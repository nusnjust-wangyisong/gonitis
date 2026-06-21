#!/usr/bin/env python3
"""
微调 timm ConvNeXt-V2 作为 KL 五分类的异构集成成员（与 BiomedCLIP 主干不同），
复用项目同一套数据管线（CLAHE 预处理 + 同样的几何增强），仅归一化改为 ImageNet。

训练完成后保存：
  <save_dir>/best_model.pth
  <out_dir>/test_probs.npy, test_labels.npy, val_probs.npy, val_labels.npy   （含 hflip TTA）
  <out_dir>/test_metrics.json
这样可被 scripts/ensemble_logits.py 以 npy 成员形式直接集成。
"""
import argparse
import json
import os
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from sklearn.metrics import f1_score, recall_score

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)
from data.dataset import ClassificationDataset, create_dataloader, get_transforms, seed_everything  # noqa: E402

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def build_loaders(data_root, img_size, batch_size, num_workers):
    train_tf = get_transforms(img_size=img_size, is_training=True, use_clahe=True, to_rgb=True,
                              clahe_clip_limit=2.0, clahe_tile_grid_size=(8, 8), percentile=(1.0, 99.0),
                              rotate_deg=10, hflip_p=0.5, random_erasing_p=0.0, normalize="imagenet")
    eval_tf = get_transforms(img_size=img_size, is_training=False, use_clahe=True, to_rgb=True,
                             clahe_clip_limit=2.0, clahe_tile_grid_size=(8, 8), percentile=(1.0, 99.0),
                             normalize="imagenet")
    loaders = {}
    for split, tf, shuffle in [("train", train_tf, True), ("val", eval_tf, False), ("test", eval_tf, False)]:
        ds = ClassificationDataset(data_root, split=split, transform=tf,
                                   error_policy="raise", include_path=False)
        loaders[split] = create_dataloader(ds, batch_size=batch_size, shuffle=shuffle,
                                            num_workers=num_workers)
        loaders[split + "_ds"] = ds
    return loaders


@torch.no_grad()
def eval_probs(model, loader, hflip):
    model.eval()
    probs, labels = [], []
    for batch in loader:
        x = batch["image"].to(DEVICE, non_blocking=True)
        logits = model(x)
        if hflip:
            logits = (logits + model(torch.flip(x, dims=[3]))) / 2
        probs.append(F.softmax(logits.float(), dim=-1).cpu().numpy())
        labels.append(batch["label"].numpy())
    return np.concatenate(probs), np.concatenate(labels)


def metrics(probs, labels):
    preds = probs.argmax(1)
    rec = recall_score(labels, preds, average=None, labels=[0, 1, 2, 3, 4], zero_division=0)
    return {
        "accuracy": float((preds == labels).mean()),
        "macro_f1": float(f1_score(labels, preds, average="macro", zero_division=0)),
        "mae": float(np.abs(preds - labels).mean()),
        "per_class_recall": {str(k): float(rec[k]) for k in range(5)},
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", required=True)
    ap.add_argument("--model", default="convnextv2_base.fcmae_ft_in22k_in1k")
    ap.add_argument("--img_size", type=int, default=224)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--patience", type=int, default=8)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--head_lr", type=float, default=1e-3)
    ap.add_argument("--weight_decay", type=float, default=0.05)
    ap.add_argument("--label_smoothing", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--save_dir", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--hflip", action="store_true", default=True)
    args = ap.parse_args()

    seed_everything(args.seed)
    os.makedirs(args.save_dir, exist_ok=True)
    os.makedirs(args.out_dir, exist_ok=True)

    loaders = build_loaders(args.data_root, args.img_size, args.batch_size, args.num_workers)
    model = timm.create_model(args.model, pretrained=True, num_classes=5).to(DEVICE)

    # 分层学习率：主干小 lr，分类头大 lr。
    # 注意：必须用 "head." 前缀精确匹配 timm 分类头（head.norm/head.fc），
    # 不能用 "fc" 子串——那会误匹配主干里所有 mlp.fc1/fc2，导致主干被高 lr 摧毁。
    def is_head(n):
        return n.startswith("head.")
    head_params = [p for n, p in model.named_parameters() if is_head(n)]
    body_params = [p for n, p in model.named_parameters() if not is_head(n)]
    print(f"[param-split] head={len(head_params)} body={len(body_params)}")
    opt = torch.optim.AdamW([
        {"params": body_params, "lr": args.lr},
        {"params": head_params, "lr": args.head_lr},
    ], weight_decay=args.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    ce = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    scaler = torch.cuda.amp.GradScaler()

    best_val, best_epoch, ckpt_path = -1.0, 0, os.path.join(args.save_dir, "best_model.pth")
    for epoch in range(1, args.epochs + 1):
        model.train()
        for batch in loaders["train"]:
            x = batch["image"].to(DEVICE, non_blocking=True)
            y = batch["label"].to(DEVICE, non_blocking=True)
            opt.zero_grad()
            with torch.cuda.amp.autocast():
                loss = ce(model(x), y)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
        sched.step()
        vp, vy = eval_probs(model, loaders["val"], args.hflip)
        vacc = (vp.argmax(1) == vy).mean()
        tag = ""
        if vacc > best_val:
            best_val, best_epoch = vacc, epoch
            torch.save(model.state_dict(), ckpt_path)
            tag = "  ✓best"
        print(f"Epoch {epoch:3d} | val_acc={vacc:.4f}{tag}", flush=True)
        if epoch - best_epoch >= args.patience:
            print(f"[Early Stopping] epoch {epoch}, best={best_val:.4f}@{best_epoch}")
            break

    model.load_state_dict(torch.load(ckpt_path, map_location=DEVICE))
    for split in ["val", "test"]:
        p, y = eval_probs(model, loaders[split], args.hflip)
        np.save(os.path.join(args.out_dir, f"{split}_probs.npy"), p)
        np.save(os.path.join(args.out_dir, f"{split}_labels.npy"), y)
        if split == "test":
            m = metrics(p, y)
            json.dump(m, open(os.path.join(args.out_dir, "test_metrics.json"), "w"), indent=2)
            print(f"[TEST] acc={m['accuracy']:.4f} F1={m['macro_f1']:.4f} MAE={m['mae']:.4f} "
                  + " ".join(f"KL{k}={m['per_class_recall'][str(k)]:.3f}" for k in range(5)))


if __name__ == "__main__":
    main()
