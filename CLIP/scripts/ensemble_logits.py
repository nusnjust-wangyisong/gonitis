#!/usr/bin/env python3
"""
软概率集成评估：加载多个 checkpoint（可不同 backbone / img_size），在 test 上
计算各自 softmax 概率并平均，再评估 acc / macro-F1 / MAE / 每类召回。

用法：
  python scripts/ensemble_logits.py \
    --data_root ../公开数据集/archive \
    --member experiments/checkpoints/checkpoints_biomedclip_bm224/archive_full/best_model.pth:biomedclip:224 \
    --member /tmp/arc_la05_ck/archive_full/best_model.pth:biomedclip:224 \
    --hflip
每个 --member 形如  路径:backbone:img_size（backbone/img_size 可省，默认 biomedclip/224）。
可选 --weights 0.6,0.4 给成员加权。
"""
import argparse
import os
import sys

import json

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import f1_score, recall_score, precision_recall_fscore_support, confusion_matrix

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from main import load_clip_state_dict, patch_state_dict_text_projection  # noqa: E402
from clip.model import build_fusion_model  # noqa: E402
from data.dataset import ClassificationDataset, create_dataloader, get_transforms, seed_everything  # noqa: E402

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def parse_member(spec):
    parts = spec.split(":")
    ckpt = parts[0]
    backbone = parts[1] if len(parts) > 1 and parts[1] else "biomedclip"
    img_size = int(parts[2]) if len(parts) > 2 and parts[2] else 224
    return ckpt, backbone, img_size


@torch.no_grad()
def member_probs(ckpt, backbone, img_size, data_root, hflip, batch_size, num_workers, split="test"):
    state_dict = patch_state_dict_text_projection(load_clip_state_dict("ViT-B/32"), 512)
    model = build_fusion_model(state_dict=state_dict, backbone=backbone, img_size=img_size,
                               freeze_text_encoder=True)
    missing, unexpected = model.load_state_dict(torch.load(ckpt, map_location="cpu"), strict=False)
    model = model.to(DEVICE).eval()
    if hasattr(model, "refresh_text_prototypes"):
        model.refresh_text_prototypes()

    normalize = (backbone == "biomedclip")
    tf = get_transforms(img_size=img_size, is_training=False, use_clahe=True, to_rgb=True,
                        clahe_clip_limit=2.0, clahe_tile_grid_size=(8, 8),
                        percentile=(1.0, 99.0), normalize=normalize)
    ds = ClassificationDataset(data_root, split=split, transform=tf,
                               error_policy="raise", include_path=False)
    loader = create_dataloader(ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    all_probs, all_labels = [], []
    for batch in loader:
        images = batch["image"].to(DEVICE, non_blocking=True)
        labels = batch["label"]
        logits = model(images)["logits_cls"]
        if hflip:
            logits_f = model(torch.flip(images, dims=[3]))["logits_cls"]
            logits = (logits + logits_f) / 2
        all_probs.append(F.softmax(logits.float(), dim=-1).cpu().numpy())
        all_labels.append(labels.numpy())
    return np.concatenate(all_probs), np.concatenate(all_labels)


def report(probs, labels, title):
    preds = probs.argmax(axis=1)
    acc = (preds == labels).mean()
    f1 = f1_score(labels, preds, average="macro", zero_division=0)
    mae = np.abs(preds - labels).mean()
    rec = recall_score(labels, preds, average=None, labels=[0, 1, 2, 3, 4], zero_division=0)
    print(f"{title:28s} acc={acc:.4f} F1={f1:.4f} MAE={mae:.4f} | "
          + " ".join(f"KL{k}={rec[k]:.3f}" for k in range(5)))
    return acc, f1, mae


def full_metrics(probs, labels):
    """返回每类 precision/recall/f1/support 与混淆矩阵，用于报告。"""
    preds = probs.argmax(axis=1)
    labs = [0, 1, 2, 3, 4]
    p, r, f, s = precision_recall_fscore_support(labels, preds, labels=labs, zero_division=0)
    cm = confusion_matrix(labels, preds, labels=labs)
    return {
        "accuracy": float((preds == labels).mean()),
        "macro_f1": float(f1_score(labels, preds, average="macro", zero_division=0)),
        "mae": float(np.abs(preds - labels).mean()),
        "per_class": {str(k): {"precision": float(p[k]), "recall": float(r[k]),
                               "f1": float(f[k]), "support": int(s[k])} for k in labs},
        "confusion_matrix": cm.tolist(),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", required=True)
    ap.add_argument("--member", action="append", required=True, help="ckpt:backbone:img_size")
    ap.add_argument("--weights", default=None, help="逗号分隔的成员权重，默认等权")
    ap.add_argument("--hflip", action="store_true")
    ap.add_argument("--batch_size", type=int, default=128)
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--calibrate", action="store_true",
                    help="在 val 上随机搜索成员权重，最大化 --calib_objective，再应用到 test")
    ap.add_argument("--calib_objective", default="accuracy", choices=["accuracy", "macro_f1"])
    ap.add_argument("--calib_iters", type=int, default=2000)
    ap.add_argument("--dump", default=None,
                    help="把最终集成的完整指标（每类 precision/recall/f1 + 混淆矩阵）写到该 json 路径")
    args = ap.parse_args()

    seed_everything(args.seed)
    members = [parse_member(m) for m in args.member]
    M = len(members)

    # 计算每个成员在 test 和（校准时）val 上的概率
    test_probs, test_labels = [], None
    val_probs, val_labels = [], None
    print("=" * 90)
    for spec, (ckpt, backbone, img_size) in zip(args.member, members):
        if spec.startswith("npy:"):
            # 预计算概率成员（如 ConvNeXt-V2）：spec = npy:<dir>，含 test/val_probs.npy
            d = spec[len("npy:"):]
            pt = np.load(os.path.join(d, "test_probs.npy")); yt = np.load(os.path.join(d, "test_labels.npy"))
            label = f"[member-npy] {os.path.basename(d.rstrip('/'))}"
            pv = yv = None
            if args.calibrate:
                pv = np.load(os.path.join(d, "val_probs.npy")); yv = np.load(os.path.join(d, "val_labels.npy"))
        else:
            pt, yt = member_probs(ckpt, backbone, img_size, args.data_root,
                                  args.hflip, args.batch_size, args.num_workers, split="test")
            label = f"[member] {os.path.basename(os.path.dirname(ckpt))}/{backbone}{img_size}"
            pv = yv = None
            if args.calibrate:
                pv, yv = member_probs(ckpt, backbone, img_size, args.data_root,
                                      args.hflip, args.batch_size, args.num_workers, split="val")
        test_labels = yt if test_labels is None else test_labels
        assert np.array_equal(test_labels, yt), "成员间 test 标签顺序不一致"
        report(pt, yt, label)
        test_probs.append(pt)
        if args.calibrate:
            val_labels = yv if val_labels is None else val_labels
            assert np.array_equal(val_labels, yv), "成员间 val 标签顺序不一致"
            val_probs.append(pv)
    test_probs = np.stack(test_probs)  # [M, N, 5]

    if args.weights:
        weights = np.array([float(x) for x in args.weights.split(",")], dtype=float)
    else:
        weights = np.ones(M, dtype=float)
    weights = weights / weights.sum()

    print("-" * 90)
    report(np.tensordot(weights, test_probs, axes=([0], [0])), test_labels, "[ENSEMBLE 等权/指定]")

    if args.calibrate:
        val_probs = np.stack(val_probs)  # [M, Nval, 5]
        def val_score(w):
            ens = np.tensordot(w, val_probs, axes=([0], [0]))
            preds = ens.argmax(1)
            if args.calib_objective == "accuracy":
                return (preds == val_labels).mean()
            return f1_score(val_labels, preds, average="macro", zero_division=0)
        rng = np.random.RandomState(args.seed)
        best_w, best_s = weights.copy(), val_score(weights)
        # 含等权与单成员 one-hot 的候选 + Dirichlet 随机搜索
        cands = [np.eye(M)[i] for i in range(M)] + [np.ones(M) / M]
        for _ in range(args.calib_iters):
            cands.append(rng.dirichlet(np.ones(M)))
        for w in cands:
            s = val_score(w)
            if s > best_s:
                best_s, best_w = s, np.array(w)
        print("-" * 90)
        print(f"[校准] val {args.calib_objective}={best_s:.4f}  最优权重={[round(float(x),3) for x in best_w]}")
        report(np.tensordot(best_w, test_probs, axes=([0], [0])), test_labels, "[ENSEMBLE 校准后]")
        weights = best_w  # 导出时用校准后的权重
    print("=" * 90)

    if args.dump:
        final_ens = np.tensordot(weights, test_probs, axes=([0], [0]))
        m = full_metrics(final_ens, test_labels)
        os.makedirs(os.path.dirname(args.dump) or ".", exist_ok=True)
        json.dump(m, open(args.dump, "w"), indent=2, ensure_ascii=False)
        print(f"[dump] 完整指标已写入 {args.dump}")
        print("混淆矩阵：")
        for row in m["confusion_matrix"]:
            print("  " + " ".join(f"{v:4d}" for v in row))


if __name__ == "__main__":
    main()
