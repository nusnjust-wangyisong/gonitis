"""
快速扫描 inference_branch_text_weight 对已训练最优模型的影响。
不重新训练，直接在测试集上用不同权重评估。

用法：
  python scripts/eval_branch_text_weight.py \
    --checkpoint experiments/checkpoints/checkpoints_dual_me_ms_fix4/MedicalExpert-split_full_dual/best_model.pth \
    --data_root /home/kaixin/code/gonitis_dataset/公开数据集/MedicalExpert-split \
    --weights 0.0 0.05 0.1 0.2 0.3 0.5
"""

import argparse
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
from sklearn.metrics import accuracy_score, f1_score

from clip.model import build_fusion_model
from data.dataset import ClassificationDataset, get_transforms
from main import load_clip_state_dict


def build_model_from_ckpt(ckpt_path, data_root, device):
    state_dict_clip = load_clip_state_dict("ViT-B/32")

    model = build_fusion_model(
        state_dict_clip,
        backbone="biomedclip",
        enable_dual_branch=True,
        convnext_multi_scale=True,
        lambda_ordinal=0.2,
        lambda_branch_text=0.1,           # 让 logits_anatomy/pathology 在 forward 里被计算
        inference_branch_text_weight=0.0,  # 先设 0，后面动态改
    )

    ckpt = torch.load(ckpt_path, map_location="cpu")
    sd = ckpt.get("model_state_dict", ckpt)
    missing, unexpected = model.load_state_dict(sd, strict=False)
    print(f"  loaded checkpoint: missing={len(missing)}, unexpected={len(unexpected)}")

    model = model.to(device).eval()
    model.refresh_text_prototypes()
    return model


def evaluate(model, loader, device, weight):
    model.inference_branch_text_weight = weight
    all_preds, all_labels = [], []
    with torch.no_grad():
        for batch in loader:
            imgs = batch["image"].to(device)
            labels = batch["label"]
            out = model(imgs)
            preds = out["logits_cls"].argmax(dim=-1).cpu()
            all_preds.extend(preds.numpy())
            all_labels.extend(labels.numpy())
    acc = accuracy_score(all_labels, all_preds)
    f1  = f1_score(all_labels, all_preds, average="macro", zero_division=0)
    return acc, f1


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data_root",  required=True)
    parser.add_argument("--weights", type=float, nargs="+",
                        default=[0.0, 0.05, 0.1, 0.15, 0.2, 0.3, 0.5])
    parser.add_argument("--split", default="test", choices=["val", "test"])
    parser.add_argument("--batch_size", type=int, default=64)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading checkpoint: {args.checkpoint}")
    model = build_model_from_ckpt(args.checkpoint, args.data_root, device)

    transform = get_transforms(img_size=224, is_training=False, normalize=True)
    dataset = ClassificationDataset(args.data_root, split=args.split, transform=transform)
    loader  = DataLoader(dataset, batch_size=args.batch_size, shuffle=False,
                         num_workers=4, pin_memory=True)
    print(f"{args.split} set: {len(dataset)} samples\n")

    print(f"{'Weight':>8}  {'Acc':>7}  {'MacroF1':>9}")
    print("-" * 32)
    best_acc, best_w = 0.0, 0.0
    for w in args.weights:
        acc, f1 = evaluate(model, loader, device, w)
        marker = " ←" if acc > best_acc else ""
        print(f"{w:>8.3f}  {acc:.4f}  {f1:.4f}{marker}")
        if acc > best_acc:
            best_acc, best_w = acc, w
    print(f"\n最优权重: {best_w}  最优 Acc: {best_acc:.4f}")


if __name__ == "__main__":
    main()
