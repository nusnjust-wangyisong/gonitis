"""
Text-Visual Prototype Blending (TVPB).

Frozen full-dual model:
  1) build visual class prototypes from train fused features;
  2) blend them with KL text prototypes;
  3) validation-select alpha (text contribution) and ensemble weight;
  4) apply to test.

This follows the recent medical VLM adaptation idea that text embeddings can be
used to inform lightweight linear/prototype heads, rather than directly
backpropagating text losses into a saturated visual backbone.
"""
import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import DataLoader

from clip.model import build_fusion_model
from data.dataset import ClassificationDataset, get_transforms
from main import load_clip_state_dict


CFG = {
    "me": ("../公开数据集/MedicalExpert-split",
           "experiments/checkpoints/checkpoints_dual_me_ms_fix4/MedicalExpert-split_full_dual/best_model.pth"),
    "ar": ("../公开数据集/archive",
           "experiments/checkpoints/checkpoints_dual_ar_ms_fix10/archive_full_dual/best_model.pth"),
    "pv": ("../私有数据集/split_result_siyouceshi_roi",
           "experiments/checkpoints/checkpoints_dual_pv_ms_fix9/split_result_siyouceshi_roi_full_dual/best_model.pth"),
}


def metrics(logits, labels):
    pred = logits.argmax(dim=-1).cpu().numpy()
    return {
        "accuracy": float(accuracy_score(labels, pred)),
        "macro_f1": float(f1_score(labels, pred, average="macro", zero_division=0)),
        "mae": float(np.abs(pred - labels).mean()),
    }


def zscore(logits):
    return (logits - logits.mean(dim=1, keepdim=True)) / logits.std(dim=1, keepdim=True).clamp(min=1e-6)


@torch.no_grad()
def extract(model, data_root, split, device, batch_size):
    tf = get_transforms(
        img_size=224,
        is_training=False,
        use_clahe=True,
        to_rgb=True,
        clahe_clip_limit=2.0,
        clahe_tile_grid_size=(8, 8),
        percentile=(1.0, 99.0),
        normalize=True,
    )
    ds = ClassificationDataset(data_root, split=split, transform=tf)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=4)
    feats, logits, labels = [], [], []
    for batch in loader:
        out = model(batch["image"].to(device))
        feats.append(F.normalize(out["img_feat"].float(), dim=-1).cpu())
        logits.append(out["logits_classifier"].float().cpu())
        labels.extend(batch["label"].numpy())
    return torch.cat(feats), torch.cat(logits), np.asarray(labels)


def build_visual_prototypes(feats, labels, num_classes=5):
    protos = []
    y = torch.as_tensor(labels)
    for c in range(num_classes):
        mask = y == c
        if not mask.any():
            protos.append(torch.zeros(feats.size(1)))
        else:
            protos.append(feats[mask].mean(dim=0))
    return F.normalize(torch.stack(protos), dim=-1)


def proto_logits(feats, visual_proto, text_proto, alpha, scale):
    if not torch.is_tensor(alpha):
        alpha = torch.full((visual_proto.size(0), 1), float(alpha), dtype=visual_proto.dtype)
    elif alpha.dim() == 1:
        alpha = alpha.view(-1, 1).to(dtype=visual_proto.dtype)
    proto = F.normalize((1.0 - alpha) * visual_proto + alpha * text_proto, dim=-1)
    return scale * feats @ proto.T


def score_key(metrics_dict, objective):
    return (
        metrics_dict[objective],
        metrics_dict["accuracy"],
        metrics_dict["macro_f1"],
        -metrics_dict["mae"],
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, choices=CFG.keys())
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--objective", choices=["accuracy", "macro_f1"], default="macro_f1")
    parser.add_argument("--classwise", action="store_true",
                        help="Coordinate-search one text blending alpha per KL class.")
    parser.add_argument("--out", default="experiments/results/text_visual_proto_blend")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_root, ckpt = CFG[args.dataset]
    sd = load_clip_state_dict("ViT-B/32")
    model = build_fusion_model(
        sd, backbone="biomedclip", enable_dual_branch=True, convnext_multi_scale=True
    ).to(device).eval()
    raw = torch.load(ckpt, map_location="cpu")
    model.load_state_dict(raw.get("model_state_dict", raw), strict=False)
    model.refresh_text_prototypes()

    train_f, _, train_y = extract(model, data_root, "train", device, args.batch_size)
    val_f, val_base, val_y = extract(model, data_root, "val", device, args.batch_size)
    test_f, test_base, test_y = extract(model, data_root, "test", device, args.batch_size)

    visual_proto = build_visual_prototypes(train_f, train_y)
    text_proto = F.normalize(model.text_feat.detach().float().cpu(), dim=-1)

    base_val = metrics(val_base, val_y)
    base_test = metrics(test_base, test_y)
    alpha_grid = [0.0, 0.02, 0.05, 0.08, 0.1, 0.15, 0.2, 0.3, 0.5, 0.8]
    weight_grid = [0.0, 0.02, 0.05, 0.08, 0.1, 0.15, 0.2, 0.3, 0.5, 0.8, 1.0]
    scale_grid = [1.0, 2.0, 5.0, 10.0, 20.0]
    best = None
    for alpha in alpha_grid:
        for scale in scale_grid:
            vp_val = proto_logits(val_f, visual_proto, text_proto, alpha, scale)
            for weight in weight_grid:
                final = zscore(val_base) + weight * zscore(vp_val)
                m = metrics(final, val_y)
                key = score_key(m, args.objective)
                if best is None or key > best["key"]:
                    best = {"key": key, "alpha": alpha, "weight": weight, "scale": scale, "val": m}

    if args.classwise and best["weight"] > 0:
        alpha_vec = torch.full((5,), float(best["alpha"]))
        best_local = dict(best)
        # Coordinate search over per-class text ratios while keeping the selected
        # ensemble weight/scale. This avoids a high-dimensional exhaustive grid.
        for _ in range(3):
            improved = False
            for c in range(5):
                cur_alpha = alpha_vec.clone()
                cur_best = best_local
                for a in alpha_grid:
                    cand_alpha = alpha_vec.clone()
                    cand_alpha[c] = a
                    vp_val = proto_logits(val_f, visual_proto, text_proto, cand_alpha, best["scale"])
                    final = zscore(val_base) + best["weight"] * zscore(vp_val)
                    m = metrics(final, val_y)
                    key = score_key(m, args.objective)
                    if key > cur_best["key"]:
                        cur_alpha = cand_alpha
                        cur_best = {
                            "key": key,
                            "alpha": cand_alpha.clone(),
                            "weight": best["weight"],
                            "scale": best["scale"],
                            "val": m,
                        }
                if not torch.equal(cur_alpha, alpha_vec):
                    alpha_vec = cur_alpha
                    best_local = cur_best
                    improved = True
            if not improved:
                break
        best = best_local

    vp_test = proto_logits(test_f, visual_proto, text_proto, best["alpha"], best["scale"])
    test_final = zscore(test_base) + best["weight"] * zscore(vp_test)
    test_m = metrics(test_final, test_y)
    result = {
        "dataset": args.dataset,
        "base_val": base_val,
        "base_test": base_test,
        "selected": {
            "alpha_text": (
                float(best["alpha"]) if not torch.is_tensor(best["alpha"]) else None
            ),
            "alpha_text_per_class": (
                [float(x) for x in best["alpha"].tolist()]
                if torch.is_tensor(best["alpha"]) else None
            ),
            "proto_weight": best["weight"],
            "proto_scale": best["scale"],
            "val": best["val"],
            "test": test_m,
        },
        "delta_test": {k: test_m[k] - base_test[k] for k in ["accuracy", "macro_f1", "mae"]},
    }
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"{args.dataset}.json"
    path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(result, indent=2, ensure_ascii=False))
    print(f"saved: {path}")


if __name__ == "__main__":
    main()
