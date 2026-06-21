"""
Text-guided cache adapter for full_dual KL grading.

Frozen full-dual baseline. This is a Tip-Adapter style multimodal head:
  - visual cache logits from train-set image features and labels;
  - text logits from KL text prototypes;
  - validation-selected fusion with the baseline classifier logits.

No model weights are changed and the no-op solution is included in the grid.
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


def score_key(m, objective):
    return (m[objective], m["accuracy"], m["macro_f1"], -m["mae"])


def zscore(logits):
    return (logits - logits.mean(dim=1, keepdim=True)) / logits.std(dim=1, keepdim=True).clamp(min=1e-6)


def confidence_gate(base_logits, tau):
    conf = torch.softmax(base_logits.float(), dim=-1).max(dim=-1).values
    return (conf < tau).float().unsqueeze(1)


@torch.no_grad()
def extract(model, data_root, split, device, batch_size, hflip=False, hflip_features_only=False):
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
    text_proto = F.normalize(model.text_feat.float(), dim=-1)
    temp = model.TEMPERATURE
    feats, base_logits, text_logits, labels = [], [], [], []
    for batch in loader:
        image = batch["image"].to(device)
        out0 = model(image)
        views = [image]
        if hflip or hflip_features_only:
            views.append(torch.flip(image, dims=[3]))
        feat_views, base_views, text_views = [], [], []
        for view in views:
            out = model(view)
            feat = F.normalize(out["img_feat"].float(), dim=-1)
            feat_views.append(feat)
            base_views.append(out["logits_classifier"].float())
            text_views.append(feat @ text_proto.T / temp)
        feat_avg = F.normalize(torch.stack(feat_views, dim=0).mean(dim=0), dim=-1)
        feats.append(feat_avg.cpu())
        if hflip and not hflip_features_only:
            base_logits.append(torch.stack(base_views, dim=0).mean(dim=0).cpu())
        else:
            base_logits.append(out0["logits_classifier"].float().cpu())
        text_logits.append(torch.stack(text_views, dim=0).mean(dim=0).cpu())
        labels.extend(batch["label"].numpy())
    return torch.cat(feats), torch.cat(base_logits), torch.cat(text_logits), np.asarray(labels)


def cache_logits(query_feats, cache_feats, cache_labels, beta, balanced):
    # query_feats: [N,D], cache_feats: [M,D]
    onehot = F.one_hot(torch.as_tensor(cache_labels, dtype=torch.long), num_classes=5).float()
    counts = onehot.sum(dim=0).clamp(min=1.0)
    out = []
    chunk = 512
    for start in range(0, query_feats.size(0), chunk):
        q = query_feats[start:start + chunk]
        sim = q @ cache_feats.T
        affinity = torch.exp(beta * (sim - 1.0))
        logits = affinity @ onehot
        if balanced:
            logits = logits / counts
        out.append(logits)
    return torch.cat(out, dim=0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, choices=CFG.keys())
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--objective", choices=["accuracy", "macro_f1"], default="macro_f1")
    parser.add_argument("--folds", type=int, default=1,
                        help="If >1, select weights by deterministic folds over val for stability.")
    parser.add_argument("--final_cache_trainval", action="store_true",
                        help="After selecting weights on val using train cache, rebuild test cache with train+val.")
    parser.add_argument("--confidence_gate", action="store_true",
                        help="Only apply cache/text residual to samples with baseline confidence below selected tau.")
    parser.add_argument("--classwise_weights", action="store_true",
                        help="After global selection, coordinate-search per-class cache/text residual weights on val.")
    parser.add_argument("--hflip", action="store_true",
                        help="Average original and horizontal-flip views when extracting logits/features.")
    parser.add_argument("--hflip_features_only", action="store_true",
                        help="Average hflip features/text/cache, but keep baseline logits from original images.")
    parser.add_argument("--out", default="experiments/results/text_cache_adapter")
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

    tr_f, _, _, tr_y = extract(
        model, data_root, "train", device, args.batch_size,
        hflip=args.hflip, hflip_features_only=args.hflip_features_only,
    )
    va_f, va_base, va_text, va_y = extract(
        model, data_root, "val", device, args.batch_size,
        hflip=args.hflip, hflip_features_only=args.hflip_features_only,
    )
    te_f, te_base, te_text, te_y = extract(
        model, data_root, "test", device, args.batch_size,
        hflip=args.hflip, hflip_features_only=args.hflip_features_only,
    )

    base_val = metrics(va_base, va_y)
    base_test = metrics(te_base, te_y)

    best = None
    beta_grid = [1.0, 2.0, 5.0, 10.0, 20.0, 40.0]
    cache_w_grid = [0.0, 0.02, 0.05, 0.08, 0.1, 0.15, 0.2, 0.3, 0.5, 0.8, 1.0]
    text_w_grid = [0.0, 0.02, 0.05, 0.08, 0.1, 0.15, 0.2, 0.3, 0.5]
    tau_grid = [1.01] if not args.confidence_gate else [0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.01]
    n_val = len(va_y)
    fold_indices = [np.arange(n_val)] if args.folds <= 1 else [
        np.arange(n_val)[i::args.folds] for i in range(args.folds)
    ]
    base_val_fold_metrics = [metrics(va_base[idx], va_y[idx]) for idx in fold_indices]

    best_cache_full = None
    best_text_full = None
    for balanced in [False, True]:
        for beta in beta_grid:
            va_cache = cache_logits(va_f, tr_f, tr_y, beta=beta, balanced=balanced)
            for cw in cache_w_grid:
                for tw in text_w_grid:
                    residual = cw * zscore(va_cache) + tw * zscore(va_text)
                    for tau in tau_grid:
                        final = zscore(va_base) + confidence_gate(va_base, tau) * residual
                        m = metrics(final, va_y)
                        if args.folds > 1:
                            fold_scores = []
                            pos_folds = 0
                            for idx, bm in zip(fold_indices, base_val_fold_metrics):
                                fm = metrics(final[idx], va_y[idx])
                                gain = fm[args.objective] - bm[args.objective]
                                fold_scores.append(gain)
                                pos_folds += int(gain > 0)
                            # Prioritize fold-stable gains; full-val metric is tie-breaker.
                            key = (
                                pos_folds,
                                float(np.mean(fold_scores)),
                                m[args.objective],
                                m["accuracy"],
                                m["macro_f1"],
                                -m["mae"],
                            )
                        else:
                            key = score_key(m, args.objective)
                        if best is None or key > best["key"]:
                            best = {
                                "key": key,
                                "balanced": balanced,
                                "beta": beta,
                                "cache_weight": cw,
                                "text_weight": tw,
                                "tau": tau,
                                "val": m,
                            }
                            best_cache_full = va_cache
                            best_text_full = va_text

    cache_class_weight = torch.full((5,), float(best["cache_weight"]))
    text_class_weight = torch.full((5,), float(best["text_weight"]))
    if args.classwise_weights and (best["cache_weight"] > 0 or best["text_weight"] > 0):
        if best_cache_full is None:
            best_cache_full = cache_logits(va_f, tr_f, tr_y, beta=best["beta"], balanced=best["balanced"])
        best_text_full = va_text
        cw_grid_local = [0.0, 0.02, 0.05, 0.08, 0.1, 0.15, 0.2, 0.3, 0.5]
        tw_grid_local = [0.0, 0.02, 0.05, 0.08, 0.1, 0.15, 0.2, 0.3, 0.5]
        current = dict(best)
        for _ in range(3):
            improved = False
            for c in range(5):
                local_best = current
                local_cw = cache_class_weight.clone()
                local_tw = text_class_weight.clone()
                for cw in cw_grid_local:
                    cand_cw = cache_class_weight.clone()
                    cand_cw[c] = cw
                    for tw in tw_grid_local:
                        cand_tw = text_class_weight.clone()
                        cand_tw[c] = tw
                        residual = (
                            zscore(best_cache_full) * cand_cw.view(1, -1)
                            + zscore(best_text_full) * cand_tw.view(1, -1)
                        )
                        final = zscore(va_base) + confidence_gate(va_base, best["tau"]) * residual
                        m = metrics(final, va_y)
                        key = score_key(m, args.objective)
                        if key > local_best["key"]:
                            local_best = {
                                "key": key,
                                "balanced": best["balanced"],
                                "beta": best["beta"],
                                "cache_weight": float(cand_cw.mean().item()),
                                "text_weight": float(cand_tw.mean().item()),
                                "tau": best["tau"],
                                "val": m,
                            }
                            local_cw = cand_cw
                            local_tw = cand_tw
                if not torch.equal(local_cw, cache_class_weight) or not torch.equal(local_tw, text_class_weight):
                    cache_class_weight = local_cw
                    text_class_weight = local_tw
                    current = local_best
                    improved = True
            if not improved:
                break
        best = current

    final_cache_f = tr_f
    final_cache_y = tr_y
    if args.final_cache_trainval:
        final_cache_f = torch.cat([tr_f, va_f], dim=0)
        final_cache_y = np.concatenate([tr_y, va_y])
    te_cache = cache_logits(te_f, final_cache_f, final_cache_y, beta=best["beta"], balanced=best["balanced"])
    te_residual = (
        zscore(te_cache) * cache_class_weight.view(1, -1)
        + zscore(te_text) * text_class_weight.view(1, -1)
    )
    te_final = zscore(te_base) + confidence_gate(te_base, best["tau"]) * te_residual
    test_m = metrics(te_final, te_y)
    result = {
        "dataset": args.dataset,
        "base_val": base_val,
        "base_test": base_test,
        "selected": {
            "balanced_cache": best["balanced"],
            "beta": best["beta"],
            "cache_weight": best["cache_weight"],
            "text_weight": best["text_weight"],
            "cache_class_weight": [float(x) for x in cache_class_weight.tolist()],
            "text_class_weight": [float(x) for x in text_class_weight.tolist()],
            "classwise_weights": bool(args.classwise_weights),
            "tau": best["tau"],
            "confidence_gate": bool(args.confidence_gate),
            "folds": args.folds,
            "final_cache_trainval": bool(args.final_cache_trainval),
            "hflip": bool(args.hflip),
            "hflip_features_only": bool(args.hflip_features_only),
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
