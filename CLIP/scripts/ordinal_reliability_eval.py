import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import numpy as np
import torch
import torch.nn as nn

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(SCRIPT_ROOT))

from calibrated_eval import (
    collect_logits,
    combine_logits,
    metrics_from_logits,
    prediction_payload,
    select_weights,
)
from main import load_clip_state_dict, patch_state_dict_text_projection, unwrap_model
from clip.model import build_fusion_model
from data.dataset import ClassificationDataset, create_dataloader, get_transforms, seed_everything


def metrics_from_preds(preds: np.ndarray, labels: torch.Tensor):
    labels_np = labels.numpy()
    num_classes = int(max(labels_np.max(), preds.max()) + 1) if len(labels_np) else 5
    precision, recall, f1 = [], [], []
    for c in range(num_classes):
        tp = float(np.sum((preds == c) & (labels_np == c)))
        fp = float(np.sum((preds == c) & (labels_np != c)))
        fn = float(np.sum((preds != c) & (labels_np == c)))
        p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        precision.append(p)
        recall.append(r)
        f1.append(2 * p * r / (p + r) if (p + r) > 0 else 0.0)
    return {
        "accuracy": float(np.mean(preds == labels_np)),
        "macro_precision": float(np.mean(precision)),
        "macro_recall": float(np.mean(recall)),
        "macro_f1": float(np.mean(f1)),
        "mae": float(np.mean(np.abs(preds - labels_np))),
    }


def score_tuple(metrics, objective):
    return (
        metrics[objective],
        metrics["accuracy"],
        metrics["macro_f1"],
        -metrics["mae"],
    )


def fit_bias(val_logits: torch.Tensor, labels: torch.Tensor, objective: str):
    num_classes = val_logits.shape[1]
    bias = torch.zeros(num_classes)
    grid = torch.linspace(-1.0, 1.0, steps=41)
    best_metrics = metrics_from_logits(val_logits, labels)

    for _ in range(4):
        improved = False
        for c in range(num_classes):
            local_best_bias = bias.clone()
            local_best_metrics = best_metrics
            for value in grid:
                candidate = bias.clone()
                candidate[c] = value
                candidate = candidate - candidate.mean()
                preds = (val_logits + candidate).argmax(dim=-1).numpy()
                metrics = metrics_from_preds(preds, labels)
                if score_tuple(metrics, objective) > score_tuple(local_best_metrics, objective):
                    local_best_bias = candidate
                    local_best_metrics = metrics
            if not torch.equal(local_best_bias, bias):
                improved = True
                bias = local_best_bias
                best_metrics = local_best_metrics
        if not improved:
            break

    return {
        "name": "class_bias",
        "bias": [float(x) for x in bias.tolist()],
        "val_metrics": best_metrics,
    }


def apply_neighbor_smoothing(probs: torch.Tensor, alpha: float):
    if alpha <= 0:
        return probs
    num_classes = probs.shape[1]
    smoothed = (1.0 - alpha) * probs
    for c in range(num_classes):
        if c > 0:
            smoothed[:, c - 1] += probs[:, c] * alpha * 0.5
        else:
            smoothed[:, c] += probs[:, c] * alpha * 0.5
        if c + 1 < num_classes:
            smoothed[:, c + 1] += probs[:, c] * alpha * 0.5
        else:
            smoothed[:, c] += probs[:, c] * alpha * 0.5
    return smoothed / smoothed.sum(dim=1, keepdim=True).clamp(min=1e-9)


def fit_smoothed_argmax(val_logits: torch.Tensor, labels: torch.Tensor, objective: str):
    probs = torch.softmax(val_logits, dim=-1)
    best = None
    for alpha in np.linspace(0.0, 0.5, 11):
        smoothed = apply_neighbor_smoothing(probs, float(alpha))
        preds = smoothed.argmax(dim=-1).numpy()
        metrics = metrics_from_preds(preds, labels)
        item = {
            "name": "neighbor_smooth_argmax",
            "alpha": float(alpha),
            "val_metrics": metrics,
        }
        if best is None or score_tuple(metrics, objective) > score_tuple(best["val_metrics"], objective):
            best = item
    return best


def fit_ordinal_thresholds(val_logits: torch.Tensor, labels: torch.Tensor, objective: str):
    num_classes = val_logits.shape[1]
    class_ids = torch.arange(num_classes, dtype=torch.float32)
    best = None
    threshold_grid = np.linspace(0.25, 3.75, 25)

    for alpha in np.linspace(0.0, 0.4, 9):
        probs = apply_neighbor_smoothing(torch.softmax(val_logits, dim=-1), float(alpha))
        expected = (probs * class_ids).sum(dim=1).numpy()
        for t0_i in range(len(threshold_grid) - 3):
            for t1_i in range(t0_i + 1, len(threshold_grid) - 2):
                for t2_i in range(t1_i + 1, len(threshold_grid) - 1):
                    for t3_i in range(t2_i + 1, len(threshold_grid)):
                        thresholds = np.array([
                            threshold_grid[t0_i],
                            threshold_grid[t1_i],
                            threshold_grid[t2_i],
                            threshold_grid[t3_i],
                        ])
                        preds = np.digitize(expected, thresholds, right=False)
                        metrics = metrics_from_preds(preds, labels)
                        item = {
                            "name": "ordinal_threshold",
                            "alpha": float(alpha),
                            "thresholds": [float(x) for x in thresholds.tolist()],
                            "val_metrics": metrics,
                        }
                        if best is None or score_tuple(metrics, objective) > score_tuple(best["val_metrics"], objective):
                            best = item
    return best


def apply_decision(logits: torch.Tensor, decision):
    if decision["name"] == "argmax":
        return logits.argmax(dim=-1).numpy()
    if decision["name"] == "class_bias":
        bias = torch.tensor(decision["bias"], dtype=logits.dtype)
        return (logits + bias).argmax(dim=-1).numpy()
    if decision["name"] == "neighbor_smooth_argmax":
        probs = apply_neighbor_smoothing(torch.softmax(logits, dim=-1), decision["alpha"])
        return probs.argmax(dim=-1).numpy()
    if decision["name"] == "ordinal_threshold":
        probs = apply_neighbor_smoothing(torch.softmax(logits, dim=-1), decision["alpha"])
        class_ids = torch.arange(logits.shape[1], dtype=torch.float32)
        expected = (probs * class_ids).sum(dim=1).numpy()
        return np.digitize(expected, np.array(decision["thresholds"]), right=False)
    raise ValueError(f"Unknown decision: {decision['name']}")


def select_decision(val_logits: torch.Tensor, labels: torch.Tensor, objective: str):
    argmax_metrics = metrics_from_logits(val_logits, labels)
    candidates = [
        {"name": "argmax", "val_metrics": argmax_metrics},
        fit_bias(val_logits, labels, objective),
        fit_smoothed_argmax(val_logits, labels, objective),
        fit_ordinal_thresholds(val_logits, labels, objective),
    ]
    return max(candidates, key=lambda item: score_tuple(item["val_metrics"], objective))


def main():
    parser = argparse.ArgumentParser(description="Ordinal reliability calibration for KL grading.")
    parser.add_argument("--data_root", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--clip_model", default="ViT-B/32")
    parser.add_argument("--pubmedbert_path", default=str(PROJECT_ROOT / "PubMedBERT"))
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--img_size", type=int, default=224)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--objective", default="macro_f1",
                        choices=["accuracy", "macro_f1", "macro_recall"])
    parser.add_argument("--hflip", action="store_true")
    parser.add_argument("--proto_grid", default="0,0.05,0.1,0.15,0.2,0.3,0.4,0.5")
    parser.add_argument("--cluster_grid", default="0,0.05,0.1,0.15,0.2,0.3,0.4")
    parser.add_argument("--temperature_grid", default="0.8,1.0,1.2")
    args = parser.parse_args()

    seed_everything(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    state_dict = patch_state_dict_text_projection(
        load_clip_state_dict(args.clip_model),
        fallback_embed_dim=512,
    )
    model = build_fusion_model(
        state_dict=state_dict,
        pubmedbert_path=args.pubmedbert_path,
        freeze_text_encoder=True,
    )
    model.load_state_dict(torch.load(args.checkpoint, map_location="cpu"))
    model = model.to(device)
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
    base = unwrap_model(model)
    if hasattr(base, "refresh_text_prototypes"):
        base.refresh_text_prototypes()

    eval_tf = get_transforms(
        img_size=args.img_size,
        is_training=False,
        use_clahe=True,
        to_rgb=True,
        clahe_clip_limit=2.0,
        clahe_tile_grid_size=(8, 8),
        percentile=(1.0, 99.0),
    )
    val_ds = ClassificationDataset(args.data_root, split="val", transform=eval_tf,
                                   error_policy="raise", include_path=False)
    test_ds = ClassificationDataset(args.data_root, split="test", transform=eval_tf,
                                    error_policy="raise", include_path=False)
    val_loader = create_dataloader(val_ds, batch_size=args.batch_size, shuffle=False,
                                   num_workers=args.num_workers)
    test_loader = create_dataloader(test_ds, batch_size=args.batch_size, shuffle=False,
                                    num_workers=args.num_workers)

    val_pack = collect_logits(model, val_loader, device, use_hflip=args.hflip)
    test_pack = collect_logits(model, test_loader, device, use_hflip=args.hflip)

    proto_grid = [float(x) for x in args.proto_grid.split(",") if x.strip()]
    cluster_grid = [float(x) for x in args.cluster_grid.split(",") if x.strip()]
    temperature_grid = [float(x) for x in args.temperature_grid.split(",") if x.strip()]
    evidence = select_weights(val_pack, proto_grid, cluster_grid, temperature_grid, args.objective)
    val_logits = combine_logits(
        val_pack, evidence["proto_weight"], evidence["cluster_weight"], evidence["temperature"]
    )
    test_logits = combine_logits(
        test_pack, evidence["proto_weight"], evidence["cluster_weight"], evidence["temperature"]
    )

    decision = select_decision(val_logits, val_pack["labels"], args.objective)
    test_preds = apply_decision(test_logits, decision)
    test_metrics = metrics_from_preds(test_preds, test_pack["labels"])

    out = {
        "data_root": args.data_root,
        "checkpoint": args.checkpoint,
        "objective": args.objective,
        "hflip_tta": bool(args.hflip),
        "evidence_selected": evidence,
        "decision_selected": decision,
        "argmax_test_metrics": metrics_from_logits(test_logits, test_pack["labels"]),
        "test_metrics": test_metrics,
        "test_predictions": {
            "labels": test_pack["labels"].numpy().tolist(),
            "preds": test_preds.tolist(),
            "confusion_matrix": prediction_payload(
                torch.nn.functional.one_hot(
                    torch.tensor(test_preds), num_classes=test_logits.shape[1]
                ).float(),
                test_pack["labels"],
            )["confusion_matrix"],
        },
    }

    os.makedirs(args.output_dir, exist_ok=True)
    with open(os.path.join(args.output_dir, "calibrated_metrics.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)

    print(json.dumps({
        "output_dir": args.output_dir,
        "evidence_selected": evidence,
        "decision_selected": decision,
        "argmax_test_metrics": out["argmax_test_metrics"],
        "test_metrics": test_metrics,
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
