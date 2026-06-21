import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(SCRIPT_ROOT))

from calibrated_eval import collect_logits, combine_logits, prediction_payload
from main import load_clip_state_dict, patch_state_dict_text_projection, unwrap_model
from clip.model import build_fusion_model
from data.dataset import ClassificationDataset, create_dataloader, get_transforms, seed_everything


def metrics_from_preds(y_true, y_pred):
    return {
        "accuracy": float(np.mean(y_pred == y_true)),
        "macro_precision": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "macro_recall": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "mae": float(np.mean(np.abs(y_pred - y_true))),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
    }


def metrics_from_logits(logits, labels):
    return metrics_from_preds(labels.numpy(), logits.argmax(dim=-1).numpy())


def parse_grid(text):
    return [float(x) for x in text.split(",") if x.strip()]


def bootstrap_score(logits, labels, rng, num_bootstrap, objective, std_penalty):
    y_true = labels.numpy()
    y_pred = logits.argmax(dim=-1).numpy()
    n = len(y_true)
    scores = []
    for _ in range(num_bootstrap):
        idx = rng.integers(0, n, size=n)
        scores.append(metrics_from_preds(y_true[idx], y_pred[idx])[objective])
    scores = np.asarray(scores, dtype=np.float64)
    return float(scores.mean() - std_penalty * scores.std()), float(scores.mean()), float(scores.std())


def select_robust_weights(
    val_pack,
    proto_grid,
    cluster_grid,
    temperature_grid,
    objective,
    num_bootstrap,
    std_penalty,
    complexity_penalty,
    seed,
    prefilter_topk,
):
    rng = np.random.default_rng(seed)
    candidates = []
    for proto_w in proto_grid:
        for cluster_w in cluster_grid:
            for temperature in temperature_grid:
                logits = combine_logits(val_pack, proto_w, cluster_w, temperature)
                val_metrics = metrics_from_logits(logits, val_pack["labels"])
                complexity = abs(proto_w) + abs(cluster_w)
                item = {
                    "proto_weight": float(proto_w),
                    "cluster_weight": float(cluster_w),
                    "temperature": float(temperature),
                    "robust_score": None,
                    "bootstrap_mean": None,
                    "bootstrap_std": None,
                    "complexity": float(complexity),
                    "metrics": val_metrics,
                }
                candidates.append(item)

    candidates = sorted(
        candidates,
        key=lambda x: (
            x["metrics"][objective],
            x["metrics"]["accuracy"],
            x["metrics"]["macro_f1"],
            -x["metrics"]["mae"],
            -x["complexity"],
        ),
        reverse=True,
    )
    candidates_for_bootstrap = candidates[:max(1, prefilter_topk)]

    best = None
    for item in candidates_for_bootstrap:
        logits = combine_logits(
            val_pack,
            item["proto_weight"],
            item["cluster_weight"],
            item["temperature"],
        )
        robust, boot_mean, boot_std = bootstrap_score(
            logits, val_pack["labels"], rng, num_bootstrap, objective, std_penalty
        )
        robust -= complexity_penalty * item["complexity"]
        item["robust_score"] = float(robust)
        item["bootstrap_mean"] = float(boot_mean)
        item["bootstrap_std"] = float(boot_std)
        key = (
            item["robust_score"],
            item["metrics"][objective],
            item["metrics"]["accuracy"],
            -item["metrics"]["mae"],
            -item["complexity"],
        )
        if best is None or key > best[0]:
            best = (key, item)
    ranked = sorted(candidates_for_bootstrap, key=lambda x: x["robust_score"], reverse=True)
    return best[1], ranked[:20], candidates[:20]


def main():
    parser = argparse.ArgumentParser(description="Bootstrap-stable evidence calibration for KL grading.")
    parser.add_argument("--data_root", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--clip_model", default="ViT-B/32")
    parser.add_argument("--pubmedbert_path", default=str(PROJECT_ROOT / "PubMedBERT"))
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--img_size", type=int, default=224)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--hflip", action="store_true")
    parser.add_argument("--objective", default="accuracy", choices=["accuracy", "macro_f1", "macro_recall"])
    parser.add_argument("--proto_grid", default="0,0.02,0.04,0.05,0.08,0.1,0.15,0.2,0.3,0.4")
    parser.add_argument("--cluster_grid", default="0,0.05,0.1,0.15,0.2,0.25,0.3,0.35,0.4,0.45,0.5")
    parser.add_argument("--temperature_grid", default="0.6,0.7,0.8,0.9,1.0")
    parser.add_argument("--num_bootstrap", type=int, default=500)
    parser.add_argument("--std_penalty", type=float, default=0.5)
    parser.add_argument("--complexity_penalty", type=float, default=0.01)
    parser.add_argument("--prefilter_topk", type=int, default=30)
    args = parser.parse_args()

    seed_everything(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    state_dict = patch_state_dict_text_projection(load_clip_state_dict(args.clip_model), fallback_embed_dim=512)
    model = build_fusion_model(
        state_dict=state_dict,
        pubmedbert_path=args.pubmedbert_path,
        freeze_text_encoder=True,
    )
    missing, unexpected = model.load_state_dict(torch.load(args.checkpoint, map_location="cpu"), strict=False)
    if missing:
        print(f"[robust] Missing keys: {missing[:8]}")
    if unexpected:
        print(f"[robust] Unexpected keys: {unexpected[:8]}")
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

    print("[robust] Collecting validation logits...")
    val_pack = collect_logits(model, val_loader, device, use_hflip=args.hflip)
    print("[robust] Collecting test logits...")
    test_pack = collect_logits(model, test_loader, device, use_hflip=args.hflip)

    selected, top_candidates, top_prefilter = select_robust_weights(
        val_pack=val_pack,
        proto_grid=parse_grid(args.proto_grid),
        cluster_grid=parse_grid(args.cluster_grid),
        temperature_grid=parse_grid(args.temperature_grid),
        objective=args.objective,
        num_bootstrap=args.num_bootstrap,
        std_penalty=args.std_penalty,
        complexity_penalty=args.complexity_penalty,
        seed=args.seed,
        prefilter_topk=args.prefilter_topk,
    )
    test_logits = combine_logits(
        test_pack,
        selected["proto_weight"],
        selected["cluster_weight"],
        selected["temperature"],
    )
    test_metrics = metrics_from_logits(test_logits, test_pack["labels"])
    raw_metrics = metrics_from_logits(test_pack["cls"], test_pack["labels"])

    out = {
        "data_root": args.data_root,
        "checkpoint": args.checkpoint,
        "hflip_tta": bool(args.hflip),
        "objective": args.objective,
        "num_bootstrap": args.num_bootstrap,
        "std_penalty": args.std_penalty,
        "complexity_penalty": args.complexity_penalty,
        "selected": selected,
        "top_candidates": top_candidates,
        "top_prefilter": top_prefilter,
        "raw_classifier_test_metrics": raw_metrics,
        "test_metrics": test_metrics,
        "test_predictions": prediction_payload(test_logits, test_pack["labels"]),
    }

    os.makedirs(args.output_dir, exist_ok=True)
    with open(os.path.join(args.output_dir, "calibrated_metrics.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)

    print(json.dumps({
        "output_dir": args.output_dir,
        "selected": selected,
        "test_metrics": test_metrics,
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
