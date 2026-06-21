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

from calibrated_eval import collect_logits, combine_logits, metrics_from_logits, prediction_payload, select_weights
from main import load_clip_state_dict, patch_state_dict_text_projection, unwrap_model
from clip.model import build_fusion_model
from data.dataset import ClassificationDataset, create_dataloader, get_transforms, seed_everything


def load_model(checkpoint, clip_state_dict, pubmedbert_path, device):
    model = build_fusion_model(
        state_dict=clip_state_dict,
        pubmedbert_path=pubmedbert_path,
        freeze_text_encoder=True,
    )
    model.load_state_dict(torch.load(checkpoint, map_location="cpu"))
    model = model.to(device)
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
    base = unwrap_model(model)
    if hasattr(base, "refresh_text_prototypes"):
        base.refresh_text_prototypes()
    return model


def mix_packs(pack_a, pack_b, alpha):
    return {
        "cls": alpha * pack_a["cls"] + (1.0 - alpha) * pack_b["cls"],
        "proto": alpha * pack_a["proto"] + (1.0 - alpha) * pack_b["proto"],
        "cluster": alpha * pack_a["cluster"] + (1.0 - alpha) * pack_b["cluster"],
        "labels": pack_a["labels"],
    }


def metric_tuple(metrics, objective):
    return (
        metrics[objective],
        metrics["accuracy"],
        metrics["macro_f1"],
        -metrics["mae"],
    )


def main():
    parser = argparse.ArgumentParser(description="Two-expert validation-calibrated ensemble evaluation.")
    parser.add_argument("--data_root", required=True)
    parser.add_argument("--checkpoint_a", required=True)
    parser.add_argument("--checkpoint_b", required=True)
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
    parser.add_argument("--alpha_grid", default="0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1")
    parser.add_argument("--proto_grid", default="0,0.05,0.1,0.15,0.2,0.3,0.4,0.5")
    parser.add_argument("--cluster_grid", default="0,0.05,0.1,0.15,0.2,0.3,0.4")
    parser.add_argument("--temperature_grid", default="0.8,1.0,1.2")
    args = parser.parse_args()

    seed_everything(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    clip_state_dict = patch_state_dict_text_projection(
        load_clip_state_dict(args.clip_model),
        fallback_embed_dim=512,
    )

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

    print("[ensemble] Loading expert A...")
    model_a = load_model(args.checkpoint_a, clip_state_dict, args.pubmedbert_path, device)
    val_a = collect_logits(model_a, val_loader, device, use_hflip=args.hflip)
    test_a = collect_logits(model_a, test_loader, device, use_hflip=args.hflip)
    del model_a
    torch.cuda.empty_cache()

    print("[ensemble] Loading expert B...")
    model_b = load_model(args.checkpoint_b, clip_state_dict, args.pubmedbert_path, device)
    val_b = collect_logits(model_b, val_loader, device, use_hflip=args.hflip)
    test_b = collect_logits(model_b, test_loader, device, use_hflip=args.hflip)
    del model_b
    torch.cuda.empty_cache()

    alpha_grid = [float(x) for x in args.alpha_grid.split(",") if x.strip()]
    proto_grid = [float(x) for x in args.proto_grid.split(",") if x.strip()]
    cluster_grid = [float(x) for x in args.cluster_grid.split(",") if x.strip()]
    temperature_grid = [float(x) for x in args.temperature_grid.split(",") if x.strip()]

    best = None
    alpha_results = {}
    for alpha in alpha_grid:
        val_pack = mix_packs(val_a, val_b, alpha)
        selected = select_weights(val_pack, proto_grid, cluster_grid, temperature_grid, args.objective)
        val_logits = combine_logits(
            val_pack,
            selected["proto_weight"],
            selected["cluster_weight"],
            selected["temperature"],
        )
        val_metrics = metrics_from_logits(val_logits, val_pack["labels"])
        item = {
            "alpha_a": float(alpha),
            "alpha_b": float(1.0 - alpha),
            "selected": selected,
            "val_metrics": val_metrics,
        }
        alpha_results[str(alpha)] = item
        if best is None or metric_tuple(val_metrics, args.objective) > metric_tuple(best["val_metrics"], args.objective):
            best = item

    test_pack = mix_packs(test_a, test_b, best["alpha_a"])
    test_logits = combine_logits(
        test_pack,
        best["selected"]["proto_weight"],
        best["selected"]["cluster_weight"],
        best["selected"]["temperature"],
    )
    test_metrics = metrics_from_logits(test_logits, test_pack["labels"])
    out = {
        "data_root": args.data_root,
        "checkpoint_a": args.checkpoint_a,
        "checkpoint_b": args.checkpoint_b,
        "objective": args.objective,
        "hflip_tta": bool(args.hflip),
        "alpha_results": alpha_results,
        "selected_ensemble": best,
        "test_metrics": test_metrics,
        "test_predictions": prediction_payload(test_logits, test_pack["labels"]),
    }

    os.makedirs(args.output_dir, exist_ok=True)
    with open(os.path.join(args.output_dir, "calibrated_metrics.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)

    print(json.dumps({
        "output_dir": args.output_dir,
        "selected_ensemble": best,
        "test_metrics": test_metrics,
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
