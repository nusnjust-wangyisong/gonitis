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

from calibrated_eval import combine_logits, metrics_from_logits, prediction_payload, select_weights
from main import load_clip_state_dict, patch_state_dict_text_projection, unwrap_model
from clip.model import build_fusion_model
from data.dataset import ClassificationDataset, create_dataloader, get_transforms, seed_everything


VIEW_NAMES = ["orig", "hflip", "vflip", "rot180", "rot90", "rot270", "hflip_rot90", "hflip_rot270"]
POLICIES = {
    "none": ["orig"],
    "hflip": ["orig", "hflip"],
    "hv180": ["orig", "hflip", "vflip", "rot180"],
    "rot4": ["orig", "rot90", "rot180", "rot270"],
    "d4": VIEW_NAMES,
}


def apply_view(images: torch.Tensor, view: str) -> torch.Tensor:
    if view == "orig":
        return images
    if view == "hflip":
        return torch.flip(images, dims=[3])
    if view == "vflip":
        return torch.flip(images, dims=[2])
    if view == "rot90":
        return torch.rot90(images, k=1, dims=[2, 3])
    if view == "rot180":
        return torch.rot90(images, k=2, dims=[2, 3])
    if view == "rot270":
        return torch.rot90(images, k=3, dims=[2, 3])
    if view == "hflip_rot90":
        return torch.flip(torch.rot90(images, k=1, dims=[2, 3]), dims=[3])
    if view == "hflip_rot270":
        return torch.flip(torch.rot90(images, k=3, dims=[2, 3]), dims=[3])
    raise ValueError(f"Unknown view: {view}")


@torch.no_grad()
def collect_policy_logits(model, loader, device):
    model.eval()
    packs = {
        name: {"cls": [], "proto": [], "cluster": [], "labels": []}
        for name in VIEW_NAMES
    }

    for batch in loader:
        images = batch["image"].to(device, non_blocking=True)
        labels = batch["label"].to(device, non_blocking=True)
        for view_name in VIEW_NAMES:
            out = model(apply_view(images, view_name), labels=None)
            packs[view_name]["cls"].append(out["logits_classifier"].float().cpu())
            packs[view_name]["proto"].append(out["logits_proto"].float().cpu())
            packs[view_name]["cluster"].append(out["cluster_logits"].float().cpu())
            packs[view_name]["labels"].append(labels.cpu())

    out = {}
    for view_name, pack in packs.items():
        out[view_name] = {
            "cls": torch.cat(pack["cls"], dim=0),
            "proto": torch.cat(pack["proto"], dim=0),
            "cluster": torch.cat(pack["cluster"], dim=0),
            "labels": torch.cat(pack["labels"], dim=0),
        }
    return out


def average_policy(view_packs, policy_name):
    views = POLICIES[policy_name]
    labels = view_packs[views[0]]["labels"]
    return {
        "cls": torch.stack([view_packs[v]["cls"] for v in views], dim=0).mean(dim=0),
        "proto": torch.stack([view_packs[v]["proto"] for v in views], dim=0).mean(dim=0),
        "cluster": torch.stack([view_packs[v]["cluster"] for v in views], dim=0).mean(dim=0),
        "labels": labels,
    }


def metric_tuple(metrics, objective):
    return (
        metrics[objective],
        metrics["accuracy"],
        metrics["macro_f1"],
        -metrics["mae"],
    )


def main():
    parser = argparse.ArgumentParser(description="Validation-selected geometric TTA policy evaluation.")
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

    print("[tta] Collecting validation view logits...")
    val_views = collect_policy_logits(model, val_loader, device)
    print("[tta] Collecting test view logits...")
    test_views = collect_policy_logits(model, test_loader, device)

    proto_grid = [float(x) for x in args.proto_grid.split(",") if x.strip()]
    cluster_grid = [float(x) for x in args.cluster_grid.split(",") if x.strip()]
    temperature_grid = [float(x) for x in args.temperature_grid.split(",") if x.strip()]

    policy_results = {}
    best = None
    for policy_name in POLICIES:
        val_pack = average_policy(val_views, policy_name)
        selected = select_weights(val_pack, proto_grid, cluster_grid, temperature_grid, args.objective)
        val_logits = combine_logits(
            val_pack,
            selected["proto_weight"],
            selected["cluster_weight"],
            selected["temperature"],
        )
        val_metrics = metrics_from_logits(val_logits, val_pack["labels"])
        item = {
            "policy": policy_name,
            "views": POLICIES[policy_name],
            "selected": selected,
            "val_metrics": val_metrics,
        }
        policy_results[policy_name] = item
        if best is None or metric_tuple(val_metrics, args.objective) > metric_tuple(best["val_metrics"], args.objective):
            best = item

    for policy_name, item in policy_results.items():
        eval_pack = average_policy(test_views, policy_name)
        eval_logits = combine_logits(
            eval_pack,
            item["selected"]["proto_weight"],
            item["selected"]["cluster_weight"],
            item["selected"]["temperature"],
        )
        item["test_metrics"] = metrics_from_logits(eval_logits, eval_pack["labels"])

    test_pack = average_policy(test_views, best["policy"])
    test_logits = combine_logits(
        test_pack,
        best["selected"]["proto_weight"],
        best["selected"]["cluster_weight"],
        best["selected"]["temperature"],
    )
    test_metrics = metrics_from_logits(test_logits, test_pack["labels"])

    out = {
        "data_root": args.data_root,
        "checkpoint": args.checkpoint,
        "objective": args.objective,
        "policy_results": policy_results,
        "selected_policy": best,
        "test_metrics": test_metrics,
        "test_predictions": prediction_payload(test_logits, test_pack["labels"]),
    }

    os.makedirs(args.output_dir, exist_ok=True)
    with open(os.path.join(args.output_dir, "calibrated_metrics.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)

    print(json.dumps({
        "output_dir": args.output_dir,
        "selected_policy": best,
        "test_metrics": test_metrics,
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
