import argparse
import itertools
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
sys.path.insert(0, str(PROJECT_ROOT))

from main import load_clip_state_dict, patch_state_dict_text_projection, unwrap_model
from clip.model import build_fusion_model
from data.dataset import ClassificationDataset, create_dataloader, get_transforms, seed_everything


def get_batch_image_label(batch, device):
    image = batch["image"].to(device, non_blocking=True)
    label = batch["label"].to(device, non_blocking=True)
    return image, label


@torch.no_grad()
def collect_logits(model, loader, device, use_hflip=False, return_view_logits=False):
    model.eval()
    cls_logits, proto_logits, cluster_logits, visual_cluster_logits, patch_logits, labels = [], [], [], [], [], []
    cls_orig_logits, proto_orig_logits, cluster_orig_logits = [], [], []
    cls_flip_logits, proto_flip_logits, cluster_flip_logits = [], [], []
    has_visual_cluster = False
    has_patch = False

    for batch in loader:
        images, y = get_batch_image_label(batch, device)
        views = [images]
        if use_hflip:
            views.append(torch.flip(images, dims=[3]))

        view_outputs = []
        for view in views:
            out = model(view, labels=None)
            vc_logits = out.get("visual_cluster_logits")
            pt_logits = out.get("patch_logits")
            view_outputs.append((
                out["logits_classifier"].float(),
                out["logits_proto"].float(),
                out["cluster_logits"].float(),
                vc_logits.float() if vc_logits is not None else None,
                pt_logits.float() if pt_logits is not None else None,
            ))

        cls = torch.stack([x[0] for x in view_outputs], dim=0).mean(dim=0)
        proto = torch.stack([x[1] for x in view_outputs], dim=0).mean(dim=0)
        cluster = torch.stack([x[2] for x in view_outputs], dim=0).mean(dim=0)
        if return_view_logits and len(view_outputs) > 1:
            cls_orig_logits.append(view_outputs[0][0].cpu())
            proto_orig_logits.append(view_outputs[0][1].cpu())
            cluster_orig_logits.append(view_outputs[0][2].cpu())
            cls_flip_logits.append(view_outputs[1][0].cpu())
            proto_flip_logits.append(view_outputs[1][1].cpu())
            cluster_flip_logits.append(view_outputs[1][2].cpu())
        visual_items = [x[3] for x in view_outputs if x[3] is not None]
        if visual_items:
            visual_cluster = torch.stack(visual_items, dim=0).mean(dim=0)
            visual_cluster_logits.append(visual_cluster.cpu())
            has_visual_cluster = True
        patch_items = [x[4] for x in view_outputs if x[4] is not None]
        if patch_items:
            patch = torch.stack(patch_items, dim=0).mean(dim=0)
            patch_logits.append(patch.cpu())
            has_patch = True

        cls_logits.append(cls.cpu())
        proto_logits.append(proto.cpu())
        cluster_logits.append(cluster.cpu())
        labels.append(y.cpu())

    pack = {
        "cls": torch.cat(cls_logits, dim=0),
        "proto": torch.cat(proto_logits, dim=0),
        "cluster": torch.cat(cluster_logits, dim=0),
        "labels": torch.cat(labels, dim=0),
    }
    if has_visual_cluster:
        pack["visual_cluster"] = torch.cat(visual_cluster_logits, dim=0)
    if has_patch:
        pack["patch"] = torch.cat(patch_logits, dim=0)
    if return_view_logits and cls_orig_logits:
        pack.update({
            "cls_orig": torch.cat(cls_orig_logits, dim=0),
            "proto_orig": torch.cat(proto_orig_logits, dim=0),
            "cluster_orig": torch.cat(cluster_orig_logits, dim=0),
            "cls_flip": torch.cat(cls_flip_logits, dim=0),
            "proto_flip": torch.cat(proto_flip_logits, dim=0),
            "cluster_flip": torch.cat(cluster_flip_logits, dim=0),
        })
    return pack


def metrics_from_logits(logits, labels):
    y_true = labels.numpy()
    y_pred = logits.argmax(dim=-1).numpy()
    return {
        "accuracy": float(np.mean(y_pred == y_true)),
        "macro_precision": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "macro_recall": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "mae": float(np.mean(np.abs(y_pred - y_true))),
    }


def prediction_payload(logits, labels):
    y_true = labels.numpy()
    y_pred = logits.argmax(dim=-1).numpy()
    return {
        "labels": y_true.tolist(),
        "preds": y_pred.tolist(),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
    }


def combine_logits(pack, proto_w, cluster_w, temperature, visual_cluster_w=0.0, patch_w=0.0):
    logits = pack["cls"] + proto_w * pack["proto"] + cluster_w * pack["cluster"]
    if "visual_cluster" in pack:
        logits = logits + visual_cluster_w * pack["visual_cluster"]
    if "patch" in pack:
        logits = logits + patch_w * pack["patch"]
    return logits / max(float(temperature), 1e-6)


def select_weights(val_pack, proto_grid, cluster_grid, temperature_grid, objective, visual_cluster_grid=None, patch_grid=None):
    best = None
    if visual_cluster_grid is None:
        visual_cluster_grid = [0.0]
    if "visual_cluster" not in val_pack:
        visual_cluster_grid = [0.0]
    if patch_grid is None:
        patch_grid = [0.0]
    if "patch" not in val_pack:
        patch_grid = [0.0]
    for proto_w, cluster_w, visual_w, patch_w, temp in itertools.product(
        proto_grid, cluster_grid, visual_cluster_grid, patch_grid, temperature_grid
    ):
        logits = combine_logits(val_pack, proto_w, cluster_w, temp, visual_w, patch_w)
        m = metrics_from_logits(logits, val_pack["labels"])
        score = m[objective]
        # Accuracy tie-breaker keeps the selected point useful for conventional tables.
        tie = (m["accuracy"], m["macro_f1"], -m["mae"])
        item = {
            "proto_weight": float(proto_w),
            "cluster_weight": float(cluster_w),
            "visual_cluster_weight": float(visual_w),
            "patch_weight": float(patch_w),
            "temperature": float(temp),
            "score": float(score),
            "metrics": m,
            "tie": tie,
        }
        if best is None or (score, tie) > (best["score"], best["tie"]):
            best = item
    best.pop("tie", None)
    return best


def score_tuple(metrics, objective):
    return (
        metrics[objective],
        metrics["accuracy"],
        metrics["macro_f1"],
        -metrics["mae"],
    )


def search_class_bias(val_logits, labels, objective):
    num_classes = val_logits.shape[1]
    bias = torch.zeros(num_classes)
    best_metrics = metrics_from_logits(val_logits, labels)
    grid = torch.linspace(-0.8, 0.8, steps=33)
    for _ in range(3):
        improved = False
        for c in range(num_classes):
            local_bias = bias.clone()
            local_metrics = best_metrics
            for value in grid:
                candidate = bias.clone()
                candidate[c] = value
                candidate = candidate - candidate.mean()
                metrics = metrics_from_logits(val_logits + candidate, labels)
                if score_tuple(metrics, objective) > score_tuple(local_metrics, objective):
                    local_bias = candidate
                    local_metrics = metrics
            if not torch.equal(local_bias, bias):
                bias = local_bias
                best_metrics = local_metrics
                improved = True
        if not improved:
            break
    return bias, best_metrics


def main():
    parser = argparse.ArgumentParser(description="Validation-calibrated multimodal logit evaluation.")
    parser.add_argument("--data_root", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--clip_model", default="ViT-B/32")
    parser.add_argument("--pubmedbert_path", default=str(PROJECT_ROOT / "PubMedBERT"))
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--img_size", type=int, default=224)
    parser.add_argument("--backbone", type=str, default="openai_clip",
                        choices=["openai_clip", "biomedclip"])
    parser.add_argument("--clip_normalize", dest="clip_normalize",
                        action="store_true", default=None)
    parser.add_argument("--no_clip_normalize", dest="clip_normalize",
                        action="store_false")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--objective", default="macro_f1",
                        choices=["accuracy", "macro_f1", "macro_recall"])
    parser.add_argument("--hflip", action="store_true")
    parser.add_argument("--enable_visual_cluster", action="store_true",
                        help="Build the full_vcf model with visual KL cluster prototypes.")
    parser.add_argument("--enable_semantic_patch", action="store_true",
                        help="Build the full_tpa model with text-guided semantic patch attention.")
    parser.add_argument("--proto_grid", default="0,0.05,0.1,0.15,0.2,0.3,0.4,0.5")
    parser.add_argument("--cluster_grid", default="0,0.05,0.1,0.15,0.2,0.3,0.4")
    parser.add_argument("--temperature_grid", default="0.8,1.0,1.2")
    parser.add_argument("--visual_cluster_grid", default="0,0.05,0.1,0.2,0.3,0.4")
    parser.add_argument("--patch_grid", default="0,0.05,0.1,0.2,0.3,0.4")
    parser.add_argument("--class_bias", action="store_true",
                        help="Search a validation-selected zero-mean per-class logit bias.")
    parser.add_argument("--calib_split", default="val", choices=["val", "trainval"],
                        help="Split(s) used for selecting logit weights.")
    args = parser.parse_args()

    seed_everything(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    state_dict = patch_state_dict_text_projection(
        load_clip_state_dict(args.clip_model),
        fallback_embed_dim=512,
    )
    if args.clip_normalize is None:
        args.clip_normalize = (args.backbone == "biomedclip")
    model = build_fusion_model(
        state_dict=state_dict,
        pubmedbert_path=args.pubmedbert_path,
        backbone=args.backbone,
        img_size=args.img_size,
        freeze_text_encoder=True,
        enable_visual_cluster=args.enable_visual_cluster,
        enable_semantic_patch=args.enable_semantic_patch,
    )
    missing, unexpected = model.load_state_dict(
        torch.load(args.checkpoint, map_location="cpu"),
        strict=False,
    )
    if missing:
        print(f"[calib] Missing keys while loading checkpoint: {missing[:8]}")
    if unexpected:
        print(f"[calib] Unexpected keys while loading checkpoint: {unexpected[:8]}")
    model = model.to(device)
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
    base = unwrap_model(model)
    if hasattr(base, "refresh_text_prototypes"):
        base.refresh_text_prototypes()

    val_tf = get_transforms(
        img_size=args.img_size,
        is_training=False,
        use_clahe=True,
        to_rgb=True,
        clahe_clip_limit=2.0,
        clahe_tile_grid_size=(8, 8),
        percentile=(1.0, 99.0),
        normalize=args.clip_normalize,
    )
    val_ds = ClassificationDataset(args.data_root, split="val", transform=val_tf,
                                   error_policy="raise", include_path=False)
    train_ds = None
    if args.calib_split == "trainval":
        train_ds = ClassificationDataset(args.data_root, split="train", transform=val_tf,
                                         error_policy="raise", include_path=False)
    test_ds = ClassificationDataset(args.data_root, split="test", transform=val_tf,
                                    error_policy="raise", include_path=False)
    val_loader = create_dataloader(val_ds, batch_size=args.batch_size, shuffle=False,
                                   num_workers=args.num_workers)
    train_loader = None
    if train_ds is not None:
        train_loader = create_dataloader(train_ds, batch_size=args.batch_size, shuffle=False,
                                         num_workers=args.num_workers)
    test_loader = create_dataloader(test_ds, batch_size=args.batch_size, shuffle=False,
                                    num_workers=args.num_workers)

    if train_loader is not None:
        print("[calib] Collecting train logits for calibration...")
        train_pack = collect_logits(model, train_loader, device, use_hflip=args.hflip)
    else:
        train_pack = None
    print("[calib] Collecting validation logits...")
    val_pack = collect_logits(model, val_loader, device, use_hflip=args.hflip)
    print("[calib] Collecting test logits...")
    test_pack = collect_logits(model, test_loader, device, use_hflip=args.hflip)

    calib_pack = val_pack
    if train_pack is not None:
        calib_pack = {}
        for key in val_pack:
            if torch.is_tensor(val_pack[key]) and key in train_pack:
                calib_pack[key] = torch.cat([train_pack[key], val_pack[key]], dim=0)
            else:
                calib_pack[key] = val_pack[key]

    proto_grid = [float(x) for x in args.proto_grid.split(",") if x.strip()]
    cluster_grid = [float(x) for x in args.cluster_grid.split(",") if x.strip()]
    temperature_grid = [float(x) for x in args.temperature_grid.split(",") if x.strip()]
    visual_cluster_grid = [float(x) for x in args.visual_cluster_grid.split(",") if x.strip()]
    patch_grid = [float(x) for x in args.patch_grid.split(",") if x.strip()]

    selected = select_weights(
        calib_pack, proto_grid, cluster_grid, temperature_grid, args.objective,
        visual_cluster_grid=visual_cluster_grid,
        patch_grid=patch_grid,
    )
    test_logits = combine_logits(
        test_pack,
        selected["proto_weight"],
        selected["cluster_weight"],
        selected["temperature"],
        selected.get("visual_cluster_weight", 0.0),
        selected.get("patch_weight", 0.0),
    )
    val_logits = combine_logits(
        calib_pack,
        selected["proto_weight"],
        selected["cluster_weight"],
        selected["temperature"],
        selected.get("visual_cluster_weight", 0.0),
        selected.get("patch_weight", 0.0),
    )

    bias = torch.zeros(test_logits.shape[1], dtype=test_logits.dtype)
    bias_val_metrics = None
    if args.class_bias:
        bias, bias_val_metrics = search_class_bias(val_logits, calib_pack["labels"], args.objective)
        test_logits = test_logits + bias.to(test_logits.dtype)
    test_metrics = metrics_from_logits(test_logits, test_pack["labels"])

    raw_metrics = metrics_from_logits(test_pack["cls"], test_pack["labels"])
    out = {
        "data_root": args.data_root,
        "checkpoint": args.checkpoint,
        "objective": args.objective,
        "hflip_tta": bool(args.hflip),
        "calib_split": args.calib_split,
        "selected": selected,
        "class_bias": [float(x) for x in bias.tolist()],
        "class_bias_val_metrics": bias_val_metrics,
        "raw_classifier_test_metrics": raw_metrics,
        "test_metrics": test_metrics,
        "test_predictions": prediction_payload(test_logits, test_pack["labels"]),
    }

    os.makedirs(args.output_dir, exist_ok=True)
    with open(os.path.join(args.output_dir, "calibrated_metrics.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)

    print(json.dumps({
        "output_dir": args.output_dir,
        "objective": args.objective,
        "hflip_tta": bool(args.hflip),
        "calib_split": args.calib_split,
        "selected": selected,
        "class_bias": [float(x) for x in bias.tolist()],
        "class_bias_val_metrics": bias_val_metrics,
        "test_metrics": test_metrics,
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
