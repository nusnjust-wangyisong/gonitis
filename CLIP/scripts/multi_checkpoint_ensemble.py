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
from clip.ablation_model import build_ablation_model
from data.dataset import ClassificationDataset, create_dataloader, get_transforms, seed_everything


def parse_model_spec(spec: str):
    if "=" not in spec:
        raise ValueError(f"Model spec must be variant=checkpoint, got: {spec}")
    variant, checkpoint = spec.split("=", 1)
    return variant.strip(), checkpoint.strip()


def build_model_for_variant(variant, clip_state_dict, pubmedbert_path):
    if variant in {"full", "full_ldl"}:
        return build_fusion_model(
            state_dict=clip_state_dict,
            pubmedbert_path=pubmedbert_path,
            freeze_text_encoder=True,
        )
    if variant == "full_adapter":
        return build_fusion_model(
            state_dict=clip_state_dict,
            pubmedbert_path=pubmedbert_path,
            freeze_text_encoder=True,
            enable_feature_adapter=True,
        )
    if variant == "full_obh":
        return build_fusion_model(
            state_dict=clip_state_dict,
            pubmedbert_path=pubmedbert_path,
            freeze_text_encoder=True,
            enable_ordinal_boundary=True,
        )
    if variant == "full_vcf":
        return build_fusion_model(
            state_dict=clip_state_dict,
            pubmedbert_path=pubmedbert_path,
            freeze_text_encoder=True,
            enable_visual_cluster=True,
        )
    return build_ablation_model(
        variant=variant,
        state_dict=clip_state_dict,
        pubmedbert_path=pubmedbert_path,
        freeze_text_encoder=True,
    )


@torch.no_grad()
def collect_logits(model, loader, device, hflip=False):
    model.eval()
    logits_list, labels_list = [], []
    for batch in loader:
        images = batch["image"].to(device, non_blocking=True)
        labels = batch["label"].to(device, non_blocking=True)
        views = [images]
        if hflip:
            views.append(torch.flip(images, dims=[3]))
        view_logits = []
        for view in views:
            out = model(view, labels=None)
            view_logits.append(out["logits_cls"].float())
        logits = torch.stack(view_logits, dim=0).mean(dim=0)
        logits_list.append(logits.cpu())
        labels_list.append(labels.cpu())
    return torch.cat(logits_list, dim=0), torch.cat(labels_list, dim=0)


def metrics_from_logits(logits, labels):
    y_true = labels.numpy()
    y_pred = logits.argmax(dim=-1).numpy()
    return {
        "accuracy": float(np.mean(y_pred == y_true)),
        "macro_precision": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "macro_recall": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "mae": float(np.mean(np.abs(y_pred - y_true))),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
    }


def normalize_logits(logits, mode):
    logits = logits.float()
    if mode == "none":
        return logits
    if mode == "zscore":
        mean = logits.mean(dim=1, keepdim=True)
        std = logits.std(dim=1, keepdim=True).clamp(min=1e-6)
        return (logits - mean) / std
    if mode == "prob":
        return torch.softmax(logits, dim=1)
    raise ValueError(f"Unknown normalize mode: {mode}")


def simplex_grid(num_models, step):
    units = int(round(1.0 / step))
    for counts in itertools.product(range(units + 1), repeat=num_models):
        if sum(counts) == units:
            yield torch.tensor([c / units for c in counts], dtype=torch.float32)


def combine_stack(stack, weights):
    return (stack * weights.view(-1, 1, 1)).sum(dim=0)


def score_tuple(metrics, objective):
    return (
        metrics[objective],
        metrics["accuracy"],
        metrics["macro_f1"],
        -metrics["mae"],
    )


def search_weights(val_stack, labels, objective, step):
    best = None
    for weights in simplex_grid(val_stack.shape[0], step):
        logits = combine_stack(val_stack, weights)
        metrics = metrics_from_logits(logits, labels)
        item = {
            "weights": [float(x) for x in weights.tolist()],
            "metrics": metrics,
        }
        if best is None or score_tuple(metrics, objective) > score_tuple(best["metrics"], objective):
            best = item
    return best


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
    return [float(x) for x in bias.tolist()], best_metrics


def main():
    parser = argparse.ArgumentParser(description="Validation-selected ensemble over multiple KL grading checkpoints.")
    parser.add_argument("--data_root", required=True)
    parser.add_argument("--models", nargs="+", required=True,
                        help="Model specs as variant=checkpoint_path")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--clip_model", default="ViT-B/32")
    parser.add_argument("--pubmedbert_path", default=str(PROJECT_ROOT / "PubMedBERT"))
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--img_size", type=int, default=224)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--hflip", action="store_true")
    parser.add_argument("--objective", default="macro_f1",
                        choices=["accuracy", "macro_f1", "macro_recall"])
    parser.add_argument("--step", type=float, default=0.05)
    parser.add_argument("--normalize", default="zscore", choices=["none", "zscore", "prob"])
    parser.add_argument("--class_bias", action="store_true")
    args = parser.parse_args()

    seed_everything(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    state_dict = patch_state_dict_text_projection(
        load_clip_state_dict(args.clip_model), fallback_embed_dim=512
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

    names, val_logits_all, test_logits_all = [], [], []
    labels_val = labels_test = None
    for spec in args.models:
        variant, checkpoint = parse_model_spec(spec)
        print(f"[ensemble] Loading {variant}: {checkpoint}")
        model = build_model_for_variant(variant, state_dict, args.pubmedbert_path)
        model.load_state_dict(torch.load(checkpoint, map_location="cpu"), strict=True)
        model = model.to(device)
        if torch.cuda.device_count() > 1:
            model = nn.DataParallel(model)
        base = unwrap_model(model)
        if hasattr(base, "refresh_text_prototypes"):
            base.refresh_text_prototypes()

        val_logits, labels_val = collect_logits(model, val_loader, device, hflip=args.hflip)
        test_logits, labels_test = collect_logits(model, test_loader, device, hflip=args.hflip)
        val_logits_all.append(normalize_logits(val_logits, args.normalize))
        test_logits_all.append(normalize_logits(test_logits, args.normalize))
        names.append(variant)
        del model
        torch.cuda.empty_cache()

    val_stack = torch.stack(val_logits_all, dim=0)
    test_stack = torch.stack(test_logits_all, dim=0)
    best = search_weights(val_stack, labels_val, args.objective, args.step)

    val_combined = combine_stack(val_stack, torch.tensor(best["weights"]))
    test_combined = combine_stack(test_stack, torch.tensor(best["weights"]))
    bias = [0.0] * val_combined.shape[1]
    bias_val_metrics = None
    if args.class_bias:
        bias, bias_val_metrics = search_class_bias(val_combined, labels_val, args.objective)
        bias_tensor = torch.tensor(bias, dtype=test_combined.dtype)
        test_combined = test_combined + bias_tensor

    test_metrics = metrics_from_logits(test_combined, labels_test)
    out = {
        "models": names,
        "model_specs": args.models,
        "normalize": args.normalize,
        "hflip_tta": bool(args.hflip),
        "objective": args.objective,
        "step": args.step,
        "selected": best,
        "class_bias": bias,
        "class_bias_val_metrics": bias_val_metrics,
        "test_metrics": test_metrics,
    }
    os.makedirs(args.output_dir, exist_ok=True)
    with open(os.path.join(args.output_dir, "ensemble_metrics.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(json.dumps(out, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
