#!/usr/bin/env python3
"""
Train a single-image classification baseline backbone for KL grading.

This script is for paper comparison experiments, not for the proposed Ours
model. It keeps the data pipeline, metrics, and output format consistent
across ResNet / MobileNetV3 / ViT / DenseNet / PVTv2 / ConvNeXt-V2 /
EfficientNet / official Spatial-Mamba baselines.
"""
import argparse
import csv
import importlib
import json
import os
import sys
from dataclasses import asdict, dataclass
from typing import Dict, Iterable, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from sklearn.metrics import f1_score, precision_score, recall_score

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)
from data.dataset import ClassificationDataset, create_dataloader, get_transforms, seed_everything  # noqa: E402


DATASETS = {
    "me": "/home/kaixin/code/gonitis_dataset/公开数据集/MedicalExpert-split",
    "ar": "/home/kaixin/code/gonitis_dataset/公开数据集/archive",
    "pv": "/home/kaixin/code/gonitis_dataset/私有数据集/split_result_siyouceshi_roi",
}

MODEL_ALIASES = {
    "resnet": "resnet50.a1_in1k",
    "resnet50": "resnet50.a1_in1k",
    "mobilenet_v3": "mobilenetv3_large_100.ra_in1k",
    "mobilenetv3": "mobilenetv3_large_100.ra_in1k",
    "vit": "vit_base_patch16_224.augreg_in21k_ft_in1k",
    "densenet": "densenet121.ra_in1k",
    "pvt_v2": "pvt_v2_b2.in1k",
    "pvtv2": "pvt_v2_b2.in1k",
    "convnext_v2": "convnextv2_base.fcmae_ft_in22k_in1k",
    "convnextv2": "convnextv2_base.fcmae_ft_in22k_in1k",
    "efficientnet": "efficientnet_b0.ra_in1k",
    "spatialmamba": "official_spatialmamba_tiny",
    "mambaout": "mambaout_base.in1k",
}

SPATIAL_MAMBA_ROOT = os.path.join(PROJECT_ROOT, "third_party", "Spatial-Mamba", "classification")
SPATIAL_MAMBA_PRETRAINED = os.path.join(PROJECT_ROOT, "experiments", "pretrained", "spatialmamba_tiny_in1k.pth")


@dataclass
class MetricBundle:
    accuracy: float
    macro_f1: float
    mae: float
    macro_precision: float
    macro_recall: float
    per_class_recall: Dict[str, float]
    per_class_f1: Dict[str, float]


def resolve_model_name(name: str) -> str:
    return MODEL_ALIASES.get(name.lower(), name)


def resolve_data_root(dataset: str, data_root: str) -> str:
    if data_root:
        return data_root
    if dataset not in DATASETS:
        raise ValueError(f"Unknown dataset '{dataset}'. Use one of {sorted(DATASETS)} or pass --data_root.")
    return DATASETS[dataset]


def build_loaders(data_root: str, img_size: int, batch_size: int, num_workers: int):
    train_tf = get_transforms(
        img_size=img_size,
        is_training=True,
        use_clahe=True,
        to_rgb=True,
        clahe_clip_limit=2.0,
        clahe_tile_grid_size=(8, 8),
        percentile=(1.0, 99.0),
        rotate_deg=10,
        hflip_p=0.5,
        random_erasing_p=0.0,
        normalize="imagenet",
    )
    eval_tf = get_transforms(
        img_size=img_size,
        is_training=False,
        use_clahe=True,
        to_rgb=True,
        clahe_clip_limit=2.0,
        clahe_tile_grid_size=(8, 8),
        percentile=(1.0, 99.0),
        normalize="imagenet",
    )

    loaders = {}
    for split, tf, shuffle in (("train", train_tf, True), ("val", eval_tf, False), ("test", eval_tf, False)):
        ds = ClassificationDataset(data_root, split=split, transform=tf, error_policy="raise", include_path=True)
        loaders[split] = create_dataloader(ds, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
        loaders[f"{split}_ds"] = ds
    return loaders


def build_model(model_name: str, num_classes: int) -> nn.Module:
    if model_name.lower() == "spatialmamba":
        return build_official_spatial_mamba(num_classes)

    resolved = resolve_model_name(model_name)
    return timm.create_model(resolved, pretrained=True, num_classes=num_classes)


def _torch_load_trusted(path: str):
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def build_official_spatial_mamba(num_classes: int) -> nn.Module:
    if not os.path.exists(SPATIAL_MAMBA_ROOT):
        raise FileNotFoundError(
            f"Official Spatial-Mamba code not found: {SPATIAL_MAMBA_ROOT}. "
            "Clone https://github.com/EdwardChasel/Spatial-Mamba.git into CLIP/third_party first."
        )
    if SPATIAL_MAMBA_ROOT not in sys.path:
        sys.path.insert(0, SPATIAL_MAMBA_ROOT)

    spatial_module = importlib.import_module("models.spatialmamba")
    model = spatial_module.SpatialMamba(
        num_classes=num_classes,
        depths=[2, 4, 8, 4],
        dims=64,
        d_state=1,
        mlp_ratio=4.0,
        dt_init="random",
        drop_path_rate=0.2,
    )

    if os.path.exists(SPATIAL_MAMBA_PRETRAINED):
        ckpt = _torch_load_trusted(SPATIAL_MAMBA_PRETRAINED)
        state = ckpt.get("model_ema") or ckpt.get("model") or ckpt
        state = {k: v for k, v in state.items() if not k.startswith("head.")}
        msg = model.load_state_dict(state, strict=False)
        print(f"[spatialmamba] loaded official ImageNet-1K Tiny weights: {SPATIAL_MAMBA_PRETRAINED}")
        print(f"[spatialmamba] missing={len(msg.missing_keys)} unexpected={len(msg.unexpected_keys)}")
    else:
        print(f"[spatialmamba] warning: official checkpoint not found: {SPATIAL_MAMBA_PRETRAINED}")
    return model


def is_classifier_param(name: str) -> bool:
    head_prefixes = ("head.", "classifier.", "fc.", "last_linear.")
    head_tokens = (".head.", ".classifier.", ".fc.", ".last_linear.")
    return name.startswith(head_prefixes) or any(tok in name for tok in head_tokens)


def split_params(model: nn.Module) -> Tuple[List[nn.Parameter], List[nn.Parameter]]:
    head, body = [], []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if is_classifier_param(name):
            head.append(param)
        else:
            body.append(param)
    if not head:
        # Some timm models expose classifier parameters under non-standard names.
        # Falling back to one LR is safer than silently dropping parameters.
        return [], [p for p in model.parameters() if p.requires_grad]
    return head, body


def metric_mode_value(metrics: MetricBundle, monitor: str) -> float:
    if monitor == "accuracy":
        return metrics.accuracy
    if monitor == "macro_f1":
        return metrics.macro_f1
    if monitor == "mae":
        return -metrics.mae
    raise ValueError(monitor)


def compute_metrics(probs: np.ndarray, labels: np.ndarray) -> MetricBundle:
    preds = probs.argmax(1)
    recall = recall_score(labels, preds, average=None, labels=[0, 1, 2, 3, 4], zero_division=0)
    f1 = f1_score(labels, preds, average=None, labels=[0, 1, 2, 3, 4], zero_division=0)
    return MetricBundle(
        accuracy=float((preds == labels).mean()),
        macro_f1=float(f1_score(labels, preds, average="macro", zero_division=0)),
        mae=float(np.abs(preds - labels).mean()),
        macro_precision=float(precision_score(labels, preds, average="macro", zero_division=0)),
        macro_recall=float(recall_score(labels, preds, average="macro", zero_division=0)),
        per_class_recall={str(k): float(recall[k]) for k in range(5)},
        per_class_f1={str(k): float(f1[k]) for k in range(5)},
    )


@torch.no_grad()
def eval_probs(model: nn.Module, loader, device: torch.device, hflip: bool) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    model.eval()
    probs, labels, paths = [], [], []
    for batch in loader:
        x = batch["image"].to(device, non_blocking=True)
        logits = model(x)
        if hflip:
            logits = (logits + model(torch.flip(x, dims=[3]))) / 2
        probs.append(F.softmax(logits.float(), dim=-1).cpu().numpy())
        labels.append(batch["label"].numpy())
        paths.extend(batch.get("path", [""] * x.shape[0]))
    return np.concatenate(probs), np.concatenate(labels), paths


def write_predictions(path: str, probs: np.ndarray, labels: np.ndarray, paths: Iterable[str]) -> None:
    preds = probs.argmax(1)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["path", "label", "pred", "prob_0", "prob_1", "prob_2", "prob_3", "prob_4"])
        for img_path, label, pred, prob in zip(paths, labels, preds, probs):
            writer.writerow([img_path, int(label), int(pred)] + [f"{float(v):.8f}" for v in prob])


def append_log(path: str, row: Dict[str, object]) -> None:
    exists = os.path.exists(path)
    with open(path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=sorted(DATASETS), default="me")
    parser.add_argument("--data_root", default="")
    parser.add_argument("--model", required=True, help="Alias or timm model name.")
    parser.add_argument("--img_size", type=int, default=224)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--head_lr", type=float, default=5e-4)
    parser.add_argument("--weight_decay", type=float, default=0.05)
    parser.add_argument("--label_smoothing", type=float, default=0.05)
    parser.add_argument("--use_class_weights", action="store_true",
                        help="Use train split class weights in cross entropy.")
    parser.add_argument("--class_weight_power", type=float, default=0.5,
                        help="Exponent for class weights. 0.5 is a mild reweighting.")
    parser.add_argument("--monitor", choices=["accuracy", "macro_f1", "mae"], default="macro_f1")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--hflip", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--run_tag", default="",
                        help="Optional suffix for hyperparameter-search runs.")
    parser.add_argument("--out_root", default="experiments/results/baseline_compare")
    parser.add_argument("--save_root", default="experiments/checkpoints/baseline_compare")
    args = parser.parse_args()

    seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_root = resolve_data_root(args.dataset, args.data_root)
    model_name = resolve_model_name(args.model)
    run_name = f"{args.dataset}_{args.model.lower()}_seed{args.seed}"
    if args.run_tag:
        run_name += f"_{args.run_tag}"
    out_dir = os.path.join(args.out_root, run_name)
    save_dir = os.path.join(args.save_root, run_name)
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(save_dir, exist_ok=True)

    print(f"[baseline] dataset={args.dataset} data_root={data_root}")
    print(f"[baseline] model={args.model} resolved={model_name} device={device}")
    loaders = build_loaders(data_root, args.img_size, args.batch_size, args.num_workers)
    model = build_model(args.model, num_classes=5).to(device)

    head_params, body_params = split_params(model)
    if head_params:
        optimizer = torch.optim.AdamW(
            [
                {"params": body_params, "lr": args.lr},
                {"params": head_params, "lr": args.head_lr},
            ],
            weight_decay=args.weight_decay,
        )
        print(f"[param-split] body={len(body_params)} head={len(head_params)} lr={args.lr} head_lr={args.head_lr}")
    else:
        optimizer = torch.optim.AdamW(body_params, lr=args.lr, weight_decay=args.weight_decay)
        print(f"[param-split] one group params={len(body_params)} lr={args.lr}")

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, args.epochs))
    class_weights = None
    if args.use_class_weights:
        class_weights = loaders["train_ds"].get_class_weights().to(device)
        class_weights = class_weights.pow(args.class_weight_power)
        class_weights = class_weights / class_weights.mean().clamp(min=1e-6)
        print(f"[class-weights] power={args.class_weight_power} weights={class_weights.detach().cpu().tolist()}")
    criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=args.label_smoothing)
    scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda"))

    config = vars(args).copy()
    config.update({"resolved_model": model_name, "data_root": data_root, "device": str(device)})
    with open(os.path.join(out_dir, "config.json"), "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)

    best_score = -1e9
    best_epoch = 0
    ckpt_path = os.path.join(save_dir, "best_model.pth")
    log_path = os.path.join(out_dir, "training_log.csv")

    for epoch in range(1, args.epochs + 1):
        model.train()
        losses = []
        for batch in loaders["train"]:
            x = batch["image"].to(device, non_blocking=True)
            y = batch["label"].to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
                loss = criterion(model(x), y)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            losses.append(float(loss.detach().cpu()))
        scheduler.step()

        val_probs, val_labels, _ = eval_probs(model, loaders["val"], device, args.hflip)
        val_metrics = compute_metrics(val_probs, val_labels)
        score = metric_mode_value(val_metrics, args.monitor)
        is_best = score > best_score
        if is_best:
            best_score = score
            best_epoch = epoch
            torch.save(model.state_dict(), ckpt_path)
        append_log(
            log_path,
            {
                "epoch": epoch,
                "train_loss": f"{float(np.mean(losses)):.6f}",
                "val_accuracy": f"{val_metrics.accuracy:.6f}",
                "val_macro_f1": f"{val_metrics.macro_f1:.6f}",
                "val_mae": f"{val_metrics.mae:.6f}",
                "best": int(is_best),
            },
        )
        print(
            f"Epoch {epoch:03d} | loss={np.mean(losses):.4f} "
            f"val_acc={val_metrics.accuracy:.4f} val_f1={val_metrics.macro_f1:.4f} "
            f"val_mae={val_metrics.mae:.4f}" + ("  ✓best" if is_best else ""),
            flush=True,
        )
        if epoch - best_epoch >= args.patience:
            print(f"[early-stop] epoch={epoch} best_epoch={best_epoch} best_score={best_score:.6f}")
            break

    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    for split in ("val", "test"):
        probs, labels, paths = eval_probs(model, loaders[split], device, args.hflip)
        np.save(os.path.join(out_dir, f"{split}_probs.npy"), probs)
        np.save(os.path.join(out_dir, f"{split}_labels.npy"), labels)
        write_predictions(os.path.join(out_dir, f"{split}_predictions.csv"), probs, labels, paths)
        metrics = compute_metrics(probs, labels)
        with open(os.path.join(out_dir, f"{split}_metrics.json"), "w", encoding="utf-8") as f:
            json.dump(asdict(metrics), f, indent=2, ensure_ascii=False)
        print(
            f"[{split.upper()}] acc={metrics.accuracy:.4f} "
            f"F1={metrics.macro_f1:.4f} MAE={metrics.mae:.4f}",
            flush=True,
        )


if __name__ == "__main__":
    main()
