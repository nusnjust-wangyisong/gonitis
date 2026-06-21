#!/usr/bin/env python3
"""Summarize baseline comparison metrics into CSV and Markdown tables."""
import argparse
import csv
import json
import os
from typing import Dict, Tuple


DATASET_ORDER = ["me", "ar", "pv"]
DATASET_LABELS = {"me": "ME", "ar": "AR", "pv": "PV"}
METHOD_ORDER = [
    "resnet",
    "mobilenet_v3",
    "vit",
    "densenet",
    "pvt_v2",
    "convnext_v2",
    "efficientnet",
    "spatialmamba",
    "ours",
]
METHOD_LABELS = {
    "resnet": "ResNet",
    "mobilenet_v3": "MobileNetV3",
    "vit": "ViT",
    "densenet": "DenseNet",
    "pvt_v2": "PVTv2",
    "convnext_v2": "ConvNeXt-V2",
    "efficientnet": "EfficientNet",
    "spatialmamba": "Spatial-Mamba-T",
    "ours": "Ours",
}

OURS = {
    "me": {
        "accuracy": 0.8839285714285714,
        "macro_f1": 0.8821111460069737,
        "mae": 0.14732142857142858,
        "macro_precision": None,
        "macro_recall": None,
    },
    "ar": {
        "accuracy": 0.7252415458937198,
        "macro_f1": 0.734563320488953,
        "mae": 0.3085748792270531,
        "macro_precision": None,
        "macro_recall": None,
    },
    "pv": {
        "accuracy": 0.7064220183486238,
        "macro_f1": 0.701229521023628,
        "mae": 0.3333333333333333,
        "macro_precision": None,
        "macro_recall": None,
    },
}

CONVNEXT_V2_OVERRIDES = {
    "me": "experiments/results/results_cvxL/medical",
    "ar": "experiments/results/results_cvxL/archive",
    "pv": "experiments/results/results_cvx_roi_L/private",
}


def parse_run_name(name: str) -> Tuple[str, str]:
    parts = name.split("_")
    if not parts:
        return "", ""
    dataset = parts[0]
    if name.startswith(f"{dataset}_mobilenet_v3"):
        return dataset, "mobilenet_v3"
    if name.startswith(f"{dataset}_pvt_v2"):
        return dataset, "pvt_v2"
    if name.startswith(f"{dataset}_convnext_v2"):
        return dataset, "convnext_v2"
    if name.startswith(f"{dataset}_spatialmamba"):
        return dataset, "spatialmamba"
    if name.startswith(f"{dataset}_resnet"):
        return dataset, "resnet"
    if len(parts) >= 2:
        return dataset, parts[1]
    return dataset, ""


def load_results(result_root: str) -> Dict[Tuple[str, str], Dict[str, float]]:
    table = {}
    if os.path.isdir(result_root):
        for name in sorted(os.listdir(result_root)):
            metrics_path = os.path.join(result_root, name, "test_metrics.json")
            if not os.path.exists(metrics_path):
                continue
            dataset, method = parse_run_name(name)
            with open(metrics_path, "r", encoding="utf-8") as f:
                table[(dataset, method)] = json.load(f)
    for dataset, metrics in load_convnext_v2_overrides().items():
        table[(dataset, "convnext_v2")] = metrics
    for dataset, metrics in load_ours_metrics(result_root).items():
        table[(dataset, "ours")] = metrics
    return table


def metrics_from_probs_dir(path: str, split: str = "test"):
    probs_path = os.path.join(path, f"{split}_probs.npy")
    labels_path = os.path.join(path, f"{split}_labels.npy")
    if not os.path.exists(probs_path) or not os.path.exists(labels_path):
        return None
    import numpy as np
    from sklearn.metrics import f1_score, precision_score, recall_score

    probs = np.load(probs_path)
    labels = np.load(labels_path)
    preds = probs.argmax(1)
    return {
        "accuracy": float((preds == labels).mean()),
        "macro_precision": float(precision_score(labels, preds, average="macro", zero_division=0)),
        "macro_recall": float(recall_score(labels, preds, average="macro", zero_division=0)),
        "macro_f1": float(f1_score(labels, preds, average="macro", zero_division=0)),
        "mae": float(np.abs(preds - labels).mean()),
    }


def load_convnext_v2_overrides():
    out = {}
    for dataset, path in CONVNEXT_V2_OVERRIDES.items():
        metrics = metrics_from_probs_dir(path)
        if metrics is not None:
            out[dataset] = metrics
    return out


def load_ours_metrics(result_root: str):
    path = os.path.join(result_root, "ours_metrics.json")
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return OURS


def fmt(value):
    if value is None:
        return ""
    return f"{float(value):.4f}"


def metric_value(table, dataset, method, metric):
    row = table.get((dataset, method))
    if not row:
        return None
    return row.get(metric)


def write_csv(table, out_csv):
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    fields = ["Method"]
    for dataset in DATASET_ORDER:
        label = DATASET_LABELS[dataset]
        fields.extend([f"{label} Acc", f"{label} F1", f"{label} MAE"])
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(fields)
        for method in METHOD_ORDER:
            row = [METHOD_LABELS[method]]
            for dataset in DATASET_ORDER:
                row.extend(
                    [
                        fmt(metric_value(table, dataset, method, "accuracy")),
                        fmt(metric_value(table, dataset, method, "macro_f1")),
                        fmt(metric_value(table, dataset, method, "mae")),
                    ]
                )
            writer.writerow(row)


def write_md(table, out_md):
    os.makedirs(os.path.dirname(out_md), exist_ok=True)
    fields = ["Method"]
    for dataset in DATASET_ORDER:
        label = DATASET_LABELS[dataset]
        fields.extend([f"{label} Acc", f"{label} F1", f"{label} MAE"])
    lines = []
    lines.append("| " + " | ".join(fields) + " |")
    lines.append("|" + "|".join(["---"] + ["---:"] * (len(fields) - 1)) + "|")
    for method in METHOD_ORDER:
        row = [METHOD_LABELS[method]]
        for dataset in DATASET_ORDER:
            row.extend(
                [
                    fmt(metric_value(table, dataset, method, "accuracy")),
                    fmt(metric_value(table, dataset, method, "macro_f1")),
                    fmt(metric_value(table, dataset, method, "mae")),
                ]
            )
        lines.append("| " + " | ".join(row) + " |")
    with open(out_md, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def write_reference_style_md(table, out_md):
    """Write one table per dataset: Model / Accuracy / Precision / Recall / Macro_f1."""
    os.makedirs(os.path.dirname(out_md), exist_ok=True)
    lines = []
    for dataset in DATASET_ORDER:
        label = DATASET_LABELS[dataset]
        lines.append(f"### {label}")
        lines.append("")
        lines.append("| Model | Accuracy | Precision | Recall | Macro_f1 |")
        lines.append("|---|---:|---:|---:|---:|")
        for method in METHOD_ORDER:
            row = table.get((dataset, method), {})
            lines.append(
                "| "
                + " | ".join(
                    [
                        METHOD_LABELS[method],
                        fmt(row.get("accuracy")),
                        fmt(row.get("macro_precision")),
                        fmt(row.get("macro_recall")),
                        fmt(row.get("macro_f1")),
                    ]
                )
                + " |"
            )
        lines.append("")
    with open(out_md, "w", encoding="utf-8") as f:
        f.write("\n".join(lines).rstrip() + "\n")


def write_reference_style_csv(table, out_csv):
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Dataset", "Model", "Accuracy", "Precision", "Recall", "Macro_f1"])
        for dataset in DATASET_ORDER:
            for method in METHOD_ORDER:
                row = table.get((dataset, method), {})
                writer.writerow(
                    [
                        DATASET_LABELS[dataset],
                        METHOD_LABELS[method],
                        fmt(row.get("accuracy")),
                        fmt(row.get("macro_precision")),
                        fmt(row.get("macro_recall")),
                        fmt(row.get("macro_f1")),
                    ]
                )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--result_root", default="experiments/results/baseline_compare")
    parser.add_argument("--out_csv", default="experiments/results/baseline_compare/comparison_summary.csv")
    parser.add_argument("--out_md", default="experiments/results/baseline_compare/comparison_summary.md")
    parser.add_argument("--out_ref_csv", default="experiments/results/baseline_compare/comparison_reference_style.csv")
    parser.add_argument("--out_ref_md", default="experiments/results/baseline_compare/comparison_reference_style.md")
    args = parser.parse_args()

    table = load_results(args.result_root)
    write_csv(table, args.out_csv)
    write_md(table, args.out_md)
    write_reference_style_csv(table, args.out_ref_csv)
    write_reference_style_md(table, args.out_ref_md)
    print(f"[summary] wrote {args.out_csv}")
    print(f"[summary] wrote {args.out_md}")
    print(f"[summary] wrote {args.out_ref_csv}")
    print(f"[summary] wrote {args.out_ref_md}")


if __name__ == "__main__":
    main()
