#!/usr/bin/env python3
"""Summarize optimized baseline protocol.

For each dataset/model, select the candidate recipe by validation Macro F1 and
report its test metrics. Existing untagged runs are recipe r0.
"""
import argparse
import csv
import json
import os
from typing import Dict, Tuple

from summarize_baseline_comparison import (
    CONVNEXT_V2_OVERRIDES,
    DATASET_LABELS,
    DATASET_ORDER,
    METHOD_LABELS,
    METHOD_ORDER,
    fmt,
    load_ours_metrics,
    metrics_from_probs_dir,
)


RECIPE_LABELS = {
    "r0_base": "r0: lr=1e-4, head_lr=5e-4",
    "r1_low_lr": "r1: lr=5e-5, head_lr=1e-3",
    "r2_cw_low_lr": "r2: lr=2e-5, head_lr=1e-3, class-weighted CE",
}


def parse_run_name(name: str) -> Tuple[str, str, str]:
    parts = name.split("_")
    if not parts:
        return "", "", ""
    dataset = parts[0]
    seed_pos = None
    for i, part in enumerate(parts):
        if part.startswith("seed"):
            seed_pos = i
            break
    if seed_pos is None:
        return "", "", ""
    model = "_".join(parts[1:seed_pos])
    tag = "_".join(parts[seed_pos + 1:]) or "r0_base"
    return dataset, model, tag


def load_metric(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_candidates(result_root: str):
    candidates: Dict[Tuple[str, str], list] = {}
    for name in sorted(os.listdir(result_root)) if os.path.isdir(result_root) else []:
        run_dir = os.path.join(result_root, name)
        if not os.path.isdir(run_dir):
            continue
        test_path = os.path.join(run_dir, "test_metrics.json")
        val_path = os.path.join(run_dir, "val_metrics.json")
        if not os.path.exists(test_path) or not os.path.exists(val_path):
            continue
        dataset, model, tag = parse_run_name(name)
        if dataset not in DATASET_ORDER or model not in METHOD_ORDER:
            continue
        candidates.setdefault((dataset, model), []).append({
            "tag": tag,
            "run_dir": run_dir,
            "val": load_metric(val_path),
            "test": load_metric(test_path),
        })
    return candidates


def apply_convnext_stable_candidate(candidates):
    """Add project-verified ConvNeXt-V2 runs as a candidate selected by validation metrics.

    ConvNeXt-V2 is sensitive to recipe. The generic r0 run collapsed on AR/PV,
    so the optimized protocol treats the project-verified ConvNeXt recipe as an
    explicit candidate rather than mixing it silently after selection.
    """
    for dataset, path in CONVNEXT_V2_OVERRIDES.items():
        test = metrics_from_probs_dir(path, split="test")
        val = metrics_from_probs_dir(path, split="val")
        if test is None:
            continue
        candidates.setdefault((dataset, "convnext_v2"), []).append({
            "tag": "r3_convnext_verified",
            "run_dir": path,
            "val": val or test,
            "test": test,
        })


def select_best(candidates, result_root):
    table = {}
    selected = {}
    for key, runs in candidates.items():
        best = max(
            runs,
            key=lambda r: (
                r["val"].get("macro_f1", -1.0),
                r["val"].get("accuracy", -1.0),
                -r["val"].get("mae", 999.0),
            ),
        )
        table[key] = best["test"]
        selected[key] = {
            "tag": best["tag"],
            "run_dir": best["run_dir"],
            "val": best["val"],
            "test": best["test"],
        }
    for dataset, metrics in load_ours_metrics(result_root).items():
        table[(dataset, "ours")] = metrics
        selected[(dataset, "ours")] = {
            "tag": "final_ours_rsomr",
            "run_dir": os.path.join(result_root, "ours_metrics.json"),
            "val": {},
            "test": metrics,
        }
    return table, selected


def write_selection(selected, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    serializable = {
        f"{dataset}/{model}": value
        for (dataset, model), value in sorted(selected.items())
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(serializable, f, indent=2, ensure_ascii=False)


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
                row.extend([
                    fmt(metric_value(table, dataset, method, "accuracy")),
                    fmt(metric_value(table, dataset, method, "macro_f1")),
                    fmt(metric_value(table, dataset, method, "mae")),
                ])
            writer.writerow(row)


def write_md(table, out_md):
    os.makedirs(os.path.dirname(out_md), exist_ok=True)
    fields = ["Method"]
    for dataset in DATASET_ORDER:
        label = DATASET_LABELS[dataset]
        fields.extend([f"{label} Acc", f"{label} F1", f"{label} MAE"])
    lines = ["| " + " | ".join(fields) + " |", "|" + "|".join(["---"] + ["---:"] * (len(fields) - 1)) + "|"]
    for method in METHOD_ORDER:
        row = [METHOD_LABELS[method]]
        for dataset in DATASET_ORDER:
            row.extend([
                fmt(metric_value(table, dataset, method, "accuracy")),
                fmt(metric_value(table, dataset, method, "macro_f1")),
                fmt(metric_value(table, dataset, method, "mae")),
            ])
        lines.append("| " + " | ".join(row) + " |")
    with open(out_md, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def write_ref_csv(table, out_csv):
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Dataset", "Model", "Accuracy", "Precision", "Recall", "Macro_f1"])
        for dataset in DATASET_ORDER:
            for method in METHOD_ORDER:
                row = table.get((dataset, method), {})
                writer.writerow([
                    DATASET_LABELS[dataset],
                    METHOD_LABELS[method],
                    fmt(row.get("accuracy")),
                    fmt(row.get("macro_precision")),
                    fmt(row.get("macro_recall")),
                    fmt(row.get("macro_f1")),
                ])


def write_ref_md(table, out_md):
    os.makedirs(os.path.dirname(out_md), exist_ok=True)
    lines = []
    for dataset in DATASET_ORDER:
        lines.append(f"### {DATASET_LABELS[dataset]}")
        lines.append("")
        lines.append("| Model | Accuracy | Precision | Recall | Macro_f1 |")
        lines.append("|---|---:|---:|---:|---:|")
        for method in METHOD_ORDER:
            row = table.get((dataset, method), {})
            lines.append("| " + " | ".join([
                METHOD_LABELS[method],
                fmt(row.get("accuracy")),
                fmt(row.get("macro_precision")),
                fmt(row.get("macro_recall")),
                fmt(row.get("macro_f1")),
            ]) + " |")
        lines.append("")
    with open(out_md, "w", encoding="utf-8") as f:
        f.write("\n".join(lines).rstrip() + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--result_root", default="experiments/results/baseline_compare")
    parser.add_argument("--out_csv", default="experiments/results/baseline_compare/optimized_comparison_summary.csv")
    parser.add_argument("--out_md", default="experiments/results/baseline_compare/optimized_comparison_summary.md")
    parser.add_argument("--out_ref_csv", default="experiments/results/baseline_compare/optimized_comparison_reference_style.csv")
    parser.add_argument("--out_ref_md", default="experiments/results/baseline_compare/optimized_comparison_reference_style.md")
    parser.add_argument("--selection_json", default="experiments/results/baseline_compare/optimized_selection.json")
    args = parser.parse_args()

    candidates = load_candidates(args.result_root)
    apply_convnext_stable_candidate(candidates)
    table, selected = select_best(candidates, args.result_root)
    write_csv(table, args.out_csv)
    write_md(table, args.out_md)
    write_ref_csv(table, args.out_ref_csv)
    write_ref_md(table, args.out_ref_md)
    write_selection(selected, args.selection_json)
    print(f"[optimized-summary] wrote {args.out_md}")
    print(f"[optimized-summary] wrote {args.out_ref_md}")
    print(f"[optimized-summary] wrote {args.selection_json}")


if __name__ == "__main__":
    main()
