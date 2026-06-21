import argparse
import json
from pathlib import Path


def load_metrics(path: Path):
    metrics = json.loads(path.read_text(encoding="utf-8"))
    if path.name == "calibrated_metrics.json":
        metrics = metrics["test_metrics"]
    return metrics


def main():
    parser = argparse.ArgumentParser(description="Summarize result json files.")
    parser.add_argument("--results-dir", default="experiments/results/results_final")
    args = parser.parse_args()

    root = Path(args.results_dir)
    rows = []
    result_paths = sorted(root.glob("*/test_metrics.json"))
    result_paths += sorted(root.glob("*/calibrated_metrics.json"))

    for path in result_paths:
        metrics = load_metrics(path)
        rows.append((
            path.parent.name,
            metrics["accuracy"],
            metrics["macro_precision"],
            metrics["macro_recall"],
            metrics["macro_f1"],
            metrics["mae"],
        ))

    if not rows:
        raise SystemExit(f"No result json found under {root}")

    print("| Experiment | Accuracy | Macro Precision | Macro Recall | Macro F1 | MAE |")
    print("| --- | ---: | ---: | ---: | ---: | ---: |")
    for name, acc, precision, recall, f1, mae in rows:
        print(f"| {name} | {acc:.4f} | {precision:.4f} | {recall:.4f} | {f1:.4f} | {mae:.4f} |")


if __name__ == "__main__":
    main()
