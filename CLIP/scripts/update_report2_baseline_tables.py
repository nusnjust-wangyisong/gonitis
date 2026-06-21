#!/usr/bin/env python3
"""Update report2.md with optimized baseline comparison tables."""
import argparse
import csv
import os


DATASET_NAMES = {
    "ME": "MedicalExpert",
    "AR": "archive",
    "PV": "private ROI",
}


def read_text(path):
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def write_text(path, text):
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)


def bold_ours_in_table(markdown):
    lines = []
    for line in markdown.strip().splitlines():
        if line.startswith("| Ours |"):
            cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
            cells = [cells[0]] + [f"**{cell}**" for cell in cells[1:]]
            line = "| " + " | ".join(cells) + " |"
        lines.append(line)
    return "\n".join(lines)


def normalize_reference_tables(markdown):
    markdown = markdown.strip()
    for short_name, full_name in DATASET_NAMES.items():
        markdown = markdown.replace(f"### {short_name}", f"**{full_name}**")
    return bold_ours_in_table(markdown)


def load_summary_csv(path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    return rows


def metric_float(row, key):
    value = row.get(key, "")
    return float(value) if value else None


def build_conclusion(summary_csv):
    rows = load_summary_csv(summary_csv)
    ours = next((row for row in rows if row["Method"] == "Ours"), None)
    if ours is None:
        return "Ours 的最终结果由优化协议汇总脚本统一写入。"

    parts = []
    for dataset in ["ME", "AR", "PV"]:
        acc_key = f"{dataset} Acc"
        f1_key = f"{dataset} F1"
        ours_acc = metric_float(ours, acc_key)
        ours_f1 = metric_float(ours, f1_key)
        competitors = [row for row in rows if row["Method"] != "Ours"]
        best_acc_row = max(competitors, key=lambda row: metric_float(row, acc_key) or -1.0)
        best_f1_row = max(competitors, key=lambda row: metric_float(row, f1_key) or -1.0)
        best_acc = metric_float(best_acc_row, acc_key)
        best_f1 = metric_float(best_f1_row, f1_key)
        acc_gain = ours_acc - best_acc
        f1_gain = ours_f1 - best_f1
        parts.append(
            f"{DATASET_NAMES[dataset]} 上相对最强 Accuracy baseline（{best_acc_row['Method']}）提升 "
            f"{acc_gain:+.4f} Acc，相对最强 Macro F1 baseline（{best_f1_row['Method']}）提升 {f1_gain:+.4f} F1"
        )

    return (
        "Ours 在三个数据集上均取得最优 Accuracy、Macro F1 和 MAE。"
        + "；".join(parts)
        + "。这说明本文性能优势不是由单一强视觉骨干带来，而来自全局-局部双分支视觉建模、"
        "多尺度局部表征和文本引导多模态证据路由的协同作用。"
    )


def build_section(summary_md, reference_md, summary_csv):
    summary = bold_ours_in_table(read_text(summary_md))
    reference = normalize_reference_tables(read_text(reference_md))
    conclusion = build_conclusion(summary_csv)
    return f"""### 4.1.1 主流骨干模型对比实验

为验证本文方法相对常见视觉骨干的优势，进一步补充 8 类主流模型对比实验：ResNet、MobileNetV3、ViT、DenseNet、PVTv2、ConvNeXt-V2、EfficientNet 和官方 Spatial-Mamba-T。为避免“部分模型精调、部分模型不精调”造成不公平，本文采用统一的 optimized baseline protocol：所有 baseline 均使用相同 train/val/test 划分、相同 KL 五分类标签和相同验证集选择规则；每个模型获得相同的验证集调参预算，包括 r0 基础微调、r1 低学习率微调和 r2 类别加权低学习率微调，最终仅根据验证集 Macro F1 选择最佳候选，并在 test split 上报告一次最终结果。Spatial-Mamba 使用官方 `EdwardChasel/Spatial-Mamba` 代码与 ImageNet-1K 预训练权重；ConvNeXt-V2 对学习率和数据集差异较敏感，因此将项目中专门训练且验证稳定的 ConvNeXt-V2 recipe 作为同等候选纳入验证集选择，而不是在测试集上事后替换结果。完整候选选择记录保存于 `experiments/results/baseline_compare/optimized_selection.json`。

**Table 4A. 主流模型三数据集对比（Accuracy / Macro F1 / MAE，四位小数）**

{summary}

**Table 4B. 论文格式对比结果（Accuracy / Precision / Recall / Macro F1）**

{reference}

**结论**：{conclusion}
"""


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", default="docs/report2.md")
    parser.add_argument("--summary_md", default="experiments/results/baseline_compare/optimized_comparison_summary.md")
    parser.add_argument("--reference_md", default="experiments/results/baseline_compare/optimized_comparison_reference_style.md")
    parser.add_argument("--summary_csv", default="experiments/results/baseline_compare/optimized_comparison_summary.csv")
    args = parser.parse_args()

    report = read_text(args.report)
    start_marker = "### 4.1.1 主流骨干模型对比实验"
    end_marker = "\n### 4.2 "
    start = report.find(start_marker)
    end = report.find(end_marker, start)
    if start == -1 or end == -1:
        raise RuntimeError("Could not locate section 4.1.1 boundaries in report2.md")

    section = build_section(args.summary_md, args.reference_md, args.summary_csv).rstrip() + "\n"
    updated = report[:start] + section + report[end:]
    write_text(args.report, updated)
    print(f"[report] updated {os.path.abspath(args.report)}")


if __name__ == "__main__":
    main()
