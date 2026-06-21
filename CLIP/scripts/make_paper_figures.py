#!/usr/bin/env python3
"""Generate publication figures for report3 from real result files.
Outputs PNGs to docs/figures/. Pure matplotlib (no seaborn)."""
import os, sys, json, csv
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BC = os.path.join(ROOT, "experiments/results/baseline_compare")
FIG = os.path.join(ROOT, "docs/figures")
os.makedirs(FIG, exist_ok=True)
DS = ["ME", "AR", "PV"]
KL = ["KL0", "KL1", "KL2", "KL3", "KL4"]
plt.rcParams.update({"font.size": 11, "axes.grid": True, "grid.alpha": 0.3,
                     "figure.dpi": 150, "savefig.bbox": "tight"})


def load_summary():
    rows = {}
    with open(os.path.join(BC, "optimized_comparison_summary.csv")) as f:
        for r in csv.DictReader(f):
            rows[r["Method"]] = {k: float(v) for k, v in r.items() if k != "Method"}
    return rows


def fig_main_comparison():
    rows = load_summary()
    # prepend the vanilla OpenAI CLIP (ViT-B/32, linear probe) reference
    clip = json.load(open(os.path.join(BC, "clip_openai_linearprobe.json")))
    key = {"ME": "me", "AR": "ar", "PV": "pv"}
    methods = ["OpenAI CLIP\n(linear probe)"] + list(rows.keys())
    fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.4))
    for j, ds in enumerate(DS):
        accs = [clip[key[ds]]["accuracy"]] + [rows[m][f"{ds} Acc"] for m in rows]
        colors = ["#7f7f7f"] + ["#bbbbbb"] * (len(rows) - 1) + ["#d62728"]
        bars = axes[j].bar(range(len(methods)), accs, color=colors)
        bars[0].set_hatch("//")
        axes[j].set_xticks(range(len(methods)))
        axes[j].set_xticklabels(methods, rotation=55, ha="right", fontsize=8)
        axes[j].set_ylabel("Accuracy") if j == 0 else None
        axes[j].set_title(ds)
        axes[j].set_ylim(min(accs) - 0.05, max(accs) + 0.04)
        axes[j].bar_label(bars, fmt="%.3f", fontsize=6, padding=1)
    fig.suptitle("Accuracy: vanilla OpenAI CLIP (ref.) vs. fine-tuned backbones vs. Ours")
    fig.savefig(os.path.join(FIG, "fig1_main_comparison.png"))
    plt.close(fig)


def fig_ablation():
    # report3 Table 2 (Accuracy)
    stages = ["BiomedCLIP\nsingle", "+Dual-branch\ncross-attn", "+Multi-scale\n& warm-start", "+Multimodal\nrouting"]
    data = {"ME": [0.8214, 0.8616, 0.8750, 0.8839],
            "AR": [0.6781, 0.7216, 0.7240, 0.7252],
            "PV": [0.6239, 0.6606, 0.6881, 0.7064]}
    fig, ax = plt.subplots(figsize=(8, 4.5))
    mk = {"ME": "o-", "AR": "s-", "PV": "^-"}
    for ds in DS:
        ax.plot(range(4), data[ds], mk[ds], label=ds, linewidth=2, markersize=8)
        for i, v in enumerate(data[ds]):
            ax.annotate(f"{v:.3f}", (i, v), textcoords="offset points", xytext=(0, 6), fontsize=7, ha="center")
    ax.set_xticks(range(4)); ax.set_xticklabels(stages, fontsize=9)
    ax.set_ylabel("Accuracy"); ax.set_title("Incremental contribution of each component")
    ax.legend()
    fig.savefig(os.path.join(FIG, "fig2_ablation.png"))
    plt.close(fig)


def _conf(ds_key):
    y, p = [], []
    with open(os.path.join(BC, f"ours_{ds_key}_predictions.csv")) as f:
        for r in csv.DictReader(f):
            y.append(int(r["label"])); p.append(int(r["pred"]))
    cm = np.zeros((5, 5), int)
    for a, b in zip(y, p):
        cm[a, b] += 1
    return cm


def fig_confusion():
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.4))
    for ax, ds, key in zip(axes, DS, ["me", "ar", "pv"]):
        cm = _conf(key)
        cmn = cm / cm.sum(1, keepdims=True).clip(min=1)
        im = ax.imshow(cmn, cmap="Blues", vmin=0, vmax=1)
        ax.set_xticks(range(5)); ax.set_yticks(range(5))
        ax.set_xticklabels(KL, fontsize=8); ax.set_yticklabels(KL, fontsize=8)
        ax.set_xlabel("Predicted"); ax.set_ylabel("True") if ds == "ME" else None
        ax.set_title(f"{ds} (acc={cm.trace()/cm.sum():.3f})")
        for i in range(5):
            for k in range(5):
                ax.text(k, i, f"{cmn[i,k]:.2f}", ha="center", va="center",
                        color="white" if cmn[i, k] > 0.5 else "black", fontsize=7)
        ax.grid(False)
    fig.suptitle("Confusion matrices of the full framework (row-normalized)")
    fig.savefig(os.path.join(FIG, "fig3_confusion.png"))
    plt.close(fig)


def fig_perclass():
    # per-class F1 from report3 tables (ME, AR): single-scale vs full framework
    me = {"single": [0.9037, 0.8467, 0.7500, 0.8302, 0.9254],
          "full":   [0.8939, 0.8392, 0.8077, 0.8929, 0.9538]}
    ar = {"single": [0.8170, 0.3657, 0.7096, 0.8299, 0.8519],
          "full":   [0.8165, 0.4500, 0.7099, 0.8430, 0.8515]}
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.2))
    x = np.arange(5); w = 0.38
    for ax, d, name in zip(axes, [me, ar], ["MedicalExpert", "archive"]):
        ax.bar(x - w/2, d["single"], w, label="single-scale dual-branch", color="#9ecae1")
        ax.bar(x + w/2, d["full"], w, label="full framework", color="#d62728")
        ax.set_xticks(x); ax.set_xticklabels(KL)
        ax.set_ylabel("F1"); ax.set_title(name); ax.set_ylim(0, 1.0); ax.legend(fontsize=8)
        for i in range(5):
            ax.annotate(f"+{d['full'][i]-d['single'][i]:+.2f}".replace("++", "+"),
                        (i, max(d['single'][i], d['full'][i])), textcoords="offset points",
                        xytext=(0, 4), fontsize=6, ha="center")
    fig.suptitle("Per-class F1: largest gain on early-OA boundary (KL1)")
    fig.savefig(os.path.join(FIG, "fig4_perclass_f1.png"))
    plt.close(fig)


if __name__ == "__main__":
    # final figures only (fig1-fig5 in report3); architecture fig via make_architecture.py
    fig_main_comparison(); fig_ablation(); fig_confusion(); fig_perclass()
    print("saved figures to", FIG)
    for f in sorted(os.listdir(FIG)):
        print(" -", f)
