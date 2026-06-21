#!/usr/bin/env python3
"""Architecture diagram (top-down) showing the three core innovations
(cross-attn dual-branch / multi-scale + warm-start / RSOMR routing),
styled after the original 原架构图.png."""
import os, sys, json, csv
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import matplotlib.font_manager as fm

plt.rcParams["axes.unicode_minus"] = False  # English labels (clean, paper-standard)
FIG = "docs/figures"
os.makedirs(FIG, exist_ok=True)

C = {"in": "#e8e8e8", "vit": "#bcd4f0", "cnx": "#fbd9a8", "fuse": "#cbb6e6",
     "feat": "#bfe3b6", "rsomr": "#ffe69a", "opc": "#ffd24d", "cbmog": "#f6c39a",
     "out": "#e8736a", "loss": "#d9d9d9", "innov": "#c0392b"}


def arch():
    fig, ax = plt.subplots(figsize=(12.5, 13)); ax.axis("off")
    ax.set_xlim(0, 12.5); ax.set_ylim(0, 13)

    def box(x, y, w, h, text, c, fs=10, bold=False):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.06",
                                    fc=c, ec="#333", lw=1.3))
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs,
                weight="bold" if bold else "normal")

    def arr(x1, y1, x2, y2, c="#333"):
        ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>",
                                     mutation_scale=16, lw=1.6, color=c))

    def innov(x, y, text):
        ax.add_patch(FancyBboxPatch((x, y), 2.5, 0.55, boxstyle="round,pad=0.04",
                                    fc="white", ec=C["innov"], lw=1.6))
        ax.text(x + 1.25, y + 0.275, text, ha="center", va="center",
                fontsize=8.5, color=C["innov"], weight="bold")

    # title + input
    ax.text(6.25, 12.7, "Text-Vision Multimodal Ordinal Framework for Knee KL Grading",
            ha="center", fontsize=13.5, weight="bold")
    box(4.7, 11.7, 3.1, 0.7, "Input: knee X-ray", C["in"], 11, True)

    # two branches
    box(1.0, 10.0, 4.0, 1.25,
        "Global branch: ViT (BiomedCLIP)\nglobal anatomy / joint alignment", C["vit"], 10)
    box(7.5, 10.0, 4.0, 1.25,
        "Local branch: ConvNeXt-V2 Large\nstage-2 fine osteophyte + stage-3 joint space\nper-sample scale-gate fusion", C["cnx"], 9)
    innov(9.6, 11.35, "Innov. 2: Multi-scale local")
    arr(6.25, 11.7, 3.0, 11.25); arr(6.25, 11.7, 9.5, 11.25)

    # cross-attn fusion
    box(3.4, 8.4, 5.7, 1.15,
        "Dual-branch Cross-Attention Fusion\nbidirectional cross-attn (global↔local) + per-sample gate", C["fuse"], 9.5)
    innov(9.6, 8.6, "Innov. 1: Cross-attn dual-branch")
    innov(0.2, 8.6, "Innov. 2 (training): warm-start")
    arr(3.0, 10.0, 5.0, 9.55); arr(9.5, 10.0, 7.5, 9.55)

    # fused feature + base logits (OCM-CLIP base; cluster-text folded in, ablation-verified non-essential)
    box(3.5, 6.95, 5.5, 1.0,
        "OCM-CLIP base output:  fuse_feat + base logits\n(legacy cluster-text fusion inside; ablation: non-essential)",
        C["feat"], 9, True)
    arr(6.25, 8.4, 6.25, 7.95)

    # multimodal evidence layer: RSOMR (single innovation = evidence pool + router)
    box(1.3, 4.5, 9.6, 1.7,
        "RSOMR  —  Reliability-aware Semantic-Ordinal Multimodal Routing\n"
        "evidence pool:  category semantic prototype  |  ordinal case memory  |  text-guided expert\n"
        "(+ ordinal transition-boundary candidate, rejected by router)\n"
        "reliability router: validation-selected per dataset, rejects unreliable evidence (w=0)",
        C["rsomr"], 9)
    innov(8.7, 6.3, "Innov. 3: RSOMR routing")
    arr(6.25, 7.0, 6.25, 6.2)

    # output
    box(4.3, 3.1, 3.9, 0.95, "Output: final KL logits (KL0-KL4, ordinal)", C["out"], 10, True)
    arr(6.25, 4.5, 6.25, 4.05)

    # training loss
    box(2.4, 1.7, 7.7, 0.85,
        "Base training loss: CE + ordinal loss (+ proto/cluster CE)\nRSOMR: inference-time routing (no training loss)", C["loss"], 9)
    arr(6.25, 3.1, 6.25, 2.55)

    fig.savefig(os.path.join(FIG, "fig_architecture_full.png"), dpi=160, bbox_inches="tight")
    plt.close(fig)
    print("saved fig_architecture_full.png")


if __name__ == "__main__":
    arch()  # clean 3-innovation architecture (OPC/CB-MOG removed)
