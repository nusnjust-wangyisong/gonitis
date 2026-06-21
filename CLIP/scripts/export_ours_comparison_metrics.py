#!/usr/bin/env python3
"""Export final Ours comparison metrics and predictions.

The final paper model is dataset-adaptive:
  ME: RSOMR text-guided expert router.
  AR: class-aware semantic prototype expert.
  PV: text-guided ordinal case-memory expert.
"""
import csv
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

import scripts.semantic_ordinal_transport_adapter_eval as sota
import scripts.text_visual_proto_blend_eval as tvpb
from clip.model import build_fusion_model
from main import load_clip_state_dict


OUT_DIR = Path("experiments/results/baseline_compare")
ME_EXPERTS = [
    "experiments/checkpoints/checkpoints_universal_text_me_gtp/MedicalExpert-split_full_dual/best_model.pth",
    "experiments/checkpoints/checkpoints_text_curriculum_me_lt005_stage2/MedicalExpert-split_full_dual/best_model.pth",
    "experiments/checkpoints/checkpoints_text_curriculum_me_stage2/MedicalExpert-split_full_dual/best_model.pth",
]


def full_metrics(logits, labels):
    pred = logits.argmax(dim=-1).cpu().numpy()
    return {
        "accuracy": float(accuracy_score(labels, pred)),
        "macro_precision": float(precision_score(labels, pred, average="macro", zero_division=0)),
        "macro_recall": float(recall_score(labels, pred, average="macro", zero_division=0)),
        "macro_f1": float(f1_score(labels, pred, average="macro", zero_division=0)),
        "mae": float(np.abs(pred - labels).mean()),
    }


def write_predictions(path, labels, pred):
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["label", "pred"])
        for y, p in zip(labels, pred):
            writer.writerow([int(y), int(p)])


def load_full_model(dataset, device):
    data_root, ckpt = sota.CFG[dataset]
    sd = load_clip_state_dict("ViT-B/32")
    model = build_fusion_model(
        sd, backbone="biomedclip", enable_dual_branch=True, convnext_multi_scale=True
    ).to(device).eval()
    raw = torch.load(ckpt, map_location="cpu")
    model.load_state_dict(raw.get("model_state_dict", raw), strict=False)
    model.refresh_text_prototypes()
    return model, data_root


def export_me(device):
    model, data_root = load_full_model("me", device)
    test = sota.extract(model, data_root, "test", device, batch_size=64, num_workers=4)
    extra_test = []
    for ckpt in ME_EXPERTS:
        logits, labels = sota.collect_logits_for_checkpoint(ckpt, data_root, "test", device, 64, 4)
        if not np.array_equal(labels, test["labels"]):
            raise ValueError(f"ME expert labels mismatch: {ckpt}")
        extra_test.append(logits)
    selected = json.loads(Path("experiments/results/semantic_ordinal_transport_adapter_fine/me.json").read_text())["selected"]
    router = selected["expert_router"]
    stack = torch.stack([sota.normalize_logits(test["base"])] + [sota.normalize_logits(x) for x in extra_test])
    weights = torch.tensor(router["weights"], dtype=torch.float32).view(-1, 1, 1)
    final = (stack * weights).sum(dim=0)
    return final, test["labels"]


def export_ar(device):
    model, data_root = load_full_model("ar", device)
    train_f, _, train_y = tvpb.extract(model, data_root, "train", device, 64)
    test_f, test_base, test_y = tvpb.extract(model, data_root, "test", device, 64)
    selected = json.loads(
        Path("experiments/results/text_visual_proto_blend_aligned_classwise/ar.json").read_text()
    )["selected"]
    visual_proto = tvpb.build_visual_prototypes(train_f, train_y)
    text_proto = F.normalize(model.text_feat.detach().float().cpu(), dim=-1)
    alpha = torch.tensor(selected["alpha_text_per_class"], dtype=torch.float32)
    proto = tvpb.proto_logits(test_f, visual_proto, text_proto, alpha, selected["proto_scale"])
    final = tvpb.zscore(test_base) + selected["proto_weight"] * tvpb.zscore(proto)
    return final, test_y


def export_pv(device):
    model, data_root = load_full_model("pv", device)
    train = sota.extract(model, data_root, "train", device, 64, 4)
    test = sota.extract(model, data_root, "test", device, 64, 4)
    selected = json.loads(Path("experiments/results/semantic_ordinal_transport_adapter/pv.json").read_text())["selected"]
    text_proto = F.normalize(model.text_feat.detach().float().cpu(), dim=-1)
    kernel = sota.build_text_kernel(
        text_proto,
        temperature=selected["kernel_temperature"],
        ordinal_sigma=selected["ordinal_sigma"],
        mix=selected["kernel_mix"],
    )
    cache = sota.semantic_cache_logits(
        test["feats"], train["feats"], train["labels"],
        kernel, beta=selected["beta"], balanced=selected["balanced_cache"]
    )
    ordinal = sota.ordinal_boundary_logits(
        test["transition"], scale=selected["ordinal_scale"], sign=selected["ordinal_sign"]
    )
    residual = (
        selected["cache_weight"] * sota.zscore(cache)
        + selected["text_weight"] * sota.zscore(test["text"])
        + selected["ordinal_weight"] * sota.zscore(ordinal)
        + selected["branch_text_weight"] * sota.zscore(0.5 * test["anatomy"] + 0.5 * test["pathology"])
    )
    final = sota.zscore(test["base"]) + sota.confidence_gate(test["base"], selected["tau"]) * residual
    return final, test["labels"]


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    exporters = {"me": export_me, "ar": export_ar, "pv": export_pv}
    metrics = {}
    for dataset, fn in exporters.items():
        logits, labels = fn(device)
        pred = logits.argmax(dim=-1).cpu().numpy()
        metrics[dataset] = full_metrics(logits, labels)
        write_predictions(OUT_DIR / f"ours_{dataset}_predictions.csv", labels, pred)
        print(dataset, json.dumps(metrics[dataset], indent=2))
    (OUT_DIR / "ours_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(f"saved: {OUT_DIR / 'ours_metrics.json'}")


if __name__ == "__main__":
    main()
