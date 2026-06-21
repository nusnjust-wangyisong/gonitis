#!/usr/bin/env python3
"""Vanilla OpenAI CLIP (ViT-B/32, general visual tower) baseline via linear probe.
Frozen CLIP image features + logistic regression; C selected on val by macro-F1;
report on test. This is the 'original CLIP' reference (before the biomedical
visual tower and the proposed framework)."""
import os, sys, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

import scripts.semantic_ordinal_transport_adapter_eval as sota
from clip.model import build_fusion_model
from data.dataset import ClassificationDataset, get_transforms
from main import load_clip_state_dict

NC = 5


@torch.no_grad()
def feats(model, data_root, split, device, bs=64, nw=4):
    tf = get_transforms(img_size=224, is_training=False, use_clahe=True, to_rgb=True,
                        clahe_clip_limit=2.0, clahe_tile_grid_size=(8, 8),
                        percentile=(1.0, 99.0), normalize=True)
    loader = DataLoader(ClassificationDataset(data_root, split=split, transform=tf),
                        batch_size=bs, shuffle=False, num_workers=nw)
    X, y = [], []
    for b in loader:
        f = model.encode_image(b["image"].to(device)).float().cpu()  # OpenAI CLIP ViT-B/32, 512-d
        X.append(f); y.extend(b["label"].numpy())
    return torch.cat(X).numpy(), np.asarray(y)


def metrics(pred, y):
    return {"accuracy": round(float(accuracy_score(y, pred)), 4),
            "macro_precision": round(float(precision_score(y, pred, average="macro", zero_division=0)), 4),
            "macro_recall": round(float(recall_score(y, pred, average="macro", zero_division=0)), 4),
            "macro_f1": round(float(f1_score(y, pred, average="macro", zero_division=0)), 4),
            "mae": round(float(np.abs(np.asarray(pred) - y).mean()), 4)}


def main():
    import argparse
    ap = argparse.ArgumentParser(); ap.add_argument("--datasets", nargs="+", default=["me", "ar", "pv"])
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    sd = load_clip_state_dict("ViT-B/32")
    model = build_fusion_model(sd, backbone="openai_clip", enable_dual_branch=False).to(device).eval()
    res = {}
    for ds in args.datasets:
        data_root, _ = sota.CFG[ds]
        Xtr, ytr = feats(model, data_root, "train", device)
        Xva, yva = feats(model, data_root, "val", device)
        Xte, yte = feats(model, data_root, "test", device)
        best, best_f1 = None, -1
        for C in [0.01, 0.05, 0.1, 0.5, 1.0, 5.0, 10.0]:
            clf = LogisticRegression(C=C, class_weight="balanced", max_iter=5000, multi_class="multinomial")
            clf.fit(Xtr, ytr)
            f1 = f1_score(yva, clf.predict(Xva), average="macro", zero_division=0)
            if f1 > best_f1:
                best_f1, best = f1, clf
        m = metrics(best.predict(Xte), yte)
        res[ds] = m
        print(f"{ds}: {json.dumps(m)}")
    out = "experiments/results/baseline_compare/clip_openai_linearprobe.json"
    json.dump(res, open(out, "w"), indent=2)
    print("saved", out)


if __name__ == "__main__":
    main()
