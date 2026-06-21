#!/usr/bin/env python3
"""Inference-time ablation: does the cluster-aware TEXT-prototype fusion in the
OCM-CLIP base actually contribute? Compare test metrics of the deployed model
with vs. without cluster_txt_feat (set to 0 -> fuse_feat = LayerNorm(img_feat)).
Also reports the mean learned gate (how much text is let into the fusion)."""
import os, sys, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, f1_score, cohen_kappa_score

import scripts.semantic_ordinal_transport_adapter_eval as sota
from clip.model import build_fusion_model
from data.dataset import ClassificationDataset, get_transforms
from main import load_clip_state_dict

NC = 5


def met(y, p):
    return {"acc": round(float(accuracy_score(y, p)), 4),
            "f1": round(float(f1_score(y, p, average="macro", zero_division=0)), 4),
            "qwk": round(float(cohen_kappa_score(y, p, weights="quadratic", labels=list(range(NC)))), 4),
            "mae": round(float(np.abs(np.asarray(p) - np.asarray(y)).mean()), 4)}


@torch.no_grad()
def collect(model, data_root, device, ablate, gate_acc):
    tf = get_transforms(img_size=224, is_training=False, use_clahe=True, to_rgb=True,
                        clahe_clip_limit=2.0, clahe_tile_grid_size=(8, 8),
                        percentile=(1.0, 99.0), normalize=True)
    loader = DataLoader(ClassificationDataset(data_root, split="test", transform=tf),
                        batch_size=64, shuffle=False, num_workers=4)
    model._ablate_cluster_text = ablate
    model._gate_store = gate_acc
    P, Y = [], []
    for b in loader:
        out = model(b["image"].to(device))
        P.extend(out["logits_classifier"].argmax(-1).cpu().numpy()); Y.extend(b["label"].numpy())
    return np.asarray(Y), np.asarray(P)


def patch(model):
    """Wrap fusion.forward: optionally zero cluster_txt_feat; record mean gate."""
    fusion = model.fusion
    orig = fusion.forward

    def fwd(img_feat, cluster_txt_feat, cluster_prob, *a, **k):
        # record mean gate magnitude on cluster_txt_feat
        inter = img_feat * cluster_txt_feat
        gi = torch.cat([img_feat, cluster_txt_feat, inter, cluster_prob], dim=-1)
        g = fusion.gate_mlp(torch.nan_to_num(gi))
        if getattr(model, "_gate_store", None) is not None:
            model._gate_store.append(float(g.mean()))
        if getattr(model, "_ablate_cluster_text", False):
            cluster_txt_feat = torch.zeros_like(cluster_txt_feat)
        return orig(img_feat, cluster_txt_feat, cluster_prob, *a, **k)
    fusion.forward = fwd


def main():
    import argparse
    ap = argparse.ArgumentParser(); ap.add_argument("--datasets", nargs="+", default=["me", "ar", "pv"])
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    res = {}
    for ds in args.datasets:
        data_root, ckpt = sota.CFG[ds]
        sd = load_clip_state_dict("ViT-B/32")
        model = build_fusion_model(sd, backbone="biomedclip", enable_dual_branch=True,
                                   convnext_multi_scale=True).to(device).eval()
        raw = torch.load(ckpt, map_location="cpu")
        model.load_state_dict(raw.get("model_state_dict", raw), strict=False)
        model.refresh_text_prototypes()
        patch(model)
        gates = []
        y, p_full = collect(model, data_root, device, ablate=False, gate_acc=gates)
        _, p_abl = collect(model, data_root, device, ablate=True, gate_acc=None)
        res[ds] = {"with_cluster_text": met(y, p_full), "without_cluster_text": met(y, p_abl),
                   "mean_fusion_gate": round(float(np.mean(gates)), 4)}
        print(f"\n=== {ds} ===")
        print("  with text :", json.dumps(res[ds]["with_cluster_text"]))
        print("  w/o  text :", json.dumps(res[ds]["without_cluster_text"]))
        print("  mean fusion gate on cluster_txt_feat =", res[ds]["mean_fusion_gate"])
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    json.dump(res, open("experiments/results/baseline_compare/cluster_text_ablation.json", "w"), indent=2)
    print("\nsaved cluster_text_ablation.json")


if __name__ == "__main__":
    main()
