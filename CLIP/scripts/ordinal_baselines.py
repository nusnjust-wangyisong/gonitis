#!/usr/bin/env python3
"""Ordinal-regression baselines on frozen BiomedCLIP features:
softmax-CE (nominal) vs CORAL vs CORN. Same features -> isolates the ordinal head.
Reports test acc / macro-F1 / QWK / MAE per dataset; epoch by val QWK."""
import os, sys, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, f1_score, cohen_kappa_score

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
        X.append(model.encode_image(b["image"].to(device)).float().cpu()); y.extend(b["label"].numpy())
    return torch.cat(X), torch.as_tensor(np.asarray(y))


def met(pred, y):
    pred = pred.numpy(); y = y.numpy()
    return {"acc": round(float(accuracy_score(y, pred)), 4),
            "f1": round(float(f1_score(y, pred, average="macro", zero_division=0)), 4),
            "qwk": round(float(cohen_kappa_score(y, pred, weights="quadratic", labels=list(range(NC)))), 4),
            "mae": round(float(np.abs(pred - y).mean()), 4)}


class Softmax(nn.Module):
    def __init__(s, d): super().__init__(); s.fc = nn.Linear(d, NC)
    def forward(s, x): return s.fc(x)
    def loss(s, x, y): return F.cross_entropy(s(x), y)
    def predict(s, x): return s(x).argmax(-1)


class CORAL(nn.Module):
    def __init__(s, d): super().__init__(); s.w = nn.Linear(d, 1, bias=False); s.b = nn.Parameter(torch.zeros(NC - 1))
    def logits(s, x): return s.w(x) + s.b
    def loss(s, x, y):
        lv = (y.unsqueeze(1) > torch.arange(NC - 1, device=y.device)).float()
        return F.binary_cross_entropy_with_logits(s.logits(x), lv)
    def predict(s, x): return (torch.sigmoid(s.logits(x)) > 0.5).sum(1)


class CORN(nn.Module):
    def __init__(s, d): super().__init__(); s.fc = nn.Linear(d, NC - 1)
    def logits(s, x): return s.fc(x)
    def loss(s, x, y):
        lo = s.logits(x); tot = 0.0
        for k in range(NC - 1):
            mask = y > (k - 1) if k > 0 else torch.ones_like(y, dtype=torch.bool)
            if mask.sum() == 0:
                continue
            tot = tot + F.binary_cross_entropy_with_logits(lo[mask, k], (y[mask] > k).float())
        return tot / (NC - 1)
    def predict(s, x):
        p = torch.sigmoid(s.logits(x)); cp = torch.cumprod(p, dim=1)
        return (cp > 0.5).sum(1)


def train_head(Head, Xtr, ytr, Xva, yva, device, epochs=300, lr=1e-2):
    torch.manual_seed(0)
    h = Head(Xtr.shape[1]).to(device)
    opt = torch.optim.Adam(h.parameters(), lr=lr, weight_decay=1e-3)
    Xtr, ytr = Xtr.to(device), ytr.to(device)
    best, best_q = None, -2
    for _ in range(epochs):
        h.train(); opt.zero_grad(); h.loss(Xtr, ytr).backward(); opt.step()
        if _ % 10 == 0:
            h.eval()
            with torch.no_grad():
                q = cohen_kappa_score(yva.numpy(), h.predict(Xva.to(device)).cpu().numpy(),
                                      weights="quadratic", labels=list(range(NC)))
            if q > best_q:
                best_q, best = q, {k: v.detach().clone() for k, v in h.state_dict().items()}
    h.load_state_dict(best); h.eval()
    return h


def main():
    import argparse
    ap = argparse.ArgumentParser(); ap.add_argument("--datasets", nargs="+", default=["me", "ar", "pv"])
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    sd = load_clip_state_dict("ViT-B/32")
    model = build_fusion_model(sd, backbone="biomedclip", enable_dual_branch=False).to(device).eval()
    res = {}
    for ds in args.datasets:
        data_root, _ = sota.CFG[ds]
        Xtr, ytr = feats(model, data_root, "train", device)
        Xva, yva = feats(model, data_root, "val", device)
        Xte, yte = feats(model, data_root, "test", device)
        res[ds] = {}
        for name, Head in [("softmax-CE", Softmax), ("CORAL", CORAL), ("CORN", CORN)]:
            h = train_head(Head, Xtr, ytr, Xva, yva, device)
            with torch.no_grad():
                pred = h.predict(Xte.to(device)).cpu()
            res[ds][name] = met(pred, yte)
            print(f"{ds} {name}: {json.dumps(res[ds][name])}")
    json.dump(res, open("experiments/results/baseline_compare/ordinal_baselines.json", "w"), indent=2)
    print("saved ordinal_baselines.json")


if __name__ == "__main__":
    main()
