"""
Frozen text residual gate evaluation.

The full-dual baseline is kept frozen. On the validation split, this script
selects a conservative text residual:

    logits = base_logits + alpha * I(max_prob < tau) * text_logits

The no-op candidate (alpha=0) is included. This tests whether text helps only on
uncertain samples without perturbing high-confidence visual decisions.
"""
import argparse
import itertools
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import DataLoader

from clip.model import build_fusion_model
from data.dataset import ClassificationDataset, get_transforms
from main import load_clip_state_dict


CFG = {
    "me": (
        "../公开数据集/MedicalExpert-split",
        "experiments/checkpoints/checkpoints_dual_me_ms_fix4/MedicalExpert-split_full_dual/best_model.pth",
    ),
    "ar": (
        "../公开数据集/archive",
        "experiments/checkpoints/checkpoints_dual_ar_ms_fix10/archive_full_dual/best_model.pth",
    ),
    "pv": (
        "../私有数据集/split_result_siyouceshi_roi",
        "experiments/checkpoints/checkpoints_dual_pv_ms_fix9/split_result_siyouceshi_roi_full_dual/best_model.pth",
    ),
}


def metrics(logits, labels):
    preds = logits.argmax(-1).cpu().numpy()
    return {
        "accuracy": float(accuracy_score(labels, preds)),
        "macro_f1": float(f1_score(labels, preds, average="macro", zero_division=0)),
        "mae": float(np.abs(preds - labels).mean()),
    }


def transition_to_class_logits(transition_logits):
    probs = torch.sigmoid(transition_logits)
    ones = torch.ones(probs.size(0), 1, device=probs.device)
    zeros = torch.zeros(probs.size(0), 1, device=probs.device)
    cdf = torch.cat([ones, probs, zeros], dim=1)
    return (cdf[:, :-1] - cdf[:, 1:]).clamp(min=1e-9).log()


def extract(model, data_root, split, device, batch_size):
    tf = get_transforms(img_size=224, is_training=False, normalize=True)
    ds = ClassificationDataset(data_root, split=split, transform=tf)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=4)

    proto_text = F.normalize(model.text_feat.float(), dim=-1)
    anat_text = F.normalize(model.global_text_feat.float(), dim=-1)
    path_text = F.normalize(model.local_text_feat.float(), dim=-1)
    trans_text = F.normalize(model.transition_text_feat.float(), dim=-1)
    temp = model.TEMPERATURE

    base, proto, branch, trans, ys = [], [], [], [], []
    with torch.no_grad():
        for batch in loader:
            image = batch["image"].to(device)
            out = model(image)
            feat = F.normalize(out["img_feat"].float(), dim=-1)
            base.append(out["logits_classifier"].float().cpu())
            proto.append((feat @ proto_text.T / temp).cpu())
            branch.append(((feat @ anat_text.T / temp) + (feat @ path_text.T / temp)).mul(0.5).cpu())
            trans.append(transition_to_class_logits(feat @ trans_text.T / temp).cpu())
            ys.extend(batch["label"].numpy())
    return {
        "base": torch.cat(base),
        "proto": torch.cat(proto),
        "branch": torch.cat(branch),
        "transition": torch.cat(trans),
        "labels": np.asarray(ys),
    }


def gated_logits(base, text, alpha, tau):
    conf = F.softmax(base, dim=-1).max(dim=-1).values
    gate = (conf < tau).float().unsqueeze(1)
    return base + alpha * gate * text


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, choices=CFG.keys())
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--out", default="experiments/results/text_residual_gate")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_root, ckpt = CFG[args.dataset]

    sd = load_clip_state_dict("ViT-B/32")
    model = build_fusion_model(
        sd,
        backbone="biomedclip",
        enable_dual_branch=True,
        convnext_multi_scale=True,
    ).to(device).eval()
    raw = torch.load(ckpt, map_location="cpu")
    model.load_state_dict(raw.get("model_state_dict", raw), strict=False)
    model.refresh_text_prototypes()

    val = extract(model, data_root, "val", device, args.batch_size)
    test = extract(model, data_root, "test", device, args.batch_size)

    base_val = metrics(val["base"], val["labels"])
    base_test = metrics(test["base"], test["labels"])

    text_names = ["proto", "branch", "transition"]
    alpha_grid = [0.0, 0.02, 0.05, 0.08, 0.1, 0.15, 0.2, 0.3, 0.5]
    tau_grid = [0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.01]

    best = None
    for name, alpha, tau in itertools.product(text_names, alpha_grid, tau_grid):
        logits = gated_logits(val["base"], val[name], alpha, tau)
        m = metrics(logits, val["labels"])
        # Prefer accuracy, then macro-F1, then lower MAE. alpha=0 remains available.
        key = (m["accuracy"], m["macro_f1"], -m["mae"])
        if best is None or key > best["key"]:
            best = {"key": key, "name": name, "alpha": alpha, "tau": tau, "val": m}

    test_logits = gated_logits(test["base"], test[best["name"]], best["alpha"], best["tau"])
    test_m = metrics(test_logits, test["labels"])
    result = {
        "dataset": args.dataset,
        "base_val": base_val,
        "base_test": base_test,
        "selected": {
            "text": best["name"],
            "alpha": best["alpha"],
            "tau": best["tau"],
            "val": best["val"],
            "test": test_m,
        },
        "delta_test": {
            k: test_m[k] - base_test[k] for k in ["accuracy", "macro_f1", "mae"]
        },
    }

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{args.dataset}.json"
    out_path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")

    print(json.dumps(result, indent=2, ensure_ascii=False))
    print(f"saved: {out_path}")


if __name__ == "__main__":
    main()

