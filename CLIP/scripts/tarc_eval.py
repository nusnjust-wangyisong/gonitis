"""
TARC: Text-Anchored Residual Calibration.

Frozen full-dual baseline + a tiny text residual calibrator:

    final_logits = base_logits + gate(confidence, margin, entropy) *
                   sum_s alpha_s * norm(text_logits_s)

Text sources:
  - proto      : standard KL text prototypes
  - anatomy    : ViT/global branch aligned to anatomy texts
  - pathology  : ConvNeXt/local branch aligned to pathology texts
  - transition : ordinal transition texts converted to class logits

Only the calibrator parameters are optimized; the visual model is frozen.
The no-op solution is the initialization, so this tests whether text provides a
transferable residual signal without perturbing the full baseline.
"""
import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
import torch.nn as nn
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


def transition_to_class_logits(transition_logits):
    probs = torch.sigmoid(transition_logits)
    ones = torch.ones(probs.size(0), 1, device=probs.device)
    zeros = torch.zeros(probs.size(0), 1, device=probs.device)
    cdf = torch.cat([ones, probs, zeros], dim=1)
    return (cdf[:, :-1] - cdf[:, 1:]).clamp(min=1e-9).log()


def metrics(logits, labels):
    preds = logits.argmax(-1).cpu().numpy()
    return {
        "accuracy": float(accuracy_score(labels, preds)),
        "macro_f1": float(f1_score(labels, preds, average="macro", zero_division=0)),
        "mae": float(np.abs(preds - labels).mean()),
    }


def confidence_features(base_logits):
    probs = F.softmax(base_logits, dim=-1)
    top2 = probs.topk(2, dim=-1).values
    conf = top2[:, 0]
    margin = top2[:, 0] - top2[:, 1]
    entropy = -(probs * probs.clamp(min=1e-9).log()).sum(dim=-1) / np.log(probs.size(1))
    return torch.stack([1.0 - conf, 1.0 - margin, entropy], dim=1)


def extract(model, data_root, split, device, batch_size):
    tf = get_transforms(img_size=224, is_training=False, normalize=True)
    ds = ClassificationDataset(data_root, split=split, transform=tf)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=4)

    proto_text = F.normalize(model.text_feat.float(), dim=-1)
    anat_text = F.normalize(model.global_text_feat.float(), dim=-1)
    path_text = F.normalize(model.local_text_feat.float(), dim=-1)
    trans_text = F.normalize(model.transition_text_feat.float(), dim=-1)
    temp = model.TEMPERATURE

    base, sources, labels = [], [], []
    with torch.no_grad():
        for batch in loader:
            image = batch["image"].to(device)
            out = model(image)
            fuse = F.normalize(out["img_feat"].float(), dim=-1)
            vit = F.normalize(model.encode_image(image).float(), dim=-1)
            cvx = F.normalize(model.visual_local(image).float(), dim=-1)
            proto = out["logits_proto"].float()
            anatomy = vit @ anat_text.T / temp
            pathology = cvx @ path_text.T / temp
            transition = transition_to_class_logits(fuse @ trans_text.T / temp)
            base.append(out["logits_classifier"].float().cpu())
            sources.append(torch.stack([
                proto.cpu(),
                anatomy.cpu(),
                pathology.cpu(),
                transition.cpu(),
            ], dim=1))
            labels.extend(batch["label"].numpy())
    return {
        "base": torch.cat(base),
        "sources": torch.cat(sources),
        "features": confidence_features(torch.cat(base)),
        "labels": torch.as_tensor(np.asarray(labels), dtype=torch.long),
    }


class TARC(nn.Module):
    def __init__(self, num_sources=4):
        super().__init__()
        self.source_weight = nn.Parameter(torch.zeros(num_sources))
        self.gate = nn.Linear(3, 1)
        nn.init.zeros_(self.gate.weight)
        nn.init.constant_(self.gate.bias, -2.0)

    def forward(self, base, sources, features):
        residual = torch.einsum("s,bsc->bc", self.source_weight, sources)
        gate = torch.sigmoid(self.gate(features))
        return base + gate * residual


def standardize(train_sources, *others):
    mean = train_sources.mean(dim=0, keepdim=True)
    std = train_sources.std(dim=0, keepdim=True).clamp(min=1e-6)
    return [(x - mean) / std for x in (train_sources,) + others]


def train_calibrator(calib, val, epochs, lr, weight_decay, l2):
    model = TARC(num_sources=calib["sources"].shape[1])
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    best = {
        "key": (
            metrics(calib["base"], calib["labels"].numpy())["accuracy"],
            metrics(calib["base"], calib["labels"].numpy())["macro_f1"],
            -metrics(calib["base"], calib["labels"].numpy())["mae"],
        ),
        "state": {k: v.detach().clone() for k, v in model.state_dict().items()},
        "val_metrics": metrics(val["base"], val["labels"].numpy()),
    }
    for _ in range(epochs):
        logits = model(calib["base"], calib["sources"], calib["features"])
        loss = F.cross_entropy(logits, calib["labels"])
        loss = loss + l2 * model.source_weight.pow(2).sum()
        opt.zero_grad()
        loss.backward()
        opt.step()

        with torch.no_grad():
            val_logits = model(val["base"], val["sources"], val["features"])
            vm = metrics(val_logits, val["labels"].numpy())
            key = (vm["accuracy"], vm["macro_f1"], -vm["mae"])
            if key > best["key"]:
                best = {
                    "key": key,
                    "state": {k: v.detach().clone() for k, v in model.state_dict().items()},
                    "val_metrics": vm,
                }
    model.load_state_dict(best["state"])
    return model, best["val_metrics"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, choices=CFG.keys())
    parser.add_argument("--calib_split", choices=["train", "val"], default="val")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=800)
    parser.add_argument("--lr", type=float, default=0.02)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--l2", type=float, default=0.05)
    parser.add_argument("--out", default="experiments/results/tarc")
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

    train = extract(model, data_root, "train", device, args.batch_size)
    val = extract(model, data_root, "val", device, args.batch_size)
    test = extract(model, data_root, "test", device, args.batch_size)

    train_s, val_s, test_s = standardize(train["sources"], val["sources"], test["sources"])
    train["sources"], val["sources"], test["sources"] = train_s, val_s, test_s
    calib = train if args.calib_split == "train" else val
    select = val

    calibrator, val_metrics = train_calibrator(
        calib, select, args.epochs, args.lr, args.weight_decay, args.l2
    )
    with torch.no_grad():
        test_logits = calibrator(test["base"], test["sources"], test["features"])
        test_metrics = metrics(test_logits, test["labels"].numpy())
        source_weight = calibrator.source_weight.detach().cpu().tolist()
        gate_weight = calibrator.gate.weight.detach().cpu().view(-1).tolist()
        gate_bias = float(calibrator.gate.bias.detach().cpu().item())

    base = {
        "train": metrics(train["base"], train["labels"].numpy()),
        "val": metrics(val["base"], val["labels"].numpy()),
        "test": metrics(test["base"], test["labels"].numpy()),
    }
    result = {
        "dataset": args.dataset,
        "calib_split": args.calib_split,
        "base": base,
        "tarc_val": val_metrics,
        "tarc_test": test_metrics,
        "delta_test": {
            k: test_metrics[k] - base["test"][k] for k in ["accuracy", "macro_f1", "mae"]
        },
        "source_names": ["proto", "anatomy", "pathology", "transition"],
        "source_weight": source_weight,
        "gate_weight": gate_weight,
        "gate_bias": gate_bias,
    }

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{args.dataset}_{args.calib_split}.json"
    out_path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(result, indent=2, ensure_ascii=False))
    print(f"saved: {out_path}")


if __name__ == "__main__":
    main()

