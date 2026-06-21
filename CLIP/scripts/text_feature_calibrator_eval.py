"""
Text-feature decision calibrator.

Frozen full-dual model. Extract base logits plus text-derived logits, then train
a regularized multinomial logistic calibrator on the train split and select C on
the validation split. This is a stronger text-side decision adapter than TARC,
while still using no test labels.
"""
import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader

from clip.model import build_fusion_model
from data.dataset import ClassificationDataset, get_transforms
from main import load_clip_state_dict


CFG = {
    "me": ("../公开数据集/MedicalExpert-split",
           "experiments/checkpoints/checkpoints_dual_me_ms_fix4/MedicalExpert-split_full_dual/best_model.pth"),
    "ar": ("../公开数据集/archive",
           "experiments/checkpoints/checkpoints_dual_ar_ms_fix10/archive_full_dual/best_model.pth"),
    "pv": ("../私有数据集/split_result_siyouceshi_roi",
           "experiments/checkpoints/checkpoints_dual_pv_ms_fix9/split_result_siyouceshi_roi_full_dual/best_model.pth"),
}


def transition_to_class_logits(transition_logits):
    probs = torch.sigmoid(transition_logits)
    ones = torch.ones(probs.size(0), 1, device=probs.device)
    zeros = torch.zeros(probs.size(0), 1, device=probs.device)
    cdf = torch.cat([ones, probs, zeros], dim=1)
    return (cdf[:, :-1] - cdf[:, 1:]).clamp(min=1e-9).log()


def metrics_from_pred(pred, labels):
    return {
        "accuracy": float(accuracy_score(labels, pred)),
        "macro_f1": float(f1_score(labels, pred, average="macro", zero_division=0)),
        "mae": float(np.abs(pred - labels).mean()),
    }


@torch.no_grad()
def extract(model, data_root, split, device, batch_size):
    tf = get_transforms(img_size=224, is_training=False, normalize=True)
    ds = ClassificationDataset(data_root, split=split, transform=tf)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=4)

    anat_text = F.normalize(model.global_text_feat.float(), dim=-1)
    path_text = F.normalize(model.local_text_feat.float(), dim=-1)
    trans_text = F.normalize(model.transition_text_feat.float(), dim=-1)
    temp = model.TEMPERATURE

    feats, labels, base_logits = [], [], []
    for batch in loader:
        image = batch["image"].to(device)
        out = model(image)
        base = out["logits_classifier"].float()
        fuse = F.normalize(out["img_feat"].float(), dim=-1)
        vit = F.normalize(model.encode_image(image).float(), dim=-1)
        cvx = F.normalize(model.visual_local(image).float(), dim=-1)
        proto = out["logits_proto"].float()
        anatomy = vit @ anat_text.T / temp
        pathology = cvx @ path_text.T / temp
        transition = transition_to_class_logits(fuse @ trans_text.T / temp)
        prob = F.softmax(base, dim=-1)
        top2 = prob.topk(2, dim=-1).values
        stats = torch.stack([
            top2[:, 0],
            top2[:, 0] - top2[:, 1],
            -(prob * prob.clamp(min=1e-9).log()).sum(dim=-1),
        ], dim=1)
        feat = torch.cat([base, proto, anatomy, pathology, transition, stats], dim=1)
        feats.append(feat.cpu().numpy())
        base_logits.append(base.cpu())
        labels.extend(batch["label"].numpy())
    return np.concatenate(feats), np.asarray(labels), torch.cat(base_logits)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, choices=CFG.keys())
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--out", default="experiments/results/text_feature_calibrator")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_root, ckpt = CFG[args.dataset]
    sd = load_clip_state_dict("ViT-B/32")
    model = build_fusion_model(
        sd, backbone="biomedclip", enable_dual_branch=True, convnext_multi_scale=True
    ).to(device).eval()
    raw = torch.load(ckpt, map_location="cpu")
    model.load_state_dict(raw.get("model_state_dict", raw), strict=False)
    model.refresh_text_prototypes()

    xtr, ytr, _ = extract(model, data_root, "train", device, args.batch_size)
    xv, yv, bv = extract(model, data_root, "val", device, args.batch_size)
    xt, yt, bt = extract(model, data_root, "test", device, args.batch_size)

    base_val = metrics_from_pred(bv.argmax(-1).numpy(), yv)
    base_test = metrics_from_pred(bt.argmax(-1).numpy(), yt)

    best = None
    for c in [0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0]:
        clf = make_pipeline(
            StandardScaler(),
            LogisticRegression(
                C=c,
                penalty="l2",
                solver="lbfgs",
                max_iter=2000,
                multi_class="multinomial",
                class_weight=None,
            ),
        )
        clf.fit(xtr, ytr)
        pv = clf.predict(xv)
        vm = metrics_from_pred(pv, yv)
        key = (vm["accuracy"], vm["macro_f1"], -vm["mae"])
        if best is None or key > best["key"]:
            best = {"key": key, "c": c, "clf": clf, "val": vm}

    pt = best["clf"].predict(xt)
    tm = metrics_from_pred(pt, yt)
    result = {
        "dataset": args.dataset,
        "base_val": base_val,
        "base_test": base_test,
        "selected_C": best["c"],
        "calibrator_val": best["val"],
        "calibrator_test": tm,
        "delta_test": {k: tm[k] - base_test[k] for k in ["accuracy", "macro_f1", "mae"]},
    }
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"{args.dataset}.json"
    path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(result, indent=2, ensure_ascii=False))
    print(f"saved: {path}")


if __name__ == "__main__":
    main()

