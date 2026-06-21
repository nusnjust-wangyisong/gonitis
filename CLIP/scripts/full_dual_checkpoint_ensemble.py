"""
Validation-selected ensemble for full_dual multi-scale checkpoints.

This is intended for testing whether a text-supervised checkpoint is
complementary to the visual full baseline even when its standalone score drops.
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
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import DataLoader

from clip.model import build_fusion_model
from data.dataset import ClassificationDataset, get_transforms
from main import load_clip_state_dict


def metrics(logits, labels):
    pred = logits.argmax(dim=-1).cpu().numpy()
    return {
        "accuracy": float(accuracy_score(labels, pred)),
        "macro_f1": float(f1_score(labels, pred, average="macro", zero_division=0)),
        "mae": float(np.abs(pred - labels).mean()),
    }


def normalize(logits, mode):
    if mode == "none":
        return logits.float()
    if mode == "zscore":
        return (logits - logits.mean(dim=1, keepdim=True)) / logits.std(dim=1, keepdim=True).clamp(min=1e-6)
    if mode == "prob":
        return F.softmax(logits.float(), dim=-1)
    raise ValueError(mode)


@torch.no_grad()
def collect(ckpt, data_root, split, device, batch_size, num_workers, normalize_mode):
    sd = load_clip_state_dict("ViT-B/32")
    model = build_fusion_model(
        sd,
        backbone="biomedclip",
        enable_dual_branch=True,
        convnext_multi_scale=True,
    ).to(device).eval()
    raw = torch.load(ckpt, map_location="cpu")
    missing, unexpected = model.load_state_dict(raw.get("model_state_dict", raw), strict=False)
    if missing or unexpected:
        print(f"[load] {ckpt}: missing={len(missing)} unexpected={len(unexpected)}")
    model.refresh_text_prototypes()

    tf = get_transforms(img_size=224, is_training=False, normalize=True)
    ds = ClassificationDataset(data_root, split=split, transform=tf)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    logits, labels = [], []
    for batch in loader:
        out = model(batch["image"].to(device))
        logits.append(normalize(out["logits_cls"].float().cpu(), normalize_mode))
        labels.extend(batch["label"].numpy())
    del model
    torch.cuda.empty_cache()
    return torch.cat(logits), np.asarray(labels)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", required=True)
    parser.add_argument("--checkpoints", nargs="+", required=True)
    parser.add_argument("--names", nargs="+", default=None)
    parser.add_argument("--normalize", choices=["none", "zscore", "prob"], default="zscore")
    parser.add_argument("--step", type=float, default=0.02)
    parser.add_argument("--objective", choices=["accuracy", "macro_f1"], default="macro_f1")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--out", default="experiments/results/full_dual_checkpoint_ensemble/result.json")
    args = parser.parse_args()

    names = args.names or [f"m{i}" for i in range(len(args.checkpoints))]
    if len(names) != len(args.checkpoints):
        raise ValueError("--names length must match --checkpoints length")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    val_logits, test_logits = [], []
    yv = yt = None
    per_model = {}
    for name, ckpt in zip(names, args.checkpoints):
        print(f"[collect] {name}: {ckpt}")
        vl, labels_v = collect(ckpt, args.data_root, "val", device, args.batch_size, args.num_workers, args.normalize)
        tl, labels_t = collect(ckpt, args.data_root, "test", device, args.batch_size, args.num_workers, args.normalize)
        yv = labels_v if yv is None else yv
        yt = labels_t if yt is None else yt
        assert np.array_equal(yv, labels_v)
        assert np.array_equal(yt, labels_t)
        val_logits.append(vl)
        test_logits.append(tl)
        per_model[name] = {"val": metrics(vl, yv), "test": metrics(tl, yt)}

    val_stack = torch.stack(val_logits)
    test_stack = torch.stack(test_logits)
    n = len(args.checkpoints)
    units = int(round(1.0 / args.step))
    best = None
    # Two-model path is common; support generic simplex by recursion-lite random-free grid.
    def grids(k, remain):
        if k == n - 1:
            yield [remain]
        else:
            for i in range(remain + 1):
                for rest in grids(k + 1, remain - i):
                    yield [i] + rest

    for counts in grids(0, units):
        w = torch.tensor([c / units for c in counts], dtype=torch.float32)
        vl = (val_stack * w.view(-1, 1, 1)).sum(dim=0)
        m = metrics(vl, yv)
        key = (m[args.objective], m["accuracy"], m["macro_f1"], -m["mae"])
        if best is None or key > best["key"]:
            best = {"key": key, "weights": w, "val": m}

    ens_test = (test_stack * best["weights"].view(-1, 1, 1)).sum(dim=0)
    test_m = metrics(ens_test, yt)
    result = {
        "names": names,
        "checkpoints": args.checkpoints,
        "normalize": args.normalize,
        "objective": args.objective,
        "per_model": per_model,
        "selected_weights": [float(x) for x in best["weights"].tolist()],
        "ensemble_val": best["val"],
        "ensemble_test": test_m,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(result, indent=2, ensure_ascii=False))
    print(f"saved: {out}")


if __name__ == "__main__":
    main()

