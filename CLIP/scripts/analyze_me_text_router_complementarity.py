"""
Analyze why the ME text-curriculum expert router improves over the baseline.

The script recomputes test logits for the baseline and text-supervised experts,
applies the validation-selected router weights, and reports:
  - corrected / harmed sample counts;
  - class-pair changes;
  - ordinal error changes;
  - expert complementarity for baseline-wrong cases.
"""
import json
import os
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import confusion_matrix
from torch.utils.data import DataLoader

from clip.model import build_fusion_model
from data.dataset import ClassificationDataset, get_transforms
from main import load_clip_state_dict


DATA_ROOT = "../公开数据集/MedicalExpert-split"
RESULT_PATH = "experiments/results/semantic_ordinal_transport_adapter_fine/me.json"


def zscore(logits):
    return (logits - logits.mean(dim=1, keepdim=True)) / logits.std(dim=1, keepdim=True).clamp(min=1e-6)


@torch.no_grad()
def collect(ckpt, device):
    sd = load_clip_state_dict("ViT-B/32")
    model = build_fusion_model(
        sd, backbone="biomedclip", enable_dual_branch=True, convnext_multi_scale=True
    ).to(device).eval()
    raw = torch.load(ckpt, map_location="cpu")
    model.load_state_dict(raw.get("model_state_dict", raw), strict=False)
    model.refresh_text_prototypes()

    tf = get_transforms(
        img_size=224,
        is_training=False,
        use_clahe=True,
        to_rgb=True,
        clahe_clip_limit=2.0,
        clahe_tile_grid_size=(8, 8),
        percentile=(1.0, 99.0),
        normalize=True,
    )
    ds = ClassificationDataset(DATA_ROOT, split="test", transform=tf)
    loader = DataLoader(ds, batch_size=64, shuffle=False, num_workers=4)
    logits, labels = [], []
    for batch in loader:
        out = model(batch["image"].to(device))
        logits.append(out["logits_classifier"].float().cpu())
        labels.extend(batch["label"].numpy())
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return torch.cat(logits), np.asarray(labels)


def main():
    with open(RESULT_PATH, "r", encoding="utf-8") as f:
        result = json.load(f)
    router = result["selected"]["expert_router"]
    names = router["names"]
    weights = torch.tensor(router["weights"], dtype=torch.float32)

    ckpts = [
        "experiments/checkpoints/checkpoints_dual_me_ms_fix4/MedicalExpert-split_full_dual/best_model.pth",
        "experiments/checkpoints/checkpoints_universal_text_me_gtp/MedicalExpert-split_full_dual/best_model.pth",
        "experiments/checkpoints/checkpoints_text_curriculum_me_lt005_stage2/MedicalExpert-split_full_dual/best_model.pth",
        "experiments/checkpoints/checkpoints_text_curriculum_me_stage2/MedicalExpert-split_full_dual/best_model.pth",
    ]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logits, labels_ref = [], None
    for name, ckpt in zip(names, ckpts):
        print(f"[collect] {name}")
        lg, labels = collect(ckpt, device)
        labels_ref = labels if labels_ref is None else labels_ref
        if not np.array_equal(labels_ref, labels):
            raise RuntimeError(f"label mismatch for {name}")
        logits.append(zscore(lg))

    stack = torch.stack(logits)
    base_pred = stack[0].argmax(dim=-1).numpy()
    ens_pred = (stack * weights.view(-1, 1, 1)).sum(dim=0).argmax(dim=-1).numpy()
    labels = labels_ref

    base_correct = base_pred == labels
    ens_correct = ens_pred == labels
    corrected = np.where((~base_correct) & ens_correct)[0]
    harmed = np.where(base_correct & (~ens_correct))[0]
    unchanged_wrong = np.where((~base_correct) & (~ens_correct) & (base_pred == ens_pred))[0]
    changed_wrong = np.where((~base_correct) & (~ens_correct) & (base_pred != ens_pred))[0]

    expert_preds = [x.argmax(dim=-1).numpy() for x in stack]
    baseline_wrong = np.where(~base_correct)[0]
    oracle_correct = 0
    for i in baseline_wrong:
        if any(p[i] == labels[i] for p in expert_preds[1:]):
            oracle_correct += 1

    transitions = Counter()
    for i in np.where(base_pred != ens_pred)[0]:
        transitions[(int(labels[i]), int(base_pred[i]), int(ens_pred[i]))] += 1

    summary = {
        "num_test": int(len(labels)),
        "baseline_correct": int(base_correct.sum()),
        "router_correct": int(ens_correct.sum()),
        "net_gain": int(ens_correct.sum() - base_correct.sum()),
        "corrected_count": int(len(corrected)),
        "harmed_count": int(len(harmed)),
        "unchanged_wrong_count": int(len(unchanged_wrong)),
        "changed_but_still_wrong_count": int(len(changed_wrong)),
        "baseline_mae": float(np.abs(base_pred - labels).mean()),
        "router_mae": float(np.abs(ens_pred - labels).mean()),
        "mean_abs_error_delta": float(np.abs(ens_pred - labels).mean() - np.abs(base_pred - labels).mean()),
        "baseline_wrong_with_text_expert_oracle_correct": int(oracle_correct),
        "baseline_wrong_count": int(len(baseline_wrong)),
        "changed_prediction_triples_label_base_router": [
            {"label": k[0], "baseline": k[1], "router": k[2], "count": v}
            for k, v in transitions.most_common()
        ],
        "baseline_confusion": confusion_matrix(labels, base_pred, labels=[0, 1, 2, 3, 4]).tolist(),
        "router_confusion": confusion_matrix(labels, ens_pred, labels=[0, 1, 2, 3, 4]).tolist(),
    }

    out = Path("experiments/results/router_complementarity/me_text_router_analysis.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"saved: {out}")


if __name__ == "__main__":
    main()
