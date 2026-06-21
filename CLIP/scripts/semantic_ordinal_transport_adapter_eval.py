"""
Semantic Ordinal Transport Adapter (SOTA) for full-dual KL grading.

Frozen multimodal adapter:
  1) KL text prototypes define a semantic label graph;
  2) train image features form a visual case-memory cache;
  3) cache evidence is transported through the text-defined label graph;
  4) transition texts produce ordinal boundary logits;
  5) validation selects a reliability-weighted fusion with the strong baseline.

No train/test labels are used for model selection. The no-op baseline is always
included in the grid, so validation selection can reject unreliable text evidence.
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


CFG = {
    "me": ("../公开数据集/MedicalExpert-split",
           "experiments/checkpoints/checkpoints_dual_me_ms_fix4/MedicalExpert-split_full_dual/best_model.pth"),
    "ar": ("../公开数据集/archive",
           "experiments/checkpoints/checkpoints_dual_ar_ms_fix10/archive_full_dual/best_model.pth"),
    "pv": ("../私有数据集/split_result_siyouceshi_roi",
           "experiments/checkpoints/checkpoints_dual_pv_ms_fix9/split_result_siyouceshi_roi_full_dual/best_model.pth"),
}


def metrics(logits, labels):
    pred = logits.argmax(dim=-1).cpu().numpy()
    return {
        "accuracy": float(accuracy_score(labels, pred)),
        "macro_f1": float(f1_score(labels, pred, average="macro", zero_division=0)),
        "mae": float(np.abs(pred - labels).mean()),
    }


def score_key(m, objective):
    return (m[objective], m["accuracy"], m["macro_f1"], -m["mae"])


def zscore(logits):
    return (logits - logits.mean(dim=1, keepdim=True)) / logits.std(dim=1, keepdim=True).clamp(min=1e-6)


def confidence_gate(base_logits, tau):
    if tau >= 1.0:
        return torch.ones(base_logits.size(0), 1)
    conf = torch.softmax(base_logits.float(), dim=-1).max(dim=-1).values
    return (conf < tau).float().unsqueeze(1)


def agreement_gate(base_logits, source_logits, tau, max_distance, min_margin):
    gate = confidence_gate(base_logits, tau)
    if max_distance >= 0:
        base_pred = base_logits.argmax(dim=-1)
        source_pred = source_logits.argmax(dim=-1)
        agree = (base_pred - source_pred).abs().le(max_distance).float().unsqueeze(1)
        gate = gate * agree
    if min_margin > 0:
        top2 = torch.topk(source_logits.float(), k=2, dim=-1).values
        margin = (top2[:, 0] - top2[:, 1]).float()
        gate = gate * margin.ge(min_margin).float().unsqueeze(1)
    return gate


def normalize_logits(logits):
    return zscore(logits.float())


@torch.no_grad()
def extract(model, data_root, split, device, batch_size, num_workers):
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
    ds = ClassificationDataset(data_root, split=split, transform=tf)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    text_proto = F.normalize(model.text_feat.detach().float(), dim=-1)
    anatomy_proto = F.normalize(model.global_text_feat.detach().float(), dim=-1)
    pathology_proto = F.normalize(model.local_text_feat.detach().float(), dim=-1)
    transition_proto = F.normalize(model.transition_text_feat.detach().float(), dim=-1)
    temp = float(model.TEMPERATURE)

    feats, base_logits, text_logits, anatomy_logits, pathology_logits, transition_evidence, labels = [], [], [], [], [], [], []
    for batch in loader:
        image = batch["image"].to(device)
        out = model(image)
        feat = F.normalize(out["img_feat"].float(), dim=-1)
        feats.append(feat.cpu())
        base_logits.append(out["logits_classifier"].float().cpu())
        text_logits.append((feat @ text_proto.T / temp).cpu())
        anatomy_logits.append((feat @ anatomy_proto.T / temp).cpu())
        pathology_logits.append((feat @ pathology_proto.T / temp).cpu())
        transition_evidence.append((feat @ transition_proto.T / temp).cpu())
        labels.extend(batch["label"].numpy())

    return {
        "feats": torch.cat(feats),
        "base": torch.cat(base_logits),
        "text": torch.cat(text_logits),
        "anatomy": torch.cat(anatomy_logits),
        "pathology": torch.cat(pathology_logits),
        "transition": torch.cat(transition_evidence),
        "labels": np.asarray(labels),
    }


@torch.no_grad()
def collect_logits_for_checkpoint(ckpt, data_root, split, device, batch_size, num_workers):
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
    ds = ClassificationDataset(data_root, split=split, transform=tf)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    logits, labels = [], []
    for batch in loader:
        out = model(batch["image"].to(device))
        logits.append(out["logits_classifier"].float().cpu())
        labels.extend(batch["label"].numpy())
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return torch.cat(logits), np.asarray(labels)


def simplex_weights(n, step):
    units = int(round(1.0 / step))

    def rec(k, remain):
        if k == n - 1:
            yield [remain]
        else:
            for i in range(remain + 1):
                for rest in rec(k + 1, remain - i):
                    yield [i] + rest

    for counts in rec(0, units):
        yield torch.tensor([c / units for c in counts], dtype=torch.float32)


def build_text_kernel(text_proto, temperature, ordinal_sigma, mix):
    num_classes = text_proto.size(0)
    sim = text_proto @ text_proto.T
    idx = torch.arange(num_classes, dtype=torch.float32)
    ordinal = torch.exp(-((idx[:, None] - idx[None, :]) ** 2) / (2.0 * ordinal_sigma ** 2))
    semantic = F.softmax(sim / temperature, dim=-1)
    ordinal = ordinal / ordinal.sum(dim=-1, keepdim=True).clamp(min=1e-6)
    kernel = (1.0 - mix) * torch.eye(num_classes) + mix * (0.5 * semantic + 0.5 * ordinal)
    return kernel / kernel.sum(dim=-1, keepdim=True).clamp(min=1e-6)


def semantic_cache_logits(query_feats, cache_feats, cache_labels, text_kernel, beta, balanced):
    onehot = F.one_hot(torch.as_tensor(cache_labels, dtype=torch.long), num_classes=text_kernel.size(0)).float()
    label_evidence = onehot @ text_kernel
    counts = onehot.sum(dim=0).clamp(min=1.0)
    if balanced:
        label_evidence = label_evidence / counts[torch.as_tensor(cache_labels, dtype=torch.long)].view(-1, 1)

    out = []
    chunk = 512
    for start in range(0, query_feats.size(0), chunk):
        q = query_feats[start:start + chunk]
        sim = q @ cache_feats.T
        affinity = torch.exp(beta * (sim - 1.0))
        out.append(affinity @ label_evidence)
    return torch.cat(out, dim=0)


def build_visual_prototypes(feats, labels, num_classes=5):
    y = torch.as_tensor(labels)
    protos = []
    for c in range(num_classes):
        mask = y == c
        if mask.any():
            protos.append(feats[mask].mean(dim=0))
        else:
            protos.append(torch.zeros(feats.size(1)))
    return F.normalize(torch.stack(protos), dim=-1)


def prototype_logits(feats, visual_proto, text_proto, alpha, scale):
    if torch.is_tensor(alpha):
        alpha_t = alpha.view(-1, 1).to(dtype=visual_proto.dtype)
    else:
        alpha_t = torch.full((visual_proto.size(0), 1), float(alpha), dtype=visual_proto.dtype)
    proto = F.normalize((1.0 - alpha_t) * visual_proto + alpha_t * text_proto, dim=-1)
    return scale * feats @ proto.T


def ordinal_boundary_logits(transition_evidence, scale=1.0, sign=1.0):
    evidence = sign * scale * transition_evidence.float()
    above = F.logsigmoid(evidence)
    below = F.logsigmoid(-evidence)
    scores = []
    num_classes = evidence.size(1) + 1
    for c in range(num_classes):
        parts = []
        if c > 0:
            parts.append(above[:, :c].sum(dim=1))
        if c < num_classes - 1:
            parts.append(below[:, c:].sum(dim=1))
        scores.append(sum(parts) if parts else torch.zeros(evidence.size(0)))
    return torch.stack(scores, dim=1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, choices=CFG.keys())
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--objective", choices=["accuracy", "macro_f1"], default="macro_f1")
    parser.add_argument("--final_cache_trainval", action="store_true")
    parser.add_argument("--extra_checkpoints", nargs="*", default=[],
                        help="Optional text-supervised expert checkpoints for semantic routing.")
    parser.add_argument("--extra_names", nargs="*", default=None)
    parser.add_argument("--expert_step", type=float, default=0.05)
    parser.add_argument("--agreement_gate", action="store_true",
                        help="Route each modality source only when it is ordinally consistent with the baseline.")
    parser.add_argument("--fixed_distance", type=int, default=None,
                        help="Fix agreement max distance instead of grid-searching it.")
    parser.add_argument("--fixed_margin", type=float, default=None,
                        help="Fix source margin threshold instead of grid-searching it.")
    parser.add_argument("--out", default="experiments/results/semantic_ordinal_transport_adapter")
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

    train = extract(model, data_root, "train", device, args.batch_size, args.num_workers)
    val = extract(model, data_root, "val", device, args.batch_size, args.num_workers)
    test = extract(model, data_root, "test", device, args.batch_size, args.num_workers)
    text_proto = F.normalize(model.text_feat.detach().float().cpu(), dim=-1)

    base_val = metrics(val["base"], val["labels"])
    base_test = metrics(test["base"], test["labels"])

    extra_val_logits, extra_test_logits, per_expert = [], [], {}
    extra_names = args.extra_names or [f"expert{i}" for i in range(len(args.extra_checkpoints))]
    if len(extra_names) != len(args.extra_checkpoints):
        raise ValueError("--extra_names length must match --extra_checkpoints")
    for name, extra_ckpt in zip(extra_names, args.extra_checkpoints):
        print(f"[extra] collect {name}: {extra_ckpt}")
        vl, vy = collect_logits_for_checkpoint(extra_ckpt, data_root, "val", device, args.batch_size, args.num_workers)
        tl, ty = collect_logits_for_checkpoint(extra_ckpt, data_root, "test", device, args.batch_size, args.num_workers)
        if not np.array_equal(vy, val["labels"]):
            raise ValueError(f"val labels mismatch for {name}")
        if not np.array_equal(ty, test["labels"]):
            raise ValueError(f"test labels mismatch for {name}")
        extra_val_logits.append(vl)
        extra_test_logits.append(tl)
        per_expert[name] = {"val": metrics(vl, val["labels"]), "test": metrics(tl, test["labels"])}

    beta_grid = [5.0, 10.0, 20.0, 40.0]
    kernel_temp_grid = [0.02, 0.1]
    kernel_mix_grid = [0.0, 0.5, 1.0]
    ordinal_sigma_grid = [0.75, 1.5]
    weight_grid = [0.0, 0.05, 0.1, 0.2, 0.3]
    fuse_weight_grid = [0.0, 0.05, 0.1, 0.2, 0.3]
    branch_weight_grid = [0.0, 0.05, 0.1]
    ord_scale_grid = [0.5, 1.0, 2.0, 5.0]
    tau_grid = [0.5, 0.6, 0.7, 0.8, 0.9, 1.01]
    if not args.agreement_gate:
        distance_grid = [4]
        margin_grid = [0.0]
    else:
        distance_grid = [args.fixed_distance] if args.fixed_distance is not None else [0, 1, 2, 4]
        margin_grid = [args.fixed_margin] if args.fixed_margin is not None else [0.0, 0.25, 0.5]

    # Stage 1: select the text-transported cache source.
    best_cache = None
    for balanced in [False, True]:
        for beta in beta_grid:
            for kt in kernel_temp_grid:
                for sigma in ordinal_sigma_grid:
                    for kmix in kernel_mix_grid:
                        kernel = build_text_kernel(text_proto, temperature=kt, ordinal_sigma=sigma, mix=kmix)
                        va_cache = semantic_cache_logits(
                            val["feats"], train["feats"], train["labels"], kernel, beta=beta, balanced=balanced
                        )
                        for wc in weight_grid:
                            residual = wc * zscore(va_cache)
                            for tau in tau_grid:
                                for dist in distance_grid:
                                    for margin in margin_grid:
                                        gate = agreement_gate(val["base"], va_cache, tau, dist, margin)
                                        final = zscore(val["base"]) + gate * residual
                                        m = metrics(final, val["labels"])
                                        key = score_key(m, args.objective)
                                        if best_cache is None or key > best_cache["key"]:
                                            best_cache = {
                                                "key": key,
                                                "balanced_cache": balanced,
                                                "beta": beta,
                                                "kernel_temperature": kt,
                                                "kernel_mix": kmix,
                                                "ordinal_sigma": sigma,
                                                "cache_weight": wc,
                                                "tau": tau,
                                                "distance": dist,
                                                "margin": margin,
                                                "val": m,
                                                "val_logits": va_cache,
                                            }

    # Stage 2: select the transition-text ordinal boundary source.
    best_ord = None
    for sign in [-1.0, 1.0]:
        for oscale in ord_scale_grid:
            va_ord = ordinal_boundary_logits(val["transition"], scale=oscale, sign=sign)
            for wo in weight_grid:
                residual = wo * zscore(va_ord)
                for tau in tau_grid:
                    for dist in distance_grid:
                        for margin in margin_grid:
                            gate = agreement_gate(val["base"], va_ord, tau, dist, margin)
                            final = zscore(val["base"]) + gate * residual
                            m = metrics(final, val["labels"])
                            key = score_key(m, args.objective)
                            if best_ord is None or key > best_ord["key"]:
                                best_ord = {
                                    "key": key,
                                    "ordinal_sign": sign,
                                    "ordinal_scale": oscale,
                                    "ordinal_weight": wo,
                                    "tau": tau,
                                    "distance": dist,
                                    "margin": margin,
                                    "val": m,
                                    "val_logits": va_ord,
                                }

    # Stage 3: select the category-level text-visual prototype source.
    visual_proto = build_visual_prototypes(train["feats"], train["labels"])
    alpha_grid = [0.0, 0.05, 0.1, 0.2, 0.3, 0.5, 0.8]
    proto_scale_grid = [1.0, 2.0, 5.0, 10.0, 20.0]
    best_proto = None
    for alpha in alpha_grid:
        for pscale in proto_scale_grid:
            va_proto = prototype_logits(val["feats"], visual_proto, text_proto, alpha, pscale)
            for wp in weight_grid:
                for tau in tau_grid:
                    for dist in distance_grid:
                        for margin in margin_grid:
                            gate = agreement_gate(val["base"], va_proto, tau, dist, margin)
                            final = zscore(val["base"]) + gate * (wp * zscore(va_proto))
                            m = metrics(final, val["labels"])
                            key = score_key(m, args.objective)
                            if best_proto is None or key > best_proto["key"]:
                                best_proto = {
                                    "key": key,
                                    "alpha": alpha,
                                    "alpha_per_class": None,
                                    "proto_scale": pscale,
                                    "proto_weight": wp,
                                    "tau": tau,
                                    "distance": dist,
                                    "margin": margin,
                                    "val": m,
                                    "val_logits": va_proto,
                                }

    if best_proto["proto_weight"] > 0:
        alpha_vec = torch.full((5,), float(best_proto["alpha"]))
        local_best = best_proto
        for _ in range(3):
            improved = False
            for c in range(5):
                class_best = local_best
                class_alpha = alpha_vec.clone()
                for alpha in alpha_grid:
                    cand_alpha = alpha_vec.clone()
                    cand_alpha[c] = alpha
                    va_proto = prototype_logits(
                        val["feats"], visual_proto, text_proto, cand_alpha, local_best["proto_scale"]
                    )
                    gate = agreement_gate(
                        val["base"], va_proto, local_best["tau"],
                        local_best["distance"], local_best["margin"]
                    )
                    final = zscore(val["base"]) + gate * (
                        local_best["proto_weight"] * zscore(va_proto)
                    )
                    m = metrics(final, val["labels"])
                    key = score_key(m, args.objective)
                    if key > class_best["key"]:
                        class_best = {
                            "key": key,
                            "alpha": None,
                            "alpha_per_class": cand_alpha.clone(),
                            "proto_scale": local_best["proto_scale"],
                            "proto_weight": local_best["proto_weight"],
                            "tau": local_best["tau"],
                            "distance": local_best["distance"],
                            "margin": local_best["margin"],
                            "val": m,
                            "val_logits": va_proto,
                        }
                        class_alpha = cand_alpha
                if not torch.equal(class_alpha, alpha_vec):
                    alpha_vec = class_alpha
                    local_best = class_best
                    improved = True
            if not improved:
                break
        best_proto = local_best

    # Stage 3: low-dimensional reliability fusion of the selected sources plus direct text sources.
    best = None
    va_cache = best_cache["val_logits"]
    va_ord = best_ord["val_logits"]
    va_proto = best_proto["val_logits"]
    for wc in fuse_weight_grid:
        for wt in fuse_weight_grid:
            for wo in fuse_weight_grid:
                for wp in fuse_weight_grid:
                    for wb in branch_weight_grid:
                        for tau in tau_grid:
                            for dist in distance_grid:
                                for margin in margin_grid:
                                    residual = (
                                        wc * agreement_gate(val["base"], va_cache, tau, dist, margin) * zscore(va_cache)
                                        + wt * agreement_gate(val["base"], val["text"], tau, dist, margin) * zscore(val["text"])
                                        + wo * agreement_gate(val["base"], va_ord, tau, dist, margin) * zscore(va_ord)
                                        + wp * agreement_gate(val["base"], va_proto, tau, dist, margin) * zscore(va_proto)
                                        + wb * agreement_gate(val["base"], val["anatomy"] + val["pathology"], tau, dist, margin)
                                        * zscore(0.5 * val["anatomy"] + 0.5 * val["pathology"])
                                    )
                                    final = zscore(val["base"]) + residual
                                    m = metrics(final, val["labels"])
                                    key = score_key(m, args.objective)
                                    if best is None or key > best["key"]:
                                        best = {
                                            "key": key,
                                            "balanced_cache": best_cache["balanced_cache"],
                                            "beta": best_cache["beta"],
                                            "kernel_temperature": best_cache["kernel_temperature"],
                                            "kernel_mix": best_cache["kernel_mix"],
                                            "ordinal_sigma": best_cache["ordinal_sigma"],
                                            "ordinal_sign": best_ord["ordinal_sign"],
                                            "ordinal_scale": best_ord["ordinal_scale"],
                                            "proto_alpha": (
                                                None if best_proto["alpha_per_class"] is not None
                                                else best_proto["alpha"]
                                            ),
                                            "proto_alpha_per_class": (
                                                None if best_proto["alpha_per_class"] is None
                                                else [float(x) for x in best_proto["alpha_per_class"].tolist()]
                                            ),
                                            "proto_scale": best_proto["proto_scale"],
                                            "cache_weight": wc,
                                            "text_weight": wt,
                                            "ordinal_weight": wo,
                                            "proto_weight": wp,
                                            "branch_text_weight": wb,
                                            "tau": tau,
                                            "distance": dist,
                                            "margin": margin,
                                            "agreement_gate": bool(args.agreement_gate),
                                            "val": m,
                                            "stage_cache_val": best_cache["val"],
                                            "stage_ordinal_val": best_ord["val"],
                                            "stage_proto_val": best_proto["val"],
                                        }

    expert_best = None
    if extra_val_logits:
        expert_val_stack = torch.stack([normalize_logits(val["base"])] + [normalize_logits(x) for x in extra_val_logits])
        expert_test_stack = torch.stack([normalize_logits(test["base"])] + [normalize_logits(x) for x in extra_test_logits])
        names = ["baseline"] + extra_names
        for weights in simplex_weights(len(names), args.expert_step):
            expert_val = (expert_val_stack * weights.view(-1, 1, 1)).sum(dim=0)
            for adapter_weight in [0.0, 0.05, 0.1, 0.2]:
                residual = (
                    best["cache_weight"] * agreement_gate(val["base"], va_cache, best["tau"], best["distance"], best["margin"]) * zscore(va_cache)
                    + best["text_weight"] * agreement_gate(val["base"], val["text"], best["tau"], best["distance"], best["margin"]) * zscore(val["text"])
                    + best["ordinal_weight"] * agreement_gate(val["base"], va_ord, best["tau"], best["distance"], best["margin"]) * zscore(va_ord)
                    + best["proto_weight"] * agreement_gate(val["base"], va_proto, best["tau"], best["distance"], best["margin"]) * zscore(va_proto)
                    + best["branch_text_weight"] * agreement_gate(val["base"], val["anatomy"] + val["pathology"], best["tau"], best["distance"], best["margin"])
                    * zscore(0.5 * val["anatomy"] + 0.5 * val["pathology"])
                )
                final = expert_val + adapter_weight * residual
                m = metrics(final, val["labels"])
                key = score_key(m, args.objective)
                if expert_best is None or key > expert_best["key"]:
                    expert_best = {
                        "key": key,
                        "names": names,
                        "weights": weights,
                        "adapter_weight": adapter_weight,
                        "val": m,
                        "test_stack": expert_test_stack,
                    }

    final_train_feats = train["feats"]
    final_train_labels = train["labels"]
    if args.final_cache_trainval:
        final_train_feats = torch.cat([train["feats"], val["feats"]], dim=0)
        final_train_labels = np.concatenate([train["labels"], val["labels"]])

    kernel = build_text_kernel(
        text_proto,
        temperature=best["kernel_temperature"],
        ordinal_sigma=best["ordinal_sigma"],
        mix=best["kernel_mix"],
    )
    te_cache = semantic_cache_logits(
        test["feats"], final_train_feats, final_train_labels,
        kernel, beta=best["beta"], balanced=best["balanced_cache"]
    )
    te_ord = ordinal_boundary_logits(
        test["transition"], scale=best["ordinal_scale"], sign=best["ordinal_sign"]
    )
    alpha = (
        torch.tensor(best["proto_alpha_per_class"], dtype=torch.float32)
        if best["proto_alpha_per_class"] is not None else best["proto_alpha"]
    )
    te_proto = prototype_logits(test["feats"], visual_proto, text_proto, alpha, best["proto_scale"])
    residual = (
        best["cache_weight"] * agreement_gate(test["base"], te_cache, best["tau"], best["distance"], best["margin"]) * zscore(te_cache)
        + best["text_weight"] * agreement_gate(test["base"], test["text"], best["tau"], best["distance"], best["margin"]) * zscore(test["text"])
        + best["ordinal_weight"] * agreement_gate(test["base"], te_ord, best["tau"], best["distance"], best["margin"]) * zscore(te_ord)
        + best["proto_weight"] * agreement_gate(test["base"], te_proto, best["tau"], best["distance"], best["margin"]) * zscore(te_proto)
        + best["branch_text_weight"] * agreement_gate(test["base"], test["anatomy"] + test["pathology"], best["tau"], best["distance"], best["margin"])
        * zscore(0.5 * test["anatomy"] + 0.5 * test["pathology"])
    )
    if expert_best is not None:
        expert_test = (expert_best["test_stack"] * expert_best["weights"].view(-1, 1, 1)).sum(dim=0)
        test_final = expert_test + expert_best["adapter_weight"] * residual
        best["expert_router"] = {
            "names": expert_best["names"],
            "weights": [float(x) for x in expert_best["weights"].tolist()],
            "adapter_weight": expert_best["adapter_weight"],
            "val": expert_best["val"],
            "per_expert": per_expert,
        }
    else:
        test_final = zscore(test["base"]) + confidence_gate(test["base"], best["tau"]) * residual
    test_m = metrics(test_final, test["labels"])

    result = {
        "dataset": args.dataset,
        "method": "semantic_ordinal_transport_adapter",
        "base_val": base_val,
        "base_test": base_test,
        "selected": {k: v for k, v in best.items() if k != "key"},
        "test": test_m,
        "delta_test": {k: test_m[k] - base_test[k] for k in ["accuracy", "macro_f1", "mae"]},
        "final_cache_trainval": bool(args.final_cache_trainval),
    }
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"{args.dataset}.json"
    path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(result, indent=2, ensure_ascii=False))
    print(f"saved: {path}")


if __name__ == "__main__":
    main()
