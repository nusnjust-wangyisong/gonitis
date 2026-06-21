import argparse
import json
import os
import pickle
import sys
from pathlib import Path

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import numpy as np
import torch
import torch.nn as nn
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, mean_absolute_error, precision_score, recall_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(SCRIPT_ROOT))

from calibrated_eval import collect_logits, combine_logits
from main import load_clip_state_dict, patch_state_dict_text_projection, unwrap_model
from clip.model import build_fusion_model
from data.dataset import ClassificationDataset, create_dataloader, get_transforms, seed_everything


def metrics_from_preds(y_true, y_pred):
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_precision": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "macro_recall": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
    }


def evidence_source_logits(pack, proto_weight=0.05, cluster_weight=0.4, visual_cluster_weight=0.0, patch_weight=0.0):
    cls = pack["cls"].float()
    proto = pack["proto"].float()
    cluster = pack["cluster"].float()
    fused = cls + float(proto_weight) * proto + float(cluster_weight) * cluster
    sources = [cls, proto, cluster]
    if "visual_cluster" in pack:
        visual_cluster = pack["visual_cluster"].float()
        fused = fused + float(visual_cluster_weight) * visual_cluster
        sources.append(visual_cluster)
    if "patch" in pack:
        patch = pack["patch"].float()
        fused = fused + float(patch_weight) * patch
        sources.append(patch)
    sources.append(fused)
    return sources


def build_evidence_features(pack, feature_mode, proto_weight=0.05, cluster_weight=0.4, visual_cluster_weight=0.0, patch_weight=0.0):
    sources = evidence_source_logits(pack, proto_weight, cluster_weight, visual_cluster_weight, patch_weight)
    cls = sources[0]
    parts = []

    if "logits" in feature_mode:
        parts.extend([logits.numpy() for logits in sources])

    if "prob" in feature_mode:
        for logits in sources:
            parts.append(torch.softmax(logits, dim=1).numpy())

    if "ord" in feature_mode:
        class_ids = torch.arange(cls.shape[1], dtype=torch.float32)
        for logits in sources:
            prob = torch.softmax(logits, dim=1)
            top2 = prob.topk(2, dim=1).values
            parts.append((prob * class_ids).sum(dim=1, keepdim=True).numpy())
            parts.append(top2[:, 0:1].numpy())
            parts.append((top2[:, 0:1] - top2[:, 1:2]).numpy())

    if "boundary" in feature_mode:
        for logits in sources:
            prob = torch.softmax(logits, dim=1)
            left_cum = torch.cumsum(prob, dim=1)[:, :-1]
            right_cum = 1.0 - left_cum
            adjacent_margin = prob[:, :-1] - prob[:, 1:]
            entropy = -(prob * prob.clamp_min(1e-8).log()).sum(dim=1, keepdim=True)
            variance = (prob * (torch.arange(prob.shape[1], dtype=torch.float32) - (prob * torch.arange(prob.shape[1], dtype=torch.float32)).sum(dim=1, keepdim=True)) ** 2).sum(dim=1, keepdim=True)
            parts.extend([
                left_cum.numpy(),
                right_cum.numpy(),
                adjacent_margin.numpy(),
                entropy.numpy(),
                variance.numpy(),
            ])

    if "disagree" in feature_mode:
        probs = [torch.softmax(logits, dim=1) for logits in sources]
        entropy_items = [
            (-(prob * prob.clamp_min(1e-8).log()).sum(dim=1, keepdim=True) / np.log(prob.shape[1]))
            for prob in probs
        ]
        entropy_stack = torch.cat(entropy_items, dim=1)
        argmax_stack = torch.stack([prob.argmax(dim=1) for prob in probs], dim=1)
        agree_max = []
        unique_ratio = []
        for row in argmax_stack:
            counts = torch.bincount(row, minlength=cls.shape[1]).float()
            agree_max.append(counts.max() / argmax_stack.shape[1])
            unique_ratio.append((counts > 0).float().sum() / argmax_stack.shape[1])
        agree_max = torch.stack(agree_max).unsqueeze(1)
        unique_ratio = torch.stack(unique_ratio).unsqueeze(1)
        parts.extend([
            entropy_stack.numpy(),
            agree_max.numpy(),
            unique_ratio.numpy(),
            entropy_stack.max(dim=1, keepdim=True).values.numpy(),
            entropy_stack.min(dim=1, keepdim=True).values.numpy(),
        ])

        for i in range(len(probs)):
            for j in range(i + 1, len(probs)):
                p, q = probs[i], probs[j]
                m = 0.5 * (p + q)
                kl_pm = (p * (p.clamp_min(1e-8).log() - m.clamp_min(1e-8).log())).sum(dim=1, keepdim=True)
                kl_qm = (q * (q.clamp_min(1e-8).log() - m.clamp_min(1e-8).log())).sum(dim=1, keepdim=True)
                parts.append((0.5 * (kl_pm + kl_qm)).numpy())

    if "tta" in feature_mode and "cls_orig" in pack and "cls_flip" in pack:
        orig_sources = [
            pack["cls_orig"].float(),
            pack["proto_orig"].float(),
            pack["cluster_orig"].float(),
        ]
        flip_sources = [
            pack["cls_flip"].float(),
            pack["proto_flip"].float(),
            pack["cluster_flip"].float(),
        ]
        orig_sources.append(orig_sources[0] + float(proto_weight) * orig_sources[1] + float(cluster_weight) * orig_sources[2])
        flip_sources.append(flip_sources[0] + float(proto_weight) * flip_sources[1] + float(cluster_weight) * flip_sources[2])
        for orig_logits, flip_logits in zip(orig_sources, flip_sources):
            p = torch.softmax(orig_logits, dim=1)
            q = torch.softmax(flip_logits, dim=1)
            m = 0.5 * (p + q)
            js = 0.5 * (
                (p * (p.clamp_min(1e-8).log() - m.clamp_min(1e-8).log())).sum(dim=1, keepdim=True)
                + (q * (q.clamp_min(1e-8).log() - m.clamp_min(1e-8).log())).sum(dim=1, keepdim=True)
            )
            p_top2 = p.topk(2, dim=1).values
            q_top2 = q.topk(2, dim=1).values
            p_entropy = -(p * p.clamp_min(1e-8).log()).sum(dim=1, keepdim=True) / np.log(p.shape[1])
            q_entropy = -(q * q.clamp_min(1e-8).log()).sum(dim=1, keepdim=True) / np.log(q.shape[1])
            argmax_agree = (p.argmax(dim=1) == q.argmax(dim=1)).float().unsqueeze(1)
            parts.extend([
                js.numpy(),
                argmax_agree.numpy(),
                torch.abs(p - q).max(dim=1, keepdim=True).values.numpy(),
                torch.abs(orig_logits - flip_logits).max(dim=1, keepdim=True).values.numpy(),
                torch.abs(p_entropy - q_entropy).numpy(),
                torch.abs((p_top2[:, 0:1] - p_top2[:, 1:2]) - (q_top2[:, 0:1] - q_top2[:, 1:2])).numpy(),
            ])

    return np.concatenate(parts, axis=1)


class CumulativeOrdinalLogistic:
    def __init__(self, c_value):
        self.c_value = c_value
        self.scaler = StandardScaler()
        self.models = []

    def fit(self, x, y):
        x = self.scaler.fit_transform(x)
        self.models = []
        for threshold in range(4):
            binary_y = (y > threshold).astype(np.int64)
            model = LogisticRegression(
                C=self.c_value,
                max_iter=5000,
                class_weight="balanced",
                solver="liblinear",
            )
            model.fit(x, binary_y)
            self.models.append(model)
        return self

    def predict_proba(self, x):
        x = self.scaler.transform(x)
        p_gt = np.stack([model.predict_proba(x)[:, 1] for model in self.models], axis=1)
        p_gt = np.minimum.accumulate(p_gt, axis=1)
        probs = np.zeros((x.shape[0], 5), dtype=np.float64)
        probs[:, 0] = 1.0 - p_gt[:, 0]
        probs[:, 1] = p_gt[:, 0] - p_gt[:, 1]
        probs[:, 2] = p_gt[:, 1] - p_gt[:, 2]
        probs[:, 3] = p_gt[:, 2] - p_gt[:, 3]
        probs[:, 4] = p_gt[:, 3]
        return np.clip(probs, 0.0, 1.0)

    def predict(self, x):
        return self.predict_proba(x).argmax(axis=1)


def build_calibrator(kind, c_value):
    if kind == "logreg_balanced":
        return make_pipeline(
            StandardScaler(),
            LogisticRegression(
                C=c_value,
                max_iter=5000,
                class_weight="balanced",
                multi_class="multinomial",
            ),
        )
    if kind == "linearsvc_balanced":
        return make_pipeline(
            StandardScaler(),
            LinearSVC(C=c_value, class_weight="balanced", max_iter=10000, dual="auto"),
        )
    if kind == "ordinal_logreg_balanced":
        return CumulativeOrdinalLogistic(c_value)
    raise ValueError(f"Unknown calibrator kind: {kind}")


def prediction_confidence_margin(calibrator, features):
    scores = prediction_scores(calibrator, features)
    scores = np.asarray(scores)
    if scores.ndim == 1:
        return np.abs(scores)
    top2 = np.sort(scores, axis=1)[:, -2:]
    return top2[:, 1] - top2[:, 0]


def prediction_scores(calibrator, features):
    if hasattr(calibrator, "predict_proba"):
        return calibrator.predict_proba(features)
    if hasattr(calibrator, "decision_function"):
        return calibrator.decision_function(features)
    return np.ones((features.shape[0], 1), dtype=np.float64)


def scores_to_probs(scores):
    scores = np.asarray(scores, dtype=np.float64)
    if scores.ndim == 1:
        scores = np.stack([-scores, scores], axis=1)
    scores = scores - scores.max(axis=1, keepdims=True)
    probs = np.exp(scores)
    return probs / probs.sum(axis=1, keepdims=True).clip(min=1e-8)


def prob_margin(prob):
    top2 = np.sort(prob, axis=1)[:, -2:]
    return top2[:, 1:2] - top2[:, 0:1]


def prob_entropy(prob):
    prob = np.asarray(prob, dtype=np.float64)
    return -(prob * np.log(np.clip(prob, 1e-8, 1.0))).sum(axis=1, keepdims=True) / np.log(prob.shape[1])


def prob_expected_grade(prob):
    class_ids = np.arange(prob.shape[1], dtype=np.float64)
    mean = (prob * class_ids).sum(axis=1, keepdims=True)
    var = (prob * (class_ids.reshape(1, -1) - mean) ** 2).sum(axis=1, keepdims=True)
    return mean, var


def smooth_ordinal_prob(prob, smoothing):
    if smoothing <= 0:
        return prob
    prob = np.asarray(prob, dtype=np.float64)
    out = prob * (1.0 - 2.0 * smoothing)
    out[:, 1:] += smoothing * prob[:, :-1]
    out[:, :-1] += smoothing * prob[:, 1:]
    out = np.clip(out, 1e-8, None)
    return out / out.sum(axis=1, keepdims=True)


def one_hot(pred, num_classes):
    out = np.zeros((pred.shape[0], num_classes), dtype=np.float64)
    out[np.arange(pred.shape[0]), pred] = 1.0
    return out


def build_selector_features(base_logits, calibrator, evidence_features):
    base_prob = torch.softmax(base_logits.float(), dim=1).numpy()
    calib_prob = scores_to_probs(prediction_scores(calibrator, evidence_features))
    if calib_prob.shape[1] != base_prob.shape[1]:
        fixed = np.zeros_like(base_prob)
        cols = min(fixed.shape[1], calib_prob.shape[1])
        fixed[:, :cols] = calib_prob[:, :cols]
        calib_prob = fixed

    base_pred = base_prob.argmax(axis=1)
    calib_pred = calib_prob.argmax(axis=1)
    base_mean, base_var = prob_expected_grade(base_prob)
    calib_mean, calib_var = prob_expected_grade(calib_prob)
    pred_gap = np.abs(base_pred - calib_pred).reshape(-1, 1).astype(np.float64) / (base_prob.shape[1] - 1)
    mean_gap = np.abs(base_mean - calib_mean)

    return np.concatenate([
        base_prob,
        calib_prob,
        np.abs(base_prob - calib_prob),
        one_hot(base_pred, base_prob.shape[1]),
        one_hot(calib_pred, base_prob.shape[1]),
        prob_margin(base_prob),
        prob_margin(calib_prob),
        prob_entropy(base_prob),
        prob_entropy(calib_prob),
        base_mean,
        calib_mean,
        base_var,
        calib_var,
        pred_gap,
        mean_gap,
    ], axis=1)


def objective_value(y_true, y_pred, objective):
    metrics = metrics_from_preds(y_true, y_pred)
    if objective == "accuracy":
        return metrics["accuracy"]
    if objective == "macro_f1":
        return metrics["macro_f1"]
    if objective == "acc_f1_mae":
        return metrics["accuracy"] + metrics["macro_f1"] - 0.25 * metrics["mae"]
    raise ValueError(f"Unknown selector objective: {objective}")


def fit_selector_gate(selector_x, selector_y, base_pred, evs_pred, objective):
    base_mae = np.abs(base_pred - selector_y)
    evs_mae = np.abs(evs_pred - selector_y)
    target = (evs_mae < base_mae).astype(np.int64)
    if len(np.unique(target)) < 2:
        return None, 1.0, target

    selector = make_pipeline(
        StandardScaler(),
        LogisticRegression(C=0.5, max_iter=5000, class_weight="balanced"),
    )
    selector.fit(selector_x, target)
    selector_prob = selector.predict_proba(selector_x)[:, 1]
    thresholds = np.unique(np.quantile(selector_prob, np.linspace(0.05, 0.95, 19)))
    best_threshold = 1.0
    best_score = objective_value(selector_y, base_pred, objective)
    for threshold in thresholds:
        pred = base_pred.copy()
        mask = selector_prob >= threshold
        pred[mask] = evs_pred[mask]
        score = objective_value(selector_y, pred, objective)
        if score > best_score:
            best_score = score
            best_threshold = float(threshold)
    return selector, best_threshold, target


def load_model(args, device):
    state_dict = patch_state_dict_text_projection(load_clip_state_dict(args.clip_model), fallback_embed_dim=512)
    model = build_fusion_model(
        state_dict=state_dict,
        pubmedbert_path=args.pubmedbert_path,
        backbone=args.backbone,
        img_size=args.img_size,
        freeze_text_encoder=True,
        enable_visual_cluster=args.enable_visual_cluster,
        enable_semantic_patch=args.enable_semantic_patch,
    )
    missing, unexpected = model.load_state_dict(torch.load(args.checkpoint, map_location="cpu"), strict=False)
    if missing:
        print(f"[stacking] Missing keys: {missing[:8]}")
    if unexpected:
        print(f"[stacking] Unexpected keys: {unexpected[:8]}")
    model = model.to(device)
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
    base = unwrap_model(model)
    if hasattr(base, "refresh_text_prototypes"):
        base.refresh_text_prototypes()
    return model


def collect_split_pack(args, model, split, transform, device):
    return_view_logits = "tta" in args.feature_mode
    if args.cache_dir:
        os.makedirs(args.cache_dir, exist_ok=True)
        cache_name = f"{split}_hflip{int(args.hflip)}_view{int(return_view_logits)}_img{args.img_size}.pt"
        cache_path = os.path.join(args.cache_dir, cache_name)
        if os.path.exists(cache_path):
            print(f"[stacking] Loading cached {split} logits: {cache_path}")
            return torch.load(cache_path, map_location="cpu")
        if not return_view_logits:
            legacy_name = f"{split}_hflip{int(args.hflip)}_img{args.img_size}.pt"
            legacy_path = os.path.join(args.cache_dir, legacy_name)
            if os.path.exists(legacy_path):
                print(f"[stacking] Loading cached {split} logits: {legacy_path}")
                return torch.load(legacy_path, map_location="cpu")

    dataset = ClassificationDataset(
        args.data_root,
        split=split,
        transform=transform,
        error_policy="raise",
        include_path=False,
    )
    loader = create_dataloader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )
    print(f"[stacking] Collecting {split} logits...")
    pack = collect_logits(model, loader, device, use_hflip=args.hflip, return_view_logits=return_view_logits)
    if args.cache_dir:
        torch.save(pack, cache_path)
    return pack


def main():
    parser = argparse.ArgumentParser(description="Evidence stacking calibration for KL grading.")
    parser.add_argument("--data_root", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--clip_model", default="ViT-B/32")
    parser.add_argument("--pubmedbert_path", default=str(PROJECT_ROOT / "PubMedBERT"))
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--img_size", type=int, default=224)
    parser.add_argument("--backbone", type=str, default="openai_clip",
                        choices=["openai_clip", "biomedclip"])
    parser.add_argument("--clip_normalize", dest="clip_normalize",
                        action="store_true", default=None)
    parser.add_argument("--no_clip_normalize", dest="clip_normalize",
                        action="store_false")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--hflip", action="store_true")
    parser.add_argument("--enable_visual_cluster", action="store_true",
                        help="Build a full_vcf model and include visual cluster logits when available.")
    parser.add_argument("--enable_semantic_patch", action="store_true",
                        help="Build a full_tpa model and include semantic patch logits when available.")
    parser.add_argument("--cache_dir", default=None,
                        help="Optional directory for cached split logits.")
    parser.add_argument("--train_splits", default="train,val",
                        help="Comma-separated splits used to train the second-stage calibrator.")
    parser.add_argument(
        "--feature_mode",
        default="logits_prob",
        choices=[
            "logits",
            "logits_prob",
            "logits_prob_ord",
            "logits_prob_boundary",
            "logits_prob_ord_boundary",
            "logits_prob_boundary_disagree",
            "logits_prob_boundary_disagree_tta",
        ],
    )
    parser.add_argument(
        "--calibrator",
        default="logreg_balanced",
        choices=["logreg_balanced", "linearsvc_balanced", "ordinal_logreg_balanced"],
    )
    parser.add_argument("--c_value", type=float, default=0.1)
    parser.add_argument("--proto_weight", type=float, default=0.05,
                        help="VC-MLC text prototype weight used to build fused evidence.")
    parser.add_argument("--cluster_weight", type=float, default=0.4,
                        help="VC-MLC cluster prototype weight used to build fused evidence.")
    parser.add_argument("--visual_cluster_weight", type=float, default=0.0,
                        help="VC-MLC visual cluster prototype weight used to build fused evidence.")
    parser.add_argument("--patch_weight", type=float, default=0.0,
                        help="Text-guided semantic patch logit weight used to build fused evidence.")
    parser.add_argument("--temperature", type=float, default=0.8,
                        help="VC-MLC temperature used for the reported base metrics.")
    parser.add_argument("--gate_quantile", type=float, default=None,
                        help="If set, only override VC-MLC predictions when calibrator confidence is above this quantile.")
    parser.add_argument("--gate_threshold_source", default="train",
                        choices=["train", "val", "trainval", "test"],
                        help="Split scores used to compute the confidence gate threshold.")
    parser.add_argument("--auto_gate_quantile", action="store_true",
                        help="Use data-scale adaptive gate quantile: smaller datasets get higher EVS coverage.")
    parser.add_argument("--small_data_threshold", type=int, default=2000,
                        help="Training size threshold used by --auto_gate_quantile.")
    parser.add_argument("--small_data_gate_quantile", type=float, default=0.15)
    parser.add_argument("--large_data_gate_quantile", type=float, default=0.25)
    parser.add_argument("--ordinal_smoothing", type=float, default=0.0,
                        help="Optional adjacent-class probability smoothing applied after gated fusion.")
    parser.add_argument("--auto_ordinal_smoothing", action="store_true",
                        help="Enable severity-class-aware ordinal neighbor smoothing for small, severely imbalanced datasets.")
    parser.add_argument("--minority_ratio_threshold", type=float, default=0.05)
    parser.add_argument("--minority_ordinal_smoothing", type=float, default=0.20)
    parser.add_argument("--large_minority_ordinal_smoothing", type=float, default=0.02)
    parser.add_argument("--selector_split", default=None,
                        help="Optional split used to train a learning-to-defer EVS selector, usually val.")
    parser.add_argument("--selector_objective", default="macro_f1",
                        choices=["accuracy", "macro_f1", "acc_f1_mae"],
                        help="Validation objective used to choose the selector threshold.")
    args = parser.parse_args()

    seed_everything(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_model(args, device)

    if args.clip_normalize is None:
        args.clip_normalize = (args.backbone == "biomedclip")
    eval_tf = get_transforms(
        img_size=args.img_size,
        is_training=False,
        use_clahe=True,
        to_rgb=True,
        clahe_clip_limit=2.0,
        clahe_tile_grid_size=(8, 8),
        percentile=(1.0, 99.0),
        normalize=args.clip_normalize,
    )

    split_names = [x.strip() for x in args.train_splits.split(",") if x.strip()]
    collect_order = []
    gate_splits = []
    gate_requested = args.gate_quantile is not None or args.auto_gate_quantile
    if gate_requested:
        if args.gate_threshold_source == "trainval":
            gate_splits = ["train", "val"]
        elif args.gate_threshold_source != "test":
            gate_splits = [args.gate_threshold_source]
    selector_splits = [args.selector_split] if args.selector_split else []
    for split in split_names + gate_splits + selector_splits + ["test"]:
        if split not in collect_order:
            collect_order.append(split)
    packs = {split: collect_split_pack(args, model, split, eval_tf, device) for split in collect_order}

    train_x = np.concatenate([
        build_evidence_features(
            packs[split], args.feature_mode,
            args.proto_weight, args.cluster_weight, args.visual_cluster_weight, args.patch_weight
        )
        for split in split_names
    ], axis=0)
    train_y = np.concatenate([packs[split]["labels"].numpy() for split in split_names], axis=0)
    test_x = build_evidence_features(
        packs["test"], args.feature_mode,
        args.proto_weight, args.cluster_weight, args.visual_cluster_weight, args.patch_weight
    )
    test_y = packs["test"]["labels"].numpy()
    effective_gate_quantile = args.gate_quantile
    if args.auto_gate_quantile:
        effective_gate_quantile = (
            args.small_data_gate_quantile
            if train_y.shape[0] < args.small_data_threshold
            else args.large_data_gate_quantile
        )
    class_counts = np.bincount(train_y, minlength=5)
    min_class_ratio = float(class_counts.min() / max(train_y.shape[0], 1))
    effective_ordinal_smoothing = float(args.ordinal_smoothing)
    if args.auto_ordinal_smoothing:
        if min_class_ratio < args.minority_ratio_threshold:
            effective_ordinal_smoothing = (
                float(args.minority_ordinal_smoothing)
                if train_y.shape[0] < args.small_data_threshold
                else float(args.large_minority_ordinal_smoothing)
            )
        else:
            effective_ordinal_smoothing = 0.0

    calibrator = build_calibrator(args.calibrator, args.c_value)
    calibrator.fit(train_x, train_y)
    test_pred = calibrator.predict(test_x)
    test_metrics = metrics_from_preds(test_y, test_pred)

    base_logits = combine_logits(
        packs["test"],
        args.proto_weight,
        args.cluster_weight,
        args.temperature,
        args.visual_cluster_weight,
        args.patch_weight,
    )
    base_pred = base_logits.argmax(dim=1).numpy()
    base_metrics = metrics_from_preds(test_y, base_pred)

    gated_pred = None
    gated_metrics = None
    gate_threshold = None
    gate_coverage = None
    if effective_gate_quantile is not None:
        if args.gate_threshold_source == "test":
            ref_scores = prediction_confidence_margin(calibrator, test_x)
        else:
            ref_parts = []
            for split in gate_splits:
                split_x = build_evidence_features(
                    packs[split], args.feature_mode,
                    args.proto_weight, args.cluster_weight, args.visual_cluster_weight, args.patch_weight
                )
                ref_parts.append(prediction_confidence_margin(calibrator, split_x))
            ref_scores = np.concatenate(ref_parts, axis=0)
        gate_threshold = float(np.quantile(ref_scores, effective_gate_quantile))
        test_scores = prediction_confidence_margin(calibrator, test_x)
        gate_mask = test_scores >= gate_threshold
        if effective_ordinal_smoothing > 0:
            base_prob = torch.softmax(base_logits.float(), dim=1).numpy()
            calib_prob = scores_to_probs(prediction_scores(calibrator, test_x))
            if calib_prob.shape[1] != base_prob.shape[1]:
                fixed = np.zeros_like(base_prob)
                cols = min(fixed.shape[1], calib_prob.shape[1])
                fixed[:, :cols] = calib_prob[:, :cols]
                calib_prob = fixed
            gated_prob = base_prob.copy()
            gated_prob[gate_mask] = calib_prob[gate_mask]
            gated_pred = smooth_ordinal_prob(gated_prob, effective_ordinal_smoothing).argmax(axis=1)
        else:
            gated_pred = base_pred.copy()
            gated_pred[gate_mask] = test_pred[gate_mask]
        gate_coverage = float(np.mean(gate_mask))
        gated_metrics = metrics_from_preds(test_y, gated_pred)

    selector_pred = None
    selector_metrics = None
    selector_threshold = None
    selector_coverage = None
    selector_positive_rate = None
    if args.selector_split:
        selector_x = build_evidence_features(
            packs[args.selector_split], args.feature_mode,
            args.proto_weight, args.cluster_weight, args.visual_cluster_weight, args.patch_weight
        )
        selector_y = packs[args.selector_split]["labels"].numpy()
        selector_base_logits = combine_logits(
            packs[args.selector_split],
            args.proto_weight,
            args.cluster_weight,
            args.temperature,
            args.visual_cluster_weight,
            args.patch_weight,
        )
        selector_base_pred = selector_base_logits.argmax(dim=1).numpy()
        selector_evs_pred = calibrator.predict(selector_x)
        selector_features = build_selector_features(selector_base_logits, calibrator, selector_x)
        selector, selector_threshold, selector_target = fit_selector_gate(
            selector_features,
            selector_y,
            selector_base_pred,
            selector_evs_pred,
            args.selector_objective,
        )
        selector_positive_rate = float(np.mean(selector_target))
        if selector is not None:
            test_selector_features = build_selector_features(base_logits, calibrator, test_x)
            selector_prob = selector.predict_proba(test_selector_features)[:, 1]
            selector_mask = selector_prob >= selector_threshold
            selector_pred = base_pred.copy()
            selector_pred[selector_mask] = test_pred[selector_mask]
            selector_coverage = float(np.mean(selector_mask))
            selector_metrics = metrics_from_preds(test_y, selector_pred)

    os.makedirs(args.output_dir, exist_ok=True)
    with open(os.path.join(args.output_dir, "stacking_calibrator.pkl"), "wb") as f:
        pickle.dump(calibrator, f)

    out = {
        "data_root": args.data_root,
        "checkpoint": args.checkpoint,
        "hflip_tta": bool(args.hflip),
        "train_splits": split_names,
        "feature_mode": args.feature_mode,
        "calibrator": args.calibrator,
        "c_value": args.c_value,
        "proto_weight": float(args.proto_weight),
        "cluster_weight": float(args.cluster_weight),
        "visual_cluster_weight": float(args.visual_cluster_weight),
        "patch_weight": float(args.patch_weight),
        "temperature": float(args.temperature),
        "gate_quantile": None if args.gate_quantile is None else float(args.gate_quantile),
        "effective_gate_quantile": None if effective_gate_quantile is None else float(effective_gate_quantile),
        "auto_gate_quantile": bool(args.auto_gate_quantile),
        "small_data_threshold": int(args.small_data_threshold),
        "small_data_gate_quantile": float(args.small_data_gate_quantile),
        "large_data_gate_quantile": float(args.large_data_gate_quantile),
        "ordinal_smoothing": float(args.ordinal_smoothing),
        "effective_ordinal_smoothing": float(effective_ordinal_smoothing),
        "auto_ordinal_smoothing": bool(args.auto_ordinal_smoothing),
        "minority_ratio_threshold": float(args.minority_ratio_threshold),
        "minority_ordinal_smoothing": float(args.minority_ordinal_smoothing),
        "large_minority_ordinal_smoothing": float(args.large_minority_ordinal_smoothing),
        "min_class_ratio": min_class_ratio,
        "class_counts": class_counts.tolist(),
        "gate_threshold_source": args.gate_threshold_source,
        "gate_threshold": gate_threshold,
        "gate_coverage": gate_coverage,
        "selector_split": args.selector_split,
        "selector_objective": args.selector_objective,
        "selector_threshold": selector_threshold,
        "selector_coverage": selector_coverage,
        "selector_positive_rate": selector_positive_rate,
        "base_calibrated_metrics": base_metrics,
        "test_metrics": test_metrics,
        "gated_test_metrics": gated_metrics,
        "selector_test_metrics": selector_metrics,
        "test_predictions": {
            "labels": test_y.tolist(),
            "preds": test_pred.tolist(),
            "gated_preds": None if gated_pred is None else gated_pred.tolist(),
            "selector_preds": None if selector_pred is None else selector_pred.tolist(),
            "confusion_matrix": test_metrics["confusion_matrix"],
            "gated_confusion_matrix": None if gated_metrics is None else gated_metrics["confusion_matrix"],
            "selector_confusion_matrix": None if selector_metrics is None else selector_metrics["confusion_matrix"],
        },
    }
    with open(os.path.join(args.output_dir, "stacking_metrics.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)

    print(json.dumps({
        "output_dir": args.output_dir,
        "base_calibrated_metrics": base_metrics,
        "test_metrics": test_metrics,
        "gated_test_metrics": gated_metrics,
        "selector_test_metrics": selector_metrics,
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
