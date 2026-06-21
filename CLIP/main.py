import sys
import os
import json
import argparse
import warnings
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import ConcatDataset
try:
    from torch.amp import GradScaler, autocast as _autocast

    def autocast_cuda(enabled=True):
        return _autocast("cuda", enabled=enabled)
except Exception:
    from torch.cuda.amp import GradScaler, autocast as _autocast

    def autocast_cuda(enabled=True):
        return _autocast(enabled=enabled)
from sklearn.metrics import classification_report, confusion_matrix, precision_score, recall_score, f1_score

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

import clip

from data.dataset import (
    ClassificationDataset,
    get_transforms,
    create_dataloader,
    create_balanced_dataloader,
    seed_everything,
)

# ── 从模型文件导入融合模型相关组件 ────────────────────────────────────────
from clip.model import (
    CLIPFusionContrastiveModel,
    build_fusion_model,
    build_optimizer,
    convert_weights,
)
from clip.ablation_model import build_ablation_model

warnings.filterwarnings("ignore")


def load_clip_state_dict(name: str, download_root: str = None) -> dict:
    """
    Load OpenAI CLIP weights as a raw state_dict. This avoids calling clip.load(),
    because this project replaces clip.model.build_model with a PubMedBERT-aware
    builder and standard CLIP weight loading should not instantiate PubMedBERT.
    """
    import importlib

    clip_loader = importlib.import_module("clip.clip")
    if name in clip_loader._MODELS:
        model_path = clip_loader._download(
            clip_loader._MODELS[name],
            download_root or os.path.expanduser("~/.cache/clip"),
        )
    elif os.path.isfile(name):
        model_path = name
    else:
        raise RuntimeError(
            f"Model {name} not found; available models = {clip.available_models()}"
        )

    with open(model_path, "rb") as opened_file:
        try:
            model = torch.jit.load(opened_file, map_location="cpu").eval()
            state_dict = model.state_dict()
            del model
        except RuntimeError:
            state_dict = torch.load(opened_file, map_location="cpu")
    return state_dict


# ════════════════════════════════════════════════════════════════════════════
#  Utility functions
# ════════════════════════════════════════════════════════════════════════════

def unwrap_model(model: nn.Module) -> nn.Module:
    """
    提取 DataParallel 封装下的原始模型。
    """
    return model.module if isinstance(model, nn.DataParallel) else model


def get_batch_image_label(batch, device):
    """
    从 batch dict 中提取图像和标签。
    支持 'image' 或 'img' 两种字段名。
    """
    if "image" in batch:
        image = batch["image"]
    elif "img" in batch:
        image = batch["img"]
    else:
        raise KeyError("batch 中没有 'image' 或 'img' 字段。")
    label = batch["label"]
    return image.to(device, non_blocking=True), label.to(device, non_blocking=True)


def patch_state_dict_text_projection(state_dict: dict, fallback_embed_dim: int) -> dict:
    """
    若 state_dict 中缺少顶层 "text_projection" 键（自定义 CLIP 将其定义为
    nn.Sequential，实际键形如 "text_projection.1.weight"），则按以下优先级补全：

      1. visual.proj          —— ViT backbone，shape [width, embed_dim]
      2. text_projection.1.weight —— nn.Sequential 中 Linear 的权重，
                                     shape [embed_dim, in_features]，取 dim 0
      3. fallback_embed_dim   —— 上两者均不存在时，用命令行 --embed_dim 占位

    返回修改后的 state_dict（原地修改，同时也返回引用）。
    """
    if "text_projection" in state_dict:
        return state_dict   # 标准 CLIP，无需处理

    # 优先级 1：ViT backbone 的 visual.proj [width, embed_dim]
    if "visual.proj" in state_dict:
        state_dict["text_projection"] = state_dict["visual.proj"]
        embed_dim = state_dict["text_projection"].shape[1]
        print(f"[patch] text_projection 缺失，已从 visual.proj 补全，embed_dim={embed_dim}")
        return state_dict

    # 优先级 2：nn.Sequential Linear 权重 text_projection.1.weight [out, in]
    seq_key = "text_projection.1.weight"
    if seq_key in state_dict:
        w = state_dict[seq_key]
        state_dict["text_projection"] = torch.zeros(w.shape[1], w.shape[0])
        embed_dim = w.shape[0]
        print(f"[patch] text_projection 缺失，已从 {seq_key} 推断，embed_dim={embed_dim}")
        return state_dict

    # 优先级 3：fallback
    state_dict["text_projection"] = torch.zeros(1, fallback_embed_dim)
    print(f"[patch] text_projection 缺失，已用 fallback embed_dim={fallback_embed_dim} 占位")
    return state_dict


def load_compatible_state_dict(model: nn.Module, checkpoint_path: str, device: str):
    """
    Load only checkpoint tensors whose names and shapes match the current model.
    This allows initializing extended variants such as full_vcf from a full model
    while leaving newly added modules randomly initialized.
    """
    current = model.state_dict()
    loaded = torch.load(checkpoint_path, map_location=device)
    compatible = {
        k: v for k, v in loaded.items()
        if k in current and tuple(v.shape) == tuple(current[k].shape)
    }
    skipped = sorted(set(loaded.keys()) - set(compatible.keys()))
    # If the target model extends the old single text-gate fusion into a
    # dual text/visual gate, migrate the compatible slices instead of starting
    # that fusion module entirely from scratch.
    if (
        getattr(model, "enable_visual_cluster", False)
        and "fusion.gate_mlp.0.weight" in loaded
        and "fusion.gate_mlp.0.weight" in current
    ):
        old_w0 = loaded["fusion.gate_mlp.0.weight"]
        new_w0 = current["fusion.gate_mlp.0.weight"].clone()
        embed_dim = getattr(model, "_embed_dim", 512)
        num_classes = getattr(model, "NUM_CLASSES", 5)

        if old_w0.shape[0] == new_w0.shape[0] and old_w0.shape[1] == embed_dim * 3 + num_classes:
            new_w0.zero_()
            # old: [img, text, img*text, cluster_prob]
            # new: [img, text, visual, img*text, img*visual, text*visual, cluster_prob, visual_prob]
            new_w0[:, 0:embed_dim] = old_w0[:, 0:embed_dim]
            new_w0[:, embed_dim:embed_dim * 2] = old_w0[:, embed_dim:embed_dim * 2]
            new_w0[:, embed_dim * 3:embed_dim * 4] = old_w0[:, embed_dim * 2:embed_dim * 3]
            old_prob_start = embed_dim * 3
            new_prob_start = embed_dim * 6
            new_w0[:, new_prob_start:new_prob_start + num_classes] = old_w0[:, old_prob_start:old_prob_start + num_classes]
            current["fusion.gate_mlp.0.weight"] = new_w0
            compatible["fusion.gate_mlp.0.weight"] = new_w0

        old_w3 = loaded.get("fusion.gate_mlp.3.weight")
        old_b3 = loaded.get("fusion.gate_mlp.3.bias")
        if old_w3 is not None and "fusion.gate_mlp.3.weight" in current:
            new_w3 = current["fusion.gate_mlp.3.weight"].clone()
            if new_w3.shape[0] == old_w3.shape[0] * 2 and new_w3.shape[1] == old_w3.shape[1]:
                new_w3.zero_()
                new_w3[:old_w3.shape[0]] = old_w3
                current["fusion.gate_mlp.3.weight"] = new_w3
                compatible["fusion.gate_mlp.3.weight"] = new_w3
        if old_b3 is not None and "fusion.gate_mlp.3.bias" in current:
            new_b3 = current["fusion.gate_mlp.3.bias"].clone()
            if new_b3.shape[0] == old_b3.shape[0] * 2:
                new_b3.zero_()
                new_b3[:old_b3.shape[0]] = old_b3
                # Keep the newly added visual gate almost closed at the start.
                # The model can learn to open it only if visual prototypes help.
                new_b3[old_b3.shape[0]:] = -5.0
                current["fusion.gate_mlp.3.bias"] = new_b3
                compatible["fusion.gate_mlp.3.bias"] = new_b3

        if (
            "cluster_prototypes" in loaded
            and "visual_cluster_prototypes" in current
            and tuple(loaded["cluster_prototypes"].shape) == tuple(current["visual_cluster_prototypes"].shape)
        ):
            compatible["visual_cluster_prototypes"] = loaded["cluster_prototypes"].clone()

    current.update(compatible)
    model.load_state_dict(current)
    return len(compatible), skipped


# ════════════════════════════════════════════════════════════════════════════
#  Trainer 类（核心训练器）
# ════════════════════════════════════════════════════════════════════════════

class Trainer:
    """
    CLIPFusionContrastiveModel 训练器。

    主要功能：
    - 分层学习率优化
    - 混合精度训练（AMP）
    - 梯度剪裁与 EMA 模型（可选）
    - 自动检查点保存与早停
    - 详细的训练日志与验证评估
    """

    def __init__(
        self,
        model: CLIPFusionContrastiveModel,
        train_loader,
        val_loader,
        device,
        lr_visual: float = 5e-5,
        lr_text_encoder: float = 1e-5,
        lr_text_proj: float = 1e-4,
        lr_fusion: float = 1e-4,
        lr_classifier: float = 1e-4,
        lr_adapter: float = None,
        lr_boundary: float = None,
        lr_cluster_proto: float = 1e-4,
        lr_prompt: float = 3e-4,
        lr_convnext: float = 1e-5,
        weight_decay: float = 0.01,
        epochs: int = 50,
        patience: int = 10,
        dataset_name: str = "dataset",
        save_dir: str = "checkpoints",
        prec: str = "amp",
        grad_clip_norm: float = 1.0,
        refresh_text_proto_each_epoch: bool = False,
        report_dir: str = "results",
        monitor_metric: str = "accuracy",
        lr_schedule: str = "warm_restart",
    ):
        """
        初始化训练器。

        Parameters
        ----------
        model : CLIPFusionContrastiveModel
            要训练的融合模型。
        train_loader, val_loader : DataLoader
            训练与验证数据加载器。
        device : str or torch.device
            计算设备。
        lr_* : float
            各模块的学习率。
        weight_decay : float
            AdamW 正则化系数。
        epochs : int
            最大训练轮数。
        patience : int
            早停耐心值。
        dataset_name : str
            数据集名称（用于日志与检查点路径）。
        save_dir : str
            检查点保存目录。
        prec : str
            精度模式，'amp' 或 'fp32'。
        grad_clip_norm : float
            梯度剪裁范数，<=0 时禁用。
        refresh_text_proto_each_epoch : bool
            是否每个 epoch 刷新文本原型（仅当文本编码器未冻结时有效）。
        """
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.epochs = epochs
        self.patience = patience
        self.prec = prec
        self.use_amp = prec == "amp" and torch.cuda.is_available()
        self.grad_clip_norm = grad_clip_norm
        self.refresh_text_proto_each_epoch = refresh_text_proto_each_epoch
        self.dataset_name = dataset_name
        self.report_dir = report_dir
        self.monitor_metric = monitor_metric

        self.scaler = GradScaler() if self.use_amp else None

        # 检查点保存目录
        self.ckpt_dir = os.path.join(save_dir, dataset_name)
        os.makedirs(self.ckpt_dir, exist_ok=True)
        self.best_model_path = os.path.join(self.ckpt_dir, "best_model.pth")
        self.best_ckpt_path = os.path.join(self.ckpt_dir, "best_checkpoint.pth")
        print(f"[Trainer] 检查点将保存至：{self.ckpt_dir}")

        # 构建优化器
        base_model = unwrap_model(self.model)
        self.optimizer = build_optimizer(
            base_model,
            lr_text_encoder=lr_text_encoder,
            lr_visual=lr_visual,
            lr_text_proj=lr_text_proj,
            lr_fusion=lr_fusion,
            lr_classifier=lr_classifier,
            lr_adapter=lr_adapter,
            lr_boundary=lr_boundary,
            lr_prompt=lr_prompt,
            lr_cluster_proto=lr_cluster_proto,
            lr_convnext=lr_convnext,
            weight_decay=weight_decay,
        )

        # 打印优化器信息
        print("[Trainer] 优化器参数组配置：")
        total_trainable = 0
        for pg in self.optimizer.param_groups:
            n = sum(p.numel() for p in pg["params"])
            total_trainable += n
            print(f"  - {pg.get('name', '?'):<20} : {n:>10,} params, lr={pg['lr']:.2e}")
        print(f"[Trainer] 可训练参数总数：{total_trainable:,}")

        # 学习率调度器：
        #   warm_restart - 余弦退火带热重启（T_0=10）；从零训练默认
        #   cosine       - 全程平滑余弦衰减，无重启；warm-start 微调推荐
        #                  （重启会在 ep10 把已收敛模型打崩，导致 val 暴跌震荡）
        if lr_schedule == "cosine":
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, T_max=epochs, eta_min=1e-7
            )
            print(f"[Trainer] 学习率调度: CosineAnnealingLR (平滑衰减, T_max={epochs})")
        else:
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
                self.optimizer, T_0=10, T_mult=1, eta_min=1e-7
            )
            print(f"[Trainer] 学习率调度: CosineAnnealingWarmRestarts (T_0=10)")

        # 训练状态跟踪
        self.best_val_acc = 0.0
        self.best_monitor_score = -float("inf")
        self.best_epoch = 0
        self.train_history = {
            "loss": [], "acc": [], "val_loss": [], "val_acc": [],
            "val_macro_f1": [], "val_mae": [],
        }

    # ──────────────────────────────────────────────────────────────────────
    #  检查点管理
    # ──────────────────────────────────────────────────────────────────────

    def _save_checkpoint(self, epoch, val_acc, monitor_score=None):
        """保存最优模型与完整检查点。"""
        state = unwrap_model(self.model).state_dict()
        torch.save(state, self.best_model_path)

        checkpoint = {
            "epoch": epoch,
            "val_acc": val_acc,
            "monitor_metric": self.monitor_metric,
            "monitor_score": float(monitor_score if monitor_score is not None else val_acc),
            "model_state_dict": state,
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "train_history": self.train_history,
        }
        torch.save(checkpoint, self.best_ckpt_path)

    def load_checkpoint(self, ckpt_path=None):
        """从检查点恢复训练状态。"""
        path = ckpt_path or self.best_ckpt_path
        if not os.path.exists(path):
            raise FileNotFoundError(f"检查点文件不存在：{path}")

        checkpoint = torch.load(path, map_location=self.device)
        unwrap_model(self.model).load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        self.best_val_acc = checkpoint["val_acc"]
        self.best_monitor_score = checkpoint.get("monitor_score", self.best_val_acc)
        self.best_epoch = checkpoint["epoch"]
        if "train_history" in checkpoint:
            self.train_history = checkpoint["train_history"]

        print(f"[Trainer] 已恢复检查点：epoch={self.best_epoch}, "
              f"val_acc={self.best_val_acc:.4f}, monitor={self.best_monitor_score:.4f}, path={path}")
        return self.best_epoch, self.best_val_acc

    # ──────────────────────────────────────────────────────────────────────
    #  前向传播与损失计算
    # ──────────────────────────────────────────────────────────────────────

    def _forward_loss(self, images, labels):
        """
        前向传播与损失计算。

        Returns
        -------
        loss : torch.Tensor
            总损失。
        cls_logits : torch.Tensor
            分类 logits [B, 5]。
        """
        out = self.model(images, labels)
        loss = out["loss"]
        if loss.ndim > 0:
            loss = loss.mean()
        cls_logits = out["logits_cls"]
        return loss, cls_logits

    # ──────────────────────────────────────────────────────────────────────
    #  单轮训练
    # ──────────────────────────────────────────────────────────────────────

    def train_one_epoch(self, epoch):
        """
        执行一个训练轮。

        Returns
        -------
        avg_loss : float
            平均损失。
        accuracy : float
            训练精度。
        """
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        total_batches = len(self.train_loader)

        for i, batch in enumerate(self.train_loader):
            images, labels = get_batch_image_label(batch, self.device)
            self.optimizer.zero_grad(set_to_none=True)

            if self.use_amp:
                with autocast_cuda(enabled=True):
                    loss, cls_logits = self._forward_loss(images, labels)
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                if self.grad_clip_norm and self.grad_clip_norm > 0:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), max_norm=self.grad_clip_norm
                    )
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss, cls_logits = self._forward_loss(images, labels)
                loss.backward()
                if self.grad_clip_norm and self.grad_clip_norm > 0:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), max_norm=self.grad_clip_norm
                    )
                self.optimizer.step()

            total_loss += loss.item() * images.size(0)
            correct += (cls_logits.argmax(dim=-1) == labels).sum().item()
            total += images.size(0)

            if (i + 1) % 20 == 0 or (i + 1) == total_batches:
                print(
                    f"  [Epoch {epoch:02d}] {i + 1:5d}/{total_batches} batches | "
                    f"Loss: {loss.item():.4f}",
                    flush=True,
                )

        self.scheduler.step()
        avg_loss = total_loss / max(total, 1)
        accuracy = correct / max(total, 1)
        return avg_loss, accuracy

    # ──────────────────────────────────────────────────────────────────────
    #  验证评估
    # ──────────────────────────────────────────────────────────────────────

    @torch.no_grad()
    def evaluate(self, loader, split_name="Val"):
        """
        在数据集上进行验证。

        Parameters
        ----------
        loader : DataLoader
            验证或测试数据加载器。
        split_name : str
            数据集名称（用于日志）。

        Returns
        -------
        avg_loss : float
            平均损失。
        accuracy : float
            准确率。
        mae : float
            平均绝对误差。
        preds : list
            预测标签列表。
        labels : list
            真实标签列表。
        """
        self.model.eval()
        ce_fn = nn.CrossEntropyLoss()

        all_preds, all_labels = [], []
        total_loss = 0.0
        total = 0

        for batch in loader:
            images, labels = get_batch_image_label(batch, self.device)

            with autocast_cuda(enabled=self.use_amp):
                out = self.model(images, labels=None)
                logits = out["logits_cls"]
                loss = ce_fn(logits, labels)

            total_loss += loss.item() * images.size(0)
            preds = logits.argmax(dim=-1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            total += images.size(0)

        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)
        avg_loss = total_loss / max(total, 1)
        accuracy = float(np.mean(all_preds == all_labels)) if len(all_labels) else 0.0
        mae = float(np.mean(np.abs(all_preds - all_labels))) if len(all_labels) else 0.0

        return avg_loss, accuracy, mae, all_preds.tolist(), all_labels.tolist()

    # ──────────────────────────────────────────────────────────────────────
    #  主训练循环
    # ──────────────────────────────────────────────────────────────────────

    def fit(self):
        """
        执行完整的训练循环，包括早停。
        """
        no_improve = 0
        base_model = unwrap_model(self.model)

        print("\n" + "=" * 90)
        print("开始训练 CLIPFusionContrastiveModel".center(90))
        print("=" * 90)

        for epoch in range(1, self.epochs + 1):
            # 训练
            train_loss, train_acc = self.train_one_epoch(epoch)

            # 可选：刷新文本原型（仅当文本编码器未冻结时）
            if self.refresh_text_proto_each_epoch and hasattr(base_model, "refresh_text_prototypes"):
                base_model.refresh_text_prototypes()
                print(f"  [Epoch {epoch}] 文本原型已刷新")

            # 验证
            val_loss, val_acc, val_mae, val_preds, val_labels = self.evaluate(
                self.val_loader, split_name="Val"
            )
            val_macro_f1 = f1_score(val_labels, val_preds, average="macro", zero_division=0)
            if self.monitor_metric == "macro_f1":
                monitor_score = val_macro_f1
                monitor_name = "Val Macro F1"
            elif self.monitor_metric == "mae":
                monitor_score = -val_mae
                monitor_name = "Val -MAE"
            else:
                monitor_score = val_acc
                monitor_name = "Val Acc"

            # 记录历史
            self.train_history["loss"].append(train_loss)
            self.train_history["acc"].append(train_acc)
            self.train_history["val_loss"].append(val_loss)
            self.train_history["val_acc"].append(val_acc)
            self.train_history["val_macro_f1"].append(float(val_macro_f1))
            self.train_history["val_mae"].append(float(val_mae))

            # 打印日志
            print(
                f"Epoch {epoch:3d} | "
                f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | "
                f"Val Loss: {val_loss:.4f} Acc: {val_acc:.4f} "
                f"MacroF1: {val_macro_f1:.4f} MAE: {val_mae:.4f}"
            )

            # 检查是否是最优模型
            if monitor_score > self.best_monitor_score:
                self.best_val_acc = val_acc
                self.best_monitor_score = monitor_score
                self.best_epoch = epoch
                self._save_checkpoint(epoch, val_acc, monitor_score)
                print(
                    f"  ✓ 更新最优模型 | {monitor_name}: {monitor_score:.4f} | "
                    f"Val Acc: {val_acc:.4f}\n"
                    f"    Model Path: {self.best_model_path}\n"
                    f"    Checkpoint: {self.best_ckpt_path}"
                )
                no_improve = 0
            else:
                no_improve += 1
                if no_improve >= self.patience:
                    print(f"\n[Early Stopping] 在 epoch {epoch} 停止，"
                          f"无改进已持续 {self.patience} 个 epoch")
                    print(
                        f"最优 epoch: {self.best_epoch}, 最优 Val Acc: {self.best_val_acc:.4f}, "
                        f"monitor={self.best_monitor_score:.4f}"
                    )
                    break

        print("\n" + "=" * 90)
        print(
            f"训练完成！最优验证准确率: {self.best_val_acc:.4f}, "
            f"monitor={self.best_monitor_score:.4f} (epoch {self.best_epoch})"
        )
        print("=" * 90)

    # ──────────────────────────────────────────────────────────────────────
    #  测试评估与报告生成
    # ──────────────────────────────────────────────────────────────────────

    def evaluate_and_report(self, test_loader, class_names, num_classes=5):
        """
        在测试集上进行评估，并生成详细报告。

        Parameters
        ----------
        test_loader : DataLoader
            测试数据加载器。
        class_names : list
            类别名称列表。
        num_classes : int
            类别数。
        """
        print("\n" + "=" * 100)
        print("测试集评估与结果报告".center(100))
        print("=" * 100)

        test_loss, test_acc, test_mae, preds, labels = self.evaluate(
            test_loader, split_name="Test"
        )

        preds_arr = np.array(preds)
        labels_arr = np.array(labels)

        # 计算宏平均指标
        precision_macro = precision_score(
            labels_arr, preds_arr, average="macro", zero_division=0
        )
        recall_macro = recall_score(
            labels_arr, preds_arr, average="macro", zero_division=0
        )
        f1_macro = f1_score(
            labels_arr, preds_arr, average="macro", zero_division=0
        )

        # 计算按类别的指标
        precision_per = precision_score(
            labels_arr, preds_arr, average=None, zero_division=0
        )
        recall_per = recall_score(
            labels_arr, preds_arr, average=None, zero_division=0
        )
        f1_per = f1_score(
            labels_arr, preds_arr, average=None, zero_division=0
        )

        # 计算按类别的准确率与 MAE
        acc_per, mae_per = [], []
        for c in range(num_classes):
            mask = labels_arr == c
            acc_c = float(np.mean(preds_arr[mask] == c)) if np.any(mask) else 0.0
            mae_c = float(np.mean(np.abs(preds_arr[mask] - c))) if np.any(mask) else 0.0
            acc_per.append(acc_c)
            mae_per.append(mae_c)

        # 打印整体指标
        print("\n【整体指标】")
        print(f"  Accuracy          : {test_acc:.4f}")
        print(f"  Macro Precision   : {precision_macro:.4f}")
        print(f"  Macro Recall      : {recall_macro:.4f}")
        print(f"  Macro F1-Score    : {f1_macro:.4f}")
        print(f"  MAE               : {test_mae:.4f}")

        # 打印按类别的详细指标
        print("\n【按类别指标】")
        print(f"{'类别':<15} {'Accuracy':>10} {'Precision':>10} "
              f"{'Recall':>10} {'F1-Score':>10} {'MAE':>10}")
        print("-" * 80)
        for i, name in enumerate(class_names):
            print(f"  {name:<13} "
                  f"{acc_per[i]:>10.4f} "
                  f"{precision_per[i]:>10.4f} "
                  f"{recall_per[i]:>10.4f} "
                  f"{f1_per[i]:>10.4f} "
                  f"{mae_per[i]:>10.4f}")
        print("-" * 80)

        # 打印分类报告
        print("\n【sklearn 分类报告】")
        print(classification_report(
            labels_arr, preds_arr,
            target_names=class_names,
            digits=4,
            zero_division=0
        ))

        # 打印混淆矩阵
        print("\n【混淆矩阵】")
        cm = confusion_matrix(labels_arr, preds_arr)
        print(cm)

        out_dir = os.path.join(self.report_dir, self.dataset_name)
        os.makedirs(out_dir, exist_ok=True)
        metrics = {
            "accuracy": test_acc,
            "macro_precision": precision_macro,
            "macro_recall": recall_macro,
            "macro_f1": f1_macro,
            "mae": test_mae,
            "per_class": {
                str(name): {
                    "accuracy": acc_per[i],
                    "precision": float(precision_per[i]),
                    "recall": float(recall_per[i]),
                    "f1": float(f1_per[i]),
                    "mae": mae_per[i],
                }
                for i, name in enumerate(class_names)
            },
            "confusion_matrix": cm.tolist(),
        }
        with open(os.path.join(out_dir, "test_metrics.json"), "w", encoding="utf-8") as f:
            json.dump(metrics, f, ensure_ascii=False, indent=2)
        with open(os.path.join(out_dir, "predictions.csv"), "w", encoding="utf-8") as f:
            f.write("label,pred\n")
            for y, p in zip(labels, preds):
                f.write(f"{y},{p}\n")
        print(f"\n结果文件已保存：{out_dir}")

        print("\n" + "=" * 100)


# ════════════════════════════════════════════════════════════════════════════
#  命令行参数解析
# ════════════════════════════════════════════════════════════════════════════

def parse_args():
    """解析命令行参数。"""
    parser = argparse.ArgumentParser(
        description="膝关节 OA KL 分级模型训练脚本（CLIPFusionContrastiveModel）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法：
  python main.py --data_root /path/to/data --epochs 50 --batch_size 32
  python main.py --data_root /path/to/data --lr_visual 5e-5 --lr_fusion 1e-4
  python main.py --data_root /path/to/data --lambda_ordinal 0.2 --lambda_proto 0.2
        """
    )

    # 数据路径与模型
    parser.add_argument("--data_root", type=str, required=True,
                        help="训练数据根目录路径")
    parser.add_argument("--extra_train_roots", type=str, default="",
                        help="可选：逗号分隔的额外数据集根目录，仅拼接其 train split 用于多源训练")
    parser.add_argument("--include_val_in_train", action="store_true",
                        help="最终训练阶段使用：将当前数据集 val split 也并入训练集")
    parser.add_argument("--clip_model", type=str, default="ViT-B/32",
                        help="CLIP 预训练模型（见 clip.available_models()）")
    parser.add_argument("--init_checkpoint", type=str, default="",
                        help="可选：加载已有融合模型权重作为训练初始化")
    parser.add_argument("--embed_dim", type=int, default=512,
                        help="嵌入空间维度")
    parser.add_argument("--img_size", type=int, default=224,
                        help="输入图像大小")
    parser.add_argument("--backbone", type=str, default="openai_clip",
                        choices=["openai_clip", "biomedclip"],
                        help="视觉主干：openai_clip(原始) 或 biomedclip(领域预训练，仅 full* 变体)")
    parser.add_argument("--clip_normalize", dest="clip_normalize", action="store_true", default=None,
                        help="对输入做 CLIP mean/std 归一化（biomedclip 默认开启）")
    parser.add_argument("--no_clip_normalize", dest="clip_normalize", action="store_false",
                        help="强制关闭 CLIP mean/std 归一化")
    parser.add_argument("--variant", type=str, default="full",
                        choices=["exp1", "exp2", "exp3", "exp4", "full", "full_ldl", "full_vcf", "full_obh", "full_adapter", "full_tpa", "full_opl", "full_ovc", "full_dual"],
                        help="实验变体：exp1/exp2/exp3/exp4 为消融，full_dual = BiomedCLIP(全局)+ConvNeXt-V2(局部)联合训练双分支")

    # 训练超参
    parser.add_argument("--batch_size", type=int, default=16,
                        help="批大小")
    parser.add_argument("--epochs", type=int, default=50,
                        help="最大训练轮数")
    parser.add_argument("--patience", type=int, default=10,
                        help="早停耐心值（验证无改进的轮数）")
    parser.add_argument("--monitor_metric", type=str, default="accuracy",
                        choices=["accuracy", "macro_f1", "mae"],
                        help="保存 best checkpoint 的验证指标")

    # 学习率（分层）
    parser.add_argument("--lr_visual", type=float, default=5e-5,
                        help="视觉 backbone 学习率")
    parser.add_argument("--lr_text_encoder", type=float, default=1e-5,
                        help="PubMedBERT 文本编码器学习率")
    parser.add_argument("--lr_text_proj", type=float, default=1e-4,
                        help="文本投影层学习率")
    parser.add_argument("--lr_fusion", type=float, default=1e-4,
                        help="融合模块学习率")
    parser.add_argument("--lr_classifier", type=float, default=1e-4,
                        help="分类 MLP 学习率")
    parser.add_argument("--lr_adapter", type=float, default=None,
                        help="视觉残差 Adapter 学习率；默认沿用 --lr_fusion")
    parser.add_argument("--lr_boundary", type=float, default=None,
                        help="OBH 边界头学习率；默认沿用 --lr_classifier")
    parser.add_argument("--lr_cluster_proto", type=float, default=1e-4,
                        help="聚类原型矩阵学习率")
    parser.add_argument("--weight_decay", type=float, default=0.01,
                        help="AdamW 权重衰减")

    # 文本编码器控制
    parser.add_argument("--freeze_text_encoder", action="store_true", default=True,
                        help="冻结 PubMedBERT（默认 True，推荐初期冻结加速）")
    parser.add_argument("--unfreeze_text_encoder", action="store_false",
                        dest="freeze_text_encoder",
                        help="解冻 PubMedBERT 进行微调")
    parser.add_argument("--refresh_text_proto", action="store_true", default=False,
                        help="每个 epoch 刷新文本原型（仅当解冻文本编码器时有效）")

    # 融合与分类 Dropout
    parser.add_argument("--fusion_dropout", type=float, default=0.1,
                        help="ClusterAwareResidualGatedFusion 的 Dropout 率")
    parser.add_argument("--cls_dropout", type=float, default=0.2,
                        help="分类 MLP 的 Dropout 率")
    parser.add_argument("--label_smoothing", type=float, default=0.05,
                        help="CrossEntropy label smoothing")
    parser.add_argument("--use_class_weights", action="store_true",
                        help="启用训练集类别权重，增强少样本 KL 等级监督")
    parser.add_argument("--class_weight_power", type=float, default=0.5,
                        help="类别权重指数，0.5 为温和重加权，1.0 为完整反频率重加权")
    parser.add_argument("--lambda_proto", type=float, default=0.2,
                        help="文本原型监督损失权重")
    parser.add_argument("--lambda_cluster", type=float, default=0.1,
                        help="聚类原型监督损失权重")
    parser.add_argument("--lambda_ordinal", type=float, default=0.2,
                        help="KL 有序损失权重")
    parser.add_argument("--lambda_ldl", type=float, default=0.0,
                        help="有序标签分布学习损失权重")
    parser.add_argument("--ldl_sigma", type=float, default=0.8,
                        help="有序标签分布学习的高斯标签宽度")
    parser.add_argument("--logit_adjust_tau", type=float, default=0.0,
                        help="Logit Adjustment 强度 tau（>0 启用，按训练集类别频率调整，专治少数类）")
    parser.add_argument("--adapter_bottleneck_dim", type=int, default=64,
                        help="视觉残差 Adapter 的瓶颈维度")
    parser.add_argument("--lambda_boundary", type=float, default=0.0,
                        help="OBH 有序边界头监督损失权重")
    parser.add_argument("--lambda_visual_cluster", type=float, default=0.0,
                        help="视觉 KL 聚类原型监督损失权重")
    parser.add_argument("--lambda_patch", type=float, default=0.0,
                        help="文本引导 patch 注意力分支监督损失权重")
    parser.add_argument("--patch_adapter_bottleneck_dim", type=int, default=64,
                        help="文本引导 patch 注意力分支的低秩 adapter 瓶颈维度")
    parser.add_argument("--lambda_proto_separation", type=float, default=0.0,
                        help="KL 原型类间分离正则权重")
    parser.add_argument("--n_ctx", type=int, default=4,
                        help="有序软提示学习的可学习 context token 数量（full_opl 变体）")
    parser.add_argument("--lambda_ordinal_prompt", type=float, default=0.1,
                        help="有序提示对比损失权重（约束文本原型保持 KL 等级单调性）")
    parser.add_argument("--lambda_ordinal_proto", type=float, default=0.05,
                        help="有序聚类原型损失权重（约束聚类原型保持 KL 等级单调性）")
    parser.add_argument("--lr_prompt", type=float, default=3e-4,
                        help="有序提示学习 ctx 向量的学习率")
    parser.add_argument("--init_visual_only", action="store_true", default=False,
                        help="从 init_checkpoint 只加载 visual.* 权重，fusion/classifier 随机初始化")
    parser.add_argument("--convnext_variant", type=str, default="convnextv2_large",
                        help="双分支 CNN 主干：convnextv2_tiny / convnextv2_base / convnextv2_large")
    parser.add_argument("--convnext_checkpoint", type=str, default="",
                        help="ConvNeXt-V2 域微调 checkpoint 路径（full_dual 变体）")
    parser.add_argument("--convnext_multi_scale", action="store_true", default=False,
                        help="多尺度局部特征：从 ConvNeXt stage-2 和 stage-3 分别提取、门控融合")
    parser.add_argument("--init_ms_from_single", type=str, default="",
                        help="多尺度热启动：从单尺度双分支 checkpoint 加载 visual.* / "
                             "visual_local.backbone.* / visual_local.norm.* 并将 "
                             "visual_local.proj → visual_local.proj_s3")
    parser.add_argument("--lr_convnext", type=float, default=1e-5,
                        help="ConvNeXt backbone 学习率（域微调初始化用较低值）")
    parser.add_argument("--lambda_emd", type=float, default=0.0,
                        help="序数 EMD 损失权重")
    parser.add_argument("--lambda_decouple", type=float, default=0.0,
                        help="双分支互补解耦正则权重")
    parser.add_argument("--lambda_branch_text", type=float, default=0.0,
                        help="分支特异性文本对齐权重（方向B：ViT↔解剖文本，ConvNeXt↔病理文本）")
    parser.add_argument("--branch_text_detach", action="store_true", default=False,
                        help="文本原型适配模式：detach 视觉特征，梯度只更新可学习文本原型")
    parser.add_argument("--inference_branch_text_weight", type=float, default=0.0,
                        help="推理时分支文本原型集成权重（anatomy+pathology 平均后与主分类器加权）")
    parser.add_argument("--lambda_transition", type=float, default=0.0,
                        help="等级过渡提示损失权重（方向A：4个二元阈值 CE，P(y>k) 文本引导）")
    parser.add_argument("--inference_transition_weight", type=float, default=0.0,
                        help="推理时过渡 logits 融合权重")
    parser.add_argument("--adaptive_text_proto", action="store_true", default=False,
                        help="ATPC：启用可学习文本原型校准向量（在 PubMedBERT 原型上叠加可学习偏移）")
    parser.add_argument("--lambda_text_ord", type=float, default=0.05,
                        help="ATPC 序数约束损失权重（保持校准文本原型在序数轴上单调有序）")
    parser.add_argument("--text_anchor_cluster", action="store_true", default=False,
                        help="OTACP：用 PubMedBERT 文本原型锚定初始化聚类原型（替代随机初始化）")
    parser.add_argument("--lambda_text_distill", type=float, default=0.0,
                        help="OTDD：序数文本分布蒸馏权重（从文本序数轴投影构造软标签）")
    parser.add_argument("--text_distill_sigma", type=float, default=0.8,
                        help="OTDD 软标签高斯核带宽")
    parser.add_argument("--lr_schedule", type=str, default="warm_restart",
                        choices=["warm_restart", "cosine"],
                        help="学习率调度：warm_restart(带热重启)/cosine(平滑衰减,warm-start微调推荐)")
    parser.add_argument("--enable_bsvla", action="store_true", default=False,
                        help="BSVLA：双分支语义专化对齐(投影头+对比对齐+跨分支解耦)")
    parser.add_argument("--lambda_bsvla", type=float, default=0.2,
                        help="BSVLA 专化对齐损失权重(ViT↔解剖, ConvNeXt↔病理)")
    parser.add_argument("--lambda_bsvla_disent", type=float, default=0.1,
                        help="BSVLA 跨分支最大熵解耦权重(off-branch 预测趋于无信息)")
    parser.add_argument("--inference_bsvla_weight", type=float, default=0.0,
                        help="推理时 BSVLA 专化对齐 logits 集成权重")
    parser.add_argument("--lambda_visual_con", type=float, default=0.0,
                        help="有序视觉对比损失权重（OVCL：在视觉特征空间按序数距离推开不同 KL 级别）")
    parser.add_argument("--visual_con_temp", type=float, default=0.1,
                        help="OVCL 对比损失温度系数")
    parser.add_argument("--visual_con_ord_weight", type=float, default=1.0,
                        help="OVCL 序数距离权重系数（越大越强调远距离对的推开）")
    parser.add_argument("--inference_proto_weight", type=float, default=0.0,
                        help="评估时融合文本原型 logits 的权重")
    parser.add_argument("--inference_cluster_weight", type=float, default=0.0,
                        help="评估时融合聚类 logits 的权重")
    parser.add_argument("--inference_visual_cluster_weight", type=float, default=0.0,
                        help="评估时融合视觉聚类 logits 的权重")
    parser.add_argument("--inference_patch_weight", type=float, default=0.0,
                        help="评估时融合文本引导 patch logits 的权重")
    parser.add_argument("--inference_boundary_weight", type=float, default=0.0,
                        help="评估时融合 OBH 边界 logits 的权重")

    # 训练工具
    parser.add_argument("--save_dir", type=str, default="checkpoints",
                        help="检查点保存目录")
    parser.add_argument("--report_dir", type=str, default="results",
                        help="测试指标与预测结果保存目录")
    parser.add_argument("--seed", type=int, default=42,
                        help="随机数种子")
    parser.add_argument("--num_workers", type=int, default=0,
                        help="数据加载器进程数")
    parser.add_argument("--prec", type=str, default="amp", choices=["fp32", "amp"],
                        help="精度模式：'amp' 混合精度，'fp32' 单精度")
    parser.add_argument("--grad_clip_norm", type=float, default=1.0,
                        help="梯度剪裁范数（<=0 禁用）")
    parser.add_argument("--no_anomaly", action="store_true", default=False,
                        help="禁用 PyTorch 异常检测")

    # PubMedBERT 路径
    parser.add_argument("--pubmedbert_path", type=str,
                        default=str(PROJECT_ROOT / "PubMedBERT"),
                        help="PubMedBERT 本地路径")

    return parser.parse_args()


# ════════════════════════════════════════════════════════════════════════════
#  主程序入口
# ════════════════════════════════════════════════════════════════════════════

def main():
    """主程序入口。"""
    args = parse_args()

    # 基本配置
    DATA_ROOT = args.data_root
    CLIP_MODEL = args.clip_model
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    DATASET_NAME = f"{os.path.basename(os.path.normpath(DATA_ROOT))}_{args.variant}"

    # backbone 校验与归一化默认值
    if args.backbone == "biomedclip" and args.variant not in {
        "full", "full_ldl", "full_vcf", "full_obh", "full_adapter", "full_opl", "full_ovc", "full_dual"
    }:
        raise ValueError(
            f"--backbone biomedclip 目前仅支持 full* 变体，收到 variant={args.variant}。"
            f"（full_tpa 需要 patch token，BiomedCLIP 视觉塔暂未提供 forward_tokens）"
        )
    if args.clip_normalize is None:
        # biomedclip 用 CLIP mean/std 预训练，默认开启归一化以保留迁移能力；
        # openai_clip 保持原有行为（[0,1] 输入，不归一化）。
        args.clip_normalize = (args.backbone == "biomedclip")

    print("\n" + "=" * 100)
    print("CLIPFusionContrastiveModel 训练脚本".center(100))
    print("=" * 100)
    print(f"[配置] 数据根目录: {DATA_ROOT}")
    print(f"[配置] CLIP 模型: {CLIP_MODEL}")
    print(f"[配置] 实验变体: {args.variant}")
    print(f"[配置] 计算设备: {DEVICE}")
    print(f"[配置] 精度模式: {args.prec}")
    print(f"[配置] 文本编码器: {'冻结' if args.freeze_text_encoder else '解冻'}")
    print(f"[配置] 视觉主干: {args.backbone} | 输入尺寸: {args.img_size} | CLIP归一化: {args.clip_normalize}")

    # 设置随机种子
    seed_everything(args.seed)
    if not args.no_anomaly:
        torch.autograd.set_detect_anomaly(True)

    # ════════════════════════════════════════════════════════════════════════
    #  1. 加载 CLIP 预训练权重
    # ════════════════════════════════════════════════════════════════════════

    print("\n[阶段 1] 加载 CLIP 预训练权重...")
    try:
        clip_state_dict = load_clip_state_dict(CLIP_MODEL)
        print(f"✓ CLIP 权重加载成功")
    except Exception as e:
        print(f"✗ CLIP 权重加载失败: {e}")
        raise

    # 补全缺失的 text_projection 顶层键
    clip_state_dict = patch_state_dict_text_projection(
        clip_state_dict, fallback_embed_dim=args.embed_dim
    )

    # ════════════════════════════════════════════════════════════════════════
    #  2. 构建 CLIPFusionContrastiveModel
    # ════════════════════════════════════════════════════════════════════════

    print("\n[阶段 2] 构建 CLIPFusionContrastiveModel...")
    try:
        if args.variant in {"full", "full_ldl", "full_vcf", "full_obh", "full_adapter", "full_tpa", "full_opl", "full_ovc", "full_dual"}:
            model = build_fusion_model(
                state_dict=clip_state_dict,
                pubmedbert_path=args.pubmedbert_path,
                backbone=args.backbone,
                img_size=args.img_size,
                freeze_text_encoder=args.freeze_text_encoder,
                fusion_dropout=args.fusion_dropout,
                cls_dropout=args.cls_dropout,
                label_smoothing=args.label_smoothing,
                lambda_proto=args.lambda_proto,
                lambda_cluster=args.lambda_cluster,
                lambda_ordinal=args.lambda_ordinal,
                lambda_ldl=args.lambda_ldl,
                ldl_sigma=args.ldl_sigma,
                logit_adjust_tau=args.logit_adjust_tau,
                enable_feature_adapter=(args.variant == "full_adapter"),
                adapter_bottleneck_dim=args.adapter_bottleneck_dim,
                enable_ordinal_boundary=(args.variant == "full_obh"),
                lambda_boundary=args.lambda_boundary,
                enable_visual_cluster=(args.variant == "full_vcf"),
                lambda_visual_cluster=args.lambda_visual_cluster,
                enable_semantic_patch=(args.variant == "full_tpa"),
                lambda_patch=args.lambda_patch,
                patch_adapter_bottleneck_dim=args.patch_adapter_bottleneck_dim,
                lambda_proto_separation=args.lambda_proto_separation,
                inference_proto_weight=args.inference_proto_weight,
                inference_cluster_weight=args.inference_cluster_weight,
                inference_visual_cluster_weight=args.inference_visual_cluster_weight,
                inference_patch_weight=args.inference_patch_weight,
                inference_boundary_weight=args.inference_boundary_weight,
                enable_prompt_learning=(args.variant == "full_opl"),
                n_ctx=args.n_ctx,
                lambda_ordinal_prompt=args.lambda_ordinal_prompt,
                lambda_ordinal_proto=args.lambda_ordinal_proto,
                lambda_visual_con=args.lambda_visual_con,
                visual_con_temperature=args.visual_con_temp,
                visual_con_ordinal_weight=args.visual_con_ord_weight,
                enable_dual_branch=(args.variant == "full_dual"),
                convnext_variant=args.convnext_variant,
                convnext_checkpoint=args.convnext_checkpoint,
                convnext_multi_scale=getattr(args, 'convnext_multi_scale', False),
                lambda_emd=getattr(args, 'lambda_emd', 0.0),
                lambda_decouple=getattr(args, 'lambda_decouple', 0.0),
                lambda_branch_text=getattr(args, 'lambda_branch_text', 0.0),
                branch_text_detach=getattr(args, 'branch_text_detach', False),
                inference_branch_text_weight=getattr(args, 'inference_branch_text_weight', 0.0),
                lambda_transition=getattr(args, 'lambda_transition', 0.0),
                inference_transition_weight=getattr(args, 'inference_transition_weight', 0.0),
                adaptive_text_proto=getattr(args, 'adaptive_text_proto', False),
                lambda_text_ord=getattr(args, 'lambda_text_ord', 0.05),
                text_anchor_cluster=getattr(args, 'text_anchor_cluster', False),
                lambda_text_distill=getattr(args, 'lambda_text_distill', 0.0),
                text_distill_sigma=getattr(args, 'text_distill_sigma', 0.8),
                enable_bsvla=getattr(args, 'enable_bsvla', False),
                lambda_bsvla=getattr(args, 'lambda_bsvla', 0.0),
                lambda_bsvla_disent=getattr(args, 'lambda_bsvla_disent', 0.0),
                inference_bsvla_weight=getattr(args, 'inference_bsvla_weight', 0.0),
            )
        else:
            model = build_ablation_model(
                variant=args.variant,
                state_dict=clip_state_dict,
                pubmedbert_path=args.pubmedbert_path,
                freeze_text_encoder=args.freeze_text_encoder,
                fusion_dropout=args.fusion_dropout,
                cls_dropout=args.cls_dropout,
            )
        model = model.to(DEVICE)
        print(f"✓ 模型构建成功")

        if getattr(args, 'init_ms_from_single', '') and getattr(args, 'convnext_multi_scale', False):
            # 多尺度热启动：只加载 backbone 权重，跳过 fusion/classifier（避免单尺度特征分布负迁移）
            # 加载范围：visual.* + visual_local.backbone.* + visual_local.norm.* + proj→proj_s3
            BACKBONE_PREFIXES = ('visual.', 'visual_local.backbone.', 'visual_local.norm.')
            current_sd = model.state_dict()
            raw = torch.load(args.init_ms_from_single, map_location=DEVICE)
            init_sd = {}
            proj_s3_mapped = False
            for k, v in raw.items():
                if any(k.startswith(p) for p in BACKBONE_PREFIXES):
                    if k in current_sd and tuple(v.shape) == tuple(current_sd[k].shape):
                        init_sd[k] = v
                # 单尺度 proj (1536→512) → 多尺度 proj_s3 (1536→512)
                if k == 'visual_local.proj.weight':
                    k_ms = 'visual_local.proj_s3.weight'
                    if k_ms in current_sd and tuple(v.shape) == tuple(current_sd[k_ms].shape):
                        init_sd[k_ms] = v
                        proj_s3_mapped = True
            model.load_state_dict({**current_sd, **init_sd})
            print(f"✓ 多尺度热启动(backbone-only): {args.init_ms_from_single}")
            print(f"  加载 {len(init_sd)} 个键，proj→proj_s3 映射: {proj_s3_mapped}")
        elif args.init_checkpoint:
            if getattr(args, 'init_visual_only', False):
                # Only load visual.* weights; leave fusion/classifier randomly initialized
                current_sd = model.state_dict()
                raw = torch.load(args.init_checkpoint, map_location=DEVICE)
                visual_only = {
                    k: v for k, v in raw.items()
                    if k.startswith("visual.") and k in current_sd
                    and tuple(v.shape) == tuple(current_sd[k].shape)
                }
                model.load_state_dict({**current_sd, **visual_only})
                print(f"✓ 已加载 visual-only 权重: {args.init_checkpoint}")
                print(f"  loaded visual keys: {len(visual_only)}")
            else:
                loaded_count, skipped = load_compatible_state_dict(
                    model, args.init_checkpoint, DEVICE
                )
                print(f"✓ 已加载初始化权重: {args.init_checkpoint}")
                print(f"  compatible tensors: {loaded_count}, skipped tensors: {len(skipped)}")
                if skipped:
                    print(f"  skipped examples: {skipped[:8]}")

        # AMP keeps trainable parameters in fp32 and casts compute safely.
        # Do not call convert_weights() during training, otherwise GradScaler
        # will see fp16 gradients on trainable parameters.

    except Exception as e:
        print(f"✗ 模型构建失败: {e}")
        raise

    # ════════════════════════════════════════════════════════════════════════
    #  3. 初始化文本原型
    # ════════════════════════════════════════════════════════════════════════

    print("\n[阶段 3] 初始化文本原型...")
    try:
        if hasattr(model, "refresh_text_prototypes"):
            model.refresh_text_prototypes()
            print(f"✓ 文本原型初始化完成")
        elif hasattr(model, "_init_text_prototypes"):
            model._init_text_prototypes()
            print(f"✓ 文本原型初始化完成")
        else:
            print("✓ 当前变体无文本原型，跳过")
    except Exception as e:
        print(f"✗ 文本原型初始化失败: {e}")
        raise

    # 打印参数统计
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"[参数统计]")
    print(f"  可训练参数: {trainable:,}")
    print(f"  总参数数: {total:,}")
    print(f"  训练占比: {100 * trainable / max(total, 1):.1f}%")

    # ════════════════════════════════════════════════════════════════════════
    #  4. 数据集与数据加载器
    # ════════════════════════════════════════════════════════════════════════

    print("\n[阶段 4] 加载数据集...")

    # 数据增强配置
    train_tf = get_transforms(
        img_size=args.img_size,
        is_training=True,
        use_clahe=True,
        to_rgb=True,
        clahe_clip_limit=2.0,
        clahe_tile_grid_size=(8, 8),
        percentile=(1.0, 99.0),
        rotate_deg=10,
        hflip_p=0.5,
        random_erasing_p=0.0,
        random_erasing_scale=(0.02, 0.15),
        random_erasing_ratio=(0.3, 3.3),
        random_erasing_value=0.0,
        normalize=args.clip_normalize,
    )
    val_tf = get_transforms(
        img_size=args.img_size,
        is_training=False,
        use_clahe=True,
        to_rgb=True,
        clahe_clip_limit=2.0,
        clahe_tile_grid_size=(8, 8),
        percentile=(1.0, 99.0),
        normalize=args.clip_normalize,
    )

    # 加载数据集
    try:
        train_ds = ClassificationDataset(
            DATA_ROOT, split="train",
            transform=train_tf, error_policy="raise",
            include_path=False
        )
        extra_train_roots = [x.strip() for x in args.extra_train_roots.split(",") if x.strip()]
        if extra_train_roots:
            train_parts = [train_ds]
            for extra_root in extra_train_roots:
                extra_ds = ClassificationDataset(
                    extra_root, split="train",
                    transform=train_tf, error_policy="raise",
                    include_path=False,
                )
                if extra_ds.class_to_idx != train_ds.class_to_idx:
                    raise ValueError(
                        f"额外训练集类别映射不一致: {extra_root} "
                        f"{extra_ds.class_to_idx} != {train_ds.class_to_idx}"
                    )
                train_parts.append(extra_ds)
            merged_train = ConcatDataset(train_parts)
            merged_train.classes = train_ds.classes
            merged_train.class_to_idx = train_ds.class_to_idx
            merged_train.labels = [y for part in train_parts for y in part.labels]
            merged_train.samples = [sample for part in train_parts for sample in part.samples]
            train_ds = merged_train
            print(f"  多源训练已启用: {len(train_parts)} 个 train split, 合计 {len(train_ds):,} 样本")
        if args.include_val_in_train:
            val_train_ds = ClassificationDataset(
                DATA_ROOT, split="val",
                transform=train_tf, error_policy="raise",
                include_path=False,
            )
            if val_train_ds.class_to_idx != train_ds.class_to_idx:
                raise ValueError(
                    f"val 类别映射不一致: {val_train_ds.class_to_idx} != {train_ds.class_to_idx}"
                )
            train_parts = [train_ds, val_train_ds]
            merged_train = ConcatDataset(train_parts)
            merged_train.classes = train_ds.classes
            merged_train.class_to_idx = train_ds.class_to_idx
            merged_train.labels = [y for part in train_parts for y in part.labels]
            merged_train.samples = [sample for part in train_parts for sample in part.samples]
            train_ds = merged_train
            print(f"  当前数据集 val 已并入训练: 合计 {len(train_ds):,} 样本")
        val_ds = ClassificationDataset(
            DATA_ROOT, split="val",
            transform=val_tf, error_policy="raise",
            include_path=False
        )
        test_ds = ClassificationDataset(
            DATA_ROOT, split="test",
            transform=val_tf, error_policy="raise",
            include_path=False
        )
        print(f"✓ 数据集加载成功")
        print(f"  训练集: {len(train_ds):,} 样本")
        print(f"  验证集: {len(val_ds):,} 样本")
        print(f"  测试集: {len(test_ds):,} 样本")
        if args.use_class_weights and hasattr(model, "ce_loss_fn"):
            class_weights = train_ds.get_class_weights().float()
            class_weights = class_weights.pow(args.class_weight_power)
            class_weights = class_weights / class_weights.mean().clamp(min=1e-6)
            model.ce_loss_fn = nn.CrossEntropyLoss(
                weight=class_weights.to(DEVICE),
                label_smoothing=args.label_smoothing,
            )
            print(f"  启用类别权重: {[round(float(x), 4) for x in class_weights.tolist()]}")
        # Logit Adjustment：用训练集类别频率设置对数先验
        if getattr(args, "logit_adjust_tau", 0.0) > 0 and hasattr(model, "set_class_prior"):
            labels = np.asarray(getattr(train_ds, "labels", []))
            counts = np.bincount(labels, minlength=model.NUM_CLASSES) if labels.size else np.ones(model.NUM_CLASSES)
            model.set_class_prior(counts)
            print(f"  启用 Logit Adjustment: tau={args.logit_adjust_tau}, 类别计数={counts.tolist()}")
    except Exception as e:
        print(f"✗ 数据集加载失败: {e}")
        raise

    # 验证类别数匹配
    assert len(train_ds.classes) == model.NUM_CLASSES, (
        f"数据集类别数 {len(train_ds.classes)} ≠ 模型类别数 {model.NUM_CLASSES}，"
        "请检查数据集或修改 CLIPFusionContrastiveModel.NUM_CLASSES"
    )
    print(f"  类别数: {len(train_ds.classes)} ({model.NUM_CLASSES} expected) ✓")

    # 创建数据加载器
    train_loader = create_balanced_dataloader(
        train_ds, batch_size=args.batch_size, num_workers=args.num_workers
    )
    val_loader = create_dataloader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers
    )
    test_loader = create_dataloader(
        test_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers
    )

    # ════════════════════════════════════════════════════════════════════════
    #  5. DataParallel（多卡支持）
    # ════════════════════════════════════════════════════════════════════════

    if torch.cuda.device_count() > 1:
        print(f"\n[多卡] 检测到 {torch.cuda.device_count()} 张 GPU，启用 DataParallel")
        model = nn.DataParallel(model)

    # ════════════════════════════════════════════════════════════════════════
    #  6. 训练器初始化与训练
    # ════════════════════════════════════════════════════════════════════════

    print("\n[阶段 5] 初始化训练器...")
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=DEVICE,
        lr_visual=args.lr_visual,
        lr_text_encoder=args.lr_text_encoder,
        lr_text_proj=args.lr_text_proj,
        lr_fusion=args.lr_fusion,
        lr_classifier=args.lr_classifier,
        lr_adapter=args.lr_adapter,
        lr_boundary=args.lr_boundary,
        lr_cluster_proto=args.lr_cluster_proto,
        lr_prompt=args.lr_prompt,
        lr_convnext=args.lr_convnext,
        weight_decay=args.weight_decay,
        epochs=args.epochs,
        patience=args.patience,
        dataset_name=DATASET_NAME,
        save_dir=args.save_dir,
        prec=args.prec,
        grad_clip_norm=args.grad_clip_norm,
        refresh_text_proto_each_epoch=(
            args.refresh_text_proto and not args.freeze_text_encoder
        ),
        report_dir=args.report_dir,
        monitor_metric=args.monitor_metric,
        lr_schedule=getattr(args, 'lr_schedule', 'warm_restart'),
    )
    print(f"✓ 训练器初始化完成")

    # 执行训练
    print("\n[阶段 6] 执行训练...")
    trainer.fit()

    # ════════════════════════════════════════════════════════════════════════
    #  7. 测试集评估
    # ════════════════════════════════════════════════════════════════════════

    print("\n[阶段 7] 在测试集上进行最终评估...")

    # 加载最优模型
    target_model = unwrap_model(model)
    target_model.load_state_dict(
        torch.load(trainer.best_model_path, map_location=DEVICE)
    )
    target_model = target_model.to(DEVICE)
    trainer.model = target_model

    # 生成测试报告
    class_names = [train_ds.classes[i] for i in range(len(train_ds.classes))]
    trainer.evaluate_and_report(
        test_loader, class_names,
        num_classes=unwrap_model(model).NUM_CLASSES,
    )

    print("\n" + "=" * 100)
    print("训练脚本完成！".center(100))
    print("=" * 100)


if __name__ == "__main__":
    main()
