#!/usr/bin/env python3
"""
Grad-CAM 关节区注意力热力图：在 X-ray 上叠加模型关注区域，佐证模型聚焦关节区
（以及 ROI 前端把注意力收敛到关节的效果）。两种主干都支持：

  - ConvNeXt-V2（纯卷积）：在最后一个卷积 stage（[B,C,7,7]）上做标准 Grad-CAM；
  - BiomedCLIP（ViT-B/16）：在最后一个 Transformer block 的 token 特征
    （[B,1+g*g,C]，去掉 CLS 后 reshape 成 g×g）上做 token 级 Grad-CAM。

每类抽若干张测试图，左原图（CLAHE 后）右热力叠加，按类拼图保存。反向传播的目标
类默认用模型预测类（--target true 改为真实类）。

用法：
  # ConvNeXt-V2（ROI 裁剪版）
  python scripts/gradcam_eval.py --backbone convnext \
    --checkpoint experiments/checkpoints/checkpoints_roi_cvx/best_model.pth --img_size 224 \
    --data_root ../私有数据集/split_result_siyouceshi_roi --out_dir figures/gradcam/cvx_roi
  # BiomedCLIP（ROI 裁剪版）
  python scripts/gradcam_eval.py --backbone biomedclip \
    --checkpoint experiments/checkpoints/checkpoints_roi_bm224/split_result_siyouceshi_roi_full/best_model.pth --img_size 224 \
    --data_root ../私有数据集/split_result_siyouceshi_roi --out_dir figures/gradcam/bm_roi
"""
import argparse
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.cm as cm

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from data.dataset import ClassificationDataset, get_transforms, seed_everything  # noqa: E402

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def build_model(backbone, checkpoint, img_size, convnext_model):
    if backbone == "convnext":
        import timm
        model = timm.create_model(convnext_model, pretrained=False, num_classes=5)
        model.load_state_dict(torch.load(checkpoint, map_location="cpu"))
        target_layer = model.stages[-1]      # [B,C,h,w]
        is_vit = False
    else:
        from main import load_clip_state_dict, patch_state_dict_text_projection
        from clip.model import build_fusion_model
        sd = patch_state_dict_text_projection(load_clip_state_dict("ViT-B/32"), 512)
        model = build_fusion_model(state_dict=sd, backbone="biomedclip", img_size=img_size,
                                   freeze_text_encoder=True)
        model.load_state_dict(torch.load(checkpoint, map_location="cpu"), strict=False)
        if hasattr(model, "refresh_text_prototypes"):
            model.refresh_text_prototypes()
        # 取最后一个 block 的 norm1（LayerNorm）输出做 Grad-CAM：已归一化、无残差 DC，
        # localization 远好于直接用 block 输出（pytorch-grad-cam 对 ViT 的标准建议）。
        target_layer = model.visual.trunk.blocks[-1].norm1   # [B,1+g*g,C]
        is_vit = True
    return model.to(DEVICE).eval(), target_layer, is_vit


def forward_logits(model, x, is_vit):
    out = model(x)
    return out["logits_cls"] if isinstance(out, dict) else out


def gradcam_map(act, grad, is_vit):
    """act/grad: 卷积为 [1,C,h,w]，ViT 为 [1,N,C]。返回 [g,g] 的 0-1 热力图。"""
    if not is_vit:
        weights = grad.mean(dim=(2, 3), keepdim=True)            # 通道重要度
        cam = F.relu((weights * act).sum(dim=1))[0]              # [h,w]
    else:
        a, g = act[0, 1:], grad[0, 1:]                           # 去掉 CLS -> [N-1,C]
        score = (a * g).sum(dim=-1)                              # 每 token 重要度 [N-1]
        score = score - score.mean()                            # 去 DC：残差让 token 有大的共有分量
        score = F.relu(score)                                   # 只保留高于平均的 token
        side = int(round(score.shape[0] ** 0.5))
        cam = score.reshape(side, side)
    cam = cam.detach().float()
    cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
    return cam.cpu().numpy()


def overlay(disp_rgb01, cam, alpha=0.45):
    """disp_rgb01: [H,W,3] in 0-1；cam: [g,g] 0-1。返回叠加后的 uint8 RGB。"""
    H, W = disp_rgb01.shape[:2]
    cam_up = np.asarray(Image.fromarray((cam * 255).astype(np.uint8)).resize((W, H), Image.BICUBIC)) / 255.0
    heat = cm.jet(cam_up)[..., :3]
    blend = (1 - alpha) * disp_rgb01 + alpha * heat
    return (np.clip(blend, 0, 1) * 255).astype(np.uint8)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backbone", choices=["convnext", "biomedclip"], required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--data_root", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--img_size", type=int, default=224)
    ap.add_argument("--convnext_model", default="convnextv2_base.fcmae_ft_in22k_in1k")
    ap.add_argument("--split", default="test")
    ap.add_argument("--num_per_class", type=int, default=3)
    ap.add_argument("--target", choices=["pred", "true"], default="pred")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    seed_everything(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)

    model, target_layer, is_vit = build_model(args.backbone, args.checkpoint, args.img_size, args.convnext_model)
    normalize = (args.backbone == "biomedclip")
    tf_model = get_transforms(img_size=args.img_size, is_training=False, use_clahe=True, to_rgb=True,
                              clahe_clip_limit=2.0, clahe_tile_grid_size=(8, 8),
                              percentile=(1.0, 99.0), normalize=normalize)
    tf_disp = get_transforms(img_size=args.img_size, is_training=False, use_clahe=True, to_rgb=True,
                             clahe_clip_limit=2.0, clahe_tile_grid_size=(8, 8),
                             percentile=(1.0, 99.0), normalize=False)
    ds = ClassificationDataset(args.data_root, split=args.split, transform=None,
                               error_policy="raise", include_path=True)

    # 钩子：抓最后一层激活与其梯度
    store = {}
    def fwd_hook(m, i, o):
        store["act"] = o
        o.register_hook(lambda g: store.__setitem__("grad", g))
    h = target_layer.register_forward_hook(fwd_hook)

    # 按类收集样本索引
    by_class = {c: [] for c in range(5)}
    for idx in range(len(ds)):
        item = ds[idx]
        c = int(item["label"])
        if len(by_class[c]) < args.num_per_class:
            by_class[c].append((idx, item["path"]))
        if all(len(v) >= args.num_per_class for v in by_class.values()):
            break

    cell = args.img_size
    rows = []
    for c in range(5):
        row_imgs = []
        for idx, path in by_class[c]:
            pil = Image.open(path)
            x = tf_model(pil).unsqueeze(0).to(DEVICE)
            disp = tf_disp(pil).permute(1, 2, 0).cpu().numpy()  # [H,W,3] 0-1
            x.requires_grad_(False)
            model.zero_grad(set_to_none=True)
            logits = forward_logits(model, x, is_vit)
            tgt = int(logits.argmax(1)) if args.target == "pred" else c
            logits[0, tgt].backward()
            cam = gradcam_map(store["act"], store["grad"], is_vit)
            ov = overlay(disp, cam)
            disp_u8 = (disp * 255).astype(np.uint8)
            pair = np.concatenate([disp_u8, ov], axis=1)  # 左原图右叠加
            row_imgs.append(pair)
        rows.append(np.concatenate(row_imgs, axis=1))
    h.remove()

    # 拼成大图：每行一个类
    maxw = max(r.shape[1] for r in rows)
    rows = [np.pad(r, ((0, 0), (0, maxw - r.shape[1]), (0, 0)), constant_values=255) for r in rows]
    grid = np.concatenate(rows, axis=0)
    out_path = os.path.join(args.out_dir, f"gradcam_{args.backbone}_{args.split}.png")
    Image.fromarray(grid).save(out_path)
    print(f"[saved] {out_path}  （每行一个类 KL0..KL4，每张为 左原图 | 右 Grad-CAM 叠加）")


if __name__ == "__main__":
    main()
