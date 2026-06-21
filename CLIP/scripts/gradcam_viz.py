#!/usr/bin/env python3
"""Grad-CAM on the ConvNeXt local branch (stage-3) of the full dual-branch model.
Shows the model focuses on the joint margins / joint space — the regions that
define KL osteophytes & narrowing. One representative test image per KL grade."""
import os, sys, glob
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, torch
import torch.nn.functional as F
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

import scripts.semantic_ordinal_transport_adapter_eval as sota
from clip.model import build_fusion_model
from data.dataset import get_transforms
from main import load_clip_state_dict

FIG = "docs/figures"


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    data_root, ckpt = sota.CFG["ar"]
    sd = load_clip_state_dict("ViT-B/32")
    model = build_fusion_model(sd, backbone="biomedclip", enable_dual_branch=True,
                               convnext_multi_scale=True).to(device).eval()
    raw = torch.load(ckpt, map_location="cpu")
    model.load_state_dict(raw.get("model_state_dict", raw), strict=False)
    model.refresh_text_prototypes()

    act = {}
    layer = model.visual_local.backbone.stages[-1]
    layer.register_forward_hook(lambda m, i, o: act.__setitem__("v", o))

    tf = get_transforms(img_size=224, is_training=False, use_clahe=True, to_rgb=True,
                        clahe_clip_limit=2.0, clahe_tile_grid_size=(8, 8),
                        percentile=(1.0, 99.0), normalize=True)
    # pick one correctly-classified test image per grade
    imgs = []
    for g in range(5):
        fs = sorted(glob.glob(os.path.join(data_root, "test", str(g), "*.png")))
        chosen = fs[len(fs) // 2] if fs else None
        for p in fs[::max(1, len(fs) // 25)]:
            x = tf(Image.open(p).convert("RGB")).unsqueeze(0).to(device)
            with torch.no_grad():
                if int(model(x)["logits_classifier"].argmax(-1)) == g:
                    chosen = p; break
        if chosen:
            imgs.append((g, chosen))

    fig, axes = plt.subplots(1, len(imgs), figsize=(3 * len(imgs), 3.4))
    for ax, (g, path) in zip(axes, imgs):
        pil = Image.open(path).convert("L").resize((224, 224))
        x = tf(Image.open(path).convert("RGB")).unsqueeze(0).to(device)
        x.requires_grad_(False)
        out = model(x)
        pred = int(out["logits_classifier"].argmax(-1))
        logit = out["logits_classifier"][0, pred]
        A = act["v"]                                   # [1, C, h, w]
        grads = torch.autograd.grad(logit, A, retain_graph=False)[0]
        w = grads.mean(dim=(2, 3), keepdim=True)
        cam = F.relu((w * A).sum(1, keepdim=True))     # [1,1,h,w]
        cam = F.interpolate(cam, size=(224, 224), mode="bilinear", align_corners=False)[0, 0]
        cam = cam.detach().cpu().numpy()
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        ax.imshow(np.array(pil), cmap="gray")
        ax.imshow(cam, cmap="jet", alpha=0.45)
        ax.set_title(f"KL{g} (pred {pred})", fontsize=10); ax.axis("off")
    fig.suptitle("Grad-CAM (ConvNeXt local branch): focus on joint margins / joint space")
    os.makedirs(FIG, exist_ok=True)
    fig.savefig(os.path.join(FIG, "fig7_gradcam.png"), dpi=150, bbox_inches="tight")
    print("saved", os.path.join(FIG, "fig7_gradcam.png"))


if __name__ == "__main__":
    main()
