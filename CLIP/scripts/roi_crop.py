#!/usr/bin/env python3
"""
膝关节 ROI 自适应裁剪前端。

私有数据集（split_result_siyouceshi）是 631×709 的单膝正位片，但裁得很松：左右
有黑边、上下带多余股骨/胫骨干，真正诊断关键的关节区只占画面中部。MedicalExpert /
archive 则已紧裁到关节。本模块把私有图裁到与之对齐的紧框，使同一条 BiomedCLIP /
ConvNeXt 管线在三个数据集上看到一致的输入。

方法（无需训练、完全可复现）：
  1. 转灰度，Otsu 阈值得到前景（骨/软组织 vs 黑背景）掩膜；
  2. 取前景行/列投影，按密度阈值定出前景包围盒，去掉左右黑边与上下空白；
  3. 以包围盒为基础取「以竖直中心（≈关节线）为中心的正方形」，加小边距，
     与 archive 的近正方形紧框对齐。

对已经基本填满的图，裁剪近似恒等，故对其它数据集也安全（可统一调用）。
"""
import numpy as np
from PIL import Image


def _otsu_threshold(gray: np.ndarray) -> int:
    hist = np.bincount(gray.ravel(), minlength=256).astype(float)
    total = gray.size
    sum_total = np.dot(np.arange(256), hist)
    sum_b = 0.0
    w_b = 0.0
    best_t, best_var = 0, -1.0
    for t in range(256):
        w_b += hist[t]
        if w_b == 0:
            continue
        w_f = total - w_b
        if w_f == 0:
            break
        sum_b += t * hist[t]
        m_b = sum_b / w_b
        m_f = (sum_total - sum_b) / w_f
        var_between = w_b * w_f * (m_b - m_f) ** 2
        if var_between > best_var:
            best_var, best_t = var_between, t
    return best_t


def crop_knee_roi(img: Image.Image, density_frac: float = 0.15, margin: float = 0.06) -> Image.Image:
    """把单膝正位片裁到以关节线为中心的紧方框。返回 RGB/灰度同输入模式不变的 PIL。"""
    mode = img.mode
    gray = np.asarray(img.convert("L"))
    H, W = gray.shape
    t = _otsu_threshold(gray)
    fg = gray > t

    col = fg.mean(axis=0)  # 每列前景占比
    row = fg.mean(axis=1)
    # 以各自最大密度的 density_frac 为阈值定边界，去掉稀疏的黑边
    cthr = col.max() * density_frac
    rthr = row.max() * density_frac
    cols = np.where(col >= cthr)[0]
    rows = np.where(row >= rthr)[0]
    if len(cols) == 0 or len(rows) == 0:
        return img
    x0, x1 = cols[0], cols[-1]
    y0, y1 = rows[0], rows[-1]
    bw, bh = x1 - x0 + 1, y1 - y0 + 1

    # 取以包围盒中心为中心的正方形（边长取较长边，关节线≈竖直中心）
    cx, cy = (x0 + x1) / 2.0, (y0 + y1) / 2.0
    side = max(bw, bh)
    side = int(side * (1 + 2 * margin))
    half = side / 2.0
    left = int(round(cx - half)); right = int(round(cx + half))
    top = int(round(cy - half)); bottom = int(round(cy + half))
    # 越界则用边界填充（pad），保证输出正方形、不拉伸
    pad_l = max(0, -left); pad_t = max(0, -top)
    pad_r = max(0, right - W); pad_b = max(0, bottom - H)
    if pad_l or pad_t or pad_r or pad_b:
        arr = np.asarray(img.convert("L"))
        arr = np.pad(arr, ((pad_t, pad_b), (pad_l, pad_r)), mode="edge")
        base = Image.fromarray(arr, mode="L")
        left += pad_l; right += pad_l; top += pad_t; bottom += pad_t
    else:
        base = img.convert("L")
    out = base.crop((left, top, right, bottom))
    return out.convert(mode) if mode != "L" else out


if __name__ == "__main__":
    import glob
    import random
    random.seed(2)
    fs = sorted(glob.glob("../私有数据集/split_result_siyouceshi/train/*/*.png"))
    pick = random.sample(fs, 10)
    cell = 160
    canvas = Image.new("L", (cell * 5, cell * 4), 255)
    for i, f in enumerate(pick):
        im = Image.open(f)
        orig = im.convert("L").resize((cell, cell))
        roi = crop_knee_roi(im).convert("L").resize((cell, cell))
        col = i % 5
        block = (i // 5) * 2  # 每 5 张占两行：上原图、下裁剪
        canvas.paste(orig, (col * cell, block * cell))
        canvas.paste(roi, (col * cell, (block + 1) * cell))
    canvas.convert("RGB").save("/tmp/roi_check.png")
    print("saved /tmp/roi_check.png  （奇数行=原图，偶数行=ROI 裁剪结果）")
