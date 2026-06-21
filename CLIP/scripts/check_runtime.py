#!/usr/bin/env python3
"""环境自检：确认关键依赖已安装、版本，以及 CUDA 是否可用。

用法：
    python scripts/check_runtime.py
缺失的包会打印 MISSING 并以非零退出，按提示 pip 安装即可。
"""
import importlib
import sys

REQUIRED = [
    "torch", "torchvision", "transformers", "timm", "open_clip",
    "numpy", "pandas", "sklearn", "cv2", "PIL", "ftfy", "regex", "tqdm",
]

missing = []
for mod in REQUIRED:
    try:
        m = importlib.import_module(mod)
        print(f"{mod:14s} {getattr(m, '__version__', 'unknown')}")
    except Exception:
        print(f"{mod:14s} MISSING")
        missing.append(mod)

try:
    import torch
    print(f"\nCUDA available : {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU count      : {torch.cuda.device_count()}")
        print(f"GPU 0          : {torch.cuda.get_device_name(0)}")
except Exception as e:
    print(f"torch import failed: {e}")

if missing:
    print(f"\n缺少依赖: {missing}")
    print("安装：pip install -r requirements.txt（open_clip 对应包名为 open_clip_torch，"
          "cv2 对应 opencv-python，PIL 对应 pillow，sklearn 对应 scikit-learn）")
    sys.exit(1)
print("\n依赖齐全。")
