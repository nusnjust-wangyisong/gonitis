import os
import json
from collections import Counter
from typing import List, Dict, Tuple, Optional

import cv2
import numpy as np
from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torch.utils.data._utils.collate import default_collate
from torchvision import transforms
from torchvision.transforms import InterpolationMode
from scipy import ndimage  # 新增：用于 random_rotate
import math  # 随机擦除用到（若你不使用随机擦除，可忽略）

# 允许的图片后缀
VALID_EXTS = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff'}

# ---------- 实用工具 ----------
def _rgb_to_gray(arr: np.ndarray) -> np.ndarray:
    """
    将任意 HxWxC 或 HxW 数组变成单通道灰度 (float32)。
    - 支持 uint8/uint16/float 等类型
    - 支持 RGB / RGBA / 灰度
    """
    # 单通道（灰度）
    if arr.ndim == 2:
        return arr.astype(np.float32, copy=False)

    # 多通道
    if arr.ndim == 3:
        # RGBA → RGB（丢弃 Alpha）
        if arr.shape[2] == 4:
            arr = arr[:, :, :3]

        # RGB → Gray
        if arr.shape[2] == 3:
            r, g, b = arr[..., 0], arr[..., 1], arr[..., 2]
            return (0.2989 * r + 0.5870 * g + 0.1140 * b).astype(np.float32, copy=False)

    raise ValueError(f"Unsupported array shape for gray conversion: {arr.shape}")



class GrayMinMax(object):
    """
    灰度 -> (可选)百分位裁剪 -> 线性映射到 uint8 [0,255] -> (可选)转回3通道PIL
    不做 CLAHE 的基础版本（可作为对照或在某些场景禁用 CLAHE 时使用）
    """
    def __init__(self,
                 to_rgb: bool = True,
                 percentile: Optional[Tuple[float, float]] = (1.0, 99.0)):
        self.to_rgb = bool(to_rgb)
        self.percentile = percentile

    def __call__(self, img: Image.Image) -> Image.Image:
        arr = np.array(img)
        if arr.ndim == 3 and arr.shape[2] == 4:
            arr = arr[:, :, :3]

        g = _rgb_to_gray(arr)  # float32
        if self.percentile is not None:
            p_low, p_high = self.percentile
            lo = np.percentile(g, p_low)
            hi = np.percentile(g, p_high)
            if hi > lo:
                g = np.clip(g, lo, hi, out=g)
        g = cv2.normalize(g, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        if self.to_rgb:
            arr3 = np.stack([g, g, g], axis=-1)
            return Image.fromarray(arr3)
        else:
            return Image.fromarray(g, mode="L")


class GrayMinMaxCLAHE(object):
    """
    灰度 -> (可选)百分位裁剪 -> 线性映射到 uint8 [0,255] -> CLAHE -> (可选)转回3通道PIL
    说明：
      - CLAHE 对 uint8/uint16 更友好；此处统一到 uint8，保证工程稳定性
      - 输出仍为 PIL Image（单通道或 3 通道），方便后续 PIL 几何增强
    """
    def __init__(self,
                 clip_limit: float = 2.0,
                 tile_grid_size: Tuple[int, int] = (8, 8),
                 to_rgb: bool = True,
                 percentile: Optional[Tuple[float, float]] = (1.0, 99.0)):
        self.clip_limit = float(clip_limit)
        self.tile_grid_size = tuple(tile_grid_size)
        self.to_rgb = bool(to_rgb)
        self.percentile = percentile

    def __call__(self, img: Image.Image) -> Image.Image:
        # 1) 尽量不提前量化，先以原动态范围读取为 numpy
        arr = np.asarray(img)
        # 2) 转灰度
        g = _rgb_to_gray(arr)  # float32
        # 3) (可选)百分位裁剪，抑制极端值影响
        if self.percentile is not None:
            p_low, p_high = self.percentile
            lo = np.percentile(g, p_low)
            hi = np.percentile(g, p_high)
            if hi > lo:
                g = np.clip(g, lo, hi, out=g)
        # 4) 线性映射到 [0,255]，再转 uint8（为 CLAHE 做准备）
        g = cv2.normalize(g, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        # 5) CLAHE
        clahe = cv2.createCLAHE(clipLimit=self.clip_limit, tileGridSize=self.tile_grid_size)
        g = clahe.apply(g)  # uint8 灰度
        # 6) 回到 PIL，单通道或三通道
        if self.to_rgb:
            arr3 = np.stack([g, g, g], axis=-1)  # HWC
            return Image.fromarray(arr3)
        else:
            return Image.fromarray(g, mode="L")


class ClassificationDataset(Dataset):
    """
    通用分类数据集结构：
      base_dir/
        class_mapping.json   # 自动生成/复用
        train/
          classA/ classB/ ...
        val/
          classA/ classB/ ...
        test/
          classA/ classB/ ...
    """
    def __init__(self,
                 base_dir: str,
                 split: str = "train",
                 transform=None,
                 class_mapping_file: Optional[str] = None,
                 error_policy: str = "raise",   # "raise" | "skip" | "placeholder"
                 placeholder_color: Tuple[int, int, int] = (127, 127, 127),
                 placeholder_size: Tuple[int, int] = (224, 224),
                 include_path: bool = True):
        self.root_dir = base_dir
        self.split = split
        self.base_dir = os.path.join(base_dir, split)
        self.transform = transform
        self.class_mapping_file = class_mapping_file or os.path.join(base_dir, "class_mapping.json")
        self.error_policy = error_policy
        self.placeholder_color = placeholder_color
        self.placeholder_size = placeholder_size
        self.include_path = include_path

        self.classes, self.class_to_idx = self._get_class_mapping()
        self.samples: List[Tuple[str, int]] = []

        # 遍历类别文件夹（排序保证可复现）
        for cls_name in sorted(self.classes):
            cls_dir = os.path.join(self.base_dir, cls_name)
            if not os.path.isdir(cls_dir):
                print(f"[警告] 类别文件夹缺失：{cls_dir}（在 {split} split），跳过该类。")
                continue
            for fname in sorted(os.listdir(cls_dir)):
                ext = os.path.splitext(fname)[1].lower()
                if ext in VALID_EXTS:
                    self.samples.append((os.path.join(cls_dir, fname), self.class_to_idx[cls_name]))

        # 统计
        print(f"{split} 集合：{len(self.samples)} 张图像，{len(self.classes)} 个类别。")
        counts = Counter([y for _, y in self.samples])
        print(f"{split} 真实标签分布: {dict(sorted(counts.items()))}")
        print(f"{split} 类别映射: {self.class_to_idx}")

        self.labels = [y for _, y in self.samples]  # 供外部统计/采样

    # ---------- 类别映射 ----------
    def _get_class_mapping(self) -> Tuple[List[str], Dict[str, int]]:
        # 若存在既有映射，优先复用（按 idx 排序恢复 classes）
        if os.path.exists(self.class_mapping_file):
            with open(self.class_mapping_file, 'r', encoding='utf-8') as f:
                class_to_idx = json.load(f)
            # 确保按 idx 排序恢复类别顺序
            classes = [c for c, _ in sorted(class_to_idx.items(), key=lambda x: x[1])]
            print(f"加载已存在的类别映射: {self.class_mapping_file}")

            # 健康检查：当前 split 的目录 vs 映射
            if os.path.isdir(self.base_dir):
                present_classes = sorted([
                    d for d in os.listdir(self.base_dir)
                    if os.path.isdir(os.path.join(self.base_dir, d)) and not d.startswith('.')
                ])
                missing = [c for c in classes if c not in present_classes]
                extra = [c for c in present_classes if c not in class_to_idx]
                if missing:
                    print(f"[警告] 映射里存在但当前 split 缺失的类: {missing}")
                if extra:
                    print(f"[警告] 当前 split 新增但映射中不存在的类: {extra}（将被忽略；如需包含，请重建映射）")
            return classes, class_to_idx

        # 否则从 train 目录生成
        base_root = os.path.dirname(self.base_dir)  # == base_dir
        train_root = os.path.join(base_root, "train")
        scan_root = train_root if os.path.isdir(train_root) else self.base_dir
        classes = sorted([
            d for d in os.listdir(scan_root)
            if os.path.isdir(os.path.join(scan_root, d)) and not d.startswith('.')
        ])
        class_to_idx = {c: i for i, c in enumerate(classes)}

        os.makedirs(os.path.dirname(self.class_mapping_file), exist_ok=True)
        with open(self.class_mapping_file, 'w', encoding='utf-8') as f:
            json.dump(class_to_idx, f, ensure_ascii=False, indent=2)
        print(f"创建并保存类别映射: {self.class_mapping_file}")
        return classes, class_to_idx

    # ---------- 类别权重（可选，供 CrossEntropyLoss 使用，避免与加权采样同时启用） ----------
    def get_class_weights(self) -> torch.Tensor:
        counts = Counter([y for _, y in self.samples])
        total = len(self.samples)
        C = len(self.classes)
        weights = []
        for i in range(C):
            n = counts.get(i, 0)
            w = total / (C * n) if n > 0 else 1.0
            weights.append(w)
        return torch.tensor(weights, dtype=torch.float32)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        try:
            # 用 with 打开并复制到内存，避免文件句柄泄漏
            with Image.open(img_path) as im:
                img = im.copy()  # 保留原动态范围与模式，后续自定义变换处理
        except Exception as e:
            if self.error_policy == "raise":
                raise RuntimeError(f"[错误] 加载失败: {img_path} :: {e}") from e
            elif self.error_policy == "skip":
                # 返回 None，交给 safe_collate 过滤（会导致该 batch 尺寸变小）
                print(f"[警告] 加载失败且选择 skip：{img_path} :: {e}")
                return None
            elif self.error_policy == "placeholder":
                print(f"[警告] 加载失败，使用占位图：{img_path} :: {e}")
                img = Image.new('RGB', self.placeholder_size, color=self.placeholder_color)
            else:
                raise ValueError(f"未知 error_policy: {self.error_policy}")

        if self.transform:
            img = self.transform(img)
        else:
            # 兜底：至少保证返回 tensor（[0,1]）
            img = transforms.ToTensor()(img)

        item = {
            "image": img,
            "label": torch.tensor(label, dtype=torch.long),
        }
        if self.include_path:
            item["path"] = img_path
        return item


# 进行随机旋转和随机翻转（保持原函数名与签名）
def random_rot_flip(image, label=None):
    k = np.random.randint(0, 4)
    image = np.rot90(image, k)
    axis = np.random.randint(0, 2)
    image = np.flip(image, axis=axis).copy()
    if label is not None:
        label = np.rot90(label, k)
        label = np.flip(label, axis=axis).copy()
        return image, label
    else:
        return image


# 随机旋转增强（保持原函数名与签名）
def random_rotate(image, label):
    angle = np.random.randint(-20, 20)
    image = ndimage.rotate(image, angle, order=0, reshape=False)
    label = ndimage.rotate(label, angle, order=0, reshape=False)
    return image, label


# ---------- 将上述 numpy 版增强包装为可用于 torchvision 的 PIL 变换 ----------
class _RandomRotFlipPIL(object):
    """把 random_rot_flip 包装成 PIL 变换；只作用于图像（分类任务无需同步标签）"""
    def __init__(self, p: float = 1.0):
        self.p = float(p)

    def __call__(self, img: Image.Image) -> Image.Image:
        if np.random.rand() > self.p:
            return img
        arr = np.asarray(img)
        out = random_rot_flip(arr, label=None)  # 仅图像
        if out.dtype != np.uint8:
            out = np.clip(out, 0, 255).astype(np.uint8)
        return Image.fromarray(out)


class _RandomRotatePIL(object):
    """把 random_rotate 包装成 PIL 变换；内部用 dummy label 调用并丢弃"""
    def __init__(self, p: float = 0.5):
        self.p = float(p)

    def __call__(self, img: Image.Image) -> Image.Image:
        if np.random.rand() > self.p:
            return img
        arr = np.asarray(img)
        dummy = np.zeros(arr.shape[:2], dtype=np.uint8)
        out, _ = random_rotate(arr, dummy)
        if out.dtype != np.uint8:
            out = np.clip(out, 0, 255).astype(np.uint8)
        return Image.fromarray(out)


# ---------- 安全 collate：允许 batch 中出现 None（来自 error_policy='skip'） ----------
def safe_collate(batch):
    batch = [b for b in batch if b is not None]
    return default_collate(batch)


# ---------- Transforms 组合 ----------
def get_transforms(img_size=224,
                   is_training=True,
                   use_clahe=True,
                   to_rgb=True,
                   # CLAHE/Min-Max 相关
                   clahe_clip_limit=2.0,
                   clahe_tile_grid_size=(8, 8),
                   percentile=(1.0, 99.0),
                   # 几何增强（旧接口参数，为兼容保留）
                   rotate_deg=10,
                   hflip_p=0.0,
                #    fill_gray=128,
                   # 新增：控制随机擦除（分类常用）
                   random_erasing_p: float = 0.5,
                   random_erasing_scale: Tuple[float, float] = (0.02, 0.2),
                   random_erasing_ratio: Tuple[float, float] = (0.3, 3.3),
                   random_erasing_value: float = 0.0,
                   normalize: bool = False):
    """
    处理顺序：
      (Gray + Percentile + MinMax + [CLAHE]) -> Resize -> [随机旋转+随机翻转/随机旋转] -> ToTensor -> [RandomErasing]
    说明：
      - 按你的要求：去掉 RandomHorizontalFlip，改用自定义的 random_rot_flip / random_rotate
      - 随机擦除在 ToTensor 之后进行（torchvision.transforms.RandomErasing）
    """
    ops = []

    if use_clahe:
        ops.append(GrayMinMaxCLAHE(
            clip_limit=clahe_clip_limit,
            tile_grid_size=clahe_tile_grid_size,
            to_rgb=to_rgb,
            percentile=percentile
        ))
    else:
        ops.append(GrayMinMax(
            to_rgb=to_rgb,
            percentile=percentile
        ))

    ops.append(transforms.Resize((img_size, img_size), interpolation=InterpolationMode.BILINEAR))

    if is_training:
        # 用 hflip_p 作为“随机旋转+随机翻转”的触发概率，保持调用兼容
        if hflip_p and hflip_p > 0.0:
            ops.append(_RandomRotFlipPIL(p=hflip_p))
        # 再额外给一次小角度随机旋转机会（概率固定为 0.5；如需参数化可自行外部控制）
        if rotate_deg and rotate_deg > 0:
            ops.append(_RandomRotatePIL(p=0.5))

    # ToTensor：输出 [0,1]
    ops.append(transforms.ToTensor())

    # 可选归一化：True/'clip' 用 CLIP mean/std（BiomedCLIP 等）；'imagenet' 用
    # ImageNet mean/std（timm ConvNeXt-V2 等）。False 则不归一化（保持原 [0,1]）。
    if normalize:
        if normalize == "imagenet":
            mean, std = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
        else:  # True 或 'clip'
            mean, std = (0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)
        ops.append(transforms.Normalize(mean=mean, std=std))

    # 随机擦除（仅训练时启用）
    if is_training and random_erasing_p and random_erasing_p > 0.0:
        ops.append(transforms.RandomErasing(
            p=random_erasing_p,
            scale=random_erasing_scale,
            ratio=random_erasing_ratio,
            value=random_erasing_value,
            inplace=True
        ))

    return transforms.Compose(ops)


# ---------- DataLoader 工具 ----------
def create_dataloader(dataset: ClassificationDataset,
                      worker_init_fn=None,
                      batch_size=32,
                      shuffle=True,
                      num_workers=4,
                      pin_memory: Optional[bool] = None,
                      persistent_workers=True) -> DataLoader:
    if pin_memory is None:
        pin_memory = torch.cuda.is_available()
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=(num_workers > 0 and persistent_workers),
        worker_init_fn=worker_init_fn,
        collate_fn=safe_collate  # 避免 error_policy='skip' 时崩溃
    )


def create_balanced_dataloader(dataset: ClassificationDataset,
                               batch_size=32,
                               num_workers=4,
                               pin_memory: Optional[bool] = None,
                               persistent_workers=True) -> DataLoader:
    """
    基于类频率的加权采样（适合训练集）。注意不要与加权损失同时使用，避免重复矫正。
    """
    if pin_memory is None:
        pin_memory = torch.cuda.is_available()

    counts = Counter(dataset.labels)
    sample_weights = [1.0 / counts[y] for _, y in dataset.samples]
    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=(num_workers > 0 and persistent_workers),
        collate_fn=safe_collate
    )


# ---------- 可选：简单的种子设置 ----------
def seed_everything(seed: int = 42):
    import random
    import numpy as np
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ---------- 使用示例（参考） ----------
if __name__ == "__main__":
    # seed_everything(42)

    base_dir = "/path/to/dataset_root"

    train_tf = get_transforms(
        img_size=224,
        is_training=True,
        use_clahe=True,
        to_rgb=True,
        clahe_clip_limit=2.0,
        clahe_tile_grid_size=(8, 8),
        percentile=(1.0, 99.0),
        rotate_deg=10,
        hflip_p=0.5,     # 作为 RandomRotFlip 的执行概率
        # fill_gray=128,   # 为兼容保留，不影响自定义旋转
        random_erasing_p=0.5,                 # 新增：随机擦除
        random_erasing_scale=(0.02, 0.2),
        random_erasing_ratio=(0.3, 3.3),
        random_erasing_value=0.0
    )
    val_tf = get_transforms(
        img_size=224,
        is_training=False,
        use_clahe=True,
        to_rgb=True,
        clahe_clip_limit=2.0,
        clahe_tile_grid_size=(8, 8),
        percentile=(1.0, 99.0)
    )

    train_ds = ClassificationDataset(
        base_dir=base_dir,
        split="train",
        transform=train_tf,
        error_policy="raise",   # 也可用 "skip" 或 "placeholder"
        include_path=True
    )
    val_ds = ClassificationDataset(
        base_dir=base_dir,
        split="val",
        transform=val_tf,
        error_policy="raise",
        include_path=True
    )

    # 二选一：普通/加权 DataLoader（不要与加权损失同时用）
    train_loader = create_balanced_dataloader(train_ds, batch_size=32, num_workers=4)
    # train_loader = create_dataloader(train_ds, batch_size=32, shuffle=True, num_workers=4)

    val_loader = create_dataloader(val_ds, batch_size=32, shuffle=False, num_workers=4)

    for batch in train_loader:
        x = batch["image"]         # [B,C,H,W], float32 in [0,1]
        y = batch["label"].long()  # [B]
        # 训练循环略
        break
