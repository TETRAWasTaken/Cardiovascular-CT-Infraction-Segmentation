"""
ImageCAS Dataset & MONAI Transform Pipeline for BaselineUNET
============================================================

Provides:
- Manifest loader for `imagecas_manifest.csv`
- Dynamic train/validation partitioning
- Foreground-biased 3D patch cropping (RandCropByPosNegLabeld)
- Sliding-window compatible validation transforms
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import List, Dict, Tuple, Optional

import pandas as pd
import torch
from monai.data import Dataset, CacheDataset, DataLoader, list_data_collate
from monai.transforms import (
    Compose,
    LoadImaged,
    EnsureChannelFirstd,
    Orientationd,
    ScaleIntensityRanged,
    RandCropByPosNegLabeld,
    RandFlipd,
    RandRotate90d,
    RandGaussianNoised,
    EnsureTyped,
)


def load_manifest_data_dicts(
    manifest_path: str | Path,
    val_ratio: float = 0.2,
    seed: int = 42,
) -> Tuple[List[Dict[str, str]], List[Dict[str, str]]]:
    """
    Load image-mask pairs from imagecas_manifest.csv and split into train/val sets.
    """
    manifest_path = Path(manifest_path).resolve()
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest file not found at: {manifest_path}")

    df = pd.read_csv(manifest_path)
    
    # Filter valid pairs
    if "has_mask" in df.columns:
        df = df[df["has_mask"] == True]
    if "status" in df.columns:
        df = df[df["status"] == "ok"]

    if len(df) == 0:
        raise ValueError(f"No valid image/mask pairs found in manifest: {manifest_path}")

    # Build data dictionaries
    data_dicts = [
        {"image": row["image_path"], "label": row["mask_path"]}
        for _, row in df.iterrows()
        if Path(row["image_path"]).exists() and Path(row["mask_path"]).exists()
    ]

    if not data_dicts:
        raise FileNotFoundError("None of the image/mask paths in the manifest exist on disk.")

    # Train / Val Split
    generator = torch.Generator().manual_seed(seed)
    indices = torch.randperm(len(data_dicts), generator=generator).tolist()
    
    val_count = max(1, int(len(data_dicts) * val_ratio)) if len(data_dicts) > 1 else 0
    train_indices = indices[val_count:]
    val_indices = indices[:val_count] if val_count > 0 else indices

    train_files = [data_dicts[i] for i in train_indices]
    val_files = [data_dicts[i] for i in val_indices]

    return train_files, val_files


def scan_directory_data_dicts(
    data_dir: str | Path,
    image_suffix: str = ".img.nii.gz",
    label_suffix: str = ".label.nii.gz",
    val_ratio: float = 0.2,
    seed: int = 42,
) -> Tuple[List[Dict[str, str]], List[Dict[str, str]]]:
    """
    Fallback directory scanner if manifest is not pre-computed.
    """
    data_dir = Path(data_dir).resolve()
    image_files = sorted(data_dir.rglob(f"*{image_suffix}"))
    
    data_dicts = []
    for img_path in image_files:
        name = img_path.name
        case_id = name[: -len(image_suffix)]
        label_name = f"{case_id}{label_suffix}"
        label_path = img_path.parent / label_name
        if not label_path.exists():
            # Try searching in raw_dir
            potential_labels = list(data_dir.rglob(label_name))
            if potential_labels:
                label_path = potential_labels[0]
            else:
                continue

        data_dicts.append({"image": str(img_path), "label": str(label_path)})

    if not data_dicts:
        raise FileNotFoundError(f"No paired {image_suffix} and {label_suffix} files found in {data_dir}")

    generator = torch.Generator().manual_seed(seed)
    indices = torch.randperm(len(data_dicts), generator=generator).tolist()
    
    val_count = max(1, int(len(data_dicts) * val_ratio)) if len(data_dicts) > 1 else 0
    train_indices = indices[val_count:]
    val_indices = indices[:val_count] if val_count > 0 else indices

    return [data_dicts[i] for i in train_indices], [data_dicts[i] for i in val_indices]


def get_train_transforms(
    patch_size: Tuple[int, int, int] = (96, 96, 96),
    num_samples: int = 4,
    hu_min: float = -100.0,
    hu_max: float = 700.0,
) -> Compose:
    """
    Training transforms with intensity clipping and foreground-biased patch cropping.
    """
    return Compose(
        [
            LoadImaged(keys=["image", "label"]),
            EnsureChannelFirstd(keys=["image", "label"]),
            Orientationd(keys=["image", "label"], axcodes="RAS"),
            ScaleIntensityRanged(
                keys=["image"],
                a_min=hu_min,
                a_max=hu_max,
                b_min=0.0,
                b_max=1.0,
                clip=True,
            ),
            RandCropByPosNegLabeld(
                keys=["image", "label"],
                label_key="label",
                spatial_size=patch_size,
                pos=2,
                neg=1,
                num_samples=num_samples,
                image_key="image",
                image_threshold=0,
            ),
            RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=0),
            RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=1),
            RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=2),
            RandRotate90d(keys=["image", "label"], prob=0.5, max_k=3),
            RandGaussianNoised(keys=["image"], prob=0.15, mean=0.0, std=0.1),
            EnsureTyped(keys=["image", "label"], track_meta=False),
        ]
    )


def get_val_transforms(
    hu_min: float = -100.0,
    hu_max: float = 700.0,
) -> Compose:
    """
    Validation transforms (whole-volume loading for sliding-window inference).
    """
    return Compose(
        [
            LoadImaged(keys=["image", "label"]),
            EnsureChannelFirstd(keys=["image", "label"]),
            Orientationd(keys=["image", "label"], axcodes="RAS"),
            ScaleIntensityRanged(
                keys=["image"],
                a_min=hu_min,
                a_max=hu_max,
                b_min=0.0,
                b_max=1.0,
                clip=True,
            ),
            EnsureTyped(keys=["image", "label"], track_meta=False),
        ]
    )


def build_dataloaders(
    train_files: List[Dict[str, str]],
    val_files: List[Dict[str, str]],
    patch_size: Tuple[int, int, int] = (96, 96, 96),
    num_samples: int = 4,
    batch_size: int = 2,
    num_workers: int = 2,
    cache_rate: float = 0.0,
    hu_min: float = -100.0,
    hu_max: float = 700.0,
) -> Tuple[DataLoader, DataLoader]:
    """
    Create PyTorch / MONAI DataLoaders for training and validation.
    """
    train_transforms = get_train_transforms(
        patch_size=patch_size,
        num_samples=num_samples,
        hu_min=hu_min,
        hu_max=hu_max,
    )
    val_transforms = get_val_transforms(
        hu_min=hu_min,
        hu_max=hu_max,
    )

    if cache_rate > 0.0:
        train_ds = CacheDataset(
            data=train_files,
            transform=train_transforms,
            cache_rate=cache_rate,
            num_workers=num_workers,
        )
        val_ds = CacheDataset(
            data=val_files,
            transform=val_transforms,
            cache_rate=cache_rate,
            num_workers=num_workers,
        )
    else:
        train_ds = Dataset(data=train_files, transform=train_transforms)
        val_ds = Dataset(data=val_files, transform=val_transforms)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=list_data_collate,
        pin_memory=torch.cuda.is_available(),
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=1,  # Full volumes evaluated 1 at a time via sliding window
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    return train_loader, val_loader
