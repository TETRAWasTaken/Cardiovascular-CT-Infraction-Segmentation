"""
ImageCAS Dataset & MONAI Transform Pipeline for BaselineUNET
============================================================

Provides:
- Manifest loader for `imagecas_manifest.csv` supporting nested folders (1-200, 201-400...)
  and official splits from `imageCAS_data_split.xlsx`
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
    If official split tags exist ('train', 'val', 'test'), uses them directly.
    """
    manifest_path = Path(manifest_path).resolve()
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest file not found at: {manifest_path}")

    df = pd.read_csv(manifest_path)
    total_raw_rows = len(df)
    
    # Robust filtering for valid image and mask paths
    if "mask_path" in df.columns and "image_path" in df.columns:
        # Check non-empty strings and not-NA
        valid_mask = (
            df["mask_path"].notna()
            & (df["mask_path"].astype(str).str.strip() != "")
            & (df["mask_path"].astype(str).str.lower() != "nan")
        )
        valid_image = (
            df["image_path"].notna()
            & (df["image_path"].astype(str).str.strip() != "")
            & (df["image_path"].astype(str).str.lower() != "nan")
        )
        df = df[valid_mask & valid_image]
    elif "has_mask" in df.columns:
        df = df[df["has_mask"].astype(str).str.lower().isin(["true", "1", "t", "yes"])]

    if len(df) == 0:
        raise ValueError(
            f"No valid image/mask pairs found in manifest ({total_raw_rows} total rows scanned in {manifest_path}).\n"
            f"Please regenerate the manifest with the updated ingestion script:\n"
            f"  python Phase1_nnUNet/data_ingestion.py --data_dir Data/extracted --output_csv BaselineUNET/imagecas_manifest.csv"
        )

    # Check if official split is provided
    if "split" in df.columns and df["split"].notna().sum() > 0 and df["split"].nunique() > 1:
        train_df = df[df["split"].astype(str).str.lower().isin(["train", "training"])]
        val_df = df[df["split"].astype(str).str.lower().isin(["val", "validation", "test"])]
        
        # If val is empty, split train
        if len(val_df) == 0:
            val_df = train_df.sample(frac=val_ratio, random_state=seed)
            train_df = train_df.drop(val_df.index)

        train_files = [
            {"image": str(Path(row["image_path"]).resolve()), "label": str(Path(row["mask_path"]).resolve())}
            for _, row in train_df.iterrows()
            if Path(row["image_path"]).exists() and Path(row["mask_path"]).exists()
        ]
        val_files = [
            {"image": str(Path(row["image_path"]).resolve()), "label": str(Path(row["mask_path"]).resolve())}
            for _, row in val_df.iterrows()
            if Path(row["image_path"]).exists() and Path(row["mask_path"]).exists()
        ]
        if train_files and val_files:
            return train_files, val_files

    # Build data dictionaries with automatic random split
    data_dicts = [
        {"image": str(Path(row["image_path"]).resolve()), "label": str(Path(row["mask_path"]).resolve())}
        for _, row in df.iterrows()
        if Path(row["image_path"]).exists() and Path(row["mask_path"]).exists()
    ]

    if not data_dicts:
        # Check first 3 missing paths for debugging
        sample_img = df.iloc[0]["image_path"] if len(df) > 0 else "N/A"
        sample_mask = df.iloc[0]["mask_path"] if len(df) > 0 else "N/A"
        raise FileNotFoundError(
            f"Found {len(df)} pairs in manifest, but none of the files exist on disk.\n"
            f"Sample missing image: {sample_img}\n"
            f"Sample missing mask : {sample_mask}\n"
            f"Please regenerate the manifest on this machine using Phase1_nnUNet/data_ingestion.py"
        )

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
    Fallback directory scanner across nested subdirectories (1-200, 201-400, etc.).
    """
    data_dir = Path(data_dir).resolve()
    image_files = sorted([
        p for p in data_dir.rglob(f"*{image_suffix}")
        if not p.name.startswith("._")
    ])

    if not image_files:
        # Fallback to general .nii.gz images
        image_files = sorted([
            p for p in data_dir.rglob("*.nii.gz")
            if "label" not in p.name.lower() and "mask" not in p.name.lower() and not p.name.startswith("._")
        ])

    data_dicts = []
    for img_path in image_files:
        name = img_path.name
        # Match case ID
        case_id = name.replace(image_suffix, "")
        label_name = f"{case_id}{label_suffix}"
        label_path = img_path.parent / label_name

        if not label_path.exists():
            potential = list(img_path.parent.glob(f"*{case_id}*label*.nii.gz"))
            if potential:
                label_path = potential[0]
            else:
                global_potential = list(data_dir.rglob(f"*{case_id}*label*.nii.gz"))
                if global_potential:
                    label_path = global_potential[0]
                else:
                    continue

        data_dicts.append({"image": str(img_path.resolve()), "label": str(label_path.resolve())})

    if not data_dicts:
        raise FileNotFoundError(f"No paired images and labels found in {data_dir}")

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
    num_workers: int = 4,
    cache_rate: float = 0.0,
    cache_dir: Optional[str | Path] = None,
    hu_min: float = -100.0,
    hu_max: float = 700.0,
) -> Tuple[DataLoader, DataLoader]:
    """
    Create PyTorch / MONAI DataLoaders for training and validation.
    Supports on-the-fly streaming Dataset (RAM < 2GB), PersistentDataset (disk cache),
    or in-memory CacheDataset.
    """
    from monai.data import PersistentDataset

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

    if cache_dir:
        cache_path = Path(cache_dir).resolve()
        cache_path.mkdir(parents=True, exist_ok=True)
        print(f"Using PersistentDataset (disk cache at {cache_path})")
        train_ds = PersistentDataset(
            data=train_files,
            transform=train_transforms,
            cache_dir=str(cache_path / "train"),
        )
        val_ds = PersistentDataset(
            data=val_files,
            transform=val_transforms,
            cache_dir=str(cache_path / "val"),
        )
    elif cache_rate > 0.0:
        print(f"Warning: CacheDataset with cache_rate={cache_rate} preloads uncompressed arrays into RAM.")
        train_ds = CacheDataset(
            data=train_files,
            transform=train_transforms,
            cache_rate=cache_rate,
            num_workers=min(num_workers, 2),
        )
        val_ds = CacheDataset(
            data=val_files,
            transform=val_transforms,
            cache_rate=cache_rate,
            num_workers=min(num_workers, 2),
        )
    else:
        # Standard memory-safe on-demand streaming
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
        batch_size=1,
        shuffle=False,
        num_workers=min(num_workers, 2),
        pin_memory=torch.cuda.is_available(),
    )

    return train_loader, val_loader
