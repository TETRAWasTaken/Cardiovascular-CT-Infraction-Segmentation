"""
Phase 2: Medical Data Pipeline & Preprocessing
=============================================
Robust MONAI dictionary transform pipeline handling 3D DICOM/NIfTI cardiac CT scans.
Optimized for x86_64 systems with NVIDIA CUDA acceleration via pinned memory.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Sequence

import nibabel as nib
import numpy as np
import torch
from monai.data import CacheDataset, DataLoader, PersistentDataset
from monai.transforms import (
    Compose,
    EnsureChannelFirstd,
    LoadImaged,
    Orientationd,
    RandCropByPosNegLabeld,
    ScaleIntensityRanged,
    Spacingd,
)


def get_transforms(
    mode: str = "train",
    spatial_size: tuple[int, int, int] = (96, 96, 96),
    num_samples: int = 4,
    pixdim: tuple[float, float, float] = (1.0, 1.0, 1.0),
    a_min: float = -100.0,
    a_max: float = 500.0,
    b_min: float = 0.0,
    b_max: float = 1.0,
) -> Compose:
    """
    Constructs sequential MONAI dictionary transforms matching requirements:
      1. LoadImaged(keys=["image", "label"])
      2. EnsureChannelFirstd(keys=["image", "label"])
      3. Orientationd(keys=["image", "label"], axcodes="RAS")
      4. Spacingd(keys=["image", "label"], pixdim=(1.0, 1.0, 1.0), mode=("bilinear", "nearest"))
      5. ScaleIntensityRanged(keys=["image"], a_min=-100, a_max=500, b_min=0.0, b_max=1.0, clip=True)
      6. RandCropByPosNegLabeld(spatial_size=(96, 96, 96), pos=1, neg=1, num_samples=4) [train mode only]
    """
    keys = ["image", "label"]

    base_transforms = [
        LoadImaged(keys=keys, image_only=False),
        EnsureChannelFirstd(keys=keys),
        Orientationd(keys=keys, axcodes="RAS"),
        Spacingd(
            keys=keys,
            pixdim=pixdim,
            mode=("bilinear", "nearest"),
        ),
        ScaleIntensityRanged(
            keys=["image"],
            a_min=a_min,
            a_max=a_max,
            b_min=b_min,
            b_max=b_max,
            clip=True,
        ),
    ]

    if mode == "train":
        train_transforms = base_transforms + [
            RandCropByPosNegLabeld(
                keys=keys,
                label_key="label",
                spatial_size=spatial_size,
                pos=1,
                neg=1,
                num_samples=num_samples,
                image_key="image",
                image_threshold=0.0,
            )
        ]
        return Compose(train_transforms)

    return Compose(base_transforms)


def build_dataset(
    data_dicts: list[dict[str, Any]],
    transforms: Compose,
    use_cache: bool = True,
    cache_rate: float = 1.0,
    cache_dir: str | Path | None = None,
    num_workers: int = 4,
) -> CacheDataset | PersistentDataset:
    """
    Initializes CacheDataset (in-memory) or PersistentDataset (on-disk cache)
    for fast I/O throughput during multi-epoch GPU training on x86_64 nodes.
    """
    if cache_dir is not None:
        os.makedirs(cache_dir, exist_ok=True)
        return PersistentDataset(
            data=data_dicts,
            transform=transforms,
            cache_dir=cache_dir,
        )
    elif use_cache:
        return CacheDataset(
            data=data_dicts,
            transform=transforms,
            cache_rate=cache_rate,
            num_workers=num_workers,
        )
    else:
        from monai.data import Dataset
        return Dataset(data=data_dicts, transform=transforms)


def build_dataloader(
    dataset: Any,
    batch_size: int = 2,
    shuffle: bool = True,
    num_workers: int = 4,
    pin_memory: bool = True,
) -> DataLoader:
    """
    Creates high-performance PyTorch DataLoader configured with pinned host memory
    for low-overhead asynchronous page-locked host-to-device transfers on NVIDIA CUDA.
    """
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory and torch.cuda.is_available(),
        persistent_workers=(num_workers > 0),
    )


def generate_synthetic_nifti(
    output_dir: str | Path,
    prefix: str = "sample_case_01",
    shape: tuple[int, int, int] = (128, 128, 128),
    spacing: tuple[float, float, float] = (1.0, 1.0, 1.0),
) -> dict[str, str]:
    """
    Generates a synthetic 3D cardiac CT volume and corresponding coronary artery label
    with high-contrast HU vessel structures and realistic spatial geometry in NIfTI format.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    d, h, w = shape
    # CT background HU: soft tissue / blood pool around 30 to 50 HU, lungs around -600 HU
    image_data = np.random.normal(loc=40.0, scale=15.0, size=shape).astype(np.float32)
    label_data = np.zeros(shape, dtype=np.uint8)

    # Simulate curved tubular artery centerlines
    z = np.linspace(15, d - 16, num=180)
    center_y = h / 2.0 + 20.0 * np.sin(z / 18.0)
    center_x = w / 2.0 + 20.0 * np.cos(z / 18.0)

    radius = 3
    for zi, yi, xi in zip(z.astype(int), center_y.astype(int), center_x.astype(int)):
        z_min, z_max = max(0, zi - radius), min(d, zi + radius + 1)
        y_min, y_max = max(0, yi - radius), min(h, yi + radius + 1)
        x_min, x_max = max(0, xi - radius), min(w, xi + radius + 1)

        for zz in range(z_min, z_max):
            for yy in range(y_min, y_max):
                for xx in range(x_min, x_max):
                    if (zz - zi) ** 2 + (yy - yi) ** 2 + (xx - xi) ** 2 <= radius ** 2:
                        label_data[zz, yy, xx] = 1
                        # High attenuation contrast agent HU (e.g. 350 to 450 HU)
                        image_data[zz, yy, xx] = np.random.uniform(350.0, 450.0)

    affine = np.diag([spacing[0], spacing[1], spacing[2], 1.0])
    affine[0, 3] = -float(w * spacing[0]) / 2.0
    affine[1, 3] = -float(h * spacing[1]) / 2.0
    affine[2, 3] = 0.0

    img_nii = nib.Nifti1Image(image_data, affine)
    lbl_nii = nib.Nifti1Image(label_data, affine)

    img_path = output_dir / f"{prefix}_img.nii.gz"
    lbl_path = output_dir / f"{prefix}_label.nii.gz"

    nib.save(img_nii, str(img_path))
    nib.save(lbl_nii, str(lbl_path))

    return {"image": str(img_path), "label": str(lbl_path)}
