"""
Unit tests for MONAI dictionary transforms and slice spatial alignment.
Verification Artifact: artifacts/transforms_check.png
"""

from __future__ import annotations

import os
import shutil
import sys
import tempfile
from pathlib import Path

# Ensure repo root is in sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import matplotlib.pyplot as plt
import numpy as np
import pytest
import torch

try:
    from Transformer.src.dataset import (
        build_dataloader,
        build_dataset,
        generate_synthetic_nifti,
        get_transforms,
    )
except ModuleNotFoundError:
    from src.dataset import (
        build_dataloader,
        build_dataset,
        generate_synthetic_nifti,
        get_transforms,
    )


@pytest.fixture(scope="module")
def temp_nifti_data():
    temp_dir = tempfile.mkdtemp(prefix="test_transforms_")
    case = generate_synthetic_nifti(temp_dir, prefix="test_case_001", shape=(128, 128, 128))
    yield [case]
    shutil.rmtree(temp_dir, ignore_errors=True)


def test_transform_shapes_and_intensity(temp_nifti_data):
    """Confirm 96x96x96 patch shape, channel first, and intensity bounds [0.0, 1.0]."""
    transforms = get_transforms(mode="train", spatial_size=(96, 96, 96), num_samples=2)
    dataset = build_dataset(temp_nifti_data, transforms=transforms, use_cache=False)
    dataloader = build_dataloader(dataset, batch_size=1, shuffle=False, num_workers=0)

    batch = next(iter(dataloader))
    image = batch["image"]
    label = batch["label"]

    # When num_samples=2, shape is [Batch, NumSamples, Channel, D, H, W] or concatenated along batch
    # In MONAI RandCropByPosNegLabeld with num_samples=2, batch dimension expands
    if image.ndim == 5:
        # [B, C, D, H, W]
        assert image.shape[-3:] == (96, 96, 96), f"Expected spatial 96x96x96, got {image.shape}"
        assert label.shape[-3:] == (96, 96, 96), f"Expected spatial 96x96x96, got {label.shape}"
    elif image.ndim == 6:
        # [B, Samples, C, D, H, W]
        assert image.shape[-3:] == (96, 96, 96)
        assert label.shape[-3:] == (96, 96, 96)

    # Validate CT intensity clipping and scale [0, 1]
    assert float(image.min()) >= 0.0, f"Image min {float(image.min())} < 0.0"
    assert float(image.max()) <= 1.0, f"Image max {float(image.max())} > 1.0"
    # Ensure binary label (0 and 1)
    unique_labels = torch.unique(label)
    assert all(val in (0, 1) for val in unique_labels.tolist()), f"Unexpected label values: {unique_labels}"


def test_slice_spatial_alignment_and_artifact(temp_nifti_data):
    """Verify foreground alignment and generate verification artifact: artifacts/transforms_check.png."""
    transforms = get_transforms(mode="train", spatial_size=(96, 96, 96), num_samples=1)
    dataset = build_dataset(temp_nifti_data, transforms=transforms, use_cache=False)
    sample = dataset[0]

    if isinstance(sample, list):
        sample = sample[0]

    img_tensor = sample["image"][0].detach().cpu().numpy()  # (96, 96, 96)
    lbl_tensor = sample["label"][0].detach().cpu().numpy()  # (96, 96, 96)

    # Check foreground overlap: where label == 1, CT intensity should reflect contrast enhancement (> 0.5)
    fg_indices = np.where(lbl_tensor > 0.5)
    assert len(fg_indices[0]) > 0, "No positive vessel voxels found in cropped patch!"
    fg_intensities = img_tensor[fg_indices]
    mean_fg_intensity = float(np.mean(fg_intensities))
    assert mean_fg_intensity > 0.5, f"Expected high CT contrast for vessel, got {mean_fg_intensity}"

    # Generate verification artifact
    artifacts_dir = Path(__file__).resolve().parents[1] / "artifacts"
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    out_png = artifacts_dir / "transforms_check.png"

    mid_z, mid_y, mid_x = 48, 48, 48

    fig, axes = plt.subplots(2, 3, figsize=(14, 9))
    # Row 0: Normalized CT Image Slices (Axial, Coronal, Sagittal)
    axes[0, 0].imshow(img_tensor[mid_z, :, :], cmap="gray")
    axes[0, 0].set_title(f"CT Axial (Z={mid_z})")
    axes[0, 1].imshow(img_tensor[:, mid_y, :], cmap="gray")
    axes[0, 1].set_title(f"CT Coronal (Y={mid_y})")
    axes[0, 2].imshow(img_tensor[:, :, mid_x], cmap="gray")
    axes[0, 2].set_title(f"CT Sagittal (X={mid_x})")

    # Row 1: Overlay Label on CT Image
    for col, (slice_img, slice_lbl, view_name) in enumerate([
        (img_tensor[mid_z, :, :], lbl_tensor[mid_z, :, :], "Axial"),
        (img_tensor[:, mid_y, :], lbl_tensor[:, mid_y, :], "Coronal"),
        (img_tensor[:, :, mid_x], lbl_tensor[:, :, mid_x], "Sagittal"),
    ]):
        axes[1, col].imshow(slice_img, cmap="gray")
        masked_lbl = np.ma.masked_where(slice_lbl == 0, slice_lbl)
        axes[1, col].imshow(masked_lbl, cmap="autumn", alpha=0.7)
        axes[1, col].set_title(f"Overlay {view_name} (Red=Artery)")

    for ax in axes.flat:
        ax.axis("off")

    plt.tight_layout()
    plt.savefig(str(out_png), dpi=150)
    plt.close(fig)

    assert out_png.exists(), f"Failed to create artifact {out_png}"
    print(f"\n[ARTIFACT CREATED] {out_png}")


def test_transforms_with_smaller_volume_dimensions():
    """
    Regression Test for ROI crop error:
    Validates that volumes with spatial dimensions smaller than spatial_size
    (e.g., shape (164, 164, 84) with spatial_size (96, 96, 96)) are automatically
    padded via SpatialPadd without raising ValueError during random cropping.
    """
    with tempfile.TemporaryDirectory(prefix="test_small_volume_") as temp_dir:
        # Create a synthetic volume with depth=84 (< 96)
        case = generate_synthetic_nifti(temp_dir, prefix="small_case_001", shape=(164, 164, 84))

        # Test Train Pipeline (Crop 96x96x96 from 164x164x84)
        train_transforms = get_transforms(mode="train", spatial_size=(96, 96, 96), num_samples=1)
        train_ds = build_dataset([case], transforms=train_transforms, use_cache=False)
        train_loader = build_dataloader(train_ds, batch_size=1, shuffle=False, num_workers=0)

        batch = next(iter(train_loader))
        train_img = batch["image"]
        train_lbl = batch["label"]

        if train_img.ndim == 6:
            b, s, c, d, h, w = train_img.shape
            train_img = train_img.view(b * s, c, d, h, w)
            train_lbl = train_lbl.view(b * s, c, d, h, w)

        assert train_img.shape[-3:] == (96, 96, 96), f"Expected (96, 96, 96), got {train_img.shape[-3:]}"
        assert train_lbl.shape[-3:] == (96, 96, 96), f"Expected (96, 96, 96), got {train_lbl.shape[-3:]}"

        # Test Val Pipeline (Whole volume padded to at least 96 in each dimension)
        val_transforms = get_transforms(mode="val", spatial_size=(96, 96, 96))
        val_ds = build_dataset([case], transforms=val_transforms, use_cache=False)
        val_loader = build_dataloader(val_ds, batch_size=1, shuffle=False, num_workers=0)

        val_batch = next(iter(val_loader))
        val_img = val_batch["image"]
        val_lbl = val_batch["label"]

        # Depth 84 was padded to 96; 164 remained 164
        assert val_img.shape[-1] >= 96, f"Expected depth >= 96, got {val_img.shape[-1]}"
        assert val_lbl.shape[-1] >= 96, f"Expected label depth >= 96, got {val_lbl.shape[-1]}"
        assert val_img.shape == val_lbl.shape, "Val image and label shapes must match"
        print(">>> Small-volume padding regression test passed successfully.")


if __name__ == "__main__":
    pytest.main(["-v", str(Path(__file__))])
