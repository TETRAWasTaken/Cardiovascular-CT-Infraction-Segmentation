"""
Unit tests for MONAI dictionary transforms and slice spatial alignment.
Verification Artifact: artifacts/transforms_check.png
"""

from __future__ import annotations

import os
import shutil
import tempfile
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pytest
import torch

from Transformer.src.dataset import (
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


if __name__ == "__main__":
    pytest.main(["-v", str(Path(__file__))])
