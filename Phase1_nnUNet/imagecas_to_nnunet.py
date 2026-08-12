"""
Phase 1 — CNN Vessel Segmentation (nnU-Net baseline)
=====================================================

Purpose
-------
nnU-Net is not a Python model class you import and call — it is a full
CLI-driven training framework. The actual "work" needed to use it is
converting your raw dataset into nnU-Net's required folder structure
and naming convention, plus generating its dataset.json config.

This script does exactly that conversion step, and is fully testable
without the real ~87GB ImageCAS download: point it at any folder of
paired (image, label) NIfTI files — including a handful of synthetic
dummy volumes — and it will lay them out correctly.

This is a genuine, permanent, dataset-independent piece of Phase 1:
whenever you're on the lab GPU machines with real ImageCAS data, you
run this same script once, then hand off to nnU-Net's own CLI:

    nnUNetv2_plan_and_preprocess -d 001 --verify_dataset_integrity
    nnUNetv2_train 001 3d_fullres 0

Expected input layout (what ImageCAS actually ships as, after unzip):
    raw_imagecas/
        1.img.nii.gz
        1.label.nii.gz
        2.img.nii.gz
        2.label.nii.gz
        ...

Produces nnU-Net v2's required layout:
    nnUNet_raw/Dataset001_ImageCAS/
        imagesTr/1_0000.nii.gz, 2_0000.nii.gz, ...
        labelsTr/1.nii.gz, 2.nii.gz, ...
        dataset.json
"""

import json
import shutil
from pathlib import Path


DATASET_ID = 1
DATASET_NAME = f"Dataset{DATASET_ID:03d}_ImageCAS"


def convert_imagecas_to_nnunet(raw_dir: str, nnunet_raw_dir: str,
                                image_suffix: str = ".img.nii.gz",
                                label_suffix: str = ".label.nii.gz") -> dict:
    """
    Convert a flat folder of ImageCAS-style paired NIfTI files into
    nnU-Net v2's required raw dataset structure.

    Parameters
    ----------
    raw_dir : path to the folder containing "<id><image_suffix>" and
              "<id><label_suffix>" file pairs (real ImageCAS or dummy data).
    nnunet_raw_dir : path to nnU-Net's expected `nnUNet_raw/` root.

    Returns
    -------
    dict summary: how many cases were found and converted.
    """
    raw_dir = Path(raw_dir)
    out_root = Path(nnunet_raw_dir) / DATASET_NAME
    images_tr = out_root / "imagesTr"
    labels_tr = out_root / "labelsTr"
    images_tr.mkdir(parents=True, exist_ok=True)
    labels_tr.mkdir(parents=True, exist_ok=True)

    image_files = sorted(raw_dir.glob(f"*{image_suffix}"))
    case_ids = []

    for img_path in image_files:
        case_id = img_path.name[: -len(image_suffix)]
        label_path = raw_dir / f"{case_id}{label_suffix}"
        if not label_path.exists():
            print(f"  [skip] {case_id}: no matching label file found")
            continue

        # nnU-Net naming convention: <case>_<channel:04d>.nii.gz for images
        shutil.copy(img_path, images_tr / f"{case_id}_0000.nii.gz")
        shutil.copy(label_path, labels_tr / f"{case_id}.nii.gz")
        case_ids.append(case_id)

    dataset_json = {
        "channel_names": {"0": "CT"},
        "labels": {"background": 0, "coronary_artery": 1},
        "numTraining": len(case_ids),
        "file_ending": ".nii.gz",
        "name": "ImageCAS",
        "description": "Coronary CT angiography vessel segmentation (Phase 1 baseline)",
    }
    with open(out_root / "dataset.json", "w") as f:
        json.dump(dataset_json, f, indent=2)

    return {"cases_found": len(image_files), "cases_converted": len(case_ids),
            "output_dir": str(out_root)}


def _make_dummy_imagecas_folder(target_dir: str, n_cases: int = 3):
    """
    Builds a handful of small synthetic NIfTI volumes shaped/named like
    real ImageCAS files, purely so this conversion script (and, later,
    `nnUNetv2_plan_and_preprocess --verify_dataset_integrity`) can be
    smoke-tested without downloading any real data.
    """
    import numpy as np
    import nibabel as nib

    target_dir = Path(target_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    affine = np.eye(4)
    affine[0, 0] = affine[1, 1] = 0.4   # realistic-ish in-plane spacing (mm)
    affine[2, 2] = 0.5                  # slice spacing (mm)

    rng = np.random.default_rng(42)
    for i in range(1, n_cases + 1):
        shape = (64, 64, 64)
        # fake CT-ish HU volume: mostly soft tissue, some vessel-bright voxels
        volume = rng.normal(loc=40, scale=60, size=shape).astype(np.float32)
        label = np.zeros(shape, dtype=np.uint8)
        # carve a fake curved "vessel" tube into the label + brighten it in the image
        for z in range(shape[2]):
            cx = 32 + int(10 * np.sin(z / 8))
            cy = 32 + int(6 * np.cos(z / 10))
            label[cx - 1:cx + 2, cy - 1:cy + 2, z] = 1
            volume[cx - 1:cx + 2, cy - 1:cy + 2, z] += 300  # calcified-plaque-like HU spike

        nib.save(nib.Nifti1Image(volume, affine), target_dir / f"{i}.img.nii.gz")
        nib.save(nib.Nifti1Image(label, affine), target_dir / f"{i}.label.nii.gz")

    return target_dir


if __name__ == "__main__":
    print("=" * 70)
    print("Phase 1 — nnU-Net baseline: dataset conversion smoke test")
    print("=" * 70)

    dummy_raw = "/home/claude/Phase1_nnUNet/_dummy_raw_imagecas"
    nnunet_raw = "/home/claude/Phase1_nnUNet/_dummy_nnUNet_raw"

    print(f"\n[1/2] Generating {3} synthetic ImageCAS-style cases -> {dummy_raw}")
    _make_dummy_imagecas_folder(dummy_raw, n_cases=3)

    print(f"[2/2] Converting to nnU-Net v2 raw dataset layout -> {nnunet_raw}")
    summary = convert_imagecas_to_nnunet(dummy_raw, nnunet_raw)

    print("\nSummary:")
    for k, v in summary.items():
        print(f"  {k}: {v}")

    print("\nOn the lab GPU machines, with real ImageCAS data, the exact next commands are:")
    print("  export nnUNet_raw=<nnunet_raw_dir>")
    print(f"  nnUNetv2_plan_and_preprocess -d {DATASET_ID} --verify_dataset_integrity")
    print(f"  nnUNetv2_train {DATASET_ID} 3d_fullres 0")
