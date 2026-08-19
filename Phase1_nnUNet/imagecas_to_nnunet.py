"""
Phase 1 — CNN Vessel Segmentation (nnU-Net baseline)
=====================================================

Purpose
-------
Converts the nested ImageCAS dataset layout (e.g. extracted/1-200, 201-400, etc.)
into nnU-Net v2's required folder structure and naming convention, plus generates
the dataset.json configuration.

Expected input layout:
    data/extracted/
        1-200/
            1.img.nii.gz
            1.label.nii.gz
            ...
        201-400/
            201.img.nii.gz
            201.label.nii.gz
            ...

Produces nnU-Net v2's required layout:
    nnUNet_raw/Dataset001_ImageCAS/
        imagesTr/1_0000.nii.gz, 2_0000.nii.gz, ...
        labelsTr/1.nii.gz, 2.nii.gz, ...
        dataset.json

Usage:
    python Phase1_nnUNet/imagecas_to_nnunet.py --data_dir data/extracted --nnunet_raw nnUNet_raw
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
from pathlib import Path

DATASET_ID = 1
DATASET_NAME = f"Dataset{DATASET_ID:03d}_ImageCAS"


def normalize_case_key(path: Path) -> str:
    name = path.name.lower()
    for ext in [".img.nii.gz", ".label.nii.gz", ".nii.gz", ".nii", ".mha", ".nrrd"]:
        if name.endswith(ext):
            name = name[: -len(ext)]
            break
    key = re.sub(r"[^0-9a-zA-Z]", "", name)
    return key


def convert_imagecas_to_nnunet(
    raw_dir: str | Path,
    nnunet_raw_dir: str | Path,
    image_suffix: str = ".img.nii.gz",
    label_suffix: str = ".label.nii.gz",
) -> dict:
    """
    Recursively scan ImageCAS folders (1-200, 201-400, etc.) and convert paired
    volumes into nnU-Net v2 raw dataset structure.
    """
    raw_dir = Path(raw_dir).resolve()
    out_root = Path(nnunet_raw_dir).resolve() / DATASET_NAME
    images_tr = out_root / "imagesTr"
    labels_tr = out_root / "labelsTr"
    images_tr.mkdir(parents=True, exist_ok=True)
    labels_tr.mkdir(parents=True, exist_ok=True)

    # Recursive scan to find all image files across subfolders
    image_files = sorted([
        p for p in raw_dir.rglob(f"*{image_suffix}")
        if not p.name.startswith("._")
    ])

    if not image_files:
        # Fallback to general .nii.gz check
        image_files = sorted([
            p for p in raw_dir.rglob("*.nii.gz")
            if "label" not in p.name.lower() and "mask" not in p.name.lower() and not p.name.startswith("._")
        ])

    case_ids = []

    for img_path in image_files:
        case_id = normalize_case_key(img_path)
        
        # Check matching label in the same directory or globally in raw_dir
        label_path = img_path.parent / f"{case_id}{label_suffix}"
        if not label_path.exists():
            potential = list(img_path.parent.glob(f"*{case_id}*label*.nii.gz"))
            if potential:
                label_path = potential[0]
            else:
                global_potential = list(raw_dir.rglob(f"*{case_id}*label*.nii.gz"))
                if global_potential:
                    label_path = global_potential[0]
                else:
                    print(f"  [skip] {case_id}: no matching label file found")
                    continue

        # nnU-Net naming convention: <case>_<channel:04d>.nii.gz for images
        shutil.copy2(img_path, images_tr / f"{case_id}_0000.nii.gz")
        shutil.copy2(label_path, labels_tr / f"{case_id}.nii.gz")
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

    return {
        "cases_found": len(image_files),
        "cases_converted": len(case_ids),
        "output_dir": str(out_root),
    }


def _make_dummy_imagecas_folder(target_dir: Path, n_cases: int = 3) -> Path:
    """Builds synthetic NIfTI volumes in 1-200 layout for testing."""
    import numpy as np
    import nibabel as nib

    subfolder = target_dir / "1-200"
    subfolder.mkdir(parents=True, exist_ok=True)
    affine = np.eye(4)
    affine[0, 0] = affine[1, 1] = 0.5
    affine[2, 2] = 0.5

    rng = np.random.default_rng(42)
    for i in range(1, n_cases + 1):
        shape = (48, 48, 48)
        volume = rng.normal(loc=40, scale=60, size=shape).astype(np.float32)
        label = np.zeros(shape, dtype=np.uint8)
        for z in range(shape[2]):
            cx = 24 + int(6 * np.sin(z / 6.0))
            cy = 24 + int(4 * np.cos(z / 8.0))
            label[cx - 1 : cx + 2, cy - 1 : cy + 2, z] = 1
            volume[cx - 1 : cx + 2, cy - 1 : cy + 2, z] += 250.0

        nib.save(nib.Nifti1Image(volume, affine), subfolder / f"{i}.img.nii.gz")
        nib.save(nib.Nifti1Image(label, affine), subfolder / f"{i}.label.nii.gz")

    return target_dir


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert ImageCAS dataset (including subfolders 1-200, 201-400...) to nnU-Net v2 raw format"
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="data/extracted",
        help="Path to extracted ImageCAS folder containing range subfolders (1-200, 201-400...)",
    )
    parser.add_argument(
        "--nnunet_raw",
        type=str,
        default="Phase1_nnUNet/nnUNet_raw",
        help="Target nnUNet_raw directory",
    )
    parser.add_argument(
        "--test_mode",
        action="store_true",
        help="Generate synthetic dummy cases to smoke test the conversion",
    )
    args = parser.parse_args()

    print("=" * 70)
    print("Phase 1 — nnU-Net baseline: ImageCAS Dataset Conversion")
    print("=" * 70)

    if args.test_mode:
        dummy_raw = Path("Phase1_nnUNet/_test_raw_imagecas")
        dummy_out = Path("Phase1_nnUNet/_test_nnUNet_raw")
        print(f"\n[Test Mode] Generating synthetic ImageCAS cases in {dummy_raw}...")
        _make_dummy_imagecas_folder(dummy_raw, n_cases=3)
        summary = convert_imagecas_to_nnunet(dummy_raw, dummy_out)
        shutil.rmtree(dummy_raw, ignore_errors=True)
        shutil.rmtree(dummy_out, ignore_errors=True)
        print("\nTest Conversion Summary:")
        for k, v in summary.items():
            print(f"  {k}: {v}")
        print("\n[SUCCESS] Test mode completed.")
        return

    raw_path = Path(args.data_dir).expanduser().resolve()
    if not raw_path.exists():
        # Try Data/extracted
        alt = raw_path.parent / "Data" / "extracted" if raw_path.name == "extracted" else None
        if alt and alt.exists():
            raw_path = alt
        else:
            raise FileNotFoundError(f"Raw directory does not exist: {raw_path}")

    summary = convert_imagecas_to_nnunet(raw_path, args.nnunet_raw)
    print("\nConversion Summary:")
    for k, v in summary.items():
        print(f"  {k}: {v}")

    print("\nNext commands for nnU-Net v2:")
    print(f"  export nnUNet_raw={Path(args.nnunet_raw).resolve()}")
    print(f"  export nnUNet_preprocessed={Path(args.nnunet_raw).resolve().parent / 'nnUNet_preprocessed'}")
    print(f"  export nnUNet_results={Path(args.nnunet_raw).resolve().parent / 'nnUNet_results'}")
    print(f"  nnUNetv2_plan_and_preprocess -d {DATASET_ID} --verify_dataset_integrity")
    print(f"  nnUNetv2_train {DATASET_ID} 3d_fullres 0")


if __name__ == "__main__":
    main()
