"""
ImageCAS Data Ingestion
=======================

Purpose:
    Discover CT volumes and corresponding segmentation masks in an ImageCAS
    dataset directory, inspect their metadata, create image/mask pairs, and
    save a dataset manifest for the later training pipeline.

IMPORTANT:
    ImageCAS folder layouts can differ depending on how the dataset was
    downloaded/extracted. This script therefore searches recursively instead
    of assuming one fixed folder structure.

Expected eventual flow:
    ImageCAS -> ingestion -> preprocessing -> Dataset/DataLoader -> 3D U-Net

Install:
    pip install SimpleITK pandas

Usage:
    python data_ingestion.py --data_dir /path/to/ImageCAS

Optional:
    python data_ingestion.py --data_dir /path/to/ImageCAS \
        --output_csv imagecas_manifest.csv
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Optional

import pandas as pd
import SimpleITK as sitk


SUPPORTED_EXTENSIONS = (
    ".nii",
    ".nii.gz",
    ".mha",
    ".mhd",
    ".nrrd",
)


def is_medical_image(path: Path) -> bool:
    """Return True if the file looks like a supported medical-image file."""
    name = path.name.lower()
    return name.endswith(SUPPORTED_EXTENSIONS)


def strip_medical_extension(path: Path) -> str:
    """Remove .nii.gz / .nii / .mha / .mhd / .nrrd from a filename."""
    name = path.name
    lower = name.lower()

    if lower.endswith(".nii.gz"):
        return name[:-7]
    if lower.endswith(".nii"):
        return name[:-4]
    if lower.endswith(".mha"):
        return name[:-4]
    if lower.endswith(".mhd"):
        return name[:-4]
    if lower.endswith(".nrrd"):
        return name[:-5]

    return path.stem


def normalize_case_key(path: Path) -> str:
    """
    Generate a normalized identifier used to match an image with its mask.

    The function removes common segmentation/mask words. This is intentionally
    conservative; the generated manifest should always be inspected before
    training.
    """
    key = strip_medical_extension(path).lower()

    replacements = [
        r"[_\-]?(label|labels|mask|masks|seg|segmentation)$",
        r"[_\-]?(label|labels|mask|masks|seg|segmentation)[_\-]?",
    ]

    for pattern in replacements:
        key = re.sub(pattern, "", key)

    key = re.sub(r"[^a-z0-9]+", "_", key).strip("_")
    return key


def looks_like_mask(path: Path) -> bool:
    """Identify likely segmentation-mask files from their names/parent folders."""
    text = " ".join(
        [
            path.name.lower(),
            path.parent.name.lower(),
            str(path.parent.parent).lower() if path.parent.parent else "",
        ]
    )

    mask_keywords = (
        "mask",
        "masks",
        "label",
        "labels",
        "seg",
        "segmentation",
        "annotation",
        "annotations",
    )

    return any(keyword in text for keyword in mask_keywords)


def discover_files(data_dir: Path) -> list[Path]:
    """Recursively find supported medical-image files."""
    files = [
        p for p in data_dir.rglob("*")
        if p.is_file() and is_medical_image(p)
    ]
    return sorted(files)


def inspect_image(path: Path) -> dict:
    """
    Read basic 3D image metadata without loading the complete volume into
    memory.
    """
    try:
        image = sitk.ReadImage(str(path))

        return {
            "status": "ok",
            "dimensions": "x".join(map(str, image.GetSize())),
            "spacing": "x".join(f"{x:.6g}" for x in image.GetSpacing()),
            "origin": "x".join(f"{x:.6g}" for x in image.GetOrigin()),
            "direction": ",".join(f"{x:.6g}" for x in image.GetDirection()),
            "component_type": image.GetPixelIDTypeAsString(),
        }

    except Exception as exc:
        return {
            "status": f"error: {exc}",
            "dimensions": "",
            "spacing": "",
            "origin": "",
            "direction": "",
            "component_type": "",
        }


def build_manifest(data_dir: Path) -> pd.DataFrame:
    """Discover files, classify likely masks, and build image/mask pairs."""
    files = discover_files(data_dir)

    if not files:
        raise FileNotFoundError(
            f"No supported medical-image files were found under: {data_dir}"
        )

    mask_files = [p for p in files if looks_like_mask(p)]
    image_files = [p for p in files if p not in mask_files]

    mask_by_key = {}
    for mask in mask_files:
        key = normalize_case_key(mask)
        mask_by_key.setdefault(key, []).append(mask)

    rows = []

    for image in image_files:
        key = normalize_case_key(image)
        possible_masks = mask_by_key.get(key, [])

        # If there is exactly one likely mask, pair it.
        mask = possible_masks[0] if len(possible_masks) == 1 else None

        metadata = inspect_image(image)

        rows.append(
            {
                "case_id": key,
                "image_path": str(image.resolve()),
                "mask_path": str(mask.resolve()) if mask else "",
                "has_mask": bool(mask),
                "status": metadata["status"],
                "dimensions": metadata["dimensions"],
                "spacing": metadata["spacing"],
                "origin": metadata["origin"],
                "direction": metadata["direction"],
                "component_type": metadata["component_type"],
            }
        )

    return pd.DataFrame(rows)


def print_summary(df: pd.DataFrame) -> None:
    """Print a compact summary useful for checking the ingestion stage."""
    print("\n" + "=" * 70)
    print("ImageCAS DATA INGESTION SUMMARY")
    print("=" * 70)

    print(f"Discovered image volumes : {len(df)}")
    print(f"Images with paired masks : {int(df['has_mask'].sum())}")
    print(f"Images without masks     : {int((~df['has_mask']).sum())}")
    print(
        f"Images read successfully: "
        f"{int((df['status'] == 'ok').sum())}"
    )

    print("\nFirst 10 discovered cases:")
    columns = [
        "case_id",
        "has_mask",
        "dimensions",
        "spacing",
        "image_path",
        "mask_path",
    ]
    print(df[columns].head(10).to_string(index=False))

    print("\nNOTE:")
    print(
        "The mask pairing is heuristic because the exact extracted ImageCAS "
        "folder structure has not been inspected yet."
    )
    print(
        "Check the generated CSV before using it for training."
    )
    print("=" * 70 + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Ingest and inspect an ImageCAS medical-image dataset."
    )

    parser.add_argument(
        "--data_dir",
        type=str,
        required=True,
        help="Root directory containing the extracted ImageCAS dataset.",
    )

    parser.add_argument(
        "--output_csv",
        type=str,
        default="imagecas_manifest.csv",
        help="Path for the generated dataset manifest CSV.",
    )

    args = parser.parse_args()

    data_dir = Path(args.data_dir).expanduser().resolve()

    if not data_dir.exists():
        raise FileNotFoundError(
            f"Dataset directory does not exist: {data_dir}"
        )

    print(f"Scanning ImageCAS directory:\n{data_dir}\n")

    manifest = build_manifest(data_dir)

    output_path = Path(args.output_csv).expanduser().resolve()
    manifest.to_csv(output_path, index=False)

    print_summary(manifest)

    print(f"Manifest saved to:\n{output_path}")


if __name__ == "__main__":
    main()
