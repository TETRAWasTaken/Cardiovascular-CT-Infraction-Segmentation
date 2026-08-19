"""
ImageCAS Data Ingestion
=======================

Purpose:
    Discover CT volumes and corresponding segmentation masks in an ImageCAS
    dataset directory (including nested ranges like extracted/1-200, 201-400, etc.),
    inspect their metadata, parse official splits if `imageCAS_data_split.xlsx` is provided,
    create image/mask pairs, and save a dataset manifest for the training pipeline.

Supported layout:
    data/
      ├── compressed/
      ├── extracted/
      │   ├── 1-200/ (1.img.nii.gz, 1.label.nii.gz, ...)
      │   ├── 201-400/
      │   ├── 401-600/
      │   ├── 601-800/
      │   └── 801-1000/
      └── imageCAS_data_split.xlsx

Usage:
    python Phase1_nnUNet/data_ingestion.py --data_dir data/extracted

With Excel split:
    python Phase1_nnUNet/data_ingestion.py \
        --data_dir data/extracted \
        --split_excel data/imageCAS_data_split.xlsx \
        --output_csv BaselineUNET/imagecas_manifest.csv
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Optional, Dict

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
    Removes keywords like .img, .label, _mask, etc. so '1.img.nii.gz' and
    '1.label.nii.gz' both produce '1'.
    """
    key = strip_medical_extension(path).lower()

    replacements = [
        r"[_\.\-]?(label|labels|mask|masks|seg|segmentation|img|image|images|ct)$",
        r"^(label|labels|mask|masks|seg|segmentation|img|image|images|ct)[_\.\-]?",
        r"[_\.\-]?(label|labels|mask|masks|seg|segmentation|img|image|images|ct)[_\.\-]?",
    ]

    for pattern in replacements:
        key = re.sub(pattern, "", key)

    key = re.sub(r"[^a-z0-9]+", "_", key).strip("_")
    return key


def looks_like_mask(path: Path) -> bool:
    """Identify likely segmentation-mask files from their names."""
    name = path.name.lower()
    mask_keywords = (
        "label",
        "labels",
        "mask",
        "masks",
        "seg",
        "segmentation",
        "annotation",
        "annotations",
    )
    return any(keyword in name for keyword in mask_keywords)


def discover_files(data_dir: Path) -> list[Path]:
    """Recursively find supported medical-image files across all subdirectories."""
    files = [
        p for p in data_dir.rglob("*")
        if p.is_file() and is_medical_image(p) and not p.name.startswith("._")
    ]
    return sorted(files)


def inspect_image(path: Path) -> dict:
    """
    Read basic 3D image metadata without loading the complete volume into memory.
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


def load_split_lookup(split_excel_path: Optional[Path]) -> Dict[str, str]:
    """
    Load official ImageCAS data split from Excel if available.
    Returns mapping from case_id / filename to split name ('train', 'val', 'test').
    """
    if split_excel_path is None or not split_excel_path.exists():
        return {}

    try:
        xls = pd.ExcelFile(split_excel_path)
        split_map = {}
        for sheet_name in xls.sheet_names:
            df = pd.read_excel(xls, sheet_name=sheet_name)
            sheet_lower = sheet_name.lower()
            split_label = "train"
            if "test" in sheet_lower:
                split_label = "test"
            elif "val" in sheet_lower:
                split_label = "val"
            elif "train" in sheet_lower:
                split_label = "train"

            for col in df.columns:
                for val in df[col].dropna():
                    val_str = str(val).strip()
                    clean_id = re.sub(r"[^0-9a-zA-Z]", "", val_str)
                    split_map[clean_id] = split_label
                    split_map[val_str] = split_label
        return split_map
    except Exception as e:
        print(f"Warning: Could not parse Excel split ({e}). Falling back to automatic split.")
        return {}


def build_manifest(data_dir: Path, split_excel_path: Optional[Path] = None) -> pd.DataFrame:
    """Discover files, classify masks and images across subdirectories, and build manifest."""
    files = discover_files(data_dir)

    if not files:
        raise FileNotFoundError(
            f"No supported medical-image files were found under: {data_dir}"
        )

    # Check for split Excel in parent if not specified
    if split_excel_path is None:
        potential_splits = [
            data_dir.parent / "imageCAS_data_split.xlsx",
            data_dir / "imageCAS_data_split.xlsx",
            data_dir.parent / "imagecas_data_split.xlsx",
        ]
        for p in potential_splits:
            if p.exists():
                split_excel_path = p
                print(f"Auto-detected official split Excel: {split_excel_path}")
                break

    split_lookup = load_split_lookup(split_excel_path)

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

        mask = possible_masks[0] if len(possible_masks) == 1 else None

        metadata = inspect_image(image)
        case_split = split_lookup.get(key, split_lookup.get(image.stem, ""))

        rows.append(
            {
                "case_id": key,
                "folder": image.parent.name,
                "image_path": str(image.resolve()),
                "mask_path": str(mask.resolve()) if mask else "",
                "has_mask": bool(mask),
                "split": case_split,
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
    """Print a compact summary of discovered subfolders and case pairings."""
    print("\n" + "=" * 70)
    print("ImageCAS DATA INGESTION SUMMARY")
    print("=" * 70)

    print(f"Discovered image volumes : {len(df)}")
    print(f"Images with paired masks : {int(df['has_mask'].sum())}")
    print(f"Images without masks     : {int((~df['has_mask']).sum())}")
    print(f"Images read successfully : {int((df['status'] == 'ok').sum())}")

    if "folder" in df.columns:
        print("\nBreakdown by subfolder:")
        print(df.groupby("folder")["has_mask"].agg(total="count", paired_masks="sum").to_string())

    if "split" in df.columns and df["split"].nunique() > 1:
        print("\nBreakdown by split:")
        print(df["split"].value_counts().to_string())

    print("\nFirst 10 discovered cases:")
    columns = [
        "case_id",
        "folder",
        "has_mask",
        "dimensions",
        "spacing",
    ]
    print(df[columns].head(10).to_string(index=False))
    print("=" * 70 + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Ingest and inspect nested ImageCAS medical-image dataset."
    )

    parser.add_argument(
        "--data_dir",
        type=str,
        default="data/extracted",
        help="Root directory containing extracted ImageCAS (e.g. data/extracted or Data/extracted).",
    )

    parser.add_argument(
        "--split_excel",
        type=str,
        default="",
        help="Path to imageCAS_data_split.xlsx if available.",
    )

    parser.add_argument(
        "--output_csv",
        type=str,
        default="BaselineUNET/imagecas_manifest.csv",
        help="Path for the generated dataset manifest CSV.",
    )

    args = parser.parse_args()

    data_dir = Path(args.data_dir).expanduser().resolve()
    split_excel = Path(args.split_excel).expanduser().resolve() if args.split_excel else None

    if not data_dir.exists():
        # Check alternative case: Data/extracted
        alt_dir = data_dir.parent / "Data" / "extracted" if data_dir.name == "extracted" else None
        if alt_dir and alt_dir.exists():
            data_dir = alt_dir
        else:
            raise FileNotFoundError(f"Dataset directory does not exist: {data_dir}")

    print(f"Scanning ImageCAS directory (recursive):\n{data_dir}\n")

    manifest = build_manifest(data_dir, split_excel_path=split_excel)

    output_path = Path(args.output_csv).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    manifest.to_csv(output_path, index=False)

    print_summary(manifest)
    print(f"Manifest saved to:\n{output_path}")


if __name__ == "__main__":
    main()
