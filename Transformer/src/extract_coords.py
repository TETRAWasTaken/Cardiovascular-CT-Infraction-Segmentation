"""
Phase 5: Coordinate Extraction & Skeletonization Post-Processing
===============================================================
Converts 3D probability maps or segmentation masks into physical world coordinate
vectors (X, Y, Z in mm) for all detected coronary arteries.

Pipeline:
1. Threshold probability map (p > 0.5)
2. 3D Largest Connected Component Analysis with cc3d to remove disconnected noise
3. 3D skeletonization via skimage.morphology.skeletonize to centerlines
4. Map voxel indices (z, y, x) to physical coordinates (X, Y, Z) using SimpleITK
5. Output structured JSON & CSV formats
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import cc3d
import numpy as np
import SimpleITK as sitk
from skimage.morphology import skeletonize


def threshold_probabilities(prob_map: np.ndarray, threshold: float = 0.5) -> np.ndarray:
    """Threshold continuous sigmoid probability map into binary mask (0 or 1)."""
    return (prob_map > threshold).astype(np.uint8)


def filter_largest_connected_components(
    binary_mask: np.ndarray,
    top_k: int = 1,
    connectivity: int = 26,
) -> np.ndarray:
    """
    Applies 3D Connected Component Analysis using cc3d to eliminate small spurious
    false-positive voxel clusters and isolate main coronary artery tree branches.
    """
    if not np.any(binary_mask):
        return np.zeros_like(binary_mask, dtype=np.uint8)

    # cc3d.largest_k extracts the k largest components
    cleaned_mask = cc3d.largest_k(
        binary_mask,
        k=top_k,
        connectivity=connectivity,
        delta=0,
    )
    return (cleaned_mask > 0).astype(np.uint8)


def extract_3d_skeleton(binary_mask: np.ndarray) -> np.ndarray:
    """
    Computes 3D topological medial axis skeletonization using skimage.morphology.skeletonize,
    reducing volumetric vessel tubes down to 1-voxel thin centerlines.
    """
    if not np.any(binary_mask):
        return np.zeros_like(binary_mask, dtype=bool)

    # 3D skeletonization (handles volumetric 3D arrays)
    skeleton = skeletonize(binary_mask.astype(bool))
    return skeleton


def map_voxel_to_physical_coordinates(
    skeleton_mask: np.ndarray,
    sitk_reference_image: sitk.Image,
) -> list[dict[str, Any]]:
    """
    Maps voxel coordinates (z, y, x) to physical world space coordinates (X, Y, Z in mm)
    incorporating spacing, origin, and directional cosine matrix via SimpleITK.
    """
    # SimpleITK indexing is (x, y, z), whereas numpy array indexing is (z, y, x)
    voxel_coords = np.argwhere(skeleton_mask)  # Array of [z, y, x]
    centerline_points: list[dict[str, Any]] = []

    for point_idx, (z, y, x) in enumerate(voxel_coords):
        sitk_index = (int(x), int(y), int(z))
        physical_point = sitk_reference_image.TransformIndexToPhysicalPoint(sitk_index)

        centerline_points.append({
            "point_id": point_idx,
            "voxel_index": {"z": int(z), "y": int(y), "x": int(x)},
            "physical_coord_mm": {
                "X": round(float(physical_point[0]), 4),
                "Y": round(float(physical_point[1]), 4),
                "Z": round(float(physical_point[2]), 4),
            },
        })

    return centerline_points


def process_volume(
    prob_or_mask: np.ndarray,
    sitk_img: sitk.Image,
    threshold: float = 0.5,
    top_k: int = 1,
) -> tuple[np.ndarray, np.ndarray, list[dict[str, Any]]]:
    """Runs complete coordinate extraction and skeletonization pipeline."""
    # 1. Thresholding
    binary = threshold_probabilities(prob_or_mask, threshold=threshold)

    # 2. 3D Largest Connected Component Analysis (cc3d)
    cleaned = filter_largest_connected_components(binary, top_k=top_k)

    # 3. 3D Skeletonization (skimage)
    skeleton = extract_3d_skeleton(cleaned)

    # 4. Map to Physical World Coordinates (SimpleITK)
    coords = map_voxel_to_physical_coordinates(skeleton, sitk_img)

    return cleaned, skeleton, coords


def export_results(
    centerline_points: list[dict[str, Any]],
    output_json: str | Path | None = None,
    output_csv: str | Path | None = None,
    metadata: dict[str, Any] | None = None,
) -> None:
    """Exports centerline coordinates to JSON and CSV formats."""
    result_data = {
        "metadata": metadata or {},
        "total_centerline_points": len(centerline_points),
        "centerline_points": centerline_points,
    }

    if output_json:
        json_path = Path(output_json)
        json_path.parent.mkdir(parents=True, exist_ok=True)
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(result_data, f, indent=2)
        print(f"[EXPORT] Saved {len(centerline_points)} centerline coordinates to {json_path}")

    if output_csv:
        csv_path = Path(output_csv)
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["point_id", "voxel_z", "voxel_y", "voxel_x", "phys_X_mm", "phys_Y_mm", "phys_Z_mm"])
            for pt in centerline_points:
                writer.writerow([
                    pt["point_id"],
                    pt["voxel_index"]["z"],
                    pt["voxel_index"]["y"],
                    pt["voxel_index"]["x"],
                    pt["physical_coord_mm"]["X"],
                    pt["physical_coord_mm"]["Y"],
                    pt["physical_coord_mm"]["Z"],
                ])
        print(f"[EXPORT] Saved CSV coordinates to {csv_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract physical world coordinates and centerlines from 3D CT artery probability maps"
    )
    parser.add_argument("--input", type=str, default="", help="Path to input 3D NIfTI probability/segmentation file")
    parser.add_argument("--threshold", type=float, default=0.5, help="Probability threshold (default: 0.5)")
    parser.add_argument("--top_k", type=int, default=1, help="Top k largest connected components (default: 1)")
    parser.add_argument(
        "--output_json",
        type=str,
        default=str(Path(__file__).resolve().parents[1] / "artifacts" / "artery_centerlines.json"),
        help="Path for output JSON file",
    )
    parser.add_argument(
        "--output_csv",
        type=str,
        default="",
        help="Path for optional output CSV file",
    )
    parser.add_argument(
        "--input_synthetic",
        action="store_true",
        help="Generate synthetic test artery volume for verification artifact",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.input_synthetic or not args.input:
        print("[POST-PROC] Creating synthetic reference volume for coordinate extraction...")
        # Create a 3D synthetic volume with curved artery tube
        d, h, w = 96, 96, 96
        prob_map = np.zeros((d, h, w), dtype=np.float32)

        # Curved tubular artery
        z_pts = np.linspace(10, 85, 120)
        y_pts = 48.0 + 15.0 * np.sin(z_pts / 12.0)
        x_pts = 48.0 + 15.0 * np.cos(z_pts / 12.0)

        for zi, yi, xi in zip(z_pts.astype(int), y_pts.astype(int), x_pts.astype(int)):
            for dz in range(-2, 3):
                for dy in range(-2, 3):
                    for dx in range(-2, 3):
                        if dz**2 + dy**2 + dx**2 <= 4:
                            prob_map[zi + dz, yi + dy, xi + dx] = 0.95

        # Add minor background noise cluster to test cc3d filtering
        prob_map[5:8, 5:8, 5:8] = 0.90

        # SimpleITK image with realistic spacing and origin
        sitk_img = sitk.GetImageFromArray(prob_map)
        sitk_img.SetSpacing((0.8, 0.8, 1.0))
        sitk_img.SetOrigin((-100.0, -100.0, 0.0))
        meta = {
            "source": "synthetic_test_volume",
            "spacing": [0.8, 0.8, 1.0],
            "origin": [-100.0, -100.0, 0.0],
            "dimensions": [w, h, d],
        }
    else:
        print(f"[POST-PROC] Loading NIfTI volume from: {args.input}")
        sitk_img = sitk.ReadImage(args.input)
        prob_map = sitk.GetArrayFromImage(sitk_img)  # Returns (z, y, x)
        meta = {
            "source": str(args.input),
            "spacing": list(sitk_img.GetSpacing()),
            "origin": list(sitk_img.GetOrigin()),
            "direction": list(sitk_img.GetDirection()),
            "dimensions": list(sitk_img.GetSize()),
        }

    cleaned_mask, skeleton_mask, coords = process_volume(
        prob_or_mask=prob_map,
        sitk_img=sitk_img,
        threshold=args.threshold,
        top_k=args.top_k,
    )

    print(f"[POST-PROC] Connected Component Filtering: {np.count_nonzero(cleaned_mask)} voxels preserved.")
    print(f"[POST-PROC] Skeletonization: {np.count_nonzero(skeleton_mask)} centerline voxels identified.")

    export_results(
        centerline_points=coords,
        output_json=args.output_json,
        output_csv=args.output_csv if args.output_csv else None,
        metadata=meta,
    )


if __name__ == "__main__":
    main()
