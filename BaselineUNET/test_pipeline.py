"""
BaselineUNET Pipeline Smoke Test
================================

Generates lightweight synthetic 3D volumes to verify the entire pipeline:
1. Synthetic CT volume generation (with simulated coronary vessels)
2. Manifest creation & validation
3. MONAI patch cropping & data loading
4. Model forward & backward pass with DiceCELoss
5. Sliding-window full-volume validation with Dice score calculation
6. Checkpoint saving

This allows complete verification without loading the 80+ GB real ImageCAS dataset.
"""

from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd

# Add project root to sys.path
project_root = Path(__file__).resolve().parents[1]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))


def create_synthetic_dataset(target_dir: Path, n_cases: int = 4) -> list[dict]:
    """Create tiny 3D NIfTI volumes with simulated curved vessel tubes."""
    target_dir.mkdir(parents=True, exist_ok=True)
    affine = np.eye(4)
    affine[0, 0] = affine[1, 1] = 0.5
    affine[2, 2] = 0.5

    rng = np.random.default_rng(42)
    rows = []

    for i in range(1, n_cases + 1):
        shape = (64, 64, 64)
        # Soft tissue background (~40 HU with standard deviation 50)
        volume = rng.normal(loc=40, scale=50, size=shape).astype(np.float32)
        label = np.zeros(shape, dtype=np.uint8)

        # Create a curved simulated coronary artery
        for z in range(shape[2]):
            cx = 32 + int(8 * np.sin(z / 6.0))
            cy = 32 + int(5 * np.cos(z / 8.0))
            label[max(0, cx - 1) : min(shape[0], cx + 2), max(0, cy - 1) : min(shape[1], cy + 2), z] = 1
            volume[max(0, cx - 1) : min(shape[0], cx + 2), max(0, cy - 1) : min(shape[1], cy + 2), z] += 250.0

        img_path = target_dir / f"{i}.img.nii.gz"
        lbl_path = target_dir / f"{i}.label.nii.gz"

        nib.save(nib.Nifti1Image(volume, affine), img_path)
        nib.save(nib.Nifti1Image(label, affine), lbl_path)

        rows.append({
            "case_id": str(i),
            "image_path": str(img_path.resolve()),
            "mask_path": str(lbl_path.resolve()),
            "has_mask": True,
            "status": "ok",
            "dimensions": "64x64x64",
            "spacing": "0.5x0.5x0.5",
        })

    manifest_path = target_dir / "test_manifest.csv"
    pd.DataFrame(rows).to_csv(manifest_path, index=False)
    print(f"Created {n_cases} synthetic 3D test cases at: {target_dir}")
    print(f"Manifest written to: {manifest_path}")
    return manifest_path


def run_smoke_test() -> None:
    test_dir = project_root / "BaselineUNET" / "_test_scratch"
    output_dir = test_dir / "runs"
    
    try:
        print("=" * 70)
        print("STEP 1: Generating Synthetic 3D Test Volumes")
        print("=" * 70)
        manifest_path = create_synthetic_dataset(test_dir / "raw_data", n_cases=4)

        print("\n" + "=" * 70)
        print("STEP 2: Executing BaselineUNET Training (2 Epochs, Synthetic Volumes)")
        print("=" * 70)
        
        from BaselineUNET.main import main as train_main
        
        # Override sys.argv to simulate CLI call
        sys.argv = [
            "main.py",
            "--manifest", str(manifest_path),
            "--patch_size", "32", "32", "32",
            "--num_samples", "2",
            "--batch_size", "2",
            "--epochs", "2",
            "--val_interval", "1",
            "--output_dir", str(output_dir),
            "--num_workers", "0",
            "--no_amp",
        ]
        
        train_main()

        print("\n" + "=" * 70)
        print("STEP 3: Verifying Artifacts")
        print("=" * 70)
        
        best_model = output_dir / "best_metric_model.pth"
        latest_ckpt = output_dir / "latest_checkpoint.pth"
        history_file = output_dir / "training_history.json"

        assert best_model.exists(), f"Missing best model: {best_model}"
        assert latest_ckpt.exists(), f"Missing latest checkpoint: {latest_ckpt}"
        assert history_file.exists(), f"Missing history log: {history_file}"

        print(f"✓ Found best model checkpoint: {best_model} ({best_model.stat().st_size / 1e6:.2f} MB)")
        print(f"✓ Found latest checkpoint: {latest_ckpt}")
        print(f"✓ Found training history log: {history_file}")
        print("\n[SUCCESS] BaselineUNET pipeline passed end-to-end smoke test!")

    finally:
        # Clean up scratch files
        if test_dir.exists():
            shutil.rmtree(test_dir)
            print(f"Cleaned up scratch directory: {test_dir}")


if __name__ == "__main__":
    run_smoke_test()
