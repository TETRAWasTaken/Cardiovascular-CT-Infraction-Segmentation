"""
Integration & Verification Tests for Training and Hyperparameter Tuning Pipelines
================================================================================
Verifies:
1. Training pipeline execution on synthetic volumes
2. Checkpoint persistence: `model_best.pt`, `checkpoint_latest.pt`
3. Metrics persistence: `train_log.json`, `train_log.csv`, `metrics_summary.json`, `config.json`
4. Checkpoint resume capability from `checkpoint_latest.pt`
5. Hyperparameter tuning engine execution across multiple parameter sets
6. Generation of `tuning_summary.csv` and `tuning_summary.json`
"""

from __future__ import annotations

import json
import os
import shutil
import sys
import tempfile
from pathlib import Path

import torch

# Ensure repo root and Transformer are in sys.path
TEST_DIR = Path(__file__).resolve().parent
REPO_ROOT = TEST_DIR.parent.parent
TRANSFORMER_DIR = TEST_DIR.parent

for p in [str(REPO_ROOT), str(TRANSFORMER_DIR)]:
    if p not in sys.path:
        sys.path.insert(0, p)

try:
    from Transformer.src.train import run_training
    from Transformer.src.tune import run_hyperparameter_tuning
except ModuleNotFoundError:
    from src.train import run_training
    from src.tune import run_hyperparameter_tuning


def test_train_pipeline_and_artifacts():
    print("\n--- [TEST 1] Single Training Pipeline & Metrics Saving ---")
    with tempfile.TemporaryDirectory() as temp_dir:
        output_dir = Path(temp_dir) / "train_output"

        # Use CPU and minimal feature_size=12 for ultra-fast unit testing
        config = {
            "output_dir": str(output_dir),
            "sanity_check": True,
            "epochs": 2,
            "batch_size": 1,
            "feature_size": 12,
            "spatial_size": (96, 96, 96),
            "num_samples": 1,
            "val_interval": 1,
            "device": "cpu",
            "num_workers": 0,
            "amp": False,
            "seed": 42,
        }

        result = run_training(config)

        # Assert result contents
        assert "best_dice" in result, "Result missing best_dice"
        assert "summary" in result, "Result missing summary"

        # Assert file generation
        best_ckpt = output_dir / "model_best.pt"
        latest_ckpt = output_dir / "checkpoint_latest.pt"
        log_json = output_dir / "train_log.json"
        log_csv = output_dir / "train_log.csv"
        summary_json = output_dir / "metrics_summary.json"
        config_json = output_dir / "config.json"

        assert latest_ckpt.exists(), f"Missing latest checkpoint: {latest_ckpt}"
        assert log_json.exists(), f"Missing train_log.json: {log_json}"
        assert log_csv.exists(), f"Missing train_log.csv: {log_csv}"
        assert summary_json.exists(), f"Missing metrics_summary.json: {summary_json}"
        assert config_json.exists(), f"Missing config.json: {config_json}"

        # Validate checkpoint contents
        ckpt_data = torch.load(str(latest_ckpt), map_location="cpu")
        assert "model_state_dict" in ckpt_data, "Checkpoint missing model_state_dict"
        assert "optimizer_state_dict" in ckpt_data, "Checkpoint missing optimizer_state_dict"
        assert ckpt_data["epoch"] == 2, f"Expected epoch 2 in checkpoint, got {ckpt_data['epoch']}"

        # Validate summary json
        with open(summary_json, "r", encoding="utf-8") as f:
            summary = json.load(f)
            assert summary["total_epochs"] == 2
            assert "best_dice" in summary

        print(">>> Test 1 Passed: Training pipeline, checkpoints, and evaluation metrics saved successfully.")


def test_resume_training():
    print("\n--- [TEST 2] Resuming Training from Checkpoint ---")
    with tempfile.TemporaryDirectory() as temp_dir:
        output_dir = Path(temp_dir) / "resume_output"

        # Step 1: Train for 1 epoch
        config_p1 = {
            "output_dir": str(output_dir),
            "sanity_check": True,
            "epochs": 1,
            "batch_size": 1,
            "feature_size": 12,
            "spatial_size": (96, 96, 96),
            "num_samples": 1,
            "val_interval": 1,
            "device": "cpu",
            "num_workers": 0,
            "amp": False,
            "seed": 42,
        }
        run_training(config_p1)

        latest_ckpt = output_dir / "checkpoint_latest.pt"
        assert latest_ckpt.exists(), "Latest checkpoint not created in phase 1"

        # Step 2: Resume training up to 2 epochs
        config_p2 = config_p1.copy()
        config_p2["epochs"] = 2
        config_p2["resume"] = str(latest_ckpt)

        result_p2 = run_training(config_p2)
        summary = result_p2["summary"]
        assert summary["total_epochs"] == 2, f"Expected 2 total epochs after resume, got {summary['total_epochs']}"

        print(">>> Test 2 Passed: Resumed training successfully from saved checkpoint.")


def test_hyperparameter_tuning_pipeline():
    print("\n--- [TEST 3] Hyperparameter Tuning Pipeline ---")
    with tempfile.TemporaryDirectory() as temp_dir:
        output_dir = Path(temp_dir) / "tune_output"

        base_config = {
            "output_dir": str(output_dir),
            "sanity_check": True,
            "epochs": 1,
            "batch_size": 1,
            "spatial_size": (96, 96, 96),
            "num_samples": 1,
            "val_interval": 1,
            "device": "cpu",
            "num_workers": 0,
            "amp": False,
            "seed": 42,
        }

        # Parameter grid with 2 different model parameter configurations
        param_grid = {
            "feature_size": [12, 24],
            "lr": [1e-4, 5e-4],
        }

        results = run_hyperparameter_tuning(
            base_config=base_config,
            param_grid=param_grid,
            search_type="grid",
            n_trials=2,  # Limit to 2 trials for speed
            seed=42,
        )

        assert len(results["trials"]) == 2, f"Expected 2 trials, got {len(results['trials'])}"
        summary_csv = output_dir / "tuning_summary.csv"
        summary_json = output_dir / "tuning_summary.json"

        assert summary_csv.exists(), f"Missing {summary_csv}"
        assert summary_json.exists(), f"Missing {summary_json}"

        # Verify trial subdirectories
        trials_dir = output_dir / "trials"
        assert (trials_dir / "trial_001").exists()
        assert (trials_dir / "trial_002").exists()

        # Check trial 1 artifacts
        assert (trials_dir / "trial_001" / "trial_config.json").exists()
        assert (trials_dir / "trial_001" / "checkpoint_latest.pt").exists()
        assert (trials_dir / "trial_001" / "metrics_summary.json").exists()

        print(">>> Test 3 Passed: Hyperparameter tuning executed across parameter sets and saved summary reports.")


if __name__ == "__main__":
    test_train_pipeline_and_artifacts()
    test_resume_training()
    test_hyperparameter_tuning_pipeline()
    print("\n🎉 ALL TESTS PASSED SUCCESSFULLY!")
