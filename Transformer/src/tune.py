"""
Hyperparameter Tuning Engine for 3D SwinUNETR Segmentation Pipeline
===================================================================
Features:
- Supports Model Architecture parameters:
    * `feature_size`: SwinUNETR embedding dimension (e.g., 24, 36, 48)
    * `drop_rate`: Transformer block dropout
    * `attn_drop_rate`: Window attention dropout
    * `dropout_path_rate`: Stochastic depth rate
    * `use_checkpoint`: Activation checkpointing
- Supports Loss parameters:
    * `lambda_dice`, `lambda_focal`, `gamma` (Focal loss modulation)
- Supports Training & Optimization parameters:
    * `lr`, `weight_decay`, `optimizer` (adamw, adam, sgd), `scheduler`, `batch_size`
- Search Strategies:
    * `grid`: Exhaustive Cartesian product of candidate parameter values
    * `random`: Random sampling of hyperparameter combinations for N trials
    * Predefined presets: `--preset model_arch`, `--preset loss_tuning`, `--preset opt_tuning`, `--preset quick_test`
    * Inline JSON or config file: `--param_grid '{"feature_size": [24, 48], "lr": [1e-4, 3e-4]}'`
- Complete Artifact & Metrics Persistence:
    * Independent trial folders: `output_dir/trials/trial_001/` with trial checkpoints & logs
    * Comparative Leaderboard Table printed to console
    * `tuning_summary.csv` and `tuning_summary.json` ranking all combinations
    * `best_trial_config.json` and `best_model_overall.pt` saved to root output directory
- Fault-Tolerant: Catches CUDA OOM or trial errors, flushes cache, and continues remaining trials.
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import os
import random
import shutil
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

# Ensure parent and repo root are in sys.path
CURRENT_DIR = Path(__file__).resolve().parent
REPO_ROOT = CURRENT_DIR.parent.parent
TRANSFORMER_DIR = CURRENT_DIR.parent

for p in [str(REPO_ROOT), str(TRANSFORMER_DIR)]:
    if p not in sys.path:
        sys.path.insert(0, p)

try:
    from Transformer.src.train import run_training
except ModuleNotFoundError:
    from src.train import run_training


PRESET_GRIDS: dict[str, dict[str, list[Any]]] = {
    "model_arch": {
        "feature_size": [24, 36, 48],
        "drop_rate": [0.0, 0.1],
        "attn_drop_rate": [0.0, 0.1],
    },
    "loss_tuning": {
        "lambda_dice": [1.0],
        "lambda_focal": [0.2, 0.5, 1.0],
        "gamma": [1.5, 2.0, 2.5],
    },
    "opt_tuning": {
        "lr": [5e-5, 1e-4, 3e-4],
        "weight_decay": [1e-5, 1e-4],
        "scheduler": ["cosine", "plateau"],
    },
    "quick_test": {
        "feature_size": [24, 48],
        "lr": [1e-4, 2e-4],
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Automated Hyperparameter Tuning Engine for SwinUNETR Segmentation"
    )
    # Target Data & Output
    parser.add_argument(
        "--data_dir",
        type=str,
        default="",
        help="Path containing NIfTI image and label pairs",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=str(TRANSFORMER_DIR / "artifacts" / "tuning_experiments"),
        help="Base directory to save tuning trials, checkpoints, and summary tables",
    )
    parser.add_argument(
        "--sanity_check",
        action="store_true",
        help="Run tuning with synthetic data for pipeline validation",
    )

    # Search Configuration
    parser.add_argument(
        "--search_type",
        type=str,
        default="grid",
        choices=["grid", "random"],
        help="Search strategy: 'grid' (all combinations) or 'random' (random sampling)",
    )
    parser.add_argument(
        "--preset",
        type=str,
        default="quick_test",
        choices=list(PRESET_GRIDS.keys()) + ["none"],
        help="Preset hyperparameter grid to tune (default: 'quick_test')",
    )
    parser.add_argument(
        "--param_grid",
        type=str,
        default="",
        help='JSON string specifying parameter candidates, e.g. \'{"feature_size": [24, 48], "lr": [1e-4, 3e-4]}\'',
    )
    parser.add_argument(
        "--config_file",
        type=str,
        default="",
        help="Path to JSON file specifying parameter search grid",
    )
    parser.add_argument(
        "--n_trials",
        type=int,
        default=5,
        help="Maximum number of trials to run (especially for random search)",
    )

    # Per-Trial Training Constraints
    parser.add_argument(
        "--epochs_per_trial",
        type=int,
        default=2,
        help="Epochs to train each trial (e.g. 5-10 for fast screening, 50-100 for full runs)",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=2,
        help="Base batch size per training step",
    )
    parser.add_argument(
        "--val_interval",
        type=int,
        default=1,
        help="Epoch interval for validation evaluation",
    )
    parser.add_argument(
        "--early_stopping_patience",
        type=int,
        default=0,
        help="Patience in validation evaluations before early stopping trial (0 = disabled)",
    )

    # Base Model & Hardware Parameters
    parser.add_argument(
        "--feature_size",
        type=int,
        default=48,
        help="Default feature_size if not specified in search grid",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to train on ('cuda' or 'cpu')",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=4,
        help="Number of dataloader background worker threads",
    )
    parser.add_argument("--seed", type=int, default=42, help="Master random seed")

    return parser.parse_args()


def generate_combinations(
    param_grid: dict[str, list[Any]],
    search_type: str = "grid",
    n_trials: int = 10,
    seed: int = 42,
) -> list[dict[str, Any]]:
    """
    Generates a list of hyperparameter parameter dictionaries either via
    exhaustive Cartesian grid or random sampling.
    """
    keys = list(param_grid.keys())
    values = [param_grid[k] if isinstance(param_grid[k], list) else [param_grid[k]] for k in keys]

    # Full Cartesian product
    all_combinations = [dict(zip(keys, combo)) for combo in itertools.product(*values)]

    if search_type == "grid":
        return all_combinations[:n_trials] if n_trials > 0 and len(all_combinations) > n_trials else all_combinations
    elif search_type == "random":
        rng = random.Random(seed)
        if len(all_combinations) <= n_trials:
            rng.shuffle(all_combinations)
            return all_combinations
        return rng.sample(all_combinations, n_trials)
    else:
        raise ValueError(f"Unknown search_type: {search_type}")


def print_leaderboard(trials_summary: list[dict[str, Any]], tuned_keys: list[str]) -> None:
    """Prints a formatted ASCII leaderboard ranking trials by Mean Dice Score."""
    print("\n" + "=" * 80)
    print("🏆 HYPERPARAMETER TUNING LEADERBOARD (Ranked by Validation Mean Dice)")
    print("=" * 80)

    # Filter successful trials
    successful = [t for t in trials_summary if t.get("status") == "COMPLETED"]
    failed = [t for t in trials_summary if t.get("status") == "FAILED"]

    successful.sort(key=lambda x: x.get("best_dice", -1.0), reverse=True)

    header = f"{'Rank':<5} | {'Trial ID':<10} | {'Mean Dice':<10} | {'HD95':<8} | {'Best Ep':<8} | "
    header += " | ".join([f"{k:<12}" for k in tuned_keys])
    print(header)
    print("-" * len(header))

    for rank, trial in enumerate(successful, start=1):
        dice_str = f"{trial.get('best_dice', 0.0):.4f}"
        hd95_val = trial.get("best_hd95", float("nan"))
        hd95_str = f"{hd95_val:.2f}" if not np.isnan(hd95_val) else "N/A"
        ep_str = str(trial.get("best_epoch", "N/A"))
        line = f"#{rank:<4} | {trial['trial_id']:<10} | {dice_str:<10} | {hd95_str:<8} | {ep_str:<8} | "
        params_str = " | ".join([f"{str(trial.get(k, '')): <12}" for k in tuned_keys])
        line += params_str
        print(line)

    if failed:
        print("\n⚠️ Failed Trials:")
        for t in failed:
            print(f"  - {t['trial_id']}: Error: {t.get('error', 'Unknown')}")

    print("=" * 80 + "\n")


def run_hyperparameter_tuning(
    base_config: dict[str, Any],
    param_grid: dict[str, list[Any]],
    search_type: str = "grid",
    n_trials: int = 10,
    seed: int = 42,
) -> dict[str, Any]:
    """
    Programmatic entry point for hyperparameter tuning.
    Iterates across parameter combinations, executes training runs, and compiles results.
    """
    output_dir = Path(base_config.get("output_dir", TRANSFORMER_DIR / "artifacts" / "tuning_experiments"))
    output_dir.mkdir(parents=True, exist_ok=True)
    trials_dir = output_dir / "trials"
    trials_dir.mkdir(parents=True, exist_ok=True)

    combinations = generate_combinations(
        param_grid=param_grid,
        search_type=search_type,
        n_trials=n_trials,
        seed=seed,
    )
    tuned_keys = list(param_grid.keys())

    print(f"\n========================================================")
    print(f"[TUNER] Starting Hyperparameter Tuning")
    print(f"[TUNER] Total Trials to Run : {len(combinations)}")
    print(f"[TUNER] Search Strategy     : {search_type.upper()}")
    print(f"[TUNER] Tuned Parameters    : {', '.join(tuned_keys)}")
    print(f"[TUNER] Output Directory    : {output_dir}")
    print(f"========================================================\n")

    trials_summary: list[dict[str, Any]] = []
    best_overall_dice = -1.0
    best_overall_trial: dict[str, Any] | None = None
    best_checkpoint_src: Path | None = None

    start_total_time = time.time()

    for idx, params in enumerate(combinations, start=1):
        trial_id = f"trial_{idx:03d}"
        trial_output_dir = trials_dir / trial_id
        trial_output_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n>>> Running [{idx}/{len(combinations)}] {trial_id} with parameters: {params}")

        # Assemble full configuration for this trial
        trial_cfg = base_config.copy()
        trial_cfg.update(params)
        trial_cfg["output_dir"] = str(trial_output_dir)
        trial_cfg["trial_id"] = trial_id

        # Save trial config
        with open(trial_output_dir / "trial_config.json", "w", encoding="utf-8") as f:
            json.dump(trial_cfg, f, indent=2)

        trial_start_time = time.time()
        try:
            # Free unused memory before each trial
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            result = run_training(trial_cfg)

            trial_duration = round(time.time() - trial_start_time, 2)
            best_dice = float(result.get("best_dice", -1.0))
            best_hd95 = float(result.get("best_hd95", float("nan")))
            best_epoch = int(result.get("best_epoch", 0))

            trial_record = {
                "trial_id": trial_id,
                "status": "COMPLETED",
                "best_dice": best_dice,
                "best_hd95": best_hd95,
                "best_epoch": best_epoch,
                "duration_sec": trial_duration,
                "trial_output_dir": str(trial_output_dir),
                **params,
            }

            if best_dice > best_overall_dice:
                best_overall_dice = best_dice
                best_overall_trial = trial_record
                checkpoint_file = Path(result["best_checkpoint"])
                if checkpoint_file.exists():
                    best_checkpoint_src = checkpoint_file

            print(f"[TUNER] {trial_id} Finished | Best Dice: {best_dice:.4f} | Time: {trial_duration}s")

        except Exception as e:
            trial_duration = round(time.time() - trial_start_time, 2)
            print(f"\n❌ [ERROR] {trial_id} failed with error: {e}")
            trial_record = {
                "trial_id": trial_id,
                "status": "FAILED",
                "best_dice": -1.0,
                "best_hd95": float("nan"),
                "best_epoch": 0,
                "duration_sec": trial_duration,
                "error": str(e),
                "trial_output_dir": str(trial_output_dir),
                **params,
            }
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        trials_summary.append(trial_record)

        # Write intermediate CSV and JSON summary after every single trial
        save_tuning_results(output_dir, trials_summary, tuned_keys)

    total_tuning_time = time.time() - start_total_time

    # Copy overall best checkpoint to root output_dir
    best_overall_checkpoint_path = output_dir / "best_model_overall.pt"
    if best_checkpoint_src and best_checkpoint_src.exists():
        shutil.copyfile(str(best_checkpoint_src), str(best_overall_checkpoint_path))
        print(f"\n[BEST] Copied overall top performing model to: {best_overall_checkpoint_path}")

    if best_overall_trial:
        with open(output_dir / "best_trial_config.json", "w", encoding="utf-8") as f:
            json.dump(best_overall_trial, f, indent=2)

    # Print summary leaderboard
    print_leaderboard(trials_summary, tuned_keys)

    print(f"[TUNER] Complete hyperparameter sweep finished in {total_tuning_time / 60.0:.2f} minutes.")
    if best_overall_trial:
        print(f"[TUNER] Optimal Configuration Found: {best_overall_trial['trial_id']} with Mean Dice = {best_overall_dice:.4f}")

    return {
        "trials": trials_summary,
        "best_trial": best_overall_trial,
        "best_dice": best_overall_dice,
        "summary_csv": str(output_dir / "tuning_summary.csv"),
        "summary_json": str(output_dir / "tuning_summary.json"),
        "best_model_overall": str(best_overall_checkpoint_path) if best_checkpoint_src else None,
    }


def save_tuning_results(
    output_dir: Path,
    trials_summary: list[dict[str, Any]],
    tuned_keys: list[str],
) -> None:
    """Serializes tuning summary to both JSON and CSV files."""
    json_path = output_dir / "tuning_summary.json"
    csv_path = output_dir / "tuning_summary.csv"

    # Save JSON
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(trials_summary, f, indent=2)

    # Save CSV
    if trials_summary:
        fieldnames = [
            "trial_id",
            "status",
            "best_dice",
            "best_hd95",
            "best_epoch",
            "duration_sec",
        ] + tuned_keys + ["trial_output_dir"]
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(trials_summary)


def main() -> None:
    args = parse_args()

    # Determine parameter grid
    param_grid: dict[str, list[Any]] = {}

    if args.param_grid:
        try:
            param_grid = json.loads(args.param_grid)
        except json.JSONDecodeError as err:
            sys.exit(f"Error: Invalid JSON for --param_grid: {err}")
    elif args.config_file and os.path.exists(args.config_file):
        with open(args.config_file, "r", encoding="utf-8") as f:
            param_grid = json.load(f)
    elif args.preset != "none" and args.preset in PRESET_GRIDS:
        param_grid = PRESET_GRIDS[args.preset]
        print(f"[TUNER] Using preset '{args.preset}' hyperparameter grid: {param_grid}")
    else:
        # Default fallback
        param_grid = PRESET_GRIDS["quick_test"]
        print(f"[TUNER] Defaulting to 'quick_test' grid: {param_grid}")

    base_config: dict[str, Any] = {
        "data_dir": args.data_dir,
        "output_dir": args.output_dir,
        "sanity_check": args.sanity_check,
        "epochs": args.epochs_per_trial,
        "batch_size": args.batch_size,
        "val_interval": args.val_interval,
        "early_stopping_patience": args.early_stopping_patience,
        "feature_size": args.feature_size,
        "device": args.device,
        "num_workers": args.num_workers,
        "seed": args.seed,
    }

    run_hyperparameter_tuning(
        base_config=base_config,
        param_grid=param_grid,
        search_type=args.search_type,
        n_trials=args.n_trials,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
