"""
GPU-Sharded Parallel Hyperparameter Tuning Engine for 3D SwinUNETR
==================================================================
Features:
- Concurrent Multi-Model Training (GPU Sharding):
    * Multi-GPU Sharding: Automatically distributes parallel model runs across
      distinct GPUs (e.g., Model 1 on `cuda:0`, Model 2 on `cuda:1`).
    * Single-GPU VRAM Sharding: Partitions a single GPU across concurrent models
      using `torch.cuda.set_per_process_memory_fraction` (e.g., 48% VRAM each)
      so two models can train simultaneously without CUDA OOM.
    * Dynamic GPU Slot Pool: Checks out GPU devices when trials start and immediately
      recycles them to the next pending trial as soon as one completes.
- Supports Model Architecture parameters:
    * `feature_size`: SwinUNETR embedding dimension (e.g., 24, 36, 48)
    * `drop_rate`: Transformer block dropout
    * `attn_drop_rate`: Window attention dropout
    * `dropout_path_rate`: Stochastic depth rate
    * `use_checkpoint`: Activation checkpointing
- Supports Loss parameters:
    * `lambda_dice`, `lambda_focal`, `gamma` (Focal loss modulation)
- Supports Training & Optimization parameters:
    * `lr`, `weight_decay`, `optimizer` (AdamW, Adam, SGD), `scheduler`, `batch_size`
- Search Strategies:
    * `grid`: Exhaustive Cartesian product of candidate parameter values
    * `random`: Random sampling of hyperparameter combinations for N trials
    * Predefined presets: `--preset model_arch`, `--preset loss_tuning`, `--preset opt_tuning`, `--preset quick_test`
- Metrics & Checkpoint Persistence:
    * Independent trial folders: `output_dir/trials/trial_001/` with trial checkpoints & logs
    * Real-time saving of `tuning_summary.csv` and `tuning_summary.json`
    * `best_trial_config.json` and `best_model_overall.pt` copied to base output directory
    * Formatted ASCII Leaderboard printed upon completion
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import os
import queue
import random
import shutil
import sys
import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.multiprocessing as mp

def sanitize_mps_environment() -> None:
    """
    Guards against NVIDIA MPS Error 805 ('MPS client failed to connect to the MPS control daemon').
    If CUDA_MPS_PIPE_DIRECTORY is set in environment but the MPS daemon is not actually running,
    unsetting it prevents CUDA calls from failing and falling back to CPU.
    """
    if "CUDA_MPS_PIPE_DIRECTORY" in os.environ:
        mps_dir = Path(os.environ["CUDA_MPS_PIPE_DIRECTORY"])
        control_pipe = mps_dir / "control"
        if not mps_dir.exists() or not control_pipe.exists():
            print(f"[NOTICE] CUDA_MPS_PIPE_DIRECTORY='{mps_dir}' was set, but no active MPS control daemon was found.")
            print("[NOTICE] Automatically unsetting CUDA_MPS_PIPE_DIRECTORY so CUDA can access GPU(s) directly.")
            os.environ.pop("CUDA_MPS_PIPE_DIRECTORY", None)
            os.environ.pop("CUDA_MPS_LOG_DIRECTORY", None)


sanitize_mps_environment()

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
        description="Automated GPU-Sharded Hyperparameter Tuning Engine for SwinUNETR Segmentation"
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

    # GPU Sharding & Concurrency Controls
    parser.add_argument(
        "--max_parallel_jobs",
        type=int,
        default=2,
        help="Number of models to train simultaneously in parallel (default: 2)",
    )
    parser.add_argument(
        "--devices",
        type=str,
        nargs="+",
        default=None,
        help="Explicit list of devices for workers (e.g. --devices cuda:0 cuda:1 or cuda:0 cuda:0)",
    )
    parser.add_argument(
        "--gpu_memory_fraction",
        type=float,
        default=0.48,
        help="VRAM fraction limit when multiple models share a single GPU (default: 0.48)",
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
        help="Default fallback device ('cuda' or 'cpu')",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=2,
        help="Number of dataloader background worker threads per model",
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


def resolve_device_pool(
    max_parallel_jobs: int,
    user_devices: list[str] | None = None,
    default_device: str = "cuda",
) -> list[tuple[str, str]]:
    """
    Resolves the pool of worker devices and assigns virtual shard labels.
    Examples for single GPU:
      - [('cuda:0', 'vGPU-0'), ('cuda:0', 'vGPU-1')]
    Examples for multi-GPU:
      - [('cuda:0', 'GPU-0'), ('cuda:1', 'GPU-1')]
    """
    if user_devices:
        return [(d, f"vGPU-{i}" if user_devices.count(d) > 1 else f"GPU-{i}") for i, d in enumerate(user_devices)]

    sanitize_mps_environment()
    cuda_available = torch.cuda.is_available()
    num_cuda_gpus = torch.cuda.device_count() if cuda_available else 0

    if default_device.startswith("cuda"):
        if not cuda_available:
            raise RuntimeError(
                f"[DEVICE ERROR] CUDA was explicitly requested (device='{default_device}'), "
                "but PyTorch reported that CUDA is NOT available on this machine!\n"
                "Reason: Likely the CUDA initialization error shown above (e.g. Error 805: NVIDIA MPS daemon not running).\n"
                "Please fix the CUDA environment (e.g. run 'unset CUDA_MPS_PIPE_DIRECTORY && rm -rf /tmp/nvidia-mps') "
                "or explicitly configure device='cpu' if you intended to train on CPU."
            )
        if num_cuda_gpus >= max_parallel_jobs:
            # Distribute across distinct physical GPUs
            return [(f"cuda:{i}", f"GPU-{i}") for i in range(max_parallel_jobs)]
        elif num_cuda_gpus > 0:
            # Single or fewer physical GPUs -> shard into virtual GPUs (vGPU-0, vGPU-1, ...)
            return [(f"cuda:{i % num_cuda_gpus}", f"vGPU-{i}") for i in range(max_parallel_jobs)]

    # Explicitly requested CPU
    return [("cpu", f"CPU-{i}") for i in range(max_parallel_jobs)]


def _isolated_worker_target(
    trial_cfg: dict[str, Any],
    device_str: str,
    virtual_shard_name: str,
    gpu_memory_fraction: float,
    result_queue: Any,
) -> None:
    """
    Executed inside a standalone spawned process for maximum CUDA & VRAM isolation.
    """
    sanitize_mps_environment()
    # Prevent CUDA caching allocator fragmentation when multiple models share one GPU
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

    trial_cfg["device"] = device_str
    trial_cfg["gpu_memory_fraction"] = gpu_memory_fraction
    trial_cfg["virtual_shard"] = virtual_shard_name

    # Set CUDA device context cleanly for this child process
    if device_str.startswith("cuda") and torch.cuda.is_available():
        gpu_id = int(device_str.split(":")[-1]) if ":" in device_str else 0
        try:
            torch.cuda.set_device(gpu_id)
            if 0.0 < gpu_memory_fraction < 1.0:
                torch.cuda.set_per_process_memory_fraction(gpu_memory_fraction, gpu_id)
        except Exception as e:
            print(f"[WARN] Failed to set per-process memory fraction on {device_str}: {e}")

    try:
        train_result = run_training(trial_cfg)
        result_queue.put({"success": True, "result": train_result, "error": None})
    except Exception as e:
        import traceback
        err_msg = f"{e}\n{traceback.format_exc()}"
        result_queue.put({"success": False, "result": None, "error": err_msg})


def execute_trial_process(
    trial_cfg: dict[str, Any],
    device_str: str,
    virtual_shard_name: str,
    gpu_memory_fraction: float,
) -> dict[str, Any]:
    """
    Spawns a clean process using PyTorch's spawn context to execute a single trial
    on its allocated GPU shard.
    """
    mp_ctx = mp.get_context("spawn")
    result_q = mp_ctx.Queue()

    proc = mp_ctx.Process(
        target=_isolated_worker_target,
        args=(trial_cfg, device_str, virtual_shard_name, gpu_memory_fraction, result_q),
    )
    proc.start()
    proc.join()

    try:
        if not result_q.empty():
            data = result_q.get()
            if data["success"]:
                return data["result"]
            else:
                raise RuntimeError(data["error"])
        else:
            raise RuntimeError(f"Trial process for {trial_cfg.get('trial_id')} exited unexpectedly (code: {proc.exitcode})")
    finally:
        result_q.close()
        result_q.join_thread()


def print_leaderboard(trials_summary: list[dict[str, Any]], tuned_keys: list[str]) -> None:
    """Prints a formatted ASCII leaderboard ranking trials by Mean Dice Score."""
    print("\n" + "=" * 85)
    print("🏆 HYPERPARAMETER TUNING LEADERBOARD (Ranked by Validation Mean Dice)")
    print("=" * 85)

    successful = [t for t in trials_summary if t.get("status") == "COMPLETED"]
    failed = [t for t in trials_summary if t.get("status") == "FAILED"]

    successful.sort(key=lambda x: x.get("best_dice", -1.0), reverse=True)

    header = f"{'Rank':<5} | {'Trial ID':<10} | {'Device':<8} | {'Mean Dice':<10} | {'HD95':<8} | {'Best Ep':<8} | "
    header += " | ".join([f"{k:<12}" for k in tuned_keys])
    print(header)
    print("-" * len(header))

    for rank, trial in enumerate(successful, start=1):
        dice_str = f"{trial.get('best_dice', 0.0):.4f}"
        hd95_val = trial.get("best_hd95", float("nan"))
        hd95_str = f"{hd95_val:.2f}" if not np.isnan(hd95_val) else "N/A"
        ep_str = str(trial.get("best_epoch", "N/A"))
        dev_str = str(trial.get("device", "N/A"))
        line = f"#{rank:<4} | {trial['trial_id']:<10} | {dev_str:<8} | {dice_str:<10} | {hd95_str:<8} | {ep_str:<8} | "
        params_str = " | ".join([f"{str(trial.get(k, '')): <12}" for k in tuned_keys])
        line += params_str
        print(line)

    if failed:
        print("\n⚠️ Failed Trials:")
        for t in failed:
            print(f"  - {t['trial_id']} ({t.get('device')}): Error: {t.get('error', 'Unknown')}")

    print("=" * 85 + "\n")


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
            "device",
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


def run_hyperparameter_tuning(
    base_config: dict[str, Any],
    param_grid: dict[str, list[Any]],
    search_type: str = "grid",
    n_trials: int = 10,
    max_parallel_jobs: int = 2,
    devices: list[str] | None = None,
    gpu_memory_fraction: float = 0.48,
    seed: int = 42,
) -> dict[str, Any]:
    """
    GPU-Sharded Concurrent Hyperparameter Tuning Engine.
    Executes multiple trials in parallel across allocated GPU shards.
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

    # Resolve device slot pool (e.g. ['cuda:0', 'cuda:1'] or ['cuda:0', 'cuda:0'])
    default_dev = base_config.get("device", "cuda" if torch.cuda.is_available() else "cpu")
    device_pool_list = resolve_device_pool(
        max_parallel_jobs=max_parallel_jobs,
        user_devices=devices,
        default_device=default_dev,
    )

    # Device slot queue for dynamic slot checkout & recycling
    device_slot_queue: queue.Queue[tuple[str, str]] = queue.Queue()
    for item in device_pool_list:
        device_slot_queue.put(item)

    # If single GPU is sharded into virtual GPUs, ensure memory fraction is applied
    distinct_phys_devices = set(d[0] for d in device_pool_list)
    needs_vram_sharding = (
        len(device_pool_list) > 1
        and len(distinct_phys_devices) < len(device_pool_list)
        and any(d[0].startswith("cuda") for d in device_pool_list)
    )
    effective_mem_fraction = gpu_memory_fraction if needs_vram_sharding else 1.0

    print(f"\n========================================================")
    print(f"[TUNER] GPU-Sharded Parallel Hyperparameter Tuning Engine")
    print(f"[TUNER] Total Trials to Run   : {len(combinations)}")
    print(f"[TUNER] Max Concurrent Models : {max_parallel_jobs}")
    print(f"[TUNER] Virtual Shards Pool   : {[f'{d} ({s})' for d, s in device_pool_list]}")
    if needs_vram_sharding:
        print(f"[TUNER] Single-GPU Sharding   : ACTIVE (VRAM Limit: {effective_mem_fraction * 100:.1f}% per model)")
    print(f"[TUNER] Search Strategy       : {search_type.upper()}")
    print(f"[TUNER] Tuned Parameters      : {', '.join(tuned_keys)}")
    print(f"[TUNER] Base Output Directory : {output_dir}")
    print(f"========================================================\n")

    trials_summary: list[dict[str, Any]] = []
    trials_lock = threading.Lock()
    best_overall_dice = -1.0
    best_overall_trial: dict[str, Any] | None = None
    best_checkpoint_src: Path | None = None

    start_total_time = time.time()

    def _run_single_trial(idx: int, params: dict[str, Any]) -> dict[str, Any]:
        nonlocal best_overall_dice, best_overall_trial, best_checkpoint_src

        # Acquire an available device and virtual shard slot
        allocated_device, shard_name = device_slot_queue.get()
        slot_label = f"{allocated_device} ({shard_name})"
        trial_id = f"trial_{idx:03d}"
        trial_output_dir = trials_dir / trial_id
        trial_output_dir.mkdir(parents=True, exist_ok=True)

        print(f"[DISPATCH] Starting [{idx}/{len(combinations)}] {trial_id} on {slot_label} with: {params}")

        trial_cfg = base_config.copy()
        trial_cfg.update(params)
        trial_cfg["output_dir"] = str(trial_output_dir)
        trial_cfg["trial_id"] = trial_id

        with open(trial_output_dir / "trial_config.json", "w", encoding="utf-8") as f:
            json.dump(trial_cfg, f, indent=2)

        trial_start_time = time.time()
        try:
            result = execute_trial_process(
                trial_cfg=trial_cfg,
                device_str=allocated_device,
                virtual_shard_name=shard_name,
                gpu_memory_fraction=effective_mem_fraction,
            )

            trial_duration = round(time.time() - trial_start_time, 2)
            best_dice = float(result.get("best_dice", -1.0))
            best_hd95 = float(result.get("best_hd95", float("nan")))
            best_epoch = int(result.get("best_epoch", 0))

            trial_record = {
                "trial_id": trial_id,
                "device": slot_label,
                "status": "COMPLETED",
                "best_dice": best_dice,
                "best_hd95": best_hd95,
                "best_epoch": best_epoch,
                "duration_sec": trial_duration,
                "trial_output_dir": str(trial_output_dir),
                **params,
            }

            with trials_lock:
                if best_dice > best_overall_dice:
                    best_overall_dice = best_dice
                    best_overall_trial = trial_record
                    ckpt_file = Path(result["best_checkpoint"])
                    if ckpt_file.exists():
                        best_checkpoint_src = ckpt_file

            print(f"[COMPLETE] {trial_id} on {slot_label} Finished | Mean Dice: {best_dice:.4f} | Time: {trial_duration}s")

        except Exception as e:
            trial_duration = round(time.time() - trial_start_time, 2)
            err_str = str(e)
            print(f"\n❌ [ERROR] {trial_id} on {slot_label} failed: {err_str}")
            if "busy or unavailable" in err_str.lower() or "device busy" in err_str.lower():
                print("\n💡 [DIAGNOSIS] CUDA device is reported busy or unavailable!")
                print("   Your GPU is configured in 'EXCLUSIVE_PROCESS' compute mode, which prevents")
                print("   multiple processes from binding to the same physical GPU simultaneously.")
                print("   Quick Solutions:")
                print("   1. Set max_parallel_jobs=1 in Transformer/run.py to train trials sequentially.")
                print("   2. Or if you have sudo: sudo nvidia-smi -c DEFAULT to allow concurrent access.")
                print("   3. Or if your workstation has multiple GPUs, assign distinct GPUs via devices=['cuda:0', 'cuda:1']")
                print("   4. Or start the NVIDIA MPS daemon to multiplex the GPU: nvidia-cuda-mps-control -d\n")

            trial_record = {
                "trial_id": trial_id,
                "device": slot_label,
                "status": "FAILED",
                "best_dice": -1.0,
                "best_hd95": float("nan"),
                "best_epoch": 0,
                "duration_sec": trial_duration,
                "error": err_str,
                "trial_output_dir": str(trial_output_dir),
                **params,
            }
        finally:
            # Recycle the virtual shard slot back to the queue immediately for waiting trials
            device_slot_queue.put((allocated_device, shard_name))

        with trials_lock:
            trials_summary.append(trial_record)
            save_tuning_results(output_dir, trials_summary, tuned_keys)

        return trial_record

    # Run trials concurrently across the worker pool
    with ThreadPoolExecutor(max_workers=max_parallel_jobs) as executor:
        futures = [
            executor.submit(_run_single_trial, idx, params)
            for idx, params in enumerate(combinations, start=1)
        ]
        for fut in as_completed(futures):
            try:
                fut.result()
            except Exception as e:
                print(f"[FATAL] Worker exception: {e}")

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

    print(f"[TUNER] All parallel trials completed in {total_tuning_time / 60.0:.2f} minutes.")
    if best_overall_trial:
        print(
            f"[TUNER] Winning Trial: {best_overall_trial['trial_id']} "
            f"with Mean Dice = {best_overall_dice:.4f} on {best_overall_trial.get('device')}"
        )

    return {
        "trials": trials_summary,
        "best_trial": best_overall_trial,
        "best_dice": best_overall_dice,
        "summary_csv": str(output_dir / "tuning_summary.csv"),
        "summary_json": str(output_dir / "tuning_summary.json"),
        "best_model_overall": str(best_overall_checkpoint_path) if best_checkpoint_src else None,
    }


def main() -> None:
    args = parse_args()

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
        max_parallel_jobs=args.max_parallel_jobs,
        devices=args.devices,
        gpu_memory_fraction=args.gpu_memory_fraction,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
