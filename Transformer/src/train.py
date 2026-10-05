"""
Phase 4: High-Performance Training & Validation Engine (SwinUNETR 3D Cardiac CT)
================================================================================
Features:
- PyTorch AMP (Automatic Mixed Precision) via torch.cuda.amp.autocast and GradScaler
- AdamW / Adam / SGD optimizer + CosineAnnealing / StepLR / ReduceLROnPlateau schedules
- Full volume sliding_window_inference validation (roi_size=(96, 96, 96), sw_batch_size=4, overlap=0.5)
- Multi-metric evaluation: Mean Dice Score, HD95 (Hausdorff Distance 95th Percentile), and Val Loss
- Checkpointing:
    * Best checkpoint saving to `model_best.pt`
    * Latest state checkpointing to `checkpoint_latest.pt` (full optimizer, scaler, scheduler, rng)
    * Periodic checkpointing via `--save_interval`
    * Full resume capability via `--resume`
- Evaluation metrics persistence:
    * `train_log.json` (Epoch-by-epoch history + summary)
    * `train_log.csv` (Spreadsheet-friendly metrics log)
    * `metrics_summary.json` (High-level benchmark metrics)
    * `config.json` (Serialized run configuration)
- Programmatic API: `run_training(config: dict | Namespace) -> dict[str, Any]` for direct
  integration into hyperparameter tuning sweeps.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from monai.inferers import sliding_window_inference
from monai.metrics import DiceMetric, HausdorffDistanceMetric
from monai.transforms import Activations, AsDiscrete, Compose
from monai.utils import set_determinism
from tqdm import tqdm

# Ensure parent and repo root are in sys.path for direct script execution
CURRENT_DIR = Path(__file__).resolve().parent
REPO_ROOT = CURRENT_DIR.parent.parent
TRANSFORMER_DIR = CURRENT_DIR.parent

for p in [str(REPO_ROOT), str(TRANSFORMER_DIR)]:
    if p not in sys.path:
        sys.path.insert(0, p)

try:
    from Transformer.src.dataset import (
        build_dataloader,
        build_dataset,
        generate_synthetic_nifti,
        get_transforms,
    )
    from Transformer.src.model import build_model
except ModuleNotFoundError:
    from src.dataset import (
        build_dataloader,
        build_dataset,
        generate_synthetic_nifti,
        get_transforms,
    )
    from src.model import build_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="SwinUNETR 3D Cardiac CT Artery Segmentation Training Engine"
    )
    # Data & Paths
    parser.add_argument(
        "--data_dir",
        type=str,
        default="",
        help="Path containing NIfTI image and label pairs",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=str(TRANSFORMER_DIR / "artifacts"),
        help="Directory to save training logs, checkpoints, and evaluation metrics",
    )
    parser.add_argument(
        "--val_split",
        type=float,
        default=0.2,
        help="Validation split ratio if data_dir is provided (default: 0.2)",
    )
    parser.add_argument(
        "--cache_rate",
        type=float,
        default=1.0,
        help="CacheDataset cache rate in memory (default: 1.0)",
    )
    parser.add_argument(
        "--sanity_check",
        action="store_true",
        help="Run quick sanity check with synthetic data (2 epochs)",
    )

    # Training Hyperparameters
    parser.add_argument("--epochs", type=int, default=100, help="Total training epochs")
    parser.add_argument("--batch_size", type=int, default=2, help="Batch size per training step")
    parser.add_argument("--lr", type=float, default=1e-4, help="Initial learning rate")
    parser.add_argument("--weight_decay", type=float, default=1e-5, help="Optimizer weight decay")
    parser.add_argument(
        "--optimizer",
        type=str,
        default="adamw",
        choices=["adamw", "adam", "sgd"],
        help="Optimizer type",
    )
    parser.add_argument(
        "--scheduler",
        type=str,
        default="cosine",
        choices=["cosine", "step", "plateau", "none"],
        help="Learning rate scheduler",
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
        help="Patience in validation evaluations before early stopping (0 = disabled)",
    )

    # Model Architecture Hyperparameters
    parser.add_argument(
        "--feature_size",
        type=int,
        default=48,
        help="SwinUNETR feature size (embedding dimension, e.g. 24, 36, 48)",
    )
    parser.add_argument(
        "--drop_rate",
        type=float,
        default=0.0,
        help="Dropout rate in SwinUNETR transformer blocks",
    )
    parser.add_argument(
        "--attn_drop_rate",
        type=float,
        default=0.0,
        help="Attention dropout rate in SwinUNETR window attention blocks",
    )
    parser.add_argument(
        "--dropout_path_rate",
        type=float,
        default=0.0,
        help="Drop-path stochastic depth rate",
    )
    parser.add_argument(
        "--use_checkpoint",
        action="store_true",
        default=True,
        help="Use activation checkpointing to save VRAM",
    )
    parser.add_argument(
        "--spatial_size",
        type=int,
        nargs=3,
        default=[96, 96, 96],
        help="Spatial patch size for training crops (D H W)",
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=4,
        help="Number of random sub-volume crops sampled per volume during training",
    )

    # Loss Hyperparameters
    parser.add_argument(
        "--lambda_dice",
        type=float,
        default=1.0,
        help="Weight for Dice component in DiceFocalLoss",
    )
    parser.add_argument(
        "--lambda_focal",
        type=float,
        default=0.5,
        help="Weight for Focal component in DiceFocalLoss",
    )
    parser.add_argument(
        "--gamma",
        type=float,
        default=2.0,
        help="Focal loss gamma focusing parameter for hard negative mining",
    )

    # Inference & Hardware Hyperparameters
    parser.add_argument(
        "--sw_batch_size",
        type=int,
        default=4,
        help="Sliding window inference batch size",
    )
    parser.add_argument(
        "--overlap",
        type=float,
        default=0.5,
        help="Sliding window inference overlap fraction",
    )
    parser.add_argument(
        "--amp",
        action="store_true",
        default=True,
        help="Use PyTorch Automatic Mixed Precision (AMP) when CUDA is available",
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
    parser.add_argument("--seed", type=int, default=42, help="Random seed for determinism")

    parser.add_argument(
        "--gpu_memory_fraction",
        type=float,
        default=1.0,
        help="Fraction of GPU VRAM allocated to this process (e.g. 0.48 to shard 1 GPU between 2 models)",
    )
    parser.add_argument(
        "--resume",
        type=str,
        default="",
        help="Path to checkpoint (.pt) to resume training from",
    )
    parser.add_argument(
        "--save_interval",
        type=int,
        default=0,
        help="Interval of epochs to save periodic checkpoints (0 = only best & latest)",
    )

    return parser.parse_args()


def train_one_epoch(
    model: nn.Module,
    train_loader: Any,
    optimizer: torch.optim.Optimizer,
    scaler: torch.cuda.amp.GradScaler | None,
    criterion: nn.Module,
    device: torch.device,
    use_amp: bool,
) -> float:
    """Runs a single training epoch with optional Automatic Mixed Precision (AMP)."""
    model.train()
    running_loss = 0.0
    step_count = 0

    pbar = tqdm(train_loader, desc="Training", leave=False)
    for batch in pbar:
        images = batch["image"].to(device, non_blocking=(device.type == "cuda"))
        labels = batch["label"].to(device, non_blocking=(device.type == "cuda"))

        optimizer.zero_grad(set_to_none=True)

        if use_amp and device.type == "cuda" and scaler is not None:
            with torch.cuda.amp.autocast(dtype=torch.float16):
                outputs = model(images)
                loss = criterion(outputs, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

        running_loss += float(loss.item())
        step_count += 1
        pbar.set_postfix({"loss": f"{loss.item():.4f}"})

    return running_loss / max(step_count, 1)


def validate(
    model: nn.Module,
    val_loader: Any,
    criterion: nn.Module,
    device: torch.device,
    dice_metric: DiceMetric,
    hd95_metric: HausdorffDistanceMetric,
    post_pred: Compose,
    post_label: Compose,
    roi_size: tuple[int, int, int] = (96, 96, 96),
    sw_batch_size: int = 4,
    overlap: float = 0.5,
    use_amp: bool = True,
) -> tuple[float, float, float]:
    """
    Sliding window validation for full-volume inference without CUDA OOM.
    Returns: (mean_dice, mean_hd95, mean_val_loss)
    """
    model.eval()
    dice_metric.reset()
    hd95_metric.reset()
    running_val_loss = 0.0
    val_steps = 0

    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Validation", leave=False):
            val_images = batch["image"].to(device, non_blocking=(device.type == "cuda"))
            val_labels = batch["label"].to(device, non_blocking=(device.type == "cuda"))

            if use_amp and device.type == "cuda":
                with torch.cuda.amp.autocast(dtype=torch.float16):
                    val_outputs = sliding_window_inference(
                        inputs=val_images,
                        roi_size=roi_size,
                        sw_batch_size=sw_batch_size,
                        predictor=model,
                        overlap=overlap,
                        mode="gaussian",
                    )
                    v_loss = criterion(val_outputs, val_labels)
            else:
                val_outputs = sliding_window_inference(
                    inputs=val_images,
                    roi_size=roi_size,
                    sw_batch_size=sw_batch_size,
                    predictor=model,
                    overlap=overlap,
                    mode="gaussian",
                )
                v_loss = criterion(val_outputs, val_labels)

            running_val_loss += float(v_loss.item())
            val_steps += 1

            # Discretize predictions and labels for metric computation
            val_outputs_discrete = [post_pred(i) for i in val_outputs]
            val_labels_discrete = [post_label(i) for i in val_labels]

            dice_metric(y_pred=val_outputs_discrete, y=val_labels_discrete)
            try:
                hd95_metric(y_pred=val_outputs_discrete, y=val_labels_discrete)
            except Exception:
                # If foreground target is absent in a slice
                pass

    try:
        mean_dice = float(dice_metric.aggregate().item())
    except Exception:
        mean_dice = 0.0

    try:
        mean_hd95 = float(hd95_metric.aggregate().item())
    except Exception:
        mean_hd95 = float("nan")

    mean_val_loss = running_val_loss / max(val_steps, 1)

    dice_metric.reset()
    hd95_metric.reset()

    return mean_dice, mean_hd95, mean_val_loss


def build_optimizer_and_scheduler(
    model: nn.Module,
    optimizer_name: str,
    lr: float,
    weight_decay: float,
    scheduler_name: str,
    total_epochs: int,
) -> tuple[torch.optim.Optimizer, Any]:
    """Factory creating optimizer and learning rate scheduler."""
    opt_lower = optimizer_name.lower()
    if opt_lower == "adamw":
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif opt_lower == "adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif opt_lower == "sgd":
        optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.99, weight_decay=weight_decay)
    else:
        raise ValueError(f"Unsupported optimizer: {optimizer_name}")

    sched_lower = scheduler_name.lower()
    if sched_lower == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_epochs, eta_min=1e-6)
    elif sched_lower == "step":
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=max(1, total_epochs // 3), gamma=0.5)
    elif sched_lower == "plateau":
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=5)
    elif sched_lower == "none":
        scheduler = None
    else:
        raise ValueError(f"Unsupported scheduler: {scheduler_name}")

    return optimizer, scheduler


def save_checkpoint(
    path: Path,
    epoch: int,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    scaler: torch.cuda.amp.GradScaler | None,
    best_dice: float,
    val_metrics: dict[str, float],
    config: dict[str, Any],
) -> None:
    """Serializes complete training state for checkpointing and safe resumption."""
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
        "scaler_state_dict": scaler.state_dict() if scaler is not None else None,
        "best_dice": best_dice,
        "val_metrics": val_metrics,
        "config": config,
    }
    torch.save(checkpoint, str(path))


def save_csv_log(csv_path: Path, history: list[dict[str, Any]]) -> None:
    """Writes tabular training history to CSV for charting and analysis."""
    if not history:
        return
    fieldnames = list(history[0].keys())
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(history)


def run_training(config: dict[str, Any] | argparse.Namespace) -> dict[str, Any]:
    """
    Main training execution function.
    Accepts either an argparse.Namespace or a dictionary, enabling easy invocation
    from command-line or from hyperparameter tuning sweeps.
    """
    if isinstance(config, argparse.Namespace):
        cfg = vars(config).copy()
    else:
        cfg = config.copy()

    # Default values for missing configuration keys
    cfg.setdefault("data_dir", "")
    cfg.setdefault("output_dir", str(TRANSFORMER_DIR / "artifacts"))
    cfg.setdefault("val_split", 0.2)
    cfg.setdefault("cache_rate", 1.0)
    cfg.setdefault("sanity_check", False)
    cfg.setdefault("epochs", 100)
    cfg.setdefault("batch_size", 2)
    cfg.setdefault("lr", 1e-4)
    cfg.setdefault("weight_decay", 1e-5)
    cfg.setdefault("optimizer", "adamw")
    cfg.setdefault("scheduler", "cosine")
    cfg.setdefault("val_interval", 1)
    cfg.setdefault("early_stopping_patience", 0)
    cfg.setdefault("feature_size", 48)
    cfg.setdefault("drop_rate", 0.0)
    cfg.setdefault("attn_drop_rate", 0.0)
    cfg.setdefault("dropout_path_rate", 0.0)
    cfg.setdefault("use_checkpoint", True)
    cfg.setdefault("spatial_size", (96, 96, 96))
    cfg.setdefault("num_samples", 4)
    cfg.setdefault("lambda_dice", 1.0)
    cfg.setdefault("lambda_focal", 0.5)
    cfg.setdefault("gamma", 2.0)
    cfg.setdefault("sw_batch_size", 4)
    cfg.setdefault("overlap", 0.5)
    cfg.setdefault("amp", True)
    cfg.setdefault("device", "cuda" if torch.cuda.is_available() else "cpu")
    cfg.setdefault("num_workers", 4)
    cfg.setdefault("seed", 42)
    cfg.setdefault("resume", "")
    cfg.setdefault("save_interval", 0)
    cfg.setdefault("gpu_memory_fraction", 1.0)

    # Normalize spatial_size to tuple
    if isinstance(cfg["spatial_size"], (list, tuple)):
        cfg["spatial_size"] = tuple(cfg["spatial_size"])

    set_determinism(seed=int(cfg["seed"]))

    output_dir = Path(cfg["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save run configuration
    config_file = output_dir / "config.json"
    with open(config_file, "w", encoding="utf-8") as f:
        # Convert non-serializable objects to string or list
        serializable_cfg = {}
        for k, v in cfg.items():
            if isinstance(v, (tuple, list)):
                serializable_cfg[k] = list(v)
            elif isinstance(v, (int, float, str, bool)) or v is None:
                serializable_cfg[k] = v
            else:
                serializable_cfg[k] = str(v)
        json.dump(serializable_cfg, f, indent=2)

    best_checkpoint_path = output_dir / "model_best.pt"
    latest_checkpoint_path = output_dir / "checkpoint_latest.pt"
    log_json_path = output_dir / "train_log.json"
    log_csv_path = output_dir / "train_log.csv"
    summary_metrics_path = output_dir / "metrics_summary.json"

    device = torch.device(cfg["device"])
    print(f"\n========================================================")
    print(f"[ENGINE] Target Device : {device} ({'CUDA GPU' if device.type == 'cuda' else 'CPU'})")
    print(f"[ENGINE] Output Dir    : {output_dir}")
    print(f"[ENGINE] Architecture  : SwinUNETR (feature_size={cfg['feature_size']})")
    print(f"[ENGINE] Optimization  : {cfg['optimizer']} (lr={cfg['lr']}, weight_decay={cfg['weight_decay']})")
    print(f"========================================================")

    if device.type == "cuda" and torch.cuda.is_available():
        gpu_id = device.index if device.index is not None else 0
        torch.cuda.set_device(gpu_id)
        mem_frac = float(cfg.get("gpu_memory_fraction", 1.0))
        if 0.0 < mem_frac < 1.0:
            torch.cuda.set_per_process_memory_fraction(mem_frac, gpu_id)
            print(f"[SHARDING] Constrained process VRAM to {mem_frac * 100:.1f}% on cuda:{gpu_id}")

        torch.backends.cudnn.benchmark = True
        print(f"[CUDA] Device Name   : {torch.cuda.get_device_name(gpu_id)}")
        print(f"[CUDA] Memory Total  : {torch.cuda.get_device_properties(gpu_id).total_memory / (1024**3):.2f} GB")

    # Data Preparation
    if cfg["sanity_check"] or not cfg["data_dir"]:
        print("[SETUP] Preparing synthetic cardiac CT scans for training/validation...")
        synth_dir = output_dir / "synthetic_cache"
        case_1 = generate_synthetic_nifti(synth_dir, prefix="case_synth_01")
        case_2 = generate_synthetic_nifti(synth_dir, prefix="case_synth_02")
        train_files = [case_1]
        val_files = [case_2]
        total_epochs = 2 if cfg["sanity_check"] else int(cfg["epochs"])
    else:
        data_path = Path(cfg["data_dir"])
        images = sorted(list(data_path.glob("*_img.nii*")) + list(data_path.glob("*_image.nii*")))
        labels = sorted(list(data_path.glob("*_label.nii*")) + list(data_path.glob("*_seg.nii*")))
        if not images:
            raise FileNotFoundError(f"No image files found in {data_path} matching '*_img.nii*' or '*_image.nii*'")

        all_files = [{"image": str(img), "label": str(lbl)} for img, lbl in zip(images, labels)]
        split_idx = max(1, int(len(all_files) * (1.0 - float(cfg["val_split"]))))
        train_files = all_files[:split_idx]
        val_files = all_files[split_idx:] if len(all_files) > 1 else all_files
        total_epochs = int(cfg["epochs"])

    print(f"[DATA] Train Volumes: {len(train_files)} | Val Volumes: {len(val_files)}")

    # Transforms & Datasets
    train_transforms = get_transforms(
        mode="train",
        spatial_size=cfg["spatial_size"],
        num_samples=int(cfg["num_samples"]),
    )
    val_transforms = get_transforms(mode="val")

    num_workers = int(cfg["num_workers"]) if device.type == "cuda" else 0
    train_ds = build_dataset(
        train_files,
        train_transforms,
        use_cache=True,
        cache_rate=float(cfg["cache_rate"]),
        num_workers=num_workers,
    )
    val_ds = build_dataset(
        val_files,
        val_transforms,
        use_cache=True,
        cache_rate=float(cfg["cache_rate"]),
        num_workers=num_workers,
    )

    train_loader = build_dataloader(
        train_ds,
        batch_size=int(cfg["batch_size"]),
        shuffle=True,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
    )
    val_loader = build_dataloader(
        val_ds,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        pin_memory=(device.type == "cuda"),
    )

    # Model & Criterion Construction
    model, criterion = build_model(
        device=device,
        img_size=cfg["spatial_size"],
        feature_size=int(cfg["feature_size"]),
        use_checkpoint=bool(cfg["use_checkpoint"]),
        drop_rate=float(cfg["drop_rate"]),
        attn_drop_rate=float(cfg["attn_drop_rate"]),
        dropout_path_rate=float(cfg["dropout_path_rate"]),
        lambda_dice=float(cfg["lambda_dice"]),
        lambda_focal=float(cfg["lambda_focal"]),
        gamma=float(cfg["gamma"]),
    )

    # Optimizer & Scheduler
    optimizer, scheduler = build_optimizer_and_scheduler(
        model=model,
        optimizer_name=cfg["optimizer"],
        lr=float(cfg["lr"]),
        weight_decay=float(cfg["weight_decay"]),
        scheduler_name=cfg["scheduler"],
        total_epochs=total_epochs,
    )

    # Mixed Precision Scaler
    use_amp = bool(cfg["amp"]) and (device.type == "cuda")
    scaler = torch.cuda.amp.GradScaler() if use_amp else None

    # Metrics & Post-processing
    dice_metric = DiceMetric(include_background=False, reduction="mean")
    hd95_metric = HausdorffDistanceMetric(percentile=95, include_background=False, reduction="mean")
    post_pred = Compose([Activations(sigmoid=True), AsDiscrete(threshold=0.5)])
    post_label = Compose([AsDiscrete(threshold=0.5)])

    start_epoch = 1
    best_dice = -1.0
    best_hd95 = float("nan")
    best_epoch = 0
    history: list[dict[str, Any]] = []

    # Resume from checkpoint if specified
    if cfg["resume"] and os.path.exists(cfg["resume"]):
        resume_path = Path(cfg["resume"])
        print(f"[RESUME] Loading checkpoint from: {resume_path}")
        checkpoint = torch.load(str(resume_path), map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        if "optimizer_state_dict" in checkpoint and checkpoint["optimizer_state_dict"]:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if "scheduler_state_dict" in checkpoint and checkpoint["scheduler_state_dict"] and scheduler:
            scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        if scaler and "scaler_state_dict" in checkpoint and checkpoint["scaler_state_dict"]:
            scaler.load_state_dict(checkpoint["scaler_state_dict"])

        start_epoch = checkpoint.get("epoch", 0) + 1
        best_dice = checkpoint.get("best_dice", -1.0)
        best_epoch = checkpoint.get("epoch", 0)
        if log_json_path.exists():
            try:
                with open(log_json_path, "r", encoding="utf-8") as f:
                    prev_data = json.load(f)
                    history = prev_data.get("history", [])
            except Exception:
                pass
        print(f"[RESUME] Resuming from epoch {start_epoch} with prior best Dice: {best_dice:.4f}")

    print(f"[TRAIN] Beginning training for {total_epochs} epochs...")
    start_time = time.time()
    epochs_no_improve = 0
    patience = int(cfg["early_stopping_patience"])

    for epoch in range(start_epoch, total_epochs + 1):
        epoch_start = time.time()
        train_loss = train_one_epoch(
            model=model,
            train_loader=train_loader,
            optimizer=optimizer,
            scaler=scaler,
            criterion=criterion,
            device=device,
            use_amp=use_amp,
        )

        val_dice, val_hd95, val_loss = float("nan"), float("nan"), float("nan")
        val_evaluated = (epoch % int(cfg["val_interval"]) == 0) or (epoch == total_epochs)

        if val_evaluated:
            val_dice, val_hd95, val_loss = validate(
                model=model,
                val_loader=val_loader,
                criterion=criterion,
                device=device,
                dice_metric=dice_metric,
                hd95_metric=hd95_metric,
                post_pred=post_pred,
                post_label=post_label,
                roi_size=cfg["spatial_size"],
                sw_batch_size=int(cfg["sw_batch_size"]),
                overlap=float(cfg["overlap"]),
                use_amp=use_amp,
            )

            # Scheduler step for plateau on validation dice
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(val_dice)
            elif scheduler is not None:
                scheduler.step()

            # Check if this is the best model so far
            if val_dice > best_dice:
                best_dice = val_dice
                best_hd95 = val_hd95
                best_epoch = epoch
                epochs_no_improve = 0

                save_checkpoint(
                    path=best_checkpoint_path,
                    epoch=epoch,
                    model=model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    scaler=scaler,
                    best_dice=best_dice,
                    val_metrics={"val_dice": val_dice, "val_hd95": val_hd95, "val_loss": val_loss},
                    config=cfg,
                )
                print(
                    f"  >>> [BEST CHECKPOINT] Epoch {epoch:03d}: Mean Dice = {best_dice:.4f} "
                    f"| HD95 = {best_hd95:.2f} -> Saved {best_checkpoint_path.name}"
                )
            else:
                epochs_no_improve += 1
        else:
            if scheduler is not None and not isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step()

        # Always save latest state for resume resilience
        save_checkpoint(
            path=latest_checkpoint_path,
            epoch=epoch,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            best_dice=best_dice,
            val_metrics={"val_dice": val_dice, "val_hd95": val_hd95, "val_loss": val_loss},
            config=cfg,
        )

        # Periodic checkpointing if requested
        if int(cfg["save_interval"]) > 0 and (epoch % int(cfg["save_interval"]) == 0):
            periodic_path = output_dir / f"checkpoint_epoch_{epoch:03d}.pt"
            save_checkpoint(
                path=periodic_path,
                epoch=epoch,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                scaler=scaler,
                best_dice=best_dice,
                val_metrics={"val_dice": val_dice, "val_hd95": val_hd95, "val_loss": val_loss},
                config=cfg,
            )

        epoch_duration = time.time() - epoch_start
        current_lr = optimizer.param_groups[0]["lr"]

        epoch_info = {
            "epoch": epoch,
            "train_loss": round(train_loss, 5),
            "val_loss": round(val_loss, 5) if not np.isnan(val_loss) else None,
            "val_dice": round(val_dice, 5) if not np.isnan(val_dice) else None,
            "val_hd95": round(val_hd95, 3) if not np.isnan(val_hd95) else None,
            "lr": current_lr,
            "duration_sec": round(epoch_duration, 2),
        }
        history.append(epoch_info)

        # Save continuous training logs after each epoch
        with open(log_json_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "best_dice": best_dice,
                    "best_hd95": best_hd95,
                    "best_epoch": best_epoch,
                    "history": history,
                },
                f,
                indent=2,
            )
        save_csv_log(log_csv_path, history)

        print(
            f"Epoch [{epoch:03d}/{total_epochs:03d}] "
            f"Train Loss: {train_loss:.4f} | "
            f"Val Loss: {val_loss:.4f} | "
            f"Val Dice: {val_dice:.4f} | "
            f"HD95: {val_hd95:.2f} | "
            f"LR: {current_lr:.6f} | "
            f"Time: {epoch_duration:.1f}s"
        )

        # Early stopping condition
        if patience > 0 and epochs_no_improve >= patience:
            print(f"\n[EARLY STOPPING] Validation Dice did not improve for {patience} checks. Stopping.")
            break

    total_time = time.time() - start_time
    print(f"\n[DONE] Finished {epoch} epochs in {total_time / 60.0:.2f} minutes.")
    print(f"[SUMMARY] Best Validation Dice: {best_dice:.4f} (at Epoch {best_epoch})")

    summary_metrics = {
        "best_dice": best_dice,
        "best_hd95": best_hd95,
        "best_epoch": best_epoch,
        "total_epochs": epoch,
        "total_duration_sec": round(total_time, 2),
        "best_checkpoint": str(best_checkpoint_path),
        "latest_checkpoint": str(latest_checkpoint_path),
        "train_log_json": str(log_json_path),
        "train_log_csv": str(log_csv_path),
        "config_json": str(config_file),
    }

    with open(summary_metrics_path, "w", encoding="utf-8") as f:
        json.dump(summary_metrics, f, indent=2)

    print(f"[METRICS] Saved metrics summary to {summary_metrics_path}")
    print(f"[METRICS] Saved tabular CSV log to {log_csv_path}")

    return {
        "best_dice": best_dice,
        "best_hd95": best_hd95,
        "best_epoch": best_epoch,
        "summary": summary_metrics,
        "output_dir": str(output_dir),
        "best_checkpoint": str(best_checkpoint_path),
    }


def main() -> None:
    args = parse_args()
    run_training(args)


if __name__ == "__main__":
    main()
