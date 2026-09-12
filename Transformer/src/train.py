"""
Phase 4: Training & Validation Engine (x86_64 + NVIDIA CUDA)
============================================================
Features:
- PyTorch AMP (Automatic Mixed Precision) via torch.cuda.amp.autocast and GradScaler
- AdamW optimizer (lr=1e-4) + CosineAnnealingLR schedule
- sliding_window_inference validation (roi_size=(96, 96, 96), sw_batch_size=4, overlap=0.5)
- Metrics: Mean Dice Score and HD95 (Hausdorff Distance 95th Percentile)
- Best checkpoint saving to artifacts/model_best.pt
"""

from __future__ import annotations

import argparse
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

from Transformer.src.dataset import (
    build_dataloader,
    build_dataset,
    generate_synthetic_nifti,
    get_transforms,
)
from Transformer.src.model import build_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="SwinUNETR 3D Cardiac CT Artery Segmentation Training Engine (CUDA)"
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="",
        help="Path containing NIfTI image and label pairs",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=str(Path(__file__).resolve().parents[1] / "artifacts"),
        help="Directory to save training logs and best checkpoints",
    )
    parser.add_argument("--epochs", type=int, default=100, help="Total training epochs")
    parser.add_argument("--batch_size", type=int, default=2, help="Batch size per training step")
    parser.add_argument("--val_interval", type=int, default=1, help="Epoch interval for validation")
    parser.add_argument("--lr", type=float, default=1e-4, help="AdamW initial learning rate")
    parser.add_argument("--weight_decay", type=float, default=1e-5, help="AdamW weight decay")
    parser.add_argument("--feature_size", type=int, default=48, help="SwinUNETR feature size")
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
    parser.add_argument(
        "--sanity_check",
        action="store_true",
        help="Run quick sanity check with synthetic data (2 epochs)",
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
    """Runs single training epoch with Automatic Mixed Precision (AMP)."""
    model.train()
    running_loss = 0.0
    step_count = 0

    pbar = tqdm(train_loader, desc="Training", leave=False)
    for batch in pbar:
        # Transfer batch to CUDA
        images = batch["image"].to(device, non_blocking=True)
        labels = batch["label"].to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        if use_amp and device.type == "cuda":
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

        running_loss += loss.item()
        step_count += 1
        pbar.set_postfix({"loss": f"{loss.item():.4f}"})

    return running_loss / max(step_count, 1)


def validate(
    model: nn.Module,
    val_loader: Any,
    device: torch.device,
    dice_metric: DiceMetric,
    hd95_metric: HausdorffDistanceMetric,
    post_pred: Compose,
    post_label: Compose,
    roi_size: tuple[int, int, int] = (96, 96, 96),
    sw_batch_size: int = 4,
    overlap: float = 0.5,
) -> tuple[float, float]:
    """
    Sliding window validation for full-volume inference without CUDA OOM.
    Logs Mean Dice Score and HD95 metrics.
    """
    model.eval()
    dice_metric.reset()
    hd95_metric.reset()

    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Validation", leave=False):
            val_images = batch["image"].to(device, non_blocking=True)
            val_labels = batch["label"].to(device, non_blocking=True)

            # PyTorch CUDA AMP during inference
            if device.type == "cuda":
                with torch.cuda.amp.autocast(dtype=torch.float16):
                    val_outputs = sliding_window_inference(
                        inputs=val_images,
                        roi_size=roi_size,
                        sw_batch_size=sw_batch_size,
                        predictor=model,
                        overlap=overlap,
                        mode="gaussian",
                    )
            else:
                val_outputs = sliding_window_inference(
                    inputs=val_images,
                    roi_size=roi_size,
                    sw_batch_size=sw_batch_size,
                    predictor=model,
                    overlap=overlap,
                    mode="gaussian",
                )

            # Discretize predictions and labels for metric computation
            val_outputs_discrete = [post_pred(i) for i in val_outputs]
            val_labels_discrete = [post_label(i) for i in val_labels]

            # Compute Dice Metric
            dice_metric(y_pred=val_outputs_discrete, y=val_labels_discrete)

            # Compute HD95 (Hausdorff Distance 95th percentile)
            try:
                hd95_metric(y_pred=val_outputs_discrete, y=val_labels_discrete)
            except Exception:
                # In case foreground is entirely absent in a slice
                pass

    mean_dice = float(dice_metric.aggregate().item())
    try:
        mean_hd95 = float(hd95_metric.aggregate().item())
    except Exception:
        mean_hd95 = float("nan")

    dice_metric.reset()
    hd95_metric.reset()

    return mean_dice, mean_hd95


def main() -> None:
    args = parse_args()
    set_determinism(seed=42)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output_dir / "model_best.pt"
    log_path = output_dir / "train_log.json"

    device = torch.device(args.device)
    print(f"[ENGINE] Target Device: {device} ({'CUDA GPU' if device.type == 'cuda' else 'CPU'})")

    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        print(f"[CUDA] Device Name : {torch.cuda.get_device_name(device)}")
        print(f"[CUDA] Memory Total: {torch.cuda.get_device_properties(device).total_memory / (1024**3):.2f} GB")

    # Data dicts setup
    if args.sanity_check or not args.data_dir:
        print("[SETUP] Preparing synthetic cardiac CT scans for sanity check...")
        synth_dir = output_dir / "synthetic_cache"
        sample_case_1 = generate_synthetic_nifti(synth_dir, prefix="case_synth_01")
        sample_case_2 = generate_synthetic_nifti(synth_dir, prefix="case_synth_02")
        train_files = [sample_case_1]
        val_files = [sample_case_2]
        total_epochs = 2 if args.sanity_check else args.epochs
    else:
        # Load user provided dataset directory
        data_path = Path(args.data_dir)
        images = sorted(list(data_path.glob("*_img.nii*")) + list(data_path.glob("*_image.nii*")))
        labels = sorted(list(data_path.glob("*_label.nii*")) + list(data_path.glob("*_seg.nii*")))
        all_files = [{"image": str(img), "label": str(lbl)} for img, lbl in zip(images, labels)]
        split = max(1, int(len(all_files) * 0.8))
        train_files = all_files[:split]
        val_files = all_files[split:] if len(all_files) > 1 else all_files
        total_epochs = args.epochs

    print(f"[DATA] Train Volumes: {len(train_files)} | Val Volumes: {len(val_files)}")

    # Transforms & DataLoaders
    train_transforms = get_transforms(mode="train", spatial_size=(96, 96, 96), num_samples=4)
    val_transforms = get_transforms(mode="val")

    train_ds = build_dataset(train_files, train_transforms, use_cache=True, cache_rate=1.0)
    val_ds = build_dataset(val_files, val_transforms, use_cache=True, cache_rate=1.0)

    train_loader = build_dataloader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers if device.type == "cuda" else 0,
        pin_memory=(device.type == "cuda"),
    )
    val_loader = build_dataloader(
        val_ds,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        pin_memory=(device.type == "cuda"),
    )

    # Model & Loss Initialization
    model, criterion = build_model(
        device=device,
        img_size=(96, 96, 96),
        feature_size=args.feature_size,
        use_checkpoint=True,
    )

    # Optimizer & Scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_epochs, eta_min=1e-6)

    # Mixed Precision Scaler
    scaler = torch.cuda.amp.GradScaler() if device.type == "cuda" else None

    # Metrics & Post-processing
    dice_metric = DiceMetric(include_background=False, reduction="mean")
    hd95_metric = HausdorffDistanceMetric(percentile=95, include_background=False, reduction="mean")

    post_pred = Compose([Activations(sigmoid=True), AsDiscrete(threshold=0.5)])
    post_label = Compose([AsDiscrete(threshold=0.5)])

    best_dice = -1.0
    history: list[dict[str, Any]] = []

    print("[TRAIN] Starting training loop...")
    start_time = time.time()

    for epoch in range(1, total_epochs + 1):
        epoch_start = time.time()
        train_loss = train_one_epoch(
            model=model,
            train_loader=train_loader,
            optimizer=optimizer,
            scaler=scaler,
            criterion=criterion,
            device=device,
            use_amp=(device.type == "cuda"),
        )
        scheduler.step()

        val_dice, val_hd95 = float("nan"), float("nan")
        if epoch % args.val_interval == 0:
            val_dice, val_hd95 = validate(
                model=model,
                val_loader=val_loader,
                device=device,
                dice_metric=dice_metric,
                hd95_metric=hd95_metric,
                post_pred=post_pred,
                post_label=post_label,
            )

            # Best model checkpointing
            if val_dice > best_dice:
                best_dice = val_dice
                checkpoint = {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "best_dice": best_dice,
                    "val_hd95": val_hd95,
                    "args": vars(args),
                }
                torch.save(checkpoint, str(checkpoint_path))
                print(f"  [CHECKPOINT] Epoch {epoch}: Best Mean Dice = {best_dice:.4f} -> Saved {checkpoint_path}")

        epoch_duration = time.time() - epoch_start
        epoch_info = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_dice": val_dice,
            "val_hd95": val_hd95,
            "lr": optimizer.param_groups[0]["lr"],
            "duration_sec": epoch_duration,
        }
        history.append(epoch_info)

        print(
            f"Epoch [{epoch:03d}/{total_epochs:03d}] "
            f"Loss: {train_loss:.4f} | "
            f"Val Dice: {val_dice:.4f} | "
            f"HD95: {val_hd95:.2f} | "
            f"Time: {epoch_duration:.1f}s"
        )

    total_time = time.time() - start_time
    print(f"\n[DONE] Finished {total_epochs} epochs in {total_time / 60.0:.2f} minutes.")
    print(f"[BEST] Top Mean Dice: {best_dice:.4f}")

    with open(log_path, "w", encoding="utf-8") as f:
        json.dump({"best_dice": best_dice, "history": history}, f, indent=2)
    print(f"[LOGS] Saved training history to {log_path}")


if __name__ == "__main__":
    main()
