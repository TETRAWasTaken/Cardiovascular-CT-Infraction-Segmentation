"""
BaselineUNET Training and Validation Script
===========================================

End-to-end training pipeline for 3D Coronary Artery Segmentation on the ImageCAS dataset.
Supports:
- Large dataset handling via manifest & on-the-fly foreground patch sampling
- Mixed precision training (AMP)
- Sliding-window validation for full-volume inference without OOM
- Best-metric checkpointing and training history logging
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
from monai.inferers import sliding_window_inference
from monai.losses import DiceCELoss
from monai.metrics import DiceMetric
from monai.transforms import Activations, AsDiscrete, Compose
from monai.utils import set_determinism
from tqdm import tqdm

# Add project root to sys.path if executed directly
project_root = Path(__file__).resolve().parents[1]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from BaselineUNET.utils.unet import ArterySegmentationModel
from BaselineUNET.utils.dataset import (
    load_manifest_data_dicts,
    scan_directory_data_dicts,
    build_dataloaders,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train 3D U-Net Baseline on ImageCAS Coronary Artery Dataset"
    )

    # Data arguments
    parser.add_argument(
        "--manifest",
        type=str,
        default="",
        help="Path to imagecas_manifest.csv produced by data_ingestion.py",
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="",
        help="Fallback directory path containing extracted ImageCAS NIfTI files",
    )
    parser.add_argument(
        "--val_ratio",
        type=float,
        default=0.2,
        help="Validation split ratio (default: 0.2)",
    )

    # Patch and Training arguments
    parser.add_argument(
        "--patch_size",
        type=int,
        nargs=3,
        default=[96, 96, 96],
        help="3D patch size for training (e.g. 96 96 96 or 128 128 128)",
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=4,
        help="Number of foreground/background patches extracted per volume per iteration",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=2,
        help="Training batch size (number of volumes loaded per batch)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=100,
        help="Number of training epochs",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-3,
        help="Initial learning rate for AdamW optimizer",
    )
    parser.add_argument(
        "--weight_decay",
        type=float,
        default=1e-4,
        help="Weight decay for AdamW",
    )
    parser.add_argument(
        "--val_interval",
        type=int,
        default=2,
        help="Epoch interval for running validation",
    )

    # Sliding-window validation arguments
    parser.add_argument(
        "--sw_batch_size",
        type=int,
        default=4,
        help="Batch size for sliding-window inference during validation",
    )
    parser.add_argument(
        "--sw_overlap",
        type=float,
        default=0.5,
        help="Overlap between sliding window patches during validation",
    )

    # Hardware & Performance
    parser.add_argument(
        "--num_workers",
        type=int,
        default=2,
        help="Number of DataLoader worker processes",
    )
    parser.add_argument(
        "--cache_rate",
        type=float,
        default=0.0,
        help="In-memory cache rate (keep 0.0 for 80GB dataset to prevent RAM exhaustion)",
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        default="",
        help="Optional disk directory for PersistentDataset caching (uses 0 RAM, fast NVMe caching)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="",
        help="Target device: 'cuda', 'mps', or 'cpu' (auto-detected if empty)",
    )
    parser.add_argument(
        "--no_amp",
        action="store_true",
        help="Disable Automatic Mixed Precision (AMP)",
    )

    # Checkpoints and outputs
    parser.add_argument(
        "--output_dir",
        type=str,
        default="BaselineUNET/runs/experiment_1",
        help="Directory to save model checkpoints and training logs",
    )
    parser.add_argument(
        "--resume",
        type=str,
        default="",
        help="Path to existing checkpoint to resume training from",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility",
    )

    return parser.parse_args()


def select_device(preferred_device: str) -> torch.device:
    if preferred_device:
        return torch.device(preferred_device)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def train_one_epoch(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_function: nn.Module,
    device: torch.device,
    scaler: torch.amp.GradScaler | None,
    epoch: int,
    total_epochs: int,
) -> float:
    model.train()
    epoch_loss = 0.0
    step = 0

    pbar = tqdm(loader, desc=f"Epoch {epoch + 1}/{total_epochs} [Train]", leave=False)
    for batch_data in pbar:
        step += 1
        inputs = batch_data["image"].to(device, non_blocking=True)
        labels = batch_data["label"].to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        if scaler is not None:
            with torch.amp.autocast(device_type=device.type):
                outputs = model(inputs)
                loss = loss_function(outputs, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(inputs)
            loss = loss_function(outputs, labels)
            loss.backward()
            optimizer.step()

        epoch_loss += loss.item()
        pbar.set_postfix({"loss": f"{loss.item():.4f}"})

    return epoch_loss / max(step, 1)


@torch.no_grad()
def validate(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    metric: DiceMetric,
    post_trans: Compose,
    patch_size: list[int],
    sw_batch_size: int,
    sw_overlap: float,
    device: torch.device,
    scaler: torch.amp.GradScaler | None,
    epoch: int,
    total_epochs: int,
) -> float:
    model.eval()
    metric.reset()

    pbar = tqdm(loader, desc=f"Epoch {epoch + 1}/{total_epochs} [Val]", leave=False)
    for val_data in pbar:
        val_inputs = val_data["image"].to(device)
        val_labels = val_data["label"].to(device)

        if scaler is not None and device.type == "cuda":
            with torch.amp.autocast(device_type=device.type):
                val_outputs = sliding_window_inference(
                    inputs=val_inputs,
                    roi_size=patch_size,
                    sw_batch_size=sw_batch_size,
                    predictor=model,
                    overlap=sw_overlap,
                )
        else:
            val_outputs = sliding_window_inference(
                inputs=val_inputs,
                roi_size=patch_size,
                sw_batch_size=sw_batch_size,
                predictor=model,
                overlap=sw_overlap,
            )

        val_outputs = [post_trans(i) for i in val_outputs]
        metric(y_pred=val_outputs, y=val_labels)

    dice_score = metric.aggregate().item()
    metric.reset()
    return float(dice_score)


def main() -> None:
    args = parse_args()
    set_determinism(seed=args.seed)

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    device = select_device(args.device)
    print(f"Using device: {device}")

    # Discover and prepare dataset
    if args.manifest:
        print(f"Loading dataset from manifest: {args.manifest}")
        train_files, val_files = load_manifest_data_dicts(
            args.manifest, val_ratio=args.val_ratio, seed=args.seed
        )
    elif args.data_dir:
        print(f"Scanning dataset from directory: {args.data_dir}")
        train_files, val_files = scan_directory_data_dicts(
            args.data_dir, val_ratio=args.val_ratio, seed=args.seed
        )
    else:
        # Default check for local manifest or default data directory
        default_manifest = Path("BaselineUNET/imagecas_manifest.csv")
        possible_dirs = [
            Path("data/extracted"),
            Path("Data/extracted"),
            Path("data"),
            Path("Data"),
        ]
        
        if default_manifest.exists():
            print(f"Found default manifest at: {default_manifest}")
            train_files, val_files = load_manifest_data_dicts(
                default_manifest, val_ratio=args.val_ratio, seed=args.seed
            )
        else:
            found_dir = None
            for p in possible_dirs:
                if p.exists() and any(p.rglob("*.nii.gz")):
                    found_dir = p
                    break

            if found_dir:
                print(f"Auto-detected dataset directory: {found_dir}")
                train_files, val_files = scan_directory_data_dicts(
                    found_dir, val_ratio=args.val_ratio, seed=args.seed
                )
            else:
                raise ValueError(
                    "Could not find default manifest or data directory. "
                    "Please provide --manifest BaselineUNET/imagecas_manifest.csv or --data_dir Data/extracted"
                )

    print(f"Dataset summary: {len(train_files)} training volumes, {len(val_files)} validation volumes")

    # Build data loaders
    train_loader, val_loader = build_dataloaders(
        train_files=train_files,
        val_files=val_files,
        patch_size=tuple(args.patch_size),
        num_samples=args.num_samples,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        cache_rate=args.cache_rate,
        cache_dir=args.cache_dir if args.cache_dir else None,
    )

    # Initialize Model, Loss, Optimizer
    model = ArterySegmentationModel(in_channels=1, out_channels=1).to(device)
    loss_function = DiceCELoss(sigmoid=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    # AMP Scaler
    use_amp = (not args.no_amp) and (device.type == "cuda")
    scaler = torch.amp.GradScaler("cuda") if use_amp else None

    # Validation evaluation transforms & metrics
    post_trans = Compose([Activations(sigmoid=True), AsDiscrete(threshold=0.5)])
    dice_metric = DiceMetric(include_background=False, reduction="mean")

    # Checkpoint state
    start_epoch = 0
    best_dice = -1.0
    history = []

    if args.resume and Path(args.resume).exists():
        print(f"Loading checkpoint from: {args.resume}")
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint["model_state"])
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        scheduler.load_state_dict(checkpoint["scheduler_state"])
        start_epoch = checkpoint["epoch"] + 1
        best_dice = checkpoint.get("best_dice", -1.0)
        history = checkpoint.get("history", [])

    print("\n" + "=" * 70)
    print("BaselineUNET 3D Training Started")
    print(f"Patch size   : {args.patch_size}")
    print(f"Batch size   : {args.batch_size} (effective patch batch: {args.batch_size * args.num_samples})")
    print(f"Total epochs : {args.epochs}")
    print(f"AMP enabled  : {use_amp}")
    print("=" * 70 + "\n")

    start_time = time.time()

    for epoch in range(start_epoch, args.epochs):
        epoch_start = time.time()
        train_loss = train_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            loss_function=loss_function,
            device=device,
            scaler=scaler,
            epoch=epoch,
            total_epochs=args.epochs,
        )
        scheduler.step()

        val_dice = None
        if (epoch + 1) % args.val_interval == 0 or (epoch + 1) == args.epochs:
            val_dice = validate(
                model=model,
                loader=val_loader,
                metric=dice_metric,
                post_trans=post_trans,
                patch_size=args.patch_size,
                sw_batch_size=args.sw_batch_size,
                sw_overlap=args.sw_overlap,
                device=device,
                scaler=scaler,
                epoch=epoch,
                total_epochs=args.epochs,
            )

        epoch_duration = time.time() - epoch_start
        status_msg = f"Epoch [{epoch + 1:03d}/{args.epochs:03d}] Loss: {train_loss:.4f} | LR: {scheduler.get_last_lr()[0]:.6f} | Time: {epoch_duration:.1f}s"
        
        if val_dice is not None:
            status_msg += f" | Val Dice: {val_dice:.4f}"
            if val_dice > best_dice:
                best_dice = val_dice
                best_model_path = output_dir / "best_metric_model.pth"
                torch.save(
                    {
                        "epoch": epoch,
                        "model_state": model.state_dict(),
                        "best_dice": best_dice,
                    },
                    best_model_path,
                )
                status_msg += f" -> [NEW BEST SAVED to {best_model_path.name}]"

        print(status_msg)

        # Record metrics history
        history.append({
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "val_dice": val_dice,
            "lr": scheduler.get_last_lr()[0],
            "duration": epoch_duration,
        })

        # Save latest checkpoint
        latest_path = output_dir / "latest_checkpoint.pth"
        torch.save(
            {
                "epoch": epoch,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "scheduler_state": scheduler.state_dict(),
                "best_dice": best_dice,
                "history": history,
            },
            latest_path,
        )

        # Save history log JSON
        with open(output_dir / "training_history.json", "w") as f:
            json.dump(history, f, indent=2)

    total_time = time.time() - start_time
    print("\n" + "=" * 70)
    print(f"Training Complete in {total_time / 60:.2f} minutes.")
    print(f"Best Validation Dice: {best_dice:.4f}")
    print(f"Checkpoints saved to: {output_dir}")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
