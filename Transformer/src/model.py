"""
Phase 3: SwinUNETR Model Construction & Loss Definition
======================================================
Architecture: 3D SwinUNETR with Shifted Window Self-Attention
Loss: Hybrid DiceFocalLoss for severe voxel sparsity (<1% foreground)
Target: NVIDIA CUDA GPUs (x86_64) with activation checkpointing
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
from monai.losses import DiceFocalLoss
from monai.networks.nets import SwinUNETR


class SwinUNETRSegmentation(nn.Module):
    """
    3D SwinUNETR wrapper configured specifically for high-resolution
    coronary artery / infarction segmentation on NVIDIA CUDA.
    """

    def __init__(
        self,
        img_size: tuple[int, int, int] = (96, 96, 96),
        in_channels: int = 1,
        out_channels: int = 1,
        feature_size: int = 48,
        use_checkpoint: bool = True,
        spatial_dims: int = 3,
        drop_rate: float = 0.0,
        attn_drop_rate: float = 0.0,
        dropout_path_rate: float = 0.0,
    ) -> None:
        super().__init__()

        self.img_size = img_size
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.feature_size = feature_size
        self.use_checkpoint = use_checkpoint

        init_kwargs: dict[str, Any] = {
            "in_channels": in_channels,
            "out_channels": out_channels,
            "feature_size": feature_size,
            "use_checkpoint": use_checkpoint,
            "spatial_dims": spatial_dims,
            "drop_rate": drop_rate,
            "attn_drop_rate": attn_drop_rate,
            "dropout_path_rate": dropout_path_rate,
        }
        import inspect
        if "img_size" in inspect.signature(SwinUNETR.__init__).parameters:
            init_kwargs["img_size"] = img_size

        self.model = SwinUNETR(**init_kwargs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        Input: Tensor of shape [B, in_channels, D, H, W]
        Output: Logits tensor of shape [B, out_channels, D, H, W]
        """
        return self.model(x)


def get_loss_function(
    sigmoid: bool = True,
    lambda_dice: float = 1.0,
    lambda_focal: float = 0.5,
    gamma: float = 2.0,
) -> DiceFocalLoss:
    """
    Constructs hybrid DiceFocalLoss.
    Focal loss addresses extreme foreground-background imbalance (<1% vessel volume),
    while Dice loss directly optimizes the volume overlap coefficient.
    """
    return DiceFocalLoss(
        sigmoid=sigmoid,
        lambda_dice=lambda_dice,
        lambda_focal=lambda_focal,
        gamma=gamma,
        batch=True,
    )


def build_model(
    device: str | torch.device = "cuda",
    img_size: tuple[int, int, int] = (96, 96, 96),
    in_channels: int = 1,
    out_channels: int = 1,
    feature_size: int = 48,
    use_checkpoint: bool = True,
    drop_rate: float = 0.0,
    attn_drop_rate: float = 0.0,
    dropout_path_rate: float = 0.0,
    lambda_dice: float = 1.0,
    lambda_focal: float = 0.5,
    gamma: float = 2.0,
) -> tuple[SwinUNETRSegmentation, DiceFocalLoss]:
    """
    Factory function returning initialized SwinUNETR model on target device and loss function.
    Supports configuring architecture hyperparameters and loss weights for hyperparameter tuning.
    """
    model = SwinUNETRSegmentation(
        img_size=img_size,
        in_channels=in_channels,
        out_channels=out_channels,
        feature_size=feature_size,
        use_checkpoint=use_checkpoint,
        spatial_dims=3,
        drop_rate=drop_rate,
        attn_drop_rate=attn_drop_rate,
        dropout_path_rate=dropout_path_rate,
    )

    if str(device).startswith("cuda") and torch.cuda.is_available():
        model = model.to(device)
        # Enable cuDNN benchmark for fixed input patch sizes
        torch.backends.cudnn.benchmark = True
    elif str(device) != "cpu":
        model = model.to(device)

    criterion = get_loss_function(
        sigmoid=True,
        lambda_dice=lambda_dice,
        lambda_focal=lambda_focal,
        gamma=gamma,
    )
    return model, criterion

