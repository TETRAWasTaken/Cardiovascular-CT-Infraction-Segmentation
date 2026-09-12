"""
Phase 3: SwinUNETR Forward-Pass Integration Test (CUDA Target)
=============================================================
Verification Target:
  Input Tensor:  [2, 1, 96, 96, 96]
  Output Tensor: [2, 1, 96, 96, 96]
  Loss: DiceFocalLoss calculation & backward gradient flow
"""

from __future__ import annotations

import pytest
import torch

from Transformer.src.model import build_model, get_loss_function


@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="Host lacks CUDA GPU. Per instructions, SwinUNETR execution tests are targeted for x86_64 CUDA nodes."
)
def test_swinunetr_cuda_forward_pass():
    """Validates 3D SwinUNETR forward pass and shape assertion on NVIDIA CUDA device."""
    device = torch.device("cuda:0")

    model, criterion = build_model(
        device=device,
        img_size=(96, 96, 96),
        feature_size=48,
        use_checkpoint=True,
    )
    model.eval()

    batch_size = 2
    in_channels = 1
    d, h, w = 96, 96, 96

    # Input: [2, 1, 96, 96, 96] on CUDA
    inputs = torch.randn((batch_size, in_channels, d, h, w), device=device, dtype=torch.float32)

    with torch.no_grad():
        outputs = model(inputs)

    # Verification Assertion: Input: [2, 1, 96, 96, 96] -> Output: [2, 1, 96, 96, 96]
    expected_shape = (batch_size, 1, d, h, w)
    assert outputs.shape == expected_shape, (
        f"Shape mismatch! Expected {expected_shape}, got {outputs.shape}"
    )
    assert not torch.isnan(outputs).any(), "Model produced NaN values in forward pass."
    print(f"\n[CUDA VERIFICATION] Input: {list(inputs.shape)} -> Output: {list(outputs.shape)}")


@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="Host lacks CUDA GPU. Targeted for x86_64 CUDA nodes."
)
def test_swinunetr_loss_and_backward_cuda():
    """Validates DiceFocalLoss computation and backward pass on CUDA."""
    device = torch.device("cuda:0")

    model, criterion = build_model(
        device=device,
        img_size=(96, 96, 96),
        feature_size=48,
        use_checkpoint=True,
    )
    model.train()

    inputs = torch.randn((2, 1, 96, 96, 96), device=device, dtype=torch.float32)
    targets = torch.randint(0, 2, (2, 1, 96, 96, 96), device=device, dtype=torch.float32)

    outputs = model(inputs)
    loss = criterion(outputs, targets)

    assert not torch.isnan(loss), "Loss computed as NaN"
    assert loss.item() > 0.0, f"Unexpected loss value: {loss.item()}"

    loss.backward()

    # Ensure gradients populated
    has_grad = any(p.grad is not None and p.grad.norm() > 0 for p in model.parameters())
    assert has_grad, "No gradients were computed in backward pass!"
    print(f"\n[CUDA LOSS & BACKPROP] Loss value: {loss.item():.4f}, Gradients successfully verified.")


if __name__ == "__main__":
    if torch.cuda.is_available():
        test_swinunetr_cuda_forward_pass()
        test_swinunetr_loss_and_backward_cuda()
    else:
        print("[INFO] Local environment has no CUDA. Forward pass test ready for x86_64 CUDA node.")
