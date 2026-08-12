import torch
import torch.nn as nn
from monai.networks.nets import UNet
from monai.losses import DiceCELoss

class ArterySegmentationModel(nn.Module):
    """
    This is the 3D UNet model which will be used for segmenting the arteries from the CT scans.
    """

    def __init__(self, in_channels: int = 1, out_channels: int = 1):
        super(ArterySegmentationModel, self).__init__()

        self.net = UNet(
            spatial_dims = 3,
            in_channels = in_channels,
            out_channels = out_channels,
            channels = (16, 32, 64, 128, 256),
            strides = (2,2,2,2),
            num_res_units = 2,
            dropout = 0.5,
            norm = "batch"
        )

    def forward(self, x):
        return self.net(x)        

if __name__ == "__main__":
    # Create the model instance
    model = ArterySegmentationModel(in_channels=1, out_channels=1)
    
    # Simulate a batch of 2 CT sub-volumes (Batch=2, Channel=1, D=64, H=128, W=128)
    dummy_ct_patch = torch.randn(2, 1, 64, 128, 128)
    
    # Forward pass
    raw_logits = model(dummy_ct_patch)
    print(f"Input Shape : {dummy_ct_patch.shape}")
    print(f"Output Shape: {raw_logits.shape}")  # Output matches input dimensions [2, 1, 64, 128, 128]

    # Combined Dice + Cross Entropy Loss (Industry standard for sparse 3D segmentation)
    loss_function = DiceCELoss(sigmoid=True)
    dummy_target = torch.randint(0, 2, (2, 1, 64, 128, 128)).float()
    
    loss = loss_function(raw_logits, dummy_target)
    print(f"Calculated Loss: {loss.item():.4f}")