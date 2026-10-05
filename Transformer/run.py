import sys
from pathlib import Path

# Set up project home directory in sys.path so 'Transformer' can be located
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    from Transformer.src.tune import run_hyperparameter_tuning
except ModuleNotFoundError:
    TRANSFORMER_DIR = Path(__file__).resolve().parent
    if str(TRANSFORMER_DIR) not in sys.path:
        sys.path.insert(0, str(TRANSFORMER_DIR))
    from src.tune import run_hyperparameter_tuning


# Base configuration applied to all trials
base_config = {
    # Dataset and Output paths
    "data_dir": "/path/to/data",  # Replace with your NIfTI dataset path
    "output_dir": str(Path(__file__).resolve().parent / "artifacts" / "tuning_adamw"),
    "val_split": 0.2,
    "cache_rate": 1.0,

    # Hardware & Performance
    "device": "cuda",             # 'cuda' or 'cpu'
    "amp": True,                  # Automatic Mixed Precision for NVIDIA CUDA
    "num_workers": 4,

    # Optimization: Locked to AdamW
    "optimizer": "adamw",
    "scheduler": "cosine",        # CosineAnnealingLR over the trial epochs
    "epochs": 50,                 # Epochs per trial
    "batch_size": 2,              # Patch batch size (increase to 4 if 24GB+ VRAM)
    "val_interval": 1,
    "seed": 42,

    # Patch size & sampling
    "spatial_size": (96, 96, 96),
    "num_samples": 4,
    "use_checkpoint": True,       # Gradient checkpointing to save VRAM
}

# Proper AdamW Hyperparameter Grid
# Balances model capacity (feature_size), transformer regularization (drop_rate),
# and AdamW dynamics (lr + weight_decay)
param_grid = {
    # 1. SwinUNETR Architecture Capacity
    # 24: Lightweight (fast convergence, low VRAM)
    # 48: Standard MONAI capacity (recommended for fine vessel details)
    "feature_size": [24, 48],

    # 2. AdamW Learning Rate
    # SwinUNETR standard range: 1e-4 to 3e-4 with CosineAnnealing
    "lr": [1e-4, 2e-4, 3e-4],

    # 3. AdamW Weight Decay (critical for Transformer generalization)
    # 1e-5: Light regularization
    # 1e-4: Standard regularization to prevent overfitting on 3D CT
    "weight_decay": [1e-5, 1e-4],

    # 4. Transformer Block Regularization
    # 0.0: No dropout
    # 0.1: Dropout to prevent attention overconfidence
    "drop_rate": [0.0, 0.1],

    # 5. Hybrid DiceFocalLoss focal weight (addressing <1% foreground artery volume)
    # "lambda_focal": [0.5, 1.0],
}

if __name__ == "__main__":
    # GPU Sharding: Trains 2 models simultaneously!
    # - If multiple physical GPUs exist (e.g. 2 GPUs), automatically assigns ['cuda:0', 'cuda:1']
    # - If single GPU exists, shards VRAM dynamically (e.g. 48% each) on ['cuda:0', 'cuda:0']
    # - You can also explicitly specify: devices=["cuda:0", "cuda:1"]
    tuning_results = run_hyperparameter_tuning(
        base_config=base_config,
        param_grid=param_grid,
        search_type="grid",           # "grid" or "random"
        max_parallel_jobs=2,          # Simultaneously train 2 models!
        devices=None,                 # Auto-detects GPUs or set e.g. ["cuda:0", "cuda:1"]
        gpu_memory_fraction=0.48,     # VRAM limit if sharding a single GPU
        # n_trials=6,                 # Uncomment to cap total trials
        seed=42,
    )

    print("\n================ TUNING COMPLETED ================")
    print("Top Trial:", tuning_results["best_trial"])
    print("Summary CSV:", tuning_results["summary_csv"])
    print("Leaderboard JSON:", tuning_results["summary_json"])
    print("Best Model Checkpoint:", tuning_results["best_model_overall"])
