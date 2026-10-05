import os
import sys
from pathlib import Path

# Prevent NVIDIA MPS Error 805 if stale MPS environment variable is active
if "CUDA_MPS_PIPE_DIRECTORY" in os.environ:
    mps_dir = Path(os.environ["CUDA_MPS_PIPE_DIRECTORY"])
    if not mps_dir.exists() or not (mps_dir / "control").exists():
        print(f"[NOTICE] Unsetting stale CUDA_MPS_PIPE_DIRECTORY='{mps_dir}' to allow direct GPU access.")
        os.environ.pop("CUDA_MPS_PIPE_DIRECTORY", None)
        os.environ.pop("CUDA_MPS_LOG_DIRECTORY", None)

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


def _detect_dataset_dir() -> str:
    env_dir = os.environ.get("DATA_DIR")
    if env_dir and Path(env_dir).exists():
        return env_dir
    candidates = [
        PROJECT_ROOT / "data" / "extracted",
        PROJECT_ROOT / "data",
        PROJECT_ROOT / "Data" / "extracted",
        PROJECT_ROOT / "Data",
    ]
    for c in candidates:
        if c.exists() and any(c.rglob("*.nii*")):
            return str(c)
    return "/path/to/data"


# Base configuration applied to all trials
base_config = {
    # Dataset and Output paths
    "data_dir": _detect_dataset_dir(),  # Set to your NIfTI dataset path, or set sanity_check: True
    "sanity_check": False,              # Set True to run with synthetic cardiac volumes
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
    # Pre-flight check: validate dataset configuration before starting trials
    data_dir_str = base_config.get("data_dir", "")
    data_path = Path(data_dir_str).expanduser()
    is_sanity = base_config.get("sanity_check", False)

    if not is_sanity and (data_dir_str == "/path/to/data" or not data_path.exists()):
        print("\n" + "=" * 75)
        print("❌ [CONFIG ERROR] Dataset directory not found or unconfigured!")
        print(f"Current setting: '{data_dir_str}'")
        print("=" * 75)
        print("Please configure 'data_dir' in Transformer/run.py with your dataset path, e.g.:")
        print("    base_config['data_dir'] = '/home/CL502-27/Cardiovascular-CT-Infraction-Segmentation/data/extracted'")
        print("\nOr provide via environment variable:")
        print("    DATA_DIR=/path/to/data python Transformer/run.py")
        print("\nOr validate the pipeline using synthetic cardiac volumes by setting:")
        print("    base_config['sanity_check'] = True")
        print("=" * 75 + "\n")
        sys.exit(1)

    # Sequential Training Mode:
    # Runs 1 trial at a time, giving each model 100% of the GPU VRAM.
    # Completely avoids 'CUDA device busy' errors in EXCLUSIVE_PROCESS compute mode.
    tuning_results = run_hyperparameter_tuning(
        base_config=base_config,
        param_grid=param_grid,
        search_type="grid",           # "grid" or "random"
        max_parallel_jobs=1,          # 1 = Sequential training (1 model at a time)
        devices=None,                 # Auto-detects primary GPU (cuda:0)
        gpu_memory_fraction=1.0,      # Full GPU VRAM available per trial
        # n_trials=6,                 # Uncomment to cap total trials
        seed=42,
    )

    print("\n================ TUNING COMPLETED ================")
    print("Top Trial:", tuning_results["best_trial"])
    print("Summary CSV:", tuning_results["summary_csv"])
    print("Leaderboard JSON:", tuning_results["summary_json"])
    print("Best Model Checkpoint:", tuning_results["best_model_overall"])
