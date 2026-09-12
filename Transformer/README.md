# SwinUNETR 3D Cardiac CT Artery Segmentation & Physical Centerline Pipeline

End-to-end deep learning and post-processing pipeline for 3D coronary artery segmentation and physical centerline coordinate extraction, targeted specifically for **x86_64 systems with NVIDIA CUDA acceleration**.

---

## Architecture & System Design

```
Raw 3D Cardiac CT (NIfTI / DICOM)
             │
             ▼
   [Phase 2: Preprocessing]
   • LoadImaged (image & label)
   • EnsureChannelFirstd
   • Orientationd(axcodes="RAS")
   • Spacingd(pixdim=(1.0, 1.0, 1.0))
   • ScaleIntensityRanged(a_min=-100, a_max=500, b_min=0.0, b_max=1.0)
   • RandCropByPosNegLabeld(96, 96, 96)
             │
             ▼
   [Phase 3: 3D SwinUNETR]
   • in_channels=1, out_channels=1
   • img_size=(96, 96, 96), feature_size=48
   • use_checkpoint=True (Activation checkpointing for VRAM efficiency)
   • Loss: DiceFocalLoss (λ_dice=1.0, λ_focal=0.5, gamma=2.0)
             │
             ▼
   [Phase 4: Training & Validation]
   • PyTorch AMP (Automatic Mixed Precision: torch.cuda.amp.autocast)
   • AdamW (lr=1e-4) + CosineAnnealingLR
   • Validation: sliding_window_inference(roi_size=(96,96,96), sw_batch_size=4)
   • Metrics: Mean Dice Score + HD95 (Hausdorff Distance 95th Percentile)
   • Checkpoint: artifacts/model_best.pt
             │
             ▼
   [Phase 5: Coordinate Extraction & Centerline Skeletonization]
   • Thresholding: p > 0.5
   • cc3d 3D Largest Connected Component Analysis (removes noise)
   • skimage.morphology.skeletonize (3D topological medial axis)
   • SimpleITK TransformIndexToPhysicalPoint (voxel (z,y,x) -> physical (X,Y,Z) in mm)
   • Export: artifacts/artery_centerlines.json & CSV
```

---

## Directory Structure

```text
Transformer/
├── requirements.txt            # Dependencies for x86_64 CUDA environment
├── README.md                   # Complete architectural and operational documentation
├── scripts/
│   └── verify_env.py          # Phase 1: CUDA compute capability, cuDNN, & spatial libs check
├── src/
│   ├── __init__.py
│   ├── dataset.py             # Phase 2: MONAI CacheDataset, transforms, synthetic generator
│   ├── model.py               # Phase 3: SwinUNETR wrapper & DiceFocalLoss definition
│   ├── train.py               # Phase 4: CUDA AMP training, sliding-window validation, HD95/Dice
│   └── extract_coords.py      # Phase 5: cc3d filter, 3D skeletonization, SimpleITK mapping
├── tests/
│   ├── __init__.py
│   ├── test_transforms.py     # Phase 2: Slice spatial alignment test & PNG artifact generator
│   └── test_model_forward.py  # Phase 3: CUDA tensor shape assertion test [2,1,96,96,96]
└── artifacts/                 # Verification outputs (PNGs, checkpoints, JSON centerlines)
```

---

## Setup & Requirements (x86_64 + NVIDIA CUDA)

### 1. Install PyTorch with CUDA Support
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

### 2. Install Project Dependencies
```bash
pip install -r Transformer/requirements.txt
```

### 3. Verify CUDA Environment
```bash
python Transformer/scripts/verify_env.py
```
Or run the single-line verification:
```bash
python -c "import torch, monai, SimpleITK; print(torch.cuda.is_available())"
```

---

## Execution Guide

### Phase 2: Run Data Preprocessing & Slice Inspection Artifact
```bash
python -m pytest Transformer/tests/test_transforms.py -v
```
Generates visual slice inspection artifact: `Transformer/artifacts/transforms_check.png`.

### Phase 3: Forward Pass & Shape Assertions (on CUDA Node)
```bash
python -m pytest Transformer/tests/test_model_forward.py -v
```
Asserts tensor shapes: Input `[2, 1, 96, 96, 96]` $\to$ Output `[2, 1, 96, 96, 96]`.

### Phase 4: Model Training (CUDA AMP)
To run full training on a CUDA node with your dataset:
```bash
python Transformer/src/train.py \
    --data_dir /path/to/nifti_data \
    --epochs 100 \
    --batch_size 2 \
    --device cuda
```
To run a 2-epoch sanity check with synthetic cardiac CT volumes:
```bash
python Transformer/src/train.py --sanity_check --device cuda
```
Best model checkpoint is saved to: `Transformer/artifacts/model_best.pt`.

### Phase 5: Artery Centerline & Physical Coordinate Extraction
Extract physical world coordinates $(X, Y, Z)$ from predicted probability maps or test volumes:
```bash
python Transformer/src/extract_coords.py \
    --input /path/to/predicted_prob_map.nii.gz \
    --threshold 0.5 \
    --top_k 1 \
    --output_json Transformer/artifacts/artery_centerlines.json \
    --output_csv Transformer/artifacts/artery_centerlines.csv
```
Or test coordinate extraction on a synthetic artery volume:
```bash
python Transformer/src/extract_coords.py \
    --input_synthetic \
    --output_json Transformer/artifacts/artery_centerlines.json
```
Output JSON format:
```json
{
  "total_centerline_points": 120,
  "centerline_points": [
    {
      "point_id": 0,
      "voxel_index": {"z": 10, "y": 48, "x": 63},
      "physical_coord_mm": {"X": -49.6, "Y": -61.6, "Z": 10.0}
    }
  ]
}
```
