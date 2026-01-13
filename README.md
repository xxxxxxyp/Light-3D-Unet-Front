# Lightweight 3D U-Net for FL Lesion Recall

This repository implements a lightweight 3D U-Net for PET-only lesion candidate detection using Follicular Lymphoma (FL) data. The system is designed for high recall with coarse masks and bounding boxes.

## Project Overview

**Core Objective**: Build and validate a lightweight 3D U-Net for lesion candidate recall using FL-70% data.

**Key Features**:
- Lightweight architecture: 16→32→64→128 channels with grouped/depthwise separable convolutions
- Path B preprocessing: Preserves original 4×4×4mm spacing
- Focal Tversky Loss for imbalanced segmentation
- Class-balanced patch sampling
- Comprehensive data augmentation
- Lesion-wise recall-focused evaluation

**Data Split**:
- Training: FL-70% (86 cases)
- Validation: FL-15% (19 cases)
- Test: FL-15% (18 cases) - Black box, not to be used

## Project Structure

```
Light-3D-Unet-Front/
├── configs/
│   └── unet_fl70.yaml          # Main configuration file
├── data/
│   ├── raw/                    # Raw data (user-provided)
│   │   ├── images/             # PET images
│   │   └── labels/             # Lesion masks
│   ├── processed/              # Preprocessed data with metadata
│   │   ├── images/             # Preprocessed PET images
│   │   ├── labels/             # Preprocessed lesion masks
│   │   ├── body_masks/         # Body masks (exclude air/background)
│   │   └── metadata/           # Per-case metadata JSON files
│   └── splits/                 # Train/val/test split files
├── models/
│   ├── checkpoints/            # Training checkpoints
│   ├── best_model.pth          # Best model (validation recall)
│   ├── unet3d.py              # Model architecture
│   ├── losses.py              # Loss functions
│   ├── dataset.py             # Dataset and data loader
│   └── metrics.py             # Evaluation metrics
├── logs/
│   ├── tensorboard/           # TensorBoard logs
│   └── training_history.json  # Training history
├── inference/
│   ├── prob_maps/             # Probability maps (NIfTI)
│   ├── bboxes/                # Candidate bounding boxes (JSON)
│   ├── metrics.csv            # Evaluation metrics
│   └── detailed_results.json  # Detailed results
├── scripts/
│   ├── split_dataset.py       # Data splitting
│   ├── preprocess_data.py     # Data preprocessing
│   ├── train.py               # Training script
│   ├── inference.py           # Inference script
│   └── evaluate.py            # Evaluation script
├── requirements.txt            # Python dependencies
└── README.md                  # This file
```

## Installation

1. **Clone the repository**:
```bash
git clone <repository-url>
cd Light-3D-Unet-Front
```

2. **Create a virtual environment** (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**:
```bash
pip install -r requirements.txt
```

4. **Verify installation**:
```bash
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
python -c "import nibabel; print(f'NiBabel: {nibabel.__version__}')"
```

## Data Preparation

### Step 1: Organize Raw Data

Place your FL data in the `data/raw/` directory with the following structure:

```
data/raw/
├── images/
│   ├── 0001_0000.nii.gz
│   ├── 0002_0000.nii.gz
│   └── ...
└── labels/
    ├── 0001.nii.gz
    ├── 0002.nii.gz
    └── ...
```

**Requirements**:
- Images must be in NIfTI format (.nii or .nii.gz)
- Spacing: 4×4×4mm (will be verified)
- SUV values must be pre-calculated
- Labels must be binary (0=background, 1=lesion)
- File naming: Image files should be `{case_id}_*.nii.gz`, label files should be `{case_id}.nii.gz`

### Step 2: Split Dataset

Split the 123 FL cases into train (70%), val (15%), test (15%):

```bash
python scripts/split_dataset.py \
    --data_root data/raw \
    --output_dir data/splits \
    --seed 42
```

This creates:
- `data/splits/train_list.txt` (86 cases)
- `data/splits/val_list.txt` (19 cases)
- `data/splits/test_list.txt` (18 cases)
- `data/split_manifest.json` (metadata)

### Step 3: Preprocess Data

Apply Path B preprocessing (intensity clipping and normalization):

```bash
python scripts/preprocess_data.py \
    --config configs/unet_fl70.yaml \
    --raw_dir data/raw \
    --processed_dir data/processed \
    --splits_dir data/splits \
    --split all  # Process train and val only
```

This applies:
- 0.5%-99.5% percentile intensity clipping
- Linear normalization to [0, 1]
- Preserves original 4×4×4mm spacing (no resampling)
- Generates metadata JSON files in `data/processed/metadata/{case_id}.json`
- Outputs files in flat structure matching raw data (images/ and labels/ subdirectories)

**Body Mask Generation** (enabled by default):

The preprocessing automatically generates body masks to exclude air/background regions outside the patient body:
- Body masks are saved in `data/processed/body_masks/{case_id}.nii.gz`
- Generated using morphological operations on normalized PET images:
  1. Thresholding (default: 0.02 on 0-1 normalized scale)
  2. Binary closing to fill holes
  3. Keep largest connected component to remove table/noise
  4. Dilation by 3-5 voxels to ensure full body coverage
- Masks are used during training to constrain background patch sampling
- Applied to validation/inference probability maps to reduce false positives in air

To disable body mask generation, set `data.body_mask.enabled: false` in your config file.

## Training

### Basic Training

Train the model with default configuration:

```bash
python scripts/train.py \
    --config configs/unet_fl70.yaml \
    --data_dir data/processed \
    --splits_dir data/splits
```

### Training Configuration

Key hyperparameters (defined in `configs/unet_fl70.yaml`):

- **Model**: Lightweight 3D U-Net (16→32→64→128 channels)
- **Loss**: Focal Tversky Loss (α=0.7, β=0.3, γ=0.75)
- **Optimizer**: AdamW (lr=1e-4, weight_decay=1e-5)
- **Scheduler**: CosineAnnealingLR (T_max=200)
- **Batch Size**: 2 (increase to 4 if GPU memory allows)
- **Patch Size**: 48×48×48 voxels (~192mm at 4mm spacing)
- **Epochs**: 200 (with early stopping patience=20)
- **Augmentation**: Random flip, rotation (±15°), scale (±10%), intensity shift (±10%), Gaussian noise

### Monitoring Training

**TensorBoard**:
```bash
tensorboard --logdir logs/tensorboard
```

**Check Training History**:
```bash
cat logs/training_history.json
```

### Model Selection

The best model is selected based on **validation lesion-wise recall**:
- Primary metric: Lesion-wise Recall@Lesion
- Tie-breaker: Voxel-wise DSC (if recall difference ≤ 1%)
- Saved to: `models/best_model.pth`

## Inference

Generate probability maps and candidate bounding boxes:

```bash
python scripts/inference.py \
    --config configs/unet_fl70.yaml \
    --model models/best_model.pth \
    --data_dir data/processed \
    --split_file data/splits/val_list.txt
```

**Outputs**:
- Probability maps: `inference/prob_maps/{case_id}_prob.nii.gz`
- Bounding boxes: `inference/bboxes/{case_id}_bboxes.json`

**BBox JSON Format**:
```json
{
  "case_id": "FL_001",
  "processing_path": "B",
  "orig_spacing": [4.0, 4.0, 4.0],
  "threshold": 0.3,
  "num_candidates": 5,
  "candidates": [
    {
      "mask_id": 1,
      "bbox_voxel": [10, 25, 30, 50, 40, 60],
      "bbox_mm": [40.0, 100.0, 120.0, 200.0, 160.0, 240.0],
      "volume_cc": 1.5,
      "confidence": 0.85
    }
  ]
}
```

## Evaluation

Evaluate model performance on validation set:

```bash
python scripts/evaluate.py \
    --config configs/unet_fl70.yaml \
    --prob_maps_dir inference/prob_maps \
    --data_dir data/processed \
    --split_file data/splits/val_list.txt \
    --output_dir inference
```

**Metrics**:
- **Lesion-wise Recall**: Primary metric (target ≥ 80%)
- **Lesion-wise Precision**: Secondary metric
- **Voxel-wise DSC**: Segmentation quality
- **FP per case**: False positives per case

**Matching Criteria**:
- IoU ≥ 0.1, OR
- Center distance ≤ 10mm

**Threshold Sensitivity Analysis**:
Evaluates at multiple thresholds [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]

**Outputs**:
- `inference/metrics.csv`: Summary metrics per threshold
- `inference/detailed_results.json`: Per-case results

## Configuration

All settings are defined in `configs/unet_fl70.yaml`. Key sections:

### Experiment
```yaml
experiment:
  name: "FL70_Lightweight_3DUNet"
  seed: 42
  processing_path: "B"
```

### Data
```yaml
data:
  spacing:
    original: [4.0, 4.0, 4.0]
    target: [4.0, 4.0, 4.0]  # No resampling
  patch_size: [48, 48, 48]
  volume_threshold:
    train_cc: 0.1   # Filter <0.1cc during training
    inference_cc: 0.5  # Filter <0.5cc during inference
  bbox_expansion_mm: 10.0  # Physical expansion
```

### Model
```yaml
model:
  encoder_channels: [16, 32, 64, 128]
  use_grouped_conv: true
  use_depthwise_separable: true
  dropout_p: 0.1
```

### Loss
```yaml
loss:
  name: "FocalTverskyLoss"
  alpha: 0.7  # False negative weight (prioritize recall)
  beta: 0.3   # False positive weight
  gamma: 0.75 # Focal parameter
```

## Reproducibility

**Fixed Random Seeds**:
- All random operations use seed=42
- Includes: NumPy, PyTorch, data splitting, augmentation

**Environment Information**:
```bash
pip freeze > environment.txt
```

**Git Commit Hash**:
Record the current commit hash in your experiment report.

## Data Isolation Rules

**MUST NOT**:
- Use DLBCL data or any external datasets
- Use FL test set (black box) for training or validation
- Recompute SUV values

**MUST**:
- Use only FL-70% for training
- Use only FL-15% (validation) for model selection
- Preserve 4×4×4mm spacing (Path B)

## Expected Performance

**Target** (discussive):
- Lesion-wise Recall ≥ 80% on validation set

**If target not met**:
Document in experiment report with:
1. Analysis of failure cases
2. 2-3 actionable improvement suggestions
3. DO NOT implement changes without approval

## Troubleshooting

### Out of Memory (OOM)

**Solution 1**: Reduce batch size
```yaml
training:
  batch_size: 1  # Reduce from 2
```

**Solution 2**: Reduce patch size
```yaml
data:
  patch_size: [32, 32, 32]  # Reduce from 48
```

### Training Instability

**Solution**: Use combined loss
```yaml
loss:
  use_combined_loss: true
  combined_loss_weights:
    focal_tversky: 0.8
    bce: 0.2
```

### Low Recall

**Possible causes**:
1. Threshold too high → Try sensitivity analysis
2. Insufficient lesion patches → Increase `lesion_patch_ratio`
3. Class imbalance → Adjust Focal Tversky α/β

## Deliverables

### Milestone 1: Data Preparation (2-4 days)
- [ ] `data/splits/` with train/val/test lists
- [ ] `data/split_manifest.json`
- [ ] `data/processed/` with preprocessed data
- [ ] Metadata JSON for each case

### Milestone 2: Initial Training (4-7 days)
- [ ] Training logs
- [ ] Multiple checkpoints
- [ ] TensorBoard logs

### Milestone 3: Hyperparameter Tuning (3-7 days)
- [ ] `models/best_model.pth`
- [ ] Validation report with metrics
- [ ] Training history

### Milestone 4: Inference & Report (2-3 days)
- [ ] Probability maps (NIfTI)
- [ ] Candidate bboxes (JSON)
- [ ] `inference/metrics.csv`
- [ ] Experiment report (PDF)

## Experiment Report

The final experiment report must include:

1. **Data Preparation**
   - Metadata JSON examples
   - Clip values and normalization parameters
   - Patch size and physical coverage
   - Voxel threshold calculations

2. **Training**
   - Training curves (loss, recall, DSC)
   - Model selection rationale
   - Best model epoch and metrics

3. **Validation**
   - Metrics table (recall, precision, DSC, FP/case)
   - Threshold sensitivity analysis
   - Per-case results

4. **Analysis**
   - Success/failure case analysis
   - Problems encountered and solutions
   - Next steps and improvement suggestions

## Citation

If you use this code, please cite:

```
[To be added]
```

## License

[To be specified]

## Contact

For questions or issues, please contact:
[To be specified]