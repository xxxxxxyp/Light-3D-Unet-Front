# Mixed FL + DLBCL Training Guide

This guide explains how to use the mixed domain training feature to train on both Follicular Lymphoma (FL) and Diffuse Large B-Cell Lymphoma (DLBCL) data while validating exclusively on FL cases.

## Overview

The mixed training feature allows you to:
- Train on a combined dataset of FL and DLBCL cases
- Control the sampling ratio between FL and DLBCL during training
- Keep validation strictly on FL cases for consistent evaluation
- Track domain statistics during training

## Configuration

### 1. Domain Ranges

Configure the case ID prefix ranges in `configs/unet_fl70.yaml`:

```yaml
data:
  domains:
    fl_prefix_max: 122        # FL cases: 0000-0122
    dlbcl_prefix_min: 1000    # DLBCL cases: 1000-1422
    dlbcl_prefix_max: 1422
```

### 2. Enable Mixed Training

Enable mixed domain training and set sampling ratios:

```yaml
training:
  mixed_domains:
    enabled: true      # Set to true to enable mixed training
    fl_ratio: 0.5      # 50% FL samples
    dlbcl_ratio: 0.5   # 50% DLBCL samples (informational, not used directly)
```

**Note:** The `fl_ratio` determines the proportion of FL samples. DLBCL samples make up the remainder.

## Dataset Structure

Ensure your data follows this structure:

```
data/
├── images/
│   ├── 0000_0000.nii.gz  # FL case
│   ├── 0001_0000.nii.gz  # FL case
│   ├── 1000_0000.nii.gz  # DLBCL case
│   └── ...
├── labels/
│   ├── 0000.nii.gz       # FL label
│   ├── 0001.nii.gz       # FL label
│   ├── 1000.nii.gz       # DLBCL label
│   └── ...
└── splits/
    ├── train_list.txt    # Contains both FL and DLBCL case IDs
    └── val_list.txt      # Can contain both, but will be filtered to FL-only
```

## Training

Run training as usual:

```bash
python scripts/train.py --config configs/unet_fl70.yaml
```

## Monitoring

### Console Output

During training, you'll see domain statistics printed each epoch:

```
Epoch 1/200
  Domain Statistics:
    FL samples: 512 (50.00%)
    DLBCL samples: 512 (50.00%)
    Total samples: 1024
  Train Loss: 0.3456
  ...
```

### TensorBoard

View domain statistics in TensorBoard:

```bash
tensorboard --logdir logs/tensorboard
```

Available metrics:
- `Domain/fl_samples` - Number of FL samples per epoch
- `Domain/dlbcl_samples` - Number of DLBCL samples per epoch
- `Domain/fl_ratio` - Actual FL sampling ratio
- `Domain/dlbcl_ratio` - Actual DLBCL sampling ratio

## Validation

Validation is **always performed on FL cases only**, regardless of what's in `val_list.txt`:

- The validation loader automatically filters to FL cases when mixed training is enabled
- This ensures consistent evaluation across different training configurations
- Early stopping and best model selection are based on FL validation performance

## Backward Compatibility

The feature is fully backward compatible:

- When `mixed_domains.enabled: false` (default), training works exactly as before
- No changes required to existing datasets or split files
- All existing configurations continue to work

## Example Configurations

### 50/50 Mixed Training

```yaml
training:
  mixed_domains:
    enabled: true
    fl_ratio: 0.5
```

### FL-Heavy Training (70% FL, 30% DLBCL)

```yaml
training:
  mixed_domains:
    enabled: true
    fl_ratio: 0.7
```

### DLBCL-Heavy Training (30% FL, 70% DLBCL)

```yaml
training:
  mixed_domains:
    enabled: true
    fl_ratio: 0.3
```

### FL-Only Training (Standard)

```yaml
training:
  mixed_domains:
    enabled: false
```

## Implementation Details

### Domain Filtering

- Case IDs are filtered based on the first 4 digits of the case ID
- FL cases: prefix ≤ `fl_prefix_max` (default: 122)
- DLBCL cases: `dlbcl_prefix_min` ≤ prefix ≤ `dlbcl_prefix_max` (default: 1000-1422)

### Sampling Strategy

- `MixedPatchDataset` creates two separate `PatchDataset` instances (FL and DLBCL)
- Each batch samples from FL with probability `fl_ratio`, otherwise from DLBCL
- Both datasets use the same patch extraction and augmentation settings
- Sample counts are tracked and logged each epoch

### Class Balancing

- Class-balanced sampling (lesion vs background) is maintained within each domain
- The `lesion_patch_ratio` applies independently to both FL and DLBCL patches

## Troubleshooting

### No FL/DLBCL cases found

**Problem:** `MixedPatchDataset: FL cases=0, DLBCL cases=...`

**Solution:** Check that:
1. Your case IDs follow the expected format (4-digit prefix)
2. The domain ranges in config match your actual case ID ranges
3. Your `train_list.txt` contains cases from both domains

### Imbalanced sampling

**Problem:** Actual ratio differs significantly from configured ratio

**Solution:**
- This is expected due to stochastic sampling
- Over a full epoch, the ratio should approximate the configured value
- Check the epoch-level statistics, not batch-level

### Validation shows zero cases

**Problem:** No validation cases after FL filtering

**Solution:** 
- Ensure `val_list.txt` contains FL cases (case IDs ≤ 122)
- Check the domain configuration ranges
