# Mixed FL + DLBCL Training Guide

This guide explains how to use the mixed domain training feature to train on both Follicular Lymphoma (FL) and Diffuse Large B-Cell Lymphoma (DLBCL) data while validating exclusively on FL cases.

## Overview

The mixed training feature supports two modes:

1. **Step-based mode (NEW, RECOMMENDED)**: `fl_epoch_plus_dlbcl`
   - Each epoch completes **one full pass through all FL batches**
   - Then adds **DLBCL batches according to a configurable step ratio**
   - Guarantees FL data is fully traversed each epoch
   - Provides precise control over FL/DLBCL training balance

2. **Probabilistic mode (OLD, backward compatibility)**: `probabilistic`
   - Stochastic sampling per batch based on FL ratio
   - May not guarantee full FL traversal per epoch
   - Epoch length = sum of FL + DLBCL dataset sizes

**Default mode**: `fl_epoch_plus_dlbcl` (step-based)

## Configuration

### 1. Domain Ranges

Configure the case ID prefix ranges in your config:

```yaml
data:
  domains:
    fl_prefix_max: 122        # FL cases: 0000-0122
    dlbcl_prefix_min: 1000    # DLBCL cases: 1000-1422
    dlbcl_prefix_max: 1422
```

### 2. Enable Step-Based Mixed Training (Recommended)

```yaml
training:
  mixed_domains:
    enabled: true
    mode: fl_epoch_plus_dlbcl  # Step-based mode
    dlbcl_steps_ratio: 1.0     # DLBCL steps = round(FL_batches * ratio)
    dlbcl_steps: null          # Optional: override with exact step count
```

**How it works:**
- Each epoch first trains on **all FL batches** (e.g., 100 batches)
- Then trains on **`round(FL_batches * dlbcl_steps_ratio)`** DLBCL batches
- Example: 100 FL batches × 1.0 ratio = 100 DLBCL steps → 200 total steps per epoch

**Key parameters:**
- `dlbcl_steps_ratio`: Multiplier for DLBCL steps relative to FL batches
  - `0.0` = FL-only (no DLBCL steps)
  - `0.5` = half as many DLBCL steps as FL batches
  - `1.0` = same number of DLBCL steps as FL batches
  - `2.0` = twice as many DLBCL steps as FL batches
- `dlbcl_steps`: Optional override to set exact DLBCL step count (ignores ratio)

### 3. Old Probabilistic Mode (Backward Compatibility)

```yaml
training:
  mixed_domains:
    enabled: true
    mode: probabilistic  # Old stochastic mode
    fl_ratio: 0.5        # 50% FL samples per batch
```

**Note:** This mode samples stochastically and doesn't guarantee full FL traversal.

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
├── body_masks/           # Body masks (shared for FL and DLBCL)
│   ├── 0000.nii.gz       # FL body mask
│   ├── 0001.nii.gz       # FL body mask
│   ├── 1000.nii.gz       # DLBCL body mask
│   └── ...
└── splits/
    ├── train_list.txt    # Contains both FL and DLBCL case IDs
    └── val_list.txt      # Can contain both, but will be filtered to FL-only
```

**Note on Body Masks:**
- Body masks are generated during preprocessing and are shared between FL and DLBCL cases
- They constrain background patch sampling to the patient body region
- This prevents sampling air-only patches and reduces false positives
- See main README for body mask configuration options

## Training

Run training as usual:

```bash
python scripts/train.py --config configs/unet_mixed_fl_dlbcl.yaml
```

## Monitoring

### Console Output (Step-Based Mode)

During training with `fl_epoch_plus_dlbcl` mode, you'll see two-stage training:

```
*** Step-Based Mixed Domain Training Enabled ***
  Mode: fl_epoch_plus_dlbcl
  FL batches per epoch: 100
  DLBCL steps per epoch: 100
  DLBCL steps ratio: 1.00
  Total steps per epoch: 200
  Validation: FL-only
  Val cases: 18 FL cases

Epoch 1/200
  Stage 1: FL training (100 batches)
  [FL progress bar]
  
  Stage 2: DLBCL training (100 steps)
  [DLBCL progress bar]
  
  Domain Statistics:
    FL steps: 100 (50.00%), avg loss: 0.3456
    DLBCL steps: 100 (50.00%), avg loss: 0.3234
    Total steps: 200, combined loss: 0.3345
  Train Loss: 0.3345
  Val Lesion-wise Recall: 0.7543 (best threshold: 0.30)
  ...
```

### Console Output (Probabilistic Mode)

With `probabilistic` mode:

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

**Step-based mode metrics:**
- `Domain/fl_steps` - FL steps per epoch
- `Domain/dlbcl_steps` - DLBCL steps per epoch
- `Domain/fl_ratio` - FL step ratio (FL steps / total steps)
- `Domain/dlbcl_ratio` - DLBCL step ratio (DLBCL steps / total steps)
- `Loss/fl_avg` - Average FL loss per epoch
- `Loss/dlbcl_avg` - Average DLBCL loss per epoch
- `Loss/combined` - Combined weighted loss per epoch
- `Loss/fl_step` - Per-step FL loss
- `Loss/dlbcl_step` - Per-step DLBCL loss

**Probabilistic mode metrics:**
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

- When `mixed_domains.enabled: false`, training works exactly as before
- Old `probabilistic` mode available for backward compatibility
- No changes required to existing datasets or split files
- All existing configurations continue to work

## Example Configurations

### Equal FL and DLBCL Steps (Step-Based)

```yaml
training:
  mixed_domains:
    enabled: true
    mode: fl_epoch_plus_dlbcl
    dlbcl_steps_ratio: 1.0  # Same steps as FL
```

### FL-Heavy Training (Step-Based)

More FL exposure (e.g., 2:1 ratio):

```yaml
training:
  mixed_domains:
    enabled: true
    mode: fl_epoch_plus_dlbcl
    dlbcl_steps_ratio: 0.5  # Half as many DLBCL steps as FL batches
```

### DLBCL-Heavy Training (Step-Based)

More DLBCL exposure (e.g., 1:2 ratio):

```yaml
training:
  mixed_domains:
    enabled: true
    mode: fl_epoch_plus_dlbcl
    dlbcl_steps_ratio: 2.0  # Twice as many DLBCL steps as FL batches
```

### Exact DLBCL Step Count (Step-Based)

```yaml
training:
  mixed_domains:
    enabled: true
    mode: fl_epoch_plus_dlbcl
    dlbcl_steps: 150  # Exactly 150 DLBCL steps per epoch
```

### FL-Only Training

```yaml
training:
  mixed_domains:
    enabled: true
    mode: fl_epoch_plus_dlbcl
    dlbcl_steps_ratio: 0.0  # No DLBCL steps
```

Or simply:

```yaml
training:
  mixed_domains:
    enabled: false  # Disabled
```

## Implementation Details

### Step-Based Mode (`fl_epoch_plus_dlbcl`)

- Creates two separate `PatchDataset` instances: one for FL, one for DLBCL
- Each has its own DataLoader with independent shuffling
- Each epoch:
  1. Iterates through **all batches** of FL loader (full pass)
  2. Samples `dlbcl_steps` batches from DLBCL loader (cycling if needed)
- `global_step` increments monotonically: FL steps first, then DLBCL steps
- Tracks and logs per-domain losses and step counts

### Probabilistic Mode (`probabilistic`)

- Uses `MixedPatchDataset` which wraps both FL and DLBCL datasets
- Each `__getitem__` call samples from FL with probability `fl_ratio`
- Dataset length = sum of FL + DLBCL dataset lengths
- Sample counts tracked but not guaranteed to match exact ratio

### Domain Filtering

- Case IDs are filtered based on the first 4 digits of the case ID
- FL cases: prefix ≤ `fl_prefix_max` (default: 122)
- DLBCL cases: `dlbcl_prefix_min` ≤ prefix ≤ `dlbcl_prefix_max` (default: 1000-1422)

### Class Balancing

- Class-balanced sampling (lesion vs background) is maintained within each domain
- The `lesion_patch_ratio` applies independently to both FL and DLBCL patches

## Troubleshooting
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
