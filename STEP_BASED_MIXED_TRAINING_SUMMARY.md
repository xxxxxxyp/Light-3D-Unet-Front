# Step-Based Mixed Training Implementation Summary

## Overview

This implementation adds a new **step-based mixed training mode** that ensures each epoch:
1. Completes **one full pass** through all FL batches
2. Adds **DLBCL batches** according to a configurable step ratio

This addresses the user requirement: **"每一轮都能将FL的数据训练一遍，然后按照比例加入DLBCL随机的数据"** (Each round trains on all FL data once, then adds DLBCL data according to a ratio).

## Key Features

### 1. Guaranteed FL Coverage
- Every epoch iterates through **all FL batches** exactly once
- No probabilistic sampling for FL data
- FL data is never skipped or under-represented

### 2. Configurable DLBCL Steps
- DLBCL steps calculated by: `dlbcl_steps = round(FL_batches × dlbcl_steps_ratio)`
- Supports exact step count override: `dlbcl_steps: 150`
- DLBCL batches are randomly sampled (shuffle enabled)
- DLBCL loader cycles if more steps needed than available batches

### 3. Two-Stage Training Loop
- **Stage 1**: Full pass through FL batches
- **Stage 2**: DLBCL steps (randomly sampled, can cycle)
- Separate progress bars for each stage
- Monotonic global_step increments across both stages

### 4. Detailed Logging
- Separate tracking of FL and DLBCL losses
- Per-domain step counts and ratios
- Combined weighted loss for overall monitoring
- TensorBoard metrics for both unified and domain-specific views

## Configuration

### Enable Step-Based Mode

```yaml
training:
  mixed_domains:
    enabled: true
    mode: fl_epoch_plus_dlbcl      # Step-based mode
    dlbcl_steps_ratio: 1.0         # DLBCL steps = FL batches × ratio
    dlbcl_steps: null              # Optional: exact step count override
```

### Common Configurations

**Equal FL and DLBCL steps (1:1 ratio):**
```yaml
dlbcl_steps_ratio: 1.0  # 100 FL batches → 100 DLBCL steps
```

**FL-heavy training (2:1 ratio):**
```yaml
dlbcl_steps_ratio: 0.5  # 100 FL batches → 50 DLBCL steps
```

**DLBCL-heavy training (1:2 ratio):**
```yaml
dlbcl_steps_ratio: 2.0  # 100 FL batches → 200 DLBCL steps
```

**Exact step count:**
```yaml
dlbcl_steps: 150  # Exactly 150 DLBCL steps per epoch (ignores ratio)
```

**FL-only (no DLBCL):**
```yaml
dlbcl_steps_ratio: 0.0  # 0 DLBCL steps
```

## Training Output Example

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
  100%|██████████| 100/100 [01:23<00:00,  1.20it/s, loss=0.3456]
  
  Stage 2: DLBCL training (100 steps)
  100%|██████████| 100/100 [01:25<00:00,  1.18it/s, loss=0.3234]
  
  Domain Statistics:
    FL steps: 100 (50.00%), avg loss: 0.3456
    DLBCL steps: 100 (50.00%), avg loss: 0.3234
    Total steps: 200, combined loss: 0.3345
  
  Val Lesion-wise Recall: 0.7543 (best threshold: 0.30)
  Val Voxel-wise DSC (macro): 0.6789
  ...
```

## TensorBoard Metrics

### Unified Metrics
- `Loss/train_step` - All training steps in one continuous graph
- `Loss/combined` - Weighted average of FL and DLBCL losses per epoch

### Domain-Specific Metrics
- `Loss/fl_step` - FL step losses
- `Loss/dlbcl_step` - DLBCL step losses
- `Loss/fl_avg` - Average FL loss per epoch
- `Loss/dlbcl_avg` - Average DLBCL loss per epoch

### Domain Statistics
- `Domain/fl_steps` - FL steps per epoch
- `Domain/dlbcl_steps` - DLBCL steps per epoch
- `Domain/fl_ratio` - FL step ratio (FL steps / total steps)
- `Domain/dlbcl_ratio` - DLBCL step ratio (DLBCL steps / total steps)

## Implementation Details

### Dataset Creation
- Two separate `PatchDataset` instances: one for FL, one for DLBCL
- Each with independent domain filtering
- Different random seeds (FL uses `seed`, DLBCL uses `seed + 1`)

### Data Loaders
- Two separate DataLoaders with independent shuffling
- FL loader: iterate completely each epoch
- DLBCL loader: sample `dlbcl_steps` batches, cycling if needed

### Global Step Calculation
```python
base_global_step = epoch * (fl_batches + dlbcl_steps)
# FL stage:   base_global_step + batch_idx
# DLBCL stage: base_global_step + fl_steps + step_idx
```

This ensures monotonic increment across epochs and stages.

### Validation
- Always FL-only, regardless of training mode
- Ensures consistent evaluation across different training configurations

## Backward Compatibility

### Old Probabilistic Mode
Still available via `mode: probabilistic`:
```yaml
training:
  mixed_domains:
    enabled: true
    mode: probabilistic
    fl_ratio: 0.5  # 50% FL samples per batch
```

### Standard Training
Works exactly as before when mixed training is disabled:
```yaml
training:
  mixed_domains:
    enabled: false
```

## Files Changed

1. **models/dataset.py**
   - Modified `get_data_loader()` to support step-based mode
   - Returns dict with `fl_loader` and `dlbcl_loader` for new mode
   - Maintains backward compatibility with old mode

2. **scripts/train.py**
   - Added `_train_epoch_step_based()` method
   - Two-stage training loop implementation
   - Separate loss tracking and logging
   - Updated initialization to handle new loader structure

3. **configs/unet_mixed_fl_dlbcl.yaml**
   - Set `mode: fl_epoch_plus_dlbcl`
   - Added `dlbcl_steps_ratio: 1.0`
   - Added `dlbcl_steps: null`

4. **configs/unet_fl70.yaml**
   - Added schema comments for new fields
   - Set default mode to `fl_epoch_plus_dlbcl`

5. **Documentation**
   - Updated `MIXED_TRAINING_GUIDE.md`
   - Updated `FEATURE_MIXED_TRAINING.md`

6. **Tests**
   - Extended `test_mixed_training.py`
   - Created `test_step_based_config.py`

## Testing

All tests pass:
```bash
$ python test_step_based_config.py
✓ All configuration tests passed successfully!
```

Configuration validation:
- ✓ Domain config schema valid
- ✓ Mixed training config schema valid
- ✓ Mode validation (fl_epoch_plus_dlbcl/probabilistic)
- ✓ DLBCL steps computation tests
- ✓ Python syntax checks passed
- ✓ YAML configs valid

Security:
- ✓ CodeQL scan passed: 0 alerts
- ✓ Code review feedback addressed

## Usage

1. Use the example config:
   ```bash
   python scripts/train.py --config configs/unet_mixed_fl_dlbcl.yaml
   ```

2. Or modify your own config to enable step-based mixed training.

3. Monitor training via console output or TensorBoard:
   ```bash
   tensorboard --logdir logs/tensorboard
   ```

## Summary

This implementation provides:
- ✅ Guaranteed full FL traversal each epoch
- ✅ Precise control over FL/DLBCL training balance
- ✅ Step-based scheduling (not probabilistic)
- ✅ Separate loss tracking per domain
- ✅ Monotonic global_step for TensorBoard
- ✅ FL-only validation for consistency
- ✅ Backward compatibility with old mode
- ✅ Comprehensive documentation and tests
- ✅ Zero security vulnerabilities

The implementation fulfills the user requirement while maintaining code quality and backward compatibility.
