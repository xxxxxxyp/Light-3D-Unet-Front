# Feature: Mixed FL + DLBCL Training

## Quick Start

To enable mixed domain training on FL + DLBCL data with the new step-based mode:

1. Use the provided example config:
   ```bash
   python scripts/train.py --config configs/unet_mixed_fl_dlbcl.yaml
   ```

2. Or enable it in your own config:
   ```yaml
   training:
     mixed_domains:
       enabled: true
       mode: fl_epoch_plus_dlbcl  # Step-based (recommended)
       dlbcl_steps_ratio: 1.0     # DLBCL steps = FL batches × ratio
   ```

## What It Does

**New Step-Based Mode (`fl_epoch_plus_dlbcl`):**
- **Trains** on FL with full epoch pass, then adds DLBCL steps by ratio
- **Guarantees** FL data is fully traversed each epoch
- **Provides** precise control over FL/DLBCL balance
- **Tracks** separate losses for FL and DLBCL stages

**Legacy Probabilistic Mode:**
- Stochastic sampling per batch (backward compatibility)
- See [MIXED_TRAINING_GUIDE.md](MIXED_TRAINING_GUIDE.md) for details

**Both modes:**
- **Validate** exclusively on FL cases for consistent evaluation
- **Log** domain statistics to console and TensorBoard each epoch
- **Maintain** backward compatibility

## Expected Output (Step-Based Mode)

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
  Stage 2: DLBCL training (100 steps)
  
  Domain Statistics:
    FL steps: 100 (50.00%), avg loss: 0.3456
    DLBCL steps: 100 (50.00%), avg loss: 0.3234
    Total steps: 200, combined loss: 0.3345
  Val Lesion-wise Recall: 0.7543 (best threshold: 0.30)
  ...
```

## Documentation

- **Usage Guide**: See [MIXED_TRAINING_GUIDE.md](MIXED_TRAINING_GUIDE.md)
- **Implementation Details**: See [IMPLEMENTATION_NOTES.md](IMPLEMENTATION_NOTES.md)
- **Example Configs**: 
  - `configs/unet_fl70.yaml` - FL-only (backward compatible)
  - `configs/unet_mixed_fl_dlbcl.yaml` - Step-based mixed training

## Requirements

- FL cases: 0000-0122 (case ID prefix ≤ 122)
- DLBCL cases: 1000-1422 (case ID prefix 1000-1422)
- Both domains in `train_list.txt`
- Standard data structure: `data/images/`, `data/labels/`

## Key Features

✅ **Step-based schedule**: Full FL epoch + configurable DLBCL steps  
✅ **Guaranteed FL coverage**: FL data fully traversed each epoch  
✅ **Precise control**: Configure exact FL/DLBCL ratio by steps  
✅ **Separate tracking**: FL and DLBCL losses logged independently  
✅ **FL-only validation**: Always evaluate on FL for consistency  
✅ **TensorBoard metrics**: Visualize domain statistics over time  
✅ **Backward compatible**: Supports old probabilistic mode  
✅ **No dataset changes**: Uses existing file structure  

## Configuration Options

### Domain Ranges (in `data.domains`)
```yaml
data:
  domains:
    fl_prefix_max: 122        # FL case ID range: 0000-0122
    dlbcl_prefix_min: 1000    # DLBCL case ID range: 1000-1422
    dlbcl_prefix_max: 1422
```

### Step-Based Mixed Training (in `training.mixed_domains`)
```yaml
training:
  mixed_domains:
    enabled: true
    mode: fl_epoch_plus_dlbcl      # Step-based mode
    dlbcl_steps_ratio: 1.0         # DLBCL steps = FL batches × ratio
    dlbcl_steps: null              # Optional: exact step count override
```

### Probabilistic Mixed Training (backward compatibility)
```yaml
training:
  mixed_domains:
    enabled: true
    mode: probabilistic            # Old stochastic mode
    fl_ratio: 0.5                  # Proportion of FL samples (0.0-1.0)
```

## Example Ratios (Step-Based Mode)

### Equal FL and DLBCL Steps
```yaml
mixed_domains:
  enabled: true
  mode: fl_epoch_plus_dlbcl
  dlbcl_steps_ratio: 1.0  # Same steps as FL
```

### FL-Heavy (2:1 ratio)
```yaml
mixed_domains:
  enabled: true
  mode: fl_epoch_plus_dlbcl
  dlbcl_steps_ratio: 0.5  # Half as many DLBCL steps
```

### DLBCL-Heavy (1:2 ratio)
```yaml
mixed_domains:
  enabled: true
  mode: fl_epoch_plus_dlbcl
  dlbcl_steps_ratio: 2.0  # Twice as many DLBCL steps
```

### Exact Step Count
```yaml
mixed_domains:
  enabled: true
  mode: fl_epoch_plus_dlbcl
  dlbcl_steps: 150  # Exactly 150 DLBCL steps
```

### FL-Only
```yaml
mixed_domains:
  enabled: false
```

## TensorBoard Metrics

View domain statistics:
```bash
tensorboard --logdir logs/tensorboard
```

**Step-based mode:**
- `Domain/fl_steps` - FL steps per epoch
- `Domain/dlbcl_steps` - DLBCL steps per epoch
- `Domain/fl_ratio` - FL step ratio
- `Domain/dlbcl_ratio` - DLBCL step ratio
- `Loss/fl_avg` - Average FL loss
- `Loss/dlbcl_avg` - Average DLBCL loss
- `Loss/combined` - Combined loss

**Probabilistic mode:**
- `Domain/fl_samples` - FL sample count per epoch
- `Domain/dlbcl_samples` - DLBCL sample count per epoch
- `Domain/fl_ratio` - Actual FL sampling ratio
- `Domain/dlbcl_ratio` - Actual DLBCL sampling ratio

## Implementation

- **Dataset Filtering**: `filter_cases_by_domain()` in `models/dataset.py`
- **Step-Based Mode**: Separate FL and DLBCL loaders with two-stage training
- **Probabilistic Mode**: `MixedPatchDataset` class with stochastic sampling
- **Data Loader**: `get_data_loader()` handles all modes
- **Training Loop**: Two-stage epoch for step-based mode

## Testing

Run the included test suite:
```bash
python test_mixed_training.py
```

Tests include:
- Domain filtering logic
- Config schema validation (including new fields)
- DLBCL steps computation
- Import verification

## Security

✅ CodeQL scan passed: 0 alerts  
✅ Code review passed: All feedback addressed  
✅ No security vulnerabilities introduced  

## Files Changed

- `models/dataset.py` - Core implementation (+251 lines)
- `scripts/train.py` - Training integration (+46 lines)
- `configs/unet_fl70.yaml` - Schema update (+13 lines)
- `configs/unet_mixed_fl_dlbcl.yaml` - Example config (new)
- `models/__init__.py` - Module exports (+2 lines)
- `MIXED_TRAINING_GUIDE.md` - Usage guide (new)
- `IMPLEMENTATION_NOTES.md` - Technical details (new)
- `test_mixed_training.py` - Test suite (new)

**Total: 1003 additions, 19 deletions across 8 files**

## Support

For questions or issues:
1. See detailed guides: MIXED_TRAINING_GUIDE.md, IMPLEMENTATION_NOTES.md
2. Check config examples: configs/unet_mixed_fl_dlbcl.yaml
3. Run tests: python test_mixed_training.py
