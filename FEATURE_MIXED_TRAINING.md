# Feature: Mixed FL + DLBCL Training

## Quick Start

To enable mixed domain training on FL + DLBCL data:

1. Use the provided example config:
   ```bash
   python scripts/train.py --config configs/unet_mixed_fl_dlbcl.yaml
   ```

2. Or enable it in your own config:
   ```yaml
   training:
     mixed_domains:
       enabled: true
       fl_ratio: 0.5  # 50% FL, 50% DLBCL
   ```

## What It Does

- **Trains** on a mix of FL and DLBCL cases with controlled sampling ratio
- **Validates** exclusively on FL cases for consistent evaluation
- **Logs** domain statistics to console and TensorBoard each epoch
- **Maintains** backward compatibility (disabled by default)

## Expected Output

```
*** Mixed Domain Training Enabled ***
  FL ratio: 50.00%
  Validation: FL-only
  Val cases: 18 FL cases

MixedPatchDataset: FL cases=87, DLBCL cases=423, FL ratio=0.50

Epoch 1/200
  Domain Statistics:
    FL samples: 512 (50.00%)
    DLBCL samples: 512 (50.00%)
    Total samples: 1024
  Train Loss: 0.3456
  Val Lesion-wise Recall: 0.7543 (best threshold: 0.30)
  ...
```

## Documentation

- **Usage Guide**: See [MIXED_TRAINING_GUIDE.md](MIXED_TRAINING_GUIDE.md)
- **Implementation Details**: See [IMPLEMENTATION_NOTES.md](IMPLEMENTATION_NOTES.md)
- **Example Configs**: 
  - `configs/unet_fl70.yaml` - FL-only (backward compatible)
  - `configs/unet_mixed_fl_dlbcl.yaml` - Mixed training enabled

## Requirements

- FL cases: 0000-0122 (case ID prefix ≤ 122)
- DLBCL cases: 1000-1422 (case ID prefix 1000-1422)
- Both domains in `train_list.txt`
- Standard data structure: `data/images/`, `data/labels/`

## Key Features

✅ **Controlled ratio**: Configure FL/DLBCL sampling ratio  
✅ **FL-only validation**: Always evaluate on FL for consistency  
✅ **Domain tracking**: Log sample counts per domain each epoch  
✅ **TensorBoard metrics**: Visualize domain statistics over time  
✅ **Backward compatible**: Works with existing configs/data  
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

### Mixed Training (in `training.mixed_domains`)
```yaml
training:
  mixed_domains:
    enabled: false           # Set to true to enable
    fl_ratio: 0.5           # Proportion of FL samples (0.0-1.0)
    dlbcl_ratio: 0.5        # Informational (= 1.0 - fl_ratio)
```

## Example Ratios

### 50/50 (Default)
```yaml
mixed_domains:
  enabled: true
  fl_ratio: 0.5
```

### FL-Heavy (70% FL, 30% DLBCL)
```yaml
mixed_domains:
  enabled: true
  fl_ratio: 0.7
```

### DLBCL-Heavy (30% FL, 70% DLBCL)
```yaml
mixed_domains:
  enabled: true
  fl_ratio: 0.3
```

### FL-Only (Backward Compatible)
```yaml
mixed_domains:
  enabled: false
```

## TensorBoard Metrics

View domain statistics:
```bash
tensorboard --logdir logs/tensorboard
```

Available metrics:
- `Domain/fl_samples` - FL sample count per epoch
- `Domain/dlbcl_samples` - DLBCL sample count per epoch
- `Domain/fl_ratio` - Actual FL sampling ratio
- `Domain/dlbcl_ratio` - Actual DLBCL sampling ratio

## Implementation

- **Dataset Filtering**: `filter_cases_by_domain()` in `models/dataset.py`
- **Mixed Dataset**: `MixedPatchDataset` class with ratio-based sampling
- **Data Loader**: `get_data_loader()` handles both standard and mixed datasets
- **Training Loop**: Automatic domain statistics tracking and logging

## Testing

Run the included test suite:
```bash
python test_mixed_training.py
```

Tests include:
- Domain filtering logic
- Config schema validation
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
