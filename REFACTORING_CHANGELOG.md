# Refactoring Changelog: Legacy Code Convergence

## Overview
This PR implements three major refactoring tasks (1B, 2B, 3A) to converge legacy/compatibility code branches, improve observability, and enforce strict body mask usage. All changes maintain backward compatibility where appropriate while providing clear deprecation warnings and enforcing strict requirements where needed.

## Changes Summary

### Task 1B: Metrics Compatibility with Deprecation Tracking ✅

#### What Changed
- **Added deprecation tracking** for old metric keys (`dsc`, `recall`, `precision`)
- **Introduced `MetricsDict` class** that wraps metric dictionaries with deprecation warnings
- **Removed deep fallback chains** in `train.py` that silently defaulted to 0.0
- **Enforced explicit key checks** to catch missing metrics early

#### Technical Details
- `models/metrics.py`:
  - Added `MetricsDict` class that extends `dict` with deprecation warnings
  - Tracks which deprecated keys have been accessed (one-time warnings per process)
  - Maps old keys to new keys: `dsc` → `voxel_wise_dsc_micro`, `recall` → `lesion_wise_recall`, `precision` → `lesion_wise_precision`
  - `calculate_metrics()` now returns `MetricsDict` instead of plain `dict`

- `scripts/train.py`:
  - Replaced multi-level `.get()` fallbacks with explicit key checks that raise `KeyError` if metrics are missing
  - Uses new metric keys (`lesion_wise_recall`, `voxel_wise_dsc_macro`, etc.) internally
  - Example old code: `val_metrics.get("best_recall", val_metrics.get("lesion_wise_recall", val_metrics.get("recall", 0.0)))`
  - Example new code: `val_metrics.get("best_recall")` with explicit `KeyError` check

- `scripts/evaluate.py`:
  - Already correct - uses `calculate_lesion_metrics()` which returns simple dict with clear keys
  - No changes needed

#### Migration Guide
**For code using metrics:**
1. **Preferred**: Update to use new keys: `lesion_wise_recall`, `lesion_wise_precision`, `voxel_wise_dsc_micro`, `voxel_wise_dsc_macro`
2. **Compatible**: Continue using old keys (`dsc`, `recall`, `precision`) - you'll see one-time deprecation warnings
3. **Not supported**: Deep `.get()` fallback chains that hide missing keys - these will now raise `KeyError`

**Example migration:**
```python
# OLD (silent failure):
recall = metrics.get("recall", 0.0)  # Returns 0.0 if key missing, hiding bugs

# NEW (explicit check):
recall = metrics.get("lesion_wise_recall")
if recall is None:
    raise KeyError("Missing expected metric: lesion_wise_recall")

# COMPATIBLE (with warning):
recall = metrics["recall"]  # Still works, but emits deprecation warning
```

#### Benefits
- **Observable**: Deprecation warnings make it clear when old keys are used
- **Fail-fast**: Missing metrics now raise errors instead of silently defaulting to 0.0
- **Maintainable**: Easier to track down metric access patterns in codebase

---

### Task 2B: Mixed Training Return Value Unification ✅

#### What Changed
- **Unified `get_data_loader()` return structure** - always returns a dict
- **Removed isinstance/tuple/dict branching** in `train.py`
- **Added explicit `mode` field** to identify training mode

#### Technical Details
- `models/dataset.py` - `get_data_loader()`:
  - **Before**: Returned `dict` (step-based), `tuple` (probabilistic), or `DataLoader` (standard)
  - **After**: Always returns `dict` with standardized structure
  
  **Return structure:**
  ```python
  # Step-based mode (fl_epoch_plus_dlbcl)
  {
      'mode': 'fl_epoch_plus_dlbcl',
      'fl_loader': DataLoader,
      'dlbcl_loader': DataLoader,
      'fl_dataset': PatchDataset,
      'dlbcl_dataset': PatchDataset
  }
  
  # Probabilistic mode
  {
      'mode': 'probabilistic',
      'train_loader': DataLoader,
      'train_dataset': MixedPatchDataset
  }
  
  # Standard mode
  {
      'mode': 'standard',
      'train_loader': DataLoader
  }
  
  # Validation mode
  {
      'mode': 'validation',
      'val_loader': DataLoader
  }
  ```

- `scripts/train.py`:
  - **Before**: Used `isinstance()` checks and pattern matching to determine training mode
  - **After**: Checks `mode` field and extracts appropriate loaders
  - Raises `TypeError` if return value is not a dict
  - Raises `ValueError` if mode is unknown

#### Migration Guide
**For code calling `get_data_loader()`:**
1. Update to expect dict return value
2. Check `result['mode']` to determine training mode
3. Access loaders via dict keys instead of unpacking

**Example migration:**
```python
# OLD (tuple unpacking):
train_loader, train_dataset = get_data_loader(...)
if isinstance(train_loader, dict):
    # Handle step-based mode
    fl_loader = train_loader['fl_loader']

# NEW (unified dict):
result = get_data_loader(...)
if result['mode'] == 'fl_epoch_plus_dlbcl':
    fl_loader = result['fl_loader']
    dlbcl_loader = result['dlbcl_loader']
elif result['mode'] == 'probabilistic':
    train_loader = result['train_loader']
    train_dataset = result['train_dataset']
elif result['mode'] == 'standard':
    train_loader = result['train_loader']
```

#### Benefits
- **Consistent API**: No more guessing return type based on configuration
- **Type-safe**: Explicit mode field prevents ambiguity
- **Maintainable**: Single code path for handling data loaders
- **Testable**: Easier to verify behavior without complex branching

---

### Task 3A: Body Mask Hard Failure Mode ✅

#### What Changed
- **Enforced strict body mask requirements** when `enabled=True` and `apply_to_*=True`
- **Modified PatchDataset** to fail if body mask missing in strict mode
- **Modified CaseDataset** to fail if body mask missing in strict mode
- **Added body_mask_config parameter** to dataset constructors

#### Technical Details
- `models/dataset.py` - `PatchDataset`:
  - Added `body_mask_config` parameter to constructor
  - Computes `body_mask_required = enabled AND apply_to_training_sampling`
  - **In strict mode** (`body_mask_required=True`):
    - Raises `FileNotFoundError` if any case is missing body mask file
    - Raises `RuntimeError` if body mask file fails to load
    - Raises `ValueError` if body mask has no valid background region
  - **In non-strict mode** (`body_mask_required=False`):
    - Issues warnings for missing/failed body masks
    - Falls back to full volume for background sampling

- `models/dataset.py` - `CaseDataset`:
  - Added `body_mask_required` parameter to constructor
  - **In strict mode** (`body_mask_required=True`):
    - Raises `FileNotFoundError` if any case is missing body mask file
    - Raises `RuntimeError` if body mask file fails to load
  - **In non-strict mode**:
    - Returns all-ones tensor as fallback for missing body masks

- `models/dataset.py` - `get_data_loader()`:
  - Extracts `body_mask_config` from config
  - Passes to all PatchDataset and MixedPatchDataset instances
  - For validation: sets `body_mask_required = apply_to_validation`

#### Configuration
Body mask behavior is controlled by config section:
```yaml
data:
  body_mask:
    enabled: true  # Enable body mask feature
    apply_to_training_sampling: true  # Strict enforcement for training
    apply_to_validation: true  # Strict enforcement for validation
    apply_to_inference: true  # Strict enforcement for inference
```

**Enforcement matrix:**
| `enabled` | `apply_to_*` | Behavior |
|-----------|--------------|----------|
| `false`   | any          | No enforcement, no warnings |
| `true`    | `false`      | Soft mode: warnings but no errors |
| `true`    | `true`       | **Strict mode: raise errors if missing** |

#### Migration Guide
**If you have body masks for all cases:**
- No action needed - existing configs will work with enforcement

**If you're missing body masks for some cases:**
- **Option 1** (recommended): Generate body masks for all cases
- **Option 2**: Disable strict enforcement:
  ```yaml
  body_mask:
    enabled: true
    apply_to_training_sampling: false  # Allow fallback to full volume
  ```
- **Option 3**: Disable body mask feature entirely:
  ```yaml
  body_mask:
    enabled: false
  ```

#### Error Messages
**Missing body mask file:**
```
FileNotFoundError: Body mask is required (enabled=True, apply_to_training_sampling=True) 
but missing for 2/10 cases: ['0001', '0005']... 
Please ensure body masks are generated for all training cases or disable body mask enforcement.
```

**Failed to load body mask:**
```
RuntimeError: Failed to load required body mask for case 0001: [error details]
```

**Invalid body mask (no background region):**
```
ValueError: Case 0001: No background voxels found within body mask. 
Body mask may be invalid or too restrictive.
```

#### Benefits
- **Fail-fast**: Catches missing body masks at dataset initialization, not during training
- **Clear errors**: Explicit error messages guide users to fix the issue
- **Configurable**: Can disable strict mode for backward compatibility
- **Safe**: Prevents silent fallback to full volume when body mask is expected

---

## Testing

All tests pass:
- ✅ `test_body_mask.py` - Includes new strict enforcement test
- ✅ `test_mixed_training.py` - Verifies unified return structure
- ✅ `test_step_based_config.py` - Validates config schema

## Backward Compatibility

### Compatible (with warnings)
- **Metrics**: Old keys (`dsc`, `recall`, `precision`) still work but emit deprecation warnings
- **Body masks**: Can disable strict enforcement via config

### Breaking Changes
- **Data loader return type**: Now always returns dict (was tuple/DataLoader)
- **Body masks**: When `enabled=True` and `apply_to_*=True`, missing body masks now raise errors instead of falling back
- **Missing metrics**: Deep `.get()` fallback chains that hid missing keys will now raise `KeyError`

## Migration Checklist

For users upgrading to this version:

1. **Update metric access patterns** (if using programmatic access):
   - [ ] Search codebase for chained `.get()` calls on metrics
   - [ ] Replace with explicit key checks or update to new key names
   - [ ] Test that metrics are correctly extracted

2. **Update data loader usage** (if calling `get_data_loader()` directly):
   - [ ] Update code to expect dict return value
   - [ ] Access loaders via dict keys instead of tuple unpacking

3. **Check body mask availability** (if body masks enabled):
   - [ ] Verify all training/validation/test cases have body masks
   - [ ] Or disable strict enforcement in config

4. **Run tests**:
   ```bash
   python test_body_mask.py
   python test_mixed_training.py
   python test_step_based_config.py
   ```

## Files Modified

### Core Changes
- `models/metrics.py` - Added MetricsDict with deprecation tracking
- `models/dataset.py` - Unified data loader returns, strict body mask enforcement
- `scripts/train.py` - Removed fallback chains, updated for unified data loader structure

### Tests
- `test_body_mask.py` - Replaced backward compat test with strict enforcement test
- `test_mixed_training.py` - Already compatible with unified structure
- `test_step_based_config.py` - Already compatible

### Documentation
- `REFACTORING_CHANGELOG.md` - This document

## Future Work

- Update `scripts/inference.py` to enforce body mask in strict mode
- Update integration tests for comprehensive body mask coverage
- Update QUICKSTART.md and IMPLEMENTATION_NOTES.md with new behaviors
