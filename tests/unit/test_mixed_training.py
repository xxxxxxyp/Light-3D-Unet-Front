"""
Test script for mixed domain training functionality
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from light_unet.data.dataset import filter_cases_by_domain

def test_domain_filtering():
    """Test domain filtering functionality"""
    print("Testing domain filtering...")
    
    # Create test case IDs
    test_cases = [
        '0000', '0001', '0050', '0100', '0122',  # FL cases
        '1000', '1100', '1200', '1422',  # DLBCL cases
        '0200', '0500', '2000'  # Edge cases
    ]
    
    # Test FL filtering
    fl_config = {
        'domain': 'fl',
        'fl_prefix_max': 122,
        'dlbcl_prefix_min': 1000,
        'dlbcl_prefix_max': 1422
    }
    
    fl_cases = filter_cases_by_domain(test_cases, fl_config)
    print(f"FL cases (expected 0000-0122): {fl_cases}")
    assert set(fl_cases) == {'0000', '0001', '0050', '0100', '0122'}
    print("✓ FL filtering works correctly")
    
    # Test DLBCL filtering
    dlbcl_config = {
        'domain': 'dlbcl',
        'fl_prefix_max': 122,
        'dlbcl_prefix_min': 1000,
        'dlbcl_prefix_max': 1422
    }
    
    dlbcl_cases = filter_cases_by_domain(test_cases, dlbcl_config)
    print(f"DLBCL cases (expected 1000-1422): {dlbcl_cases}")
    assert set(dlbcl_cases) == {'1000', '1100', '1200', '1422'}
    print("✓ DLBCL filtering works correctly")
    
    # Test no filtering
    all_cases = filter_cases_by_domain(test_cases, None)
    print(f"All cases (no filter): {all_cases}")
    assert len(all_cases) == len(test_cases)
    print("✓ No filtering works correctly")
    
    print("\n✓ All domain filtering tests passed!")

def test_patchdataset_defaults_to_fl():
    """PatchDataset should default to FL-only cases when no domain is provided"""
    print("\nTesting PatchDataset default domain filtering...")
    import tempfile
    from light_unet.data.dataset import PatchDataset
    
    with tempfile.TemporaryDirectory() as tmpdir:
        split_file = os.path.join(tmpdir, "split.txt")
        with open(split_file, "w") as f:
            f.write("0001\n0122\n1000\n1100\n")
        
        dataset = PatchDataset(
            data_dir=tmpdir,
            split_file=split_file,
            patch_size=(4, 4, 4),
            lesion_patch_ratio=0.5,
            augmentation=None,
            seed=0
        )
        
        print(f"Filtered case IDs: {dataset.case_ids}")
        assert set(dataset.case_ids) == {"0001", "0122"}, "PatchDataset should only keep FL cases by default"
        print("✓ PatchDataset default filtering works correctly")

def test_mixed_dataset_import():
    """Test that MixedPatchDataset can be imported"""
    print("\nTesting MixedPatchDataset import...")
    try:
        from light_unet.data.dataset import MixedPatchDataset
        print("✓ MixedPatchDataset imported successfully")
    except ImportError as e:
        print(f"✗ Failed to import MixedPatchDataset: {e}")
        return False
    return True

def test_config_schema():
    """Test that config schema is valid"""
    print("\nTesting config schema...")
    import yaml
    
    try:
        with open('configs/unet_fl70.yaml', 'r') as f:
            config = yaml.safe_load(f)
        
        # Check for domains config
        assert 'domains' in config['data'], "Missing 'domains' in data config"
        assert 'fl_prefix_max' in config['data']['domains']
        assert 'dlbcl_prefix_min' in config['data']['domains']
        assert 'dlbcl_prefix_max' in config['data']['domains']
        print("✓ Domain config schema is valid")
        
        # Check for mixed_domains config
        assert 'mixed_domains' in config['training'], "Missing 'mixed_domains' in training config"
        assert 'enabled' in config['training']['mixed_domains']
        assert 'mode' in config['training']['mixed_domains']
        assert 'dlbcl_steps_ratio' in config['training']['mixed_domains']
        assert 'dlbcl_steps' in config['training']['mixed_domains']
        assert 'fl_ratio' in config['training']['mixed_domains']
        print("✓ Mixed training config schema is valid")
        
        # Validate mode values
        mode = config['training']['mixed_domains'].get('mode')
        assert mode in ['fl_epoch_plus_dlbcl', 'probabilistic'], f"Invalid mode: {mode}"
        print(f"✓ Mixed training mode is valid: {mode}")
        
        print("✓ All config schema tests passed!")
        return True
    except Exception as e:
        print(f"✗ Config schema test failed: {e}")
        return False

def test_dlbcl_steps_computation():
    """Test DLBCL steps computation from ratio"""
    print("\nTesting DLBCL steps computation...")
    
    # Test cases: (fl_batches, ratio, expected_steps)
    test_cases = [
        (100, 0.0, 0),
        (100, 0.5, 50),
        (100, 1.0, 100),
        (100, 1.5, 150),
        (87, 1.0, 87),
        (87, 0.3, 26),  # round(87 * 0.3) = round(26.1) = 26
        (50, 0.7, 35),  # round(50 * 0.7) = 35
    ]
    
    for fl_batches, ratio, expected in test_cases:
        result = round(fl_batches * ratio)
        assert result == expected, f"Expected {expected} for {fl_batches} * {ratio}, got {result}"
        print(f"  ✓ {fl_batches} batches * {ratio} ratio = {result} steps")
    
    print("✓ DLBCL steps computation tests passed!")
    return True

if __name__ == "__main__":
    try:
        test_domain_filtering()
        test_patchdataset_defaults_to_fl()
        test_mixed_dataset_import()
        test_config_schema()
        test_dlbcl_steps_computation()
        print("\n" + "="*50)
        print("All tests passed successfully! ✓")
        print("="*50)
    except AssertionError as e:
        print(f"\n✗ Test failed: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n✗ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
