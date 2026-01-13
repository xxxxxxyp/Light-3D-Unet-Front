#!/usr/bin/env python
"""
Simple test to validate step-based mixed training configuration
without requiring full dependencies.
"""
import yaml

def test_configs():
    """Test that configs have correct schema"""
    print("Testing configuration schemas...")
    
    # Test unet_fl70.yaml
    with open('configs/unet_fl70.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    # Check domains config
    assert 'domains' in config['data'], "Missing 'domains' in data config"
    assert 'fl_prefix_max' in config['data']['domains']
    assert 'dlbcl_prefix_min' in config['data']['domains']
    assert 'dlbcl_prefix_max' in config['data']['domains']
    print("  ✓ Domain config schema is valid")
    
    # Check mixed_domains config
    assert 'mixed_domains' in config['training'], "Missing 'mixed_domains' in training config"
    assert 'enabled' in config['training']['mixed_domains']
    assert 'mode' in config['training']['mixed_domains']
    assert 'dlbcl_steps_ratio' in config['training']['mixed_domains']
    assert 'dlbcl_steps' in config['training']['mixed_domains']
    assert 'fl_ratio' in config['training']['mixed_domains']
    print("  ✓ Mixed training config schema is valid")
    
    # Validate mode values
    mode = config['training']['mixed_domains'].get('mode')
    assert mode in ['fl_epoch_plus_dlbcl', 'probabilistic'], f"Invalid mode: {mode}"
    print(f"  ✓ Mixed training mode is valid: {mode}")
    
    # Test unet_mixed_fl_dlbcl.yaml
    with open('configs/unet_mixed_fl_dlbcl.yaml', 'r') as f:
        mixed_config = yaml.safe_load(f)
    
    assert mixed_config['training']['mixed_domains']['enabled'] == True
    assert mixed_config['training']['mixed_domains']['mode'] == 'fl_epoch_plus_dlbcl'
    assert 'dlbcl_steps_ratio' in mixed_config['training']['mixed_domains']
    print("  ✓ Mixed config has step-based mode enabled")
    
    print("\n✓ All config schema tests passed!")

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

if __name__ == "__main__":
    try:
        test_configs()
        test_dlbcl_steps_computation()
        print("\n" + "="*60)
        print("All configuration tests passed successfully! ✓")
        print("="*60)
    except AssertionError as e:
        print(f"\n✗ Test failed: {e}")
        exit(1)
    except Exception as e:
        print(f"\n✗ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        exit(1)
