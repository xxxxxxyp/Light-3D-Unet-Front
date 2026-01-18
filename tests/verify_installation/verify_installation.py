#!/usr/bin/env python
"""
System Verification Script
Tests that all components are properly installed and functional
"""

import sys
import importlib
from pathlib import Path


def check_python_version():
    """Check Python version"""
    print("="*60)
    print("Checking Python Version...")
    print("="*60)
    version = sys.version_info
    print(f"Python {version.major}.{version.minor}.{version.micro}")
    
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print("❌ Python 3.8+ required")
        return False
    print("✓ Python version OK")
    return True


def check_dependencies():
    """Check required packages"""
    print("\n" + "="*60)
    print("Checking Dependencies...")
    print("="*60)
    
    required = {
        'torch': 'PyTorch',
        'torchvision': 'TorchVision',
        'numpy': 'NumPy',
        'nibabel': 'NiBabel',
        'scipy': 'SciPy',
        'sklearn': 'scikit-learn',
        'matplotlib': 'Matplotlib',
        'tensorboard': 'TensorBoard',
        'yaml': 'PyYAML',
        'tqdm': 'tqdm',
        'pandas': 'Pandas'
    }
    
    all_ok = True
    for module, name in required.items():
        try:
            mod = importlib.import_module(module)
            version = getattr(mod, '__version__', 'unknown')
            print(f"✓ {name:20s} {version}")
        except ImportError:
            print(f"❌ {name:20s} NOT FOUND")
            all_ok = False
    
    return all_ok


def check_cuda():
    """Check CUDA availability"""
    print("\n" + "="*60)
    print("Checking CUDA...")
    print("="*60)
    
    try:
        import torch
        if torch.cuda.is_available():
            print(f"✓ CUDA available")
            print(f"  Device count: {torch.cuda.device_count()}")
            print(f"  Current device: {torch.cuda.current_device()}")
            print(f"  Device name: {torch.cuda.get_device_name(0)}")
            print(f"  CUDA version: {torch.version.cuda}")
        else:
            print("⚠ CUDA not available (will use CPU)")
        return True
    except Exception as e:
        print(f"❌ Error checking CUDA: {e}")
        return False


def check_project_structure():
    """Check project directory structure"""
    print("\n" + "="*60)
    print("Checking Project Structure...")
    print("="*60)
    
    required_dirs = [
        'configs',
        'models',
        'scripts',
        'data/raw',
        'data/processed',
        'data/splits',
        'logs',
        'inference/prob_maps',
        'inference/bboxes'
    ]
    
    all_ok = True
    for dir_path in required_dirs:
        path = Path(dir_path)
        if path.exists():
            print(f"✓ {dir_path}")
        else:
            print(f"❌ {dir_path} NOT FOUND")
            all_ok = False
    
    return all_ok


def check_config_files():
    """Check configuration files"""
    print("\n" + "="*60)
    print("Checking Configuration Files...")
    print("="*60)
    
    required_files = [
        'configs/unet_fl70.yaml',
        'requirements.txt',
        'README.md',
        'main.py'
    ]
    
    all_ok = True
    for file_path in required_files:
        path = Path(file_path)
        if path.exists():
            print(f"✓ {file_path}")
        else:
            print(f"❌ {file_path} NOT FOUND")
            all_ok = False
    
    return all_ok


def test_model():
    """Test model instantiation"""
    print("\n" + "="*60)
    print("Testing Model...")
    print("="*60)
    
    try:
        from light_unet.unet3d import Lightweight3DUNet
        import torch
        
        model = Lightweight3DUNet(
            in_channels=1,
            out_channels=1,
            start_channels=16,
            encoder_channels=[16, 32, 64, 128]
        )
        
        params = model.count_parameters()
        print(f"✓ Model created successfully")
        print(f"  Total parameters: {params['total']:,}")
        print(f"  Trainable parameters: {params['trainable']:,}")
        
        # Test forward pass
        x = torch.randn(1, 1, 48, 48, 48)
        with torch.no_grad():
            y = model(x)
        
        print(f"✓ Forward pass successful")
        print(f"  Input shape: {x.shape}")
        print(f"  Output shape: {y.shape}")
        
        return True
    except Exception as e:
        print(f"❌ Model test failed: {e}")
        return False


def test_loss():
    """Test loss function"""
    print("\n" + "="*60)
    print("Testing Loss Function...")
    print("="*60)
    
    try:
        from light_unet.losses import FocalTverskyLoss
        import torch
        
        loss_fn = FocalTverskyLoss(alpha=0.7, beta=0.3, gamma=0.75)
        
        pred = torch.rand(2, 1, 48, 48, 48)
        target = torch.randint(0, 2, (2, 1, 48, 48, 48)).float()
        
        loss = loss_fn(pred, target)
        
        print(f"✓ Loss function working")
        print(f"  Loss value: {loss.item():.4f}")
        
        return True
    except Exception as e:
        print(f"❌ Loss test failed: {e}")
        return False


def test_config_loading():
    """Test configuration loading"""
    print("\n" + "="*60)
    print("Testing Configuration Loading...")
    print("="*60)
    
    try:
        import yaml
        
        with open('configs/unet_fl70.yaml', 'r') as f:
            config = yaml.safe_load(f)
        
        print(f"✓ Configuration loaded successfully")
        print(f"  Experiment: {config['experiment']['name']}")
        print(f"  Seed: {config['experiment']['seed']}")
        print(f"  Processing path: {config['experiment']['processing_path']}")
        
        return True
    except Exception as e:
        print(f"❌ Config loading failed: {e}")
        return False


def main():
    """Run all checks"""
    print("\n" + "="*60)
    print("FL-70% LIGHTWEIGHT 3D U-NET SYSTEM VERIFICATION")
    print("="*60)
    
    results = {
        'Python Version': check_python_version(),
        'Dependencies': check_dependencies(),
        'CUDA': check_cuda(),
        'Project Structure': check_project_structure(),
        'Config Files': check_config_files(),
        'Model': test_model(),
        'Loss Function': test_loss(),
        'Configuration': test_config_loading()
    }
    
    print("\n" + "="*60)
    print("VERIFICATION SUMMARY")
    print("="*60)
    
    for check, passed in results.items():
        status = "✓ PASS" if passed else "❌ FAIL"
        print(f"{check:30s} {status}")
    
    all_passed = all(results.values())
    
    print("\n" + "="*60)
    if all_passed:
        print("✓ ALL CHECKS PASSED - SYSTEM READY!")
        print("="*60)
        print("\nNext steps:")
        print("1. Place your FL data in data/raw/")
        print("2. Run: python main.py --mode all")
        print("3. Monitor: tensorboard --logdir logs/tensorboard")
        return 0
    else:
        print("❌ SOME CHECKS FAILED - PLEASE FIX ISSUES")
        print("="*60)
        print("\nTroubleshooting:")
        print("1. Run: pip install -r requirements.txt")
        print("2. Check Python version (requires 3.8+)")
        print("3. Ensure all required directories exist")
        return 1


if __name__ == "__main__":
    sys.exit(main())
