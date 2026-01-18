"""
Training Entrypoint for Lightweight 3D U-Net
"""

import os
import sys
import argparse
from pathlib import Path

# [FIX] Robust path setup
# Get the absolute path of the project root (2 levels up from scripts/train.py)
current_file = Path(__file__).resolve()
project_root = current_file.parent.parent

# Force insert project root to the BEGINNING of sys.path
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# Debug: Verify paths
print(f"[System] Project root set to: {project_root}")
print(f"[System] Checking models package: {(project_root / 'models' / '__init__.py').exists()}")

from light_unet.core.trainer import Trainer
from light_unet.core.config import ConfigManager

def main():
    parser = argparse.ArgumentParser(description="Train Lightweight 3D U-Net")
    parser.add_argument("--config", type=str, default="configs/unet_fl70.yaml",
                        help="Path to configuration file")
    parser.add_argument("--data_dir", type=str, default=None,
                        help="Path to processed data directory (overrides config)")
    parser.add_argument("--splits_dir", type=str, default=None,
                        help="Path to splits directory (overrides config)")
    
    args = parser.parse_args()
    
    # Load config
    config = ConfigManager.load(args.config)
    
    # Apply overrides
    if args.data_dir:
        config["data_dir"] = args.data_dir
    else:
        # Ensure default relative path is resolved relative to project root
        if not os.path.isabs(config.get("data_dir", "data/processed")):
            config["data_dir"] = str(project_root / config.get("data_dir", "data/processed"))
    
    if args.splits_dir:
        config["splits_dir"] = args.splits_dir
    else:
        if not os.path.isabs(config.get("splits_dir", "data/splits")):
            config["splits_dir"] = str(project_root / config.get("splits_dir", "data/splits"))
    
    # Save updated config
    ConfigManager.save(config, args.config)
    
    # Start training
    trainer = Trainer(config)
    trainer.train()

if __name__ == "__main__":
    main()