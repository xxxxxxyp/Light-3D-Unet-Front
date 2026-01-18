"""
Training Entrypoint for Lightweight 3D U-Net
"""

import os
import sys
import yaml
import argparse

# Ensure root directory is in path to import models/light_unet
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from light_unet.core.trainer import Trainer

def main():
    parser = argparse.ArgumentParser(description="Train Lightweight 3D U-Net")
    parser.add_argument("--config", type=str, default="configs/unet_fl70.yaml",
                        help="Path to configuration file")
    parser.add_argument("--data_dir", type=str, default=None,
                        help="Path to processed data directory (overrides config)")
    parser.add_argument("--splits_dir", type=str, default=None,
                        help="Path to splits directory (overrides config)")
    
    args = parser.parse_args()
    
    # Load config and override if specified
    with open(args.config, "r", encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    if args.data_dir:
        config["data_dir"] = args.data_dir
    else:
        config["data_dir"] = config.get("data_dir", "data/processed")
    
    if args.splits_dir:
        config["splits_dir"] = args.splits_dir
    else:
        config["splits_dir"] = config.get("splits_dir", "data/splits")
    
    # Save updated config
    with open(args.config, "w") as f:
        yaml.dump(config, f, default_flow_style=False)
    
    # Create trainer and start training
    trainer = Trainer(args.config)
    trainer.train()

if __name__ == "__main__":
    main()