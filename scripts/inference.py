"""
Inference Entrypoint for Lightweight 3D U-Net
"""

import os
import sys
import argparse
from pathlib import Path

# [FIX] Robust path setup
current_file = Path(__file__).resolve()
project_root = current_file.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from light_unet.core.inferencer import Inferencer
from light_unet.core.config import ConfigManager

def main():
    parser = argparse.ArgumentParser(description="Inference with Lightweight 3D U-Net")
    parser.add_argument("--config", type=str, default="configs/unet_fl70.yaml",
                        help="Path to configuration file")
    parser.add_argument("--model", type=str, default="models/best_model.pth",
                        help="Path to model checkpoint")
    parser.add_argument("--data_dir", type=str, default="data/processed",
                        help="Path to processed data directory")
    parser.add_argument("--split_file", type=str, default="data/splits/val_list.txt",
                        help="Path to split file to run inference on")
    parser.add_argument("--case_id", type=str, default=None,
                        help="Specific case ID to infer (optional)")
    parser.add_argument("--threshold", type=float, default=None,
                        help="Probability threshold (overrides config)")
    
    args = parser.parse_args()
    
    # Load config
    config = ConfigManager.load(args.config)
    
    # Resolve data paths relative to project root if they are relative
    if not os.path.isabs(args.data_dir):
        data_dir = str(project_root / args.data_dir)
    else:
        data_dir = args.data_dir
        
    if not os.path.isabs(args.split_file):
        split_file = str(project_root / args.split_file)
    else:
        split_file = args.split_file

    # Override threshold if specified
    if args.threshold is not None:
        config["validation"]["default_threshold"] = args.threshold
    
    # Create inferencer
    inferencer = Inferencer(config, args.model)
    
    # Run inference
    if args.case_id:
        inferencer.infer_case(args.case_id, data_dir)
    else:
        inferencer.infer_split(split_file, data_dir)

if __name__ == "__main__":
    main()