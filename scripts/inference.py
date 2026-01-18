"""
Inference Entrypoint for Lightweight 3D U-Net
"""

import os
import sys
import argparse

# Ensure root directory is in path to import models/light_unet
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from light_unet.core.inferencer import Inferencer

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
    
    # Create inferencer
    inferencer = Inferencer(args.config, args.model)
    
    # Override threshold if specified
    if args.threshold is not None:
        inferencer.config["validation"]["default_threshold"] = args.threshold
    
    # Run inference
    if args.case_id:
        inferencer.infer_case(args.case_id, args.data_dir)
    else:
        inferencer.infer_split(args.split_file, args.data_dir)

if __name__ == "__main__":
    main()