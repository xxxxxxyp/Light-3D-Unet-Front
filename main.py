"""
Main Execution Script for FL-70% Lightweight 3D U-Net Pipeline
This script orchestrates the complete workflow from data preparation to evaluation
"""

import os
import sys
import argparse
import subprocess
from pathlib import Path


def run_command(cmd, description):
    """Run a command and handle errors"""
    print(f"\n{'='*80}")
    print(f"{description}")
    print(f"{'='*80}")
    print(f"Command: {' '.join(cmd)}")
    
    result = subprocess.run(cmd, capture_output=False)
    
    if result.returncode != 0:
        print(f"Error: {description} failed with return code {result.returncode}")
        sys.exit(1)
    
    print(f"✓ {description} completed successfully")


def main():
    parser = argparse.ArgumentParser(
        description="FL-70% Lightweight 3D U-Net Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run complete pipeline
  python main.py --mode all
  
  # Run individual stages
  python main.py --mode split
  python main.py --mode preprocess
  python main.py --mode train
  python main.py --mode inference
  python main.py --mode evaluate
  
  # Custom paths
  python main.py --mode all --data_root /path/to/data --config custom_config.yaml
        """
    )
    
    parser.add_argument("--mode", type=str, required=True,
                        choices=["all", "split", "preprocess", "train", "inference", "evaluate"],
                        help="Execution mode")
    parser.add_argument("--config", type=str, default="configs/unet_fl70.yaml",
                        help="Path to configuration file")
    parser.add_argument("--data_root", type=str, default="data/raw",
                        help="Path to raw data directory")
    parser.add_argument("--processed_dir", type=str, default="data/processed",
                        help="Path to processed data directory")
    parser.add_argument("--splits_dir", type=str, default="data/splits",
                        help="Path to splits directory")
    parser.add_argument("--model_path", type=str, default="models/best_model.pth",
                        help="Path to model checkpoint (for inference)")
    parser.add_argument("--skip_split", action="store_true",
                        help="Skip data splitting if already done")
    parser.add_argument("--skip_preprocess", action="store_true",
                        help="Skip preprocessing if already done")
    
    args = parser.parse_args()
    
    # Ensure directories exist
    Path("data/raw").mkdir(parents=True, exist_ok=True)
    Path("data/processed").mkdir(parents=True, exist_ok=True)
    Path("data/splits").mkdir(parents=True, exist_ok=True)
    Path("models/checkpoints").mkdir(parents=True, exist_ok=True)
    Path("logs").mkdir(parents=True, exist_ok=True)
    Path("inference/prob_maps").mkdir(parents=True, exist_ok=True)
    Path("inference/bboxes").mkdir(parents=True, exist_ok=True)
    
    # Define stages
    stages = {
        "split": {
            "cmd": [
                "python", "scripts/split_dataset.py",
                "--data_root", args.data_root,
                "--output_dir", args.splits_dir,
                "--seed", "42"
            ],
            "description": "Step 1: Data Splitting",
            "skip": args.skip_split
        },
        "preprocess": {
            "cmd": [
                "python", "scripts/preprocess_data.py",
                "--config", args.config,
                "--raw_dir", args.data_root,
                "--processed_dir", args.processed_dir,
                "--splits_dir", args.splits_dir,
                "--split", "all"
            ],
            "description": "Step 2: Data Preprocessing",
            "skip": args.skip_preprocess
        },
        "train": {
            "cmd": [
                "python", "scripts/train.py",
                "--config", args.config,
                "--data_dir", args.processed_dir,
                "--splits_dir", args.splits_dir
            ],
            "description": "Step 3: Model Training",
            "skip": False
        },
        "inference": {
            "cmd": [
                "python", "scripts/inference.py",
                "--config", args.config,
                "--model", args.model_path,
                "--data_dir", args.processed_dir,
                "--split_file", f"{args.splits_dir}/val_list.txt"
            ],
            "description": "Step 4: Inference on Validation Set",
            "skip": False
        },
        "evaluate": {
            "cmd": [
                "python", "scripts/evaluate.py",
                "--config", args.config,
                "--prob_maps_dir", "inference/prob_maps",
                "--data_dir", args.processed_dir,
                "--split_file", f"{args.splits_dir}/val_list.txt",
                "--output_dir", "inference"
            ],
            "description": "Step 5: Evaluation",
            "skip": False
        }
    }
    
    # Determine which stages to run
    if args.mode == "all":
        stages_to_run = ["split", "preprocess", "train", "inference", "evaluate"]
    else:
        stages_to_run = [args.mode]
    
    # Run stages
    print("\n" + "="*80)
    print("FL-70% LIGHTWEIGHT 3D U-NET PIPELINE")
    print("="*80)
    print(f"Mode: {args.mode}")
    print(f"Config: {args.config}")
    print(f"Data root: {args.data_root}")
    print(f"Processed dir: {args.processed_dir}")
    print(f"Splits dir: {args.splits_dir}")
    
    for stage_name in stages_to_run:
        stage = stages[stage_name]
        
        if stage["skip"]:
            print(f"\n⊗ Skipping {stage['description']}")
            continue
        
        run_command(stage["cmd"], stage["description"])
    
    print("\n" + "="*80)
    print("PIPELINE COMPLETED SUCCESSFULLY")
    print("="*80)
    
    # Print summary
    if args.mode in ["all", "evaluate"]:
        print("\nNext steps:")
        print("1. Review metrics: cat inference/metrics.csv")
        print("2. Check TensorBoard: tensorboard --logdir logs/tensorboard")
        print("3. Review training history: cat logs/training_history.json")
        print("4. Prepare experiment report with findings and analysis")


if __name__ == "__main__":
    main()
