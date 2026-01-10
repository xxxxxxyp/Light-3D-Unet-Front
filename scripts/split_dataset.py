"""
Data Splitting Script for FL Dataset
Splits 123 FL cases into train (70%), val (15%), test (15%)
Generates train_list.txt, val_list.txt, test_list.txt and split_manifest.json
"""

import os
import json
import random
import argparse
from pathlib import Path
from datetime import datetime


def split_dataset(data_root, output_dir, train_ratio=0.70, val_ratio=0.15, test_ratio=0.15, seed=42):
    """
    Split dataset into train/val/test sets
    
    Args:
        data_root: Path to raw data directory (expects flat structure with images/ and labels/ subdirs)
        output_dir: Path to output directory for splits
        train_ratio: Training set ratio
        val_ratio: Validation set ratio
        test_ratio: Test set ratio
        seed: Random seed for reproducibility
    """
    # Set random seed
    random.seed(seed)
    
    # Validate ratios
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, \
        f"Split ratios must sum to 1.0, got {train_ratio + val_ratio + test_ratio}"
    
    # Get all case IDs from raw data
    # Data structure: data/raw/images/*.nii.gz and data/raw/labels/*.nii.gz
    data_root = Path(data_root)
    
    case_ids = set()
    
    # Extract case IDs from label files (labels have format: XXXX.nii.gz or XXXX.nii)
    labels_dir = data_root / "labels"
    if labels_dir.exists():
        for label_file in labels_dir.glob("*.nii*"):
            # Only process .nii and .nii.gz files
            if label_file.suffix not in ['.nii', '.gz'] or (label_file.suffix == '.gz' and not label_file.stem.endswith('.nii')):
                continue
            # Extract case ID from filename using Path.stem
            # For .nii.gz files, stem gives us "case_id.nii", so we need to remove .nii
            case_id = label_file.stem
            if case_id.endswith('.nii'):
                case_id = case_id[:-4]
            case_ids.add(case_id)
    
    # Verify that corresponding image files exist (images have format: XXXX_0000.nii.gz)
    images_dir = data_root / "images"
    valid_cases = []
    if images_dir.exists() and len(case_ids) > 0:
        for case_id in sorted(case_ids):
            # Look for image file with pattern case_id_*.nii*
            image_files = list(images_dir.glob(f"{case_id}_*.nii*"))
            if len(image_files) > 0:
                valid_cases.append(case_id)
    
    case_dirs = valid_cases
    
    # If no cases found, create placeholder structure for demonstration
    if len(case_dirs) == 0:
        print(f"Warning: No valid cases found in {data_root}")
        print("Expected structure: data/raw/images/XXXX_*.nii.gz and data/raw/labels/XXXX.nii.gz")
        print("Creating placeholder case list for 123 FL cases...")
        # Generate placeholder case IDs
        case_dirs = [f"{i:04d}" for i in range(1, 124)]
    
    total_cases = len(case_dirs)
    print(f"Total cases found: {total_cases}")
    
    # Shuffle cases
    random.shuffle(case_dirs)
    
    # Calculate split sizes
    train_size = int(total_cases * train_ratio)
    val_size = int(total_cases * val_ratio)
    test_size = total_cases - train_size - val_size  # Remaining goes to test
    
    # Split cases
    train_cases = sorted(case_dirs[:train_size])
    val_cases = sorted(case_dirs[train_size:train_size + val_size])
    test_cases = sorted(case_dirs[train_size + val_size:])
    
    print(f"\nSplit summary:")
    print(f"  Train: {len(train_cases)} cases ({len(train_cases)/total_cases*100:.1f}%)")
    print(f"  Val:   {len(val_cases)} cases ({len(val_cases)/total_cases*100:.1f}%)")
    print(f"  Test:  {len(test_cases)} cases ({len(test_cases)/total_cases*100:.1f}%)")
    
    # Create output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Write split files
    with open(output_dir / "train_list.txt", "w") as f:
        f.write("\n".join(train_cases) + "\n")
    
    with open(output_dir / "val_list.txt", "w") as f:
        f.write("\n".join(val_cases) + "\n")
    
    with open(output_dir / "test_list.txt", "w") as f:
        f.write("\n".join(test_cases) + "\n")
    
    # Create split manifest JSON
    manifest = {
        "dataset": "Follicular_Lymphoma",
        "total_cases": total_cases,
        "split_date": datetime.now().isoformat(),
        "seed": seed,
        "split_ratios": {
            "train": train_ratio,
            "val": val_ratio,
            "test": test_ratio
        },
        "split_sizes": {
            "train": len(train_cases),
            "val": len(val_cases),
            "test": len(test_cases)
        },
        "splits": {
            "train": train_cases,
            "val": val_cases,
            "test": test_cases
        },
        "processing_path": "B",
        "spacing": [4.0, 4.0, 4.0],
        "notes": [
            "Test set is black-box and should not be used for training or validation",
            "All cases preserve original 4×4×4mm spacing (Path B)",
            "SUV values are pre-calculated and should not be recomputed"
        ]
    }
    
    # Write manifest
    manifest_path = output_dir.parent / "split_manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    
    print(f"\nFiles created:")
    print(f"  {output_dir / 'train_list.txt'}")
    print(f"  {output_dir / 'val_list.txt'}")
    print(f"  {output_dir / 'test_list.txt'}")
    print(f"  {manifest_path}")
    
    return manifest


def main():
    parser = argparse.ArgumentParser(description="Split FL dataset into train/val/test sets")
    parser.add_argument("--data_root", type=str, default="data/raw",
                        help="Path to raw data directory")
    parser.add_argument("--output_dir", type=str, default="data/splits",
                        help="Path to output directory for split files")
    parser.add_argument("--train_ratio", type=float, default=0.70,
                        help="Training set ratio")
    parser.add_argument("--val_ratio", type=float, default=0.15,
                        help="Validation set ratio")
    parser.add_argument("--test_ratio", type=float, default=0.15,
                        help="Test set ratio")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility")
    
    args = parser.parse_args()
    
    manifest = split_dataset(
        data_root=args.data_root,
        output_dir=args.output_dir,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed
    )
    
    print("\nSplit manifest created successfully!")


if __name__ == "__main__":
    main()
