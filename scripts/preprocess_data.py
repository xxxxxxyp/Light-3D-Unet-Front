"""
Data Preprocessing Script for FL Dataset
Path B: Preserve 4×4×4mm spacing, apply intensity clipping and normalization
"""

import os
import json
import argparse
import numpy as np
import nibabel as nib
from pathlib import Path
from datetime import datetime
from tqdm import tqdm


def clip_and_normalize(image, low_percentile=0.5, high_percentile=99.5, target_range=(0, 1)):
    """
    Clip intensity values to percentiles and normalize to target range
    
    Args:
        image: Input image array
        low_percentile: Lower percentile for clipping
        high_percentile: Upper percentile for clipping
        target_range: Target range for normalization (min, max)
    
    Returns:
        normalized_image: Normalized image
        metadata: Dictionary with clip values and normalization parameters
    """
    # Calculate clip values
    clip_min = np.percentile(image, low_percentile)
    clip_max = np.percentile(image, high_percentile)
    
    # Clip values
    clipped = np.clip(image, clip_min, clip_max)
    
    # Normalize to target range
    if clip_max > clip_min:
        normalized = (clipped - clip_min) / (clip_max - clip_min)
        normalized = normalized * (target_range[1] - target_range[0]) + target_range[0]
    else:
        normalized = np.ones_like(clipped) * target_range[0]
    
    metadata = {
        "clip_values": {
            "min": float(clip_min),
            "max": float(clip_max),
            "low_percentile": low_percentile,
            "high_percentile": high_percentile
        },
        "normalization_range": list(target_range)
    }
    
    return normalized, metadata


def calculate_voxel_thresholds(spacing, volume_cc_list):
    """
    Calculate voxel count thresholds for given volume in cubic centimeters
    
    Args:
        spacing: Voxel spacing in mm [z, y, x]
        volume_cc_list: List of volume thresholds in cubic centimeters
    
    Returns:
        Dictionary mapping volume_cc to voxel count
    """
    # Calculate voxel volume in cubic millimeters
    voxel_volume_mm3 = spacing[0] * spacing[1] * spacing[2]
    
    # Convert to cubic centimeters (1 cc = 1000 mm³)
    voxel_volume_cc = voxel_volume_mm3 / 1000.0
    
    thresholds = {}
    for volume_cc in volume_cc_list:
        voxel_count = volume_cc / voxel_volume_cc
        thresholds[f"{volume_cc}cc"] = {
            "volume_cc": volume_cc,
            "voxel_count": int(np.ceil(voxel_count)),
            "formula": f"ceil({volume_cc}cc / {voxel_volume_cc:.6f}cc/voxel)"
        }
    
    return thresholds


def preprocess_case(case_id, raw_dir, processed_dir, config):
    """
    Preprocess a single case
    
    Args:
        case_id: Case identifier (e.g., "0001")
        raw_dir: Path to raw data directory (contains images/ and labels/ subdirs)
        processed_dir: Path to processed output directory
        config: Preprocessing configuration
    
    Returns:
        success: Boolean indicating success
        metadata: Case metadata dictionary
    """
    raw_dir = Path(raw_dir)
    images_dir = raw_dir / "images"
    labels_dir = raw_dir / "labels"
    
    # Find image and label files for this case
    # Images have pattern: case_id_*.nii or case_id_*.nii.gz (e.g., 0001_0000.nii.gz)
    # Labels have pattern: case_id.nii or case_id.nii.gz (e.g., 0001.nii.gz)
    image_files = []
    label_files = []
    
    if images_dir.exists():
        for pattern in [f"{case_id}_*.nii.gz", f"{case_id}_*.nii"]:
            image_files.extend(images_dir.glob(pattern))
    
    if labels_dir.exists():
        for pattern in [f"{case_id}.nii.gz", f"{case_id}.nii"]:
            label_files.extend(labels_dir.glob(pattern))
    
    # Check if files exist
    if len(image_files) == 0 or len(label_files) == 0:
        print(f"Warning: Case {case_id} missing files (images: {len(image_files)}, labels: {len(label_files)}), skipping...")
        return False, None
    
    # Create output directories (flat structure matching raw)
    processed_dir = Path(processed_dir)
    processed_images_dir = processed_dir / "images"
    processed_labels_dir = processed_dir / "labels"
    processed_metadata_dir = processed_dir / "metadata"
    processed_images_dir.mkdir(parents=True, exist_ok=True)
    processed_labels_dir.mkdir(parents=True, exist_ok=True)
    processed_metadata_dir.mkdir(parents=True, exist_ok=True)
    
    # Process each image file (typically one PET scan per case)
    metadata_list = []
    for img_file in image_files:
        # Load image
        img_nii = nib.load(img_file)
        img_data = img_nii.get_fdata()
        affine = img_nii.affine
        header = img_nii.header
        
        # Extract spacing from header
        spacing = header.get_zooms()[:3]  # z, y, x
        spacing = [float(s) for s in spacing]
        
        # Verify spacing (should be 4×4×4mm for Path B)
        expected_spacing = config["spacing"]["target"]
        if not np.allclose(spacing, expected_spacing, atol=0.1):
            print(f"Warning: Case {case_id} has spacing {spacing}, expected {expected_spacing}")
        
        # Apply intensity clipping and normalization
        normalized_img, intensity_metadata = clip_and_normalize(
            img_data,
            low_percentile=config["intensity"]["clip_percentile_low"],
            high_percentile=config["intensity"]["clip_percentile_high"],
            target_range=config["intensity"]["normalization_range"]
        )
        
        # Calculate voxel thresholds
        voxel_thresholds = calculate_voxel_thresholds(
            spacing,
            [config["volume_threshold"]["train_cc"], config["volume_threshold"]["inference_cc"]]
        )
        
        # Save processed image
        output_img_path = processed_images_dir / img_file.name
        output_nii = nib.Nifti1Image(normalized_img.astype(np.float32), affine, header)
        nib.save(output_nii, output_img_path)
        
        # Create metadata
        case_metadata = {
            "case_id": case_id,
            "orig_spacing": spacing,
            "image_size": list(img_data.shape),  # x, y, z
            "suv_calculated": True,  # Assume SUV is pre-calculated
            "clip_values": intensity_metadata["clip_values"],
            "normalization_range": intensity_metadata["normalization_range"],
            "patch_size": config["patch_size"],
            "voxel_thresholds": voxel_thresholds,
            "processing_timestamp": datetime.now().isoformat(),
            "processing_path": "B",
            "seed": config["seed"],
            "bbox_expansion_mm": config["bbox_expansion_mm"],
            "bbox_expansion_voxels": config["bbox_expansion_voxels"]
        }
        
        metadata_list.append(case_metadata)
    
    # Copy label files (no processing needed)
    for label_file in label_files:
        output_label_path = processed_labels_dir / label_file.name
        label_nii = nib.load(label_file)
        nib.save(label_nii, output_label_path)
    
    # Save metadata JSON
    if metadata_list:
        metadata_path = processed_metadata_dir / f"{case_id}.json"
        with open(metadata_path, "w") as f:
            json.dump(metadata_list[0] if len(metadata_list) == 1 else metadata_list, f, indent=2)
        
        return True, metadata_list[0] if len(metadata_list) == 1 else metadata_list
    
    return False, None


def preprocess_dataset(split_file, raw_dir, processed_dir, config):
    """
    Preprocess all cases in a split
    
    Args:
        split_file: Path to split list file (train_list.txt, val_list.txt)
        raw_dir: Path to raw data directory
        processed_dir: Path to processed output directory
        config: Preprocessing configuration
    
    Returns:
        Summary statistics
    """
    # Load case list
    with open(split_file, "r") as f:
        case_ids = [line.strip() for line in f if line.strip()]
    
    print(f"Processing {len(case_ids)} cases from {split_file}")
    
    # Process each case
    successful = 0
    failed = []
    all_metadata = []
    
    for case_id in tqdm(case_ids, desc="Preprocessing"):
        success, metadata = preprocess_case(case_id, raw_dir, processed_dir, config)
        if success:
            successful += 1
            all_metadata.append(metadata)
        else:
            failed.append(case_id)
    
    print(f"\nPreprocessing complete:")
    print(f"  Successful: {successful}/{len(case_ids)}")
    print(f"  Failed: {len(failed)}")
    if failed:
        print(f"  Failed cases: {', '.join(failed[:10])}{'...' if len(failed) > 10 else ''}")
    
    return {
        "total": len(case_ids),
        "successful": successful,
        "failed": len(failed),
        "failed_cases": failed,
        "metadata": all_metadata
    }


def main():
    parser = argparse.ArgumentParser(description="Preprocess FL dataset")
    parser.add_argument("--config", type=str, default="configs/unet_fl70.yaml",
                        help="Path to configuration file")
    parser.add_argument("--raw_dir", type=str, default="data/raw",
                        help="Path to raw data directory")
    parser.add_argument("--processed_dir", type=str, default="data/processed",
                        help="Path to processed output directory")
    parser.add_argument("--splits_dir", type=str, default="data/splits",
                        help="Path to splits directory")
    parser.add_argument("--split", type=str, choices=["train", "val", "test", "all"],
                        default="all", help="Which split to process")
    
    args = parser.parse_args()
    
    # Load configuration
    import yaml
    with open(args.config, "r", encoding='utf-8') as f:
        full_config = yaml.safe_load(f)
    
    # Extract preprocessing configuration
    config = {
        "spacing": full_config["data"]["spacing"],
        "intensity": full_config["data"]["intensity"],
        "patch_size": full_config["data"]["patch_size"],
        "volume_threshold": full_config["data"]["volume_threshold"],
        "bbox_expansion_mm": full_config["data"]["bbox_expansion_mm"],
        "bbox_expansion_voxels": full_config["data"]["bbox_expansion_voxels"],
        "seed": full_config["experiment"]["seed"]
    }
    
    # Determine which splits to process
    splits_to_process = []
    if args.split == "all":
        splits_to_process = ["train", "val"]  # Don't process test (black box)
    else:
        if args.split == "test":
            print("Warning: Test set is black box and should not be processed at this stage")
            response = input("Continue anyway? (y/N): ")
            if response.lower() != 'y':
                print("Aborting.")
                return
        splits_to_process = [args.split]
    
    # Process each split
    all_summaries = {}
    for split_name in splits_to_process:
        split_file = Path(args.splits_dir) / f"{split_name}_list.txt"
        if not split_file.exists():
            print(f"Warning: Split file {split_file} not found, skipping...")
            continue
        
        print(f"\n{'='*60}")
        print(f"Processing {split_name} split")
        print(f"{'='*60}")
        
        summary = preprocess_dataset(split_file, args.raw_dir, args.processed_dir, config)
        all_summaries[split_name] = summary
    
    # Save summary
    summary_path = Path(args.processed_dir) / "preprocessing_summary.json"
    with open(summary_path, "w") as f:
        json.dump({
            "config": config,
            "summaries": all_summaries,
            "timestamp": datetime.now().isoformat()
        }, f, indent=2)
    
    print(f"\nPreprocessing summary saved to {summary_path}")


if __name__ == "__main__":
    main()
