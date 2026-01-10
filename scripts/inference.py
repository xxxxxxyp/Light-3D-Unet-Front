"""
Inference Script for Lightweight 3D U-Net
Generates probability maps and candidate bounding boxes
"""

import os
import sys
import json
import yaml
import argparse
import numpy as np
import nibabel as nib
import torch
from pathlib import Path
from tqdm import tqdm
from scipy import ndimage

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.unet3d import Lightweight3DUNet
from models.metrics import get_connected_components, calculate_metrics


class Inferencer:
    """Inference class for generating predictions"""
    
    def __init__(self, config_path, model_path):
        """Initialize inferencer"""
        # Load configuration
        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)
        
        # Setup device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        
        # Load model
        self.model = Lightweight3DUNet(
            in_channels=1,
            out_channels=self.config["model"]["output_channels"],
            start_channels=self.config["model"]["start_channels"],
            encoder_channels=self.config["model"]["encoder_channels"],
            use_depthwise_separable=self.config["model"]["use_depthwise_separable"],
            use_grouped=self.config["model"]["use_grouped_conv"],
            groups=self.config["model"]["groups"],
            dropout_p=0.0  # No dropout during inference
        ).to(self.device)
        
        # Load checkpoint
        checkpoint = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.eval()
        
        print(f"Loaded model from {model_path}")
        print(f"Best epoch: {checkpoint.get('best_epoch', 'N/A')}")
        print(f"Best metric: {checkpoint.get('best_metric', 'N/A'):.4f}")
        
        # Output directories
        self.prob_maps_dir = Path(self.config["output"]["prob_maps_dir"])
        self.bboxes_dir = Path(self.config["output"]["bboxes_dir"])
        self.prob_maps_dir.mkdir(parents=True, exist_ok=True)
        self.bboxes_dir.mkdir(parents=True, exist_ok=True)
    
    def sliding_window_inference(self, image, window_size=(48, 48, 48), stride=24):
        """
        Perform sliding window inference on full volume
        
        Args:
            image: Input image [D, H, W]
            window_size: Size of sliding window
            stride: Stride for sliding window
        
        Returns:
            prob_map: Probability map [D, H, W]
        """
        d, h, w = image.shape
        wd, wh, ww = window_size
        
        # Output probability map
        prob_map = np.zeros_like(image, dtype=np.float32)
        count_map = np.zeros_like(image, dtype=np.float32)
        
        # Sliding window
        for z in range(0, d - wd + 1, stride):
            for y in range(0, h - wh + 1, stride):
                for x in range(0, w - ww + 1, stride):
                    # Extract patch
                    patch = image[z:z+wd, y:y+wh, x:x+ww]
                    
                    # Predict
                    with torch.no_grad():
                        patch_tensor = torch.from_numpy(patch).unsqueeze(0).unsqueeze(0).float().to(self.device)
                        pred = self.model(patch_tensor)
                        pred = pred.cpu().numpy()[0, 0]
                    
                    # Accumulate
                    prob_map[z:z+wd, y:y+wh, x:x+ww] += pred
                    count_map[z:z+wd, y:y+wh, x:x+ww] += 1
        
        # Average overlapping predictions
        prob_map = np.divide(prob_map, count_map, where=count_map > 0)
        
        return prob_map
    
    def extract_bboxes(self, prob_map, threshold=0.3, min_volume_cc=0.5, spacing=(4.0, 4.0, 4.0)):
        """
        Extract bounding boxes from probability map
        
        Args:
            prob_map: Probability map [D, H, W]
            threshold: Probability threshold
            min_volume_cc: Minimum volume in cubic centimeters
            spacing: Voxel spacing (z, y, x) in mm
        
        Returns:
            bboxes: List of bounding box dictionaries
        """
        # Binarize
        binary_mask = (prob_map >= threshold).astype(np.int32)
        
        # Calculate minimum voxel count
        voxel_volume_mm3 = spacing[0] * spacing[1] * spacing[2]
        voxel_volume_cc = voxel_volume_mm3 / 1000.0
        min_voxels = int(np.ceil(min_volume_cc / voxel_volume_cc))
        
        # Get connected components
        labeled, num_components = get_connected_components(binary_mask, min_size=min_voxels)
        
        bboxes = []
        
        for component_id in range(1, num_components + 1):
            component_mask = labeled == component_id
            
            # Find bounding box
            coords = np.argwhere(component_mask)
            if len(coords) == 0:
                continue
            
            z_min, y_min, x_min = coords.min(axis=0)
            z_max, y_max, x_max = coords.max(axis=0)
            
            # Expand bbox by 10mm (3 voxels at 4mm spacing)
            expansion_voxels = self.config["data"]["bbox_expansion_voxels"]
            
            z_min_exp = max(0, z_min - expansion_voxels)
            z_max_exp = min(prob_map.shape[0] - 1, z_max + expansion_voxels)
            y_min_exp = max(0, y_min - expansion_voxels)
            y_max_exp = min(prob_map.shape[1] - 1, y_max + expansion_voxels)
            x_min_exp = max(0, x_min - expansion_voxels)
            x_max_exp = min(prob_map.shape[2] - 1, x_max + expansion_voxels)
            
            # Convert to mm coordinates
            z_min_mm = z_min_exp * spacing[0]
            z_max_mm = z_max_exp * spacing[0]
            y_min_mm = y_min_exp * spacing[1]
            y_max_mm = y_max_exp * spacing[1]
            x_min_mm = x_min_exp * spacing[2]
            x_max_mm = x_max_exp * spacing[2]
            
            # Calculate volume
            volume_voxels = component_mask.sum()
            volume_cc = volume_voxels * voxel_volume_cc
            
            # Calculate max confidence in region
            confidence = prob_map[component_mask].max()
            
            bbox = {
                "mask_id": component_id,
                "bbox_voxel": [
                    int(z_min_exp), int(z_max_exp),
                    int(y_min_exp), int(y_max_exp),
                    int(x_min_exp), int(x_max_exp)
                ],
                "bbox_mm": [
                    float(z_min_mm), float(z_max_mm),
                    float(y_min_mm), float(y_max_mm),
                    float(x_min_mm), float(x_max_mm)
                ],
                "volume_cc": float(volume_cc),
                "confidence": float(confidence)
            }
            
            bboxes.append(bbox)
        
        return bboxes
    
    def infer_case(self, case_id, data_dir, threshold=0.3):
        """
        Perform inference on a single case
        
        Args:
            case_id: Case identifier
            data_dir: Path to processed data directory
            threshold: Probability threshold
        
        Returns:
            success: Boolean indicating success
        """
        # Load image from flat structure
        data_dir = Path(data_dir)
        images_dir = data_dir / "images"
        
        # Find image file for this case
        image_files = []
        for pattern in [f"{case_id}_*.nii.gz", f"{case_id}_*.nii"]:
            image_files.extend(images_dir.glob(pattern))
        
        if len(image_files) == 0:
            print(f"Warning: No image files found for {case_id}")
            return False
        
        image_path = image_files[0]
        image_nii = nib.load(image_path)
        image = image_nii.get_fdata().astype(np.float32)
        affine = image_nii.affine
        header = image_nii.header
        
        # Get spacing
        spacing = header.get_zooms()[:3]
        spacing = [float(s) for s in spacing]
        
        # Load metadata
        metadata_path = case_dir / "metadata.json"
        if metadata_path.exists():
            with open(metadata_path, "r") as f:
                metadata = json.load(f)
        else:
            metadata = {}
        
        # Perform inference
        print(f"Running inference on {case_id}...")
        prob_map = self.sliding_window_inference(
            image,
            window_size=tuple(self.config["data"]["patch_size"]),
            stride=24  # 50% overlap
        )
        
        # Save probability map
        prob_map_path = self.prob_maps_dir / f"{case_id}_prob.nii.gz"
        prob_nii = nib.Nifti1Image(prob_map, affine, header)
        nib.save(prob_nii, prob_map_path)
        print(f"Saved probability map: {prob_map_path}")
        
        # Extract bounding boxes
        bboxes = self.extract_bboxes(
            prob_map,
            threshold=threshold,
            min_volume_cc=self.config["data"]["volume_threshold"]["inference_cc"],
            spacing=spacing
        )
        
        # Create bbox JSON
        bbox_json = {
            "case_id": case_id,
            "processing_path": "B",
            "orig_spacing": spacing,
            "threshold": threshold,
            "num_candidates": len(bboxes),
            "candidates": bboxes
        }
        
        # Save bbox JSON
        bbox_path = self.bboxes_dir / f"{case_id}_bboxes.json"
        with open(bbox_path, "w") as f:
            json.dump(bbox_json, f, indent=2)
        
        print(f"Found {len(bboxes)} candidates, saved to {bbox_path}")
        
        return True
    
    def infer_split(self, split_file, data_dir):
        """
        Perform inference on all cases in a split
        
        Args:
            split_file: Path to split list file
            data_dir: Path to processed data directory
        """
        # Load case list
        with open(split_file, "r") as f:
            case_ids = [line.strip() for line in f if line.strip()]
        
        print(f"Performing inference on {len(case_ids)} cases...")
        
        # Infer each case
        successful = 0
        failed = []
        
        threshold = self.config["validation"]["default_threshold"]
        
        for case_id in tqdm(case_ids, desc="Inference"):
            success = self.infer_case(case_id, data_dir, threshold=threshold)
            if success:
                successful += 1
            else:
                failed.append(case_id)
        
        print(f"\nInference complete:")
        print(f"  Successful: {successful}/{len(case_ids)}")
        print(f"  Failed: {len(failed)}")
        if failed:
            print(f"  Failed cases: {', '.join(failed)}")


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
        # Single case inference
        inferencer.infer_case(args.case_id, args.data_dir)
    else:
        # Full split inference
        inferencer.infer_split(args.split_file, args.data_dir)


if __name__ == "__main__":
    main()
