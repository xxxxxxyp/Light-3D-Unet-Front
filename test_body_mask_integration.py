"""
Integration test for body mask feature - end-to-end workflow
"""

import sys
import os
import tempfile
import yaml
import json
import numpy as np
import nibabel as nib
from pathlib import Path

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from scripts.preprocess_data import preprocess_case
from models.dataset import PatchDataset, CaseDataset


def test_end_to_end_workflow():
    """Test complete workflow: preprocessing -> dataset loading -> sampling"""
    print("Testing end-to-end body mask workflow...")
    
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        
        # Setup directories
        raw_dir = tmpdir / "raw"
        processed_dir = tmpdir / "processed"
        raw_images = raw_dir / "images"
        raw_labels = raw_dir / "labels"
        
        raw_images.mkdir(parents=True)
        raw_labels.mkdir(parents=True)
        
        # Create synthetic case
        case_id = "0001"
        image_shape = (50, 50, 50)
        
        # Create PET image with body-like intensity distribution
        image = np.zeros(image_shape, dtype=np.float32)
        # Body region (center sphere)
        center = np.array([25, 25, 25])
        for z in range(image_shape[0]):
            for y in range(image_shape[1]):
                for x in range(image_shape[2]):
                    dist = np.sqrt((z - center[0])**2 + (y - center[1])**2 + (x - center[2])**2)
                    if dist < 15:
                        image[z, y, x] = 200 + 100 * np.random.rand()
                    else:
                        image[z, y, x] = 10 * np.random.rand()
        
        # Create label with lesion
        label = np.zeros(image_shape, dtype=np.float32)
        label[20:25, 20:25, 20:25] = 1.0
        
        # Save raw data
        affine = np.eye(4)
        affine[:3, :3] *= 4.0  # 4mm spacing
        
        image_path = raw_images / f"{case_id}_0000.nii.gz"
        nib.save(nib.Nifti1Image(image, affine), image_path)
        
        label_path = raw_labels / f"{case_id}.nii.gz"
        nib.save(nib.Nifti1Image(label, affine), label_path)
        
        print(f"  ✓ Created synthetic case {case_id}")
        
        # Create preprocessing config
        config = {
            "spacing": {"target": [4.0, 4.0, 4.0]},
            "intensity": {
                "clip_percentile_low": 0.5,
                "clip_percentile_high": 99.5,
                "normalization_range": [0, 1]
            },
            "patch_size": [16, 16, 16],
            "volume_threshold": {
                "train_cc": 0.1,
                "inference_cc": 0.5
            },
            "bbox_expansion_mm": 10.0,
            "bbox_expansion_voxels": 3,
            "body_mask": {
                "enabled": True,
                "threshold": 0.02,
                "closing_voxels": 3,
                "keep_largest_component": True,
                "dilate_voxels": 3
            },
            "seed": 42
        }
        
        # Run preprocessing
        success, metadata = preprocess_case(
            case_id=case_id,
            raw_dir=raw_dir,
            processed_dir=processed_dir,
            config=config
        )
        
        assert success, "Preprocessing should succeed"
        print(f"  ✓ Preprocessing completed")
        
        # Verify outputs
        processed_image = processed_dir / "images" / f"{case_id}_0000.nii.gz"
        processed_label = processed_dir / "labels" / f"{case_id}.nii.gz"
        body_mask_file = processed_dir / "body_masks" / f"{case_id}.nii.gz"
        metadata_file = processed_dir / "metadata" / f"{case_id}.json"
        
        assert processed_image.exists(), "Processed image should exist"
        assert processed_label.exists(), "Processed label should exist"
        assert body_mask_file.exists(), "Body mask should exist"
        assert metadata_file.exists(), "Metadata should exist"
        print(f"  ✓ All output files created")
        
        # Load and verify body mask
        body_mask_nii = nib.load(body_mask_file)
        body_mask = body_mask_nii.get_fdata().astype(bool)
        assert body_mask.dtype == bool or body_mask.dtype == np.uint8
        assert body_mask.shape == image_shape
        mask_coverage = body_mask.sum() / body_mask.size
        assert mask_coverage > 0.1, f"Body mask should cover some volume (got {mask_coverage:.2%})"
        print(f"  ✓ Body mask valid: {body_mask.sum()} voxels ({mask_coverage:.1%} coverage)")
        
        # Verify metadata contains body mask info
        with open(metadata_file) as f:
            meta = json.load(f)
        assert "body_mask" in meta, "Metadata should contain body_mask info"
        assert "threshold" in meta["body_mask"]
        assert "voxel_counts" in meta["body_mask"]
        print(f"  ✓ Metadata contains body mask info")
        
        # Create split file for dataset
        split_file = tmpdir / "train_list.txt"
        with open(split_file, "w") as f:
            f.write(f"{case_id}\n")
        
        # Test PatchDataset loading
        dataset = PatchDataset(
            data_dir=str(processed_dir),
            split_file=str(split_file),
            patch_size=(16, 16, 16),
            lesion_patch_ratio=0.5,
            augmentation=None,
            seed=42,
            domain_config=None
        )
        
        assert len(dataset.cases) == 1, "Should load 1 case"
        assert dataset.cases[0]["body_mask_path"] is not None, "Should have body mask path"
        assert len(dataset.background_locations) > 0, "Should have background locations"
        print(f"  ✓ PatchDataset loaded: {len(dataset.background_locations)} background locations")
        
        # Test CaseDataset loading with body mask
        val_dataset = CaseDataset(
            data_dir=str(processed_dir),
            split_file=str(split_file),
            domain_config=None,
            return_body_mask=True
        )
        
        assert len(val_dataset) == 1, "Should load 1 case"
        
        # Get sample and verify body mask is returned
        image_t, label_t, case_id_ret, spacing, body_mask_t = val_dataset[0]
        assert body_mask_t is not None, "Should return body mask"
        assert body_mask_t.shape[0] == 1, "Body mask should have channel dimension"
        print(f"  ✓ CaseDataset returns body mask: {body_mask_t.shape}")
        
        print("\n  ✓✓✓ End-to-end workflow test PASSED ✓✓✓")


def test_config_validation():
    """Test that config files have proper body mask configuration"""
    print("\nTesting config file validation...")
    
    configs = [
        "configs/unet_fl70.yaml",
        "configs/unet_mixed_fl_dlbcl.yaml"
    ]
    
    for config_path in configs:
        if not Path(config_path).exists():
            print(f"  ⚠ Skipping {config_path} (not found)")
            continue
            
        with open(config_path) as f:
            config = yaml.safe_load(f)
        
        # Check body_mask configuration
        assert "body_mask" in config["data"], f"{config_path} missing body_mask config"
        body_mask_config = config["data"]["body_mask"]
        
        assert "enabled" in body_mask_config
        assert "threshold" in body_mask_config
        assert "dilate_voxels" in body_mask_config
        assert "closing_voxels" in body_mask_config
        assert "keep_largest_component" in body_mask_config
        assert "apply_to_training_sampling" in body_mask_config
        assert "apply_to_validation" in body_mask_config
        assert "apply_to_inference" in body_mask_config
        
        # Validate reasonable values
        assert 0.0 < body_mask_config["threshold"] < 0.5, "Threshold should be between 0 and 0.5"
        assert 2 <= body_mask_config["dilate_voxels"] <= 10, "Dilate voxels should be 2-10"
        
        print(f"  ✓ {config_path} has valid body_mask config")
    
    print("  ✓ All config files validated")


if __name__ == "__main__":
    try:
        test_config_validation()
        test_end_to_end_workflow()
        
        print("\n" + "="*60)
        print("All integration tests passed successfully! ✓✓✓")
        print("="*60)
    except AssertionError as e:
        print(f"\n✗ Integration test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    except Exception as e:
        print(f"\n✗ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
