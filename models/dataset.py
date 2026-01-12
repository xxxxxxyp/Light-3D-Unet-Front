"""
Dataset and Data Loader for FL Dataset
Implements patch extraction with class-balanced sampling and augmentation
"""

import os
import json
import random
import warnings
import numpy as np
import nibabel as nib
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from scipy import ndimage
from scipy.ndimage import rotate, zoom

from .utils import find_case_files


def filter_cases_by_domain(case_ids, domain_config):
    """
    Filter case IDs by domain based on case ID prefix.
    
    Args:
        case_ids: List of case IDs (e.g., ['0001', '0002', '1000'])
        domain_config: Domain configuration dict with keys:
            - fl_prefix_max: Maximum FL case prefix (e.g., 122)
            - dlbcl_prefix_min: Minimum DLBCL case prefix (e.g., 1000)
            - dlbcl_prefix_max: Maximum DLBCL case prefix (e.g., 1422)
            - domain: 'fl', 'dlbcl', or None (None = all cases)
    
    Returns:
        Filtered list of case IDs
    """
    if domain_config is None or domain_config.get('domain') is None:
        return case_ids
    
    domain = domain_config.get('domain', '').lower()
    fl_prefix_max = domain_config.get('fl_prefix_max', 122)
    dlbcl_prefix_min = domain_config.get('dlbcl_prefix_min', 1000)
    dlbcl_prefix_max = domain_config.get('dlbcl_prefix_max', 1422)
    
    filtered = []
    for case_id in case_ids:
        try:
            # Extract numeric prefix from case ID (first 4 digits)
            prefix = int(case_id[:4])
            
            if domain == 'fl':
                # FL cases: 0000-0122
                if prefix <= fl_prefix_max:
                    filtered.append(case_id)
            elif domain == 'dlbcl':
                # DLBCL cases: 1000-1422
                if dlbcl_prefix_min <= prefix <= dlbcl_prefix_max:
                    filtered.append(case_id)
            else:
                # Unknown domain, keep all
                filtered.append(case_id)
        except (ValueError, IndexError):
            # If case ID doesn't start with numbers, keep it
            warnings.warn(f"Case ID {case_id} doesn't match expected format, skipping filter")
            filtered.append(case_id)
    
    return filtered


class CaseDataset(Dataset):
    """
    Dataset for full-case validation/inference
    Returns full volume (or ROI), label, case_id, and spacing
    """
    def __init__(self, data_dir, split_file, domain_config=None):
        """
        Args:
            data_dir: Path to processed data directory
            split_file: Path to split list file
            domain_config: Optional domain filtering configuration
        """
        self.data_dir = Path(data_dir)

        with open(split_file, "r") as f:
            all_case_ids = [line.strip() for line in f if line.strip()]
        
        # Filter by domain if specified
        self.case_ids = filter_cases_by_domain(all_case_ids, domain_config)

        self.cases = []
        for case_id in self.case_ids:
            image_files = find_case_files(self.data_dir, case_id, file_type="image")
            label_files = find_case_files(self.data_dir, case_id, file_type="label")

            if len(image_files) > 0 and len(label_files) > 0:
                metadata_path = self.data_dir / "metadata" / f"{case_id}.json"
                self.cases.append({
                    "case_id": case_id,
                    "image_path": str(image_files[0]),
                    "label_path": str(label_files[0]),
                    "metadata_path": str(metadata_path) if metadata_path.exists() else None
                })
            else:
                warnings.warn(
                    f"Case {case_id} missing required files (images found: {len(image_files)}, labels found: {len(label_files)}). "
                    f"Expected images under {self.data_dir / 'images'} and labels under {self.data_dir / 'labels'}. Skipping case.",
                    UserWarning
                )

        print(f"Loaded {len(self.cases)} cases from {split_file}")

    def __len__(self):
        return len(self.cases)

    def __getitem__(self, idx):
        case = self.cases[idx]
        image_nii = nib.load(case["image_path"])
        label_nii = nib.load(case["label_path"])

        image = image_nii.get_fdata().astype(np.float32)
        label = label_nii.get_fdata().astype(np.float32)

        spacing = tuple(float(s) for s in image_nii.header.get_zooms()[:3])

        image = torch.from_numpy(image).unsqueeze(0)
        label = torch.from_numpy(label).unsqueeze(0)

        return image, label, case["case_id"], spacing


class PatchDataset(Dataset):
    """
    Dataset for 3D patch extraction with class-balanced sampling
    """
    def __init__(self, data_dir, split_file, patch_size=(48, 48, 48),
                 lesion_patch_ratio=0.5, augmentation=None, seed=42, domain_config=None):
        """
        Args:
            data_dir: Path to processed data directory
            split_file: Path to split list file (train_list.txt, val_list.txt)
            patch_size: Size of patches to extract (z, y, x)
            lesion_patch_ratio: Minimum ratio of lesion-containing patches per batch
            augmentation: Augmentation configuration dictionary
            seed: Random seed
            domain_config: Optional domain filtering configuration
        """
        self.data_dir = Path(data_dir)
        self.patch_size = patch_size
        self.lesion_patch_ratio = lesion_patch_ratio
        self.augmentation = augmentation
        self.seed = seed
        
        # Set random seed
        random.seed(seed)
        np.random.seed(seed)
        
        # Default to FL-only training data when no domain is specified
        if domain_config is None:
            domain_config = {
                'domain': 'fl',
                'fl_prefix_max': 122,
                'dlbcl_prefix_min': 1000,
                'dlbcl_prefix_max': 1422
            }
        
        # Load case list
        with open(split_file, "r") as f:
            all_case_ids = [line.strip() for line in f if line.strip()]
        
        # Filter by domain if specified
        self.case_ids = filter_cases_by_domain(all_case_ids, domain_config)
        
        # Load case data
        self.cases = []
        
        for case_id in self.case_ids:
            # Find image and label files for this case
            image_files = find_case_files(self.data_dir, case_id, file_type="image")
            label_files = find_case_files(self.data_dir, case_id, file_type="label")
            
            if len(image_files) > 0 and len(label_files) > 0:
                # Metadata is stored in metadata/{case_id}.json
                metadata_path = self.data_dir / "metadata" / f"{case_id}.json"
                self.cases.append({
                    "case_id": case_id,
                    "image_path": str(image_files[0]),
                    "label_path": str(label_files[0]),
                    "metadata_path": str(metadata_path) if metadata_path.exists() else None
                })
            else:
                print(f"Warning: Case {case_id} not found (images: {len(image_files)}, labels: {len(label_files)}), skipping...")
        
        print(f"Loaded {len(self.cases)} cases from {split_file}")
        
        # Pre-compute lesion locations for class-balanced sampling
        self.lesion_locations = []
        self.background_locations = []
        
        for case_idx, case in enumerate(self.cases):
            label_nii = nib.load(case["label_path"])
            label = label_nii.get_fdata()
            
            # Find lesion voxels
            lesion_coords = np.argwhere(label > 0)
            
            if len(lesion_coords) > 0:
                # Sample lesion centers
                for _ in range(max(10, len(lesion_coords) // 1000)):  # Sample ~10-100 per case
                    idx = np.random.randint(len(lesion_coords))
                    self.lesion_locations.append((case_idx, lesion_coords[idx]))
            
            # Sample background locations
            background_coords = np.argwhere(label == 0)
            if len(background_coords) > 0:
                for _ in range(max(10, len(background_coords) // 5000)):  # Sample background
                    idx = np.random.randint(len(background_coords))
                    self.background_locations.append((case_idx, background_coords[idx]))
        
        print(f"Found {len(self.lesion_locations)} lesion locations, "
              f"{len(self.background_locations)} background locations")
    
    def __len__(self):
        # Return total number of possible patches (lesion + background)
        return len(self.lesion_locations) + len(self.background_locations)
    
    def _extract_patch(self, image, label, center):
        """Extract patch centered at given location"""
        pz, py, px = self.patch_size
        z, y, x = center
        
        # Calculate patch boundaries
        z_start = max(0, z - pz // 2)
        z_end = min(image.shape[0], z_start + pz)
        y_start = max(0, y - py // 2)
        y_end = min(image.shape[1], y_start + py)
        x_start = max(0, x - px // 2)
        x_end = min(image.shape[2], x_start + px)
        
        # Extract patch
        img_patch = image[z_start:z_end, y_start:y_end, x_start:x_end]
        label_patch = label[z_start:z_end, y_start:y_end, x_start:x_end]
        
        # Pad if necessary
        if img_patch.shape != tuple(self.patch_size):
            pad_z = pz - img_patch.shape[0]
            pad_y = py - img_patch.shape[1]
            pad_x = px - img_patch.shape[2]
            
            img_patch = np.pad(img_patch,
                              ((0, pad_z), (0, pad_y), (0, pad_x)),
                              mode='constant', constant_values=0)
            label_patch = np.pad(label_patch,
                                ((0, pad_z), (0, pad_y), (0, pad_x)),
                                mode='constant', constant_values=0)
        
        return img_patch, label_patch
    
    def _augment(self, image, label):
        """Apply data augmentation"""
        if self.augmentation is None:
            return image, label
        
        # Random flip
        if self.augmentation.get("random_flip", {}).get("enabled", False):
            if np.random.rand() < self.augmentation["random_flip"].get("prob", 0.5):
                axes = self.augmentation["random_flip"].get("axes", [0, 1, 2])
                axis = random.choice(axes)
                image = np.flip(image, axis=axis).copy()
                label = np.flip(label, axis=axis).copy()
        
        # Random rotation
        if self.augmentation.get("random_rotation", {}).get("enabled", False):
            if np.random.rand() < self.augmentation["random_rotation"].get("prob", 0.5):
                angle_range = self.augmentation["random_rotation"].get("angle_range", [-15, 15])
                angle = np.random.uniform(angle_range[0], angle_range[1])
                axes = self.augmentation["random_rotation"].get("axes", [[0, 1], [0, 2], [1, 2]])
                axis_pair = random.choice(axes)
                image = rotate(image, angle, axes=axis_pair, reshape=False, order=1, mode='constant', cval=0)
                label = rotate(label, angle, axes=axis_pair, reshape=False, order=0, mode='constant', cval=0)
        
        # Random scale
        if self.augmentation.get("random_scale", {}).get("enabled", False):
            if np.random.rand() < self.augmentation["random_scale"].get("prob", 0.3):
                scale_range = self.augmentation["random_scale"].get("scale_range", [0.9, 1.1])
                scale = np.random.uniform(scale_range[0], scale_range[1])
                image = zoom(image, scale, order=1, mode='constant', cval=0)
                label = zoom(label, scale, order=0, mode='constant', cval=0)
                
                # Crop or pad to original size
                if image.shape != tuple(self.patch_size):
                    # Simple center crop/pad
                    image = self._resize_to_patch_size(image)
                    label = self._resize_to_patch_size(label)
        
        # Intensity shift
        if self.augmentation.get("intensity_shift", {}).get("enabled", False):
            if np.random.rand() < self.augmentation["intensity_shift"].get("prob", 0.5):
                shift_range = self.augmentation["intensity_shift"].get("shift_range", [-0.1, 0.1])
                shift = np.random.uniform(shift_range[0], shift_range[1])
                image = np.clip(image + shift, 0, 1)
        
        # Gaussian noise
        if self.augmentation.get("gaussian_noise", {}).get("enabled", False):
            if np.random.rand() < self.augmentation["gaussian_noise"].get("prob", 0.3):
                sigma = self.augmentation["gaussian_noise"].get("sigma", 0.01)
                noise = np.random.normal(0, sigma, image.shape)
                image = np.clip(image + noise, 0, 1)
        
        return image, label
    
    def _resize_to_patch_size(self, array):
        """Resize array to patch size by center crop or pad"""
        pz, py, px = self.patch_size
        z, y, x = array.shape
        
        # Crop if larger
        if z > pz:
            start = (z - pz) // 2
            array = array[start:start+pz, :, :]
        if y > py:
            start = (y - py) // 2
            array = array[:, start:start+py, :]
        if x > px:
            start = (x - px) // 2
            array = array[:, :, start:start+px]
        
        # Pad if smaller
        z, y, x = array.shape
        if z < pz or y < py or x < px:
            pad_z = max(0, pz - z)
            pad_y = max(0, py - y)
            pad_x = max(0, px - x)
            array = np.pad(array,
                          ((0, pad_z), (0, pad_y), (0, pad_x)),
                          mode='constant', constant_values=0)
        
        return array
    
    def __getitem__(self, idx):
        """Get a patch sample"""
        # Decide whether to sample lesion or background patch
        if np.random.rand() < self.lesion_patch_ratio and len(self.lesion_locations) > 0:
            # Sample lesion patch
            location_idx = np.random.randint(len(self.lesion_locations))
            case_idx, center = self.lesion_locations[location_idx]
        else:
            # Sample background patch
            if len(self.background_locations) > 0:
                location_idx = np.random.randint(len(self.background_locations))
                case_idx, center = self.background_locations[location_idx]
            elif len(self.lesion_locations) > 0:
                # Fallback to lesion if no background
                location_idx = np.random.randint(len(self.lesion_locations))
                case_idx, center = self.lesion_locations[location_idx]
            else:
                raise ValueError("No valid locations found")
        
        # Load case data
        case = self.cases[case_idx]
        image_nii = nib.load(case["image_path"])
        label_nii = nib.load(case["label_path"])
        
        image = image_nii.get_fdata().astype(np.float32)
        label = label_nii.get_fdata().astype(np.float32)
        
        # Extract patch
        img_patch, label_patch = self._extract_patch(image, label, center)
        
        # Apply augmentation
        img_patch, label_patch = self._augment(img_patch, label_patch)
        
        # Convert to tensor
        img_patch = torch.from_numpy(img_patch).unsqueeze(0)  # Add channel dimension
        label_patch = torch.from_numpy(label_patch).unsqueeze(0)
        
        return img_patch, label_patch


class MixedPatchDataset(Dataset):
    """
    Mixed dataset that samples from both FL and DLBCL datasets with controlled ratio.
    Implements ratio-based sampling for mixed domain training.
    """
    def __init__(self, data_dir, split_file, patch_size=(48, 48, 48),
                 lesion_patch_ratio=0.5, augmentation=None, seed=42,
                 domain_config=None, fl_ratio=0.5):
        """
        Args:
            data_dir: Path to processed data directory
            split_file: Path to split list file
            patch_size: Size of patches to extract
            lesion_patch_ratio: Ratio of lesion-containing patches
            augmentation: Augmentation configuration
            seed: Random seed
            domain_config: Domain configuration with FL and DLBCL ranges
            fl_ratio: Ratio of FL samples in each epoch (0.0-1.0)
        """
        self.seed = seed
        self.fl_ratio = fl_ratio
        
        # Create FL dataset
        fl_config = {
            'domain': 'fl',
            'fl_prefix_max': domain_config.get('fl_prefix_max', 122) if domain_config else 122,
            'dlbcl_prefix_min': domain_config.get('dlbcl_prefix_min', 1000) if domain_config else 1000,
            'dlbcl_prefix_max': domain_config.get('dlbcl_prefix_max', 1422) if domain_config else 1422
        }
        
        self.fl_dataset = PatchDataset(
            data_dir=data_dir,
            split_file=split_file,
            patch_size=patch_size,
            lesion_patch_ratio=lesion_patch_ratio,
            augmentation=augmentation,
            seed=seed,
            domain_config=fl_config
        )
        
        # Create DLBCL dataset
        dlbcl_config = {
            'domain': 'dlbcl',
            'fl_prefix_max': domain_config.get('fl_prefix_max', 122) if domain_config else 122,
            'dlbcl_prefix_min': domain_config.get('dlbcl_prefix_min', 1000) if domain_config else 1000,
            'dlbcl_prefix_max': domain_config.get('dlbcl_prefix_max', 1422) if domain_config else 1422
        }
        
        self.dlbcl_dataset = PatchDataset(
            data_dir=data_dir,
            split_file=split_file,
            patch_size=patch_size,
            lesion_patch_ratio=lesion_patch_ratio,
            augmentation=augmentation,
            seed=seed + 1,  # Different seed for DLBCL
            domain_config=dlbcl_config
        )
        
        # Track sample counts for logging
        self.fl_sample_count = 0
        self.dlbcl_sample_count = 0
        
        print(f"MixedPatchDataset: FL cases={len(self.fl_dataset.cases)}, "
              f"DLBCL cases={len(self.dlbcl_dataset.cases)}, "
              f"FL ratio={fl_ratio:.2f}")
    
    def __len__(self):
        """Return combined dataset length"""
        # Return sum of both datasets to provide enough samples per epoch
        # This ensures we can achieve the target ratio over a full epoch
        return len(self.fl_dataset) + len(self.dlbcl_dataset)
    
    def __getitem__(self, idx):
        """Sample from FL or DLBCL based on ratio"""
        # Note: We use random sampling independent of idx to achieve desired ratio
        # The sampling is stochastic but reproducible within each epoch due to
        # DataLoader's worker seeding
        if np.random.rand() < self.fl_ratio and len(self.fl_dataset) > 0:
            # Sample from FL
            fl_idx = np.random.randint(len(self.fl_dataset))
            self.fl_sample_count += 1
            return self.fl_dataset[fl_idx]
        elif len(self.dlbcl_dataset) > 0:
            # Sample from DLBCL
            dlbcl_idx = np.random.randint(len(self.dlbcl_dataset))
            self.dlbcl_sample_count += 1
            return self.dlbcl_dataset[dlbcl_idx]
        elif len(self.fl_dataset) > 0:
            # Fallback to FL if DLBCL empty
            fl_idx = np.random.randint(len(self.fl_dataset))
            self.fl_sample_count += 1
            return self.fl_dataset[fl_idx]
        else:
            raise ValueError("Both FL and DLBCL datasets are empty")
    
    def reset_sample_counts(self):
        """Reset sample counts for new epoch"""
        self.fl_sample_count = 0
        self.dlbcl_sample_count = 0
    
    def get_sample_counts(self):
        """Get current sample counts"""
        return {
            'fl_samples': self.fl_sample_count,
            'dlbcl_samples': self.dlbcl_sample_count,
            'total_samples': self.fl_sample_count + self.dlbcl_sample_count
        }


def get_data_loader(data_dir, split_file, config, is_train=True):
    """
    Create data loader from configuration
    
    Args:
        data_dir: Path to processed data directory
        split_file: Path to split list file
        config: Full configuration dictionary
        is_train: Whether this is for training (enables augmentation)
    
    Returns:
        data_loader: PyTorch DataLoader (or tuple of (loader, dataset) for mixed training)
    """
    augmentation = config["augmentation"] if is_train else None
    
    if is_train:
        # Check if mixed domain training is enabled
        mixed_config = config.get("training", {}).get("mixed_domains", {})
        use_mixed = mixed_config.get("enabled", False)
        
        if use_mixed:
            # Use MixedPatchDataset for mixed FL + DLBCL training
            domain_config = config.get("data", {}).get("domains", {})
            fl_ratio = mixed_config.get("fl_ratio", 0.5)
            
            dataset = MixedPatchDataset(
                data_dir=data_dir,
                split_file=split_file,
                patch_size=config["data"]["patch_size"],
                lesion_patch_ratio=config["training"]["class_balanced_sampling"]["lesion_patch_ratio"],
                augmentation=augmentation,
                seed=config["experiment"]["seed"],
                domain_config=domain_config,
                fl_ratio=fl_ratio
            )
        else:
            # Use standard PatchDataset
            dataset = PatchDataset(
                data_dir=data_dir,
                split_file=split_file,
                patch_size=config["data"]["patch_size"],
                lesion_patch_ratio=config["training"]["class_balanced_sampling"]["lesion_patch_ratio"],
                augmentation=augmentation,
                seed=config["experiment"]["seed"]
            )
        
        batch_size = config["training"]["batch_size"]
        shuffle = True
    else:
        # Validation: Always use FL-only when mixed training is enabled
        mixed_config = config.get("training", {}).get("mixed_domains", {})
        use_mixed = mixed_config.get("enabled", False)
        
        if use_mixed:
            # Filter validation to FL-only
            domain_config = config.get("data", {}).get("domains", {})
            fl_config = {
                'domain': 'fl',
                'fl_prefix_max': domain_config.get('fl_prefix_max', 122),
                'dlbcl_prefix_min': domain_config.get('dlbcl_prefix_min', 1000),
                'dlbcl_prefix_max': domain_config.get('dlbcl_prefix_max', 1422)
            }
            dataset = CaseDataset(
                data_dir=data_dir,
                split_file=split_file,
                domain_config=fl_config
            )
        else:
            # Standard validation (no filtering)
            dataset = CaseDataset(
                data_dir=data_dir,
                split_file=split_file
            )
        
        batch_size = 1
        shuffle = False
    
    data_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=16,
        pin_memory=True
    )
    
    # Return both loader and dataset for mixed training (to access sample counts)
    if is_train and isinstance(dataset, MixedPatchDataset):
        return data_loader, dataset
    
    return data_loader


if __name__ == "__main__":
    # Test dataset
    import yaml
    
    config_path = "configs/unet_fl70.yaml"
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    
    # Create dummy data for testing
    data_dir = "data/processed"
    split_file = "data/splits/train_list.txt"
    
    if Path(split_file).exists():
        loader = get_data_loader(data_dir, split_file, config, is_train=True)
        
        print(f"DataLoader created with {len(loader.dataset)} samples")
        
        # Test one batch
        for img_batch, label_batch in loader:
            print(f"Image batch shape: {img_batch.shape}")
            print(f"Label batch shape: {label_batch.shape}")
            print(f"Image range: [{img_batch.min():.3f}, {img_batch.max():.3f}]")
            print(f"Label unique values: {torch.unique(label_batch)}")
            break
    else:
        print(f"Split file {split_file} not found. Run split_dataset.py first.")
