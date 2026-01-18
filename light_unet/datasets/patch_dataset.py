"""
Dataset for 3D patch extraction with class-balanced sampling
"""

import random
import numpy as np
import nibabel as nib
import torch
from pathlib import Path
from torch.utils.data import Dataset
from scipy.ndimage import rotate, zoom

from light_unet.utils import find_case_files
from .constants import DEFAULT_FL_DOMAIN_CONFIG, DEFAULT_FL_PREFIX_MAX, DEFAULT_DLBCL_PREFIX_MIN, DEFAULT_DLBCL_PREFIX_MAX
from .utils import filter_cases_by_domain, create_missing_body_mask_error

class PatchDataset(Dataset):
    """Dataset for 3D patch extraction with class-balanced sampling"""
    def __init__(self, data_dir, split_file, patch_size=(48, 48, 48),
                 lesion_patch_ratio=0.5, augmentation=None, seed=42, domain_config=None,
                 body_mask_config=None):
        self.data_dir = Path(data_dir)
        self.patch_size = patch_size
        self.lesion_patch_ratio = lesion_patch_ratio
        self.augmentation = augmentation
        self.seed = seed
        
        self.body_mask_enabled = body_mask_config.get("enabled", False) if body_mask_config else False
        self.body_mask_required = (self.body_mask_enabled and body_mask_config.get("apply_to_training_sampling", False)) if body_mask_config else False
        
        random.seed(seed)
        np.random.seed(seed)
        
        if domain_config is None:
            domain_config = DEFAULT_FL_DOMAIN_CONFIG.copy()
        
        with open(split_file, "r") as f:
            all_case_ids = [line.strip() for line in f if line.strip()]
        
        self.case_ids = filter_cases_by_domain(all_case_ids, domain_config)
        self.cases = self._load_cases()
        
        print(f"Loaded {len(self.cases)} cases from {split_file}")
        self._check_body_masks()
        
        self.lesion_locations, self.background_locations = self._sample_locations()
        print(f"Found {len(self.lesion_locations)} lesion locations, {len(self.background_locations)} background locations")

    def _load_cases(self):
        cases = []
        for case_id in self.case_ids:
            image_files = find_case_files(self.data_dir, case_id, file_type="image")
            label_files = find_case_files(self.data_dir, case_id, file_type="label")
            
            if len(image_files) > 0 and len(label_files) > 0:
                metadata_path = self.data_dir / "metadata" / f"{case_id}.json"
                body_mask_path = self.data_dir / "body_masks" / f"{case_id}.nii.gz"
                cases.append({
                    "case_id": case_id,
                    "image_path": str(image_files[0]),
                    "label_path": str(label_files[0]),
                    "metadata_path": str(metadata_path) if metadata_path.exists() else None,
                    "body_mask_path": str(body_mask_path) if body_mask_path.exists() else None
                })
        return cases

    def _check_body_masks(self):
        if self.body_mask_required:
            cases_with_masks = sum(1 for c in self.cases if c["body_mask_path"] is not None)
            if cases_with_masks < len(self.cases):
                missing_cases = [c["case_id"] for c in self.cases if c["body_mask_path"] is None]
                raise create_missing_body_mask_error(len(self.cases) - cases_with_masks, len(self.cases), missing_cases, "training")

    def _sample_locations(self):
        lesion_locs = []
        bg_locs = []
        
        for case_idx, case in enumerate(self.cases):
            label = nib.load(case["label_path"]).get_fdata()
            body_mask = self._load_body_mask_array(case)
            
            lesion_coords = np.argwhere(label > 0)
            if len(lesion_coords) > 0:
                num_samples = max(10, len(lesion_coords) // 1000)
                indices = np.random.randint(len(lesion_coords), size=num_samples)
                for idx in indices:
                    lesion_locs.append((case_idx, lesion_coords[idx]))
            
            if body_mask is not None:
                bg_coords = np.argwhere((label == 0) & body_mask)
            else:
                bg_coords = np.argwhere(label == 0)
                
            if len(bg_coords) > 0:
                num_samples = max(10, len(bg_coords) // 5000)
                indices = np.random.randint(len(bg_coords), size=num_samples)
                for idx in indices:
                    bg_locs.append((case_idx, bg_coords[idx]))
        
        return lesion_locs, bg_locs

    def _load_body_mask_array(self, case):
        if case["body_mask_path"] is not None:
            try:
                return nib.load(case["body_mask_path"]).get_fdata().astype(bool)
            except Exception as e:
                if self.body_mask_required:
                    raise RuntimeError(f"Failed to load body mask for {case['case_id']}: {e}")
        return None

    def __len__(self):
        return len(self.lesion_locations) + len(self.background_locations)

    def __getitem__(self, idx):
        if np.random.rand() < self.lesion_patch_ratio and len(self.lesion_locations) > 0:
            idx = np.random.randint(len(self.lesion_locations))
            case_idx, center = self.lesion_locations[idx]
        else:
            if len(self.background_locations) > 0:
                idx = np.random.randint(len(self.background_locations))
                case_idx, center = self.background_locations[idx]
            else:
                idx = np.random.randint(len(self.lesion_locations))
                case_idx, center = self.lesion_locations[idx]
        
        case = self.cases[case_idx]
        image = nib.load(case["image_path"]).get_fdata().astype(np.float32)
        label = nib.load(case["label_path"]).get_fdata().astype(np.float32)
        
        img_patch, label_patch = self._extract_patch(image, label, center)
        if self.augmentation:
            img_patch, label_patch = self._augment(img_patch, label_patch)
            
        return torch.from_numpy(img_patch).unsqueeze(0), torch.from_numpy(label_patch).unsqueeze(0)

    def _extract_patch(self, image, label, center):
        pz, py, px = self.patch_size
        z, y, x = center
        z_start = max(0, z - pz // 2)
        z_end = min(image.shape[0], z_start + pz)
        y_start = max(0, y - py // 2)
        y_end = min(image.shape[1], y_start + py)
        x_start = max(0, x - px // 2)
        x_end = min(image.shape[2], x_start + px)
        
        img_patch = image[z_start:z_end, y_start:y_end, x_start:x_end]
        label_patch = label[z_start:z_end, y_start:y_end, x_start:x_end]
        
        if img_patch.shape != tuple(self.patch_size):
            pad_config = [(0, pz - img_patch.shape[0]), (0, py - img_patch.shape[1]), (0, px - img_patch.shape[2])]
            img_patch = np.pad(img_patch, pad_config, mode='constant', constant_values=0)
            label_patch = np.pad(label_patch, pad_config, mode='constant', constant_values=0)
            
        return img_patch, label_patch

    def _augment(self, image, label):
        if self.augmentation is None:
            return image, label
        
        if self.augmentation.get("random_flip", {}).get("enabled", False):
            if np.random.rand() < self.augmentation["random_flip"].get("prob", 0.5):
                axes = self.augmentation["random_flip"].get("axes", [0, 1, 2])
                axis = random.choice(axes)
                image = np.flip(image, axis=axis).copy()
                label = np.flip(label, axis=axis).copy()
        
        if self.augmentation.get("random_rotation", {}).get("enabled", False):
            if np.random.rand() < self.augmentation["random_rotation"].get("prob", 0.5):
                angle_range = self.augmentation["random_rotation"].get("angle_range", [-15, 15])
                angle = np.random.uniform(angle_range[0], angle_range[1])
                axes = self.augmentation["random_rotation"].get("axes", [[0, 1], [0, 2], [1, 2]])
                axis_pair = random.choice(axes)
                image = rotate(image, angle, axes=axis_pair, reshape=False, order=1, mode='constant', cval=0)
                label = rotate(label, angle, axes=axis_pair, reshape=False, order=0, mode='constant', cval=0)
        
        if self.augmentation.get("random_scale", {}).get("enabled", False):
            if np.random.rand() < self.augmentation["random_scale"].get("prob", 0.3):
                scale_range = self.augmentation["random_scale"].get("scale_range", [0.9, 1.1])
                scale = np.random.uniform(scale_range[0], scale_range[1])
                image = zoom(image, scale, order=1, mode='constant', cval=0)
                label = zoom(label, scale, order=0, mode='constant', cval=0)
                
                if image.shape != tuple(self.patch_size):
                    pz, py, px = self.patch_size
                    z, y, x = image.shape
                    
                    if z > pz:
                        start = (z - pz) // 2
                        image = image[start:start+pz, :, :]
                        label = label[start:start+pz, :, :]
                    if y > py:
                        start = (y - py) // 2
                        image = image[:, start:start+py, :]
                        label = label[:, start:start+py, :]
                    if x > px:
                        start = (x - px) // 2
                        image = image[:, :, start:start+px]
                        label = label[:, :, start:start+px]
                        
                    z, y, x = image.shape
                    if z < pz or y < py or x < px:
                        pad_z = max(0, pz - z)
                        pad_y = max(0, py - y)
                        pad_x = max(0, px - x)
                        image = np.pad(image, ((0, pad_z), (0, pad_y), (0, pad_x)), mode='constant', constant_values=0)
                        label = np.pad(label, ((0, pad_z), (0, pad_y), (0, pad_x)), mode='constant', constant_values=0)

        if self.augmentation.get("intensity_shift", {}).get("enabled", False):
            if np.random.rand() < self.augmentation["intensity_shift"].get("prob", 0.5):
                shift_range = self.augmentation["intensity_shift"].get("shift_range", [-0.1, 0.1])
                shift = np.random.uniform(shift_range[0], shift_range[1])
                image = np.clip(image + shift, 0, 1)
        
        if self.augmentation.get("gaussian_noise", {}).get("enabled", False):
            if np.random.rand() < self.augmentation["gaussian_noise"].get("prob", 0.3):
                sigma = self.augmentation["gaussian_noise"].get("sigma", 0.01)
                noise = np.random.normal(0, sigma, image.shape)
                image = np.clip(image + noise, 0, 1)
        
        return image, label


class MixedPatchDataset(Dataset):
    """Mixed dataset that samples from both FL and DLBCL datasets."""
    def __init__(self, data_dir, split_file, patch_size=(48, 48, 48),
                 lesion_patch_ratio=0.5, augmentation=None, seed=42,
                 domain_config=None, fl_ratio=0.5, body_mask_config=None):
        self.seed = seed
        self.fl_ratio = fl_ratio
        
        fl_config = self._make_domain_config('fl', domain_config)
        self.fl_dataset = PatchDataset(data_dir, split_file, patch_size, lesion_patch_ratio, 
                                     augmentation, seed, fl_config, body_mask_config)
        
        dlbcl_config = self._make_domain_config('dlbcl', domain_config)
        self.dlbcl_dataset = PatchDataset(data_dir, split_file, patch_size, lesion_patch_ratio, 
                                        augmentation, seed + 1, dlbcl_config, body_mask_config)
        
        self.reset_sample_counts()
        print(f"MixedPatchDataset: FL={len(self.fl_dataset.cases)}, DLBCL={len(self.dlbcl_dataset.cases)}, FL ratio={fl_ratio:.2f}")

    def _make_domain_config(self, domain, base_config):
        return {
            'domain': domain,
            'fl_prefix_max': base_config.get('fl_prefix_max', DEFAULT_FL_PREFIX_MAX) if base_config else DEFAULT_FL_PREFIX_MAX,
            'dlbcl_prefix_min': base_config.get('dlbcl_prefix_min', DEFAULT_DLBCL_PREFIX_MIN) if base_config else DEFAULT_DLBCL_PREFIX_MIN,
            'dlbcl_prefix_max': base_config.get('dlbcl_prefix_max', DEFAULT_DLBCL_PREFIX_MAX) if base_config else DEFAULT_DLBCL_PREFIX_MAX
        }
    
    def __len__(self):
        return len(self.fl_dataset) + len(self.dlbcl_dataset)
    
    def __getitem__(self, idx):
        if np.random.rand() < self.fl_ratio and len(self.fl_dataset) > 0:
            self.fl_sample_count += 1
            return self.fl_dataset[np.random.randint(len(self.fl_dataset))]
        elif len(self.dlbcl_dataset) > 0:
            self.dlbcl_sample_count += 1
            return self.dlbcl_dataset[np.random.randint(len(self.dlbcl_dataset))]
        return self.fl_dataset[np.random.randint(len(self.fl_dataset))]

    def reset_sample_counts(self):
        self.fl_sample_count = 0
        self.dlbcl_sample_count = 0
    
    def get_sample_counts(self):
        return {'fl_samples': self.fl_sample_count, 'dlbcl_samples': self.dlbcl_sample_count, 
                'total_samples': self.fl_sample_count + self.dlbcl_sample_count}