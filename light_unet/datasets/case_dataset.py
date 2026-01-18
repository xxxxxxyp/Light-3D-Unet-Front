"""
Dataset for full-case validation/inference
"""

import warnings
import numpy as np
import nibabel as nib
import torch
from pathlib import Path
from torch.utils.data import Dataset

from models.utils import find_case_files
from .utils import filter_cases_by_domain, create_missing_body_mask_error

class CaseDataset(Dataset):
    """Dataset for full-case validation/inference"""
    def __init__(self, data_dir, split_file, domain_config=None, return_body_mask=False, body_mask_required=False):
        self.data_dir = Path(data_dir)
        self.return_body_mask = return_body_mask
        self.body_mask_required = body_mask_required

        with open(split_file, "r") as f:
            all_case_ids = [line.strip() for line in f if line.strip()]
        
        self.case_ids = filter_cases_by_domain(all_case_ids, domain_config)

        self.cases = []
        for case_id in self.case_ids:
            image_files = find_case_files(self.data_dir, case_id, file_type="image")
            label_files = find_case_files(self.data_dir, case_id, file_type="label")

            if len(image_files) > 0 and len(label_files) > 0:
                metadata_path = self.data_dir / "metadata" / f"{case_id}.json"
                body_mask_path = self.data_dir / "body_masks" / f"{case_id}.nii.gz"
                
                self.cases.append({
                    "case_id": case_id,
                    "image_path": str(image_files[0]),
                    "label_path": str(label_files[0]),
                    "metadata_path": str(metadata_path) if metadata_path.exists() else None,
                    "body_mask_path": str(body_mask_path) if body_mask_path.exists() else None
                })
            else:
                warnings.warn(f"Case {case_id} missing files, skipping.", UserWarning)

        print(f"Loaded {len(self.cases)} cases from {split_file}")
        self._check_body_masks()

    def _check_body_masks(self):
        if self.body_mask_required:
            cases_with_masks = sum(1 for c in self.cases if c["body_mask_path"] is not None)
            if cases_with_masks < len(self.cases):
                missing_cases = [c["case_id"] for c in self.cases if c["body_mask_path"] is None]
                raise create_missing_body_mask_error(len(self.cases) - cases_with_masks, len(self.cases), missing_cases, "validation/inference")
        elif self.return_body_mask:
            cases_with_masks = sum(1 for c in self.cases if c["body_mask_path"] is not None)
            print(f"Body masks available for {cases_with_masks}/{len(self.cases)} cases")

    def __len__(self):
        return len(self.cases)

    def __getitem__(self, idx):
        case = self.cases[idx]
        image_nii = nib.load(case["image_path"])
        label_nii = nib.load(case["label_path"])

        image = torch.from_numpy(image_nii.get_fdata().astype(np.float32)).unsqueeze(0)
        label = torch.from_numpy(label_nii.get_fdata().astype(np.float32)).unsqueeze(0)
        spacing = tuple(float(s) for s in image_nii.header.get_zooms()[:3])

        if self.return_body_mask:
            body_mask = self._load_body_mask(case, label)
            return image, label, case["case_id"], spacing, body_mask
        else:
            return image, label, case["case_id"], spacing

    def _load_body_mask(self, case, label_tensor):
        if case["body_mask_path"] is not None:
            try:
                body_mask = nib.load(case["body_mask_path"]).get_fdata().astype(np.float32)
                return torch.from_numpy(body_mask).unsqueeze(0)
            except Exception as e:
                if self.body_mask_required:
                    raise RuntimeError(f"Failed to load required body mask for {case['case_id']}: {e}") from e
                warnings.warn(f"Failed to load body mask for {case['case_id']}: {e}. Using full volume.")
        elif self.body_mask_required:
            raise FileNotFoundError(f"Body mask required but missing for {case['case_id']}")
        
        return torch.ones_like(label_tensor)