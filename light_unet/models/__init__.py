"""
Models package for Lightweight 3D U-Net
"""

from .unet3d import Lightweight3DUNet
from .losses import FocalTverskyLoss, CombinedLoss, DiceLoss, get_loss_function
from .metrics import (
    calculate_dsc,
    calculate_lesion_metrics,
    calculate_metrics,
    get_connected_components
)

# [CHANGE] Import dataset components from the new light_unet.datasets package
# This acts as a compatibility shim
from light_unet.datasets import (
    CaseDataset, 
    PatchDataset, 
    MixedPatchDataset, 
    get_data_loader, 
    filter_cases_by_domain
)

__all__ = [
    'Lightweight3DUNet',
    'FocalTverskyLoss',
    'CombinedLoss',
    'DiceLoss',
    'get_loss_function',
    'CaseDataset',
    'PatchDataset',
    'MixedPatchDataset',
    'get_data_loader',
    'filter_cases_by_domain',
    'calculate_dsc',
    'calculate_lesion_metrics',
    'calculate_metrics',
    'get_connected_components',
]