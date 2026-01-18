"""
Models package for Lightweight 3D U-Net
"""

# 1. 从当前目录 (models) 导出原有的模型和损失函数
from .unet3d import Lightweight3DUNet
from .losses import FocalTverskyLoss, CombinedLoss, DiceLoss, get_loss_function
from .metrics import (
    calculate_dsc,
    calculate_lesion_metrics,
    calculate_metrics,
    get_connected_components
)
# 注意：models/utils.py 不需要在这里导出，它可以直接通过 from models.utils import ... 使用

# 2. [兼容性层] 从新的 light_unet.datasets 导出数据组件
# 这一步至关重要！它确保了原来的代码（如果还有残留）引用 models.PatchDataset 时不会报错
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