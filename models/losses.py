"""
Loss Functions for Lesion Segmentation
Focal Tversky Loss with configurable parameters
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalTverskyLoss(nn.Module):
    """
    Focal Tversky Loss for imbalanced segmentation
    
    Args:
        alpha: Weight for false negatives (higher = prioritize recall)
        beta: Weight for false positives (higher = prioritize precision)
        gamma: Focal parameter (higher = focus on hard examples)
        smooth: Smoothing factor to avoid division by zero
    """
    def __init__(self, alpha=0.7, beta=0.3, gamma=0.75, smooth=1e-6):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.smooth = smooth
        
        assert abs(alpha + beta - 1.0) < 1e-6, f"alpha + beta must equal 1.0, got {alpha + beta}"
    
    def forward(self, pred, target):
        """
        Args:
            pred: Predicted probabilities [B, C, D, H, W]
            target: Ground truth binary masks [B, C, D, H, W]
        
        Returns:
            loss: Focal Tversky loss value
        """
        # Flatten spatial dimensions
        pred = pred.view(-1)
        target = target.view(-1)
        
        # Calculate true positives, false positives, false negatives
        tp = (pred * target).sum()
        fp = (pred * (1 - target)).sum()
        fn = ((1 - pred) * target).sum()
        
        # Tversky index
        tversky_index = (tp + self.smooth) / (tp + self.alpha * fn + self.beta * fp + self.smooth)
        
        # Focal Tversky loss
        focal_tversky = (1 - tversky_index) ** self.gamma
        
        return focal_tversky


class CombinedLoss(nn.Module):
    """
    Combined Focal Tversky Loss and Binary Cross Entropy
    Fallback option if training is unstable
    """
    def __init__(self, ftl_weight=0.8, bce_weight=0.2, alpha=0.7, beta=0.3, gamma=0.75):
        super().__init__()
        self.ftl_weight = ftl_weight
        self.bce_weight = bce_weight
        
        self.focal_tversky = FocalTverskyLoss(alpha=alpha, beta=beta, gamma=gamma)
        self.bce = nn.BCELoss()
        
        assert abs(ftl_weight + bce_weight - 1.0) < 1e-6, \
            f"Weights must sum to 1.0, got {ftl_weight + bce_weight}"
    
    def forward(self, pred, target):
        """
        Args:
            pred: Predicted probabilities [B, C, D, H, W]
            target: Ground truth binary masks [B, C, D, H, W]
        
        Returns:
            loss: Combined loss value
        """
        ftl = self.focal_tversky(pred, target)
        bce = self.bce(pred.view(-1), target.view(-1))
        
        return self.ftl_weight * ftl + self.bce_weight * bce


class DiceLoss(nn.Module):
    """
    Dice Loss for segmentation (for comparison/debugging)
    """
    def __init__(self, smooth=1e-6):
        super().__init__()
        self.smooth = smooth
    
    def forward(self, pred, target):
        """
        Args:
            pred: Predicted probabilities [B, C, D, H, W]
            target: Ground truth binary masks [B, C, D, H, W]
        
        Returns:
            loss: Dice loss value
        """
        pred = pred.view(-1)
        target = target.view(-1)
        
        intersection = (pred * target).sum()
        union = pred.sum() + target.sum()
        
        dice = (2.0 * intersection + self.smooth) / (union + self.smooth)
        
        return 1.0 - dice


def get_loss_function(config):
    """
    Factory function to create loss function from config
    
    Args:
        config: Loss configuration dictionary
    
    Returns:
        loss_fn: Loss function instance
    """
    loss_name = config.get("name", "FocalTverskyLoss")
    
    if config.get("use_combined_loss", False):
        # Use combined loss (FTL + BCE)
        weights = config.get("combined_loss_weights", {"focal_tversky": 0.8, "bce": 0.2})
        return CombinedLoss(
            ftl_weight=weights["focal_tversky"],
            bce_weight=weights["bce"],
            alpha=config.get("alpha", 0.7),
            beta=config.get("beta", 0.3),
            gamma=config.get("gamma", 0.75)
        )
    elif loss_name == "FocalTverskyLoss":
        return FocalTverskyLoss(
            alpha=config.get("alpha", 0.7),
            beta=config.get("beta", 0.3),
            gamma=config.get("gamma", 0.75)
        )
    elif loss_name == "DiceLoss":
        return DiceLoss()
    else:
        raise ValueError(f"Unknown loss function: {loss_name}")


def test_losses():
    """Test loss functions"""
    # Create dummy data
    pred = torch.rand(2, 1, 48, 48, 48)  # Batch of 2, single channel
    target = torch.randint(0, 2, (2, 1, 48, 48, 48)).float()
    
    # Test Focal Tversky Loss
    ftl = FocalTverskyLoss(alpha=0.7, beta=0.3, gamma=0.75)
    loss_ftl = ftl(pred, target)
    print(f"Focal Tversky Loss: {loss_ftl.item():.4f}")
    
    # Test Combined Loss
    combined = CombinedLoss(ftl_weight=0.8, bce_weight=0.2)
    loss_combined = combined(pred, target)
    print(f"Combined Loss: {loss_combined.item():.4f}")
    
    # Test Dice Loss
    dice = DiceLoss()
    loss_dice = dice(pred, target)
    print(f"Dice Loss: {loss_dice.item():.4f}")
    
    print("Loss function tests passed!")


if __name__ == "__main__":
    test_losses()
