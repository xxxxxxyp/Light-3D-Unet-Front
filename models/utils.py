"""
Utility functions for file operations and inference
"""

import numpy as np
import torch
from pathlib import Path
from typing import List, Union, Tuple


def sliding_window_inference_3d(
    image: np.ndarray,
    model: torch.nn.Module,
    patch_size: Tuple[int, int, int] = (48, 48, 48),
    overlap: float = 0.5,
    device: torch.device = None,
    use_gaussian: bool = True
) -> np.ndarray:
    """
    Perform 3D sliding-window inference without MONAI.
    
    Args:
        image: Input image volume [D, H, W] or [1, D, H, W]
        model: PyTorch model for inference
        patch_size: Size of sliding window (z, y, x)
        overlap: Overlap ratio between patches (0.0-1.0)
        device: Device to run inference on
        use_gaussian: Whether to use Gaussian weighting for overlap blending
    
    Returns:
        prob_map: Probability map [D, H, W]
    """
    if device is None:
        device = next(model.parameters()).device
    
    # Remove batch dimension if present
    if len(image.shape) == 4 and image.shape[0] == 1:
        image = image[0]
    
    if len(image.shape) != 3:
        raise ValueError(f"Expected 3D image [D, H, W], got shape {image.shape}")
    
    d, h, w = image.shape
    pd, ph, pw = patch_size
    
    # Calculate stride from overlap
    stride_d = max(1, int(pd * (1 - overlap)))
    stride_h = max(1, int(ph * (1 - overlap)))
    stride_w = max(1, int(pw * (1 - overlap)))
    
    # Create Gaussian importance map if requested
    if use_gaussian:
        importance_map = _get_gaussian_importance_map(patch_size)
    else:
        importance_map = np.ones(patch_size, dtype=np.float32)
    
    # Output probability map and count map for averaging
    prob_map = np.zeros((d, h, w), dtype=np.float32)
    count_map = np.zeros((d, h, w), dtype=np.float32)
    
    # Generate all sliding window positions
    # Use max(0, ...) to handle volumes smaller than patch size
    z_positions = list(range(0, max(0, d - pd + 1), stride_d)) if d >= pd else []
    y_positions = list(range(0, max(0, h - ph + 1), stride_h)) if h >= ph else []
    x_positions = list(range(0, max(0, w - pw + 1), stride_w)) if w >= pw else []
    
    # Ensure we cover the entire volume for dimensions larger than patch
    if d > pd and (len(z_positions) == 0 or z_positions[-1] + pd < d):
        z_positions.append(d - pd)
    if h > ph and (len(y_positions) == 0 or y_positions[-1] + ph < h):
        y_positions.append(h - ph)
    if w > pw and (len(x_positions) == 0 or x_positions[-1] + pw < w):
        x_positions.append(w - pw)
    
    # Handle case where volume is smaller than patch size
    if len(z_positions) == 0:
        z_positions = [0]
    if len(y_positions) == 0:
        y_positions = [0]
    if len(x_positions) == 0:
        x_positions = [0]
    
    # Sliding window inference
    model.eval()
    with torch.no_grad():
        for z in z_positions:
            for y in y_positions:
                for x in x_positions:
                    # Calculate actual region to extract
                    z_end = min(z + pd, d)
                    y_end = min(y + ph, h)
                    x_end = min(x + pw, w)
                    
                    actual_d = z_end - z
                    actual_h = y_end - y
                    actual_w = x_end - x
                    
                    # Extract patch
                    patch = image[z:z_end, y:y_end, x:x_end]
                    
                    # Pad if necessary (when volume is smaller than patch)
                    needs_padding = patch.shape != patch_size
                    if needs_padding:
                        pad_d = pd - patch.shape[0]
                        pad_h = ph - patch.shape[1]
                        pad_w = pw - patch.shape[2]
                        patch = np.pad(
                            patch,
                            ((0, pad_d), (0, pad_h), (0, pad_w)),
                            mode='constant',
                            constant_values=0
                        )
                    
                    # Convert to tensor and add batch and channel dimensions
                    patch_tensor = torch.from_numpy(patch).unsqueeze(0).unsqueeze(0).float().to(device)
                    
                    # Predict
                    pred = model(patch_tensor)
                    # Safely extract prediction - handle different output shapes
                    pred = pred.squeeze().cpu().numpy()
                    # Ensure we have 3D output
                    if pred.ndim != 3:
                        raise ValueError(f"Expected 3D model output, got shape {pred.shape}")
                    
                    # Remove padding if it was added
                    if needs_padding:
                        pred = pred[:actual_d, :actual_h, :actual_w]
                    
                    # Get importance weights for the actual region size
                    weights = importance_map[:actual_d, :actual_h, :actual_w]
                    
                    # Accumulate weighted predictions
                    prob_map[z:z_end, y:y_end, x:x_end] += pred * weights
                    count_map[z:z_end, y:y_end, x:x_end] += weights
    
    # Average overlapping predictions
    prob_map = np.divide(prob_map, count_map, where=count_map > 0, out=prob_map)
    
    return prob_map


def _get_gaussian_importance_map(patch_size: Tuple[int, int, int]) -> np.ndarray:
    """
    Generate a 3D Gaussian importance map for patch blending.
    
    Args:
        patch_size: Size of the patch (z, y, x)
    
    Returns:
        importance_map: 3D Gaussian weight map
    """
    # Create 1D Gaussian for each dimension
    def gaussian_1d(length):
        # Center at middle of the patch
        center = length / 2.0
        # Sigma such that 3*sigma ≈ length/2 (covers most of the patch)
        sigma = length / 6.0
        x = np.arange(length)
        g = np.exp(-((x - center) ** 2) / (2 * sigma ** 2))
        return g
    
    # Generate 3D importance map as outer product of 1D Gaussians
    z_gaussian = gaussian_1d(patch_size[0])
    y_gaussian = gaussian_1d(patch_size[1])
    x_gaussian = gaussian_1d(patch_size[2])
    
    # Create 3D map
    importance_map = np.einsum('i,j,k->ijk', z_gaussian, y_gaussian, x_gaussian)
    
    # Normalize to [0, 1]
    importance_map = importance_map / importance_map.max()
    
    return importance_map.astype(np.float32)


def find_case_files(base_dir: Union[Path, str], case_id: str, file_type: str = "image") -> List[Path]:
    """
    Find image or label files for a specific case
    
    Args:
        base_dir: Base directory containing images/ or labels/ subdirectory
        case_id: Case identifier (e.g., "0001")
        file_type: Type of file to find ("image" or "label")
    
    Returns:
        List of matching file paths (sorted for consistent ordering)
    """
    base_dir = Path(base_dir)
    
    if file_type == "image":
        # Images have pattern: case_id_*.nii or case_id_*.nii.gz (e.g., 0001_0000.nii.gz)
        subdir = base_dir / "images"
        patterns = [f"{case_id}_*.nii.gz", f"{case_id}_*.nii"]
    elif file_type == "label":
        # Labels have pattern: case_id.nii or case_id.nii.gz (e.g., 0001.nii.gz)
        subdir = base_dir / "labels"
        patterns = [f"{case_id}.nii.gz", f"{case_id}.nii"]
    else:
        raise ValueError(f"Invalid file_type: {file_type}. Must be 'image' or 'label'")
    
    files = []
    if subdir.exists():
        for pattern in patterns:
            files.extend(subdir.glob(pattern))
    
    # Sort for consistent ordering across systems
    return sorted(files)
