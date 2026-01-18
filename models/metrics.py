"""
Evaluation Metrics for Lesion Segmentation
Includes lesion-wise recall, precision, voxel-wise DSC, and FP per case
"""

import numpy as np
import warnings
from scipy import ndimage
from sklearn.metrics import precision_score, recall_score

DEFAULT_SPACING = (4.0, 4.0, 4.0)
SMOOTH = 1e-6
SPATIAL_DIMENSIONS = 3

# Deprecated metric key mappings
DEPRECATED_KEYS = {
    'dsc': 'voxel_wise_dsc_micro',
    'recall': 'lesion_wise_recall',
    'precision': 'lesion_wise_precision'
}

# Track which deprecated keys have been warned about (per process)
_deprecated_warnings_issued = set()


class MetricsDict(dict):
    """
    Dictionary wrapper that tracks access to deprecated metric keys.
    Issues one-time warnings when deprecated keys are accessed.
    """
    
    def __getitem__(self, key):
        if key in DEPRECATED_KEYS and key not in _deprecated_warnings_issued:
            new_key = DEPRECATED_KEYS[key]
            warnings.warn(
                f"Metric key '{key}' is deprecated. Use '{new_key}' instead. "
                f"Deprecated keys are maintained for backward compatibility but may be removed in the future.",
                DeprecationWarning,
                stacklevel=2
            )
            _deprecated_warnings_issued.add(key)
        return super().__getitem__(key)
    
    def get(self, key, default=None):
        """Override get to also trigger deprecation warnings"""
        if key in self:
            return self[key]
        return default


def calculate_dsc(pred, target, smooth=SMOOTH):
    """
    Calculate Dice Similarity Coefficient (DSC)
    
    Args:
        pred: Predicted binary mask
        target: Ground truth binary mask
        smooth: Smoothing factor
    
    Returns:
        dsc: Dice coefficient
    """
    pred = np.ravel(pred)
    target = np.ravel(target)
    
    intersection = (pred * target).sum()
    union = pred.sum() + target.sum()
    
    dsc = (2.0 * intersection + smooth) / (union + smooth)
    
    return dsc


def get_connected_components(mask, min_size=0):
    """
    Get connected components from binary mask
    
    Args:
        mask: Binary mask
        min_size: Minimum component size (in voxels)
    
    Returns:
        labeled: Labeled array where each component has unique integer
        num_components: Number of components
    """
    labeled, num_components = ndimage.label(mask)
    
    if min_size > 0:
        # Filter small components
        component_sizes = np.bincount(labeled.ravel())
        small_components = component_sizes < min_size
        small_components[0] = False  # Keep background
        
        labeled[small_components[labeled]] = 0
        
        # Relabel
        labeled, num_components = ndimage.label(labeled > 0)
    
    return labeled, num_components


def calculate_iou(pred_component, target_component):
    """Calculate IoU between two binary masks"""
    intersection = np.logical_and(pred_component, target_component).sum()
    union = np.logical_or(pred_component, target_component).sum()
    
    if union == 0:
        return 0.0
    
    return intersection / union


def calculate_center_distance(pred_component, target_component, spacing=(1.0, 1.0, 1.0)):
    """
    Calculate distance between centers of mass
    
    Args:
        pred_component: Predicted component mask
        target_component: Target component mask
        spacing: Voxel spacing (z, y, x) in mm
    
    Returns:
        distance: Distance in mm
    """
    pred_center = ndimage.center_of_mass(pred_component)
    target_center = ndimage.center_of_mass(target_component)
    
    # Handle potential 4D output from center_of_mass (batch dimension)
    if len(pred_center) > 3:
        pred_center = pred_center[:3]
    if len(target_center) > 3:
        target_center = target_center[:3]
    
    # Convert to physical coordinates
    pred_center_mm = np.array(pred_center) * np.array(spacing)
    target_center_mm = np.array(target_center) * np.array(spacing)
    
    distance = np.linalg.norm(pred_center_mm - target_center_mm)
    
    return distance


def _compute_component_centers(labeled):
    """Compute centers of mass for all components in a labeled volume."""
    if labeled.size == 0:
        return np.empty((0, 3), dtype=np.float64)
    num_components = int(labeled.max())
    if num_components == 0:
        return np.empty((0, 3), dtype=np.float64)
    
    indices = np.arange(1, num_components + 1)
    centers = ndimage.center_of_mass(
        np.ones_like(labeled, dtype=np.float32),
        labels=labeled,
        index=indices
    )
    centers = np.atleast_2d(np.asarray(centers, dtype=np.float64))
    if centers.shape[1] > 3:
        centers = centers[:, :3]
    return centers


def match_components(pred_labeled, target_labeled, iou_threshold=0.1,
                    distance_threshold_mm=10.0, spacing=(4.0, 4.0, 4.0)):
    """
    Match predicted components to target components
    
    Args:
        pred_labeled: Labeled prediction array
        target_labeled: Labeled target array
        iou_threshold: IoU threshold for matching
        distance_threshold_mm: Center distance threshold in mm
        spacing: Voxel spacing (z, y, x) in mm
    
    Returns:
        matches: List of (pred_id, target_id) tuples
        unmatched_pred: List of unmatched prediction component IDs
        unmatched_target: List of unmatched target component IDs
    """
    num_pred = int(pred_labeled.max())
    num_target = int(target_labeled.max())
    
    if num_pred == 0 or num_target == 0:
        return [], list(range(1, num_pred + 1)), list(range(1, num_target + 1))
    
    pred_flat = np.ravel(pred_labeled).astype(np.int64, copy=False)
    target_flat = np.ravel(target_labeled).astype(np.int64, copy=False)
    
    # Intersection counts for all component pairs using encoded indices
    pair_offset = np.int64(num_target + 1)
    max_index = (np.int64(num_pred) + 1) * pair_offset
    combined_idx = pred_flat * pair_offset + target_flat
    intersection = np.bincount(
        combined_idx,
        minlength=max_index
    ).reshape(num_pred + 1, num_target + 1)
    intersection[0, :] = 0
    intersection[:, 0] = 0
    
    pred_sizes = np.bincount(pred_flat, minlength=num_pred + 1)
    target_sizes = np.bincount(target_flat, minlength=num_target + 1)
    
    union = pred_sizes[:, None] + target_sizes[None, :] - intersection
    iou_matrix = np.divide(
        intersection,
        union,
        out=np.zeros_like(intersection, dtype=np.float32),
        where=union > 0
    )
    
    # Precompute component centers (in mm)
    spacing_arr = np.asarray(spacing, dtype=np.float64)
    pred_centers = _compute_component_centers(pred_labeled) * spacing_arr
    target_centers = _compute_component_centers(target_labeled) * spacing_arr
    if pred_centers.size and target_centers.size:
        diff = pred_centers[:, None, :] - target_centers[None, :, :]
        distance_matrix = np.linalg.norm(diff, axis=2)
    else:
        # If either side is empty, use inf to prevent distance-based matches
        distance_matrix = np.full((num_pred, num_target), np.inf, dtype=np.float64)
    
    matches = []
    matched_pred = set()
    matched_target_mask = np.zeros(num_target, dtype=bool)
    
    for pred_id in range(1, num_pred + 1):
        iou_row = iou_matrix[pred_id, 1:]
        if distance_matrix.size > 0:
            distance_row = distance_matrix[pred_id - 1]
            distance_criteria = distance_row <= distance_threshold_mm
            valid_mask = ~matched_target_mask & (
                (iou_row >= iou_threshold) | distance_criteria
            )
        else:
            valid_mask = ~matched_target_mask & (iou_row >= iou_threshold)
        if not np.any(valid_mask):
            continue
        
        candidate_ious = np.where(valid_mask, iou_row, -np.inf)
        best_target_idx = int(np.argmax(candidate_ious))
        best_target_id = best_target_idx + 1
        matches.append((pred_id, best_target_id))
        matched_pred.add(pred_id)
        matched_target_mask[best_target_idx] = True
    
    unmatched_pred = [i for i in range(1, num_pred + 1) if i not in matched_pred]
    unmatched_target = [i for i in range(1, num_target + 1) if not matched_target_mask[i - 1]]
    
    return matches, unmatched_pred, unmatched_target


def calculate_lesion_metrics(pred, target, threshold=0.5, min_size_voxels=0,
                             iou_threshold=0.1, distance_threshold_mm=10.0,
                             spacing=(4.0, 4.0, 4.0)):
    """
    Calculate lesion-wise metrics
    
    Args:
        pred: Predicted probability map or binary mask [D, H, W] or [B, 1, D, H, W]
        target: Ground truth binary mask [D, H, W] or [B, 1, D, H, W]
        threshold: Probability threshold for binarization
        min_size_voxels: Minimum lesion size in voxels
        iou_threshold: IoU threshold for matching
        distance_threshold_mm: Center distance threshold
        spacing: Voxel spacing (z, y, x) in mm
    
    Returns:
        metrics: Dictionary with recall, precision, f1, tp, fp, fn
    """
    # Handle batch dimension
    if len(pred.shape) == 5:
        pred = pred[:, 0, :, :, :]  # Remove channel dimension
    if len(target.shape) == 5:
        target = target[:, 0, :, :, :]
    if len(pred.shape) == 4 and pred.shape[0] == 1:
        pred = pred[0]
    if len(target.shape) == 4 and target.shape[0] == 1:
        target = target[0]
    
    # Binarize prediction
    pred_binary = (pred >= threshold).astype(np.int32)
    target_binary = (target >= 0.5).astype(np.int32)
    
    # Get connected components
    pred_labeled, num_pred = get_connected_components(pred_binary, min_size=min_size_voxels)
    target_labeled, num_target = get_connected_components(target_binary, min_size=min_size_voxels)
    
    if num_target == 0:
        # No ground truth lesions
        if num_pred == 0:
            return {"recall": 1.0, "precision": 1.0, "f1": 1.0, "tp": 0, "fp": 0, "fn": 0}
        else:
            return {"recall": 0.0, "precision": 0.0, "f1": 0.0, "tp": 0, "fp": num_pred, "fn": 0}
    
    if num_pred == 0:
        # No predictions
        return {"recall": 0.0, "precision": 0.0, "f1": 0.0, "tp": 0, "fp": 0, "fn": num_target}
    
    # Match components
    matches, unmatched_pred, unmatched_target = match_components(
        pred_labeled, target_labeled,
        iou_threshold=iou_threshold,
        distance_threshold_mm=distance_threshold_mm,
        spacing=spacing
    )
    
    tp = len(matches)
    fp = len(unmatched_pred)
    fn = len(unmatched_target)
    
    # Calculate recall and precision
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    
    return {
        "recall": recall,
        "precision": precision,
        "f1": f1,
        "tp": tp,
        "fp": fp,
        "fn": fn
    }


def _normalize_spacing_per_case(spacing, num_cases):
    """Return a spacing list for each case."""
    if num_cases == 0:
        return []
    if isinstance(spacing, np.ndarray):
        spacing = spacing.tolist()
    if isinstance(spacing, (list, tuple)):
        if len(spacing) == 0:
            return [tuple(map(float, DEFAULT_SPACING)) for _ in range(num_cases)]
        if len(spacing) == num_cases and isinstance(spacing[0], (list, tuple, np.ndarray)):
            return [tuple(map(float, s)) for s in spacing]
        if len(spacing) == SPATIAL_DIMENSIONS and all(isinstance(s, (int, float, np.floating)) for s in spacing):
            return [tuple(map(float, spacing)) for _ in range(num_cases)]
    return [tuple(map(float, DEFAULT_SPACING)) for _ in range(num_cases)]


def calculate_metrics(predictions, labels, threshold=0.5, spacing=DEFAULT_SPACING):
    """
    Calculate all metrics for a batch of predictions
    
    Args:
        predictions: Predicted probability maps [B, 1, D, H, W] or list of arrays
        labels: Ground truth binary masks [B, 1, D, H, W] or list of arrays
        threshold: Probability threshold
        spacing: Voxel spacing (single tuple or list of tuples per case)
    
    Returns:
        metrics: Dictionary with all metrics including:
            - lesion_wise_recall, lesion_wise_precision, lesion_wise_f1
            - voxel_wise_dsc_micro (global DSC across all voxels)
            - voxel_wise_dsc_macro (mean of per-case DSC)
            - fp_per_case
            - tp, fp, fn (lesion counts)
            - Backward compatibility aliases: dsc, recall, precision
    """
    if not isinstance(predictions, (list, tuple)) and not (hasattr(predictions, "shape") and hasattr(predictions, "__getitem__")):
        raise TypeError("predictions must be a list/tuple or array-like object with shape and indexing support")
    if not isinstance(labels, (list, tuple)) and not (hasattr(labels, "shape") and hasattr(labels, "__getitem__")):
        raise TypeError("labels must be a list/tuple or array-like object with shape and indexing support")

    if isinstance(predictions, (list, tuple)):
        pred_list = list(predictions)
    else:
        pred_list = [predictions[i] for i in range(predictions.shape[0])]

    if isinstance(labels, (list, tuple)):
        label_list = list(labels)
    else:
        label_list = [labels[i] for i in range(labels.shape[0])]

    num_cases = len(pred_list)
    spacing_list = _normalize_spacing_per_case(spacing, num_cases)

    total_tp = 0
    total_fp = 0
    total_fn = 0
    intersection_sum = 0.0
    union_sum = 0.0
    per_case_dsc_list = []

    for pred, target, spacing_item in zip(pred_list, label_list, spacing_list):
        pred_array = np.asarray(pred)
        target_array = np.asarray(target)

        pred_binary = (pred_array >= threshold).astype(np.int32)
        target_binary = (target_array >= 0.5).astype(np.int32)

        # Accumulate for micro DSC
        intersection_sum += (pred_binary * target_binary).sum()
        union_sum += pred_binary.sum() + target_binary.sum()

        # Calculate per-case DSC for macro DSC
        case_dsc = calculate_dsc(pred_binary, target_binary, smooth=SMOOTH)
        per_case_dsc_list.append(case_dsc)

        lesion_metrics = calculate_lesion_metrics(
            pred_array,
            target_array,
            threshold=threshold,
            min_size_voxels=0,
            iou_threshold=0.1,
            distance_threshold_mm=10.0,
            spacing=spacing_item
        )

        total_tp += lesion_metrics["tp"]
        total_fp += lesion_metrics["fp"]
        total_fn += lesion_metrics["fn"]

    # Voxel-wise metrics
    voxel_dsc_micro = (2.0 * intersection_sum + SMOOTH) / (union_sum + SMOOTH)
    voxel_dsc_macro = np.mean(per_case_dsc_list) if per_case_dsc_list else 0.0
    
    # Lesion-wise metrics
    lesion_recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
    lesion_precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    lesion_f1 = (2 * lesion_precision * lesion_recall) / (lesion_precision + lesion_recall) if (lesion_precision + lesion_recall) > 0 else 0.0
    fp_per_case = total_fp / num_cases if num_cases > 0 else 0.0

    return MetricsDict({
        # New clearer keys
        "lesion_wise_recall": lesion_recall,
        "lesion_wise_precision": lesion_precision,
        "lesion_wise_f1": lesion_f1,
        "voxel_wise_dsc_micro": voxel_dsc_micro,
        "voxel_wise_dsc_macro": voxel_dsc_macro,
        "fp_per_case": fp_per_case,
        "tp": total_tp,
        "fp": total_fp,
        "fn": total_fn,
        # Backward compatibility aliases (will trigger deprecation warning on access)
        "dsc": voxel_dsc_micro,
        "recall": lesion_recall,
        "precision": lesion_precision
    })


def test_metrics():
    """Test metrics calculation"""
    # Create dummy data
    pred = np.random.rand(2, 1, 48, 48, 48)
    target = np.random.randint(0, 2, (2, 1, 48, 48, 48)).astype(np.float32)
    
    metrics = calculate_metrics(pred, target, threshold=0.5)
    
    print("Metrics:")
    print(f"  DSC: {metrics['dsc']:.4f}")
    print(f"  Recall: {metrics['recall']:.4f}")
    print(f"  Precision: {metrics['precision']:.4f}")
    print(f"  FP per case: {metrics['fp_per_case']:.4f}")
    print(f"  TP: {metrics['tp']}, FP: {metrics['fp']}, FN: {metrics['fn']}")


if __name__ == "__main__":
    test_metrics()
