"""
Focused tests for BBox recall metric integration.
"""

import numpy as np
import os
import sys
import importlib.util

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

METRICS_PATH = os.path.join(REPO_ROOT, "light_unet", "models", "metrics.py")
spec = importlib.util.spec_from_file_location("metrics", METRICS_PATH)
metrics = importlib.util.module_from_spec(spec)
spec.loader.exec_module(metrics)

calculate_bbox_recall = metrics.calculate_bbox_recall
calculate_metrics = metrics.calculate_metrics


def test_calculate_bbox_recall_respects_expansion():
    """BBox expansion should recover nearby GT lesions."""
    prob_map = np.zeros((12, 12, 12), dtype=np.float32)
    label = np.zeros((12, 12, 12), dtype=np.float32)

    prob_map[1:2, 1:2, 1:2] = 1.0
    label[3:4, 3:4, 3:4] = 1.0

    hits_no_expand, num_gt_no_expand = calculate_bbox_recall(
        prob_map, label, threshold=0.5, expansion_voxels=0
    )
    hits_expand, num_gt_expand = calculate_bbox_recall(
        prob_map, label, threshold=0.5, expansion_voxels=2
    )

    assert num_gt_no_expand == 1 and num_gt_expand == 1
    assert hits_no_expand == 0
    assert hits_expand == 1


def test_calculate_metrics_contains_bbox_recall():
    """Batch metrics should include aggregated bbox_recall."""
    pred_1 = np.zeros((10, 10, 10), dtype=np.float32)
    label_1 = np.zeros((10, 10, 10), dtype=np.float32)
    pred_1[2:4, 2:4, 2:4] = 1.0
    label_1[2:4, 2:4, 2:4] = 1.0

    pred_2 = np.zeros((10, 10, 10), dtype=np.float32)
    label_2 = np.zeros((10, 10, 10), dtype=np.float32)
    pred_2[1:2, 1:2, 1:2] = 1.0
    label_2[7:8, 7:8, 7:8] = 1.0

    metrics = calculate_metrics(
        [pred_1, pred_2],
        [label_1, label_2],
        threshold=0.5,
        expansion_voxels=0
    )

    assert "bbox_recall" in metrics
    assert np.isclose(metrics["bbox_recall"], 0.5)


if __name__ == "__main__":
    test_calculate_bbox_recall_respects_expansion()
    test_calculate_metrics_contains_bbox_recall()
    print("All bbox recall metric tests passed! ✓")
