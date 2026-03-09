"""
Focused tests for slice-wise prompt bbox recall metrics.
"""

import importlib.util
import os
import sys

import numpy as np

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

METRICS_PATH = os.path.join(REPO_ROOT, "light_unet", "models", "metrics.py")
spec = importlib.util.spec_from_file_location("metrics", METRICS_PATH)
metrics = importlib.util.module_from_spec(spec)
spec.loader.exec_module(metrics)

calculate_bbox_recall = metrics.calculate_bbox_recall
calculate_metrics = metrics.calculate_metrics


def test_calculate_bbox_recall_uses_tp_slice_boxes():
    """Only TP slice boxes that overlap a GT lesion slice should count as hits."""
    label = np.zeros((6, 10, 10), dtype=np.float32)
    label[1:3, 2:5, 2:5] = 1.0
    label[4:5, 6:8, 6:8] = 1.0

    prompts = {
        "TP": [
            {"z": 1, "box_2d": [2, 2, 4, 4]},
            {"z": 4, "box_2d": [0, 0, 1, 1]},
        ],
        "FP": [
            {"z": 4, "box_2d": [6, 6, 7, 7]},
        ],
    }

    hits, num_gt = calculate_bbox_recall(prompts, label)

    assert num_gt == 2
    assert hits == 1


def test_calculate_metrics_accepts_case_prompts():
    """Batch metrics should aggregate bbox_recall from per-case prompt payloads."""
    pred_1 = np.zeros((6, 8, 8), dtype=np.float32)
    label_1 = np.zeros((6, 8, 8), dtype=np.float32)
    pred_1[1:3, 1:3, 1:3] = 1.0
    label_1[1:3, 1:3, 1:3] = 1.0

    pred_2 = np.zeros((6, 8, 8), dtype=np.float32)
    label_2 = np.zeros((6, 8, 8), dtype=np.float32)
    pred_2[4:5, 5:7, 5:7] = 1.0
    label_2[4:5, 5:7, 5:7] = 1.0

    case_prompts = [
        {"TP": [{"z": 1, "box_2d": [1, 1, 2, 2]}], "FP": []},
        {"TP": [], "FP": [{"z": 4, "box_2d": [5, 5, 6, 6]}]},
    ]

    batch_metrics = calculate_metrics(
        [pred_1, pred_2],
        [label_1, label_2],
        threshold=0.5,
        expansion_voxels=0,
        case_prompts=case_prompts,
    )

    assert "bbox_recall" in batch_metrics
    assert np.isclose(batch_metrics["bbox_recall"], 0.5)


if __name__ == "__main__":
    test_calculate_bbox_recall_uses_tp_slice_boxes()
    test_calculate_metrics_accepts_case_prompts()
    print("All bbox recall metric tests passed! ✓")
