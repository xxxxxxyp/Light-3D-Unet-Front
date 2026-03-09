"""
Focused tests for body-mask filtering during prompt export.
"""

import importlib.util
import os
import tempfile
from pathlib import Path

import nibabel as nib
import numpy as np


REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
EXPORT_BBOXES_PATH = os.path.join(REPO_ROOT, "scripts", "export_bboxes.py")


def load_export_bboxes_module():
    spec = importlib.util.spec_from_file_location("export_bboxes", EXPORT_BBOXES_PATH)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load export_bboxes module from {EXPORT_BBOXES_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def save_volume(path, array):
    nib.save(nib.Nifti1Image(np.asarray(array, dtype=np.float32), affine=np.eye(4)), str(path))


def test_body_mask_filter_removes_outside_slice_boxes():
    export_bboxes = load_export_bboxes_module()

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        pred_path = tmpdir / "0001_pred.nii.gz"
        gt_path = tmpdir / "0001.nii.gz"
        body_mask_path = tmpdir / "0001_body.nii.gz"

        pred = np.zeros((3, 8, 8), dtype=np.float32)
        pred[0, 1:4, 1:4] = 1.0
        pred[1, 1:4, 1:4] = 1.0
        gt = np.zeros((3, 8, 8), dtype=np.float32)
        body_mask = np.zeros((3, 8, 8), dtype=np.float32)
        body_mask[0, 1:4, 1:4] = 1.0

        save_volume(pred_path, pred)
        save_volume(gt_path, gt)
        save_volume(body_mask_path, body_mask)

        prompts = export_bboxes.process_single_case(
            pred_path=pred_path,
            gt_path=gt_path,
            body_mask_path=body_mask_path,
            threshold=0.5,
            expansion_voxels=0,
            body_mask_ratio_threshold=0.1,
        )

    assert prompts["TP"] == []
    assert prompts["FP"] == [{"z": 0, "box_2d": [1, 1, 3, 3]}]


def test_body_mask_filter_keeps_slice_when_ratio_reaches_threshold():
    export_bboxes = load_export_bboxes_module()

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        pred_path = tmpdir / "0002_pred.nii.gz"
        gt_path = tmpdir / "0002.nii.gz"
        body_mask_path = tmpdir / "0002_body.nii.gz"

        pred = np.zeros((2, 10, 10), dtype=np.float32)
        pred[0, 2:5, 2:5] = 1.0
        gt = np.zeros((2, 10, 10), dtype=np.float32)
        body_mask = np.zeros((2, 10, 10), dtype=np.float32)
        body_mask[0, 2, 2] = 1.0  # 1/9 > 0.1 threshold

        save_volume(pred_path, pred)
        save_volume(gt_path, gt)
        save_volume(body_mask_path, body_mask)

        prompts = export_bboxes.process_single_case(
            pred_path=pred_path,
            gt_path=gt_path,
            body_mask_path=body_mask_path,
            threshold=0.5,
            expansion_voxels=0,
            body_mask_ratio_threshold=0.1,
        )

    assert prompts["FP"] == [{"z": 0, "box_2d": [2, 2, 4, 4]}]


if __name__ == "__main__":
    test_body_mask_filter_removes_outside_slice_boxes()
    test_body_mask_filter_keeps_slice_when_ratio_reaches_threshold()
    print("Body mask prompt-filter tests passed! ✓")
