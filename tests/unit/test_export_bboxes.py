"""
Tests for exporting slice-wise TP/FP prompt boxes.
"""

import importlib.util
import json
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


def test_process_single_case_generates_tp_fp_slice_boxes():
    export_bboxes = load_export_bboxes_module()

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        pred_path = tmpdir / "0001_pred.nii.gz"
        gt_path = tmpdir / "0001.nii.gz"
        body_mask_path = tmpdir / "0001_body.nii.gz"

        pred = np.zeros((5, 10, 10), dtype=np.float32)
        pred[1:3, 2:5, 3:6] = 1.0  # TP component over z=1,2
        pred[3:5, 6:8, 1:3] = 1.0  # FP component over z=3,4

        gt = np.zeros((5, 10, 10), dtype=np.float32)
        gt[1:3, 2:5, 3:6] = 1.0

        body_mask = np.zeros((5, 10, 10), dtype=np.float32)
        body_mask[:, 1:9, 1:9] = 1.0
        body_mask[4, :, :] = 0.0  # Filter one FP slice completely outside body mask

        save_volume(pred_path, pred)
        save_volume(gt_path, gt)
        save_volume(body_mask_path, body_mask)

        prompts = export_bboxes.process_single_case(
            pred_path=pred_path,
            gt_path=gt_path,
            body_mask_path=body_mask_path,
            threshold=0.5,
            expansion_voxels=1,
            body_mask_ratio_threshold=0.1,
        )

    assert prompts == {
        "TP": [
            {"z": 1, "box_2d": [1, 2, 5, 6]},
            {"z": 2, "box_2d": [1, 2, 5, 6]},
        ],
        "FP": [
            {"z": 3, "box_2d": [5, 0, 8, 3]},
        ],
    }


def test_export_bboxes_writes_new_json_structure():
    export_bboxes = load_export_bboxes_module()

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        pred_dir = tmpdir / "pred"
        gt_dir = tmpdir / "labels"
        body_mask_dir = tmpdir / "body_masks"
        output_json = tmpdir / "exports" / "prompts.json"
        pred_dir.mkdir()
        gt_dir.mkdir()
        body_mask_dir.mkdir()

        pred = np.zeros((4, 8, 8), dtype=np.float32)
        pred[0:2, 2:4, 2:4] = 1.0
        gt = np.zeros((4, 8, 8), dtype=np.float32)
        gt[0:2, 2:4, 2:4] = 1.0
        body_mask = np.ones((4, 8, 8), dtype=np.float32)

        save_volume(pred_dir / "case_0000_pred.nii.gz", pred)
        save_volume(gt_dir / "case_0000.nii.gz", gt)
        save_volume(body_mask_dir / "case_0000.nii.gz", body_mask)

        exported = export_bboxes.export_bboxes(
            pred_dir=pred_dir,
            gt_dir=gt_dir,
            body_mask_dir=body_mask_dir,
            output_json=output_json,
            threshold=0.5,
            expansion_voxels=0,
            body_mask_ratio_threshold=0.1,
        )

        payload = output_json.read_text(encoding="utf-8")

    assert exported == {
        "case_0000": {
            "TP": [
                {"z": 0, "box_2d": [2, 2, 3, 3]},
                {"z": 1, "box_2d": [2, 2, 3, 3]},
            ],
            "FP": [],
        }
    }
    assert json.loads(payload) == exported
    assert payload == '{"case_0000":{"TP":[{"z":0,"box_2d":[2,2,3,3]},{"z":1,"box_2d":[2,2,3,3]}],"FP":[]}}'


if __name__ == "__main__":
    test_process_single_case_generates_tp_fp_slice_boxes()
    test_export_bboxes_writes_new_json_structure()
    print("Export bbox tests passed! ✓")
