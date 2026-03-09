"""
Integration test for body-mask-aware prompt export.
"""

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import nibabel as nib
import numpy as np


REPO_ROOT = Path(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
SCRIPT_PATH = REPO_ROOT / "scripts" / "export_bboxes.py"


def save_volume(path, array):
    nib.save(nib.Nifti1Image(np.asarray(array, dtype=np.float32), affine=np.eye(4)), str(path))


def test_export_bboxes_cli_filters_fp_slices_with_body_mask():
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        pred_dir = tmpdir / "pred"
        gt_dir = tmpdir / "labels"
        body_mask_dir = tmpdir / "body_masks"
        output_json = tmpdir / "prompts.json"
        pred_dir.mkdir()
        gt_dir.mkdir()
        body_mask_dir.mkdir()

        case_id = "case_0100"
        pred = np.zeros((4, 8, 8), dtype=np.float32)
        pred[1:3, 2:4, 2:4] = 1.0  # TP slices
        pred[2:4, 5:7, 5:7] = 1.0  # FP slices, only z=2 survives body mask
        gt = np.zeros((4, 8, 8), dtype=np.float32)
        gt[1:3, 2:4, 2:4] = 1.0
        body_mask = np.ones((4, 8, 8), dtype=np.float32)
        body_mask[3, :, :] = 0.0

        save_volume(pred_dir / f"{case_id}_pred.nii.gz", pred)
        save_volume(gt_dir / f"{case_id}.nii.gz", gt)
        save_volume(body_mask_dir / f"{case_id}.nii.gz", body_mask)

        result = subprocess.run(
            [
                sys.executable,
                str(SCRIPT_PATH),
                "--pred_dir",
                str(pred_dir),
                "--gt_dir",
                str(gt_dir),
                "--body_mask_dir",
                str(body_mask_dir),
                "--output_json",
                str(output_json),
                "--threshold",
                "0.5",
                "--expansion_voxels",
                "0",
                "--body_mask_ratio_threshold",
                "0.1",
            ],
            cwd=str(REPO_ROOT),
            capture_output=True,
            text=True,
            check=True,
        )

        exported = json.loads(output_json.read_text(encoding="utf-8"))

    assert exported == {
        case_id: {
            "TP": [
                {"z": 1, "box_2d": [2, 2, 3, 3]},
                {"z": 2, "box_2d": [2, 2, 3, 3]},
            ],
            "FP": [
                {"z": 2, "box_2d": [5, 5, 6, 6]},
            ],
        }
    }
    assert "Body-mask filtered FP slices in total: 1" in result.stderr


if __name__ == "__main__":
    test_export_bboxes_cli_filters_fp_slices_with_body_mask()
    print("Body mask integration tests passed! ✓")
