"""
Tests for exporting bounding boxes from saved probability maps.
"""

import importlib.util
import json
import os
import tempfile

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


def save_prob_map(path, array):
    nib.save(nib.Nifti1Image(array.astype(np.float32), affine=np.eye(4)), path)


def test_process_single_case_extracts_expanded_bboxes():
    export_bboxes = load_export_bboxes_module()

    with tempfile.TemporaryDirectory() as tmpdir:
        prob_map_path = os.path.join(tmpdir, "0001_prob.nii.gz")
        prob_map = np.zeros((1, 12, 12, 12), dtype=np.float32)
        prob_map[0, 1:4, 1:4, 1:4] = 1.0
        prob_map[0, 7:10, 6:9, 5:8] = 1.0
        save_prob_map(prob_map_path, prob_map)

        bboxes = export_bboxes.process_single_case(prob_map_path, threshold=0.5, expansion_voxels=2)

    assert bboxes == [
        [0, 6, 0, 6, 0, 6],
        [5, 12, 4, 11, 3, 10],
    ]


def test_export_bboxes_writes_compact_json():
    export_bboxes = load_export_bboxes_module()

    with tempfile.TemporaryDirectory() as tmpdir:
        prob_maps_dir = os.path.join(tmpdir, "prob_maps")
        os.makedirs(prob_maps_dir, exist_ok=True)
        output_json = os.path.join(tmpdir, "exports", "bboxes.json")

        case_2 = np.zeros((12, 12, 12), dtype=np.float32)
        case_2[6:9, 6:9, 6:9] = 1.0
        save_prob_map(os.path.join(prob_maps_dir, "0002_prob.nii.gz"), case_2)

        case_1 = np.zeros((12, 12, 12), dtype=np.float32)
        case_1[2:5, 3:6, 4:7] = 1.0
        save_prob_map(os.path.join(prob_maps_dir, "0001_prob.nii.gz"), case_1)

        exported = export_bboxes.export_bboxes(
            prob_maps_dir=prob_maps_dir,
            output_json=output_json,
            threshold=0.5,
            expansion_voxels=1,
        )

        with open(output_json, "r", encoding="utf-8") as handle:
            payload = handle.read()

    assert exported == {
        "0001": [[1, 6, 2, 7, 3, 8]],
        "0002": [[5, 10, 5, 10, 5, 10]],
    }
    assert json.loads(payload) == exported
    assert payload == '{"0001":[[1,6,2,7,3,8]],"0002":[[5,10,5,10,5,10]]}'


if __name__ == "__main__":
    test_process_single_case_extracts_expanded_bboxes()
    test_export_bboxes_writes_compact_json()
    print("Export bbox tests passed! ✓")
