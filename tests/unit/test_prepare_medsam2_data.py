"""
Tests for preparing MedSAM2 NPZ data from processed NIfTI files.
"""

import importlib.util
import os
import tempfile
from pathlib import Path

import nibabel as nib
import numpy as np


REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
SCRIPT_PATH = os.path.join(REPO_ROOT, "scripts", "prepare_medsam2_data.py")


def load_prepare_module():
    spec = importlib.util.spec_from_file_location("prepare_medsam2_data", SCRIPT_PATH)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load module from {SCRIPT_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def save_nifti(path, array):
    nib.save(nib.Nifti1Image(np.asarray(array), affine=np.eye(4)), str(path))


def test_convert_to_npz_transposes_and_quantizes_data():
    prepare = load_prepare_module()

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        img_path = tmpdir / "0001_0000.nii.gz"
        label_path = tmpdir / "0001.nii.gz"
        out_path = tmpdir / "0001.npz"

        image = np.array(
            [
                [[-5.0, 0.0, 7.5], [15.0, 30.0, 3.75]],
                [[1.5, 5.0, 10.0], [12.0, 14.0, 16.0]],
            ],
            dtype=np.float32,
        )
        label = np.array(
            [
                [[0.0, 0.0, 2.0], [0.0, -1.0, 0.0]],
                [[3.0, 0.0, 0.0], [0.0, 0.0, 4.0]],
            ],
            dtype=np.float32,
        )

        save_nifti(img_path, image)
        save_nifti(label_path, label)

        prepare.convert_to_npz(img_path, label_path, out_path, clip_min=0.0, clip_max=15.0)

        payload = np.load(out_path)
        expected_imgs = np.transpose(image, (2, 0, 1))
        expected_imgs = np.clip(expected_imgs, 0.0, 15.0)
        expected_imgs = ((expected_imgs / 15.0) * 255.0).astype(np.uint8)
        expected_gts = (np.transpose(label, (2, 0, 1)) > 0).astype(np.uint8)

        assert payload["imgs"].dtype == np.uint8
        assert payload["gts"].dtype == np.uint8
        assert payload["imgs"].shape == (3, 2, 2)
        assert payload["gts"].shape == (3, 2, 2)
        np.testing.assert_array_equal(payload["imgs"], expected_imgs)
        np.testing.assert_array_equal(payload["gts"], expected_gts)


def test_prepare_dataset_writes_train_and_val_npz_files():
    prepare = load_prepare_module()

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        data_dir = tmpdir / "processed"
        images_dir = data_dir / "images"
        labels_dir = data_dir / "labels"
        splits_dir = tmpdir / "splits"
        output_dir = tmpdir / "medsam2_npz"

        images_dir.mkdir(parents=True)
        labels_dir.mkdir(parents=True)
        splits_dir.mkdir(parents=True)

        train_case_id = "0001"
        val_case_id = "0002"

        save_nifti(images_dir / f"{train_case_id}_0000.nii.gz", np.arange(24, dtype=np.float32).reshape(2, 3, 4))
        save_nifti(labels_dir / f"{train_case_id}.nii.gz", np.array(np.arange(24).reshape(2, 3, 4) % 2, dtype=np.float32))
        save_nifti(images_dir / f"{val_case_id}_0000.nii.gz", np.full((2, 3, 4), 15.0, dtype=np.float32))
        save_nifti(labels_dir / f"{val_case_id}.nii.gz", np.zeros((2, 3, 4), dtype=np.float32))

        (splits_dir / "train_list.txt").write_text(f"{train_case_id}\n", encoding="utf-8")
        (splits_dir / "val_list.txt").write_text(f"{val_case_id}\n", encoding="utf-8")

        prepare.prepare_dataset(
            data_dir=data_dir,
            splits_dir=splits_dir,
            output_dir=output_dir,
            clip_min=0.0,
            clip_max=15.0,
        )

        train_npz = output_dir / "train" / f"{train_case_id}.npz"
        val_npz = output_dir / "val" / f"{val_case_id}.npz"

        assert train_npz.exists()
        assert val_npz.exists()

        train_payload = np.load(train_npz)
        val_payload = np.load(val_npz)

        assert set(train_payload.files) == {"imgs", "gts"}
        assert set(val_payload.files) == {"imgs", "gts"}
        assert train_payload["imgs"].shape == (4, 2, 3)
        assert train_payload["gts"].shape == (4, 2, 3)
        assert val_payload["imgs"].dtype == np.uint8
        assert val_payload["gts"].dtype == np.uint8
        assert np.all(val_payload["imgs"] == 255)
        assert np.all(val_payload["gts"] == 0)


def test_convert_to_npz_serializes_tp_and_fp_prompt_boxes():
    prepare = load_prepare_module()

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        img_path = tmpdir / "0003_0000.nii.gz"
        label_path = tmpdir / "0003.nii.gz"
        out_path = tmpdir / "0003.npz"

        save_nifti(img_path, np.ones((2, 2, 2), dtype=np.float32))
        save_nifti(label_path, np.zeros((2, 2, 2), dtype=np.float32))

        prepare.convert_to_npz(
            img_path,
            label_path,
            out_path,
            clip_min=0.0,
            clip_max=15.0,
            case_prompts={
                "TP": [{"z": 1, "box_2d": [2, 3, 4, 5]}],
                "FP": [{"z": 0, "box_2d": [5, 6, 7, 8]}],
            },
        )

        payload = np.load(out_path)

    np.testing.assert_array_equal(payload["tp_boxes"], np.array([[1, 2, 3, 4, 5]], dtype=np.int16))
    np.testing.assert_array_equal(payload["fp_boxes"], np.array([[0, 5, 6, 7, 8]], dtype=np.int16))


if __name__ == "__main__":
    test_convert_to_npz_transposes_and_quantizes_data()
    test_prepare_dataset_writes_train_and_val_npz_files()
    test_convert_to_npz_serializes_tp_and_fp_prompt_boxes()
    print("Prepare MedSAM2 data tests passed! ✓")
