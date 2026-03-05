"""
Focused tests for PatchDataset label dilation augmentation.
"""

import sys
import numpy as np
import types


def _load_patch_dataset_class():
    if "torch" not in sys.modules:
        fake_torch = types.ModuleType("torch")
        fake_torch_nn = types.ModuleType("torch.nn")
        fake_torch_utils = types.ModuleType("torch.utils")
        fake_torch_utils_data = types.ModuleType("torch.utils.data")
        fake_torch_utils_data.Dataset = object
        fake_torch_utils_data.DataLoader = object
        fake_torch_nn.Module = object
        fake_torch.device = object
        fake_torch.nn = fake_torch_nn
        fake_torch.utils = fake_torch_utils
        fake_torch_utils.data = fake_torch_utils_data
        sys.modules["torch"] = fake_torch
        sys.modules["torch.nn"] = fake_torch_nn
        sys.modules["torch.utils"] = fake_torch_utils
        sys.modules["torch.utils.data"] = fake_torch_utils_data

    from light_unet.datasets.patch_dataset import PatchDataset
    return PatchDataset


PatchDataset = _load_patch_dataset_class()


def _make_dataset_stub(augmentation):
    dataset = PatchDataset.__new__(PatchDataset)
    dataset.augmentation = augmentation
    dataset.patch_size = (5, 5, 5)
    return dataset


def test_label_dilation_enabled_expands_mask():
    image = np.zeros((5, 5, 5), dtype=np.float32)
    label = np.zeros((5, 5, 5), dtype=np.float32)
    label[2, 2, 2] = 1.0

    dataset = _make_dataset_stub({
        "label_dilation": {"enabled": True, "voxels": 1}
    })
    _, augmented_label = dataset._augment(image, label)

    assert augmented_label.dtype == np.float32
    assert int(augmented_label.sum()) == 7  # center + 6-neighborhood
    assert augmented_label[2, 2, 2] == 1.0
    assert augmented_label[1, 2, 2] == 1.0
    assert augmented_label[3, 2, 2] == 1.0
    assert augmented_label[2, 1, 2] == 1.0
    assert augmented_label[2, 3, 2] == 1.0
    assert augmented_label[2, 2, 1] == 1.0
    assert augmented_label[2, 2, 3] == 1.0


def test_label_dilation_disabled_keeps_mask():
    image = np.zeros((5, 5, 5), dtype=np.float32)
    label = np.zeros((5, 5, 5), dtype=np.float32)
    label[2, 2, 2] = 1.0

    dataset = _make_dataset_stub({
        "label_dilation": {"enabled": False, "voxels": 1}
    })
    _, augmented_label = dataset._augment(image, label)

    assert np.array_equal(augmented_label, label)


if __name__ == "__main__":
    try:
        test_label_dilation_enabled_expands_mask()
        test_label_dilation_disabled_keeps_mask()
        print("All label dilation tests passed! ✓")
    except AssertionError as e:
        print(f"✗ Test failed: {e}")
        sys.exit(1)
