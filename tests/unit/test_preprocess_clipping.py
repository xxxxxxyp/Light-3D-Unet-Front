import numpy as np

from scripts.preprocess_data import clip_and_normalize


def test_clip_and_normalize_uses_absolute_bounds_and_metadata():
    image = np.array([-5.0, 0.0, 7.5, 15.0, 30.0], dtype=np.float32)

    normalized, metadata = clip_and_normalize(image, clip_min=0.0, clip_max=15.0, target_range=(0, 1))

    np.testing.assert_allclose(normalized, np.array([0.0, 0.0, 0.5, 1.0, 1.0], dtype=np.float32))
    assert metadata["clip_values"] == {"min": 0.0, "max": 15.0, "method": "absolute_value"}
    assert metadata["normalization_range"] == [0, 1]
