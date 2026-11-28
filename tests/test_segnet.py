from pathlib import Path

import numpy as np
import pytest

from slopewatch.segnet import TorchSegmenter, prepare_image, prepare_mask

CHECKPOINT = Path(__file__).resolve().parents[1] / "models" / "segmentation.pt"


def test_prepare_image_pads_to_square():
    image = np.zeros((480, 640, 3), dtype=np.uint8)
    tensor, (left, top, width, height), (raw_h, raw_w) = prepare_image(image)
    assert tensor.shape == (1, 3, 256, 256)
    assert (raw_h, raw_w) == (480, 640)
    assert width == 256 and height == 192
    assert top == 32 and left == 0


def test_prepare_mask_uses_ignore_value():
    mask = np.ones((480, 640), dtype=np.uint8)
    tensor = prepare_mask(mask)
    assert tensor.shape == (256, 256)
    assert int(tensor[0, 0]) == 255
    assert int(tensor[128, 128]) == 1


@pytest.mark.skipif(not CHECKPOINT.is_file(), reason="segmentation checkpoint missing")
def test_torch_segmenter_predicts_classes():
    segmenter = TorchSegmenter(CHECKPOINT)
    image = (np.random.default_rng(0).random((120, 160, 3)) * 255).astype(np.uint8)
    prediction = segmenter.predict(image)
    assert prediction.shape == (120, 160)
    assert prediction.dtype == np.uint8
    assert set(np.unique(prediction)).issubset(set(range(5)))
