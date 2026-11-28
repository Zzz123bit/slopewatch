import numpy as np

from slopewatch.fusion import segmentation_status
from slopewatch.photogram import transform_points_to_comparison
from slopewatch.vggt_recon import ViewMapping


def test_view_mapping_resizes_and_pads():
    mapping = ViewMapping(
        original_size=(640, 480),
        resized_size=(518, 392),
        crop_y=0,
        canvas_size=(518, 518),
        offset=(0, 63),
    )
    mask = np.zeros((480, 640), dtype=np.uint8)
    mask[:, :320] = 3
    mapped = mapping.map_mask(mask)
    assert mapped.shape == (518, 518)
    assert (mapped[:63] == 255).all() and (mapped[-63:] == 255).all()
    body = mapped[63:63 + 392]
    assert (body[:, :250] == 3).mean() > 0.97
    assert (body[:, 280:] == 0).mean() > 0.97


def test_view_mapping_rejects_size_mismatch():
    mapping = ViewMapping((100, 80), (518, 518), 0, (518, 518), (0, 0))
    try:
        mapping.map_mask(np.zeros((90, 100), dtype=np.uint8))
    except ValueError as error:
        assert "尺寸" in str(error)
    else:
        raise AssertionError("expected ValueError")


def test_transform_points_applies_similarity_and_detrend():
    diagnostics = {
        "alignment": {"scale": 2.0, "rotation": np.eye(3).tolist(), "translation": [1.0, 0.0, 0.0]},
        "scale_mm_per_unit": 1000.0,
        "plane_rotation": np.eye(3).tolist(),
        "detrend": {"applied": True, "plane_mm": [10.0, 0.0, 0.0]},
    }
    mapped = transform_points_to_comparison(np.array([[1.0, 2.0, 3.0]]), diagnostics)
    assert np.allclose(mapped, [[3000.0, 4000.0, 5990.0]])


def test_segmentation_status_shape():
    status = segmentation_status()
    assert set(status) >= {"available", "detail", "model"}
    assert isinstance(status["available"], bool)

