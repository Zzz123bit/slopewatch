import numpy as np

from slopewatch.photogram import align_similarity, compare_reconstructions


def _scene(n=15000, seed=0):
    rng = np.random.default_rng(seed)
    x = rng.uniform(0, 200, n)
    y = rng.uniform(0, 120, n)
    z = (
        60
        - 0.2 * x
        + 2.2 * np.sin(x / 17)
        + 1.6 * np.cos(y / 11)
        + 3.0 * np.exp(-(((x - 70) / 16) ** 2 + ((y - 40) / 12) ** 2))
        + 2.0 * np.exp(-(((x - 150) / 14) ** 2 + ((y - 80) / 10) ** 2))
        + 0.06 * rng.normal(size=n)
    )
    return np.column_stack([x, y, z])


def _with_change(points):
    changed = points.copy()
    changed[:, 2] -= 8.0 * np.exp(-(((changed[:, 0] - 120) / 18) ** 2 + ((changed[:, 1] - 60) / 10) ** 2))
    return changed


def _pose(points, scale=0.37, translation=(5.0, -3.0, 2.0)):
    rotation = np.array([[0.36, -0.48, 0.8], [0.8, 0.5, -0.2], [-0.48, 0.72, 0.5]])
    left, _, right_t = np.linalg.svd(rotation)
    rotation = left @ right_t
    return scale * (points @ rotation.T) + np.array(translation), rotation


def test_alignment_recovers_scale_and_change():
    before = _scene()
    after = _with_change(before)
    world, _ = _pose(after)
    comparison, diagnostics = compare_reconstructions(
        before, world, grid_step_mm=1.0, threshold_mm=2.0, up_mode="z_up"
    )
    assert abs(diagnostics["alignment"]["scale"] * 0.37 - 1.0) < 0.05
    assert comparison.metrics["max_erosion_depth_mm"] > 5.5
    assert comparison.metrics["erosion_volume_l"] > 0
    assert comparison.metrics["valid_cells"] > 1000


def test_gravity_up_frame_is_applied():
    before = _scene(n=8000)
    after = _with_change(before)
    # Put both clouds into a VGGT-like frame (Y down, Z forward) instead of Z-up.
    inverse = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]])
    comparison, diagnostics = compare_reconstructions(
        before @ inverse.T, after @ inverse.T, grid_step_mm=1.0, threshold_mm=2.0
    )
    assert diagnostics["coordinate_system"].startswith("Z-up")
    assert comparison.metrics["max_erosion_depth_mm"] > 5.5


def test_invalid_scale_rejected():
    points = _scene(n=2000)
    try:
        compare_reconstructions(points, points, scale_mm_per_unit=0.0)
    except ValueError as error:
        assert "尺度" in str(error)
    else:
        raise AssertionError("expected ValueError")
