import numpy as np

from slopewatch.core import Comparison
from slopewatch.weak_labels import COLLAPSE, GULLY, OTHER, UNKNOWN, label_change


def _comparison():
    step = 1.0
    x_coords = np.arange(0.0, 200.0, step)
    y_coords = np.arange(0.0, 120.0, step)
    x, y = np.meshgrid(x_coords, y_coords)
    before = 50.0 - 0.1 * x
    after = before.copy()
    # long thin channel along y at x ~ 30
    channel = -8.0 * np.exp(-(((x - 30.0) / 4.0) ** 2)) * ((y > 5) & (y < 115))
    # compact pit at (120, 60) with an adjacent deposition mound
    pit = -20.0 * np.exp(-(((x - 120.0) ** 2 + (y - 60.0) ** 2) / 12.0**2))
    mound = 6.0 * np.exp(-(((x - 145.0) ** 2 + (y - 60.0) ** 2) / 12.0**2))
    after = after + channel + pit + mound
    valid = np.ones_like(before, dtype=bool)
    valid[:, :2] = False
    delta = after - before
    metrics = {"erosion_threshold_mm": 3.0, "grid_step_mm": step}
    return Comparison(x, y, before, after, delta, valid, metrics, [], step)


def test_elongated_channel_is_gully_and_pit_is_collapse():
    comparison = _comparison()
    result = label_change(comparison)
    labels = result.labels
    assert labels[60, 30] == GULLY
    assert labels[60, 120] == COLLAPSE
    assert labels[5, 100] == OTHER
    assert (labels[:, :2] == UNKNOWN).all()
    assert result.stats["gully_cells"] > 100
    assert result.stats["collapse_cells"] > 50
    assert len(result.regions) >= 2


def test_rules_are_configurable():
    comparison = _comparison()
    strict = label_change(comparison, {"gully_min_length_mm": 500.0})
    assert strict.labels[60, 30] != GULLY


def test_flat_surface_has_no_targets():
    step = 1.0
    x_coords = np.arange(0.0, 50.0, step)
    y_coords = np.arange(0.0, 50.0, step)
    x, y = np.meshgrid(x_coords, y_coords)
    surface = 10.0 + 0.01 * x
    delta = np.zeros_like(surface)
    valid = np.ones_like(surface, dtype=bool)
    comparison = Comparison(x, y, surface, surface, delta, valid, {"erosion_threshold_mm": 3.0}, [], step)
    result = label_change(comparison)
    assert result.stats["gully_cells"] == 0
    assert result.stats["collapse_cells"] == 0
    assert (result.labels[valid] == OTHER).all()
