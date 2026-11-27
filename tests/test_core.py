import struct
from pathlib import Path

import numpy as np
import pytest

from slopewatch.core import compare_point_clouds, load_point_cloud, make_demo_comparison


def test_ascii_ply_loads_xyz_and_scales_meters():
    content = b"ply\nformat ascii 1.0\nelement vertex 3\nproperty float x\nproperty float y\nproperty float z\nend_header\n0 0 1\n1 0 2\n0 1 3\n"
    points = load_point_cloud(content, "surface.ply", unit="m")
    np.testing.assert_allclose(points, [[0, 0, 1000], [1000, 0, 2000], [0, 1000, 3000]])


def test_csv_loads_from_path(tmp_path: Path):
    csv_path = tmp_path / "points.csv"
    csv_path.write_text("x,y,z\n0,0,1\n1,0,2\n0,1,3\n", encoding="utf-8")
    points = load_point_cloud(csv_path, csv_path.name)
    np.testing.assert_allclose(points, [[0, 0, 1], [1, 0, 2], [0, 1, 3]])


def test_binary_ply_loads_xyz():
    header = b"ply\nformat binary_little_endian 1.0\nelement vertex 3\nproperty float x\nproperty float y\nproperty float z\nend_header\n"
    body = b"".join(struct.pack("<fff", *row) for row in [(0, 0, 1), (1, 0, 2), (0, 1, 3)])
    points = load_point_cloud(header + body, "surface.ply")
    np.testing.assert_allclose(points, [[0, 0, 1], [1, 0, 2], [0, 1, 3]])


def test_comparison_reports_erosion_and_deposition():
    x, y = np.meshgrid(np.arange(5.0), np.arange(5.0))
    before = np.column_stack((x.ravel(), y.ravel(), np.full(x.size, 10.0)))
    change = np.zeros_like(x)
    change[1, 1] = -2.0
    change[3, 3] = 4.0
    after = np.column_stack((x.ravel(), y.ravel(), (10.0 + change).ravel()))
    result = compare_point_clouds(before, after, grid_step_mm=1.0, erosion_threshold_mm=1.0, elapsed_hours=2.0)
    assert result.metrics["max_erosion_depth_mm"] == 2.0
    assert result.metrics["max_deposition_height_mm"] == 4.0
    assert result.metrics["erosion_volume_l"] == pytest.approx(0.000002)
    assert result.metrics["deposition_volume_l"] == pytest.approx(0.000004)
    assert result.metrics["max_change_rate_mm_per_hour"] == 2.0


def test_volume_threshold_filters_small_changes():
    x, y = np.meshgrid(np.arange(5.0), np.arange(5.0))
    before = np.column_stack((x.ravel(), y.ravel(), np.full(x.size, 10.0)))
    change = np.full_like(x, -0.5)
    change[2, 2] = -2.0
    after = np.column_stack((x.ravel(), y.ravel(), (10.0 + change).ravel()))
    result = compare_point_clouds(before, after, grid_step_mm=1.0, erosion_threshold_mm=1.0)
    assert result.metrics["erosion_volume_l"] == pytest.approx(0.000002)
    assert result.metrics["erosion_volume_raw_l"] > result.metrics["erosion_volume_l"]


def test_demo_result_is_repeatable_and_has_change():
    first = make_demo_comparison()
    second = make_demo_comparison()
    assert first.metrics == second.metrics
    assert first.metrics["erosion_volume_l"] > 0
    assert first.metrics["deposition_volume_l"] > 0
    assert first.metrics["gully_count_est"] >= 1

