import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from slopewatch.registration import Transform, fit_control_frame, refine_stable_icp, verify_control_frame


def test_surveyed_planar_frame_recovers_metric_scale_and_rejects_one_wrong_id():
    world = np.array([[0, 0, 0], [80, 0, 0], [80, 50, 0], [0, 50, 0], [160, 50, 0]])
    world_to_local = Transform(0.023, Rotation.from_euler("xz", [83, 25], degrees=True).as_matrix(),
                               np.array([8.6, -3.2, 9.4]))
    local = world_to_local.apply(world)
    local[-1] = local[1] + [1.0, 1.0, 2.0]  # Mistaken marker observation.
    frame, report = fit_control_frame(local, world, max_error_mm=0.2)
    assert report["inlier_indices"] == [0, 1, 2, 3]
    expected_error = np.linalg.norm(local[-1] - world_to_local.apply(world[-1])) / world_to_local.scale
    assert report["residuals_mm"][-1] == pytest.approx(expected_error)
    assert report["scale_mm_per_unit"] == pytest.approx(1 / world_to_local.scale)
    assert np.linalg.det(frame.rotation) == pytest.approx(1)
    raised_target = np.array([[40, 30, 17], [30, 10, -4]])
    np.testing.assert_allclose(frame.apply(world_to_local.apply(raised_target)), raised_target, atol=1e-10)


def test_refinement_moves_only_rigid_frame_and_preserves_erosion_outside_stable_area():
    rng = np.random.default_rng(24)
    fixed = rng.uniform(-120, 120, size=(90, 3))
    shift = Transform(1.0, Rotation.from_euler("y", 1, degrees=True).as_matrix(), [1, -0.8, 0.7])
    moving = shift.apply(fixed)
    # Two views have partial overlap, shuffled order and different point counts.
    correction, report = refine_stable_icp(fixed[:-4], moving[7:][::-1], max_distance_mm=10,
                                            max_rmse_mm=0.5, min_overlap=0.85)
    assert correction.scale == 1
    np.testing.assert_allclose(correction.apply(moving), fixed, atol=1e-9)
    assert report["overlap"] >= 0.85
    before = np.array([[12, 34, -5], [-50, -60, 1]])
    erosion = np.array([[0, 0, -15], [0, 0, -8]])
    after = shift.apply(before + erosion)
    np.testing.assert_allclose(correction.apply(after) - before, erosion, atol=1e-9)
    composed = shift.then(correction)
    assert verify_control_frame(composed, fixed[:4], fixed[:4], [0, 1, 2, 3])["rmse_mm"] < 1e-9
    with pytest.raises(ValueError, match="控制点偏差"):
        verify_control_frame(shift, fixed[:4], fixed[:4], [0, 1, 2, 3], max_error_mm=0.01)


def test_degenerate_or_inconsistent_controls_rejected():
    line = np.array([[0, 0, 0], [1, 0, 0], [2, 0.001, 0], [3, 0, 0]])
    with pytest.raises(ValueError, match="不共线"):
        fit_control_frame(line, line)
    rng = np.random.default_rng(42)
    with pytest.raises(ValueError, match="公制坐标"):
        fit_control_frame(rng.normal(size=(7, 3)), rng.normal(size=(7, 3)), max_error_mm=1e-4)


def test_icp_rejects_nonoverlap_and_large_stable_residuals():
    rng = np.random.default_rng(24)
    fixed = rng.uniform(-100, 100, size=(60, 3))
    # All after points have a close match, but only 25% of before points do.
    with pytest.raises(ValueError, match="双向重叠率"):
        refine_stable_icp(fixed, fixed[:15], max_distance_mm=1)
    moving = fixed + rng.uniform(-1, 1, size=fixed.shape)
    with pytest.raises(ValueError, match="RMSE"):
        refine_stable_icp(fixed, moving, max_distance_mm=5, max_rmse_mm=0.1)
    with pytest.raises(ValueError, match="近邻不足"):
        refine_stable_icp(fixed, fixed + 2000)
