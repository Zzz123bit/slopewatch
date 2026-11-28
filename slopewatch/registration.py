"""Calibrate each camera reconstruction into a common, Z-up millimetre frame."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from math import ceil, comb

import numpy as np
from scipy.spatial import cKDTree


@dataclass(frozen=True)
class Transform:
    """Column-vector convention: output = scale * rotation @ input + translation."""

    scale: float
    rotation: np.ndarray
    translation: np.ndarray

    def apply(self, points):
        return self.scale * (np.asarray(points) @ self.rotation.T) + self.translation

    def then(self, following: Transform) -> Transform:
        """Compose two transforms in application order."""
        return Transform(
            self.scale * following.scale,
            following.rotation @ self.rotation,
            following.apply(self.translation),
        )

    def as_dict(self) -> dict:
        return {
            "scale": float(self.scale),
            "rotation": self.rotation.tolist(),
            "translation": self.translation.tolist(),
        }


def _xyz(points, label, minimum):
    values = np.asarray(points, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != 3 or len(values) < minimum:
        raise ValueError(f"{label}至少需要 {minimum} 个三维点")
    if not np.isfinite(values).all():
        raise ValueError(f"{label}含非有限三维坐标")
    return values


def _positive(value, label):
    if not np.isfinite(value) or value <= 0:
        raise ValueError(f"{label}必须为有限正数")


def _check_geometry(centered):
    # A planar target is sufficient. A line leaves rotation about its axis unconstrained.
    axes = np.linalg.svd(centered, compute_uv=False)
    if axes[0] <= 0 or axes[1] < 0.01 * axes[0]:
        raise ValueError("点分布退化，请选取分散且不共线的点")


def _fit(source, target, estimate_scale):
    source_center, target_center = source.mean(axis=0), target.mean(axis=0)
    source_d, target_d = source - source_center, target - target_center
    _check_geometry(source_d)
    _check_geometry(target_d)
    left, singular_values, right_t = np.linalg.svd(target_d.T @ source_d)
    signs = np.ones(3)
    signs[-1] = np.sign(np.linalg.det(left @ right_t))
    rotation = (left * signs) @ right_t
    scale = 1.0
    if estimate_scale:
        scale = float(singular_values @ signs / np.sum(source_d**2))
        _positive(scale, "标定尺度")
    return Transform(scale, rotation, target_center - scale * (rotation @ source_center))


def _control_triplets(count):
    if comb(count, 3) <= 512:
        yield from combinations(range(count), 3)
        return
    rng = np.random.default_rng(0)
    seen = set()
    while len(seen) < 512:
        triple = tuple(sorted(rng.choice(count, 3, replace=False)))
        if triple not in seen:
            seen.add(triple)
            yield triple


def _rmse(residuals):
    return float(np.sqrt(np.mean(np.square(residuals))))


def fit_control_frame(source_xyz, world_xyz_mm, *, max_error_mm=3.0):
    """Robust Sim(3) from same-ID marker pairs. World XYZ must use millimetres with Z up.

    A triangle fixes the frame. Redundant landmarks permit outlier rejection.
    Require at least 70% consensus and refit on all inliers. Residuals are 3D.
    """
    source = _xyz(source_xyz, "重建控制点", 3)
    world = _xyz(world_xyz_mm, "实测控制点", 3)
    if len(world) != len(source):
        raise ValueError("两组控制点须按相同 ID 对齐")
    _positive(max_error_mm, "控制点误差门限")
    for points in (source, world):
        if len(np.unique(points, axis=0)) != len(points):
            raise ValueError("控制点有重复坐标，请核对 ID")

    best_score = (0, -np.inf)
    best_inliers = None
    required = max(3, ceil(0.7 * len(source)))
    for triple in _control_triplets(len(source)):
        indices = list(triple)
        try:
            candidate = _fit(source[indices], world[indices], estimate_scale=True)
        except ValueError:  # A valid frame may contain some collinear triples.
            continue
        residuals = np.linalg.norm(candidate.apply(source) - world, axis=1)
        inliers = residuals <= max_error_mm
        support = int(inliers.sum())
        if support >= required:
            score = (support, -_rmse(residuals[inliers]))
            if score > best_score:
                best_inliers, best_score = inliers, score
    if best_inliers is None:
        for points in (source, world):
            _check_geometry(points - points.mean(axis=0))
        raise ValueError("控制点无法建立一致的公制坐标系，请核对 ID、单位与实测 XYZ")

    # Re-evaluate the fitted consensus so the returned residuals obey the gate.
    inliers = best_inliers
    for _ in range(10):
        transform = _fit(source[inliers], world[inliers], estimate_scale=True)
        residuals = np.linalg.norm(transform.apply(source) - world, axis=1)
        updated = residuals <= max_error_mm
        if np.array_equal(inliers, updated):
            break
        if updated.sum() < required:
            raise ValueError("重估后的控制点一致性不足")
        inliers = updated
    else:
        raise ValueError("控制点内点集合未收敛，请核对标记观测")
    return transform, {
        "scale_mm_per_unit": transform.scale,
        "rmse_mm": _rmse(residuals[inliers]),
        "max_inlier_error_mm": float(residuals[inliers].max()),
        "inlier_indices": np.flatnonzero(inliers).tolist(),
        "residuals_mm": residuals.tolist(),
        "max_error_mm": float(max_error_mm),
    }


def verify_control_frame(transform, source_xyz, world_xyz_mm, inlier_indices, *, max_error_mm=3.0):
    """Recheck already accepted survey controls after applying a rigid refinement."""
    source = _xyz(source_xyz, "重建控制点", 3)
    world = _xyz(world_xyz_mm, "实测控制点", 3)
    indices = np.asarray(inlier_indices)
    if (len(source) != len(world) or indices.ndim != 1 or indices.dtype.kind not in "iu"
            or len(np.unique(indices)) < 3 or np.any(indices < 0) or np.any(indices >= len(source))):
        raise ValueError("精配准复核需要至少 3 个有效控制点索引")
    _positive(max_error_mm, "控制点误差门限")
    errors = np.linalg.norm(transform.apply(source[indices]) - world[indices], axis=1)
    peak = float(errors.max())
    if not np.isfinite(peak) or peak > max_error_mm:
        raise ValueError(f"精配准后控制点偏差 {peak:.2f} mm 超过门限 {max_error_mm:.2f} mm")
    return {"rmse_mm": _rmse(errors), "max_error_mm": peak}


_MIN_STABLE = 12


def _stable_sample(points, label):
    points = np.unique(_xyz(points, label, _MIN_STABLE), axis=0)
    if len(points) < _MIN_STABLE:
        raise ValueError(f"{label}至少需要 {_MIN_STABLE} 个不同的稳定点")
    if len(points) > 20000:
        # Bound nearest-neighbour search costs for dense multi-view pointmaps.
        indices = np.random.default_rng(0).choice(len(points), 20000, replace=False)
        points = points[indices]
    _check_geometry(points - points.mean(axis=0))
    return points


def refine_stable_icp(before_stable_mm, after_stable_mm, *, max_distance_mm=10.0,
                      max_rmse_mm=3.0, min_overlap=0.6):
    """Return rigid after-to-before correction using only caller-selected stable points.

    Inputs are in the same coarse, metric frame and may be unpaired/unequal length.
    Nearest-neighbour fitting uses unique target correspondences. Quality gates
    measure distances in both directions on the full bounded samples before trimming.
    """
    _positive(max_distance_mm, "稳定区匹配半径")
    _positive(max_rmse_mm, "稳定区 RMSE 门限")
    if not 0 < min_overlap <= 1:
        raise ValueError("稳定区重叠率门限须在 (0, 1] 内")
    fixed = _stable_sample(before_stable_mm, "前时相稳定区")
    moving = _stable_sample(after_stable_mm, "后时相稳定区")
    tree = cKDTree(fixed)
    result = Transform(1.0, np.eye(3), np.zeros(3))
    for iteration in range(1, 61):
        aligned = result.apply(moving)
        distances, nearest = tree.query(aligned)
        matches = np.flatnonzero(distances <= max_distance_mm)
        if len(matches) < _MIN_STABLE:
            raise ValueError("稳定区近邻不足，请检查两期区域选择与粗配准")
        ordered = matches[np.argsort(distances[matches])]
        # Dense duplicates must not dominate the alignment. Keep the closest
        # source for each target and trim the largest 15% of remaining errors.
        _, first = np.unique(nearest[ordered], return_index=True)
        correspondences = ordered[np.sort(first)]
        if len(correspondences) < _MIN_STABLE:
            raise ValueError("稳定区独立对应点不足")
        keep = correspondences[:max(_MIN_STABLE, ceil(0.85 * len(correspondences)))]
        step = _fit(aligned[keep], fixed[nearest[keep]], estimate_scale=False)
        result = result.then(step)
        max_step = np.linalg.norm(step.apply(aligned) - aligned, axis=1).max()
        if max_step < max(1e-6, 0.001 * min(max_rmse_mm, max_distance_mm)):
            break
    else:
        raise ValueError("稳定区 ICP 未收敛，请检查区域或改善控制点布局")

    aligned = result.apply(moving)
    forward = tree.query(aligned)[0]
    backward = cKDTree(aligned).query(fixed)[0]
    f_ok, b_ok = forward <= max_distance_mm, backward <= max_distance_mm
    f_overlap, b_overlap = float(f_ok.mean()), float(b_ok.mean())
    overlap = min(f_overlap, b_overlap)
    if overlap < min_overlap:
        raise ValueError(f"稳定区双向重叠率 {overlap:.1%} 低于门限 {min_overlap:.1%}")
    rmse = max(_rmse(forward[f_ok]), _rmse(backward[b_ok]))
    if rmse > max_rmse_mm:
        raise ValueError(f"稳定区 RMSE {rmse:.2f} mm 超过门限 {max_rmse_mm:.2f} mm")
    return result, {
        "iterations": iteration, "rmse_mm": rmse, "overlap": overlap,
        "after_overlap": f_overlap, "before_overlap": b_overlap,
        "before_sample_count": len(fixed), "after_sample_count": len(moving),
        "max_distance_mm": float(max_distance_mm), "max_rmse_mm": float(max_rmse_mm),
        "min_overlap": float(min_overlap),
    }
