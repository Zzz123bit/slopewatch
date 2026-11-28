"""Similarity alignment and two-epoch comparison for VGGT photo reconstructions.

VGGT returns every epoch in its own arbitrary scale and camera frame, so the
two epochs must be brought into a common frame before a DEM of difference can
be computed.  This module provides a robust Sim(3) ICP, a gravity-up heuristic
and the bridge into :func:`slopewatch.core.compare_point_clouds`.
"""

from __future__ import annotations

from dataclasses import dataclass

import os

import numpy as np
from scipy.spatial import cKDTree

from .core import Comparison, compare_point_clouds


@dataclass(frozen=True)
class Similarity:
    """Column-vector convention: output = scale * rotation @ input + translation."""

    scale: float
    rotation: np.ndarray
    translation: np.ndarray

    def apply(self, points: np.ndarray) -> np.ndarray:
        points = np.asarray(points, dtype=np.float64)
        return self.scale * (points @ self.rotation.T) + self.translation


def _umeyama(source: np.ndarray, target: np.ndarray, with_scale: bool = True) -> Similarity:
    """Least-squares similarity mapping source onto paired target points."""
    source = np.asarray(source, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    if source.shape != target.shape or source.ndim != 2 or source.shape[1] != 3:
        raise ValueError("对应点须为同形状的 N×3 数组")
    if len(source) < 3:
        raise ValueError("至少需要 3 对对应点")

    source_mean = source.mean(axis=0)
    target_mean = target.mean(axis=0)
    source_centered = source - source_mean
    target_centered = target - target_mean

    covariance = target_centered.T @ source_centered / len(source)
    left, singular, right_t = np.linalg.svd(covariance)
    correction = np.eye(3)
    if np.linalg.det(left) * np.linalg.det(right_t) < 0:
        correction[2, 2] = -1.0
    rotation = left @ correction @ right_t

    scale = 1.0
    if with_scale:
        variance = float((source_centered**2).sum()) / len(source)
        if variance <= 0:
            raise ValueError("源点退化，无法估计尺度")
        scale = float(np.trace(np.diag(singular) @ correction) / variance)
        if not np.isfinite(scale) or scale <= 0:
            raise ValueError("尺度估计失败")
    translation = target_mean - scale * (rotation @ source_mean)
    return Similarity(scale, rotation, translation)


def _subsample(points: np.ndarray, limit: int, seed: int = 0) -> np.ndarray:
    if len(points) <= limit:
        return points
    indices = np.random.default_rng(seed).choice(len(points), limit, replace=False)
    return points[indices]


def _pca_rotation(moving: np.ndarray, fixed: np.ndarray) -> list[np.ndarray]:
    """Four proper-rotation candidates from principal axes, sign ambiguity resolved later."""
    def axes(points: np.ndarray) -> np.ndarray:
        centered = points - points.mean(axis=0)
        values, vectors = np.linalg.eigh(centered.T @ centered)
        ordered = vectors[:, np.argsort(values)[::-1]]
        if np.linalg.det(ordered) < 0:
            ordered[:, 2] *= -1
        return ordered

    base = axes(fixed) @ axes(moving).T
    candidates = []
    for sx, sy in ((1, 1), (1, -1), (-1, 1), (-1, -1)):
        flip = np.diag([sx, sy, sx * sy]).astype(np.float64)
        rotation = base @ flip
        if np.linalg.det(rotation) < 0:
            rotation[:, 0] *= -1
        candidates.append(rotation)
    return candidates


def align_similarity(
    moving: np.ndarray,
    fixed: np.ndarray,
    *,
    max_iterations: int = 45,
    trim_fraction: float = 0.8,
    sample_limit: int = 40_000,
) -> tuple[np.ndarray, dict]:
    """Align ``moving`` onto ``fixed`` with a robust scale-aware ICP.

    Returns the transformed moving points and a diagnostics dictionary.  The
    transform is estimated on bounded samples only; it is then applied to every
    input point.
    """
    moving_all = np.asarray(moving, dtype=np.float64)
    fixed_all = np.asarray(fixed, dtype=np.float64)
    for label, points in (("前时相", fixed_all), ("后时相", moving_all)):
        if points.ndim != 2 or points.shape[1] != 3 or len(points) < 100:
            raise ValueError(f"{label}点云至少需要 100 个三维点")
    moving_sample = _subsample(moving_all, sample_limit, seed=1)
    fixed_sample = _subsample(fixed_all, sample_limit, seed=2)
    tree = cKDTree(fixed_sample)

    moving_mean, fixed_mean = moving_sample.mean(axis=0), fixed_sample.mean(axis=0)
    moving_radius = float(np.sqrt(((moving_sample - moving_mean) ** 2).sum(axis=1).mean()))
    fixed_radius = float(np.sqrt(((fixed_sample - fixed_mean) ** 2).sum(axis=1).mean()))
    if moving_radius <= 0 or fixed_radius <= 0:
        raise ValueError("点云退化，无法估计初始尺度")
    initial_scale = fixed_radius / moving_radius

    probe = _subsample(moving_sample, 2_000, seed=3)
    best = None
    init_mode = os.environ.get("SLOPEWATCH_ALIGN_INIT", "auto").lower()
    if init_mode == "identity":
        candidates = [np.eye(3)]
    elif init_mode == "pca":
        candidates = _pca_rotation(moving_sample, fixed_sample)
    else:
        candidates = [np.eye(3), *_pca_rotation(moving_sample, fixed_sample)]
    for rotation in candidates:
        candidate = Similarity(initial_scale, rotation, fixed_mean - initial_scale * (rotation @ moving_mean))
        distances = tree.query(candidate.apply(probe))[0]
        score = float(np.median(distances))
        if best is None or score < best[0]:
            best = (score, candidate)
    transform = best[1]

    previous = np.inf
    for iteration in range(1, max_iterations + 1):
        aligned = transform.apply(moving_sample)
        distances, nearest = tree.query(aligned)
        cutoff = np.quantile(distances, trim_fraction)
        keep = distances <= max(cutoff, 1e-12)
        if keep.sum() < 12:
            raise ValueError("两期点云重叠不足，无法配准")
        transform = _umeyama(moving_sample[keep], fixed_sample[nearest[keep]], with_scale=True)
        median = float(np.median(distances[keep]))
        if abs(previous - median) <= max(1e-9, 1e-5 * median):
            converged = True
            break
        previous = median
    else:
        converged = False

    aligned = transform.apply(moving_sample)
    distances = tree.query(aligned)[0]
    info = {
        "scale": float(transform.scale),
        "rotation": transform.rotation.tolist(),
        "translation": transform.translation.tolist(),
        "iterations": iteration,
        "converged": converged,
        "median_residual": float(np.median(distances)),
        "rmse_residual": float(np.sqrt(np.mean(np.square(distances)))),
        "overlap_fraction": float(np.mean(distances <= np.quantile(distances, 0.95))),
        "sample_points": int(len(moving_sample)),
    }
    return transform.apply(moving_all), info


def _estimate_normals(points: np.ndarray, neighbours: int = 12) -> np.ndarray:
    tree = cKDTree(points)
    _, indices = tree.query(points, k=neighbours)
    patches = points[indices]
    centered = patches - patches.mean(axis=1, keepdims=True)
    covariance = np.einsum("nki,nkj->nij", centered, centered)
    _, vectors = np.linalg.eigh(covariance)
    normals = vectors[:, :, 0]
    return normals / np.maximum(np.linalg.norm(normals, axis=1, keepdims=True), 1e-12)


def refine_point_to_plane(
    source: np.ndarray,
    target: np.ndarray,
    transform: Similarity,
    *,
    iterations: int = 20,
    sample_limit: int = 25_000,
) -> Similarity:
    """Point-to-plane ICP; on smooth heightfields it avoids the tangential drift
    that a point-to-point fit trades against real surface change."""
    source_sample = _subsample(source, sample_limit, seed=11)
    target_sample = _subsample(target, sample_limit * 2, seed=12)
    normals = _estimate_normals(target_sample)
    tree = cKDTree(target_sample)

    scale = transform.scale
    rotation = transform.rotation.copy()
    translation = transform.translation.copy()
    for _ in range(iterations):
        moved = scale * (source_sample @ rotation.T) + translation
        distances, nearest = tree.query(moved)
        keep = distances <= np.quantile(distances, 0.9)
        points = moved[keep]
        normals_k = normals[nearest[keep]]
        targets = target_sample[nearest[keep]]
        design = np.column_stack((np.cross(points, normals_k), normals_k))
        residual = -np.einsum("ij,ij->i", normals_k, points - targets)
        solution, *_ = np.linalg.lstsq(design, residual, rcond=None)
        omega, shift = solution[:3], solution[3:]
        angle = float(np.linalg.norm(omega))
        if angle < 1e-12:
            delta_rotation = np.eye(3)
        else:
            axis = omega / angle
            skew = np.array([[0.0, -axis[2], axis[1]], [axis[2], 0.0, -axis[0]], [-axis[1], axis[0], 0.0]])
            delta_rotation = np.eye(3) + np.sin(angle) * skew + (1 - np.cos(angle)) * (skew @ skew)
        rotation = delta_rotation @ rotation
        translation = delta_rotation @ translation + shift
    return Similarity(scale, rotation, translation)


def trim_surface_outliers(points: np.ndarray, rounds: int = 3, gate: float = 5.0) -> np.ndarray:
    """Drop background/far points that sit far from the dominant slope plane."""
    cloud = np.asarray(points, dtype=np.float64)
    if len(cloud) < 200:
        return cloud
    keep = np.ones(len(cloud), dtype=bool)
    for _ in range(rounds):
        subset = cloud[keep]
        center = subset.mean(axis=0)
        centered = subset - center
        values, vectors = np.linalg.eigh(centered.T @ centered)
        normal = vectors[:, 0]
        distances = np.abs((cloud - center) @ normal)
        median = np.median(distances[keep])
        mad = np.median(np.abs(distances[keep] - median))
        limit = median + gate * max(mad, 1e-9)
        updated = distances <= limit
        if updated.sum() < 500 or np.array_equal(updated, keep):
            break
        keep = updated
    return cloud[keep]


def _rotation_between(source: np.ndarray, target: np.ndarray) -> np.ndarray:
    source = source / np.linalg.norm(source)
    target = target / np.linalg.norm(target)
    cross = np.cross(source, target)
    sine = np.linalg.norm(cross)
    cosine = float(np.dot(source, target))
    if sine < 1e-9:
        return np.eye(3) if cosine > 0 else np.diag([1.0, -1.0, -1.0])
    axis = cross / sine
    skew = np.array([[0.0, -axis[2], axis[1]], [axis[2], 0.0, -axis[0]], [-axis[1], axis[0], 0.0]])
    return np.eye(3) + sine * skew + (1 - cosine) * (skew @ skew)


def plane_up_rotation(points: np.ndarray, trim: float = 0.9) -> np.ndarray:
    """Rotate the dominant terrain plane onto XY so the DoD is a heightfield.

    VGGT's world frame follows the first camera, so a pitched camera makes a
    fixed gravity assumption wrong by tens of degrees.  Fitting the slope plane
    is stable for both hand-held and drone captures; the sign is chosen so the
    surface normal points back towards the camera (+Z of the VGGT frame).
    """
    cloud = np.asarray(points, dtype=np.float64)
    if cloud.ndim != 2 or cloud.shape[1] != 3 or len(cloud) < 100:
        raise ValueError("点云至少需要 100 个三维点")
    center = cloud.mean(axis=0)
    normal = np.array([0.0, 0.0, 1.0])
    for _ in range(4):
        centered = cloud - center
        values, vectors = np.linalg.eigh(centered.T @ centered)
        normal = vectors[:, 0]
        distances = centered @ normal
        keep = np.abs(distances) <= np.quantile(np.abs(distances), trim)
        if keep.sum() < 100:
            break
        cloud = cloud[keep]
        center = cloud.mean(axis=0)
    if normal[2] < 0:
        normal = -normal
    return _rotation_between(normal, np.array([0.0, 0.0, 1.0]))


def gravity_up_rotation(points: np.ndarray) -> np.ndarray:
    """Heuristic pose from a hand-held/drone capture: VGGT world Y points down.

    The first camera of VGGT defines the world frame with OpenCV axes (X right,
    Y down, Z forward).  Rotating -Y onto +Z turns the reconstruction into the
    Z-up frame used by the DEM comparison.
    """
    points = np.asarray(points)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("点云须为 N×3 数组")
    return np.array([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, -1.0, 0.0]])


def compare_reconstructions(
    before_points: np.ndarray,
    after_points: np.ndarray,
    *,
    grid_step_mm: float = 2.0,
    threshold_mm: float = 3.0,
    elapsed_hours: float = 1.0,
    scale_mm_per_unit: float = 1.0,
    max_points: int = 300_000,
    up_mode: str = "plane",
) -> tuple[Comparison, dict]:
    """Align two VGGT epochs and run the shared DoD comparison.

    ``scale_mm_per_unit`` converts reconstruction units to millimetres.  Without
    a surveyed scale bar it must be supplied by the operator; the diagnostics
    flag the result as uncalibrated when it is left at the default 1.0.
    ``up_mode`` selects the vertical axis: ``y_down`` for raw VGGT world frames,
    ``z_up`` when the points are already in a Z-up survey frame.
    """
    if up_mode not in ("plane", "y_down", "z_up"):
        raise ValueError("up_mode 仅支持 plane、y_down 或 z_up")
    before = np.asarray(before_points, dtype=np.float64)
    after = np.asarray(after_points, dtype=np.float64)
    if not np.isfinite(before).all() or not np.isfinite(after).all():
        raise ValueError("重建点云包含非有限坐标")
    if scale_mm_per_unit <= 0 or not np.isfinite(scale_mm_per_unit):
        raise ValueError("尺度系数须为有限正数")

    before = trim_surface_outliers(before)
    after = trim_surface_outliers(after)
    if len(after) > max_points:
        after = _subsample(after, max_points, seed=4)
    if len(before) > max_points:
        before = _subsample(before, max_points, seed=5)

    aligned_after, alignment = align_similarity(after, before)
    if up_mode == "plane":
        rotation = plane_up_rotation(before)
    elif up_mode == "y_down":
        rotation = gravity_up_rotation(before)
    else:
        rotation = np.eye(3)
    before_mm = (before * scale_mm_per_unit) @ rotation.T
    after_mm = (aligned_after * scale_mm_per_unit) @ rotation.T

    comparison = compare_point_clouds(
        before_mm,
        after_mm,
        grid_step_mm=grid_step_mm,
        erosion_threshold_mm=threshold_mm,
        elapsed_hours=elapsed_hours,
    )

    # Residual tilt after a global similarity fit is a well-known DEM artefact.
    # A robust plane fitted to the common cells removes it without eating the
    # local erosion signal (Huber IRLS ignores the changed cells).
    detrend = {"applied": False}
    valid = comparison.valid
    if int(valid.sum()) > 200:
        xs = comparison.x_mm[valid]
        ys = comparison.y_mm[valid]
        dz = comparison.delta_mm[valid]
        design = np.column_stack((np.ones_like(xs), xs, ys))
        weights = np.ones_like(dz)
        coefficients = np.zeros(3)
        for _ in range(6):
            weighted = design * weights[:, None]
            coefficients, *_ = np.linalg.lstsq(weighted, dz * weights, rcond=None)
            residual = dz - design @ coefficients
            scale = max(1.4826 * np.median(np.abs(residual - np.median(residual))), 1e-9)
            weights = 1.0 / np.maximum(1.0, np.abs(residual) / (1.5 * scale))
        ax, ay = after_mm[:, 0], after_mm[:, 1]
        correction = coefficients[0] + coefficients[1] * ax + coefficients[2] * ay
        after_mm = after_mm.copy()
        after_mm[:, 2] -= correction
        comparison = compare_point_clouds(
            before_mm,
            after_mm,
            grid_step_mm=grid_step_mm,
            erosion_threshold_mm=threshold_mm,
            elapsed_hours=elapsed_hours,
        )
        detrend = {
            "applied": True,
            "plane_mm": [float(value) for value in coefficients],
            "note": "稳健平面去趋势，抑制全局相似配准残余倾斜",
        }

    diagnostics = {
        "detrend": detrend,
        "alignment": alignment,
        "scale_mm_per_unit": float(scale_mm_per_unit),
        "scale_calibrated": not np.isclose(scale_mm_per_unit, 1.0),
        "coordinate_system": f"Z-up ({up_mode}), millimetres",
        "points": {"before": int(len(before)), "after": int(len(after))},
        "bbox_before_mm": [before_mm.min(axis=0).round(3).tolist(), before_mm.max(axis=0).round(3).tolist()],
        "bbox_after_mm": [after_mm.min(axis=0).round(3).tolist(), after_mm.max(axis=0).round(3).tolist()],
    }
    return comparison, diagnostics
