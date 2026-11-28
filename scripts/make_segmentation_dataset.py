"""Generate a synthetic slope dataset with per-pixel semantic labels.

Classes follow ``slopewatch.fusion``: 0 other, 1 gully, 2 collapse candidate,
3 protection, 4 unprotected reference.  Each scene randomises the slope, gully
layout, a collapse pit, a protection mat and a bare reference strip, then
renders two hand-held viewpoints with splatting.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

WIDTH, HEIGHT = 640, 480
FOCAL = 780.0
GRID = 0.0016
X_RANGE = (-0.55, 0.55)
Y_RANGE = (-0.38, 0.38)


@dataclass
class Scene:
    slope: float
    gully_x: float
    gully_w: float
    gully_depth: float
    gully2_x: float
    collapse_x: float
    collapse_y: float
    collapse_depth: float
    protect_x: tuple[float, float]
    reference_x: tuple[float, float]
    seed: int


def sample_scene(rng) -> Scene:
    protect_lo = float(rng.uniform(-0.45, -0.15))
    return Scene(
        slope=float(rng.uniform(0.35, 0.65)),
        gully_x=float(rng.uniform(-0.05, 0.10)),
        gully_w=float(rng.uniform(0.030, 0.055)),
        gully_depth=float(rng.uniform(0.05, 0.11)),
        gully2_x=float(rng.uniform(0.18, 0.32)),
        collapse_x=float(rng.uniform(0.18, 0.38)),
        collapse_y=float(rng.uniform(-0.15, 0.05)),
        collapse_depth=float(rng.uniform(0.06, 0.12)),
        protect_x=(protect_lo, protect_lo + float(rng.uniform(0.16, 0.24))),
        reference_x=(protect_lo + 0.28, protect_lo + 0.46),
        seed=int(rng.integers(0, 1_000_000)),
    )


def _ramp(y):
    return np.clip((y + 0.30) / 0.55, 0.0, 1.0)


def protect_mask(x, scene):
    return (x >= scene.protect_x[0]) & (x <= scene.protect_x[1])


def reference_mask(x, scene):
    return (x >= scene.reference_x[0]) & (x <= scene.reference_x[1])


def fields(x, y, scene):
    rough = 0.003 * np.sin(41 * x + 1.3) * np.cos(29 * y) + 0.002 * np.sin(63 * y)
    gully = -scene.gully_depth * _ramp(y) * np.exp(-(((x - scene.gully_x - 0.05 * np.sin(2.1 * y)) / scene.gully_w) ** 2))
    gully2 = -0.6 * scene.gully_depth * _ramp(y) * np.exp(-(((x - scene.gully2_x) / (scene.gully_w * 0.8)) ** 2))
    collapse = -scene.collapse_depth * np.exp(
        -(((x - scene.collapse_x) ** 2 + (y - scene.collapse_y) ** 2) / (0.062**2))
    )
    mat = protect_mask(x, scene)
    gully = np.where(mat, gully * 0.05, gully)
    gully2 = np.where(mat, gully2 * 0.05, gully2)
    base = -scene.slope * x + 0.06
    return base + rough + gully + gully2 + collapse, gully, gully2, collapse


def labels_for(x, y, gully, gully2, collapse, scene):
    labels = np.zeros(x.shape, dtype=np.uint8)
    labels[reference_mask(x, scene)] = 4
    labels[protect_mask(x, scene)] = 3
    gully_gate = 0.35 * scene.gully_depth
    labels[(np.abs(gully) + np.abs(gully2) > gully_gate) & (labels == 0)] = 1
    labels[(np.abs(collapse) > 0.25 * scene.collapse_depth) & (labels == 0)] = 2
    return labels


def albedo(x, y, labels, rng, scene):
    grain = rng.uniform(-0.05, 0.05, x.shape)
    base = np.stack([0.55 + grain, 0.44 + grain, 0.32 + grain], axis=-1)
    mat = labels == 3
    if mat.any():
        grid = ((np.sin(70 * x) > 0.6) | (np.sin(70 * y) > 0.6)).astype(np.float32) * 0.12
        mat_colour = np.stack([0.22 + grid, 0.42 + grid, 0.28 + grid], axis=-1)
        base[mat] = mat_colour[mat]
    ref = labels == 4
    if ref.any():
        base[ref] = np.stack([0.62 + grain[ref], 0.50 + grain[ref], 0.36 + grain[ref]], axis=-1)
        lo, hi = scene.reference_x
        frame = ref & (((x < lo + 0.012) | (x > hi - 0.012)))
        base[frame] = np.array([0.88, 0.88, 0.84])
    return np.clip(base, 0.05, 0.95)


def normals(x, y, scene):
    dx = 1e-4
    dzdx = (fields(x + dx, y, scene)[0] - fields(x - dx, y, scene)[0]) / (2 * dx)
    dzdy = (fields(x, y + dx, scene)[0] - fields(x, y - dx, scene)[0]) / (2 * dx)
    normal = np.stack([-dzdx, -dzdy, np.ones_like(dzdx)], axis=-1)
    return normal / np.linalg.norm(normal, axis=-1, keepdims=True)


def _look_at(eye, target):
    forward = target - eye
    forward /= np.linalg.norm(forward)
    right = np.cross(forward, np.array([0.0, 0.0, 1.0]))
    right /= np.linalg.norm(right)
    up = np.cross(right, forward)
    return np.stack([right, up, forward]), eye


def cameras(rng):
    result = []
    for _ in range(2):
        azimuth = rng.uniform(-0.35, 0.35)
        distance = rng.uniform(1.05, 1.35)
        eye = np.array([distance * np.sin(azimuth), -0.25 + rng.uniform(-0.05, 0.05),
                        0.85 + rng.uniform(-0.06, 0.06)])
        result.append(_look_at(eye, np.array([0.0, 0.02, -0.02])))
    return result


def _splat(points, channels, rotation, eye):
    local = (points - eye) @ rotation.T
    depth = local[:, 2]
    valid = depth > 0.05
    local, channels, depth = local[valid], channels[valid], depth[valid]
    u = (FOCAL * local[:, 0] / depth + WIDTH / 2).round().astype(np.int32)
    v = (FOCAL * local[:, 1] / depth + HEIGHT / 2).round().astype(np.int32)
    inside = (u >= 0) & (u < WIDTH) & (v >= 0) & (v < HEIGHT)
    u, v, channels, depth = u[inside], v[inside], channels[inside], depth[inside]
    # A 3x3 footprint closes the gaps left by point splatting at ~1 px spacing.
    us = np.concatenate([u + dx for dx in (-1, 0, 1)])
    vs = np.concatenate([v + dy for dy in (-1, 0, 1)])
    cs = np.concatenate([channels] * 3)
    ds = np.concatenate([depth] * 3)
    ok = (us >= 0) & (us < WIDTH) & (vs >= 0) & (vs < HEIGHT)
    us, vs, cs, ds = us[ok], vs[ok], cs[ok], ds[ok]
    order = np.argsort(-ds)
    canvas = np.zeros((HEIGHT, WIDTH, channels.shape[1]), dtype=np.float32)
    canvas[vs[order], us[order]] = cs[order]
    covered = np.zeros((HEIGHT, WIDTH), dtype=np.uint8)
    covered[vs, us] = 255
    return canvas, covered


def render(points, colours, classes, rotation, eye):
    rgb, covered = _splat(points, colours, rotation, eye)
    labels, _ = _splat(points, classes[:, None].astype(np.float32), rotation, eye)
    labels = labels[..., 0]
    holes = cv2.bitwise_not(covered)
    rgb = cv2.inpaint((rgb * 255).astype(np.uint8), holes, 6, cv2.INPAINT_TELEA)
    labels = labels.astype(np.uint8)
    kernel = np.ones((3, 3), np.uint8)
    for _ in range(8):
        missing = covered == 0
        if not missing.any():
            break
        grown = cv2.dilate(labels, kernel)
        labels[missing] = grown[missing]
        covered = cv2.dilate(covered, kernel)
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), labels


def main(out: Path, scenes: int = 120):
    rng = np.random.default_rng(20261128)
    (out / "images").mkdir(parents=True, exist_ok=True)
    (out / "masks").mkdir(parents=True, exist_ok=True)
    xs = np.arange(*X_RANGE, GRID)
    ys = np.arange(*Y_RANGE, GRID)
    gx, gy = np.meshgrid(xs, ys)
    gx = gx + rng.uniform(-GRID / 3, GRID / 3, gx.shape)
    gy = gy + rng.uniform(-GRID / 3, GRID / 3, gy.shape)
    index = 0
    for scene_index in range(scenes):
        scene = sample_scene(rng)
        z, gully, gully2, collapse = fields(gx, gy, scene)
        labels = labels_for(gx, gy, gully, gully2, collapse, scene)
        normal = normals(gx, gy, scene)
        light = np.array([-0.55, -0.35, 0.76])
        light = light / np.linalg.norm(light)
        lambert = np.clip(normal @ light, 0.15, 1.0)
        colours = np.clip(albedo(gx, gy, labels, rng, scene) * (0.45 + 0.75 * lambert)[..., None], 0.05, 0.98)
        points = np.stack([gx.ravel(), gy.ravel(), z.ravel()], axis=-1)
        flat_colours = colours.reshape(-1, 3).astype(np.float32)
        flat_labels = labels.ravel()
        for rotation, eye in cameras(rng):
            rgb, mask = render(points, flat_colours, flat_labels, rotation, eye)
            cv2.imwrite(str(out / "images" / f"{index:04d}.png"), rgb)
            cv2.imwrite(str(out / "masks" / f"{index:04d}.png"), mask)
            index += 1
    meta = {"images": index, "classes": {"0": "other", "1": "gully", "2": "collapse_candidate",
                                         "3": "protection", "4": "unprotected_reference"}}
    (out / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(meta, ensure_ascii=False))


if __name__ == "__main__":
    import sys

    target = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parents[1] / "data" / "segmentation"
    main(target)
