"""Render a synthetic two-epoch multi-view slope dataset for VGGT experiments.

The terrain is an analytic heightfield with known erosion, rendered by splatting
the surface into eight pinhole cameras.  Ground-truth volumes are written next to
the images so the reconstruction pipeline can be scored end to end.
"""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

WIDTH, HEIGHT = 640, 480
FOCAL = 900.0
GRID = 0.0012  # metres between surface samples
X_RANGE = (-0.55, 0.55)
Y_RANGE = (-0.38, 0.38)


def _ramp(y):
    return np.clip((y + 0.30) / 0.55, 0.0, 1.0)


def terrain(x, y, stage):
    base = -0.52 * x + 0.05
    rocks = 0.035 * np.exp(-(((x + 0.28) ** 2 + (y - 0.10) ** 2) / 0.02**2))
    rocks += 0.030 * np.exp(-(((x - 0.05) ** 2 + (y + 0.22) ** 2) / 0.018**2))
    rocks += 0.025 * np.exp(-(((x - 0.36) ** 2 + (y - 0.18) ** 2) / 0.016**2))
    rough = 0.0035 * np.sin(38 * x + 2.1) * np.cos(27 * y) + 0.002 * np.sin(70 * y)
    gully_axis = 0.02 + 0.06 * np.sin(2.4 * y)
    gully = -0.120 * stage * _ramp(y) * np.exp(-(((x - gully_axis) / 0.035) ** 2))
    gully2 = -0.070 * stage * _ramp(y) * np.exp(-(((x - 0.20 - 0.03 * np.sin(3 * y)) / 0.022) ** 2))
    collapse = -0.100 * stage * np.exp(-(((x - 0.27) ** 2 + (y + 0.06) ** 2) / 0.030**2))
    fan = 0.040 * stage * np.exp(-(((x - 0.02) ** 2) / 0.16**2 + ((y - 0.30) ** 2) / 0.05**2))
    return base + rough + rocks + gully + gully2 + collapse + fan


def _normal(x, y, stage=0.0):
    dx, dy = 1e-4, 1e-4
    dzdx = (terrain(x + dx, y, stage) - terrain(x - dx, y, stage)) / (2 * dx)
    dzdy = (terrain(x, y + dy, stage) - terrain(x, y - dy, stage)) / (2 * dy)
    normal = np.stack([-dzdx, -dzdy, np.ones_like(dzdx)], axis=-1)
    return normal / np.linalg.norm(normal, axis=-1, keepdims=True)


def _albedo(x, y, rng):
    grain = rng.uniform(-0.05, 0.05, x.shape)
    broad = 0.06 * np.sin(6.0 * x) * np.cos(4.5 * y) + 0.04 * np.sin(11.0 * y + 1.0)
    speckle = (rng.random(x.shape) < 0.05).astype(np.float32) * 0.16
    base = np.stack([0.55 + grain + broad, 0.45 + grain + broad, 0.33 + grain + broad], axis=-1)
    mat = (x > -0.45) & (x < -0.25)
    if mat.any():
        grid = ((np.sin(80 * x) > 0.6) | (np.sin(80 * y) > 0.6)).astype(np.float32) * 0.12
        base[mat] = np.stack([0.22 + grid[mat], 0.42 + grid[mat], 0.28 + grid[mat]], axis=-1)
    reference = (x > -0.21) & (x < -0.02)
    if reference.any():
        base[reference] = np.stack([0.62 + grain[reference], 0.50 + grain[reference], 0.36 + grain[reference]], axis=-1)
        frame = reference & ((x < -0.198) | (x > -0.032))
        base[frame] = np.array([0.88, 0.88, 0.84])
    return np.clip(base - speckle[..., None], 0.05, 0.95)


def _look_at(eye, target):
    forward = target - eye
    forward = forward / np.linalg.norm(forward)
    right = np.cross(forward, np.array([0.0, 0.0, 1.0]))
    right /= np.linalg.norm(right)
    up = np.cross(right, forward)
    return np.stack([right, up, forward]), eye


def _render(points, colours, rotation, eye, focal=FOCAL):
    local = (points - eye) @ rotation.T
    depth = local[:, 2]
    valid = depth > 0.05
    local, colours, depth = local[valid], colours[valid], depth[valid]
    u = (focal * local[:, 0] / depth + WIDTH / 2).round().astype(np.int32)
    v = (focal * local[:, 1] / depth + HEIGHT / 2).round().astype(np.int32)
    inside = (u >= 0) & (u < WIDTH) & (v >= 0) & (v < HEIGHT)
    u, v, colours, depth = u[inside], v[inside], colours[inside], depth[inside]

    us = np.concatenate([u + dx for dx in (-1, 0, 1)])
    vs = np.concatenate([v + dy for dy in (-1, 0, 1)])
    cs = np.concatenate([colours] * 3)
    ds = np.concatenate([depth] * 3)
    ok = (us >= 0) & (us < WIDTH) & (vs >= 0) & (vs < HEIGHT)
    us, vs, cs, ds = us[ok], vs[ok], cs[ok], ds[ok]
    order = np.argsort(-ds)  # far to near, nearest wins
    image = np.full((HEIGHT, WIDTH, 3), float(np.median(cs.min(axis=0))), dtype=np.float32)
    image[vs[order], us[order]] = cs[order]
    covered = np.zeros((HEIGHT, WIDTH), dtype=np.uint8)
    covered[vs, us] = 255

    holes = cv2.bitwise_not(cv2.dilate(covered, np.ones((3, 3), np.uint8)))
    rgb = cv2.inpaint((image * 255).astype(np.uint8), holes, 4, cv2.INPAINT_TELEA)
    points2d = np.column_stack(np.nonzero(covered))[:, ::-1]
    silhouette = np.zeros((HEIGHT, WIDTH), np.uint8)
    if len(points2d) >= 3:
        cv2.fillPoly(silhouette, [cv2.convexHull(points2d)], 255)
    rgb[silhouette == 0] = 240
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


def _cameras():
    cameras = []
    for index in range(8):
        t = index / 7
        x = -0.46 + 0.92 * t
        y = -0.30 + 0.06 * np.cos(3.0 * t)
        eye = np.array([x, y, 0.95 + 0.05 * np.sin(2.2 * t)])
        target = np.array([0.0, 0.02, -0.04])
        rotation, origin = _look_at(eye, target)
        cameras.append((rotation, origin))
    return cameras


def main(out_root: Path):
    rng = np.random.default_rng(7)
    xs = np.arange(*X_RANGE, GRID)
    ys = np.arange(*Y_RANGE, GRID)
    gx, gy = np.meshgrid(xs, ys)
    gx = gx + rng.uniform(-GRID / 3, GRID / 3, gx.shape)
    gy = gy + rng.uniform(-GRID / 3, GRID / 3, gy.shape)

    before_z = terrain(gx, gy, 0.0)
    after_z = terrain(gx, gy, 1.0)
    albedo = _albedo(gx, gy, rng)
    light = np.array([-0.55, -0.25, 0.80])
    light = light / np.linalg.norm(light)

    for epoch, z, stage in (("before", before_z, 0.0), ("after", after_z, 1.0)):
        normal = _normal(gx, gy, stage)
        lambert = np.clip(normal @ light, 0.12, 1.0)
        shade = (0.22 + 0.95 * lambert)[..., None]
        points = np.stack([gx.ravel(), gy.ravel(), z.ravel()], axis=-1)
        colours = np.clip(albedo * shade, 0.03, 0.98).reshape(-1, 3).astype(np.float32)
        directory = out_root / epoch
        directory.mkdir(parents=True, exist_ok=True)
        for index, (rotation, eye) in enumerate(_cameras()):
            frame = _render(points, colours, rotation, eye)
            cv2.imwrite(str(directory / f"view_{index:02d}.png"), frame)

    cell = (GRID**2)
    eroded = np.maximum(before_z - after_z, 0.0).sum() * cell
    deposited = np.maximum(after_z - before_z, 0.0).sum() * cell
    truth = {
        "units": "m",
        "grid_m": GRID,
        "erosion_volume_m3": float(eroded),
        "erosion_volume_l": float(eroded * 1000),
        "deposition_volume_m3": float(deposited),
        "deposition_volume_l": float(deposited * 1000),
        "slope_deg": float(np.degrees(np.arctan(0.52))),
        "max_erosion_depth_mm": float(np.max(before_z - after_z) * 1000),
        "cameras": len(_cameras()),
        "image_size": [WIDTH, HEIGHT],
    }
    (out_root / "ground_truth.json").write_text(json.dumps(truth, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(truth, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    import sys

    target = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parents[1] / "data" / "samples" / "synthetic_slope"
    main(target)
