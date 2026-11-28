from __future__ import annotations

import csv
import io
import json
import math
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO

import numpy as np
from scipy.interpolate import griddata
from scipy.ndimage import label
from scipy.spatial import cKDTree


PLY_TYPES = {
    "char": "i1",
    "int8": "i1",
    "uchar": "u1",
    "uint8": "u1",
    "short": "<i2",
    "int16": "<i2",
    "ushort": "<u2",
    "uint16": "<u2",
    "int": "<i4",
    "int32": "<i4",
    "uint": "<u4",
    "uint32": "<u4",
    "float": "<f4",
    "float32": "<f4",
    "double": "<f8",
    "float64": "<f8",
}
PLY_BIG_ENDIAN_TYPES = {
    name: dtype.replace("<", ">") for name, dtype in PLY_TYPES.items()
}


@dataclass
class Comparison:
    x_mm: np.ndarray
    y_mm: np.ndarray
    before_mm: np.ndarray
    after_mm: np.ndarray
    delta_mm: np.ndarray
    valid: np.ndarray
    metrics: dict
    gullies: list[dict]
    grid_step_mm: float


def _read_ply(data: bytes) -> np.ndarray:
    stream = io.BytesIO(data)
    header_lines = []
    while True:
        line = stream.readline()
        if not line:
            raise ValueError("PLY 文件缺少 end_header")
        header_lines.append(line.decode("ascii", errors="strict").strip())
        if header_lines[-1] == "end_header":
            break

    file_format = None
    vertex_count = None
    properties = []
    current_element = None
    for line in header_lines:
        parts = line.split()
        if len(parts) == 3 and parts[0] == "format":
            file_format = parts[1]
        elif len(parts) == 3 and parts[0] == "element":
            current_element = parts[1]
            if current_element == "vertex":
                vertex_count = int(parts[2])
        elif len(parts) == 3 and parts[0] == "property" and current_element == "vertex":
            properties.append((parts[1], parts[2]))
        elif parts and parts[0] == "property" and current_element == "vertex":
            raise ValueError("PLY 顶点属性包含列表字段，当前无法解析")

    if file_format is None or vertex_count is None:
        raise ValueError("PLY 文件头缺少 format 或 vertex 数量")

    names = [name.lower() for _, name in properties]
    if not all(axis in names for axis in ("x", "y", "z")):
        raise ValueError("PLY 顶点属性必须包含 x、y、z")

    if file_format == "ascii":
        rows = np.loadtxt(stream, dtype=np.float64, max_rows=vertex_count, ndmin=2)
        if rows.shape[0] != vertex_count or rows.shape[1] < len(properties):
            raise ValueError("PLY 顶点数据不完整")
        columns = {name.lower(): index for index, (_, name) in enumerate(properties)}
        return rows[:, [columns["x"], columns["y"], columns["z"]]]

    if file_format not in ("binary_little_endian", "binary_big_endian"):
        raise ValueError(f"不支持的 PLY 格式 {file_format}")

    type_map = PLY_TYPES if file_format == "binary_little_endian" else PLY_BIG_ENDIAN_TYPES
    try:
        dtype = np.dtype([(name, type_map[ply_type]) for ply_type, name in properties])
    except KeyError as error:
        raise ValueError(f"PLY 属性类型不支持 {error.args[0]}") from error
    vertices = np.frombuffer(stream.read(), dtype=dtype, count=vertex_count)
    if len(vertices) != vertex_count:
        raise ValueError("PLY 顶点数据不完整")
    return np.column_stack((vertices["x"], vertices["y"], vertices["z"])).astype(np.float64)


def _read_text_cloud(data: bytes, suffix: str) -> np.ndarray:
    text = data.decode("utf-8-sig", errors="replace")
    if suffix == ".csv":
        rows = list(csv.reader(io.StringIO(text)))
        rows = [row for row in rows if row and any(value.strip() for value in row)]
        if not rows:
            raise ValueError("CSV 文件为空")
        first = [cell.strip().lower() for cell in rows[0]]
        if all(axis in first for axis in ("x", "y", "z")):
            indexes = [first.index(axis) for axis in ("x", "y", "z")]
            rows = rows[1:]
        else:
            indexes = [0, 1, 2]
        points = []
        for row in rows:
            if len(row) > max(indexes):
                try:
                    points.append([float(row[index]) for index in indexes])
                except ValueError:
                    continue
        return np.asarray(points, dtype=np.float64)

    points = []
    for line in text.splitlines():
        fields = line.split()
        if not fields or fields[0].startswith("#"):
            continue
        try:
            values = [float(field) for field in fields]
        except ValueError:
            continue
        # COLMAP points3D.txt starts with point ID followed by XYZ.
        xyz = values[1:4] if len(values) >= 8 else values[:3]
        if len(xyz) == 3:
            points.append(xyz)
    return np.asarray(points, dtype=np.float64)


def load_point_cloud(source, filename: str, unit: str = "mm") -> np.ndarray:
    """Load XYZ coordinates from ASCII/binary PLY, XYZ/TXT, or CSV."""
    suffix = Path(filename).suffix.lower()
    if isinstance(source, (str, Path)):
        data = Path(source).read_bytes()
    elif hasattr(source, "read"):
        data = source.read()
    else:
        data = bytes(source)
    if suffix == ".ply":
        points = _read_ply(data)
    elif suffix in (".xyz", ".txt", ".csv"):
        points = _read_text_cloud(data, suffix)
    else:
        raise ValueError("点云格式请使用 PLY、XYZ、TXT 或 CSV")

    if points.ndim != 2 or points.shape[1] != 3 or len(points) < 3:
        raise ValueError("点云至少需要 3 个有效 XYZ 点")
    points = points[np.isfinite(points).all(axis=1)]
    if len(points) < 3:
        raise ValueError("点云中有效 XYZ 点不足")
    if unit == "m":
        points = points * 1000.0
    elif unit != "mm":
        raise ValueError("单位仅支持 mm 或 m")
    return points


def _unique_xy(points: np.ndarray, rounding_mm: float) -> np.ndarray:
    xy = np.round(points[:, :2] / rounding_mm).astype(np.int64)
    _, inverse = np.unique(xy, axis=0, return_inverse=True)
    counts = np.bincount(inverse)
    sums = np.column_stack(
        [np.bincount(inverse, weights=points[:, column]) for column in range(3)]
    )
    return sums / counts[:, None]


def _measure_gullies(erosion_depth: np.ndarray, valid: np.ndarray, threshold_mm: float, step_mm: float):
    active = (erosion_depth >= threshold_mm) & valid
    labels, count = label(active)
    gullies = []
    cell_area = step_mm * step_mm
    for region_id in range(1, count + 1):
        row, col = np.where(labels == region_id)
        if len(row) < 4:
            continue
        coords = np.column_stack((col * step_mm, row * step_mm))
        centered = coords - coords.mean(axis=0)
        eigenvalues, eigenvectors = np.linalg.eigh(np.cov(centered, rowvar=False))
        major_axis = eigenvectors[:, np.argmax(eigenvalues)]
        projection = centered @ major_axis
        length_mm = max(float(projection.max() - projection.min() + step_mm), step_mm)
        area_mm2 = len(row) * cell_area
        local_depth = erosion_depth[row, col]
        gullies.append(
            {
                "id": len(gullies) + 1,
                "area_mm2": round(area_mm2, 2),
                "length_mm_est": round(length_mm, 2),
                "width_mm_est": round(area_mm2 / length_mm, 2),
                "max_depth_mm": round(float(local_depth.max()), 2),
                "mean_depth_mm": round(float(local_depth.mean()), 2),
            }
        )
    gullies.sort(key=lambda item: item["area_mm2"], reverse=True)
    return gullies[:20]


def _build_comparison(
    x_mm: np.ndarray,
    y_mm: np.ndarray,
    before_mm: np.ndarray,
    after_mm: np.ndarray,
    valid: np.ndarray,
    grid_step_mm: float,
    erosion_threshold_mm: float,
    elapsed_hours: float,
) -> Comparison:
    delta = after_mm - before_mm
    erosion_depth = np.maximum(-delta, 0.0)
    deposition_height = np.maximum(delta, 0.0)
    changed = valid & (np.abs(delta) >= erosion_threshold_mm)
    erosion_mask = valid & (delta <= -erosion_threshold_mm)
    deposition_mask = valid & (delta >= erosion_threshold_mm)

    cell_area = grid_step_mm * grid_step_mm
    erosion_volume_raw_mm3 = float(erosion_depth[valid].sum() * cell_area)
    deposition_volume_raw_mm3 = float(deposition_height[valid].sum() * cell_area)
    erosion_volume_mm3 = float(erosion_depth[erosion_mask].sum() * cell_area)
    deposition_volume_mm3 = float(deposition_height[deposition_mask].sum() * cell_area)
    changed_erosion = erosion_depth[erosion_mask]
    changed_deposition = deposition_height[deposition_mask]
    erosion_mean = float(changed_erosion.mean()) if changed_erosion.size else 0.0
    deposition_mean = float(changed_deposition.mean()) if changed_deposition.size else 0.0
    max_abs_delta = float(np.abs(delta[valid]).max())
    metrics = {
        "valid_area_m2": round(float(valid.sum() * cell_area / 1_000_000), 6),
        "grid_step_mm": round(float(grid_step_mm), 3),
        "valid_cells": int(valid.sum()),
        "erosion_volume_l": round(erosion_volume_mm3 / 1_000_000, 6),
        "deposition_volume_l": round(deposition_volume_mm3 / 1_000_000, 6),
        "erosion_volume_raw_l": round(erosion_volume_raw_mm3 / 1_000_000, 6),
        "deposition_volume_raw_l": round(deposition_volume_raw_mm3 / 1_000_000, 6),
        "max_erosion_depth_mm": round(float(erosion_depth[valid].max()), 3),
        "mean_erosion_depth_mm": round(erosion_mean, 3),
        "max_deposition_height_mm": round(float(deposition_height[valid].max()), 3),
        "mean_deposition_height_mm": round(deposition_mean, 3),
        "changed_area_m2": round(float(changed.sum() * cell_area / 1_000_000), 6),
        "max_abs_change_mm": round(max_abs_delta, 3),
        "elapsed_hours": round(float(elapsed_hours), 3),
        "max_change_rate_mm_per_hour": round(max_abs_delta / elapsed_hours, 3) if elapsed_hours > 0 else None,
        "erosion_threshold_mm": round(float(erosion_threshold_mm), 3),
    }
    gullies = _measure_gullies(erosion_depth, valid, erosion_threshold_mm, grid_step_mm)
    metrics["gully_count_est"] = len(gullies)
    return Comparison(x_mm, y_mm, before_mm, after_mm, delta, valid, metrics, gullies, grid_step_mm)


def compare_point_clouds(
    before_points: np.ndarray,
    after_points: np.ndarray,
    grid_step_mm: float = 2.0,
    erosion_threshold_mm: float = 3.0,
    elapsed_hours: float = 1.0,
    bounds_xy_mm: tuple[float, float, float, float] | None = None,
) -> Comparison:
    if grid_step_mm <= 0 or erosion_threshold_mm < 0 or elapsed_hours < 0:
        raise ValueError("网格间距需大于 0，阈值和时长不能为负")

    x_min = max(float(before_points[:, 0].min()), float(after_points[:, 0].min()))
    x_max = min(float(before_points[:, 0].max()), float(after_points[:, 0].max()))
    y_min = max(float(before_points[:, 1].min()), float(after_points[:, 1].min()))
    y_max = min(float(before_points[:, 1].max()), float(after_points[:, 1].max()))
    if x_max <= x_min or y_max <= y_min:
        raise ValueError("两期点云没有重叠的平面范围")

    if bounds_xy_mm is not None:
        # Survey volumes integrate complete horizontal cells inside the measured plot.
        x_min, y_min, x_max, y_max = bounds_xy_mm
        nx = math.floor((x_max - x_min) / grid_step_mm)
        ny = math.floor((y_max - y_min) / grid_step_mm)
        x_min += grid_step_mm / 2
        y_min += grid_step_mm / 2
    else:
        nx = math.ceil((x_max - x_min) / grid_step_mm + 0.25)
        ny = math.ceil((y_max - y_min) / grid_step_mm + 0.25)
    if min(nx, ny) < 1 or nx * ny > 1_000_000:
        raise ValueError("分析范围须至少容纳一格，且总网格不能超过 100 万格")
    x_coords = x_min + np.arange(nx) * grid_step_mm
    y_coords = y_min + np.arange(ny) * grid_step_mm
    grid_x, grid_y = np.meshgrid(x_coords, y_coords)
    grid = (grid_x, grid_y)

    rounding_mm = max(grid_step_mm / 2.0, 0.01)
    before_unique = _unique_xy(before_points, rounding_mm)
    after_unique = _unique_xy(after_points, rounding_mm)
    if len(before_unique) > 200_000:
        before_unique = before_unique[:: math.ceil(len(before_unique) / 200_000)]
    if len(after_unique) > 200_000:
        after_unique = after_unique[:: math.ceil(len(after_unique) / 200_000)]

    try:
        before_grid = griddata(before_unique[:, :2], before_unique[:, 2], grid, method="linear")
        after_grid = griddata(after_unique[:, :2], after_unique[:, 2], grid, method="linear")
    except Exception as error:
        raise ValueError(f"点云无法插值为高程面，请检查 XY 覆盖和分布。{error}") from error

    before_tree = cKDTree(before_unique[:, :2])
    after_tree = cKDTree(after_unique[:, :2])
    before_sample = before_unique[:: max(1, len(before_unique) // 10_000), :2]
    after_sample = after_unique[:: max(1, len(after_unique) // 10_000), :2]
    before_nearest = before_tree.query(before_sample, k=2)[0][:, 1]
    after_nearest = after_tree.query(after_sample, k=2)[0][:, 1]
    typical_spacing = float(np.median(np.concatenate((before_nearest, after_nearest))))
    max_supported_gap_mm = max(grid_step_mm * 2.0, min(grid_step_mm * 5.0, typical_spacing * 3.0))
    grid_points = np.column_stack((grid_x.ravel(), grid_y.ravel()))
    before_distance = before_tree.query(grid_points, k=1)[0].reshape(grid_x.shape)
    after_distance = after_tree.query(grid_points, k=1)[0].reshape(grid_x.shape)
    valid = (
        np.isfinite(before_grid)
        & np.isfinite(after_grid)
        & (before_distance <= max_supported_gap_mm)
        & (after_distance <= max_supported_gap_mm)
    )
    if valid.sum() < 9:
        raise ValueError("两期点云共同有效区域不足，请检查配准和覆盖范围")
    before_grid = np.where(valid, before_grid, 0.0)
    after_grid = np.where(valid, after_grid, 0.0)
    return _build_comparison(
        grid_x,
        grid_y,
        before_grid,
        after_grid,
        valid,
        grid_step_mm,
        erosion_threshold_mm,
        elapsed_hours,
    )


def make_demo_comparison() -> Comparison:
    step = 2.0
    x_coords = np.arange(0.0, 202.0, step)
    y_coords = np.arange(0.0, 122.0, step)
    x, y = np.meshgrid(x_coords, y_coords)
    surface = 72.0 - 0.19 * x + 1.3 * np.sin(x / 31.0) + 0.7 * np.cos(y / 17.0)
    channel_center = 58.0 + 0.12 * x + 5.0 * np.sin(x / 38.0)
    channel = 9.5 * np.exp(-((y - channel_center) / 5.0) ** 2)
    fan_center = 58.0 + 0.12 * 180.0 + 5.0 * np.sin(180.0 / 38.0)
    deposit = 4.5 * np.exp(-((x - 176.0) / 18.0) ** 2 - ((y - fan_center) / 15.0) ** 2)
    after = surface - channel + deposit
    valid = np.ones_like(surface, dtype=bool)
    return _build_comparison(x, y, surface, after, valid, step, 2.5, 0.5)


def comparison_csv(comparison: Comparison) -> bytes:
    output = io.StringIO(newline="")
    writer = csv.writer(output)
    writer.writerow(["x_mm", "y_mm", "z_before_mm", "z_after_mm", "delta_mm", "valid"])
    for row in range(comparison.valid.shape[0]):
        for col in range(comparison.valid.shape[1]):
            if comparison.valid[row, col]:
                writer.writerow(
                    [
                        round(float(comparison.x_mm[row, col]), 3),
                        round(float(comparison.y_mm[row, col]), 3),
                        round(float(comparison.before_mm[row, col]), 4),
                        round(float(comparison.after_mm[row, col]), 4),
                        round(float(comparison.delta_mm[row, col]), 4),
                        True,
                    ]
                )
    return output.getvalue().encode("utf-8-sig")


def comparison_json(comparison: Comparison, metadata: dict | None = None) -> bytes:
    payload = {
        "metadata": metadata or {},
        "metrics": comparison.metrics,
        "gullies": comparison.gullies,
    }
    return json.dumps(payload, ensure_ascii=False, indent=2).encode("utf-8")

