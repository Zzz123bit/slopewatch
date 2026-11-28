"""Fuse post-event semantic observations with DoD in the calibrated XY frame."""

from __future__ import annotations

import io
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy import ndimage as ndi

from .core import Comparison


OTHER, GULLY, COLLAPSE, PROTECTION, REFERENCE, UNKNOWN = 0, 1, 2, 3, 4, 255
CLASSES = {
    OTHER: "other",
    GULLY: "gully",
    COLLAPSE: "collapse_candidate",
    PROTECTION: "protection",
    REFERENCE: "unprotected_reference",
    UNKNOWN: "unclassified",
}
RGB = {
    OTHER: (222, 223, 211),
    GULLY: (36, 116, 167),
    COLLAPSE: (177, 81, 36),
    PROTECTION: (31, 133, 95),
    REFERENCE: (137, 111, 171),
    UNKNOWN: (153, 163, 167),
}
NO_DATA_COLOR = (39, 48, 54)


def _check_ids(labels: np.ndarray) -> np.ndarray:
    labels = np.asarray(labels)
    if labels.dtype.kind not in "iu" or not np.all(np.isin(labels, list(CLASSES))):
        raise ValueError("语义掩码只接受类别编号 0、1、2、3、4、255")
    return labels.astype(np.uint8, copy=False)


def read_label_mask(path_or_bytes: Path | str | bytes, expected_size=None) -> np.ndarray:
    """Read raw 8-bit grayscale/palette PNG IDs. expected_size is (width, height)."""
    source = io.BytesIO(path_or_bytes) if isinstance(path_or_bytes, bytes) else path_or_bytes
    with Image.open(source) as image:
        if image.format != "PNG" or image.mode not in ("L", "P") or image.n_frames != 1:
            raise ValueError("请使用单帧灰度或索引色 PNG 保存原始语义类别编号")
        if expected_size is not None and image.size != tuple(expected_size):
            raise ValueError(f"掩码尺寸 {image.size} 与原始照片尺寸 {tuple(expected_size)} 不一致")
        return _check_ids(np.array(image))


def load_segmenter(model_path=None, *, input_size=None):
    """Load one OpenCV DNN per job. Accepts a path or uint8 RGB image for prediction.

    Models must accept 1x3xSxS RGB / 255 and output 1x5xHxW finite class logits
    ordered by class ID. Resize the image isotropically, pad with black, then
    bilinearly restore each logit channel before classification.
    """
    model_path = model_path or os.environ.get("SLOPEWATCH_SEGMENTATION_ONNX")
    if not model_path:
        return None
    import cv2

    side = int(input_size or os.environ.get("SLOPEWATCH_SEGMENTATION_SIZE", "512"))
    if not 16 <= side <= 2048:
        raise ValueError("分割模型输入边长须在 16 至 2048 像素之间")
    if not Path(model_path).is_file():
        raise ValueError(f"找不到配置的 ONNX 分割模型 {Path(model_path).name}")
    net = cv2.dnn.readNetFromONNX(str(model_path))

    def predict(image) -> np.ndarray:
        if isinstance(image, (Path, str)):
            with Image.open(image) as source:
                image = np.array(source.convert("RGB"))
        image = np.asarray(image)
        if image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3:
            raise ValueError("分割输入需要 uint8 RGB 图像")
        height, width = image.shape[:2]
        scale = side / max(width, height)
        w, h = max(1, round(width * scale)), max(1, round(height * scale))
        left, top = (side - w) // 2, (side - h) // 2
        padded = np.zeros((side, side, 3), dtype=np.uint8)
        padded[top:top+h, left:left+w] = cv2.resize(image, (w, h))
        net.setInput(cv2.dnn.blobFromImage(padded, scalefactor=1 / 255.0, swapRB=False))
        output = net.forward()
        if (output.ndim != 4 or tuple(output.shape[:2]) != (1, 5)
                or not np.isfinite(output).all()):
            raise ValueError("ONNX 分割输出必须是有限的 1×5×H×W 类别分数")
        # Keep just the winning score and label at native resolution.
        prediction = np.zeros((height, width), dtype=np.uint8)
        best = np.full((height, width), -np.inf, dtype=np.float32)
        for class_id, logits in enumerate(output[0]):
            unpadded = cv2.resize(logits, (side, side))[top:top+h, left:left+w]
            scores = cv2.resize(unpadded, (width, height))
            improved = scores > best
            prediction[improved] = class_id
            np.maximum(scores, best, out=best)
        return prediction

    return predict


def segment_image(image_path: Path | str) -> np.ndarray:
    predictor = load_segmenter()
    if predictor is None:
        raise ValueError("请先配置 SLOPEWATCH_SEGMENTATION_ONNX")
    return predictor(image_path)


def project_label_views(comparison: Comparison, views) -> np.ndarray:
    """Vote once per view per cell, so dense foreground views cannot dominate.

    Each (xyz_mm, labels) pair has shape (..., 3) and (...,). Coordinates must
    already include the metric transform and post-event rigid refinement.
    A strict majority is required both within a view and across contributing
    views. Unseen, invalid and contested cells get ID 255.
    """
    rows, cols = comparison.valid.shape
    cells = rows * cols
    step = comparison.grid_step_mm
    x_start, y_start = comparison.x_mm[0, 0] - step/2, comparison.y_mm[0, 0] - step/2
    votes = np.zeros((5, cells), dtype=np.uint32)
    for points, labels in views:
        points, labels = np.asarray(points), _check_ids(labels)
        if points.shape != (*labels.shape, 3):
            raise ValueError("语义像素和三维点的尺寸不一致")
        points, labels = points.reshape(-1, 3), labels.ravel()
        inside = (np.isfinite(points).all(axis=1) & (labels != UNKNOWN)
                  & (points[:, 0] >= x_start) & (points[:, 0] < x_start + cols * step)
                  & (points[:, 1] >= y_start) & (points[:, 1] < y_start + rows * step))
        col = np.floor((points[inside, 0] - x_start) / step).astype(np.int64)
        row = np.floor((points[inside, 1] - y_start) / step).astype(np.int64)
        # Rare floating-point rounding at the uppermost edge can produce n.
        indices = np.minimum(row, rows - 1) * cols + np.minimum(col, cols - 1)
        if not indices.size:
            continue
        occupied, inv = np.unique(indices, return_inverse=True)
        count = len(occupied)
        buckets = labels[inside].astype(np.int64) * count + inv
        histogram = np.bincount(buckets, minlength=5 * count).reshape(5, count)
        winners = histogram.argmax(axis=0)
        decisive = 2 * histogram.max(axis=0) > histogram.sum(axis=0)
        votes[winners[decisive], occupied[decisive]] += 1
    winner = votes.argmax(axis=0).reshape(rows, cols).astype(np.uint8)
    supported = (2 * votes.max(axis=0) > votes.sum(axis=0)).reshape(rows, cols)
    winner[~(supported & comparison.valid)] = UNKNOWN
    return winner


def _components(mask: np.ndarray, min_cells: int):
    regions, count = ndi.label(mask, structure=np.ones((3, 3), dtype=bool))
    sizes = np.bincount(regions.ravel(), minlength=count + 1)
    for region_id, bounds in enumerate(ndi.find_objects(regions), 1):
        if sizes[region_id] >= min_cells:
            row, col = np.nonzero(regions[bounds] == region_id)
            yield row + bounds[0].start, col + bounds[1].start


def _axis_length(row: np.ndarray, col: np.ndarray, step: float) -> float:
    """PCA projected extent of grid squares along the dominant plan-view axis."""
    if row.size == 1:
        return step
    xy = np.column_stack((col - col.mean(), row - row.mean()))
    _, axes = np.linalg.eigh(xy.T @ xy)
    major = axes[:, -1]
    return float((np.ptp(xy @ major) + abs(major).sum()) * step)


def _geometry_labels(comparison: Comparison, erosion: np.ndarray, deposition: np.ndarray):
    labels = np.full(erosion.shape, UNKNOWN, dtype=np.uint8)
    # Adjacent deposition is an extra test for compact failure candidates.
    adjacent_deposition = ndi.binary_dilation(deposition, structure=np.ones((3, 3)), iterations=2)
    for row, col in _components(erosion, min_cells=4):
        axis_length = _axis_length(row, col, step=1)
        aspect = axis_length**2 / row.size
        if aspect >= 3:
            labels[row, col] = GULLY
        elif row.size >= 9 and np.count_nonzero(adjacent_deposition[row, col]) >= 3:
            labels[row, col] = COLLAPSE
    return labels


def _class_metrics(mask, erosion, deposition, delta, cell_area):
    down, up = mask & erosion, mask & deposition
    return {
        "cells": int(mask.sum()),
        "area_mm2": float(mask.sum() * cell_area),
        "eroded_area_mm2": float(down.sum() * cell_area),
        "deposited_area_mm2": float(up.sum() * cell_area),
        "erosion_volume_l": float(-delta[down].sum() * cell_area / 1e6),
        "deposition_volume_l": float(delta[up].sum() * cell_area / 1e6),
        "max_erosion_depth_mm": float(np.max(-delta[down], initial=0)),
        "max_deposition_height_mm": float(np.max(delta[up], initial=0)),
    }


def _protection_comparison(classes):
    """Area-normalized erosion contrast requires explicit protection and reference."""
    protected = classes[CLASSES[PROTECTION]]
    reference = classes[CLASSES[REFERENCE]]
    if not protected["cells"] or not reference["cells"]:
        return {"status": "missing_regions", "erosion_reduction_pct": None,
                "note": "请标注防护区及同工况下的未防护对照区。"}
    p = protected["erosion_volume_l"] * 1e6 / protected["area_mm2"]
    r = reference["erosion_volume_l"] * 1e6 / reference["area_mm2"]
    return {
        "status": "observed_contrast" if r > 0 else "zero_reference_erosion",
        "protected_mean_erosion_mm": p,
        "reference_mean_erosion_mm": r,
        "erosion_reduction_pct": 100 * (1 - p / r) if r > 0 else None,
        "note": "按各区有效面积归一化阈值以上冲刷量。百分比描述本期观测差异，因果归因需匹配坡度、土质与降雨工况。",
    }


@dataclass
class FusionResult:
    labels: np.ndarray
    summary: dict


def analyze_fusion(comparison: Comparison, labels=None, *, source="manual", threshold_mm=None) -> FusionResult:
    threshold = float(comparison.metrics["erosion_threshold_mm"] if threshold_mm is None else threshold_mm)
    if not np.isfinite(threshold) or threshold < 0:
        raise ValueError("变化检测阈值须为非负有限数值")
    delta, valid = comparison.delta_mm, comparison.valid
    erosion = valid & (delta <= -threshold) & (delta < 0)
    deposition = valid & (delta >= threshold) & (delta > 0)
    if labels is None:
        source = "geometry"
        labels = _geometry_labels(comparison, erosion, deposition)
        method = ("8 邻域内至少 4 个冲刷格，主轴长与面积等效宽之比 ≥3 时列为冲沟候选。"
                  "较紧凑的冲刷斑块至少 9 格且至少 3 格在堆积区 2 格邻域内时列为坍塌候选，需现场复核。")
    else:
        if source not in {"manual", "model", "synthetic"}:
            raise ValueError("请指定掩码来源 manual、model 或 synthetic")
        labels = _check_ids(labels)
        if labels.shape != valid.shape:
            raise ValueError("分类栅格尺寸与共同高程面不一致")
        labels = np.where(valid, labels, np.uint8(UNKNOWN))
        method = "使用观测掩码与阈值以上两期高程差的交集统计。未分类区单独汇总。"
    step = comparison.grid_step_mm
    classes = {name: _class_metrics(valid & (labels == code), erosion, deposition, delta, step**2)
               for code, name in CLASSES.items()}
    gullies = []
    for row, col in _components(valid & (labels == GULLY), min_cells=1):
        length = _axis_length(row, col, step)
        area = float(len(row) * step**2)
        x, y = comparison.x_mm[row, col], comparison.y_mm[row, col]
        depth = np.maximum(-delta[row, col], 0)
        gullies.append({
            "id": len(gullies) + 1,
            "centroid_xy_mm": [float(x.mean()), float(y.mean())],
            "bbox_xyxy_mm": [float(x.min()-step/2), float(y.min()-step/2),
                             float(x.max()+step/2), float(y.max()+step/2)],
            "length_mm_est": length, "width_mm_est": area / length, "area_mm2": area,
            "max_depth_mm": float(depth.max()), "mean_depth_mm": float(depth.mean()),
        })
    classified = int(((labels != UNKNOWN) & valid).sum())
    return FusionResult(labels, {
        "source": source, "method": method, "labels_legend": CLASSES.copy(),
        "threshold_mm": threshold, "classes": classes, "gullies": gullies,
        "gully_measurement": "平面主轴投影长度，平均等效宽度等于面积除以投影长度，深度为两期高程下降量。",
        "coverage": {"classified_cells": classified, "valid_cells": int(valid.sum()),
                     "classified_fraction": classified / max(1, int(valid.sum()))},
        "protection": _protection_comparison(classes),
    })


def _font(size):
    for name in ("DejaVuSans.ttf", "arial.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default()


def atlas_png(comparison: Comparison, result: FusionResult) -> bytes:
    """Render two XY maps with the same scale, pixel footprint and coordinates."""
    valid, delta = comparison.valid, comparison.delta_mm
    limit = float(np.abs(delta[valid]).max())
    neutral = np.array((247, 245, 236))
    # Blue loss and ochre gain are distinguished from the no-data background.
    low, high = np.array((32, 91, 151)), np.array((201, 93, 40))
    fraction = np.clip(np.abs(delta) / (limit or 1), 0, 1)[..., None]
    dod = (neutral + fraction * (np.where(delta[..., None] < 0, low, high) - neutral)).astype(np.uint8)
    dod[~valid] = NO_DATA_COLOR
    color_table = np.zeros((256, 3), dtype=np.uint8)
    for class_id in CLASSES:
        color_table[class_id] = RGB[class_id]
    labels = color_table[result.labels]
    labels[~valid] = NO_DATA_COLOR

    canvas = Image.new("RGB", (1120, 628), "#ffffff")
    draw = ImageDraw.Draw(canvas)
    heading, subheading, small = _font(27), _font(18), _font(15)
    ink = "#233a42"
    draw.text((35, 16), "Slope change atlas", font=heading, fill=ink)
    source = result.summary["source"]
    threshold = result.summary["threshold_mm"]
    draw.text((35, 55), f"Semantic source: {source}    |    Change threshold: {threshold:g} mm",
              font=small, fill=ink)
    rows, cols = valid.shape
    scale = min(490 / cols, 318 / rows)
    size = max(1, round(scale * cols)), max(1, round(scale * rows))
    step = comparison.grid_step_mm
    x_min, x_max = comparison.x_mm[0, 0] - step/2, comparison.x_mm[0, -1] + step/2
    y_min, y_max = comparison.y_mm[0, 0] - step/2, comparison.y_mm[-1, 0] + step/2
    for x0, title, data in ((35, "Elevation change (after - before)", dod),
                            (580, "Mapped features", labels)):
        draw.text((x0, 91), title, font=subheading, fill=ink)
        x = x0 + (490 - size[0]) // 2
        y = 125 + (318 - size[1]) // 2
        panel = Image.fromarray(np.flipud(data)).resize(size, Image.Resampling.NEAREST)
        canvas.paste(panel, (x, y))
        draw.text((x0, 457), f"X  {x_min:g} .. {x_max:g} mm    |    Y  {y_min:g} .. {y_max:g} mm",
                  font=small, fill=ink)

    grad = np.linspace(-1, 1, 256)[:, None]
    bar = neutral + np.abs(grad) * (np.where(grad < 0, low, high) - neutral)
    canvas.paste(Image.fromarray(bar.astype(np.uint8)[None, :, :]).resize((370, 14)), (35, 505))
    draw.text((35, 526), f"Loss  {-limit:g}", font=small, fill=ink)
    draw.text((214, 526), "0", font=small, fill=ink)
    draw.text((405, 526), f"Gain  {limit:g} mm", font=small, fill=ink, anchor="ra")
    names = ("Other", "Gully", "Collapse candidate", "Protection", "Unprotected reference", "Unclassified")
    for index, (code, name) in enumerate(zip(CLASSES, names)):
        x = 580 + (index % 2) * 240
        y = 504 + (index // 2) * 29
        draw.rectangle((x, y, x+15, y+15), fill=RGB[code])
        draw.text((x+23, y-1), name, font=small, fill=ink)
    draw.rectangle((35, 585, 50, 600), fill=NO_DATA_COLOR)
    draw.text((58, 584), f"No common DEM coverage    |    Grid: {step:g} mm    |    +Y up, +X right",
              font=small, fill=ink)
    output = io.BytesIO()
    canvas.save(output, format="PNG")
    return output.getvalue()
