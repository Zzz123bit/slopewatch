"""Rule-based (weakly supervised) geomorphological labelling from a DoD.

Instead of hand-painting 2D masks, derive labels from the measured surface
change using published morphological criteria:

* rill / gully: elongated erosion channel (aspect ratio >= 3), length >= 10 cm,
  width <= 40 cm.  The 30 cm depth/width boundary between rills and gullies is
  the common FAO / USDA-NRCS convention; here the classes are merged into one
  "gully" class because the reconstruction cannot resolve that boundary reliably.
* collapse / bank failure: compact erosion patch (aspect < 3), area >= 4 cm^2,
  max depth >= 10 mm, adjacent deposition or a steep headwall.
* sheet erosion: broad, shallow lowering below the gully threshold -> "other".

References for the criteria (used as documentation, not as data):
- SL 190-2007《土壤侵蚀分类分级标准》
- GB/T 20465-2006《水土保持术语》
- USDA-NRCS rill/gully definitions (gully: depth > 30 cm, width > 30 cm)

The rules are intentionally conservative: uncertain cells are left at 255 and
ignored by training.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import ndimage as ndi

from .core import Comparison

OTHER, GULLY, COLLAPSE, UNKNOWN = 0, 1, 2, 255

DEFAULT_RULES = {
    "min_depth_mm": 5.0,
    "gully_min_length_mm": 100.0,
    "gully_max_width_mm": 400.0,
    "gully_min_aspect": 3.0,
    "collapse_min_area_mm2": 400.0,
    "collapse_max_aspect": 2.5,
    "collapse_min_depth_mm": 10.0,
    "deposition_adjacency_mm": 40.0,
}


@dataclass
class WeakLabelResult:
    labels: np.ndarray
    regions: list[dict]
    rules: dict
    stats: dict


def _region_metrics(rows: np.ndarray, cols: np.ndarray, comparison: Comparison) -> dict:
    step = comparison.grid_step_mm
    xy = np.column_stack((cols * step, rows * step)).astype(np.float64)
    centered = xy - xy.mean(axis=0)
    if len(xy) >= 3:
        values, vectors = np.linalg.eigh(centered.T @ centered)
        major = vectors[:, -1]
        projection = centered @ major
        length = float(np.ptp(projection) + step)
        orientation = float(np.degrees(np.arctan2(major[1], major[0])))
    else:
        length = float(step)
        orientation = 0.0
    area = float(len(rows) * step * step)
    width = area / max(length, step)
    depth = np.maximum(-comparison.delta_mm[rows, cols], 0.0)
    return {
        "cells": int(len(rows)),
        "area_mm2": round(area, 1),
        "length_mm": round(length, 1),
        "width_mm": round(width, 1),
        "aspect": round(length / max(width, step), 2),
        "max_depth_mm": round(float(depth.max()), 2),
        "mean_depth_mm": round(float(depth.mean()), 2),
        "centroid_xy_mm": [round(float(xy[:, 0].mean()), 1), round(float(xy[:, 1].mean()), 1)],
        "orientation_deg": round(orientation, 1),
    }


def label_change(comparison: Comparison, rules: dict | None = None) -> WeakLabelResult:
    """Assign gully / collapse / other labels from a DEM of difference."""
    params = {**DEFAULT_RULES, **(rules or {})}
    delta = comparison.delta_mm
    valid = comparison.valid
    min_depth = float(params["min_depth_mm"])

    erosion = valid & (delta <= -min_depth)
    deposition = valid & (delta >= min_depth)

    labels = np.full(delta.shape, UNKNOWN, dtype=np.uint8)
    labels[valid] = OTHER
    regions: list[dict] = []

    adjacency_cells = max(1, int(round(params["deposition_adjacency_mm"] / comparison.grid_step_mm)))
    near_deposition = ndi.binary_dilation(
        deposition, structure=np.ones((3, 3), dtype=bool), iterations=adjacency_cells
    )
    components, count = ndi.label(erosion, structure=np.ones((3, 3), dtype=bool))
    for region_id in range(1, count + 1):
        rows, cols = np.nonzero(components == region_id)
        if len(rows) < 2:
            continue
        metrics = _region_metrics(rows, cols, comparison)
        touches_deposition = bool(near_deposition[rows, cols].any())
        if (metrics["aspect"] >= params["gully_min_aspect"]
                and metrics["length_mm"] >= params["gully_min_length_mm"]
                and metrics["width_mm"] <= params["gully_max_width_mm"]):
            label = GULLY
            reason = "细长下切（长宽比≥3、长度≥10 cm、宽度≤40 cm）"
        elif (metrics["area_mm2"] >= params["collapse_min_area_mm2"]
              and metrics["max_depth_mm"] >= params["collapse_min_depth_mm"]
              and metrics["aspect"] < params["collapse_max_aspect"]
              and touches_deposition):
            label = COLLAPSE
            reason = "紧凑深坑且邻近堆积（面积≥4 cm²、深≥10 mm、长宽比<2.5）"
        else:
            label = OTHER
            reason = "面蚀或形态不满足冲沟/坍塌判据"
        labels[rows, cols] = label
        metrics.update(region_id=len(regions) + 1, label=int(label), reason=reason,
                       touches_deposition=touches_deposition)
        regions.append(metrics)

    eroded_cells = int(erosion.sum())
    gully_cells = int((labels == GULLY).sum())
    collapse_cells = int((labels == COLLAPSE).sum())
    stats = {
        "valid_cells": int(valid.sum()),
        "eroded_cells": eroded_cells,
        "gully_cells": gully_cells,
        "collapse_cells": collapse_cells,
        "unknown_cells": int((labels == UNKNOWN).sum()),
        "gully_fraction_of_eroded": round(gully_cells / eroded_cells, 3) if eroded_cells else 0.0,
        "regions": len(regions),
        "rules": params,
    }
    return WeakLabelResult(labels, regions, params, stats)


def labels_to_rgb(labels: np.ndarray) -> np.ndarray:
    """Colour map used by reports and quick previews (RGB uint8)."""
    palette = np.array([
        (222, 223, 211),
        (36, 116, 167),
        (177, 81, 36),
        (60, 66, 70),
    ], dtype=np.uint8)
    index = np.where(labels == UNKNOWN, 3, np.clip(labels, 0, 2))
    return palette[index]
