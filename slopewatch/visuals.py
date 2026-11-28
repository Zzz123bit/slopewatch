"""Server-side figures for the photo-compare page (works without any CDN)."""

from __future__ import annotations

import io
from pathlib import Path

import numpy as np

from .core import Comparison

_FONT = Path(r"C:\Windows\Fonts\msyh.ttc")
_font_ready = False


def _use_cjk_font():
    global _font_ready
    if _font_ready:
        return
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib import font_manager, rcParams

    if _FONT.is_file():
        font_manager.fontManager.addfont(str(_FONT))
        rcParams["font.family"] = ["Microsoft YaHei", "DejaVu Sans"]
    rcParams["axes.unicode_minus"] = False
    _font_ready = True


def _stride(array: np.ndarray, maximum_side: int) -> np.ndarray:
    step = max(1, int(np.ceil(max(array.shape) / maximum_side)))
    return array[::step, ::step]


def surface_payload(comparison: Comparison, maximum_side: int = 110) -> dict:
    """Downsampled grids for optional interactive 3D rendering."""
    before = _stride(comparison.before_mm, maximum_side)
    after = _stride(comparison.after_mm, maximum_side)
    delta = _stride(comparison.delta_mm, maximum_side)
    valid = _stride(comparison.valid, maximum_side)
    x = _stride(comparison.x_mm, maximum_side)[0, :]
    y = _stride(comparison.y_mm, maximum_side)[:, 0]

    def clean(array):
        return [[None if not valid[row, col] else round(float(array[row, col]), 3)
                 for col in range(array.shape[1])] for row in range(array.shape[0])]

    return {
        "x_mm": [round(float(value), 2) for value in x],
        "y_mm": [round(float(value), 2) for value in y],
        "before_mm": clean(before),
        "after_mm": clean(after),
        "delta_mm": clean(delta),
        "valid": valid.astype(bool).tolist(),
    }


def render_surface_png(comparison: Comparison, maximum_side: int = 90) -> bytes:
    """Two shaded 3D views of the slope, before and after."""
    _use_cjk_font()
    import matplotlib.pyplot as plt

    before = _stride(comparison.before_mm, maximum_side)
    after = _stride(comparison.after_mm, maximum_side)
    valid = _stride(comparison.valid, maximum_side)
    x = _stride(comparison.x_mm, maximum_side)[0, :]
    y = _stride(comparison.y_mm, maximum_side)[:, 0]
    grid_x, grid_y = np.meshgrid(x, y)
    mask = ~valid
    before = np.ma.array(before, mask=mask)
    after = np.ma.array(after, mask=mask)
    values = np.concatenate([before.compressed(), after.compressed()]) if valid.any() else np.array([0.0, 1.0])
    low, high = float(values.min()), float(values.max())
    if high - low < 1e-6:
        high = low + 1.0

    figure = plt.figure(figsize=(11, 4.6), dpi=150)
    for index, (title, surface) in enumerate((("雨前坡面", before), ("雨后坡面", after)), start=1):
        axes = figure.add_subplot(1, 2, index, projection="3d")
        axes.plot_surface(grid_x, grid_y, surface, cmap="terrain", linewidth=0, antialiased=True,
                          vmin=low, vmax=high)
        axes.set_title(title)
        axes.set_xlabel("X (mm)")
        axes.set_ylabel("Y (mm)")
        axes.set_zlabel("Z (mm)")
        axes.set_zlim(low, high)
        axes.view_init(elev=32, azim=-62)
        axes.tick_params(labelsize=7)
    figure.tight_layout()
    output = io.BytesIO()
    figure.savefig(output, format="png", bbox_inches="tight", facecolor="white")
    plt.close(figure)
    return output.getvalue()


def render_dod_png(comparison: Comparison) -> bytes:
    """Plan-view DEM of difference with a symmetric colour scale."""
    _use_cjk_font()
    import matplotlib.pyplot as plt

    delta = np.where(comparison.valid, comparison.delta_mm, np.nan)
    limit = float(np.nanmax(np.abs(delta))) if np.isfinite(delta).any() else 1.0
    limit = max(limit, 1e-6)
    step = comparison.grid_step_mm
    x0 = comparison.x_mm[0, 0] - step / 2
    x1 = comparison.x_mm[0, -1] + step / 2
    y0 = comparison.y_mm[0, 0] - step / 2
    y1 = comparison.y_mm[-1, 0] + step / 2

    figure = plt.figure(figsize=(6.6, 4.8), dpi=150)
    axes = figure.add_subplot(1, 1, 1)
    image = axes.imshow(delta, origin="lower", extent=(x0, x1, y0, y1), cmap="RdBu",
                        vmin=-limit, vmax=limit, interpolation="nearest")
    axes.set_title("冲淤高程差图 (雨后 − 雨前)")
    axes.set_xlabel("X (mm)")
    axes.set_ylabel("Y (mm)")
    bar = figure.colorbar(image, ax=axes, fraction=0.046, pad=0.03)
    bar.set_label("ΔZ (mm)")
    figure.tight_layout()
    output = io.BytesIO()
    figure.savefig(output, format="png", bbox_inches="tight", facecolor="white")
    plt.close(figure)
    return output.getvalue()
