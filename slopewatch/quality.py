from __future__ import annotations

import io
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageOps


def inspect_image(data: bytes | str | Path, filename: str) -> dict:
    try:
        image_source = io.BytesIO(data) if isinstance(data, bytes) else data
        image = Image.open(image_source)
    except Exception as error:
        raise ValueError(f"无法读取图片 {filename}") from error

    width, height = image.size
    if width < 64 or height < 64:
        raise ValueError(f"图片分辨率过小 {filename}")
    if width * height > 80_000_000:
        raise ValueError(f"图片像素数超过 8000 万，无法安全检查 {filename}")
    image = ImageOps.exif_transpose(image).convert("RGB")

    scale = min(1.0, 1200.0 / max(width, height))
    small = image.resize((max(1, int(width * scale)), max(1, int(height * scale))))
    rgb = np.asarray(small)
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    sharpness = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    brightness = float(gray.mean())
    clipped_dark = float(np.mean(gray <= 5))
    clipped_bright = float(np.mean(gray >= 250))

    warnings = []
    if sharpness < 35:
        warnings.append("可能模糊")
    if brightness < 45:
        warnings.append("偏暗")
    if brightness > 220:
        warnings.append("偏亮")
    if clipped_dark + clipped_bright > 0.35:
        warnings.append("高光或暗部细节可能丢失")

    marker_count = None
    aruco = getattr(cv2, "aruco", None)
    if aruco is not None and hasattr(aruco, "DICT_4X4_50"):
        dictionary = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)
        detector = aruco.ArucoDetector(dictionary)
        corners, ids, _ = detector.detectMarkers(gray)
        marker_count = 0 if ids is None else len(ids)

    return {
        "filename": Path(filename).name,
        "width": width,
        "height": height,
        "sharpness_laplacian": round(sharpness, 2),
        "brightness_mean": round(brightness, 2),
        "clipped_dark_fraction": round(clipped_dark, 4),
        "clipped_bright_fraction": round(clipped_bright, 4),
        "aruco_marker_count": marker_count,
        "warnings": warnings,
    }


def inspect_images(images: list[tuple[str, bytes | str | Path]]) -> dict:
    if not images:
        raise ValueError("请至少选择一张图片")
    reports = [inspect_image(data, name) for name, data in images]
    return {
        "image_count": len(reports),
        "warning_count": sum(bool(item["warnings"]) for item in reports),
        "images": reports,
    }

