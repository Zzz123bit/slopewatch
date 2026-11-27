from __future__ import annotations

import io
import shutil
import zipfile
from pathlib import Path


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".webp"}
MAX_IMAGE_COUNT = 300
MAX_IMAGE_BYTES = 50 * 1024 * 1024
MAX_TOTAL_IMAGE_BYTES = 1024 * 1024 * 1024


def extract_image_archive(archive: Path | bytes, images_dir: Path) -> list[Path]:
    """Extract a bounded number of image files without retaining the ZIP in memory."""
    source = io.BytesIO(archive) if isinstance(archive, bytes) else archive
    images_dir.mkdir(parents=True, exist_ok=True)
    extracted = []
    total_bytes = 0

    with zipfile.ZipFile(source) as zipped:
        for member in zipped.infolist():
            if member.is_dir():
                continue
            member_path = Path(member.filename)
            if member_path.is_absolute() or ".." in member_path.parts:
                raise ValueError("ZIP 中包含不安全路径")
            if member_path.suffix.lower() not in IMAGE_SUFFIXES:
                continue
            if member.file_size > MAX_IMAGE_BYTES:
                raise ValueError(f"单张图片不能超过 {MAX_IMAGE_BYTES // (1024 * 1024)} MB")
            total_bytes += member.file_size
            if total_bytes > MAX_TOTAL_IMAGE_BYTES:
                raise ValueError("ZIP 解压后的图片总量不能超过 1 GB")
            if len(extracted) >= MAX_IMAGE_COUNT:
                raise ValueError(f"每个时相最多处理 {MAX_IMAGE_COUNT} 张图片")

            suffix = member_path.suffix.lower()
            destination = images_dir / f"{len(extracted) + 1:04d}{suffix}"
            with zipped.open(member) as image_stream, destination.open("wb") as output_stream:
                shutil.copyfileobj(image_stream, output_stream, length=1024 * 1024)
            extracted.append(destination)

    if len(extracted) < 8:
        raise ValueError("照片重建至少需要 8 张图片，建议每个时相拍摄 25–40 张")
    return extracted

