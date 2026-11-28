"""Pre-annotate a folder of slope photos with the current segmentation model.

The output masks are ordinary class-ID PNGs; open them in ``annotate.py`` (or any
editor) and fix the mistakes instead of painting from scratch.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from slopewatch.fusion import load_segmenter

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def main():
    parser = argparse.ArgumentParser(description="用现有模型预标注照片")
    parser.add_argument("--images", required=True, help="原始照片目录")
    parser.add_argument("--masks", required=True, help="输出掩膜目录")
    parser.add_argument("--overwrite", action="store_true", help="覆盖已有掩膜")
    args = parser.parse_args()

    source = Path(args.images)
    if not source.is_dir():
        raise SystemExit(
            f"找不到照片目录：{source}\n"
            "请先新建目录并把照片复制进去，例如：\n"
            "  mkdir data\\real_photos\n"
            "  copy 你的照片 data\\real_photos\n"
            "然后再运行本命令。"
        )
    images = sorted(p for p in source.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES)
    if not images:
        raise SystemExit(f"{args.images} 中没有图片")
    masks_dir = Path(args.masks)
    masks_dir.mkdir(parents=True, exist_ok=True)

    segmenter = load_segmenter()
    if segmenter is None:
        raise SystemExit("没有可用的分割模型，请先运行 scripts/train_segmentation.py")

    written = 0
    for index, path in enumerate(images, 1):
        target = masks_dir / f"{path.stem}.png"
        if target.exists() and not args.overwrite:
            continue
        mask = segmenter(path)
        cv2.imwrite(str(target), mask)
        written += 1
        print(f"[{index}/{len(images)}] {path.name} -> {target.name}", flush=True)
    print(f"完成：新写入 {written} 张预标注掩膜，目录 {masks_dir}")


if __name__ == "__main__":
    main()
