"""Normalise the downloaded Zenodo landslide datasets into images/masks pairs.

Sources (CC-BY-4.0):
- Nepal landslide dataset for semantic segmentation, DOI 10.5281/zenodo.3688363
- Italy landslide dataset for semantic segmentation, DOI 10.5281/zenodo.4287425
- Japan landslide dataset for semantic segmentation, DOI 10.5281/zenodo.3775870

Each dataset is a set of small satellite tiles with a binary landslide mask.
They are external reference data (landslide vs background), not the five-class
slope scheme used by this project.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import cv2
import numpy as np

SOURCES = {
    "nepal": {
        "doi": "10.5281/zenodo.3688363",
        "title": "Nepal landslide dataset for semantic segmentation",
        "pairs": [
            ("nepal/Nepal landslide dataset/Training/image", "nepal/Nepal landslide dataset/Training/label", "nepal_mask"),
            ("nepal/Nepal landslide dataset/Validation/images", "nepal/Nepal landslide dataset/Validation/masks", "nepal_mask"),
            ("nepal/Nepal landslide dataset/Test/image", "nepal/Nepal landslide dataset/Test/mask", "nepal_mask"),
        ],
    },
    "italy": {
        "doi": "10.5281/zenodo.4287425",
        "title": "Italy landslide dataset for semantic segmentation",
        "pairs": [("italy/image", "italy/masks", "prefix_masks")],
    },
    "japan": {
        "doi": "10.5281/zenodo.3775870",
        "title": "Japan landslide dataset for semantic segmentation",
        "pairs": [("japan/image", "japan/label", "prefix_masks")],
    },
}


def find_mask(image: Path, mask_dir: Path, rule: str) -> Path | None:
    stem = image.stem
    if rule == "prefix_masks":
        candidate = mask_dir / f"masks_{stem}.png"
        return candidate if candidate.is_file() else None
    # nepal: b_COMP_0_1.tiff -> COMP_0_1_mask.png
    base = stem[2:] if stem.startswith("b_") else stem
    candidate = mask_dir / f"{base}_mask.png"
    return candidate if candidate.is_file() else None


def main():
    parser = argparse.ArgumentParser(description="整理 Zenodo 滑坡数据集为 images/masks")
    parser.add_argument("--external", default="data/external")
    parser.add_argument("--out", default="data/external/prepared")
    args = parser.parse_args()
    external = Path(args.external)
    out_root = Path(args.out)
    manifest = []
    for name, meta in SOURCES.items():
        images_dir = out_root / name / "images"
        masks_dir = out_root / name / "masks"
        images_dir.mkdir(parents=True, exist_ok=True)
        masks_dir.mkdir(parents=True, exist_ok=True)
        index = 0
        for image_rel, mask_rel, rule in meta["pairs"]:
            image_path = external / image_rel
            mask_dir = external / mask_rel
            if not image_path.is_dir():
                continue
            for image in sorted(image_path.iterdir()):
                if image.suffix.lower() not in {".tif", ".tiff", ".png", ".jpg"}:
                    continue
                mask_path = find_mask(image, mask_dir, rule)
                if mask_path is None:
                    continue
                frame = cv2.imread(str(image), cv2.IMREAD_COLOR)
                mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
                if frame is None or mask is None:
                    continue
                if mask.shape != frame.shape[:2]:
                    mask = cv2.resize(mask, (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_NEAREST)
                binary = (mask > 127).astype(np.uint8)
                if binary.sum() == 0:
                    continue
                index += 1
                cv2.imwrite(str(images_dir / f"{index:04d}.png"), frame)
                cv2.imwrite(str(masks_dir / f"{index:04d}.png"), binary)
        manifest.append({
            "name": name,
            "title": meta["title"],
            "doi": meta["doi"],
            "license": "CC-BY-4.0",
            "pairs": index,
            "positive_pixel_fraction": None,
        })
        print(f"{name}: {index} pairs")
    (out_root / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print("manifest ->", out_root / "manifest.json")


if __name__ == "__main__":
    main()
