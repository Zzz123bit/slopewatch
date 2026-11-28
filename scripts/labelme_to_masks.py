"""Convert LabelMe polygon annotations into class-ID PNG masks.

Usage:
    python scripts/labelme_to_masks.py --json DIR_WITH_JSON --masks OUT_DIR
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

LABELS = {
    "other": 0, "其他": 0, "背景": 0,
    "gully": 1, "冲沟": 1, "细沟": 1,
    "collapse": 2, "collapse_candidate": 2, "坍塌": 2, "崩塌": 2,
    "protection": 3, "防护": 3, "防护垫": 3, "防护区": 3,
    "reference": 4, "unprotected_reference": 4, "对照": 4, "对照区": 4, "裸土对照": 4,
}


def convert(json_path: Path, out_dir: Path) -> Path:
    payload = json.loads(json_path.read_text(encoding="utf-8-sig"))
    height = int(payload["imageHeight"])
    width = int(payload["imageWidth"])
    mask = np.full((height, width), 255, dtype=np.uint8)
    for shape in payload.get("shapes", []):
        label = str(shape.get("label", "")).strip()
        if label not in LABELS:
            raise ValueError(f"{json_path.name}: 未知类别 {label!r}，请使用 {sorted(LABELS)}")
        points = np.asarray(shape.get("points", []), dtype=np.float32)
        if len(points) < 3:
            continue
        polygon = np.round(points).astype(np.int32)
        cv2.fillPoly(mask, [polygon], int(LABELS[label]))
    out_dir.mkdir(parents=True, exist_ok=True)
    target = out_dir / f"{json_path.stem}.png"
    cv2.imwrite(str(target), mask)
    return target


def main():
    parser = argparse.ArgumentParser(description="LabelMe JSON -> 类别 ID 掩膜")
    parser.add_argument("--json", required=True, help="包含 *.json 的目录")
    parser.add_argument("--masks", required=True, help="输出掩膜目录")
    args = parser.parse_args()
    files = sorted(Path(args.json).glob("*.json"))
    if not files:
        raise SystemExit("没有找到 LabelMe JSON")
    for path in files:
        target = convert(path, Path(args.masks))
        print(f"{path.name} -> {target.name}")


if __name__ == "__main__":
    main()

