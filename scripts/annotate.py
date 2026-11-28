"""Tiny OpenCV brush tool for five-class slope masks.

Run:
    python scripts/annotate.py --images data/real_photos --masks data/real_masks

Controls
--------
0-4     选择类别（其他/冲沟/坍塌/防护/对照）
5       橡皮（未知 255）
[ ]     减小/增大笔刷
s       保存当前掩膜
a / d   上一张 / 下一张（自动保存）
z       撤销上一笔
q       退出
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
PALETTE = np.array([
    (222, 223, 211),   # other
    (36, 116, 167),    # gully
    (177, 81, 36),     # collapse
    (31, 133, 95),     # protection
    (137, 111, 171),   # reference
], dtype=np.uint8)
NAMES = ("other", "gully", "collapse", "protection", "reference", "erase")
MAX_SIDE = 1100


class Annotator:
    def __init__(self, images, masks_dir: Path, start: int = 0, alpha: float = 0.45):
        self.images = images
        self.masks_dir = masks_dir
        self.index = start
        self.alpha = alpha
        self.class_id = 1
        self.brush = 12
        self.dirty = False
        self.history: list[np.ndarray] = []
        self.image = None
        self.mask = None
        self.display = None
        self.scale = 1.0
        self.window = "SlopeWatch annotate"

    # ---------- data ----------
    def load(self):
        path = self.images[self.index]
        self.image = cv2.imread(str(path))
        if self.image is None:
            raise ValueError(f"无法读取 {path}")
        mask_path = self.masks_dir / f"{path.stem}.png"
        if mask_path.is_file():
            self.mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
            if self.mask.shape != self.image.shape[:2]:
                raise ValueError(f"{mask_path.name} 尺寸与照片不一致")
        else:
            self.mask = np.zeros(self.image.shape[:2], dtype=np.uint8)
        self.dirty = False
        self.history.clear()
        height, width = self.image.shape[:2]
        self.scale = min(1.0, MAX_SIDE / max(height, width))
        self.refresh()

    def save(self):
        if not self.dirty:
            return
        target = self.masks_dir / f"{self.images[self.index].stem}.png"
        cv2.imwrite(str(target), self.mask)
        self.dirty = False
        print(f"已保存 {target.name}")

    # ---------- rendering ----------
    def refresh(self):
        colours = PALETTE[np.clip(self.mask, 0, 4)]
        overlay = cv2.addWeighted(self.image, 1 - self.alpha, colours, self.alpha, 0)
        unknown = self.mask == 255
        overlay[unknown] = cv2.addWeighted(self.image[unknown], 0.7,
                                           np.full_like(self.image[unknown], 40), 0.3, 0)
        canvas = overlay.copy()
        for class_id in range(5):
            x = 12 + class_id * 150
            cv2.rectangle(canvas, (x, 10), (x + 18, 28), PALETTE[class_id][::-1].tolist(), -1)
            cv2.putText(canvas, NAMES[class_id], (x + 24, 25), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (255, 255, 255), 1, cv2.LINE_AA)
        title = f"{self.images[self.index].name}  [{self.index+1}/{len(self.images)}]  class={NAMES[self.class_id]}  brush={self.brush}"
        cv2.putText(canvas, title, (12, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
        if self.scale < 1.0:
            canvas = cv2.resize(canvas, None, fx=self.scale, fy=self.scale)
        self.display = canvas
        cv2.imshow(self.window, self.display)

    # ---------- editing ----------
    def paint(self, x: int, y: int):
        if self.image is None:
            return
        height, width = self.mask.shape
        x, y = int(round(x)), int(round(y))
        if not (0 <= x < width and 0 <= y < height):
            return
        value = 255 if self.class_id == 5 else self.class_id
        cv2.circle(self.mask, (x, y), self.brush, value, -1)
        self.dirty = True

    def on_mouse(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.history.append(self.mask.copy())
            if len(self.history) > 30:
                self.history.pop(0)
            self.paint(x / self.scale, y / self.scale)
            self.refresh()
        elif event == cv2.EVENT_MOUSEMOVE and flags & cv2.EVENT_FLAG_LBUTTON:
            self.paint(x / self.scale, y / self.scale)
            self.refresh()

    # ---------- loop ----------
    def go(self, delta: int) -> bool:
        self.save()
        target = self.index + delta
        if 0 <= target < len(self.images):
            self.index = target
            self.load()
            return True
        return False

    def run(self):
        cv2.namedWindow(self.window, cv2.WINDOW_AUTOSIZE)
        cv2.setMouseCallback(self.window, self.on_mouse)
        self.load()
        while True:
            key = cv2.waitKey(20) & 0xFF
            if key == ord("q"):
                self.save()
                break
            if key == ord("s"):
                self.save()
            elif key == ord("z") and self.history:
                self.mask = self.history.pop()
                self.dirty = True
                self.refresh()
            elif key in (ord("d"), 255):
                if key == ord("d"):
                    self.go(1)
            elif key == ord("a"):
                self.go(-1)
            elif key == ord("["):
                self.brush = max(2, self.brush - 2)
                self.refresh()
            elif key == ord("]"):
                self.brush = min(80, self.brush + 2)
                self.refresh()
            elif ord("0") <= key <= ord("5"):
                self.class_id = key - ord("0")
                self.refresh()
        cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(description="五类坡面掩膜标注工具")
    parser.add_argument("--images", required=True)
    parser.add_argument("--masks", required=True)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--dry-run", action="store_true", help="只检查目录，不打开窗口")
    args = parser.parse_args()

    images = sorted(p for p in Path(args.images).iterdir() if p.suffix.lower() in IMAGE_SUFFIXES)
    if not images:
        raise SystemExit(f"{args.images} 中没有图片")
    masks_dir = Path(args.masks)
    masks_dir.mkdir(parents=True, exist_ok=True)
    print(f"共 {len(images)} 张图片，掩膜目录 {masks_dir}")
    print("类别：0 其他 / 1 冲沟 / 2 坍塌 / 3 防护 / 4 对照 / 5 橡皮")
    print("快捷键：[ ] 笔刷  s 保存  a/d 上一张/下一张  z 撤销  q 退出")
    if args.dry_run:
        for path in images[:5]:
            print("dry-run:", path.name)
        return
    Annotator(images, masks_dir, start=args.start).run()


if __name__ == "__main__":
    main()
