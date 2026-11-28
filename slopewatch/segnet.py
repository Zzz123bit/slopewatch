"""Compact five-class segmentation network shared by training and inference.

The same architecture is used by ``scripts/train_segmentation.py`` and by the
runtime ``TorchSegmenter`` so a fine-tuned checkpoint can be dropped in without
touching the API.  Pre/post-processing mirrors ``fusion.load_segmenter``.
"""

from __future__ import annotations

import threading
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

SIDE = 256
CLASSES = 5
CLASS_NAMES = ("other", "gully", "collapse_candidate", "protection", "unprotected_reference")


class UNet(nn.Module):
    def __init__(self, base: int = 24):
        super().__init__()

        def block(cin, cout):
            return nn.Sequential(
                nn.Conv2d(cin, cout, 3, padding=1), nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
                nn.Conv2d(cout, cout, 3, padding=1), nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
            )

        self.enc1 = block(3, base)
        self.enc2 = block(base, base * 2)
        self.enc3 = block(base * 2, base * 4)
        self.enc4 = block(base * 4, base * 8)
        self.pool = nn.MaxPool2d(2)
        self.up3 = nn.ConvTranspose2d(base * 8, base * 4, 2, stride=2)
        self.dec3 = block(base * 8, base * 4)
        self.up2 = nn.ConvTranspose2d(base * 4, base * 2, 2, stride=2)
        self.dec2 = block(base * 4, base * 2)
        self.up1 = nn.ConvTranspose2d(base * 2, base, 2, stride=2)
        self.dec1 = block(base * 2, base)
        self.out = nn.Conv2d(base, CLASSES, 1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))
        d3 = self.dec3(torch.cat([self.up3(e4), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))
        return self.out(d1)


def prepare_image(image_rgb: np.ndarray, side: int = SIDE):
    """Isotropic resize + black padding, matching the ONNX preprocessing."""
    height, width = image_rgb.shape[:2]
    scale = side / max(width, height)
    resized = cv2.resize(image_rgb, (max(1, round(width * scale)), max(1, round(height * scale))))
    canvas = np.zeros((side, side, 3), dtype=np.uint8)
    top = (side - resized.shape[0]) // 2
    left = (side - resized.shape[1]) // 2
    canvas[top:top + resized.shape[0], left:left + resized.shape[1]] = resized
    tensor = torch.from_numpy(canvas).permute(2, 0, 1).float() / 255.0
    return tensor.unsqueeze(0), (left, top, resized.shape[1], resized.shape[0]), (height, width)


def prepare_mask(mask: np.ndarray, side: int = SIDE):
    """Mask version of :func:`prepare_image`; padding uses 255 (ignore)."""
    height, width = mask.shape[:2]
    scale = side / max(width, height)
    resized = cv2.resize(mask, (max(1, round(width * scale)), max(1, round(height * scale))),
                         interpolation=cv2.INTER_NEAREST)
    canvas = np.full((side, side), 255, dtype=np.uint8)
    top = (side - resized.shape[0]) // 2
    left = (side - resized.shape[1]) // 2
    canvas[top:top + resized.shape[0], left:left + resized.shape[1]] = resized
    return torch.from_numpy(canvas).long()


class TorchSegmenter:
    """Callable with the same contract as the ONNX predictor in ``fusion``."""

    def __init__(self, checkpoint: Path, device: str = "cpu"):
        self.checkpoint = Path(checkpoint)
        self.device = torch.device(device)
        self.lock = threading.Lock()
        self.model = UNet()
        state = torch.load(str(self.checkpoint), map_location="cpu", weights_only=True)
        if isinstance(state, dict) and "model" in state:
            state = state["model"]
        self.model.load_state_dict(state)
        self.model.eval().to(self.device)

    def predict(self, image) -> np.ndarray:
        if isinstance(image, (str, Path)):
            frame = cv2.imread(str(image))
            if frame is None:
                raise ValueError(f"无法读取图片 {image}")
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        elif isinstance(image, np.ndarray):
            frame = image
        else:
            raise ValueError("分割输入需要路径或 RGB uint8 数组")
        if frame.dtype != np.uint8 or frame.ndim != 3 or frame.shape[2] != 3:
            raise ValueError("分割输入需要 uint8 RGB 图像")
        tensor, (left, top, width, height), (raw_h, raw_w) = prepare_image(frame)
        with self.lock, torch.inference_mode():
            logits = self.model(tensor.to(self.device))
            logits = logits[:, :, top:top + height, left:left + width]
            logits = F.interpolate(logits, size=(raw_h, raw_w), mode="bilinear", align_corners=False)
            prediction = logits.argmax(dim=1)[0].cpu().numpy().astype(np.uint8)
        return prediction

    def __call__(self, image):
        return self.predict(image)
