"""Train a compact U-Net for the five-class slope segmentation and export ONNX.

The preprocessing matches ``slopewatch.fusion.load_segmenter``: isotropic resize
to 256 px, black padding to a square, RGB / 255, logits shaped 1x5xHxW.
"""

from __future__ import annotations

import json
import random
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

SIDE = 256
CLASSES = 5


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


def preprocess(image_bgr: np.ndarray, mask: np.ndarray | None):
    height, width = image_bgr.shape[:2]
    scale = SIDE / max(width, height)
    w, h = max(1, round(width * scale)), max(1, round(height * scale))
    left, top = (SIDE - w) // 2, (SIDE - h) // 2
    canvas = np.zeros((SIDE, SIDE, 3), dtype=np.uint8)
    canvas[top:top + h, left:left + w] = cv2.resize(image_bgr, (w, h))
    image = torch.from_numpy(canvas[:, :, ::-1].copy()).permute(2, 0, 1).float() / 255.0
    if mask is None:
        return image, None
    label_canvas = np.full((SIDE, SIDE), 255, dtype=np.uint8)
    label_canvas[top:top + h, left:left + w] = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
    return image, torch.from_numpy(label_canvas).long()


class SlopeSegmentation(Dataset):
    def __init__(self, items, augment=False):
        self.items = items
        self.augment = augment

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        image_path, mask_path = self.items[index]
        image = cv2.imread(str(image_path))
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if self.augment and random.random() < 0.5:
            image = cv2.flip(image, 1)
            mask = cv2.flip(mask, 1)
        image_t, mask_t = preprocess(image, mask)
        return image_t, mask_t


def iou_per_class(predictions, targets):
    scores = []
    for class_id in range(CLASSES):
        pred = predictions == class_id
        true = targets == class_id
        union = (pred | true).sum().item()
        scores.append(float((pred & true).sum().item()) / union if union else float("nan"))
    return scores


def main():
    root = Path(__file__).resolve().parents[1]
    data = root / "data" / "segmentation"
    items = [(p, data / "masks" / p.name) for p in sorted((data / "images").glob("*.png"))]
    if len(items) < 50:
        raise SystemExit("请先运行 scripts/make_segmentation_dataset.py 生成数据")
    random.seed(7)
    random.shuffle(items)
    split = int(len(items) * 0.9)
    train_items, val_items = items[:split], items[split:]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        free, _ = torch.cuda.mem_get_info()
        if free < 1.5 * 1024**3:
            device = "cpu"
    print(f"device={device} train={len(train_items)} val={len(val_items)}", flush=True)

    model = UNet().to(device)
    weights = torch.tensor([0.35, 2.0, 2.5, 0.8, 0.8], device=device)
    criterion = nn.CrossEntropyLoss(weight=weights, ignore_index=255)
    optimiser = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    epochs = 36
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimiser, T_max=epochs)
    train_loader = DataLoader(SlopeSegmentation(train_items, augment=True), batch_size=8, shuffle=True, num_workers=0)
    val_loader = DataLoader(SlopeSegmentation(val_items), batch_size=8, num_workers=0)

    best = -1.0
    history = []
    for epoch in range(1, epochs + 1):
        model.train()
        total = 0.0
        for images, masks in train_loader:
            images, masks = images.to(device), masks.to(device)
            optimiser.zero_grad(set_to_none=True)
            loss = criterion(model(images), masks)
            loss.backward()
            optimiser.step()
            total += float(loss.item())
        scheduler.step()

        model.eval()
        ious, count = np.zeros(CLASSES), 0
        with torch.no_grad():
            for images, masks in val_loader:
                logits = model(images.to(device))
                prediction = logits.argmax(1).cpu()
                values = iou_per_class(prediction, masks)
                ious += np.nan_to_num(values, nan=0.0)
                count += 1
        ious /= max(count, 1)
        miou = float(np.nanmean(ious[[0, 1, 2, 3, 4]]))
        history.append({"epoch": epoch, "loss": total / len(train_loader), "miou": miou,
                        "iou": [round(float(v), 4) for v in ious]})
        print(f"epoch {epoch:02d} loss {total/len(train_loader):.3f} mIoU {miou:.3f} "
              f"iou {np.round(ious, 2).tolist()}", flush=True)
        if miou > best:
            best = miou
            (root / "models").mkdir(exist_ok=True)
            torch.save(model.state_dict(), root / "models" / "segmentation.pt")

    model.load_state_dict(torch.load(root / "models" / "segmentation.pt", map_location=device, weights_only=True))
    model.eval().cpu()
    dummy = torch.zeros(1, 3, SIDE, SIDE)
    onnx_path = root / "models" / "segmentation.onnx"
    torch.onnx.export(
        model, dummy, onnx_path, opset_version=17, dynamo=False,
        input_names=["input"], output_names=["logits"],
    )
    (root / "models" / "segmentation_metrics.json").write_text(
        json.dumps({"best_miou": best, "history": history, "side": SIDE}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"ONNX written to {onnx_path} ({onnx_path.stat().st_size/1e6:.1f} MB), best mIoU {best:.3f}")


if __name__ == "__main__":
    main()
