"""Fine-tune the five-class segmenter on real annotated slope photos.

Example:
    python scripts/finetune_segmentation.py --images data/real_photos --masks data/real_masks --epochs 20

The best checkpoint overwrites ``models/segmentation.pt``; the runtime picks it
up automatically (ONNX is used only when present).  An ONNX export is attempted
when the ``onnx`` package is importable.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import cv2
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from slopewatch.segnet import CLASSES, UNet, prepare_image, prepare_mask

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def augment(image: np.ndarray, mask: np.ndarray, rng: random.Random):
    if rng.random() < 0.5:
        image = cv2.flip(image, 1)
        mask = cv2.flip(mask, 1)
    if rng.random() < 0.7:
        factor = rng.uniform(0.8, 1.25)
        image = np.clip(image.astype(np.float32) * factor, 0, 255).astype(np.uint8)
    if rng.random() < 0.5:
        shift = rng.uniform(-18, 18)
        image = np.clip(image.astype(np.float32) + shift, 0, 255).astype(np.uint8)
    if rng.random() < 0.6:
        angle = rng.uniform(-8, 8)
        height, width = image.shape[:2]
        matrix = cv2.getRotationMatrix2D((width / 2, height / 2), angle, rng.uniform(0.92, 1.08))
        image = cv2.warpAffine(image, matrix, (width, height), flags=cv2.INTER_LINEAR,
                               borderMode=cv2.BORDER_REFLECT)
        mask = cv2.warpAffine(mask, matrix, (width, height), flags=cv2.INTER_NEAREST,
                             borderValue=255)
    return image, mask


class MaskDataset(Dataset):
    def __init__(self, pairs, train: bool, seed: int = 0):
        self.pairs = pairs
        self.train = train
        self.rng = random.Random(seed)

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, index):
        image_path, mask_path = self.pairs[index]
        image = cv2.cvtColor(cv2.imread(str(image_path)), cv2.COLOR_BGR2RGB)
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if self.train:
            image, mask = augment(image, mask, self.rng)
        image_tensor, _, _ = prepare_image(image)
        mask_tensor = prepare_mask(mask)
        return image_tensor[0], mask_tensor


def iou_per_class(prediction, target):
    scores = []
    for class_id in range(CLASSES):
        predicted = prediction == class_id
        truth = (target == class_id)
        union = (predicted | truth).sum().item()
        scores.append(float((predicted & truth).sum().item()) / union if union else float("nan"))
    return scores


def main():
    parser = argparse.ArgumentParser(description="在真实标注照片上微调分割模型")
    parser.add_argument("--images", required=True)
    parser.add_argument("--masks", required=True)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--val-fraction", type=float, default=0.15)
    parser.add_argument("--init", default="models/segmentation.pt")
    parser.add_argument("--out", default="models/segmentation.pt")
    args = parser.parse_args()

    images = {p.stem: p for p in Path(args.images).iterdir() if p.suffix.lower() in IMAGE_SUFFIXES}
    masks = {p.stem: p for p in Path(args.masks).glob("*.png")}
    pairs = [(images[stem], masks[stem]) for stem in sorted(set(images) & set(masks))]
    if len(pairs) < 4:
        raise SystemExit("至少需要 4 张“照片 + 同名 PNG 掩膜”")
    random.Random(7).shuffle(pairs)
    split = max(1, int(len(pairs) * args.val_fraction))
    val_pairs, train_pairs = pairs[:split], pairs[split:]
    print(f"训练 {len(train_pairs)} 张，验证 {len(val_pairs)} 张")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda" and torch.cuda.mem_get_info()[0] < 1.5 * 1024**3:
        device = "cpu"
    print("device:", device)

    model = UNet()
    init_path = Path(args.init)
    if init_path.is_file():
        state = torch.load(str(init_path), map_location="cpu", weights_only=True)
        model.load_state_dict(state)
        print(f"从 {init_path} 初始化")
    model.to(device)

    counts = np.zeros(CLASSES, dtype=np.float64)
    for _, mask_path in train_pairs:
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        counts += np.bincount(mask.reshape(-1), minlength=256)[:CLASSES]
    frequency = counts / max(counts.sum(), 1)
    weights = 1.0 / np.sqrt(frequency + 1e-3)
    weights = np.clip(weights / weights.mean(), 0.25, 5.0).astype(np.float32)
    print("类别权重:", np.round(weights, 2).tolist())
    criterion = nn.CrossEntropyLoss(weight=torch.tensor(weights, device=device), ignore_index=255)
    optimiser = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimiser, T_max=max(1, args.epochs))

    train_loader = DataLoader(MaskDataset(train_pairs, True), batch_size=args.batch, shuffle=True)
    val_loader = DataLoader(MaskDataset(val_pairs, False), batch_size=args.batch)

    best = -1.0
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    history = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        total = 0.0
        for images_t, masks_t in train_loader:
            images_t, masks_t = images_t.to(device), masks_t.to(device)
            optimiser.zero_grad(set_to_none=True)
            loss = criterion(model(images_t), masks_t)
            loss.backward()
            optimiser.step()
            total += float(loss.item())
        scheduler.step()

        model.eval()
        ious, seen = np.zeros(CLASSES), 0
        with torch.inference_mode():
            for images_t, masks_t in val_loader:
                prediction = model(images_t.to(device)).argmax(1).cpu()
                ious += np.nan_to_num(iou_per_class(prediction, masks_t), nan=0.0)
                seen += 1
        ious /= max(seen, 1)
        miou = float(np.mean(ious))
        history.append({"epoch": epoch, "loss": total / len(train_loader), "miou": miou,
                        "iou": [round(float(v), 4) for v in ious]})
        print(f"epoch {epoch:02d} loss {total/len(train_loader):.3f} mIoU {miou:.3f} "
              f"iou {np.round(ious, 2).tolist()}", flush=True)
        if miou > best:
            best = miou
            torch.save(model.state_dict(), out_path)

    print(f"最佳 mIoU {best:.3f}，checkpoint 已保存到 {out_path}")

    onnx_path = out_path.with_suffix(".onnx")
    try:
        import onnx  # noqa: F401
        model.load_state_dict(torch.load(str(out_path), map_location="cpu", weights_only=True))
        torch.onnx.export(model.eval().cpu(), torch.zeros(1, 3, 256, 256), onnx_path,
                          opset_version=17, dynamo=False, input_names=["input"], output_names=["logits"])
        print(f"ONNX 已导出 {onnx_path}（运行时会优先使用它）")
    except Exception as error:  # noqa: BLE001 - export is optional
        print(f"跳过 ONNX 导出（{type(error).__name__}），运行时会自动使用 {out_path.name}")

    (out_path.parent / "segmentation_finetune_metrics.json").write_text(
        json.dumps({"samples": len(pairs), "best_miou": best, "history": history}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
