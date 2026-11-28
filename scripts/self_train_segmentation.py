"""Self-supervised labelling: turn a two-epoch photo pair into training masks.

Pipeline:
    before/after photos -> VGGT reconstruction -> DoD -> geomorphological rules
    -> per-pixel labels on each after view -> ready-to-train image/mask folder

Then fine-tune with:
    python scripts/finetune_segmentation.py --images OUT/images --masks OUT/masks --epochs 15
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from slopewatch.photogram import compare_reconstructions, transform_points_to_comparison
from slopewatch.vggt_recon import reconstruct_vggt
from slopewatch.weak_labels import label_change, labels_to_rgb

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def list_images(folder: Path):
    return sorted(p for p in folder.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES)


def project_grid_to_views(comparison, labels, cloud, diagnostics):
    """For every after view, look up the rule label of each reconstructed pixel."""
    step = comparison.grid_step_mm
    rows_count, cols_count = labels.shape
    x0 = comparison.x_mm[0, 0] - step / 2
    y0 = comparison.y_mm[0, 0] - step / 2
    outputs = []
    threshold = float(cloud.stats["confidence_threshold"])
    for points, confidence, mapping, path in zip(
        cloud.view_points, cloud.view_confidence, cloud.view_mappings, cloud.image_paths
    ):
        flat = points.reshape(-1, 3)
        metric = transform_points_to_comparison(flat, diagnostics)
        cols = np.floor((metric[:, 0] - x0) / step).astype(np.int64, copy=False)
        rows = np.floor((metric[:, 1] - y0) / step).astype(np.int64, copy=False)
        valid = (np.isfinite(metric).all(axis=1) & (confidence.reshape(-1) >= threshold)
                 & (cols >= 0) & (cols < cols_count) & (rows >= 0) & (rows < rows_count))
        mask = np.full(len(flat), 255, dtype=np.uint8)
        mask[valid] = labels[rows[valid], cols[valid]]
        outputs.append((path, mask.reshape(points.shape[:2])))
    return outputs


def main():
    parser = argparse.ArgumentParser(description="由两期照片生成自监督（规则）标签")
    parser.add_argument("--scene", required=True, help="包含 before/ 与 after/ 的场景目录")
    parser.add_argument("--out", required=True, help="输出目录（images/、masks/）")
    parser.add_argument("--grid-step", type=float, default=1.5)
    parser.add_argument("--threshold", type=float, default=3.0)
    parser.add_argument("--scale", type=float, default=1000.0, help="mm/单位")
    parser.add_argument("--max-views", type=int, default=4)
    parser.add_argument("--finetune-epochs", type=int, default=0)
    args = parser.parse_args()

    scene = Path(args.scene)
    before_dir, after_dir = scene / "before", scene / "after"
    if not before_dir.is_dir() or not after_dir.is_dir():
        raise SystemExit(f"{scene} 需要 before/ 与 after/ 两个子目录")
    before_images, after_images = list_images(before_dir), list_images(after_dir)
    if len(before_images) < 2 or len(after_images) < 2:
        raise SystemExit("每个时相至少 2 张照片")

    print(f"雨前 {len(before_images)} 张，雨后 {len(after_images)} 张", flush=True)
    before_cloud = reconstruct_vggt(before_images, max_views=args.max_views)
    after_cloud = reconstruct_vggt(after_images, max_views=args.max_views)
    comparison, diagnostics = compare_reconstructions(
        before_cloud.points, after_cloud.points,
        grid_step_mm=args.grid_step, threshold_mm=args.threshold, scale_mm_per_unit=args.scale,
    )
    weak = label_change(comparison)
    print("规则标签统计:", json.dumps(weak.stats, ensure_ascii=False)[:300], flush=True)

    out = Path(args.out)
    images_out, masks_out = out / "images", out / "masks"
    images_out.mkdir(parents=True, exist_ok=True)
    masks_out.mkdir(parents=True, exist_ok=True)
    views = project_grid_to_views(comparison, weak.labels, after_cloud, diagnostics)
    kept, unknown_fraction = 0, []
    for index, (path, mask) in enumerate(views):
        if (mask != 255).mean() < 0.05:
            continue
        frame = cv2.imread(str(path))
        if frame is None:
            continue
        kept += 1
        cv2.imwrite(str(images_out / f"{index:02d}.png"), frame)
        cv2.imwrite(str(masks_out / f"{index:02d}.png"), mask)
        unknown_fraction.append(float((mask == 255).mean()))
        print(f"  视图 {index}: {path.name} 未标注比例 {unknown_fraction[-1]:.1%}", flush=True)

    cv2.imwrite(str(out / "weak_grid.png"), labels_to_rgb(weak.labels)[:, :, ::-1])
    summary = {
        "scene": str(scene),
        "views_kept": kept,
        "unknown_fraction": round(float(np.mean(unknown_fraction)), 3) if unknown_fraction else None,
        "weak_stats": weak.stats,
        "scale_mm_per_unit": args.scale,
    }
    (out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"生成 {kept} 张自监督训练图 -> {out}", flush=True)

    if args.finetune_epochs > 0 and kept >= 4:
        import subprocess

        model = out / "segmentation_self.pt"
        command = [sys.executable, "scripts/finetune_segmentation.py",
                   "--images", str(images_out), "--masks", str(masks_out),
                   "--epochs", str(args.finetune_epochs), "--out", str(model)]
        print("开始微调:", " ".join(command), flush=True)
        subprocess.run(command, check=False)
        print(f"自监督模型: {model}")


if __name__ == "__main__":
    main()
