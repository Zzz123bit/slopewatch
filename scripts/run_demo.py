"""One-command demo runner for the SlopeWatch competition demo.

Modes
-----
quick : synthetic two-epoch slope, no GPU, ~2 seconds.  Produces the full
        report bundle (metrics, PDF, 3D surface, DoD map, semantic atlas).
photo : real pipeline on a two-epoch photo scene (needs VGGT weights and a
        GPU or CPU); produces the same bundle plus weak labels.

Usage
-----
    python scripts/run_demo.py --mode quick
    python scripts/run_demo.py --mode photo --scene data/samples/synthetic_sem
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from slopewatch.core import comparison_csv, comparison_json, make_demo_comparison
from slopewatch.fusion import analyze_fusion, atlas_png
from slopewatch.report import build_report_pdf
from slopewatch.visuals import render_dod_png, render_surface_png

ROOT = Path(__file__).resolve().parents[1]


def _write_bundle(comparison, out: Path, metadata: dict, *, title: str) -> dict:
    out.mkdir(parents=True, exist_ok=True)
    fusion = analyze_fusion(comparison)
    (out / "metrics.json").write_bytes(comparison_json(comparison, metadata))
    (out / "change.csv").write_bytes(comparison_csv(comparison))
    (out / "report.pdf").write_bytes(build_report_pdf(comparison.metrics, comparison.gullies, metadata))
    (out / "surface.png").write_bytes(render_surface_png(comparison))
    (out / "dod.png").write_bytes(render_dod_png(comparison))
    (out / "atlas.png").write_bytes(atlas_png(comparison, fusion))
    (out / "fusion.json").write_text(
        json.dumps({"summary": fusion.summary}, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    summary = {
        "title": title,
        "metrics": comparison.metrics,
        "regions": len(comparison.gullies),
        "files": sorted(p.name for p in out.iterdir() if p.is_file()),
    }
    (out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def run_quick(out: Path) -> dict:
    started = time.time()
    comparison = make_demo_comparison()
    summary = _write_bundle(
        comparison,
        out,
        {"source": "synthetic_demo", "说明": "程序生成的降雨前后坡面，用于离线演示", "soil": "示例土",
         "rainfall_mm_h": 60},
        title="合成样例演示",
    )
    summary["seconds"] = round(time.time() - started, 2)
    (out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def _zip_images(images: list[Path], target: Path):
    with zipfile.ZipFile(target, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for image in images:
            archive.write(image, image.name)


def run_photo(scene: Path, out: Path, max_views: int = 4, scale: float = 1000.0) -> dict:
    from slopewatch import jobs

    before_dir, after_dir = scene / "before", scene / "after"
    if not before_dir.is_dir() or not after_dir.is_dir():
        raise SystemExit(f"{scene} 需要 before/ 与 after/ 子目录")
    before_images = sorted(p for p in before_dir.iterdir() if p.suffix.lower() in {".png", ".jpg", ".jpeg"})
    after_images = sorted(p for p in after_dir.iterdir() if p.suffix.lower() in {".png", ".jpg", ".jpeg"})
    if len(before_images) < 2 or len(after_images) < 2:
        raise SystemExit("每个时相至少 2 张照片")

    stage = ROOT / "demo_output" / "_staging"
    stage.mkdir(parents=True, exist_ok=True)
    before_zip, after_zip = stage / "before.zip", stage / "after.zip"
    _zip_images(before_images, before_zip)
    _zip_images(after_images, after_zip)

    job_id, directory = jobs.create_job("photo_comparison", {"demo": True, "scene": str(scene)})
    (directory / "before.zip").write_bytes(before_zip.read_bytes())
    (directory / "after.zip").write_bytes(after_zip.read_bytes())
    started = time.time()
    jobs.run_photo_job(
        job_id, "before.zip", "after.zip",
        grid_step_mm=1.5, erosion_threshold_mm=3.0, elapsed_hours=1.0,
        scale_mm_per_unit=scale, max_views=max_views, metadata={"source": "demo", "scene": scene.name},
    )
    record = jobs.read_job(job_id)
    if record["status"] != "complete":
        raise SystemExit(f"演示任务失败：{record.get('error')}")

    out.mkdir(parents=True, exist_ok=True)
    for name in ("metrics.json", "change.csv", "report.pdf", "surface.png", "dod.png",
                 "atlas.png", "fusion.json", "weak_labels.png", "weak_labels.json", "diagnostics.json"):
        source = directory / name
        if source.is_file():
            shutil.copy2(source, out / name)
    summary = {
        "title": "照片双时相对比演示",
        "job_id": job_id,
        "scene": str(scene),
        "metrics": record["result"]["metrics"],
        "semantic": record["result"].get("semantic", {}),
        "weak_labels": record["result"].get("weak_labels", {}),
        "seconds": round(time.time() - started, 1),
    }
    (out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def main():
    parser = argparse.ArgumentParser(description="坡镜智安一键演示")
    parser.add_argument("--mode", choices=["quick", "photo"], default="quick")
    parser.add_argument("--scene", default="data/samples/synthetic_sem")
    parser.add_argument("--out", default="demo_output")
    parser.add_argument("--max-views", type=int, default=4)
    parser.add_argument("--scale", type=float, default=1000.0)
    args = parser.parse_args()

    out = ROOT / args.out / args.mode
    if args.mode == "quick":
        summary = run_quick(out)
    else:
        summary = run_photo(ROOT / args.scene, out, args.max_views, args.scale)

    metrics = summary["metrics"]
    print("\n=== 演示完成 ===")
    print(f"模式：{args.mode}")
    print(f"冲刷体积：{metrics['erosion_volume_l']:.4f} L")
    print(f"堆积体积：{metrics['deposition_volume_l']:.4f} L")
    print(f"最大冲刷深度：{metrics['max_erosion_depth_mm']:.2f} mm")
    print(f"冲沟候选：{metrics['gully_count_est']} 个")
    if summary.get("seconds"):
        print(f"耗时：{summary['seconds']} 秒")
    print(f"产物目录：{out}")
    print("打开：report.pdf / surface.png / dod.png / atlas.png")


if __name__ == "__main__":
    main()
