from __future__ import annotations

import json
import os
import uuid
from datetime import datetime, timezone
from pathlib import Path

from .core import (
    comparison_csv,
    comparison_json,
    compare_point_clouds,
    load_point_cloud,
    make_demo_comparison,
)
from .photogram import compare_reconstructions
from .quality import inspect_images
from .reconstruction import reconstruct_with_colmap
from .report import build_report_pdf
from .uploads import extract_image_archive
from .vggt_recon import reconstruct_vggt
from .visuals import render_dod_png, render_surface_png, surface_payload


PROJECT_ROOT = Path(__file__).resolve().parents[1]
JOBS_ROOT = PROJECT_ROOT / "data" / "jobs"
JOBS_ROOT.mkdir(parents=True, exist_ok=True)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def job_dir(job_id: str) -> Path:
    if not job_id or any(character not in "0123456789abcdef-" for character in job_id.lower()):
        raise ValueError("无效的 job_id")
    path = JOBS_ROOT / job_id
    if not path.is_dir():
        raise FileNotFoundError("任务不存在")
    return path


def save_job(job_id: str, payload: dict):
    directory = JOBS_ROOT / job_id
    directory.mkdir(parents=True, exist_ok=True)
    temporary = directory / "job.json.tmp"
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(temporary, directory / "job.json")


def read_job(job_id: str) -> dict:
    path = job_dir(job_id) / "job.json"
    return json.loads(path.read_text(encoding="utf-8"))


def create_job(kind: str, metadata: dict | None = None) -> tuple[str, Path]:
    job_id = str(uuid.uuid4())
    directory = JOBS_ROOT / job_id
    directory.mkdir(parents=True, exist_ok=False)
    save_job(
        job_id,
        {
            "job_id": job_id,
            "kind": kind,
            "status": "queued",
            "created_at": utc_now(),
            "updated_at": utc_now(),
            "metadata": metadata or {},
            "progress": 0,
        },
    )
    return job_id, directory


def update_job(job_id: str, **updates):
    job = read_job(job_id)
    job.update(updates)
    job["updated_at"] = utc_now()
    save_job(job_id, job)


def _complete_comparison(job_id: str, comparison, metadata: dict):
    directory = job_dir(job_id)
    (directory / "metrics.json").write_bytes(comparison_json(comparison, metadata))
    (directory / "change.csv").write_bytes(comparison_csv(comparison))
    (directory / "report.pdf").write_bytes(build_report_pdf(comparison.metrics, comparison.gullies, metadata))
    update_job(
        job_id,
        status="complete",
        progress=100,
        result={
            "metrics": comparison.metrics,
            "gullies": comparison.gullies,
            "files": ["metrics.json", "change.csv", "report.pdf"],
        },
    )


def run_demo_job(job_id: str):
    try:
        update_job(job_id, status="running", progress=20, message="生成演示地形")
        comparison = make_demo_comparison()
        _complete_comparison(job_id, comparison, {"source": "synthetic_demo"})
    except Exception as error:
        update_job(job_id, status="failed", error=str(error))


def run_compare_job(
    job_id: str,
    before_filename: str,
    after_filename: str,
    unit: str,
    grid_step_mm: float,
    erosion_threshold_mm: float,
    elapsed_hours: float,
    metadata: dict,
):
    try:
        update_job(job_id, status="running", progress=15, message="读取点云")
        directory = job_dir(job_id)
        before_path = directory / before_filename
        after_path = directory / after_filename
        before = load_point_cloud(before_path, before_path.name, unit)
        update_job(job_id, progress=35, message="插值为共同高程网格")
        after = load_point_cloud(after_path, after_path.name, unit)
        comparison = compare_point_clouds(
            before,
            after,
            grid_step_mm=grid_step_mm,
            erosion_threshold_mm=erosion_threshold_mm,
            elapsed_hours=elapsed_hours,
        )
        _complete_comparison(job_id, comparison, metadata)
    except Exception as error:
        update_job(job_id, status="failed", error=str(error))


def run_photo_job(
    job_id: str,
    before_archive: str,
    after_archive: str,
    *,
    grid_step_mm: float,
    erosion_threshold_mm: float,
    elapsed_hours: float,
    scale_mm_per_unit: float,
    max_views: int,
    metadata: dict,
):
    """Reconstruct both epochs with VGGT and compare them in a common frame."""
    try:
        directory = job_dir(job_id)
        update_job(job_id, status="running", progress=4, message="解压雨前/雨后照片")
        before_images = extract_image_archive(directory / before_archive, directory / "before")
        after_images = extract_image_archive(directory / after_archive, directory / "after")
        if len(before_images) < 8 or len(after_images) < 8:
            raise ValueError("每个时相至少需要 8 张照片，建议 25–40 张并保持高重叠")

        update_job(job_id, progress=8, message="检查照片质量")
        quality = {
            "before": inspect_images([(path.name, path) for path in before_images]),
            "after": inspect_images([(path.name, path) for path in after_images]),
        }
        (directory / "photo_quality.json").write_text(
            json.dumps(quality, ensure_ascii=False, indent=2), encoding="utf-8"
        )

        def before_progress(value, message):
            update_job(job_id, progress=10 + int(value * 0.38), message=f"雨前：{message}")

        def after_progress(value, message):
            update_job(job_id, progress=50 + int(value * 0.38), message=f"雨后：{message}")

        update_job(job_id, progress=10, message="雨前时相 VGGT 重建")
        before_cloud = reconstruct_vggt(before_images, max_views=max_views, progress=before_progress)
        update_job(job_id, progress=50, message="雨后时相 VGGT 重建")
        after_cloud = reconstruct_vggt(after_images, max_views=max_views, progress=after_progress)

        update_job(job_id, progress=88, message="相似变换配准与高程差计算")
        comparison, diagnostics = compare_reconstructions(
            before_cloud.points,
            after_cloud.points,
            grid_step_mm=grid_step_mm,
            threshold_mm=erosion_threshold_mm,
            elapsed_hours=elapsed_hours,
            scale_mm_per_unit=scale_mm_per_unit,
        )
        diagnostics["reconstruction"] = {"before": before_cloud.stats, "after": after_cloud.stats}
        diagnostics["photos"] = {"before": len(before_images), "after": len(after_images)}
        diagnostics["backend"] = "VGGT-1B (fp32 weights, bf16 autocast)"

        update_job(job_id, progress=93, message="生成三维坡面、冲淤图与报告")
        report_metadata = {
            **metadata,
            "重建方式": "VGGT-1B 多视角重建",
            "相似变换配准": (
                f"尺度 {diagnostics['alignment']['scale']:.4f}，"
                f"残差中位数 {diagnostics['alignment']['median_residual']:.3f} 单位"
            ),
            "尺度是否标定": "是" if diagnostics["scale_calibrated"] else "否（按 1 单位=1 mm 输出，需现场标尺复核）",
            "前后照片数": f"{len(before_images)} / {len(after_images)}",
        }
        (directory / "metrics.json").write_bytes(comparison_json(comparison, report_metadata))
        (directory / "change.csv").write_bytes(comparison_csv(comparison))
        (directory / "report.pdf").write_bytes(
            build_report_pdf(comparison.metrics, comparison.gullies, report_metadata)
        )
        (directory / "surface.png").write_bytes(render_surface_png(comparison))
        (directory / "dod.png").write_bytes(render_dod_png(comparison))
        (directory / "surface.json").write_text(
            json.dumps(surface_payload(comparison), ensure_ascii=False), encoding="utf-8"
        )
        (directory / "diagnostics.json").write_text(
            json.dumps(diagnostics, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        update_job(
            job_id,
            status="complete",
            progress=100,
            message="分析完成",
            result={
                "metrics": comparison.metrics,
                "gullies": comparison.gullies,
                "diagnostics": {
                    "scale_calibrated": diagnostics["scale_calibrated"],
                    "similarity_scale": diagnostics["alignment"]["scale"],
                    "median_residual": diagnostics["alignment"]["median_residual"],
                    "views": diagnostics["reconstruction"],
                },
                "files": [
                    "metrics.json",
                    "change.csv",
                    "report.pdf",
                    "surface.png",
                    "dod.png",
                    "surface.json",
                    "diagnostics.json",
                    "photo_quality.json",
                ],
            },
        )
    except Exception as error:  # noqa: BLE001 - job errors are reported to the client
        update_job(job_id, status="failed", error=str(error))


def run_reconstruction_job(job_id: str, max_image_size: int):
    try:
        directory = job_dir(job_id)
        update_job(job_id, status="running", progress=5, message="解压并检查照片")
        images_dir = directory / "images"
        image_files = extract_image_archive(directory / "photos.zip", images_dir)
        image_reports = inspect_images([(path.name, path) for path in image_files])
        (directory / "photo_quality.json").write_text(
            json.dumps(image_reports, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        update_job(job_id, progress=20, message="执行 COLMAP 重建")
        cloud_path = reconstruct_with_colmap(images_dir, directory / "reconstruction", max_image_size)
        update_job(
            job_id,
            status="complete",
            progress=100,
            message="完成。点云为 COLMAP 自由尺度坐标",
            result={
                "files": [
                    "reconstruction/dense/fused.ply",
                    "photo_quality.json",
                    "reconstruction/colmap.log",
                ],
                "coordinate_note": "单次照片重建结果尚未恢复实物尺度或统一到其他时相坐标系",
                "point_cloud": str(cloud_path.relative_to(directory)).replace("\\", "/"),
            },
        )
    except Exception as error:
        update_job(job_id, status="failed", error=str(error))

