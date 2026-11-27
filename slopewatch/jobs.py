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
from .quality import inspect_images
from .reconstruction import reconstruct_with_colmap
from .report import build_report_pdf
from .uploads import extract_image_archive


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

