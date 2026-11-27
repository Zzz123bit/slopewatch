from __future__ import annotations

import json
from pathlib import Path

from fastapi import APIRouter, BackgroundTasks, FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse

from slopewatch.jobs import (
    create_job,
    job_dir,
    read_job,
    run_compare_job,
    run_demo_job,
    run_reconstruction_job,
)
from slopewatch.reconstruction import colmap_status


app = FastAPI(
    title="坡镜智安 API",
    version="0.1.0",
    description="室内坡面多时相点云差分与演示任务接口",
)
MAX_UPLOAD_BYTES = 300 * 1024 * 1024
ALLOWED_CLOUD_SUFFIXES = {".ply", ".xyz", ".txt", ".csv"}
submissions = APIRouter()


def _read_upload(upload: UploadFile, limit: int = MAX_UPLOAD_BYTES) -> bytes:
    data = upload.file.read(limit + 1)
    if len(data) > limit:
        raise HTTPException(status_code=413, detail="上传文件超过 300 MB")
    if not data:
        raise HTTPException(status_code=400, detail="上传文件为空")
    return data


@app.get("/api/v1/health")
def health():
    return {"status": "ok", "service": "slopewatch", "colmap": colmap_status()}


@submissions.post("/api/v1/jobs/demo", status_code=202)
def start_demo(background_tasks: BackgroundTasks):
    job_id, _ = create_job("synthetic_demo")
    background_tasks.add_task(run_demo_job, job_id)
    return {"job_id": job_id, "status": "queued", "status_url": f"/api/v1/jobs/{job_id}"}


@submissions.post("/api/v1/jobs/compare", status_code=202)
def start_comparison(
    background_tasks: BackgroundTasks,
    before: UploadFile = File(...),
    after: UploadFile = File(...),
    unit: str = Form("mm"),
    grid_step_mm: float = Form(2.0, gt=0.0, le=100.0),
    erosion_threshold_mm: float = Form(3.0, ge=0.0, le=100.0),
    elapsed_hours: float = Form(1.0, ge=0.0, le=1000000.0),
    metadata_json: str = Form("{}"),
):
    if unit not in ("mm", "m"):
        raise HTTPException(status_code=422, detail="unit 仅支持 mm 或 m")
    for upload in (before, after):
        suffix = Path(upload.filename or "").suffix.lower()
        if suffix not in ALLOWED_CLOUD_SUFFIXES:
            raise HTTPException(status_code=415, detail="点云格式请使用 PLY、XYZ、TXT 或 CSV")
    try:
        metadata = json.loads(metadata_json)
        if not isinstance(metadata, dict):
            raise ValueError("metadata_json 须为 JSON 对象")
    except ValueError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error

    before_data = _read_upload(before)
    after_data = _read_upload(after)
    job_id, directory = create_job("point_cloud_comparison", metadata)
    before_name = f"before{Path(before.filename or '').suffix.lower()}"
    after_name = f"after{Path(after.filename or '').suffix.lower()}"
    (directory / before_name).write_bytes(before_data)
    (directory / after_name).write_bytes(after_data)
    background_tasks.add_task(
        run_compare_job,
        job_id,
        before_name,
        after_name,
        unit,
        grid_step_mm,
        erosion_threshold_mm,
        elapsed_hours,
        metadata,
    )
    return {"job_id": job_id, "status": "queued", "status_url": f"/api/v1/jobs/{job_id}"}


@submissions.post("/api/v1/reconstructions", status_code=202)
def start_reconstruction(
    background_tasks: BackgroundTasks,
    photos_zip: UploadFile = File(...),
    max_image_size: int = Form(1600),
):
    if max_image_size < 512 or max_image_size > 4000:
        raise HTTPException(status_code=422, detail="max_image_size 范围为 512 到 4000")
    if Path(photos_zip.filename or "").suffix.lower() != ".zip":
        raise HTTPException(status_code=415, detail="照片请打包为 ZIP 上传")
    data = _read_upload(photos_zip)
    if not colmap_status()["available"]:
        raise HTTPException(
            status_code=503,
            detail="当前环境未检测到 COLMAP。点云分析和演示数据可独立运行",
        )
    job_id, directory = create_job("photo_reconstruction")
    (directory / "photos.zip").write_bytes(data)
    background_tasks.add_task(run_reconstruction_job, job_id, max_image_size)
    return {"job_id": job_id, "status": "queued", "status_url": f"/api/v1/jobs/{job_id}"}


@app.get("/api/v1/jobs/{job_id}")
def get_job(job_id: str):
    try:
        return read_job(job_id)
    except (ValueError, FileNotFoundError) as error:
        raise HTTPException(status_code=404, detail=str(error)) from error


@app.get("/api/v1/jobs/{job_id}/result")
def get_result(job_id: str):
    try:
        job = read_job(job_id)
        if job["status"] != "complete":
            raise HTTPException(status_code=409, detail=f"任务状态为 {job['status']}")
        if "result" not in job:
            raise HTTPException(status_code=404, detail="任务没有结构化结果")
        return job["result"]
    except (ValueError, FileNotFoundError) as error:
        raise HTTPException(status_code=404, detail=str(error)) from error


@app.get("/api/v1/jobs/{job_id}/files/{filename:path}")
def get_job_file(job_id: str, filename: str):
    try:
        directory = job_dir(job_id)
    except (ValueError, FileNotFoundError) as error:
        raise HTTPException(status_code=404, detail=str(error)) from error
    file_path = (directory / filename).resolve()
    if directory.resolve() not in file_path.parents or not file_path.is_file():
        raise HTTPException(status_code=404, detail="文件不存在")
    return FileResponse(file_path, filename=file_path.name)


app.include_router(submissions)

