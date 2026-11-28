"""Bearer-authenticated entry point for the SlopeWatch HTTPS tunnel."""

import hmac
import os
import re
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import APIRouter, Depends, FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse
from fastapi.security import HTTPBearer

from api import submissions
from slopewatch.jobs import job_dir, read_job
from slopewatch.reconstruction import colmap_status


@asynccontextmanager
async def lifespan(app: FastAPI):
    token = os.environ.get("SLOPEWATCH_API_TOKEN", "")
    if not re.fullmatch(r"[A-Za-z0-9_-]{32,}", token):
        raise RuntimeError("请将 SLOPEWATCH_API_TOKEN 配置为至少 32 位的随机 URL 安全字符（A-Z、a-z、0-9、-、_）")
    app.state.access_token = token.encode("ascii")
    try:
        yield
    finally:
        app.state.access_token = None


app = FastAPI(
    title="坡镜智安远程接口",
    version="1.0",
    description="点击 Authorize 输入团队访问令牌，可以提交试验任务并下载结果。",
    lifespan=lifespan,
)
PUBLIC_PATHS = {"/", "/livez", "/docs", "/docs/oauth2-redirect", "/redoc", "/openapi.json"}
EXPORT_NAMES = {
    "change.csv", "metrics.json", "report.pdf", "photo_quality.json",
    "surface.png", "dod.png", "surface.json", "diagnostics.json", "fusion.json", "atlas.png",
    "reconstruction/dense/fused.ply",
}
WEB_PAGE = Path(__file__).resolve().parent / "web" / "index.html"


@app.middleware("http")
async def authorize(request: Request, call_next):
    expected = getattr(request.app.state, "access_token", None)
    if not expected:
        return JSONResponse({"detail": "远程服务未配置"}, status_code=503)
    if request.url.path not in PUBLIC_PATHS:
        headers = request.headers.getlist("authorization")
        scheme, _, credential = (headers[0] if len(headers) == 1 else "").partition(" ")
        if scheme.lower() != "bearer" or not hmac.compare_digest(credential.encode("utf-8"), expected):
            return JSONResponse({"detail": "请输入有效的访问令牌"}, status_code=401,
                                headers={"WWW-Authenticate": "Bearer", "Cache-Control": "no-store"})
    response = await call_next(request)
    response.headers["Cache-Control"] = "no-store"
    response.headers["X-Content-Type-Options"] = "nosniff"
    return response


@app.get("/", include_in_schema=False)
def web_client():
    return FileResponse(WEB_PAGE)


@app.get("/livez")
def livez():
    return {"service": "slopewatch-public", "status": "ok"}


# Middleware runs before multipart parsing. The dependency also advertises
# the same Bearer scheme to Swagger UI and generated clients.
business = APIRouter(dependencies=[Depends(HTTPBearer(scheme_name="TeamToken"))])
business.include_router(submissions)


@business.get("/api/v1/health")
def health():
    return {"status": "ok", "service": "slopewatch", "colmap": {"available": colmap_status()["available"]}}


def public_job(job_id: str) -> dict:
    try:
        job = read_job(job_id)
    except (ValueError, FileNotFoundError) as error:
        raise HTTPException(status_code=404, detail="任务不存在") from error
    if job.get("status") == "failed":
        job["error"] = "任务处理失败。请核对输入，或将任务编号提供给演示机负责人查看详情。"
    if "result" in job:
        result = job["result"]
        result["files"] = [name for name in result.get("files", []) if name in EXPORT_NAMES]
    return job


@business.get("/api/v1/jobs/{job_id}")
def status(job_id: str):
    return public_job(job_id)


@business.get("/api/v1/jobs/{job_id}/result")
def result(job_id: str):
    job = public_job(job_id)
    if job["status"] != "complete":
        raise HTTPException(status_code=409, detail=f"任务状态为 {job['status']}")
    if "result" not in job:
        raise HTTPException(status_code=404, detail="任务没有结构化结果")
    return job["result"]


@business.get("/api/v1/jobs/{job_id}/files/{filename:path}")
def download(job_id: str, filename: str):
    artifacts = result(job_id)
    if filename not in artifacts["files"]:
        raise HTTPException(status_code=404, detail="未找到结果文件")
    directory = job_dir(job_id).resolve()
    file = (directory / filename).resolve()
    if not file.is_relative_to(directory) or not file.is_file():
        raise HTTPException(status_code=404, detail="未找到结果文件")
    return FileResponse(file, filename=file.name)


app.include_router(business)
