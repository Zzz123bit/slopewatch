"""Adapter for the official VGGT 1B checkpoint: multi-view photos -> coloured points.

The checkpoint is loaded once per process and reused.  Inference is deliberately
bounded (view count, output points) so a laptop GPU can serve a demo job.
"""

from __future__ import annotations

import json
import os
import struct
import threading
from dataclasses import dataclass
from pathlib import Path

import numpy as np

CHECKPOINT_ENV = "SLOPEWATCH_VGGT_CHECKPOINT"
DTYPE_ENV = "SLOPEWATCH_VGGT_DTYPE"
DEVICE_ENV = "SLOPEWATCH_VGGT_DEVICE"
DEFAULT_CHECKPOINT = Path(r"D:\models\VGGT-1B\model.safetensors")
MAX_VIEWS = 24
_GPU_LOCK = threading.Lock()
_MODEL_CACHE: dict = {"path": None, "model": None}


class VGGTError(RuntimeError):
    def __init__(self, code: str, detail: str):
        super().__init__(detail)
        self.code = code


@dataclass
class Reconstruction:
    points: np.ndarray          # (N, 3) float64 in arbitrary VGGT units
    colors: np.ndarray          # (N, 3) uint8
    confidence: np.ndarray      # (N,) float32
    stats: dict


def _safetensors_complete(path: Path) -> bool:
    """A truncated safetensors file has a valid header but missing tensor data."""
    try:
        with path.open("rb") as stream:
            raw = stream.read(8)
            if len(raw) < 8:
                return False
            header_size = int.from_bytes(raw, "little")
            if not 0 < header_size < 64 * 1024 * 1024:
                return False
            header = stream.read(header_size)
            if len(header) < header_size:
                return False
            info = json.loads(header)
            data_end = max(
                (int(value["data_offsets"][1]) for key, value in info.items() if key != "__metadata__"),
                default=0,
            )
        return path.stat().st_size >= 8 + header_size + data_end
    except (OSError, ValueError, KeyError):
        return False


def _checkpoint() -> Path:
    value = os.environ.get(CHECKPOINT_ENV)
    path = Path(value) if value else DEFAULT_CHECKPOINT
    if not path.is_file():
        raise VGGTError("checkpoint_missing", f"找不到 VGGT 权重文件 {path}")
    if path.suffix.lower() == ".safetensors" and not _safetensors_complete(path):
        raise VGGTError("checkpoint_incomplete", f"权重文件 {path.name} 尚未下载完成，请等待下载结束后重试")
    parts = path.with_name(path.name + ".parts.json")
    if parts.is_file():
        try:
            completed = len({int(x) for x in json.loads(parts.read_text())})
        except (OSError, ValueError):
            completed = 0
        expected = -(-path.stat().st_size // (4 * 1024 * 1024))
        if completed < expected:
            percent = 100.0 * completed / max(expected, 1)
            raise VGGTError(
                "checkpoint_incomplete",
                f"权重仍在下载：{completed}/{expected} 块（{percent:.0f}%），完成后自动可用",
            )
    if path.stat().st_size < 1_000_000_000:
        raise VGGTError("checkpoint_incomplete", f"权重文件 {path.name} 不完整，请重新下载")
    return path


def _imports():
    # OpenCV and PyTorch ship separate Intel OpenMP runtimes; loading both in one
    # process aborts unless the duplicate runtime is tolerated.
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
    try:
        import torch
        from safetensors.torch import load_file as load_safetensors
        from vggt.models.vggt import VGGT
        from vggt.utils.load_fn import load_and_preprocess_images
    except (ImportError, OSError, RuntimeError) as exc:
        raise VGGTError("dependencies_missing", "请安装 torch、safetensors 与官方 vggt 包") from exc
    return torch, load_safetensors, VGGT, load_and_preprocess_images


def vggt_status() -> dict:
    """Check checkpoint, imports and CUDA without loading the weights."""
    try:
        path = _checkpoint()
        torch, _, _, _ = _imports()
    except VGGTError as exc:
        return {"available": False, "code": exc.code, "detail": str(exc)}
    if not torch.cuda.is_available():
        return {"available": False, "code": "cuda_unavailable", "detail": "需要 NVIDIA GPU 与 CUDA 版 PyTorch"}
    return {
        "available": True,
        "code": "ready",
        "detail": "已检测到权重与 CUDA，提交照片后加载模型",
        "checkpoint": str(path),
        "checkpoint_gb": round(path.stat().st_size / 1024**3, 2),
    }


def _load_model(torch, load_safetensors, model_type, path: Path, device: str):
    key = str(path.resolve())
    with _GPU_LOCK:
        if _MODEL_CACHE["model"] is not None and _MODEL_CACHE["path"] == key:
            return _MODEL_CACHE["model"]
        model = model_type()
        try:
            if path.suffix.lower() == ".safetensors":
                state = load_safetensors(str(path), device="cpu")
            else:
                state = torch.load(str(path), map_location="cpu", weights_only=True)
            if isinstance(state, dict):
                state = state.get("model", state.get("state_dict", state))
            model.load_state_dict(state, strict=True)
            del state
        except Exception as exc:  # noqa: BLE001 - surface a readable job error
            raise VGGTError("checkpoint_invalid", f"VGGT 权重加载失败：{exc}") from exc
        model.eval().to(device)
        dtype = os.environ.get(DTYPE_ENV, "fp32").lower()
        if dtype in ("bf16", "bfloat16"):
            model = model.to(torch.bfloat16)
        elif dtype in ("fp16", "float16", "half"):
            model = model.to(torch.float16)
        _MODEL_CACHE.update(path=key, model=model)
        return model


def _select_views(paths: list[Path], max_views: int) -> list[Path]:
    if len(paths) <= max_views:
        return paths
    indices = np.linspace(0, len(paths) - 1, max_views).round().astype(int)
    return [paths[i] for i in sorted(set(indices.tolist()))]


def reconstruct_vggt(
    image_paths,
    *,
    max_views: int = 8,
    confidence_percentile: float = 35.0,
    max_points: int = 200_000,
    progress=None,
) -> Reconstruction:
    """Run VGGT on one epoch and return a filtered, coloured point cloud."""
    paths = [Path(path) for path in image_paths]
    if len(paths) < 2:
        raise ValueError("每个时相至少需要 2 张重叠照片")
    if len(paths) > MAX_VIEWS:
        raise ValueError(f"单个时相最多 {MAX_VIEWS} 张照片，避免显存溢出")
    paths = _select_views(paths, max(2, int(max_views)))
    if path_missing := [p.name for p in paths if not p.is_file()]:
        raise ValueError(f"照片不存在：{', '.join(path_missing[:3])}")

    torch, load_safetensors, model_type, preprocess = _imports()
    device = os.environ.get(DEVICE_ENV, "auto").lower()
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
        if device == "cuda":
            # Laptop GPUs are often already busy with the desktop compositor;
            # paging a 1B model through shared memory is far slower than CPU.
            free, _ = torch.cuda.mem_get_info()
            if free < 6 * 1024**3:
                device = "cpu"
    if device == "cuda" and not torch.cuda.is_available():
        raise VGGTError("cuda_unavailable", "未检测到可用的 CUDA 设备")
    if device not in ("cuda", "cpu"):
        raise VGGTError("device_invalid", f"不支持的推理设备 {device}")
    checkpoint = _checkpoint()
    if progress:
        progress(15, f"预处理 {len(paths)} 张照片")
    try:
        images = preprocess([str(path) for path in paths]).to(device)
    except Exception as exc:  # noqa: BLE001
        raise VGGTError("preprocessing_failed", f"照片预处理失败：{exc}") from exc

    if progress:
        progress(30, "加载 VGGT 权重")
    model = _load_model(torch, load_safetensors, model_type, checkpoint, device)
    if progress:
        progress(45, f"VGGT 推理（{len(paths)} 视图）")
    try:
        param_dtype = next(model.parameters()).dtype
        autocast_dtype = torch.float16 if param_dtype == torch.float16 else torch.bfloat16
        with torch.inference_mode(), torch.autocast(device, dtype=autocast_dtype):
            predictions = model(images)
    except torch.OutOfMemoryError as exc:
        if device == "cuda":
            torch.cuda.empty_cache()
        raise VGGTError("gpu_memory", "显存不足：请减少每期照片数或降低重建点数") from exc
    except Exception as exc:  # noqa: BLE001
        raise VGGTError("inference_failed", f"VGGT 推理失败：{exc}") from exc

    points = predictions["world_points"][0].float().cpu().numpy()
    confidence = predictions["world_points_conf"][0].float().cpu().numpy()
    rgb = predictions["images"][0].permute(0, 2, 3, 1).float().cpu().numpy()
    del predictions
    torch.cuda.empty_cache()

    if points.ndim != 4 or points.shape[-1] != 3 or confidence.shape != points.shape[:3]:
        raise VGGTError("inference_failed", "VGGT 输出维度异常")

    valid = np.isfinite(points).all(axis=-1) & np.isfinite(confidence)
    if valid.sum() < 100:
        raise VGGTError("empty_reconstruction", "有效三维点不足，请检查照片重叠度与清晰度")
    threshold = float(np.quantile(confidence[valid], confidence_percentile / 100.0))
    valid &= confidence >= threshold

    flat_points = points[valid].astype(np.float64)
    flat_conf = confidence[valid].astype(np.float32)
    flat_rgb = np.clip(rgb[valid] * 255.0, 0, 255).astype(np.uint8)

    if len(flat_points) > max_points:
        stride = int(np.ceil(len(flat_points) / max_points))
        flat_points = flat_points[::stride]
        flat_conf = flat_conf[::stride]
        flat_rgb = flat_rgb[::stride]

    stats = {
        "views": len(paths),
        "valid_pixels": int(valid.sum()),
        "points": int(len(flat_points)),
        "confidence_threshold": threshold,
        "confidence_percentile": float(confidence_percentile),
        "image_size": [int(images.shape[-1]), int(images.shape[-2])],
        "checkpoint": checkpoint.name,
    }
    return Reconstruction(flat_points, flat_rgb, flat_conf, stats)
