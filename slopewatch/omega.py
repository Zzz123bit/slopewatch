"""VGGT-Ω geometry adapter for the official 1B-512 checkpoint.

Upstream: https://github.com/facebookresearch/vggt-omega/tree/48b23c8ce72c9a8fcf987f0d7f43d8e4870759f6
Weights: https://huggingface.co/facebook/VGGT-Omega (vggt_omega_1b_512.pt).
Torch and the upstream package are optional until a photo job is requested.
"""

from __future__ import annotations

import os
import threading
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
from PIL import Image, ImageOps


_GPU_LOCK = threading.Lock()
CHECKPOINT_ENV = "SLOPEWATCH_OMEGA_CHECKPOINT"


class OmegaError(RuntimeError):
    def __init__(self, code: str, detail: str):
        super().__init__(detail)
        self.code = code


@dataclass(frozen=True)
class ViewMapping:
    """Sizes are (W,H). Source pixels refer to the EXIF-upright original."""

    source_size: tuple[int, int]
    crop_box: tuple[int, int, int, int]
    resized_size: tuple[int, int]
    canvas_size: tuple[int, int]

    @property
    def offset(self) -> tuple[int, int]:
        return tuple((total - used) // 2 for total, used in zip(self.canvas_size, self.resized_size))

    def map_pixels(self, xy: np.ndarray) -> np.ndarray:
        """Map pixel centers (x,y), returning NaN for centers outside the crop."""
        xy = np.asarray(xy, dtype=np.float64)
        if xy.ndim == 0 or xy.shape[-1] != 2:
            raise ValueError("像素坐标须为 (x, y)")
        origin = np.array(self.crop_box[:2])
        limit = np.array(self.crop_box[2:])
        valid = ((xy >= origin) & (xy <= limit - 1)).all(axis=-1)
        scale = np.array(self.resized_size) / (limit - origin)
        mapped = (xy - origin + 0.5) * scale - 0.5 + np.array(self.offset)
        return np.where(valid[..., None], mapped, np.nan)

    def map_mask(self, mask: np.ndarray) -> np.ndarray:
        """Map upright uint8 class IDs with nearest sampling, padding with 255."""
        mask = np.asarray(mask)
        if mask.ndim != 2 or mask.shape[::-1] != self.source_size:
            raise ValueError("标签 PNG 必须为单通道，尺寸与按 EXIF 转正后的原图一致")
        if mask.dtype.kind not in "biu" or np.min(mask) < 0 or np.max(mask) > 255:
            raise ValueError("标签必须为 0–255 的整数")
        labels = Image.fromarray(mask.astype(np.uint8, copy=False)).crop(self.crop_box)
        labels = labels.resize(self.resized_size, Image.Resampling.NEAREST)
        canvas = Image.new("L", self.canvas_size, 255)
        canvas.paste(labels, self.offset)
        return np.asarray(canvas)


@dataclass
class OmegaScene:
    """Points in a single arbitrary-scale world frame, indexed [view,row,col].

    Color, confidence and coordinates share the processed image grid. Confidence
    is the raw depth-head score (1 + exp(logit)), not a probability. Invalid
    geometry and padding have NaN coordinates and confidence -inf.
    """

    points: np.ndarray
    confidence: np.ndarray
    rgb: np.ndarray
    views: tuple[ViewMapping, ...]
    image_names: tuple[str, ...]


def _checkpoint() -> Path:
    value = os.environ.get(CHECKPOINT_ENV)
    if not value or not Path(value).is_file():
        raise OmegaError("checkpoint_missing", f"请取得官方 vggt_omega_1b_512.pt 并设置 {CHECKPOINT_ENV}")
    return Path(value)


def _runtime():
    try:
        import torch
        from vggt_omega.models import VGGTOmega
        from vggt_omega.utils.load_fn import load_and_preprocess_images
        from vggt_omega.utils.pose_enc import encoding_to_camera
    except (ImportError, OSError, RuntimeError) as exc:
        raise OmegaError("dependencies_missing", "请安装匹配的 CUDA PyTorch、torchvision 和官方 vggt-omega 包") from exc
    if not torch.cuda.is_available():
        raise OmegaError("cuda_unavailable", "请在装有 NVIDIA GPU 且 PyTorch CUDA 可用的环境运行")
    return torch, VGGTOmega, load_and_preprocess_images, encoding_to_camera


def omega_status() -> dict:
    """Check configured checkpoint presence, imports and CUDA without loading weights."""
    try:
        _checkpoint()
        _runtime()
    except OmegaError as exc:
        return {"available": False, "code": exc.code, "detail": str(exc)}
    return {"available": True, "code": "ready", "detail": "已检测到权重文件、官方包和 CUDA，提交照片后加载权重"}


def _crop_box(size: tuple[int, int]) -> tuple[int, int, int, int]:
    # The pinned upstream loader center-crops aspect ratios outside [1/2, 2].
    width, height = size
    crop_w, crop_h = min(width, height * 2), min(height, width * 2)
    left, top = (width - crop_w) // 2, (height - crop_h) // 2
    return left, top, left + crop_w, top + crop_h


def _prepare_images(paths, torch, preprocess):
    frames, sizes = [], []
    with TemporaryDirectory(prefix="slopewatch-omega-") as directory:
        for index, path in enumerate(paths):
            source = path
            with Image.open(path) as image:
                if image.getexif().get(274, 1) != 1:
                    upright = ImageOps.exif_transpose(image)
                    # Upstream composites RGBA on white; keep the alpha channel.
                    source = Path(directory) / f"{index}.png"
                    upright.convert("RGBA" if "A" in upright.getbands() else "RGB").save(source)
                    sizes.append(upright.size)
                else:
                    sizes.append(image.size)
            # A single-view call exposes the exact unpadded dimensions produced
            # by upstream, without duplicating its patch-grid rounding algorithm.
            frame = preprocess([str(source)], mode="max_size", image_resolution=512)[0]
            frames.append(frame)
    shape = (max(frame.shape[2] for frame in frames), max(frame.shape[1] for frame in frames))
    mappings, padded = [], []
    for frame, size in zip(frames, sizes):
        height, width = frame.shape[-2:]
        mapping = ViewMapping(size, _crop_box(size), (width, height), shape)
        left, top = mapping.offset
        padding = (left, shape[0] - width - left, top, shape[1] - height - top)
        padded.append(torch.nn.functional.pad(frame, padding, value=1.0))
        mappings.append(mapping)
    return torch.stack(padded), tuple(mappings)


def _world_geometry(depth, scores, extrinsics, intrinsics, views):
    """Invert upstream OpenCV world-to-camera poses; depth is camera Z."""
    count = len(views)
    width, height = views[0].canvas_size
    if (depth.shape != (count, height, width, 1) or scores.shape != (count, height, width)
            or extrinsics.shape != (count, 3, 4) or intrinsics.shape != (count, 3, 3)):
        raise ValueError("深度、置信度或相机矩阵的维度与模型输入不符")
    depth = depth[..., 0]
    confidence = scores.astype(np.float32, copy=True)
    points = np.empty((count, height, width, 3), np.float32)
    u = np.arange(width, dtype=np.float32)
    v = np.arange(height, dtype=np.float32)[:, None]
    for i, view in enumerate(views):
        K = intrinsics[i]
        pose = extrinsics[i]
        if not (np.isfinite(K).all() and np.isfinite(pose).all() and K[0, 0] > 0 and K[1, 1] > 0):
            raise ValueError("模型返回无效相机参数")
        z = depth[i]
        camera = np.stack(((u - K[0, 2]) / K[0, 0] * z, (v - K[1, 2]) / K[1, 1] * z, z), axis=-1)
        points[i] = (camera - pose[:, 3]) @ pose[:, :3]
        valid = (z > 0) & np.isfinite(points[i]).all(axis=-1) & np.isfinite(confidence[i])
        left, top = view.offset
        w, h = view.resized_size
        valid[:top] = valid[top + h:] = False
        valid[:, :left] = valid[:, left + w:] = False
        confidence[i, ~valid] = -np.inf
        points[i, ~valid] = np.nan
    return points, confidence


def reconstruct_omega(image_paths: Sequence[Path]) -> OmegaScene:
    """Run all views of one epoch jointly through official VGGT-Ω on CUDA."""
    paths = tuple(Path(path) for path in image_paths)
    if not 2 <= len(paths) <= 32:
        raise ValueError("每期 VGGT-Ω 重建需要 2–32 张重叠照片")
    checkpoint = _checkpoint()
    torch, model_type, preprocess, decode_camera = _runtime()
    with _GPU_LOCK:
        try:
            images, views = _prepare_images(paths, torch, preprocess)
        except (OSError, ValueError, Image.DecompressionBombError) as exc:
            raise OmegaError("preprocessing_failed", "无法读取照片或预处理图像，请检查文件格式和尺寸") from exc
        try:
            model = model_type().eval()
            weights = torch.load(checkpoint, map_location="cpu", weights_only=True)
            if isinstance(weights, dict):
                weights = weights.get("model", weights.get("state_dict", weights))
            model.load_state_dict(weights, strict=True)
            del weights
        except Exception as exc:
            raise OmegaError("checkpoint_invalid", "VGGT-Ω 权重加载失败，请核实官方 1B-512 检查点和可用内存") from exc
        try:
            with torch.inference_mode():
                predictions = model.to("cuda")(images.to("cuda"))
                extrinsics, intrinsics = decode_camera(predictions["pose_enc"], predictions["images"].shape[-2:])
                depth = predictions["depth"][0].float().cpu().numpy()
                scores = predictions["depth_conf"][0].float().cpu().numpy()
                points, confidence = _world_geometry(
                    depth, scores, extrinsics[0].float().cpu().numpy(), intrinsics[0].float().cpu().numpy(), views,
                )
        except torch.OutOfMemoryError as exc:
            raise OmegaError("gpu_memory", "VGGT-Ω 显存不足，请释放显存、减少每期照片数或换用显存更大的 GPU") from exc
        except (RuntimeError, KeyError, ValueError) as exc:
            raise OmegaError("inference_failed", "VGGT-Ω 几何推理失败，请检查模型版本、CUDA 和输入照片") from exc
    rgb = images.permute(0, 2, 3, 1).mul(255).round().byte().numpy()
    return OmegaScene(points, confidence, rgb, views, tuple(path.name for path in paths))
