"""Two-epoch metric surveys. World XYZ are millimetres, with Z vertically up."""

from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Annotated, Literal
from zipfile import ZipFile

import cv2
import numpy as np
from PIL import Image, ImageOps
from pydantic import BaseModel, ConfigDict, Field, model_validator

from . import omega
from .core import Comparison, compare_point_clouds
from .fusion import analyze_fusion, load_segmenter, project_label_views, read_label_mask
from .registration import fit_control_frame, refine_stable_icp


class SurveyInput(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)


class Target(SurveyInput):
    id: int = Field(ge=0, lt=50)
    xyz_mm: tuple[float, float, float]
    role: Literal["control", "check"] = "control"


# min XYZ followed by max XYZ. Height limits exclude walls and background.
Box = tuple[float, float, float, float, float, float]


class SurveySettings(SurveyInput):
    targets: Annotated[list[Target], Field(min_length=4, max_length=50)]
    plot_box_mm: Box
    stable_boxes_mm: Annotated[list[Box], Field(min_length=1, max_length=16)]
    grid_step_mm: float = Field(default=2, gt=0, le=100)
    threshold_mm: float = Field(default=3, gt=0, le=100)
    tolerance_mm: float = Field(default=3, gt=0, le=100)
    elapsed_hours: float = Field(gt=0)
    min_depth_confidence: float = Field(default=2.0, ge=1)
    min_coverage: float = Field(default=0.7, gt=0, le=1)

    @model_validator(mode="after")
    def check_geometry(self):
        ids = {target.id for target in self.targets}
        if len(ids) != len(self.targets):
            raise ValueError("靶标 ID 必须唯一")
        controls = sum(target.role == "control" for target in self.targets)
        if controls < 3 or controls == len(ids):
            raise ValueError("至少需要三个控制点和一个独立检查点")
        for box in [self.plot_box_mm, *self.stable_boxes_mm]:
            if any(lo >= hi for lo, hi in zip(box[:3], box[3:])):
                raise ValueError("区域须按 xmin,ymin,zmin,xmax,ymax,zmax 排列")
        return self


def unpack_survey(archive: Path, root: Path):
    """Extract only the named survey inputs, validating the archive before inference."""
    with ZipFile(archive) as zipped:
        files = [item for item in zipped.infolist() if not item.is_dir()]
        if len(files) > 97 or sum(item.file_size for item in files) > 1024**3:
            raise ValueError("试验包超过 97 个文件或 1 GB 展开大小")
        if sum(item.filename == "survey.json" for item in files) != 1:
            raise ValueError("试验包根目录须有一份 survey.json")
        if zipped.getinfo("survey.json").file_size > 64 * 1024:
            raise ValueError("survey.json 太大")
        settings = SurveySettings.model_validate_json(zipped.read("survey.json"))
        seen = set()
        images = {"before": [], "after": []}
        masks = {}
        for item in files:
            name = PurePosixPath(item.filename)
            if item.filename == "survey.json":
                continue
            # Store only generated numeric names. User filenames are lookup keys.
            parts = name.parts
            key = (parts[:-1], name.stem.casefold())
            if key in seen or item.file_size > 50 * 1024**2 or not item.file_size:
                raise ValueError("试验包包含同名、空文件或超过 50 MB 的文件")
            seen.add(key)
            if len(parts) == 2 and parts[0] in images and name.suffix.lower() in {".jpg", ".jpeg", ".png"}:
                entry = images[parts[0]]
                output = root / parts[0] / (str(len(entry)) + name.suffix.lower())
                entry.append((name.stem.casefold(), output))
            elif parts[:-1] == ("masks", "after") and name.suffix.lower() == ".png":
                output = root / "labels" / f"{len(masks)}.png"
                masks[name.stem.casefold()] = output
            else:
                raise ValueError("试验包中有无法识别的文件，请按示例目录打包")
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_bytes(zipped.read(item))
        for epoch, entries in images.items():
            if not 2 <= len(entries) <= 32:
                raise ValueError(f"{epoch} 目录须有 2–32 张多视角照片")
            entries.sort(key=lambda entry: entry[0])
        if masks.keys() - {stem for stem, _ in images["after"]}:
            raise ValueError("语义掩码须与 after 中的一张照片同名")
    return settings, images, masks


def _detect_centres(rgb):
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    corners, ids, _ = cv2.aruco.ArucoDetector(dictionary).detectMarkers(cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY))
    centres = {}
    for marker, marker_id in zip(corners, [] if ids is None else ids.ravel()):
        a, b, c, d = np.column_stack((marker.reshape(4, 2), np.ones(4)))
        centre = np.cross(np.cross(a, c), np.cross(b, d))
        if abs(centre[2]) > 1e-10:
            if int(marker_id) in centres:
                raise ValueError("同一照片出现重复 ArUco ID")
            centres[int(marker_id)] = centre[:2] / centre[2]
    return centres


@dataclass
class Epoch:
    points: np.ndarray
    views: list
    landmarks: dict
    control_ids: list
    transform: object
    calibration: dict


def prepare_epoch(entries, masks, settings, segmenter=None):
    scene = omega.reconstruct_omega([path for _, path in entries])
    observations = defaultdict(list)
    points, votes = [], []
    for index, (stem, path) in enumerate(entries):
        with Image.open(path) as image:
            rgb = np.asarray(ImageOps.exif_transpose(image).convert("RGB"))
        xyz = scene.points[index]
        valid = np.isfinite(xyz).all(axis=-1) & np.isfinite(scene.confidence[index])
        valid &= scene.confidence[index] >= settings.min_depth_confidence
        mapping = scene.views[index]
        for marker_id, centre in _detect_centres(rgb).items():
            pixel = np.asarray(mapping.map_pixels(centre.reshape(1, 2)))[0]
            if not np.isfinite(pixel).all():
                continue
            u, v = np.rint(pixel).astype(int)
            if not (0 <= u < valid.shape[1] and 0 <= v < valid.shape[0]) or not valid[v, u]:
                continue
            observations[marker_id].append(xyz[v, u].copy())
        # Bound DEM interpolation work independently of camera count.
        stride = max(1, int(np.ceil(np.sqrt(len(entries) * valid.size / 400_000))))
        select = valid[::stride, ::stride]
        view_points = xyz[::stride, ::stride][select]
        points.append(view_points)
        labels = None
        if stem in masks:
            labels = read_label_mask(masks[stem])
            if labels.shape != rgb.shape[:2]:
                raise ValueError("语义掩码与 EXIF 校正后的对应照片尺寸不一致")
        elif segmenter is not None:
            labels = segmenter(rgb)
        if labels is not None:
            votes.append((view_points, mapping.map_mask(labels)[::stride, ::stride][select]))
    targets = {item.id: item for item in settings.targets}
    landmarks = {key: np.median(rows, axis=0) for key, rows in observations.items()
                 if key in targets and len(rows) >= 2}
    control_ids = sorted(key for key in landmarks if targets[key].role == "control")
    if len(control_ids) < 3:
        raise ValueError("可用控制点不足：至少三个不同控制靶的中心各需在两张照片中通过置信度筛选")
    frame, stats = fit_control_frame(np.array([landmarks[i] for i in control_ids]),
                                    np.array([targets[i].xyz_mm for i in control_ids]),
                                    max_error_mm=settings.tolerance_mm)
    stats["observed_target_views"] = {str(key): len(rows) for key, rows in observations.items() if key in targets}
    stats["control_ids"] = control_ids
    return Epoch(frame.apply(np.concatenate(points)), [(frame.apply(p), ids) for p, ids in votes],
                 landmarks, control_ids, frame, stats)


def inside(points, box):
    return np.all((points >= box[:3]) & (points <= box[3:]), axis=1)


def _stable_points(points, boxes):
    return points[np.logical_or.reduce([inside(points, box) for box in boxes])]


@dataclass
class SurveyResult:
    comparison: Comparison
    fusion: object
    diagnostics: dict


def analyze_survey(archive: Path, root: Path, progress):
    settings, images, masks = unpack_survey(archive, root)
    progress(10, "VGGT-Ω 基准期重建与靶标校准")
    before = prepare_epoch(images["before"], {}, settings)
    progress(42, "VGGT-Ω 复测期重建、标定与语义提取")
    # Partial hand-labelled views use only their supplied masks, with explicit unknown cells.
    segmenter = None if masks else load_segmenter()
    after = prepare_epoch(images["after"], masks, settings, segmenter)
    progress(75, "固定区域精配准与独立检查点检验")
    rigid, refinement = refine_stable_icp(_stable_points(before.points, settings.stable_boxes_mm),
                                          _stable_points(after.points, settings.stable_boxes_mm),
                                          max_distance_mm=3 * settings.tolerance_mm,
                                          max_rmse_mm=settings.tolerance_mm)
    final = after.transform.then(rigid)
    targets = {t.id: t for t in settings.targets}
    for idx in after.calibration["inlier_indices"]:
        key = after.control_ids[idx]
        if np.linalg.norm(final.apply(after.landmarks[key][None])[0] - targets[key].xyz_mm) > settings.tolerance_mm:
            raise ValueError("稳定区精配准导致控制靶漂移超限")
    checks = []
    for target in settings.targets:
        if target.role != "check":
            continue
        if target.id not in before.landmarks or target.id not in after.landmarks:
            raise ValueError(f"独立检查靶 {target.id} 在两期均须至少被两张照片观测")
        b = before.transform.apply(before.landmarks[target.id][None])[0]
        a = final.apply(after.landmarks[target.id][None])[0]
        residual = max(np.linalg.norm(a - target.xyz_mm), np.linalg.norm(b - target.xyz_mm))
        if residual > settings.tolerance_mm:
            raise ValueError(f"独立检查靶 {target.id} 超过毫米误差容限")
        checks.append({"id": target.id, "max_position_error_mm": float(residual), "delta_z_mm": float(a[2] - b[2])})
    threshold = max(settings.threshold_mm, max(abs(c["delta_z_mm"]) for c in checks))
    after_points = rigid.apply(after.points)
    box = settings.plot_box_mm
    p, q = before.points[inside(before.points, box)], after_points[inside(after_points, box)]
    if min(len(p), len(q)) < 9:
        raise ValueError("两期测区有效点不足，请核对靶标世界坐标和 plot_box_mm")
    progress(86, "生成高程差、语义图谱与分区指标")
    comparison = compare_point_clouds(p, q, grid_step_mm=settings.grid_step_mm,
                                      erosion_threshold_mm=threshold, elapsed_hours=settings.elapsed_hours,
                                      bounds_xy_mm=(box[0], box[1], box[3], box[4]))
    coverage = float(comparison.valid.mean())
    if coverage < settings.min_coverage:
        raise ValueError(f"两期有效测区覆盖率仅 {coverage:.1%}，未达到设定值")
    votes = []
    for view_points, labels in after.views:
        aligned = rigid.apply(view_points)
        selection = inside(aligned, box)
        votes.append((aligned[selection], labels[selection]))
    labels = project_label_views(comparison, votes) if votes else None
    fusion = analyze_fusion(comparison, labels, source="manual" if masks else "model", threshold_mm=threshold)
    diagnostics = {
        "backend": "VGGT-Ω", "coordinate_system": "survey frame, XYZ mm, Z up",
        "settings": settings.model_dump(), "images": {k: len(v) for k, v in images.items()},
        "before": before.calibration, "after": after.calibration,
        "before_to_mm": before.transform.as_dict(), "after_to_mm": final.as_dict(),
        "stable_icp": refinement, "check_targets": checks, "common_coverage": coverage,
        "effective_threshold_mm": threshold,
    }
    return SurveyResult(comparison, fusion, diagnostics)
