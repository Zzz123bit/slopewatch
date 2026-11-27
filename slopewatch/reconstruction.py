from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path


def find_colmap() -> str | None:
    configured = os.environ.get("COLMAP_EXE")
    if configured and Path(configured).is_file() and Path(configured).suffix.lower() != ".bat":
        return configured
    executable = shutil.which("colmap.exe") or shutil.which("colmap")
    if executable and Path(executable).suffix.lower() != ".bat":
        return executable
    return None


def colmap_status() -> dict:
    executable = find_colmap()
    return {"available": executable is not None, "executable": executable}


def _run(command: list[str], working_directory: Path, log_file: Path, timeout_seconds: int = 3600):
    result = subprocess.run(
        command,
        cwd=working_directory,
        capture_output=True,
        text=True,
        timeout=timeout_seconds,
        check=False,
    )
    with log_file.open("a", encoding="utf-8") as stream:
        stream.write("$ " + " ".join(command) + "\n")
        stream.write(result.stdout or "")
        stream.write(result.stderr or "")
        stream.write("\n")
    if result.returncode != 0:
        details = (result.stderr or result.stdout or "未知错误").strip()[-3000:]
        raise RuntimeError(f"COLMAP 命令失败。{details}")


def reconstruct_with_colmap(images_dir: Path, output_dir: Path, max_image_size: int = 1600) -> Path:
    executable = find_colmap()
    if executable is None:
        raise RuntimeError("未检测到 COLMAP，请安装 COLMAP 并配置 PATH 或 COLMAP_EXE")
    images_dir = images_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    database = output_dir / "database.db"
    sparse_dir = output_dir / "sparse"
    dense_dir = output_dir / "dense"
    sparse_dir.mkdir(exist_ok=True)
    log_file = output_dir / "colmap.log"
    for path in (database,):
        if path.exists():
            path.unlink()

    prefix = [executable]
    _run(
        prefix
        + [
            "feature_extractor",
            "--database_path",
            str(database),
            "--image_path",
            str(images_dir),
            "--ImageReader.single_camera",
            "1",
            "--ImageReader.camera_model",
            "SIMPLE_RADIAL",
        ],
        output_dir,
        log_file,
    )
    _run(prefix + ["exhaustive_matcher", "--database_path", str(database)], output_dir, log_file)
    _run(
        prefix
        + [
            "mapper",
            "--database_path",
            str(database),
            "--image_path",
            str(images_dir),
            "--output_path",
            str(sparse_dir),
        ],
        output_dir,
        log_file,
    )

    models = sorted(
        [folder for folder in sparse_dir.iterdir() if folder.is_dir()],
        key=lambda folder: sum(1 for _ in folder.iterdir()),
        reverse=True,
    )
    if not models:
        raise RuntimeError("COLMAP 没有生成稀疏模型，请检查照片重叠度和清晰度")

    dense_dir.mkdir(exist_ok=True)
    _run(
        prefix
        + [
            "image_undistorter",
            "--image_path",
            str(images_dir),
            "--input_path",
            str(models[0]),
            "--output_path",
            str(dense_dir),
            "--output_type",
            "COLMAP",
        ],
        output_dir,
        log_file,
    )
    _run(
        prefix
        + [
            "patch_match_stereo",
            "--workspace_path",
            str(dense_dir),
            "--workspace_format",
            "COLMAP",
            "--PatchMatchStereo.geom_consistency",
            "true",
            "--PatchMatchStereo.max_image_size",
            str(max_image_size),
        ],
        output_dir,
        log_file,
        timeout_seconds=7200,
    )
    cloud_path = dense_dir / "fused.ply"
    _run(
        prefix
        + [
            "stereo_fusion",
            "--workspace_path",
            str(dense_dir),
            "--workspace_format",
            "COLMAP",
            "--input_type",
            "geometric",
            "--output_path",
            str(cloud_path),
        ],
        output_dir,
        log_file,
        timeout_seconds=7200,
    )
    return cloud_path

