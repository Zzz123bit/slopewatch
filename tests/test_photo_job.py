import io
import json
import zipfile

import numpy as np
from PIL import Image

from slopewatch import jobs
from slopewatch.vggt_recon import Reconstruction


def _zip_of_images(count: int) -> bytes:
    rng = np.random.default_rng(0)
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        for index in range(count):
            array = rng.integers(0, 255, (96, 96, 3), dtype=np.uint8)
            image = io.BytesIO()
            Image.fromarray(array).save(image, format="PNG")
            archive.writestr(f"view_{index:02d}.png", image.getvalue())
    return buffer.getvalue()


def _cloud(points: np.ndarray) -> Reconstruction:
    return Reconstruction(
        points=points.astype(np.float64),
        colors=np.full((len(points), 3), 128, dtype=np.uint8),
        confidence=np.ones(len(points), dtype=np.float32),
        stats={"views": 8, "points": int(len(points))},
    )


def test_photo_job_writes_result(monkeypatch, tmp_path):
    monkeypatch.setattr(jobs, "JOBS_ROOT", tmp_path / "jobs")
    jobs.JOBS_ROOT.mkdir(parents=True)

    rng = np.random.default_rng(1)
    n = 6000
    x = rng.uniform(0, 200, n)
    y = rng.uniform(0, 120, n)
    z = 60 - 0.2 * x + 2.0 * np.sin(x / 17) + 1.5 * np.cos(y / 11)
    before = np.column_stack([x, y, z])
    after = before.copy()
    after[:, 2] -= 8.0 * np.exp(-(((x - 120) / 18) ** 2 + ((y - 60) / 10) ** 2))

    inverse = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]])
    calls = {"n": 0}

    def fake_reconstruct(paths, **kwargs):
        calls["n"] += 1
        points = (before if calls["n"] == 1 else after) @ inverse.T
        return _cloud(points)

    monkeypatch.setattr(jobs, "reconstruct_vggt", fake_reconstruct)

    job_id, directory = jobs.create_job("photo_comparison", {})
    (directory / "before.zip").write_bytes(_zip_of_images(8))
    (directory / "after.zip").write_bytes(_zip_of_images(8))

    jobs.run_photo_job(
        job_id,
        "before.zip",
        "after.zip",
        grid_step_mm=1.0,
        erosion_threshold_mm=2.0,
        elapsed_hours=1.0,
        scale_mm_per_unit=1.0,
        max_views=8,
        metadata={"source": "unit-test"},
    )

    record = jobs.read_job(job_id)
    assert record["status"] == "complete", record.get("error")
    for name in ("metrics.json", "change.csv", "report.pdf", "surface.png", "dod.png", "surface.json"):
        assert (directory / name).is_file(), name
    metrics = json.loads((directory / "metrics.json").read_text(encoding="utf-8"))["metrics"]
    assert metrics["erosion_volume_l"] > 0
    assert metrics["max_erosion_depth_mm"] > 4.0
