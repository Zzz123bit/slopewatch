import secrets

import pytest
from fastapi.testclient import TestClient

import public_api
from api import app as local_app
from slopewatch import jobs


@pytest.fixture
def service(monkeypatch, tmp_path):
    token = secrets.token_urlsafe(32)
    monkeypatch.setenv("SLOPEWATCH_API_TOKEN", token)
    monkeypatch.setattr(jobs, "JOBS_ROOT", tmp_path)
    with TestClient(public_api.app) as client:
        yield client, {"Authorization": f"Bearer {token}"}


@pytest.mark.parametrize("token", ["", "short", "含中文" * 16])
def test_requires_configured_token(monkeypatch, token):
    monkeypatch.setenv("SLOPEWATCH_API_TOKEN", token)
    with pytest.raises(RuntimeError, match="SLOPEWATCH_API_TOKEN"), TestClient(public_api.app):
        pass


def test_auth_precedes_upload_and_docs_advertise_bearer(service):
    client, headers = service
    assert client.get("/livez").json() == {"service": "slopewatch-public", "status": "ok"}
    assert client.get("/docs").status_code == 200
    schema = client.get("/openapi.json").json()
    security = schema["paths"]["/api/v1/jobs/compare"]["post"]["security"]
    assert {"TeamToken": []} in security
    response = client.post(
        "/api/v1/jobs/compare",
        headers={"Content-Type": "multipart/form-data", "Authorization": "Bearer invalid"},
        content=b"",
    )
    assert response.status_code == 401
    assert response.headers["www-authenticate"] == "Bearer"
    assert client.get("/api/v1/health").status_code == 401
    assert set(client.get("/api/v1/health", headers=headers).json()["colmap"]) == {"available"}
    with TestClient(local_app) as local:
        assert local.get("/api/v1/health").status_code == 200


def test_remote_demo_status_result_and_download(service):
    client, headers = service
    response = client.post("/api/v1/jobs/demo", headers=headers)
    assert response.status_code == 202
    url = response.json()["status_url"]
    assert client.get(url, headers=headers).json()["status"] == "complete"
    assert client.get(url + "/result", headers=headers).json()["metrics"]["valid_cells"] > 0
    for path in [url, url + "/result", url + "/files/report.pdf"]:
        assert client.get(path).status_code == 401
    assert client.get(url + "/files/report.pdf", headers=headers).content.startswith(b"%PDF")
    assert client.get(url + "/files/job.json", headers=headers).status_code == 404


def test_private_log_and_task_errors_stay_local(service):
    client, headers = service
    job_id, directory = jobs.create_job("photo_reconstruction")
    log_name = "reconstruction/colmap.log"
    (directory / "reconstruction").mkdir()
    (directory / log_name).write_text("diagnostic")
    jobs.update_job(job_id, status="complete", result={"files": [log_name]})
    url = f"/api/v1/jobs/{job_id}"
    assert client.get(url + "/result", headers=headers).json()["files"] == []
    assert client.get(url + "/files/" + log_name, headers=headers).status_code == 404
    jobs.update_job(job_id, status="failed", error="internal-diagnostic-path")
    assert "internal-diagnostic-path" not in client.get(url, headers=headers).text
