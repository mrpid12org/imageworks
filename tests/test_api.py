import io
import numpy as np
from PIL import Image
import pytest

try:
    from fastapi.testclient import TestClient
    from imageworks.apps.competition_checker.api.main import app

    client = TestClient(app)
except Exception:  # pragma: no cover - optional dependency in dev
    TestClient = None
    client = None


def _png_bytes(rgb: np.ndarray) -> bytes:
    buf = io.BytesIO()
    Image.fromarray(rgb).save(buf, format="PNG")
    return buf.getvalue()


@pytest.mark.skipif(client is None, reason="httpx not installed for TestClient")
def test_healthz():
    r = client.get("/healthz")
    assert r.status_code == 200 and r.json().get("status") == "ok"


@pytest.mark.skipif(client is None, reason="httpx not installed for TestClient")
def test_mono_check_upload():
    # Neutral gray 32x32
    rgb = np.full((32, 32, 3), 128, np.uint8)
    files = {"image": ("im.png", _png_bytes(rgb), "image/png")}
    r = client.post("/mono/check", files=files)
    assert r.status_code == 200
    data = r.json()
    assert data["verdict"] == "pass" and data["mode"] == "neutral"
