from __future__ import annotations

from fastapi.testclient import TestClient

from services.serving.api.app import app


def test_health() -> None:
    client = TestClient(app)
    assert client.get("/health").json()["status"] == "ok"


def test_generate() -> None:
    client = TestClient(app)
    data = client.post("/generate", json={"prompt": "hi", "max_tokens": 5}).json()
    assert "generated" in data["text"]
