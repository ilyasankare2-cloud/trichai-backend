"""Smoke tests for TrichAI backend.

Run with: pytest tests/ -v
These don't require the model — they test the security middleware,
validators and endpoint shapes that we control directly.
"""
import io
import struct
from unittest.mock import patch
import pytest


@pytest.fixture
def client():
    """Lazy import so tests can run even if model file is missing."""
    with patch("inference.InferenceEngine") as mock_engine:
        mock_engine.return_value.predict.return_value = {
            "label": "bud",
            "display": "Cogollo seco",
            "confidence": 0.9,
            "quality": "Alta",
            "thc_min": 15,
            "thc_max": 30,
            "thc_estimate": 22,
            "description": "test",
            "varieties": ["Test"],
            "all_probs": {"bud": 0.9, "hash": 0.05, "other": 0.03, "plant": 0.02},
        }
        from fastapi.testclient import TestClient
        import main
        yield TestClient(main.app)


def _jpeg_bytes(payload: bytes = b"\x00" * 100) -> bytes:
    return b"\xff\xd8\xff" + payload


def _png_bytes(payload: bytes = b"\x00" * 100) -> bytes:
    return b"\x89PNG\r\n\x1a\n" + payload


# ── HEALTH ───────────────────────────────────────────────────────────────────

def test_health_returns_ok(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


# ── MAGIC BYTES VALIDATION ───────────────────────────────────────────────────

def test_analyze_rejects_text_file(client):
    r = client.post(
        "/analyze",
        files={"file": ("foo.jpg", b"not an image at all", "image/jpeg")},
    )
    assert r.status_code == 400
    assert "imagen" in r.json()["detail"].lower()


def test_analyze_rejects_unsupported_mime(client):
    r = client.post(
        "/analyze",
        files={"file": ("foo.gif", _jpeg_bytes(), "image/gif")},
    )
    assert r.status_code == 400


def test_analyze_accepts_jpeg(client):
    r = client.post(
        "/analyze",
        files={"file": ("foo.jpg", _jpeg_bytes(), "image/jpeg")},
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["success"] is True
    assert body["result"]["label"] == "bud"


def test_analyze_accepts_png(client):
    r = client.post(
        "/analyze",
        files={"file": ("foo.png", _png_bytes(), "image/png")},
    )
    assert r.status_code == 200


# ── CONTRIBUTE VALIDATION ────────────────────────────────────────────────────

def test_contribute_rejects_invalid_label(client):
    r = client.post(
        "/contribute",
        files={"file": ("foo.jpg", _jpeg_bytes(), "image/jpeg")},
        data={"label": "garbage"},
    )
    assert r.status_code == 400


def test_contribute_accepts_valid_label(client):
    r = client.post(
        "/contribute",
        files={"file": ("foo.jpg", _jpeg_bytes(), "image/jpeg")},
        data={"label": "hash"},
    )
    assert r.status_code == 200
    assert r.json()["label"] == "hash"


# ── STATS PROTECTION ─────────────────────────────────────────────────────────

def test_stats_requires_key_when_set(client, monkeypatch):
    monkeypatch.setattr("main.STATS_KEY", "secret-test-key")
    r = client.get("/stats")
    assert r.status_code == 403

    r2 = client.get("/stats", headers={"x-api-key": "secret-test-key"})
    assert r2.status_code == 200


def test_stats_open_when_no_key_set(client, monkeypatch):
    monkeypatch.setattr("main.STATS_KEY", "")
    r = client.get("/stats")
    assert r.status_code == 200


# ── VISUAL TRAITS ────────────────────────────────────────────────────────────

def test_visual_traits_module_imports():
    from visual_traits import analyze
    assert callable(analyze)
