"""Smoke tests for TrichAI backend.

Run with: pytest tests/ -v

These don't require the actual ONNX model or R2 credentials. We patch
`inference.ensure_model_local` and `inference.InferenceEngine` so main
can import even when the model is absent. Visual traits are patched
because the fake JPEG/PNG byte sequences used here are not real images
and don't decode in Pillow.
"""
import sys
from unittest.mock import patch

import pytest


@pytest.fixture
def client():
    # Force a fresh main import so module-level side effects (model loading)
    # see our patches.
    sys.modules.pop("main", None)
    with patch("inference.InferenceEngine") as mock_engine, \
         patch("inference.ensure_model_local"):
        mock_engine.return_value.predict.return_value = {
            "label":        "bud",
            "display":      "Cogollo seco",
            "confidence":   0.9,
            "quality":      "Alta",
            "thc_min":      15,
            "thc_max":      30,
            "thc_estimate": 22,
            "description":  "test",
            "varieties":    ["Test"],
            "all_probs":    {"bud": 0.9, "hash": 0.05, "other": 0.03, "plant": 0.02},
        }
        from fastapi.testclient import TestClient
        import main
        with patch.object(main, "analyze_traits", return_value={
            "brightness":        50.0,
            "roughness":         30.0,
            "texture":           "Media",
            "saturation":        40.0,
            "trichome_coverage": 5.0,
            "trichomes":         "Media",
            "dominant_color":    [100, 80, 60],
            "cure":              "Bien curada",
            "warmth":            15.0,
            "uniformity":        "Media",
            "green_tint":        False,
        }):
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


# ── FEEDBACK VALIDATION ──────────────────────────────────────────────────────

def test_feedback_accepts_positive(client):
    r = client.post(
        "/feedback",
        files={"file": ("foo.jpg", _jpeg_bytes(), "image/jpeg")},
        data={"predicted": "bud", "is_correct": "true", "confidence": "0.92"},
    )
    assert r.status_code == 200, r.text
    assert r.json()["success"] is True


def test_feedback_requires_real_label_on_negative(client):
    r = client.post(
        "/feedback",
        files={"file": ("foo.jpg", _jpeg_bytes(), "image/jpeg")},
        data={"predicted": "bud", "is_correct": "false", "confidence": "0.55"},
    )
    assert r.status_code == 400


def test_feedback_accepts_negative_with_real_label(client):
    r = client.post(
        "/feedback",
        files={"file": ("foo.jpg", _jpeg_bytes(), "image/jpeg")},
        data={
            "predicted":   "bud",
            "is_correct":  "false",
            "real_label":  "hash",
            "confidence":  "0.55",
        },
    )
    assert r.status_code == 200, r.text


def test_feedback_rejects_invalid_predicted(client):
    r = client.post(
        "/feedback",
        files={"file": ("foo.jpg", _jpeg_bytes(), "image/jpeg")},
        data={"predicted": "garbage", "is_correct": "true"},
    )
    assert r.status_code == 400


# ── STATS PROTECTION (SEC-02: fail-closed) ───────────────────────────────────

def test_stats_requires_key_when_set(client, monkeypatch):
    monkeypatch.setattr("main.STATS_KEY", "secret-test-key")
    r = client.get("/stats")
    assert r.status_code == 403

    r2 = client.get("/stats", headers={"x-api-key": "secret-test-key"})
    assert r2.status_code == 200


def test_stats_fails_closed_when_no_key_set(client, monkeypatch):
    monkeypatch.setattr("main.STATS_KEY", "")
    r = client.get("/stats")
    # SEC-02: if no key configured, endpoint is disabled (503), not open.
    assert r.status_code == 503


# ── VISUAL TRAITS ────────────────────────────────────────────────────────────

def test_visual_traits_module_imports():
    from visual_traits import analyze
    assert callable(analyze)
