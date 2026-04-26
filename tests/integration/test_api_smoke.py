"""Integration smoke tests for the FastAPI translation API.

These tests keep model loading out of CI by swapping in a tiny fake engine.
The goal is to verify request routing, schemas, and endpoint wiring without
requiring GPU/MPS memory or downloading Hugging Face models.
"""

import pytest
from fastapi.testclient import TestClient

import src.serving.fastapi_app as fastapi_app


class FakeTranslationEngine:
    model_version = "test"
    params = {"marian": {"fallback_confidence_threshold": 0.4}}

    def load_models(self):
        return None

    @property
    def is_ready(self):
        return True

    @property
    def whisper_loaded(self):
        return True

    @property
    def marian_loaded(self):
        return True

    def translate_text(self, text: str):
        return (
            "hello",
            0.91,
            [{"navi": text, "en": "hello", "found": True}],
        )


@pytest.mark.integration
def test_translate_text_smoke(monkeypatch):
    monkeypatch.setattr(fastapi_app, "engine", FakeTranslationEngine())

    with TestClient(fastapi_app.app) as client:
        response = client.post("/translate/text", json={"text": "kaltxì"})

    assert response.status_code == 200
    body = response.json()
    assert body["english"] == "hello"
    assert body["confidence"] == 0.91
    assert body["word_breakdown"][0]["found"] is True


@pytest.mark.integration
def test_health_and_ready_smoke(monkeypatch):
    monkeypatch.setattr(fastapi_app, "engine", FakeTranslationEngine())

    with TestClient(fastapi_app.app) as client:
        health = client.get("/health")
        ready = client.get("/ready")

    assert health.status_code == 200
    assert health.json()["status"] == "ok"
    assert ready.status_code == 200
    assert ready.json()["ready"] is True
