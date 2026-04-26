"""Unit tests for Pydantic schemas."""

import pytest
from pydantic import ValidationError

from src.serving.schemas import (
    AudioTranslationResponse,
    HealthResponse,
    TextTranslationRequest,
    VocabSubmissionRequest,
)


class TestTextTranslationRequest:
    def test_valid_request(self):
        req = TextTranslationRequest(text="kaltxì")
        assert req.text == "kaltxì"

    def test_empty_text_rejected(self):
        with pytest.raises(ValidationError):
            TextTranslationRequest(text="")

    def test_too_long_text_rejected(self):
        with pytest.raises(ValidationError):
            TextTranslationRequest(text="x" * 501)


class TestVocabSubmission:
    def test_valid_submission(self):
        req = VocabSubmissionRequest(navi_word="fya'o", english_meaning="path, way")
        assert req.navi_word == "fya'o"
        assert req.audio_b64 is None

    def test_with_audio(self):
        req = VocabSubmissionRequest(
            navi_word="fya'o", english_meaning="path", audio_b64="base64data"
        )
        assert req.audio_b64 == "base64data"


class TestAudioTranslationResponse:
    def test_valid_response(self):
        resp = AudioTranslationResponse(
            navi_text="kaltxì", english="hello", confidence=0.85, latency_ms=150
        )
        assert resp.confidence == 0.85

    def test_confidence_bounds(self):
        with pytest.raises(ValidationError):
            AudioTranslationResponse(
                navi_text="kaltxì", english="hello", confidence=1.5, latency_ms=150
            )


class TestHealthResponse:
    def test_health(self):
        resp = HealthResponse(status="ok", model_version="v1", uptime_s=123.4)
        assert resp.status == "ok"
