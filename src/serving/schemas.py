"""Pydantic schemas — define the shape of all API requests and responses.

Pydantic models validate data automatically. If someone sends
{ "text": 123 } to an endpoint expecting a string, FastAPI returns
a clear 422 error without you writing any validation code.

Key learning points:
- BaseModel defines a schema with typed fields
- Optional fields have default values
- FastAPI uses these schemas for request validation AND auto-generated docs
- The Swagger UI at /docs lets you test endpoints interactively
"""

from pydantic import BaseModel, Field


# --- Request Schemas ---

class TextTranslationRequest(BaseModel):
    """Request body for POST /translate/text."""

    text: str = Field(..., min_length=1, max_length=500, description="Na'vi text to translate")


class VocabSubmissionRequest(BaseModel):
    """Request body for POST /vocab/submit."""

    navi_word: str = Field(..., min_length=1, max_length=100, description="Na'vi word")
    english_meaning: str = Field(..., min_length=1, max_length=200, description="English meaning")
    audio_b64: str | None = Field(None, description="Base64-encoded audio recording (optional)")


# --- Response Schemas ---

class AudioTranslationResponse(BaseModel):
    """Response for POST /translate/audio."""

    navi_text: str = Field(..., description="Transcribed Na'vi text from audio")
    english: str = Field(..., description="English translation")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Translation confidence score")
    latency_ms: int = Field(..., description="End-to-end processing time in milliseconds")


class TextTranslationResponse(BaseModel):
    """Response for POST /translate/text."""

    english: str = Field(..., description="English translation")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Translation confidence score")
    word_breakdown: list[dict] = Field(default_factory=list, description="Per-word translation breakdown")
    latency_ms: int = Field(..., description="Processing time in milliseconds")


class VocabSubmissionResponse(BaseModel):
    """Response for POST /vocab/submit."""

    accepted: bool
    message: str


class HealthResponse(BaseModel):
    """Response for GET /health."""

    status: str
    model_version: str
    uptime_s: float


class ReadyResponse(BaseModel):
    """Response for GET /ready."""

    ready: bool
    whisper_loaded: bool
    marian_loaded: bool


class ErrorResponse(BaseModel):
    """Structured error response (used for all error cases)."""

    error: str
    detail: str
    request_id: str
