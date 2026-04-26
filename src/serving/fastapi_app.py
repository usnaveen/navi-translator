"""Na'vi Language Translator — FastAPI Backend.

This is the main API server. It loads models at startup and serves
translation requests. All endpoints return structured JSON.

Key learning points:
- @app.on_event("startup") runs code once when the server starts
- UploadFile handles file uploads (audio files for translation)
- Response model= tells FastAPI the expected output schema
- Middleware adds request_id to every request for traceability
- The /metrics endpoint returns Prometheus-formatted text
"""

import logging
import time
import uuid

from fastapi import FastAPI, File, HTTPException, Request, Response, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from src.monitoring.prometheus_exporter import (
    get_metrics,
    record_confidence,
    record_fuzzy_corrections,
    record_oov_words,
    record_translation,
    record_vocab_submission,
    update_fallback_rate,
    update_model_version,
    update_oov_rate,
)
from src.serving.inference import TranslationEngine
from src.serving.schemas import (
    AudioTranslationResponse,
    ErrorResponse,
    HealthResponse,
    ReadyResponse,
    TextTranslationRequest,
    TextTranslationResponse,
    VocabSubmissionRequest,
    VocabSubmissionResponse,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Na'vi Language Translator",
    description="Translates Na'vi speech and text to English",
    version="1.0.0",
)

# CORS: allow frontend to call the API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global state
engine = TranslationEngine()
_start_time = time.time()

# Track OOV and fallback rates
_recent_oov_counts: list[bool] = []
_recent_fallback_counts: list[bool] = []
_window_size = 100


@app.on_event("startup")
async def startup():
    """Load models when the server starts."""
    logger.info("Loading translation models...")
    try:
        engine.load_models()
        update_model_version(engine.model_version)
        logger.info("Server ready.")
    except Exception as e:
        logger.error("Failed to load models: %s", e)


@app.middleware("http")
async def add_request_id(request: Request, call_next):
    """Add a unique request_id to every request for traceability."""
    request.state.request_id = str(uuid.uuid4())[:8]
    response = await call_next(request)
    response.headers["X-Request-ID"] = request.state.request_id
    return response


# --- Translation Endpoints ---

@app.post("/translate/audio", response_model=AudioTranslationResponse)
async def translate_audio(request: Request, file: UploadFile = File(...)):
    """Translate Na'vi audio to English.

    Accepts WAV, MP3, or OGG audio files.
    Chain: audio → Whisper (ASR) → Na'vi text → MarianMT → English
    """
    request_id = request.state.request_id
    start = time.time()

    # Validate content type
    allowed_types = {"audio/wav", "audio/mpeg", "audio/ogg", "audio/x-wav", "audio/wave"}
    content_type = file.content_type or ""
    if content_type and content_type not in allowed_types and not file.filename.endswith((".wav", ".mp3", ".ogg")):
        raise HTTPException(
            status_code=415,
            detail=ErrorResponse(
                error="Unsupported Media Type",
                detail=f"Expected audio file, got {content_type}",
                request_id=request_id,
            ).model_dump(),
        )

    if not engine.is_ready:
        raise HTTPException(
            status_code=503,
            detail=ErrorResponse(
                error="Service Unavailable",
                detail="Models not loaded yet",
                request_id=request_id,
            ).model_dump(),
        )

    try:
        audio_bytes = await file.read()

        # Check audio length (rough estimate: 16kHz * 2 bytes * 30s = ~960KB)
        if len(audio_bytes) > 30 * 16000 * 4:
            raise HTTPException(
                status_code=413,
                detail=ErrorResponse(
                    error="Request Entity Too Large",
                    detail="Audio too long (> 30 seconds)",
                    request_id=request_id,
                ).model_dump(),
            )

        # Step 1: Transcribe audio → Na'vi text
        navi_text, asr_confidence = engine.transcribe_audio(audio_bytes)

        # Step 2: Translate Na'vi → English
        english, nmt_confidence, word_breakdown = engine.translate_text(navi_text)

        latency_ms = int((time.time() - start) * 1000)
        confidence = round((asr_confidence + nmt_confidence) / 2, 3)

        # Track metrics
        _track_oov(word_breakdown)
        _track_fallback(nmt_confidence < engine.params["marian"]["fallback_confidence_threshold"])
        record_translation("audio", "success", latency_ms)
        record_confidence("audio", confidence)
        record_oov_words([b["navi"] for b in word_breakdown if not b.get("found")])
        record_fuzzy_corrections(word_breakdown)

        # Reconstruct the corrected Na'vi sentence from breakdown for the UI
        corrected_text = " ".join(b.get("navi_corrected", b["navi"]) for b in word_breakdown)
        fuzzy_count = sum(1 for b in word_breakdown if b.get("fuzzy"))

        return AudioTranslationResponse(
            navi_text=navi_text,
            navi_corrected=corrected_text,
            english=english,
            confidence=confidence,
            asr_confidence=round(asr_confidence, 3),
            nmt_confidence=round(nmt_confidence, 3),
            fuzzy_corrections=fuzzy_count,
            latency_ms=latency_ms,
        )

    except HTTPException:
        raise
    except Exception as e:
        latency_ms = int((time.time() - start) * 1000)
        record_translation("audio", "error", latency_ms)
        logger.error("Translation error [%s]: %s", request_id, e, exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=ErrorResponse(
                error="Internal Server Error",
                detail=str(e),
                request_id=request_id,
            ).model_dump(),
        )


@app.post("/translate/text", response_model=TextTranslationResponse)
async def translate_text(request: Request, body: TextTranslationRequest):
    """Translate Na'vi text to English."""
    request_id = request.state.request_id
    start = time.time()

    if not engine.is_ready:
        raise HTTPException(
            status_code=503,
            detail=ErrorResponse(
                error="Service Unavailable",
                detail="Models not loaded yet",
                request_id=request_id,
            ).model_dump(),
        )

    try:
        english, confidence, word_breakdown = engine.translate_text(body.text)
        latency_ms = int((time.time() - start) * 1000)

        _track_oov(word_breakdown)
        _track_fallback(confidence < engine.params["marian"]["fallback_confidence_threshold"])
        record_translation("text", "success", latency_ms)
        record_confidence("text", confidence)
        record_oov_words([b["navi"] for b in word_breakdown if not b.get("found")])
        record_fuzzy_corrections(word_breakdown)

        return TextTranslationResponse(
            english=english,
            confidence=confidence,
            word_breakdown=word_breakdown,
            latency_ms=latency_ms,
        )

    except Exception as e:
        latency_ms = int((time.time() - start) * 1000)
        record_translation("text", "error", latency_ms)
        logger.error("Translation error [%s]: %s", request_id, e, exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


# --- Vocabulary Submission ---

@app.post("/vocab/submit", response_model=VocabSubmissionResponse)
async def vocab_submit(body: VocabSubmissionRequest):
    """Submit a new Na'vi word with optional audio."""
    # Validate Na'vi character set
    allowed = engine.params["data"]["navi_charset"]
    if not all(c in allowed for c in body.navi_word):
        return VocabSubmissionResponse(
            accepted=False,
            message="Word contains invalid Na'vi characters",
        )

    # Store submission (in production, this would write to data/raw/community/)
    record_vocab_submission()
    logger.info("Vocab submission: %s = %s", body.navi_word, body.english_meaning)

    return VocabSubmissionResponse(
        accepted=True,
        message=f"Thank you! '{body.navi_word}' has been submitted for review.",
    )


# --- Health & Monitoring ---

@app.get("/health", response_model=HealthResponse)
def health():
    return HealthResponse(
        status="ok",
        model_version=engine.model_version,
        uptime_s=round(time.time() - _start_time, 1),
    )


@app.get("/ready", response_model=ReadyResponse)
def ready():
    return ReadyResponse(
        ready=engine.is_ready,
        whisper_loaded=engine.whisper_loaded,
        marian_loaded=engine.marian_loaded,
    )


@app.get("/metrics")
def metrics():
    """Expose Prometheus metrics."""
    return Response(content=get_metrics(), media_type="text/plain; charset=utf-8")


# --- Internal helpers ---

def _track_oov(word_breakdown: list[dict]):
    """Update rolling OOV rate from word breakdown."""
    if not word_breakdown:
        return
    oov = any(not w.get("found", True) for w in word_breakdown)
    _recent_oov_counts.append(oov)
    if len(_recent_oov_counts) > _window_size:
        _recent_oov_counts.pop(0)
    rate = sum(_recent_oov_counts) / len(_recent_oov_counts)
    update_oov_rate(rate)


def _track_fallback(used_fallback: bool):
    """Update rolling fallback rate."""
    _recent_fallback_counts.append(used_fallback)
    if len(_recent_fallback_counts) > _window_size:
        _recent_fallback_counts.pop(0)
    rate = sum(_recent_fallback_counts) / len(_recent_fallback_counts)
    update_fallback_rate(rate)
