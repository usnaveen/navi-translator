"""Na'vi Language Translator — FastAPI Backend (stub for Phase 1)."""

import time

from fastapi import FastAPI

app = FastAPI(
    title="Na'vi Language Translator",
    description="Translates Na'vi speech/text to English",
    version="0.1.0",
)

_start_time = time.time()


@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_version": "none",
        "uptime_s": round(time.time() - _start_time, 1),
    }


@app.get("/ready")
def ready():
    return {
        "ready": False,
        "whisper_loaded": False,
        "marian_loaded": False,
    }


@app.get("/metrics")
def metrics():
    return "# No metrics yet\n"
