"""Prometheus metrics exporter — instruments the FastAPI backend.

Prometheus works by "scraping" — it polls the /metrics endpoint every 10-15s
and stores the values in a time-series database. Grafana then queries
Prometheus to build dashboards.

Key learning points:
- Counter: only goes up (e.g., total requests). Use rate() in Prometheus to get requests/sec
- Histogram: tracks distributions (e.g., latency). Auto-creates buckets + sum + count
- Gauge: can go up AND down (e.g., current OOV rate, current model version)
- Info: key-value metadata (e.g., which model version is loaded)
- generate_latest() serializes all metrics into Prometheus text format
"""

from prometheus_client import (
    CollectorRegistry,
    Counter,
    Gauge,
    Histogram,
    Info,
    generate_latest,
)

# Use a custom registry to avoid conflicts with default metrics
REGISTRY = CollectorRegistry()

# --- Counters ---
TRANSLATION_REQUESTS = Counter(
    "navi_translation_requests_total",
    "Total translation requests",
    ["input_type", "status"],  # labels: audio/text, success/error
    registry=REGISTRY,
)

VOCAB_SUBMISSIONS = Counter(
    "navi_vocab_submissions_total",
    "Total community vocabulary submissions received",
    registry=REGISTRY,
)

# --- Histogram ---
TRANSLATION_LATENCY = Histogram(
    "navi_translation_latency_ms",
    "End-to-end translation latency in milliseconds",
    buckets=[50, 100, 250, 500, 1000, 2000, 3000, 5000, 10000],
    registry=REGISTRY,
)

# --- Gauges ---
WHISPER_WER_LIVE = Gauge(
    "navi_whisper_wer_live",
    "Running WER estimate on recent requests with ground truth",
    registry=REGISTRY,
)

MARIAN_BLEU_LIVE = Gauge(
    "navi_marian_bleu_live",
    "Running BLEU estimate on recent text requests",
    registry=REGISTRY,
)

OOV_RATE = Gauge(
    "navi_oov_rate",
    "Fraction of Na'vi tokens in last N requests not in words.json",
    registry=REGISTRY,
)

FALLBACK_RATE = Gauge(
    "navi_fallback_rate",
    "Fraction of requests that fell back to Reykunyu word lookup",
    registry=REGISTRY,
)

# --- Info ---
MODEL_VERSION = Info(
    "navi_model_version",
    "Currently loaded model version string from MLflow registry",
    registry=REGISTRY,
)

# --- Quality metrics ---
CONFIDENCE_HIST = Histogram(
    "navi_translation_confidence",
    "Distribution of combined translation confidence scores",
    buckets=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 1.0],
    labelnames=["input_type"],
    registry=REGISTRY,
)

HIGH_CONFIDENCE_TOTAL = Counter(
    "navi_high_confidence_total",
    "Translations with combined confidence >= 0.7",
    ["input_type"],
    registry=REGISTRY,
)

# --- Drift / data-quality metric ---
OOV_WORDS = Counter(
    "navi_oov_words_total",
    "Count of Na'vi tokens not found in the dictionary, labeled by word",
    ["word"],
    registry=REGISTRY,
)

# --- Fuzzy correction (phonetic post-correction layer) ---
FUZZY_CORRECTIONS = Counter(
    "navi_fuzzy_corrections_total",
    "Tokens rescued by the fuzzy-match (Levenshtein) post-correction layer",
    ["edit_distance"],
    registry=REGISTRY,
)

TOKENS_PROCESSED = Counter(
    "navi_tokens_processed_total",
    "Total Na'vi tokens that passed through the dictionary stage",
    registry=REGISTRY,
)


def get_metrics() -> bytes:
    """Generate Prometheus text format metrics."""
    return generate_latest(REGISTRY)


def record_translation(input_type: str, status: str, latency_ms: float):
    """Record a translation request."""
    TRANSLATION_REQUESTS.labels(input_type=input_type, status=status).inc()
    TRANSLATION_LATENCY.observe(latency_ms)


def record_vocab_submission():
    """Record a vocabulary submission."""
    VOCAB_SUBMISSIONS.inc()


def update_oov_rate(rate: float):
    """Update the current OOV rate gauge."""
    OOV_RATE.set(rate)


def update_fallback_rate(rate: float):
    """Update the fallback rate gauge."""
    FALLBACK_RATE.set(rate)


def update_model_version(version: str):
    """Update the model version info."""
    MODEL_VERSION.info({"version": version})


def record_confidence(input_type: str, confidence: float):
    """Record a translation's confidence score and bump high-conf counter."""
    CONFIDENCE_HIST.labels(input_type=input_type).observe(confidence)
    if confidence >= 0.7:
        HIGH_CONFIDENCE_TOTAL.labels(input_type=input_type).inc()


def record_fuzzy_corrections(breakdown: list[dict]):
    """Track tokens corrected by the Levenshtein post-correction layer."""
    for entry in breakdown:
        TOKENS_PROCESSED.inc()
        d = entry.get("edit_distance", 0)
        if d > 0:
            FUZZY_CORRECTIONS.labels(edit_distance=str(d)).inc()


# --- User activity ---
REQUESTS_BY_USER = Counter(
    "navi_requests_by_user_total",
    "Translation requests per user IP",
    ["user_ip", "input_type"],
    registry=REGISTRY,
)

OOV_BY_USER = Counter(
    "navi_oov_by_user_total",
    "OOV tokens submitted per user IP — high value = likely gibberish sender",
    ["user_ip"],
    registry=REGISTRY,
)

VOCAB_BY_USER = Counter(
    "navi_vocab_by_user_total",
    "Vocabulary contributions per user IP",
    ["user_ip"],
    registry=REGISTRY,
)

# --- Service load ---
ACTIVE_REQUESTS = Gauge(
    "navi_active_requests",
    "Currently in-flight translation requests",
    registry=REGISTRY,
)

PROCESS_MEMORY_MB = Gauge(
    "navi_process_memory_mb",
    "RSS memory usage of the backend process in MB",
    registry=REGISTRY,
)

PROCESS_CPU_PERCENT = Gauge(
    "navi_process_cpu_percent",
    "CPU utilisation of the backend process (percent)",
    registry=REGISTRY,
)


def record_user_request(user_ip: str, input_type: str):
    REQUESTS_BY_USER.labels(user_ip=user_ip, input_type=input_type).inc()


def record_oov_by_user(user_ip: str, count: int = 1):
    if count > 0:
        OOV_BY_USER.labels(user_ip=user_ip).inc(count)


def record_vocab_by_user(user_ip: str):
    VOCAB_BY_USER.labels(user_ip=user_ip).inc()


def update_process_metrics():
    """Refresh process-level CPU and memory gauges (call periodically)."""
    try:
        import os
        import psutil
        proc = psutil.Process(os.getpid())
        PROCESS_MEMORY_MB.set(proc.memory_info().rss / 1024 / 1024)
        PROCESS_CPU_PERCENT.set(proc.cpu_percent(interval=None))
    except Exception:
        pass


def record_oov_words(words: list[str]):
    """Increment OOV counter for each unresolved Na'vi token.

    We cap label cardinality by truncating very long tokens (Prometheus
    advice: keep label values short and bounded).
    """
    for w in words:
        if not w:
            continue
        # Strip non-alphanumeric noise and bound length to keep cardinality sane
        clean = "".join(c for c in w.lower() if c.isalnum() or c in "ìäáéíóúñ")[:20]
        if clean:
            OOV_WORDS.labels(word=clean).inc()
