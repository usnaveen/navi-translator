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
