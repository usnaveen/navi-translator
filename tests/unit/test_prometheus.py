"""Unit tests for Prometheus metric registration."""

from src.monitoring.prometheus_exporter import (
    FALLBACK_RATE,
    OOV_RATE,
    TRANSLATION_LATENCY,
    TRANSLATION_REQUESTS,
    VOCAB_SUBMISSIONS,
    get_metrics,
    record_translation,
    record_vocab_submission,
)


class TestMetricRegistration:
    def test_metrics_endpoint_returns_bytes(self):
        result = get_metrics()
        assert isinstance(result, bytes)

    def test_record_translation(self):
        record_translation("text", "success", 150.0)
        metrics_text = get_metrics().decode()
        assert "navi_translation_requests_total" in metrics_text
        assert "navi_translation_latency_ms" in metrics_text

    def test_record_vocab_submission(self):
        record_vocab_submission()
        metrics_text = get_metrics().decode()
        assert "navi_vocab_submissions_total" in metrics_text

    def test_all_metrics_present(self):
        metrics_text = get_metrics().decode()
        expected = [
            "navi_translation_requests_total",
            "navi_translation_latency_ms",
            "navi_oov_rate",
            "navi_fallback_rate",
            "navi_vocab_submissions_total",
        ]
        for metric_name in expected:
            assert metric_name in metrics_text, f"Missing metric: {metric_name}"
