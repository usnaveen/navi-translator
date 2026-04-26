"""Unit tests for drift detection.

Tests that drift detection correctly identifies when OOV rate
exceeds the baseline threshold.
"""

from src.monitoring.drift import compute_live_oov_rate


class TestOOVDrift:
    def test_no_oov(self):
        """All tokens in vocabulary — OOV rate should be 0."""
        vocab = {"kaltxì", "oel", "ngati"}
        requests = [{"navi_text": "kaltxì oel ngati"}]
        rate = compute_live_oov_rate(requests, vocab)
        assert rate == 0.0

    def test_full_oov(self):
        """No tokens in vocabulary — OOV rate should be 1."""
        vocab = {"kaltxì"}
        requests = [{"navi_text": "zxyw qrst"}]
        rate = compute_live_oov_rate(requests, vocab)
        assert rate == 1.0

    def test_partial_oov(self):
        """Mix of known and unknown tokens."""
        vocab = {"kaltxì", "oel"}
        requests = [{"navi_text": "kaltxì oel newword"}]
        rate = compute_live_oov_rate(requests, vocab)
        assert abs(rate - 1 / 3) < 0.01

    def test_empty_requests(self):
        """No requests — rate should be 0."""
        vocab = {"kaltxì"}
        rate = compute_live_oov_rate([], vocab)
        assert rate == 0.0

    def test_20pct_oov_spike_detected(self):
        """Simulate injecting 20% OOV tokens — should exceed threshold."""
        vocab = {"a", "b", "c", "d", "e", "f", "g", "h", "i", "j"}
        # 8 known + 2 unknown = 20% OOV
        requests = [{"navi_text": "a b c d e f g h unknown1 unknown2"}]
        rate = compute_live_oov_rate(requests, vocab)
        assert rate == 0.2

        baseline_oov = 0.05  # 5% baseline
        threshold = 0.15  # 15% increase triggers alert
        assert rate > baseline_oov * (1 + threshold)  # drift detected!
