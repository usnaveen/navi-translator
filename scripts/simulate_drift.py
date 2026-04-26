"""Simulate input distribution drift against the running translator backend.

Why this exists
---------------
Drift detection is one of the central MLOps concerns: the model was trained
on a fixed Reykunyu vocabulary (~3,230 words), but real-world traffic will
include novel constructions, typos, and code-switched English. We want a
reproducible way to *induce* that scenario on demand so reviewers can watch
the dashboard react in real time.

What it does
------------
Fires translation requests through three phases:

  Phase 1 — BASELINE   : real Na'vi phrases (high coverage, low fallback)
  Phase 2 — DRIFT      : pseudo-Na'vi gibberish (forces dictionary misses)
  Phase 3 — RECOVERY   : real Na'vi phrases again (metrics return to normal)

The phases are spaced so the Grafana panels (5-min rate windows) clearly
show the spike-and-recover pattern. Pair this script with the dashboard
open and the story tells itself.

Usage
-----
    python scripts/simulate_drift.py
    python scripts/simulate_drift.py --base http://localhost:8000 --duration 180
"""
from __future__ import annotations

import argparse
import random
import string
import sys
import time
from urllib import request, error

# Real Na'vi phrases the trained pipeline should handle gracefully.
BASELINE_PHRASES = [
    "kaltxì",
    "irayo",
    "kìyevame",
    "eywa",
    "tìyawn",
    "sìltsan",
    "oel ngati kameie",
    "nitram",
    "txon",
    "sran",
    "kehe",
    "fya'o",
    "nga yawne lu oer",
    "oeyä tsmukan",
    "ma frapo",
]

# Phonotactically Na'vi-shaped gibberish: real Na'vi consonant clusters
# (tx, kx, px, ng) and vowels (ì, ä) but combined into nonsense words
# nobody trained on. Looks like Na'vi to a casual eye but every token is OOV.
NAVI_PHONEMES_C = ["tx", "kx", "px", "ng", "ts", "fy", "sl", "fp", "kk"]
NAVI_PHONEMES_V = ["a", "e", "i", "o", "u", "ì", "ä"]


def _gibberish_token(rng: random.Random) -> str:
    parts = []
    for _ in range(rng.randint(2, 4)):
        parts.append(rng.choice(NAVI_PHONEMES_C))
        parts.append(rng.choice(NAVI_PHONEMES_V))
    return "".join(parts)


def _gibberish_phrase(rng: random.Random) -> str:
    n_words = rng.randint(1, 4)
    return " ".join(_gibberish_token(rng) for _ in range(n_words))


def _post_text(base_url: str, text: str, timeout: float = 30.0) -> dict | None:
    import json

    payload = json.dumps({"text": text}).encode("utf-8")
    req = request.Request(
        f"{base_url}/translate/text",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except (error.URLError, error.HTTPError, TimeoutError) as e:
        print(f"  ! request failed: {e}", file=sys.stderr)
        return None


def run_phase(label: str, phrases: list[str], rng: random.Random,
              base_url: str, duration_s: float, qps: float) -> dict:
    """Fire requests for *duration_s* seconds at the requested rate."""
    print(f"\n=== Phase: {label} ({duration_s:.0f}s @ {qps:.1f} req/s) ===")
    end = time.time() + duration_s
    interval = 1.0 / max(qps, 0.1)

    sent = 0
    high_conf = 0
    low_conf = 0

    while time.time() < end:
        phrase = rng.choice(phrases)
        result = _post_text(base_url, phrase)
        sent += 1
        if result:
            conf = result.get("confidence", 0.0)
            if conf >= 0.7:
                high_conf += 1
                tag = "✓"
            else:
                low_conf += 1
                tag = "✗"
            print(f"  {tag} {phrase[:30]:<30} conf={conf*100:5.1f}%  →  {result.get('english','')[:32]}")
        time.sleep(interval)

    print(f"  Phase done: {sent} requests, {high_conf} high-conf, {low_conf} low-conf")
    return {"sent": sent, "high_conf": high_conf, "low_conf": low_conf}


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--base", default="http://localhost:8000", help="backend base URL")
    p.add_argument("--duration", type=float, default=120.0, help="seconds per phase")
    p.add_argument("--qps", type=float, default=1.5, help="requests per second")
    p.add_argument("--seed", type=int, default=42, help="rng seed for reproducibility")
    args = p.parse_args()

    rng = random.Random(args.seed)
    drift_phrases = [_gibberish_phrase(rng) for _ in range(80)]

    # Sanity ping
    try:
        with request.urlopen(f"{args.base}/health", timeout=5) as r:
            health = r.read().decode()
            print(f"Backend health: {health[:200]}")
    except Exception as e:
        print(f"Cannot reach backend at {args.base}: {e}")
        return 1

    overall = {"baseline": None, "drift": None, "recovery": None}
    overall["baseline"] = run_phase("BASELINE (real Na'vi)", BASELINE_PHRASES, rng, args.base, args.duration, args.qps)
    overall["drift"]    = run_phase("DRIFT (gibberish)",      drift_phrases,    rng, args.base, args.duration, args.qps)
    overall["recovery"] = run_phase("RECOVERY (real Na'vi)",  BASELINE_PHRASES, rng, args.base, args.duration, args.qps)

    print("\n=== Summary ===")
    for k, v in overall.items():
        if v:
            ratio = v["high_conf"] / max(v["sent"], 1) * 100
            print(f"  {k:>10}: sent={v['sent']:>3}  high-conf%={ratio:5.1f}")

    print("\nNow open Grafana — you should see:")
    print("  • Top-10 OOV Words panel populated with new gibberish tokens")
    print("  • Fallback Rate spiked during the DRIFT window")
    print("  • Confidence histogram heatmap shifted left during DRIFT")
    print("  • High-Confidence Rate dipped from green to red and back")
    return 0


if __name__ == "__main__":
    sys.exit(main())
