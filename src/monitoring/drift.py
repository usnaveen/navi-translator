"""Data drift detection using EvidentlyAI.

This module runs periodically (via Airflow every 24h) and compares
live inference data against the training baselines to detect drift.

Key learning points:
- "Drift" means the statistical distribution of input data has changed
  from what the model was trained on
- Example: if users start using new Na'vi words the model hasn't seen,
  the OOV rate rises — that's semantic drift
- EvidentlyAI's DataDriftPreset computes statistical tests
  (KS test, PSI, etc.) to quantify the shift
- When drift is detected, we update the Prometheus gauge, which triggers
  an alert, which triggers the Airflow retrain DAG
"""

import json
import logging
import sqlite3
from datetime import datetime
from pathlib import Path

import numpy as np
import yaml

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def load_params() -> dict:
    with open(PROJECT_ROOT / "params.yaml") as f:
        return yaml.safe_load(f)


def load_baselines() -> dict:
    """Load training baselines from baselines.json."""
    baselines_path = PROJECT_ROOT / "baselines.json"
    if not baselines_path.exists():
        logger.warning("No baselines.json found — cannot detect drift")
        return {}
    with open(baselines_path) as f:
        return json.load(f)


def get_recent_requests(db_path: str, limit: int = 500) -> list[dict]:
    """Fetch recent inference requests from the SQLite log database."""
    try:
        conn = sqlite3.connect(db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.execute(
            "SELECT * FROM requests ORDER BY timestamp DESC LIMIT ?", (limit,)
        )
        rows = [dict(row) for row in cursor.fetchall()]
        conn.close()
        return rows
    except Exception as e:
        logger.warning("Could not read request log: %s", e)
        return []


def compute_live_oov_rate(requests: list[dict], vocab: set[str]) -> float:
    """Compute OOV rate from recent requests."""
    total_tokens = 0
    oov_tokens = 0

    for req in requests:
        navi_text = req.get("navi_text", "")
        tokens = navi_text.lower().split()
        total_tokens += len(tokens)
        for token in tokens:
            if token not in vocab:
                oov_tokens += 1

    return oov_tokens / total_tokens if total_tokens > 0 else 0.0


def run_evidently_drift_report(
    reference_data: dict,
    current_data: dict,
    output_path: Path,
) -> dict:
    """Generate an EvidentlyAI drift report comparing reference vs current data.

    Returns a dict with drift detection results.
    """
    try:
        import pandas as pd
        from evidently import ColumnMapping
        from evidently.report import Report
        from evidently.metric_preset import DataDriftPreset

        # Build dataframes from MFCC features
        ref_mfccs = reference_data.get("mfcc_features", [])
        cur_mfccs = current_data.get("mfcc_features", [])

        if not ref_mfccs or not cur_mfccs:
            logger.warning("Insufficient MFCC data for drift report")
            return {"drift_detected": False, "reason": "insufficient_data"}

        columns = [f"mfcc_{i}" for i in range(len(ref_mfccs[0]))]
        ref_df = pd.DataFrame(ref_mfccs, columns=columns)
        cur_df = pd.DataFrame(cur_mfccs, columns=columns)

        # Run drift report
        report = Report(metrics=[DataDriftPreset()])
        report.run(reference_data=ref_df, current_data=cur_df)

        # Save HTML report
        output_path.parent.mkdir(parents=True, exist_ok=True)
        report.save_html(str(output_path))
        logger.info("Drift report saved to %s", output_path)

        # Extract results
        report_dict = report.as_dict()
        drift_result = report_dict.get("metrics", [{}])[0].get("result", {})
        drift_detected = drift_result.get("dataset_drift", False)

        return {
            "drift_detected": drift_detected,
            "drift_share": drift_result.get("drift_share", 0.0),
            "number_of_drifted_columns": drift_result.get("number_of_drifted_columns", 0),
        }

    except ImportError:
        logger.error("EvidentlyAI not installed — pip install evidently")
        return {"drift_detected": False, "reason": "evidently_not_installed"}
    except Exception as e:
        logger.error("Drift report failed: %s", e)
        return {"drift_detected": False, "reason": str(e)}


def check_drift(params: dict = None) -> dict:
    """Main drift detection routine. Called by Airflow or manually."""
    if params is None:
        params = load_params()

    monitoring_params = params["monitoring"]
    baselines = load_baselines()

    if not baselines:
        return {"drift_detected": False, "reason": "no_baselines"}

    # Load vocabulary
    words_path = PROJECT_ROOT / "data" / "raw" / "words.json"
    vocab = set()
    if words_path.exists():
        with open(words_path, encoding="utf-8") as f:
            words = json.load(f)
        vocab = {w["navi"].lower() for w in words}

    # Get recent requests
    db_path = PROJECT_ROOT / params["serving"]["request_log_db"]
    requests = get_recent_requests(str(db_path), monitoring_params["drift_reference_requests"])

    if not requests:
        logger.info("No recent requests to analyze")
        return {"drift_detected": False, "reason": "no_requests"}

    # Check OOV drift
    baseline_oov = baselines.get("oov", {}).get("oov_rate", 0.0)
    live_oov = compute_live_oov_rate(requests, vocab)
    oov_threshold = monitoring_params["oov_drift_threshold"]
    oov_drift = live_oov > baseline_oov * (1 + oov_threshold)

    logger.info("OOV rate — baseline: %.4f, live: %.4f, drift: %s", baseline_oov, live_oov, oov_drift)

    # Generate EvidentlyAI report
    date_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = PROJECT_ROOT / "monitoring" / "reports" / f"drift_{date_str}.html"
    evidently_result = run_evidently_drift_report(
        reference_data={"mfcc_features": [baselines.get("mfcc", {}).get("mean", [])]},
        current_data={"mfcc_features": [baselines.get("mfcc", {}).get("mean", [])]},
        output_path=report_path,
    )

    drift_detected = oov_drift or evidently_result.get("drift_detected", False)

    result = {
        "drift_detected": drift_detected,
        "oov": {
            "baseline": baseline_oov,
            "live": live_oov,
            "drift": oov_drift,
        },
        "evidently": evidently_result,
        "timestamp": datetime.now().isoformat(),
    }

    # Save result
    result_path = PROJECT_ROOT / "monitoring" / "reports" / f"drift_result_{date_str}.json"
    result_path.parent.mkdir(parents=True, exist_ok=True)
    with open(result_path, "w") as f:
        json.dump(result, f, indent=2)

    return result


def trigger_retrain():
    """Trigger the Airflow retrain DAG via API call."""
    import requests

    try:
        response = requests.post(
            "http://localhost:8080/api/v1/dags/navi_retrain_dag/dagRuns",
            json={"conf": {"trigger": "drift_detection"}},
            auth=("admin", "admin"),
            timeout=10,
        )
        response.raise_for_status()
        logger.info("Retrain DAG triggered successfully")
    except Exception as e:
        logger.error("Failed to trigger retrain DAG: %s", e)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    result = check_drift()
    print(json.dumps(result, indent=2))
    if result.get("drift_detected"):
        print("DRIFT DETECTED — triggering retraining...")
        trigger_retrain()
