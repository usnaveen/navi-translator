"""Baseline statistics — DVC stage 'baseline_stats'.

Computes statistical baselines from the training data that will later
be used by EvidentlyAI to detect data drift in production.

Key learning points:
- Drift detection compares live data distributions against a baseline
- We compute: MFCC stats, OOV rate, audio duration distributions
- These baselines are saved to baselines.json (tracked as a DVC metric)
- When the monitoring system sees values deviating significantly from
  these baselines, it triggers a retraining pipeline
- This is the core of the "living feedback loop" in the spec
"""

import json
import logging
import sys
from pathlib import Path

import librosa
import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
)
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def load_params() -> dict:
    with open(PROJECT_ROOT / "params.yaml") as f:
        return yaml.safe_load(f)


def compute_mfcc_baselines(audio_dir: Path, manifest: list[dict], sr: int) -> dict:
    """Compute mean and std of MFCC features across training audio.

    MFCCs (Mel-Frequency Cepstral Coefficients) capture the spectral
    shape of audio — essentially which frequencies are present and how
    loud they are. They're the standard feature for speech recognition.

    If the distribution of MFCCs shifts (e.g., different microphone,
    accent, or background noise), that's "data drift".
    """
    all_mfccs = []

    for item in manifest:
        filepath = audio_dir / item["filename"]
        if not filepath.exists():
            continue

        try:
            audio, _ = librosa.load(str(filepath), sr=sr, mono=True)
            mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)
            # Take mean across time axis for each coefficient
            mean_mfccs = np.mean(mfccs, axis=1)
            all_mfccs.append(mean_mfccs)
        except Exception as e:
            logger.warning("Failed to compute MFCC for %s: %s", filepath.name, e)

    if not all_mfccs:
        return {"mean": [0.0] * 13, "std": [0.0] * 13, "count": 0}

    all_mfccs = np.array(all_mfccs)
    return {
        "mean": np.mean(all_mfccs, axis=0).tolist(),
        "std": np.std(all_mfccs, axis=0).tolist(),
        "count": len(all_mfccs),
    }


def compute_oov_rate(val_pairs_path: Path, words_path: Path) -> dict:
    """Compute Out-Of-Vocabulary rate on the validation set.

    OOV rate = fraction of Na'vi tokens in validation data that don't
    appear in words.json. A rising OOV rate in production means users
    are using new Na'vi words that the model hasn't seen.
    """
    import csv

    # Load known vocabulary
    with open(words_path, encoding="utf-8") as f:
        words = json.load(f)
    vocab = {w["navi"].lower() for w in words}

    # Count OOV tokens in validation set
    total_tokens = 0
    oov_tokens = 0

    with open(val_pairs_path, encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            tokens = row["navi"].lower().split()
            total_tokens += len(tokens)
            for token in tokens:
                if token not in vocab:
                    oov_tokens += 1

    oov_rate = oov_tokens / total_tokens if total_tokens > 0 else 0.0

    return {
        "oov_rate": round(oov_rate, 4),
        "total_tokens": total_tokens,
        "oov_tokens": oov_tokens,
        "vocab_size": len(vocab),
    }


def compute_audio_duration_stats(manifest: list[dict]) -> dict:
    """Compute audio duration distribution statistics."""
    if not manifest:
        return {"mean": 0.0, "std": 0.0, "count": 0}

    durations = [m["duration_s"] for m in manifest]
    return {
        "mean": round(float(np.mean(durations)), 3),
        "std": round(float(np.std(durations)), 3),
        "p25": round(float(np.percentile(durations, 25)), 3),
        "p75": round(float(np.percentile(durations, 75)), 3),
        "min": round(float(min(durations)), 3),
        "max": round(float(max(durations)), 3),
        "count": len(durations),
    }


def main():
    params = load_params()
    data_params = params["data"]

    train_dir = PROJECT_ROOT / "data" / "processed" / "train"
    val_dir = PROJECT_ROOT / "data" / "processed" / "val"
    audio_dir = PROJECT_ROOT / "data" / "processed" / "audio"
    words_path = PROJECT_ROOT / "data" / "raw" / "words.json"
    plots_dir = PROJECT_ROOT / "data" / "processed" / "baseline_plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    baselines = {}

    # --- MFCC baselines from training audio ---
    logger.info("Computing MFCC baselines...")
    audio_manifest_path = train_dir / "audio_manifest.json"
    if audio_manifest_path.exists():
        with open(audio_manifest_path) as f:
            train_audio_manifest = json.load(f)
        baselines["mfcc"] = compute_mfcc_baselines(
            audio_dir, train_audio_manifest, data_params["audio_sample_rate"]
        )
        logger.info("  MFCC computed from %d files", baselines["mfcc"]["count"])
    else:
        logger.warning("No training audio manifest — skipping MFCC baselines")
        baselines["mfcc"] = {"mean": [0.0] * 13, "std": [0.0] * 13, "count": 0}

    # --- OOV rate on validation set ---
    logger.info("Computing OOV rate...")
    val_pairs_path = val_dir / "pairs.tsv"
    if val_pairs_path.exists():
        baselines["oov"] = compute_oov_rate(val_pairs_path, words_path)
        logger.info("  OOV rate: %.4f", baselines["oov"]["oov_rate"])
    else:
        logger.warning("No validation pairs — skipping OOV baseline")
        baselines["oov"] = {"oov_rate": 0.0, "total_tokens": 0, "oov_tokens": 0}

    # --- Audio duration distribution ---
    logger.info("Computing audio duration stats...")
    if audio_manifest_path.exists():
        baselines["audio_duration"] = compute_audio_duration_stats(train_audio_manifest)
        logger.info("  Mean duration: %.3fs", baselines["audio_duration"]["mean"])
    else:
        baselines["audio_duration"] = {"mean": 0.0, "std": 0.0, "count": 0}

    # --- Save baselines ---
    baselines_path = PROJECT_ROOT / "baselines.json"
    with open(baselines_path, "w") as f:
        json.dump(baselines, f, indent=2)

    logger.info("=== Baseline statistics saved to %s ===", baselines_path)
    logger.info("These values will be used by EvidentlyAI for drift detection")


if __name__ == "__main__":
    main()
