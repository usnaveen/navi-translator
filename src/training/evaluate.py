"""Model evaluation and promotion — compares against registry models.

This script:
1. Loads a trained model from an MLflow run
2. Evaluates it on the test set (WER for Whisper, BLEU for MarianMT)
3. Compares against the current 'Production' model in MLflow registry
4. Promotes the new model if it's better by the threshold defined in params.yaml

Key learning points:
- MLflow Model Registry has stages: None → Staging → Production → Archived
- We compare the new model's metric against the current Production model
- Auto-promotion only happens if improvement exceeds a threshold (prevents noise)
- This is a core MLOps pattern: automated model validation gates
"""

import argparse
import csv
import json
import logging
import sys
from pathlib import Path

import mlflow
import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def load_params(config_path: str = None) -> dict:
    path = config_path or str(PROJECT_ROOT / "params.yaml")
    with open(path) as f:
        return yaml.safe_load(f)


def evaluate_whisper(run_id: str, params: dict) -> float:
    """Evaluate Whisper model on test set and return WER."""
    from jiwer import wer
    from transformers import WhisperForConditionalGeneration, WhisperProcessor
    import librosa

    sr = params["data"]["audio_sample_rate"]
    test_dir = PROJECT_ROOT / "data" / "processed" / "test"
    audio_dir = PROJECT_ROOT / "data" / "processed" / "audio"

    manifest_path = test_dir / "audio_manifest.json"
    if not manifest_path.exists():
        logger.warning("No test audio manifest")
        return 1.0

    with open(manifest_path) as f:
        manifest = json.load(f)

    # Load words for reference transcriptions
    with open(PROJECT_ROOT / "data" / "raw" / "words.json", encoding="utf-8") as f:
        words = json.load(f)
    word_map = {w["navi"].lower(): w["navi"] for w in words}

    # Load model from MLflow artifacts
    artifact_path = mlflow.artifacts.download_artifacts(run_id=run_id, artifact_path="whisper-navi-lora")
    processor = WhisperProcessor.from_pretrained(artifact_path)
    model = WhisperForConditionalGeneration.from_pretrained(artifact_path)

    predictions = []
    references = []

    for item in manifest:
        filepath = audio_dir / item["filename"]
        if not filepath.exists():
            continue

        reference = word_map.get(filepath.stem.lower(), filepath.stem)

        try:
            audio, _ = librosa.load(str(filepath), sr=sr, mono=True)
            input_features = processor.feature_extractor(
                audio, sampling_rate=sr, return_tensors="pt"
            ).input_features

            predicted_ids = model.generate(input_features)
            transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]

            predictions.append(transcription)
            references.append(reference)
        except Exception as e:
            logger.warning("Eval failed for %s: %s", filepath.name, e)

    if not predictions:
        return 1.0

    wer_score = wer(references, predictions)
    logger.info("Test WER: %.4f (%d samples)", wer_score, len(predictions))
    return round(wer_score, 4)


def evaluate_marian(run_id: str, params: dict) -> float:
    """Evaluate MarianMT model on test set and return BLEU."""
    import sacrebleu
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

    test_dir = PROJECT_ROOT / "data" / "processed" / "test"
    pairs_path = test_dir / "pairs.tsv"

    if not pairs_path.exists():
        logger.warning("No test pairs")
        return 0.0

    # Load test data
    sources = []
    references = []
    with open(pairs_path, encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            sources.append(row["navi"])
            references.append(row["en"])

    # Load model from MLflow
    artifact_path = mlflow.artifacts.download_artifacts(run_id=run_id, artifact_path="marian-navi-en")
    tokenizer = AutoTokenizer.from_pretrained(artifact_path)
    model = AutoModelForSeq2SeqLM.from_pretrained(artifact_path)

    # Translate
    predictions = []
    batch_size = 32
    for i in range(0, len(sources), batch_size):
        batch = sources[i : i + batch_size]
        inputs = tokenizer(batch, return_tensors="pt", padding=True, truncation=True, max_length=128)
        outputs = model.generate(**inputs, max_length=128)
        decoded = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        predictions.extend(decoded)

    # BLEU
    result = sacrebleu.corpus_bleu(predictions, [references])
    bleu_score = round(result.score / 100, 4)
    logger.info("Test BLEU: %.4f (%d samples)", bleu_score, len(predictions))
    return bleu_score


def check_and_promote(
    model_type: str,
    run_id: str,
    new_metric: float,
    params: dict,
) -> bool:
    """Compare against current Production model and promote if better."""
    mlflow_params = params["mlflow"]
    client = mlflow.tracking.MlflowClient()

    if model_type == "whisper":
        registry_name = mlflow_params["model_registry_name_whisper"]
        threshold = mlflow_params["promotion_threshold_wer"]
        metric_name = "wer"
        better = lambda new, old: old - new >= threshold  # lower is better
    else:
        registry_name = mlflow_params["model_registry_name_marian"]
        threshold = mlflow_params["promotion_threshold_bleu"]
        metric_name = "bleu"
        better = lambda new, old: new - old >= threshold  # higher is better

    # Try to get current production model's metric
    try:
        latest_versions = client.get_latest_versions(registry_name, stages=["Production"])
        if latest_versions:
            prod_run_id = latest_versions[0].run_id
            prod_run = client.get_run(prod_run_id)
            old_metric = float(prod_run.data.metrics.get(f"final_{metric_name}", 0))
            logger.info("Current Production %s: %.4f, New: %.4f", metric_name, old_metric, new_metric)

            if better(new_metric, old_metric):
                logger.info("New model is better by >= %.4f — promoting!", threshold)
            else:
                logger.info("New model not sufficiently better — keeping current Production")
                return False
        else:
            logger.info("No Production model found — promoting first model")
    except Exception:
        logger.info("Registry '%s' not found — creating and promoting", registry_name)

    # Register and promote
    model_uri = f"runs:/{run_id}/{model_type.replace('whisper', 'whisper-navi-lora').replace('marian', 'marian-navi-en')}"
    try:
        mv = mlflow.register_model(model_uri, registry_name)
        client.transition_model_version_stage(
            name=registry_name,
            version=mv.version,
            stage="Production",
            archive_existing_versions=True,
        )
        logger.info("Promoted %s v%s to Production", registry_name, mv.version)
        return True
    except Exception as e:
        logger.error("Failed to promote model: %s", e)
        return False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-type", required=True, choices=["whisper", "marian"])
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--config", default=str(PROJECT_ROOT / "params.yaml"))
    args = parser.parse_args()

    params = load_params(args.config)
    mlflow_params = params["mlflow"]

    mlflow.set_tracking_uri(mlflow_params["tracking_uri"])

    # Evaluate
    if args.model_type == "whisper":
        metric = evaluate_whisper(args.run_id, params)
        mlflow.log_metric("test_wer", metric)
    else:
        metric = evaluate_marian(args.run_id, params)
        mlflow.log_metric("test_bleu", metric)

    # Promote if better
    promoted = check_and_promote(args.model_type, args.run_id, metric, params)

    logger.info("=== Evaluation complete. Promoted: %s ===", promoted)


if __name__ == "__main__":
    main()
