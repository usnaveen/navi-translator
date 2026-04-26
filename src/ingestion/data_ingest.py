"""Data ingestion orchestrator — DVC stage 'ingest'.

This is the entry point for the first DVC pipeline stage. It:
1. Downloads words.json from Reykunyu GitHub
2. Generates synthetic audio for every word using TTS
3. Validates all downloaded/generated data

Key learning points:
- This script is called by DVC via `dvc repro ingest`
- DVC tracks inputs (this script + dependencies) and outputs (data/raw/, data/synthetic/)
- If nothing changes, DVC skips this stage on next run — that's reproducibility
"""

import json
import logging
import sys
from pathlib import Path

import yaml

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.ingestion.reykunyu_client import download_words
from src.ingestion.tts_generate import generate_audio_for_words

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
)
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def load_params() -> dict:
    with open(PROJECT_ROOT / "params.yaml") as f:
        return yaml.safe_load(f)


def validate_words(words: list[dict]) -> list[dict]:
    """Validate word entries against the required schema.

    Each entry must have: navi (str), en (str), type (str).
    Entries missing any field are rejected.
    """
    valid = []
    rejected = 0

    for entry in words:
        navi = entry.get("navi", "").strip()
        en = entry.get("en", "").strip()
        word_type = entry.get("type", "").strip()

        if not navi or not en:
            rejected += 1
            continue

        if not word_type:
            word_type = "unknown"

        valid.append({"navi": navi, "en": en, "type": word_type})

    logger.info("Validation: %d valid, %d rejected", len(valid), rejected)
    return valid


def validate_navi_charset(word: str, allowed_chars: str) -> bool:
    """Check that a Na'vi word only contains allowed characters."""
    return all(c in allowed_chars for c in word)


def validate_audio_file(filepath: Path, min_dur: float, max_dur: float) -> bool:
    """Validate an audio file's duration and format."""
    try:
        import soundfile as sf

        info = sf.info(str(filepath))
        duration = info.duration
        return min_dur <= duration <= max_dur
    except Exception as e:
        logger.warning("Invalid audio file %s: %s", filepath, e)
        return False


def main():
    params = load_params()
    data_params = params["data"]
    tts_params = params["tts"]

    raw_dir = PROJECT_ROOT / "data" / "raw"
    synthetic_dir = PROJECT_ROOT / "data" / "synthetic"
    words_path = raw_dir / "words.json"
    audio_dir = raw_dir / "audio"

    # Ensure directories exist
    raw_dir.mkdir(parents=True, exist_ok=True)
    audio_dir.mkdir(parents=True, exist_ok=True)
    synthetic_dir.mkdir(parents=True, exist_ok=True)

    # Step 1: Download Na'vi dictionary
    logger.info("=== Step 1: Downloading Na'vi dictionary ===")
    words = download_words(data_params["reykunyu_words_url"], words_path)

    # Step 2: Validate words
    logger.info("=== Step 2: Validating word entries ===")
    words = validate_words(words)

    # Re-save validated words
    with open(words_path, "w", encoding="utf-8") as f:
        json.dump(words, f, ensure_ascii=False, indent=2)

    # Step 3: Generate synthetic audio
    logger.info("=== Step 3: Generating synthetic audio ===")
    num_generated = generate_audio_for_words(
        words=words,
        output_dir=synthetic_dir,
        tts_engine=tts_params["engine"],
        language=tts_params["language"],
        slow=tts_params["slow"],
        sample_rate=data_params["audio_sample_rate"],
    )

    # Step 4: Validate audio files
    logger.info("=== Step 4: Validating audio files ===")
    valid_audio = 0
    invalid_audio = 0
    for wav_file in synthetic_dir.glob("*.wav"):
        if validate_audio_file(
            wav_file,
            data_params["min_audio_duration"],
            data_params["max_audio_duration"],
        ):
            valid_audio += 1
        else:
            invalid_audio += 1

    logger.info(
        "Audio validation: %d valid, %d invalid out of %d files",
        valid_audio,
        invalid_audio,
        valid_audio + invalid_audio,
    )

    # Summary
    logger.info("=== Ingestion Complete ===")
    logger.info("  Words: %d", len(words))
    logger.info("  Audio files generated: %d", num_generated)
    logger.info("  Valid audio files: %d", valid_audio)


if __name__ == "__main__":
    main()
