"""Audio preprocessing — DVC stage 'preprocess_audio'.

This script processes all audio files (both real recordings and synthetic TTS)
into the format that Whisper expects:
- 16kHz sample rate
- Mono channel
- WAV format

It also validates each file against the schema requirements from params.yaml.

Key learning points:
- librosa.load() can read almost any audio format and resample in one step
- Whisper expects 16kHz mono audio — this is the standard for speech models
- Log-mel spectrograms are a visual representation of audio frequencies over time
  (Whisper computes these internally, but we validate the raw audio here)
- soundfile (sf) is used for writing WAV files efficiently
"""

import json
import logging
import sys
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf
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


def process_audio_file(
    input_path: Path,
    output_path: Path,
    target_sr: int = 16000,
    min_duration: float = 0.3,
    max_duration: float = 10.0,
) -> dict | None:
    """Load, validate, resample, and save a single audio file.

    Returns metadata dict on success, None on failure.
    """
    try:
        # librosa.load handles format conversion and resampling
        audio, sr = librosa.load(str(input_path), sr=target_sr, mono=True)
        duration = len(audio) / sr

        # Validate duration
        if duration < min_duration:
            logger.warning("Too short (%.2fs): %s", duration, input_path.name)
            return None
        if duration > max_duration:
            logger.warning("Too long (%.2fs): %s", duration, input_path.name)
            return None

        # Save as WAV
        output_path.parent.mkdir(parents=True, exist_ok=True)
        sf.write(str(output_path), audio, target_sr)

        return {
            "filename": output_path.name,
            "original": input_path.name,
            "duration_s": round(duration, 3),
            "sample_rate": target_sr,
            "samples": len(audio),
        }

    except Exception as e:
        logger.warning("Failed to process %s: %s", input_path.name, e)
        return None


def process_all_audio(
    source_dirs: list[Path],
    output_dir: Path,
    target_sr: int,
    min_duration: float,
    max_duration: float,
) -> list[dict]:
    """Process all audio files from source directories."""
    output_dir.mkdir(parents=True, exist_ok=True)
    metadata = []

    audio_extensions = {".wav", ".mp3", ".ogg", ".flac", ".m4a"}

    for source_dir in source_dirs:
        if not source_dir.exists():
            logger.warning("Source directory does not exist: %s", source_dir)
            continue

        files = [f for f in source_dir.rglob("*") if f.suffix.lower() in audio_extensions]
        logger.info("Processing %d audio files from %s", len(files), source_dir)

        for i, audio_file in enumerate(files):
            output_path = output_dir / f"{audio_file.stem}.wav"

            result = process_audio_file(
                input_path=audio_file,
                output_path=output_path,
                target_sr=target_sr,
                min_duration=min_duration,
                max_duration=max_duration,
            )

            if result:
                result["source_dir"] = source_dir.name
                metadata.append(result)

            if (i + 1) % 200 == 0:
                logger.info("  Processed %d/%d files", i + 1, len(files))

    return metadata


def compute_audio_stats(metadata: list[dict]) -> dict:
    """Compute summary statistics for processed audio."""
    if not metadata:
        return {"count": 0}

    durations = [m["duration_s"] for m in metadata]
    return {
        "count": len(metadata),
        "total_duration_s": round(sum(durations), 2),
        "mean_duration_s": round(np.mean(durations), 3),
        "std_duration_s": round(np.std(durations), 3),
        "min_duration_s": round(min(durations), 3),
        "max_duration_s": round(max(durations), 3),
    }


def main():
    params = load_params()
    data_params = params["data"]

    source_dirs = [
        PROJECT_ROOT / "data" / "raw" / "audio",
        PROJECT_ROOT / "data" / "synthetic",
    ]
    output_dir = PROJECT_ROOT / "data" / "processed" / "audio"

    metadata = process_all_audio(
        source_dirs=source_dirs,
        output_dir=output_dir,
        target_sr=data_params["audio_sample_rate"],
        min_duration=data_params["min_audio_duration"],
        max_duration=data_params["max_audio_duration"],
    )

    # Save metadata
    with open(output_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    # Compute and log stats
    stats = compute_audio_stats(metadata)
    logger.info("=== Audio Processing Complete ===")
    for key, value in stats.items():
        logger.info("  %s: %s", key, value)

    with open(output_dir / "stats.json", "w") as f:
        json.dump(stats, f, indent=2)


if __name__ == "__main__":
    main()
