"""TTS audio generator — creates synthetic Na'vi audio for ASR training.

Since Na'vi is a constructed language with limited real audio recordings,
we bootstrap the ASR training data by generating synthetic speech using
text-to-speech (TTS).

Key learning points:
- gTTS converts text to speech using Google's TTS engine
- We use Indonesian ('id') as the closest phonetic match for Na'vi
- Audio is saved as WAV at 16kHz mono — the format Whisper expects
- Each word gets its own audio file, named by the Na'vi word
"""

import json
import logging
import re
from io import BytesIO
from pathlib import Path

import soundfile as sf
import yaml

logger = logging.getLogger(__name__)


def load_params(params_file: str = "params.yaml") -> dict:
    with open(params_file) as f:
        return yaml.safe_load(f)


def sanitize_filename(word: str) -> str:
    """Convert a Na'vi word to a safe filename."""
    # Replace special characters with underscores
    safe = re.sub(r"[^a-zA-Z0-9äìÄÌ]", "_", word)
    return safe.strip("_")[:100]  # limit length


def generate_audio_for_words(
    words: list[dict],
    output_dir: Path,
    tts_engine: str = "gtts",
    language: str = "id",
    slow: bool = True,
    sample_rate: int = 16000,
) -> int:
    """Generate synthetic audio for each Na'vi word.

    Args:
        words: List of word dicts with 'navi' key.
        output_dir: Directory to save WAV files.
        tts_engine: TTS engine to use ('gtts').
        language: Language code for TTS (default 'id' for Indonesian).
        slow: Whether to use slow speech (better for training).
        sample_rate: Target sample rate in Hz.

    Returns:
        Number of audio files successfully generated.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    generated = 0
    skipped = 0

    for i, word_entry in enumerate(words):
        navi_word = word_entry["navi"]
        filename = sanitize_filename(navi_word)
        output_path = output_dir / f"{filename}.wav"

        # Skip if already generated
        if output_path.exists():
            skipped += 1
            continue

        try:
            _generate_single(
                text=navi_word,
                output_path=output_path,
                engine=tts_engine,
                language=language,
                slow=slow,
                sample_rate=sample_rate,
            )
            generated += 1

            if (i + 1) % 100 == 0:
                logger.info("Generated %d/%d audio files", i + 1, len(words))

        except Exception as e:
            logger.warning("Failed to generate audio for '%s': %s", navi_word, e)

    logger.info(
        "TTS complete: %d generated, %d skipped (already exist), %d total words",
        generated,
        skipped,
        len(words),
    )
    return generated


def _generate_single(
    text: str,
    output_path: Path,
    engine: str,
    language: str,
    slow: bool,
    sample_rate: int,
) -> None:
    """Generate a single audio file using the specified TTS engine."""
    if engine == "gtts":
        _generate_gtts(text, output_path, language, slow, sample_rate)
    else:
        raise ValueError(f"Unsupported TTS engine: {engine}")


def _generate_gtts(
    text: str,
    output_path: Path,
    language: str,
    slow: bool,
    sample_rate: int,
) -> None:
    """Generate audio using Google Text-to-Speech (gTTS).

    gTTS outputs MP3, so we convert to WAV at the target sample rate.
    """
    from gtts import gTTS

    tts = gTTS(text=text, lang=language, slow=slow)

    # gTTS writes MP3 — we need to convert to WAV
    mp3_buffer = BytesIO()
    tts.write_to_fp(mp3_buffer)
    mp3_buffer.seek(0)

    # Use librosa to load MP3 and resample to target rate
    import librosa

    audio, _ = librosa.load(mp3_buffer, sr=sample_rate, mono=True)
    sf.write(str(output_path), audio, sample_rate)
