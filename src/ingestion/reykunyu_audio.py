"""Reykunyu Real Audio Downloader.

Downloads ~4,958 community-recorded Na'vi audio clips from Reykunyu
(speakers: Plumps, tsyili, Tekre). REAL fluent Na'vi recordings.

Two-phase to avoid librosa thread-safety issues:
  Phase 1: parallel HTTP downloads of MP3s -> data/raw/audio/
  Phase 2: serial resample -> data/processed/audio/<stem>.wav (16kHz mono)

Output:
    data/raw/audio/<stem>.mp3
    data/processed/audio/<stem>.wav
    data/processed/audio/metadata.json
    data/processed/audio/stats.json

Attribution required for academic use. Add credits to README.
"""

from __future__ import annotations

import json
import logging
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from urllib.parse import quote

import requests
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
)
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
REYKUNYU_LIST = "https://reykunyu.lu/api/list/all"
REYKUNYU_AUDIO_BASE = "https://reykunyu.lu/fam/"

MAX_CONCURRENT_DOWNLOADS = 8
REQUEST_TIMEOUT = 30


def safe_filename(navi_word: str, speaker: str, word_type: str) -> str:
    safe = re.sub(r"[/\\:*?\"<>|]", "_", navi_word)
    safe = safe.replace(" ", "_")
    return f"{speaker}__{safe}__{word_type or 'x'}"


def load_params() -> dict:
    with open(PROJECT_ROOT / "params.yaml") as f:
        return yaml.safe_load(f)


def fetch_word_list() -> list[dict]:
    logger.info("Fetching full Reykunyu word list with audio metadata...")
    response = requests.get(REYKUNYU_LIST, timeout=REQUEST_TIMEOUT)
    response.raise_for_status()
    data = response.json()
    logger.info("Got %d entries", len(data))
    return data


def collect_audio_targets(words: list[dict]) -> list[dict]:
    targets = []
    for entry in words:
        navi = entry.get("na'vi") or entry.get("navi")
        word_type = entry.get("type", "x")
        if not navi:
            continue
        for pron in entry.get("pronunciation", []) or []:
            if not isinstance(pron, dict):
                continue
            for audio in pron.get("audio", []) or []:
                file = audio.get("file")
                speaker = audio.get("speaker", "unknown")
                if not file:
                    continue
                stem = safe_filename(navi, speaker, word_type)
                targets.append({
                    "navi": navi,
                    "speaker": speaker,
                    "word_type": word_type,
                    "remote_file": file,
                    "local_stem": stem,
                })
    return targets


# ---------------- Phase 1: downloads ----------------

def download_one_mp3(target: dict, raw_dir: Path) -> bool:
    raw_path = raw_dir / f"{target['local_stem']}.mp3"
    if raw_path.exists() and raw_path.stat().st_size > 100:
        return True
    url = REYKUNYU_AUDIO_BASE + quote(target["remote_file"], safe="/")
    try:
        r = requests.get(url, timeout=REQUEST_TIMEOUT)
        r.raise_for_status()
        raw_path.write_bytes(r.content)
        return True
    except Exception as e:
        logger.warning("DL fail %s: %s", target["remote_file"], e)
        return False


def phase1_download(targets: list[dict], raw_dir: Path) -> int:
    logger.info("=== Phase 1: Downloading %d MP3s ===", len(targets))
    raw_dir.mkdir(parents=True, exist_ok=True)
    ok = 0
    with ThreadPoolExecutor(max_workers=MAX_CONCURRENT_DOWNLOADS) as pool:
        futures = {pool.submit(download_one_mp3, t, raw_dir): t for t in targets}
        for i, fut in enumerate(as_completed(futures), 1):
            if fut.result():
                ok += 1
            if i % 200 == 0:
                logger.info("  Phase 1 progress: %d/%d ok=%d", i, len(targets), ok)
    logger.info("Phase 1 complete: %d/%d successful", ok, len(targets))
    return ok


# ---------------- Phase 2: serial resample ----------------

def resample_one(
    target: dict,
    raw_dir: Path,
    processed_dir: Path,
    target_sr: int,
    min_duration: float,
    max_duration: float,
) -> dict | None:
    """Serial resample. Imports librosa locally to avoid early native init."""
    import librosa
    import soundfile as sf

    raw_path = raw_dir / f"{target['local_stem']}.mp3"
    wav_path = processed_dir / f"{target['local_stem']}.wav"

    if not raw_path.exists():
        return None

    # If already resampled, validate and reuse
    if wav_path.exists():
        try:
            info = sf.info(str(wav_path))
            duration = info.frames / info.samplerate
            if min_duration <= duration <= max_duration:
                return {
                    "filename": wav_path.name,
                    "navi": target["navi"],
                    "speaker": target["speaker"],
                    "word_type": target["word_type"],
                    "duration_s": round(duration, 3),
                    "sample_rate": info.samplerate,
                    "samples": info.frames,
                    "source_dir": "reykunyu_real",
                }
        except Exception:
            pass

    try:
        audio, _ = librosa.load(str(raw_path), sr=target_sr, mono=True)
        duration = len(audio) / target_sr
        if duration < min_duration or duration > max_duration:
            return None
        sf.write(str(wav_path), audio, target_sr)
        return {
            "filename": wav_path.name,
            "navi": target["navi"],
            "speaker": target["speaker"],
            "word_type": target["word_type"],
            "duration_s": round(duration, 3),
            "sample_rate": target_sr,
            "samples": len(audio),
            "source_dir": "reykunyu_real",
        }
    except Exception as e:
        logger.warning("Resample fail %s: %s", raw_path.name, e)
        return None


def phase2_resample(
    targets: list[dict],
    raw_dir: Path,
    processed_dir: Path,
    target_sr: int,
    min_duration: float,
    max_duration: float,
) -> list[dict]:
    logger.info("=== Phase 2: Resampling MP3 -> 16kHz mono WAV ===")
    processed_dir.mkdir(parents=True, exist_ok=True)
    metadata = []
    skipped = 0
    for i, t in enumerate(targets, 1):
        result = resample_one(t, raw_dir, processed_dir, target_sr, min_duration, max_duration)
        if result:
            metadata.append(result)
        else:
            skipped += 1
        if i % 200 == 0:
            logger.info("  Phase 2 progress: %d/%d ok=%d skipped=%d",
                        i, len(targets), len(metadata), skipped)
    logger.info("Phase 2 complete: %d/%d resampled, %d skipped/invalid",
                len(metadata), len(targets), skipped)
    return metadata


# ---------------- Main ----------------

def main():
    params = load_params()
    data_params = params["data"]
    target_sr = data_params["audio_sample_rate"]
    min_dur = data_params["min_audio_duration"]
    max_dur = data_params["max_audio_duration"]

    raw_dir = PROJECT_ROOT / "data" / "raw" / "audio"
    processed_dir = PROJECT_ROOT / "data" / "processed" / "audio"

    # Step 1: get word list with audio fields
    words = fetch_word_list()

    # Save enriched words.json (with pronunciation/audio nested fields)
    normalized = []
    for w in words:
        entry = dict(w)
        if "na'vi" in entry and "navi" not in entry:
            entry["navi"] = entry["na'vi"]
        if "translations" in entry and entry["translations"]:
            entry["en"] = entry["translations"][0].get("en", entry.get("en", ""))
        normalized.append(entry)
    with open(PROJECT_ROOT / "data" / "raw" / "words.json", "w", encoding="utf-8") as f:
        json.dump(normalized, f, ensure_ascii=False, indent=2)
    logger.info("Saved enriched words.json (%d entries)", len(normalized))

    targets = collect_audio_targets(words)
    logger.info("Found %d audio files (%d unique words)",
                len(targets), len({t["navi"] for t in targets}))

    # Phase 1: parallel HTTP downloads
    t0 = time.time()
    phase1_download(targets, raw_dir)
    logger.info("Phase 1 wall time: %.1fs", time.time() - t0)

    # Phase 2: serial resample
    t1 = time.time()
    metadata = phase2_resample(targets, raw_dir, processed_dir,
                                target_sr, min_dur, max_dur)
    logger.info("Phase 2 wall time: %.1fs", time.time() - t1)

    # Write metadata
    with open(processed_dir / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    # Stats
    speakers = {}
    durations = []
    for m in metadata:
        speakers[m["speaker"]] = speakers.get(m["speaker"], 0) + 1
        durations.append(m["duration_s"])

    stats = {
        "count": len(metadata),
        "speakers": speakers,
        "total_duration_s": round(sum(durations), 2),
        "mean_duration_s": round(sum(durations) / len(durations), 3) if durations else 0,
        "min_duration_s": min(durations) if durations else 0,
        "max_duration_s": max(durations) if durations else 0,
    }
    with open(processed_dir / "stats.json", "w") as f:
        json.dump(stats, f, indent=2)

    logger.info("=== Reykunyu Audio Download Complete ===")
    for k, v in stats.items():
        logger.info("  %s: %s", k, v)


if __name__ == "__main__":
    main()
