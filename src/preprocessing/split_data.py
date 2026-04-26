"""Data splitting — DVC stage 'split'.

Splits both text pairs and audio data into train/val/test sets.
Uses stratified splitting based on word frequency tiers so that
rare words are represented proportionally in all splits.

Key learning points:
- A naive random split might put all rare Na'vi words in one set
- Stratified split ensures each set has the same proportion of
  common/uncommon/rare words
- We use a fixed random seed (from params.yaml) for reproducibility
- The split ratios (80/10/10) are also in params.yaml so they can
  be tuned without changing code
"""

import csv
import json
import logging
import shutil
import sys
from collections import Counter
from pathlib import Path

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


def assign_frequency_tier(words: list[dict]) -> list[dict]:
    """Assign frequency tiers to words for stratified splitting.

    Tiers:
    - 'common': top 30% most common word types
    - 'mid': middle 40%
    - 'rare': bottom 30%
    """
    type_counts = Counter(w["type"] for w in words)
    sorted_types = sorted(type_counts.keys(), key=lambda t: type_counts[t], reverse=True)

    tier_map = {}
    n = len(sorted_types)
    for i, t in enumerate(sorted_types):
        if i < n * 0.3:
            tier_map[t] = "common"
        elif i < n * 0.7:
            tier_map[t] = "mid"
        else:
            tier_map[t] = "rare"

    for w in words:
        w["frequency_tier"] = tier_map.get(w.get("type", "unknown"), "rare")

    return words


def stratified_split(
    items: list[dict],
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    seed: int,
    stratify_key: str = "frequency_tier",
) -> tuple[list[dict], list[dict], list[dict]]:
    """Split items into train/val/test with stratification."""
    rng = np.random.RandomState(seed)

    # Group by stratum
    groups: dict[str, list[dict]] = {}
    for item in items:
        key = item.get(stratify_key, "unknown")
        groups.setdefault(key, []).append(item)

    train, val, test = [], [], []

    for stratum, group_items in groups.items():
        rng.shuffle(group_items)
        n = len(group_items)
        n_train = max(1, int(n * train_ratio))
        n_val = max(1, int(n * val_ratio)) if n > 2 else 0
        # rest goes to test
        train.extend(group_items[:n_train])
        val.extend(group_items[n_train : n_train + n_val])
        test.extend(group_items[n_train + n_val :])

    # Final shuffle within each split
    rng.shuffle(train)
    rng.shuffle(val)
    rng.shuffle(test)

    return train, val, test


def save_text_split(pairs: list[dict], output_path: Path) -> None:
    """Save text pairs as TSV."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f, delimiter="\t")
        writer.writerow(["navi", "en"])
        for p in pairs:
            writer.writerow([p["navi"], p["en"]])


def save_audio_manifest(items: list[dict], output_path: Path) -> None:
    """Save audio file manifest for a split."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(items, f, indent=2)


def main():
    params = load_params()
    split_params = params["split"]

    text_dir = PROJECT_ROOT / "data" / "processed" / "text"
    audio_dir = PROJECT_ROOT / "data" / "processed" / "audio"
    words_path = PROJECT_ROOT / "data" / "raw" / "words.json"

    train_dir = PROJECT_ROOT / "data" / "processed" / "train"
    val_dir = PROJECT_ROOT / "data" / "processed" / "val"
    test_dir = PROJECT_ROOT / "data" / "processed" / "test"

    # Clean output dirs
    for d in [train_dir, val_dir, test_dir]:
        if d.exists():
            shutil.rmtree(d)
        d.mkdir(parents=True)

    # Load words for frequency tier assignment
    with open(words_path, encoding="utf-8") as f:
        words = json.load(f)

    words = assign_frequency_tier(words)
    word_tier_map = {w["navi"]: w.get("frequency_tier", "rare") for w in words}

    # --- Split text pairs ---
    pairs_path = text_dir / "pairs.tsv"
    text_items = []
    with open(pairs_path, encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            # Assign tier based on first word
            first_word = row["navi"].split()[0]
            row["frequency_tier"] = word_tier_map.get(first_word, "rare")
            text_items.append(row)

    train_text, val_text, test_text = stratified_split(
        text_items,
        split_params["train_ratio"],
        split_params["val_ratio"],
        split_params["test_ratio"],
        split_params["random_seed"],
    )

    save_text_split(train_text, train_dir / "pairs.tsv")
    save_text_split(val_text, val_dir / "pairs.tsv")
    save_text_split(test_text, test_dir / "pairs.tsv")

    logger.info("Text split: train=%d, val=%d, test=%d", len(train_text), len(val_text), len(test_text))

    # --- Split audio ---
    audio_meta_path = audio_dir / "metadata.json"
    if audio_meta_path.exists():
        with open(audio_meta_path) as f:
            audio_items = json.load(f)

        # Assign tiers
        for item in audio_items:
            stem = Path(item["filename"]).stem
            item["frequency_tier"] = word_tier_map.get(stem, "rare")

        train_audio, val_audio, test_audio = stratified_split(
            audio_items,
            split_params["train_ratio"],
            split_params["val_ratio"],
            split_params["test_ratio"],
            split_params["random_seed"],
        )

        save_audio_manifest(train_audio, train_dir / "audio_manifest.json")
        save_audio_manifest(val_audio, val_dir / "audio_manifest.json")
        save_audio_manifest(test_audio, test_dir / "audio_manifest.json")

        logger.info("Audio split: train=%d, val=%d, test=%d", len(train_audio), len(val_audio), len(test_audio))
    else:
        logger.warning("No audio metadata found — skipping audio split")

    # Save split summary
    summary = {
        "text": {"train": len(train_text), "val": len(val_text), "test": len(test_text)},
    }
    if audio_meta_path.exists():
        summary["audio"] = {"train": len(train_audio), "val": len(val_audio), "test": len(test_audio)}

    with open(train_dir.parent / "split_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    logger.info("=== Data splitting complete ===")


if __name__ == "__main__":
    main()
