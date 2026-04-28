"""Build Na'vi → English sentence pairs — DVC stage 'preprocess_text'.

This script takes the validated words.json and produces a TSV file
of Na'vi → English translation pairs for MarianMT training.

Key learning points:
- MarianMT expects parallel text: source sentence \t target sentence
- We create pairs at multiple levels: single words, short phrases,
  and simple constructed sentences to give the model more varied training data
- TSV (tab-separated values) is the standard format for parallel corpora
"""

import csv
import json
import logging
import random
import sys
from pathlib import Path

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


def build_word_pairs(words: list[dict]) -> list[tuple[str, str]]:
    """Create simple word-level translation pairs."""
    pairs = []
    for entry in words:
        navi = entry["navi"]
        en = entry["en"]
        # If multiple meanings separated by ';', use the first one as primary
        primary_en = en.split(";")[0].strip()
        if primary_en:
            pairs.append((navi, primary_en))
    return pairs


def build_phrase_pairs(words: list[dict]) -> list[tuple[str, str]]:
    """Build simple phrase pairs by combining words.

    Creates short Na'vi phrases by combining 2-3 words
    and their English translations. This gives MarianMT
    training signal for multi-word input.
    """
    pairs = []
    random.seed(42)  # reproducible

    # Group words by type for more natural combinations
    by_type: dict[str, list[dict]] = {}
    for w in words:
        word_type = w.get("type", "unknown")
        by_type.setdefault(word_type, []).append(w)

    # Create 2-word combinations
    for _ in range(min(500, len(words))):
        w1, w2 = random.sample(words, 2)
        navi_phrase = f"{w1['navi']} {w2['navi']}"
        en1 = w1["en"].split(";")[0].strip()
        en2 = w2["en"].split(";")[0].strip()
        en_phrase = f"{en1} {en2}"
        pairs.append((navi_phrase, en_phrase))

    # Create 3-word combinations (fewer)
    for _ in range(min(200, len(words) // 2)):
        w1, w2, w3 = random.sample(words, 3)
        navi_phrase = f"{w1['navi']} {w2['navi']} {w3['navi']}"
        en1 = w1["en"].split(";")[0].strip()
        en2 = w2["en"].split(";")[0].strip()
        en3 = w3["en"].split(";")[0].strip()
        en_phrase = f"{en1} {en2} {en3}"
        pairs.append((navi_phrase, en_phrase))

    return pairs


def deduplicate_pairs(pairs: list[tuple[str, str]]) -> list[tuple[str, str]]:
    """Remove duplicate pairs, preserving order."""
    seen = set()
    unique = []
    for pair in pairs:
        key = (pair[0].lower(), pair[1].lower())
        if key not in seen:
            seen.add(key)
            unique.append(pair)
    return unique


def write_pairs_tsv(pairs: list[tuple[str, str]], output_path: Path) -> None:
    """Write translation pairs to a TSV file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f, delimiter="\t")
        writer.writerow(["navi", "en"])
        for navi, en in pairs:
            writer.writerow([navi, en])
    logger.info("Wrote %d pairs to %s", len(pairs), output_path)


def main():
    load_params()
    words_path = PROJECT_ROOT / "data" / "raw" / "words.json"
    output_dir = PROJECT_ROOT / "data" / "processed" / "text"

    # Load validated words
    with open(words_path, encoding="utf-8") as f:
        words = json.load(f)

    logger.info("Loaded %d words from %s", len(words), words_path)

    # Build pairs at different levels
    word_pairs = build_word_pairs(words)
    logger.info("Built %d word-level pairs", len(word_pairs))

    phrase_pairs = build_phrase_pairs(words)
    logger.info("Built %d phrase-level pairs", len(phrase_pairs))

    # Combine and deduplicate
    all_pairs = deduplicate_pairs(word_pairs + phrase_pairs)
    logger.info("Total unique pairs: %d", len(all_pairs))

    # Write output
    write_pairs_tsv(all_pairs, output_dir / "pairs.tsv")

    # Also save a manifest
    manifest = {
        "total_pairs": len(all_pairs),
        "word_pairs": len(word_pairs),
        "phrase_pairs": len(phrase_pairs),
        "source_words": len(words),
    }
    import json as json_mod

    with open(output_dir / "manifest.json", "w") as f:
        json_mod.dump(manifest, f, indent=2)

    logger.info("=== Text preprocessing complete ===")


if __name__ == "__main__":
    main()
