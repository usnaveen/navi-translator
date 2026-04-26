"""Reykunyu API client — fetches the Na'vi dictionary.

The Reykunyu project (reykunyu.lu) maintains the most comprehensive
Na'vi dictionary. This module downloads the full words.json and
provides lookup utilities.

Key learning points:
- requests.get() fetches data from a URL
- response.json() parses the JSON body into a Python dict/list
- We validate the schema of each entry to catch bad data early
"""

import json
import logging
from pathlib import Path

import requests
import yaml

logger = logging.getLogger(__name__)


def load_params(params_file: str = "params.yaml") -> dict:
    """Load parameters from the YAML config file."""
    with open(params_file) as f:
        return yaml.safe_load(f)


def download_words(url: str, output_path: Path) -> list[dict]:
    """Download the full Na'vi dictionary from Reykunyu GitHub.

    The words.json file contains entries like:
        { "na'vi": "kaltxì", "en": "hello", "type": "intj", ... }

    Returns the list of validated word entries.
    """
    logger.info("Downloading Na'vi dictionary from %s", url)
    response = requests.get(url, timeout=30)
    response.raise_for_status()

    raw_data = response.json()
    words = _parse_words(raw_data)

    logger.info("Downloaded %d words, %d passed validation", len(raw_data), len(words))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(words, f, ensure_ascii=False, indent=2)

    return words


def _parse_words(raw_data: list | dict) -> list[dict]:
    """Parse and validate word entries from the raw Reykunyu data.

    The raw format can vary — sometimes it's a list of entries,
    sometimes a dict keyed by word. We normalize to a flat list.
    """
    words = []

    # Handle both list and dict formats
    if isinstance(raw_data, dict):
        entries = []
        for key, value in raw_data.items():
            if isinstance(value, list):
                for item in value:
                    if isinstance(item, dict):
                        item.setdefault("na'vi", key)
                        entries.append(item)
            elif isinstance(value, dict):
                value.setdefault("na'vi", key)
                entries.append(value)
    else:
        entries = raw_data

    for entry in entries:
        word = _extract_word(entry)
        if word:
            words.append(word)

    return words


def _extract_word(entry: dict) -> dict | None:
    """Extract a normalized word record from a raw entry.

    Returns None if the entry is invalid (missing required fields).
    """
    # Try different key names the API might use
    navi = entry.get("na'vi") or entry.get("navi") or entry.get("word", "")
    navi = navi.strip() if isinstance(navi, str) else ""

    # English translations can be in different formats
    en_raw = entry.get("en") or entry.get("translations", {}).get("en", [])
    if isinstance(en_raw, str):
        en = en_raw.strip()
    elif isinstance(en_raw, list):
        # Join multiple English meanings
        meanings = []
        for item in en_raw:
            if isinstance(item, str):
                meanings.append(item.strip())
            elif isinstance(item, dict):
                meaning = item.get("en", "") or item.get("meaning", "")
                if meaning:
                    meanings.append(meaning.strip())
        en = "; ".join(meanings)
    elif isinstance(en_raw, dict):
        en = en_raw.get("en", "") or en_raw.get("meaning", "")
    else:
        en = ""

    word_type = entry.get("type") or entry.get("pos") or "unknown"

    if not navi or not en:
        return None

    return {
        "navi": navi,
        "en": en,
        "type": word_type,
    }


def lookup_word(word: str, words: list[dict]) -> dict | None:
    """Look up a single Na'vi word in the loaded dictionary."""
    word_lower = word.lower().strip()
    for entry in words:
        if entry["navi"].lower() == word_lower:
            return entry
    return None


def load_words(words_path: Path) -> list[dict]:
    """Load previously downloaded words.json from disk."""
    with open(words_path, encoding="utf-8") as f:
        return json.load(f)
