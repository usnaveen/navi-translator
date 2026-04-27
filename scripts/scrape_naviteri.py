"""Scrape canonical Na'vi sentences from naviteri.org (Paul Frommer's blog).

naviteri.org is the authoritative source for the Na'vi language — maintained
by the linguist who created it. Each post contains real Na'vi sentences paired
with English translations, written in this consistent pattern:

    <strong>Na'vi sentence here.</strong>
    'English translation here.'

This script:
1. Fetches all 291+ post URLs from the WordPress sitemap
2. Downloads each post and extracts sentence pairs
3. Filters out single-word vocabulary entries (we already have those from Reykunyu)
4. Writes real Na'vi sentence pairs to data/raw/naviteri_sentences.tsv

Why this matters: the existing training data (build_pairs.py) generates fake
sentences by randomly concatenating dictionary words. This scraper provides
real sentences with correct Na'vi grammar — free word order, infixes, case
markings — which are essential for meaningful BLEU evaluation.
"""

import csv
import logging
import re
import time
import xml.etree.ElementTree as ET
from pathlib import Path

import requests

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
)
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[1]

SITEMAP_URL = "https://naviteri.org/wp-sitemap-posts-post-1.xml"
REQUEST_DELAY = 1.0  # seconds between requests — be polite to the server
REQUEST_TIMEOUT = 20

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.5",
}

# Na'vi uses these unique characters; presence is a strong signal
NAVI_CHARS = set("ìäéù'")

# Minimum words for something to count as a "sentence" not a vocab entry
MIN_NAVI_WORDS = 2


def fetch_post_urls(sitemap_url: str) -> list[str]:
    """Return all post URLs from the WordPress sitemap."""
    logger.info("Fetching sitemap: %s", sitemap_url)
    resp = requests.get(sitemap_url, headers=HEADERS, timeout=REQUEST_TIMEOUT)
    resp.raise_for_status()

    root = ET.fromstring(resp.content)
    # WordPress sitemaps use the standard sitemap namespace
    ns = {"sm": "http://www.sitemaps.org/schemas/sitemap/0.9"}
    urls = [loc.text.strip() for loc in root.findall(".//sm:loc", ns) if loc.text]
    logger.info("Found %d post URLs", len(urls))
    return urls


def fetch_post_html(url: str) -> str | None:
    """Fetch a single blog post and return its raw HTML."""
    try:
        resp = requests.get(url, headers=HEADERS, timeout=REQUEST_TIMEOUT)
        resp.raise_for_status()
        return resp.text
    except requests.RequestException as e:
        logger.warning("Failed to fetch %s: %s", url, e)
        return None


def extract_pairs_from_html(html: str) -> list[tuple[str, str]]:
    """Extract Na'vi → English sentence pairs from a post's HTML.

    The pattern on naviteri.org is:
        <strong>Na'vi sentence.</strong>
        'English translation.'

    We parse the post body as plain text (stripping tags) and use regex
    to find bold Na'vi lines immediately followed by single-quoted English.
    """
    # Strip HTML tags but keep newlines at block boundaries so the
    # line-by-line pattern (bold Na'vi then quoted English) is preserved.
    text = _html_to_text(html)
    return _extract_pairs_from_text(text)


def _html_to_text(html: str) -> str:
    """Convert HTML to plain text, inserting newlines at block elements."""
    # Replace block-level closers with newlines before stripping all tags
    html = re.sub(r"</(?:p|br|div|li|blockquote|h[1-6])[^>]*>", "\n", html, flags=re.IGNORECASE)
    html = re.sub(r"<br\s*/?>", "\n", html, flags=re.IGNORECASE)
    # Strip remaining tags
    text = re.sub(r"<[^>]+>", "", html)
    # Decode common HTML entities
    text = text.replace("&amp;", "&").replace("&lt;", "<").replace("&gt;", ">")
    text = text.replace("&#8216;", "‘").replace("&#8217;", "’")
    text = text.replace("&nbsp;", " ").replace("&#39;", "'")
    # Normalise unicode apostrophes/quotes to ASCII for easier matching
    text = text.replace("‘", "'").replace("’", "'")
    text = text.replace("“", '"').replace("”", '"')
    return text


def _extract_pairs_from_text(text: str) -> list[tuple[str, str]]:
    """Find (Na'vi sentence, English translation) pairs in plain text.

    Looks for lines that end with a period (or !) and are followed within
    the next 1-2 non-empty lines by a single-quoted English translation.
    """
    lines = [line.strip() for line in text.splitlines()]
    pairs = []

    i = 0
    while i < len(lines):
        line = lines[i]

        # Skip empty or very short lines
        if len(line) < 4:
            i += 1
            continue

        # Check if this line looks like a Na'vi sentence candidate:
        # - multiple words
        # - contains Na'vi-specific characters OR looks like a Na'vi phrase
        words = line.split()
        if len(words) >= MIN_NAVI_WORDS and _looks_like_navi(line):
            # Look ahead up to 3 lines for a single-quoted English translation
            for j in range(i + 1, min(i + 4, len(lines))):
                candidate = lines[j]
                if not candidate:
                    continue
                english = _extract_quoted_translation(candidate)
                if english:
                    navi_clean = _clean_navi(line)
                    if navi_clean and len(navi_clean.split()) >= MIN_NAVI_WORDS:
                        pairs.append((navi_clean, english))
                    break
                # If we hit another non-empty line that isn't a translation, stop
                if candidate and not _extract_quoted_translation(candidate):
                    break

        i += 1

    return pairs


def _looks_like_navi(text: str) -> bool:
    """Return True if the text is likely a Na'vi phrase.

    Heuristics:
    - Contains Na'vi-specific characters (ì, ä, é, ù, or apostrophe mid-word)
    - Is not an obvious English sentence (starts with common English words)
    """
    text_lower = text.lower()

    # Reject obvious English
    english_starters = (
        "the ", "a ", "an ", "this ", "that ", "it ", "he ", "she ",
        "we ", "they ", "you ", "i ", "in ", "on ", "at ", "for ",
        "note:", "see ", "from ", "as ", "but ", "so ", "if ",
    )
    if any(text_lower.startswith(s) for s in english_starters):
        return False

    # Reject lines that are obviously metadata
    if text_lower.startswith("http") or text_lower.startswith("©"):
        return False

    # Strong signal: Na'vi-specific characters
    if any(c in text for c in NAVI_CHARS):
        return True

    # Moderate signal: mid-word apostrophe (Na'vi glottal stops)
    if re.search(r"\w'\w", text):
        return True

    return False


def _extract_quoted_translation(text: str) -> str | None:
    """Extract English text wrapped in single quotes: 'like this'."""
    # Match text that starts with ' and ends with ' (possibly with trailing punctuation)
    m = re.match(r"^'(.+?)'\.?\s*$", text)
    if m:
        return m.group(1).strip()

    # Also handle lines that are just a quoted phrase mid-sentence
    m = re.search(r"'([A-Z][^']{5,}[.!?])'", text)
    if m:
        return m.group(1).strip()

    return None


def _clean_navi(text: str) -> str:
    """Remove leading/trailing punctuation noise from a Na'vi sentence."""
    # Strip leading bullet chars, numbers, etc.
    text = re.sub(r"^[\d\.\-\*\•\–\—\s]+", "", text)
    text = text.strip()
    # Reject vocabulary entries — they contain pronunciation guides like (n., ...)
    if re.search(r"\([a-z]+\.\,?\s", text):
        return ""
    # Reject lines with parenthetical part-of-speech markers
    if re.search(r"\((?:n|v|adj|adv|intj|part|conj|adp)\.", text):
        return ""
    return text


def deduplicate(pairs: list[tuple[str, str]]) -> list[tuple[str, str]]:
    seen = set()
    unique = []
    for navi, en in pairs:
        key = navi.lower().strip()
        if key not in seen:
            seen.add(key)
            unique.append((navi, en))
    return unique


def write_tsv(pairs: list[tuple[str, str]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f, delimiter="\t")
        writer.writerow(["navi", "en"])
        for navi, en in pairs:
            writer.writerow([navi, en])
    logger.info("Wrote %d sentence pairs to %s", len(pairs), output_path)


def main():
    output_path = Path.home() / "Downloads" / "naviteri_sentences.tsv"

    # Step 1: get all post URLs
    urls = fetch_post_urls(SITEMAP_URL)

    # Step 2: scrape each post
    all_pairs: list[tuple[str, str]] = []
    for idx, url in enumerate(urls, 1):
        logger.info("[%d/%d] %s", idx, len(urls), url)
        html = fetch_post_html(url)
        if html:
            pairs = extract_pairs_from_html(html)
            if pairs:
                logger.info("  → %d pairs", len(pairs))
            all_pairs.extend(pairs)
        time.sleep(REQUEST_DELAY)

    # Step 3: deduplicate and write
    unique_pairs = deduplicate(all_pairs)
    logger.info("Total unique sentence pairs: %d (from %d raw)", len(unique_pairs), len(all_pairs))
    write_tsv(unique_pairs, output_path)
    logger.info("Done. Run `dvc repro preprocess_text` to rebuild training pairs.")


if __name__ == "__main__":
    main()
