"""Unit tests for text pair building."""

from src.preprocessing.build_pairs import build_word_pairs, deduplicate_pairs


class TestBuildWordPairs:
    def test_basic_pairs(self):
        words = [
            {"navi": "kaltxì", "en": "hello", "type": "intj"},
            {"navi": "oel", "en": "I (ergative)", "type": "pn"},
        ]
        pairs = build_word_pairs(words)
        assert len(pairs) == 2
        assert ("kaltxì", "hello") in pairs

    def test_semicolon_meanings(self):
        """Should use first meaning when multiple are separated by ';'."""
        words = [{"navi": "fya'o", "en": "path; way; manner", "type": "n"}]
        pairs = build_word_pairs(words)
        assert len(pairs) == 1
        assert pairs[0] == ("fya'o", "path")

    def test_empty_en_skipped(self):
        words = [{"navi": "test", "en": "", "type": "n"}]
        pairs = build_word_pairs(words)
        assert len(pairs) == 0


class TestDeduplicate:
    def test_removes_duplicates(self):
        pairs = [("a", "b"), ("a", "b"), ("c", "d")]
        result = deduplicate_pairs(pairs)
        assert len(result) == 2

    def test_case_insensitive_dedup(self):
        pairs = [("Hello", "World"), ("hello", "world")]
        result = deduplicate_pairs(pairs)
        assert len(result) == 1
