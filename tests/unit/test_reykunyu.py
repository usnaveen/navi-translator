"""Unit tests for the Reykunyu client."""


from src.ingestion.reykunyu_client import _extract_word, lookup_word


def validate_words(words):
    """Re-implement simple validation for testing."""
    valid = []
    for entry in words:
        if entry.get("navi") and entry.get("en"):
            valid.append(entry)
    return valid


class TestExtractWord:
    def test_basic_word(self):
        entry = {"na'vi": "kaltxì", "en": "hello", "type": "intj"}
        result = _extract_word(entry)
        assert result is not None
        assert result["navi"] == "kaltxì"
        assert result["en"] == "hello"
        assert result["type"] == "intj"

    def test_missing_navi_returns_none(self):
        entry = {"en": "hello", "type": "intj"}
        result = _extract_word(entry)
        assert result is None

    def test_missing_en_returns_none(self):
        entry = {"na'vi": "kaltxì", "type": "intj"}
        result = _extract_word(entry)
        assert result is None

    def test_list_english_meanings(self):
        entry = {"na'vi": "tsun", "en": ["can", "able to"], "type": "vim"}
        result = _extract_word(entry)
        assert result is not None
        assert "can" in result["en"]

    def test_dict_english_meaning(self):
        entry = {"na'vi": "lu", "en": {"en": "to be"}, "type": "v"}
        result = _extract_word(entry)
        assert result is not None
        assert result["en"] == "to be"


class TestLookupWord:
    def test_found(self):
        words = [{"navi": "kaltxì", "en": "hello", "type": "intj"}]
        result = lookup_word("kaltxì", words)
        assert result is not None
        assert result["en"] == "hello"

    def test_case_insensitive(self):
        words = [{"navi": "Kaltxì", "en": "hello", "type": "intj"}]
        result = lookup_word("kaltxì", words)
        assert result is not None

    def test_not_found(self):
        words = [{"navi": "kaltxì", "en": "hello", "type": "intj"}]
        result = lookup_word("nonexistent", words)
        assert result is None
