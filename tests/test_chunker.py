"""
Tests for app.services.chunker — no DB or network required.
"""
import pytest
from app.services.chunker import chunk_text


def test_empty_string_returns_empty_list():
    assert chunk_text("") == []


def test_whitespace_only_returns_empty_list():
    assert chunk_text("   \n\t  ") == []


def test_short_text_becomes_single_chunk():
    result = chunk_text("Hello world", size=800, overlap=100)
    assert result == ["Hello world"]


def test_long_text_is_split():
    text = "A" * 2000
    result = chunk_text(text, size=800, overlap=100)
    assert len(result) > 1


def test_first_chunk_has_correct_size():
    text = "B" * 2000
    result = chunk_text(text, size=800, overlap=100)
    assert len(result[0]) == 800


def test_overlap_content_appears_in_next_chunk():
    # 800 A's + 800 B's; overlap=100 → chunk[1] starts at offset 700 → first 100 chars are A's
    text = "A" * 800 + "B" * 800
    result = chunk_text(text, size=800, overlap=100)
    assert result[1][:100] == "A" * 100


def test_exact_chunk_count():
    # 1600 chars, size=800, overlap=100 → step=700
    # chunk 0: [0:800], chunk 1: [700:1500], chunk 2: [1400:1600]
    text = "X" * 1600
    result = chunk_text(text, size=800, overlap=100)
    assert len(result) == 3


def test_leading_trailing_whitespace_stripped():
    text = "  " + "Z" * 100 + "  "
    result = chunk_text(text)
    assert result[0][0] == "Z"
    assert result[0][-1] == "Z"


def test_custom_small_chunk_size():
    text = "abcde" * 4  # 20 chars
    result = chunk_text(text, size=5, overlap=0)
    assert all(len(c) <= 5 for c in result)


def test_zero_overlap_no_repetition():
    text = "0123456789" * 10  # 100 chars, perfectly divisible by size=10
    result = chunk_text(text, size=10, overlap=0)
    joined = "".join(result)
    assert joined == text


def test_unicode_text():
    text = "Асан Байтемиров " * 100
    result = chunk_text(text, size=200, overlap=50)
    assert len(result) > 1
    for chunk in result:
        assert len(chunk) > 0


def test_single_char_chunk_size():
    result = chunk_text("abc", size=1, overlap=0)
    assert result == ["a", "b", "c"]


def test_chunk_text_default_params():
    """Default size=800, overlap=100. Text shorter than 800 → one chunk."""
    text = "Краткий текст"
    result = chunk_text(text)
    assert len(result) == 1
    assert result[0] == text


def test_all_chunks_nonempty():
    """Every returned chunk must contain at least one non-whitespace character."""
    text = "Word " * 300
    result = chunk_text(text, size=100, overlap=20)
    for chunk in result:
        assert chunk.strip() != ""