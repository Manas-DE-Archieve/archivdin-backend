"""
Duplicate detection service tests — fully mocked, no DB or OpenAI needed.
"""
import uuid
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from app.services.duplicate import find_duplicates

MOCK_EMBEDDING = [0.1] * 1536


def _make_db_mock(trgm_rows=None, vec_rows=None):
    """Returns an AsyncMock db that yields trgm_rows on first call, vec_rows on second."""
    trgm_rows = trgm_rows or []
    vec_rows = vec_rows or []
    call_count = 0

    async def fake_execute(query, params=None):
        nonlocal call_count
        m = MagicMock()
        if call_count == 0:
            m.mappings.return_value.all.return_value = trgm_rows
        else:
            m.mappings.return_value.all.return_value = vec_rows
        call_count += 1
        return m

    db = AsyncMock()
    db.execute = fake_execute
    return db


@pytest.mark.asyncio
@patch("app.services.duplicate.embed_text", new_callable=AsyncMock, return_value=MOCK_EMBEDDING)
async def test_empty_db_returns_empty_list(mock_emb):
    db = _make_db_mock()
    result = await find_duplicates(db, "Новый Человек")
    assert result == []


@pytest.mark.asyncio
@patch("app.services.duplicate.embed_text", new_callable=AsyncMock, return_value=MOCK_EMBEDDING)
async def test_single_trgm_match_returned(mock_emb):
    pid = uuid.uuid4()
    trgm_row = {"id": pid, "full_name": "Алиев Марат", "birth_year": 1900, "region": "Ош", "score": 0.80}
    db = _make_db_mock(trgm_rows=[trgm_row])
    result = await find_duplicates(db, "Алиев Марат")
    assert len(result) == 1
    assert result[0]["full_name"] == "Алиев Марат"


@pytest.mark.asyncio
@patch("app.services.duplicate.embed_text", new_callable=AsyncMock, return_value=MOCK_EMBEDDING)
async def test_same_person_in_both_phases_deduplicated(mock_emb):
    """Person appearing in both trgm AND vector results should appear only once."""
    pid = uuid.uuid4()
    row = {"id": pid, "full_name": "Алиев Марат", "birth_year": 1900, "region": "Ош", "score": 0.85}
    vec_row = {"id": pid, "full_name": "Алиев Марат", "birth_year": 1900, "region": "Ош", "score": 0.90}
    db = _make_db_mock(trgm_rows=[row], vec_rows=[vec_row])
    result = await find_duplicates(db, "Алиев Марат")
    assert len(result) == 1


@pytest.mark.asyncio
@patch("app.services.duplicate.embed_text", new_callable=AsyncMock, return_value=MOCK_EMBEDDING)
async def test_highest_score_kept_after_deduplication(mock_emb):
    """When same person appears in both phases, the HIGHER score wins."""
    pid = uuid.uuid4()
    trgm_row = {"id": pid, "full_name": "X", "birth_year": None, "region": None, "score": 0.70}
    vec_row  = {"id": pid, "full_name": "X", "birth_year": None, "region": None, "score": 0.95}
    db = _make_db_mock(trgm_rows=[trgm_row], vec_rows=[vec_row])
    result = await find_duplicates(db, "X")
    assert result[0]["similarity_score"] == 0.95


@pytest.mark.asyncio
@patch("app.services.duplicate.embed_text", new_callable=AsyncMock, return_value=MOCK_EMBEDDING)
async def test_results_sorted_by_score_descending(mock_emb):
    """Results must be sorted highest score first."""
    id1, id2 = uuid.uuid4(), uuid.uuid4()
    trgm_rows = [
        {"id": id1, "full_name": "А", "birth_year": None, "region": None, "score": 0.60},
        {"id": id2, "full_name": "Б", "birth_year": None, "region": None, "score": 0.90},
    ]
    db = _make_db_mock(trgm_rows=trgm_rows)
    result = await find_duplicates(db, "test")
    assert result[0]["similarity_score"] >= result[1]["similarity_score"]


@pytest.mark.asyncio
@patch("app.services.duplicate.embed_text", new_callable=AsyncMock, return_value=MOCK_EMBEDDING)
async def test_limit_respected(mock_emb):
    """find_duplicates should return at most `limit` results."""
    rows = [
        {"id": uuid.uuid4(), "full_name": f"Person {i}", "birth_year": None, "region": None, "score": 0.8}
        for i in range(10)
    ]
    db = _make_db_mock(trgm_rows=rows)
    result = await find_duplicates(db, "Person", limit=3)
    assert len(result) <= 3


@pytest.mark.asyncio
@patch("app.services.duplicate.embed_text", new_callable=AsyncMock, return_value=MOCK_EMBEDDING)
async def test_score_rounded_to_4_decimals(mock_emb):
    """similarity_score field should have at most 4 decimal places."""
    pid = uuid.uuid4()
    row = {"id": pid, "full_name": "Test", "birth_year": None, "region": None, "score": 0.123456789}
    db = _make_db_mock(trgm_rows=[row])
    result = await find_duplicates(db, "Test")
    score_str = str(result[0]["similarity_score"])
    decimals = len(score_str.split(".")[-1]) if "." in score_str else 0
    assert decimals <= 4