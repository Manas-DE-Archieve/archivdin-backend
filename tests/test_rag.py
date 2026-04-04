"""
RAG service tests — all OpenAI calls mocked.
"""
import json
import pytest
from unittest.mock import AsyncMock, MagicMock, patch


# ── Helpers ───────────────────────────────────────────────

def _make_db_mock_with_chunks(rows):
    """DB mock that returns given rows from execute().mappings().all()"""
    db = AsyncMock()
    mock_result = MagicMock()
    mock_result.mappings.return_value.all.return_value = rows
    db.execute = AsyncMock(return_value=mock_result)
    return db


def _make_stream_mock(tokens: list[str]):
    """Build an async iterator that yields fake OpenAI streaming chunks."""
    async def _aiter():
        for t in tokens:
            event = MagicMock()
            event.choices = [MagicMock()]
            event.choices[0].delta.content = t
            yield event
        # final chunk with no content
        end = MagicMock()
        end.choices = [MagicMock()]
        end.choices[0].delta.content = None
        yield end

    return _aiter()


# ── retrieve_chunks ───────────────────────────────────────

@pytest.mark.asyncio
@patch("app.services.rag.embed_text", new_callable=AsyncMock, return_value=[0.0] * 1536)
async def test_retrieve_chunks_returns_correct_shape(mock_emb):
    from app.services.rag import retrieve_chunks

    row = {
        "id": "chunk-uuid",
        "chunk_text": "Исторический текст",
        "document_id": "doc-uuid",
        "filename": "delo.txt",
        "score": 0.88,
    }
    db = _make_db_mock_with_chunks([row])

    chunks = await retrieve_chunks(db, "вопрос", top_k=3)
    assert len(chunks) == 1
    assert chunks[0]["document_name"] == "delo.txt"
    assert chunks[0]["chunk_text"] == "Исторический текст"
    assert chunks[0]["score"] == 0.88


@pytest.mark.asyncio
@patch("app.services.rag.embed_text", new_callable=AsyncMock, return_value=[0.0] * 1536)
async def test_retrieve_chunks_empty_db_returns_empty(mock_emb):
    from app.services.rag import retrieve_chunks

    db = _make_db_mock_with_chunks([])
    chunks = await retrieve_chunks(db, "вопрос без ответа")
    assert chunks == []


@pytest.mark.asyncio
@patch("app.services.rag.embed_text", new_callable=AsyncMock, return_value=[0.0] * 1536)
async def test_retrieve_chunks_score_rounded(mock_emb):
    from app.services.rag import retrieve_chunks

    row = {
        "id": "x",
        "chunk_text": "text",
        "document_id": "doc-x",
        "filename": "f.txt",
        "score": 0.9999999,
    }
    db = _make_db_mock_with_chunks([row])

    chunks = await retrieve_chunks(db, "q")
    score_str = str(chunks[0]["score"])
    decimals = len(score_str.split(".")[-1]) if "." in score_str else 0
    assert decimals <= 4


@pytest.mark.asyncio
@patch("app.services.rag.embed_text", new_callable=AsyncMock, return_value=[0.0] * 1536)
async def test_retrieve_chunks_returns_document_id(mock_emb):
    from app.services.rag import retrieve_chunks

    row = {
        "id": "chunk-1",
        "chunk_text": "текст чанка",
        "document_id": "doc-123",
        "filename": "archive.txt",
        "score": 0.75,
    }
    db = _make_db_mock_with_chunks([row])

    chunks = await retrieve_chunks(db, "вопрос")
    assert "document_id" in chunks[0]
    assert chunks[0]["document_id"] == "doc-123"


# ── stream_rag_answer ─────────────────────────────────────

@pytest.mark.asyncio
@patch("app.services.rag.embed_text", new_callable=AsyncMock, return_value=[0.0] * 1536)
@patch("app.services.rag.AsyncOpenAI")
async def test_stream_emits_sources_event_first(mock_openai_cls, mock_emb):
    from app.services.rag import stream_rag_answer

    db = _make_db_mock_with_chunks([
        {"id": "c1", "chunk_text": "Текст", "document_id": "d1", "filename": "doc.txt", "score": 0.9}
    ])

    mock_client = MagicMock()
    mock_client.chat.completions.create = AsyncMock(return_value=_make_stream_mock(["Ответ"]))
    mock_openai_cls.return_value = mock_client

    events = []
    async for line in stream_rag_answer(db, "Вопрос", []):
        events.append(line)

    assert events, "No events emitted"
    first = json.loads(events[0][6:])
    assert first["type"] == "sources"


@pytest.mark.asyncio
@patch("app.services.rag.embed_text", new_callable=AsyncMock, return_value=[0.0] * 1536)
@patch("app.services.rag.AsyncOpenAI")
async def test_stream_emits_done_event_last(mock_openai_cls, mock_emb):
    from app.services.rag import stream_rag_answer

    db = _make_db_mock_with_chunks([])

    mock_client = MagicMock()
    mock_client.chat.completions.create = AsyncMock(return_value=_make_stream_mock(["Hello"]))
    mock_openai_cls.return_value = mock_client

    events = []
    async for line in stream_rag_answer(db, "Q", []):
        events.append(line)

    last = json.loads(events[-1][6:])
    assert last["type"] == "done"


@pytest.mark.asyncio
@patch("app.services.rag.embed_text", new_callable=AsyncMock, return_value=[0.0] * 1536)
@patch("app.services.rag.AsyncOpenAI")
async def test_stream_emits_token_events(mock_openai_cls, mock_emb):
    from app.services.rag import stream_rag_answer

    db = _make_db_mock_with_chunks([])

    mock_client = MagicMock()
    mock_client.chat.completions.create = AsyncMock(
        return_value=_make_stream_mock(["Го", "лос", " из", " архива"])
    )
    mock_openai_cls.return_value = mock_client

    token_events = []
    async for line in stream_rag_answer(db, "Q", []):
        payload = json.loads(line[6:])
        if payload["type"] == "token":
            token_events.append(payload["data"])

    assert "".join(token_events) == "Голос из архива"


@pytest.mark.asyncio
@patch("app.services.rag.embed_text", new_callable=AsyncMock, return_value=[0.0] * 1536)
@patch("app.services.rag.AsyncOpenAI")
async def test_stream_sources_contain_chunk_fields(mock_openai_cls, mock_emb):
    from app.services.rag import stream_rag_answer

    db = _make_db_mock_with_chunks([
        {"id": "abc", "chunk_text": "Важный фрагмент", "document_id": "d1", "filename": "archive.txt", "score": 0.77}
    ])

    mock_client = MagicMock()
    mock_client.chat.completions.create = AsyncMock(return_value=_make_stream_mock([]))
    mock_openai_cls.return_value = mock_client

    async for line in stream_rag_answer(db, "Q", []):
        payload = json.loads(line[6:])
        if payload["type"] == "sources":
            src = payload["data"][0]
            assert src["document_name"] == "archive.txt"
            assert src["chunk_text"] == "Важный фрагмент"
            assert src["score"] == 0.77
            break


@pytest.mark.asyncio
@patch("app.services.rag.embed_text", new_callable=AsyncMock, return_value=[0.0] * 1536)
@patch("app.services.rag.AsyncOpenAI")
async def test_stream_event_count_sources_plus_tokens_plus_done(mock_openai_cls, mock_emb):
    from app.services.rag import stream_rag_answer

    db = _make_db_mock_with_chunks([])

    mock_client = MagicMock()
    mock_client.chat.completions.create = AsyncMock(
        return_value=_make_stream_mock(["A", "B", "C"])
    )
    mock_openai_cls.return_value = mock_client

    types = []
    async for line in stream_rag_answer(db, "Q", []):
        types.append(json.loads(line[6:])["type"])

    # sources → token × 3 → done
    assert types[0] == "sources"
    assert types[-1] == "done"
    assert types.count("token") == 3