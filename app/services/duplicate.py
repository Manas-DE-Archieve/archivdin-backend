import json
from typing import List
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import text, select
from openai import AsyncOpenAI
from app.services.embedding import embed_text
from app.config import get_settings

settings = get_settings()


async def find_duplicates(
    db: AsyncSession, full_name: str, threshold: float = 0.4, limit: int = 5
) -> List[dict]:
    """
    Two-phase duplicate detection for persons:
    1. Fuzzy name match via pg_trgm (fast)
    2. Semantic similarity via pgvector (precise)
    Returns merged, deduplicated list ranked by best score.
    """
    trgm_result = await db.execute(
        text("""
            SELECT id, full_name, birth_year, region,
                   similarity(full_name, :name) AS score
            FROM persons
            WHERE similarity(full_name, :name) > :threshold
            ORDER BY score DESC
            LIMIT :limit
        """),
        {"name": full_name, "threshold": threshold, "limit": limit}
    )
    trgm_rows = trgm_result.mappings().all()

    name_embedding = await embed_text(full_name)
    vec_str = "[" + ",".join(str(x) for x in name_embedding) + "]"

    vec_result = await db.execute(
        text("""
            SELECT id, full_name, birth_year, region,
                1 - (name_embedding <=> CAST(:vec AS vector)) AS score
            FROM persons
            WHERE name_embedding IS NOT NULL
              AND 1 - (name_embedding <=> CAST(:vec AS vector)) > :threshold
            ORDER BY name_embedding <=> CAST(:vec AS vector)
            LIMIT :limit
        """),
        {"vec": vec_str, "limit": limit, "threshold": threshold}
    )
    vec_rows = vec_result.mappings().all()

    merged: dict[str, dict] = {}
    for row in list(trgm_rows) + list(vec_rows):
        pid = str(row["id"])
        score = float(row["score"])
        if pid not in merged or merged[pid]["similarity_score"] < score:
            merged[pid] = {
                "id": row["id"],
                "full_name": row["full_name"],
                "birth_year": row["birth_year"],
                "region": row["region"],
                "similarity_score": round(score, 4),
            }

    candidates = sorted(merged.values(), key=lambda x: x["similarity_score"], reverse=True)
    return candidates[:limit]


async def find_similar_documents(
    db: AsyncSession, raw_text: str, threshold: float = 0.60, limit: int = 5
) -> List[dict]:
    """
    Phase 1: vector similarity search on chunk embeddings.
    Uses first 3000 chars for embedding to better capture overall content.
    """
    sample = raw_text[:3000].strip()
    if not sample:
        return []

    embedding = await embed_text(sample)
    vec_str = "[" + ",".join(str(x) for x in embedding) + "]"

    result = await db.execute(
        text("""
            SELECT
                d.id,
                d.filename,
                d.raw_text,
                AVG(1 - (c.embedding <=> CAST(:vec AS vector))) AS avg_score
            FROM chunks c
            JOIN documents d ON d.id = c.document_id
            WHERE c.embedding IS NOT NULL
            GROUP BY d.id, d.filename, d.raw_text
            HAVING AVG(1 - (c.embedding <=> CAST(:vec AS vector))) > :threshold
            ORDER BY avg_score DESC
            LIMIT :limit
        """),
        {"vec": vec_str, "threshold": threshold, "limit": limit},
    )
    rows = result.mappings().all()
    return [
        {
            "id": row["id"],
            "filename": row["filename"],
            "raw_text": row["raw_text"] or "",
            "similarity_score": round(float(row["avg_score"]), 4),
        }
        for row in rows
    ]


async def validate_duplicates_with_llm(
    uploaded_text: str,
    candidates: List[dict],
) -> List[dict]:
    """
    Phase 2: LLM decides which candidates are TRUE duplicates.
    A "duplicate" = same document even if slightly edited/reformatted.
    """
    if not candidates:
        return []

    client = AsyncOpenAI(api_key=settings.openai_api_key)

    candidate_blocks = ""
    for i, c in enumerate(candidates, 1):
        snippet = c["raw_text"][:800].replace("\n", " ")
        candidate_blocks += f"\n[Документ {i}] id={c['id']} filename={c['filename']}\n{snippet}\n"

    uploaded_snippet = uploaded_text[:800].replace("\n", " ")

    prompt = f"""Ты — система обнаружения дубликатов архивных документов.

Загружаемый документ:
{uploaded_snippet}

Кандидаты из архива:
{candidate_blocks}

Твоя задача: определить, является ли загружаемый документ дубликатом любого из кандидатов.

ПРАВИЛА:
- Дубликат = тот же документ, даже если изменено несколько слов, фраз, дат или имён
- Дубликат = тот же документ с небольшими правками, дополнениями или сокращениями
- НЕ дубликат = разные документы на похожую тему о разных людях или событиях

Верни ТОЛЬКО JSON-массив (без markdown):
[{{"id": "<uuid>", "is_duplicate": true/false, "score": 0.0-1.0}}]
"""

    try:
        response = await client.chat.completions.create(
            model=settings.chat_model,
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
            max_tokens=300,
        )
        content = response.choices[0].message.content or "{}"
        parsed = json.loads(content)
        if isinstance(parsed, dict):
            arr = next((v for v in parsed.values() if isinstance(v, list)), [])
        else:
            arr = parsed
    except Exception as e:
        print(f"WARNING: LLM duplicate validation failed: {e}. Falling back to vector scores.")
        # Fallback: return candidates with score > 0.70
        return [
            {"id": str(c["id"]), "filename": c["filename"], "similarity_score": c["similarity_score"]}
            for c in candidates if c["similarity_score"] > 0.70
        ]

    id_to_meta = {str(c["id"]): c for c in candidates}
    confirmed = []
    for item in arr:
        if item.get("is_duplicate") and str(item.get("id")) in id_to_meta:
            meta = id_to_meta[str(item["id"])]
            confirmed.append({
                "id": meta["id"],
                "filename": meta["filename"],
                "similarity_score": round(float(item.get("score", meta["similarity_score"])), 4),
            })

    return confirmed