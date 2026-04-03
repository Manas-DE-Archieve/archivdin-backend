"""
Generates 0–3 historical facts from a document's text and saves them to the DB.
Used both during upload and by the backfill script.
"""
import json
from typing import List
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from openai import AsyncOpenAI
from app.config import get_settings
from app.models.fact import Fact

settings = get_settings()


async def generate_and_save_facts(
    db: AsyncSession,
    document_id,
    filename: str,
    raw_text: str,
) -> List[Fact]:
    """Generate 0–3 facts from document text and persist them. Idempotent."""

    # Skip if facts already exist for this document
    existing = await db.execute(
        select(Fact).where(Fact.document_id == document_id).limit(1)
    )
    if existing.scalar_one_or_none():
        return []

    snippet = raw_text[:3000].strip()
    if not snippet:
        return []

    client = AsyncOpenAI(api_key=settings.openai_api_key)
    prompt = f"""Ты — историк-архивист проекта «Архивдин Үнү» (Голос из архива).
Прочитай документ и извлеки 1–3 интересных исторических факта, которые будут полезны широкой аудитории.
Если документ содержит только технические данные без исторической ценности — верни пустой массив.

Документ: {filename}
Текст:
{snippet}

Верни ТОЛЬКО JSON (без markdown):
[
  {{
    "icon": "🏔",
    "category": "Название категории (1–3 слова)",
    "title": "Краткий заголовок факта (до 10 слов)",
    "body": "Развёрнутое описание 2–3 предложения с историческим контекстом."
  }}
]
Максимум 3 факта. Если нечего извлечь — верни [].
"""

    try:
        response = await client.chat.completions.create(
            model=settings.chat_model,
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
            max_tokens=800,
        )
        content = response.choices[0].message.content or "{}"
        parsed = json.loads(content)
        if isinstance(parsed, dict):
            arr = next((v for v in parsed.values() if isinstance(v, list)), [])
        else:
            arr = parsed
    except Exception as e:
        print(f"WARNING: facts generation failed for '{filename}': {e}")
        return []

    facts = []
    for item in arr[:3]:
        if not item.get("title") or not item.get("body"):
            continue
        fact = Fact(
            document_id=document_id,
            source_filename=filename,
            icon=item.get("icon", "📖"),
            category=item.get("category", "История"),
            title=item["title"],
            body=item["body"],
        )
        db.add(fact)
        facts.append(fact)

    if facts:
        await db.commit()
        print(f"INFO:     Generated {len(facts)} facts from '{filename}'")

    return facts