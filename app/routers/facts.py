import json
from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import text
from openai import AsyncOpenAI
from app.database import get_db
from app.config import get_settings

router = APIRouter(prefix="/api/facts", tags=["facts"])
settings = get_settings()


@router.get("")
async def get_interesting_facts(db: AsyncSession = Depends(get_db)):
    """
    Generates Wikipedia-style interesting facts about Kyrgyz repressions,
    enriched with real stats from the database.
    """
    # Get real stats from DB
    stats_result = await db.execute(
        text("""
            SELECT
                COUNT(*) AS total,
                COUNT(CASE WHEN rehabilitation_date IS NOT NULL THEN 1 END) AS rehabilitated,
                COUNT(DISTINCT region) AS regions,
                MIN(birth_year) AS earliest_birth,
                COUNT(DISTINCT occupation) AS occupations
            FROM persons
            WHERE full_name IS NOT NULL
        """)
    )
    stats = stats_result.mappings().first()

    top_region_result = await db.execute(
        text("""
            SELECT region, COUNT(*) as cnt
            FROM persons WHERE region IS NOT NULL
            GROUP BY region ORDER BY cnt DESC LIMIT 1
        """)
    )
    top_region = top_region_result.mappings().first()

    db_context = f"""
Реальные данные из архива:
- Записей в базе: {stats['total']}
- Реабилитировано: {stats['rehabilitated']}
- Охвачено регионов: {stats['regions']}
- Самый ранний год рождения: {stats['earliest_birth']}
- Разных профессий: {stats['occupations']}
- Самый пострадавший регион: {top_region['region'] if top_region else 'неизвестно'} ({top_region['cnt'] if top_region else 0} чел.)
"""

    prompt = f"""Ты — историк-архивист проекта «Архивдин Үнү» — цифрового мемориала жертв политических репрессий 1918–1953 годов в Кыргызстане.

{db_context}

Сгенерируй 6 интересных исторических фактов для раздела «Знаете ли вы?» в стиле Википедии.
Используй реальные данные из архива там, где они уместны.
Факты должны быть: исторически точными, разнообразными по тематике, интересными для широкой аудитории.

Верни ТОЛЬКО JSON (без markdown):
[
  {{
    "icon": "🏔",
    "category": "Название категории",
    "title": "Короткий заголовок факта",
    "body": "Развёрнутое описание 2-3 предложения."
  }}
]

Темы для покрытия: масштаб репрессий, профессии жертв, geography, методы репрессий, реабилитация, конкретные судьбы.
"""

    client = AsyncOpenAI(api_key=settings.openai_api_key)
    try:
        response = await client.chat.completions.create(
            model=settings.chat_model,
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
            max_tokens=1200,
        )
        content = response.choices[0].message.content or "{}"
        parsed = json.loads(content)
        # unwrap if nested
        if isinstance(parsed, dict):
            facts = next((v for v in parsed.values() if isinstance(v, list)), [])
        else:
            facts = parsed
    except Exception as e:
        print(f"ERROR: facts generation failed: {e}")
        facts = []

    return {"facts": facts, "db_stats": dict(stats)}