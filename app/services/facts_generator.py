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
    prompt = f"""Ты — автор сильных документальных историй о репрессиях, работающий в стиле Netflix / BBC / Meduza.

Твоя задача — найти в архивном документе 1–3 факта, которые:
— пробирают до мурашек
— вызывают сочувствие, шок или ощущение абсурда системы
— выглядят как начало мощной человеческой истории

❗️Пиши так, чтобы читатель НЕ СМОГ пролистать дальше.

---

🎯 Что искать в тексте:

1. ЧЕЛОВЕЧЕСКАЯ ТРАГЕДИЯ  
— судьба конкретного человека  
— арест в важный момент (день рождения, свадьба, ночью)  
— исчезновение без объяснения  
— судьба семьи  

2. АБСУРД СИСТЕМЫ  
— нелепые обвинения (шпионаж, "враг народа" без доказательств)  
— противоречия  
— мелочь → тяжёлое наказание  

3. МАСШТАБ УЖАСА  
— много людей за короткое время  
— массовые приговоры  
— “конвейер”  

4. СОЦИАЛЬНЫЙ СРЕЗ  
— кого именно забирали (учителей, крестьян, стариков, подростков)  
— разрушение обычной жизни  

---

✍️ Как писать:

— Начинай с сильного крючка (как первая строка фильма)  
— Делай текст живым и визуальным  
— Можно использовать короткие, рубленые предложения  
— НО: всё должно быть строго основано на документе (никакой выдумки)  

Примеры начала:
— "Ему было всего 19, когда..."  
— "В одну ночь исчезла целая семья..."  
— "Его обвинили в шпионаже за..."  

---

📦 Формат ответа:

Верни JSON:

{{
  "facts": [
    {{
      "icon": "💔",
      "category": "ТРАГЕДИЯ / АБСУРД / МАСШТАБ",
      "title": "Очень цепляющий заголовок (как новость или фильм)",
      "body": "Короткая, эмоциональная, но точная история (2–4 предложения)."
    }}
  ]
}}

---

📌 Ограничения:

— НЕ выдумывай факты  
— НЕ добавляй то, чего нет в тексте  
— НЕ используй общий язык типа “многие пострадали”  
— Каждый факт должен быть конкретным  

---

Документ: {filename}  
Текст: {snippet}
"""

    try:
        response = await client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
            max_tokens=800,
        )
        content = response.choices[0].message.content or "{}"
        parsed = json.loads(content)
        arr = parsed.get("facts", [])
        if not arr and isinstance(parsed, dict):
            for v in parsed.values():
                if isinstance(v, list):
                    arr = v
                    break
    except Exception as e:
        import traceback
        print(f"WARNING: facts generation failed for '{filename}': {e}")
        print(traceback.format_exc())
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