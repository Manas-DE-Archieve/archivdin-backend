from uuid import UUID
from fastapi import APIRouter, Depends, Query
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func, text
from app.database import get_db
from app.models.fact import Fact
from pydantic import BaseModel
from typing import Optional, List
from datetime import datetime

router = APIRouter(prefix="/api/facts", tags=["facts"])


class FactOut(BaseModel):
    id: UUID
    document_id: Optional[UUID] = None
    source_filename: Optional[str] = None
    icon: Optional[str] = None
    category: Optional[str] = None
    title: str
    body: str
    created_at: datetime

    class Config:
        from_attributes = True


class FactsResponse(BaseModel):
    items: List[FactOut]
    total: int
    remaining: int  # unseen facts count after this batch


@router.get("", response_model=FactsResponse)
async def get_facts(
    limit: int = Query(6, ge=1, le=20),
    seen_ids: str = Query("", description="Comma-separated UUIDs already seen"),
    db: AsyncSession = Depends(get_db),
):
    """
    Returns `limit` unseen facts in random order.
    Client passes seen_ids to exclude already-shown facts.
    When all are seen, returns empty list with remaining=0.
    """
    total = (await db.execute(select(func.count()).select_from(Fact))).scalar_one()

    exclude: List[UUID] = []
    if seen_ids.strip():
        for s in seen_ids.split(","):
            try:
                exclude.append(UUID(s.strip()))
            except ValueError:
                pass

    query = select(Fact)
    if exclude:
        query = query.where(Fact.id.notin_(exclude))

    # Random order via PostgreSQL RANDOM()
    query = query.order_by(text("RANDOM()")).limit(limit)
    rows = (await db.execute(query)).scalars().all()

    unseen_count_q = select(func.count()).select_from(Fact)
    if exclude:
        unseen_count_q = unseen_count_q.where(Fact.id.notin_(exclude))
    unseen_total = (await db.execute(unseen_count_q)).scalar_one()
    remaining = max(0, unseen_total - len(rows))

    return FactsResponse(items=list(rows), total=total, remaining=remaining)