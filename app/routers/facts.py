from uuid import UUID
from fastapi import APIRouter, Depends, Query
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func
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
    page: int
    limit: int


@router.get("", response_model=FactsResponse)
async def list_facts(
    page: int = Query(1, ge=1),
    limit: int = Query(12, ge=1, le=50),
    db: AsyncSession = Depends(get_db),
):
    count = (await db.execute(select(func.count()).select_from(Fact))).scalar_one()
    offset = (page - 1) * limit
    rows = (await db.execute(
        select(Fact).order_by(Fact.created_at.desc()).offset(offset).limit(limit)
    )).scalars().all()
    return FactsResponse(items=list(rows), total=count, page=page, limit=limit)