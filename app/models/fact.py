import uuid
from datetime import datetime
from sqlalchemy import Column, DateTime, Text, ForeignKey
from sqlalchemy.dialects.postgresql import UUID
from app.database import Base


class Fact(Base):
    __tablename__ = "facts"

    id          = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    document_id = Column(UUID(as_uuid=True), ForeignKey("documents.id", ondelete="CASCADE"), nullable=True)
    source_filename = Column(Text, nullable=True)
    icon        = Column(Text, nullable=True)
    category    = Column(Text, nullable=True)
    title       = Column(Text, nullable=False)
    body        = Column(Text, nullable=False)
    created_at  = Column(DateTime(timezone=True), default=datetime.utcnow)