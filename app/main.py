from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.database import init_db, AsyncSessionLocal
from app.routers import auth, persons, documents, chat, admin, facts
from sqlalchemy import text


async def ensure_pg_extensions():
    """Enable pg_trgm extension required for similarity() function."""
    async with AsyncSessionLocal() as db:
        try:
            await db.execute(text("CREATE EXTENSION IF NOT EXISTS pg_trgm"))
            await db.commit()
            print("INFO:     pg_trgm extension ensured.")
        except Exception as e:
            print(f"WARNING:  Could not create pg_trgm extension: {e}")
            await db.rollback()


@asynccontextmanager
async def lifespan(app: FastAPI):
    await init_db()
    await ensure_pg_extensions()
    yield


app = FastAPI(
    title="Архивдин Үнү API",
    description="Digital archive of repressed people (1918–1953) with RAG assistant",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:5173", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(auth.router)
app.include_router(persons.router)
app.include_router(documents.router)
app.include_router(chat.router)
app.include_router(admin.router)
app.include_router(facts.router)


@app.get("/health")
async def health():
    return {"status": "ok"}