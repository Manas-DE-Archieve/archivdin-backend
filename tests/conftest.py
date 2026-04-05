#
# ПРАВИЛЬНОЕ СОДЕРЖИМОЕ ФАЙЛА backend/tests/conftest.py
#
import os
import pytest
import pytest_asyncio
from dotenv import load_dotenv

# Load test env vars if .env.test exists
load_dotenv(os.path.join(os.path.dirname(__file__), '..', '.env.test'), override=False)
from httpx import AsyncClient, ASGITransport
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
from sqlalchemy import text

DB_USER = os.getenv("POSTGRES_USER", "postgres")
DB_PASS = os.getenv("POSTGRES_PASSWORD", os.getenv("DB_PASSWORD", "strongpassword123"))
DB_HOST = os.getenv("POSTGRES_HOST", "localhost")
DB_PORT = os.getenv("POSTGRES_PORT", "5433")
DB_NAME = os.getenv("POSTGRES_TEST_DB", "archive_test")

TEST_DB_URL = f"postgresql+asyncpg://{DB_USER}:{DB_PASS}@{DB_HOST}:{DB_PORT}/{DB_NAME}"


@pytest_asyncio.fixture(scope="function")  # Изоляция для каждого теста
async def db_engine():
    """Создает новый движок и чистую базу данных для КАЖДОГО теста."""
    engine = create_async_engine(TEST_DB_URL, echo=False)

    from app.database import Base
    async with engine.begin() as conn:
        await conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
        await conn.execute(text("CREATE EXTENSION IF NOT EXISTS pg_trgm"))
        # Пересоздаем таблицы перед каждым тестом для полной изоляции
        await conn.run_sync(Base.metadata.drop_all)
        await conn.run_sync(Base.metadata.create_all)

    yield engine

    # Безопасно уничтожаем движок после завершения теста
    await engine.dispose()


@pytest_asyncio.fixture
async def client(db_engine):
    """Свежий HTTP клиент и свежая сессия БД для каждого теста."""
    from app.main import app
    from app.database import get_db

    TestSession = async_sessionmaker(db_engine, class_=AsyncSession, expire_on_commit=False)

    async def override_get_db():
        async with TestSession() as session:
            yield session

    app.dependency_overrides[get_db] = override_get_db

    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as ac:
        yield ac

    app.dependency_overrides.pop(get_db, None)


@pytest_asyncio.fixture
async def db(db_engine):
    TestSession = async_sessionmaker(db_engine, class_=AsyncSession, expire_on_commit=False)
    async with TestSession() as session:
        yield session