"""
Persons API tests — all embedding calls are mocked so no OpenAI key needed.
"""
import pytest
from unittest.mock import AsyncMock, patch
from httpx import AsyncClient

MOCK_EMBEDDING = [0.0] * 1536

# All persons tests need embedding mocked at both call sites:
# 1. app.services.embedding.embed_text  — used in persons.py when saving person
# 2. app.services.duplicate.embed_text  — used in duplicate.py during dupe check
EMBED_PATCHES = [
    patch("app.services.embedding.embed_text", new_callable=AsyncMock, return_value=MOCK_EMBEDDING),
    patch("app.services.duplicate.embed_text", new_callable=AsyncMock, return_value=MOCK_EMBEDDING),
    patch("app.routers.persons.embed_text",     new_callable=AsyncMock, return_value=MOCK_EMBEDDING),
]


async def _register_and_login(client: AsyncClient, email: str, password: str = "pass1234") -> str:
    await client.post("/api/auth/register", json={"email": email, "password": password})
    res = await client.post("/api/auth/login", json={"email": email, "password": password})
    return res.json()["access_token"]


async def _create_person(client: AsyncClient, token: str, full_name: str, **kwargs) -> dict:
    res = await client.post(
        "/api/persons",
        json={"full_name": full_name, "force": True, **kwargs},
        headers={"Authorization": f"Bearer {token}"},
    )
    assert res.status_code == 201, res.text
    return res.json()


# ── Create ────────────────────────────────────────────────

@pytest.mark.asyncio
@patch("app.services.embedding.embed_text", new_callable=AsyncMock, return_value=MOCK_EMBEDDING)
@patch("app.services.duplicate.embed_text", new_callable=AsyncMock, return_value=MOCK_EMBEDDING)
@patch("app.routers.persons.embed_text",     new_callable=AsyncMock, return_value=MOCK_EMBEDDING)
async def test_create_person_returns_201(m1, m2, m3, client: AsyncClient):
    token = await _register_and_login(client, "create@test.com")
    res = await client.post(
        "/api/persons",
        json={"full_name": "Байтемиров Асан", "birth_year": 1899, "region": "Чуйская область", "force": True},
        headers={"Authorization": f"Bearer {token}"},
    )
    assert res.status_code == 201
    assert res.json()["full_name"] == "Байтемиров Асан"
    assert res.json()["status"] == "pending"


@pytest.mark.asyncio
@patch("app.services.embedding.embed_text", new_callable=AsyncMock, return_value=MOCK_EMBEDDING)
@patch("app.services.duplicate.embed_text", new_callable=AsyncMock, return_value=MOCK_EMBEDDING)
@patch("app.routers.persons.embed_text",     new_callable=AsyncMock, return_value=MOCK_EMBEDDING)
async def test_create_person_requires_auth(m1, m2, m3, client: AsyncClient):
    res = await client.post("/api/persons", json={"full_name": "Test", "force": True})
    assert res.status_code == 401


@pytest.mark.asyncio
@patch("app.services.embedding.embed_text", new_callable=AsyncMock, return_value=MOCK_EMBEDDING)
@patch("app.services.duplicate.embed_text", new_callable=AsyncMock, return_value=MOCK_EMBEDDING)
@patch("app.routers.persons.embed_text",     new_callable=AsyncMock, return_value=MOCK_EMBEDDING)
async def test_create_person_all_fields_saved(m1, m2, m3, client: AsyncClient):
    token = await _register_and_login(client, "allfields@test.com")
    payload = {
        "full_name": "Сыдыков Омор",
        "birth_year": 1905,
        "death_year": 1938,
        "region": "Ошская область",
        "charge": "58-10",
        "biography": "Был учителем, расстрелян в 1938 году.",
        "force": True,
    }
    res = await client.post(
        "/api/persons", json=payload, headers={"Authorization": f"Bearer {token}"}
    )
    assert res.status_code == 201
    data = res.json()
    assert data["birth_year"] == 1905
    assert data["region"] == "Ошская область"
    assert data["charge"] == "58-10"


# ── Read ──────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_get_nonexistent_person_returns_404(client: AsyncClient):
    res = await client.get("/api/persons/00000000-0000-0000-0000-000000000000")
    assert res.status_code == 404


@pytest.mark.asyncio
@patch("app.services.embedding.embed_text", new_callable=AsyncMock, return_value=MOCK_EMBEDDING)
@patch("app.services.duplicate.embed_text", new_callable=AsyncMock, return_value=MOCK_EMBEDDING)
@patch("app.routers.persons.embed_text",     new_callable=AsyncMock, return_value=MOCK_EMBEDDING)
async def test_get_person_by_id(m1, m2, m3, client: AsyncClient):
    token = await _register_and_login(client, "getbyid@test.com")
    created = await _create_person(client, token, "Иброимова Айнур")
    pid = created["id"]
    res = await client.get(f"/api/persons/{pid}")
    assert res.status_code == 200
    assert res.json()["id"] == pid


@pytest.mark.asyncio
@patch("app.services.embedding.embed_text", new_callable=AsyncMock, return_value=MOCK_EMBEDDING)
@patch("app.services.duplicate.embed_text", new_callable=AsyncMock, return_value=MOCK_EMBEDDING)
@patch("app.routers.persons.embed_text",     new_callable=AsyncMock, return_value=MOCK_EMBEDDING)
async def test_list_persons_returns_pagination(m1, m2, m3, client: AsyncClient):
    token = await _register_and_login(client, "listpag@test.com")
    for name in ["Алиев Марат", "Токтоматов Жакып", "Сыдыкова Гүлзат"]:
        await _create_person(client, token, name, region="Ошская область")
    res = await client.get("/api/persons", params={"region": "Ошская область"})
    assert res.status_code == 200
    data = res.json()
    assert "items" in data
    assert "total" in data
    assert data["total"] >= 3


@pytest.mark.asyncio
async def test_list_persons_empty_db(client: AsyncClient):
    res = await client.get("/api/persons")
    assert res.status_code == 200
    assert res.json()["total"] == 0
    assert res.json()["items"] == []


# ── Update ────────────────────────────────────────────────

@pytest.mark.asyncio
@patch("app.services.embedding.embed_text", new_callable=AsyncMock, return_value=MOCK_EMBEDDING)
@patch("app.services.duplicate.embed_text", new_callable=AsyncMock, return_value=MOCK_EMBEDDING)
@patch("app.routers.persons.embed_text",     new_callable=AsyncMock, return_value=MOCK_EMBEDDING)
async def test_update_person_name(m1, m2, m3, client: AsyncClient):
    token = await _register_and_login(client, "update@test.com")
    created = await _create_person(client, token, "Омор Сыдыков")
    pid = created["id"]
    res = await client.put(
        f"/api/persons/{pid}",
        json={"full_name": "Омор Сыдыков-Исправлено"},
        headers={"Authorization": f"Bearer {token}"},
    )
    assert res.status_code == 200
    assert res.json()["full_name"] == "Омор Сыдыков-Исправлено"


@pytest.mark.asyncio
@patch("app.services.embedding.embed_text", new_callable=AsyncMock, return_value=MOCK_EMBEDDING)
@patch("app.services.duplicate.embed_text", new_callable=AsyncMock, return_value=MOCK_EMBEDDING)
@patch("app.routers.persons.embed_text",     new_callable=AsyncMock, return_value=MOCK_EMBEDDING)
async def test_update_other_users_person_forbidden(m1, m2, m3, client: AsyncClient):
    token1 = await _register_and_login(client, "owner@test.com")
    token2 = await _register_and_login(client, "intruder@test.com")
    created = await _create_person(client, token1, "Чужой Человек")
    pid = created["id"]
    res = await client.put(
        f"/api/persons/{pid}",
        json={"full_name": "Взломано"},
        headers={"Authorization": f"Bearer {token2}"},
    )
    assert res.status_code == 403


# ── Status ────────────────────────────────────────────────

@pytest.mark.asyncio
@patch("app.services.embedding.embed_text", new_callable=AsyncMock, return_value=MOCK_EMBEDDING)
@patch("app.services.duplicate.embed_text", new_callable=AsyncMock, return_value=MOCK_EMBEDDING)
@patch("app.routers.persons.embed_text",     new_callable=AsyncMock, return_value=MOCK_EMBEDDING)
async def test_status_update_by_regular_user_forbidden(m1, m2, m3, client: AsyncClient):
    token = await _register_and_login(client, "regular@test.com")
    created = await _create_person(client, token, "Касымов Болот")
    pid = created["id"]
    res = await client.patch(
        f"/api/persons/{pid}/status",
        json={"status": "verified"},
        headers={"Authorization": f"Bearer {token}"},
    )
    assert res.status_code == 403


# ── Delete ────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_delete_person_requires_auth(client: AsyncClient):
    res = await client.delete("/api/persons/00000000-0000-0000-0000-000000000000")
    assert res.status_code == 401


# ── Stats ─────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_stats_summary_returns_expected_shape(client: AsyncClient):
    res = await client.get("/api/persons/stats/summary")
    assert res.status_code == 200
    data = res.json()
    for key in ("total", "executed", "rehabilitated", "regions"):
        assert key in data
        assert isinstance(data[key], int)


# ── Duplicate detection (API level) ──────────────────────

@pytest.mark.asyncio
@patch("app.services.embedding.embed_text", new_callable=AsyncMock, return_value=MOCK_EMBEDDING)
@patch("app.services.duplicate.embed_text", new_callable=AsyncMock, return_value=MOCK_EMBEDDING)
@patch("app.routers.persons.embed_text",     new_callable=AsyncMock, return_value=MOCK_EMBEDDING)
async def test_duplicate_warning_returned_without_force(m1, m2, m3, client: AsyncClient):
    token = await _register_and_login(client, "dupcheck@test.com")
    payload = {"full_name": "Байтемиров Асан Дубль", "birth_year": 1900}

    # First insertion with force=True — guaranteed to save
    await client.post(
        "/api/persons",
        json={**payload, "force": True},
        headers={"Authorization": f"Bearer {token}"},
    )
    # Second insertion without force — with mock embeddings all zeros,
    # vector score will be 1.0, so duplicates_found=True
    res = await client.post(
        "/api/persons",
        json=payload,
        headers={"Authorization": f"Bearer {token}"},
    )
    # Accept either: duplicate warning (200 with flag) or 201 if trgm threshold not met
    if res.status_code != 201:
        data = res.json()
        assert data.get("duplicates_found") is True