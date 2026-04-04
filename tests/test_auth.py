"""
Auth endpoint tests.
Uses real HTTP client with in-memory test DB (configured in conftest.py).
"""
import pytest
from httpx import AsyncClient


@pytest.mark.asyncio
async def test_register_success(client: AsyncClient):
    res = await client.post("/api/auth/register", json={"email": "new_user@test.com", "password": "secret123"})
    assert res.status_code == 201
    data = res.json()
    assert data["email"] == "new_user@test.com"
    assert data["role"] == "user"
    assert "id" in data
    assert "password_hash" not in data


@pytest.mark.asyncio
async def test_register_duplicate_email_rejected(client: AsyncClient):
    payload = {"email": "dup@test.com", "password": "secret123"}
    await client.post("/api/auth/register", json=payload)
    res = await client.post("/api/auth/register", json=payload)
    assert res.status_code == 400


@pytest.mark.asyncio
async def test_login_returns_tokens(client: AsyncClient):
    await client.post("/api/auth/register", json={"email": "login@test.com", "password": "pass1234"})
    res = await client.post("/api/auth/login", json={"email": "login@test.com", "password": "pass1234"})
    assert res.status_code == 200
    tokens = res.json()
    assert "access_token" in tokens
    assert "refresh_token" in tokens
    assert tokens["token_type"] == "bearer"


@pytest.mark.asyncio
async def test_login_wrong_password_returns_401(client: AsyncClient):
    await client.post("/api/auth/register", json={"email": "wrong@test.com", "password": "correct"})
    res = await client.post("/api/auth/login", json={"email": "wrong@test.com", "password": "incorrect"})
    assert res.status_code == 401


@pytest.mark.asyncio
async def test_login_nonexistent_user_returns_401(client: AsyncClient):
    res = await client.post("/api/auth/login", json={"email": "ghost@test.com", "password": "pass"})
    assert res.status_code == 401


@pytest.mark.asyncio
async def test_me_endpoint_requires_auth(client: AsyncClient):
    res = await client.get("/api/auth/me")
    assert res.status_code == 401


@pytest.mark.asyncio
async def test_me_endpoint_returns_user(client: AsyncClient):
    await client.post("/api/auth/register", json={"email": "me@test.com", "password": "pass1234"})
    login = await client.post("/api/auth/login", json={"email": "me@test.com", "password": "pass1234"})
    token = login.json()["access_token"]

    res = await client.get("/api/auth/me", headers={"Authorization": f"Bearer {token}"})
    assert res.status_code == 200
    assert res.json()["email"] == "me@test.com"


@pytest.mark.asyncio
async def test_me_with_invalid_token_returns_401(client: AsyncClient):
    res = await client.get("/api/auth/me", headers={"Authorization": "Bearer invalidtoken"})
    assert res.status_code == 401


@pytest.mark.asyncio
async def test_refresh_token_works(client: AsyncClient):
    await client.post("/api/auth/register", json={"email": "refresh@test.com", "password": "pass1234"})
    login = await client.post("/api/auth/login", json={"email": "refresh@test.com", "password": "pass1234"})
    refresh_token = login.json()["refresh_token"]

    res = await client.post("/api/auth/refresh", params={"token": refresh_token})
    assert res.status_code == 200
    assert "access_token" in res.json()


@pytest.mark.asyncio
async def test_refresh_with_access_token_fails(client: AsyncClient):
    """Access token must not be accepted as a refresh token."""
    await client.post("/api/auth/register", json={"email": "badrefresh@test.com", "password": "pass1234"})
    login = await client.post("/api/auth/login", json={"email": "badrefresh@test.com", "password": "pass1234"})
    access_token = login.json()["access_token"]

    res = await client.post("/api/auth/refresh", params={"token": access_token})
    assert res.status_code == 400


@pytest.mark.asyncio
async def test_setup_super_admin_creates_admin(client: AsyncClient):
    res = await client.post(
        "/api/auth/setup-super-admin",
        json={"email": "admin@test.com", "password": "adminpass"},
    )
    assert res.status_code == 201
    assert res.json()["role"] == "super_admin"


@pytest.mark.asyncio
async def test_setup_super_admin_second_call_fails(client: AsyncClient):
    """Only one super_admin can be created via this endpoint."""
    await client.post(
        "/api/auth/setup-super-admin",
        json={"email": "admin1@test.com", "password": "adminpass"},
    )
    res = await client.post(
        "/api/auth/setup-super-admin",
        json={"email": "admin2@test.com", "password": "adminpass"},
    )
    assert res.status_code == 400