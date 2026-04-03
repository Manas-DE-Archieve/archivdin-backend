"""
Auth endpoint tests — uses real HTTP client against in-memory app (no live DB needed for logic tests).
The conftest.py already sets up a test DB; these tests mock at the service level where possible.
"""
import pytest
from unittest.mock import AsyncMock, patch
from httpx import AsyncClient


@pytest.mark.asyncio
async def test_register_success(client: AsyncClient):
    res = await client.post("/api/auth/register", json={"email": "new_user@test.com", "password": "secret123"})
    assert res.status_code == 201
    data = res.json()
    assert data["email"] == "new_user@test.com"
    assert data["role"] == "user"
    assert "id" in data
    assert "password_hash" not in data  # never expose hash


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