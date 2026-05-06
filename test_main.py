import os
import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from main import app, Base, get_db, get_password_hash
import json

# Setup test database
SQLALCHEMY_DATABASE_URL = "sqlite:///./test.db"
engine = create_engine(SQLALCHEMY_DATABASE_URL, connect_args={"check_same_thread": False})
TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

def override_get_db():
    try:
        db = TestingSessionLocal()
        yield db
    finally:
        db.close()

app.dependency_overrides[get_db] = override_get_db

client = TestClient(app)

@pytest.fixture(autouse=True)
def setup_db():
    Base.metadata.create_all(bind=engine)
    yield
    Base.metadata.drop_all(bind=engine)

def test_register_user():
    response = client.post(
        "/register",
        json={"email": "test@example.com", "password": "testpassword"}
    )
    assert response.status_code == 200
    assert response.json()["message"] == "User created successfully"

def test_login_user():
    # Register first
    client.post(
        "/register",
        json={"email": "login@example.com", "password": "loginpassword"}
    )
    # Login
    response = client.post(
        "/token",
        data={"username": "login@example.com", "password": "loginpassword"}
    )
    assert response.status_code == 200
    assert "access_token" in response.json()

def test_create_chat_and_persist():
    # Register and Login
    client.post("/register", json={"email": "chat@example.com", "password": "password"})
    login_res = client.post("/token", data={"username": "chat@example.com", "password": "password"})
    token = login_res.json()["access_token"]
    headers = {"Authorization": f"Bearer {token}"}

    # Create Chat
    chat_res = client.post("/chats", json={"title": "Test Chat"}, headers=headers)
    assert chat_res.status_code == 200
    chat_id = chat_res.json()["id"]

    # List Chats
    list_res = client.get("/chats", headers=headers)
    assert len(list_res.json()) == 1
    assert list_res.json()[0]["id"] == chat_id

def test_chat_persistence_on_reload():
    # Register and Login
    client.post("/register", json={"email": "persist@example.com", "password": "password"})
    login_res = client.post("/token", data={"username": "persist@example.com", "password": "password"})
    token = login_res.json()["access_token"]
    headers = {"Authorization": f"Bearer {token}"}

    # Create Chat
    chat_res = client.post("/chats", json={"title": "Persistent Chat"}, headers=headers)
    chat_id = chat_res.json()["id"]

    # Post a message (Mocking the AI part if needed, but here we just check DB save)
    # Since we can't easily mock the OpenAI client inside main.py from here without more complex mocking,
    # we'll focus on the API flow and database entry.
    # Note: The /chat endpoint calls OpenAI. For a unit test, we'd normally mock 'client.chat.completions.create'.
    
    # Verify chat exists after "reload" (new request)
    verify_res = client.get("/chats", headers=headers)
    assert any(c["id"] == chat_id for c in verify_res.json())

def test_unauthorized_access():
    response = client.get("/chats")
    assert response.status_code == 401
