from fastapi import FastAPI
import pytest
from fastapi.testclient import TestClient
from rtrec.serving.app import create_app

# Mock secret token for tests
SECRET_TOKEN = "fake_secret_token"

@pytest.fixture
def client():
    """Fixture to provide a TestClient with a fresh app instance."""
    app = create_app()
    return TestClient(app)

def test_read_root(client):
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"message": "Recommender System API is running"}

def test_fit(client):
    interactions = [
        {"user": "user1", "item": "item1", "timestamp": 1672531200.0, "rating": 5.0},
        {"user": "user1", "item": "item2", "timestamp": 1672617600.0, "rating": 3.0},
        {"user": "user2", "item": "item1", "timestamp": 1672704000.0, "rating": 4.0},
    ]

    response = client.post(
        "/fit",
        json=interactions,
        headers={"X-Token": SECRET_TOKEN}
    )
    assert response.status_code == 200
    assert response.json() == {"message": "Training successful"}

def test_fit_invalid_token(client):
    interactions = [
        {"user": "user1", "item": "item1", "timestamp": 1672531200.0, "rating": 5.0},
    ]

    response = client.post(
        "/fit",
        json=interactions,
        headers={"X-Token": "wrong_token"}
    )
    assert response.status_code == 400
    assert response.json() == {"detail": "Invalid X-Token header"}

def test_recommend(client):
    # Train the model first
    interactions = [
        {"user": "user1", "item": "item1", "timestamp": 1672531200.0, "rating": 5.0},
        {"user": "user1", "item": "item2", "timestamp": 1672617600.0, "rating": 3.0},
        {"user": "user2", "item": "item1", "timestamp": 1672704000.0, "rating": 4.0},
        {"user": "user2", "item": "item3", "timestamp": 1672704000.0, "rating": 4.0},
        {"user": "user2", "item": "item4", "timestamp": 1672704000.0, "rating": 3.0},
    ]
    client.post("/fit", json=interactions, headers={"X-Token": SECRET_TOKEN})

    # Request recommendations
    request_payload = {
        "user": "user1",
        "top_k": 5,
        "filter_interacted": True
    }
    response = client.post(
        "/recommend",
        json=request_payload,
        headers={"X-Token": SECRET_TOKEN}
    )
    assert response.status_code == 200

    response_json = response.json()
    assert response_json["user"] == "user1"
    assert "recommendations" in response_json
    assert isinstance(response_json["recommendations"], list)
    assert len(response_json["recommendations"]) == 2
    # item3 and item4 are recommended
    assert all(item in ["item3", "item4"] for item in response_json["recommendations"])

def test_recommend_integers(client):
    # Train the model first
    interactions = [
        {"user": 1, "item": 1, "timestamp": 1672531200.0, "rating": 5.0},
        {"user": 1, "item": 2, "timestamp": 1672617600.0, "rating": 3.0},
        {"user": 2, "item": 1, "timestamp": 1672704000.0, "rating": 4.0},
        {"user": 2, "item": 3, "timestamp": 1672704000.0, "rating": 4.0},
        {"user": 2, "item": 4, "timestamp": 1672704000.0, "rating": 3.0},
    ]
    client.post("/fit", json=interactions, headers={"X-Token": SECRET_TOKEN})

    # Request recommendations
    request_payload = {
        "user": 1,
        "top_k": 5,
        "filter_interacted": True
    }
    response = client.post(
        "/recommend",
        json=request_payload,
        headers={"X-Token": SECRET_TOKEN}
    )
    assert response.status_code == 200

    response_json = response.json()
    assert response_json["user"] == 1
    assert "recommendations" in response_json
    assert isinstance(response_json["recommendations"], list)
    assert len(response_json["recommendations"]) == 2
    assert all(item in [3, 4] for item in response_json["recommendations"])

def test_recommend_invalid_token(client):
    request_payload = {
        "user": "user1",
        "top_k": 5,
        "filter_interacted": True
    }
    response = client.post(
        "/recommend",
        json=request_payload,
        headers={"X-Token": "wrong_token"}
    )
    assert response.status_code == 400
    assert response.json() == {"detail": "Invalid X-Token header"}

if __name__ == "__main__":
    pytest.main()
