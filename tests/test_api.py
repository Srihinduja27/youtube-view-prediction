import pytest
from starlette.testclient import TestClient
from api.main import app


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


def test_root_endpoint(client):
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert data["service"] == "YouTube Prediction API"
    assert data["status"] == "online"


def test_health_endpoint(client):
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["model_loaded"] is True
    assert data["preprocessor_loaded"] is True


def test_predict_single_video(client):
    payload = {
        "category_id": 10,
        "comments_disabled": 0,
        "ratings_disabled": 0,
        "video_error_or_removed": 0,
        "days_to_trending": 2.0,
        "publish_year": 2018,
        "publish_month": 5,
        "publish_day": 12,
        "publish_dayofweek": 5,
        "publish_hour": 14,
        "trending_year": 2018,
        "trending_month": 5,
        "trending_day": 14,
        "trending_dayofweek": 0,
        "title_length": 45,
        "title_word_count": 8,
        "description_length": 350,
        "description_word_count": 40,
        "tag_count": 15,
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "predicted_views" in data
    assert data["predicted_views"] >= 0
    assert "predicted_log_views" in data
    assert "request_id" in data


def test_predict_batch(client):
    video = {
        "category_id": 24,
        "comments_disabled": 0,
        "ratings_disabled": 0,
        "video_error_or_removed": 0,
        "days_to_trending": 1.0,
        "publish_year": 2018,
        "publish_month": 6,
        "publish_day": 1,
        "publish_dayofweek": 4,
        "publish_hour": 18,
        "trending_year": 2018,
        "trending_month": 6,
        "trending_day": 2,
        "trending_dayofweek": 5,
        "title_length": 30,
        "title_word_count": 5,
        "description_length": 150,
        "description_word_count": 20,
        "tag_count": 8,
    }
    batch_payload = {"videos": [video, video, video]}
    response = client.post("/predict/batch", json=batch_payload)
    assert response.status_code == 200
    data = response.json()
    assert data["total_records"] == 3
    assert len(data["predictions"]) == 3


def test_predict_invalid_month(client):
    payload = {
        "category_id": 10,
        "publish_month": 15,  # Invalid month > 12
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 422
