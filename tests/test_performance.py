import time
import pytest
from starlette.testclient import TestClient
from api.main import app


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


def test_prediction_latency_benchmark(client):
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

    # Warmup
    client.post("/predict", json=payload)

    latencies = []
    for _ in range(20):
        t0 = time.perf_counter()
        res = client.post("/predict", json=payload)
        latencies.append((time.perf_counter() - t0) * 1000)
        assert res.status_code == 200

    avg_latency = sum(latencies) / len(latencies)
    print(f"\nAverage prediction latency: {avg_latency:.2f} ms")
    assert avg_latency < 100.0, f"Average latency too high: {avg_latency} ms"
