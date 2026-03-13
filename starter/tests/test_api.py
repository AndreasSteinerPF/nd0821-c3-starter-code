from __future__ import annotations

import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from main import app, load_artifacts

client = TestClient(app)
SAMPLE_REQUESTS_PATH = (
    Path(__file__).resolve().parents[1] / "model" / "sample_requests.json"
)


@pytest.fixture(scope="module")
def sample_requests() -> dict[str, dict[str, int | str]]:
    load_artifacts.cache_clear()
    load_artifacts()
    return json.loads(SAMPLE_REQUESTS_PATH.read_text(encoding="utf-8"))


def test_get_root_returns_welcome_message() -> None:
    response = client.get("/")

    assert response.status_code == 200
    assert response.json() == {
        "message": "Welcome to the census income prediction API."
    }


def test_post_predict_returns_less_or_equal_50k(sample_requests) -> None:
    response = client.post("/predict", json=sample_requests["<=50K"])

    assert response.status_code == 200
    assert response.json() == {"prediction": "<=50K"}


def test_post_predict_returns_greater_than_50k(sample_requests) -> None:
    response = client.post("/predict", json=sample_requests[">50K"])

    assert response.status_code == 200
    assert response.json() == {"prediction": ">50K"}
