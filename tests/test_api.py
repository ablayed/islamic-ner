"""API smoke tests for Phase 5 endpoints."""

from __future__ import annotations

import importlib

import pytest
from fastapi.testclient import TestClient

import api.main as api_main


@pytest.fixture()
def client(monkeypatch):
    monkeypatch.setenv("ISLAMIC_NER_SKIP_MODEL_LOAD", "1")
    monkeypatch.setenv("ISLAMIC_NER_SKIP_NEO4J", "1")
    importlib.reload(api_main)
    with TestClient(api_main.app) as test_client:
        yield test_client


def test_health_endpoint(client: TestClient) -> None:
    response = client.get("/health")
    assert response.status_code == 200
    payload = response.json()

    assert payload["status"] == "healthy"
    assert "model_loaded" in payload
    assert "neo4j_connected" in payload


def test_ner_endpoint_with_sample_hadith(client: TestClient) -> None:
    sample_text = "قال الإمام البخاري في صحيح البخاري إن الربا من الكبائر."
    response = client.post(
        "/ner",
        json={"text": sample_text, "return_tokens": True},
    )

    assert response.status_code == 200
    payload = response.json()

    assert payload["text"] == sample_text
    assert isinstance(payload["normalized_text"], str)
    assert isinstance(payload["entities"], list)
    assert isinstance(payload["tokens"], list)
    assert payload["tokens"], "Expected token-level output when return_tokens=true."

    # With model skipped in tests, gazetteer fallback should still find known entities.
    assert any(entity["type"] in {"SCHOLAR", "BOOK", "CONCEPT"} for entity in payload["entities"])
