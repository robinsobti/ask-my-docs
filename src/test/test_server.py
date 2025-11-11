import importlib

from fastapi.testclient import TestClient


def _reload_server(monkeypatch, **env_overrides):
    for key, value in env_overrides.items():
        monkeypatch.setenv(key, value)

    import src.server as server  # noqa: WPS433  # local import for test reloads

    return importlib.reload(server)


def test_health_endpoint_reports_build_info(monkeypatch):
    server = _reload_server(
        monkeypatch,
        APP_BUILD_VERSION="1.2.3",
        APP_BUILD_SHA="abc123",
        APP_BUILD_TIME="2024-07-04T12:00:00Z",
    )
    client = TestClient(server.app)

    response = client.get("/health")

    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "ok"
    assert payload["build"] == {
        "version": "1.2.3",
        "sha": "abc123",
        "time": "2024-07-04T12:00:00Z",
    }


def test_query_endpoint_returns_hits(monkeypatch):
    server = _reload_server(monkeypatch)
    monkeypatch.setattr(
        server,
        "retrieve",
        lambda *args, **kwargs: [
            {"id": "doc::1", "title": "Doc", "text": "Answer", "source": "unit-test", "score": 0.9}
        ],
    )

    client = TestClient(server.app)
    response = client.post("/query", json={"query": "hello", "mode": "bm25"})

    assert response.status_code == 200
    payload = response.json()
    assert payload["count"] == 1
    assert payload["results"][0]["id"] == "doc::1"
    assert payload["results"][0]["score"] == 0.9


def test_query_endpoint_rejects_empty_query(monkeypatch):
    server = _reload_server(monkeypatch)
    client = TestClient(server.app)

    response = client.post("/query", json={"query": "    ", "mode": "bm25"})

    assert response.status_code == 400
    assert response.json()["detail"] == "Query must not be empty."
