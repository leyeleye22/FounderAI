from fastapi.testclient import TestClient

from app.main import app


client = TestClient(app)


def test_expected_routes_are_registered() -> None:
    paths = {route.path for route in app.routes}

    expected_paths = {
        "/",
        "/health",
        "/agents/catalog",
        "/agents/problem/challenge",
        "/agents/interview/analyze",
        "/agents/sprint/plan",
        "/agents/chat",
    }

    missing = expected_paths - paths
    assert not missing, f"Missing API routes: {sorted(missing)}"


def test_health_endpoint_contract() -> None:
    response = client.get("/health")

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert "environment" in body
    assert "model_dir" in body


def test_chat_endpoint_contract() -> None:
    payload = {
        "project": {"workspace_id": "ws_123", "project_id": "proj_123"},
        "module": {
            "module_key": "problem-statement",
            "label": "Problem",
            "filled_fields": [],
            "empty_fields": ["problemStatement", "who", "cost"],
            "raw_content": None,
        },
        "message": "Aide-moi à formuler mon problème.",
        "locale": "fr",
        "conversation_history": [],
    }

    response = client.post("/agents/chat", json=payload)

    assert response.status_code == 200
    body = response.json()
    assert isinstance(body["reply"], str)
    assert isinstance(body["actions"], list)
    assert isinstance(body["supporting_context"], list)
    assert body["module_key"] == "problem-statement"
