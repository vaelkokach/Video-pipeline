from fastapi.testclient import TestClient

from src.api.app import create_app
from src.api.state import RuntimeState


def test_api_health_and_publish():
    app = create_app(state=RuntimeState())
    client = TestClient(app)

    health = client.get("/health")
    assert health.status_code == 200
    assert health.json()["ok"] is True

    payload = {"frame_index": 10, "students": [{"student_id": 1}], "events": []}
    publish = client.post("/realtime/publish", json=payload)
    assert publish.status_code == 200

    latest = client.get("/realtime/latest")
    assert latest.status_code == 200
    assert latest.json()["frame_index"] == 10
