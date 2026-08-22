import pytest
import os
import sys
import io
import json
from fastapi.testclient import TestClient

# Add backend app to python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.main import app
from app.db.database import Base, engine

# Ensure fresh DB tables for tests
Base.metadata.drop_all(bind=engine)
Base.metadata.create_all(bind=engine)

client = TestClient(app)

def test_root_endpoint():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json()["status"] == "online"

def test_active_test_session_not_found_initially():
    response = client.get("/tests/active")
    assert response.status_code == 404
    assert "No active test session found" in response.json()["detail"]

def test_start_test_session():
    payload = {"title": "Midterm Exam Room 101"}
    response = client.post("/tests/start", json=payload)
    assert response.status_code == 201
    data = response.json()
    assert data["test_id"].startswith("test_")
    assert data["title"] == "Midterm Exam Room 101"
    assert data["status"] == "active"
    assert data["start_time"] is not None
    assert data["end_time"] is None

def test_get_active_test_session():
    response = client.get("/tests/active")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "active"
    assert data["title"] == "Midterm Exam Room 101"

def test_list_test_sessions():
    response = client.get("/tests")
    assert response.status_code == 200
    data = response.json()
    assert isinstance(data, list)
    assert len(data) >= 1
    assert data[0]["status"] == "active"

def test_end_test_session():
    active_res = client.get("/tests/active")
    assert active_res.status_code == 200
    test_id = active_res.json()["test_id"]

    response = client.post(f"/tests/{test_id}/end")
    assert response.status_code == 200
    data = response.json()
    assert data["test_id"] == test_id
    assert data["status"] == "completed"
    assert data["end_time"] is not None

def test_get_active_test_session_after_ending():
    response = client.get("/tests/active")
    assert response.status_code == 404

def test_end_nonexistent_test_session():
    response = client.post("/tests/nonexistent_test_id/end")
    assert response.status_code == 404

def test_create_behavior_event_multipart_success():
    event_dict = {
        "event_id": "test_evt_mp_001",
        "test_id": "test_session_01",
        "timestamp": "2026-08-22T10:00:00Z",
        "event_type": "looking_away",
        "confidence": 0.95,
        "risk_score": 40.0,
        "risk_category": "high",
        "frame_number": 100,
        "bounding_box": [100.0, 150.0, 200.0, 250.0],
        "metadata": {"test": True}
    }
    fake_img_bytes = b"\xff\xd8\xff\xe0\x00\x10JFIF\x00\x01\x01\x01\x00`\x00`\x00\x00\xff\xdb\x00C\x00"
    response = client.post(
        "/behavior-events",
        data={"event": json.dumps(event_dict)},
        files={"snapshot": ("test_frame.jpg", io.BytesIO(fake_img_bytes), "image/jpeg")}
    )
    assert response.status_code == 201
    data = response.json()
    assert data["test_id"] == "test_session_01"
    assert data["risk_category"] == "high"
    assert data["bounding_box"] == [100.0, 150.0, 200.0, 250.0]
    assert data["snapshot_path"].startswith("/static/snapshots/evt_")
    assert data["id"] is not None

def test_create_behavior_event_multipart_missing_snapshot():
    event_dict = {
        "test_id": "test_session_01",
        "timestamp": "2026-08-22T10:00:00Z",
        "event_type": "looking_away",
        "confidence": 0.80,
        "risk_score": 20.0,
        "risk_category": "low",
        "frame_number": 101,
        "bounding_box": [10.0, 10.0, 50.0, 50.0]
    }
    response = client.post(
        "/behavior-events",
        data={"event": json.dumps(event_dict)}
    )
    assert response.status_code == 422

def test_create_behavior_event_multipart_invalid_file_type():
    event_dict = {
        "test_id": "test_session_01",
        "timestamp": "2026-08-22T10:00:00Z",
        "event_type": "looking_away",
        "confidence": 0.80,
        "risk_score": 20.0,
        "risk_category": "low",
        "frame_number": 102,
        "bounding_box": [10.0, 10.0, 50.0, 50.0]
    }
    fake_txt_bytes = b"Hello world text file"
    response = client.post(
        "/behavior-events",
        data={"event": json.dumps(event_dict)},
        files={"snapshot": ("document.txt", io.BytesIO(fake_txt_bytes), "text/plain")}
    )
    assert response.status_code == 400
    assert "Only JPEG and PNG images are allowed" in response.json()["detail"]

def test_invalid_behavior_event_confidence():
    payload = {
        "test_id": "test_session_01",
        "timestamp": "2026-08-22T10:00:00Z",
        "event_type": "phone_detected",
        "confidence": 1.5,
        "risk_score": 85.0,
        "risk_category": "high",
        "frame_number": 105,
        "bounding_box": [10.0, 20.0, 30.0, 40.0]
    }
    fake_img_bytes = b"fake image"
    response = client.post(
        "/behavior-events",
        data={"event": json.dumps(payload)},
        files={"snapshot": ("test_frame.jpg", io.BytesIO(fake_img_bytes), "image/jpeg")}
    )
    assert response.status_code == 422

def test_get_behavior_events():
    response = client.get("/behavior-events?test_id=test_session_01")
    assert response.status_code == 200
    data = response.json()
    assert isinstance(data, list)
    assert len(data) >= 1
    assert data[0]["test_id"] == "test_session_01"

def test_create_behavior_events_batch_success():
    payload = {
        "events": [
            {
                "test_id": "test_batch_01",
                "timestamp": "2026-08-22T11:00:00Z",
                "event_type": "looking_away",
                "confidence": 0.85,
                "risk_score": 30.0,
                "risk_category": "low",
                "frame_number": 200,
                "bounding_box": [50.0, 50.0, 100.0, 100.0]
            },
            {
                "test_id": "test_batch_01",
                "timestamp": "2026-08-22T11:01:00Z",
                "event_type": "phone_detected",
                "confidence": 0.95,
                "risk_score": 80.0,
                "risk_category": "high",
                "frame_number": 210,
                "bounding_box": [60.0, 60.0, 120.0, 120.0]
            },
            {
                "test_id": "test_batch_02",
                "timestamp": "2026-08-22T11:00:30Z",
                "event_type": "head_turn",
                "confidence": 0.75,
                "risk_score": 15.0,
                "risk_category": "low",
                "frame_number": 150,
                "bounding_box": [70.0, 70.0, 90.0, 90.0]
            }
        ]
    }
    response = client.post("/behavior-events/batch", json=payload)
    assert response.status_code == 201
    data = response.json()
    assert isinstance(data, list)
    assert len(data) == 3
    test_ids = {event["test_id"] for event in data}
    assert test_ids == {"test_batch_01", "test_batch_02"}

def test_create_behavior_events_batch_atomic_rollback():
    target_test_id = "test_atomic_01"
    payload = {
        "events": [
            {
                "test_id": target_test_id,
                "timestamp": "2026-08-22T11:05:00Z",
                "event_type": "looking_away",
                "confidence": 0.80,
                "risk_score": 25.0,
                "risk_category": "low",
                "frame_number": 300,
                "bounding_box": [10.0, 10.0, 50.0, 50.0]
            },
            {
                "test_id": "test_atomic_02",
                "timestamp": "2026-08-22T11:05:05Z",
                "event_type": "phone_detected",
                "confidence": 1.5,
                "risk_score": 90.0,
                "risk_category": "high",
                "frame_number": 305,
                "bounding_box": [20.0, 20.0, 60.0, 60.0]
            }
        ]
    }
    response = client.post("/behavior-events/batch", json=payload)
    assert response.status_code == 422

    get_res = client.get(f"/behavior-events?test_id={target_test_id}")
    assert get_res.status_code == 200
    assert len(get_res.json()) == 0

def test_create_behavior_events_batch_empty():
    payload = {"events": []}
    response = client.post("/behavior-events/batch", json=payload)
    assert response.status_code == 422

def test_websocket_broadcast_risk_category_filtering():
    with client.websocket_connect("/ws") as websocket:
        # High risk event -> triggers WebSocket broadcast
        high_risk_event = {
            "test_id": "test_session_ws",
            "timestamp": "2026-08-22T12:10:00Z",
            "event_type": "phone_detected",
            "confidence": 0.95,
            "risk_score": 90.0,
            "risk_category": "high",
            "frame_number": 600,
            "bounding_box": [100.0, 100.0, 200.0, 200.0]
        }
        res_high = client.post(
            "/behavior-events",
            data={"event": json.dumps(high_risk_event)},
            files={"snapshot": ("high_risk.jpg", io.BytesIO(b"fake_high_image"), "image/jpeg")}
        )
        assert res_high.status_code == 201

        ws_msg = websocket.receive_json()
        assert ws_msg["risk_category"] == "high"
        assert ws_msg["test_id"] == "test_session_ws"

        # Low risk event -> does NOT trigger WebSocket broadcast
        low_risk_event = {
            "test_id": "test_session_ws",
            "timestamp": "2026-08-22T12:11:00Z",
            "event_type": "head_turn",
            "confidence": 0.70,
            "risk_score": 15.0,
            "risk_category": "low",
            "frame_number": 601,
            "bounding_box": [10.0, 10.0, 50.0, 50.0]
        }
        res_low = client.post(
            "/behavior-events",
            data={"event": json.dumps(low_risk_event)},
            files={"snapshot": ("low_risk.jpg", io.BytesIO(b"fake_low_image"), "image/jpeg")}
        )
        assert res_low.status_code == 201

        # Confirm low-risk event is saved in DB
        get_res = client.get(f"/behavior-events?test_id=test_session_ws&risk_category=low")
        assert get_res.status_code == 200
        assert len(get_res.json()) >= 1
