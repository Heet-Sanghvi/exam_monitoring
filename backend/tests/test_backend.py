import pytest
import os
import sys
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

def test_create_behavior_event():
    payload = {
        "event_id": "test_evt_001",
        "student_id": "student_test_01",
        "timestamp": "2026-08-22T10:00:00Z",
        "event_type": "looking_away",
        "confidence": 0.95,
        "risk_score": 40.0,
        "frame_number": 100,
        "bounding_box": [10.0, 20.0, 100.0, 200.0],
        "metadata": {"test": True}
    }
    response = client.post("/behavior-events", json=payload)
    assert response.status_code == 201
    data = response.json()
    assert data["student_id"] == "student_test_01"
    assert data["risk_score"] == 40.0
    assert data["id"] is not None

def test_invalid_behavior_event_confidence():
    payload = {
        "student_id": "student_test_02",
        "timestamp": "2026-08-22T10:00:00Z",
        "event_type": "phone_detected",
        "confidence": 1.5,  # Invalid: > 1.0
        "risk_score": 85.0
    }
    response = client.post("/behavior-events", json=payload)
    assert response.status_code == 422  # Validation error

def test_get_behavior_events():
    response = client.get("/behavior-events")
    assert response.status_code == 200
    data = response.json()
    assert isinstance(data, list)
    assert len(data) >= 1

def test_get_student_risk_scores_basic():
    response = client.get("/students/risk-scores")
    assert response.status_code == 200
    data = response.json()
    assert isinstance(data, list)
    assert len(data) >= 1
    student = next(s for s in data if s["student_id"] == "student_test_01")
    assert student["risk_level"] == "Medium"

def test_risk_score_aggregation_complex():
    """
    Tests:
    - Multiple events per student
    - Latest risk score vs max risk score distinction
    - Correct event count
    - Risk level threshold classification (High >= 70, Medium >= 35, Low < 35)
    - Multiple distinct students
    """
    # Student A: Event 1 (Medium risk: 40), Event 2 (High risk: 85), Event 3 (Latest event, low risk: 20)
    events_student_a = [
        {"student_id": "student_A", "timestamp": "2026-08-22T10:01:00Z", "event_type": "looking_away", "confidence": 0.8, "risk_score": 40.0},
        {"student_id": "student_A", "timestamp": "2026-08-22T10:02:00Z", "event_type": "phone_detected", "confidence": 0.9, "risk_score": 85.0},
        {"student_id": "student_A", "timestamp": "2026-08-22T10:03:00Z", "event_type": "head_turn", "confidence": 0.7, "risk_score": 20.0},
    ]

    # Student B: Event 1 (Low risk: 15), Event 2 (Low risk: 25)
    events_student_b = [
        {"student_id": "student_B", "timestamp": "2026-08-22T10:01:00Z", "event_type": "head_turn", "confidence": 0.75, "risk_score": 15.0},
        {"student_id": "student_B", "timestamp": "2026-08-22T10:04:00Z", "event_type": "head_turn", "confidence": 0.80, "risk_score": 25.0},
    ]

    for evt in events_student_a + events_student_b:
        res = client.post("/behavior-events", json=evt)
        assert res.status_code == 201

    response = client.get("/students/risk-scores")
    assert response.status_code == 200
    scores = response.json()
    
    student_a_summary = next(s for s in scores if s["student_id"] == "student_A")
    assert student_a_summary["event_count"] == 3
    assert student_a_summary["max_risk_score"] == 85.0
    assert student_a_summary["latest_risk_score"] == 20.0
    assert student_a_summary["last_seen"] == "2026-08-22T10:03:00Z"
    assert student_a_summary["risk_level"] == "High"

    student_b_summary = next(s for s in scores if s["student_id"] == "student_B")
    assert student_b_summary["event_count"] == 2
    assert student_b_summary["max_risk_score"] == 25.0
    assert student_b_summary["latest_risk_score"] == 25.0
    assert student_b_summary["last_seen"] == "2026-08-22T10:04:00Z"
    assert student_b_summary["risk_level"] == "Low"

def test_create_behavior_events_batch_success():
    payload = {
        "events": [
            {
                "student_id": "student_batch_01",
                "timestamp": "2026-08-22T11:00:00Z",
                "event_type": "looking_away",
                "confidence": 0.85,
                "risk_score": 30.0
            },
            {
                "student_id": "student_batch_01",
                "timestamp": "2026-08-22T11:01:00Z",
                "event_type": "phone_detected",
                "confidence": 0.95,
                "risk_score": 80.0
            },
            {
                "student_id": "student_batch_02",
                "timestamp": "2026-08-22T11:00:30Z",
                "event_type": "head_turn",
                "confidence": 0.75,
                "risk_score": 15.0
            }
        ]
    }
    response = client.post("/behavior-events/batch", json=payload)
    assert response.status_code == 201
    data = response.json()
    assert isinstance(data, list)
    assert len(data) == 3
    student_ids = {event["student_id"] for event in data}
    assert student_ids == {"student_batch_01", "student_batch_02"}

def test_create_behavior_events_batch_atomic_rollback():
    valid_student_id = "student_atomic_01"
    payload = {
        "events": [
            {
                "student_id": valid_student_id,
                "timestamp": "2026-08-22T11:05:00Z",
                "event_type": "looking_away",
                "confidence": 0.80,
                "risk_score": 25.0
            },
            {
                "student_id": "student_atomic_02",
                "timestamp": "2026-08-22T11:05:05Z",
                "event_type": "phone_detected",
                "confidence": 1.5,  # Invalid: > 1.0
                "risk_score": 90.0
            }
        ]
    }
    response = client.post("/behavior-events/batch", json=payload)
    assert response.status_code == 422

    # Verify no partial insertion occurred for valid_student_id
    get_res = client.get(f"/behavior-events?student_id={valid_student_id}")
    assert get_res.status_code == 200
    assert len(get_res.json()) == 0

def test_create_behavior_events_batch_empty():
    payload = {"events": []}
    response = client.post("/behavior-events/batch", json=payload)
    assert response.status_code == 422

def test_create_single_behavior_event_after_batch():
    payload = {
        "event_id": "test_evt_post_batch",
        "student_id": "student_post_batch_01",
        "timestamp": "2026-08-22T11:10:00Z",
        "event_type": "body_rotation",
        "confidence": 0.88,
        "risk_score": 35.0
    }
    response = client.post("/behavior-events", json=payload)
    assert response.status_code == 201
    data = response.json()
    assert data["student_id"] == "student_post_batch_01"
    assert data["risk_score"] == 35.0

