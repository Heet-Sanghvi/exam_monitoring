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

def test_get_student_risk_scores():
    response = client.get("/students/risk-scores")
    assert response.status_code == 200
    data = response.json()
    assert isinstance(data, list)
    assert len(data) >= 1
    student = next(s for s in data if s["student_id"] == "student_test_01")
    assert student["risk_level"] == "Medium"
