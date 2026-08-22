# System Architecture — AI Exam Monitoring & Proctoring System

## Overview

The AI Exam Monitoring system is divided into two distinct components interacting via a shared contract:

1. **Perception Pipeline** (`perception/`): Handled by Person 1 (Mac). Analyzes video streams, detects persons, tracks poses, evaluates sustained rolling-window behavior, and generates structured events.
2. **Backend & Dashboard** (`backend/`): Handled by Person 2 (Windows). Validates incoming events, persists them in SQLite, broadcasts real-time alerts via WebSocket, and serves a judge-friendly Streamlit dashboard.

---

## Data Flow Diagram

```text
Camera / Video Stream
         │
         ▼
Perception Pipeline
 ├── Person Detection & Tracking (ByteTrack/DeepSORT)
 ├── Pose & Keypoint Extraction (YOLO / Pose estimation)
 ├── Temporal Behavior Rules & Rolling-Window Risk Analysis
 └── Object Detection (Phone, Chit detection)
         │
         ▼ (HTTP POST /behavior-events)
Shared Data Schema (schema_behavior_event.json)
         │
         ▼
FastAPI Backend (app/main.py)
 ├── Pydantic Event Validation
 ├── SQLite Storage (exam_monitoring.db)
 └── WebSocket Connection Manager (app/websocket/manager.py)
         │
         ▼
Streamlit Monitoring Dashboard (dashboard/dashboard_app.py)
 ├── 🚨 Live Alerts Feed
 ├── 📊 Student Risk Scores View
 └── 📋 Event Log & Audit Export
```

---

## Shared Data Contract

All events posted to the backend MUST conform to `shared/schema_behavior_event.json`:

```json
{
  "event_id": "evt_12345",
  "student_id": "student_01",
  "timestamp": "2026-08-22T10:30:00Z",
  "event_type": "looking_away",
  "confidence": 0.92,
  "risk_score": 35.0,
  "frame_number": 1450,
  "bounding_box": [100, 150, 300, 450],
  "metadata": { "camera_id": "cam_01" }
}
```
