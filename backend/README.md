# Exam Monitoring Backend & Dashboard

This directory contains the FastAPI backend, SQLite persistence layer, WebSocket live broadcaster, and Streamlit monitoring dashboard.

---

## Architecture Overview

```text
Perception Pipeline (or Dummy Generator)
             │
   HTTP POST /behavior-events
             ▼
    FastAPI Backend API
    ├── Pydantic Event Validation
    ├── SQLite Storage (SQLAlchemy ORM)
    └── WebSocket Broadcaster (/ws)
             │
             ▼
    Streamlit Dashboard
    ├── 🚨 Live Alerts Feed
    ├── 📊 Student Risk Scores
    └── 📋 Filterable Event Log
```

---

## 🚀 Quickstart Guide

### 1. Install Dependencies

```bash
pip install -r backend/requirements.txt
```

### 2. Run the FastAPI Backend

From the repository root directory:

```bash
uvicorn app.main:app --reload --host 127.0.0.1 --port 8000 --app-dir backend
```

API Interactive Documentation (Swagger UI): `http://127.0.0.1:8000/docs`

### 3. Launch the Streamlit Monitoring Dashboard

In a separate terminal window:

```bash
streamlit run backend/dashboard/dashboard_app.py
```

Dashboard Web UI: `http://localhost:8501`

### 4. Run the Dummy Event Generator (Testing)

To test the backend and dashboard without needing the perception ML model running:

```bash
python backend/dummy_data/dummy_event_generator.py --interval 2.0
```

---

## 🧪 Running Automated Unit & Integration Tests

```bash
python -m pytest backend/tests/test_backend.py
```
