# Setup & Execution Instructions

## Prerequisites
- Python 3.10+
- `pip`

---

## 1. Backend & Dashboard Setup (Person 2 — Windows)

```bash
# Clone & navigate to project root
cd exam-monitoring-hackathon

# Install backend dependencies
pip install -r backend/requirements.txt

# Run backend API
uvicorn app.main:app --reload --host 127.0.0.1 --port 8000 --app-dir backend
```

In a new terminal:
```bash
# Launch Streamlit dashboard
streamlit run backend/dashboard/dashboard_app.py
```

In a third terminal (for testing backend independently):
```bash
# Run dummy event generator
python backend/dummy_data/dummy_event_generator.py --interval 2.0
```

---

## 2. Shared Schema Contract

Check `shared/README.md` for details on event payload validation.
