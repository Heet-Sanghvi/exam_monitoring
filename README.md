# AI Exam Monitoring & Proctoring System 🛡️

An AI-powered automated exam proctoring system that monitors students during examinations using computer vision and temporal behavior analysis.

## Key Features
- **Temporal Anomaly & Risk Analysis**: Avoids false positives from single frames by tracking rolling-window suspicious behavior.
- **Multi-Behavior Detection**: Supports `looking_away`, `head_turn`, `body_rotation`, `suspicious_hand_movement`, `phone_detected`, `chit_detected`, `multiple_persons_detected`, and `sustained_absence`.
- **FastAPI & SQLite Backend**: Real-time event ingestion API with Pydantic contract validation.
- **WebSocket Live Stream**: Instant event broadcasting to proctoring monitors.
- **Judge-Friendly Streamlit Dashboard**: Includes real-time alert feed, student risk score aggregation, and filterable audit event log.

---

## Workspace Structure

```text
exam-monitoring-hackathon/
├── shared/             # Data schemas (JSON schema contracts)
├── backend/            # FastAPI backend, SQLite DB, WebSocket, Streamlit dashboard
├── perception/         # Person detection, tracking, pose estimation, risk scoring (Mac)
├── docs/               # System architecture & setup documentation
└── demo_assets/        # Screenshots & presentation assets
```

For setup and development instructions, see [docs/setup_instructions.md](docs/setup_instructions.md).
