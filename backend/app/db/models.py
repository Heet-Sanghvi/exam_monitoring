import uuid
from sqlalchemy import Column, Integer, String, Float, DateTime, JSON
from datetime import datetime, timezone
from .database import Base

def utc_now():
    return datetime.now(timezone.utc)

def generate_test_id():
    return f"test_{uuid.uuid4().hex[:8]}"

class TestSessionDB(Base):
    __tablename__ = "exam_test_sessions"

    test_id = Column(String, primary_key=True, index=True, default=generate_test_id)
    title = Column(String, nullable=True)
    status = Column(String, nullable=False, default="active", index=True)  # "active" | "completed"
    start_time = Column(DateTime, default=utc_now, nullable=False)
    end_time = Column(DateTime, nullable=True)

class BehaviorEventDB(Base):
    __tablename__ = "behavior_events"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    event_id = Column(String, nullable=True, index=True)
    test_id = Column(String, nullable=False, index=True)
    timestamp = Column(String, nullable=False, index=True)  # ISO 8601 string format
    event_type = Column(String, nullable=False, index=True)
    confidence = Column(Float, nullable=False)
    risk_score = Column(Float, nullable=False)
    risk_category = Column(String, nullable=False, index=True)  # "low" or "high"
    frame_number = Column(Integer, nullable=False)
    bounding_box = Column(JSON, nullable=False)  # [x, y, w, h] format
    snapshot_path = Column(String, nullable=True)
    metadata_json = Column(JSON, nullable=True)
    created_at = Column(DateTime, default=utc_now, nullable=False)
