from sqlalchemy import Column, Integer, String, Float, DateTime, JSON, Text
from datetime import datetime, timezone
from .database import Base

def utc_now():
    return datetime.now(timezone.utc)

class BehaviorEventDB(Base):
    __tablename__ = "behavior_events"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    event_id = Column(String, nullable=True, index=True)
    student_id = Column(String, nullable=False, index=True)
    timestamp = Column(String, nullable=False, index=True)  # ISO 8601 string format
    event_type = Column(String, nullable=False, index=True)
    confidence = Column(Float, nullable=False)
    risk_score = Column(Float, nullable=False)
    frame_number = Column(Integer, nullable=True)
    bounding_box = Column(JSON, nullable=True)
    metadata_json = Column(JSON, nullable=True)
    created_at = Column(DateTime, default=utc_now, nullable=False)
