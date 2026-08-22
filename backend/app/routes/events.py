from typing import List, Optional, Dict, Any
from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field, ConfigDict
from sqlalchemy.orm import Session
from datetime import datetime

from app.db.database import get_db
from app.db.models import BehaviorEventDB
from app.websocket.manager import manager

router = APIRouter()

class BehaviorEventCreate(BaseModel):
    event_id: Optional[str] = None
    student_id: str = Field(..., description="Unique student or track identifier")
    timestamp: str = Field(..., description="ISO 8601 string timestamp")
    event_type: str = Field(..., description="Type of behavior event")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence score 0.0 to 1.0")
    risk_score: float = Field(..., ge=0.0, le=100.0, description="Risk score 0.0 to 100.0")
    frame_number: Optional[int] = Field(None, ge=0)
    bounding_box: Optional[List[float]] = None
    metadata: Optional[Dict[str, Any]] = Field(default_factory=dict)

class BehaviorEventResponse(BehaviorEventCreate):
    id: int
    created_at: datetime

    model_config = ConfigDict(from_attributes=True)

@router.post("/behavior-events", response_model=BehaviorEventResponse, status_code=201)
async def create_behavior_event(event: BehaviorEventCreate, db: Session = Depends(get_db)):
    db_event = BehaviorEventDB(
        event_id=event.event_id,
        student_id=event.student_id,
        timestamp=event.timestamp,
        event_type=event.event_type,
        confidence=event.confidence,
        risk_score=event.risk_score,
        frame_number=event.frame_number,
        bounding_box=event.bounding_box,
        metadata_json=event.metadata,
    )
    db.add(db_event)
    db.commit()
    db.refresh(db_event)

    # Broadcast event payload via WebSocket
    broadcast_payload = {
        "id": db_event.id,
        "event_id": db_event.event_id,
        "student_id": db_event.student_id,
        "timestamp": db_event.timestamp,
        "event_type": db_event.event_type,
        "confidence": db_event.confidence,
        "risk_score": db_event.risk_score,
        "frame_number": db_event.frame_number,
        "bounding_box": db_event.bounding_box,
        "metadata": db_event.metadata_json,
        "created_at": db_event.created_at.isoformat() if db_event.created_at else None
    }
    await manager.broadcast(broadcast_payload)

    return BehaviorEventResponse(
        id=db_event.id,
        event_id=db_event.event_id,
        student_id=db_event.student_id,
        timestamp=db_event.timestamp,
        event_type=db_event.event_type,
        confidence=db_event.confidence,
        risk_score=db_event.risk_score,
        frame_number=db_event.frame_number,
        bounding_box=db_event.bounding_box,
        metadata=db_event.metadata_json or {},
        created_at=db_event.created_at
    )

@router.get("/behavior-events", response_model=List[BehaviorEventResponse])
def get_behavior_events(
    student_id: Optional[str] = None,
    event_type: Optional[str] = None,
    limit: int = Query(100, ge=1, le=1000),
    offset: int = Query(0, ge=0),
    db: Session = Depends(get_db)
):
    query = db.query(BehaviorEventDB)
    if student_id:
        query = query.filter(BehaviorEventDB.student_id == student_id)
    if event_type:
        query = query.filter(BehaviorEventDB.event_type == event_type)
    
    events = query.order_by(BehaviorEventDB.id.desc()).offset(offset).limit(limit).all()
    
    return [
        BehaviorEventResponse(
            id=e.id,
            event_id=e.event_id,
            student_id=e.student_id,
            timestamp=e.timestamp,
            event_type=e.event_type,
            confidence=e.confidence,
            risk_score=e.risk_score,
            frame_number=e.frame_number,
            bounding_box=e.bounding_box,
            metadata=e.metadata_json or {},
            created_at=e.created_at
        )
        for e in events
    ]

@router.get("/students/risk-scores")
def get_student_risk_scores(db: Session = Depends(get_db)):
    events = db.query(BehaviorEventDB).all()
    student_summary: Dict[str, Dict[str, Any]] = {}

    for e in events:
        sid = e.student_id
        if sid not in student_summary:
            student_summary[sid] = {
                "student_id": sid,
                "latest_risk_score": e.risk_score,
                "max_risk_score": e.risk_score,
                "event_count": 0,
                "last_seen": e.timestamp,
                "risk_level": "Low"
            }
        
        student_summary[sid]["event_count"] += 1
        student_summary[sid]["latest_risk_score"] = e.risk_score
        if e.risk_score > student_summary[sid]["max_risk_score"]:
            student_summary[sid]["max_risk_score"] = e.risk_score
        student_summary[sid]["last_seen"] = e.timestamp

    for sid, summary in student_summary.items():
        score = summary["max_risk_score"]
        if score >= 70:
            summary["risk_level"] = "High"
        elif score >= 35:
            summary["risk_level"] = "Medium"
        else:
            summary["risk_level"] = "Low"

    return list(student_summary.values())
