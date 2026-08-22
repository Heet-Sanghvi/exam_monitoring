from typing import List, Optional, Dict, Any
from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field, ConfigDict
from sqlalchemy.orm import Session
from sqlalchemy import func
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

class BehaviorEventBatchCreate(BaseModel):
    events: List[BehaviorEventCreate] = Field(..., min_length=1, description="Non-empty list of behavior events")

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

@router.post("/behavior-events/batch", response_model=List[BehaviorEventResponse], status_code=201)
async def create_behavior_events_batch(batch: BehaviorEventBatchCreate, db: Session = Depends(get_db)):
    # Note: if any event in `batch.events` fails BehaviorEventCreate validation,
    # FastAPI/Pydantic rejects the whole request with a 422 before this handler
    # ever runs, so there is no risk of partial insertion from bad payloads.
    db_events = [
        BehaviorEventDB(
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
        for event in batch.events
    ]

    try:
        db.add_all(db_events)
        db.commit()
    except Exception:
        db.rollback()
        raise HTTPException(status_code=500, detail="Batch insertion failed; transaction rolled back")

    for db_event in db_events:
        db.refresh(db_event)

    responses = []
    for db_event in db_events:
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

        responses.append(BehaviorEventResponse(
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
        ))

    return responses


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
    # Subquery to aggregate max_id, max_risk_score, and event_count per student
    stats_subquery = (
        db.query(
            BehaviorEventDB.student_id.label("student_id"),
            func.max(BehaviorEventDB.id).label("max_id"),
            func.max(BehaviorEventDB.risk_score).label("max_risk_score"),
            func.count(BehaviorEventDB.id).label("event_count")
        )
        .group_by(BehaviorEventDB.student_id)
        .subquery()
    )

    # Join back to get the latest event's risk_score and timestamp for each student
    results = (
        db.query(
            stats_subquery.c.student_id,
            BehaviorEventDB.risk_score.label("latest_risk_score"),
            stats_subquery.c.max_risk_score,
            stats_subquery.c.event_count,
            BehaviorEventDB.timestamp.label("last_seen")
        )
        .join(BehaviorEventDB, BehaviorEventDB.id == stats_subquery.c.max_id)
        .all()
    )

    summaries = []
    for row in results:
        max_score = float(row.max_risk_score)
        if max_score >= 70:
            risk_level = "High"
        elif max_score >= 35:
            risk_level = "Medium"
        else:
            risk_level = "Low"

        summaries.append({
            "student_id": row.student_id,
            "latest_risk_score": float(row.latest_risk_score),
            "max_risk_score": max_score,
            "event_count": int(row.event_count),
            "last_seen": row.last_seen,
            "risk_level": risk_level
        })

    return summaries

