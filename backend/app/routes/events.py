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
    test_id: str = Field(..., description="Unique test or exam session identifier")
    timestamp: str = Field(..., description="ISO 8601 string timestamp")
    event_type: str = Field(..., description="Type of behavior event")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence score 0.0 to 1.0")
    risk_score: float = Field(..., ge=0.0, le=100.0, description="Risk score 0.0 to 100.0")
    risk_category: str = Field(..., description="Risk category: 'low' or 'high'")
    frame_number: int = Field(..., ge=0, description="Video frame number")
    bounding_box: List[float] = Field(..., min_length=4, max_length=4, description="Bounding box [x, y, w, h]")
    snapshot_path: Optional[str] = Field(None, description="Path or URL to saved snapshot image")
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
        test_id=event.test_id,
        timestamp=event.timestamp,
        event_type=event.event_type,
        confidence=event.confidence,
        risk_score=event.risk_score,
        risk_category=event.risk_category,
        frame_number=event.frame_number,
        bounding_box=event.bounding_box,
        snapshot_path=event.snapshot_path,
        metadata_json=event.metadata,
    )
    db.add(db_event)
    db.commit()
    db.refresh(db_event)

    # Broadcast event payload via WebSocket
    broadcast_payload = {
        "id": db_event.id,
        "event_id": db_event.event_id,
        "test_id": db_event.test_id,
        "timestamp": db_event.timestamp,
        "event_type": db_event.event_type,
        "confidence": db_event.confidence,
        "risk_score": db_event.risk_score,
        "risk_category": db_event.risk_category,
        "frame_number": db_event.frame_number,
        "bounding_box": db_event.bounding_box,
        "snapshot_path": db_event.snapshot_path,
        "metadata": db_event.metadata_json,
        "created_at": db_event.created_at.isoformat() if db_event.created_at else None
    }
    await manager.broadcast(broadcast_payload)

    return BehaviorEventResponse(
        id=db_event.id,
        event_id=db_event.event_id,
        test_id=db_event.test_id,
        timestamp=db_event.timestamp,
        event_type=db_event.event_type,
        confidence=db_event.confidence,
        risk_score=db_event.risk_score,
        risk_category=db_event.risk_category,
        frame_number=db_event.frame_number,
        bounding_box=db_event.bounding_box,
        snapshot_path=db_event.snapshot_path,
        metadata=db_event.metadata_json or {},
        created_at=db_event.created_at
    )

@router.post("/behavior-events/batch", response_model=List[BehaviorEventResponse], status_code=201)
async def create_behavior_events_batch(batch: BehaviorEventBatchCreate, db: Session = Depends(get_db)):
    db_events = [
        BehaviorEventDB(
            event_id=event.event_id,
            test_id=event.test_id,
            timestamp=event.timestamp,
            event_type=event.event_type,
            confidence=event.confidence,
            risk_score=event.risk_score,
            risk_category=event.risk_category,
            frame_number=event.frame_number,
            bounding_box=event.bounding_box,
            snapshot_path=event.snapshot_path,
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
            "test_id": db_event.test_id,
            "timestamp": db_event.timestamp,
            "event_type": db_event.event_type,
            "confidence": db_event.confidence,
            "risk_score": db_event.risk_score,
            "risk_category": db_event.risk_category,
            "frame_number": db_event.frame_number,
            "bounding_box": db_event.bounding_box,
            "snapshot_path": db_event.snapshot_path,
            "metadata": db_event.metadata_json,
            "created_at": db_event.created_at.isoformat() if db_event.created_at else None
        }
        await manager.broadcast(broadcast_payload)

        responses.append(BehaviorEventResponse(
            id=db_event.id,
            event_id=db_event.event_id,
            test_id=db_event.test_id,
            timestamp=db_event.timestamp,
            event_type=db_event.event_type,
            confidence=db_event.confidence,
            risk_score=db_event.risk_score,
            risk_category=db_event.risk_category,
            frame_number=db_event.frame_number,
            bounding_box=db_event.bounding_box,
            snapshot_path=db_event.snapshot_path,
            metadata=db_event.metadata_json or {},
            created_at=db_event.created_at
        ))

    return responses

@router.get("/behavior-events", response_model=List[BehaviorEventResponse])
def get_behavior_events(
    test_id: Optional[str] = None,
    risk_category: Optional[str] = None,
    event_type: Optional[str] = None,
    limit: int = Query(100, ge=1, le=1000),
    offset: int = Query(0, ge=0),
    db: Session = Depends(get_db)
):
    query = db.query(BehaviorEventDB)
    if test_id:
        query = query.filter(BehaviorEventDB.test_id == test_id)
    if risk_category:
        query = query.filter(BehaviorEventDB.risk_category == risk_category)
    if event_type:
        query = query.filter(BehaviorEventDB.event_type == event_type)
    
    events = query.order_by(BehaviorEventDB.id.desc()).offset(offset).limit(limit).all()
    
    return [
        BehaviorEventResponse(
            id=e.id,
            event_id=e.event_id,
            test_id=e.test_id,
            timestamp=e.timestamp,
            event_type=e.event_type,
            confidence=e.confidence,
            risk_score=e.risk_score,
            risk_category=e.risk_category,
            frame_number=e.frame_number,
            bounding_box=e.bounding_box,
            snapshot_path=e.snapshot_path,
            metadata=e.metadata_json or {},
            created_at=e.created_at
        )
        for e in events
    ]
