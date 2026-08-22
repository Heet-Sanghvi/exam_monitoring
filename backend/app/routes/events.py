import os
import uuid
from typing import List, Optional, Dict, Any
from fastapi import APIRouter, Depends, HTTPException, Query, Form, File, UploadFile, status
from pydantic import BaseModel, Field, ConfigDict, ValidationError, field_validator
from sqlalchemy.orm import Session
from datetime import datetime

from app.db.database import get_db
from app.db.models import BehaviorEventDB
from app.websocket.manager import manager

router = APIRouter()

MAX_FILE_SIZE_BYTES = 10 * 1024 * 1024  # 10 MB
ALLOWED_CONTENT_TYPES = {"image/jpeg", "image/png", "image/jpg"}
ALLOWED_REVIEW_STATUSES = {"unreviewed", "correct", "incorrect"}

# Ensure static/snapshots directory exists relative to current file
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
SNAPSHOTS_DIR = os.path.join(BASE_DIR, "static", "snapshots")
os.makedirs(SNAPSHOTS_DIR, exist_ok=True)

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
    review_status: str = Field(default="unreviewed", description="Teacher review status: 'unreviewed', 'correct', or 'incorrect'")
    metadata: Optional[Dict[str, Any]] = Field(default_factory=dict)

class BehaviorEventResponse(BehaviorEventCreate):
    id: int
    created_at: datetime

    model_config = ConfigDict(from_attributes=True)

class BehaviorEventBatchCreate(BaseModel):
    events: List[BehaviorEventCreate] = Field(..., min_length=1, description="Non-empty list of behavior events")

class BehaviorEventReviewUpdate(BaseModel):
    review_status: str = Field(..., description="Teacher review status: 'unreviewed', 'correct', or 'incorrect'")

    @field_validator("review_status")
    @classmethod
    def validate_review_status(cls, v: str) -> str:
        if v not in ALLOWED_REVIEW_STATUSES:
            raise ValueError("Invalid review_status. Allowed values: 'unreviewed', 'correct', 'incorrect'")
        return v

@router.post("/behavior-events", response_model=BehaviorEventResponse, status_code=201)
async def create_behavior_event(
    event: str = Form(..., description="JSON string containing behavior event payload matching schema"),
    snapshot: UploadFile = File(..., description="Snapshot image file (JPEG or PNG)"),
    db: Session = Depends(get_db)
):
    # 1. Validate JSON payload against BehaviorEventCreate schema
    try:
        event_obj = BehaviorEventCreate.model_validate_json(event)
    except ValidationError as ve:
        raise HTTPException(
            status_code=422,
            detail=ve.errors()
        )
    except Exception as e:
        raise HTTPException(
            status_code=422,
            detail=f"Invalid JSON string format in 'event' field: {str(e)}"
        )

    # 2. Validate snapshot image file content type
    content_type = (snapshot.content_type or "").lower()
    if content_type not in ALLOWED_CONTENT_TYPES:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid file type '{snapshot.content_type}'. Only JPEG and PNG images are allowed."
        )

    # 3. Read image bytes and validate max file size
    contents = await snapshot.read()
    if len(contents) > MAX_FILE_SIZE_BYTES:
        raise HTTPException(
            status_code=400,
            detail=f"File size ({len(contents)} bytes) exceeds the maximum allowed limit of {MAX_FILE_SIZE_BYTES} bytes (10MB)."
        )

    # 4. Save image to disk with unique filename
    ext = ".png" if "png" in content_type else ".jpg"
    filename = f"evt_{uuid.uuid4().hex[:12]}{ext}"
    file_save_path = os.path.join(SNAPSHOTS_DIR, filename)

    try:
        with open(file_save_path, "wb") as f:
            f.write(contents)
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to save snapshot image to disk: {str(e)}"
        )

    relative_snapshot_url = f"/static/snapshots/{filename}"

    # 5. Insert event into Database
    db_event = BehaviorEventDB(
        event_id=event_obj.event_id,
        test_id=event_obj.test_id,
        timestamp=event_obj.timestamp,
        event_type=event_obj.event_type,
        confidence=event_obj.confidence,
        risk_score=event_obj.risk_score,
        risk_category=event_obj.risk_category,
        frame_number=event_obj.frame_number,
        bounding_box=event_obj.bounding_box,
        snapshot_path=relative_snapshot_url,
        review_status=event_obj.review_status or "unreviewed",
        metadata_json=event_obj.metadata,
    )
    db.add(db_event)
    db.commit()
    db.refresh(db_event)

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
        "review_status": db_event.review_status,
        "metadata": db_event.metadata_json,
        "created_at": db_event.created_at.isoformat() if db_event.created_at else None
    }

    # 6. WebSocket Broadcast rule: ONLY broadcast when risk_category == "high"
    if db_event.risk_category == "high":
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
        review_status=db_event.review_status,
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
            review_status=event.review_status or "unreviewed",
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
            "review_status": db_event.review_status,
            "metadata": db_event.metadata_json,
            "created_at": db_event.created_at.isoformat() if db_event.created_at else None
        }

        # WebSocket Broadcast rule: ONLY broadcast when risk_category == "high"
        if db_event.risk_category == "high":
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
            review_status=db_event.review_status,
            metadata=db_event.metadata_json or {},
            created_at=db_event.created_at
        ))

    return responses

@router.patch("/behavior-events/{event_id}", response_model=BehaviorEventResponse)
def update_event_review_status(
    event_id: str,
    review_in: BehaviorEventReviewUpdate,
    db: Session = Depends(get_db)
):
    db_event = None
    if event_id.isdigit():
        db_event = db.query(BehaviorEventDB).filter(BehaviorEventDB.id == int(event_id)).first()
    if not db_event:
        db_event = db.query(BehaviorEventDB).filter(BehaviorEventDB.event_id == event_id).first()

    if not db_event:
        raise HTTPException(status_code=404, detail=f"Behavior event '{event_id}' not found")

    db_event.review_status = review_in.review_status
    db.commit()
    db.refresh(db_event)

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
        review_status=db_event.review_status,
        metadata=db_event.metadata_json or {},
        created_at=db_event.created_at
    )

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
            review_status=e.review_status,
            metadata=e.metadata_json or {},
            created_at=e.created_at
        )
        for e in events
    ]
