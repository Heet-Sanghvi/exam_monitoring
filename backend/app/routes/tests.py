import uuid
from typing import List, Optional
from fastapi import APIRouter, Depends, HTTPException, Query, status
from pydantic import BaseModel, Field, ConfigDict
from sqlalchemy.orm import Session
from datetime import datetime, timezone

from app.db.database import get_db
from app.db.models import TestSessionDB, BehaviorEventDB, utc_now
from app.routes.events import BehaviorEventResponse

router = APIRouter()

class TestSessionStart(BaseModel):
    test_id: Optional[str] = Field(None, description="Optional custom test session identifier")
    title: Optional[str] = Field(None, description="Optional human-readable title for test session")

class TestSessionResponse(BaseModel):
    test_id: str
    title: Optional[str] = None
    status: str
    start_time: datetime
    end_time: Optional[datetime] = None

    model_config = ConfigDict(from_attributes=True)

@router.post("/tests/start", response_model=TestSessionResponse, status_code=201)
def start_test_session(session_in: Optional[TestSessionStart] = None, db: Session = Depends(get_db)):
    title = session_in.title if session_in else None
    custom_id = session_in.test_id if session_in else None
    
    test_id = custom_id or f"test_{uuid.uuid4().hex[:8]}"

    # Check if custom test_id already exists
    if custom_id:
        existing = db.query(TestSessionDB).filter(TestSessionDB.test_id == custom_id).first()
        if existing:
            raise HTTPException(status_code=400, detail=f"Test session '{custom_id}' already exists")

    db_session = TestSessionDB(
        test_id=test_id,
        title=title,
        status="active",
        start_time=utc_now()
    )
    db.add(db_session)
    db.commit()
    db.refresh(db_session)

    return db_session

@router.get("/tests/active", response_model=TestSessionResponse)
def get_active_test_session(db: Session = Depends(get_db)):
    active_session = (
        db.query(TestSessionDB)
        .filter(TestSessionDB.status == "active")
        .order_by(TestSessionDB.start_time.desc())
        .first()
    )

    if not active_session:
        raise HTTPException(status_code=404, detail="No active test session found")

    return active_session

@router.post("/tests/{test_id}/end", response_model=TestSessionResponse)
def end_test_session(test_id: str, db: Session = Depends(get_db)):
    session = db.query(TestSessionDB).filter(TestSessionDB.test_id == test_id).first()

    if not session:
        raise HTTPException(status_code=404, detail=f"Test session '{test_id}' not found")

    session.status = "completed"
    session.end_time = utc_now()
    db.commit()
    db.refresh(session)

    return session

@router.get("/tests/{test_id}/events", response_model=List[BehaviorEventResponse])
def get_test_session_events(
    test_id: str,
    risk_category: Optional[str] = Query(None, description="Filter by risk category: 'low' or 'high'"),
    db: Session = Depends(get_db)
):
    session = db.query(TestSessionDB).filter(TestSessionDB.test_id == test_id).first()

    if not session:
        raise HTTPException(status_code=404, detail=f"Test session '{test_id}' not found")

    query = db.query(BehaviorEventDB).filter(BehaviorEventDB.test_id == test_id)
    if risk_category:
        query = query.filter(BehaviorEventDB.risk_category == risk_category)

    events = query.order_by(BehaviorEventDB.id.desc()).all()

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

@router.get("/tests", response_model=List[TestSessionResponse])
def list_test_sessions(db: Session = Depends(get_db)):
    return db.query(TestSessionDB).order_by(TestSessionDB.start_time.desc()).all()
