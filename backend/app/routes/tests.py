import uuid
from typing import List, Optional
from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field, ConfigDict
from sqlalchemy.orm import Session
from datetime import datetime, timezone

from app.db.database import get_db
from app.db.models import TestSessionDB, utc_now

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

@router.get("/tests", response_model=List[TestSessionResponse])
def list_test_sessions(db: Session = Depends(get_db)):
    return db.query(TestSessionDB).order_by(TestSessionDB.start_time.desc()).all()
