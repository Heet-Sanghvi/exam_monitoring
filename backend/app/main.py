from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from app.db.database import engine, Base
from app.routes.events import router as events_router
from app.routes.tests import router as tests_router
from app.websocket.manager import manager

# Create database tables automatically
Base.metadata.create_all(bind=engine)

app = FastAPI(
    title="Exam Monitoring Proctoring API",
    description="Backend API for receiving perception events, storing in SQLite, and broadcasting to dashboard",
    version="1.0.0"
)

# Enable CORS for local Streamlit & frontend access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(events_router)
app.include_router(tests_router)

@app.get("/")
def root():
    return {
        "status": "online",
        "service": "Exam Monitoring Backend",
        "docs_url": "/docs"
    }

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await manager.connect(websocket)
    try:
        while True:
            # Keep connection alive; clients can also send ping/pong messages
            data = await websocket.receive_text()
    except WebSocketDisconnect:
        manager.disconnect(websocket)
