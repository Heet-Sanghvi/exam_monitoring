"""
Live Local Integration Test: Perception Pipeline Dispatch to HTTP Backend Server.

Spins up a lightweight Python HTTP server on an ephemeral port matching Heet's exact endpoint
contract (GET /tests/active and POST /behavior-events multipart upload) and executes live
end-to-end dispatch from PipelineOrchestrator without external dependencies.
"""

import json
import os
import sys
import tempfile
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Dict, List
from urllib.parse import urlparse

import numpy as np
import pytest

PERCEPTION_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PERCEPTION_DIR)

from src.main import PipelineConfig, PipelineOrchestrator


class MockBackendHandler(BaseHTTPRequestHandler):
    db_events: List[Dict[str, Any]] = []
    active_test_session = {"test_id": "test_live_integration_101", "status": "active"}

    def do_GET(self):
        parsed = urlparse(self.path)
        if parsed.path == "/tests/active":
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps(self.active_test_session).encode("utf-8"))
        elif parsed.path == "/backend-test/received-events":
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps(self.db_events).encode("utf-8"))
        else:
            self.send_response(404)
            self.end_headers()

    def do_POST(self):
        parsed = urlparse(self.path)
        if parsed.path == "/behavior-events":
            content_length = int(self.headers.get("Content-Length", 0))
            body_bytes = self.rfile.read(content_length)

            # Check that multipart body contains both 'event' json and 'snapshot' file
            body_str = body_bytes.decode("utf-8", errors="ignore")
            assert 'name="event"' in body_str, "Multipart payload must contain 'event' form field"
            assert 'name="snapshot"' in body_str, "Multipart payload must contain 'snapshot' file field"

            # Extract event JSON
            import re
            match = re.search(r'name="event"\r\n\r\n(.*?)\r\n--', body_str, re.DOTALL)
            event_json = json.loads(match.group(1)) if match else {}

            record = {
                "id": len(self.db_events) + 1,
                "event_id": event_json.get("event_id"),
                "test_id": event_json.get("test_id"),
                "event_type": event_json.get("event_type"),
                "risk_score": event_json.get("risk_score"),
                "risk_category": event_json.get("risk_category"),
                "bounding_box": event_json.get("bounding_box"),
                "received_bytes": content_length,
            }
            self.db_events.append(record)

            self.send_response(201)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps(record).encode("utf-8"))
        else:
            self.send_response(404)
            self.end_headers()

    def log_message(self, format, *args):
        pass  # Silence HTTP server logs during test


def test_live_backend_multipart_dispatch():
    """
    Start local HTTP server on port 8899, run PipelineOrchestrator,
    and verify real HTTP multipart upload of event metadata + snapshot.
    """
    MockBackendHandler.db_events = []
    server = ThreadingHTTPServer(("127.0.0.1", 8899), MockBackendHandler)
    server_thread = threading.Thread(target=server.serve_forever, daemon=True)
    server_thread.start()
    time.sleep(0.3)

    try:
        with tempfile.TemporaryDirectory() as tmp_dir:
            config = PipelineConfig(
                backend_url="http://127.0.0.1:8899",
                snapshot_base_dir=tmp_dir,
                debounce_frames=1,
            )
            orchestrator = PipelineOrchestrator(config=config)

            # 1. Startup session discovery
            active_id = orchestrator.initialize_backend_session()
            assert active_id == "test_live_integration_101"
            assert orchestrator.event_dispatcher._online is True

            # 2. Process synthetic frame with physical reach violation
            frame = np.zeros((480, 640, 3), dtype=np.uint8) + 40
            mock_pose_output = {
                "frame_number": 5,
                "timestamp": "2026-08-22T14:30:00Z",
                "tracks": [
                    {
                        "track_id": 1,
                        "student_id": "student_001",
                        "bbox": [100.0, 100.0, 250.0, 350.0],
                        "confidence": 0.90,
                        "keypoints": [
                            {"name": "nose", "x": 175.0, "y": 120.0, "confidence": 0.90},
                            {"name": "left_shoulder", "x": 135.0, "y": 180.0, "confidence": 0.90},
                            {"name": "right_shoulder", "x": 215.0, "y": 180.0, "confidence": 0.90},
                            {"name": "left_hip", "x": 140.0, "y": 260.0, "confidence": 0.85},
                            {"name": "right_hip", "x": 210.0, "y": 260.0, "confidence": 0.85},
                            {"name": "left_wrist", "x": 140.0, "y": 340.0, "confidence": 0.85},  # Below hips
                            {"name": "right_wrist", "x": 210.0, "y": 220.0, "confidence": 0.85},
                        ],
                    }
                ],
            }

            from unittest.mock import patch
            with patch.object(orchestrator.pose_estimator, "process_frame", return_value=mock_pose_output):
                result = orchestrator.process_frame(frame, frame_number=5)

            assert result["flagged_events_count"] == 1
            assert result["dispatched_count"] == 1
            assert orchestrator.total_events_dispatched == 1

            # 3. Query backend to verify multipart receipt
            import requests
            resp = requests.get("http://127.0.0.1:8899/backend-test/received-events")
            assert resp.status_code == 200
            records = resp.json()
            assert len(records) == 1
            rec = records[0]
            assert rec["test_id"] == "test_live_integration_101"
            assert rec["event_type"] == "hand_out_of_bounds"
            assert rec["received_bytes"] > 0
            assert len(rec["bounding_box"]) == 4

    finally:
        server.shutdown()
        server.server_close()
