"""
Unit tests for perception/src/main.py (Pipeline Orchestrator).

Covers:
- test_orchestrator_init
- test_initialize_backend_session_online
- test_initialize_backend_session_404_or_offline_fallback
- test_process_frame_empty_image
- test_process_frame_with_movement_violation_and_snapshots
- test_red_and_green_box_snapshot_rendering
- test_process_frame_dispatches_without_student_id_or_seat_zone
- test_pipeline_video_processing_mock
"""

import json
import os
import sys
import tempfile
from unittest.mock import MagicMock, patch

import cv2
import numpy as np
import pytest

# Ensure perception/src is on python path
PERCEPTION_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
REPO_DIR = os.path.abspath(os.path.join(PERCEPTION_DIR, ".."))
sys.path.insert(0, PERCEPTION_DIR)

from src.main import PipelineConfig, PipelineOrchestrator
from src.behavior_rules import MovementFlaggingConfig, MovementFlaggingEngine
from src.object_detect import ObjectDetectionConfig
from src.send_events import EventDispatcherConfig


class TestPipelineOrchestrator:

    @pytest.fixture
    def orchestrator(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            config = PipelineConfig(
                backend_url="http://127.0.0.1:8000",
                default_test_id="test_unit_001",
                snapshot_base_dir=tmp_dir,
                debounce_frames=1,
            )
            yield PipelineOrchestrator(config=config)

    def test_orchestrator_init(self, orchestrator):
        """Verify orchestrator properly initializes all pipeline components."""
        assert orchestrator.pose_estimator is not None
        assert orchestrator.movement_flagger is not None
        assert orchestrator.object_detector is not None
        assert orchestrator.event_dispatcher is not None
        assert orchestrator.device in ("mps", "cuda", "cpu")

    def test_initialize_backend_session_online(self, orchestrator):
        """When backend has active session, orchestrator must adopt test_id."""
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"test_id": "session_live_888", "status": "active"}
        mock_resp.raise_for_status = MagicMock()

        with patch("src.send_events.requests.get", return_value=mock_resp):
            test_id = orchestrator.initialize_backend_session()
            assert test_id == "session_live_888"
            assert orchestrator.test_id == "session_live_888"

    def test_initialize_backend_session_404_or_offline_fallback(self, orchestrator):
        """When backend returns 404 or fails, fallback test_id must be used without crashing."""
        with patch("src.send_events.requests.get", side_effect=Exception("Connection refused")):
            test_id = orchestrator.initialize_backend_session()
            assert test_id == "test_unit_001"
            assert orchestrator.test_id == "test_unit_001"

    def test_process_frame_empty_image(self, orchestrator):
        """Processing blank frame should succeed without errors or false flags."""
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        result = orchestrator.process_frame(frame, frame_number=0)
        assert result["frame_number"] == 0
        assert result["tracks_count"] == 0
        assert result["flagged_events_count"] == 0
        assert result["latency_ms"] >= 0.0

    def test_red_and_green_box_snapshot_rendering(self, orchestrator):
        """
        Verify multi-student snapshot rendering draws RED alert box on the
        triggering student and GREEN boxes on non-triggering students.
        """
        frame = np.zeros((480, 640, 3), dtype=np.uint8) + 50

        # Two detected students: Student 1 (violator) and Student 2 (normal)
        violator_track = {
            "track_id": 1,
            "student_id": "student_001",
            "bbox": [100.0, 100.0, 200.0, 300.0],
            "keypoints": [],
        }
        normal_track = {
            "track_id": 2,
            "student_id": "student_002",
            "bbox": [350.0, 100.0, 450.0, 300.0],
            "keypoints": [],
        }
        all_tracks = [violator_track, normal_track]

        flagged_event = {
            "event_type": "hand_out_of_bounds",
            "confidence": 0.90,
            "risk_score": 85.0,
            "risk_category": "high",
            "bounding_box": [100.0, 100.0, 200.0, 300.0],
            "reason": "wrist below desk",
        }

        annotated_img, snap_path = orchestrator.movement_flagger.render_and_save_snapshot(
            frame=frame,
            flagged_event=flagged_event,
            frame_number=5,
            test_id="test_draw",
            all_tracks=all_tracks,
        )

        assert os.path.isfile(snap_path)
        assert annotated_img.shape == frame.shape

        # Sample pixel near violator box top-left (100, 100) -> should have RED color (BGR: 0, 0, 255)
        # Sample pixel near normal student box top-left (350, 100) -> should have GREEN color (BGR: 0, 200, 0)
        # Check that both red and green colors are present in the annotated snapshot
        b, g, r = cv2.split(annotated_img)
        has_red = np.any((r > 200) & (g < 50) & (b < 50))
        has_green = np.any((g > 180) & (r < 50) & (b < 50))
        assert has_red, "Annotated snapshot must contain RED alert box on violator"
        assert has_green, "Annotated snapshot must contain GREEN bounding box on non-violating student"

    def test_process_frame_dispatches_without_student_id_or_seat_zone(self, orchestrator):
        """Verify dispatched event payload strictly omits student_id and seat_zone."""
        dispatched_payloads = []

        def mock_dispatch(flagged_event, snapshot_path, frame_number, timestamp=None):
            payload = orchestrator.event_dispatcher.build_event_payload(
                flagged_event=flagged_event,
                frame_number=frame_number,
                timestamp=timestamp,
            )
            dispatched_payloads.append(payload)
            return True

        orchestrator.event_dispatcher.dispatch = mock_dispatch

        # Frame with synthetic violation
        frame = np.zeros((480, 640, 3), dtype=np.uint8) + 40
        # Mock pose estimator returning 1 track with downward reaching wrist
        mock_pose_output = {
            "frame_number": 1,
            "timestamp": "2026-08-22T12:00:00Z",
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
                        {"name": "left_wrist", "x": 140.0, "y": 340.0, "confidence": 0.85},  # Below hips (340 > 260+25)
                        {"name": "right_wrist", "x": 210.0, "y": 220.0, "confidence": 0.85},
                    ],
                }
            ],
        }

        with patch.object(orchestrator.pose_estimator, "process_frame", return_value=mock_pose_output):
            result = orchestrator.process_frame(frame, frame_number=1)

        assert result["flagged_events_count"] >= 1
        assert len(dispatched_payloads) >= 1

        for payload in dispatched_payloads:
            assert "student_id" not in payload, "Dispatched payload must NOT contain student_id"
            assert "seat_zone" not in payload, "Dispatched payload must NOT contain seat_zone"
            assert "test_id" in payload
            assert "bounding_box" in payload
            assert len(payload["bounding_box"]) == 4
            assert payload["risk_category"] in ("low", "high")
