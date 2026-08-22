"""
Tests for Phase 5: Behavior Flagging, Snapshot Routing, and Event Dispatching.

Covers:
- test_bbox_conversion            : [x1,y1,x2,y2] → [x,y,w,h] conversion (explicitly required)
- test_body_movement_features     : vertical nose drop and shoulder tilt calculations
- test_hand_movement_features     : wrist below hip / lateral extension calculations
- test_flagging_engine_no_flag    : no violation when student is upright (below threshold)
- test_flagging_engine_body_flag  : body_movement flag triggers correctly
- test_flagging_engine_hand_flag  : hand_out_of_bounds flag triggers correctly
- test_risk_score_and_category    : risk_score range and risk_category derivation
- test_debounce_cooldown          : same violation not re-fired within debounce_frames window
- test_snapshot_routing           : snapshot saved to correct low_risk / high_risk folder
- test_send_events_offline        : offline backend → local fallback (no crash, JSON saved)
- test_send_events_payload_fields : payload includes all required fields, no student_id/seat_zone
"""

import json
import os
import sys
import tempfile
from typing import Any, Dict
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

# Make sure src is importable from tests/
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.behavior_features import (
    compute_body_movement_features,
    compute_hand_movement_features,
    extract_keypoints_map,
)
from src.behavior_rules import MovementFlaggingConfig, MovementFlaggingEngine
from src.send_events import EventDispatcher, EventDispatcherConfig, convert_bbox_xyxy_to_xywh


# ──────────────────────────────────────────────────────────────────────────────
# Helper Fixtures
# ──────────────────────────────────────────────────────────────────────────────

def make_keypoints_map(
    nose_y: float = 100.0,
    shoulder_y: float = 200.0,
    hip_y: float = 350.0,
    wrist_y: float = 300.0,
    shoulder_tilt: float = 0.0,
    conf: float = 0.9,
) -> Dict[str, Dict[str, float]]:
    """Create a synthetic keypoints dict simulating a seated student."""
    cx = 300.0
    half_w = 40.0
    return {
        "nose":           {"x": cx,               "y": nose_y,           "confidence": conf},
        "left_shoulder":  {"x": cx - half_w,       "y": shoulder_y + shoulder_tilt, "confidence": conf},
        "right_shoulder": {"x": cx + half_w,       "y": shoulder_y,       "confidence": conf},
        "left_hip":       {"x": cx - 30.0,         "y": hip_y,            "confidence": conf},
        "right_hip":      {"x": cx + 30.0,         "y": hip_y,            "confidence": conf},
        "left_wrist":     {"x": cx - 20.0,         "y": wrist_y,          "confidence": conf},
        "right_wrist":    {"x": cx + 20.0,         "y": wrist_y,          "confidence": conf},
    }


def make_track_dict(
    student_id: str = "student_001",
    nose_y: float = 100.0,
    shoulder_y: float = 200.0,
    hip_y: float = 350.0,
    wrist_y: float = 300.0,
    shoulder_tilt: float = 0.0,
    bbox: list = None,
    conf: float = 0.85,
) -> Dict[str, Any]:
    """Create a synthetic track dict as produced by track.py / pose.py."""
    if bbox is None:
        bbox = [260.0, 80.0, 380.0, 480.0]  # [x1, y1, x2, y2]
    cx = 300.0
    half_w = 40.0
    keypoints = [
        {"name": "nose",           "x": cx,          "y": nose_y,           "confidence": conf},
        {"name": "left_shoulder",  "x": cx - half_w, "y": shoulder_y + shoulder_tilt, "confidence": conf},
        {"name": "right_shoulder", "x": cx + half_w, "y": shoulder_y,       "confidence": conf},
        {"name": "left_hip",       "x": cx - 30.0,   "y": hip_y,            "confidence": conf},
        {"name": "right_hip",      "x": cx + 30.0,   "y": hip_y,            "confidence": conf},
        {"name": "left_wrist",     "x": cx - 20.0,   "y": wrist_y,          "confidence": conf},
        {"name": "right_wrist",    "x": cx + 20.0,   "y": wrist_y,          "confidence": conf},
    ]
    return {
        "student_id": student_id,
        "bbox": bbox,
        "keypoints": keypoints,
        "confidence": conf,
    }


# ──────────────────────────────────────────────────────────────────────────────
# 1. Bounding Box Conversion (EXPLICITLY REQUIRED UNIT TEST)
# ──────────────────────────────────────────────────────────────────────────────

class TestBboxConversion:
    """Explicit unit tests for [x1,y1,x2,y2] → [x,y,w,h] conversion."""

    def test_basic_conversion(self):
        result = convert_bbox_xyxy_to_xywh([10.0, 20.0, 110.0, 220.0])
        assert result == [10.0, 20.0, 100.0, 200.0], f"Got {result}"

    def test_top_left_is_preserved(self):
        """x and y must equal the original x1, y1."""
        x1, y1, x2, y2 = 55.5, 99.1, 200.3, 350.7
        result = convert_bbox_xyxy_to_xywh([x1, y1, x2, y2])
        assert result[0] == x1, "x must equal x1"
        assert result[1] == y1, "y must equal y1"

    def test_width_height_calculation(self):
        """w = x2 - x1 and h = y2 - y1 exactly."""
        result = convert_bbox_xyxy_to_xywh([100.0, 50.0, 300.0, 250.0])
        assert result[2] == 200.0, f"Width should be 200.0, got {result[2]}"
        assert result[3] == 200.0, f"Height should be 200.0, got {result[3]}"

    def test_output_length_is_4(self):
        result = convert_bbox_xyxy_to_xywh([0.0, 0.0, 640.0, 480.0])
        assert len(result) == 4

    def test_zero_width_box(self):
        """Degenerate case: x1 == x2 → width = 0."""
        result = convert_bbox_xyxy_to_xywh([50.0, 50.0, 50.0, 150.0])
        assert result[2] == 0.0

    def test_wrong_input_length_raises(self):
        with pytest.raises(ValueError):
            convert_bbox_xyxy_to_xywh([10.0, 20.0, 30.0])


# ──────────────────────────────────────────────────────────────────────────────
# 2. Body Movement Feature Extraction
# ──────────────────────────────────────────────────────────────────────────────

class TestBodyMovementFeatures:

    def test_nose_drop_with_baseline(self):
        """Nose drop should be measured from the baseline upright position."""
        baseline = make_keypoints_map(nose_y=100.0)
        current = make_keypoints_map(nose_y=250.0)  # 150px drop
        feats = compute_body_movement_features(current, baseline_kpts=baseline)
        assert feats["vertical_nose_drop"] == pytest.approx(150.0, abs=1.0)

    def test_no_drop_when_upright(self):
        """No drop should be reported when the nose is at baseline level."""
        baseline = make_keypoints_map(nose_y=100.0)
        current = make_keypoints_map(nose_y=100.0)
        feats = compute_body_movement_features(current, baseline_kpts=baseline)
        assert feats["vertical_nose_drop"] == 0.0

    def test_shoulder_tilt_horizontal(self):
        """Level shoulders should produce near-zero tilt."""
        kpts = make_keypoints_map(shoulder_tilt=0.0)
        feats = compute_body_movement_features(kpts)
        assert feats["shoulder_tilt_deg"] < 5.0

    def test_shoulder_tilt_large(self):
        """Pronounced shoulder tilt should produce a significant angle."""
        kpts = make_keypoints_map(shoulder_tilt=60.0)  # 60px vertical offset over 80px horizontal span
        feats = compute_body_movement_features(kpts)
        assert feats["shoulder_tilt_deg"] > 15.0


# ──────────────────────────────────────────────────────────────────────────────
# 3. Hand Movement Feature Extraction
# ──────────────────────────────────────────────────────────────────────────────

class TestHandMovementFeatures:

    def test_no_extension_when_hands_above_hips(self):
        """Hands well above hip level → zero below-hip metric."""
        kpts = make_keypoints_map(hip_y=400.0, wrist_y=280.0)
        bbox = [260.0, 80.0, 380.0, 500.0]
        feats = compute_hand_movement_features(kpts, bbox)
        assert feats["max_wrist_below_hip"] == 0.0

    def test_extension_below_hip(self):
        """Wrist below hip → non-zero below-hip distance."""
        kpts = make_keypoints_map(hip_y=300.0, wrist_y=380.0)  # 80px below hip
        bbox = [260.0, 80.0, 380.0, 480.0]
        feats = compute_hand_movement_features(kpts, bbox)
        assert feats["max_wrist_below_hip"] == pytest.approx(80.0, abs=2.0)

    def test_lateral_reach(self):
        """Wrists far outside bbox bounds → non-zero lateral metric."""
        kpts = make_keypoints_map()
        kpts["right_wrist"]["x"] = 500.0  # Far outside bbox x2=380
        bbox = [260.0, 80.0, 380.0, 480.0]
        feats = compute_hand_movement_features(kpts, bbox)
        assert feats["max_wrist_reach_lateral"] > 0.0


# ──────────────────────────────────────────────────────────────────────────────
# 4. Movement Flagging Engine
# ──────────────────────────────────────────────────────────────────────────────

class TestMovementFlaggingEngine:

    def _make_engine(self, **kwargs) -> MovementFlaggingEngine:
        # Allow callers to override any field; set sensible defaults for unspecified ones
        defaults = {"snapshot_base_dir": tempfile.mkdtemp(), "debounce_frames": 5}
        defaults.update(kwargs)
        cfg = MovementFlaggingConfig(**defaults)
        return MovementFlaggingEngine(config=cfg)

    def test_no_flag_when_upright(self):
        """Normal upright posture should produce no flag."""
        engine = self._make_engine()
        track = make_track_dict(nose_y=100.0, shoulder_y=200.0, hip_y=350.0, wrist_y=300.0)
        # Feed 10 baseline frames first
        for i in range(10):
            engine.evaluate_track(track, frame_number=i)
        result = engine.evaluate_track(track, frame_number=10)
        assert result is None

    def test_body_movement_flag_triggers(self):
        """Severe head bending below baseline should trigger body_movement."""
        engine = self._make_engine(body_bend_drop_px=80.0, debounce_frames=5)
        # Register baseline with upright nose
        baseline_track = make_track_dict(nose_y=100.0, shoulder_y=220.0)
        for i in range(6):
            engine.evaluate_track(baseline_track, frame_number=i)

        # Now simulate deep forward bend (nose drops 200px below baseline)
        bent_track = make_track_dict(nose_y=350.0, shoulder_y=220.0, student_id="student_001")
        result = engine.evaluate_track(bent_track, frame_number=20)
        assert result is not None
        assert result["event_type"] == "body_movement"

    def test_hand_out_of_bounds_flag_triggers(self):
        """Wrist reaching far below hips should trigger hand_out_of_bounds."""
        engine = self._make_engine(hand_below_hip_px=30.0, debounce_frames=5)
        # Normal baseline
        for i in range(6):
            engine.evaluate_track(make_track_dict(hip_y=350.0, wrist_y=280.0), frame_number=i)
        # Now reach far below hips
        reach_track = make_track_dict(hip_y=350.0, wrist_y=450.0)  # 100px below hip
        result = engine.evaluate_track(reach_track, frame_number=20)
        assert result is not None
        assert result["event_type"] == "hand_out_of_bounds"

    def test_risk_score_is_bounded(self):
        """risk_score must be in [0.0, 100.0]."""
        engine = self._make_engine(hand_below_hip_px=5.0, debounce_frames=1)
        for i in range(3):
            engine.evaluate_track(make_track_dict(hip_y=350.0, wrist_y=280.0), frame_number=i)
        track = make_track_dict(hip_y=200.0, wrist_y=600.0)  # Extreme reach
        result = engine.evaluate_track(track, frame_number=10)
        if result:
            assert 0.0 <= result["risk_score"] <= 100.0

    def test_risk_category_low_high(self):
        """risk_category must be 'low' when score < threshold, 'high' otherwise."""
        engine = self._make_engine(high_risk_threshold=60.0, hand_below_hip_px=5.0, debounce_frames=1)
        # Build a minimal flag
        track = make_track_dict(hip_y=300.0, wrist_y=380.0)
        result = engine.evaluate_track(track, frame_number=5)
        if result:
            expected_cat = "high" if result["risk_score"] >= 60.0 else "low"
            assert result["risk_category"] == expected_cat

    def test_debounce_prevents_repeated_flags(self):
        """Same violation must not re-fire within debounce_frames window."""
        engine = self._make_engine(hand_below_hip_px=10.0, debounce_frames=20)
        track = make_track_dict(hip_y=300.0, wrist_y=400.0)
        first = engine.evaluate_track(track, frame_number=5)
        assert first is not None  # Should fire once
        second = engine.evaluate_track(track, frame_number=10)  # Only 5 frames apart < 20
        assert second is None  # Should be suppressed by debounce

    def test_snapshot_routes_to_correct_folder(self):
        """Snapshot should be placed in low_risk/ or high_risk/ matching risk_category."""
        engine = self._make_engine(hand_below_hip_px=5.0, high_risk_threshold=60.0, debounce_frames=1)
        track = make_track_dict(hip_y=300.0, wrist_y=380.0)
        result = engine.evaluate_track(track, frame_number=5)
        assert result is not None

        dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        _, filepath = engine.render_and_save_snapshot(
            frame=dummy_frame,
            flagged_event=result,
            frame_number=5,
            test_id="test_snap_001",
        )
        assert os.path.isfile(filepath), f"Snapshot not saved at {filepath}"
        if result["risk_category"] == "high":
            assert "high_risk" in filepath
        else:
            assert "low_risk" in filepath


# ──────────────────────────────────────────────────────────────────────────────
# 5. Event Dispatcher — Offline Resilience & Payload Contract
# ──────────────────────────────────────────────────────────────────────────────

class TestEventDispatcher:

    def _make_dispatcher(self, tmp_dir: str) -> EventDispatcher:
        cfg = EventDispatcherConfig(
            base_url="http://127.0.0.1:9999",  # Non-existent port
            offline_fallback_dir=tmp_dir,
            request_timeout_seconds=0.5,
            default_test_id="test_offline_fallback",
        )
        return EventDispatcher(config=cfg)

    def _sample_flagged_event(self) -> Dict:
        return {
            "student_id": "student_001",
            "event_type": "hand_out_of_bounds",
            "confidence": 0.87,
            "risk_score": 72.5,
            "risk_category": "high",
            "bounding_box": [100.0, 50.0, 300.0, 400.0],  # [x1,y1,x2,y2]
            "reason": "wrist below hip (80.0px >= 30.0px)",
        }

    def test_offline_fetch_returns_default(self):
        """fetch_active_test_id must not raise even when backend is down."""
        with tempfile.TemporaryDirectory() as tmp:
            dispatcher = self._make_dispatcher(tmp)
            test_id = dispatcher.fetch_active_test_id()
            assert test_id == "test_offline_fallback"

    def test_offline_dispatch_saves_locally(self):
        """dispatch() must not raise on connection failure and must save JSON locally."""
        with tempfile.TemporaryDirectory() as tmp:
            dispatcher = self._make_dispatcher(tmp)
            dispatcher.test_id = "test_offline_001"

            flagged = self._sample_flagged_event()
            # Create a real dummy snapshot file
            snap_path = os.path.join(tmp, "dummy_snap.jpg")
            dummy_img = np.zeros((100, 100, 3), dtype=np.uint8)
            import cv2
            cv2.imwrite(snap_path, dummy_img)

            success = dispatcher.dispatch(
                flagged_event=flagged,
                snapshot_path=snap_path,
                frame_number=42,
            )
            assert success is False  # Backend unreachable

            # Verify local fallback JSON was written
            json_files = [f for f in os.listdir(tmp) if f.endswith(".json")]
            assert len(json_files) == 1, f"Expected 1 offline JSON, found: {json_files}"

            with open(os.path.join(tmp, json_files[0])) as f:
                saved = json.load(f)
            assert saved["event_type"] == "hand_out_of_bounds"
            assert saved["test_id"] == "test_offline_001"
            assert saved["frame_number"] == 42

    def test_payload_bbox_is_xywh(self):
        """build_event_payload must convert bbox from [x1,y1,x2,y2] → [x,y,w,h]."""
        with tempfile.TemporaryDirectory() as tmp:
            dispatcher = self._make_dispatcher(tmp)
            flagged = self._sample_flagged_event()
            # bbox input: [100.0, 50.0, 300.0, 400.0] → expected [100, 50, 200, 350]
            payload = dispatcher.build_event_payload(flagged, frame_number=1)
            bbox = payload["bounding_box"]
            assert bbox[0] == 100.0, "x should equal x1"
            assert bbox[1] == 50.0,  "y should equal y1"
            assert bbox[2] == 200.0, "w should equal x2-x1"
            assert bbox[3] == 350.0, "h should equal y2-y1"

    def test_payload_has_required_fields(self):
        """Payload must include all required API fields."""
        with tempfile.TemporaryDirectory() as tmp:
            dispatcher = self._make_dispatcher(tmp)
            flagged = self._sample_flagged_event()
            payload = dispatcher.build_event_payload(flagged, frame_number=99)

            required_fields = [
                "event_id", "test_id", "frame_number", "timestamp",
                "event_type", "confidence", "risk_score", "risk_category",
                "bounding_box", "metadata",
            ]
            for field_name in required_fields:
                assert field_name in payload, f"Missing field: {field_name}"

    def test_payload_has_no_student_id(self):
        """Per finalized schema contract — payload must NOT contain student_id or seat_zone."""
        with tempfile.TemporaryDirectory() as tmp:
            dispatcher = self._make_dispatcher(tmp)
            flagged = self._sample_flagged_event()
            payload = dispatcher.build_event_payload(flagged, frame_number=1)
            assert "student_id" not in payload, "student_id must NOT be in payload (schema decision)"
            assert "seat_zone" not in payload, "seat_zone must NOT be in payload (schema decision)"

    def test_fetch_active_test_id_from_mock_backend(self):
        """When backend returns test_id, dispatcher must adopt it."""
        with tempfile.TemporaryDirectory() as tmp:
            dispatcher = self._make_dispatcher(tmp)
            mock_resp = MagicMock()
            mock_resp.json.return_value = {"test_id": "test_from_backend_99"}
            mock_resp.raise_for_status = MagicMock()
            with patch("src.send_events.requests.get", return_value=mock_resp):
                test_id = dispatcher.fetch_active_test_id()
            assert test_id == "test_from_backend_99"
            assert dispatcher.test_id == "test_from_backend_99"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
