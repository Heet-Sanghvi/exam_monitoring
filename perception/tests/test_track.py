import os
import sys
import numpy as np
import pytest

# Ensure perception/src is on python path
PERCEPTION_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PERCEPTION_DIR)

from src.track import PersonTracker, generate_track_color

def test_tracker_init():
    """Verify PersonTracker initializes with default parameters."""
    tracker = PersonTracker(conf_threshold=0.35)
    assert tracker.conf_threshold == 0.35
    assert tracker.student_counter == 0
    assert len(tracker.track_to_student) == 0

def test_persistent_student_id_mapping():
    """Verify raw tracker IDs map to stable student_xxx strings."""
    tracker = PersonTracker()
    
    # Assign new IDs
    sid_1 = tracker.get_or_create_student_id(1)
    sid_2 = tracker.get_or_create_student_id(2)
    sid_3 = tracker.get_or_create_student_id(5)

    assert sid_1 == "student_001"
    assert sid_2 == "student_002"
    assert sid_3 == "student_003"

    # Re-query existing IDs (must remain consistent)
    assert tracker.get_or_create_student_id(1) == "student_001"
    assert tracker.get_or_create_student_id(2) == "student_002"
    assert tracker.get_or_create_student_id(5) == "student_003"
    assert tracker.student_counter == 3

def test_tracker_reset():
    """Verify reset clears student mappings and trail history."""
    tracker = PersonTracker()
    tracker.get_or_create_student_id(1)
    tracker.get_or_create_student_id(2)
    assert len(tracker.track_to_student) == 2

    tracker.reset()
    assert len(tracker.track_to_student) == 0
    assert tracker.student_counter == 0
    assert len(tracker.trail_history) == 0

def test_track_color_palette():
    """Verify distinct colors generated for tracks."""
    c1 = generate_track_color(1)
    c2 = generate_track_color(2)
    assert len(c1) == 3
    assert len(c2) == 3
    assert all(0 <= v <= 255 for v in c1)

def test_track_empty_frame():
    """Verify update handles empty or None frames."""
    tracker = PersonTracker()
    res = tracker.update(np.zeros((0, 0, 3), dtype=np.uint8))
    assert res["tracks"] == []
    assert res["active_tracks_count"] == 0

    none_res = tracker.update(None)
    assert none_res["tracks"] == []
    assert none_res["active_tracks_count"] == 0

def test_draw_tracks():
    """Verify rendering of track bounding boxes and student badges."""
    tracker = PersonTracker()
    frame = np.zeros((480, 640, 3), dtype=np.uint8)

    mock_track_result = {
        "tracks": [
            {
                "track_id": 1,
                "student_id": "student_001",
                "bbox": [50.0, 80.0, 200.0, 350.0],
                "confidence": 0.88,
                "class_id": 0,
                "class_name": "person",
                "keypoints": np.zeros((17, 3), dtype=np.float32),
            },
            {
                "track_id": 2,
                "student_id": "student_002",
                "bbox": [250.0, 80.0, 400.0, 350.0],
                "confidence": 0.91,
                "class_id": 0,
                "class_name": "person",
                "keypoints": np.zeros((17, 3), dtype=np.float32),
            },
        ],
        "active_tracks_count": 2,
        "total_unique_students": 2,
        "inference_time_ms": 22.4,
        "device": tracker.device,
    }

    annotated = tracker.draw_tracks(frame, mock_track_result, fps=28.0)
    assert annotated is not None
    assert annotated.shape == frame.shape
    assert annotated.dtype == np.uint8
    assert np.any(annotated > 0)
