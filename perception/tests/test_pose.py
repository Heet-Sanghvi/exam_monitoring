import os
import sys
import json
import numpy as np
import pytest
import jsonschema

# Ensure perception/src is on python path
PERCEPTION_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
REPO_DIR = os.path.abspath(os.path.join(PERCEPTION_DIR, ".."))
sys.path.insert(0, PERCEPTION_DIR)

from src.pose import (
    PoseEstimator,
    calculate_head_pose_angles,
    COCO_KEYPOINT_NAMES,
    COCO_SKELETON,
)

SCHEMA_POSE_PATH = os.path.join(REPO_DIR, "shared", "schema_pose_output.json")


def load_pose_schema() -> dict:
    with open(SCHEMA_POSE_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def test_coco_keypoints_constants():
    """Verify that exactly 17 COCO keypoints and skeleton pairs are defined."""
    assert len(COCO_KEYPOINT_NAMES) == 17
    assert "nose" in COCO_KEYPOINT_NAMES
    assert "left_wrist" in COCO_KEYPOINT_NAMES
    assert "right_wrist" in COCO_KEYPOINT_NAMES
    assert len(COCO_SKELETON) > 10


def test_head_pose_calculation_straight():
    """Verify head pose returns yaw near 0 for frontal face."""
    kpts_map = {
        "nose": {"x": 100.0, "y": 100.0, "confidence": 0.9},
        "left_eye": {"x": 80.0, "y": 80.0, "confidence": 0.9},
        "right_eye": {"x": 120.0, "y": 80.0, "confidence": 0.9},
        "left_ear": {"x": 60.0, "y": 85.0, "confidence": 0.8},
        "right_ear": {"x": 140.0, "y": 85.0, "confidence": 0.8},
    }
    angles = calculate_head_pose_angles(kpts_map)
    assert angles is not None
    assert "yaw" in angles
    assert "pitch" in angles
    assert "roll" in angles
    assert abs(angles["yaw"]) < 5.0  # Centered
    assert abs(angles["roll"]) < 5.0  # Horizontal


def test_head_pose_low_confidence():
    """Verify head pose returns None when keypoint confidence is too low."""
    kpts_map = {
        "nose": {"x": 100.0, "y": 100.0, "confidence": 0.1},
        "left_eye": {"x": 80.0, "y": 80.0, "confidence": 0.9},
        "right_eye": {"x": 120.0, "y": 80.0, "confidence": 0.9},
    }
    angles = calculate_head_pose_angles(kpts_map)
    assert angles is None


def test_pose_estimator_schema_conformance():
    """Verify output strictly matches shared/schema_pose_output.json."""
    estimator = PoseEstimator()
    frame = np.zeros((480, 640, 3), dtype=np.uint8)

    output = estimator.process_frame(frame, frame_number=10)
    schema = load_pose_schema()

    # Create schema-compliant payload (without internal metadata helper)
    payload = {
        "frame_number": output["frame_number"],
        "timestamp": output["timestamp"],
        "tracks": output["tracks"],
    }

    # Validate against JSON schema
    jsonschema.validate(instance=payload, schema=schema)
    assert output["frame_number"] == 10
    assert isinstance(output["tracks"], list)


def test_pose_drawing():
    """Verify rendering of skeleton and student annotations."""
    estimator = PoseEstimator()
    frame = np.zeros((480, 640, 3), dtype=np.uint8)

    mock_keypoints = [
        {"name": name, "x": 100.0 + i * 5, "y": 120.0 + i * 5, "confidence": 0.85}
        for i, name in enumerate(COCO_KEYPOINT_NAMES)
    ]

    mock_pose_output = {
        "frame_number": 1,
        "timestamp": "2026-08-22T10:00:00Z",
        "tracks": [
            {
                "track_id": 1,
                "student_id": "student_001",
                "bbox": [50.0, 50.0, 200.0, 350.0],
                "keypoints": mock_keypoints,
                "head_pose": {"yaw": 12.0, "pitch": -5.0, "roll": 2.0},
            }
        ],
        "metadata": {"inference_time_ms": 18.0},
    }

    annotated = estimator.draw_pose_frame(frame, mock_pose_output, fps=30.0)
    assert annotated is not None
    assert annotated.shape == frame.shape
    assert np.any(annotated > 0)
