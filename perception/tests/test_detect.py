import os
import sys
import numpy as np
import pytest

# Ensure perception/src is on python path
PERCEPTION_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PERCEPTION_DIR)

from src.detect import PersonDetector, get_optimal_device

def test_optimal_device_detection():
    """Verify that device resolution correctly handles MPS on Apple Silicon."""
    device = get_optimal_device()
    assert device in ["mps", "cuda", "cpu"]
    
    # Check explicit override
    assert get_optimal_device("cpu") == "cpu"
    assert get_optimal_device("mps") == "mps"

def test_person_detector_init():
    """Verify detector initialization and model loading."""
    detector = PersonDetector(conf_threshold=0.4)
    assert detector.conf_threshold == 0.4
    assert detector.model is not None

def test_detect_empty_frame():
    """Verify detector gracefully handles empty/None frames."""
    detector = PersonDetector()
    
    empty_result = detector.detect(np.zeros((0, 0, 3), dtype=np.uint8))
    assert empty_result["person_count"] == 0
    assert empty_result["detections"] == []

    none_result = detector.detect(None)
    assert none_result["person_count"] == 0
    assert none_result["detections"] == []

def test_detect_synthetic_frame_structure():
    """Verify detection response schema on a valid frame."""
    detector = PersonDetector()
    
    # Create synthetic test frame (640x480)
    test_frame = np.full((480, 640, 3), 128, dtype=np.uint8)
    
    result = detector.detect(test_frame)
    assert "detections" in result
    assert "person_count" in result
    assert "inference_time_ms" in result
    assert "device" in result
    assert isinstance(result["person_count"], int)
    assert isinstance(result["inference_time_ms"], float)
    assert result["inference_time_ms"] >= 0.0

def test_draw_detections():
    """Verify drawing bounding boxes, labels, and HUD onto a frame."""
    detector = PersonDetector()
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    
    mock_detection_result = {
        "detections": [
            {
                "bbox": [100.0, 150.0, 300.0, 450.0],
                "confidence": 0.92,
                "class_id": 0,
                "class_name": "person",
                "keypoints": np.zeros((17, 3), dtype=np.float32),
            }
        ],
        "person_count": 1,
        "inference_time_ms": 15.5,
        "device": detector.device,
    }
    
    annotated = detector.draw_detections(frame, mock_detection_result, fps=32.5)
    assert annotated is not None
    assert annotated.shape == frame.shape
    assert annotated.dtype == np.uint8
    # Frame should have been modified (non-zero pixels where HUD/bbox drawn)
    assert np.any(annotated > 0)
