import os
import json
import pytest

PERCEPTION_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
REPO_ROOT = os.path.abspath(os.path.join(PERCEPTION_ROOT, ".."))

def test_perception_directory_structure():
    """Verify that all required perception folders exist."""
    required_dirs = [
        os.path.join(PERCEPTION_ROOT, "src"),
        os.path.join(PERCEPTION_ROOT, "data"),
        os.path.join(PERCEPTION_ROOT, "data", "sample_videos"),
        os.path.join(PERCEPTION_ROOT, "data", "test_clips", "normal"),
        os.path.join(PERCEPTION_ROOT, "data", "test_clips", "suspicious"),
        os.path.join(PERCEPTION_ROOT, "models"),
        os.path.join(PERCEPTION_ROOT, "output"),
        os.path.join(PERCEPTION_ROOT, "tests"),
    ]
    for directory in required_dirs:
        assert os.path.isdir(directory), f"Missing directory: {directory}"

def test_perception_src_modules_exist():
    """Verify that all planned perception pipeline module files exist."""
    expected_modules = [
        "__init__.py",
        "detect.py",
        "track.py",
        "pose.py",
        "behavior_features.py",
        "behavior_rules.py",
        "object_detect.py",
        "risk_score.py",
        "send_events.py",
        "main.py",
    ]
    src_dir = os.path.join(PERCEPTION_ROOT, "src")
    for mod in expected_modules:
        mod_path = os.path.join(src_dir, mod)
        assert os.path.isfile(mod_path), f"Missing module file: {mod_path}"

def test_shared_schemas_accessible_and_valid():
    """Verify that shared schemas are present and contain valid JSON."""
    schema_behavior_path = os.path.join(REPO_ROOT, "shared", "schema_behavior_event.json")
    schema_pose_path = os.path.join(REPO_ROOT, "shared", "schema_pose_output.json")

    assert os.path.isfile(schema_behavior_path), "shared/schema_behavior_event.json not found"
    assert os.path.isfile(schema_pose_path), "shared/schema_pose_output.json not found"

    with open(schema_behavior_path, "r", encoding="utf-8") as f:
        behavior_schema = json.load(f)
        assert "properties" in behavior_schema
        assert "event_type" in behavior_schema["properties"]

    with open(schema_pose_path, "r", encoding="utf-8") as f:
        pose_schema = json.load(f)
        assert "properties" in pose_schema
        assert "tracks" in pose_schema["properties"]

def test_environment_core_imports():
    """Verify that core packages required for perception are importable."""
    import numpy as np
    import cv2
    import requests
    import pydantic
    import scipy

    assert np is not None
    assert cv2 is not None
    assert requests is not None
    assert pydantic is not None
    assert scipy is not None
