"""
Behavior Features Extraction Module
Part of the Exam Monitoring Perception Pipeline.

Calculates physical motion and kinematic features from 2D pose keypoints:
- Vertical head/torso displacement (bending towards floor/desk)
- Upper body tilt / shoulder roll
- Wrist reaching out of bounding box / reaching below hip level
- Relative hand motion distances
"""

import math
from typing import Dict, Any, Optional, List, Tuple, Union
import numpy as np

from src.pose import COCO_KEYPOINT_NAMES


def extract_keypoints_map(keypoints: Union[List[Dict[str, Any]], np.ndarray]) -> Dict[str, Dict[str, float]]:
    """
    Convert keypoint list or array into a lookup dict by keypoint name.

    Returns:
        Dict mapping name to {'x': float, 'y': float, 'confidence': float}
    """
    kpts_map: Dict[str, Dict[str, float]] = {}

    if isinstance(keypoints, np.ndarray):
        for idx, name in enumerate(COCO_KEYPOINT_NAMES):
            if idx < len(keypoints):
                kx = float(keypoints[idx][0])
                ky = float(keypoints[idx][1])
                kc = float(keypoints[idx][2]) if len(keypoints[idx]) > 2 else 1.0
                kpts_map[name] = {"x": kx, "y": ky, "confidence": kc}
    elif isinstance(keypoints, list):
        for kp in keypoints:
            if isinstance(kp, dict) and "name" in kp:
                kpts_map[kp["name"]] = {
                    "x": float(kp.get("x", 0.0)),
                    "y": float(kp.get("y", 0.0)),
                    "confidence": float(kp.get("confidence", 0.0)),
                }

    return kpts_map


def compute_body_movement_features(
    current_kpts: Dict[str, Dict[str, float]],
    baseline_kpts: Optional[Dict[str, Dict[str, float]]] = None,
) -> Dict[str, float]:
    """
    Calculate body bending, torso inclination, and vertical displacement metrics.

    Args:
        current_kpts: Current frame keypoints dict.
        baseline_kpts: Optional reference/initial upright keypoints dict.

    Returns:
        Dict containing:
            - vertical_nose_drop: Downward pixel drop of nose relative to baseline (or shoulders)
            - shoulder_tilt_deg: Absolute angle of shoulder line from horizontal
            - torso_height_px: Distance between shoulders midpoint and hips midpoint
            - is_bending: Boolean indicator of pronounced downward bending
    """
    nose = current_kpts.get("nose")
    l_sh = current_kpts.get("left_shoulder")
    r_sh = current_kpts.get("right_shoulder")
    l_hip = current_kpts.get("left_hip")
    r_hip = current_kpts.get("right_hip")

    # 1. Shoulder Tilt Angle
    shoulder_tilt_deg = 0.0
    if l_sh and r_sh and l_sh["confidence"] > 0.3 and r_sh["confidence"] > 0.3:
        dx = r_sh["x"] - l_sh["x"]
        dy = r_sh["y"] - l_sh["y"]
        if abs(dx) > 1e-3:
            shoulder_tilt_deg = abs(math.degrees(math.atan2(dy, dx)))
            # Normalize so 0 is horizontal (level shoulders)
            if shoulder_tilt_deg > 90.0:
                shoulder_tilt_deg = abs(180.0 - shoulder_tilt_deg)

    # 2. Vertical Head / Nose Drop
    vertical_nose_drop = 0.0
    if nose and nose["confidence"] > 0.25:
        if baseline_kpts and "nose" in baseline_kpts and baseline_kpts["nose"]["confidence"] > 0.25:
            # Drop relative to baseline upright head position
            vertical_nose_drop = max(0.0, nose["y"] - baseline_kpts["nose"]["y"])
        elif l_sh and r_sh:
            # Drop relative to shoulder midpoint
            sh_mid_y = (l_sh["y"] + r_sh["y"]) / 2.0
            # In upright posture, nose is well above shoulders (smaller y). If nose y >= sh_mid_y, deep bend
            vertical_nose_drop = max(0.0, nose["y"] - (sh_mid_y - 40.0))

    # 3. Torso compression / height
    torso_height_px = 0.0
    if l_sh and r_sh and l_hip and r_hip:
        sh_mid_y = (l_sh["y"] + r_sh["y"]) / 2.0
        hip_mid_y = (l_hip["y"] + r_hip["y"]) / 2.0
        torso_height_px = max(0.0, hip_mid_y - sh_mid_y)

    return {
        "vertical_nose_drop": round(float(vertical_nose_drop), 2),
        "shoulder_tilt_deg": round(float(shoulder_tilt_deg), 2),
        "torso_height_px": round(float(torso_height_px), 2),
    }


def compute_hand_movement_features(
    current_kpts: Dict[str, Dict[str, float]],
    bbox: List[float],
) -> Dict[str, float]:
    """
    Calculate hand extension, reaching below desk/hip, and lateral reach out of bounds.

    Args:
        current_kpts: Current frame keypoints dict.
        bbox: Bounding box [x1, y1, x2, y2].

    Returns:
        Dict containing:
            - max_wrist_below_hip: Maximum distance wrists extend below hips (reaching down)
            - max_wrist_reach_lateral: Maximum distance wrists extend laterally from torso
            - is_wrist_out_of_bounds: Whether wrist exceeds the normal desk/torso envelope
    """
    l_wrist = current_kpts.get("left_wrist")
    r_wrist = current_kpts.get("right_wrist")
    l_hip = current_kpts.get("left_hip")
    r_hip = current_kpts.get("right_hip")
    l_sh = current_kpts.get("left_shoulder")
    r_sh = current_kpts.get("right_shoulder")

    max_wrist_below_hip = 0.0
    max_wrist_reach_lateral = 0.0

    # Reference hip Y level (lower torso boundary)
    hip_y = None
    if l_hip and r_hip and (l_hip["confidence"] > 0.3 or r_hip["confidence"] > 0.3):
        hip_y = max(l_hip["y"], r_hip["y"])
    elif l_sh and r_sh:
        # Approximate hip Y if occluded by desk (~1.5x torso height below shoulders)
        sh_y = (l_sh["y"] + r_sh["y"]) / 2.0
        hip_y = sh_y + 120.0

    # Reference torso center X
    torso_cx = (bbox[0] + bbox[2]) / 2.0
    torso_half_w = (bbox[2] - bbox[0]) / 2.0

    for wrist in [l_wrist, r_wrist]:
        if wrist and wrist["confidence"] > 0.25:
            # Check downward reach below hip
            if hip_y is not None:
                drop_below_hip = wrist["y"] - hip_y
                if drop_below_hip > max_wrist_below_hip:
                    max_wrist_below_hip = drop_below_hip

            # Check lateral extension beyond bounding box / torso
            lat_dist = abs(wrist["x"] - torso_cx) - torso_half_w
            if lat_dist > max_wrist_reach_lateral:
                max_wrist_reach_lateral = lat_dist

    return {
        "max_wrist_below_hip": round(float(max(0.0, max_wrist_below_hip)), 2),
        "max_wrist_reach_lateral": round(float(max(0.0, max_wrist_reach_lateral)), 2),
    }
