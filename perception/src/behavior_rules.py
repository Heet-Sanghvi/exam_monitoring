"""
Behavioral Rules & Movement-Flagging Module
Part of the Exam Monitoring Perception Pipeline.

Evaluates tracked student keypoints against configurable thresholds to detect:
- hand_out_of_bounds (reaching below desk/hips or extending laterally into neighboring space)
- body_movement (pronounced bending downwards towards desk/floor or sharp shoulder tilt)

When a flag triggers:
1. Renders a visibly distinct alert box around the violating student on the frame.
2. Computes numeric risk_score and derives risk_category ("low" or "high").
3. Saves a timestamped snapshot image into output/snapshots/low_risk/ or high_risk/.
4. Produces a structured event payload for dispatch.
"""

import os
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
import cv2

from src.behavior_features import (
    extract_keypoints_map,
    compute_body_movement_features,
    compute_hand_movement_features,
)


@dataclass
class MovementFlaggingConfig:
    """Configurable threshold parameters for movement anomaly detection."""

    # Body Movement Thresholds
    body_bend_drop_px: float = 65.0
    """Vertical nose drop (px) from baseline sufficient to flag bending on its own."""

    body_tilt_deg: float = 35.0
    """Shoulder tilt angle (degrees) from horizontal. Raised from 20° to cut static-posture
    false positives from wide-angle CCTV where students naturally sit at slight angles.
    Tilt alone only fires when body_tilt_min_nose_drop_px is also satisfied."""

    body_tilt_min_nose_drop_px: float = 30.0
    """Minimum nose drop (px) that must co-occur with a tilt event before body_movement is
    raised. This is the key false-positive guard: a student sitting at an angle with a static
    head position will have near-zero nose drop and will NOT trigger. A student actually
    bending forward to reach/copy will have both tilt AND a significant nose drop.
    Set to 0.0 to disable the co-condition and revert to OR logic."""

    # Hand / Wrist Out of Bounds Thresholds
    hand_below_hip_px: float = 25.0       # Wrist extending below hip / desk baseline
    hand_lateral_reach_px: float = 30.0   # Wrist extending laterally out of student envelope

    # Risk Scoring Configuration
    high_risk_threshold: float = 60.0     # Score >= threshold produces "high" category, else "low"

    # Cooldown & Storage
    debounce_frames: int = 15             # Minimum frames between flags for the same student
    snapshot_base_dir: str = "perception/output/snapshots"


class MovementFlaggingEngine:
    """Rule engine that evaluates student movements and captures violation snapshots."""

    def __init__(self, config: Optional[MovementFlaggingConfig] = None):
        self.config = config or MovementFlaggingConfig()

        # Upright baseline keypoints per student: student_id -> {keypoint_name: {x, y, conf}}
        self.student_baselines: Dict[str, Dict[str, Dict[str, float]]] = {}

        # Frame cooldown tracker: (student_id, event_type) -> last_triggered_frame
        self.last_triggered_frame: Dict[Tuple[str, str], int] = {}

        # Ensure snapshot destination folders exist
        self.low_risk_dir = os.path.join(self.config.snapshot_base_dir, "low_risk")
        self.high_risk_dir = os.path.join(self.config.snapshot_base_dir, "high_risk")
        os.makedirs(self.low_risk_dir, exist_ok=True)
        os.makedirs(self.high_risk_dir, exist_ok=True)

    def reset(self) -> None:
        """Clear baselines and cooldown state."""
        self.student_baselines.clear()
        self.last_triggered_frame.clear()

    def update_student_baseline(self, student_id: str, kpts_map: Dict[str, Dict[str, float]]) -> None:
        """Update running upright baseline for a student when in normal seated posture."""
        nose = kpts_map.get("nose")
        l_sh = kpts_map.get("left_shoulder")
        r_sh = kpts_map.get("right_shoulder")

        if nose and l_sh and r_sh and nose["confidence"] > 0.4:
            sh_y = (l_sh["y"] + r_sh["y"]) / 2.0
            # Only record baseline when nose is clearly above shoulders (upright posture)
            if nose["y"] < sh_y:
                if student_id not in self.student_baselines:
                    self.student_baselines[student_id] = kpts_map
                else:
                    # Exponential smoothing of upright baseline nose position
                    prev_y = self.student_baselines[student_id]["nose"]["y"]
                    # If current nose is higher (smaller y), update upright reference
                    if nose["y"] < prev_y:
                        self.student_baselines[student_id] = kpts_map

    def evaluate_track(
        self,
        track_dict: Dict[str, Any],
        frame_number: int,
    ) -> Optional[Dict[str, Any]]:
        """
        Evaluate a single student track for physical movement anomalies.

        Returns:
            Dict with violation details if a flag is triggered, otherwise None.
        """
        student_id = track_dict.get("student_id", "student_unknown")
        bbox = track_dict.get("bbox", [0.0, 0.0, 0.0, 0.0])
        raw_kpts = track_dict.get("keypoints", [])
        track_conf = track_dict.get("confidence", 0.8)

        kpts_map = extract_keypoints_map(raw_kpts)
        self.update_student_baseline(student_id, kpts_map)
        baseline = self.student_baselines.get(student_id)

        # 1. Compute Features
        body_feats = compute_body_movement_features(kpts_map, baseline_kpts=baseline)
        hand_feats = compute_hand_movement_features(kpts_map, bbox=bbox)

        nose_drop = body_feats["vertical_nose_drop"]
        tilt_deg = body_feats["shoulder_tilt_deg"]
        wrist_below_hip = hand_feats["max_wrist_below_hip"]
        wrist_lateral = hand_feats["max_wrist_reach_lateral"]

        event_type: Optional[str] = None
        reasons: List[str] = []
        severity: float = 0.0

        # 2. Check Hand Out of Bounds Rule
        if wrist_below_hip >= self.config.hand_below_hip_px or wrist_lateral >= self.config.hand_lateral_reach_px:
            event_type = "hand_out_of_bounds"
            if wrist_below_hip >= self.config.hand_below_hip_px:
                reasons.append(f"wrist below hip ({wrist_below_hip:.1f}px >= {self.config.hand_below_hip_px}px)")
                severity += min(45.0, (wrist_below_hip / self.config.hand_below_hip_px) * 35.0)
            if wrist_lateral >= self.config.hand_lateral_reach_px:
                reasons.append(f"lateral wrist extension ({wrist_lateral:.1f}px >= {self.config.hand_lateral_reach_px}px)")
                severity += min(35.0, (wrist_lateral / self.config.hand_lateral_reach_px) * 30.0)

        # 3. Check Body Movement Rule (Bending / Deep Inclination)
        #
        # Sub-rule A: nose_drop alone meets the bend threshold → clear bending forward.
        # Sub-rule B: tilt meets threshold AND nose has dropped enough to rule out a
        #             static angled seated posture (body_tilt_min_nose_drop_px co-condition).
        #             When body_tilt_min_nose_drop_px == 0.0, reverts to pure OR logic.
        else:
            tilt_with_drop = (
                tilt_deg >= self.config.body_tilt_deg
                and nose_drop >= self.config.body_tilt_min_nose_drop_px
            )
            body_flag = nose_drop >= self.config.body_bend_drop_px or tilt_with_drop

            if body_flag:
                event_type = "body_movement"
                if nose_drop >= self.config.body_bend_drop_px:
                    reasons.append(f"vertical head drop ({nose_drop:.1f}px >= {self.config.body_bend_drop_px}px)")
                    severity += min(55.0, (nose_drop / self.config.body_bend_drop_px) * 40.0)
                if tilt_with_drop:
                    reasons.append(
                        f"shoulder tilt ({tilt_deg:.1f}° >= {self.config.body_tilt_deg}°) "
                        f"+ nose drop ({nose_drop:.1f}px >= {self.config.body_tilt_min_nose_drop_px}px)"
                    )
                    severity += min(35.0, (tilt_deg / self.config.body_tilt_deg) * 30.0)


        if event_type is None:
            return None

        # Check frame debounce cooldown
        last_frame = self.last_triggered_frame.get((student_id, event_type), -999)
        if frame_number - last_frame < self.config.debounce_frames:
            return None

        self.last_triggered_frame[(student_id, event_type)] = frame_number

        # Calculate final numeric risk score (0.0 to 100.0)
        base_confidence_factor = max(0.5, min(1.0, track_conf))
        calculated_risk_score = round(min(100.0, max(15.0, (25.0 + severity) * base_confidence_factor)), 1)
        risk_category = "high" if calculated_risk_score >= self.config.high_risk_threshold else "low"

        return {
            "student_id": student_id,
            "event_type": event_type,
            "confidence": round(float(track_conf), 2),
            "risk_score": calculated_risk_score,
            "risk_category": risk_category,
            "bounding_box": bbox,  # [x1, y1, x2, y2]
            "reason": " | ".join(reasons),
            "metrics": {
                "vertical_nose_drop": nose_drop,
                "shoulder_tilt_deg": tilt_deg,
                "max_wrist_below_hip": wrist_below_hip,
                "max_wrist_reach_lateral": wrist_lateral,
            },
        }

    def render_and_save_snapshot(
        self,
        frame: np.ndarray,
        flagged_event: Dict[str, Any],
        frame_number: int,
        test_id: str = "test_001",
        timestamp: Optional[str] = None,
        all_tracks: Optional[List[Dict[str, Any]]] = None,
    ) -> Tuple[np.ndarray, str]:
        """
        Draw distinct alert box on the violating student (RED/AMBER) and normal boxes on all
        other detected students (GREEN), then save snapshot to appropriate risk folder.

        Returns:
            Tuple of (annotated_snapshot_frame, saved_snapshot_filepath)
        """
        if timestamp is None:
            timestamp = datetime.now(timezone.utc).isoformat()

        snapshot_img = frame.copy()
        h, w = snapshot_img.shape[:2]

        bbox = flagged_event["bounding_box"]  # [x1, y1, x2, y2]
        event_type = flagged_event["event_type"]
        risk_score = flagged_event["risk_score"]
        risk_cat = flagged_event["risk_category"]
        reason = flagged_event.get("reason", "")

        x1, y1, x2, y2 = map(int, bbox)
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w - 1, x2), min(h - 1, y2)

        # 1. Draw Green Bounding Boxes on all other detected students
        if all_tracks:
            for tr in all_tracks:
                tb = tr.get("bbox")
                if not tb or len(tb) != 4:
                    continue
                tx1, ty1, tx2, ty2 = map(int, tb)
                tx1, ty1 = max(0, tx1), max(0, ty1)
                tx2, ty2 = min(w - 1, tx2), min(h - 1, ty2)

                # Skip if this is the triggering bounding box (IoU / overlap check)
                inter_x1 = max(x1, tx1)
                inter_y1 = max(y1, ty1)
                inter_x2 = min(x2, tx2)
                inter_y2 = min(y2, ty2)
                inter_w = max(0, inter_x2 - inter_x1)
                inter_h = max(0, inter_y2 - inter_y1)
                inter_area = inter_w * inter_h
                box_area = (x2 - x1) * (y2 - y1)
                if box_area > 0 and (inter_area / box_area) > 0.65:
                    continue  # This is the triggering student

                # Draw subtle green box on non-violating student
                cv2.rectangle(snapshot_img, (tx1, ty1), (tx2, ty2), (0, 200, 0), 2)
                tag_g = "STUDENT"
                (gtw, gth), _ = cv2.getTextSize(tag_g, cv2.FONT_HERSHEY_SIMPLEX, 0.40, 1)
                cv2.rectangle(snapshot_img, (tx1, max(0, ty1 - gth - 6)), (tx1 + gtw + 6, ty1), (0, 200, 0), -1)
                cv2.putText(snapshot_img, tag_g, (tx1 + 3, max(gth + 1, ty1 - 3)), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (0, 0, 0), 1, cv2.LINE_AA)

        # 2. Draw Visibly Distinct Alert Box (Bright Crimson / Ruby Red) on Trigger
        alert_color = (0, 0, 255) if risk_cat == "high" else (0, 140, 255)  # Red for high, Amber for low
        # Double rectangle border for visual prominence
        cv2.rectangle(snapshot_img, (x1, y1), (x2, y2), alert_color, 3)
        cv2.rectangle(snapshot_img, (x1 - 2, y1 - 2), (x2 + 2, y2 + 2), (255, 255, 255), 1)

        # Alert Banner Tag
        tag_text = f"VIOLATION: {event_type.upper()} | Risk: {risk_score:.0f} ({risk_cat.upper()})"
        (tw, th), baseline = cv2.getTextSize(tag_text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
        cv2.rectangle(snapshot_img, (x1, max(0, y1 - th - 12)), (x1 + tw + 12, y1), alert_color, -1)
        cv2.putText(
            snapshot_img,
            tag_text,
            (x1 + 6, max(th + 2, y1 - 6)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

        # Top Overlay Stamp
        stamp_h = 45
        overlay = snapshot_img.copy()
        cv2.rectangle(overlay, (0, 0), (w, stamp_h), (15, 15, 15), -1)
        cv2.addWeighted(overlay, 0.85, snapshot_img, 0.15, 0, snapshot_img)
        
        banner_info = f"TEST: {test_id} | FRAME: {frame_number} | TIME: {timestamp} | FLAG: {event_type} | {reason}"
        cv2.putText(snapshot_img, banner_info, (15, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1, cv2.LINE_AA)

        # Determine target folder (high_risk vs low_risk)
        target_dir = self.high_risk_dir if risk_cat == "high" else self.low_risk_dir
        clean_time = timestamp.replace(":", "-").replace(".", "_")
        filename = f"snapshot_{clean_time}_{event_type}_f{frame_number}.jpg"
        filepath = os.path.join(target_dir, filename)

        # Save snapshot
        cv2.imwrite(filepath, snapshot_img)

        return snapshot_img, filepath

