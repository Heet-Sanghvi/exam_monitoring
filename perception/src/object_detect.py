"""
Object Detection Module: Cell Phone & Paper Chit Detection
Part of the Exam Monitoring Perception Pipeline.

Detects unauthorized items in the exam hall:
1. Phone Detection: Uses pretrained YOLOv8 (COCO class 67: 'cell phone') on Apple Silicon MPS/GPU.
2. Paper / Chit Detection: Wrist-anchored geometric and contrast heuristic detecting suspicious
   small paper slips near hands/desks.

Outputs events formatted to match shared/schema_behavior_event.json with:
- event_type: "phone_detected" | "chit_detected"
- bounding_box: [x, y, w, h] (top-left origin + dimensions)
- risk_score: 0.0 - 100.0
- risk_category: "low" | "high"
- No persistent student identity or seat_zone fields.
"""

import os
import math
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple, Union

import cv2
import numpy as np
import torch
from ultralytics import YOLO

from src.send_events import convert_bbox_xyxy_to_xywh


@dataclass
class ObjectDetectionConfig:
    """Configurable thresholds and parameters for object anomaly detection."""

    # Phone Detection Settings
    phone_model_path: str = "perception/models/yolov8n.pt"
    phone_conf_threshold: float = 0.35
    phone_coco_class_id: int = 67  # COCO class 67 is 'cell phone'
    phone_risk_base: float = 50.0
    phone_risk_multiplier: float = 50.0

    # Paper / Chit Detection Settings (Wrist-anchored geometric & contrast heuristic)
    chit_roi_radius_px: int = 50
    chit_min_area_px: float = 80.0
    chit_max_area_px: float = 1200.0  # Caps area to distinguish small chits from large A4 answer booklets (~3000-8000px^2)
    chit_min_solidity: float = 0.70
    chit_min_aspect_ratio: float = 0.2
    chit_max_aspect_ratio: float = 5.0
    chit_max_wrist_dist_px: float = 40.0
    chit_min_wrist_conf: float = 0.40
    chit_brightness_threshold: int = 175  # Luminance threshold for bright paper slip
    chit_risk_base: float = 40.0
    chit_risk_multiplier: float = 45.0

    # Risk Scoring and Cooldown
    high_risk_threshold: float = 60.0
    debounce_frames: int = 15
    device: Optional[str] = None


class ObjectDetector:
    """Detects phones and paper chits in video frames."""

    def __init__(self, config: Optional[ObjectDetectionConfig] = None):
        self.config = config or ObjectDetectionConfig()

        # Resolve compute device
        if self.config.device is not None:
            self.device = self.config.device
        elif torch.backends.mps.is_available():
            self.device = "mps"
        elif torch.cuda.is_available():
            self.device = "cuda"
        else:
            self.device = "cpu"

        # Load YOLO model
        if not os.path.isfile(self.config.phone_model_path):
            # Ultralytics will auto-download to path if needed
            os.makedirs(os.path.dirname(os.path.abspath(self.config.phone_model_path)), exist_ok=True)

        self.model = YOLO(self.config.phone_model_path)
        self.model.to(self.device)

        # Frame cooldown tracker: (event_type, region_key) -> last_triggered_frame
        self.last_triggered_frame: Dict[Tuple[str, str], int] = {}

    def reset(self) -> None:
        """Clear detection cooldown state."""
        self.last_triggered_frame.clear()

    def detect_phones(self, frame: np.ndarray) -> List[Dict[str, Any]]:
        """
        Run YOLOv8 inference to detect cell phones in the frame.

        Returns:
            List of raw phone detection dicts: [{"bbox_xyxy": [...], "confidence": float}]
        """
        results = self.model(
            frame,
            classes=[self.config.phone_coco_class_id],
            conf=self.config.phone_conf_threshold,
            device=self.device,
            verbose=False,
        )

        detections = []
        for r in results:
            boxes = r.boxes
            if boxes is None or len(boxes) == 0:
                continue
            for box in boxes:
                xyxy = box.xyxy[0].cpu().numpy().tolist()
                conf = float(box.conf[0].cpu().item())
                detections.append({
                    "bbox_xyxy": [round(v, 1) for v in xyxy],
                    "confidence": round(conf, 3),
                })
        return detections

    def detect_chits(
        self,
        frame: np.ndarray,
        tracks: Optional[List[Dict[str, Any]]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Evaluate wrists for small rectangular high-contrast paper chits.

        Args:
            frame: BGR video frame.
            tracks: Optional list of tracks with keypoints from PoseEstimator.

        Returns:
            List of raw chit detection dicts: [{"bbox_xyxy": [...], "confidence": float, "metrics": {...}}]
        """
        if not tracks:
            return []

        h, w = frame.shape[:2]
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        detections = []

        # Find all valid wrist keypoints
        wrist_points: List[Tuple[float, float, float]] = []
        for track in tracks:
            for kp in track.get("keypoints", []):
                name = kp.get("name")
                conf = kp.get("confidence", 0.0)
                if name in ("left_wrist", "right_wrist") and conf >= self.config.chit_min_wrist_conf:
                    wrist_points.append((float(kp["x"]), float(kp["y"]), float(conf)))

        if not wrist_points:
            return []

        # Analyze each wrist ROI
        radius = self.config.chit_roi_radius_px
        for wx, wy, w_conf in wrist_points:
            x1 = max(0, int(wx - radius))
            y1 = max(0, int(wy - radius))
            x2 = min(w, int(wx + radius))
            y2 = min(h, int(wy + radius))

            if x2 - x1 < 10 or y2 - y1 < 10:
                continue

            roi_gray = gray[y1:y2, x1:x2]

            # Contrast thresholding: paper slips are typically brighter than background desks
            # Binary threshold targeting high-contrast slips
            _, thresh = cv2.threshold(
                roi_gray,
                self.config.chit_brightness_threshold,
                255,
                cv2.THRESH_BINARY,
            )

            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            for cnt in contours:
                area = cv2.contourArea(cnt)
                if not (self.config.chit_min_area_px <= area <= self.config.chit_max_area_px):
                    continue

                # Bounding box of contour in ROI coordinates
                bx, by, bw, bh = cv2.boundingRect(cnt)
                if bw == 0 or bh == 0:
                    continue

                # Aspect ratio check
                aspect_ratio = float(bw) / float(bh)
                if not (self.config.chit_min_aspect_ratio <= aspect_ratio <= self.config.chit_max_aspect_ratio):
                    continue

                # Solidity check (area / bounding box area)
                solidity = area / float(bw * bh)
                if solidity < self.config.chit_min_solidity:
                    continue

                # Global coordinates
                gx1 = x1 + bx
                gy1 = y1 + by
                gx2 = gx1 + bw
                gy2 = gy1 + bh

                # Distance from contour centroid to wrist
                cx = gx1 + bw / 2.0
                cy = gy1 + bh / 2.0
                dist_to_wrist = math.hypot(cx - wx, cy - wy)

                if dist_to_wrist > self.config.chit_max_wrist_dist_px:
                    continue

                # Confidence heuristic: combines shape solidity and proximity
                proximity_factor = max(0.5, 1.0 - (dist_to_wrist / self.config.chit_max_wrist_dist_px) * 0.5)
                chit_conf = round(min(0.95, max(0.45, solidity * proximity_factor * w_conf)), 3)

                detections.append({
                    "bbox_xyxy": [float(gx1), float(gy1), float(gx2), float(gy2)],
                    "confidence": chit_conf,
                    "metrics": {
                        "area": round(area, 1),
                        "solidity": round(solidity, 2),
                        "aspect_ratio": round(aspect_ratio, 2),
                        "dist_to_wrist_px": round(dist_to_wrist, 1),
                    },
                })

        return detections

    def process_frame(
        self,
        frame: np.ndarray,
        frame_number: int,
        tracks: Optional[List[Dict[str, Any]]] = None,
        test_id: str = "test_001",
        timestamp: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Process a video frame, detect phones and chits, and format events conforming to shared schema.

        Args:
            frame: Video frame (BGR).
            frame_number: Current frame index.
            tracks: Optional tracked person keypoints from PoseEstimator.
            test_id: Test session identifier.
            timestamp: ISO 8601 string.

        Returns:
            List of schema-compliant event dicts.
        """
        if timestamp is None:
            timestamp = datetime.now(timezone.utc).isoformat()

        events: List[Dict[str, Any]] = []

        # 1. Detect Phones
        phone_dets = self.detect_phones(frame)
        for det in phone_dets:
            conf = det["confidence"]
            bbox_xyxy = det["bbox_xyxy"]

            # Spatial grid key for debounce
            region_key = f"{int(bbox_xyxy[0] // 80)}_{int(bbox_xyxy[1] // 80)}"
            last_f = self.last_triggered_frame.get(("phone_detected", region_key), -999)
            if frame_number - last_f < self.config.debounce_frames:
                continue

            self.last_triggered_frame[("phone_detected", region_key)] = frame_number

            # Calculate risk score
            risk_score = round(min(100.0, max(20.0, self.config.phone_risk_base + conf * self.config.phone_risk_multiplier)), 1)
            risk_cat = "high" if risk_score >= self.config.high_risk_threshold else "low"

            bbox_xywh = convert_bbox_xyxy_to_xywh(bbox_xyxy)
            event = {
                "event_id": str(uuid.uuid4()),
                "test_id": test_id,
                "frame_number": frame_number,
                "timestamp": timestamp,
                "event_type": "phone_detected",
                "confidence": conf,
                "risk_score": risk_score,
                "risk_category": risk_cat,
                "bounding_box": bbox_xywh,
                "metadata": {
                    "reason": f"cell phone detected with confidence {conf:.2f}",
                    "bbox_xyxy": bbox_xyxy,
                },
            }
            events.append(event)

        # 2. Detect Paper Chits
        chit_dets = self.detect_chits(frame, tracks=tracks)
        for det in chit_dets:
            conf = det["confidence"]
            bbox_xyxy = det["bbox_xyxy"]

            region_key = f"{int(bbox_xyxy[0] // 80)}_{int(bbox_xyxy[1] // 80)}"
            last_f = self.last_triggered_frame.get(("chit_detected", region_key), -999)
            if frame_number - last_f < self.config.debounce_frames:
                continue

            self.last_triggered_frame[("chit_detected", region_key)] = frame_number

            # Calculate risk score
            risk_score = round(min(100.0, max(20.0, self.config.chit_risk_base + conf * self.config.chit_risk_multiplier)), 1)
            risk_cat = "high" if risk_score >= self.config.high_risk_threshold else "low"

            bbox_xywh = convert_bbox_xyxy_to_xywh(bbox_xyxy)
            event = {
                "event_id": str(uuid.uuid4()),
                "test_id": test_id,
                "frame_number": frame_number,
                "timestamp": timestamp,
                "event_type": "chit_detected",
                "confidence": conf,
                "risk_score": risk_score,
                "risk_category": risk_cat,
                "bounding_box": bbox_xywh,
                "metadata": {
                    "reason": f"paper chit detected near wrist (area={det['metrics']['area']}px, solidity={det['metrics']['solidity']})",
                    "metrics": det["metrics"],
                    "bbox_xyxy": bbox_xyxy,
                },
            }
            events.append(event)

        return events

    def draw_detections(
        self,
        frame: np.ndarray,
        detected_events: List[Dict[str, Any]],
    ) -> np.ndarray:
        """
        Render visual bounding boxes and tags for detected objects on the frame.

        Args:
            frame: Video frame (BGR).
            detected_events: List of event dicts from process_frame().

        Returns:
            Annotated BGR frame.
        """
        annotated = frame.copy()
        h, w = annotated.shape[:2]

        for ev in detected_events:
            etype = ev["event_type"]
            risk_cat = ev["risk_category"]
            risk_score = ev["risk_score"]
            conf = ev["confidence"]

            # [x, y, w, h] -> [x1, y1, x2, y2]
            bx, by, bw, bh = ev["bounding_box"]
            x1, y1 = int(max(0, bx)), int(max(0, by))
            x2, y2 = int(min(w - 1, bx + bw)), int(min(h - 1, by + bh))

            color = (0, 0, 255) if risk_cat == "high" else (0, 165, 255)

            # Draw alert box
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)

            # Label tag
            tag_label = f"{etype.upper()} ({conf:.2f}) | Risk: {risk_score:.0f}"
            (tw, th), _ = cv2.getTextSize(tag_label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
            cv2.rectangle(annotated, (x1, max(0, y1 - th - 8)), (x1 + tw + 6, y1), color, -1)
            cv2.putText(
                annotated,
                tag_label,
                (x1 + 3, max(th + 2, y1 - 4)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )

        return annotated
