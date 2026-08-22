"""
Pose Estimation Module
Part of the Exam Monitoring Perception Pipeline.

Extracts 17 COCO keypoints per tracked student, calculates geometric head orientation,
and produces frame-level structured output conforming to shared/schema_pose_output.json.
"""

import os
import time
import json
import math
import argparse
from datetime import datetime, timezone
from typing import List, Dict, Any, Optional, Tuple, Union
import numpy as np
import cv2

from src.detect import DEFAULT_MODEL_PATH, get_optimal_device
from src.track import PersonTracker, generate_track_color

# 17 COCO Keypoint Standard Names
COCO_KEYPOINT_NAMES = [
    "nose",            # 0
    "left_eye",        # 1
    "right_eye",       # 2
    "left_ear",        # 3
    "right_ear",       # 4
    "left_shoulder",   # 5
    "right_shoulder",  # 6
    "left_elbow",      # 7
    "right_elbow",     # 8
    "left_wrist",      # 9
    "right_wrist",     # 10
    "left_hip",        # 11
    "right_hip",       # 12
    "left_knee",       # 13
    "right_knee",      # 14
    "left_ankle",      # 15
    "right_ankle",     # 16
]

# Skeleton limb connections (pairs of keypoint indices)
COCO_SKELETON = [
    (0, 1), (0, 2), (1, 3), (2, 4),           # Face
    (5, 6), (5, 7), (7, 9),                   # Left arm
    (6, 8), (8, 10),                          # Right arm
    (5, 11), (6, 12), (11, 12),               # Torso
    (11, 13), (13, 15),                       # Left leg
    (12, 14), (14, 16),                       # Right leg
]


def calculate_head_pose_angles(kpts_map: Dict[str, Dict[str, float]]) -> Optional[Dict[str, float]]:
    """
    Estimate coarse head orientation (yaw, pitch, roll in degrees) from 2D facial keypoints.

    Args:
        kpts_map: Dict mapping keypoint name to dict with 'x', 'y', 'confidence'.

    Returns:
        Dict with 'pitch', 'yaw', 'roll' in degrees, or None if insufficient keypoint visibility.
    """
    nose = kpts_map.get("nose")
    l_eye = kpts_map.get("left_eye")
    r_eye = kpts_map.get("right_eye")
    l_ear = kpts_map.get("left_ear")
    r_ear = kpts_map.get("right_ear")

    if not (nose and l_eye and r_eye):
        return None

    # Require minimum detection confidence
    if nose["confidence"] < 0.25 or l_eye["confidence"] < 0.25 or r_eye["confidence"] < 0.25:
        return None

    dx_eyes = r_eye["x"] - l_eye["x"]
    dy_eyes = r_eye["y"] - l_eye["y"]
    eye_dist = math.hypot(dx_eyes, dy_eyes)
    if eye_dist < 1e-4:
        return None

    # Roll: In-plane rotation angle of the eye line
    roll = math.degrees(math.atan2(dy_eyes, dx_eyes))

    # Eye midpoint
    mid_x = (l_eye["x"] + r_eye["x"]) / 2.0
    mid_y = (l_eye["y"] + r_eye["y"]) / 2.0

    # Yaw: Horizontal displacement of nose relative to eye midpoint
    # Normalized by eye distance (-1.0 = looking far left, +1.0 = looking far right)
    yaw_ratio = (nose["x"] - mid_x) / (eye_dist / 2.0)
    yaw = float(np.clip(yaw_ratio * 45.0, -90.0, 90.0))

    # Pitch: Vertical displacement of nose below eye midpoint
    # When head pitches down, nose moves downward relative to eyes
    pitch_ratio = (nose["y"] - mid_y) / (eye_dist / 2.0) - 0.7  # Baseline offset
    pitch = float(np.clip(pitch_ratio * 40.0, -80.0, 80.0))

    return {
        "pitch": round(pitch, 2),
        "yaw": round(yaw, 2),
        "roll": round(roll, 2),
    }


class PoseEstimator:
    """Extracts 17 COCO keypoints and head pose per tracked student."""

    def __init__(
        self,
        model_path: str = DEFAULT_MODEL_PATH,
        device: Optional[str] = None,
        conf_threshold: float = 0.30,
        tracker_type: str = "bytetrack.yaml",
    ):
        self.tracker = PersonTracker(
            model_path=model_path,
            device=device,
            conf_threshold=conf_threshold,
            tracker_type=tracker_type,
        )
        self.device = self.tracker.device

    def reset(self) -> None:
        """Reset internal tracker state."""
        self.tracker.reset()

    def process_frame(
        self,
        frame: np.ndarray,
        frame_number: int = 0,
        timestamp: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Run tracking and pose extraction for a single frame, formatting according to schema_pose_output.json.

        Args:
            frame: Input BGR frame.
            frame_number: Current frame index.
            timestamp: ISO 8601 timestamp string.

        Returns:
            Dict strictly matching shared/schema_pose_output.json schema.
        """
        if timestamp is None:
            timestamp = datetime.now(timezone.utc).isoformat()

        track_result = self.tracker.update(frame)
        tracks_data: List[Dict[str, Any]] = []

        for tr in track_result.get("tracks", []):
            raw_kpts = tr.get("keypoints")  # Shape (17, 3) [x, y, conf]
            keypoints_list: List[Dict[str, Any]] = []
            kpts_map: Dict[str, Dict[str, float]] = {}

            if raw_kpts is not None and len(raw_kpts) >= 17:
                for idx, name in enumerate(COCO_KEYPOINT_NAMES):
                    kx = float(raw_kpts[idx][0])
                    ky = float(raw_kpts[idx][1])
                    kc = float(raw_kpts[idx][2]) if len(raw_kpts[idx]) > 2 else 1.0
                    
                    kpt_entry = {
                        "name": name,
                        "x": round(kx, 2),
                        "y": round(ky, 2),
                        "confidence": round(kc, 4),
                    }
                    keypoints_list.append(kpt_entry)
                    kpts_map[name] = kpt_entry
            else:
                # Fill missing keypoints with 0 confidence
                for name in COCO_KEYPOINT_NAMES:
                    kpt_entry = {"name": name, "x": 0.0, "y": 0.0, "confidence": 0.0}
                    keypoints_list.append(kpt_entry)
                    kpts_map[name] = kpt_entry

            head_pose = calculate_head_pose_angles(kpts_map)

            track_entry: Dict[str, Any] = {
                "track_id": int(tr["track_id"]),
                "student_id": tr["student_id"],
                "bbox": [round(float(c), 2) for c in tr["bbox"]],
                "keypoints": keypoints_list,
            }
            if head_pose is not None:
                track_entry["head_pose"] = head_pose

            tracks_data.append(track_entry)

        return {
            "frame_number": int(frame_number),
            "timestamp": timestamp,
            "tracks": tracks_data,
            "metadata": {
                "active_count": len(tracks_data),
                "inference_time_ms": track_result.get("inference_time_ms", 0.0),
                "device": self.device,
            },
        }

    def draw_pose_frame(
        self,
        frame: np.ndarray,
        pose_output: Dict[str, Any],
        fps: Optional[float] = None,
        kpt_conf_thresh: float = 0.35,
    ) -> np.ndarray:
        """
        Draw skeleton limbs, keypoint joints, bounding boxes, student badges, and HUD.

        Args:
            frame: Input BGR frame.
            pose_output: Frame output dict from self.process_frame().
            fps: Current FPS value.
            kpt_conf_thresh: Minimum confidence to draw a keypoint/limb.

        Returns:
            Annotated BGR frame.
        """
        annotated = frame.copy()
        h, w = annotated.shape[:2]
        tracks = pose_output.get("tracks", [])
        meta = pose_output.get("metadata", {})
        inference_ms = meta.get("inference_time_ms", 0.0)

        for track in tracks:
            tid = track["track_id"]
            sid = track["student_id"]
            bbox = track["bbox"]
            kpts = track.get("keypoints", [])
            color = generate_track_color(tid)

            # Map keypoint name to dict
            kdict = {k["name"]: k for k in kpts}

            # 1. Draw Skeleton Limbs
            for i1, i2 in COCO_SKELETON:
                name1 = COCO_KEYPOINT_NAMES[i1]
                name2 = COCO_KEYPOINT_NAMES[i2]
                kp1, kp2 = kdict.get(name1), kdict.get(name2)

                if kp1 and kp2 and kp1["confidence"] >= kpt_conf_thresh and kp2["confidence"] >= kpt_conf_thresh:
                    pt1 = (int(kp1["x"]), int(kp1["y"]))
                    pt2 = (int(kp2["x"]), int(kp2["y"]))
                    if 0 <= pt1[0] < w and 0 <= pt1[1] < h and 0 <= pt2[0] < w and 0 <= pt2[1] < h:
                        cv2.line(annotated, pt1, pt2, color, 2, cv2.LINE_AA)

            # 2. Draw Keypoint Joints
            for kp in kpts:
                if kp["confidence"] >= kpt_conf_thresh:
                    kx, ky = int(kp["x"]), int(kp["y"])
                    if 0 <= kx < w and 0 <= ky < h:
                        # Highlight wrists and nose with distinct colors
                        if "wrist" in kp["name"]:
                            cv2.circle(annotated, (kx, ky), 5, (0, 0, 255), -1)  # Red for wrists/hands
                        elif kp["name"] == "nose":
                            cv2.circle(annotated, (kx, ky), 4, (0, 255, 255), -1)  # Yellow for nose
                        else:
                            cv2.circle(annotated, (kx, ky), 3, (0, 255, 128), -1)  # Green for others

            # 3. Draw Bounding Box & Student Badge
            x1, y1, x2, y2 = map(int, bbox)
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w - 1, x2), min(h - 1, y2)
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)

            # Badge with head pose info if available
            hp = track.get("head_pose")
            pose_str = f" | Y:{hp['yaw']:+.0f}° P:{hp['pitch']:+.0f}°" if hp else ""
            label = f"{sid}{pose_str}"

            (lw, lh), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
            cv2.rectangle(annotated, (x1, max(0, y1 - lh - 8)), (x1 + lw + 8, y1), color, -1)
            cv2.putText(
                annotated,
                label,
                (x1 + 4, max(lh + 2, y1 - 4)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (0, 0, 0),
                1,
                cv2.LINE_AA,
            )

        # 4. Draw Top-Left HUD
        hud_w, hud_h = 330, 115
        overlay = annotated.copy()
        cv2.rectangle(overlay, (10, 10), (10 + hud_w, 10 + hud_h), (20, 20, 20), -1)
        cv2.addWeighted(overlay, 0.75, annotated, 0.25, 0, annotated)
        cv2.rectangle(annotated, (10, 10), (10 + hud_w, 10 + hud_h), (120, 120, 120), 1)

        cv2.putText(annotated, "EXAM MONITORING: POSE ESTIMATION", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 215, 255), 1, cv2.LINE_AA)
        cv2.putText(annotated, f"Tracked Students: {len(tracks)} (17 Keypoints/Track)", (20, 52), cv2.FONT_HERSHEY_SIMPLEX, 0.46, (0, 255, 0), 1, cv2.LINE_AA)
        cv2.putText(annotated, f"Device: {self.device.upper()}  |  Latency: {inference_ms:.1f} ms", (20, 72), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (200, 200, 200), 1, cv2.LINE_AA)
        cv2.putText(annotated, "Skeleton: COCO 17 Keypoints + Head Pose", (20, 92), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (180, 180, 180), 1, cv2.LINE_AA)

        if fps is not None:
            cv2.putText(annotated, f"FPS: {fps:.1f}", (20, 112), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1, cv2.LINE_AA)

        return annotated

    def process_video(
        self,
        video_source: Union[str, int],
        output_video_path: Optional[str] = None,
        output_jsonl_path: Optional[str] = None,
        max_frames: Optional[int] = None,
        show_preview: bool = False,
    ) -> Dict[str, Any]:
        """
        Process a video, extract 17 keypoints per student per frame, log to jsonl, and save video.

        Args:
            video_source: Path to video file or webcam index.
            output_video_path: Path to write annotated MP4.
            output_jsonl_path: Path to write schema_pose_output.json lines.
            max_frames: Max frames to process.
            show_preview: Whether to display preview.

        Returns:
            Dict containing detailed pose estimation and keypoint analytics.
        """
        self.reset()
        cap = cv2.VideoCapture(video_source)
        if not cap.isOpened():
            raise ValueError(f"Could not open video source: {video_source}")

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        source_fps = cap.get(cv2.CAP_PROP_FPS)
        if source_fps <= 0 or np.isnan(source_fps):
            source_fps = 25.0

        video_writer = None
        if output_video_path:
            os.makedirs(os.path.dirname(os.path.abspath(output_video_path)), exist_ok=True)
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            video_writer = cv2.VideoWriter(output_video_path, fourcc, source_fps, (width, height))

        jsonl_file = None
        if output_jsonl_path:
            os.makedirs(os.path.dirname(os.path.abspath(output_jsonl_path)), exist_ok=True)
            jsonl_file = open(output_jsonl_path, "w", encoding="utf-8")

        frame_count = 0
        total_latency_ms = 0.0
        fps_buffer: List[float] = []
        overall_start = time.perf_counter()

        # Keypoint analytics collection
        kpt_confidences: Dict[str, List[float]] = {name: [] for name in COCO_KEYPOINT_NAMES}
        tracks_per_frame_history: List[int] = []

        print(f"Starting Pose Estimation on source: {video_source}")
        print(f"Frame resolution: {width}x{height} | Target device: {self.device}")

        try:
            while True:
                if max_frames and frame_count >= max_frames:
                    break

                t0 = time.perf_counter()
                ret, frame = cap.read()
                if not ret:
                    break

                pose_out = self.process_frame(frame, frame_number=frame_count)
                latency_ms = pose_out["metadata"]["inference_time_ms"]
                total_latency_ms += latency_ms

                loop_time = time.perf_counter() - t0
                current_fps = 1.0 / loop_time if loop_time > 0 else 0.0
                fps_buffer.append(current_fps)
                if len(fps_buffer) > 30:
                    fps_buffer.pop(0)
                rolling_fps = float(np.mean(fps_buffer))

                tracks_per_frame_history.append(len(pose_out["tracks"]))

                # Record keypoint confidence stats
                for tr in pose_out["tracks"]:
                    for kp in tr.get("keypoints", []):
                        kpt_confidences[kp["name"]].append(kp["confidence"])

                # Write JSONL log if enabled
                if jsonl_file is not None:
                    # Strip metadata helper for strict schema compliance
                    schema_compliant_obj = {
                        "frame_number": pose_out["frame_number"],
                        "timestamp": pose_out["timestamp"],
                        "tracks": pose_out["tracks"],
                    }
                    jsonl_file.write(json.dumps(schema_compliant_obj) + "\n")

                # Write video frame if enabled
                annotated = self.draw_pose_frame(frame, pose_out, fps=rolling_fps)
                if video_writer is not None:
                    video_writer.write(annotated)

                frame_count += 1

                if show_preview:
                    cv2.imshow("Exam Monitoring - Pose Estimation", annotated)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break

                if frame_count % 200 == 0:
                    print(
                        f"Frame {frame_count} | Active: {len(pose_out['tracks'])} | "
                        f"FPS: {rolling_fps:.1f} | Latency: {latency_ms:.1f}ms"
                    )

        finally:
            cap.release()
            if video_writer is not None:
                video_writer.release()
            if jsonl_file is not None:
                jsonl_file.close()
            if show_preview:
                cv2.destroyAllWindows()

        total_elapsed = time.perf_counter() - overall_start
        avg_fps = frame_count / total_elapsed if total_elapsed > 0 else 0.0
        avg_latency = total_latency_ms / frame_count if frame_count > 0 else 0.0

        # Summarize keypoint confidence stats by region
        kpt_summary = {}
        for name, vals in kpt_confidences.items():
            if vals:
                kpt_summary[name] = {
                    "mean_conf": round(float(np.mean(vals)), 3),
                    "median_conf": round(float(np.median(vals)), 3),
                    "pct_visible_gt_35": round(float(np.mean(np.array(vals) >= 0.35) * 100), 1),
                }

        return {
            "total_frames": frame_count,
            "total_time_s": total_elapsed,
            "avg_fps": avg_fps,
            "avg_latency_ms": avg_latency,
            "avg_tracks_per_frame": float(np.mean(tracks_per_frame_history)) if tracks_per_frame_history else 0.0,
            "keypoint_stats": kpt_summary,
            "device": self.device,
            "output_video_path": output_video_path,
            "output_jsonl_path": output_jsonl_path,
        }


def main():
    parser = argparse.ArgumentParser(description="Exam Monitoring - Pose Estimation Module")
    parser.add_argument("--source", type=str, default="0", help="Video file path or webcam index (default: 0)")
    parser.add_argument("--device", type=str, default=None, help="Device ('mps', 'cuda', 'cpu')")
    parser.add_argument("--conf", type=float, default=0.30, help="Confidence threshold (default: 0.30)")
    parser.add_argument("--output-video", type=str, default=None, help="Path to save annotated video")
    parser.add_argument("--output-jsonl", type=str, default=None, help="Path to save pose_output.jsonl")
    parser.add_argument("--max-frames", type=int, default=None, help="Maximum frames to process")
    parser.add_argument("--preview", action="store_true", help="Show GUI preview")
    args = parser.parse_args()

    video_source = int(args.source) if args.source.isdigit() else args.source

    estimator = PoseEstimator(device=args.device, conf_threshold=args.conf)
    estimator.process_video(
        video_source=video_source,
        output_video_path=args.output_video,
        output_jsonl_path=args.output_jsonl,
        max_frames=args.max_frames,
        show_preview=args.preview,
    )


if __name__ == "__main__":
    main()
