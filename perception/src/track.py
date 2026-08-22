"""
Multi-Person Tracking Module
Part of the Exam Monitoring Perception Pipeline.

Integrates ByteTrack via YOLOv8 Pose to maintain persistent student IDs (student_001, student_002, ...)
across frames, smoothing over brief seated occlusions, bending postures, and camera noise.
"""

import os
import time
import argparse
from typing import List, Dict, Any, Optional, Tuple, Union
import numpy as np
import cv2
import torch
from ultralytics import YOLO

from src.detect import DEFAULT_MODEL_PATH, get_optimal_device


def generate_track_color(track_id: int) -> Tuple[int, int, int]:
    """Generate a distinct, aesthetically pleasing BGR color for each student ID."""
    palette = [
        (255, 128, 0),    # Vivid Orange
        (0, 200, 255),    # Bright Amber/Yellow
        (50, 220, 50),    # Emerald Green
        (255, 50, 150),   # Magenta/Pink
        (0, 255, 255),    # Cyan
        (180, 105, 255),  # Lavender
        (255, 215, 0),    # Gold
        (0, 165, 255),    # Coral
        (144, 238, 144),  # Light Green
        (238, 130, 238),  # Violet
        (75, 0, 130),     # Indigo
        (255, 99, 71),    # Tomato
    ]
    return palette[track_id % len(palette)]


class PersonTracker:
    """Multi-person tracker with ByteTrack and persistent student ID assignment."""

    def __init__(
        self,
        model_path: str = DEFAULT_MODEL_PATH,
        device: Optional[str] = None,
        conf_threshold: float = 0.30,
        tracker_type: str = "bytetrack.yaml",
    ):
        self.device = get_optimal_device(device)
        self.conf_threshold = conf_threshold
        self.model_path = model_path
        self.tracker_type = tracker_type

        # Load YOLO model for tracking
        self.model = YOLO(self.model_path)

        # Student ID mapping: raw_track_id (int) -> student_id (str, e.g. "student_001")
        self.track_to_student: Dict[int, str] = {}
        self.student_counter: int = 0

        # Motion trail history: student_id -> list of center points (x, y)
        self.trail_history: Dict[str, List[Tuple[int, int]]] = {}
        self.max_trail_length: int = 20

    def reset(self) -> None:
        """Reset internal tracker state and student ID registry."""
        self.track_to_student.clear()
        self.student_counter = 0
        self.trail_history.clear()

    def get_or_create_student_id(self, raw_track_id: int) -> str:
        """Assign or retrieve persistent student_xxx identifier for a given track ID."""
        if raw_track_id not in self.track_to_student:
            self.student_counter += 1
            self.track_to_student[raw_track_id] = f"student_{self.student_counter:03d}"
        return self.track_to_student[raw_track_id]

    def update(self, frame: np.ndarray) -> Dict[str, Any]:
        """
        Process a single frame through YOLO Pose + ByteTrack.

        Args:
            frame: Input BGR frame.

        Returns:
            Dict containing:
                - tracks: List of active track objects (student_id, track_id, bbox, confidence, keypoints)
                - active_tracks_count: Number of currently tracked persons in frame
                - total_unique_students: Cumulative unique students registered
                - inference_time_ms: Combined inference & tracking latency in ms
                - device: Compute device
        """
        if frame is None or frame.size == 0:
            return {
                "tracks": [],
                "active_tracks_count": 0,
                "total_unique_students": len(self.track_to_student),
                "inference_time_ms": 0.0,
                "device": self.device,
            }

        start_time = time.perf_counter()

        # Run YOLO Tracking with ByteTrack
        results = self.model.track(
            source=frame,
            persist=True,
            tracker=self.tracker_type,
            device=self.device,
            conf=self.conf_threshold,
            classes=[0],  # Class 0 = person
            verbose=False,
        )

        inference_time_ms = (time.perf_counter() - start_time) * 1000.0

        tracks: List[Dict[str, Any]] = []

        if results and len(results) > 0:
            result = results[0]
            boxes = result.boxes
            keypoints_obj = result.keypoints

            if boxes is not None and len(boxes) > 0:
                xyxy = boxes.xyxy.cpu().numpy()
                confs = boxes.conf.cpu().numpy()
                cls_ids = boxes.cls.cpu().numpy().astype(int)

                # Extract track IDs from boxes
                raw_ids = boxes.id.cpu().numpy().astype(int) if boxes.id is not None else None

                # Extract keypoints
                kpts_data = None
                if keypoints_obj is not None and keypoints_obj.data is not None:
                    kpts_data = keypoints_obj.data.cpu().numpy()

                for i in range(len(boxes)):
                    raw_id = int(raw_ids[i]) if raw_ids is not None and i < len(raw_ids) else (i + 1)
                    student_id = self.get_or_create_student_id(raw_id)
                    bbox = [float(coord) for coord in xyxy[i]]

                    # Update center trail
                    cx = int((bbox[0] + bbox[2]) / 2)
                    cy = int((bbox[1] + bbox[3]) / 2)
                    if student_id not in self.trail_history:
                        self.trail_history[student_id] = []
                    self.trail_history[student_id].append((cx, cy))
                    if len(self.trail_history[student_id]) > self.max_trail_length:
                        self.trail_history[student_id].pop(0)

                    track_entry: Dict[str, Any] = {
                        "track_id": raw_id,
                        "student_id": student_id,
                        "bbox": bbox,
                        "confidence": float(confs[i]),
                        "class_id": int(cls_ids[i]),
                        "class_name": "person",
                        "keypoints": kpts_data[i] if kpts_data is not None and i < len(kpts_data) else None,
                    }
                    tracks.append(track_entry)

        return {
            "tracks": tracks,
            "active_tracks_count": len(tracks),
            "total_unique_students": len(self.track_to_student),
            "inference_time_ms": inference_time_ms,
            "device": self.device,
        }

    def draw_tracks(
        self,
        frame: np.ndarray,
        tracking_result: Dict[str, Any],
        fps: Optional[float] = None,
        draw_keypoints: bool = True,
        draw_trails: bool = True,
    ) -> np.ndarray:
        """
        Render tracking bounding boxes, persistent student ID badges, motion trails, and HUD.

        Args:
            frame: Input BGR frame.
            tracking_result: Output dict from self.update().
            fps: Current processing FPS.
            draw_keypoints: Render keypoints if available.
            draw_trails: Render center movement history trail.

        Returns:
            Annotated BGR frame.
        """
        annotated = frame.copy()
        h, w = annotated.shape[:2]
        tracks = tracking_result.get("tracks", [])
        active_count = tracking_result.get("active_tracks_count", len(tracks))
        total_unique = tracking_result.get("total_unique_students", len(self.track_to_student))
        inference_ms = tracking_result.get("inference_time_ms", 0.0)
        device_name = tracking_result.get("device", self.device).upper()

        # Draw motion trails
        if draw_trails:
            for track in tracks:
                sid = track["student_id"]
                tid = track["track_id"]
                color = generate_track_color(tid)
                pts = self.trail_history.get(sid, [])
                for idx in range(1, len(pts)):
                    alpha = idx / len(pts)
                    thickness = int(1 + alpha * 2)
                    cv2.line(annotated, pts[idx - 1], pts[idx], color, thickness)

        # Draw student bounding boxes & ID tags
        for track in tracks:
            bbox = track["bbox"]
            conf = track["confidence"]
            sid = track["student_id"]
            tid = track["track_id"]
            color = generate_track_color(tid)

            x1, y1, x2, y2 = map(int, bbox)
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w - 1, x2), min(h - 1, y2)

            # Draw bounding box
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)

            # Student Badge
            label = f"{sid} ({conf:.2f})"
            (lw, lh), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(
                annotated,
                (x1, max(0, y1 - lh - 8)),
                (x1 + lw + 8, y1),
                color,
                -1,
            )
            cv2.putText(
                annotated,
                label,
                (x1 + 4, max(lh + 2, y1 - 4)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 0, 0),
                1,
                cv2.LINE_AA,
            )

            # Draw keypoint dots
            kpts = track.get("keypoints")
            if draw_keypoints and kpts is not None:
                for kpt in kpts:
                    kx, ky = int(kpt[0]), int(kpt[1])
                    kconf = kpt[2] if len(kpt) > 2 else 1.0
                    if kconf > 0.4 and 0 <= kx < w and 0 <= ky < h:
                        cv2.circle(annotated, (kx, ky), 3, (0, 255, 200), -1)

        # Draw HUD Panel at Top-Left
        hud_w, hud_h = 320, 110
        overlay = annotated.copy()
        cv2.rectangle(overlay, (10, 10), (10 + hud_w, 10 + hud_h), (20, 20, 20), -1)
        cv2.addWeighted(overlay, 0.75, annotated, 0.25, 0, annotated)
        cv2.rectangle(annotated, (10, 10), (10 + hud_w, 10 + hud_h), (120, 120, 120), 1)

        # HUD Text elements
        cv2.putText(annotated, "EXAM MONITORING: MULTI-PERSON TRACKING", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 215, 255), 1, cv2.LINE_AA)
        cv2.putText(annotated, f"Active Tracked: {active_count}  |  Total Unique: {total_unique}", (20, 52), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 255, 0), 1, cv2.LINE_AA)
        cv2.putText(annotated, f"Tracker: ByteTrack  |  Device: {device_name}", (20, 72), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (200, 200, 200), 1, cv2.LINE_AA)
        cv2.putText(annotated, f"Inference Latency: {inference_ms:.1f} ms", (20, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (200, 200, 200), 1, cv2.LINE_AA)

        if fps is not None:
            cv2.putText(annotated, f"Pipeline FPS: {fps:.1f}", (20, 108), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1, cv2.LINE_AA)

        return annotated

    def process_video(
        self,
        video_source: Union[str, int],
        output_path: Optional[str] = None,
        max_frames: Optional[int] = None,
        show_preview: bool = False,
    ) -> Dict[str, Any]:
        """
        Process a video stream or file with tracking, compute continuity stats and save output.

        Args:
            video_source: File path or webcam index.
            output_path: Optional output video path.
            max_frames: Maximum frames to process.
            show_preview: Show preview window.

        Returns:
            Dict containing detailed tracking performance and continuity metrics.
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

        writer = None
        if output_path:
            os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(output_path, fourcc, source_fps, (width, height))

        frame_count = 0
        total_inference_ms = 0.0
        fps_buffer: List[float] = []
        overall_start = time.perf_counter()

        # Track history analysis
        student_frame_appearances: Dict[str, List[int]] = {}
        active_counts_history: List[int] = []

        print(f"Starting Multi-Person ByteTrack on source: {video_source}")
        print(f"Frame resolution: {width}x{height} | Target device: {self.device}")

        try:
            while True:
                if max_frames and frame_count >= max_frames:
                    break

                loop_start = time.perf_counter()
                ret, frame = cap.read()
                if not ret:
                    break

                track_result = self.update(frame)
                inference_ms = track_result["inference_time_ms"]
                total_inference_ms += inference_ms

                loop_time = time.perf_counter() - loop_start
                current_fps = 1.0 / loop_time if loop_time > 0 else 0.0
                fps_buffer.append(current_fps)
                if len(fps_buffer) > 30:
                    fps_buffer.pop(0)
                rolling_fps = float(np.mean(fps_buffer))

                active_count = track_result["active_tracks_count"]
                active_counts_history.append(active_count)

                # Record student frame appearances
                for tr in track_result["tracks"]:
                    sid = tr["student_id"]
                    if sid not in student_frame_appearances:
                        student_frame_appearances[sid] = []
                    student_frame_appearances[sid].append(frame_count)

                annotated = self.draw_tracks(frame, track_result, fps=rolling_fps)

                if writer is not None:
                    writer.write(annotated)

                frame_count += 1

                if show_preview:
                    cv2.imshow("Exam Monitoring - Multi-Person Tracking", annotated)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break

                if frame_count % 200 == 0:
                    print(
                        f"Frame {frame_count} | Active: {active_count} | Total Unique: {len(self.track_to_student)} | "
                        f"FPS: {rolling_fps:.1f} | Latency: {inference_ms:.1f}ms"
                    )

        finally:
            cap.release()
            if writer is not None:
                writer.release()
            if show_preview:
                cv2.destroyAllWindows()

        total_elapsed = time.perf_counter() - overall_start
        avg_fps = frame_count / total_elapsed if total_elapsed > 0 else 0.0
        avg_inference_ms = total_inference_ms / frame_count if frame_count > 0 else 0.0

        # Calculate student track continuity & lifespan metrics
        student_lifespans = {
            sid: {
                "frames_tracked": len(frames),
                "lifespan_pct": round((len(frames) / frame_count) * 100, 1),
                "first_frame": frames[0],
                "last_frame": frames[-1],
            }
            for sid, frames in student_frame_appearances.items()
        }

        # Count gaps/interruptions per student track
        track_gaps: Dict[str, int] = {}
        for sid, frames in student_frame_appearances.items():
            gaps = 0
            for k in range(1, len(frames)):
                if frames[k] - frames[k - 1] > 1:
                    gaps += 1
            track_gaps[sid] = gaps

        return {
            "total_frames": frame_count,
            "total_time_s": total_elapsed,
            "avg_fps": avg_fps,
            "avg_inference_ms": avg_inference_ms,
            "total_unique_students": len(self.track_to_student),
            "student_lifespans": student_lifespans,
            "track_gaps": track_gaps,
            "avg_active_tracks": float(np.mean(active_counts_history)) if active_counts_history else 0.0,
            "device": self.device,
            "output_path": output_path,
        }


def main():
    parser = argparse.ArgumentParser(description="Exam Monitoring - Multi-Person Tracking Module")
    parser.add_argument("--source", type=str, default="0", help="Video file path or webcam index (default: 0)")
    parser.add_argument("--device", type=str, default=None, help="Device to use ('mps', 'cuda', 'cpu')")
    parser.add_argument("--conf", type=float, default=0.30, help="Confidence threshold (default: 0.30)")
    parser.add_argument("--output", type=str, default=None, help="Path to save output annotated video")
    parser.add_argument("--max-frames", type=int, default=None, help="Maximum number of frames to process")
    parser.add_argument("--preview", action="store_true", help="Show GUI preview window")
    args = parser.parse_args()

    video_source = int(args.source) if args.source.isdigit() else args.source

    tracker = PersonTracker(device=args.device, conf_threshold=args.conf)
    tracker.process_video(
        video_source=video_source,
        output_path=args.output,
        max_frames=args.max_frames,
        show_preview=args.preview,
    )


if __name__ == "__main__":
    main()
