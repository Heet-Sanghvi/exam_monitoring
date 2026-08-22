"""
Exam Monitoring System: Perception Pipeline Orchestrator
Main entry point for CV pipeline execution and real-time event dispatch.

Sequences:
1. Video Input (Camera / MKV / MP4)
2. Person Detection + Short-term Association + 17 COCO Keypoints (PoseEstimator)
3. Physical Movement Anomaly Engine (MovementFlaggingEngine: hand_out_of_bounds, body_movement)
4. Unauthorized Object Detection (ObjectDetector: phone_detected, chit_detected)
5. Multi-Student Snapshot Rendering (Red alert box on trigger, Green boxes on other students)
6. Multipart Event Dispatch to Backend API (POST /behavior-events with JSON + JPEG)
"""

import argparse
import json
import os
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple, Union

import cv2
import numpy as np
import torch

# Ensure perception directory is on python path
PERCEPTION_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PERCEPTION_DIR not in sys.path:
    sys.path.insert(0, PERCEPTION_DIR)

from src.behavior_rules import MovementFlaggingConfig, MovementFlaggingEngine
from src.object_detect import ObjectDetectionConfig, ObjectDetector
from src.pose import PoseEstimator
from src.send_events import EventDispatcher, EventDispatcherConfig


@dataclass
class PipelineConfig:
    """Master configuration for the Exam Monitoring Perception Pipeline."""

    # Backend API configuration
    backend_url: str = field(default_factory=lambda: os.environ.get("EXAM_BACKEND_URL", "http://127.0.0.1:8000"))
    active_test_endpoint: str = "/tests/active"
    behavior_events_endpoint: str = "/behavior-events"
    default_test_id: str = "test_default"
    connect_timeout_seconds: float = 3.0
    request_timeout_seconds: float = field(default_factory=lambda: float(os.environ.get("EXAM_REQUEST_TIMEOUT", "6.0")))

    # Device & Models
    device: Optional[str] = None
    pose_conf_threshold: float = 0.30
    phone_conf_threshold: float = 0.35

    # Storage & Cooldown
    snapshot_base_dir: str = "perception/output/snapshots"
    debounce_frames: int = 15

    # Sub-module configs
    movement_config: Optional[MovementFlaggingConfig] = None
    object_config: Optional[ObjectDetectionConfig] = None
    dispatcher_config: Optional[EventDispatcherConfig] = None


class PipelineOrchestrator:
    """
    Orchestrates the entire CV perception pipeline and dispatches behavior
    events to the backend API.
    """

    def __init__(self, config: Optional[PipelineConfig] = None):
        self.config = config or PipelineConfig()

        # Resolve compute device
        if self.config.device is not None:
            self.device = self.config.device
        elif torch.backends.mps.is_available():
            self.device = "mps"
        elif torch.cuda.is_available():
            self.device = "cuda"
        else:
            self.device = "cpu"

        # Initialize sub-module configs
        m_cfg = self.config.movement_config or MovementFlaggingConfig(
            debounce_frames=self.config.debounce_frames,
            snapshot_base_dir=self.config.snapshot_base_dir,
        )
        o_cfg = self.config.object_config or ObjectDetectionConfig(
            phone_conf_threshold=self.config.phone_conf_threshold,
            debounce_frames=self.config.debounce_frames,
            device=self.device,
        )
        d_cfg = self.config.dispatcher_config or EventDispatcherConfig(
            base_url=self.config.backend_url,
            active_test_endpoint=self.config.active_test_endpoint,
            behavior_events_endpoint=self.config.behavior_events_endpoint,
            default_test_id=self.config.default_test_id,
            connect_timeout_seconds=self.config.connect_timeout_seconds,
            request_timeout_seconds=self.config.request_timeout_seconds,
        )

        # Initialize pipeline modules
        self.pose_estimator = PoseEstimator(
            device=self.device,
            conf_threshold=self.config.pose_conf_threshold,
        )
        self.movement_flagger = MovementFlaggingEngine(config=m_cfg)
        self.object_detector = ObjectDetector(config=o_cfg)
        self.event_dispatcher = EventDispatcher(config=d_cfg)

        self.test_id: str = self.config.default_test_id
        self.total_frames_processed: int = 0
        self.total_events_dispatched: int = 0

    def initialize_backend_session(self) -> str:
        """
        Discover active test session from backend API on pipeline startup.
        Executes pre-flight health check, followed by session discovery.
        Handles active session, 404 (no active session), and backend-offline gracefully.
        """
        print(f"[Pipeline] Pre-flight connectivity check to: {self.config.backend_url} ...")
        reachable, msg = self.event_dispatcher.check_backend_connectivity()

        if reachable:
            print(f"[Pipeline] ✅ {msg}")
            self.test_id = self.event_dispatcher.fetch_active_test_id()
            if self.event_dispatcher._online and self.test_id != self.config.default_test_id:
                print(f"[Pipeline] ✅ Active exam session discovered: '{self.test_id}'")
            else:
                print(f"[Pipeline] ℹ️  Backend reachable, but no active session found. Operating with session ID: '{self.test_id}'")
        else:
            self.test_id = self.config.default_test_id
            print(f"[Pipeline] ⚠️  {msg}")
            print(f"[Pipeline]    Action items if unexpected:")
            print(f"      1. Verify Heet's server is running (e.g. uvicorn app.main:app --host 0.0.0.0 --port 8000)")
            print(f"      2. Verify both laptops are on the same Wi-Fi / LAN hotspot and the IP address is correct.")
            print(f"      3. Check firewall settings allowing inbound connections on port 8000.")
            print(f"[Pipeline] -> Continuing in OFFLINE mode (snapshots & events saved locally to {self.config.snapshot_base_dir}).")

        return self.test_id

    def process_frame(
        self,
        frame: np.ndarray,
        frame_number: int,
        timestamp: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Execute full perception pipeline on a single video frame.

        Sequence:
        1. Pose Estimation + Track association (PoseEstimator)
        2. Behavior / Movement rule checking (MovementFlaggingEngine)
        3. Object detection (ObjectDetector: phone & chit)
        4. Multi-student Snapshot Rendering & Event Dispatch (EventDispatcher)

        Returns:
            Dict containing frame analytics, detected events, and latency.
        """
        if timestamp is None:
            timestamp = datetime.now(timezone.utc).isoformat()

        t0 = time.perf_counter()
        flagged_events: List[Dict[str, Any]] = []

        # 1. Pose Estimation & Tracking
        pose_output = self.pose_estimator.process_frame(frame, frame_number=frame_number)
        tracks = pose_output.get("tracks", [])

        # 2. Movement / Behavioral Rules (hand_out_of_bounds, body_movement)
        for tr in tracks:
            track_dict = {
                "student_id": tr.get("student_id", "student_unknown"),
                "bbox": tr.get("bbox", [0.0, 0.0, 0.0, 0.0]),
                "keypoints": tr.get("keypoints", []),
                "confidence": tr.get("confidence", 0.85),
            }
            violation = self.movement_flagger.evaluate_track(track_dict, frame_number=frame_number)
            if violation:
                flagged_events.append(violation)

        # 3. Object Detection (phone_detected, chit_detected)
        object_events = self.object_detector.process_frame(
            frame=frame,
            frame_number=frame_number,
            tracks=tracks,
            test_id=self.test_id,
            timestamp=timestamp,
        )
        for obj_ev in object_events:
            # Map object detector output for snapshot rendering
            # Object detector bounding box is [x, y, w, h] -> convert to [x1, y1, x2, y2] for renderer
            bx, by, bw, bh = obj_ev["bounding_box"]
            flagged_obj = {
                "event_type": obj_ev["event_type"],
                "confidence": obj_ev["confidence"],
                "risk_score": obj_ev["risk_score"],
                "risk_category": obj_ev["risk_category"],
                "bounding_box": [bx, by, bx + bw, by + bh],
                "reason": obj_ev["metadata"].get("reason", obj_ev["event_type"]),
                "metrics": obj_ev["metadata"].get("metrics", {}),
            }
            flagged_events.append(flagged_obj)

        # 4. Snapshot Rendering & Dispatch for all flagged events
        dispatched_count = 0
        for event in flagged_events:
            # Render snapshot with RED alert on trigger and GREEN on all other detected students
            annotated_snap, snap_path = self.movement_flagger.render_and_save_snapshot(
                frame=frame,
                flagged_event=event,
                frame_number=frame_number,
                test_id=self.test_id,
                timestamp=timestamp,
                all_tracks=tracks,
            )

            # Dispatch multipart to backend API
            success = self.event_dispatcher.dispatch(
                flagged_event=event,
                snapshot_path=snap_path,
                frame_number=frame_number,
                timestamp=timestamp,
            )
            if success:
                dispatched_count += 1
                self.total_events_dispatched += 1

        self.total_frames_processed += 1
        elapsed_ms = (time.perf_counter() - t0) * 1000.0

        return {
            "frame_number": frame_number,
            "timestamp": timestamp,
            "tracks_count": len(tracks),
            "flagged_events_count": len(flagged_events),
            "dispatched_count": dispatched_count,
            "flagged_events": flagged_events,
            "tracks": tracks,
            "latency_ms": elapsed_ms,
        }

    def process_video(
        self,
        video_source: Union[str, int],
        max_frames: Optional[int] = None,
        output_video_path: Optional[str] = None,
        show_preview: bool = False,
    ) -> Dict[str, Any]:
        """
        Run the end-to-end perception pipeline on a video file or live camera stream.

        Args:
            video_source: Video file path or webcam integer index.
            max_frames: Optional limit on frames to process.
            output_video_path: Optional path to save annotated visualization MP4.
            show_preview: Whether to display OpenCV GUI window.

        Returns:
            Dict of execution analytics and violation summary.
        """
        self.initialize_backend_session()

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

        frame_count = 0
        total_latency_ms = 0.0
        fps_buffer: List[float] = []
        overall_start = time.perf_counter()
        events_summary: Dict[str, int] = {}

        print(f"\n{'='*70}")
        print(f"EXAM MONITORING PIPELINE RUNNING")
        print(f"Source: {video_source} ({width}x{height} @ {source_fps:.1f} FPS)")
        print(f"Device: {self.device.upper()} | Backend URL: {self.config.backend_url}")
        print(f"Active Test Session ID: {self.test_id}")
        print(f"{'='*70}\n")

        try:
            while True:
                if max_frames and frame_count >= max_frames:
                    break

                t0 = time.perf_counter()
                ret, frame = cap.read()
                if not ret:
                    break

                result = self.process_frame(frame, frame_number=frame_count)
                latency_ms = result["latency_ms"]
                total_latency_ms += latency_ms

                loop_time = time.perf_counter() - t0
                fps_val = 1.0 / loop_time if loop_time > 0 else 0.0
                fps_buffer.append(fps_val)
                if len(fps_buffer) > 30:
                    fps_buffer.pop(0)
                rolling_fps = float(np.mean(fps_buffer))

                for ev in result["flagged_events"]:
                    etype = ev["event_type"]
                    events_summary[etype] = events_summary.get(etype, 0) + 1
                    print(
                        f"  [FRAME {frame_count:5d}] FLAG: {etype:22s} | "
                        f"risk={ev['risk_score']:4.1f} ({ev['risk_category']:4s}) | "
                        f"latency={latency_ms:.1f}ms"
                    )

                # Render display frame
                if video_writer is not None or show_preview:
                    annotated = self.pose_estimator.draw_pose_frame(
                        frame,
                        {"tracks": result["tracks"], "metadata": {"inference_time_ms": latency_ms}},
                        fps=rolling_fps,
                    )
                    # Overlay active session info
                    cv2.putText(
                        annotated,
                        f"TEST: {self.test_id} | FLAGS: {sum(events_summary.values())}",
                        (20, height - 20),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.55,
                        (0, 255, 255),
                        2,
                        cv2.LINE_AA,
                    )

                    if video_writer is not None:
                        video_writer.write(annotated)

                    if show_preview:
                        cv2.imshow("Exam Monitoring - Perception Pipeline", annotated)
                        if cv2.waitKey(1) & 0xFF == ord("q"):
                            break

                frame_count += 1
                if frame_count % 150 == 0:
                    print(
                        f"Progress: Frame {frame_count:4d} | Students: {result['tracks_count']} | "
                        f"Speed: {rolling_fps:.1f} FPS | Total Flags: {sum(events_summary.values())}"
                    )

        finally:
            cap.release()
            if video_writer is not None:
                video_writer.release()
            if show_preview:
                cv2.destroyAllWindows()

        total_elapsed = time.perf_counter() - overall_start
        avg_fps = frame_count / total_elapsed if total_elapsed > 0 else 0.0
        avg_latency = total_latency_ms / frame_count if frame_count > 0 else 0.0

        print(f"\n{'='*70}")
        print(f"PIPELINE RUN SUMMARY")
        print(f"{'='*70}")
        print(f"  Total Frames Processed : {frame_count}")
        print(f"  Elapsed Time           : {total_elapsed:.1f}s ({avg_fps:.1f} FPS)")
        print(f"  Average Latency        : {avg_latency:.1f} ms/frame")
        print(f"  Total Flagged Events   : {sum(events_summary.values())}")
        for etype, count in events_summary.items():
            print(f"    - {etype:22s}: {count}")
        print(f"  Total Dispatched HTTP  : {self.total_events_dispatched}")
        print(f"{'='*70}\n")

        return {
            "total_frames": frame_count,
            "total_time_s": total_elapsed,
            "avg_fps": avg_fps,
            "avg_latency_ms": avg_latency,
            "events_summary": events_summary,
            "total_dispatched": self.total_events_dispatched,
            "device": self.device,
            "test_id": self.test_id,
        }


def main():
    default_backend = os.environ.get("EXAM_BACKEND_URL", "http://127.0.0.1:8000")
    default_timeout = float(os.environ.get("EXAM_REQUEST_TIMEOUT", "6.0"))

    parser = argparse.ArgumentParser(description="Exam Monitoring - Perception Pipeline Orchestrator")
    parser.add_argument("--source", type=str, default="0", help="Video file path or camera index (default: 0)")
    parser.add_argument("--backend-url", type=str, default=default_backend, help=f"Backend API base URL (default: {default_backend})")
    parser.add_argument("--timeout", type=float, default=default_timeout, help=f"HTTP request timeout in seconds (default: {default_timeout}s)")
    parser.add_argument("--device", type=str, default=None, help="Compute device ('mps', 'cuda', 'cpu')")
    parser.add_argument("--max-frames", type=int, default=None, help="Maximum frames to process")
    parser.add_argument("--output-video", type=str, default=None, help="Path to save annotated MP4 video")
    parser.add_argument("--preview", action="store_true", help="Display real-time OpenCV preview window")
    args = parser.parse_args()

    video_source = int(args.source) if args.source.isdigit() else args.source

    config = PipelineConfig(
        backend_url=args.backend_url,
        request_timeout_seconds=args.timeout,
        device=args.device,
    )
    orchestrator = PipelineOrchestrator(config=config)
    orchestrator.process_video(
        video_source=video_source,
        max_frames=args.max_frames,
        output_video_path=args.output_video,
        show_preview=args.preview,
    )


if __name__ == "__main__":
    main()
