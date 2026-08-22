"""
Person Detection Module
Part of the Exam Monitoring Perception Pipeline.

Uses YOLOv8 Pose (yolov8n-pose.pt) for person detection and keypoint extraction,
optimized with Apple Silicon Metal Performance Shaders (MPS) device acceleration.
"""

import os
import time
import argparse
from typing import List, Dict, Any, Optional, Tuple, Union
import numpy as np
import cv2
import torch
from ultralytics import YOLO

# Default model path in local perception/models folder or root fallback
DEFAULT_MODEL_PATH = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "models", "yolov8n-pose.pt")
)
if not os.path.exists(DEFAULT_MODEL_PATH):
    # Fallback to root or default ultralytics cache
    DEFAULT_MODEL_PATH = "yolov8n-pose.pt"


def get_optimal_device(requested_device: Optional[str] = None) -> str:
    """Determine the best available compute device (MPS > CUDA > CPU)."""
    if requested_device:
        return requested_device.lower()
    
    if torch.backends.mps.is_available():
        return "mps"
    elif torch.cuda.is_available():
        return "cuda"
    return "cpu"


class PersonDetector:
    """Person detector and keypoint extractor powered by YOLOv8 Pose."""

    def __init__(
        self,
        model_path: str = DEFAULT_MODEL_PATH,
        device: Optional[str] = None,
        conf_threshold: float = 0.35,
    ):
        self.device = get_optimal_device(device)
        self.conf_threshold = conf_threshold
        self.model_path = model_path
        
        # Load YOLO model
        self.model = YOLO(self.model_path)

    def detect(self, frame: np.ndarray) -> Dict[str, Any]:
        """
        Run person detection and pose estimation on a single frame.

        Args:
            frame: BGR image numpy array (H, W, 3).

        Returns:
            Dict containing:
                - detections: List of detection dicts (bbox, confidence, keypoints, class_id)
                - person_count: Number of persons detected
                - inference_time_ms: Model prediction latency in milliseconds
                - device: Compute device used for inference
        """
        if frame is None or frame.size == 0:
            return {
                "detections": [],
                "person_count": 0,
                "inference_time_ms": 0.0,
                "device": self.device,
            }

        start_time = time.perf_counter()
        
        # Run YOLO inference
        results = self.model.predict(
            source=frame,
            device=self.device,
            conf=self.conf_threshold,
            classes=[0],  # Class 0 = person
            verbose=False,
        )
        
        inference_time_ms = (time.perf_counter() - start_time) * 1000.0

        detections: List[Dict[str, Any]] = []
        
        if results and len(results) > 0:
            result = results[0]
            boxes = result.boxes
            keypoints_obj = result.keypoints

            if boxes is not None and len(boxes) > 0:
                xyxy = boxes.xyxy.cpu().numpy()
                confs = boxes.conf.cpu().numpy()
                cls_ids = boxes.cls.cpu().numpy().astype(int)

                kpts_data = None
                if keypoints_obj is not None and keypoints_obj.data is not None:
                    kpts_data = keypoints_obj.data.cpu().numpy()  # Shape: (N, 17, 3) [x, y, conf]

                for i in range(len(boxes)):
                    detection_entry: Dict[str, Any] = {
                        "bbox": [float(coord) for coord in xyxy[i]],
                        "confidence": float(confs[i]),
                        "class_id": int(cls_ids[i]),
                        "class_name": "person",
                        "keypoints": kpts_data[i] if kpts_data is not None and i < len(kpts_data) else None,
                    }
                    detections.append(detection_entry)

        return {
            "detections": detections,
            "person_count": len(detections),
            "inference_time_ms": inference_time_ms,
            "device": self.device,
        }

    def draw_detections(
        self,
        frame: np.ndarray,
        detection_result: Dict[str, Any],
        fps: Optional[float] = None,
        draw_keypoints: bool = True,
    ) -> np.ndarray:
        """
        Render bounding boxes, confidence labels, and HUD overlays onto the frame.

        Args:
            frame: Input BGR frame.
            detection_result: Output dict from self.detect().
            fps: Optional current FPS value for HUD display.
            draw_keypoints: Whether to render pose keypoints and skeleton connections.

        Returns:
            Annotated BGR frame.
        """
        annotated = frame.copy()
        h, w = annotated.shape[:2]
        detections = detection_result.get("detections", [])
        person_count = detection_result.get("person_count", len(detections))
        inference_ms = detection_result.get("inference_time_ms", 0.0)
        device_name = detection_result.get("device", self.device).upper()

        # Draw individual person bounding boxes & labels
        for det in detections:
            bbox = det["bbox"]
            conf = det["confidence"]
            x1, y1, x2, y2 = map(int, bbox)

            # Clamp coordinates to frame boundaries
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w - 1, x2), min(h - 1, y2)

            # Bounding box color (Cyan/Blue for active student detection)
            box_color = (255, 180, 0)  # BGR
            cv2.rectangle(annotated, (x1, y1), (x2, y2), box_color, 2)

            # Label badge
            label = f"Person: {conf:.2f}"
            (lw, lh), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(
                annotated,
                (x1, max(0, y1 - lh - 8)),
                (x1 + lw + 8, y1),
                box_color,
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

            # Draw keypoints if available and requested
            kpts = det.get("keypoints")
            if draw_keypoints and kpts is not None:
                for kpt in kpts:
                    kx, ky = int(kpt[0]), int(kpt[1])
                    kconf = kpt[2] if len(kpt) > 2 else 1.0
                    if kconf > 0.4 and 0 <= kx < w and 0 <= ky < h:
                        cv2.circle(annotated, (kx, ky), 4, (0, 255, 128), -1)

        # Draw HUD Panel at Top-Left
        hud_w, hud_h = 280, 95
        overlay = annotated.copy()
        cv2.rectangle(overlay, (10, 10), (10 + hud_w, 10 + hud_h), (20, 20, 20), -1)
        cv2.addWeighted(overlay, 0.75, annotated, 0.25, 0, annotated)
        cv2.rectangle(annotated, (10, 10), (10 + hud_w, 10 + hud_h), (100, 100, 100), 1)

        # HUD Text items
        header_color = (0, 215, 255)  # Gold/Amber
        cv2.putText(annotated, "EXAM MONITORING: PERCEPTION", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.45, header_color, 1, cv2.LINE_AA)

        # Person Count with status indicator
        count_color = (0, 255, 0) if person_count == 1 else ((0, 165, 255) if person_count == 0 else (0, 0, 255))
        cv2.putText(annotated, f"Persons Detected: {person_count}", (20, 52), cv2.FONT_HERSHEY_SIMPLEX, 0.5, count_color, 1, cv2.LINE_AA)

        # Hardware Device & Latency
        cv2.putText(annotated, f"Device: {device_name} ({inference_ms:.1f} ms)", (20, 72), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (200, 200, 200), 1, cv2.LINE_AA)

        # FPS indicator
        if fps is not None:
            cv2.putText(annotated, f"FPS: {fps:.1f}", (20, 92), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1, cv2.LINE_AA)

        return annotated

    def process_video(
        self,
        video_source: Union[str, int],
        output_path: Optional[str] = None,
        max_frames: Optional[int] = None,
        show_preview: bool = False,
    ) -> Dict[str, Any]:
        """
        Process a video stream or file, detect persons, calculate FPS, and optionally save output.

        Args:
            video_source: File path (str) or webcam index (int).
            output_path: Optional file path to save annotated output video.
            max_frames: Optional maximum number of frames to process.
            show_preview: Whether to display a preview window (requires GUI support).

        Returns:
            Dict containing processing metrics (total_frames, avg_fps, avg_inference_ms, total_time_s).
        """
        cap = cv2.VideoCapture(video_source)
        if not cap.isOpened():
            raise ValueError(f"Could not open video source: {video_source}")

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        source_fps = cap.get(cv2.CAP_PROP_FPS)
        if source_fps <= 0 or np.isnan(source_fps):
            source_fps = 30.0

        writer = None
        if output_path:
            os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(output_path, fourcc, source_fps, (width, height))

        frame_count = 0
        total_inference_ms = 0.0
        fps_buffer: List[float] = []
        overall_start = time.perf_counter()

        print(f"Starting person detection on source: {video_source}")
        print(f"Frame resolution: {width}x{height} | Target device: {self.device}")

        try:
            while True:
                if max_frames and frame_count >= max_frames:
                    break

                loop_start = time.perf_counter()
                ret, frame = cap.read()
                if not ret:
                    break

                # Run detection
                det_result = self.detect(frame)
                inference_ms = det_result["inference_time_ms"]
                total_inference_ms += inference_ms

                loop_time = time.perf_counter() - loop_start
                current_fps = 1.0 / loop_time if loop_time > 0 else 0.0
                fps_buffer.append(current_fps)
                if len(fps_buffer) > 30:
                    fps_buffer.pop(0)
                rolling_fps = float(np.mean(fps_buffer))

                # Annotate frame
                annotated_frame = self.draw_detections(frame, det_result, fps=rolling_fps)

                if writer is not None:
                    writer.write(annotated_frame)

                frame_count += 1

                if show_preview:
                    cv2.imshow("Exam Monitoring - Person Detection", annotated_frame)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break

                if frame_count % 30 == 0:
                    print(f"Processed {frame_count} frames | Rolling FPS: {rolling_fps:.1f} | Latency: {inference_ms:.1f}ms")

        finally:
            cap.release()
            if writer is not None:
                writer.release()
            if show_preview:
                cv2.destroyAllWindows()

        total_elapsed = time.perf_counter() - overall_start
        avg_fps = frame_count / total_elapsed if total_elapsed > 0 else 0.0
        avg_inference_ms = total_inference_ms / frame_count if frame_count > 0 else 0.0

        print(f"\n--- Detection Processing Summary ---")
        print(f"Total Frames Processed: {frame_count}")
        print(f"Total Time: {total_elapsed:.2f} s")
        print(f"Average Pipeline FPS: {avg_fps:.2f} FPS")
        print(f"Average Model Latency: {avg_inference_ms:.2f} ms")
        print(f"Device: {self.device}")

        return {
            "total_frames": frame_count,
            "total_time_s": total_elapsed,
            "avg_fps": avg_fps,
            "avg_inference_ms": avg_inference_ms,
            "device": self.device,
            "output_path": output_path,
        }


def main():
    parser = argparse.ArgumentParser(description="Exam Monitoring - Person Detection Module")
    parser.add_argument("--source", type=str, default="0", help="Video file path or webcam index (default: 0)")
    parser.add_argument("--device", type=str, default=None, help="Device to use ('mps', 'cuda', 'cpu')")
    parser.add_argument("--conf", type=float, default=0.35, help="Confidence threshold (default: 0.35)")
    parser.add_argument("--output", type=str, default=None, help="Path to save output annotated video")
    parser.add_argument("--max-frames", type=int, default=None, help="Maximum number of frames to process")
    parser.add_argument("--preview", action="store_true", help="Show GUI preview window")
    args = parser.parse_args()

    # Parse source as int if webcam index
    video_source = int(args.source) if args.source.isdigit() else args.source

    detector = PersonDetector(device=args.device, conf_threshold=args.conf)
    detector.process_video(
        video_source=video_source,
        output_path=args.output,
        max_frames=args.max_frames,
        show_preview=args.preview,
    )


if __name__ == "__main__":
    main()
