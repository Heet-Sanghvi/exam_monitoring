"""
Evaluation script for Phase 6 (object_detect.py) on the real Seat 12 clip:
"Seat No. 12 was seen taking a piece of paper from the desk.mkv" (1,936 frames).

Runs PoseEstimator + ObjectDetector concurrently to measure:
1. Phone detection flags (should be 0 since this clip is a paper-taking incident).
2. Chit detection flags around the paper retrieval frames (frames ~20-70, ~370-400, ~600-640).
"""

import os
import sys
import time
from collections import defaultdict

import cv2

PERCEPTION_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PERCEPTION_DIR)

from src.object_detect import ObjectDetectionConfig, ObjectDetector
from src.pose import PoseEstimator

CLIP = os.path.join(PERCEPTION_DIR, "data", "test_clips", "suspicious", "Seat No. 12 was seen taking a piece of paper from the desk.mkv")

config = ObjectDetectionConfig(
    phone_conf_threshold=0.35,
    chit_roi_radius_px=60,
    chit_min_area_px=100.0,
    chit_max_area_px=6000.0,
    chit_min_solidity=0.65,
    chit_max_wrist_dist_px=45.0,
    debounce_frames=15,
)

estimator = PoseEstimator()
detector = ObjectDetector(config=config)

cap = cv2.VideoCapture(CLIP)
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
fps_src = cap.get(cv2.CAP_PROP_FPS)

print(f"Clip: {CLIP}")
print(f"Total frames: {total_frames}  Source FPS: {fps_src:.1f}")

events_by_type = defaultdict(list)
t_start = time.perf_counter()
frame_num = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame_num += 1

    # Extract pose and keypoints
    pose_output = estimator.process_frame(frame, frame_number=frame_num)
    tracks = pose_output.get("tracks", [])

    # Run object detector with wrist tracking
    events = detector.process_frame(
        frame=frame,
        frame_number=frame_num,
        tracks=tracks,
        test_id="eval_seat12_obj",
    )

    for ev in events:
        events_by_type[ev["event_type"]].append(ev)
        print(
            f"  [FRAME {frame_num:5d}] EVENT: {ev['event_type']:18s} | "
            f"risk={ev['risk_score']:4.1f} ({ev['risk_category']:4s}) | "
            f"conf={ev['confidence']:.2f} | {ev['metadata'].get('reason', '')}"
        )

cap.release()
elapsed = time.perf_counter() - t_start
avg_fps = frame_num / elapsed if elapsed > 0 else 0

print(f"\n{'='*70}")
print("EVALUATION SUMMARY — Object Detection on Seat 12 Clip")
print(f"{'='*70}")
print(f"  Total frames processed : {frame_num}")
print(f"  Elapsed time           : {elapsed:.1f}s  ({avg_fps:.1f} FPS)")
print(f"  Total events flagged   : {sum(len(v) for v in events_by_type.values())}")

for etype, evts in events_by_type.items():
    scores = [e["risk_score"] for e in evts]
    frames = [e["frame_number"] for e in evts]
    high = [e for e in evts if e["risk_category"] == "high"]
    print(f"\n  [{etype}]  count={len(evts)}  frames={frames[:10]}{'...' if len(frames)>10 else ''}")
    print(f"    risk_score: min={min(scores):.1f}  max={max(scores):.1f}  avg={sum(scores)/len(scores):.1f}")
    print(f"    high-risk: {len(high)}, low-risk: {len(evts) - len(high)}")

if not events_by_type.get("phone_detected"):
    print("\n  [phone_detected] count=0  (Correct: no mobile phones present in this clip)")
