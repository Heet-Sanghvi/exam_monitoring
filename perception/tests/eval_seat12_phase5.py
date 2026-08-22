"""
Phase 5 evaluation script — run full pipeline (track+pose → flag)
against the Seat 12 suspicious clip and report which events are triggered.

PoseEstimator.process_frame() internally calls the tracker, so we only need:
    pose_output = estimator.process_frame(frame, frame_number)
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import time
import cv2
from collections import defaultdict

from src.pose import PoseEstimator
from src.behavior_rules import MovementFlaggingEngine, MovementFlaggingConfig

CLIP = "perception/data/test_clips/suspicious/Seat No. 12 was seen taking a piece of paper from the desk.mkv"

config = MovementFlaggingConfig(
    body_bend_drop_px=65.0,
    body_tilt_deg=20.0,
    hand_below_hip_px=25.0,
    hand_lateral_reach_px=30.0,
    high_risk_threshold=60.0,
    debounce_frames=15,
    snapshot_base_dir="perception/output/snapshots",
)

estimator = PoseEstimator()
flagger = MovementFlaggingEngine(config=config)

cap = cv2.VideoCapture(CLIP)
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
fps_src = cap.get(cv2.CAP_PROP_FPS)
print(f"Clip: {CLIP}")
print(f"Total frames: {total_frames}  Source FPS: {fps_src:.1f}")
print(f"Thresholds → body_bend_drop={config.body_bend_drop_px}px  tilt={config.body_tilt_deg}°  "
      f"hand_below_hip={config.hand_below_hip_px}px  lateral={config.hand_lateral_reach_px}px\n")

events_by_type = defaultdict(list)
snapshots_saved = []

t_start = time.perf_counter()
frame_num = 0
while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame_num += 1

    # Pose estimation (calls tracker internally)
    pose_output = estimator.process_frame(frame, frame_number=frame_num)

    for track in pose_output.get("tracks", []):
        # Confidence is not in pose output schema; use a default
        track_conf = track.get("confidence", 0.85)

        track_dict = {
            "student_id": track.get("student_id", "student_unknown"),
            "bbox":       track.get("bbox", [0.0, 0.0, 0.0, 0.0]),
            "keypoints":  track.get("keypoints", []),
            "confidence": track_conf,
        }

        flagged = flagger.evaluate_track(track_dict, frame_number=frame_num)
        if flagged:
            events_by_type[flagged["event_type"]].append({
                "frame": frame_num,
                "risk_score": flagged["risk_score"],
                "risk_category": flagged["risk_category"],
                "reason": flagged.get("reason", ""),
            })
            _, snap_path = flagger.render_and_save_snapshot(
                frame=frame,
                flagged_event=flagged,
                frame_number=frame_num,
                test_id="eval_seat12",
            )
            snapshots_saved.append(snap_path)
            print(f"  [FRAME {frame_num:5d}] FLAG: {flagged['event_type']:25s} | "
                  f"risk={flagged['risk_score']:5.1f} ({flagged['risk_category']:4s}) | {flagged.get('reason','')}")

cap.release()
elapsed = time.perf_counter() - t_start
avg_fps = frame_num / elapsed if elapsed > 0 else 0

print(f"\n{'='*70}")
print(f"EVALUATION SUMMARY — Seat 12 Clip")
print(f"{'='*70}")
print(f"  Total frames processed : {frame_num}")
print(f"  Elapsed time           : {elapsed:.1f}s  ({avg_fps:.1f} FPS)")
print(f"  Total events flagged   : {sum(len(v) for v in events_by_type.values())}")

for etype, evts in events_by_type.items():
    scores = [e["risk_score"] for e in evts]
    frames = [e["frame"] for e in evts]
    print(f"\n  [{etype}]  count={len(evts)}  frames={frames[:8]}{'...' if len(frames)>8 else ''}")
    print(f"    risk_score: min={min(scores):.1f}  max={max(scores):.1f}  avg={sum(scores)/len(scores):.1f}")
    high = [e for e in evts if e["risk_category"] == "high"]
    print(f"    high-risk events: {len(high)}, low-risk events: {len(evts)-len(high)}")
    print(f"    reasons sample: {evts[0]['reason']}")

print(f"\n  Snapshots saved: {len(snapshots_saved)}")
if snapshots_saved:
    low  = [p for p in snapshots_saved if "low_risk"  in p]
    high = [p for p in snapshots_saved if "high_risk" in p]
    print(f"    → high_risk/ : {len(high)}")
    print(f"    → low_risk/  : {len(low)}")
    if high:
        print(f"    First high-risk: {high[0]}")
    if low:
        print(f"    First low-risk:  {low[0]}")
