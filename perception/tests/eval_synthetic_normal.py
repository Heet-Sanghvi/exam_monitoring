"""
Synthetic false-positive test: simulate a classroom of 8 students
sitting at various natural angles (0–30°) for 300 frames with static
head positions. Under the new tuned config (body_tilt_deg=35°,
body_tilt_min_nose_drop_px=30px), NONE of these should trigger.

This substitutes for a real normal/ clip since none is available.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import math
import random
from src.behavior_rules import MovementFlaggingEngine, MovementFlaggingConfig
from src.behavior_features import extract_keypoints_map

random.seed(42)

config = MovementFlaggingConfig(
    body_bend_drop_px=65.0,
    body_tilt_deg=35.0,
    body_tilt_min_nose_drop_px=30.0,
    hand_below_hip_px=25.0,
    hand_lateral_reach_px=30.0,
    high_risk_threshold=60.0,
    debounce_frames=15,
    snapshot_base_dir="/tmp/eval_normal_test",
)
os.makedirs("/tmp/eval_normal_test/low_risk", exist_ok=True)
os.makedirs("/tmp/eval_normal_test/high_risk", exist_ok=True)

engine = MovementFlaggingEngine(config=config)

# 8 synthetic students with varying static seated angles (0°–28°)
# and minor frame-to-frame jitter (±3px) simulating normal movement
students = [
    {"id": f"student_{i+1:03d}", "seat_tilt_deg": random.uniform(0, 28)}
    for i in range(8)
]

false_positives = 0
total_evaluations = 0

for frame_num in range(300):
    for stu in students:
        tilt_deg = stu["seat_tilt_deg"]
        # Shoulder span = 80px, compute Y offset for the given angle
        sh_y_offset = 80.0 * math.tan(math.radians(tilt_deg))
        jitter = random.uniform(-3, 3)

        cx = random.uniform(250, 400)
        sh_y = 200.0 + jitter
        nose_y = sh_y - 100.0 + jitter  # Nose well above shoulders

        keypoints = [
            {"name": "nose",           "x": cx,       "y": nose_y,             "confidence": 0.88},
            {"name": "left_shoulder",  "x": cx - 40,  "y": sh_y + sh_y_offset, "confidence": 0.92},
            {"name": "right_shoulder", "x": cx + 40,  "y": sh_y,               "confidence": 0.92},
            {"name": "left_hip",       "x": cx - 30,  "y": sh_y + 150 + jitter,"confidence": 0.85},
            {"name": "right_hip",      "x": cx + 30,  "y": sh_y + 150 + jitter,"confidence": 0.85},
            {"name": "left_wrist",     "x": cx - 20,  "y": sh_y + 100 + jitter,"confidence": 0.80},
            {"name": "right_wrist",    "x": cx + 20,  "y": sh_y + 100 + jitter,"confidence": 0.80},
        ]
        track_dict = {
            "student_id": stu["id"],
            "bbox": [cx - 60, nose_y - 10, cx + 60, sh_y + 180],
            "keypoints": keypoints,
            "confidence": 0.87,
        }
        result = engine.evaluate_track(track_dict, frame_number=frame_num)
        total_evaluations += 1
        if result is not None:
            false_positives += 1
            print(f"  FALSE POSITIVE  frame={frame_num:4d}  student={stu['id']}  "
                  f"type={result['event_type']}  risk={result['risk_score']}  reason={result['reason']}")

print(f"\n{'='*60}")
print(f"SYNTHETIC NORMAL SITTING — False-Positive Test")
print(f"{'='*60}")
print(f"  Students simulated  : {len(students)}")
print(f"  Frames              : 300")
print(f"  Total evaluations   : {total_evaluations}")
print(f"  False positives     : {false_positives}")
print(f"  FP rate             : {false_positives / total_evaluations * 100:.2f}%")

if false_positives == 0:
    print("\n  RESULT: ✅ ZERO false positives on synthetic normal sitting postures")
else:
    print(f"\n  RESULT: ⚠️  {false_positives} false positives — thresholds need further tuning")
