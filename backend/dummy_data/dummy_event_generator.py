import time
import random
import uuid
from datetime import datetime, timezone
import requests
import argparse

API_URL = "http://127.0.0.1:8000/behavior-events"

STUDENT_IDS = [f"student_{i:02d}" for i in range(1, 16)]

EVENT_TYPES = [
    ("looking_away", (15.0, 35.0)),
    ("head_turn", (10.0, 30.0)),
    ("body_rotation", (20.0, 45.0)),
    ("suspicious_hand_movement", (25.0, 50.0)),
    ("phone_detected", (70.0, 95.0)),
    ("chit_detected", (65.0, 90.0)),
    ("multiple_persons_detected", (80.0, 100.0)),
    ("sustained_absence", (40.0, 75.0)),
]

def generate_random_event(student_id: str = None) -> dict:
    if not student_id:
        student_id = random.choice(STUDENT_IDS)
    
    event_type, (min_risk, max_risk) = random.choice(EVENT_TYPES)
    confidence = round(random.uniform(0.70, 0.99), 2)
    risk_score = round(random.uniform(min_risk, max_risk), 1)
    frame_number = random.randint(100, 10000)
    
    # Bounding box [x_min, y_min, x_max, y_max]
    x_min = random.uniform(50.0, 300.0)
    y_min = random.uniform(50.0, 200.0)
    bounding_box = [
        round(x_min, 1),
        round(y_min, 1),
        round(x_min + random.uniform(100, 200), 1),
        round(y_min + random.uniform(150, 250), 1)
    ]
    
    return {
        "event_id": f"evt_{uuid.uuid4().hex[:8]}",
        "student_id": student_id,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "event_type": event_type,
        "confidence": confidence,
        "risk_score": risk_score,
        "frame_number": frame_number,
        "bounding_box": bounding_box,
        "metadata": {
            "camera_id": f"cam_{random.randint(1, 4):02d}",
            "simulated": True
        }
    }

def run_generator(api_url: str = API_URL, interval: float = 2.0, count: int = 0):
    print(f"Starting dummy event generator targeting {api_url} (interval={interval}s)...")
    sent = 0
    try:
        while True:
            event = generate_random_event()
            try:
                response = requests.post(api_url, json=event, timeout=5)
                if response.status_code == 201:
                    print(f"[{sent+1}] Sent event: Student={event['student_id']} | Type={event['event_type']} | Risk={event['risk_score']}")
                else:
                    print(f"Failed to post event ({response.status_code}): {response.text}")
            except requests.exceptions.RequestException as e:
                print(f"Connection error posting to API: {e}")

            sent += 1
            if count > 0 and sent >= count:
                print(f"Finished generating {count} dummy events.")
                break
            time.sleep(interval)
    except KeyboardInterrupt:
        print("\nDummy event generator stopped by user.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate simulated behavior events for backend testing")
    parser.add_argument("--url", type=str, default=API_URL, help="Backend API endpoint URL")
    parser.add_argument("--interval", type=float, default=2.0, help="Interval in seconds between events")
    parser.add_argument("--count", type=int, default=0, help="Total number of events to send (0 = infinite)")
    args = parser.parse_args()
    
    run_generator(api_url=args.url, interval=args.interval, count=args.count)
