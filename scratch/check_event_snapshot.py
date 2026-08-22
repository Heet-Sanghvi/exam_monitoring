import requests
import json

API_BASE_URL = "http://127.0.0.1:8000"

def inspect_event_frame_700():
    res = requests.get(f"{API_BASE_URL}/behavior-events?limit=50", timeout=3)
    if res.status_code != 200:
        print(f"Failed to fetch events: Status {res.status_code}")
        return

    events = res.json()
    print(f"Total events retrieved: {len(events)}")
    
    target_event = None
    for evt in events:
        if evt.get("frame_number") == 700 or "13:00:00" in str(evt.get("timestamp")):
            target_event = evt
            print("\nFound Target Event:")
            print(json.dumps(evt, indent=2))
            break

    if not target_event:
        print("Target event frame 700 / 13:00:00 not found among recent events.")

if __name__ == "__main__":
    inspect_event_frame_700()
