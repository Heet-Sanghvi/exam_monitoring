import requests
import sys
import os

# Add backend and dashboard to sys.path
sys.path.insert(0, os.path.abspath("backend/dashboard"))

API_BASE_URL = "http://127.0.0.1:8000"

def test_fetch_and_render_logic():
    print("--- 1. Testing GET /tests ---")
    try:
        res = requests.get(f"{API_BASE_URL}/tests", timeout=3)
        print(f"Status Code: {res.status_code}")
        sessions = res.json()
        print(f"Sessions count: {len(sessions)}")
        for s in sessions:
            print(f"  - test_id: {s['test_id']}, title: {s.get('title')}, status: {s['status']}")
    except Exception as e:
        print(f"Error fetching /tests: {e}")
        return False

    print("\n--- 2. Testing GET /behavior-events ---")
    try:
        res_events = requests.get(f"{API_BASE_URL}/behavior-events?limit=10", timeout=3)
        print(f"Status Code: {res_events.status_code}")
        events = res_events.json()
        print(f"Events count: {len(events)}")
    except Exception as e:
        print(f"Error fetching /behavior-events: {e}")

    print("\n--- 3. Testing module imports and function invocation ---")
    try:
        import pages.test_sessions as ts
        import pages.risk_scores as rs
        import pages.event_log as el
        print("Imports successful!")
    except Exception as e:
        print(f"Import error: {e}")
        return False

    return True

if __name__ == "__main__":
    test_fetch_and_render_logic()
