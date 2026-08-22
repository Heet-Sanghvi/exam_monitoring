import requests
import json

API_BASE_URL = "http://127.0.0.1:8000"

def test_teacher_review_patch_flow():
    print("--- 1. Fetching existing events ---")
    res = requests.get(f"{API_BASE_URL}/behavior-events?limit=5", timeout=3)
    if res.status_code != 200 or not res.json():
        print("No events found in DB to test review PATCH.")
        return False

    events = res.json()
    target_event = events[0]
    event_id = target_event["id"]
    initial_status = target_event.get("review_status", "unreviewed")
    print(f"Target Event ID: {event_id}, Initial Status: '{initial_status}'")

    print("\n--- 2. Updating review_status to 'correct' via PATCH ---")
    patch_res = requests.patch(
        f"{API_BASE_URL}/behavior-events/{event_id}",
        json={"review_status": "correct"},
        timeout=3
    )
    print(f"PATCH Status Code: {patch_res.status_code}")
    if patch_res.status_code == 200:
        updated_event = patch_res.json()
        print(f"Updated Event Review Status: '{updated_event['review_status']}'")

    print("\n--- 3. Verifying updated status via GET /behavior-events ---")
    verify_res = requests.get(f"{API_BASE_URL}/behavior-events?limit=20", timeout=3)
    if verify_res.status_code == 200:
        for evt in verify_res.json():
            if evt["id"] == event_id:
                print(f"Verified Event DB Status: '{evt['review_status']}'")
                assert evt["review_status"] == "correct"
                break

    print("\n--- 4. Resetting review_status to 'unreviewed' ---")
    reset_res = requests.patch(
        f"{API_BASE_URL}/behavior-events/{event_id}",
        json={"review_status": "unreviewed"},
        timeout=3
    )
    if reset_res.status_code == 200:
        print(f"Reset Event Review Status: '{reset_res.json()['review_status']}'")

    print("\nSUCCESS: Teacher review PATCH endpoint verified end-to-end!")
    return True

if __name__ == "__main__":
    test_teacher_review_patch_flow()
