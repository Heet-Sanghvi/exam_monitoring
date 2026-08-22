# Shared Data Schemas & API Contract

This folder defines the formal data contract between the **Perception Pipeline** (Person 1) and the **Backend System** (Person 2).

---

## 1. Behavior Event Schema (`schema_behavior_event.json`)

When the perception pipeline identifies a suspicious behavior event or updates a student's risk profile over a rolling window, it sends a payload conforming to this schema via HTTP POST to the backend endpoint:

```http
POST /behavior-events
Content-Type: application/json
```

### JSON Schema Field Summary

| Field | Type | Required | Description | Example |
| :--- | :--- | :--- | :--- | :--- |
| `event_id` | `string` | No | UUID v4 / string event ID | `"evt_12345"` |
| `student_id` | `string` | **Yes** | Student / Track identifier | `"student_01"` |
| `timestamp` | `string` | **Yes** | ISO 8601 UTC timestamp | `"2026-08-22T10:30:00Z"` |
| `event_type` | `string` | **Yes** | Enum of detected behavior | `"looking_away"` |
| `confidence` | `number` | **Yes** | Model confidence (0.0 to 1.0) | `0.92` |
| `risk_score` | `number` | **Yes** | Risk score (0.0 to 100.0) | `35.0` |
| `frame_number` | `integer` | No | Video frame index | `1450` |
| `bounding_box` | `array[4]` | No | `[x_min, y_min, x_max, y_max]` | `[100, 150, 300, 450]` |
| `metadata` | `object` | No | Extra key-value metadata | `{"camera_id": "cam_01"}` |

### Allowed `event_type` Values
- `looking_away`
- `head_turn`
- `body_rotation`
- `suspicious_hand_movement`
- `phone_detected`
- `chit_detected`
- `multiple_persons_detected`
- `sustained_absence`

---

## 2. Pose Output Schema (`schema_pose_output.json`)

Used internally by perception for frame-level pose keypoint logging (`output/pose_output.jsonl`).

---

## Example Payload

```json
{
  "event_id": "evt_98765",
  "student_id": "student_03",
  "timestamp": "2026-08-22T10:42:31Z",
  "event_type": "looking_away",
  "confidence": 0.91,
  "risk_score": 35.0,
  "frame_number": 1280,
  "bounding_box": [120.0, 80.0, 340.0, 410.0],
  "metadata": {
    "camera_id": "room1_cam2",
    "duration_seconds": 3.5
  }
}
```
