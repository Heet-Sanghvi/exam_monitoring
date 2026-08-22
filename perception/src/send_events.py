"""
Event Dispatcher Module
Part of the Exam Monitoring Perception Pipeline.

Responsibilities:
1. On startup, call GET /tests/active to fetch the current test_id once.
   Falls back to a configurable default if backend is unreachable.
2. On each flagged violation event, POST multipart to /behavior-events:
   - JSON payload field: event metadata (no student_id or seat_zone — invigilator identifies visually)
   - File field: annotated snapshot image
3. Bounding box conversion: [x1, y1, x2, y2] → [x, y, w, h] before sending.
4. Graceful offline handling: never crash perception; save event JSON locally on failure.

Bounding Box Contract (backend enforced):
    Input (internal):  [x1, y1, x2, y2] (top-left, bottom-right corners)
    Output (API):      [x, y, w, h]  where x=x1, y=y1, w=x2-x1, h=y2-y1
"""

import json
import os
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

import requests


@dataclass
class EventDispatcherConfig:
    """Configuration for backend communication endpoints."""
    base_url: str = field(default_factory=lambda: os.environ.get("EXAM_BACKEND_URL", "http://127.0.0.1:8000"))
    active_test_endpoint: str = "/tests/active"
    behavior_events_endpoint: str = "/behavior-events"
    default_test_id: str = "test_default"
    connect_timeout_seconds: float = 3.0
    request_timeout_seconds: float = field(default_factory=lambda: float(os.environ.get("EXAM_REQUEST_TIMEOUT", "6.0")))
    offline_fallback_dir: str = "perception/output/offline_events"


def convert_bbox_xyxy_to_xywh(bbox_xyxy: List[float]) -> List[float]:
    """
    Convert bounding box from [x1, y1, x2, y2] (corner format) to
    [x, y, w, h] (top-left origin + dimensions) as required by the backend API.

    Args:
        bbox_xyxy: [x1, y1, x2, y2] where (x1,y1) is top-left and (x2,y2) is bottom-right.

    Returns:
        [x, y, w, h] where x=x1, y=y1, w=x2-x1, h=y2-y1.
    """
    if len(bbox_xyxy) != 4:
        raise ValueError(f"Expected 4 bbox values, got {len(bbox_xyxy)}: {bbox_xyxy}")
    x1, y1, x2, y2 = bbox_xyxy
    return [x1, y1, x2 - x1, y2 - y1]


class EventDispatcher:
    """
    Dispatches flagged behavior events to the backend API.
    Handles startup test_id discovery, pre-flight connectivity, and per-event multipart POST.
    """

    def __init__(self, config: Optional[EventDispatcherConfig] = None):
        self.config = config or EventDispatcherConfig()
        self.test_id: str = self.config.default_test_id
        self._online: bool = False

        os.makedirs(self.config.offline_fallback_dir, exist_ok=True)

    def check_backend_connectivity(self) -> Tuple[bool, str]:
        """
        Perform a fast pre-flight reachability check to verify backend status.

        Returns:
            Tuple of (is_reachable: bool, message: str)
        """
        url = self.config.base_url.rstrip("/") + "/"
        timeout = (self.config.connect_timeout_seconds, self.config.connect_timeout_seconds)
        try:
            resp = requests.get(url, timeout=timeout)
            if resp.status_code == 200:
                try:
                    data = resp.json()
                    svc_name = data.get("service", "Backend API")
                    return True, f"Connected to {svc_name} at {self.config.base_url}"
                except Exception:
                    return True, f"Connected to backend at {self.config.base_url}"
            return True, f"Backend reached at {self.config.base_url} (HTTP {resp.status_code})"
        except requests.exceptions.ConnectionError:
            return False, f"Connection refused to {self.config.base_url}. Verify Heet's server is running on the host/LAN."
        except requests.exceptions.Timeout:
            return False, f"Connection to {self.config.base_url} timed out ({self.config.connect_timeout_seconds}s). Check IP, Wi-Fi network, and firewall."
        except Exception as exc:
            return False, f"Cannot reach {self.config.base_url} ({type(exc).__name__}: {exc})"

    def fetch_active_test_id(self) -> str:
        """
        Fetch the active test_id from the backend on startup.
        Returns the fetched test_id, or the configured default if backend is unavailable.
        """
        url = self.config.base_url.rstrip("/") + self.config.active_test_endpoint
        timeout = (self.config.connect_timeout_seconds, self.config.request_timeout_seconds)
        try:
            resp = requests.get(url, timeout=timeout)
            if resp.status_code == 404:
                self.test_id = self.config.default_test_id
                self._online = True  # Backend is online, but no session created yet
                print(f"[EventDispatcher] Backend is online, but no active exam session found (HTTP 404). Using fallback test_id='{self.test_id}'.")
                return self.test_id

            resp.raise_for_status()
            data = resp.json()
            # Accept either {"test_id": "..."} or {"id": "..."} from different backend versions
            fetched_id = data.get("test_id") or data.get("id") or self.config.default_test_id
            self.test_id = str(fetched_id)
            self._online = True
            print(f"[EventDispatcher] Active test_id fetched: '{self.test_id}'")
        except Exception as exc:
            self.test_id = self.config.default_test_id
            self._online = False
            print(f"[EventDispatcher] Could not fetch active test_id ({type(exc).__name__}). Using fallback test_id='{self.test_id}'.")
        return self.test_id

    def build_event_payload(
        self,
        flagged_event: Dict[str, Any],
        frame_number: int,
        timestamp: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Build the JSON event payload from a flagged violation dict.
        Converts bounding box from [x1,y1,x2,y2] → [x,y,w,h].

        Args:
            flagged_event: Output dict from MovementFlaggingEngine.evaluate_track().
            frame_number: Current frame index.
            timestamp: ISO 8601 string; defaults to now(UTC).
            metadata: Optional extra dict (e.g. camera_id, clip_name).

        Returns:
            Dict ready for JSON serialization and multipart POST.
        """
        if timestamp is None:
            timestamp = datetime.now(timezone.utc).isoformat()

        bbox_xyxy = flagged_event.get("bounding_box", [0.0, 0.0, 0.0, 0.0])
        bbox_xywh = convert_bbox_xyxy_to_xywh(bbox_xyxy)

        payload = {
            "event_id": str(uuid.uuid4()),
            "test_id": self.test_id,
            "frame_number": frame_number,
            "timestamp": timestamp,
            "event_type": flagged_event.get("event_type", "unknown"),
            "confidence": flagged_event.get("confidence", 0.0),
            "risk_score": flagged_event.get("risk_score", 0.0),
            "risk_category": flagged_event.get("risk_category", "low"),
            "bounding_box": bbox_xywh,
            "metadata": metadata or {},
        }
        # Optionally embed reason string in metadata if provided
        reason = flagged_event.get("reason")
        if reason:
            payload["metadata"]["reason"] = reason
        metrics = flagged_event.get("metrics")
        if metrics:
            payload["metadata"]["metrics"] = metrics

        return payload

    def dispatch(
        self,
        flagged_event: Dict[str, Any],
        snapshot_path: str,
        frame_number: int,
        timestamp: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """
        Send a flagged behavior event + snapshot to the backend via multipart POST.
        If the backend is unreachable, saves the payload JSON locally and returns False.

        Args:
            flagged_event: Output from MovementFlaggingEngine.evaluate_track().
            snapshot_path: Absolute or relative path to the saved snapshot image.
            frame_number: Frame index at which the violation occurred.
            timestamp: ISO 8601 string; defaults to now(UTC).
            metadata: Optional extra context dict.

        Returns:
            True if successfully sent to backend, False if saved locally only.
        """
        payload = self.build_event_payload(
            flagged_event=flagged_event,
            frame_number=frame_number,
            timestamp=timestamp,
            metadata=metadata,
        )

        url = self.config.base_url.rstrip("/") + self.config.behavior_events_endpoint

        try:
            if not os.path.isfile(snapshot_path):
                raise FileNotFoundError(f"Snapshot not found: {snapshot_path}")

            with open(snapshot_path, "rb") as img_file:
                # Multipart: JSON event string in 'event' field, image bytes in 'snapshot' field
                files = {
                    "snapshot": (
                        os.path.basename(snapshot_path),
                        img_file,
                        "image/jpeg",
                    )
                }
                data = {"event": json.dumps(payload)}
                resp = requests.post(
                    url,
                    data=data,
                    files=files,
                    timeout=(self.config.connect_timeout_seconds, self.config.request_timeout_seconds),
                )
            resp.raise_for_status()
            print(
                f"[EventDispatcher] Sent event '{payload['event_type']}' "
                f"frame={frame_number} risk={payload['risk_score']} → HTTP {resp.status_code}"
            )
            return True

        except Exception as exc:
            print(f"[EventDispatcher] Dispatch failed ({exc}). Saving event locally.")
            self._save_locally(payload, snapshot_path)
            return False

    def _save_locally(self, payload: Dict[str, Any], snapshot_path: str) -> None:
        """
        Persist event JSON to the offline fallback directory so no event is lost.
        """
        event_id = payload.get("event_id", "unknown")
        json_path = os.path.join(self.config.offline_fallback_dir, f"event_{event_id}.json")
        payload["_local_snapshot_path"] = snapshot_path
        try:
            with open(json_path, "w") as f:
                json.dump(payload, f, indent=2)
            print(f"[EventDispatcher] Event saved locally: {json_path}")
        except Exception as exc:
            print(f"[EventDispatcher] WARNING: Could not save locally either: {exc}")
