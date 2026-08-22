import sys
import os
import streamlit as st
import requests

API_BASE_URL = "http://127.0.0.1:8000"

def render_live_alert_banner(api_base_url=API_BASE_URL):
    """
    Renders an unmissable ambient live alert banner when a new behavior event
    (high or low risk) arrives. Designed for invigilators monitoring across the room.
    """
    # 1. Fetch latest behavior event from backend API
    latest_event = None
    try:
        res = requests.get(f"{api_base_url}/behavior-events?limit=1", timeout=2)
        if res.status_code == 200:
            events = res.json()
            if events:
                latest_event = events[0]
    except Exception:
        return  # Fail silently if backend API is temporarily offline

    if not latest_event:
        return

    latest_id = latest_event.get("id", 0)

    # 2. Initialize last_seen_event_id on fresh session startup to prevent stale alerts
    if "last_seen_event_id" not in st.session_state:
        st.session_state["last_seen_event_id"] = latest_id
        return

    # If no new event has arrived since last seen, return cleanly
    if latest_id <= st.session_state["last_seen_event_id"]:
        return

    # 3. New Event Detected -> Extract parameters
    risk_category = latest_event.get("risk_category", "low")
    event_type = latest_event.get("event_type", "unknown").replace("_", " ").title()
    test_id = latest_event.get("test_id", "N/A")
    frame_num = latest_event.get("frame_number", "N/A")
    is_high_risk = (risk_category == "high")

    # Styling: High-risk -> Bright Red fast pulse, Low-risk -> Bright Orange slow pulse
    banner_bg = "linear-gradient(135deg, #991B1B 0%, #EF4444 100%)" if is_high_risk else "linear-gradient(135deg, #9A3412 0%, #F97316 100%)"
    border_color = "#FCA5A5" if is_high_risk else "#FFEDD5"
    pulse_rgb = "239, 68, 68" if is_high_risk else "249, 115, 22"
    pulse_speed = "1.0s" if is_high_risk else "2.2s"
    title_text = "🚨 NEW HIGH-RISK DETECTION" if is_high_risk else "⚠️ NEW LOW-RISK DETECTION"

    # Render Floating Alert Box in Sidebar Top
    st.sidebar.markdown(f"""
    <style>
        @keyframes alert_pulse_box {{
            0% {{ box-shadow: 0 0 0 0 rgba({pulse_rgb}, 0.9); }}
            70% {{ box-shadow: 0 0 0 16px rgba({pulse_rgb}, 0); }}
            100% {{ box-shadow: 0 0 0 0 rgba(0, 0, 0, 0); }}
        }}
        .ambient-alert-card {{
            background: {banner_bg};
            color: white;
            padding: 14px 16px;
            border-radius: 10px;
            border: 2px solid {border_color};
            animation: alert_pulse_box {pulse_speed} infinite;
            margin-bottom: 12px;
            font-family: system-ui, -apple-system, sans-serif;
        }}
        .ambient-alert-title {{
            font-weight: 800;
            font-size: 0.95rem;
            margin-bottom: 4px;
            letter-spacing: 0.5px;
        }}
        .ambient-alert-body {{
            font-size: 0.85rem;
            opacity: 0.95;
            margin-bottom: 8px;
        }}
    </style>

    <div class="ambient-alert-card">
        <div class="ambient-alert-title">{title_text}</div>
        <div class="ambient-alert-body">
            <b>{event_type}</b> detected in <code>{test_id}</code> (Frame #{frame_num})
        </div>
    </div>
    """, unsafe_allow_html=True)

    btn_col1, btn_col2 = st.sidebar.columns([3, 2])
    if btn_col1.button("View Snapshot →", key=f"alert_btn_{latest_id}"):
        st.session_state["last_seen_event_id"] = latest_id
        st.session_state["selected_test_id"] = test_id
        st.session_state["jump_to_event_id"] = latest_id
        st.session_state["jump_risk_category"] = risk_category
        try:
            st.switch_page("pages/test_sessions.py")
        except Exception:
            st.rerun()

    if btn_col2.button("Dismiss ✖", key=f"dismiss_btn_{latest_id}"):
        st.session_state["last_seen_event_id"] = latest_id
        st.rerun()

    st.sidebar.markdown("---")
