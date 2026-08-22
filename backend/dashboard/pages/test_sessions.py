import sys
import os
import streamlit as st
import requests
from datetime import datetime

API_BASE_URL = "http://127.0.0.1:8000"

def render_test_sessions_view(api_base_url=API_BASE_URL):
    st.subheader("🎓 Exam Test Sessions")
    st.caption("Browse active and completed exam proctoring sessions.")

    # --- Start New Session Form ---
    with st.expander("➕ Start New Test Session", expanded=False):
        with st.form("start_test_form"):
            new_title = st.text_input("Test Session Title", placeholder="e.g. CS101 Final Exam - Room 3")
            submit_start = st.form_submit_button("🚀 Start Exam Session")
            if submit_start:
                try:
                    payload = {"title": new_title} if new_title.strip() else {}
                    res = requests.post(f"{api_base_url}/tests/start", json=payload, timeout=5)
                    if res.status_code == 201:
                        created_session = res.json()
                        st.success(f"Session started! ID: `{created_session['test_id']}`")
                        st.session_state["selected_test_id"] = created_session['test_id']
                        st.rerun()
                    else:
                        st.error(f"Failed to start session: {res.text}")
                except Exception as e:
                    st.error(f"Backend connection error: {e}")

    # --- Fetch All Sessions ---
    sessions = []
    try:
        res = requests.get(f"{api_base_url}/tests", timeout=5)
        if res.status_code == 200:
            sessions = res.json()
        else:
            st.error(f"Failed to fetch test sessions from backend (Status {res.status_code}): {res.text}")
    except Exception as e:
        st.error(f"Error connecting to backend API at {api_base_url}/tests: {e}")
        return

    if not sessions:
        st.info("No exam test sessions recorded yet. Click 'Start New Test Session' above to begin.")
        return

    # Check if a session is currently selected
    selected_test_id = st.session_state.get("selected_test_id")

    if selected_test_id:
        # Render Detail View for Selected Session
        render_session_detail_view(selected_test_id, api_base_url)
    else:
        # Render Sessions List
        render_sessions_list(sessions, api_base_url)


def render_sessions_list(sessions, api_base_url=API_BASE_URL):
    st.markdown(f"**Total Sessions ({len(sessions)})**")

    for s in sessions:
        test_id = s.get("test_id")
        title = s.get("title") or test_id
        status = s.get("status", "active")
        start_time = s.get("start_time", "")
        end_time = s.get("end_time") or "In Progress"

        status_color = "#16A34A" if status == "active" else "#64748B"
        status_bg = "#DCFCE7" if status == "active" else "#F1F5F9"
        status_label = "🟢 ACTIVE" if status == "active" else "⚪ COMPLETED"

        status_html = f'<span style="background-color: {status_bg}; color: {status_color}; padding: 4px 12px; border-radius: 12px; font-weight: 600; font-size: 0.85rem;">{status_label}</span>'

        with st.container():
            col1, col2, col3 = st.columns([3, 2, 1])
            with col1:
                st.markdown(f"#### 📝 {title}")
                st.caption(f"ID: `{test_id}` | Started: `{start_time}`")
            with col2:
                st.markdown(status_html, unsafe_allow_html=True)
                st.caption(f"Ended: `{end_time}`")
            with col3:
                if st.button("View Events 🔍", key=f"view_{test_id}"):
                    st.session_state["selected_test_id"] = test_id
                    st.rerun()
            st.markdown("---")


def render_session_detail_view(test_id, api_base_url=API_BASE_URL):
    col_back, col_title, col_end = st.columns([1, 4, 2])
    with col_back:
        if st.button("⬅️ Back to List"):
            st.session_state["selected_test_id"] = None
            st.rerun()

    with col_title:
        st.markdown(f"### Session Details: `{test_id}`")

    # Fetch session info
    session_info = None
    try:
        res_list = requests.get(f"{api_base_url}/tests", timeout=5)
        if res_list.status_code == 200:
            for s in res_list.json():
                if s["test_id"] == test_id:
                    session_info = s
                    break
    except Exception:
        pass

    if session_info:
        with col_end:
            if session_info.get("status") == "active":
                if st.button("🛑 End Test Session", key=f"end_{test_id}"):
                    try:
                        end_res = requests.post(f"{api_base_url}/tests/{test_id}/end", timeout=5)
                        if end_res.status_code == 200:
                            st.success("Session ended!")
                            st.rerun()
                    except Exception as e:
                        st.error(f"Error: {e}")
            else:
                st.info("Status: Completed")

    st.markdown("---")

    # High Risk & Low Risk Tabs
    tab_high, tab_low = st.tabs(["🚨 High Risk Events", "ℹ️ Low Risk Events"])

    with tab_high:
        render_event_group(test_id, "high", api_base_url)

    with tab_low:
        render_event_group(test_id, "low", api_base_url)


def render_event_group(test_id, risk_category, api_base_url=API_BASE_URL):
    try:
        res = requests.get(f"{api_base_url}/tests/{test_id}/events?risk_category={risk_category}", timeout=5)
        if res.status_code == 200:
            events = res.json()
        else:
            events = []
    except Exception as e:
        st.error(f"Failed to fetch events: {e}")
        return

    if not events:
        st.info(f"No {risk_category}-risk behavior events recorded for this session.")
        return

    st.markdown(f"**Found {len(events)} {risk_category}-risk event(s)**")

    # Render events in a grid
    cols = st.columns(2)
    for idx, evt in enumerate(events):
        with cols[idx % 2]:
            render_event_card(evt, risk_category, api_base_url)


def render_event_card(evt, risk_category, api_base_url=API_BASE_URL):
    event_db_id = evt.get("id") or evt.get("event_id")
    event_type = evt.get("event_type", "unknown").replace("_", " ").title()
    confidence = evt.get("confidence", 0.0)
    risk_score = evt.get("risk_score", 0.0)
    frame_num = evt.get("frame_number", "N/A")
    timestamp = evt.get("timestamp", "")
    snapshot_path = evt.get("snapshot_path")
    review_status = evt.get("review_status", "unreviewed")

    is_high_risk = (risk_category == "high")

    if is_high_risk:
        card_style = "border: 2px solid #EF4444; border-left: 8px solid #DC2626; background: linear-gradient(135deg, #FEF2F2 0%, #FFF1F2 100%); border-radius: 10px; padding: 14px; margin-bottom: 14px; box-shadow: 0 4px 12px rgba(220, 38, 38, 0.15);"
        badge_html = '<span style="background-color: #DC2626; color: white; padding: 4px 12px; border-radius: 14px; font-weight: 700; font-size: 0.85rem; letter-spacing: 0.5px;">🚨 HIGH RISK FLAG</span>'
    else:
        card_style = "border: 1px solid #E2E8F0; border-left: 4px solid #3B82F6; background-color: #F8FAFC; border-radius: 8px; padding: 12px; margin-bottom: 12px;"
        badge_html = '<span style="background-color: #3B82F6; color: white; padding: 3px 10px; border-radius: 12px; font-weight: 500; font-size: 0.8rem;">ℹ️ Low Risk</span>'

    card_header_html = f'<div style="{card_style}"><div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 6px;"><h4 style="margin: 0; color: #1E293B;">⚠️ {event_type}</h4>{badge_html}</div><small style="color: #475569; font-weight: 500;">Frame #{frame_num} &nbsp;|&nbsp; Timestamp: {timestamp}</small></div>'

    with st.container():
        st.markdown(card_header_html, unsafe_allow_html=True)

        if snapshot_path:
            image_url = f"{api_base_url}{snapshot_path}"
            caption_prefix = "🚨 High-Risk Anomaly Snapshot" if is_high_risk else "Low-Risk Snapshot"
            st.image(image_url, use_container_width=True, caption=f"{caption_prefix} ({event_type})")
        else:
            st.warning("No snapshot image attached")

        col1, col2 = st.columns(2)
        col1.metric("Confidence", f"{confidence:.0%}")
        col2.metric("Risk Score", f"{risk_score:.1f}")

        # --- Teacher Review Controls ---
        st.markdown("##### 👩‍🏫 Teacher Review")

        if review_status == "correct":
            correct_badge_html = '<div style="background-color: #DCFCE7; color: #15803D; padding: 6px 12px; border-radius: 6px; font-weight: bold; margin-bottom: 8px;">✅ Verified Correct Flag</div>'
            st.markdown(correct_badge_html, unsafe_allow_html=True)
            if st.button("🔄 Reset Review Status", key=f"reset_{event_db_id}"):
                update_review_status(api_base_url, event_db_id, "unreviewed")
        elif review_status == "incorrect":
            incorrect_badge_html = '<div style="background-color: #FEE2E2; color: #B91C1C; padding: 6px 12px; border-radius: 6px; font-weight: bold; margin-bottom: 8px;">❌ Flagged Incorrect (False Alarm)</div>'
            st.markdown(incorrect_badge_html, unsafe_allow_html=True)
            if st.button("🔄 Reset Review Status", key=f"reset_{event_db_id}"):
                update_review_status(api_base_url, event_db_id, "unreviewed")
        else:
            st.caption("Status: `unreviewed`")
            btn_col1, btn_col2 = st.columns(2)
            if btn_col1.button("✅ Correct", key=f"correct_{event_db_id}"):
                update_review_status(api_base_url, event_db_id, "correct")
            if btn_col2.button("❌ Incorrect", key=f"incorrect_{event_db_id}"):
                update_review_status(api_base_url, event_db_id, "incorrect")


def update_review_status(api_base_url, event_db_id, new_status):
    try:
        url = f"{api_base_url}/behavior-events/{event_db_id}"
        res = requests.patch(url, json={"review_status": new_status}, timeout=5)
        if res.status_code == 200:
            st.success(f"Updated review status to '{new_status}'!")
            st.rerun()
        else:
            st.error(f"Failed to update review status (Status {res.status_code}): {res.text}")
    except Exception as e:
        st.error(f"Network error updating review status: {e}")

if __name__ == "__main__":
    render_test_sessions_view(API_BASE_URL)
