import streamlit as st
import pandas as pd
from datetime import datetime

def render_alerts_page(events, api_url):
    st.subheader("🚨 Live Proctoring Alert Feed")
    st.caption("Real-time stream of detected temporal behavior events and security anomalies.")

    if not events:
        st.info("No behavior events detected yet. Ensure the backend and perception pipeline (or dummy generator) are running.")
        return

    # Filter critical/suspicious alerts
    high_alerts = [e for e in events if e.get("risk_score", 0) >= 35.0]

    st.markdown(f"**Showing {len(events)} recent events ({len(high_alerts)} high/medium severity alerts)**")

    for evt in events[:15]:
        risk = evt.get("risk_score", 0)
        event_type = evt.get("event_type", "unknown").replace("_", " ").title()
        student_id = evt.get("student_id", "N/A")
        confidence = evt.get("confidence", 0.0)
        timestamp = evt.get("timestamp", "")
        frame_num = evt.get("frame_number", "N/A")

        if risk >= 70:
            badge_color = "#DC2626"
            border_color = "#FCA5A5"
            bg_color = "#FEF2F2"
            severity = "HIGH RISK"
        elif risk >= 35:
            badge_color = "#D97706"
            border_color = "#FDE68A"
            bg_color = "#FFFBEB"
            severity = "MEDIUM RISK"
        else:
            badge_color = "#2563EB"
            border_color = "#BFDBFE"
            bg_color = "#EFF6FF"
            severity = "LOW RISK"

        st.markdown(f"""
        <div style="
            background-color: {bg_color};
            border-left: 6px solid {badge_color};
            border-radius: 6px;
            padding: 14px 18px;
            margin-bottom: 12px;
            box-shadow: 0 1px 3px rgba(0,0,0,0.05);
        ">
            <div style="display: flex; justify-content: space-between; align-items: center;">
                <h4 style="margin: 0; color: #1E293B;">🎓 <strong>{student_id}</strong> &nbsp;—&nbsp; <span style="color: {badge_color};">{event_type}</span></h4>
                <span style="
                    background-color: {badge_color};
                    color: white;
                    padding: 3px 10px;
                    border-radius: 12px;
                    font-size: 0.8rem;
                    font-weight: 600;
                ">{severity} (Risk: {risk:.1f})</span>
            </div>
            <div style="margin-top: 8px; font-size: 0.9rem; color: #475569;">
                <b>Confidence:</b> {confidence:.0%} &nbsp;|&nbsp;
                <b>Frame #:</b> {frame_num} &nbsp;|&nbsp;
                <b>Timestamp:</b> {timestamp}
            </div>
        </div>
        """, unsafe_allow_html=True)
