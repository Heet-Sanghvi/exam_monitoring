import sys
import os
import streamlit as st
import pandas as pd
import requests

# Ensure dashboard directory is in python search path for module imports
DASHBOARD_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if DASHBOARD_DIR not in sys.path:
    sys.path.insert(0, DASHBOARD_DIR)

from utils.live_alerts import render_live_alert_banner

API_BASE_URL = "http://127.0.0.1:8000"

def render_event_log_page(events=None):
    st.subheader("📋 Comprehensive Behavior Event Log")
    st.caption("Search, filter, and audit all recorded proctoring events.")

    if events is None:
        try:
            res = requests.get(f"{API_BASE_URL}/behavior-events?limit=500", timeout=3)
            events = res.json() if res.status_code == 200 else []
        except Exception:
            events = []

    if not events:
        st.info("No behavior events recorded in the database.")
        return

    df = pd.DataFrame(events)

    # Filtering UI
    col1, col2, col3 = st.columns(3)
    
    with col1:
        test_ids = ["All"] + sorted(list(df["test_id"].dropna().unique())) if "test_id" in df.columns else ["All"]
        selected_test_id = st.selectbox("Filter by Test Session", test_ids)

    with col2:
        event_types = ["All"] + sorted(list(df["event_type"].dropna().unique())) if "event_type" in df.columns else ["All"]
        selected_event_type = st.selectbox("Filter by Event Type", event_types)

    with col3:
        risk_categories = ["All"] + sorted(list(df["risk_category"].dropna().unique())) if "risk_category" in df.columns else ["All"]
        selected_risk_cat = st.selectbox("Filter by Risk Category", risk_categories)

    # Apply filters
    filtered_df = df.copy()
    if selected_test_id != "All":
        filtered_df = filtered_df[filtered_df["test_id"] == selected_test_id]
    if selected_event_type != "All":
        filtered_df = filtered_df[filtered_df["event_type"] == selected_event_type]
    if selected_risk_cat != "All":
        filtered_df = filtered_df[filtered_df["risk_category"] == selected_risk_cat]

    st.markdown(f"**Showing {len(filtered_df)} of {len(events)} events**")

    # Reorder columns for display
    display_cols = ["id", "test_id", "risk_category", "event_type", "risk_score", "confidence", "review_status", "timestamp", "frame_number", "snapshot_path"]
    display_cols = [c for c in display_cols if c in filtered_df.columns]

    st.dataframe(filtered_df[display_cols], use_container_width=True)

    # CSV Export
    csv_data = filtered_df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="📥 Export Filtered Log as CSV",
        data=csv_data,
        file_name=f"event_log_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv",
        mime="text/csv"
    )

if __name__ == "__main__":
    render_event_log_page()
