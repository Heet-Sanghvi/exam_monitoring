import sys
import os
import streamlit as st

# Ensure dashboard directory is in python search path for module imports
DASHBOARD_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if DASHBOARD_DIR not in sys.path:
    sys.path.insert(0, DASHBOARD_DIR)

from utils.live_alerts import render_live_alert_banner

API_BASE_URL = "http://127.0.0.1:8000"

def render_risk_scores_page(risk_scores=None):
    st.subheader("📊 Student Risk Scores (Deprecated)")
    st.info("Student-level persistent risk scoring has been deprecated in favor of event-based snapshot detection per exam test session.")

if __name__ == "__main__":
    render_risk_scores_page()
