import streamlit as st
import requests
import pandas as pd
from datetime import datetime

st.set_page_config(
    page_title="AI Exam Monitoring & Proctoring Dashboard",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Modern Custom CSS Styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1E293B;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.0rem;
        color: #64748B;
        margin-bottom: 1.5rem;
    }
    .stMetric {
        background-color: #F8FAFC;
        padding: 12px;
        border-radius: 8px;
        border: 1px solid #E2E8F0;
    }
</style>
""", unsafe_allow_html=True)

API_BASE_URL = "http://127.0.0.1:8000"

def fetch_events(limit=200):
    try:
        res = requests.get(f"{API_BASE_URL}/behavior-events?limit={limit}", timeout=3)
        if res.status_code == 200:
            return res.json()
    except Exception:
        pass
    return []

def fetch_risk_scores():
    try:
        res = requests.get(f"{API_BASE_URL}/students/risk-scores", timeout=3)
        if res.status_code == 200:
            return res.json()
    except Exception:
        pass
    return []

st.sidebar.title("🛡️ Proctoring Portal")
page = st.sidebar.radio(
    "Navigation",
    ["🚨 Live Alerts", "📊 Student Risk Scores", "📋 Filterable Event Log"]
)

st.sidebar.markdown("---")
st.sidebar.subheader("System Status")

# System health ping
try:
    health_res = requests.get(API_BASE_URL, timeout=2)
    if health_res.status_code == 200:
        st.sidebar.success("Backend API: Online 🟢")
    else:
        st.sidebar.error("Backend API: Error 🔴")
except Exception:
    st.sidebar.error("Backend API: Offline 🔴")

st.sidebar.markdown("""
<small style="color: #64748B;">
AI Proctoring System v1.0<br/>
Real-time temporal behavior monitoring
</small>
""", unsafe_allow_html=True)

# Top Bar Header
st.markdown('<div class="main-header">Exam Monitoring & Proctoring Center</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Automated rolling-window behavior anomaly detection & live proctor alerts</div>', unsafe_allow_html=True)

events = fetch_events()
risk_scores = fetch_risk_scores()

col1, col2, col3, col4 = st.columns(4)
high_risk_count = sum(1 for s in risk_scores if s.get("risk_level") == "High")
medium_risk_count = sum(1 for s in risk_scores if s.get("risk_level") == "Medium")

col1.metric("Total Events Detected", len(events))
col2.metric("High Risk Students 🚨", high_risk_count)
col3.metric("Medium Risk Students ⚠️", medium_risk_count)
col4.metric("Monitored Students 🎓", len(risk_scores))

st.markdown("---")

# Import page modules dynamically or route content
if page == "🚨 Live Alerts":
    from pages.alerts import render_alerts_page
    render_alerts_page(events, API_BASE_URL)

elif page == "📊 Student Risk Scores":
    from pages.risk_scores import render_risk_scores_page
    render_risk_scores_page(risk_scores)

elif page == "📋 Filterable Event Log":
    from pages.event_log import render_event_log_page
    render_event_log_page(events)
