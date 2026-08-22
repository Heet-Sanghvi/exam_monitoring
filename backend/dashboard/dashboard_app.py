import sys
import os
import streamlit as st
import requests
from streamlit_autorefresh import st_autorefresh

# Ensure dashboard directory is in python search path for page imports
DASHBOARD_DIR = os.path.dirname(os.path.abspath(__file__))
if DASHBOARD_DIR not in sys.path:
    sys.path.insert(0, DASHBOARD_DIR)

from utils.live_alerts import render_live_alert_banner

st.set_page_config(
    page_title="AI Exam Monitoring & Proctoring Dashboard",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

API_BASE_URL = "http://127.0.0.1:8000"

# Render ambient live alert banner
render_live_alert_banner(API_BASE_URL)

# Custom CSS Styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.0rem;
        font-weight: 700;
        color: #1E293B;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 0.95rem;
        color: #64748B;
        margin-bottom: 1.2rem;
    }
    .stMetric {
        background-color: #F8FAFC;
        padding: 12px;
        border-radius: 8px;
        border: 1px solid #E2E8F0;
    }
    .stMetric * {
        color: #0F172A !important;
    }
</style>
""", unsafe_allow_html=True)

st.sidebar.title("🛡️ Proctoring Portal")
page = st.sidebar.radio(
    "Navigation",
    ["🎓 Test Sessions", "📋 All Behavior Events"]
)

st.sidebar.markdown("---")
st.sidebar.subheader("🔄 Auto-Refresh Controls")
auto_refresh_enabled = st.sidebar.checkbox("Enable Auto-Refresh", value=True, help="Automatically refresh active session events and test list")
refresh_interval_sec = st.sidebar.slider("Interval (seconds)", min_value=1, max_value=15, value=3, step=1)

if auto_refresh_enabled:
    st_autorefresh(interval=refresh_interval_sec * 1000, key="proctoring_dashboard_autorefresh")

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
AI Proctoring System v2.0<br/>
Event-based Snapshot Proctoring
</small>
""", unsafe_allow_html=True)

# Top Bar Header
st.markdown('<div class="main-header">Exam Monitoring & Proctoring Center</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Real-time behavior anomaly detection & snapshot event proctoring</div>', unsafe_allow_html=True)

# Fetch Top Metrics
active_session_id = "None"
total_events_count = 0
high_risk_count = 0
sessions_count = 0

try:
    s_res = requests.get(f"{API_BASE_URL}/tests", timeout=3)
    if s_res.status_code == 200:
        sessions = s_res.json()
        sessions_count = len(sessions)
        for s in sessions:
            if s.get("status") == "active":
                active_session_id = s.get("test_id")
                break
except Exception:
    pass

try:
    e_res = requests.get(f"{API_BASE_URL}/behavior-events?limit=500", timeout=3)
    if e_res.status_code == 200:
        all_events = e_res.json()
        total_events_count = len(all_events)
        high_risk_count = sum(1 for e in all_events if e.get("risk_category") == "high")
except Exception:
    pass

col1, col2, col3, col4 = st.columns(4)
col1.metric("Active Session 🎓", active_session_id)
col2.metric("Total Test Sessions 📋", sessions_count)
col3.metric("High-Risk Flags 🚨", high_risk_count)
col4.metric("Total Events Recorded 👁️", total_events_count)

st.markdown("---")

if page == "🎓 Test Sessions":
    from pages.test_sessions import render_test_sessions_view
    render_test_sessions_view(API_BASE_URL)
elif page == "📋 All Behavior Events":
    from pages.event_log import render_event_log_page
    render_event_log_page(all_events if 'all_events' in locals() else [])
