import streamlit as st

def render_risk_scores_page(risk_scores=None):
    st.subheader("📊 Student Risk Scores (Deprecated)")
    st.info("Student-level persistent risk scoring has been deprecated in favor of event-based snapshot detection per exam test session.")

if __name__ == "__main__":
    render_risk_scores_page()
