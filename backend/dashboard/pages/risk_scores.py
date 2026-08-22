import streamlit as st
import pandas as pd

def render_risk_scores_page(risk_scores):
    st.subheader("📊 Student Accumulated Risk Scores")
    st.caption("Temporal rolling-window risk assessment per student.")

    if not risk_scores:
        st.info("No student risk scores recorded yet.")
        return

    df = pd.DataFrame(risk_scores)
    
    # Filter controls
    risk_level_filter = st.multiselect(
        "Filter by Risk Level",
        options=["High", "Medium", "Low"],
        default=["High", "Medium", "Low"]
    )

    filtered_df = df[df["risk_level"].isin(risk_level_filter)]

    # Display summary statistics
    st.markdown("### Risk Overview")
    
    chart_data = filtered_df.set_index("student_id")[["max_risk_score", "latest_risk_score"]]
    st.bar_chart(chart_data)

    st.markdown("### Student Details")
    
    def color_risk(val):
        if val == "High":
            return 'background-color: #FEE2E2; color: #991B1B; font-weight: bold;'
        elif val == "Medium":
            return 'background-color: #FEF3C7; color: #92400E; font-weight: bold;'
        return 'background-color: #D1FAE5; color: #065F46;'

    styled_df = filtered_df.style.applymap(color_risk, subset=["risk_level"])
    st.dataframe(styled_df, use_container_width=True)
