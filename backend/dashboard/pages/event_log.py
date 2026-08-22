import streamlit as st
import pandas as pd

def render_event_log_page(events):
    st.subheader("📋 Comprehensive Behavior Event Log")
    st.caption("Search, filter, and audit all recorded proctoring events.")

    if not events:
        st.info("No behavior events recorded in the database.")
        return

    df = pd.DataFrame(events)

    # Filtering UI
    col1, col2, col3 = st.columns(3)
    
    with col1:
        students = ["All"] + sorted(list(df["student_id"].unique())) if "student_id" in df.columns else ["All"]
        selected_student = st.selectbox("Filter by Student", students)

    with col2:
        event_types = ["All"] + sorted(list(df["event_type"].unique())) if "event_type" in df.columns else ["All"]
        selected_event_type = st.selectbox("Filter by Event Type", event_types)

    with col3:
        min_risk = st.slider("Minimum Risk Score", 0.0, 100.0, 0.0)

    # Apply filters
    filtered_df = df.copy()
    if selected_student != "All":
        filtered_df = filtered_df[filtered_df["student_id"] == selected_student]
    if selected_event_type != "All":
        filtered_df = filtered_df[filtered_df["event_type"] == selected_event_type]
    filtered_df = filtered_df[filtered_df["risk_score"] >= min_risk]

    st.markdown(f"**Showing {len(filtered_df)} of {len(events)} events**")

    # Reorder columns for display
    display_cols = ["id", "student_id", "event_type", "risk_score", "confidence", "timestamp", "frame_number"]
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
