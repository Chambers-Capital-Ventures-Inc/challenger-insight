import streamlit as st
import pandas as pd
import plotly.express as px

st.set_page_config(page_title="Disengagement Dashboard", layout="wide")

@st.cache_data
def load_data():
    eeg_only = pd.read_csv("data/clean_eeg_only_disengagement.csv")
    eeg_full = pd.read_csv("data/clean_eeg.csv")
    disengage_eeg = pd.read_csv("data/clean_disengagement_eeg.csv")
    session_summary = pd.read_csv("data/challenger_insight_session_summary.csv")
    segments = pd.read_csv("data/challenger_insight_eeg_segments_5000.csv")
    return eeg_only, eeg_full, disengage_eeg, session_summary, segments

eeg_only, eeg_full, disengage_eeg, session_summary, segments = load_data()

st.title("Disengagement Analysis Dashboard")

st.sidebar.header("Filters")
subject_filter = st.sidebar.selectbox("Select Subject", ["All"] + sorted(session_summary.subject_id.unique().tolist()))
modality_filter = st.sidebar.multiselect("Modality", session_summary.modality.unique())
difficulty_filter = st.sidebar.multiselect("Difficulty", session_summary.task_difficulty.unique())

filtered_sessions = session_summary.copy()
if subject_filter != "All":
    filtered_sessions = filtered_sessions[filtered_sessions.subject_id == subject_filter]
if modality_filter:
    filtered_sessions = filtered_sessions[filtered_sessions.modality.isin(modality_filter)]
if difficulty_filter:
    filtered_sessions = filtered_sessions[filtered_sessions.task_difficulty.isin(difficulty_filter)]

st.subheader("Session Summary")

col1, col2, col3, col4 = st.columns(4)
col1.metric("Avg Cognitive Load", round(filtered_sessions.mean_cog_load.mean(), 3))
col2.metric("Avg % Engaged", f"{round(filtered_sessions.pct_engaged.mean(),1)}%")
col3.metric("Avg Disengagement Risk", round(filtered_sessions.mean_disengage_risk.mean(), 3))
col4.metric("Total Sessions", len(filtered_sessions))

st.markdown("---")

st.subheader("Engagement Curve (Per Session)")

if subject_filter != "All":
    user_segments = segments[segments.subject_id == subject_filter]
else:
    user_segments = segments.copy()

if not user_segments.empty:
    curve = px.line(
        user_segments,
        x="timestamp",
        y="engaged_label",
        color="session_id",
        title="Engagement Over Time",
    )
    st.plotly_chart(curve, use_container_width=True)
else:
    st.write("No data for selected filters.")

st.subheader("Top Disengagement Triggers")

trigger_counts = user_segments.groupby("recommended_trigger").size().reset_index(name="count")
trigger_counts = trigger_counts[trigger_counts.recommended_trigger != 0]

if not trigger_counts.empty:
    bar = px.bar(
        trigger_counts.sort_values("count", ascending=False),
        x="count",
        y="recommended_trigger",
        orientation="h",
        title="Disengagement Triggers",
    )
    st.plotly_chart(bar, use_container_width=True)
else:
    st.write("No trigger data.")

st.subheader("EEG Feature Distributions")

feature_cols = [
    "delta_power","theta_power","alpha_power","beta_power","gamma_power",
    "theta_alpha_ratio","beta_alpha_ratio","theta_beta_ratio","spectral_entropy"
]

feature = st.selectbox("Select EEG Feature", feature_cols)
hist = px.histogram(eeg_full, x=feature, nbins=40, title=f"Distribution of {feature}")
st.plotly_chart(hist, use_container_width=True)

st.subheader("Disengagement Risk Over Time")

risk_plot = px.line(
    user_segments,
    x="timestamp",
    y="disengagement_risk",
    color="session_id",
    title="Disengagement Risk Curve",
)
st.plotly_chart(risk_plot, use_container_width=True)