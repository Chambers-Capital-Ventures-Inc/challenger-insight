import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import joblib
import os
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, classification_report,
    mean_squared_error, mean_absolute_error, r2_score
)
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor

# ------------------------------------------------------------
# PAGE CONFIGURATION
# ------------------------------------------------------------
st.set_page_config(page_title="DeepHumanX Disengagement Dashboard", layout="wide")

# ------------------------------------------------------------
# BRANDING (DeepHumanX CSS)
# ------------------------------------------------------------
st.markdown("""
<style>
/* Sidebar background only */
section[data-testid="stSidebar"] {
    background-color: #3a4252 !important;
}

/* Sidebar text general styling */
section[data-testid="stSidebar"] p,
section[data-testid="stSidebar"] label,
section[data-testid="stSidebar"] h1,
section[data-testid="stSidebar"] h2,
section[data-testid="stSidebar"] h3 {
    color: white !important;
    font-family: 'Montserrat', sans-serif !important;
}

/* Selectbox and multiselect readable */
section[data-testid="stSidebar"] .stSelectbox div[data-baseweb="select"],
section[data-testid="stSidebar"] .stMultiSelect div[data-baseweb="select"] {
    background-color: white !important;
    color: black !important;
    border-radius: 6px !important;
}

/* Arrow icon visible */
section[data-testid="stSidebar"] .stSelectbox svg,
section[data-testid="stSidebar"] .stMultiSelect svg {
    fill: black !important;
}
</style>
""", unsafe_allow_html=True)

# ------------------------------------------------------------
# LOAD DATA
# ------------------------------------------------------------
@st.cache_data
def load_data():
    base = "data/"
    eeg_segments = pd.read_csv(base + "challenger_insight_eeg_segments_5000.csv")
    session_summary = pd.read_csv(base + "challenger_insight_session_summary.csv")
    eeg_full = pd.read_csv(base + "clean_eeg.csv")
    eeg_only = pd.read_csv(base + "clean_eeg_only_disengagement.csv")
    clean_eeg = pd.read_csv(base + "clean_disengagement_eeg.csv")
    return eeg_segments, session_summary, eeg_full, eeg_only, clean_eeg

eeg_segments, session_summary, eeg_full, eeg_only, clean_eeg = load_data()

# ------------------------------------------------------------
# LOAD TRAINED MODEL (Random Forest Regressor)
# ------------------------------------------------------------
MODEL_PATH = "notebooks/tuned_rf_disengagement.pkl"
model = joblib.load(MODEL_PATH)

# ------------------------------------------------------------
# TRIGGER LABELS
# ------------------------------------------------------------
trigger_map = {0: "No Trigger", 3: "Cognitive Fatigue"}

# ------------------------------------------------------------
# TIME FORMATTER
# ------------------------------------------------------------
def format_time_difference(t0, t1):
    diff_seconds = (t1 - t0).total_seconds()
    diff_minutes = diff_seconds / 60
    return f"{int(diff_seconds)} sec ({diff_minutes:.1f} min)"

# ------------------------------------------------------------
# METRIC FUNCTIONS
# ------------------------------------------------------------
def get_disengagement_start(df):
    disengaged = df[df["engaged_label"] == 0]
    if disengaged.empty:
        return "N/A"
    t0 = pd.to_datetime(df["timestamp"].iloc[0])
    t1 = pd.to_datetime(disengaged["timestamp"].iloc[0])
    return format_time_difference(t0, t1)

def get_critical_drop(df):
    df = df.copy()
    df["risk_diff"] = df["disengagement_risk"].diff()

    if df["risk_diff"].isna().all():
        return "N/A"

    idx = df["risk_diff"].idxmax()
    t0 = pd.to_datetime(df["timestamp"].iloc[0])
    t1 = pd.to_datetime(df.loc[idx, "timestamp"])

    return format_time_difference(t0, t1)

def get_leaving_before_end(df):
    cutoff = int(len(df) * 0.8)
    final = df.iloc[cutoff:]
    if len(final) == 0:
        return "N/A"
    disengaged = (final["engaged_label"] == 0).sum()
    pct = disengaged / len(final) * 100
    return f"{pct:.1f}%"

# ------------------------------------------------------------
# HEADER + LOGO
# ------------------------------------------------------------
st.title("Disengagement Analysis Dashboard")
st.image("data/color_deephumanx_logo.png", width=200)
st.markdown("### **AI-Augmented • Human-Enhanced • Purpose-Driven™**")

# ------------------------------------------------------------
# SIDEBAR FILTERS
# ------------------------------------------------------------
st.sidebar.header("Filters")
subject_filter = st.sidebar.selectbox(
    "Select Subject",
    ["All"] + sorted(session_summary.subject_id.unique().tolist())
)
modality_filter = st.sidebar.multiselect("Modality", session_summary.modality.unique())
difficulty_filter = st.sidebar.multiselect("Difficulty", session_summary.task_difficulty.unique())

filtered_sessions = session_summary.copy()
if subject_filter != "All":
    filtered_sessions = filtered_sessions[filtered_sessions.subject_id == subject_filter]
if modality_filter:
    filtered_sessions = filtered_sessions[filtered_sessions.modality.isin(modality_filter)]
if difficulty_filter:
    filtered_sessions = filtered_sessions[filtered_sessions.task_difficulty.isin(difficulty_filter)]

# ------------------------------------------------------------
# SESSION SUMMARY
# ------------------------------------------------------------
st.subheader("Session Summary")
c1, c2, c3, c4 = st.columns(4)
c1.metric("Avg Cognitive Load", round(filtered_sessions.mean_cog_load.mean(), 3))
c2.metric("Avg % Engaged", f"{round(filtered_sessions.pct_engaged.mean(), 1)}%")
c3.metric("Avg Disengagement Risk", round(filtered_sessions.mean_disengage_risk.mean(), 3))
c4.metric("Total Sessions", len(filtered_sessions))

st.markdown("---")

# ------------------------------------------------------------
# SEGMENTS FILTERING
# ------------------------------------------------------------
user_segments = eeg_segments.copy()
if subject_filter != "All":
    user_segments = user_segments[user_segments.subject_id == subject_filter]

# ------------------------------------------------------------
# ENGAGEMENT CURVE
# ------------------------------------------------------------
st.subheader("Engagement Curve")
if not user_segments.empty:
    fig = px.line(
        user_segments,
        x="timestamp", y="engaged_label",
        color="session_id",
        color_discrete_sequence=["#f1580c"]
    )
    st.plotly_chart(fig, use_container_width=True)
else:
    st.write("No data available.")

# ------------------------------------------------------------
# TOP TRIGGERS
# ------------------------------------------------------------
st.subheader("Top Disengagement Triggers")
trigger_counts = user_segments.groupby("recommended_trigger").size().reset_index(name="count")
trigger_counts = trigger_counts[trigger_counts.recommended_trigger != 0]
trigger_counts["trigger_label"] = trigger_counts["recommended_trigger"].map(trigger_map)

if not trigger_counts.empty:
    fig = px.bar(
        trigger_counts,
        x="count", y="trigger_label",
        orientation="h",
        color_discrete_sequence=["#f1580c"]
    )
    st.plotly_chart(fig, use_container_width=True)
else:
    st.write("No triggers found.")

# ------------------------------------------------------------
# KEY INSIGHTS METRICS
# ------------------------------------------------------------
st.markdown("---")
st.subheader("Key Disengagement Insights")

if not user_segments.empty:
    diseng_start = get_disengagement_start(user_segments)
    critical = get_critical_drop(user_segments)
    leaving = get_leaving_before_end(user_segments)

    most_common = (
        trigger_map.get(trigger_counts.sort_values("count", ascending=False).iloc[0]["recommended_trigger"], "N/A")
        if not trigger_counts.empty else "N/A"
    )

    a, b, c, d = st.columns(4)
    a.markdown(f"<div class='metric-box'><h3>Disengagement Starts</h3><p>{diseng_start}</p></div>", unsafe_allow_html=True)
    b.markdown(f"<div class='metric-box'><h3>Critical Drop-off</h3><p>{critical}</p></div>", unsafe_allow_html=True)
    c.markdown(f"<div class='metric-box'><h3>Leaving Before End</h3><p>{leaving}</p></div>", unsafe_allow_html=True)
    d.markdown(f"<div class='metric-box'><h3>Most Common Trigger</h3><p>{most_common}</p></div>", unsafe_allow_html=True)

# ------------------------------------------------------------
# PCA & t-SNE VISUALIZATION
# ------------------------------------------------------------
st.markdown("---")
st.subheader("EEG Feature Structure (PCA & t-SNE)")

feature_cols = [
    "delta_power","theta_power","alpha_power","beta_power","gamma_power",
    "theta_alpha_ratio","beta_alpha_ratio","theta_beta_ratio","spectral_entropy"
]

X = clean_eeg[feature_cols]
y_risk = clean_eeg["disengagement_risk"]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

st.write("### PCA (2D)")
pca_fig = px.scatter(
    x=X_pca[:,0], y=X_pca[:,1],
    color=y_risk,
    color_continuous_scale="RdBu",
    labels={"color": "Disengagement Risk"}
)
st.plotly_chart(pca_fig, use_container_width=True)

st.write("### t-SNE")
tsne = TSNE(n_components=2, perplexity=30, learning_rate=200)
X_tsne = tsne.fit_transform(X_scaled)

tsne_fig = px.scatter(
    x=X_tsne[:,0], y=X_tsne[:,1],
    color=y_risk,
    color_continuous_scale="RdBu"
)
st.plotly_chart(tsne_fig, use_container_width=True)

# ------------------------------------------------------------
# RISK PREDICTION (Using Tuned Random Forest)
# ------------------------------------------------------------
st.markdown("---")
st.subheader("Predict Disengagement Risk (Model Inference)")

uploaded = st.file_uploader("Upload EEG CSV (matching feature schema)", type=["csv"])
if uploaded:
    new_df = pd.read_csv(uploaded)
    if all(col in new_df.columns for col in feature_cols):
        preds = model.predict(new_df[feature_cols])
        new_df["predicted_risk"] = preds
        st.write(new_df.head())
        st.download_button(
            "Download predictions",
            new_df.to_csv(index=False),
            file_name="predicted_risk.csv"
        )
    else:
        st.error("Missing required EEG feature columns.")

# ------------------------------------------------------------
# PREDICTION ERROR PLOT (Actual vs Predicted Risk)
# ------------------------------------------------------------
st.subheader("Prediction Error Plot")

if uploaded and "predicted_risk" in new_df.columns:

    import plotly.graph_objects as go

    fig = go.Figure()

    # Scatter of actual vs predicted
    fig.add_trace(go.Scatter(
        x=new_df["disengagement_risk"],
        y=new_df["predicted_risk"],
        mode="markers",
        marker=dict(size=6, color="#f1580c", opacity=0.7),
        name="Predictions"
    ))

    # Perfect prediction line
    min_val = min(new_df["disengagement_risk"].min(), new_df["predicted_risk"].min())
    max_val = max(new_df["disengagement_risk"].max(), new_df["predicted_risk"].max())

    fig.add_trace(go.Scatter(
        x=[min_val, max_val],
        y=[min_val, max_val],
        mode="lines",
        line=dict(color="black", dash="dash"),
        name="Perfect Prediction"
    ))

    fig.update_layout(
        xaxis_title="Actual Disengagement Risk",
        yaxis_title="Predicted Disengagement Risk",
        title="Actual vs Predicted Disengagement Risk",
        height=500
    )

    st.plotly_chart(fig, use_container_width=True)

else:
    st.write("Upload a file with disengagement_risk + EEG features to view prediction error.")


# ------------------------------------------------------------
# FEATURE IMPORTANCE
# ------------------------------------------------------------
st.markdown("---")
st.subheader("EEG Feature Importance (Random Forest)")

importances = model.feature_importances_
imp_df = pd.DataFrame({"feature": feature_cols, "importance": importances})
fig_imp = px.bar(
    imp_df.sort_values("importance", ascending=False),
    x="importance", y="feature",
    orientation="h",
    color="importance",
    color_continuous_scale="Inferno"
)
st.plotly_chart(fig_imp, use_container_width=True)

# ------------------------------------------------------------
# DISENGAGEMENT RISK CURVE
# ------------------------------------------------------------
st.markdown("---")
st.subheader("Disengagement Risk Over Time")

risk_fig = px.line(
    user_segments,
    x="timestamp", y="disengagement_risk",
    color="session_id",
    color_discrete_sequence=["#f1580c"]
)
st.plotly_chart(risk_fig, use_container_width=True)

# ------------------------------------------------------------
# END
# ------------------------------------------------------------
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666666; padding: 20px;'>
    <p>© DeepHumanX — Adaptive Learning Platform • 2025</p>
</div>
""", unsafe_allow_html=True)
