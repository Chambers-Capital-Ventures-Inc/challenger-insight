import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt

# -------------------------
# 1. Load data & model
# -------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("notebooks/eeg_dashboard.csv")
    return df

@st.cache_resource
def load_model():
    model = joblib.load("notebooks/tuned_rf_disengagement.pkl")
    return model

df = load_data()
model = load_model()

# Define feature columns (EEG only)
feature_cols = [
    "delta_power","theta_power","alpha_power","beta_power","gamma_power",
    "theta_alpha_ratio","beta_alpha_ratio","theta_beta_ratio",
    "spectral_entropy"
]

st.title("Adaptive Learning EEG Dashboard")

# -------------------------
# 2. Sidebar controls
# -------------------------
st.sidebar.header("Controls")

# Subject/session filters (if available)
if "subject_id" in df.columns:
    subjects = ["All"] + sorted(df["subject_id"].unique().tolist())
    chosen_subject = st.sidebar.selectbox("Subject", subjects, index=0)
else:
    chosen_subject = "All"

if "session_id" in df.columns:
    sessions = ["All"] + sorted(df["session_id"].unique().tolist())
    chosen_session = st.sidebar.selectbox("Session", sessions, index=0)
else:
    chosen_session = "All"

# Filter dataframe
filtered_df = df.copy()
if chosen_subject != "All":
    filtered_df = filtered_df[filtered_df["subject_id"] == chosen_subject]
if chosen_session != "All":
    filtered_df = filtered_df[filtered_df["session_id"] == chosen_session]

st.sidebar.write(f"Rows after filtering: {len(filtered_df)}")

# Risk threshold
default_thresh = 0.45
risk_thresh = st.sidebar.slider(
    "Disengagement Risk Threshold",
    min_value=0.0,
    max_value=1.0,
    value=default_thresh,
    step=0.01
)

# -------------------------
# 3. Model predictions
# -------------------------
X = filtered_df[feature_cols]
y_true_risk = filtered_df["disengagement_risk"]

y_pred_risk = model.predict(X)
filtered_df["predicted_risk"] = y_pred_risk

# Binary flags based on threshold
filtered_df["true_disengaged"] = (filtered_df["disengagement_risk"] >= risk_thresh).astype(int)
filtered_df["pred_disengaged"] = (filtered_df["predicted_risk"] >= risk_thresh).astype(int)

# -------------------------
# 4. Overview: Engagement vs Disengagement
# -------------------------
st.header("Engagement vs. Disengagement")

col1, col2 = st.columns(2)

with col1:
    st.subheader("True Disengagement (from risk)")
    disengaged_rate_true = filtered_df["true_disengaged"].mean()
    st.metric("Disengaged segments", f"{disengaged_rate_true*100:.1f}%")

with col2:
    st.subheader("Predicted Disengagement (model)")
    disengaged_rate_pred = filtered_df["pred_disengaged"].mean()
    st.metric("Predicted disengaged", f"{disengaged_rate_pred*100:.1f}%")

# Timeline if timestamp available
if "timestamp" in filtered_df.columns:
    st.subheader("Risk Over Time")
    tmp = filtered_df.sort_values("timestamp")
    st.line_chart(
        tmp.set_index("timestamp")[["disengagement_risk","predicted_risk"]]
    )

# -------------------------
# 5. Brainwave patterns
# -------------------------
st.header("Brainwave Patterns")

# Compare EEG bands for engaged vs disengaged (true, based on risk)
engaged_df = filtered_df[filtered_df["true_disengaged"] == 0]
disengaged_df = filtered_df[filtered_df["true_disengaged"] == 1]

mean_engaged = engaged_df[["delta_power","theta_power","alpha_power","beta_power","gamma_power"]].mean()
mean_disengaged = disengaged_df[["delta_power","theta_power","alpha_power","beta_power","gamma_power"]].mean()

brain_df = pd.DataFrame({
    "Engaged": mean_engaged,
    "Disengaged": mean_disengaged
})

st.bar_chart(brain_df)

st.caption("Average band power by state (based on disengagement_risk threshold).")

# -------------------------
# 6. Model prediction diagnostics
# -------------------------
st.header("Model Predictions")

# Scatter: true vs predicted risk
st.subheader("True vs Predicted Disengagement Risk")

scatter_df = pd.DataFrame({
    "True risk": y_true_risk,
    "Predicted risk": y_pred_risk
})

st.scatter_chart(scatter_df)

# Simple error metrics
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

mse = mean_squared_error(y_true_risk, y_pred_risk)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_true_risk, y_pred_risk)
r2 = r2_score(y_true_risk, y_pred_risk)

st.write(f"**MSE**: {mse:.4f}  |  **RMSE**: {rmse:.4f}  |  **MAE**: {mae:.4f}  |  **R²**: {r2:.3f}")

# Feature importance
st.subheader("Feature Importance")

if hasattr(model, "feature_importances_"):
    importances = model.feature_importances_
    fi_df = pd.DataFrame({
        "feature": feature_cols,
        "importance": importances
    }).sort_values("importance", ascending=False)
    st.bar_chart(fi_df.set_index("feature"))

# -------------------------
# 7. Adaptive learning recommendations
# -------------------------
st.header("Adaptive Learning Recommendations")

# Aggregate indicators for this filtered slice
avg_risk = y_true_risk.mean()
avg_pred_risk = y_pred_risk.mean()
avg_cog_load = filtered_df["cognitive_load_score"].mean() if "cognitive_load_score" in filtered_df.columns else None
avg_engagement_sr = filtered_df["self_report_engagement_5pt"].mean() if "self_report_engagement_5pt" in filtered_df.columns else None
avg_fatigue_sr = filtered_df["self_report_fatigue_5pt"].mean() if "self_report_fatigue_5pt" in filtered_df.columns else None

st.write(f"- **Avg true disengagement risk**: {avg_risk:.3f}")
st.write(f"- **Avg predicted risk**: {avg_pred_risk:.3f}")
if avg_cog_load is not None:
    st.write(f"- **Avg cognitive load score**: {avg_cog_load:.2f}")
if avg_engagement_sr is not None:
    st.write(f"- **Self-report engagement (1–5)**: {avg_engagement_sr:.2f}")
if avg_fatigue_sr is not None:
    st.write(f"- **Self-report fatigue (1–5)**: {avg_fatigue_sr:.2f}")

st.subheader("Recommended Actions")

recommendations = []

# Example rules (you can tune thresholds)
high_risk = avg_pred_risk >= risk_thresh
very_high_risk = avg_pred_risk >= (risk_thresh + 0.15)
high_load = (avg_cog_load is not None) and (avg_cog_load >= 0.7)
low_load = (avg_cog_load is not None) and (avg_cog_load <= 0.3)
high_fatigue = (avg_fatigue_sr is not None) and (avg_fatigue_sr >= 4)

if very_high_risk and high_fatigue:
    recommendations.append("Disengagement and fatigue are both high → recommend a **microbreak (2–5 minutes)** and a brief reflective pause.")
elif very_high_risk and high_load:
    recommendations.append("High disengagement risk and high cognitive load → **simplify the task** or provide **worked examples** before continuing.")
elif high_risk and low_load:
    recommendations.append("High disengagement risk but low cognitive load → learner may be **bored**. Recommend **increasing difficulty** or adding challenge questions.")
elif high_risk:
    recommendations.append("Disengagement risk is elevated → introduce a **short interactive check-in** (poll, quiz, or discussion) to re-engage attention.")
else:
    recommendations.append("Disengagement risk appears moderate or low → maintain current difficulty but **monitor** for spikes in risk over time.")

if recommendations:
    for rec in recommendations:
        st.markdown(f"- {rec}")
