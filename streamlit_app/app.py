import streamlit as st
import pandas as pd
import plotly.express as px

st.set_page_config(page_title="Challenger Insight", layout="wide")

st.title("🧠 Challenger Insight – EEG Engagement Dashboard")

# Sidebar upload box
uploaded_file = st.sidebar.file_uploader("Upload EEG CSV", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.subheader("Raw Data")
    st.dataframe(df.head())

    # Placeholder visualization
    st.subheader("PCA Visualization (Demo)")
    fig = px.scatter(df, x=df.columns[0], y=df.columns[1])
    st.plotly_chart(fig, use_container_width=True)

else:
    st.info("Upload EEG CSV to begin analysis.")
