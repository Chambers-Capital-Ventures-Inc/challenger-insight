import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from data_utils import (
    load_and_process_data,
    calculate_engagement_curve,
    calculate_disengagement_triggers,
    calculate_key_metrics,
    calculate_disengagement_reasons,
    generate_recommendations
)

# Page configuration
st.set_page_config(
    page_title="Disengagement Analysis Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS for styling
st.markdown("""
<style>
    .main {
        background-color: #6d0000;  /* Dark red background */
        color: white;
    }
    .block-container {
        padding: 2rem 2rem 2rem 2rem;
        background-color: #6d0000;
    }
    /* Only style h1 in specific contexts, not all h1 */
    .main > div > div > div > h1 {
        color: #dbdbdb !important;  /* Light gray for section headers */
    }
    h2, h3 {
        color: #dbdbdb !important;  /* Light gray for h2, h3 */
    }
    [data-testid="stMetricValue"] {
        font-size: 28px;
        color: white;
    }
    /* Make sidebar match */
    [data-testid="stSidebar"] {
        background-color: #6d0000;
    }
    /* Style dividers */
    hr {
        border-color: #dbdbdb;
    }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown("""
<div style='text-align: center; padding: 30px; background-color: #000000; border-radius: 10px; margin-bottom: 20px; border: 2px solid #dbdbdb;'>
    <h1 style='color: #dbdbdb; margin: 0; font-size: 42px; font-weight: bold;'>DISENGAGEMENT ANALYSIS DASHBOARD</h1>
</div>
""", unsafe_allow_html=True)

# Load data
@st.cache_data
def load_dashboard_data():
    """Load and process all data for the dashboard"""
    try:
        # Try to load real data
        segments_df, sessions_df = load_and_process_data()
        use_real_data = True
    except FileNotFoundError:
        st.warning("⚠️ Data files not found. Using demo data. Please add segments_df.csv and sessions_df.csv to use real data.")
        # Create demo data
        segments_df = pd.DataFrame({
            'segment_idx': range(300),
            'disengaged': np.random.choice([0, 1], 300, p=[0.3, 0.7]),
            'session_id': ['session_' + str(i//60) for i in range(300)]
        })
        
        sessions_df = pd.DataFrame({
            'session_id': [f'session_{i}' for i in range(100)],
            'stimulus_type': np.random.choice(['Passive Content', 'Long Session', 'Survey Fatigue', 'Low Relevance'], 100),
            'task_difficulty': np.random.choice(['Easy', 'Moderate', 'Hard'], 100),
            'modality': np.random.choice(['Live Workshop', 'Async Video'], 100),
            'disengaged': np.random.choice([0, 1], 100, p=[0.41, 0.59])
        })
        use_real_data = False
    
    # Calculate all metrics
    time_labels, engagement_pct = calculate_engagement_curve(segments_df)
    triggers, trigger_counts = calculate_disengagement_triggers(sessions_df)
    metrics = calculate_key_metrics(segments_df, sessions_df)
    reasons = calculate_disengagement_reasons(sessions_df, segments_df)
    
    # Generate recommendations (or use defaults for demo)
    if use_real_data:
        recommendations = generate_recommendations(sessions_df, segments_df, metrics)
    else:
        recommendations = [
            "Break passive segment into 7-10 min interactive blocks",
            "Move survey to end of session (reduces 54% mid-drop)",
            "Add relevance cue to preview to prevent early exit"
        ]
    
    return {
        'segments_df': segments_df,
        'sessions_df': sessions_df,
        'time_labels': time_labels,
        'engagement_pct': engagement_pct,
        'triggers': triggers,
        'trigger_counts': trigger_counts,
        'metrics': metrics,
        'reasons': reasons,
        'recommendations': recommendations,
        'use_real_data': use_real_data
    }

data = load_dashboard_data()

# Main disengagement curve
st.markdown("### Disengagement Curve (Engagement % vs Time in Session)")

fig_curve = go.Figure()
fig_curve.add_trace(go.Scatter(
    x=data['time_labels'],
    y=data['engagement_pct'],
    mode='lines+markers',
    line=dict(color='white', width=3),
    marker=dict(size=8, color='white'),
    fill='tonexty',
    fillcolor='rgba(255, 255, 255, 0.1)'
))

fig_curve.update_layout(
    plot_bgcolor='#000000',
    paper_bgcolor='#6d0000',  # Dark red background
    font=dict(color='white', size=12),
    xaxis=dict(
        title='Minutes Into Session',
        gridcolor='#333333',
        showgrid=True,
        zeroline=False
    ),
    yaxis=dict(
        title='% Engaged',
        gridcolor='#333333',
        showgrid=True,
        zeroline=False
    ),
    height=400,
    margin=dict(l=50, r=50, t=20, b=50),
    hovermode='x unified'
)

st.plotly_chart(fig_curve, use_container_width=True)

# Triggers and Why Users Disengage
col1, col2 = st.columns([1, 1])

with col1:
    st.markdown("### Top Disengagement Triggers")
    
    if len(data['triggers']) > 0:
        fig_triggers = go.Figure(go.Bar(
            x=data['trigger_counts'],
            y=data['triggers'],
            orientation='h',
            marker=dict(color='#CCCCCC'),  # Light gray bars
            text=data['trigger_counts'],
            textposition='outside'
        ))
        
        fig_triggers.update_layout(
            plot_bgcolor='#000000',
            paper_bgcolor='#6d0000',  # Dark red background
            font=dict(color='white', size=12),
            xaxis=dict(
                gridcolor='#333333',
                showgrid=True,
                zeroline=False,
                title='Count'
            ),
            yaxis=dict(
                gridcolor='#333333',
                zeroline=False
            ),
            height=300,
            margin=dict(l=150, r=20, t=20, b=40),
            showlegend=False
        )
        
        st.plotly_chart(fig_triggers, use_container_width=True)
    else:
        st.info("No disengagement triggers detected in the current dataset.")

with col2:
    st.markdown("### Why Users Disengage")
    
    # Display reasons
    reasons_html = "<div style='background-color: #6d0000; padding: 20px; border-radius: 10px; margin-top: 20px; border: 2px solid #dbdbdb;'>"
    reasons_html += "<ul style='color: white; font-size: 14px; line-height: 2;'>"
    
    if len(data['reasons']) > 0:
        for reason in data['reasons'][:4]:  # Show top 4 reasons
            reasons_html += f"<li>{reason}</li>"
    else:
        # Default reasons if none calculated
        reasons_html += """
        <li>68% disengage after first passive segment</li>
        <li>54% drop when survey appears mid-session</li>
        <li>47% leave after 25 minutes of nonstop content</li>
        <li>39% lost interest if the topic feels irrelevant</li>
        """
    
    reasons_html += "</ul></div>"
    st.markdown(reasons_html, unsafe_allow_html=True)


# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666666; padding: 20px;'>
    <p>Powered by DeepHumanX & Challenger Insights Inc.</p>
</div>
""", unsafe_allow_html=True)

# Sidebar for additional options
with st.sidebar:
    st.markdown("### Dashboard Settings")
    st.markdown("---")
    
    st.markdown("#### Data Source")
    if data['use_real_data']:
        st.success("✅ Using real data")
    else:
        st.info("ℹ️ Using demo data")
    
    st.markdown("---")
    st.markdown("#### Quick Stats")
    st.metric("Total Sessions", len(data['sessions_df']))
    st.metric("Total Segments", len(data['segments_df']))
    st.metric("Disengagement Rate", f"{data['metrics']['pct_leaving_before_end']}%")
