"""
Data processing utilities for the disengagement dashboard.
This script loads and processes data from your notebook's CSV files.
"""

import pandas as pd
import numpy as np

def load_and_process_data(segments_path='segments_df.csv', sessions_path='sessions_df.csv'):
    """
    Load the segments data. We'll aggregate to session level ourselves.
    
    Parameters:
    -----------
    segments_path : str
        Path to the segments CSV file
    sessions_path : str
        Not used - we aggregate from segments
        
    Returns:
    --------
    segments_df : DataFrame
        Processed segments data
    sessions_df : DataFrame
        Aggregated session-level data
    """
    # Load segments data
    segments_df = pd.read_csv(segments_path)
    
    # Convert timestamp to datetime if needed
    if 'timestamp' in segments_df.columns:
        segments_df['timestamp'] = pd.to_datetime(segments_df['timestamp'])
    
    # Aggregate to session level
    sessions_df = segments_df.groupby('session_id').agg({
        'subject_id': 'first',
        'stimulus_type': 'first',
        'task_difficulty': 'first',
        'modality': 'first',
        'disengaged': 'max',  # Session is disengaged if ANY segment is disengaged
        'segment_idx': 'count'  # Number of segments in session
    }).reset_index()
    
    sessions_df.rename(columns={'segment_idx': 'n_segments'}, inplace=True)
    
    return segments_df, sessions_df

def calculate_engagement_curve(segments_df, time_bins=5):
    """
    Calculate engagement percentage over time in actual minutes.
    
    Parameters:
    -----------
    segments_df : DataFrame
        Segments data with 'segment_idx' and 'disengaged' columns
    time_bins : int
        Number of time bins to divide the session into
        
    Returns:
    --------
    time_labels : list
        Labels for time points in minutes
    engagement_pct : list
        Engagement percentages at each time point
    """
    # Assuming each segment is 10 seconds (0.167 minutes)
    SEGMENT_DURATION_MINUTES = 10 / 60
    
    # Group segments into time bins
    max_segment = segments_df['segment_idx'].max()
    max_time_minutes = max_segment * SEGMENT_DURATION_MINUTES
    
    time_bin_minutes = max_time_minutes / time_bins
    
    time_labels = []
    engagement_pct = []
    
    for i in range(time_bins):
        start_time = i * time_bin_minutes
        end_time = (i + 1) * time_bin_minutes
        
        # Convert time back to segment indices
        start_idx = start_time / SEGMENT_DURATION_MINUTES
        end_idx = end_time / SEGMENT_DURATION_MINUTES
        
        # Filter segments in this time bin
        bin_segments = segments_df[
            (segments_df['segment_idx'] >= start_idx) & 
            (segments_df['segment_idx'] < end_idx)
        ]
        
        # Calculate engagement (percentage NOT disengaged)
        if len(bin_segments) > 0:
            engaged_pct = (1 - bin_segments['disengaged'].mean()) * 100
            engagement_pct.append(round(engaged_pct, 1))
        else:
            engagement_pct.append(0)
        
        # Create label showing minute range
        time_labels.append(f"{int(start_time)}-{int(end_time)} min")
    
    return time_labels, engagement_pct

def calculate_disengagement_triggers(sessions_df):
    """
    Calculate the frequency of different disengagement triggers.
    
    Parameters:
    -----------
    sessions_df : DataFrame
        Sessions data with 'stimulus_type' and 'disengaged' columns
        
    Returns:
    --------
    triggers : list
        List of trigger names
    trigger_counts : list
        Frequency of each trigger
    """
    # Filter to only disengaged sessions
    disengaged_sessions = sessions_df[sessions_df['disengaged'] == 1]
    
    if len(disengaged_sessions) > 0 and 'stimulus_type' in disengaged_sessions.columns:
        # Count by stimulus type (the main trigger according to feature importance)
        trigger_counts_series = disengaged_sessions['stimulus_type'].value_counts()
        
        return trigger_counts_series.index.tolist(), trigger_counts_series.values.tolist()
    else:
        # Return empty if no data
        return [], []

def calculate_key_metrics(segments_df, sessions_df):
    """
    Calculate key dashboard metrics.
    
    Returns:
    --------
    dict with keys:
        - disengagement_starts: Minutes when disengagement typically starts
        - critical_dropoff: Minutes when critical drop-off occurs
        - pct_leaving_before_end: Percentage leaving before completion
        - most_common_trigger: Most common disengagement trigger
    """
    # Percentage of sessions with disengagement
    pct_leaving = (sessions_df['disengaged'].mean()) * 100
    
    # Find when disengagement starts (first quartile of disengaged segments)
    disengaged_segments = segments_df[segments_df['disengaged'] == 1]
    if len(disengaged_segments) > 0:
        # Assuming each segment is ~10 seconds, convert to minutes
        disengagement_starts = int(disengaged_segments['segment_idx'].quantile(0.25) * 10 / 60)
        critical_dropoff = int(disengaged_segments['segment_idx'].quantile(0.75) * 10 / 60)
    else:
        disengagement_starts = 0
        critical_dropoff = 0
    
    # Most common trigger among disengaged sessions
    disengaged_sessions = sessions_df[sessions_df['disengaged'] == 1]
    if len(disengaged_sessions) > 0 and 'stimulus_type' in disengaged_sessions.columns:
        most_common_trigger = disengaged_sessions['stimulus_type'].value_counts().index[0]
    else:
        most_common_trigger = "Unknown"
    
    return {
        'disengagement_starts': disengagement_starts,
        'critical_dropoff': critical_dropoff,
        'pct_leaving_before_end': int(pct_leaving),
        'most_common_trigger': most_common_trigger
    }

def calculate_disengagement_reasons(sessions_df, segments_df):
    """
    Calculate statistics about why users disengage based on actual data patterns.
    
    Returns:
    --------
    reasons : list of str
        List of formatted reason strings
    """
    reasons = []
    
    # Calculate disengagement rate by stimulus type
    if 'stimulus_type' in sessions_df.columns:
        for stimulus_type in sessions_df['stimulus_type'].unique():
            subset = sessions_df[sessions_df['stimulus_type'] == stimulus_type]
            if len(subset) > 5:  # Need enough samples
                pct = (subset['disengaged'].mean()) * 100
                if pct > 50:  # Show high disengagement patterns
                    reasons.append(f"{int(pct)}% disengage during {stimulus_type}")
    
    # Calculate disengagement by modality
    if 'modality' in sessions_df.columns:
        for modality in sessions_df['modality'].unique():
            subset = sessions_df[sessions_df['modality'] == modality]
            if len(subset) > 5:
                pct = (subset['disengaged'].mean()) * 100
                if pct > 50:
                    reasons.append(f"{int(pct)}% drop in {modality} format")
    
    # Calculate disengagement by task difficulty
    if 'task_difficulty' in sessions_df.columns:
        for difficulty in sessions_df['task_difficulty'].unique():
            subset = sessions_df[sessions_df['task_difficulty'] == difficulty]
            if len(subset) > 5:
                pct = (subset['disengaged'].mean()) * 100
                if pct > 50:
                    reasons.append(f"{int(pct)}% disengage when content is {difficulty}")
    
    # Add segment-level insight about when disengagement happens
    disengaged_segs = segments_df[segments_df['disengaged'] == 1]
    if len(disengaged_segs) > 0:
        avg_disengage_time = disengaged_segs['segment_idx'].mean() * 10 / 60  # Convert to minutes
        reasons.append(f"Average disengagement occurs at {int(avg_disengage_time)} minutes into session")
    
    # If no strong patterns found, return general insights
    if len(reasons) == 0:
        overall_rate = sessions_df['disengaged'].mean() * 100
        reasons = [
            f"{int(overall_rate)}% of sessions show disengagement",
            "Patterns vary across different content types",
            "Engagement levels fluctuate throughout sessions"
        ]
    
    return reasons[:4]  # Return top 4

def generate_recommendations(sessions_df, segments_df, metrics):
    """
    Generate data-driven recommendations based on disengagement patterns.
    
    Returns:
    --------
    recommendations : list of str
        List of actionable recommendations
    """
    recommendations = []
    
    # Recommendation based on most common trigger
    if metrics['most_common_trigger'] != "Unknown":
        trigger = metrics['most_common_trigger']
        if 'Discussion' in trigger:
            recommendations.append(f"Make {trigger} sessions more interactive with Q&A breaks every 7-10 minutes")
        elif 'Video' in trigger or 'Lecture' in trigger:
            recommendations.append(f"Break {trigger} content into shorter segments with engagement checks")
        elif 'Lab' in trigger or 'Code' in trigger:
            recommendations.append(f"Add guidance prompts during {trigger} to maintain engagement")
        else:
            recommendations.append(f"Redesign {trigger} format to increase interactivity")
    
    # Recommendation based on when disengagement starts
    if metrics['disengagement_starts'] > 0:
        start_time = metrics['disengagement_starts']
        recommendations.append(f"Add engagement break at {start_time} minute mark (before disengagement typically starts)")
    
    # Recommendation based on modality if available
    if 'modality' in sessions_df.columns:
        modality_disengage = sessions_df.groupby('modality')['disengaged'].mean()
        if len(modality_disengage) > 0:
            worst_modality = modality_disengage.idxmax()
            pct = modality_disengage.max() * 100
            if pct > 60:
                recommendations.append(f"Redesign {worst_modality} format ({int(pct)}% disengagement rate)")
    
    # Default recommendations if we don't have enough data
    if len(recommendations) == 0:
        recommendations = [
            "Implement regular engagement checks throughout sessions",
            "Vary content format to maintain student interest",
            "Monitor real-time engagement data to intervene early"
        ]
    
    return recommendations[:3]  # Return top 3
