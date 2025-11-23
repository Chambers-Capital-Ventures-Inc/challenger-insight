"""
Quick script to export data from your notebook.
Add this cell to your Jupyter notebook to export the data.
"""

# Export segments data
print("Exporting segments_df...")
segments_df.to_csv('segments_df.csv', index=False)
print(f"✅ Exported {len(segments_df)} segments to segments_df.csv")

# Export sessions data
print("\nExporting sessions_df...")
sessions_df.to_csv('sessions_df.csv', index=False)
print(f"✅ Exported {len(sessions_df)} sessions to sessions_df.csv")

print("\n🎉 Data export complete! You can now run your Streamlit dashboard.")
print("\nNext steps:")
print("1. Make sure these CSV files are in the same directory as app.py")
print("2. Run: streamlit run app.py")
