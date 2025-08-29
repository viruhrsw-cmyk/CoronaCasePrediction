import streamlit as st
import pandas as pd

def create_metrics_display(metrics):
    """Create a nice display for forecast metrics"""
    
    if not metrics:
        st.warning("No metrics available")
        return
    
    # Create columns for metrics
    cols = st.columns(len(metrics))
    
    for i, (metric_name, metric_value) in enumerate(metrics.items()):
        with cols[i]:
            # Format metric name
            display_name = metric_name.upper().replace('_', ' ')
            
            # Format metric value
            if isinstance(metric_value, float):
                if metric_value < 1:
                    formatted_value = f"{metric_value:.4f}"
                elif metric_value < 100:
                    formatted_value = f"{metric_value:.2f}"
                else:
                    formatted_value = f"{metric_value:.0f}"
            else:
                formatted_value = str(metric_value)
            
            # Display metric
            st.metric(
                label=display_name,
                value=formatted_value
            )

def create_download_button(data, filename):
    """Create a download button for data"""
    
    if data is not None and not data.empty:
        csv = data.to_csv(index=False)
        st.download_button(
            label="📥 Download Data",
            data=csv,
            file_name=filename,
            mime="text/csv"
        )

def display_data_info(data, target_var):
    """Display information about the loaded data"""
    
    if data is None or data.empty:
        st.warning("No data available")
        return
    
    # Basic info
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Total Days", len(data))
    
    with col2:
        st.metric("Date Range", f"{data['date'].min().strftime('%Y-%m-%d')} to {data['date'].max().strftime('%Y-%m-%d')}")
    
    with col3:
        st.metric("Missing Values", data[target_var].isna().sum())
    
    # Data statistics
    st.subheader("Data Statistics")
    st.write(data[target_var].describe())
