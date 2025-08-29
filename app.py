import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

from src.data_loader import DataLoader
from src.forecasting import COVIDForecaster
from src.utils import create_metrics_display

# Page configuration
st.set_page_config(
    page_title="COVID-19 Forecasting Dashboard",
    page_icon="��",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .forecast-plot {
        background-color: white;
        padding: 1rem;
        border-radius: 0.5rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
</style>
""", unsafe_allow_html=True)

def main():
    # Header
    st.markdown('<h1 class="main-header">🦠 COVID-19 Forecasting Dashboard</h1>', unsafe_allow_html=True)
    
    # Sidebar
    st.sidebar.header("📊 Configuration")
    
    # Data source selection
    data_source = st.sidebar.selectbox(
        "Data Source",
        ["Our World in Data", "Upload CSV"],
        help="Choose data source for COVID-19 data"
    )
    
    if data_source == "Our World in Data":
        st.sidebar.info("Using Our World in Data COVID-19 dataset")
        data_loader = DataLoader()
        
        # Country/Region selection
        countries = data_loader.get_available_countries()
        selected_country = st.sidebar.selectbox(
            "Select Country/Region",
            countries,
            index=countries.index("India") if "India" in countries else 0
        )
        
        # Target variable selection
        target_var = st.sidebar.selectbox(
            "Target Variable",
            ["new_cases", "new_cases_smoothed", "new_deaths", "new_deaths_smoothed"],
            index=1,
            help="Select the variable to forecast"
        )
        
        # Forecast horizon
        forecast_horizon = st.sidebar.slider(
            "Forecast Horizon (days)",
            min_value=7,
            max_value=30,
            value=14,
            step=7
        )
        
        # Model parameters
        st.sidebar.subheader("Model Parameters")
        use_seasonality = st.sidebar.checkbox("Use Weekly Seasonality", value=True)
        transform_data = st.sidebar.checkbox("Log Transform Data", value=True)
        
        # Load data
        if st.sidebar.button("🔄 Load Data & Forecast"):
            with st.spinner("Loading data and generating forecasts..."):
                try:
                    # Load data
                    data = data_loader.load_country_data(selected_country, target_var)
                    
                    if data is not None and not data.empty:
                        st.success(f"✅ Data loaded successfully for {selected_country}")
                        
                        # Initialize forecaster
                        forecaster = COVIDForecaster(
                            use_seasonality=use_seasonality,
                            transform_data=transform_data
                        )
                        
                        # Generate forecast
                        forecast_result = forecaster.forecast(
                            data, 
                            target_var, 
                            forecast_horizon
                        )
                        
                        if forecast_result:
                            # Display results
                            display_results(data, forecast_result, selected_country, target_var)
                        else:
                            st.error("❌ Forecasting failed. Please check the data and parameters.")
                    else:
                        st.error("❌ No data available for the selected country and variable.")
                        
                except Exception as e:
                    st.error(f"❌ Error: {str(e)}")
    
    else:
        st.sidebar.info("Upload your own CSV file")
        uploaded_file = st.sidebar.file_uploader(
            "Choose a CSV file",
            type=['csv'],
            help="Upload CSV with columns: date, location, new_cases, etc."
        )
        
        if uploaded_file is not None:
            try:
                data = pd.read_csv(uploaded_file)
                st.success("✅ File uploaded successfully!")
                st.write("Data preview:", data.head())
            except Exception as e:
                st.error(f"❌ Error reading file: {str(e)}")

def display_results(data, forecast_result, country, target_var):
    """Display forecasting results"""
    
    # Create tabs for different views
    tab1, tab2, tab3, tab4 = st.tabs(["📈 Forecast", "📊 Metrics", "🔍 Analysis", "📋 Data"])
    
    with tab1:
        st.subheader(f"Forecast for {country} - {target_var.replace('_', ' ').title()}")
        
        # Create forecast plot
        fig = create_forecast_plot(data, forecast_result, target_var)
        st.plotly_chart(fig, use_container_width=True)
        
        # Forecast summary
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Forecast Horizon", f"{len(forecast_result['forecast'])} days")
        with col2:
            st.metric("Last Actual Value", f"{data[target_var].iloc[-1]:.0f}")
        with col3:
            st.metric("First Forecast Value", f"{forecast_result['forecast'][0]:.0f}")
    
    with tab2:
        st.subheader("Model Performance Metrics")
        
        if 'metrics' in forecast_result:
            metrics = forecast_result['metrics']
            create_metrics_display(metrics)
        else:
            st.info("No metrics available for this forecast.")
    
    with tab3:
        st.subheader("Data Analysis")
        
        # Time series decomposition
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Data Statistics**")
            st.write(data[target_var].describe())
        
        with col2:
            st.write("**Trend Analysis**")
            # Simple trend calculation
            data_clean = data[target_var].dropna()
            if len(data_clean) > 1:
                trend = np.polyfit(range(len(data_clean)), data_clean, 1)[0]
                st.write(f"Trend slope: {trend:.2f}")
                if trend > 0:
                    st.write("📈 Increasing trend")
                else:
                    st.write("📉 Decreasing trend")
    
    with tab4:
        st.subheader("Raw Data")
        st.dataframe(data, use_container_width=True)
        
        # Download button
        csv = data.to_csv(index=False)
        st.download_button(
            label="📥 Download Data as CSV",
            data=csv,
            file_name=f"{country}_{target_var}_data.csv",
            mime="text/csv"
        )

def create_forecast_plot(data, forecast_result, target_var):
    """Create interactive forecast plot using Plotly"""
    
    # Prepare data for plotting
    dates = pd.date_range(
        start=data['date'].iloc[-1] + timedelta(days=1),
        periods=len(forecast_result['forecast']),
        freq='D'
    )
    
    forecast_df = pd.DataFrame({
        'date': dates,
        'forecast': forecast_result['forecast'],
        'lower_ci': forecast_result.get('lower_ci', forecast_result['forecast'] * 0.8),
        'upper_ci': forecast_result.get('upper_ci', forecast_result['forecast'] * 1.2)
    })
    
    # Create subplot
    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=('Forecast', 'Forecast vs Actual'),
        vertical_spacing=0.1,
        row_heights=[0.7, 0.3]
    )
    
    # Main forecast plot
    fig.add_trace(
        go.Scatter(
            x=data['date'],
            y=data[target_var],
            mode='lines+markers',
            name='Actual',
            line=dict(color='blue', width=2),
            marker=dict(size=4)
        ),
        row=1, col=1
    )
    
    fig.add_trace(
        go.Scatter(
            x=forecast_df['date'],
            y=forecast_df['forecast'],
            mode='lines+markers',
            name='Forecast',
            line=dict(color='red', width=2, dash='dash'),
            marker=dict(size=4)
        ),
        row=1, col=1
    )
    
    # Confidence interval
    fig.add_trace(
        go.Scatter(
            x=forecast_df['date'].tolist() + forecast_df['date'].tolist()[::-1],
            y=forecast_df['upper_ci'].tolist() + forecast_df['lower_ci'].tolist()[::-1],
            fill='toself',
            fillcolor='rgba(255,0,0,0.2)',
            line=dict(color='rgba(255,255,255,0)'),
            name='Confidence Interval',
            showlegend=False
        ),
        row=1, col=1
    )
    
    # Forecast vs Actual comparison (last 30 days)
    recent_data = data.tail(30)
    recent_forecast = forecast_df.head(30)
    
    fig.add_trace(
        go.Scatter(
            x=recent_data['date'],
            y=recent_data[target_var],
            mode='lines+markers',
            name='Recent Actual',
            line=dict(color='blue', width=2),
            showlegend=False
        ),
        row=2, col=1
    )
    
    fig.add_trace(
        go.Scatter(
            x=recent_forecast['date'],
            y=recent_forecast['forecast'],
            mode='lines+markers',
            name='Recent Forecast',
            line=dict(color='red', width=2, dash='dash'),
            showlegend=False
        ),
        row=2, col=1
    )
    
    # Update layout
    fig.update_layout(
        title=f'COVID-19 {target_var.replace("_", " ").title()} Forecast',
        xaxis_title='Date',
        yaxis_title=target_var.replace('_', ' ').title(),
        height=600,
        showlegend=True,
        hovermode='x unified'
    )
    
    fig.update_xaxes(title_text="Date", row=2, col=1)
    fig.update_yaxes(title_text=target_var.replace('_', ' ').title(), row=2, col=1)
    
    return fig

if __name__ == "__main__":
    main()
