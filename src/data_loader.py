import pandas as pd
import numpy as np
import requests
from datetime import datetime, timedelta
import streamlit as st

class DataLoader:
    """Class to handle data loading and preprocessing for COVID-19 data"""
    
    def __init__(self):
        self.owid_url = "https://covid.ourworldindata.org/data/owid-covid-data.csv"
        self.cache = {}
        
    def get_available_countries(self):
        """Get list of available countries from OWID dataset"""
        try:
            # Load a small sample to get country list
            sample_data = pd.read_csv(self.owid_url, nrows=1000)
            countries = sorted(sample_data['location'].unique().tolist())
            return countries
        except Exception as e:
            st.error(f"Error loading countries: {str(e)}")
            return ["India", "United States", "United Kingdom", "Germany", "France"]
    
    def load_country_data(self, country, target_var, start_date=None, end_date=None):
        """Load COVID-19 data for a specific country and target variable"""
        
        cache_key = f"{country}_{target_var}_{start_date}_{end_date}"
        if cache_key in self.cache:
            return self.cache[cache_key]
        
        try:
            # Load data
            data = pd.read_csv(self.owid_url)
            
            # Filter for specific country
            country_data = data[data['location'] == country].copy()
            
            if country_data.empty:
                st.error(f"No data found for {country}")
                return None
            
            # Convert date column
            country_data['date'] = pd.to_datetime(country_data['date'])
            
            # Filter by date range if specified
            if start_date:
                country_data = country_data[country_data['date'] >= pd.to_datetime(start_date)]
            if end_date:
                country_data = country_data[country_data['date'] <= pd.to_datetime(end_date)]
            
            # Sort by date
            country_data = country_data.sort_values('date').reset_index(drop=True)
            
            # Clean target variable
            if target_var in country_data.columns:
                # Replace negative values with NaN
                country_data.loc[country_data[target_var] < 0, target_var] = np.nan
                
                # Forward fill missing values
                country_data[target_var] = country_data[target_var].fillna(method='ffill')
                
                # Remove rows with still missing values
                country_data = country_data.dropna(subset=[target_var])
                
                if len(country_data) < 30:
                    st.warning(f"Limited data available for {country}: {len(country_data)} days")
                    return None
                
                # Cache the result
                self.cache[cache_key] = country_data
                return country_data
            else:
                st.error(f"Target variable '{target_var}' not found in data")
                return None
                
        except Exception as e:
            st.error(f"Error loading data: {str(e)}")
            return None
    
    def get_data_summary(self, data, target_var):
        """Get summary statistics for the data"""
        if data is None or data.empty:
            return None
            
        summary = {
            'total_days': len(data),
            'date_range': f"{data['date'].min().strftime('%Y-%m-%d')} to {data['date'].max().strftime('%Y-%m-%d')}",
            'mean': data[target_var].mean(),
            'std': data[target_var].std(),
            'min': data[target_var].min(),
            'max': data[target_var].max(),
            'missing_values': data[target_var].isna().sum()
        }
        
        return summary
