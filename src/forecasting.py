import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error
import warnings
warnings.filterwarnings('ignore')

try:
    from pmdarima import auto_arima
    PMDARIMA_AVAILABLE = True
except ImportError:
    PMDARIMA_AVAILABLE = False

try:
    import statsmodels.api as sm
    from statsmodels.tsa.statespace.sarimax import SARIMAX
    STATSMODELS_AVAILABLE = True
except ImportError:
    STATSMODELS_AVAILABLE = False

class COVIDForecaster:
    """Class for COVID-19 time series forecasting"""
    
    def __init__(self, use_seasonality=True, transform_data=True):
        self.use_seasonality = use_seasonality
        self.transform_data = transform_data
        self.model = None
        self.fitted_model = None
        
    def preprocess_data(self, data, target_var):
        """Preprocess the data for forecasting"""
        # Ensure data is sorted by date
        data = data.sort_values('date').reset_index(drop=True)
        
        # Create time series
        ts_data = data[target_var].values
        
        # Apply log transformation if requested
        if self.transform_data:
            ts_data = np.log1p(ts_data)  # log(1+x) to handle zeros
            
        return ts_data
    
    def fit_sarimax(self, data, target_var):
        """Fit SARIMAX model to the data"""
        if not STATSMODELS_AVAILABLE:
            raise ImportError("Statsmodels is required for SARIMAX modeling")
        
        ts_data = self.preprocess_data(data, target_var)
        
        # Determine model parameters
        if self.use_seasonality:
            # Weekly seasonality (7 days)
            seasonal_period = 7
            order = (1, 1, 1)
            seasonal_order = (1, 0, 1, seasonal_period)
        else:
            order = (1, 1, 1)
            seasonal_order = (0, 0, 0, 0)
        
        try:
            # Fit SARIMAX model
            model = SARIMAX(
                ts_data,
                order=order,
                seasonal_order=seasonal_order,
                enforce_stationarity=False,
                enforce_invertibility=False
            )
            
            fitted_model = model.fit(disp=False)
            self.fitted_model = fitted_model
            
            return True
            
        except Exception as e:
            print(f"Error fitting SARIMAX model: {e}")
            return False
    
    def fit_auto_arima(self, data, target_var):
        """Fit auto ARIMA model using pmdarima"""
        if not PMDARIMA_AVAILABLE:
            raise ImportError("pmdarima is required for auto ARIMA")
        
        ts_data = self.preprocess_data(data, target_var)
        
        try:
            # Auto ARIMA with seasonal components
            if self.use_seasonality:
                model = auto_arima(
                    ts_data,
                    seasonal=True,
                    m=7,  # Weekly seasonality
                    stepwise=True,
                    suppress_warnings=True,
                    error_action='ignore',
                    max_p=3, max_q=3, max_P=2, max_Q=2
                )
            else:
                model = auto_arima(
                    ts_data,
                    seasonal=False,
                    stepwise=True,
                    suppress_warnings=True,
                    error_action='ignore',
                    max_p=3, max_q=3
                )
            
            self.model = model
            return True
            
        except Exception as e:
            print(f"Error fitting auto ARIMA model: {e}")
            return False
    
    def forecast(self, data, target_var, horizon):
        """Generate forecast for the specified horizon"""
        
        if data is None or data.empty:
            return None
        
        try:
            # Try SARIMAX first
            if self.fit_sarimax(data, target_var):
                return self._generate_sarimax_forecast(data, target_var, horizon)
            
            # Fallback to auto ARIMA
            elif PMDARIMA_AVAILABLE and self.fit_auto_arima(data, target_var):
                return self._generate_arima_forecast(data, target_var, horizon)
            
            else:
                # Simple moving average fallback
                return self._generate_simple_forecast(data, target_var, horizon)
                
        except Exception as e:
            print(f"Error in forecasting: {e}")
            return self._generate_simple_forecast(data, target_var, horizon)
    
    def _generate_sarimax_forecast(self, data, target_var, horizon):
        """Generate forecast using fitted SARIMAX model"""
        try:
            # Get forecast
            forecast = self.fitted_model.get_forecast(steps=horizon)
            forecast_values = forecast.predicted_mean.values
            
            # Inverse transform if log transformation was applied
            if self.transform_data:
                forecast_values = np.expm1(forecast_values)
            
            # Get confidence intervals
            conf_int = forecast.conf_int()
            lower_ci = np.expm1(conf_int.iloc[:, 0].values) if self.transform_data else conf_int.iloc[:, 0].values
            upper_ci = np.expm1(conf_int.iloc[:, 1].values) if self.transform_data else conf_int.iloc[:, 1].values
            
            # Calculate metrics on training data
            ts_data = self.preprocess_data(data, target_var)
            fitted_values = self.fitted_model.fittedvalues
            
            if self.transform_data:
                fitted_values = np.expm1(fitted_values)
                actual_values = data[target_var].values
            else:
                actual_values = data[target_var].values
            
            # Calculate metrics
            metrics = self._calculate_metrics(actual_values, fitted_values)
            
            return {
                'forecast': forecast_values,
                'lower_ci': lower_ci,
                'upper_ci': upper_ci,
                'metrics': metrics,
                'model_type': 'SARIMAX'
            }
            
        except Exception as e:
            print(f"Error generating SARIMAX forecast: {e}")
            return None
    
    def _generate_arima_forecast(self, data, target_var, horizon):
        """Generate forecast using fitted ARIMA model"""
        try:
            # Get forecast
            forecast_values = self.model.predict(n_periods=horizon)
            
            # Inverse transform if log transformation was applied
            if self.transform_data:
                forecast_values = np.expm1(forecast_values)
            
            # Calculate metrics
            ts_data = self.preprocess_data(data, target_var)
            fitted_values = self.model.predict_in_sample()
            
            if self.transform_data:
                fitted_values = np.expm1(fitted_values)
                actual_values = data[target_var].values
            else:
                actual_values = data[target_var].values
            
            metrics = self._calculate_metrics(actual_values, fitted_values)
            
            return {
                'forecast': forecast_values,
                'metrics': metrics,
                'model_type': 'Auto ARIMA'
            }
            
        except Exception as e:
            print(f"Error generating ARIMA forecast: {e}")
            return None
    
    def _generate_simple_forecast(self, data, target_var, horizon):
        """Generate simple forecast using moving average"""
        try:
            # Use simple moving average
            window_size = min(7, len(data) // 4)
            ma_values = data[target_var].rolling(window=window_size, min_periods=1).mean()
            
            # Simple trend projection
            last_values = ma_values.tail(window_size).values
            if len(last_values) > 1:
                trend = np.mean(np.diff(last_values))
                forecast_values = [last_values[-1] + trend * (i + 1) for i in range(horizon)]
            else:
                forecast_values = [last_values[-1]] * horizon
            
            # Ensure non-negative values
            forecast_values = np.maximum(forecast_values, 0)
            
            # Calculate simple metrics
            metrics = {
                'mae': mean_absolute_error(data[target_var].values, ma_values.values),
                'rmse': np.sqrt(mean_squared_error(data[target_var].values, ma_values.values)),
                'mape': mean_absolute_percentage_error(data[target_var].values, ma_values.values) * 100
            }
            
            return {
                'forecast': forecast_values,
                'metrics': metrics,
                'model_type': 'Simple Moving Average'
            }
            
        except Exception as e:
            print(f"Error generating simple forecast: {e}")
            return None
    
    def _calculate_metrics(self, actual, predicted):
        """Calculate forecast accuracy metrics"""
        try:
            # Remove any NaN values
            mask = ~(np.isnan(actual) | np.isnan(predicted))
            actual_clean = actual[mask]
            predicted_clean = predicted[mask]
            
            if len(actual_clean) == 0:
                return {}
            
            metrics = {
                'mae': mean_absolute_error(actual_clean, predicted_clean),
                'rmse': np.sqrt(mean_squared_error(actual_clean, predicted_clean)),
                'mape': mean_absolute_percentage_error(actual_clean, predicted_clean) * 100
            }
            
            return metrics
            
        except Exception as e:
            print(f"Error calculating metrics: {e}")
            return {}
