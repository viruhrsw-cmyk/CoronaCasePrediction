# 🦠 COVID-19 Forecasting Dashboard

An interactive Streamlit application for forecasting COVID-19 cases using time series models.

## 🚀 Features

- **Interactive Dashboard**: User-friendly interface for COVID-19 data analysis
- **Multiple Forecasting Models**: SARIMAX, Auto ARIMA, and fallback models
- **Real-time Data**: Integration with Our World in Data COVID-19 dataset
- **Customizable Parameters**: Adjust forecast horizon, seasonality, and transformations
- **Visual Analytics**: Interactive plots with Plotly
- **Performance Metrics**: MAE, RMSE, and MAPE calculations

## 🛠️ Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/covid-forecast-streamlit.git
cd covid-forecast-streamlit
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Run the application:
```bash
streamlit run app.py
```

## �� Deployment

### Deploy to Streamlit Cloud

1. Push your code to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect your GitHub account
4. Select this repository
5. Deploy!

### Deploy to Heroku

1. Create a `Procfile`:
2. Deploy using Heroku CLI or GitHub integration

## �� Usage

1. **Select Data Source**: Choose between Our World in Data or upload your own CSV
2. **Choose Country/Region**: Select from available countries
3. **Set Parameters**: Configure forecast horizon and model parameters
4. **Generate Forecast**: Click to load data and generate predictions
5. **Analyze Results**: View forecasts, metrics, and data analysis

## 🔧 Configuration

- **Forecast Horizon**: 7-30 days
- **Seasonality**: Weekly patterns (7-day cycles)
- **Data Transformation**: Optional log transformation
- **Model Selection**: Automatic model selection with fallbacks

## 📁 Project Structure
covid-forecast-streamlit/
├── app.py # Main Streamlit application
├── src/ # Source code modules
│ ├── data_loader.py # Data loading and preprocessing
│ ├── forecasting.py # Forecasting models
│ └── utils.py # Utility functions
├── requirements.txt # Python dependencies
└── .streamlit/ # Streamlit configuration
## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## �� Acknowledgments

- [Our World in Data](https://ourworldindata.org/) for COVID-19 data
- Streamlit team for the amazing framework
- Open source community for statistical modeling libraries
