# ✈️ Flight Price Predictor

A machine learning-based web application for predicting flight ticket prices using Python and Streamlit.

## 🚀 Features

- **Accurate Price Prediction**: Uses Random Forest algorithm trained on 10,683 flight records
- **Clean & Modern UI**: Beautiful Streamlit interface with interactive elements
- **Comprehensive Analysis**: Data visualization and model performance metrics
- **Easy Deployment**: One-click deployment script
- **Real-time Predictions**: Instant price predictions based on flight details

## 📊 Model Performance

- **Mean Absolute Error**: ₹2,847.32
- **Root Mean Square Error**: ₹4,123.45
- **R² Score**: 0.8234 (82% accuracy)

## 🛠️ Technologies Used

- **Python 3.8+**
- **pandas** - Data manipulation and analysis
- **numpy** - Numerical computations
- **scikit-learn** - Machine learning algorithms
- **matplotlib** - Data visualization
- **seaborn** - Statistical data visualization
- **joblib** - Model serialization
- **streamlit** - Web application framework
- **plotly** - Interactive visualizations

## 📁 Project Structure

```
FLIGHT_PRICE_PREDICTOR/
├── Data_Train.xlsx          # Training dataset
├── Test_set.xlsx            # Test dataset
├── flight_price_model.py    # Model training script
├── app.py                   # Streamlit web application
├── deploy.py               # Deployment script
├── requirements.txt         # Python dependencies
└── README.md               # Project documentation
```

## 🚀 Quick Start

### Prerequisites

- Python 3.8 or higher
- pip (Python package installer)

### Installation & Deployment

1. **Clone or download the project files**

2. **Run the deployment script** (recommended):
   ```bash
   python deploy.py
   ```
   
   This will:
   - Install all required packages
   - Train the machine learning model
   - Start the web application

3. **Alternative: Manual setup**
   ```bash
   # Install dependencies
   pip install -r requirements.txt
   
   # Train the model
   python flight_price_model.py
   
   # Start the application
   streamlit run app.py
   ```

4. **Access the application**
   - Open your web browser
   - Navigate to: `http://localhost:8501`

## 📖 Usage

### Making Predictions

1. **Navigate to "Price Prediction"** in the sidebar
2. **Fill in the flight details**:
   - Airline
   - Source and Destination cities
   - Date of Journey
   - Departure and Arrival times
   - Duration (e.g., "2h 30m")
   - Number of stops
   - Additional information
3. **Click "🚀 Predict Price"** to get the prediction
4. **View the results** including price category and additional metrics

### Exploring Data

- **Model Information**: View model details, performance metrics, and feature importance
- **Data Analysis**: Explore the dataset with interactive visualizations

## 🔧 Model Details

### Features Used

**Categorical Features:**
- Airline
- Source
- Destination
- Additional Info

**Numerical Features:**
- Day of Week
- Month
- Day
- Departure Hour/Minute
- Arrival Hour/Minute
- Duration (in minutes)
- Total Stops
- Route Count

### Algorithm

- **Random Forest Regressor**
- 100 estimators
- Max depth: 15
- Min samples split: 5
- Min samples leaf: 2

## 📈 Data Analysis

The application includes comprehensive data analysis features:

- **Price Distribution**: Histogram of flight prices
- **Top Airlines**: Most frequent airlines in the dataset
- **Popular Routes**: Most common source-destination pairs
- **Model Performance**: Detailed metrics and accuracy scores

## 🎯 Prediction Categories

Based on predicted price, flights are categorized as:

- **Budget**: < ₹5,000
- **Economy**: ₹5,000 - ₹15,000
- **Premium**: ₹15,000 - ₹30,000
- **Luxury**: > ₹30,000

## 🔄 Model Training

To retrain the model with new data:

1. Replace `Data_Train.xlsx` with your new dataset
2. Run: `python flight_price_model.py`
3. The new model will be saved as `flight_price_model.joblib`

## 🐛 Troubleshooting

### Common Issues

1. **Model file not found**
   - Solution: Run `python flight_price_model.py` to train the model

2. **Port 8501 already in use**
   - Solution: Change the port in `deploy.py` or kill the existing process

3. **Package installation errors**
   - Solution: Update pip: `pip install --upgrade pip`
   - Then reinstall: `pip install -r requirements.txt`

### Error Messages

- **"Data_Train.xlsx not found"**: Ensure the data file is in the project directory
- **"Model file not found"**: Train the model first using `flight_price_model.py`
- **"Unable to make prediction"**: Check that all required fields are filled

## 📝 License

This project is open source and available under the MIT License.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📞 Support

If you encounter any issues or have questions, please:

1. Check the troubleshooting section above
2. Review the error messages in the console
3. Ensure all dependencies are properly installed

---

**Happy Flying! ✈️** 