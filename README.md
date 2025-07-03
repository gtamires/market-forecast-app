# 📈 Market Forecast App

This is a Flask-based web application for forecasting stock prices using machine learning models like LSTM, Random Forest, and GRU. The system fetches historical stock data via Yahoo Finance and provides interactive visualizations and performance metrics.

---

## 🚀 Features

- 🔍 Real-time financial data collection using `yFinance`
- 📊 Visualization of price charts, moving averages, and candlesticks
- 🤖 Forecasts using:
  - LSTM (Long Short-Term Memory)
  - Random Forest
  - GRU (Gated Recurrent Unit)
- 🧪 Model evaluation with metrics: MSE, RMSE, MAE
- 📈 Comparative plots: actual vs. predicted values
- 🧾 CSV export of forecast results
- 💸 Expected return calculation and 95% confidence interval

---

## 🧰 Technologies Used

- Python
- Flask
- yFinance
- scikit-learn
- Keras / TensorFlow
- Matplotlib / Plotly
- Pandas / NumPy

---

## 📦 How to Run the Project Locally

### 1. Clone the repository
```bash
git clone https://github.com/gtamires/market-forecast-app.git
cd market-forecast-app
