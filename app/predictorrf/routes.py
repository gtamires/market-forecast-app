# /predictorrf/routes.py

from flask import Blueprint, request, render_template
import os, csv, datetime
import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib
import matplotlib.dates as mdates
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from scipy import stats

# Blueprint
predictorrf_bp = Blueprint('predictorrf', __name__)

# Constants
STOCK_OPTIONS = ["AAPL", "MSFT", "GOOGL", "TSLA", "AMZN", "META", "NFLX", "NVDA", "JPM", "V"]
SEQUENCE_LENGTH = 60
DEFAULT_DAYS_AHEAD = 7
DEFAULT_START_DATE = '2015-01-01'

# === Main route ===
@predictorrf_bp.route("/rf", methods=["GET", "POST"])
def rf():
    context = {
        "stock_options": STOCK_OPTIONS,
        "selected_symbol": STOCK_OPTIONS[0],
        "days_ahead": DEFAULT_DAYS_AHEAD,
        "error": None,
        "past_plot_url": None, "future_plot_url": None,
        "metrics": None, "future_price": None,
        "summary": None, "csv_link": None,
        "last_price": None, "last_date": None,
        "investment": None, "expected_return": None
    }

    if request.method == "POST":
        context["selected_symbol"] = request.form["symbol"].upper()
        context["days_ahead"] = int(request.form["days"])
        context["investment"] = request.form.get("investment")

        try:
            result = predict_with_rf(context["selected_symbol"], context["days_ahead"])
            if not result:
                context["error"] = "No data found or insufficient data."
            else:
                (context["past_plot_url"], context["future_plot_url"], context["metrics"],
                 future_price, context["summary"], future_prices, 
                 context["last_price"], context["last_date"]) = result

                context["future_price"] = round(float(future_price), 2)
                context["csv_link"] = "empty link"

                if context["investment"]:
                    try:
                        inv = float(context["investment"])
                        context["expected_return"] = round((context["future_price"] / context["last_price"]) * inv, 2)
                    except ValueError:
                        context["error"] = "Invalid investment amount."
        except Exception as e:
            context["error"] = f"An error occurred: {str(e)}"

    return render_template("predictor/rf.html", **context)

# === Core Logic ===
def predict_with_rf(symbol, days_ahead=DEFAULT_DAYS_AHEAD):
    data = load_stock_data(symbol)
    if data.empty or len(data) <= SEQUENCE_LENGTH:
        return None

    close_data = data['Close'].values.reshape(-1, 1)
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(close_data) #normalize

    X, y = create_sequences(scaled_data)
    model = RandomForestRegressor(n_estimators=100, random_state=42) #100 trees
    model.fit(X, y) #train

    predicted = model.predict(X) #RandomForestRegressor method
    predicted_prices = scaler.inverse_transform(predicted.reshape(-1, 1)).flatten()  #desnormalize 
    actual_prices = close_data[SEQUENCE_LENGTH:].flatten()

    metrics = compute_metrics(actual_prices, predicted_prices)

    # Forecast future prices
    future_scaled = forecast_future_prices(model, scaled_data[-SEQUENCE_LENGTH:], days_ahead)
    future_prices = scaler.inverse_transform(future_scaled.reshape(-1, 1)).flatten()

    last_price = float(close_data[-1])

    today = datetime.datetime.now().date()
    yesterday = today - datetime.timedelta(days=1)
    last_date = yesterday.strftime('%Y-%m-%d') 
    
    summary = summarize_forecast(close_data, future_prices, days_ahead)

    past_plot_url = plot_past_predictions(symbol, actual_prices, predicted_prices, data.index.tolist())
    future_plot_url = plot_future_forecast(symbol, future_prices, last_price, last_date)

    return past_plot_url, future_plot_url, metrics, future_prices[-1], summary, future_prices, last_price, last_date

# === Utility Functions ===
def load_stock_data(symbol):
    end_date = (datetime.datetime.now() - datetime.timedelta(days=1)).strftime('%Y-%m-%d')
    return yf.download(symbol, start=DEFAULT_START_DATE, end=end_date) #Open, High, Low, Close, adjClosePrice, tradingVolume

def create_sequences(data):
    X, y = [], []
    for i in range(SEQUENCE_LENGTH, len(data)):
        X.append(data[i - SEQUENCE_LENGTH:i, 0]) #past
        y.append(data[i, 0]) #future
    return np.array(X), np.array(y)

def compute_metrics(actual, predicted):
    return {
        "MSE": round(mean_squared_error(actual, predicted), 4),
        "RMSE": round(np.sqrt(mean_squared_error(actual, predicted)), 4),
        "MAE": round(mean_absolute_error(actual, predicted), 4)
    }

def forecast_future_prices(model, last_seq, steps):
    future = []
    seq = last_seq.flatten()
    for _ in range(steps):
        next_pred = model.predict([seq])[0]
        future.append(next_pred)
        seq = np.append(seq[1:], next_pred)
    return np.array(future)

def summarize_forecast(actual, forecasted, days):
    percent_change = ((forecasted[-1] - actual[-1][0]) / actual[-1][0]) * 100
    direction = "increase" if percent_change > 0 else "decrease"
    forecast_deltas = np.diff(forecasted)
    forecast_std = np.std(forecast_deltas)
    margin = stats.norm.ppf(0.975) * forecast_std  #95%
    lower = round(float(forecasted[-1] - margin), 2)
    upper = round(float(forecasted[-1] + margin), 2)
    return f"The forecast for the next {days} days suggests a {abs(round(percent_change, 2))}% {direction} with 95% confidence. Expected price range: ${lower} - ${upper}"

# === Plotting Functions ===
def plot_past_predictions(symbol, actual, predicted, dates):
    actual, predicted = actual[-105:], predicted[-105:]
    dates = dates[-len(actual):]

    plt.figure(figsize=(12, 6))
    plt.plot(dates, actual, label='Actual', color='black')
    plt.plot(dates, predicted, label='Predicted', color='green')
    plt.title(f"{symbol} - Last 5 Months Actual vs Predicted Prices")
    plt.xlabel("Date")
    plt.ylabel("Price")
    plt.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()

    path = os.path.join("app", "static", "past_plot.png")
    plt.savefig(path)
    plt.close()
    return os.path.join("static", "past_plot.png")

def plot_future_forecast(symbol, future, last_price, last_date):
    start = pd.to_datetime(last_date) + pd.Timedelta(days=1)
    future_dates = [start + pd.Timedelta(days=i) for i in range(len(future))]

    plt.figure(figsize=(12, 6))
    plt.plot(future_dates, future, marker='o', linestyle='-', color='blue', label='Forecast')
    for x, y in zip(future_dates, future):
        plt.annotate(f"${float(y):.2f}", (x, y), textcoords="offset points", xytext=(0, 8), ha='center', fontsize=8)
    plt.title(f"{symbol} - {len(future)}-Day Forecast")
    plt.xlabel("Date")
    plt.ylabel("Forecasted Price")
    plt.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()

    path = os.path.join("app", "static", "future_plot.png")
    plt.savefig(path)
    plt.close()
    return os.path.join("static", "future_plot.png")
