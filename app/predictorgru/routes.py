# --- Imports ---
import os
import csv
import datetime
import numpy as np
import matplotlib
import matplotlib.dates as mdates
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
import pandas as pd
from flask import Blueprint, Flask, request, render_template
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from keras.layers import GRU, Dense
from keras.models import Sequential
from scipy import stats
import yfinance as yf  # For downloading historical stock data

# --- Flask Setup ---
app = Flask(__name__)
predictorgru_bp = Blueprint('predictorgru', __name__)  # Blueprint for modular routing

# --- Constants ---
STOCK_OPTIONS = [  # Dropdown options for user to select a stock
    "AAPL", "MSFT", "GOOGL", "TSLA", "AMZN", 
    "META", "NFLX", "NVDA", "JPM", "V"
]

SEQUENCE_LENGTH = 60  # Number of past days to use for prediction window
DEFAULT_DAYS_AHEAD = 7  # Number of future days to forecast by default
DEFAULT_START_DATE = '2015-01-01'  # Earliest date for stock data fetching


# --- Route for GRU prediction page ---
@predictorgru_bp.route("/gru", methods=["GET", "POST"])
def gru():
    # Context dictionary passed to the HTML template
    context = {
        "past_plot_url": None, "future_plot_url": None, "metrics": None, 
        "future_price": None, "summary": None, "csv_link": None, "error": None,
        "last_price": None, "last_date": None, "investment": None,
        "expected_return": None, "stock_options": STOCK_OPTIONS,
        "selected_symbol": STOCK_OPTIONS[0], "days_ahead": DEFAULT_DAYS_AHEAD
    }

    if request.method == "POST":
        # Capture form inputs
        context["selected_symbol"] = request.form["symbol"].upper()
        context["days_ahead"] = int(request.form["days"])
        context["investment"] = request.form.get("investment")

        try:
            # Run prediction logic
            results = predict_stock(context["selected_symbol"], days_ahead=context["days_ahead"])
            if not results:
                context["error"] = "No data found or insufficient data for prediction."
            else:
                # Unpack results and update context
                (
                    context["past_plot_url"], context["future_plot_url"], context["metrics"], context["future_price"],
                    context["summary"], future_prices, context["last_price"], context["last_date"]
                ) = results

                # Export predictions to CSV
                context["csv_link"] = "empty link"

                # Calculate expected return from investment
                if context["investment"]:
                    try:
                        inv = float(context["investment"])
                        context["expected_return"] = round((context["future_price"] / context["last_price"]) * inv, 2)
                    except ValueError:
                        context["error"] = "Invalid investment amount. Please enter a valid number."
        except IndexError:
            context["error"] = "Insufficient data to make predictions. Try a different symbol."
        except Exception as e:
            context["error"] = f"An unexpected error occurred: {str(e)}"

    return render_template("predictor/gru.html", **context)


# --- Core prediction logic using GRU ---
def predict_stock(symbol, start_date=None, end_date=None, days_ahead=DEFAULT_DAYS_AHEAD):
    # Set default date range
    today = datetime.datetime.now().date()
    yesterday = today - datetime.timedelta(days=1)

    start_date = start_date or DEFAULT_START_DATE
    end_date = end_date or yesterday.strftime('%Y-%m-%d')

    # Download stock data
    data = yf.download(symbol, start=start_date, end=end_date) #Open, High, Low, Close, adjClosePrice, tradingVolume
    if data.empty or len(data) <= SEQUENCE_LENGTH:
        return None

    # Normalize closing prices
    close_data = data['Close'].values.reshape(-1, 1)
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(close_data) #normalize

    # Prepare training sequences
    X_train, y_train = create_sequences(scaled_data)
    if X_train.size == 0:
        return None

    # Reshape input to fit GRU model
    X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))

    # Build and train GRU model
    model = build_gru_model(input_shape=(SEQUENCE_LENGTH, 1))
    model.fit(X_train, y_train, epochs=3, batch_size=32, verbose=0)

    # Evaluate model using recent historical data
    test_data = scaled_data[-(SEQUENCE_LENGTH + 100):]
    predicted_prices = predict_with_model(model, test_data, scaler)
    actual_prices = close_data[-len(predicted_prices):]

    # Compute evaluation metrics
    metrics = calculate_metrics(actual_prices, predicted_prices)
    last_price = float(close_data[-1][0])
    last_date = yesterday.strftime('%Y-%m-%d')  # explicitly set to yesterday

    # Forecast future prices
    future_prices = forecast_future_prices(model, scaled_data, scaler, days_ahead)
    future_summary = summarize_forecast(close_data, future_prices, days_ahead)

    # Generate plot image
    dates = data.index.tolist() #covert pandas to python list
    past_plot_url = plot_past_predictions(symbol, actual_prices, predicted_prices, dates)
    future_plot_url = plot_future_forecast(symbol, future_prices, last_price, last_date)

    return past_plot_url, future_plot_url, metrics, round(future_prices[-1], 2), future_summary, future_prices, last_price, last_date


# --- Helper: Create input-output sequences for training ---
def create_sequences(data):
    """
    Transforms time series into overlapping sequences for LSTM input.
    Each input is a window of 60 past days and its target is the 61st day.
    """    
    X, y = [], []
    for i in range(SEQUENCE_LENGTH, len(data)):
        X.append(data[i - SEQUENCE_LENGTH:i, 0]) #past
        y.append(data[i, 0])  #future
    return np.array(X), np.array(y)


# --- Helper: Build a two-layer GRU model architecture ---
def build_gru_model(input_shape):
    model = Sequential()
    model.add(GRU(50, return_sequences=True, input_shape=input_shape))
    model.add(GRU(50))  # Second GRU layer
    model.add(Dense(1))  # Output layer
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model


# --- Helper: Predict stock prices using trained model ---
def predict_with_model(model, data, scaler):
    X_test = [data[i - SEQUENCE_LENGTH:i, 0] for i in range(SEQUENCE_LENGTH, len(data))]
    X_test = np.array(X_test).reshape(-1, SEQUENCE_LENGTH, 1)
    predictions = model.predict(X_test)
    return scaler.inverse_transform(predictions) # Convert to actual prices


# --- Helper: Calculate MSE, RMSE, MAE ---
def calculate_metrics(actual, predicted):
    mse = mean_squared_error(actual, predicted)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(actual, predicted)
    return {"MSE": round(mse, 4), "RMSE": round(rmse, 4), "MAE": round(mae, 4)}


# --- Helper: Forecast future prices using recursive prediction ---
def forecast_future_prices(model, scaled_data, scaler, days):
    input_seq = scaled_data[-SEQUENCE_LENGTH:].reshape(1, SEQUENCE_LENGTH, 1)
    future = []
    for _ in range(days):
        next_val = model.predict(input_seq)[0][0]
        future.append(next_val)
        input_seq = np.append(input_seq[:, 1:, :], [[[next_val]]], axis=1)
    return scaler.inverse_transform(np.array(future).reshape(-1, 1)).flatten()


# --- Helper: Generate a summary with confidence interval ---
def summarize_forecast(actual, forecasted, days):
    percent_change = ((forecasted[-1] - actual[-1][0]) / actual[-1][0]) * 100
    direction = "increase" if percent_change > 0 else "decrease"

    # Estimate uncertainty using standard deviation
    forecast_deltas = np.diff(forecasted)
    forecast_std = np.std(forecast_deltas)
    margin = stats.norm.ppf(0.975) * forecast_std
    lower = round(forecasted[-1] - margin, 2)
    upper = round(forecasted[-1] + margin, 2)

    return (
        f"The forecast for the next {days} days suggests a "
        f"{abs(round(percent_change, 2))}% {direction} with 95% confidence. "
        f"Expected price range: ${lower} - ${upper}"
    )


def plot_past_predictions(symbol, actual, predicted, dates):
    """Plot last 5 months of actual and predicted data."""
    # Use the last ~105 trading days (~5 months)
    max_points = 105
    actual = actual[-max_points:]
    predicted = predicted[-max_points:]
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


def plot_future_forecast(symbol, future, last_actual_price, last_date):
    """Plot future forecast data with dates."""
    forecast_start = pd.to_datetime(last_date) + pd.Timedelta(days=1)
    future_dates = [forecast_start + pd.Timedelta(days=i) for i in range(len(future))]

    plt.figure(figsize=(12, 6))
    plt.plot(future_dates, future, marker='o', linestyle='-', color='blue', label='Forecast')

    # Annotate with forecasted values
    for i, (x, y) in enumerate(zip(future_dates, future)):
        plt.annotate(f"${y:.2f}", (x, y), textcoords="offset points", xytext=(0, 8), ha='center', fontsize=8)

    plt.title(f"{symbol} - {len(future)}-Day Forecast")
    plt.xlabel("Date")
    plt.ylabel("Forecasted Price")
    plt.legend()
    plt.xticks(rotation=45)
    plt.tight_layout() #avoid overlaps
    path = os.path.join("app", "static", "future_plot.png")
    plt.savefig(path)
    plt.close()
    return os.path.join("static", "future_plot.png")

