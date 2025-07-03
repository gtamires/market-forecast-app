### File: analysis/routes.py

import os
import datetime
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from flask import Blueprint, request, render_template
import yfinance as yf

# Create blueprint for EDA
analysis_bp = Blueprint('analysis', __name__, template_folder='templates/analysis')

# Constants
STOCK_OPTIONS = [
    "AAPL", "MSFT", "GOOGL", "TSLA", "AMZN",
    "META", "NFLX", "NVDA", "JPM", "V"
]

today = datetime.datetime.now().date()  #class, module
yesterday = today - datetime.timedelta(days=1)

DEFAULT_START_DATE = '2015-01-01'
DEFAULT_END_DATE = yesterday.strftime('%Y-%m-%d')

@analysis_bp.route("/analysis", methods=["GET", "POST"])
def analysis():
    """
    Renders the Exploratory Data Analysis page.
    Handles both displaying the form (GET) and performing EDA (POST).
    """
    context = {
        'stock_options': STOCK_OPTIONS,
        'selected_symbol': STOCK_OPTIONS[0],
        'start_date': DEFAULT_START_DATE,
        'end_date': DEFAULT_END_DATE,
        'summary_stats': None,
        'plots': [],
        'error': None
    }

    if request.method == 'POST':
        symbol = request.form.get('symbol').upper()
        start_date = request.form.get('start_date') or DEFAULT_START_DATE
        end_date = request.form.get('end_date') or DEFAULT_END_DATE
        context.update({
            'selected_symbol': symbol,
            'start_date': start_date,
            'end_date': end_date
        })
        try:
            # Fetch data
            data = yf.download(symbol, start=start_date, end=end_date)
            if data.empty:
                raise ValueError("No data fetched for symbol.")

            # Compute summary statistics
            returns = data['Close'].pct_change().dropna()
            summary = data['Close'].describe().to_dict() #o	count, mean, std, min, 25%, 50%, 75%, max
            summary['volatility'] = returns.std()
            summary['mean_return'] = returns.mean()
            # Round stats
            summary = {k: round(v, 4) if isinstance(v, (int, float, np.number)) else v for k, v in summary.items()}
            context['summary_stats'] = summary

            # Create static folder if not exists
            static_dir = os.path.join('app', 'static', 'analysis')
            os.makedirs(static_dir, exist_ok=True)

            # Plot 1: Closing Price over time
            plt.figure(figsize=(10, 4))
            plt.plot(data.index, data['Close'])
            plt.title(f"{symbol} Closing Price")
            plt.xlabel("Date")
            plt.ylabel("Price (USD)")
            plt.xticks(rotation=45)
            plt.tight_layout()
            price_plot = os.path.join(static_dir, 'closing_price.png')
            plt.savefig(price_plot)
            plt.close()

            # Plot 2: Histogram of daily returns   (percentage changes from one day to the next)
            plt.figure(figsize=(8, 4))
            plt.hist(returns, bins=50)
            plt.title(f"{symbol} Daily Returns Distribution")
            plt.xlabel("Daily Return")
            plt.ylabel("Frequency")
            plt.tight_layout()
            returns_plot = os.path.join(static_dir, 'returns_hist.png')
            plt.savefig(returns_plot)
            plt.close()

            # Plot 3: Rolling volatility (30-day)
            rolling_vol = returns.rolling(window=30).std()
            plt.figure(figsize=(10, 4))
            plt.plot(rolling_vol.index, rolling_vol)
            plt.title(f"{symbol} 30-Day Rolling Volatility")
            plt.xlabel("Date")
            plt.ylabel("Volatility")
            plt.xticks(rotation=45)
            plt.tight_layout()
            vol_plot = os.path.join(static_dir, 'rolling_vol.png')
            plt.savefig(vol_plot)
            plt.close()

            context['plots'] = [
                'analysis/closing_price.png',
                'analysis/returns_hist.png',
                'analysis/rolling_vol.png'
            ]
        except Exception as e:
            context['error'] = str(e)

    return render_template('analysis/analysis.html', **context)