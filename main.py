import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta

# Define the stock ticker and time period
ticker = "TSLA"  # Tesla
end_date = datetime.now().date()  # Today’s date
start_date = end_date - timedelta(days=30)  # 30 days ago to match NewsAPI

# Fetch stock data from yFinance
try:
    stock = yf.Ticker(ticker)
    stock_data = stock.history(start=start_date, end=end_date)

    # Check if data is empty
    if stock_data.empty:
        print(f"No data found for {ticker}. Check ticker or internet connection.")
        exit()

    # Reset index to make 'Date' a column
    stock_data = stock_data.reset_index()

    # Select relevant columns and rename for clarity
    stock_data = stock_data[['Date', 'Open', 'High', 'Low', 'Close', 'Volume']]
    stock_data['Date'] = pd.to_datetime(stock_data['Date']).dt.date  # Keep only date (no time)

    # Save to CSV
    stock_data.to_csv('tsla_stock_data.csv', index=False)

    # Display the first few rows
    print(f"Stock data for {ticker}:")
    print(stock_data.head())

except Exception as e:
    print(f"Error fetching data for {ticker}: {e}")
    