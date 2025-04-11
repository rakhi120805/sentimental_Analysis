import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

# Load stock and sentiment data
stock_data = pd.read_csv("tsla_stock_data.csv")
sentiment_data = pd.read_csv("tsla_news_sentiment.csv")

# Convert Date columns to datetime
stock_data["Date"] = pd.to_datetime(stock_data["Date"]).dt.date
sentiment_data["Date"] = pd.to_datetime(sentiment_data["Date"]).dt.date

# Merge datasets on Date (inner join to keep only matching dates)
merged_data = pd.merge(stock_data, sentiment_data, on="Date", how="inner")

# Create features
# 1. Daily return: (Close - Open) / Open
merged_data["Daily_Return"] = (merged_data["Close"] - merged_data["Open"]) / merged_data["Open"]

# 2. Price range: High - Low
merged_data["Price_Range"] = merged_data["High"] - merged_data["Low"]

# 3. Volatility proxy: abs(Daily_Return)
merged_data["Volatility"] = merged_data["Daily_Return"].abs()

# 4. Sentiment (already included)
# Optionally smooth sentiment to reduce noise (3-day moving average)
merged_data["Sentiment_Smoothed"] = merged_data["Sentiment"].rolling(window=3, min_periods=1).mean()

# Select features for PCA
features = ["Open", "Close", "Volume", "Daily_Return", "Price_Range", "Volatility", "Sentiment", "Sentiment_Smoothed"]
feature_data = merged_data[features]

# Handle missing values (should be minimal after merge)
feature_data = feature_data.fillna(0)  # Replace NaNs with 0 (e.g., for early moving averages)

# Standardize features (PCA requires scaled data)
scaler = StandardScaler()
scaled_features = scaler.fit_transform(feature_data)

# Create a DataFrame with scaled features
scaled_df = pd.DataFrame(scaled_features, columns=features, index=merged_data["Date"])

# Save merged and scaled data
merged_data.to_csv("tsla_merged_data.csv", index=False)
scaled_df.to_csv("tsla_scaled_features.csv")

# Display results
print("Merged Data Preview:")
print(merged_data[["Date", "Close", "Sentiment", "Daily_Return", "Sentiment_Smoothed"]].head())
print("\nScaled Features Preview:")
print(scaled_df.head())
print(f"\nData range: {merged_data['Date'].min()} to {merged_data['Date'].max()}")
print(f"Number of days: {len(merged_data)}")
