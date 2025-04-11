import requests
import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from datetime import datetime, timedelta

# NewsAPI setup
api_key = "YOUR_NEWSAPI_KEY"  # Replace with your NewsAPI key
query = "Tesla OR Tesla Inc OR TSLA OR Elon Musk OR Tesla Motors"  # Broad query
end_date = datetime(2025, 4, 10).date()  # Match stock data
start_date = datetime(2025, 3, 11).date()  # Match stock data

# Function to fetch news with pagination
def fetch_news(api_key, query, start_date, end_date):
    url = "https://newsapi.org/v2/everything"
    articles = []
    page = 1
    max_pages = 5  # Limit to avoid hitting API quota
    while page <= max_pages:
        params = {
            "q": query,
            "from": start_date.strftime("%Y-%m-%d"),
            "to": end_date.strftime("%Y-%m-%d"),
            "apiKey": api_key,
            "language": "en",
            "sortBy": "popularity",  # Prioritize high-impact articles
            "page": page,
            "pageSize": 100
        }
        try:
            response = requests.get(url, params=params)
            if response.status_code == 429:
                print("Rate limit reached. Try again later.")
                break
            response.raise_for_status()
            data = response.json()
            new_articles = data.get("articles", [])
            articles.extend(new_articles)
            print(f"Fetched {len(new_articles)} articles on page {page} (total: {len(articles)})")
            if len(new_articles) < 100:
                break
            page += 1
        except Exception as e:
            print(f"Error fetching page {page}: {e}")
            break
    if not articles:
        print("No news articles found.")
    return articles

# Fetch news
print(f"Fetching news from {start_date} to {end_date}...")
articles = fetch_news(api_key, query, start_date, end_date)
print(f"Total articles fetched: {len(articles)}")

# Log articles for debugging
with open("news_articles_log.txt", "w", encoding="utf-8") as f:
    for i, article in enumerate(articles, 1):
        f.write(f"Article {i}:\n")
        f.write(f"Title: {article.get('title', 'N/A')}\n")
        f.write(f"Description: {article.get('description', 'N/A')}\n")
        f.write(f"Published: {article.get('publishedAt', 'N/A')}\n")
        f.write(f"Source: {article.get('source', {}).get('name', 'N/A')}\n")
        f.write("-" * 50 + "\n")

# Process news into a DataFrame
news_data = []
for article in articles:
    title = article.get("title", "")
    description = article.get("description", "")
    published_at = article.get("publishedAt", "")
    if title or description:  # Accept either
        text = f"{title} {description or ''}".strip()
        try:
            date = pd.to_datetime(published_at).date()
            if start_date <= date <= end_date:
                news_data.append({"Date": date, "Text": text})
            else:
                print(f"Skipping article outside range: {published_at}")
        except ValueError:
            print(f"Skipping article with invalid date: {published_at}")

news_df = pd.DataFrame(news_data)
if news_df.empty:
    print("No valid news data after processing. Check news_articles_log.txt.")
    exit()

# Calculate sentiment using VADER
analyzer = SentimentIntensityAnalyzer()
def get_sentiment(text):
    scores = analyzer.polarity_scores(text)
    return scores["compound"]

news_df["Sentiment"] = news_df["Text"].apply(get_sentiment)

# Aggregate sentiment by date (mean sentiment per day)
sentiment_df = news_df.groupby("Date")["Sentiment"].mean().reset_index()

# Ensure all dates are covered
all_dates = pd.DataFrame({"Date": pd.date_range(start=start_date, end=end_date).date})
sentiment_df = all_dates.merge(sentiment_df, on="Date", how="left")
sentiment_df["Sentiment"] = sentiment_df["Sentiment"].fillna(0)  # Neutral for no news

# Save to CSV
sentiment_df.to_csv("tsla_news_sentiment.csv", index=False)

# Display results
print("\nNews sentiment for Tesla:")
print(sentiment_df)
print(f"\nSentiment data range: {sentiment_df['Date'].min()} to {sentiment_df['Date'].max()}")

# Verify stock data alignment
stock_data = pd.read_csv("tsla_stock_data.csv")
stock_dates = pd.to_datetime(stock_data["Date"]).dt.date
print("\nStock data range:")
print(f"From {stock_dates.min()} to {stock_dates.max()}")

# Count non-zero sentiment days
non_zero_days = len(sentiment_df[sentiment_df["Sentiment"] != 0])
print(f"Days with non-zero sentiment: {non_zero_days}")
