# src/data_preprocess.py

import numpy as np
import pandas as pd
from .config import NIFTY_CSV, NEWS_CSV, BAR_FREQ_MINUTES, TRADING_START, TRADING_END, RANDOM_SEED

np.random.seed(RANDOM_SEED)

def load_nifty():
    """
    Load intraday NIFTY 50 data from CSV with columns:
    date, open, high, low, close, volume
    where date is like '09-01-2015 09:15'
    """
    df = pd.read_csv(NIFTY_CSV)
    df.rename(columns=str.lower, inplace=True)
    df["date"] = pd.to_datetime(df["date"], format="%d-%m-%Y %H:%M")
    df = df.set_index("date").sort_index()

    # Ensure numeric
    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Filter regular trading hours
    df = df.between_time(TRADING_START, TRADING_END)
    return df


def load_news():
    """
    Load macro / financial news CSV with columns:
    Date, Headline, Source, Market_Event, Market_Index,
    Index_Change_Percent, Trading_Volume, Sentiment, Sector,
    Impact_Level, Related_Company
    """
    news = pd.read_csv(NEWS_CSV)
    news.rename(columns=str.strip, inplace=True)
    news.rename(columns=str.lower, inplace=True)

    # Parse date (assumes '21-05-2025' style)
    news["date"] = pd.to_datetime(news["date"], format="%d-%m-%Y")

    # Basic sentiment score mapping if 'sentiment' column is text
    # (Positive / Neutral / Negative). Empty cells become Neutral.
    def map_sentiment(s):
        if isinstance(s, str):
            s = s.strip().lower()
        else:
            return 0.0
        if "pos" in s:
            return 1.0
        if "neg" in s:
            return -1.0
        return 0.0

    if "sentiment" in news.columns:
        news["sentiment_score"] = news["sentiment"].apply(map_sentiment)
    else:
        news["sentiment_score"] = 0.0

    # We treat each row as a daily macro event; later we aggregate to intraday windows.
    return news


def compute_intraday_returns(df):
    df = df.copy()
    df["log_ret"] = np.log(df["close"]).diff()
    df["abs_ret"] = df["log_ret"].abs()
    return df


def resample_to_5min(df):
    """
    In case your raw CSV is not exactly 5-min spaced,
    this enforces a regular 5-min OHLCV grid.
    """
    df_5m = (df[["open", "high", "low", "close", "volume"]]
             .resample(f"{BAR_FREQ_MINUTES}T")
             .agg({
                 "open": "first",
                 "high": "max",
                 "low": "min",
                 "close": "last",
                 "volume": "sum"
             }))
    df_5m.dropna(subset=["close"], inplace=True)
    return df_5m


def prepare_base_nifty():
    df = load_nifty()
    df = resample_to_5min(df)
    df = compute_intraday_returns(df)
    return df


def prepare_news_events():
    news = load_news()
    # aggregate by date; we will map to intraday later
    daily = (news.groupby("date")
             .agg({"sentiment_score": "mean"})
             .rename(columns={"sentiment_score": "daily_sentiment"}))
    return news, daily
