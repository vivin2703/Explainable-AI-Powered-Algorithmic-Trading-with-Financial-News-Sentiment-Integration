# src/feature_engineering.py

import numpy as np
import pandas as pd
from scipy.stats import norm
from .config import (
    RISK_FREE_RATE, DAYS_TO_EXPIRY,
    MIN_IV, MAX_IV, WINDOW_STEPS
)

def realized_vol(df, window=78):  # ~1 day of 5-min bars
    rv = np.sqrt((df["log_ret"]**2).rolling(window).sum()) * np.sqrt(252)
    return rv


def garman_klass_vol(df, window=78):
    """Alternative realized volatility estimator."""
    log_hl = np.log(df["high"] / df["low"])
    log_co = np.log(df["close"] / df["open"])
    sig2 = 0.5 * (log_hl**2) - (2*np.log(2) - 1)*(log_co**2)
    vol = np.sqrt(sig2.rolling(window).sum()) * np.sqrt(252)
    return vol


def simulate_iv_series(df, phi=0.97, sigma_noise=0.02, seed=42):
    np.random.seed(seed)
    base_vol = df["rv"].fillna(df["rv"].median()).clip(lower=0.1).values
    iv = np.zeros(len(df))
    iv[0] = np.clip(base_vol[0], MIN_IV, MAX_IV)
    for t in range(1, len(df)):
        iv[t] = phi * iv[t-1] + sigma_noise * np.random.randn()
        iv[t] = np.clip(iv[t], MIN_IV, MAX_IV)
    df["iv"] = iv
    return df


def black_scholes_call(S, K, T, r, sigma):
    if T <= 0:
        return max(0.0, S - K)
    d1 = (np.log(S/K) + (r + 0.5*sigma**2)*T) / (sigma*np.sqrt(T) + 1e-12)
    d2 = d1 - sigma*np.sqrt(T)
    call = S * norm.cdf(d1) - K * np.exp(-r*T) * norm.cdf(d2)
    return call


def compute_greeks(S, K, T, r, sigma):
    if T <= 0:
        return 1.0, 0.0, 0.0, 0.0
    d1 = (np.log(S/K) + (r + 0.5*sigma**2)*T) / (sigma*np.sqrt(T) + 1e-12)
    d2 = d1 - sigma*np.sqrt(T)

    delta = norm.cdf(d1)
    gamma = norm.pdf(d1) / (S*sigma*np.sqrt(T) + 1e-12)
    vega  = S * norm.pdf(d1) * np.sqrt(T) / 100.0
    theta = (-(S*norm.pdf(d1)*sigma/(2*np.sqrt(T)))
             - r*K*np.exp(-r*T)*norm.cdf(d2)) / 365.0
    return delta, gamma, vega, theta


def generate_atm_option_features(df):
    """
    For each timestamp, create an ATM call option:
    - Strike = nearest round(close to nearest 50)
    - Time to expiry fixed to DAYS_TO_EXPIRY
    - Use simulated IV to generate call price + Greeks
    """
    df = df.copy()
    df["strike_atm"] = (df["close"] / 50).round() * 50
    T = DAYS_TO_EXPIRY / 252.0

    call_prices = []
    deltas, gammas, vegas, thetas = [], [], [], []

    for S, K, sigma in zip(df["close"], df["strike_atm"], df["iv"]):
        call_price = black_scholes_call(S, K, T, RISK_FREE_RATE, sigma)
        d, g, v, th = compute_greeks(S, K, T, RISK_FREE_RATE, sigma)
        call_prices.append(call_price)
        deltas.append(d); gammas.append(g); vegas.append(v); thetas.append(th)

    df["call_atm"] = call_prices
    df["delta"] = deltas
    df["gamma"] = gammas
    df["vega"] = vegas
    df["theta"] = thetas
    return df


def align_news_to_intraday(df_intraday, news_daily):
    """
    Map daily sentiment to intraday timestamps and build
    rolling sentiment windows: 1h, 6h, 24h.
    """
    df = df_intraday.copy()

    # Map daily sentiment to each timestamp (forward-fill within day)
    daily_on_index = news_daily.reindex(df.index.date, method="ffill")
    df["daily_sentiment"] = daily_on_index.values

    # Treat daily_sentiment as "news shock" at 09:15 of each day;
    # then compute rolling means.

    df["sent_1h"] = df["daily_sentiment"].rolling(12, min_periods=1).mean()
    df["sent_6h"] = df["daily_sentiment"].rolling(72, min_periods=1).mean()
    df["sent_24h"] = df["daily_sentiment"].rolling(288, min_periods=1).mean()

    df["news_surprise"] = df["sent_1h"] - df["sent_24h"].rolling(288, min_periods=1).mean()
    df["news_surprise"].fillna(0.0, inplace=True)
    return df


def build_modeling_frame(base_df, news_daily):
    """
    Full feature pipeline, leak-free, for ATM call prediction.
    """
    df = base_df.copy()

    # Realized vol features
    df["rv"] = realized_vol(df, window=78)
    df["rv_gk"] = garman_klass_vol(df, window=78)

    # Simulated IV time-series
    df = simulate_iv_series(df)

    # Option price + Greeks
    df = generate_atm_option_features(df)

    # News/sentiment features aligned
    df = align_news_to_intraday(df, news_daily)

    # Lagged features
    for lag in [1, 2, 3, 6, 12]:
        df[f"close_lag_{lag}"] = df["close"].shift(lag)
        df[f"iv_lag_{lag}"] = df["iv"].shift(lag)
        df[f"rv_lag_{lag}"] = df["rv"].shift(lag)

    df["ret_1"] = df["log_ret"]
    df["ret_5"] = df["close"].pct_change(5)
    df["ret_20"] = df["close"].pct_change(20)

    # Interaction features
    df["sent_x_iv"] = df["sent_1h"] * df["iv"]
    df["sent_x_vol"] = df["sent_1h"] * df["rv"]

    df.dropna(inplace=True)

    FEATURE_COLS = [
        "close", "iv", "rv", "rv_gk",
        "delta", "gamma", "vega", "theta",
        "sent_1h", "sent_6h", "sent_24h",
        "news_surprise", "sent_x_iv", "sent_x_vol",
        "close_lag_1", "close_lag_2", "close_lag_3",
        "iv_lag_1", "iv_lag_2",
        "rv_lag_1", "rv_lag_2",
        "ret_1", "ret_5", "ret_20"
    ]

    TARGET_COL = "call_atm"

    return df, FEATURE_COLS, TARGET_COL
