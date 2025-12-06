# src/config.py

from pathlib import Path

# Paths
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
RESULTS_DIR = PROJECT_ROOT / "results"
MODELS_DIR = PROJECT_ROOT / "models"

NIFTY_CSV = DATA_DIR / "nifty_intraday_2015_2025.csv"
NEWS_CSV = DATA_DIR / "global_news_sentiment.csv"

# Time / sampling
BAR_FREQ_MINUTES = 5              # 5-minute bars
TRADING_START = "09:15"
TRADING_END = "15:30"

# Option config (ATM call)
RISK_FREE_RATE = 0.06             # annual
DAYS_TO_EXPIRY = 7                # weekly options
MIN_IV = 0.07
MAX_IV = 0.40

# LSTM / training
WINDOW_STEPS = 12                 # last 1 hour of 5-min data
TRAIN_DAYS = 30
TEST_DAYS = 7
BATCH_SIZE = 64
EPOCHS = 40
VAL_SPLIT = 0.1
EARLY_STOP_PATIENCE = 4

RANDOM_SEED = 42
