# src/train_walk_forward.py

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from pathlib import Path
import tensorflow as tf

from .config import (
    TRAIN_DAYS, TEST_DAYS, WINDOW_STEPS,
    BATCH_SIZE, EPOCHS, VAL_SPLIT, EARLY_STOP_PATIENCE,
    RANDOM_SEED, RESULTS_DIR, MODELS_DIR
)
from .data_preprocess import prepare_base_nifty, prepare_news_events
from .feature_engineering import build_modeling_frame
from .models import build_lstm_model, moving_average_baseline, random_walk_baseline

np.random.seed(RANDOM_SEED)
tf.random.set_seed(RANDOM_SEED)
RESULTS_DIR.mkdir(exist_ok=True, parents=True)
MODELS_DIR.mkdir(exist_ok=True, parents=True)


def make_sequences(values, targets, window=WINDOW_STEPS):
    X, y = [], []
    for i in range(window, len(values)):
        X.append(values[i-window:i])
        y.append(targets[i])
    return np.array(X), np.array(y)


def walk_forward_train():
    # 1. Load and build full modeling frame
    base_df = prepare_base_nifty()
    news_raw, news_daily = prepare_news_events()
    full_df, feature_cols, target_col = build_modeling_frame(base_df, news_daily)

    values = full_df[feature_cols].values
    target = full_df[target_col].values
    timestamps = full_df.index

    bars_per_day = int(6.25 * 60 / 5)  # approx 75 5-min bars per session
    train_len = TRAIN_DAYS * bars_per_day
    test_len = TEST_DAYS * bars_per_day

    start = 0
    all_metrics = []
    all_preds = []

    fold_id = 0
    while start + train_len + test_len < len(full_df):
        end_train = start + train_len
        end_test = end_train + test_len

        X_train_raw = values[start:end_train]
        y_train = target[start:end_train]
        X_test_raw = values[end_train:end_test]
        y_test = target[end_train:end_test]
        ts_test = timestamps[end_train:end_test]

        scaler = MinMaxScaler()
        X_train_scaled = scaler.fit_transform(X_train_raw)
        X_test_scaled = scaler.transform(X_test_raw)

        X_train_seq, y_train_seq = make_sequences(X_train_scaled, y_train)
        X_test_seq, y_test_seq = make_sequences(X_test_scaled, y_test)

        n_features = X_train_seq.shape[-1]
        model = build_lstm_model(n_features)

        es = tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=EARLY_STOP_PATIENCE,
            restore_best_weights=True,
            verbose=1
        )

        history = model.fit(
            X_train_seq, y_train_seq,
            validation_split=VAL_SPLIT,
            epochs=EPOCHS,
            batch_size=BATCH_SIZE,
            callbacks=[es],
            verbose=0
        )

        y_pred = model.predict(X_test_seq).flatten()

        # Baselines for the same test window
        ma_pred = moving_average_baseline(y_test_seq, window=5)
        rw_pred = random_walk_baseline(y_test_seq)

        metrics = {
            "fold": fold_id,
            "start_idx": int(start),
            "mae_lstm": mean_absolute_error(y_test_seq, y_pred),
            "rmse_lstm": mean_squared_error(y_test_seq, y_pred, squared=False),
            "r2_lstm": r2_score(y_test_seq, y_pred),
            "mae_ma": mean_absolute_error(y_test_seq, ma_pred),
            "mae_rw": mean_absolute_error(y_test_seq, rw_pred)
        }

        # directional accuracy
        sign_true = np.sign(np.diff(y_test_seq))
        sign_pred = np.sign(np.diff(y_pred))
        dir_acc = (sign_true == sign_pred).mean()
        metrics["dir_acc_lstm"] = dir_acc

        all_metrics.append(metrics)

        preds_df = pd.DataFrame({
            "timestamp": ts_test[WINDOW_STEPS:],
            "y_true": y_test_seq,
            "y_pred": y_pred
        })
        preds_df["fold"] = fold_id
        all_preds.append(preds_df)

        # save model snapshot per fold (optional)
        model.save(MODELS_DIR / f"lstm_fold_{fold_id}.keras")

        fold_id += 1
        start += test_len  # slide window

    metrics_df = pd.DataFrame(all_metrics)
    preds_all_df = pd.concat(all_preds, ignore_index=True)

    metrics_df.to_csv(RESULTS_DIR / "walk_forward_metrics.csv", index=False)
    preds_all_df.to_csv(RESULTS_DIR / "walk_forward_predictions.csv", index=False)


if __name__ == "__main__":
    walk_forward_train()
