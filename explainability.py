# src/explainability.py

import numpy as np
import pandas as pd
import shap
from lime.lime_tabular import LimeTabularExplainer
import tensorflow as tf
from pathlib import Path

from .config import RESULTS_DIR, MODELS_DIR, WINDOW_STEPS
from .data_preprocess import prepare_base_nifty, prepare_news_events
from .feature_engineering import build_modeling_frame

def prepare_last_window(feature_cols, target_col):
    base_df = prepare_base_nifty()
    _, news_daily = prepare_news_events()
    full_df, feature_cols, target_col = build_modeling_frame(base_df, news_daily)

    values = full_df[feature_cols].values
    target = full_df[target_col].values

    X_seq = []
    y_seq = []
    for i in range(WINDOW_STEPS, len(values)):
        X_seq.append(values[i-WINDOW_STEPS:i])
        y_seq.append(target[i])
    return np.array(X_seq), np.array(y_seq), feature_cols


def run_shap_for_last_fold():
    # Load last fold model
    model_paths = sorted(MODELS_DIR.glob("lstm_fold_*.keras"))
    if not model_paths:
        raise FileNotFoundError("No fold models found in models/ directory.")
    model = tf.keras.models.load_model(model_paths[-1])

    X_seq, y_seq, feature_cols = prepare_last_window(feature_cols=None, target_col=None)
    # Use a subset for SHAP
    X_sample = X_seq[-500:]

    # DeepExplainer for sequence model (uses background sample)
    explainer = shap.DeepExplainer(model, X_sample[:100])
    shap_values = explainer.shap_values(X_sample[:200])[0]   # (n_samples, time, features)

    # Aggregate over time dimension (mean |shap| per feature)
    shap_abs_mean = np.mean(np.abs(shap_values), axis=1)     # (n_samples, features)

    shap.summary_plot(shap_abs_mean, features=X_sample.mean(axis=1),
                      feature_names=feature_cols, show=False)
    shap.plt.gcf().savefig(RESULTS_DIR / "shap_summary.png", dpi=200, bbox_inches="tight")


def run_lime_example():
    model_paths = sorted(MODELS_DIR.glob("lstm_fold_*.keras"))
    if not model_paths:
        raise FileNotFoundError("No fold models found in models/ directory.")
    model = tf.keras.models.load_model(model_paths[-1])

    X_seq, y_seq, feature_cols = prepare_last_window(feature_cols=None, target_col=None)
    X_flat = X_seq.mean(axis=1)  # simple aggregation over time

    explainer = LimeTabularExplainer(
        training_data=X_flat,
        feature_names=feature_cols,
        mode="regression"
    )

    def predict_fn(x):
        # Expand back to sequence length with constant profiles (approximation)
        x_seq = np.repeat(x[:, None, :], repeats=WINDOW_STEPS, axis=1)
        return model.predict(x_seq)

    exp = explainer.explain_instance(
        X_flat[-1],
        predict_fn,
        num_features=10
    )
    fig = exp.as_pyplot_figure()
    fig.savefig(RESULTS_DIR / "lime_example.png", dpi=200, bbox_inches="tight")


if __name__ == "__main__":
    run_shap_for_last_fold()
    run_lime_example()
