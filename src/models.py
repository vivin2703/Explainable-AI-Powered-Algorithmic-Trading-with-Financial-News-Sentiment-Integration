# src/models.py

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from .config import WINDOW_STEPS

def build_lstm_model(n_features):
    model = models.Sequential([
        layers.Input(shape=(WINDOW_STEPS, n_features)),
        layers.LSTM(64, return_sequences=True),
        layers.Dropout(0.3),
        layers.LSTM(32),
        layers.Dense(32, activation="relu"),
        layers.Dense(1)
    ])
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-3),
                  loss="mse",
                  metrics=["mae"])
    return model


def moving_average_baseline(y, window=5):
    y = np.asarray(y)
    ma = np.convolve(y, np.ones(window)/window, mode="same")
    return ma


def random_walk_baseline(y):
    """
    Next value = previous value.
    """
    y = np.asarray(y)
    rw = np.roll(y, 1)
    rw[0] = y[0]
    return rw
