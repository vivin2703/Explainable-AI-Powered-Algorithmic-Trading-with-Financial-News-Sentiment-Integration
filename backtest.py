# src/backtest.py

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from .config import RESULTS_DIR

def compute_equity_curve(preds_df, cost_bp=2):
    """
    Simple one-step strategy:
    - Long call if model predicts price increase next bar
    - Short (or exit) if predicts decrease
    """
    df = preds_df.sort_values("timestamp").reset_index(drop=True)
    equity = [0.0]

    for i in range(1, len(df)):
        p_prev = df["y_pred"].iloc[i-1]
        p_now = df["y_pred"].iloc[i]
        direction = 1 if p_now > p_prev else -1

        true_prev = df["y_true"].iloc[i-1]
        true_now = df["y_true"].iloc[i]

        pnl = direction * (true_now - true_prev)
        # transaction cost in basis points
        pnl -= abs(true_now) * (cost_bp / 10000.0)
        equity.append(equity[-1] + pnl)

    df["equity"] = equity
    return df


def sharpe_ratio(returns, risk_free=0.0):
    if returns.std() == 0:
        return 0.0
    return (returns.mean() - risk_free) / returns.std() * np.sqrt(252)


def sortino_ratio(returns, risk_free=0.0):
    downside = returns[returns < 0]
    if downside.std() == 0:
        return 0.0
    return (returns.mean() - risk_free) / downside.std() * np.sqrt(252)


def max_drawdown(equity_curve):
    roll_max = equity_curve.cummax()
    dd = equity_curve - roll_max
    return dd.min()


def run_backtest():
    preds = pd.read_csv(RESULTS_DIR / "walk_forward_predictions.csv", parse_dates=["timestamp"])
    eq_df = compute_equity_curve(preds)
    eq_df.to_csv(RESULTS_DIR / "backtest_equity.csv", index=False)

    # metrics
    ret = eq_df["equity"].diff().dropna()
    sharpe = sharpe_ratio(ret)
    sortino = sortino_ratio(ret)
    mdd = max_drawdown(eq_df["equity"])

    metrics = {
        "sharpe": sharpe,
        "sortino": sortino,
        "max_drawdown": mdd,
        "hit_rate": (ret > 0).mean()
    }
    pd.Series(metrics).to_csv(RESULTS_DIR / "backtest_metrics.csv")

    # plot
    plt.figure(figsize=(10,4))
    plt.plot(eq_df["timestamp"], eq_df["equity"])
    plt.title("Backtest Equity Curve (ATM Call Strategy)")
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "backtest_equity.png", dpi=200)


if __name__ == "__main__":
    run_backtest()
