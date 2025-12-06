# main.py

from src.train_walk_forward import walk_forward_train
from src.explainability import run_shap_for_last_fold, run_lime_example
from src.backtest import run_backtest

if __name__ == "__main__":
    walk_forward_train()
    run_shap_for_last_fold()
    run_lime_example()
    run_backtest()
