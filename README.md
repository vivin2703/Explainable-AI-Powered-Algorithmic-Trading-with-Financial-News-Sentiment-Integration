# Explainable AI-Based NIFTY 50 Option Price Prediction Using Financial News Sentiment

## Abstract
This project presents an explainable deep learning framework for predicting **NIFTY 50 ATM call option prices** by integrating historical intraday market data with financial news sentiment. Due to the unavailability of long-horizon historical option price data, option premiums are reconstructed using the **Black–Scholes pricing model** driven by real underlying index movements and simulated implied volatility. A **Long Short-Term Memory (LSTM)** network is employed to capture temporal dependencies, while **SHAP** and **LIME** techniques are used to interpret model predictions. The system is evaluated using walk-forward validation, baseline comparisons, and a simple trading backtest.

---

## 1. Introduction
Option prices are influenced by complex interactions among underlying asset movements, volatility dynamics, and market sentiment. Traditional statistical models often fail to capture such nonlinear and time-dependent relationships. Recent advances in deep learning, particularly sequence models such as LSTM networks, provide an effective mechanism to model these dependencies.  

This project aims to build a robust and interpretable framework for predicting **NIFTY 50 ATM call option prices** by combining:
- High-frequency historical market data  
- Quantified financial news sentiment  
- Deep learning-based time-series forecasting  
- Explainable AI (XAI) methodologies  

---

## 2. Data Sources

### 2.1 Market Data
- **Instrument**: NIFTY 50 Index  
- **Frequency**: 5-minute intraday data  
- **Period**: 2015–2025  
- **Fields**: Open, High, Low, Close, Volume  

This dataset provides approximately 975,000 observations, enabling long-horizon temporal modeling.

### 2.2 News and Sentiment Data
A curated dataset of approximately 3,000 global financial news events was used. Each record contains:
- Publication date  
- Headline and source  
- Market event type  
- Related index and sector  
- Qualitative sentiment label (Positive / Neutral / Negative)  

Sentiment labels are numerically encoded and aggregated into intraday windows to align with market data.

---

## 3. Feature Engineering

### 3.1 Market-Derived Features
- Log returns and momentum features  
- Lagged price and volatility terms  
- Realized volatility (rolling variance of returns)  
- Garman–Klass volatility estimator  

### 3.2 Option Pricing Features
Since historical option prices are not publicly available over long horizons, **ATM call option premiums** are reconstructed using:
- Black–Scholes option pricing model  
- Weekly fixed time-to-expiry assumption  
- Simulated implied volatility time series  

Additionally, key option Greeks are computed:
- Delta  
- Gamma  
- Vega  
- Theta  

### 3.3 Sentiment Features
Daily sentiment scores are aligned with intraday timestamps and transformed into:
- 1-hour sentiment average  
- 6-hour sentiment average  
- 24-hour sentiment average  
- News surprise metric (short-term vs long-term sentiment deviation)  

Interaction features combining sentiment and volatility are also included.

---

## 4. Model Architecture

A **Long Short-Term Memory (LSTM)** neural network is used for option price prediction. The model operates on sliding windows of past observations to capture temporal dependencies.

**Architecture Overview**:
- Two stacked LSTM layers  
- Dropout for regularization  
- Fully connected layers for regression output  

The target variable is the **ATM call option premium** at the next time step.

---

## 5. Training and Validation Strategy

To avoid look-ahead bias, a **walk-forward validation** approach is adopted:
- Fixed-length training window  
- Forward-only rolling test window  
- Repeated over the full dataset  

Baseline models are implemented for comparison:
- Moving average baseline  
- Random walk baseline  

---

## 6. Evaluation Metrics
Model performance is assessed using:
- Mean Absolute Error (MAE)  
- Root Mean Squared Error (RMSE)  
- R² score  
- Directional accuracy  

---

## 7. Explainability and Interpretability

To enhance transparency and trust:
- **SHAP (SHapley Additive exPlanations)** is used to identify globally influential features affecting option prices.  
- **LIME (Local Interpretable Model-agnostic Explanations)** provides instance-level explanations for individual predictions.  

These techniques enable interpretation of how market variables and sentiment contribute to predicted option premiums.

---

## 8. Backtesting and Trading Simulation

A simplified rule-based backtest is implemented:
- Long position initiated when predicted option price increases  
- Performance evaluated using:
  - Equity curve  
  - Sharpe ratio  
  - Sortino ratio  
  - Maximum drawdown  
  - Hit rate  

This evaluation demonstrates the economic relevance of the predictive model.

---

## 9. Project Structure

ExplainableAI-NiftyOptions/
│
├── data/
│ ├── nifty_intraday_2015_2025.csv
│ └── global_news_sentiment.csv
│
├── src/
│ ├── config.py
│ ├── data_preprocess.py
│ ├── feature_engineering.py
│ ├── models.py
│ ├── train_walk_forward.py
│ ├── explainability.py
│ └── backtest.py
│
├── main.py
└── requirements.txt

---

## 10. Reproducibility
The pipeline is modular and fully reproducible. All preprocessing steps, model configurations, and evaluation metrics are parameterized through a centralized configuration file.

---

## 11. Limitations and Future Work
- Synthetic option prices are used due to lack of historical option-chain data.  
- Future work may incorporate real-time option data feeds, transformer-based architectures, and reinforcement learning for dynamic trading strategies.

---

## 12. Conclusion
This project demonstrates a comprehensive and interpretable deep learning framework for NIFTY 50 option price prediction. By combining market dynamics, financial news sentiment, and explainable AI techniques, the system provides both predictive accuracy and transparency, making it suitable for academic research and practical financial analysis.

---







