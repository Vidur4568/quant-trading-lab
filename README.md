# Quant Trading Lab

Machine learning research on hourly Bitcoin returns: lagged-return features, a PyTorch model, and long/short signals evaluated on a held-out chronological test set.

## Overview

This project asks a focused question: do the three most recent hourly Bitcoin returns contain enough information to predict the direction of the next hourly return, and can that be turned into a simple systematic long/short strategy?

It covers the full research loop on a small scale: data loading, log-return and feature construction, exploratory analysis, model training, signal generation, and out-of-sample evaluation. The results are preliminary, and the limitations and roadmap below matter as much as the numbers.

## Key Results

Evaluated on a chronological 70% / 30% train/test split:

| Metric | Result |
|---|---|
| Asset | Bitcoin (BTC/USDT) |
| Data frequency | Hourly |
| Observations analyzed | 43K+ |
| Lagged-return features | 3 |
| Training / testing split | 70% / 30% |
| Out-of-sample observations | 13K+ |
| Directional accuracy | 50.7% |
| Gross annualized Sharpe ratio | 1.49 |

> **Important:** The Sharpe ratio is gross of transaction costs, fees, and slippage, and does not subtract a risk-free rate. A 50.7% hit rate is a small edge that has not been shown to be statistically significant. These are research results, not live trading performance.

## Data

Hourly BTC/USDT bars (`BTCUSDT-1h.csv`) indexed by `open_time`, with 43K+ observations.

## Features

Hourly log returns are computed from closing prices:

$$r_t = \ln\left(\frac{P_t}{P_{t-1}}\right)$$

where $P_t$ is the closing price at hour $t$. Three lagged features are built from this series:

- `close_log_return_lag_1`: return one hour earlier
- `close_log_return_lag_2`: return two hours earlier
- `close_log_return_lag_3`: return three hours earlier

Rows with missing lags are dropped. The model never sees the return it is predicting, only the three returns before it.

Exploratory analysis in the notebook includes correlation matrices, pair plots, and return distributions.

## Model

A PyTorch model takes the three lagged returns as input and predicts the current hour's log return.

The data is split chronologically: the model is developed on the first 70% of observations and evaluated on the remaining 30% (13K+ observations) that were not used for training. A chronological split avoids the look-ahead bias that random shuffling would introduce in time-series data. Trained weights are saved to `notebooks/model.pth`.

## Signals and Backtest

The sign of the predicted return sets the position:

| Predicted return | Position |
|---|---|
| Positive | Long (+1) |
| Negative | Short (-1) |

Strategy log returns multiply each position by the realized return:

$$r_t^{strategy} = s_t \cdot r_t$$

where $s_t \in \{-1, +1\}$ is the signal. Cumulative log returns are converted to a gross equity curve using $100 \cdot e^{\sum r^{strategy}}$.

## Evaluation

- **Directional accuracy:** the share of test hours where the sign of the signal matches the sign of the realized return (about 50.7%).
- **Annualized Sharpe ratio:** mean divided by standard deviation of hourly strategy log returns, scaled by $\sqrt{365 \times 24}$ because Bitcoin trades continuously (about 1.49, gross).

## Limitations

- **No transaction costs.** The backtest excludes fees, bid-ask spread, slippage, and funding costs. The strategy can change position every hour, so costs could remove much of the gross performance.
- **Baselines not yet compared.** Results have not yet been compared to buy-and-hold or to the share of hours Bitcoin simply rose.
- **Limited features.** The model uses only three lagged returns.
- **Single asset and single split.** Evaluation uses one asset and one 70/30 chronological split, not walk-forward validation.
- **Statistical significance.** 50.7% directional accuracy on about 13K observations does not by itself demonstrate a significant predictive edge, and returns are serially dependent.
- **Market regimes.** Performance has not been checked across high-volatility, low-volatility, trending, or mean-reverting periods.
- **Execution realism.** This is a simplified simulation and does not show that signals could be executed at the assumed prices.

## Project Structure

```text
quant-trading-lab/
├── data/            # Historical market data
├── notebooks/       # Jupyter notebooks and saved model weights
├── src/             # Python modules
├── requirements.txt
└── README.md
```

## Technologies

Python, PyTorch, Pandas, NumPy, Seaborn, Jupyter Notebook, Git, GitHub

## How to Run

```bash
git clone https://github.com/Vidur4568/quant-trading-lab.git
cd quant-trading-lab
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
jupyter notebook notebooks/
```

## Roadmap

- [ ] Add transaction costs and slippage, and report turnover
- [ ] Compare against buy-and-hold and simple baselines
- [ ] Add maximum drawdown and volatility metrics
- [ ] Implement walk-forward validation and test statistical significance
- [ ] Compare the neural model with linear and logistic regression
- [ ] Test additional features such as volatility and volume
- [ ] Evaluate across market regimes and additional assets

## Disclaimer

This repository is for educational and research purposes only. Results are experimental, do not represent live trading performance, and are not investment advice.
