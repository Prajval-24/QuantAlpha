# 📈 QuantAlpha: ML-Driven Market Analytics Platform

> An institutional-grade quantitative finance research platform built to ingest historical financial time-series data, execute out-of-sample machine learning models without look-ahead bias, and simulate path-dependent portfolio allocations with realistic transaction friction.

[![Live Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://your-streamlit-app-url.streamlit.app)
[![Python Version](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

## 🏛️ Overview

**QuantAlpha** bridges the gap between raw financial data ingestion, out-of-sample machine learning prediction, and rigorous portfolio risk execution. It provides both a backend REST API framework and an interactive frontend dashboard for quantitative researchers and developers.

---

## 🚀 Key Features

* **⚡ FastAPI & SQLite Backend:** Ingests, processes, and serves historical financial time-series data with optimized database queries for low-latency retrieval.
* **🌲 Scikit-Learn Walk-Forward ML Pipeline:** Implements rolling walk-forward Random Forest classifiers to predict out-of-sample directional returns, strictly eliminating look-ahead bias.
* **📊 Multi-Asset Alpha Strategies:** Supports Momentum (Moving Average), Mean Reversion (Z-Score), and Supervised Machine Learning alpha generation across multiple equities.
* **⚖️ Institutional Risk & Allocation Engine:** Translates raw alpha signals into optimal portfolio asset weights using Equal Weighting, Inverse Volatility, Volatility Targeting, and Maximum Diversification while enforcing long-only constraints.
* **📉 Path-Dependent Backtesting:** Simulates portfolio drift, dynamic turnover calculation, and transaction cost/slippage friction.
* **🌐 Cloud-Deployed Interactive UI:** Built using **Streamlit** for real-time visualization of performance metrics, trade data, and cumulative equity curves.

---

## 🛠️ Tech Stack

* **Programming Language:** Python
* **Backend Framework:** FastAPI, Uvicorn
* **Machine Learning:** Scikit-Learn (Random Forest Classifiers)
* **Data Processing & Storage:** Pandas, NumPy, SQLite, yFinance
* **Dashboard & Visualization:** Streamlit
* **Version Control:** Git, GitHub

---

## 📂 Project Architecture

```text
QuantAlpha/
│
├── app.py                     # Streamlit frontend analytics dashboard
├── requirements.txt           # Project Python dependencies
├── data/                      # Local data storage (raw OHLCV CSVs & SQLite DB)
│   └── raw/
├── src/                       # Core application logic
│   ├── api/                   # FastAPI backend endpoints
│   ├── alphas/                # Alpha signal generators (Momentum, Mean Reversion, ML)
│   ├── data/                  # Data loaders and feature preprocessing
│   ├── backtest/              # Performance metrics and evaluation engines
│   └── portfolio/             # Portfolio construction, risk controls, and execution engine
└── tests/                     # Unit and integration test suites