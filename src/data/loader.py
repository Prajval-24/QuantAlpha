from datetime import date, timedelta
from pathlib import Path

import pandas as pd
import yfinance as yf


TICKERS = {
    "RELIANCE": "RELIANCE.NS",
    "TCS": "TCS.NS",
    "INFY": "INFY.NS",
    "HDFCBANK": "HDFCBANK.NS",
    "ICICIBANK": "ICICIBANK.NS",
}

RAW_DATA_DIR = Path("data/raw")


def download_stock_data(
    symbol: str,
    start: str = "2021-01-01",
    end: str | None = None,
) -> pd.DataFrame:
    """Download complete historical daily OHLCV data."""

    if symbol not in TICKERS:
        raise ValueError(
            f"Unsupported symbol: {symbol}. "
            f"Available symbols: {list(TICKERS.keys())}"
        )

    ticker = TICKERS[symbol]

    # yfinance treats `end` as exclusive.
    # Using tomorrow as the end date when the user explicitly
    # provides an end date preserves that requested range.
    #
    # When end=None, use tomorrow so that the current completed
    # trading day is included while avoiding a future-date issue.
    if end is None:
        end_date = date.today() + timedelta(days=1)
        end = end_date.isoformat()

    data = yf.download(
        ticker,
        start=start,
        end=end,
        interval="1d",
        auto_adjust=True,
        progress=False,
    )

    if data.empty:
        raise ValueError(
            f"No market data returned for {symbol}."
        )

    data = data.reset_index()

    # yfinance can sometimes return MultiIndex columns.
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)

    required_columns = [
        "Date",
        "Open",
        "High",
        "Low",
        "Close",
        "Volume",
    ]

    missing_columns = [
        column
        for column in required_columns
        if column not in data.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {missing_columns}"
        )

    data = data[required_columns].copy()

    # ---------------------------------------------------------
    # FIX: Drop rows where Yahoo Finance returned NaN prices
    # (e.g., market holidays, trading halts)
    # ---------------------------------------------------------
    data = data.dropna(subset=["Open", "High", "Low", "Close"]).copy()

    data["Symbol"] = symbol

    # ---------------------------------------------------------
    # FIX: Standardize timezone to prevent downstream merge crashes
    # ---------------------------------------------------------
    data["Date"] = pd.to_datetime(data["Date"]).dt.tz_localize(None)

    # Remove any incomplete current-day bar that may be returned
    # by the data provider.
    today = pd.Timestamp.today().normalize()

    data = data[
        data["Date"].dt.normalize() < today
    ].copy()

    data = (
        data
        .sort_values("Date")
        .drop_duplicates(subset="Date")
        .reset_index(drop=True)
    )

    if data.empty:
        raise ValueError(
            f"No completed daily market data available for {symbol}."
        )

    return data


def save_raw_data(
    data: pd.DataFrame,
    symbol: str,
) -> Path:
    """Save downloaded market data locally."""

    if symbol not in TICKERS:
        raise ValueError(
            f"Unsupported symbol: {symbol}. "
            f"Available symbols: {list(TICKERS.keys())}"
        )

    RAW_DATA_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    file_path = (
        RAW_DATA_DIR
        / f"{symbol.lower()}.csv"
    )

    data.to_csv(
        file_path,
        index=False,
    )

    return file_path


def load_raw_data(
    symbol: str,
) -> pd.DataFrame:
    """Load previously downloaded market data from local storage."""

    if symbol not in TICKERS:
        raise ValueError(
            f"Unsupported symbol: {symbol}. "
            f"Available symbols: {list(TICKERS.keys())}"
        )

    file_path = (
        RAW_DATA_DIR
        / f"{symbol.lower()}.csv"
    )

    if not file_path.exists():
        raise FileNotFoundError(
            f"No local data found for {symbol}: {file_path}"
        )

    data = pd.read_csv(
        file_path
    )

    data["Date"] = pd.to_datetime(
        data["Date"]
    )

    data = (
        data
        .sort_values("Date")
        .drop_duplicates(subset="Date")
        .reset_index(drop=True)
    )

    return data
