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
    """Download historical daily OHLCV data for a supported stock."""

    if symbol not in TICKERS:
        raise ValueError(
            f"Unsupported symbol: {symbol}. "
            f"Available symbols: {list(TICKERS.keys())}"
        )

    ticker = TICKERS[symbol]

    data = yf.download(
        ticker,
        start=start,
        end=end,
        interval="1d",
        auto_adjust=True,
        progress=False,
    )

    if data.empty:
        raise ValueError(f"No market data returned for {symbol}.")

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
        column for column in required_columns
        if column not in data.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {missing_columns}"
        )

    data = data[required_columns].copy()

    data["Symbol"] = symbol
    data["Date"] = pd.to_datetime(data["Date"])

    data = data.sort_values("Date").reset_index(drop=True)

    return data


def save_raw_data(data: pd.DataFrame, symbol: str) -> Path:
    """Save downloaded market data locally."""

    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)

    file_path = RAW_DATA_DIR / f"{symbol.lower()}.csv"

    data.to_csv(file_path, index=False)

    return file_path

def load_raw_data(symbol: str) -> pd.DataFrame:
    """Load previously downloaded market data from local storage."""

    if symbol not in TICKERS:
        raise ValueError(
            f"Unsupported symbol: {symbol}. "
            f"Available symbols: {list(TICKERS.keys())}"
        )

    file_path = RAW_DATA_DIR / f"{symbol.lower()}.csv"

    if not file_path.exists():
        raise FileNotFoundError(
            f"No local data found for {symbol}: {file_path}"
        )

    data = pd.read_csv(file_path)

    data["Date"] = pd.to_datetime(data["Date"])

    return data