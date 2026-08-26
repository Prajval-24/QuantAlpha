import pandas as pd


REQUIRED_COLUMNS = [
    "Date",
    "Open",
    "High",
    "Low",
    "Close",
    "Volume",
    "Symbol",
]


def validate_market_data(data: pd.DataFrame) -> None:
    """Validate the structure and quality of market data."""

    # Check required columns
    missing_columns = [
        column
        for column in REQUIRED_COLUMNS
        if column not in data.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {missing_columns}"
        )

    # Check empty dataset
    if data.empty:
        raise ValueError("Market data is empty.")

    # Check duplicate dates
    if data["Date"].duplicated().any():
        raise ValueError("Duplicate dates found in market data.")

    # Check chronological order
    if not data["Date"].is_monotonic_increasing:
        raise ValueError("Market data is not sorted by date.")

    # Check missing values
    price_columns = [
        "Open",
        "High",
        "Low",
        "Close",
        "Volume",
    ]

    missing_values = data[price_columns].isna().sum()

    if missing_values.any():
        raise ValueError(
            f"Missing values found:\n{missing_values}"
        )

    # Prices must be positive
    if (data[price_columns] < 0).any().any():
        raise ValueError("Negative market values found.")

    # OHLC consistency
    invalid_high = data["High"] < data[["Open", "Close"]].max(axis=1)
    invalid_low = data["Low"] > data[["Open", "Close"]].min(axis=1)

    if invalid_high.any():
        raise ValueError(
            "Invalid OHLC data: High is below Open or Close."
        )

    if invalid_low.any():
        raise ValueError(
            "Invalid OHLC data: Low is above Open or Close."
        )


def add_basic_features(data: pd.DataFrame) -> pd.DataFrame:
    """Add basic time-series features used by later strategies."""

    data = data.copy()

    data["Return_1D"] = data["Close"].pct_change()

    data["Return_5D"] = data["Close"].pct_change(5)

    data["Return_20D"] = data["Close"].pct_change(20)

    data["Volatility_20D"] = (
        data["Return_1D"]
        .rolling(window=20)
        .std()
    )

    data["MA_20"] = (
        data["Close"]
        .rolling(window=20)
        .mean()
    )

    data["MA_Distance"] = (
        data["Close"] / data["MA_20"] - 1
    )

    data["Volume_Change"] = (
        data["Volume"].pct_change()
    )

    return data