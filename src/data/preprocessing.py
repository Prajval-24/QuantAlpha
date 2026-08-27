import numpy as np
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


def validate_market_data(
    data: pd.DataFrame,
) -> None:
    """Validate the structure and quality of market data."""

    # --------------------------------------------------
    # Required columns
    # --------------------------------------------------

    missing_columns = [
        column
        for column in REQUIRED_COLUMNS
        if column not in data.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {missing_columns}"
        )

    # --------------------------------------------------
    # Empty dataset
    # --------------------------------------------------

    if data.empty:
        raise ValueError(
            "Market data is empty."
        )

    # --------------------------------------------------
    # Date validation
    # --------------------------------------------------

    if data["Date"].isna().any():
        raise ValueError(
            "Missing dates found in market data."
        )

    if data["Date"].duplicated().any():
        raise ValueError(
            "Duplicate dates found in market data."
        )

    if not data["Date"].is_monotonic_increasing:
        raise ValueError(
            "Market data is not sorted by date."
        )

    # --------------------------------------------------
    # Numeric columns
    # --------------------------------------------------

    price_columns = [
        "Open",
        "High",
        "Low",
        "Close",
    ]

    numeric_columns = (
        price_columns
        + ["Volume"]
    )

    non_numeric = [
        column
        for column in numeric_columns
        if not pd.api.types.is_numeric_dtype(
            data[column]
        )
    ]

    if non_numeric:
        raise ValueError(
            f"Non-numeric market columns: {non_numeric}"
        )

    # --------------------------------------------------
    # Missing values
    # --------------------------------------------------

    missing_values = (
        data[numeric_columns]
        .isna()
        .sum()
    )

    if missing_values.any():
        raise ValueError(
            f"Missing values found:\n{missing_values}"
        )

    # --------------------------------------------------
    # Infinite values
    # --------------------------------------------------

    if np.isinf(
        data[numeric_columns].to_numpy()
    ).any():
        raise ValueError(
            "Infinite market values found."
        )

    # --------------------------------------------------
    # Price validation
    # --------------------------------------------------

    if (
        data[price_columns] <= 0
    ).any().any():
        raise ValueError(
            "Market prices must be strictly positive."
        )

    # --------------------------------------------------
    # Volume validation
    # --------------------------------------------------

    if (
        data["Volume"] < 0
    ).any():
        raise ValueError(
            "Volume cannot be negative."
        )

    # --------------------------------------------------
    # OHLC consistency
    # --------------------------------------------------

    invalid_high = (
        data["High"]
        < data[
            ["Open", "Close"]
        ].max(axis=1)
    )

    invalid_low = (
        data["Low"]
        > data[
            ["Open", "Close"]
        ].min(axis=1)
    )

    invalid_range = (
        data["High"]
        < data["Low"]
    )

    if invalid_high.any():
        raise ValueError(
            "Invalid OHLC data: "
            "High is below Open or Close."
        )

    if invalid_low.any():
        raise ValueError(
            "Invalid OHLC data: "
            "Low is above Open or Close."
        )

    if invalid_range.any():
        raise ValueError(
            "Invalid OHLC data: "
            "High is below Low."
        )


def add_basic_features(
    data: pd.DataFrame,
) -> pd.DataFrame:
    """Add basic time-series features used by later strategies."""

    data = (
        data
        .copy()
        .sort_values("Date")
        .reset_index(drop=True)
    )

    data["Return_1D"] = (
        data["Close"].pct_change()
    )

    data["Return_5D"] = (
        data["Close"].pct_change(5)
    )

    data["Return_20D"] = (
        data["Close"].pct_change(20)
    )

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
        data["Close"]
        / data["MA_20"]
        - 1
    )

    # pct_change can produce +/-inf when the
    # previous volume is zero. Treat these as
    # unavailable observations rather than valid features.
    data["Volume_Change"] = (
        data["Volume"]
        .pct_change()
        .replace(
            [np.inf, -np.inf],
            np.nan,
        )
    )

    return data