import numpy as np
import pandas as pd


FEATURE_COLUMNS = [
    "Return_1D",
    "Return_5D",
    "Return_20D",
    "MA_Distance_10D",
    "MA_Distance_20D",
    "Volatility_20D",
    "Volume_Change",
    "Volume_Ratio_20D",
]


def build_ml_features(
    data: pd.DataFrame,
) -> pd.DataFrame:
    """
    Build machine-learning features from OHLCV data.

    All features use information available on or before
    the current trading day.

    Target:
        1 = next-day return is positive
        0 = next-day return is non-positive
    """

    df = data.copy()

    # --------------------------------------------------
    # 1. Validate required columns
    # --------------------------------------------------

    required_columns = {
        "Date",
        "Open",
        "High",
        "Low",
        "Close",
        "Volume",
    }

    missing = required_columns - set(df.columns)

    if missing:
        raise ValueError(
            f"Missing required columns: {sorted(missing)}"
        )

    # --------------------------------------------------
    # 2. Sort chronologically
    # --------------------------------------------------

    df = (
        df
        .sort_values("Date")
        .reset_index(drop=True)
    )

    # --------------------------------------------------
    # 3. One-day return
    # --------------------------------------------------

    df["Return_1D"] = (
        df["Close"].pct_change()
    )

    # --------------------------------------------------
    # 4. Momentum features
    # --------------------------------------------------

    df["Return_5D"] = (
        df["Close"].pct_change(5)
    )

    df["Return_20D"] = (
        df["Close"].pct_change(20)
    )

    # --------------------------------------------------
    # 5. Moving-average distance
    # --------------------------------------------------

    ma_10 = (
        df["Close"]
        .rolling(10)
        .mean()
    )

    ma_20 = (
        df["Close"]
        .rolling(20)
        .mean()
    )

    df["MA_Distance_10D"] = (
        df["Close"] / ma_10 - 1
    )

    df["MA_Distance_20D"] = (
        df["Close"] / ma_20 - 1
    )

    # --------------------------------------------------
    # 6. Historical volatility
    # --------------------------------------------------

    df["Volatility_20D"] = (
        df["Return_1D"]
        .rolling(20)
        .std()
    )

    # --------------------------------------------------
    # 7. Volume features
    # --------------------------------------------------

    df["Volume_Change"] = (
        df["Volume"].pct_change()
    )

    average_volume_20 = (
        df["Volume"]
        .rolling(20)
        .mean()
    )

    df["Volume_Ratio_20D"] = (
        df["Volume"] / average_volume_20
    )

    # --------------------------------------------------
    # 8. Replace infinite values
    # --------------------------------------------------
    #
    # pct_change() or division can produce:
    #     +inf
    #     -inf
    #
    # These are invalid ML inputs.
    # Convert them to NaN so they can be removed below.
    # --------------------------------------------------

    df = df.replace(
        [np.inf, -np.inf],
        np.nan,
    )

    # --------------------------------------------------
    # 9. Prediction target
    # --------------------------------------------------

    next_day_return = (
        df["Close"].shift(-1)
        / df["Close"]
        - 1
    )

    df["Target"] = (
        next_day_return > 0
    ).where(
        next_day_return.notna()
    )

    # --------------------------------------------------
    # 10. Remove rows where features or target
    #     cannot be calculated
    # --------------------------------------------------

    df = (
        df
        .dropna(
            subset=FEATURE_COLUMNS + ["Target"]
        )
        .reset_index(drop=True)
    )

    # --------------------------------------------------
    # 11. Convert target to integer
    # --------------------------------------------------

    df["Target"] = (
        df["Target"]
        .astype(int)
    )

    return df


def get_ml_dataset(
    data: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.Series]:
    """
    Return feature matrix X and target vector y.
    """

    df = build_ml_features(data)

    X = df[
        FEATURE_COLUMNS
    ].copy()

    y = df[
        "Target"
    ].copy()

    return X, y


def temporal_train_test_split(
    data: pd.DataFrame,
    test_size: float = 0.2,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split ML data chronologically.

    The earliest observations are used for training.
    The most recent observations are reserved for
    out-of-sample testing.

    No random shuffling is performed.
    """

    if not 0 < test_size < 1:
        raise ValueError(
            "test_size must be between 0 and 1."
        )

    df = data.copy()

    if "Target" not in df.columns:
        raise ValueError(
            "Input data must contain Target."
        )

    # --------------------------------------------------
    # Ensure chronological ordering
    # --------------------------------------------------

    df = (
        df
        .sort_values("Date")
        .reset_index(drop=True)
    )

    # --------------------------------------------------
    # Calculate split point
    # --------------------------------------------------

    split_index = int(
        len(df) * (1 - test_size)
    )

    # --------------------------------------------------
    # Create train/test datasets
    # --------------------------------------------------

    train = (
        df
        .iloc[:split_index]
        .copy()
    )

    test = (
        df
        .iloc[split_index:]
        .copy()
    )

    # --------------------------------------------------
    # Validate split
    # --------------------------------------------------

    if train.empty or test.empty:
        raise ValueError(
            "Train/test split produced an empty dataset."
        )

    if train["Date"].max() >= test["Date"].min():
        raise ValueError(
            "Temporal split is invalid: "
            "train data overlaps test data."
        )

    return train, test