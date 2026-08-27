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


REQUIRED_COLUMNS = {
    "Date",
    "Open",
    "High",
    "Low",
    "Close",
    "Volume",
}


def _build_features_for_group(
    group: pd.DataFrame,
) -> pd.DataFrame:
    """
    Build time-series features for ONE asset.

    All sequential operations such as pct_change(),
    rolling(), and shift() are therefore guaranteed
    to stay within the same asset.
    """

    df = (
        group
        .sort_values("Date")
        .copy()
    )

    # --------------------------------------------------
    # Returns
    # --------------------------------------------------

    df["Return_1D"] = (
        df["Close"]
        .pct_change()
    )

    df["Return_5D"] = (
        df["Close"]
        .pct_change(5)
    )

    df["Return_20D"] = (
        df["Close"]
        .pct_change(20)
    )

    # --------------------------------------------------
    # Moving-average distance
    # --------------------------------------------------

    ma_10 = (
        df["Close"]
        .rolling(
            window=10,
            min_periods=10,
        )
        .mean()
    )

    ma_20 = (
        df["Close"]
        .rolling(
            window=20,
            min_periods=20,
        )
        .mean()
    )

    df["MA_Distance_10D"] = (
        df["Close"] / ma_10 - 1.0
    )

    df["MA_Distance_20D"] = (
        df["Close"] / ma_20 - 1.0
    )

    # --------------------------------------------------
    # Historical volatility
    # --------------------------------------------------

    df["Volatility_20D"] = (
        df["Return_1D"]
        .rolling(
            window=20,
            min_periods=20,
        )
        .std()
    )

    # --------------------------------------------------
    # Volume features
    # --------------------------------------------------

    df["Volume_Change"] = (
        df["Volume"]
        .pct_change()
    )

    average_volume_20 = (
        df["Volume"]
        .rolling(
            window=20,
            min_periods=20,
        )
        .mean()
    )

    df["Volume_Ratio_20D"] = (
        df["Volume"]
        / average_volume_20
    )

    # --------------------------------------------------
    # Replace invalid numerical values
    # --------------------------------------------------

    df = df.replace(
        [np.inf, -np.inf],
        np.nan,
    )

    # --------------------------------------------------
    # Next-day target
    #
    # IMPORTANT:
    # shift(-1) happens INSIDE this asset only.
    # --------------------------------------------------

    next_day_return = (
        df["Close"].shift(-1)
        / df["Close"]
        - 1.0
    )

    df["Target"] = (
        next_day_return > 0
    ).where(
        next_day_return.notna()
    )

    return df


def build_ml_features(
    data: pd.DataFrame,
    drop_unlabeled: bool = True,
) -> pd.DataFrame:
    """
    Build machine-learning features from OHLCV data.

    Parameters
    ----------
    data:
        OHLCV dataframe.

        If a "Symbol" column exists, all sequential
        operations are performed independently for
        each symbol.

    drop_unlabeled:
        If True, remove rows whose next-day target
        cannot be calculated.

        This is appropriate for training.

        If False, retain the latest row with Target=NaN.
        This is useful for live inference.

    Returns
    -------
    pd.DataFrame
        Original data plus ML features and Target.
    """

    df = data.copy()

    # --------------------------------------------------
    # Validate required columns
    # --------------------------------------------------

    missing = (
        REQUIRED_COLUMNS
        - set(df.columns)
    )

    if missing:
        raise ValueError(
            "Missing required columns: "
            f"{sorted(missing)}"
        )

    if df.empty:
        raise ValueError(
            "Input data is empty."
        )

    # --------------------------------------------------
    # Normalize dates
    # --------------------------------------------------

    df["Date"] = pd.to_datetime(
        df["Date"]
    )

    # --------------------------------------------------
    # Validate ordering keys
    # --------------------------------------------------

    if "Symbol" in df.columns:

        duplicate_mask = (
            df.duplicated(
                subset=["Symbol", "Date"],
                keep=False,
            )
        )

        if duplicate_mask.any():
            duplicates = (
                df.loc[
                    duplicate_mask,
                    ["Symbol", "Date"],
                ]
                .head(10)
            )

            raise ValueError(
                "Duplicate Symbol/Date observations "
                "found in ML input.\n"
                f"{duplicates}"
            )

        # --------------------------------------------------
        # Multi-asset data
        #
        # Features are constructed independently per
        # symbol. This prevents cross-asset contamination.
        # --------------------------------------------------

        result = (
    df
    .groupby(
        "Symbol",
        group_keys=False,
        sort=False,
    )
    .apply(
        _build_features_for_group,
    )
)

    else:

        # --------------------------------------------------
        # Single-asset data
        # --------------------------------------------------

        if df["Date"].duplicated().any():
            raise ValueError(
                "Duplicate dates found in ML input."
            )

        result = _build_features_for_group(
            df
        )

    # --------------------------------------------------
    # Sort final result
    # --------------------------------------------------

    sort_columns = (
        ["Symbol", "Date"]
        if "Symbol" in result.columns
        else ["Date"]
    )

    result = (
        result
        .sort_values(sort_columns)
        .reset_index(drop=True)
    )

    # --------------------------------------------------
    # Validate numerical features
    # --------------------------------------------------

    finite_check = (
        result[FEATURE_COLUMNS]
        .replace(
            [np.inf, -np.inf],
            np.nan,
        )
    )

    # --------------------------------------------------
    # Drop rows where features cannot exist
    #
    # Target is handled separately so that the latest
    # inference row can optionally be retained.
    # --------------------------------------------------

    result = (
        result
        .dropna(
            subset=FEATURE_COLUMNS
        )
        .copy()
    )

    # --------------------------------------------------
    # Training mode
    # --------------------------------------------------

    if drop_unlabeled:

        result = (
            result
            .dropna(
                subset=["Target"]
            )
            .copy()
        )

        result["Target"] = (
            result["Target"]
            .astype(int)
        )

    # --------------------------------------------------
    # Final safety check
    # --------------------------------------------------

    if not np.isfinite(
        result[FEATURE_COLUMNS]
        .to_numpy(dtype=float)
    ).all():
        raise ValueError(
            "Non-finite ML feature values remain "
            "after feature construction."
        )

    return result


def get_ml_dataset(
    data: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.Series]:
    """
    Return feature matrix X and target vector y
    for model training.
    """

    df = build_ml_features(
        data,
        drop_unlabeled=True,
    )

    X = (
        df[FEATURE_COLUMNS]
        .copy()
    )

    y = (
        df["Target"]
        .copy()
    )

    return X, y


def temporal_train_test_split(
    data: pd.DataFrame,
    test_size: float = 0.2,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split ML data chronologically using DATE boundaries.

    For multi-asset data, every asset belonging to the same
    trading date remains entirely in either train or test.

    No random shuffling is performed.
    """

    if not 0 < test_size < 1:
        raise ValueError(
            "test_size must be between 0 and 1."
        )

    df = data.copy()

    if "Date" not in df.columns:
        raise ValueError(
            "Input data must contain Date."
        )

    if "Target" not in df.columns:
        raise ValueError(
            "Input data must contain Target."
        )

    df["Date"] = pd.to_datetime(
        df["Date"]
    )

    df = (
        df
        .sort_values("Date")
        .reset_index(drop=True)
    )

    unique_dates = (
        pd.DatetimeIndex(
            df["Date"]
            .drop_duplicates()
            .sort_values()
        )
    )

    n_dates = len(
        unique_dates
    )

    if n_dates < 2:
        raise ValueError(
            "At least two unique dates are required "
            "for temporal train/test splitting."
        )

    test_date_count = max(
        1,
        int(
            np.ceil(
                n_dates * test_size
            )
        ),
    )

    if test_date_count >= n_dates:
        test_date_count = n_dates - 1

    split_position = (
        n_dates - test_date_count
    )

    cutoff_date = (
        unique_dates[split_position]
    )

    train = (
        df[
            df["Date"] < cutoff_date
        ]
        .copy()
    )

    test = (
        df[
            df["Date"] >= cutoff_date
        ]
        .copy()
    )

    # --------------------------------------------------
    # Validate split
    # --------------------------------------------------

    if train.empty or test.empty:
        raise ValueError(
            "Train/test split produced an "
            "empty dataset."
        )

    train_max = train["Date"].max()
    test_min = test["Date"].min()

    if train_max >= test_min:
        raise ValueError(
            "Temporal split is invalid: "
            "train data overlaps test data."
        )

    return train, test