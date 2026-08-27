import numpy as np
import pandas as pd


def _validate_long_only_signals(
    signals: pd.DataFrame,
) -> None:
    """
    Validate signals for the current long-only portfolio architecture.

    Allowed values:
        1 = long
        0 = no position

    Short signals (-1) are intentionally rejected rather than silently
    discarded.
    """

    if not isinstance(signals, pd.DataFrame):
        raise TypeError(
            "Signals must be a pandas DataFrame."
        )

    if signals.empty:
        raise ValueError(
            "Signals DataFrame cannot be empty."
        )

    if not signals.index.is_monotonic_increasing:
        raise ValueError(
            "Signals index must be sorted chronologically."
        )

    if signals.columns.duplicated().any():
        raise ValueError(
            "Signals contain duplicate asset columns."
        )

    values = signals.to_numpy(dtype=float)

    if not np.isfinite(values).all():
        raise ValueError(
            "Signals contain NaN or infinite values."
        )

    unique_values = np.unique(values)

    invalid_values = unique_values[
        ~np.isin(unique_values, [0.0, 1.0])
    ]

    if len(invalid_values) > 0:
        raise ValueError(
            "Long-only portfolio accepts only signals "
            "0 and 1. Invalid values found: "
            f"{invalid_values.tolist()}"
        )


def equal_weight(
    signals: pd.DataFrame,
) -> pd.DataFrame:
    """
    Allocate equal weight across all active long positions.

    Parameters
    ----------
    signals:
        DataFrame where columns are asset symbols and values are
        trading signals:

            1 = long
            0 = no position

    Returns
    -------
    pd.DataFrame
        Portfolio weights with the same index and columns as signals.

    Notes
    -----
    This portfolio construction is intentionally long-only.
    Short signals are rejected rather than silently discarded.
    """

    _validate_long_only_signals(signals)

    active_count = signals.sum(axis=1)

    weights = signals.div(
        active_count.replace(0, np.nan),
        axis=0,
    )

    return weights.fillna(0.0)


def inverse_volatility_weight(
    returns: pd.DataFrame,
    signals: pd.DataFrame,
    lookback: int = 20,
) -> pd.DataFrame:
    """
    Allocate capital using inverse-volatility weighting.

    Assets with lower recent volatility receive higher weight.

        Weight_i ∝ 1 / volatility_i

    Only assets with an active long signal receive capital.

    Parameters
    ----------
    returns:
        Historical asset returns. Rows must be chronological and
        columns must correspond to signals.

    signals:
        Long-only trading signals:

            1 = long
            0 = no position

    lookback:
        Number of observations used to estimate volatility.

    Returns
    -------
    pd.DataFrame
        Portfolio weights with the same shape as signals.
    """

    if not isinstance(returns, pd.DataFrame):
        raise TypeError(
            "Returns must be a pandas DataFrame."
        )

    if not isinstance(signals, pd.DataFrame):
        raise TypeError(
            "Signals must be a pandas DataFrame."
        )

    if returns.empty:
        raise ValueError(
            "Returns DataFrame cannot be empty."
        )

    if signals.empty:
        raise ValueError(
            "Signals DataFrame cannot be empty."
        )

    if lookback <= 0:
        raise ValueError(
            "Lookback must be greater than zero."
        )

    if not returns.index.equals(signals.index):
        raise ValueError(
            "Returns and signals must have the same index."
        )

    if not returns.columns.equals(signals.columns):
        raise ValueError(
            "Returns and signals must have the same columns "
            "in the same order."
        )

    if not returns.index.is_monotonic_increasing:
        raise ValueError(
            "Returns index must be sorted chronologically."
        )

    _validate_long_only_signals(signals)

    return_values = returns.to_numpy(dtype=float)

    if not np.isfinite(return_values).all():
        raise ValueError(
            "Returns contain NaN or infinite values."
        )

    volatility = (
        returns
        .rolling(
            window=lookback,
            min_periods=lookback,
        )
        .std(ddof=1)
    )

    inverse_volatility = (
        1.0
        / volatility.replace(0.0, np.nan)
    )

    active_inverse_volatility = (
        inverse_volatility * signals
    )

    total_inverse_volatility = (
        active_inverse_volatility.sum(axis=1)
    )

    weights = active_inverse_volatility.div(
        total_inverse_volatility.replace(0.0, np.nan),
        axis=0,
    )

    return weights.fillna(0.0)


def validate_weights(
    weights: pd.DataFrame,
    tolerance: float = 1e-8,
) -> None:
    """
    Validate long-only portfolio weights.

    Ensures:
    - weights are finite
    - no materially negative weights
    - total gross exposure does not exceed 100%

    Cash is permitted, so total exposure may be below 100%.
    """

    if not isinstance(weights, pd.DataFrame):
        raise TypeError(
            "Weights must be a pandas DataFrame."
        )

    if weights.empty:
        raise ValueError(
            "Weights DataFrame cannot be empty."
        )

    if tolerance < 0:
        raise ValueError(
            "Tolerance cannot be negative."
        )

    values = weights.to_numpy(dtype=float)

    if not np.isfinite(values).all():
        raise ValueError(
            "Portfolio weights contain NaN or infinite values."
        )

    if (weights < -tolerance).any().any():
        raise ValueError(
            "Portfolio contains negative weights. "
            "The current portfolio architecture is long-only."
        )

    total_exposure = weights.sum(axis=1)

    if (total_exposure > 1.0 + tolerance).any():
        raise ValueError(
            "Portfolio exposure exceeds 100%."
        )