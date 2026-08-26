import pandas as pd


def equal_weight(
    signals: pd.DataFrame,
) -> pd.DataFrame:
    """
    Allocate equal weight across all active positions.

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
        Portfolio weights with the same shape as signals.
    """

    active = signals.clip(lower=0)

    active_count = active.sum(axis=1)

    weights = active.div(
        active_count.replace(0, pd.NA),
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
    """

    if lookback <= 0:
        raise ValueError(
            "Lookback must be greater than zero."
        )

    if not returns.index.equals(signals.index):
        raise ValueError(
            "Returns and signals must have the same index."
        )

    volatility = (
        returns
        .rolling(lookback)
        .std()
    )

    inverse_volatility = 1 / volatility.replace(
        0,
        pd.NA,
    )

    active_inverse_volatility = (
        inverse_volatility
        * signals.clip(lower=0)
    )

    total_inverse_volatility = (
        active_inverse_volatility.sum(axis=1)
    )

    weights = active_inverse_volatility.div(
        total_inverse_volatility.replace(0, pd.NA),
        axis=0,
    )

    return weights.fillna(0.0)


def validate_weights(
    weights: pd.DataFrame,
    tolerance: float = 1e-8,
) -> None:
    """
    Validate portfolio weights.

    Ensures:
    - no negative weights
    - total exposure does not exceed 100%
    """

    if (weights < -tolerance).any().any():
        raise ValueError(
            "Portfolio contains negative weights."
        )

    total_exposure = weights.sum(axis=1)

    if (total_exposure > 1 + tolerance).any():
        raise ValueError(
            "Portfolio exposure exceeds 100%."
        )