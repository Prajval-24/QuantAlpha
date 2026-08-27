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


def volatility_target_weight(
    returns: pd.DataFrame,
    signals: pd.DataFrame,
    target_volatility: float = 0.15,
    lookback: int = 20,
    periods_per_year: int = 252,
) -> pd.DataFrame:
    """
    Construct portfolio weights scaled to target a constant annualized volatility.
    """
    base_weights = inverse_volatility_weight(
        returns=returns, signals=signals, lookback=lookback
    )
    
    # Compute rolling sample covariance matrix and portfolio variance correctly: w^T * Cov * w
    # We estimate rolling covariance using historical returns over the lookback window
    scaled_weights_list = []
    
    # Loop over rolling windows safely to avoid look-ahead bias and matrix shape crashes
    for i in range(len(returns)):
        if i < lookback:
            scaled_weights_list.append(base_weights.iloc[i] * 0.0)
            continue
            
        window_returns = returns.iloc[i - lookback : i]
        cov_matrix = window_returns.cov() * periods_per_year
        w = base_weights.iloc[i].to_numpy(dtype=float)
        
        # Realized annualized portfolio variance = w^T * Cov * w
        port_var = np.dot(w.T, np.dot(cov_matrix.to_numpy(), w))
        port_vol = np.sqrt(max(port_var, 1e-12))
        
        # Scaling factor = Target Vol / Realized Vol (capped at 2.0x leverage)
        scaling_factor = min(target_volatility / port_vol, 2.0) if port_vol > 0 else 1.0
        
        scaled_weights_list.append(base_weights.iloc[i] * scaling_factor)
        
    scaled_weights = pd.DataFrame(scaled_weights_list, index=returns.index)
    return scaled_weights.fillna(0.0)


def maximum_diversification_weight(
    returns: pd.DataFrame,
    signals: pd.DataFrame,
    lookback: int = 60,
) -> pd.DataFrame:
    """
    Construct weights that maximize the diversification ratio.
    """
    inv_vol = inverse_volatility_weight(
        returns=returns, signals=signals, lookback=lookback
    )
    
    adjusted_weights_list = []
    
    for i in range(len(returns)):
        if i < lookback:
            adjusted_weights_list.append(inv_vol.iloc[i] * 0.0)
            continue
            
        window_returns = returns.iloc[i - lookback : i]
        corr_matrix = window_returns.corr().fillna(0.0)
        
        # Average correlation per asset relative to others
        avg_corr = corr_matrix.mean(axis=1)
        div_factor = 1.0 / (1.0 + avg_corr.clip(lower=0.0))
        
        row_weight = inv_vol.iloc[i] * div_factor
        total_w = row_weight.sum()
        
        if total_w > 0:
            row_weight = row_weight / total_w
            
        adjusted_weights_list.append(row_weight)
        
    adjusted_weights = pd.DataFrame(adjusted_weights_list, index=returns.index)
    return adjusted_weights.fillna(0.0)