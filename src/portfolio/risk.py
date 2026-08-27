import numpy as np
import pandas as pd


WEIGHT_TOLERANCE = 1e-8


def _validate_weights(
    weights: pd.DataFrame,
) -> None:
    """Validate portfolio weights for risk-control operations."""

    if not isinstance(weights, pd.DataFrame):
        raise TypeError(
            "Weights must be a pandas DataFrame."
        )

    if weights.empty:
        raise ValueError(
            "Weights DataFrame cannot be empty."
        )

    if weights.columns.duplicated().any():
        raise ValueError(
            "Weights contain duplicate asset columns."
        )

    values = weights.to_numpy(dtype=float)

    if not np.isfinite(values).all():
        raise ValueError(
            "Weights contain NaN or infinite values."
        )

    if (
        weights < -WEIGHT_TOLERANCE
    ).any().any():
        raise ValueError(
            "Negative weights are not supported. "
            "The current portfolio architecture is long-only."
        )


def cap_weights(
    weights: pd.DataFrame,
    max_weight: float = 0.25,
) -> pd.DataFrame:
    """
    Cap individual asset weights and redistribute excess
    capital across remaining active positions.

    Parameters
    ----------
    weights:
        Target portfolio weights.

    max_weight:
        Maximum allowed weight per asset.

    Returns
    -------
    pd.DataFrame
        Risk-controlled portfolio weights.

    Notes
    -----
    Existing total exposure is preserved whenever possible.

    The function does NOT normalize the input portfolio to 100%
    exposure. Therefore, existing cash is preserved.

    If the number of active positions is insufficient to
    redistribute all capital under max_weight, the remaining
    capital stays in cash.
    """

    if not np.isfinite(max_weight):
        raise ValueError(
            "max_weight must be finite."
        )

    if max_weight <= 0:
        raise ValueError(
            "max_weight must be positive."
        )

    if max_weight > 1:
        raise ValueError(
            "max_weight cannot exceed 1."
        )

    _validate_weights(weights)

    weights = weights.astype(float).copy()

    result = pd.DataFrame(
        0.0,
        index=weights.index,
        columns=weights.columns,
    )

    for date, row in weights.iterrows():

        current = row.copy()

        total_exposure = current.sum()

        if total_exposure <= WEIGHT_TOLERANCE:
            continue

        # --------------------------------------------------
        # Iteratively cap overweight positions and
        # redistribute their excess across eligible
        # active positions.
        # --------------------------------------------------

        remaining = current.copy()

        while True:

            overweight = (
                remaining > max_weight + WEIGHT_TOLERANCE
            )

            if not overweight.any():
                break

            excess = (
                remaining[overweight] - max_weight
            ).sum()

            remaining.loc[overweight] = max_weight

            eligible = (
                (remaining > WEIGHT_TOLERANCE)
                & (
                    remaining
                    < max_weight - WEIGHT_TOLERANCE
                )
            )

            if not eligible.any():
                # No active position can absorb more capital.
                # The unreallocated amount remains as cash.
                break

            eligible_total = (
                remaining[eligible].sum()
            )

            if eligible_total <= WEIGHT_TOLERANCE:
                break

            allocation = (
                excess
                * remaining[eligible]
                / eligible_total
            )

            remaining.loc[eligible] += allocation

        # Remove tiny numerical noise.
        remaining = remaining.clip(lower=0.0)

        result.loc[date] = remaining

    return result


def exposure_cap(
    weights: pd.DataFrame,
    max_exposure: float = 1.0,
) -> pd.DataFrame:
    """
    Limit total portfolio exposure.

    Parameters
    ----------
    weights:
        Portfolio weights.

    max_exposure:
        Maximum total portfolio exposure.

    Returns
    -------
    pd.DataFrame
        Exposure-controlled weights.

    Notes
    -----
    Exposure below max_exposure is preserved.
    """

    if not np.isfinite(max_exposure):
        raise ValueError(
            "max_exposure must be finite."
        )

    if max_exposure <= 0:
        raise ValueError(
            "max_exposure must be positive."
        )

    if max_exposure > 1:
        raise ValueError(
            "max_exposure cannot exceed 1."
        )

    _validate_weights(weights)

    weights = weights.astype(float).copy()

    exposure = weights.sum(axis=1)

    scale = (
        max_exposure
        / exposure.replace(0.0, np.nan)
    ).clip(upper=1.0)

    result = weights.mul(
        scale.fillna(0.0),
        axis=0,
    )

    return result


def apply_risk_controls(
    weights: pd.DataFrame,
    max_weight: float = 0.25,
    max_exposure: float = 1.0,
) -> pd.DataFrame:
    """
    Apply portfolio risk controls.

    Pipeline
    --------
    1. Individual position cap.
    2. Total exposure cap.

    The portfolio remains long-only.
    Existing cash is preserved unless exposure exceeds
    max_exposure.
    """

    controlled = cap_weights(
        weights,
        max_weight=max_weight,
    )

    controlled = exposure_cap(
        controlled,
        max_exposure=max_exposure,
    )

    return controlled