import pandas as pd


def cap_weights(
    weights: pd.DataFrame,
    max_weight: float = 0.25,
) -> pd.DataFrame:
    """
    Cap individual asset weights and redistribute
    excess capital across remaining active positions.

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
    """

    if max_weight <= 0:
        raise ValueError("max_weight must be positive.")

    if max_weight > 1:
        raise ValueError("max_weight cannot exceed 1.")

    weights = weights.copy().astype(float)

    if (weights < 0).any().any():
        raise ValueError(
            "Negative weights are not supported."
        )

    result = pd.DataFrame(
        0.0,
        index=weights.index,
        columns=weights.columns,
    )

    for date, row in weights.iterrows():

        current = row.copy()

        if current.sum() <= 0:
            continue

        # Normalize first.
        current = current / current.sum()

        # Iteratively cap overweight positions
        # and redistribute excess capital.
        remaining = current.copy()

        while True:

            overweight = remaining > max_weight

            if not overweight.any():
                break

            excess = (
                remaining[overweight] - max_weight
            ).sum()

            remaining[overweight] = max_weight

            eligible = (
                (remaining > 0)
                & (remaining < max_weight)
            )

            if not eligible.any():
                break

            eligible_total = remaining[eligible].sum()

            if eligible_total <= 0:
                break

            remaining[eligible] += (
                excess
                * remaining[eligible]
                / eligible_total
            )

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
    """

    if max_exposure <= 0:
        raise ValueError(
            "max_exposure must be positive."
        )

    weights = weights.copy().astype(float)

    exposure = weights.sum(axis=1)

    scale = (
        max_exposure
        / exposure.replace(0, pd.NA)
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

    Pipeline:
        1. Individual position cap
        2. Total exposure cap
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