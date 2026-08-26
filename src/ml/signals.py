import pandas as pd


def probability_signal(
    probabilities: pd.Series,
    upper_threshold: float = 0.55,
    lower_threshold: float = 0.45,
) -> pd.Series:
    """
    Convert predicted probabilities into ML trading signals.

    Returns:
        1  -> bullish
        0  -> neutral
       -1  -> bearish
    """

    if not 0 < lower_threshold < upper_threshold < 1:
        raise ValueError(
            "Thresholds must satisfy "
            "0 < lower < upper < 1."
        )

    signals = pd.Series(
        0,
        index=probabilities.index,
        dtype=int,
        name="ML_Signal",
    )

    signals.loc[
        probabilities >= upper_threshold
    ] = 1

    signals.loc[
        probabilities <= lower_threshold
    ] = -1

    return signals