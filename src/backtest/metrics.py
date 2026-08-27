import numpy as np
import pandas as pd


TRADING_DAYS_PER_YEAR = 252


def total_return(returns: pd.Series) -> float:
    """Calculate cumulative strategy return."""

    returns = pd.Series(returns).dropna()

    if returns.empty:
        return 0.0

    equity = (1.0 + returns).cumprod()

    return float(equity.iloc[-1] - 1.0)


def annualized_return(
    returns: pd.Series,
    periods_per_year: int = TRADING_DAYS_PER_YEAR,
) -> float:
    """Calculate annualized compounded return."""

    returns = pd.Series(returns).dropna()

    if returns.empty:
        return 0.0

    if periods_per_year <= 0:
        raise ValueError(
            "periods_per_year must be positive."
        )

    periods = len(returns)

    cumulative = float(
        (1.0 + returns).prod()
    )

    # A portfolio that reaches zero or below
    # cannot have a meaningful positive-base
    # annualized compound return.
    if cumulative <= 0:
        return -1.0

    return float(
        cumulative ** (
            periods_per_year / periods
        ) - 1.0
    )


def annualized_volatility(
    returns: pd.Series,
    periods_per_year: int = TRADING_DAYS_PER_YEAR,
) -> float:
    """Calculate annualized return volatility."""

    returns = pd.Series(returns).dropna()

    if returns.empty:
        return 0.0

    if periods_per_year <= 0:
        raise ValueError(
            "periods_per_year must be positive."
        )

    volatility = returns.std(
        ddof=1
    )

    if pd.isna(volatility):
        return 0.0

    return float(
        volatility
        * np.sqrt(periods_per_year)
    )


def sharpe_ratio(
    returns: pd.Series,
    risk_free_rate: float = 0.0,
    periods_per_year: int = TRADING_DAYS_PER_YEAR,
) -> float:
    """Calculate annualized Sharpe ratio."""

    returns = pd.Series(returns).dropna()

    if returns.empty:
        return 0.0

    if periods_per_year <= 0:
        raise ValueError(
            "periods_per_year must be positive."
        )

    # risk_free_rate is an annualized rate.
    daily_risk_free_rate = (
        risk_free_rate
        / periods_per_year
    )

    excess_returns = (
        returns
        - daily_risk_free_rate
    )

    volatility = excess_returns.std(
        ddof=1
    )

    if (
        pd.isna(volatility)
        or volatility == 0
    ):
        return 0.0

    return float(
        excess_returns.mean()
        / volatility
        * np.sqrt(periods_per_year)
    )


def maximum_drawdown(
    returns: pd.Series,
) -> float:
    """Calculate maximum portfolio drawdown."""

    returns = pd.Series(returns).dropna()

    if returns.empty:
        return 0.0

    equity_curve = (
        1.0 + returns
    ).cumprod()

    running_peak = (
        equity_curve.cummax()
    )

    drawdown = (
        equity_curve
        / running_peak
    ) - 1.0

    return float(
        drawdown.min()
    )


def calculate_metrics(
    returns: pd.Series,
    trades: int,
) -> dict:
    """Calculate the complete performance summary."""

    returns = pd.Series(returns).dropna()

    return {
        "total_return": total_return(
            returns
        ),
        "annualized_return": annualized_return(
            returns
        ),
        "annualized_volatility":
            annualized_volatility(
                returns
            ),
        "sharpe_ratio": sharpe_ratio(
            returns
        ),
        "max_drawdown": maximum_drawdown(
            returns
        ),
        "trades": int(trades),
    }