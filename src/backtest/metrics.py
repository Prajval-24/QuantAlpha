import numpy as np
import pandas as pd


TRADING_DAYS_PER_YEAR = 252


def total_return(returns: pd.Series) -> float:
    """Calculate cumulative strategy return."""

    if returns.empty:
        return 0.0

    return float((1 + returns).prod() - 1)


def annualized_return(
    returns: pd.Series,
    periods_per_year: int = TRADING_DAYS_PER_YEAR,
) -> float:
    """Calculate annualized return."""

    if returns.empty:
        return 0.0

    cumulative = (1 + returns).prod()
    periods = len(returns)

    if periods == 0:
        return 0.0

    return float(
        cumulative ** (periods_per_year / periods) - 1
    )


def annualized_volatility(
    returns: pd.Series,
    periods_per_year: int = TRADING_DAYS_PER_YEAR,
) -> float:
    """Calculate annualized volatility."""

    if returns.empty:
        return 0.0

    return float(
        returns.std(ddof=1) * np.sqrt(periods_per_year)
    )


def sharpe_ratio(
    returns: pd.Series,
    risk_free_rate: float = 0.0,
    periods_per_year: int = TRADING_DAYS_PER_YEAR,
) -> float:
    """Calculate annualized Sharpe ratio."""

    if returns.empty:
        return 0.0

    excess_return = returns - (
        risk_free_rate / periods_per_year
    )

    volatility = excess_return.std(ddof=1)

    if volatility == 0 or np.isnan(volatility):
        return 0.0

    return float(
        excess_return.mean()
        / volatility
        * np.sqrt(periods_per_year)
    )


def maximum_drawdown(returns: pd.Series) -> float:
    """Calculate maximum portfolio drawdown."""

    if returns.empty:
        return 0.0

    equity_curve = (1 + returns).cumprod()

    running_peak = equity_curve.cummax()

    drawdown = (
        equity_curve / running_peak
    ) - 1

    return float(drawdown.min())


def calculate_metrics(
    returns: pd.Series,
    trades: int,
) -> dict:
    """Calculate the complete performance summary."""

    return {
        "total_return": total_return(returns),
        "annualized_return": annualized_return(returns),
        "annualized_volatility": annualized_volatility(returns),
        "sharpe_ratio": sharpe_ratio(returns),
        "max_drawdown": maximum_drawdown(returns),
        "trades": int(trades),
    }