import numpy as np
import pandas as pd


TRADING_DAYS_PER_YEAR = 252


def total_return(returns: pd.Series) -> float:
    """Calculate cumulative strategy return."""
    returns = pd.Series(returns).dropna()
    if returns.empty:
        return 0.0
    return float((1 + returns).prod() - 1.0)


def annualized_return(
    returns: pd.Series,
    periods_per_year: int = TRADING_DAYS_PER_YEAR,
) -> float:
    """Calculate annualized return (CAGR)."""
    returns = pd.Series(returns).dropna()
    if returns.empty:
        return 0.0

    cumulative = (1 + returns).prod()
    if cumulative <= 0:
        return -1.0

    periods = len(returns)
    if periods == 0:
        return 0.0

    return float(
        cumulative ** (periods_per_year / periods) - 1.0
    )


def annualized_volatility(
    returns: pd.Series,
    periods_per_year: int = TRADING_DAYS_PER_YEAR,
) -> float:
    """Calculate annualized volatility."""
    returns = pd.Series(returns).dropna()
    if len(returns) < 2:
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
    returns = pd.Series(returns).dropna()
    if len(returns) < 2:
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


def sortino_ratio(
    returns: pd.Series,
    risk_free_rate: float = 0.0,
    periods_per_year: int = TRADING_DAYS_PER_YEAR,
) -> float:
    """Calculate annualized Sortino ratio using downside deviation."""
    returns = pd.Series(returns).dropna()
    if len(returns) < 2:
        return 0.0

    excess_return = returns - (
        risk_free_rate / periods_per_year
    )
    
    downside_returns = excess_return[excess_return < 0]
    if downside_returns.empty:
        return float('inf') if excess_return.mean() > 0 else 0.0

    downside_std = np.sqrt((downside_returns ** 2).mean()) * np.sqrt(periods_per_year)
    if downside_std == 0 or np.isnan(downside_std):
        return 0.0

    return float(
        (excess_return.mean() * periods_per_year)
        / downside_std
    )


def maximum_drawdown(returns: pd.Series) -> float:
    """Calculate maximum portfolio drawdown."""
    returns = pd.Series(returns).dropna()
    if returns.empty:
        return 0.0

    equity_curve = (1 + returns).cumprod()
    running_peak = equity_curve.cummax()
    drawdown = (equity_curve / running_peak) - 1.0

    return float(drawdown.min())


def calmar_ratio(
    returns: pd.Series,
    periods_per_year: int = TRADING_DAYS_PER_YEAR,
) -> float:
    """Calculate Calmar ratio (Annualized Return / Absolute Max Drawdown)."""
    returns = pd.Series(returns).dropna()
    if returns.empty:
        return 0.0

    cagr = annualized_return(returns, periods_per_year=periods_per_year)
    mdd = maximum_drawdown(returns)

    if mdd == 0 or np.isnan(mdd):
        return 0.0

    return float(cagr / abs(mdd))


def win_rate(returns: pd.Series) -> float:
    """Calculate the ratio of positive return periods."""
    returns = pd.Series(returns).dropna()
    if returns.empty:
        return 0.0
    return float((returns > 0).mean())


def profit_factor(returns: pd.Series) -> float:
    """Calculate profit factor (Gross Gains / Gross Losses)."""
    returns = pd.Series(returns).dropna()
    if returns.empty:
        return 0.0

    gains = returns[returns > 0].sum()
    losses = -returns[returns < 0].sum()

    if losses == 0:
        return float('inf') if gains > 0 else 0.0

    return float(gains / losses)


def information_ratio(
    strategy_returns: pd.Series,
    benchmark_returns: pd.Series,
    periods_per_year: int = TRADING_DAYS_PER_YEAR,
) -> float:
    """Calculate Information Ratio relative to a benchmark."""
    strat = pd.Series(strategy_returns).dropna()
    bench = pd.Series(benchmark_returns).dropna()
    
    aligned = pd.concat([strat, bench], axis=1).dropna()
    if len(aligned) < 2:
        return 0.0

    active_returns = aligned.iloc[:, 0] - aligned.iloc[:, 1]
    tracking_error = active_returns.std(ddof=1) * np.sqrt(periods_per_year)

    if tracking_error == 0 or np.isnan(tracking_error):
        return 0.0

    return float(
        (active_returns.mean() * periods_per_year)
        / tracking_error
    )


def calculate_metrics(
    returns: pd.Series,
    trades: int,
    benchmark_returns: pd.Series | None = None,
) -> dict:
    """Calculate the comprehensive performance summary dictionary."""
    returns = pd.Series(returns).dropna()

    metrics = {
        "total_return": total_return(returns),
        "annualized_return": annualized_return(returns),
        "annualized_volatility": annualized_volatility(returns),
        "sharpe_ratio": sharpe_ratio(returns),
        "sortino_ratio": sortino_ratio(returns),
        "max_drawdown": maximum_drawdown(returns),
        "calmar_ratio": calmar_ratio(returns),
        "win_rate": win_rate(returns),
        "profit_factor": profit_factor(returns),
        "trades": int(trades),
    }

    if benchmark_returns is not None:
        bench = pd.Series(benchmark_returns).dropna()
        if not bench.empty:
            metrics["information_ratio"] = information_ratio(returns, bench)
            metrics["benchmark_total_return"] = total_return(bench)
            metrics["benchmark_annualized_return"] = annualized_return(bench)
            metrics["benchmark_sharpe_ratio"] = sharpe_ratio(bench)
            metrics["benchmark_max_drawdown"] = maximum_drawdown(bench)

    return metrics