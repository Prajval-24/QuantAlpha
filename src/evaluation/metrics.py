import numpy as np
import pandas as pd


def calculate_transaction_costs(
    signals: pd.Series,
    cost_per_trade: float = 0.001,
) -> pd.Series:
    """
    Calculate transaction costs from changes in position.

    cost_per_trade = 0.001 means 10 bps per position change.
    """

    signals = signals.astype(float)

    turnover = signals.diff().abs()

    # Opening the first position is also a trade.
    turnover.iloc[0] = abs(signals.iloc[0])

    return turnover * cost_per_trade


def calculate_net_returns(
    gross_returns: pd.Series,
    signals: pd.Series,
    cost_per_trade: float = 0.001,
) -> pd.Series:
    """
    Calculate strategy returns after transaction costs.
    """

    costs = calculate_transaction_costs(
        signals,
        cost_per_trade,
    )

    return gross_returns - costs


def calculate_total_return(
    returns: pd.Series,
) -> float:
    """
    Compound periodic returns.
    """

    returns = returns.fillna(0.0)

    return float(
        (1.0 + returns).prod() - 1.0
    )


def calculate_sharpe(
    returns: pd.Series,
) -> float:
    """
    Annualized Sharpe ratio assuming daily returns.
    """

    returns = returns.fillna(0.0)

    volatility = returns.std()

    if volatility == 0 or np.isnan(volatility):
        return 0.0

    return float(
        returns.mean()
        / volatility
        * np.sqrt(252)
    )


def calculate_max_drawdown(
    returns: pd.Series,
) -> float:
    """
    Calculate maximum peak-to-trough drawdown.
    """

    returns = returns.fillna(0.0)

    equity = (
        1.0 + returns
    ).cumprod()

    running_max = equity.cummax()

    drawdown = (
        equity / running_max
    ) - 1.0

    return float(drawdown.min())


def calculate_win_rate(
    returns: pd.Series,
) -> float:
    """
    Percentage of non-zero return periods that are profitable.
    """

    active_returns = returns[
        returns != 0
    ]

    if len(active_returns) == 0:
        return 0.0

    return float(
        (active_returns > 0).mean()
    )


def calculate_profit_factor(
    returns: pd.Series,
) -> float:
    """
    Gross profits divided by gross losses.
    """

    profits = returns[
        returns > 0
    ].sum()

    losses = abs(
        returns[
            returns < 0
        ].sum()
    )

    if losses == 0:
        return np.inf if profits > 0 else 0.0

    return float(
        profits / losses
    )


def calculate_all_metrics(
    returns: pd.Series,
) -> dict:
    """
    Calculate the complete strategy metric set.
    """

    return {
        "return": calculate_total_return(
            returns
        ),
        "sharpe": calculate_sharpe(
            returns
        ),
        "max_drawdown": calculate_max_drawdown(
            returns
        ),
        "win_rate": calculate_win_rate(
            returns
        ),
        "profit_factor": calculate_profit_factor(
            returns
        ),
    }