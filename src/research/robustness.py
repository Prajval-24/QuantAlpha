import pandas as pd
from src.portfolio.research import (
    PortfolioExperimentConfig,
    run_portfolio_experiment,
)


def run_parameter_sensitivity_sweep(
    symbols: list[str],
    strategy_name: str,
    lookback_values: list[int],
    transaction_cost: float = 0.001,
) -> pd.DataFrame:
    """
    Evaluate strategy and portfolio sensitivity across multiple lookback windows
    (e.g., for inverse volatility or strategy indicators).
    """
    results = []

    for lb in lookback_values:
        config = PortfolioExperimentConfig(
            symbols=symbols,
            strategy_name=strategy_name,
            portfolio_method="inverse_volatility",
            lookback=lb,
            transaction_cost=transaction_cost,
        )
        _, metrics = run_portfolio_experiment(config)

        results.append({
            "strategy": strategy_name,
            "lookback": lb,
            "transaction_cost": transaction_cost,
            "total_return": metrics["total_return"],
            "annualized_return": metrics["annualized_return"],
            "annualized_volatility": metrics["annualized_volatility"],
            "sharpe_ratio": metrics["sharpe_ratio"],
            "max_drawdown": metrics["max_drawdown"],
            "profit_factor": metrics["profit_factor"],
            "trades": metrics["trades"],
        })

    return pd.DataFrame(results)


def run_transaction_cost_sensitivity_sweep(
    symbols: list[str],
    strategy_name: str,
    cost_values: list[float],
    portfolio_method: str = "equal_weight",
    lookback: int = 20,
) -> pd.DataFrame:
    """
    Evaluate performance degradation across various transaction cost assumptions
    (e.g., 0.0%, 0.05%, 0.10%, 0.20%, 0.50%).
    """
    results = []

    for cost in cost_values:
        config = PortfolioExperimentConfig(
            symbols=symbols,
            strategy_name=strategy_name,
            portfolio_method=portfolio_method,
            lookback=lookback,
            transaction_cost=cost,
        )
        _, metrics = run_portfolio_experiment(config)

        results.append({
            "strategy": strategy_name,
            "portfolio_method": portfolio_method,
            "transaction_cost": cost,
            "total_return": metrics["total_return"],
            "annualized_return": metrics["annualized_return"],
            "sharpe_ratio": metrics["sharpe_ratio"],
            "max_drawdown": metrics["max_drawdown"],
            "average_turnover": metrics["average_turnover"],
            "final_equity": metrics["final_equity"],
        })

    return pd.DataFrame(results)