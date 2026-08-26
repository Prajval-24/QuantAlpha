import pandas as pd
from .construction import (
    equal_weight,
    inverse_volatility_weight,
)
from src.alphas import get_strategy
from src.data.loader import load_raw_data
from src.data.preprocessing import (
    add_basic_features,
    validate_market_data,
)

from .construction import equal_weight
from .engine import PortfolioEngine


def build_signal_matrix(
    symbols: list[str],
    strategy_name: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Build aligned signal and return matrices
    for multiple assets.
    """

    signals = {}
    returns = {}

    strategy = get_strategy(
        strategy_name
    )

    for symbol in symbols:

        data = load_raw_data(symbol)

        validate_market_data(data)

        data = add_basic_features(data)

        data = strategy.generate_signal(data)

        data = data.set_index("Date")

        signals[symbol] = data["Signal"]

        returns[symbol] = data["Close"].pct_change()

    signal_matrix = pd.DataFrame(
        signals
    ).fillna(0)

    return_matrix = pd.DataFrame(
        returns
    ).fillna(0)

    # Keep only dates where all assets
    # have aligned observations.
    common_index = (
        signal_matrix.index
        .intersection(return_matrix.index)
        .sort_values()
    )

    signal_matrix = signal_matrix.loc[
        common_index
    ]

    return_matrix = return_matrix.loc[
        common_index
    ]

    return signal_matrix, return_matrix


def run_portfolio_research(
    symbols: list[str],
    strategy_name: str,
    transaction_cost: float = 0.001,
) -> tuple[pd.DataFrame, dict]:
    """
    Run a multi-asset portfolio experiment.
    """

    signals, returns = build_signal_matrix(
        symbols=symbols,
        strategy_name=strategy_name,
    )

    weights = equal_weight(
        signals
    )

    engine = PortfolioEngine(
        transaction_cost=transaction_cost
    )

    result, metrics = engine.run(
        returns=returns,
        weights=weights,
    )

    return result, metrics

def run_risk_aware_portfolio_research(
    symbols: list[str],
    strategy_name: str,
    lookback: int = 20,
    transaction_cost: float = 0.001,
) -> tuple[pd.DataFrame, dict]:
    """
    Run a portfolio experiment using inverse-volatility
    position sizing.
    """

    signals, returns = build_signal_matrix(
        symbols=symbols,
        strategy_name=strategy_name,
    )

    weights = inverse_volatility_weight(
        returns=returns,
        signals=signals,
        lookback=lookback,
    )

    engine = PortfolioEngine(
        transaction_cost=transaction_cost
    )

    result, metrics = engine.run(
        returns=returns,
        weights=weights,
    )

    return result, metrics

def compare_portfolio_methods(
    symbols: list[str],
    strategies: list[str] | None = None,
    transaction_cost: float = 0.001,
    lookback: int = 20,
) -> pd.DataFrame:
    """
    Compare portfolio construction methods
    across multiple alpha strategies.
    """

    if strategies is None:
        strategies = [
            "momentum",
            "mean_reversion",
        ]

    results = []

    for strategy_name in strategies:

        # ----------------------------------------------
        # Equal-weight portfolio
        # ----------------------------------------------

        _, equal_metrics = run_portfolio_research(
            symbols=symbols,
            strategy_name=strategy_name,
            transaction_cost=transaction_cost,
        )

        results.append(
            {
                "strategy": strategy_name,
                "portfolio_method": "equal_weight",
                "total_return": equal_metrics[
                    "total_return"
                ],
                "annualized_return": equal_metrics[
                    "annualized_return"
                ],
                "annualized_volatility": equal_metrics[
                    "annualized_volatility"
                ],
                "sharpe_ratio": equal_metrics[
                    "sharpe_ratio"
                ],
                "max_drawdown": equal_metrics[
                    "max_drawdown"
                ],
                "trades": equal_metrics[
                    "trades"
                ],
                "average_turnover": equal_metrics[
                    "average_turnover"
                ],
                "final_equity": equal_metrics[
                    "final_equity"
                ],
            }
        )

        # ----------------------------------------------
        # Inverse-volatility portfolio
        # ----------------------------------------------

        _, inverse_metrics = (
            run_risk_aware_portfolio_research(
                symbols=symbols,
                strategy_name=strategy_name,
                lookback=lookback,
                transaction_cost=transaction_cost,
            )
        )

        results.append(
            {
                "strategy": strategy_name,
                "portfolio_method": "inverse_volatility",
                "total_return": inverse_metrics[
                    "total_return"
                ],
                "annualized_return": inverse_metrics[
                    "annualized_return"
                ],
                "annualized_volatility": inverse_metrics[
                    "annualized_volatility"
                ],
                "sharpe_ratio": inverse_metrics[
                    "sharpe_ratio"
                ],
                "max_drawdown": inverse_metrics[
                    "max_drawdown"
                ],
                "trades": inverse_metrics[
                    "trades"
                ],
                "average_turnover": inverse_metrics[
                    "average_turnover"
                ],
                "final_equity": inverse_metrics[
                    "final_equity"
                ],
            }
        )

    return pd.DataFrame(results)