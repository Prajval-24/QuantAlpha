from dataclasses import dataclass, field
import pandas as pd

from src.alphas import get_strategy
from src.data.loader import load_raw_data
from src.data.preprocessing import (
    add_basic_features,
    validate_market_data,
)

from .construction import (
    equal_weight,
    inverse_volatility_weight,
)
from .risk import apply_risk_controls
from .engine import PortfolioEngine


@dataclass
class PortfolioExperimentConfig:
    symbols: list[str] = field(default_factory=lambda: ["RELIANCE", "TCS", "INFY"])
    strategy_name: str = "mean_reversion"
    portfolio_method: str = "equal_weight"  # "equal_weight" or "inverse_volatility"
    lookback: int = 20
    max_weight: float = 0.25
    max_exposure: float = 1.0
    transaction_cost: float = 0.001


def build_signal_matrix(
    symbols: list[str],
    strategy_name: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Build aligned signal and return matrices for multiple assets.

    The portfolio research layer uses only dates for which every
    requested asset has a genuine market observation.

    Signals:
        1 = long
        0 = no position

    Returns:
        Daily close-to-close returns for each asset.
    """

    if not symbols:
        raise ValueError(
            "symbols cannot be empty."
        )

    if len(set(symbols)) != len(symbols):
        raise ValueError(
            "symbols must not contain duplicates."
        )

    if not strategy_name:
        raise ValueError(
            "strategy_name cannot be empty."
        )

    signals: dict[str, pd.Series] = {}
    returns: dict[str, pd.Series] = {}

    strategy = get_strategy(
        strategy_name
    )

    for symbol in symbols:

        data = load_raw_data(
            symbol
        )

        validate_market_data(
            data
        )

        data = add_basic_features(
            data
        )

        data = strategy.generate_signal(
            data
        )

        data = (
            data
            .sort_values("Date")
            .set_index("Date")
        )

        signals[symbol] = (
            data["Signal"]
            .astype(float)
        )

        returns[symbol] = (
            data["Close"]
            .pct_change()
            .astype(float)
        )

    signal_matrix = pd.DataFrame(
        signals
    )

    return_matrix = pd.DataFrame(
        returns
    )

    # --------------------------------------------------
    # Align on dates where EVERY asset has real data.
    # --------------------------------------------------

    valid_signal_dates = (
        signal_matrix
        .dropna()
        .index
    )

    valid_return_dates = (
        return_matrix
        .dropna()
        .index
    )

    common_index = (
        valid_signal_dates
        .intersection(valid_return_dates)
        .sort_values()
    )

    if common_index.empty:
        raise ValueError(
            "No common dates exist across all requested assets."
        )

    signal_matrix = (
        signal_matrix
        .loc[common_index]
        .copy()
    )

    return_matrix = (
        return_matrix
        .loc[common_index]
        .copy()
    )

    # --------------------------------------------------
    # Final numerical safety checks.
    # --------------------------------------------------

    if signal_matrix.isna().any().any():
        raise ValueError(
            "Signal matrix contains missing values after alignment."
        )

    if return_matrix.isna().any().any():
        raise ValueError(
            "Return matrix contains missing values after alignment."
        )

    return signal_matrix, return_matrix


def run_portfolio_experiment(
    config: PortfolioExperimentConfig,
) -> tuple[pd.DataFrame, dict]:
    """
    Run a fully configured, reproducible portfolio experiment.
    Binds strategy, construction, risk controls, and execution into an audited result.
    """
    signals, returns = build_signal_matrix(
        symbols=config.symbols,
        strategy_name=config.strategy_name,
    )

    if config.portfolio_method == "equal_weight":
        weights = equal_weight(signals)
    elif config.portfolio_method == "inverse_volatility":
        weights = inverse_volatility_weight(
            returns=returns,
            signals=signals,
            lookback=config.lookback,
        )
    else:
        raise ValueError(
            f"Unknown portfolio construction method: {config.portfolio_method}"
        )

    # Enforce risk controls (Single Source of Truth)
    weights = apply_risk_controls(
        weights=weights,
        max_weight=config.max_weight,
        max_exposure=config.max_exposure,
    )

    engine = PortfolioEngine(
        transaction_cost=config.transaction_cost
    )

    result, metrics = engine.run(
        returns=returns,
        weights=weights,
    )

    metrics["experiment_config"] = {
        "symbols": config.symbols,
        "strategy_name": config.strategy_name,
        "portfolio_method": config.portfolio_method,
        "lookback": config.lookback,
        "max_weight": config.max_weight,
        "max_exposure": config.max_exposure,
        "transaction_cost": config.transaction_cost,
    }

    return result, metrics


def run_portfolio_research(
    symbols: list[str],
    strategy_name: str,
    transaction_cost: float = 0.001,
) -> tuple[pd.DataFrame, dict]:
    """
    Run an equal-weight multi-asset portfolio experiment.
    """
    config = PortfolioExperimentConfig(
        symbols=symbols,
        strategy_name=strategy_name,
        portfolio_method="equal_weight",
        transaction_cost=transaction_cost,
    )
    return run_portfolio_experiment(config)


def run_risk_aware_portfolio_research(
    symbols: list[str],
    strategy_name: str,
    lookback: int = 20,
    transaction_cost: float = 0.001,
) -> tuple[pd.DataFrame, dict]:
    """
    Run a portfolio experiment using inverse-volatility position sizing.
    """
    config = PortfolioExperimentConfig(
        symbols=symbols,
        strategy_name=strategy_name,
        portfolio_method="inverse_volatility",
        lookback=lookback,
        transaction_cost=transaction_cost,
    )
    return run_portfolio_experiment(config)


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

    if not symbols:
        raise ValueError(
            "symbols cannot be empty."
        )

    if strategies is None:
        strategies = [
            "momentum",
            "mean_reversion",
        ]

    if not strategies:
        raise ValueError(
            "strategies cannot be empty."
        )

    results = []

    for strategy_name in strategies:

        # --------------------------------------------------
        # Equal-weight portfolio
        # --------------------------------------------------

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

        # --------------------------------------------------
        # Inverse-volatility portfolio
        # --------------------------------------------------

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

    return pd.DataFrame(
        results
    )