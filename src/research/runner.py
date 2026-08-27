from pathlib import Path

import pandas as pd

from src.alphas import get_strategy
from src.backtest.engine import BacktestEngine
from src.data.loader import (
    TICKERS,
    download_stock_data,
    load_raw_data,
    save_raw_data,  # FIX: Added missing import
)
from src.data.preprocessing import (
    add_basic_features,
    validate_market_data,
)


RESEARCH_RESULTS_PATH = Path(
    "data/research_results.csv"
)


def run_single_experiment(
    symbol: str,
    strategy_name: str,
    transaction_cost: float = 0.001,
) -> dict:
    """
    Run one stock-strategy experiment.

    Uses locally cached market data when available.
    Downloads fresh data only when no local CSV exists.
    """

    # --------------------------------------------------
    # 1. Load market data
    # --------------------------------------------------

    try:
        data = load_raw_data(symbol)

        print(
            f"Using cached data for {symbol}..."
        )

    except FileNotFoundError:
        print(
            f"No cached data for {symbol}. "
            f"Downloading..."
        )

        data = download_stock_data(symbol)

        # FIX: Actually save the data to cache so subsequent
        # runs don't hit the API repeatedly.
        save_raw_data(data, symbol)

    # --------------------------------------------------
    # 2. Validate market data
    # --------------------------------------------------

    validate_market_data(data)

    # --------------------------------------------------
    # 3. Feature engineering
    # --------------------------------------------------

    data = add_basic_features(data)

    # --------------------------------------------------
    # 4. Create strategy
    # --------------------------------------------------

    strategy = get_strategy(
        strategy_name
    )

    # --------------------------------------------------
    # 5. Generate trading signals
    # --------------------------------------------------

    data = strategy.generate_signal(data)

    # --------------------------------------------------
    # 6. Run backtest
    # --------------------------------------------------

    engine = BacktestEngine(
        transaction_cost=transaction_cost
    )

    result, metrics = engine.run(data)

    # --------------------------------------------------
    # 7. Build standardized research result
    # --------------------------------------------------

    return {
        "symbol": symbol,
        "strategy": strategy_name,

        "start_date": result["Date"].min(),
        "end_date": result["Date"].max(),

        "total_return": metrics[
            "total_return"
        ],

        "annualized_return": metrics[
            "annualized_return"
        ],

        "annualized_volatility": metrics[
            "annualized_volatility"
        ],

        "sharpe_ratio": metrics[
            "sharpe_ratio"
        ],

        "max_drawdown": metrics[
            "max_drawdown"
        ],

        "trades": metrics[
            "trades"
        ],

        "benchmark_total_return": metrics[
            "benchmark_total_return"
        ],

        "benchmark_annualized_return": metrics[
            "benchmark_annualized_return"
        ],

        "benchmark_sharpe_ratio": metrics[
            "benchmark_sharpe_ratio"
        ],

        "benchmark_max_drawdown": metrics[
            "benchmark_max_drawdown"
        ],

        "excess_return": metrics[
            "excess_return"
        ],
    }


def run_research(
    symbols: list[str] | None = None,
    strategies: list[str] | None = None,
    transaction_cost: float = 0.001,
) -> pd.DataFrame:
    """
    Run systematic research across multiple
    stocks and strategies.
    """

    # --------------------------------------------------
    # Default stock universe
    # --------------------------------------------------

    if symbols is None:
        symbols = list(TICKERS.keys())

    # --------------------------------------------------
    # Default strategies
    # --------------------------------------------------

    if strategies is None:
        strategies = [
            "momentum",
            "mean_reversion",
        ]

    experiments = []

    # --------------------------------------------------
    # Run every stock-strategy combination
    # --------------------------------------------------

    for symbol in symbols:

        for strategy_name in strategies:

            print(
                f"Running {strategy_name} "
                f"on {symbol}..."
            )

            experiment = run_single_experiment(
                symbol=symbol,
                strategy_name=strategy_name,
                transaction_cost=transaction_cost,
            )

            experiments.append(
                experiment
            )

    # --------------------------------------------------
    # Convert results to DataFrame
    # --------------------------------------------------

    return pd.DataFrame(
        experiments
    )


def save_research_results(
    results: pd.DataFrame,
    path: str | Path = RESEARCH_RESULTS_PATH,
) -> None:
    """
    Save research results to CSV.
    """

    path = Path(path)

    # Make sure the parent directory exists.
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    results.to_csv(
        path,
        index=False,
    )


def load_research_results(
    path: str | Path = RESEARCH_RESULTS_PATH,
) -> pd.DataFrame:
    """
    Load previously saved research results.
    """

    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(
            f"Research results not found: {path}"
        )

    return pd.read_csv(path)


def summarize_research(
    results: pd.DataFrame,
) -> dict:
    """
    Generate high-level research summaries
    from experiment results.
    """

    if results.empty:
        raise ValueError(
            "Cannot summarize empty research results."
        )

    strategy_summary = (
        results
        .groupby("strategy")
        .agg(
            experiments=("symbol", "count"),
            average_return=("total_return", "mean"),
            average_sharpe=("sharpe_ratio", "mean"),
            average_drawdown=("max_drawdown", "mean"),
            average_excess_return=("excess_return", "mean"),
        )
        .reset_index()
    )

    positive_excess = (
        results["excess_return"] > 0
    ).sum()

    total_experiments = len(results)

    positive_excess_ratio = (
        positive_excess / total_experiments
    )

    best_experiment = results.loc[
        results["sharpe_ratio"].idxmax()
    ]

    return {
        "strategy_summary": strategy_summary,
        "positive_excess_experiments": int(
            positive_excess
        ),
        "positive_excess_ratio": float(
            positive_excess_ratio
        ),
        "best_experiment": best_experiment,
    }
