import pandas as pd

from src.data.loader import load_raw_data
from src.data.preprocessing import add_basic_features

from src.ml.features import (
    build_ml_features,
    temporal_train_test_split,
    FEATURE_COLUMNS,
)

from src.ml.model import MLAlphaModel

from src.alphas import get_strategy

from src.backtest.engine import BacktestEngine


def run_ml_filter_experiment(
    symbol: str = "RELIANCE",
    threshold: float = 0.55,
) -> tuple[pd.DataFrame, dict, dict]:
    """
    Test whether an ML probability filter improves
    the existing mean-reversion strategy.

    Workflow:

        Raw market data
                ↓
        ML feature construction
                ↓
        Temporal train/test split
                ↓
        Train Logistic Regression
                ↓
        Predict test probabilities
                ↓
        Generate mean-reversion signals
                ↓
        Apply ML probability filter
                ↓
        Backtest
                ↓
        Compare against baseline

    Parameters
    ----------
    symbol:
        Stock symbol to test.

    threshold:
        Minimum ML probability required to allow
        a mean-reversion long signal.

        Example:
            0.55 means P(next-day return > 0)
            must be at least 55%.

    Returns
    -------
    result:
        Backtest result of the ML-filtered strategy.

    metrics:
        Metrics of the ML-filtered strategy plus
        comparison statistics.

    baseline_metrics:
        Metrics of the unfiltered mean-reversion
        strategy on the same test period.
    """

    # --------------------------------------------------
    # 1. Validate threshold
    # --------------------------------------------------

    if not 0.5 < threshold < 1.0:
        raise ValueError(
            "threshold must be between 0.5 and 1.0."
        )

    # --------------------------------------------------
    # 2. Load raw market data
    # --------------------------------------------------

    raw_data = load_raw_data(symbol)

    # --------------------------------------------------
    # 3. Build ML features
    # --------------------------------------------------

    ml_data = build_ml_features(
        raw_data.copy()
    )

    # --------------------------------------------------
    # 4. Temporal train/test split
    # --------------------------------------------------

    train, test = temporal_train_test_split(
        ml_data
    )

    # --------------------------------------------------
    # 5. Train ML model
    # --------------------------------------------------

    model = MLAlphaModel()

    model.fit(
        train[FEATURE_COLUMNS],
        train["Target"],
    )

    # --------------------------------------------------
    # 6. Generate ML probabilities
    #    ONLY on the unseen test period
    # --------------------------------------------------

    probabilities = (
        model.predict_probability(
            test[FEATURE_COLUMNS]
        )
    )

    # --------------------------------------------------
    # 7. Build strategy features
    # --------------------------------------------------
    #
    # Mean Reversion requires:
    #
    #     MA_20
    #     MA_Distance
    #
    # These must be calculated using the complete
    # historical price series BEFORE selecting
    # the test period.
    #
    # This preserves the historical context required
    # for the rolling moving average.
    #
    # --------------------------------------------------

    strategy_data = add_basic_features(
        raw_data.copy()
    )

    # --------------------------------------------------
    # 8. Select only the unseen test period
    # --------------------------------------------------

    test_market_data = strategy_data[
        strategy_data["Date"].isin(
            test["Date"]
        )
    ].copy()

    test_market_data = (
        test_market_data
        .sort_values("Date")
        .reset_index(drop=True)
    )

    # --------------------------------------------------
    # 9. Get mean-reversion strategy
    # --------------------------------------------------

    strategy = get_strategy(
        "mean_reversion"
    )

    # --------------------------------------------------
    # 10. Generate baseline signals
    # --------------------------------------------------

    baseline_data = strategy.generate_signal(
        test_market_data
    )

    # --------------------------------------------------
    # 11. Prepare ML probability data
    # --------------------------------------------------

    probability_data = pd.DataFrame(
        {
            "Date": test["Date"].values,
            "Probability": probabilities.values,
        }
    )

    probability_data = (
        probability_data
        .sort_values("Date")
        .reset_index(drop=True)
    )

    # --------------------------------------------------
    # 12. Align baseline signals and ML probabilities
    # --------------------------------------------------

    result_data = baseline_data.merge(
        probability_data,
        on="Date",
        how="inner",
    )

    # --------------------------------------------------
    # 13. Preserve original strategy signal
    # --------------------------------------------------

    result_data[
        "Baseline_Signal"
    ] = result_data["Signal"]

    # --------------------------------------------------
    # 14. Apply ML probability filter
    # --------------------------------------------------
    #
    # Mean-reversion strategy:
    #
    #     Signal = 1
    #         → long
    #
    # ML:
    #
    #     Probability >= threshold
    #         → allow trade
    #
    #     Probability < threshold
    #         → reject trade
    #
    # --------------------------------------------------

    result_data["Signal"] = (
        (
            result_data["Baseline_Signal"] == 1
        )
        &
        (
            result_data["Probability"]
            >= threshold
        )
    ).astype(int)

    # --------------------------------------------------
    # 15. Run ML-filtered backtest
    # --------------------------------------------------

    backtest = BacktestEngine()

    filtered_result, filtered_metrics = (
        backtest.run(
            result_data
        )
    )

    # --------------------------------------------------
    # 16. Run baseline backtest
    # --------------------------------------------------

    baseline_result, baseline_metrics = (
        backtest.run(
            baseline_data
        )
    )

    # --------------------------------------------------
    # 17. Copy metrics so we don't mutate the
    #     BacktestEngine's original dictionary
    # --------------------------------------------------

    filtered_metrics = dict(
        filtered_metrics
    )

    baseline_metrics = dict(
        baseline_metrics
    )

    # --------------------------------------------------
    # 18. Add experiment metadata
    # --------------------------------------------------

    filtered_metrics[
        "symbol"
    ] = symbol

    filtered_metrics[
        "strategy"
    ] = "mean_reversion_ml_filter"

    filtered_metrics[
        "ml_threshold"
    ] = threshold

    filtered_metrics[
        "ml_train_rows"
    ] = len(train)

    filtered_metrics[
        "ml_test_rows"
    ] = len(test)

    # --------------------------------------------------
    # 19. Add baseline metrics
    # --------------------------------------------------

    filtered_metrics[
        "baseline_total_return"
    ] = baseline_metrics[
        "total_return"
    ]

    filtered_metrics[
        "baseline_sharpe_ratio"
    ] = baseline_metrics[
        "sharpe_ratio"
    ]

    filtered_metrics[
        "baseline_max_drawdown"
    ] = baseline_metrics[
        "max_drawdown"
    ]

    filtered_metrics[
        "baseline_trades"
    ] = baseline_metrics[
        "trades"
    ]

    # --------------------------------------------------
    # 20. Calculate incremental performance
    # --------------------------------------------------

    filtered_metrics[
        "return_difference"
    ] = (
        filtered_metrics["total_return"]
        - baseline_metrics["total_return"]
    )

    filtered_metrics[
        "sharpe_difference"
    ] = (
        filtered_metrics["sharpe_ratio"]
        - baseline_metrics["sharpe_ratio"]
    )

    filtered_metrics[
        "drawdown_difference"
    ] = (
        filtered_metrics["max_drawdown"]
        - baseline_metrics["max_drawdown"]
    )

    filtered_metrics[
        "trade_difference"
    ] = (
        filtered_metrics["trades"]
        - baseline_metrics["trades"]
    )

    # --------------------------------------------------
    # 21. Return results
    # --------------------------------------------------

    return (
        filtered_result,
        filtered_metrics,
        baseline_metrics,
    )


def print_ml_filter_report(
    metrics: dict,
) -> None:
    """
    Print a concise ML filter research report.
    """

    print()
    print(
        "ML FILTER RESEARCH"
    )
    print(
        "=" * 50
    )

    print(
        f"Symbol:             "
        f"{metrics.get('symbol', 'N/A')}"
    )

    print(
        f"ML threshold:       "
        f"{metrics['ml_threshold']:.2f}"
    )

    print(
        f"Training rows:      "
        f"{metrics['ml_train_rows']}"
    )

    print(
        f"Test rows:          "
        f"{metrics['ml_test_rows']}"
    )

    print()

    print(
        "BASELINE — MEAN REVERSION"
    )

    print(
        f"Return:             "
        f"{metrics['baseline_total_return']:.6f}"
    )

    print(
        f"Sharpe:             "
        f"{metrics['baseline_sharpe_ratio']:.6f}"
    )

    print(
        f"Max Drawdown:       "
        f"{metrics['baseline_max_drawdown']:.6f}"
    )

    print(
        f"Trades:             "
        f"{metrics['baseline_trades']}"
    )

    print()

    print(
        "ML-FILTERED MEAN REVERSION"
    )

    print(
        f"Return:             "
        f"{metrics['total_return']:.6f}"
    )

    print(
        f"Sharpe:             "
        f"{metrics['sharpe_ratio']:.6f}"
    )

    print(
        f"Max Drawdown:       "
        f"{metrics['max_drawdown']:.6f}"
    )

    print(
        f"Trades:             "
        f"{metrics['trades']}"
    )

    print()

    print(
        "INCREMENTAL VALUE"
    )

    print(
        f"Return Difference:  "
        f"{metrics['return_difference']:.6f}"
    )

    print(
        f"Sharpe Difference:  "
        f"{metrics['sharpe_difference']:.6f}"
    )

    print(
        f"Drawdown Difference: "
        f"{metrics['drawdown_difference']:.6f}"
    )

    print(
        f"Trade Difference:   "
        f"{metrics['trade_difference']}"
    )


def compare_ml_filter(
    symbol: str = "RELIANCE",
    thresholds: list[float] | None = None,
) -> pd.DataFrame:
    """
    Run ML-filter experiments for multiple thresholds.

    IMPORTANT:
    This function is intended for research/diagnostics.
    Threshold selection should NOT be performed using
    the final untouched test set.

    Default thresholds are provided for exploratory
    analysis only.
    """

    if thresholds is None:
        thresholds = [
            0.55,
            0.60,
            0.65,
        ]

    rows = []

    for threshold in thresholds:

        print(
            f"Running ML filter "
            f"threshold={threshold:.2f}..."
        )

        _, metrics, _ = (
            run_ml_filter_experiment(
                symbol=symbol,
                threshold=threshold,
            )
        )

        rows.append(
            {
                "symbol": symbol,
                "threshold": threshold,
                "total_return": (
                    metrics["total_return"]
                ),
                "sharpe_ratio": (
                    metrics["sharpe_ratio"]
                ),
                "max_drawdown": (
                    metrics["max_drawdown"]
                ),
                "trades": (
                    metrics["trades"]
                ),
                "baseline_return": (
                    metrics[
                        "baseline_total_return"
                    ]
                ),
                "baseline_sharpe": (
                    metrics[
                        "baseline_sharpe_ratio"
                    ]
                ),
                "return_difference": (
                    metrics[
                        "return_difference"
                    ]
                ),
                "sharpe_difference": (
                    metrics[
                        "sharpe_difference"
                    ]
                ),
            }
        )

    return pd.DataFrame(rows)

def run_multi_asset_ml_research(
    symbols: list[str],
    threshold: float = 0.55,
) -> pd.DataFrame:
    """
    Evaluate the ML filter consistently across
    multiple assets using the same threshold.
    """

    results = []

    for symbol in symbols:

        print(
            f"Running ML filter on {symbol}..."
        )

        _, metrics, _ = (
            run_ml_filter_experiment(
                symbol=symbol,
                threshold=threshold,
            )
        )

        results.append(
            {
                "symbol": symbol,
                "threshold": threshold,
                "baseline_return": (
                    metrics[
                        "baseline_total_return"
                    ]
                ),
                "ml_return": (
                    metrics[
                        "total_return"
                    ]
                ),
                "baseline_sharpe": (
                    metrics[
                        "baseline_sharpe_ratio"
                    ]
                ),
                "ml_sharpe": (
                    metrics[
                        "sharpe_ratio"
                    ]
                ),
                "baseline_drawdown": (
                    metrics[
                        "baseline_max_drawdown"
                    ]
                ),
                "ml_drawdown": (
                    metrics[
                        "max_drawdown"
                    ]
                ),
                "baseline_trades": (
                    metrics[
                        "baseline_trades"
                    ]
                ),
                "ml_trades": (
                    metrics[
                        "trades"
                    ]
                ),
                "return_difference": (
                    metrics[
                        "return_difference"
                    ]
                ),
                "sharpe_difference": (
                    metrics[
                        "sharpe_difference"
                    ]
                ),
            }
        )

    return pd.DataFrame(results)