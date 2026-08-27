import numpy as np
import pandas as pd

from src.ml.model import MLAlphaModel
from src.ml.features import FEATURE_COLUMNS
from src.validation.walk_forward import (
    generate_walk_forward_splits,
    validate_walk_forward_splits,
)
from src.portfolio.risk import apply_risk_controls


def _safe_sharpe(returns: pd.Series) -> float:
    returns = returns.dropna()

    if len(returns) < 2:
        return 0.0

    volatility = returns.std()

    if volatility == 0 or pd.isna(volatility):
        return 0.0

    return float(
        returns.mean()
        / volatility
        * np.sqrt(252)
    )


def _max_drawdown(returns: pd.Series) -> float:
    returns = returns.dropna()

    if returns.empty:
        return 0.0

    equity = (1 + returns).cumprod()
    peak = equity.cummax()
    drawdown = equity / peak - 1

    return float(drawdown.min())


def _calculate_trade_metrics(
    returns: pd.Series,
    turnover: pd.Series,
) -> dict:

    returns = returns.dropna()

    if returns.empty:
        return {
            "win_rate": 0.0,
            "profit_factor": 0.0,
        }

    positive = returns[returns > 0]
    negative = returns[returns < 0]

    win_rate = (
        len(positive) / len(returns)
        if len(returns) > 0
        else 0.0
    )

    gross_profit = positive.sum()
    gross_loss = abs(negative.sum())

    if gross_loss == 0:
        profit_factor = (
            float("inf")
            if gross_profit > 0
            else 0.0
        )
    else:
        profit_factor = (
            float(gross_profit / gross_loss)
        )

    return {
        "win_rate": float(win_rate),
        "profit_factor": profit_factor,
    }


def _portfolio_metrics(
    returns: pd.Series,
    turnover: pd.Series,
) -> dict:

    returns = returns.fillna(0.0)

    total_return = float(
        (1 + returns).prod() - 1
    )

    sharpe = _safe_sharpe(returns)
    drawdown = _max_drawdown(returns)

    trade_metrics = _calculate_trade_metrics(
        returns,
        turnover,
    )

    return {
        "return": total_return,
        "sharpe": sharpe,
        "drawdown": drawdown,
        "win_rate": trade_metrics["win_rate"],
        "profit_factor": trade_metrics["profit_factor"],
        "trades": int((turnover > 0).sum()),
    }


def _build_equal_weights(
    signals: pd.DataFrame,
) -> pd.DataFrame:

    active = signals.clip(lower=0)

    active_count = active.sum(axis=1)

    weights = active.div(
        active_count.replace(0, np.nan),
        axis=0,
    )

    return weights.fillna(0.0)


def _build_ml_weights(
    train: pd.DataFrame,
    test: pd.DataFrame,
    threshold: float,
) -> pd.DataFrame:

    model = MLAlphaModel()

    model.fit(
        train[FEATURE_COLUMNS],
        train["Target"],
    )

    probabilities = model.predict_proba(
        test[FEATURE_COLUMNS]
    )

    ml_signal = (
        probabilities >= threshold
    ).astype(float)

    return pd.DataFrame(
        ml_signal,
        index=test.index,
        columns=["Signal"],
    )


def _apply_risk_controls_to_weights(
    weights: pd.DataFrame,
    max_weight: float,
    max_exposure: float,
) -> pd.DataFrame:

    return apply_risk_controls(
        weights,
        max_weight=max_weight,
        max_exposure=max_exposure,
    )


def run_portfolio_walk_forward(
    data: dict[str, pd.DataFrame],
    threshold: float = 0.55,
    transaction_cost: float = 0.001,
    max_weight: float = 0.25,
    max_exposure: float = 1.0,
) -> pd.DataFrame:

    symbols = list(data.keys())

    prepared = {}

    for symbol in symbols:

        df = data[symbol].copy()

        df = df.sort_index()

        prepared[symbol] = df

    common_index = None

    for symbol in symbols:

        index = prepared[symbol].index

        if common_index is None:
            common_index = index
        else:
            common_index = common_index.intersection(
                index
            )

    common_index = common_index.sort_values()

    for symbol in symbols:

        prepared[symbol] = prepared[symbol].loc[
            common_index
        ]

    combined = []

    for symbol in symbols:

        df = prepared[symbol].copy()

        df["Symbol"] = symbol

        combined.append(df)

    all_data = pd.concat(combined)

    splits = generate_walk_forward_splits(
        all_data
    )

    validate_walk_forward_splits(
        all_data,
        splits,
    )

    results = []

    for window_number, (train_idx, test_idx) in enumerate(
        splits,
        start=1,
    ):

        print(
            f"\nRunning portfolio walk-forward window "
            f"{window_number}..."
        )

        train = all_data.iloc[train_idx]
        test = all_data.iloc[test_idx]

        baseline_weights = pd.DataFrame(
            0.0,
            index=common_index,
            columns=symbols,
        )

        ml_weights = pd.DataFrame(
            0.0,
            index=common_index,
            columns=symbols,
        )

        for symbol in symbols:

            train_symbol = train[
                train["Symbol"] == symbol
            ]

            test_symbol = test[
                test["Symbol"] == symbol
            ]

            if train_symbol.empty or test_symbol.empty:
                continue

            baseline_signal = (
                test_symbol["Signal"]
                .astype(float)
            )

            baseline_weights.loc[
                test_symbol.index,
                symbol,
            ] = baseline_signal.values

            try:

                ml_signal = _build_ml_weights(
                    train_symbol,
                    test_symbol,
                    threshold,
                )

                ml_weights.loc[
                    test_symbol.index,
                    symbol,
                ] = ml_signal["Signal"].values

            except Exception:

                ml_weights.loc[
                    test_symbol.index,
                    symbol,
                ] = 0.0

        baseline_weights = _build_equal_weights(
            baseline_weights
        )

        ml_weights = _build_equal_weights(
            ml_weights
        )

        baseline_weights = (
            _apply_risk_controls_to_weights(
                baseline_weights,
                max_weight,
                max_exposure,
            )
        )

        ml_weights = (
            _apply_risk_controls_to_weights(
                ml_weights,
                max_weight,
                max_exposure,
            )
        )

        baseline_turnover = (
            baseline_weights
            .diff()
            .abs()
            .sum(axis=1)
            .fillna(0.0)
        )

        ml_turnover = (
            ml_weights
            .diff()
            .abs()
            .sum(axis=1)
            .fillna(0.0)
        )

        baseline_returns = pd.DataFrame(
            index=common_index,
            columns=symbols,
            dtype=float,
        )

        ml_returns = baseline_returns.copy()

        for symbol in symbols:

            symbol_data = prepared[symbol]

            returns = symbol_data[
                "Close"
            ].pct_change()

            baseline_returns[
                symbol
            ] = returns.reindex(common_index).fillna(0.0)

            ml_returns[
                symbol
            ] = returns.reindex(common_index).fillna(0.0)

        baseline_gross = (
            baseline_weights.shift(1).fillna(0.0)
            * baseline_returns
        ).sum(axis=1)

        ml_gross = (
            ml_weights.shift(1).fillna(0.0)
            * ml_returns
        ).sum(axis=1)

        baseline_cost = (
            baseline_turnover
            * transaction_cost
        )

        ml_cost = (
            ml_turnover
            * transaction_cost
        )

        baseline_net = (
            baseline_gross
            - baseline_cost
        )

        ml_net = (
            ml_gross
            - ml_cost
        )

        benchmark_returns = (
            baseline_returns
            .mean(axis=1)
        )

        baseline_metrics = _portfolio_metrics(
            baseline_net.loc[test.index],
            baseline_turnover.loc[test.index],
        )

        ml_metrics = _portfolio_metrics(
            ml_net.loc[test.index],
            ml_turnover.loc[test.index],
        )

        benchmark_metrics = _portfolio_metrics(
            benchmark_returns.loc[test.index],
            pd.Series(
                0.0,
                index=test.index,
            ),
        )

        results.append(
            {
                "Window": window_number,
                "Benchmark_Return": benchmark_metrics[
                    "return"
                ],
                "Baseline_Net_Return": baseline_metrics[
                    "return"
                ],
                "ML_Net_Return": ml_metrics[
                    "return"
                ],
                "Baseline_Net_Sharpe": baseline_metrics[
                    "sharpe"
                ],
                "ML_Net_Sharpe": ml_metrics[
                    "sharpe"
                ],
                "Baseline_Net_Drawdown": baseline_metrics[
                    "drawdown"
                ],
                "ML_Net_Drawdown": ml_metrics[
                    "drawdown"
                ],
                "Baseline_Trades": baseline_metrics[
                    "trades"
                ],
                "ML_Trades": ml_metrics[
                    "trades"
                ],
            }
        )

    return pd.DataFrame(results)


def print_portfolio_walk_forward_report(
    results: pd.DataFrame,
) -> None:

    print(
        "\nPORTFOLIO WALK-FORWARD EVALUATION"
    )

    print("=" * 95)

    print("WINDOW RESULTS")
    print("=" * 95)

    print(
        results.to_string(
            index=False,
            float_format=lambda x: f"{x:.6f}",
        )
    )

    print(
        "\nAGGREGATE OUT-OF-SAMPLE RESULTS"
    )

    print("=" * 95)

    benchmark_return = (
        (1 + results["Benchmark_Return"])
        .prod()
        - 1
    )

    baseline_return = (
        (1 + results["Baseline_Net_Return"])
        .prod()
        - 1
    )

    ml_return = (
        (1 + results["ML_Net_Return"])
        .prod()
        - 1
    )

    baseline_sharpe = _safe_sharpe(
        results["Baseline_Net_Return"]
    )

    ml_sharpe = _safe_sharpe(
        results["ML_Net_Return"]
    )

    benchmark_sharpe = _safe_sharpe(
        results["Benchmark_Return"]
    )

    baseline_drawdown = (
        results["Baseline_Net_Drawdown"].min()
    )

    ml_drawdown = (
        results["ML_Net_Drawdown"].min()
    )

    print(
        f"Benchmark Return:       "
        f"{benchmark_return:.6f}"
    )

    print(
        f"Baseline Net Return:    "
        f"{baseline_return:.6f}"
    )

    print(
        f"ML Net Return:           "
        f"{ml_return:.6f}"
    )

    print()

    print(
        f"Benchmark Sharpe:       "
        f"{benchmark_sharpe:.6f}"
    )

    print(
        f"Baseline Net Sharpe:    "
        f"{baseline_sharpe:.6f}"
    )

    print(
        f"ML Net Sharpe:          "
        f"{ml_sharpe:.6f}"
    )

    print()

    print(
        f"Baseline Max Drawdown:  "
        f"{baseline_drawdown:.6f}"
    )

    print(
        f"ML Max Drawdown:        "
        f"{ml_drawdown:.6f}"
    )

    print()

    print(
        f"ML vs Benchmark:        "
        f"{ml_return - benchmark_return:.6f}"
    )

    print(
        f"ML vs Baseline:         "
        f"{ml_return - baseline_return:.6f}"
    )