import numpy as np
import pandas as pd

from src.data.loader import load_raw_data
from src.data.preprocessing import (
    add_basic_features,
    validate_market_data,
)
from src.ml.features import (
    FEATURE_COLUMNS,
    build_ml_features,
)
from src.ml.model import MLAlphaModel
from src.alphas.mean_reversion import MeanReversionStrategy
from src.portfolio.risk import apply_risk_controls
from src.portfolio.engine import PortfolioEngine


DEFAULT_TRANSACTION_COST = 0.001
DEFAULT_MAX_WEIGHT = 0.25
DEFAULT_MAX_EXPOSURE = 1.0
DEFAULT_N_WINDOWS = 4


# ============================================================
# METRICS
# ============================================================

def _total_return(
    returns: pd.Series,
) -> float:

    returns = (
        pd.Series(returns)
        .fillna(0.0)
    )

    if returns.empty:
        return 0.0

    return float(
        (1.0 + returns).prod() - 1.0
    )


def _safe_sharpe(
    returns: pd.Series,
) -> float:

    returns = (
        pd.Series(returns)
        .dropna()
    )

    if len(returns) < 2:
        return 0.0

    volatility = returns.std()

    if (
        volatility == 0
        or pd.isna(volatility)
    ):
        return 0.0

    return float(
        returns.mean()
        / volatility
        * np.sqrt(252)
    )


def _max_drawdown(
    returns: pd.Series,
) -> float:

    returns = (
        pd.Series(returns)
        .fillna(0.0)
    )

    if returns.empty:
        return 0.0

    equity = (
        1.0 + returns
    ).cumprod()

    peak = equity.cummax()

    drawdown = (
        equity / peak - 1.0
    )

    return float(
        drawdown.min()
    )


def _win_rate(
    returns: pd.Series,
) -> float:

    returns = (
        pd.Series(returns)
        .dropna()
    )

    if returns.empty:
        return 0.0

    return float(
        (returns > 0).mean()
    )


def _profit_factor(
    returns: pd.Series,
) -> float:

    returns = (
        pd.Series(returns)
        .dropna()
    )

    gains = (
        returns[returns > 0]
        .sum()
    )

    losses = (
        -returns[returns < 0]
        .sum()
    )

    if losses == 0:

        if gains > 0:
            return float("inf")

        return 0.0

    return float(
        gains / losses
    )


def _calculate_metrics(
    returns: pd.Series,
    trades: int,
) -> dict:

    returns = (
        pd.Series(returns)
        .fillna(0.0)
    )

    return {
        "return": _total_return(
            returns
        ),
        "sharpe": _safe_sharpe(
            returns
        ),
        "drawdown": _max_drawdown(
            returns
        ),
        "win_rate": _win_rate(
            returns
        ),
        "profit_factor": _profit_factor(
            returns
        ),
        "trades": int(trades),
    }


# ============================================================
# DATA PREPARATION
# ============================================================

def _build_asset_data(
    symbol: str,
) -> pd.DataFrame:
    """
    Build a clean dataset for one asset.
    """

    print(
        f"Loading {symbol}..."
    )

    raw = load_raw_data(
        symbol
    )

    validate_market_data(
        raw
    )

    ml_data = build_ml_features(
        raw
    )

    strategy = MeanReversionStrategy()

    strategy_data = (
        raw.copy()
    )

    strategy_data = (
        add_basic_features(
            strategy_data
        )
    )

    strategy_data = (
        strategy.generate_signal(
            strategy_data
        )
    )

    ml_data = (
        ml_data
        .set_index("Date")
    )

    strategy_data = (
        strategy_data
        .set_index("Date")
    )

    strategy_data = (
        strategy_data[
            [
                "Close",
                "Signal",
            ]
        ]
        .copy()
    )

    strategy_data = (
        strategy_data.rename(
            columns={
                "Signal":
                    "Baseline_Signal"
            }
        )
    )

    ml_data = (
        ml_data[
            FEATURE_COLUMNS
            + ["Target"]
        ]
        .copy()
    )

    data = (
        strategy_data
        .join(
            ml_data,
            how="inner",
        )
    )

    data["Return"] = (
        data["Close"]
        .pct_change()
    )

    data = data.replace(
        [np.inf, -np.inf],
        np.nan,
    )

    data = data.dropna(
        subset=(
            FEATURE_COLUMNS
            + [
                "Target",
                "Return",
                "Baseline_Signal",
            ]
        )
    )

    data = (
        data
        .sort_index()
        .copy()
    )

    return data


# ============================================================
# WALK-FORWARD WINDOWS
# ============================================================

def _walk_forward_windows(
    index: pd.DatetimeIndex,
    n_windows: int = DEFAULT_N_WINDOWS,
) -> list[
    tuple[
        pd.DatetimeIndex,
        pd.DatetimeIndex,
    ]
]:

    if n_windows < 1:
        raise ValueError(
            "n_windows must be at least 1."
        )

    index = (
        pd.DatetimeIndex(index)
        .sort_values()
        .unique()
    )

    n = len(index)

    if n < 100:
        raise ValueError(
            "Not enough observations for "
            "walk-forward validation."
        )

    test_size = (
        n // (n_windows + 1)
    )

    if test_size < 1:
        raise ValueError(
            "Test window size is too small."
        )

    windows = []

    for i in range(n_windows):

        train_end = (
            n
            - (n_windows - i)
            * test_size
        )

        test_start = train_end

        test_end = min(
            test_start + test_size,
            n,
        )

        train_index = index[
            :train_end
        ]

        test_index = index[
            test_start:test_end
        ]

        if len(train_index) == 0:
            continue

        if len(test_index) == 0:
            continue

        if (
            train_index[-1]
            >= test_index[0]
        ):
            raise ValueError(
                "Walk-forward split overlaps."
            )

        windows.append(
            (
                train_index,
                test_index,
            )
        )

    return windows


# ============================================================
# ML MODEL
# ============================================================

def _fit_ml_model(
    train: pd.DataFrame,
) -> MLAlphaModel:

    missing = [
        column
        for column in FEATURE_COLUMNS
        if column not in train.columns
    ]

    if missing:
        raise ValueError(
            f"Missing ML features: {missing}"
        )

    if "Target" not in train.columns:
        raise ValueError(
            "Training data must contain Target."
        )

    X_train = (
        train[
            FEATURE_COLUMNS
        ]
        .copy()
    )

    y_train = (
        train["Target"]
        .copy()
    )

    if y_train.nunique() < 2:
        raise ValueError(
            "Training target contains only "
            "one class."
        )

    model = MLAlphaModel()

    model.fit(
        X_train,
        y_train,
    )

    return model


def _generate_ml_signal(
    model: MLAlphaModel,
    test: pd.DataFrame,
    threshold: float,
) -> pd.Series:

    if not 0 < threshold < 1:
        raise ValueError(
            "threshold must be between 0 and 1."
        )

    X = (
        test[
            FEATURE_COLUMNS
        ]
        .copy()
    )

    probabilities = (
        model.predict_probability(
            X
        )
    )

    probabilities = pd.Series(
        probabilities,
        index=test.index,
        dtype=float,
    )

    signal = (
        (
            test["Baseline_Signal"]
            > 0
        )
        & (
            probabilities
            >= threshold
        )
    ).astype(float)

    return signal


# ============================================================
# PORTFOLIO WEIGHTS
# ============================================================

def _build_weights(
    signals: pd.DataFrame,
    max_weight: float,
    max_exposure: float,
) -> pd.DataFrame:

    if max_weight <= 0:
        raise ValueError(
            "max_weight must be positive."
        )

    if max_exposure <= 0:
        raise ValueError(
            "max_exposure must be positive."
        )

    active = (
        signals
        .clip(
            lower=0.0,
            upper=1.0,
        )
    )

    active_count = (
        active.sum(axis=1)
    )

    weights = (
        active.div(
            active_count.replace(
                0,
                np.nan,
            ),
            axis=0,
        )
        .fillna(0.0)
    )

    weights = apply_risk_controls(
        weights,
        max_weight=max_weight,
        max_exposure=max_exposure,
    )

    return weights


# ============================================================
# MAIN WALK-FORWARD VALIDATION
# ============================================================

def run_portfolio_walk_forward(
    symbols: list[str],
    ml_threshold: float = 0.55,
    transaction_cost: float = DEFAULT_TRANSACTION_COST,
    max_weight: float = DEFAULT_MAX_WEIGHT,
    max_exposure: float = DEFAULT_MAX_EXPOSURE,
    n_windows: int = DEFAULT_N_WINDOWS,
) -> dict:
    """
    Multi-asset walk-forward validation with CONTINUOUS portfolio execution.
    """

    if not symbols:
        raise ValueError(
            "symbols cannot be empty."
        )

    if not 0 < ml_threshold < 1:
        raise ValueError(
            "ml_threshold must be between 0 and 1."
        )

    if transaction_cost < 0:
        raise ValueError(
            "transaction_cost cannot be negative."
        )

    if max_weight <= 0:
        raise ValueError(
            "max_weight must be positive."
        )

    if max_exposure <= 0:
        raise ValueError(
            "max_exposure must be positive."
        )

    if n_windows < 1:
        raise ValueError(
            "n_windows must be at least 1."
        )

    symbols = list(dict.fromkeys(symbols))

    print("PORTFOLIO WALK-FORWARD VALIDATION")
    print("=" * 95)
    print("Symbols:", ", ".join(symbols))
    print(f"ML threshold: {ml_threshold}")

    # 1. Load data
    asset_data = {}
    for symbol in symbols:
        asset_data[symbol] = _build_asset_data(symbol)

    # 2. Extract common dates
    common_index = None
    for symbol in symbols:
        index = asset_data[symbol].index
        if common_index is None:
            common_index = index
        else:
            common_index = common_index.intersection(index)

    if common_index is None:
        raise ValueError("No common dates found.")

    common_index = pd.DatetimeIndex(common_index).sort_values().unique()

    if len(common_index) < 100:
        raise ValueError("Insufficient common observations.")

    for symbol in symbols:
        asset_data[symbol] = asset_data[symbol].loc[common_index].copy()

    # 3. Create Windows
    windows = _walk_forward_windows(common_index, n_windows=n_windows)

    # --------------------------------------------------------
    # 4. Generate Continuous Model Predictions
    # --------------------------------------------------------

    all_baseline_weights = []
    all_ml_weights = []
    window_metadata = []

    for window_number, (train_dates, test_dates) in enumerate(windows, start=1):
        print(f"\nRunning portfolio walk-forward window {window_number}...")

        # Build pooled train dataset
        train_parts = []
        for symbol in symbols:
            train_part = asset_data[symbol].loc[train_dates].copy()
            train_part["Symbol"] = symbol
            train_parts.append(train_part)

        train = pd.concat(train_parts, axis=0)
        model = _fit_ml_model(train)

        # To evaluate on the first day of the test window, the portfolio engine
        # requires the target weight generated on the final day of the training set.
        prediction_dates = [train_dates[-1]] + list(test_dates)

        baseline_signals = pd.DataFrame(0.0, index=prediction_dates, columns=symbols)
        ml_signals = pd.DataFrame(0.0, index=prediction_dates, columns=symbols)

        for symbol in symbols:
            pred_asset = asset_data[symbol].loc[prediction_dates].copy()

            baseline_signals.loc[prediction_dates, symbol] = (
                pred_asset["Baseline_Signal"].astype(float).values
            )

            ml_signal = _generate_ml_signal(model, pred_asset, ml_threshold)
            ml_signals.loc[prediction_dates, symbol] = ml_signal.values

        baseline_weights = _build_weights(
            baseline_signals, max_weight=max_weight, max_exposure=max_exposure
        )

        ml_weights = _build_weights(
            ml_signals, max_weight=max_weight, max_exposure=max_exposure
        )

        all_baseline_weights.append(baseline_weights)
        all_ml_weights.append(ml_weights)

        # Record slices for metrics. First window absorbs the initial setup cost day.
        slice_dates = prediction_dates if window_number == 1 else list(test_dates)

        window_metadata.append({
            "window": window_number,
            "train_rows": len(train),
            "slice_dates": slice_dates
        })

    # --------------------------------------------------------
    # 5. Stitch Weights and Run Continuous Portfolio Engine
    # --------------------------------------------------------
    print("\nStitching continuous portfolio and applying precise mathematical execution...")

    # .groupby(level=0).last() gracefully overwrites overlapping boundary days
    # with the target weight generated by the freshly retrained walk-forward model.
    master_baseline_weights = (
        pd.concat(all_baseline_weights)
        .groupby(level=0)
        .last()
        .sort_index()
    )

    master_ml_weights = (
        pd.concat(all_ml_weights)
        .groupby(level=0)
        .last()
        .sort_index()
    )

    master_benchmark_weights = pd.DataFrame(
        1.0 / len(symbols),
        index=master_baseline_weights.index,
        columns=symbols,
    )

    master_returns = pd.DataFrame(
        index=master_baseline_weights.index,
        columns=symbols
    )
    for symbol in symbols:
        master_returns[symbol] = asset_data[symbol]["Return"]

    master_returns = master_returns.astype(float)

    strategy_engine = PortfolioEngine(transaction_cost=transaction_cost)
    benchmark_engine = PortfolioEngine(transaction_cost=0.0)

    baseline_result, _ = strategy_engine.run(master_returns, master_baseline_weights)
    ml_result, _ = strategy_engine.run(master_returns, master_ml_weights)
    benchmark_result, _ = benchmark_engine.run(master_returns, master_benchmark_weights)

    # --------------------------------------------------------
    # 6. Slice Global Results for Window Reporting
    # --------------------------------------------------------

    window_results = []

    for meta in window_metadata:
        w_idx = meta["slice_dates"]

        b_net = baseline_result["Net_Return"].loc[w_idx]
        b_turn = baseline_result["Turnover"].loc[w_idx]

        m_net = ml_result["Net_Return"].loc[w_idx]
        m_turn = ml_result["Turnover"].loc[w_idx]

        bm_net = benchmark_result["Net_Return"].loc[w_idx]

        b_metrics = _calculate_metrics(b_net, trades=int((b_turn > 1e-8).sum()))
        m_metrics = _calculate_metrics(m_net, trades=int((m_turn > 1e-8).sum()))
        bm_metrics = _calculate_metrics(bm_net, trades=0)

        window_results.append({
            "Window": meta["window"],
            "Train_Rows": meta["train_rows"],
            "Test_Rows": len(w_idx),
            "Test_Start": w_idx[0],
            "Test_End": w_idx[-1],
            "Benchmark_Return": bm_metrics["return"],
            "Baseline_Net_Return": b_metrics["return"],
            "ML_Net_Return": m_metrics["return"],
            "Baseline_Net_Sharpe": b_metrics["sharpe"],
            "ML_Net_Sharpe": m_metrics["sharpe"],
            "Baseline_Net_Drawdown": b_metrics["drawdown"],
            "ML_Net_Drawdown": m_metrics["drawdown"],
            "Baseline_Win_Rate": b_metrics["win_rate"],
            "ML_Win_Rate": m_metrics["win_rate"],
            "Baseline_Profit_Factor": b_metrics["profit_factor"],
            "ML_Profit_Factor": m_metrics["profit_factor"],
            "Baseline_Trades": b_metrics["trades"],
            "ML_Trades": m_metrics["trades"],
        })

    # --------------------------------------------------------
    # 7. Aggregate OOS Metrics
    # --------------------------------------------------------

    b_full_net = baseline_result["Net_Return"]
    b_full_turn = baseline_result["Turnover"]

    m_full_net = ml_result["Net_Return"]
    m_full_turn = ml_result["Turnover"]

    bm_full_net = benchmark_result["Net_Return"]

    agg_b_metrics = _calculate_metrics(b_full_net, trades=int((b_full_turn > 1e-8).sum()))
    agg_m_metrics = _calculate_metrics(m_full_net, trades=int((m_full_turn > 1e-8).sum()))
    agg_bm_metrics = _calculate_metrics(bm_full_net, trades=0)

    aggregate = {
        "benchmark_return": agg_bm_metrics["return"],
        "baseline_return": agg_b_metrics["return"],
        "ml_return": agg_m_metrics["return"],

        "benchmark_sharpe": agg_bm_metrics["sharpe"],
        "baseline_sharpe": agg_b_metrics["sharpe"],
        "ml_sharpe": agg_m_metrics["sharpe"],

        "baseline_drawdown": agg_b_metrics["drawdown"],
        "ml_drawdown": agg_m_metrics["drawdown"],

        "baseline_win_rate": agg_b_metrics["win_rate"],
        "ml_win_rate": agg_m_metrics["win_rate"],

        "baseline_profit_factor": agg_b_metrics["profit_factor"],
        "ml_profit_factor": agg_m_metrics["profit_factor"],

        "baseline_trades": agg_b_metrics["trades"],
        "ml_trades": agg_m_metrics["trades"],

        "baseline_vs_benchmark": agg_b_metrics["return"] - agg_bm_metrics["return"],
        "ml_vs_benchmark": agg_m_metrics["return"] - agg_bm_metrics["return"],
        "ml_vs_baseline": agg_m_metrics["return"] - agg_b_metrics["return"],

        "oos_days": len(m_full_net),
    }

    # --------------------------------------------------------
    # 8. Result DataFrames.
    # --------------------------------------------------------

    window_df = pd.DataFrame(window_results)

    daily_df = pd.DataFrame({
        "Benchmark_Return": bm_full_net,
        "Baseline_Net_Return": b_full_net,
        "ML_Net_Return": m_full_net,
    })

    return {
        "windows": window_df,
        "window_results": window_df,
        "daily_returns": daily_df,
        "aggregate": aggregate,
    }


# ============================================================
# REPORT
# ============================================================

def print_portfolio_walk_forward_report(
    results: dict,
) -> None:
    """
    Print portfolio walk-forward report.
    """

    window_df = results.get(
        "window_results",
        results.get("windows"),
    )

    aggregate = results[
        "aggregate"
    ]

    print()

    print(
        "PORTFOLIO WALK-FORWARD EVALUATION"
    )

    print(
        "=" * 95
    )

    print(
        "WINDOW RESULTS"
    )

    print(
        "=" * 95
    )

    display_columns = [
        "Window",
        "Benchmark_Return",
        "Baseline_Net_Return",
        "ML_Net_Return",
        "Baseline_Net_Sharpe",
        "ML_Net_Sharpe",
        "Baseline_Net_Drawdown",
        "ML_Net_Drawdown",
        "Baseline_Trades",
        "ML_Trades",
    ]

    print(
        window_df[
            display_columns
        ].to_string(
            index=False
        )
    )

    print()

    print(
        "AGGREGATE OUT-OF-SAMPLE RESULTS"
    )

    print(
        "=" * 95
    )

    print(
        f"Benchmark Return:       "
        f"{aggregate['benchmark_return']:.6f}"
    )

    print(
        f"Baseline Net Return:    "
        f"{aggregate['baseline_return']:.6f}"
    )

    print(
        f"ML Net Return:          "
        f"{aggregate['ml_return']:.6f}"
    )

    print()

    print(
        f"Benchmark Sharpe:       "
        f"{aggregate['benchmark_sharpe']:.6f}"
    )

    print(
        f"Baseline Net Sharpe:    "
        f"{aggregate['baseline_sharpe']:.6f}"
    )

    print(
        f"ML Net Sharpe:          "
        f"{aggregate['ml_sharpe']:.6f}"
    )

    print()

    print(
        f"Baseline Max Drawdown:  "
        f"{aggregate['baseline_drawdown']:.6f}"
    )

    print(
        f"ML Max Drawdown:        "
        f"{aggregate['ml_drawdown']:.6f}"
    )

    print()

    print(
        f"Baseline Win Rate:      "
        f"{aggregate['baseline_win_rate']:.4f}"
    )

    print(
        f"ML Win Rate:            "
        f"{aggregate['ml_win_rate']:.4f}"
    )

    print()

    print(
        f"Baseline Profit Factor: "
        f"{aggregate['baseline_profit_factor']:.4f}"
    )

    print(
        f"ML Profit Factor:       "
        f"{aggregate['ml_profit_factor']:.4f}"
    )

    print()

    print(
        f"Baseline vs Benchmark:  "
        f"{aggregate['baseline_vs_benchmark']:.6f}"
    )

    print(
        f"ML vs Benchmark:        "
        f"{aggregate['ml_vs_benchmark']:.6f}"
    )

    print(
        f"ML vs Baseline:         "
        f"{aggregate['ml_vs_baseline']:.6f}"
    )

    print()

    print(
        f"OOS Days:               "
        f"{aggregate['oos_days']}"
    )
