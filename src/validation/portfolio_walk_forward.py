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


DEFAULT_TRANSACTION_COST = 0.001
DEFAULT_MAX_WEIGHT = 0.25
DEFAULT_MAX_EXPOSURE = 1.0
DEFAULT_N_WINDOWS = 4


# ============================================================
# METRICS
# ============================================================

def _total_return(returns: pd.Series) -> float:
    returns = pd.Series(returns).fillna(0.0)

    if returns.empty:
        return 0.0

    return float((1.0 + returns).prod() - 1.0)


def _sharpe(returns: pd.Series) -> float:
    returns = pd.Series(returns).dropna()

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
    returns = pd.Series(returns).fillna(0.0)

    if returns.empty:
        return 0.0

    equity = (1.0 + returns).cumprod()
    peak = equity.cummax()

    drawdown = equity / peak - 1.0

    return float(drawdown.min())


def _win_rate(returns: pd.Series) -> float:
    returns = pd.Series(returns).dropna()

    if returns.empty:
        return 0.0

    return float((returns > 0).mean())


def _profit_factor(returns: pd.Series) -> float:
    returns = pd.Series(returns).dropna()

    gains = returns[returns > 0].sum()
    losses = -returns[returns < 0].sum()

    if losses == 0:
        if gains > 0:
            return float("inf")

        return 0.0

    return float(gains / losses)


def _calculate_metrics(
    returns: pd.Series,
    trades: int,
) -> dict:

    returns = pd.Series(returns).fillna(0.0)

    return {
        "return": _total_return(returns),
        "sharpe": _sharpe(returns),
        "drawdown": _max_drawdown(returns),
        "win_rate": _win_rate(returns),
        "profit_factor": _profit_factor(returns),
        "trades": int(trades),
    }


# ============================================================
# DATA PREPARATION
# ============================================================

def _build_asset_data(
    symbol: str,
) -> pd.DataFrame:
    """
    Build one clean dataset for one asset.

    Contains:
        Date
        Close
        Baseline_Signal
        ML features
        Target
        Return
    """

    print(f"Loading {symbol}...")

    raw = load_raw_data(symbol)

    validate_market_data(raw)

    # --------------------------------------------------------
    # ML pipeline
    # --------------------------------------------------------

    ml_data = build_ml_features(raw)

    # --------------------------------------------------------
    # Baseline strategy pipeline
    # --------------------------------------------------------

    strategy = MeanReversionStrategy()

    strategy_data = raw.copy()

    strategy_data = add_basic_features(
        strategy_data
    )

    strategy_data = strategy.generate_signal(
        strategy_data
    )

    # --------------------------------------------------------
    # Index by Date
    # --------------------------------------------------------

    ml_data = ml_data.set_index("Date")

    strategy_data = strategy_data.set_index("Date")

    # --------------------------------------------------------
    # Keep only required strategy columns
    # --------------------------------------------------------

    strategy_data = strategy_data[
        [
            "Close",
            "Signal",
        ]
    ].copy()

    strategy_data = strategy_data.rename(
        columns={
            "Signal": "Baseline_Signal"
        }
    )

    # --------------------------------------------------------
    # Keep ML columns
    # --------------------------------------------------------

    ml_data = ml_data[
        FEATURE_COLUMNS + ["Target"]
    ].copy()

    # --------------------------------------------------------
    # Combine
    # --------------------------------------------------------

    data = strategy_data.join(
        ml_data,
        how="inner",
    )

    # --------------------------------------------------------
    # Calculate next-period return
    # --------------------------------------------------------

    data["Return"] = (
        data["Close"].pct_change()
    )

    # --------------------------------------------------------
    # Clean data
    # --------------------------------------------------------

    data = data.replace(
        [np.inf, -np.inf],
        np.nan,
    )

    data = data.dropna(
        subset=FEATURE_COLUMNS
        + ["Target", "Return"]
    )

    data = data.sort_index()

    return data


# ============================================================
# WALK-FORWARD WINDOWS
# ============================================================

def _walk_forward_windows(
    index: pd.DatetimeIndex,
    n_windows: int = DEFAULT_N_WINDOWS,
) -> list[tuple[pd.DatetimeIndex, pd.DatetimeIndex]]:

    index = pd.DatetimeIndex(
        sorted(index.unique())
    )

    n = len(index)

    if n < 100:
        raise ValueError(
            "Not enough observations for "
            "walk-forward validation."
        )

    test_size = n // (n_windows + 1)

    if test_size < 1:
        raise ValueError(
            "Test window size is too small."
        )

    windows = []

    for i in range(n_windows):

        train_end = (
            n
            - (n_windows - i) * test_size
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

        if train_index[-1] >= test_index[0]:
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

    X_train = train[
        FEATURE_COLUMNS
    ].copy()

    y_train = train[
        "Target"
    ].copy()

    model = MLAlphaModel()

    model.fit(
        X_train,
        y_train,
    )

    return model


def _ml_signal(
    model: MLAlphaModel,
    data: pd.DataFrame,
    threshold: float,
) -> pd.Series:

    X = data[
        FEATURE_COLUMNS
    ].copy()

    probabilities = model.predict_probability(
        X
    )

    probabilities = np.asarray(
        probabilities
    )

    # Handle models returning Nx2 probabilities.
    if probabilities.ndim == 2:

        if probabilities.shape[1] >= 2:
            probabilities = probabilities[:, 1]
        else:
            probabilities = probabilities[:, 0]

    probabilities = pd.Series(
        probabilities,
        index=data.index,
    )

    return (
        probabilities >= threshold
    ).astype(float)


# ============================================================
# PORTFOLIO WEIGHTS
# ============================================================

def _build_weights(
    signals: pd.DataFrame,
    max_weight: float,
    max_exposure: float,
) -> pd.DataFrame:
    """
    Equal-weight active positions followed by
    portfolio risk controls.
    """

    active = signals.clip(
        lower=0,
        upper=1,
    )

    active_count = active.sum(
        axis=1
    )

    weights = active.div(
        active_count.replace(
            0,
            np.nan,
        ),
        axis=0,
    ).fillna(0.0)

    weights = apply_risk_controls(
        weights,
        max_weight=max_weight,
        max_exposure=max_exposure,
    )

    return weights


# ============================================================
# PORTFOLIO ENGINE
# ============================================================

def _run_portfolio(
    returns: pd.DataFrame,
    weights: pd.DataFrame,
    transaction_cost: float,
) -> tuple[pd.DataFrame, dict]:

    if not returns.index.equals(
        weights.index
    ):
        raise ValueError(
            "Returns and weights must have "
            "identical indices."
        )

    if not returns.columns.equals(
        weights.columns
    ):
        raise ValueError(
            "Returns and weights must have "
            "identical columns."
        )

    # Target weights generated at t
    # become positions at t+1.
    position_weights = (
        weights
        .shift(1)
        .fillna(0.0)
    )

    gross_returns = (
        position_weights * returns
    ).sum(axis=1)

    turnover = (
        weights
        .diff()
        .abs()
        .sum(axis=1)
        .fillna(0.0)
    )

    costs = (
        turnover
        * transaction_cost
    )

    net_returns = (
        gross_returns
        - costs
    )

    equity = (
        1.0 + net_returns
    ).cumprod()

    trades = int(
        (turnover > 0).sum()
    )

    result = pd.DataFrame(
        {
            "Gross_Return": gross_returns,
            "Transaction_Cost": costs,
            "Net_Return": net_returns,
            "Equity": equity,
        },
        index=returns.index,
    )

    metrics = _calculate_metrics(
        net_returns,
        trades,
    )

    metrics["final_equity"] = float(
        equity.iloc[-1]
    )

    metrics["average_turnover"] = float(
        turnover.mean()
    )

    return result, metrics


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

    if not symbols:
        raise ValueError(
            "symbols cannot be empty."
        )

    if not 0 < ml_threshold < 1:
        raise ValueError(
            "ml_threshold must be between 0 and 1."
        )

    print(
        "PORTFOLIO WALK-FORWARD VALIDATION"
    )

    print(
        "=" * 95
    )

    print(
        f"Symbols: {', '.join(symbols)}"
    )

    print(
        f"ML threshold: {ml_threshold}"
    )

    # --------------------------------------------------------
    # Load each asset
    # --------------------------------------------------------

    asset_data = {}

    for symbol in symbols:

        asset_data[symbol] = (
            _build_asset_data(symbol)
        )

    # --------------------------------------------------------
    # Common dates
    # --------------------------------------------------------

    common_index = None

    for symbol in symbols:

        index = asset_data[
            symbol
        ].index

        if common_index is None:
            common_index = index
        else:
            common_index = (
                common_index
                .intersection(index)
            )

    common_index = (
        common_index
        .sort_values()
    )

    if len(common_index) < 100:
        raise ValueError(
            "Not enough common observations "
            "across assets."
        )

    for symbol in symbols:

        asset_data[symbol] = (
            asset_data[symbol]
            .loc[common_index]
            .copy()
        )

    # --------------------------------------------------------
    # Build portfolio-level matrices
    # --------------------------------------------------------

    baseline_signals = pd.DataFrame(
        {
            symbol: asset_data[
                symbol
            ]["Baseline_Signal"]
            for symbol in symbols
        },
        index=common_index,
    )

    returns = pd.DataFrame(
        {
            symbol: asset_data[
                symbol
            ]["Return"]
            for symbol in symbols
        },
        index=common_index,
    )

    # --------------------------------------------------------
    # Walk-forward windows
    # --------------------------------------------------------

    windows = _walk_forward_windows(
        common_index,
        n_windows=n_windows,
    )

    window_results = []

    all_baseline_returns = []
    all_ml_returns = []
    all_benchmark_returns = []

    # --------------------------------------------------------
    # Process each window
    # --------------------------------------------------------

    for window_number, (
        train_dates,
        test_dates,
    ) in enumerate(
        windows,
        start=1,
    ):

        print(
            f"\nRunning portfolio walk-forward "
            f"window {window_number}..."
        )

        # ----------------------------------------------------
        # Build training dataset
        # ----------------------------------------------------

        train_parts = []

        for symbol in symbols:

            train_part = (
                asset_data[symbol]
                .loc[train_dates]
                .copy()
            )

            train_part["Symbol"] = symbol

            train_parts.append(
                train_part
            )

        train = pd.concat(
            train_parts
        )

        # ----------------------------------------------------
        # Fit ML model
        # ----------------------------------------------------

        model = _fit_ml_model(
            train
        )

        # ----------------------------------------------------
        # Generate ML signals
        # ----------------------------------------------------

        ml_signals = pd.DataFrame(
            0.0,
            index=test_dates,
            columns=symbols,
        )

        for symbol in symbols:

            test_asset = (
                asset_data[symbol]
                .loc[test_dates]
            )

            ml_signals.loc[
                test_dates,
                symbol,
            ] = _ml_signal(
                model,
                test_asset,
                ml_threshold,
            ).values

        # ----------------------------------------------------
        # Baseline signals
        # ----------------------------------------------------

        baseline_window_signals = (
            baseline_signals
            .loc[test_dates]
            .copy()
        )

        # ----------------------------------------------------
        # Build risk-controlled weights
        # ----------------------------------------------------

        baseline_weights = _build_weights(
            baseline_window_signals,
            max_weight=max_weight,
            max_exposure=max_exposure,
        )

        ml_weights = _build_weights(
            ml_signals,
            max_weight=max_weight,
            max_exposure=max_exposure,
        )

        # ----------------------------------------------------
        # Test returns
        # ----------------------------------------------------

        test_returns = (
            returns.loc[test_dates]
            .copy()
        )

        # ----------------------------------------------------
        # Baseline portfolio
        # ----------------------------------------------------

        _, baseline_metrics = _run_portfolio(
            returns=test_returns,
            weights=baseline_weights,
            transaction_cost=transaction_cost,
        )

        # ----------------------------------------------------
        # ML portfolio
        # ----------------------------------------------------

        _, ml_metrics = _run_portfolio(
            returns=test_returns,
            weights=ml_weights,
            transaction_cost=transaction_cost,
        )

        # ----------------------------------------------------
        # Benchmark
        # Equal-weight buy-and-hold daily benchmark
        # ----------------------------------------------------

        benchmark_returns = (
            test_returns.mean(axis=1)
        )

        benchmark_metrics = (
            _calculate_metrics(
                benchmark_returns,
                trades=0,
            )
        )

        # ----------------------------------------------------
        # Save window returns
        # ----------------------------------------------------

        all_baseline_returns.append(
            _run_portfolio(
                test_returns,
                baseline_weights,
                transaction_cost,
            )[0]["Net_Return"]
        )

        all_ml_returns.append(
            _run_portfolio(
                test_returns,
                ml_weights,
                transaction_cost,
            )[0]["Net_Return"]
        )

        all_benchmark_returns.append(
            benchmark_returns
        )

        # ----------------------------------------------------
        # Window result
        # ----------------------------------------------------

        window_results.append(
            {
                "Window": window_number,
                "Benchmark_Return":
                    benchmark_metrics["return"],
                "Baseline_Net_Return":
                    baseline_metrics["return"],
                "ML_Net_Return":
                    ml_metrics["return"],
                "Baseline_Net_Sharpe":
                    baseline_metrics["sharpe"],
                "ML_Net_Sharpe":
                    ml_metrics["sharpe"],
                "Baseline_Net_Drawdown":
                    baseline_metrics["drawdown"],
                "ML_Net_Drawdown":
                    ml_metrics["drawdown"],
                "Baseline_Trades":
                    baseline_metrics["trades"],
                "ML_Trades":
                    ml_metrics["trades"],
            }
        )

    # ========================================================
    # AGGREGATE OOS RESULTS
    # ========================================================

    benchmark_oos = pd.concat(
        all_benchmark_returns
    )

    baseline_oos = pd.concat(
        all_baseline_returns
    )

    ml_oos = pd.concat(
        all_ml_returns
    )

    benchmark_metrics = _calculate_metrics(
        benchmark_oos,
        trades=0,
    )

    baseline_metrics = _calculate_metrics(
        baseline_oos,
        trades=int(
            (baseline_oos != 0).sum()
        ),
    )

    ml_metrics = _calculate_metrics(
        ml_oos,
        trades=int(
            (ml_oos != 0).sum()
        ),
    )

    aggregate = {
        "benchmark": benchmark_metrics,
        "baseline": baseline_metrics,
        "ml": ml_metrics,
        "baseline_vs_benchmark":
            baseline_metrics["return"]
            - benchmark_metrics["return"],
        "ml_vs_benchmark":
            ml_metrics["return"]
            - benchmark_metrics["return"],
        "ml_vs_baseline":
            ml_metrics["return"]
            - baseline_metrics["return"],
        "oos_days": len(
            ml_oos
        ),
    }

    return {
        "windows": pd.DataFrame(
            window_results
        ),
        "aggregate": aggregate,
    }


# ============================================================
# REPORT
# ============================================================

def print_portfolio_walk_forward_report(
    results: dict,
) -> None:

    windows = results[
        "windows"
    ]

    aggregate = results[
        "aggregate"
    ]

    benchmark = aggregate[
        "benchmark"
    ]

    baseline = aggregate[
        "baseline"
    ]

    ml = aggregate[
        "ml"
    ]

    print(
        "\nPORTFOLIO WALK-FORWARD EVALUATION"
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

    print(
        windows.to_string(
            index=False
        )
    )

    print(
        "\nAGGREGATE OUT-OF-SAMPLE RESULTS"
    )

    print(
        "=" * 95
    )

    print(
        f"Benchmark Return:       "
        f"{benchmark['return']:.6f}"
    )

    print(
        f"Baseline Net Return:    "
        f"{baseline['return']:.6f}"
    )

    print(
        f"ML Net Return:          "
        f"{ml['return']:.6f}"
    )

    print()

    print(
        f"Benchmark Sharpe:       "
        f"{benchmark['sharpe']:.6f}"
    )

    print(
        f"Baseline Net Sharpe:    "
        f"{baseline['sharpe']:.6f}"
    )

    print(
        f"ML Net Sharpe:          "
        f"{ml['sharpe']:.6f}"
    )

    print()

    print(
        f"Baseline Max Drawdown:  "
        f"{baseline['drawdown']:.6f}"
    )

    print(
        f"ML Max Drawdown:        "
        f"{ml['drawdown']:.6f}"
    )

    print()

    print(
        f"Baseline Win Rate:      "
        f"{baseline['win_rate']:.4f}"
    )

    print(
        f"ML Win Rate:            "
        f"{ml['win_rate']:.4f}"
    )

    print()

    print(
        f"Baseline Profit Factor: "
        f"{baseline['profit_factor']:.4f}"
    )

    print(
        f"ML Profit Factor:       "
        f"{ml['profit_factor']:.4f}"
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