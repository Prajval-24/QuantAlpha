import pandas as pd
import numpy as np

from src.ml.model import MLAlphaModel
from src.ml.features import FEATURE_COLUMNS
from src.alphas.mean_reversion import MeanReversionStrategy
from src.validation.walk_forward import (
    generate_walk_forward_splits,
    validate_walk_forward_splits,
)


def calculate_strategy_metrics(
    returns: pd.Series,
    signals: pd.Series,
) -> dict:
    """
    Calculate basic trading performance metrics.
    """

    returns = returns.fillna(0.0)

    equity = (
        1.0 + returns
    ).cumprod()

    total_return = (
        equity.iloc[-1] - 1.0
    )

    volatility = returns.std()

    if volatility > 0:
        sharpe = (
            returns.mean()
            / volatility
            * np.sqrt(252)
        )
    else:
        sharpe = 0.0

    running_max = equity.cummax()

    drawdown = (
        equity / running_max
    ) - 1.0

    max_drawdown = drawdown.min()

    trades = (
        signals.diff()
        .abs()
        .fillna(0)
        .sum()
    )

    return {
        "total_return": total_return,
        "sharpe_ratio": sharpe,
        "max_drawdown": max_drawdown,
        "trades": int(trades),
    }


def prepare_strategy_data(
    data: pd.DataFrame,
) -> pd.DataFrame:
    """
    Prepare the market data required by the
    mean-reversion strategy.

    The strategy requires MA_20 and MA_Distance.
    """

    data = data.copy()

    if "Close" not in data.columns:
        raise ValueError(
            "Data must contain a Close column."
        )

    data["MA_20"] = (
        data["Close"]
        .rolling(20)
        .mean()
    )

    data["MA_Distance"] = (
        data["Close"]
        / data["MA_20"]
        - 1.0
    )

    data = data.dropna(
        subset=[
            "MA_20",
            "MA_Distance",
        ]
    ).copy()

    return data


def calculate_forward_returns(
    data: pd.DataFrame,
) -> pd.Series:
    """
    Calculate next-period returns.

    Signal generated at time t is applied to
    the return from t to t+1.
    """

    return (
        data["Close"]
        .pct_change()
        .shift(-1)
        .fillna(0.0)
    )


def run_walk_forward_strategy(
    data: pd.DataFrame,
    threshold: float = 0.55,
) -> pd.DataFrame:
    """
    Run complete walk-forward strategy evaluation.

    Each model is trained only on historical data
    available before the corresponding test window.

    Returns one row per out-of-sample observation.
    """

    data = prepare_strategy_data(data)

    splits = generate_walk_forward_splits(data)

    validate_walk_forward_splits(splits)

    strategy = MeanReversionStrategy()

    all_results = []

    for window_number, (train, test) in enumerate(
        splits,
        start=1,
    ):

        print(
            f"Running strategy walk-forward "
            f"window {window_number}..."
        )

        # ------------------------------------------
        # Train ML model using historical data only
        # ------------------------------------------

        model = MLAlphaModel()

        model.fit(
            train[FEATURE_COLUMNS],
            train["Target"],
        )

        # ------------------------------------------
        # Generate baseline strategy signal
        # ------------------------------------------

        baseline_output = strategy.generate_signal(test)

        # Strategy implementations may return either:
        #   1. a Series / array of signals
        #   2. a DataFrame containing a Signal column

        if isinstance(
            baseline_output,
            pd.DataFrame,
        ):

            if "Signal" in baseline_output.columns:

                baseline_signal = (
                    baseline_output["Signal"]
                    .to_numpy()
                )

            elif "signal" in baseline_output.columns:

                baseline_signal = (
                    baseline_output["signal"]
                    .to_numpy()
                )

            else:
                raise ValueError(
                    "Strategy output DataFrame does not "
                    "contain a 'Signal' column."
                )

        elif isinstance(
            baseline_output,
            pd.Series,
        ):

            baseline_signal = (
                baseline_output.to_numpy()
            )

        else:

            baseline_signal = (
                np.asarray(baseline_output)
            )

        # Make sure signal length matches test data.
        if len(baseline_signal) != len(test):

            raise ValueError(
                "Strategy signal length does not "
                "match test data length. "
                f"Signals: {len(baseline_signal)}, "
                f"Test rows: {len(test)}"
            )

        baseline_signal = pd.Series(
            baseline_signal,
            index=test.index,
            dtype=float,
        )

        # ------------------------------------------
        # ML probability
        # ------------------------------------------

        probability = model.predict_probability(
            test[FEATURE_COLUMNS]
        )

        probability = pd.Series(
            probability,
            index=test.index,
            dtype=float,
        )

        # ------------------------------------------
        # ML filter
        # ------------------------------------------

        ml_signal = (
            baseline_signal
            * (probability >= threshold)
        )

        ml_signal = ml_signal.astype(float)

        # ------------------------------------------
        # Forward returns
        # ------------------------------------------

        forward_returns = (
            calculate_forward_returns(test)
        )

        # ------------------------------------------
        # Strategy returns
        # ------------------------------------------

        baseline_returns = (
            baseline_signal
            * forward_returns
        )

        ml_returns = (
            ml_signal
            * forward_returns
        )

        # ------------------------------------------
        # Store out-of-sample results
        # ------------------------------------------

        window_results = pd.DataFrame(
            {
                "Date": test[
                    "Date"
                ].values,

                "Window": window_number,

                "Close": test[
                    "Close"
                ].values,

                "Baseline_Signal": (
                    baseline_signal.values
                ),

                "Probability": (
                    probability.values
                ),

                "ML_Signal": (
                    ml_signal.values
                ),

                "Forward_Return": (
                    forward_returns.values
                ),

                "Baseline_Return": (
                    baseline_returns.values
                ),

                "ML_Return": (
                    ml_returns.values
                ),
            }
        )

        all_results.append(
            window_results
        )

    if not all_results:

        raise ValueError(
            "No walk-forward results generated."
        )

    return pd.concat(
        all_results,
        ignore_index=True,
    )


def evaluate_strategy_windows(
    results: pd.DataFrame,
) -> pd.DataFrame:
    """
    Calculate performance metrics independently
    for each walk-forward test window.
    """

    window_results = []

    for window, group in results.groupby(
        "Window"
    ):

        baseline_metrics = (
            calculate_strategy_metrics(
                group["Baseline_Return"],
                group["Baseline_Signal"],
            )
        )

        ml_metrics = (
            calculate_strategy_metrics(
                group["ML_Return"],
                group["ML_Signal"],
            )
        )

        window_results.append(
            {
                "Window": window,

                "Baseline_Return": (
                    baseline_metrics[
                        "total_return"
                    ]
                ),

                "ML_Return": (
                    ml_metrics[
                        "total_return"
                    ]
                ),

                "Baseline_Sharpe": (
                    baseline_metrics[
                        "sharpe_ratio"
                    ]
                ),

                "ML_Sharpe": (
                    ml_metrics[
                        "sharpe_ratio"
                    ]
                ),

                "Baseline_Drawdown": (
                    baseline_metrics[
                        "max_drawdown"
                    ]
                ),

                "ML_Drawdown": (
                    ml_metrics[
                        "max_drawdown"
                    ]
                ),

                "Baseline_Trades": (
                    baseline_metrics[
                        "trades"
                    ]
                ),

                "ML_Trades": (
                    ml_metrics[
                        "trades"
                    ]
                ),
            }
        )

    return pd.DataFrame(
        window_results
    )


def print_strategy_walk_forward_report(
    results: pd.DataFrame,
) -> None:
    """
    Print complete walk-forward strategy report.
    """

    window_results = (
        evaluate_strategy_windows(
            results
        )
    )

    print()

    print(
        "WALK-FORWARD STRATEGY EVALUATION"
    )

    print(
        "=" * 75
    )

    print(
        "WINDOW RESULTS"
    )

    print(
        window_results.to_string(
            index=False
        )
    )

    print()

    print(
        "AGGREGATE RESULTS"
    )

    print(
        "=" * 75
    )

    baseline_returns = results[
        "Baseline_Return"
    ]

    ml_returns = results[
        "ML_Return"
    ]

    baseline_equity = (
        1.0 + baseline_returns
    ).cumprod()

    ml_equity = (
        1.0 + ml_returns
    ).cumprod()

    baseline_total_return = (
        baseline_equity.iloc[-1] - 1.0
    )

    ml_total_return = (
        ml_equity.iloc[-1] - 1.0
    )

    baseline_volatility = (
        baseline_returns.std()
    )

    ml_volatility = (
        ml_returns.std()
    )

    if baseline_volatility > 0:
        baseline_sharpe = (
            baseline_returns.mean()
            / baseline_volatility
            * np.sqrt(252)
        )
    else:
        baseline_sharpe = 0.0

    if ml_volatility > 0:
        ml_sharpe = (
            ml_returns.mean()
            / ml_volatility
            * np.sqrt(252)
        )
    else:
        ml_sharpe = 0.0

    baseline_drawdown = (
        baseline_equity
        / baseline_equity.cummax()
        - 1.0
    ).min()

    ml_drawdown = (
        ml_equity
        / ml_equity.cummax()
        - 1.0
    ).min()

    print(
        f"Baseline Return:       "
        f"{baseline_total_return:.6f}"
    )

    print(
        f"ML Filter Return:      "
        f"{ml_total_return:.6f}"
    )

    print()

    print(
        f"Baseline Sharpe:       "
        f"{baseline_sharpe:.6f}"
    )

    print(
        f"ML Filter Sharpe:      "
        f"{ml_sharpe:.6f}"
    )

    print()

    print(
        f"Baseline Max Drawdown: "
        f"{baseline_drawdown:.6f}"
    )

    print(
        f"ML Filter Max Drawdown:"
        f" {ml_drawdown:.6f}"
    )

    print()

    print(
        f"Return Difference:     "
        f"{ml_total_return - baseline_total_return:.6f}"
    )

    print(
        f"Sharpe Difference:     "
        f"{ml_sharpe - baseline_sharpe:.6f}"
    )

    print(
        f"Drawdown Difference:   "
        f"{ml_drawdown - baseline_drawdown:.6f}"
    )