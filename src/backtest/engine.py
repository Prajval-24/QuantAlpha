import pandas as pd

from .metrics import calculate_metrics


class BacktestEngine:
    """Vectorized single-asset backtesting engine for QuantAlpha."""

    def __init__(
        self,
        transaction_cost: float = 0.001,
    ):
        """
        Parameters
        ----------
        transaction_cost:
            Proportional transaction cost applied to
            absolute position changes.

            Example:
                0.001 = 0.10%
        """

        if transaction_cost < 0:
            raise ValueError(
                "Transaction cost cannot be negative."
            )

        self.transaction_cost = float(
            transaction_cost
        )

    def run(
        self,
        data: pd.DataFrame,
    ) -> tuple[pd.DataFrame, dict]:
        """
        Run a single-asset backtest using precomputed signals.

        Signal[t] determines Position[t+1].
        This prevents look-ahead bias when signals are
        generated using information available at t.

        BacktestEngine intentionally supports only one asset.
        Multi-asset data must be handled by PortfolioEngine.
        """

        # --------------------------------------------------
        # 1. Validate input structure
        # --------------------------------------------------

        required_columns = [
            "Date",
            "Close",
            "Signal",
        ]

        missing = [
            column
            for column in required_columns
            if column not in data.columns
        ]

        if missing:
            raise ValueError(
                f"Missing required columns: {missing}"
            )

        if data.empty:
            raise ValueError(
                "Backtest data cannot be empty."
            )

        # --------------------------------------------------
        # 2. Reject multi-asset input
        # --------------------------------------------------

        if "Symbol" in data.columns:

            symbols = (
                data["Symbol"]
                .dropna()
                .unique()
            )

            if len(symbols) > 1:
                raise ValueError(
                    "BacktestEngine only supports "
                    "single-asset data. Use PortfolioEngine "
                    "for multi-asset portfolios."
                )

        # --------------------------------------------------
        # 3. Validate Date
        # --------------------------------------------------

        if data["Date"].isna().any():
            raise ValueError(
                "Date column contains missing values."
            )

        if data["Date"].duplicated().any():
            raise ValueError(
                "Duplicate dates found in backtest data."
            )

        # --------------------------------------------------
        # 4. Validate Close
        # --------------------------------------------------

        if data["Close"].isna().any():
            raise ValueError(
                "Close column contains missing values."
            )

        if (data["Close"] <= 0).any():
            raise ValueError(
                "Close prices must be strictly positive."
            )

        # --------------------------------------------------
        # 5. Validate Signal
        # --------------------------------------------------

        if data["Signal"].isna().any():
            raise ValueError(
                "Signal column contains missing values."
            )

        valid_signals = {-1, 0, 1}

        invalid_signals = set(
            data["Signal"].unique()
        ) - valid_signals

        if invalid_signals:
            raise ValueError(
                "Signal must contain only "
                "-1, 0, or 1. "
                f"Invalid values: {sorted(invalid_signals)}"
            )

        # --------------------------------------------------
        # 6. Copy and sort chronologically
        # --------------------------------------------------

        result = (
            data.copy()
            .sort_values("Date")
            .reset_index(drop=True)
        )

        # --------------------------------------------------
        # 7. Market returns
        # --------------------------------------------------
        #
        # Return[t] is the price return from t-1 to t.
        #

        result["Market_Return"] = (
            result["Close"]
            .pct_change()
        )

        # --------------------------------------------------
        # 8. Shift signal
        # --------------------------------------------------
        #
        # Signal[t] becomes Position[t+1].
        #
        # Therefore a signal generated after the close
        # on day t cannot earn the return from t-1 to t.
        #

        result["Position"] = (
            result["Signal"]
            .shift(1)
            .fillna(0.0)
        )

        # --------------------------------------------------
        # 9. Strategy return before costs
        # --------------------------------------------------

        result["Strategy_Return"] = (
            result["Position"]
            * result["Market_Return"]
        )

        # --------------------------------------------------
        # 10. Position changes / turnover
        # --------------------------------------------------
        #
        # The first position is entered from zero capital
        # exposure, so its full absolute size is turnover.
        #
        # Example:
        #
        #   0 -> +1 = 1 unit
        #   +1 -> 0 = 1 unit
        #   +1 -> -1 = 2 units
        #

        previous_position = (
            result["Position"]
            .shift(1)
            .fillna(0.0)
        )

        result["Position_Change"] = (
            result["Position"]
            - previous_position
        ).abs()

        # --------------------------------------------------
        # 11. Transaction costs
        # --------------------------------------------------

        result["Transaction_Cost"] = (
            result["Position_Change"]
            * self.transaction_cost
        )

        # --------------------------------------------------
        # 12. Net strategy return
        # --------------------------------------------------

        result["Net_Return"] = (
            result["Strategy_Return"]
            - result["Transaction_Cost"]
        )

        # --------------------------------------------------
        # 13. Strategy equity
        # --------------------------------------------------

        result["Equity"] = (
            1.0
            + result["Net_Return"].fillna(0.0)
        ).cumprod()

        # --------------------------------------------------
        # 14. Buy-and-hold benchmark
        # --------------------------------------------------

        result["Benchmark_Equity"] = (
            1.0
            + result["Market_Return"].fillna(0.0)
        ).cumprod()

        # --------------------------------------------------
        # 15. Turnover units
        # --------------------------------------------------
        #
        # This is NOT a round-trip trade count.
        # It measures cumulative absolute position changes.
        #

        turnover_units = float(
            result["Position_Change"].sum()
        )

        # Preserve the existing "trades" metric for
        # compatibility with the rest of the project.
        trades = int(
            turnover_units
        )

        # --------------------------------------------------
        # 16. Strategy metrics
        # --------------------------------------------------

        strategy_returns = (
            result["Net_Return"]
            .dropna()
        )

        strategy_metrics = calculate_metrics(
            strategy_returns,
            trades,
        )

        strategy_metrics["turnover_units"] = (
            turnover_units
        )

        # --------------------------------------------------
        # 17. Benchmark metrics
        # --------------------------------------------------

        benchmark_returns = (
            result["Market_Return"]
            .dropna()
        )

        benchmark_metrics = calculate_metrics(
            benchmark_returns,
            0,
        )

        # --------------------------------------------------
        # 18. Benchmark comparison
        # --------------------------------------------------

        strategy_metrics[
            "benchmark_total_return"
        ] = benchmark_metrics[
            "total_return"
        ]

        strategy_metrics[
            "benchmark_annualized_return"
        ] = benchmark_metrics[
            "annualized_return"
        ]

        strategy_metrics[
            "benchmark_sharpe_ratio"
        ] = benchmark_metrics[
            "sharpe_ratio"
        ]

        strategy_metrics[
            "benchmark_max_drawdown"
        ] = benchmark_metrics[
            "max_drawdown"
        ]

        # --------------------------------------------------
        # 19. Excess return
        # --------------------------------------------------

        strategy_metrics["excess_return"] = (
            strategy_metrics["total_return"]
            - strategy_metrics[
                "benchmark_total_return"
            ]
        )

        return result, strategy_metrics