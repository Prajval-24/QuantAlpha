import pandas as pd

from .metrics import calculate_metrics


class BacktestEngine:
    """Vectorized backtesting engine for QuantAlpha strategies."""

    def __init__(self, transaction_cost: float = 0.001):
        """
        Parameters
        ----------
        transaction_cost:
            Transaction cost applied when the position changes.
            0.001 = 0.10%.
        """

        if transaction_cost < 0:
            raise ValueError(
                "Transaction cost cannot be negative."
            )

        self.transaction_cost = transaction_cost

    def run(
        self,
        data: pd.DataFrame,
    ) -> tuple[pd.DataFrame, dict]:
        """
        Run a backtest using precomputed trading signals.

        Signal[t] determines Position[t+1] to avoid
        look-ahead bias.
        """

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

        result = data.copy()

        # Ensure chronological order.
        result = (
            result
            .sort_values("Date")
            .reset_index(drop=True)
        )

        # --------------------------------------------------
        # 1. Market returns
        # --------------------------------------------------

        result["Market_Return"] = (
            result["Close"].pct_change()
        )

        # --------------------------------------------------
        # 2. Shift signal to avoid look-ahead bias
        # --------------------------------------------------

        result["Position"] = (
            result["Signal"].shift(1)
        )

        result["Position"] = (
            result["Position"].fillna(0)
        )

        # --------------------------------------------------
        # 3. Strategy returns before transaction costs
        # --------------------------------------------------

        result["Strategy_Return"] = (
            result["Position"]
            * result["Market_Return"]
        )

        # --------------------------------------------------
        # 4. Detect position changes
        # --------------------------------------------------

        result["Position_Change"] = (
            result["Position"]
            .diff()
            .abs()
            .fillna(0)
        )

        # --------------------------------------------------
        # 5. Transaction costs
        # --------------------------------------------------

        result["Transaction_Cost"] = (
            result["Position_Change"]
            * self.transaction_cost
        )

        # --------------------------------------------------
        # 6. Net strategy return
        # --------------------------------------------------

        result["Net_Return"] = (
            result["Strategy_Return"]
            - result["Transaction_Cost"]
        )

        # --------------------------------------------------
        # 7. Strategy equity curve
        # --------------------------------------------------

        result["Equity"] = (
            1 + result["Net_Return"].fillna(0)
        ).cumprod()

        # --------------------------------------------------
        # 8. Buy-and-hold benchmark
        # --------------------------------------------------

        result["Benchmark_Equity"] = (
            1 + result["Market_Return"].fillna(0)
        ).cumprod()

        # --------------------------------------------------
        # 9. Count position changes
        # --------------------------------------------------

        trades = int(
            result["Position_Change"].sum()
        )

        # --------------------------------------------------
        # 10. Strategy performance metrics
        # --------------------------------------------------

        strategy_metrics = calculate_metrics(
            result["Net_Return"].dropna(),
            trades,
        )

        # --------------------------------------------------
        # 11. Benchmark performance metrics
        # --------------------------------------------------

        benchmark_metrics = calculate_metrics(
            result["Market_Return"].dropna(),
            0,
        )

        # --------------------------------------------------
        # 12. Add benchmark comparison
        # --------------------------------------------------

        strategy_metrics["benchmark_total_return"] = (
            benchmark_metrics["total_return"]
        )

        strategy_metrics["benchmark_annualized_return"] = (
            benchmark_metrics["annualized_return"]
        )

        strategy_metrics["benchmark_sharpe_ratio"] = (
            benchmark_metrics["sharpe_ratio"]
        )

        strategy_metrics["benchmark_max_drawdown"] = (
            benchmark_metrics["max_drawdown"]
        )

        # --------------------------------------------------
        # 13. Excess return
        # --------------------------------------------------

        strategy_metrics["excess_return"] = (
            strategy_metrics["total_return"]
            - strategy_metrics["benchmark_total_return"]
        )

        return result, strategy_metrics