import pandas as pd

from src.backtest.metrics import calculate_metrics


class PortfolioEngine:
    """Simulate a multi-asset portfolio from target weights."""

    def __init__(
        self,
        transaction_cost: float = 0.001,
    ):
        """
        Parameters
        ----------
        transaction_cost:
            Cost applied to portfolio turnover.
            0.001 = 0.10%.
        """

        if transaction_cost < 0:
            raise ValueError(
                "Transaction cost cannot be negative."
            )

        self.transaction_cost = transaction_cost

    def run(
        self,
        returns: pd.DataFrame,
        weights: pd.DataFrame,
    ) -> tuple[pd.DataFrame, dict]:
        """
        Run portfolio simulation.

        Target weights at day t are applied to returns
        at day t+1.
        """

        if not returns.index.equals(weights.index):
            raise ValueError(
                "Returns and weights must have identical indices."
            )

        if not returns.columns.equals(weights.columns):
            raise ValueError(
                "Returns and weights must have identical columns."
            )

        returns = returns.copy()
        weights = weights.copy()

        # --------------------------------------------------
        # 1. Validate weights
        # --------------------------------------------------

        if (weights < 0).any().any():
            raise ValueError(
                "Negative portfolio weights are not allowed."
            )

        exposure = weights.sum(axis=1)

        if (exposure > 1 + 1e-8).any():
            raise ValueError(
                "Portfolio exposure exceeds 100%."
            )

        # --------------------------------------------------
        # 2. Shift weights to prevent look-ahead bias
        # --------------------------------------------------

        position_weights = weights.shift(1).fillna(0)

        # --------------------------------------------------
        # 3. Calculate asset-level contribution
        # --------------------------------------------------

        asset_returns = (
            position_weights * returns
        )

        portfolio_gross_return = (
            asset_returns.sum(axis=1)
        )

        # --------------------------------------------------
        # 4. Calculate turnover
        # --------------------------------------------------

        turnover = (
            weights
            .diff()
            .abs()
            .sum(axis=1)
            .fillna(0)
        )

        # --------------------------------------------------
        # 5. Transaction costs
        # --------------------------------------------------

        transaction_costs = (
            turnover
            * self.transaction_cost
        )

        # --------------------------------------------------
        # 6. Net portfolio return
        # --------------------------------------------------

        portfolio_return = (
            portfolio_gross_return
            - transaction_costs
        )

        # --------------------------------------------------
        # 7. Portfolio equity curve
        # --------------------------------------------------

        equity = (
            1 + portfolio_return.fillna(0)
        ).cumprod()

        result = pd.DataFrame(
            {
                "Gross_Return": portfolio_gross_return,
                "Turnover": turnover,
                "Transaction_Cost": transaction_costs,
                "Net_Return": portfolio_return,
                "Equity": equity,
            },
            index=returns.index,
        )

        # --------------------------------------------------
        # 8. Performance metrics
        # --------------------------------------------------

        trades = int(
            (turnover > 0).sum()
        )

        metrics = calculate_metrics(
            portfolio_return.dropna(),
            trades,
        )

        metrics["average_turnover"] = float(
            turnover.mean()
        )

        metrics["final_equity"] = float(
            equity.iloc[-1]
        )

        return result, metrics