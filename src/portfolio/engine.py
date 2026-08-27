import numpy as np
import pandas as pd

from src.backtest.metrics import calculate_metrics


WEIGHT_TOLERANCE = 1e-8


class PortfolioEngine:
    """Simulate a multi-asset long-only portfolio from target weights."""

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

        Target weights generated at day t are applied to
        returns at day t+1.

        Transaction turnover is measured against the
        portfolio's actual holdings immediately before
        rebalancing.

        The initial allocation is measured against a
        100% cash portfolio and therefore incurs
        transaction costs.
        """

        # --------------------------------------------------
        # 0. Validate inputs
        # --------------------------------------------------

        if not isinstance(returns, pd.DataFrame):
            raise TypeError(
                "Returns must be a pandas DataFrame."
            )

        if not isinstance(weights, pd.DataFrame):
            raise TypeError(
                "Weights must be a pandas DataFrame."
            )

        if returns.empty:
            raise ValueError(
                "Returns DataFrame cannot be empty."
            )

        if weights.empty:
            raise ValueError(
                "Weights DataFrame cannot be empty."
            )

        if not returns.index.equals(weights.index):
            raise ValueError(
                "Returns and weights must have identical indices."
            )

        if not returns.columns.equals(weights.columns):
            raise ValueError(
                "Returns and weights must have identical columns."
            )

        if not returns.index.is_monotonic_increasing:
            raise ValueError(
                "Returns index must be sorted chronologically."
            )

        if returns.index.duplicated().any():
            raise ValueError(
                "Returns contain duplicate index values."
            )

        if weights.index.duplicated().any():
            raise ValueError(
                "Weights contain duplicate index values."
            )

        # Work on copies so caller-owned data is never modified.
        returns = returns.astype(float).copy()
        weights = weights.astype(float).copy()

        # --------------------------------------------------
        # 1. Validate numerical inputs
        # --------------------------------------------------

        if not np.isfinite(
            returns.to_numpy()
        ).all():
            raise ValueError(
                "Returns contain NaN or infinite values."
            )

        if not np.isfinite(
            weights.to_numpy()
        ).all():
            raise ValueError(
                "Weights contain NaN or infinite values."
            )

        # Reject materially negative weights.
        if (
            weights < -WEIGHT_TOLERANCE
        ).any().any():
            raise ValueError(
                "Negative portfolio weights are not allowed. "
                "The current portfolio architecture is long-only."
            )

        # Reject exposure above 100%.
        exposure = weights.sum(axis=1)

        if (
            exposure > 1.0 + WEIGHT_TOLERANCE
        ).any():
            raise ValueError(
                "Portfolio exposure exceeds 100%."
            )

        # Remove tiny numerical negative values such as
        # -1e-12 caused by floating-point arithmetic.
        weights = weights.clip(lower=0.0)

        # --------------------------------------------------
        # 2. Position weights used for returns
        # --------------------------------------------------

        # Target weights generated at t are applied during
        # t+1. Therefore:
        #
        #     Position[t] = TargetWeight[t-1]
        #
        # The first row starts from cash.
        position_weights = (
            weights
            .shift(1)
            .fillna(0.0)
        )

        # --------------------------------------------------
        # 3. Calculate gross portfolio returns
        # --------------------------------------------------

        asset_returns = (
            position_weights
            * returns
        )

        portfolio_gross_return = (
            asset_returns.sum(axis=1)
        )

        # --------------------------------------------------
        # 4. Calculate actual portfolio weights after drift
        # --------------------------------------------------

        # The holdings represented by position_weights
        # experience today's asset returns.
        asset_values_after_return = (
            position_weights
            * (1.0 + returns)
        )

        invested_value_after_return = (
            asset_values_after_return.sum(axis=1)
        )

        # Any uninvested capital is treated as cash.
        # Cash earns zero return.
        cash_weight_before_return = (
            1.0
            - position_weights.sum(axis=1)
        )

        cash_value_after_return = (
            cash_weight_before_return
        )

        total_portfolio_value = (
            invested_value_after_return
            + cash_value_after_return
        )

        # Prevent division by zero in pathological cases.
        if (
            total_portfolio_value <= 0
        ).any():
            raise ValueError(
                "Portfolio value became non-positive."
            )

        # Actual asset weights immediately before
        # the next rebalance.
        drifted_weights = (
            asset_values_after_return.div(
                total_portfolio_value,
                axis=0,
            )
        )

        drifted_weights = (
            drifted_weights
            .fillna(0.0)
        )

        # --------------------------------------------------
        # 5. Calculate turnover
        # --------------------------------------------------

        # For t > 0:
        #
        #     turnover[t]
        #         = |target[t] - drifted[t]|
        #
        # For t = 0:
        #
        #     target[0] is purchased from cash.
        #
        # Therefore the first row must use:
        #
        #     turnover[0] = sum(|target[0]|)
        #
        # rather than comparing against drifted_weights[0],
        # which represents zero holdings after the initial
        # no-position return.

        turnover = (
            weights
            - drifted_weights
        ).abs().sum(axis=1)

        turnover.iloc[0] = (
            weights.iloc[0]
            .abs()
            .sum()
        )

        turnover = turnover.fillna(0.0)

        # --------------------------------------------------
        # 6. Transaction costs
        # --------------------------------------------------

        transaction_costs = (
            turnover
            * self.transaction_cost
        )

        # --------------------------------------------------
        # 7. Net portfolio return
        # --------------------------------------------------

        # Exact accounting:
        # End_Wealth = Start_Wealth * (1 + Gross_Return) * (1 - Transaction_Costs)
        portfolio_return = (
            (1.0 + portfolio_gross_return)
            * (1.0 - transaction_costs)
        ) - 1.0

        # A return below -100% is mathematically impossible
        # for this long-only, cash-limited architecture.
        if (
            portfolio_return < -1.0
        ).any():
            raise ValueError(
                "Portfolio return cannot be less than -100%."
            )

        # --------------------------------------------------
        # 8. Portfolio equity curve
        # --------------------------------------------------

        equity = (
            1.0
            + portfolio_return.fillna(0.0)
        ).cumprod()

        if not np.isfinite(
            equity.to_numpy()
        ).all():
            raise ValueError(
                "Portfolio equity contains NaN or infinite values."
            )

        if (
            equity <= 0
        ).any():
            raise ValueError(
                "Portfolio equity became non-positive."
            )

        # --------------------------------------------------
        # 9. Build result dataframe
        # --------------------------------------------------

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
        # 10. Performance metrics
        # --------------------------------------------------

        # Count days on which portfolio turnover occurred.
        #
        # This is an execution-event count, not a traditional
        # round-trip trade count.
        trades = int(
            (
                turnover
                > WEIGHT_TOLERANCE
            ).sum()
        )

        metrics = calculate_metrics(
            portfolio_return.dropna(),
            trades,
        )

        # Additional portfolio-specific metrics.
        metrics["average_turnover"] = float(
            turnover.mean()
        )

        metrics["final_equity"] = float(
            equity.iloc[-1]
        )

        return result, metrics
