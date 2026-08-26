import pandas as pd

from .base import AlphaStrategy


class MeanReversionStrategy(AlphaStrategy):
    """Mean-reversion strategy based on distance from moving average."""

    name = "Mean Reversion"

    def __init__(self, threshold: float = 0.05):
        self.threshold = threshold

    def generate_signal(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate signals when price deviates from its moving average."""

        required_columns = [
            "Close",
            "MA_20",
            "MA_Distance",
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

        result["Signal"] = 0

        oversold = result["MA_Distance"] < -self.threshold

        result.loc[oversold, "Signal"] = 1

        return result