import pandas as pd

from .base import AlphaStrategy


class MomentumStrategy(AlphaStrategy):
    """Simple multi-horizon momentum strategy."""

    name = "Momentum"

    def __init__(
        self,
        short_window: int = 5,
        long_window: int = 20,
    ):
        self.short_window = short_window
        self.long_window = long_window

    def generate_signal(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate long/neutral signals using price momentum."""

        required_columns = [
            "Close",
            "Return_5D",
            "Return_20D",
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

        bullish = (
            (result["Return_5D"] > 0)
            & (result["Return_20D"] > 0)
        )

        result.loc[bullish, "Signal"] = 1

        return result