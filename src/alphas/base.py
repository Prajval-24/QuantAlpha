from abc import ABC, abstractmethod

import pandas as pd


class AlphaStrategy(ABC):
    """Base interface for all QuantAlpha trading strategies."""

    name: str = "Base Strategy"

    @abstractmethod
    def generate_signal(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Generate trading signals from market data.

        The returned DataFrame must contain a `Signal` column:
            1  = Long
            0  = No position
           -1  = Short
        """
        raise NotImplementedError