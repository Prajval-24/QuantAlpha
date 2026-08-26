from .base import AlphaStrategy
from .momentum import MomentumStrategy
from .mean_reversion import MeanReversionStrategy


STRATEGIES = {
    "momentum": MomentumStrategy,
    "mean_reversion": MeanReversionStrategy,
}


def get_strategy(name: str) -> AlphaStrategy:
    """Create a strategy instance by name."""

    if name not in STRATEGIES:
        available = ", ".join(STRATEGIES.keys())

        raise ValueError(
            f"Unknown strategy '{name}'. "
            f"Available strategies: {available}"
        )

    return STRATEGIES[name]()