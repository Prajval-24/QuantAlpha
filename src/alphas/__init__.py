from src.alphas.momentum import MomentumStrategy
from src.alphas.mean_reversion import MeanReversionStrategy
from src.alphas.ml_strategy import MachineLearningAlphaStrategy

def get_strategy(strategy_name: str):
    if strategy_name == "momentum":
        return MomentumStrategy()
    elif strategy_name == "mean_reversion":
        return MeanReversionStrategy()
    elif strategy_name == "ml_supervised":
        return MachineLearningAlphaStrategy()
    else:
        raise ValueError(f"Unknown strategy name: {strategy_name}")