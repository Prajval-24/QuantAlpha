from fastapi import FastAPI, HTTPException
import pandas as pd
from src.data.loader import load_raw_data
from src.portfolio.research import PortfolioExperimentConfig, run_portfolio_experiment

app = FastAPI(title="QuantAlpha API", version="1.0.0")

@app.get("/")
def read_root():
    return {"message": "Welcome to QuantAlpha ML-Driven Market Analytics API"}

@app.get("/data/{symbol}")
def get_symbol_data(symbol: str, limit: int = 100):
    try:
        df = load_raw_data(symbol)
        if df.empty:
            raise HTTPException(status_code=404, detail=f"No data found for symbol: {symbol}")
        # Return last N rows as JSON
        data_subset = df.tail(limit).to_dict(orient="records")
        return {"symbol": symbol, "rows_returned": len(data_subset), "data": data_subset}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/backtest/run")
def run_backtest_api(symbol_list: list[str], strategy: str = "momentum", method: str = "equal_weight"):
    try:
        config = PortfolioExperimentConfig(
            symbols=symbol_list,
            strategy_name=strategy,
            portfolio_method=method
        )
        _, metrics = run_portfolio_experiment(config)
        return {"status": "success", "metrics": metrics}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))