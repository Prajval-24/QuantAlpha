import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from src.alphas.base import AlphaStrategy


class MachineLearningAlphaStrategy(AlphaStrategy):
    """
    Supervised learning alpha strategy using Scikit-learn 
    to classify next-day direction (up/down) and generate trading signals.
    """
    def __init__(self, lookback_train: int = 252, n_estimators: int = 50):
        self.lookback_train = lookback_train
        self.n_estimators = n_estimators
        self.name = "ml_supervised"

    def generate_signal(self, data: pd.DataFrame) -> pd.DataFrame:
        df = data.copy()
        
        if "Returns" not in df.columns:
            df["Returns"] = df["Close"].pct_change()
        if "SMA_20" not in df.columns:
            df["SMA_20"] = df["Close"].rolling(20).mean()
        if "Volatility_20" not in df.columns:
            df["Volatility_20"] = df["Returns"].rolling(20).std()
            
        df["Target"] = (df["Returns"].shift(-1) > 0).astype(int)
        
        feature_cols = ["Returns", "SMA_20", "Volatility_20"]
        df = df.dropna()
        
        signals = pd.Series(0.0, index=df.index)
        model = RandomForestClassifier(n_estimators=self.n_estimators, random_state=42)
        
        min_train_size = min(self.lookback_train, len(df) // 2)
        
        if len(df) > min_train_size + 10:
            for i in range(min_train_size, len(df)):
                train_data = df.iloc[i - min_train_size : i]
                X_train = train_data[feature_cols]
                y_train = train_data["Target"]
                
                model.fit(X_train, y_train)
                
                current_features = df.iloc[[i]][feature_cols]
                pred = model.predict(current_features)[0]
                
                signals.iloc[i] = float(pred)
                
        df["Signal"] = signals
        return df