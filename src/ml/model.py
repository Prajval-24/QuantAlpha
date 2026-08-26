import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .features import FEATURE_COLUMNS


class MLAlphaModel:
    """
    Logistic Regression model for next-day return direction.
    """

    def __init__(
        self,
        C: float = 1.0,
        max_iter: int = 1000,
    ):
        """
        Parameters
        ----------
        C:
            Inverse regularization strength.

        max_iter:
            Maximum number of optimization iterations.
        """

        if C <= 0:
            raise ValueError(
                "C must be greater than zero."
            )

        self.model = Pipeline(
            [
                (
                    "scaler",
                    StandardScaler(),
                ),
                (
                    "classifier",
                    LogisticRegression(
                        C=C,
                        max_iter=max_iter,
                        random_state=42,
                    ),
                ),
            ]
        )

        self.is_fitted = False

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
    ) -> "MLAlphaModel":
        """
        Fit the model on historical training data.
        """

        missing = [
            column
            for column in FEATURE_COLUMNS
            if column not in X.columns
        ]

        if missing:
            raise ValueError(
                f"Missing required features: {missing}"
            )

        if len(X) != len(y):
            raise ValueError(
                "X and y must have the same number of rows."
            )

        self.model.fit(
            X[FEATURE_COLUMNS],
            y,
        )

        self.is_fitted = True

        return self

    def predict_probability(
        self,
        X: pd.DataFrame,
    ) -> pd.Series:
        """
        Predict probability of a positive next-day return.
        """

        if not self.is_fitted:
            raise RuntimeError(
                "Model must be fitted before prediction."
            )

        probabilities = (
            self.model
            .predict_proba(
                X[FEATURE_COLUMNS]
            )[:, 1]
        )

        return pd.Series(
            probabilities,
            index=X.index,
            name="Probability",
        )

    def predict(
        self,
        X: pd.DataFrame,
    ) -> pd.Series:
        """
        Predict binary next-day direction.
        """

        if not self.is_fitted:
            raise RuntimeError(
                "Model must be fitted before prediction."
            )

        predictions = (
            self.model
            .predict(
                X[FEATURE_COLUMNS]
            )
        )

        return pd.Series(
            predictions,
            index=X.index,
            name="Prediction",
        )

    def coefficients(
        self,
    ) -> pd.Series:
        """
        Return standardized model coefficients.
        """

        if not self.is_fitted:
            raise RuntimeError(
                "Model must be fitted before inspection."
            )

        classifier = (
            self.model
            .named_steps["classifier"]
        )

        return pd.Series(
            classifier.coef_[0],
            index=FEATURE_COLUMNS,
            name="Coefficient",
        )