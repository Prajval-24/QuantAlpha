import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .features import FEATURE_COLUMNS


class MLAlphaModel:
    """
    Logistic Regression model for predicting
    next-day return direction.
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

        if max_iter < 1:
            raise ValueError(
                "max_iter must be at least 1."
            )

        self.C = float(C)
        self.max_iter = int(max_iter)

        self.model = Pipeline(
            [
                (
                    "scaler",
                    StandardScaler(),
                ),
                (
                    "classifier",
                    LogisticRegression(
                        C=self.C,
                        max_iter=self.max_iter,
                        random_state=42,
                    ),
                ),
            ]
        )

        self.is_fitted = False

    def _validate_features(
        self,
        X: pd.DataFrame,
    ) -> None:
        """
        Validate the feature matrix before passing
        it to sklearn.
        """

        if not isinstance(
            X,
            pd.DataFrame,
        ):
            raise TypeError(
                "X must be a pandas DataFrame."
            )

        missing = [
            column
            for column in FEATURE_COLUMNS
            if column not in X.columns
        ]

        if missing:
            raise ValueError(
                f"Missing required features: {missing}"
            )

        values = (
            X[FEATURE_COLUMNS]
            .to_numpy(dtype=float)
        )

        if not np.isfinite(values).all():
            raise ValueError(
                "X contains NaN or infinite values."
            )

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
    ) -> "MLAlphaModel":
        """
        Fit the model using historical training data.
        """

        self._validate_features(
            X
        )

        y = pd.Series(
            y,
            index=X.index,
        )

        if len(X) != len(y):
            raise ValueError(
                "X and y must have the same "
                "number of rows."
            )

        if y.isna().any():
            raise ValueError(
                "Target contains NaN values."
            )

        y_values = (
            y.to_numpy()
        )

        unique_classes = (
            np.unique(y_values)
        )

        if not np.isin(
            unique_classes,
            [0, 1],
        ).all():
            raise ValueError(
                "Target must contain only 0 and 1."
            )

        if len(unique_classes) < 2:
            raise ValueError(
                "Training target must contain "
                "both classes: 0 and 1."
            )

        self.model.fit(
            X[FEATURE_COLUMNS],
            y.astype(int),
        )

        self.is_fitted = True

        return self

    def predict_probability(
        self,
        X: pd.DataFrame,
    ) -> pd.Series:
        """
        Predict probability of a positive
        next-day return.
        """

        if not self.is_fitted:
            raise RuntimeError(
                "Model must be fitted before prediction."
            )

        self._validate_features(
            X
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

        Returns
        -------
        pd.Series
            1 = positive next-day return
            0 = non-positive next-day return
        """

        if not self.is_fitted:
            raise RuntimeError(
                "Model must be fitted before prediction."
            )

        self._validate_features(
            X
        )

        predictions = (
            self.model
            .predict(
                X[FEATURE_COLUMNS]
            )
        )

        return pd.Series(
            predictions.astype(int),
            index=X.index,
            name="Prediction",
        )

    def coefficients(
        self,
    ) -> pd.Series:
        """
        Return coefficients from the logistic
        regression classifier.

        Features are standardized by the pipeline,
        so coefficients are on standardized feature
        units.
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