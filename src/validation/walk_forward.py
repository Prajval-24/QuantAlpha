import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)

from src.ml.model import MLAlphaModel
from src.ml.features import FEATURE_COLUMNS


def generate_walk_forward_splits(
    data: pd.DataFrame,
    train_size: float = 0.60,
    test_size: float = 0.10,
    step_size: float = 0.10,
) -> list[tuple[pd.DataFrame, pd.DataFrame]]:
    """
    Generate expanding-window walk-forward splits.

    CRITICAL FIX: Splitting is strictly based on unique trading
    dates rather than absolute row counts. This guarantees that
    cross-sectional multi-asset data for a single day is never
    split across the train and test sets, which would leak
    future factor information.

    The final incomplete test window is discarded.
    """

    if data.empty:
        raise ValueError(
            "Input data cannot be empty."
        )

    if not 0 < train_size < 1:
        raise ValueError(
            "train_size must be between 0 and 1."
        )

    if not 0 < test_size < 1:
        raise ValueError(
            "test_size must be between 0 and 1."
        )

    if not 0 < step_size < 1:
        raise ValueError(
            "step_size must be between 0 and 1."
        )

    if train_size + test_size > 1:
        raise ValueError(
            "train_size + test_size cannot exceed 1."
        )

    if "Date" not in data.columns:
        raise ValueError(
            "Data must contain a 'Date' column for temporal splitting."
        )

    # ----------------------------------------------------
    # Calculate lengths based on unique chronological dates
    # ----------------------------------------------------

    unique_dates = (
        data["Date"]
        .sort_values()
        .unique()
    )

    n_dates = len(unique_dates)

    train_length = int(
        n_dates * train_size
    )

    test_length = max(
        1,
        int(n_dates * test_size)
    )

    step_length = max(
        1,
        int(n_dates * step_size)
    )

    splits = []

    train_end_idx = train_length

    # ----------------------------------------------------
    # Generate windows
    # ----------------------------------------------------

    while train_end_idx < n_dates:

        test_end_idx = train_end_idx + test_length

        # Do not create an incomplete final test window.
        if test_end_idx > n_dates:
            break

        train_cutoff = unique_dates[train_end_idx - 1]
        test_start_date = unique_dates[train_end_idx]
        test_end_date = unique_dates[test_end_idx - 1]

        train = data[
            data["Date"] <= train_cutoff
        ].copy()

        test = data[
            (data["Date"] >= test_start_date)
            & (data["Date"] <= test_end_date)
        ].copy()

        splits.append(
            (
                train,
                test,
            )
        )

        train_end_idx += step_length

    return splits


def validate_walk_forward_splits(
    splits: list[
        tuple[pd.DataFrame, pd.DataFrame]
    ],
) -> None:
    """
    Validate temporal ordering and absence
    of train/test overlap.
    """

    if not splits:
        raise ValueError(
            "No walk-forward splits provided."
        )

    previous_test_end = None
    previous_train_end = None

    for i, (train, test) in enumerate(
        splits,
        start=1,
    ):

        if train.empty:
            raise AssertionError(
                f"Split {i}: training set is empty."
            )

        if test.empty:
            raise AssertionError(
                f"Split {i}: test set is empty."
            )

        train_last_date = train[
            "Date"
        ].max()

        test_first_date = test[
            "Date"
        ].min()

        assert (
            train_last_date < test_first_date
        ), (
            f"Split {i}: training data overlaps "
            f"with test data."
        )

        if previous_test_end is not None:

            assert (
                test["Date"].min()
                > previous_test_end
            ), (
                f"Split {i}: test windows overlap "
                f"or move backwards."
            )

        if previous_train_end is not None:

            assert (
                train_last_date
                > previous_train_end
            ), (
                f"Split {i}: training window "
                f"did not expand."
            )

        previous_test_end = test[
            "Date"
        ].max()

        previous_train_end = train_last_date


def evaluate_walk_forward(
    data: pd.DataFrame,
) -> pd.DataFrame:
    """
    Train and evaluate the ML model independently
    on every walk-forward window.

    The model is retrained for every window using
    only data available before that window's test
    period.

    Returns
    -------
    DataFrame
        One row per walk-forward window.
    """

    splits = generate_walk_forward_splits(
        data
    )

    validate_walk_forward_splits(
        splits
    )

    results = []

    for window_number, (train, test) in enumerate(
        splits,
        start=1,
    ):

        print(
            f"Evaluating walk-forward "
            f"window {window_number}..."
        )

        # ------------------------------------------
        # Train a completely fresh model
        # ------------------------------------------

        model = MLAlphaModel()

        model.fit(
            train[FEATURE_COLUMNS],
            train["Target"],
        )

        # ------------------------------------------
        # Out-of-sample predictions
        # ------------------------------------------

        probabilities = (
            model.predict_probability(
                test[FEATURE_COLUMNS]
            )
        )

        predictions = model.predict(
            test[FEATURE_COLUMNS]
        )

        actual = test["Target"]

        # ------------------------------------------
        # Classification metrics
        # ------------------------------------------

        accuracy = accuracy_score(
            actual,
            predictions,
        )

        precision = precision_score(
            actual,
            predictions,
            zero_division=0,
        )

        recall = recall_score(
            actual,
            predictions,
            zero_division=0,
        )

        f1 = f1_score(
            actual,
            predictions,
            zero_division=0,
        )

        # ROC-AUC requires both classes to be present.
        if actual.nunique() == 2:

            roc_auc = roc_auc_score(
                actual,
                probabilities,
            )

        else:

            roc_auc = float("nan")

        # ------------------------------------------
        # Baseline majority-class accuracy
        # ------------------------------------------

        majority_class = (
            actual
            .value_counts()
            .idxmax()
        )

        baseline_accuracy = (
            actual == majority_class
        ).mean()

        # ------------------------------------------
        # Store results
        # ------------------------------------------

        results.append(
            {
                "window": window_number,

                "train_start": train[
                    "Date"
                ].min(),

                "train_end": train[
                    "Date"
                ].max(),

                "test_start": test[
                    "Date"
                ].min(),

                "test_end": test[
                    "Date"
                ].max(),

                "train_rows": len(train),

                "test_rows": len(test),

                "accuracy": accuracy,

                "baseline_accuracy": (
                    baseline_accuracy
                ),

                "accuracy_vs_baseline": (
                    accuracy
                    - baseline_accuracy
                ),

                "precision": precision,

                "recall": recall,

                "f1_score": f1,

                "roc_auc": roc_auc,
            }
        )

    return pd.DataFrame(
        results
    )


def summarize_walk_forward(
    results: pd.DataFrame,
) -> dict:
    """
    Aggregate walk-forward ML performance.
    """

    if results.empty:
        raise ValueError(
            "Walk-forward results are empty."
        )

    return {
        "windows": len(results),

        "average_accuracy": (
            results["accuracy"].mean()
        ),

        "average_baseline_accuracy": (
            results[
                "baseline_accuracy"
            ].mean()
        ),

        "average_accuracy_vs_baseline": (
            results[
                "accuracy_vs_baseline"
            ].mean()
        ),

        "average_precision": (
            results["precision"].mean()
        ),

        "average_recall": (
            results["recall"].mean()
        ),

        "average_f1": (
            results["f1_score"].mean()
        ),

        "average_roc_auc": (
            results["roc_auc"].mean()
        ),

        "median_roc_auc": (
            results["roc_auc"].median()
        ),
    }


def print_walk_forward_summary(
    splits: list[
        tuple[pd.DataFrame, pd.DataFrame]
    ],
) -> None:
    """
    Print a readable summary of walk-forward windows.
    """

    print()

    print(
        "WALK-FORWARD VALIDATION WINDOWS"
    )

    print(
        "=" * 75
    )

    for i, (train, test) in enumerate(
        splits,
        start=1,
    ):

        train_start = train[
            "Date"
        ].min()

        train_end = train[
            "Date"
        ].max()

        test_start = test[
            "Date"
        ].min()

        test_end = test[
            "Date"
        ].max()

        print(
            f"Window {i}:"
        )

        print(
            f"  Train: "
            f"{train_start} → {train_end} "
            f"({len(train)} rows)"
        )

        print(
            f"  Test:  "
            f"{test_start} → {test_end} "
            f"({len(test)} rows)"
        )

        print()


def print_model_walk_forward_report(
    results: pd.DataFrame,
) -> None:
    """
    Print detailed walk-forward ML results.
    """

    summary = summarize_walk_forward(
        results
    )

    print()

    print(
        "WALK-FORWARD ML EVALUATION"
    )

    print(
        "=" * 75
    )

    print(
        f"Windows:                    "
        f"{summary['windows']}"
    )

    print(
        f"Average Accuracy:           "
        f"{summary['average_accuracy']:.4f}"
    )

    print(
        f"Average Baseline Accuracy:  "
        f"{summary['average_baseline_accuracy']:.4f}"
    )

    print(
        f"Average Accuracy vs Baseline:"
        f" {summary['average_accuracy_vs_baseline']:.4f}"
    )

    print(
        f"Average Precision:           "
        f"{summary['average_precision']:.4f}"
    )

    print(
        f"Average Recall:              "
        f"{summary['average_recall']:.4f}"
    )

    print(
        f"Average F1:                  "
        f"{summary['average_f1']:.4f}"
    )

    print(
        f"Average ROC-AUC:             "
        f"{summary['average_roc_auc']:.4f}"
    )

    print(
        f"Median ROC-AUC:              "
        f"{summary['median_roc_auc']:.4f}"
    )

    print()

    print(
        "WINDOW RESULTS"
    )

    print(
        results[
            [
                "window",
                "train_rows",
                "test_rows",
                "accuracy",
                "baseline_accuracy",
                "accuracy_vs_baseline",
                "roc_auc",
            ]
        ].to_string(
            index=False
        )
    )
