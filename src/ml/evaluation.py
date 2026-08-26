import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def evaluate_classifier(
    y_true: pd.Series,
    predictions: pd.Series,
    probabilities: pd.Series,
) -> dict:
    """
    Evaluate binary classification performance.
    """

    if not (
        len(y_true)
        == len(predictions)
        == len(probabilities)
    ):
        raise ValueError(
            "y_true, predictions and probabilities "
            "must have the same length."
        )

    accuracy = accuracy_score(
        y_true,
        predictions,
    )

    precision = precision_score(
        y_true,
        predictions,
        zero_division=0,
    )

    recall = recall_score(
        y_true,
        predictions,
        zero_division=0,
    )

    f1 = f1_score(
        y_true,
        predictions,
        zero_division=0,
    )

    roc_auc = roc_auc_score(
        y_true,
        probabilities,
    )

    # Predict the majority class every time.
    baseline_prediction = (
        y_true
        .value_counts()
        .idxmax()
    )

    baseline_accuracy = (
        y_true == baseline_prediction
    ).mean()

    matrix = confusion_matrix(
        y_true,
        predictions,
    )

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "roc_auc": roc_auc,
        "baseline_accuracy": baseline_accuracy,
        "accuracy_vs_baseline": (
            accuracy - baseline_accuracy
        ),
        "confusion_matrix": matrix,
    }


def print_evaluation(
    metrics: dict,
) -> None:
    """
    Print classifier evaluation results.
    """

    print("ML Classification Results")
    print("-" * 40)

    print(
        f"Accuracy:              "
        f"{metrics['accuracy']:.4f}"
    )

    print(
        f"Precision:             "
        f"{metrics['precision']:.4f}"
    )

    print(
        f"Recall:                "
        f"{metrics['recall']:.4f}"
    )

    print(
        f"F1 Score:              "
        f"{metrics['f1_score']:.4f}"
    )

    print(
        f"ROC-AUC:               "
        f"{metrics['roc_auc']:.4f}"
    )

    print(
        f"Baseline Accuracy:     "
        f"{metrics['baseline_accuracy']:.4f}"
    )

    print(
        f"Accuracy vs Baseline:  "
        f"{metrics['accuracy_vs_baseline']:.4f}"
    )

    print()
    print("Confusion Matrix:")
    print(
        metrics["confusion_matrix"]
    )