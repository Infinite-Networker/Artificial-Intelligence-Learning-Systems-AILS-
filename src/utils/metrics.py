"""
AILS Model Metrics Utilities
Precision, Recall, F1-Score, AUC-ROC, and reporting.
Created by Cherry Computer Ltd.
"""

import numpy as np
import logging
from typing import Dict, Optional


def evaluate_model(y_true: np.ndarray,
                   y_pred: np.ndarray,
                   y_prob: Optional[np.ndarray] = None,
                   labels: Optional[list] = None) -> Dict:
    """
    Comprehensive AILS model evaluation.

    Args:
        y_true: Ground truth binary labels.
        y_pred: Predicted binary labels.
        y_prob: Predicted probabilities (for AUC-ROC, optional).
        labels: List of class names for the report.

    Returns:
        Dictionary with all evaluation metrics.
    """
    from sklearn.metrics import (
        classification_report, confusion_matrix,
        precision_score, recall_score,
        f1_score, roc_auc_score, accuracy_score
    )

    logger = logging.getLogger("AILS.Metrics")

    metrics = {
        "accuracy":  round(float(accuracy_score(y_true, y_pred)), 4),
        "precision": round(float(precision_score(y_true, y_pred,
                                                  average="weighted",
                                                  zero_division=0)), 4),
        "recall":    round(float(recall_score(y_true, y_pred,
                                               average="weighted",
                                               zero_division=0)), 4),
        "f1_score":  round(float(f1_score(y_true, y_pred,
                                           average="weighted",
                                           zero_division=0)), 4),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
        "classification_report": classification_report(
            y_true, y_pred, target_names=labels, zero_division=0
        ),
    }

    if y_prob is not None:
        try:
            metrics["auc_roc"] = round(
                float(roc_auc_score(y_true, y_prob)), 4
            )
        except ValueError as e:
            logger.warning(f"AUC-ROC could not be computed: {e}")

    logger.info(
        f"📊 Accuracy={metrics['accuracy']}, "
        f"Precision={metrics['precision']}, "
        f"Recall={metrics['recall']}, "
        f"F1={metrics['f1_score']}"
    )
    return metrics


def print_metrics_report(metrics: Dict) -> None:
    """Pretty-print an evaluation metrics report."""
    print("\n" + "=" * 50)
    print("  AILS MODEL EVALUATION REPORT")
    print("  Cherry Computer Ltd.")
    print("=" * 50)
    for k, v in metrics.items():
        if k == "classification_report":
            print(f"\nClassification Report:\n{v}")
        elif k == "confusion_matrix":
            print(f"\nConfusion Matrix:\n{np.array(v)}")
        else:
            print(f"  {k:<20}: {v}")
    print("=" * 50)
