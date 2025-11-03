from __future__ import annotations

import json
from dataclasses import dataclass

import numpy as np
from sklearn.metrics import auc, confusion_matrix, roc_curve


@dataclass
class EvalReport:
    """Container for evaluation metrics."""

    threshold: float
    accuracy: float
    recall: float
    precision: float
    f1: float
    gmean: float
    roc_auc: float
    tn: int
    fp: int
    fn: int
    tp: int

    def to_json(self, indent: int = 2) -> str:
        """Serialize to JSON string."""
        return json.dumps(
            {
                "threshold": round(self.threshold, 4),
                "accuracy": round(self.accuracy, 4),
                "recall": round(self.recall, 4),
                "precision": round(self.precision, 4),
                "f1": round(self.f1, 4),
                "gmean": round(self.gmean, 4),
                "roc_auc": round(self.roc_auc, 4),
                "confusion_matrix": {
                    "tn": int(self.tn),
                    "fp": int(self.fp),
                    "fn": int(self.fn),
                    "tp": int(self.tp),
                },
            },
            indent=indent,
        )


def compute_metrics(y_true: np.ndarray, y_proba: np.ndarray) -> EvalReport:
    """Compute evaluation metrics with optimal threshold based on G-mean.
    
    Args:
        y_true: True binary labels
        y_proba: Predicted probabilities for positive class
        
    Returns:
        EvalReport with metrics at optimal threshold
    """
    # Compute ROC curve
    fpr, tpr, thresholds = roc_curve(y_true, y_proba)
    roc_auc = auc(fpr, tpr)

    # Find optimal threshold using G-mean
    gmeans = np.sqrt(tpr * (1 - fpr))
    ix = np.argmax(gmeans)
    optimal_threshold = thresholds[ix]
    optimal_gmean = gmeans[ix]

    # Predict with optimal threshold
    y_pred = (y_proba >= optimal_threshold).astype(int)

    # Compute confusion matrix
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    # Compute metrics
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

    return EvalReport(
        threshold=optimal_threshold,
        accuracy=accuracy,
        recall=recall,
        precision=precision,
        f1=f1,
        gmean=optimal_gmean,
        roc_auc=roc_auc,
        tn=tn,
        fp=fp,
        fn=fn,
        tp=tp,
    )
