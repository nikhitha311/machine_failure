

import time
import numpy as np
import pandas as pd
from sklearn.metrics import (
    precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix,
)


def compute_metrics(y_true, y_pred, y_proba=None):
    """Compute precision/recall/F1/AUC + confusion matrix."""
    metrics = {
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
    }
    if y_proba is not None and len(np.unique(y_true)) > 1:
        metrics["roc_auc"] = roc_auc_score(y_true, y_proba)
    else:
        metrics["roc_auc"] = float("nan")
    return metrics


def measure_inference_time(model, X, repeat=5):
    """Average inference time in milliseconds."""
    _ = model.predict_proba(X)
    times = []
    for _ in range(repeat):
        t0 = time.perf_counter()
        _ = model.predict_proba(X)
        times.append(time.perf_counter() - t0)
    return float(np.mean(times) * 1000.0)


def create_shifted_dataset(X, sensor_cols, random_state=42):
    """Controlled operating-condition shift."""
    rng = np.random.RandomState(random_state)
    Xs = X.copy()

    c = sensor_cols.get("air_temp")
    if c in Xs.columns:
        Xs[c] = Xs[c] * (1 + rng.uniform(0.05, 0.10, size=len(Xs)))

    c = sensor_cols.get("process_temp")
    if c in Xs.columns:
        Xs[c] = Xs[c] * (1 + rng.uniform(0.04, 0.06, size=len(Xs)))

    c = sensor_cols.get("torque")
    if c in Xs.columns:
        Xs[c] = Xs[c] * (1 + rng.uniform(0.08, 0.12, size=len(Xs)))

    c = sensor_cols.get("rpm")
    if c in Xs.columns:
        Xs[c] = Xs[c] * (1 - rng.uniform(0.03, 0.07, size=len(Xs)))

    c = sensor_cols.get("tool_wear")
    if c in Xs.columns:
        Xs[c] = Xs[c] * (1 + rng.uniform(0.08, 0.12, size=len(Xs)))

    return Xs


def risk_level(prob):
    """Map failure probability to human-readable risk."""
    if prob < 0.30:
        return "NORMAL"
    elif prob < 0.70:
        return "WARNING"
    else:
        return "CRITICAL"


def print_metrics_block(title, metrics, runtime_ms=None):
    print(f"\n===== {title} =====")
    print(f"Precision : {metrics['precision']:.4f}")
    print(f"Recall    : {metrics['recall']:.4f}")
    print(f"F1 Score  : {metrics['f1']:.4f}")
    if not np.isnan(metrics.get("roc_auc", float("nan"))):
        print(f"ROC-AUC   : {metrics['roc_auc']:.4f}")
    if runtime_ms is not None:
        print(f"Runtime   : {runtime_ms:.2f} ms")
    print(f"Confusion : {metrics['confusion_matrix']}")