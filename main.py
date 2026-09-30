"""
==================================================
ROBUST MACHINE FAILURE PREDICTION
OptiForge 2026 - Command-line pipeline
==================================================
"""

import sys
import numpy as np
import pandas as pd

from src.preprocess import (
    load_dataset, clean_and_encode,
    get_feature_target, get_sensor_columns,
    split_data,
)
from src.train import build_model, train_model, build_robust_model
from src.evaluate import (
    compute_metrics, measure_inference_time,
    create_shifted_dataset, print_metrics_block,
)
from src.robustness import robustness_report
from src.predict import predict_one


def main():
    print("=" * 50)
    print("ROBUST MACHINE FAILURE PREDICTION")
    print("=" * 50)

    # ---------------- Load ----------------
    try:
        df = load_dataset("data/ai4i2020.csv")
    except FileNotFoundError as e:
        print(f"\n[ERROR] {e}")
        sys.exit(1)

    df = clean_and_encode(df)
    X, y, feature_names = get_feature_target(df)
    sensor_cols = get_sensor_columns(feature_names)

    print("\nDataset:")
    print(f"Total samples    : {len(X)}")
    print(f"Failure ratio    : {y.mean():.4f}")
    print(f"Features         : {len(feature_names)}")

    # ---------------- Split ----------------
    X_train, X_test, y_train, y_test = split_data(X, y)
    print(f"Training samples : {len(X_train)}")
    print(f"Testing samples  : {len(X_test)}")

    # ---------------- Baseline model ----------------
    print("\nModel: Random Forest (baseline)")
    model = build_model()
    model, train_time = train_model(model, X_train, y_train)
    print(f"Train time       : {train_time:.2f} s")

    # Normal evaluation
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    normal_metrics = compute_metrics(y_test, y_pred, y_proba)
    runtime_ms = measure_inference_time(model, X_test)
    print_metrics_block("NORMAL PERFORMANCE", normal_metrics, runtime_ms)

    # ---------------- Shifted test ----------------
    X_test_shifted = create_shifted_dataset(X_test, sensor_cols)
    y_pred_s = model.predict(X_test_shifted)
    y_proba_s = model.predict_proba(X_test_shifted)[:, 1]
    shifted_metrics = compute_metrics(y_test, y_pred_s, y_proba_s)
    print_metrics_block("SHIFTED CONDITION TEST (Baseline Model)", shifted_metrics)

    drop = (normal_metrics["f1"] - shifted_metrics["f1"]) * 100.0
    print(f"\nF1 Drop (baseline) : {drop:.2f}%")

    # ---------------- Robust model ----------------
    print("\nTraining robust model with augmented data...")
    robust_model = build_robust_model(
        X_train, y_train, feature_names, sensor_cols,
        random_state=42, n_estimators=200, n_augment=1,
    )

    # Evaluate robust model on normal + shifted
    y_pred_rn = robust_model.predict(X_test)
    y_proba_rn = robust_model.predict_proba(X_test)[:, 1]
    robust_normal = compute_metrics(y_test, y_pred_rn, y_proba_rn)

    y_pred_rs = robust_model.predict(X_test_shifted)
    y_proba_rs = robust_model.predict_proba(X_test_shifted)[:, 1]
    robust_shifted = compute_metrics(y_test, y_pred_rs, y_proba_rs)

    print_metrics_block("ROBUST MODEL - NORMAL", robust_normal)
    print_metrics_block("ROBUST MODEL - SHIFTED", robust_shifted)

    comp = robustness_report(shifted_metrics, robust_shifted)
    print(f"\nBaseline F1 (shifted) : {comp['baseline_f1']:.4f}")
    print(f"Robust   F1 (shifted) : {comp['robust_f1']:.4f}")
    print(f"Improvement           : {comp['improvement_pct']:.2f}%")

    # ---------------- Feature Importance ----------------
    print("\n===== FEATURE IMPORTANCE (Robust Model) =====")
    importances = robust_model.feature_importances_
    order = np.argsort(importances)[::-1]
    for rank, i in enumerate(order[:5], 1):
        print(f"{rank}. {feature_names[i]:30s} : {importances[i]:.4f}")

    # ---------------- Sample Prediction ----------------
    print("\n===== SAMPLE PREDICTION (Highest-risk test sample) =====")
    high_risk_idx = int(np.argmax(y_proba_rs))
    sample = X_test.iloc[[high_risk_idx]].copy()
    result = predict_one(
        robust_model, sample, feature_names, sensor_cols, top_k=5
    )
    print(f"Failure Probability: {result['probability']*100:.2f}%")
    print(f"Risk Level         : {result['risk_level']}")
    print("\nExplanation:")
    for line in result["explanation"]:
        print(f"  - {line}")
    print(f"\nRecommendation:\n{result['recommendation']}")

    print("\n" + "=" * 50)
    print("PIPELINE COMPLETE")
    print("=" * 50)


if __name__ == "__main__":
    main()