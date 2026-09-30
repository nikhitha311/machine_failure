"""
OptiForge 2026 - Submission Script
Team: code crafters (OPT-26-3407)
Track: Machine Learning & AI
Project: Robust Machine Failure Prediction Under Operating Condition Shift
"""

from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score


RANDOM_STATE = 42
N_ESTIMATORS = 200
TEST_SIZE = 0.2


def build_model(n_estimators=N_ESTIMATORS, random_state=RANDOM_STATE):
    """Deterministic Random Forest with balanced class weights."""
    return RandomForestClassifier(
        n_estimators=n_estimators,
        class_weight="balanced",
        random_state=random_state,
        n_jobs=-1,
    )


def preprocess(df):
    """Drop IDs + failure subtypes, encode Type, coerce numerics."""
    drop_cols = [
        c for c in df.columns
        if c.lower() in ("udi", "product id", "product_id",
                         "twf", "hdf", "pwf", "osf", "rnf")
    ]
    df = df.drop(columns=drop_cols, errors="ignore")

    if "Type" in df.columns:
        df["Type"] = (
            df["Type"].astype(str).str.strip().str.upper()
            .map({"L": 0, "M": 1, "H": 2})
            .fillna(0)
            .astype(int)
        )

    for c in df.columns:
        if "failure" not in c.lower():
            df[c] = pd.to_numeric(df[c], errors="coerce")

    num_cols = df.select_dtypes(include=[np.number]).columns
    for c in num_cols:
        if df[c].isna().any():
            df[c] = df[c].fillna(df[c].median())

    return df


def create_shifted_dataset(X, random_state=RANDOM_STATE):
    """Simulate realistic operating-condition shift on sensor features."""
    rng = np.random.RandomState(random_state)
    Xs = X.copy()

    sensor_shifts = {
        "air temperature": (0.05, 0.10),
        "process temperature": (0.04, 0.06),
        "torque": (0.08, 0.12),
        "rotational speed": (-0.07, -0.03),
        "tool wear": (0.08, 0.12),
    }

    for col in Xs.columns:
        col_lower = col.lower()
        for key, (lo, hi) in sensor_shifts.items():
            if key in col_lower:
                Xs[col] = Xs[col] * (1 + rng.uniform(lo, hi, size=len(Xs)))
                break

    return Xs


def solve(X_train, y_train, X_test):
    """Entry point for OptiForge automated evaluation."""
    if not isinstance(X_train, pd.DataFrame):
        X_train = pd.DataFrame(X_train)
    if not isinstance(X_test, pd.DataFrame):
        X_test = pd.DataFrame(X_test)

    common = [c for c in X_train.columns if c in X_test.columns]
    if not common:
        raise ValueError("No common features between train and test")

    X_train = X_train[common].fillna(0)
    X_test = X_test[common].fillna(0)

    X_aug = pd.concat(
        [X_train, create_shifted_dataset(X_train)],
        axis=0,
    ).reset_index(drop=True)
    y_aug = np.concatenate([np.asarray(y_train), np.asarray(y_train)])

    model = build_model()
    model.fit(X_aug.values, y_aug)
    return model.predict(X_test.values)


def main():
    """Local test harness — run with AI4I dataset."""
    df = pd.read_csv("data/ai4i2020.csv")
    df = preprocess(df)

    target_col = [c for c in df.columns if "failure" in c.lower()][0]
    y = df[target_col].astype(int).values
    X = df.drop(columns=[target_col])

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=TEST_SIZE,
        random_state=RANDOM_STATE, stratify=y,
    )

    print("=" * 55)
    print("ROBUST MACHINE FAILURE PREDICTION")
    print("=" * 55)
    print(f"Dataset: {len(X)} samples, {X.shape[1]} features")
    print(f"Failure ratio: {y.mean():.4f}")

    baseline = build_model()
    baseline.fit(X_tr.values, y_tr)
    y_pred = baseline.predict(X_te.values)
    y_proba = baseline.predict_proba(X_te.values)[:, 1]

    print("\n===== NORMAL PERFORMANCE (Baseline) =====")
    print(f"Precision : {precision_score(y_te, y_pred, zero_division=0):.4f}")
    print(f"Recall    : {recall_score(y_te, y_pred, zero_division=0):.4f}")
    print(f"F1 Score  : {f1_score(y_te, y_pred, zero_division=0):.4f}")
    print(f"ROC-AUC   : {roc_auc_score(y_te, y_proba):.4f}")

    X_te_shifted = create_shifted_dataset(X_te)
    y_pred_s = baseline.predict(X_te_shifted.values)
    y_proba_s = baseline.predict_proba(X_te_shifted.values)[:, 1]

    print("\n===== SHIFTED CONDITION (Baseline) =====")
    print(f"Precision : {precision_score(y_te, y_pred_s, zero_division=0):.4f}")
    print(f"Recall    : {recall_score(y_te, y_pred_s, zero_division=0):.4f}")
    print(f"F1 Score  : {f1_score(y_te, y_pred_s, zero_division=0):.4f}")
    print(f"ROC-AUC   : {roc_auc_score(y_te, y_proba_s):.4f}")

    y_pred_rs = solve(X_tr, y_tr, X_te_shifted)

    print("\n===== SHIFTED CONDITION (Robust Model) =====")
    print(f"Precision : {precision_score(y_te, y_pred_rs, zero_division=0):.4f}")
    print(f"Recall    : {recall_score(y_te, y_pred_rs, zero_division=0):.4f}")
    print(f"F1 Score  : {f1_score(y_te, y_pred_rs, zero_division=0):.4f}")

    print("\n" + "=" * 55)
    print("PIPELINE COMPLETE")
    print("=" * 55)


if __name__ == "__main__":
    main()