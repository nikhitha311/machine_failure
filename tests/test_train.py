"""
Random Forest training for machine failure prediction.
"""

import time
import numpy as np
from sklearn.ensemble import RandomForestClassifier


def build_model(n_estimators=200, random_state=42):
    """Deterministic Random Forest with balanced class weights."""
    return RandomForestClassifier(
        n_estimators=n_estimators,
        class_weight="balanced",
        random_state=random_state,
        n_jobs=-1,
    )


def train_model(model, X_train, y_train):
    """Fit and return (model, training_time_seconds)."""
    t0 = time.perf_counter()
    model.fit(X_train, y_train)
    train_time = time.perf_counter() - t0
    return model, train_time


def build_robust_model(X_train, y_train, feature_names,
                       sensor_cols, random_state=42,
                       n_estimators=200, n_augment=1):
    """
    Robust training: augment training data with realistic,
    controlled sensor perturbations.
    """
    rng = np.random.RandomState(random_state)

    X_aug = [X_train.values]
    y_aug = [y_train]

    for _ in range(n_augment):
        Xp = X_train.copy()

        c = sensor_cols.get("air_temp")
        if c in Xp.columns:
            Xp[c] = Xp[c] * (1 + rng.uniform(0.05, 0.10, size=len(Xp)))

        c = sensor_cols.get("process_temp")
        if c in Xp.columns:
            Xp[c] = Xp[c] * (1 + rng.uniform(0.04, 0.06, size=len(Xp)))

        c = sensor_cols.get("torque")
        if c in Xp.columns:
            Xp[c] = Xp[c] * (1 + rng.uniform(0.08, 0.12, size=len(Xp)))

        c = sensor_cols.get("rpm")
        if c in Xp.columns:
            Xp[c] = Xp[c] * (1 - rng.uniform(0.03, 0.07, size=len(Xp)))

        c = sensor_cols.get("tool_wear")
        if c in Xp.columns:
            Xp[c] = Xp[c] * (1 + rng.uniform(0.08, 0.12, size=len(Xp)))

        X_aug.append(Xp.values)
        y_aug.append(y_train)

    X_aug = np.vstack(X_aug)
    y_aug = np.concatenate(y_aug)

    model = build_model(n_estimators=n_estimators, random_state=random_state)
    model.fit(X_aug, y_aug)
    return model