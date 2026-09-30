"""
Train robust model and save artifacts for API.
Run once before starting the API.
"""

import os
import sys
import joblib

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.preprocess import (
    load_dataset, clean_and_encode, get_feature_target,
    get_sensor_columns, split_data,
)
from src.train import build_robust_model


def main():
    os.makedirs("models", exist_ok=True)

    print("Loading dataset...")
    df = load_dataset("data/ai4i2020.csv")
    df = clean_and_encode(df)

    X, y, feature_names = get_feature_target(df)
    sensor_cols = get_sensor_columns(feature_names)

    print(f"Dataset: {len(X)} samples, {len(feature_names)} features")
    print(f"Failure ratio: {y.mean():.4f}")

    X_train, X_test, y_train, y_test = split_data(X, y)

    print("Training robust model...")
    model = build_robust_model(
        X_train, y_train, feature_names, sensor_cols,
        random_state=42, n_estimators=200, n_augment=2,
    )

    joblib.dump(model, "models/rf_robust.pkl")
    joblib.dump(feature_names, "models/feature_names.pkl")

    print("Model saved to models/rf_robust.pkl")
    print("Feature names saved to models/feature_names.pkl")


if __name__ == "__main__":
    main()
