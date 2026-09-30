"""Singleton model loader — loads once, caches in memory."""

import os
import joblib
from fastapi import HTTPException

_MODEL = None
_FEATURE_NAMES = None


def get_model():
    """Lazy-load model once, cache in memory."""
    global _MODEL, _FEATURE_NAMES

    if _MODEL is not None:
        return _MODEL, _FEATURE_NAMES

    model_path = os.getenv("MODEL_PATH", "models/rf_robust.pkl")
    features_path = os.getenv("FEATURE_NAMES_PATH", "models/feature_names.pkl")

    if not os.path.exists(model_path):
        raise HTTPException(
            status_code=503,
            detail=f"Model not found at {model_path}. Run: python scripts/train_and_save.py",
        )

    _MODEL = joblib.load(model_path)

    if os.path.exists(features_path):
        _FEATURE_NAMES = joblib.load(features_path)

    return _MODEL, _FEATURE_NAMES