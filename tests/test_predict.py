"""Tests for prediction module."""
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from src.predict import predict_one


def _make_two_class_model(n=20, random_state=42):
    """Helper: train a Random Forest on a 2-class dataset."""
    rng = np.random.RandomState(random_state)
    X = pd.DataFrame({
        "Torque [Nm]": rng.uniform(20, 80, n),
        "Tool wear [min]": rng.uniform(0, 250, n),
        "Rotational speed [rpm]": rng.uniform(1200, 1800, n),
    })
    # Binary labels — both classes present
    y = (X["Torque [Nm]"] > 50).astype(int).values
    model = RandomForestClassifier(n_estimators=10, random_state=random_state)
    model.fit(X.values, y)
    return model, X


def test_predict_returns_dict():
    """Predict returns a dict with required keys."""
    model, X = _make_two_class_model()
    result = predict_one(model, X.iloc[[0]], list(X.columns), {}, top_k=2)

    assert "probability" in result
    assert "risk_level" in result
    assert "explanation" in result
    assert "recommendation" in result


def test_predict_probability_range():
    """Probability is between 0 and 1."""
    model, X = _make_two_class_model()
    result = predict_one(model, X.iloc[[0]], list(X.columns), {}, top_k=2)

    assert 0 <= result["probability"] <= 1


def test_predict_risk_levels():
    """Risk level is one of three valid values."""
    model, X = _make_two_class_model()
    result = predict_one(model, X.iloc[[0]], list(X.columns), {}, top_k=2)

    assert result["risk_level"] in ["NORMAL", "WARNING", "CRITICAL"]