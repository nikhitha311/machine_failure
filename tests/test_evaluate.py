import numpy as np
import pandas as pd
from src.evaluate import compute_metrics, create_shifted_dataset


def test_metrics_perfect():
    y = np.array([0, 0, 1, 1])
    p = np.array([0, 0, 1, 1])
    m = compute_metrics(y, p, p.astype(float))
    assert m["precision"] == 1.0
    assert m["recall"] == 1.0
    assert m["f1"] == 1.0


def test_shifted_dataset_changes_values():
    X = pd.DataFrame({
        "Air temperature [K]": [300.0],
        "Torque [Nm]": [40.0],
        "Rotational speed [rpm]": [1500.0],
        "Tool wear [min]": [100.0],
        "Process temperature [K]": [310.0],
    })
    sensor_cols = {
        "air_temp": "Air temperature [K]",
        "torque": "Torque [Nm]",
        "rpm": "Rotational speed [rpm]",
        "tool_wear": "Tool wear [min]",
        "process_temp": "Process temperature [K]",
    }
    Xs = create_shifted_dataset(X, sensor_cols)
    assert Xs["Torque [Nm]"].iloc[0] != X["Torque [Nm]"].iloc[0]
