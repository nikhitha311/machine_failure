import pandas as pd
from src.preprocess import clean_and_encode, get_feature_target


def test_clean_drops_id_columns():
    df = pd.DataFrame({
        "UDI": [1, 2], "Product ID": ["A", "B"],
        "Type": ["L", "M"], "Torque [Nm]": [10.0, 20.0],
        "Machine failure": [0, 1],
    })
    out = clean_and_encode(df)
    assert "UDI" not in out.columns
    assert "Product ID" not in out.columns


def test_type_encoded():
    df = pd.DataFrame({
        "Type": ["L", "M", "H"],
        "Torque [Nm]": [1.0, 2.0, 3.0],
        "Machine failure": [0, 0, 1],
    })
    out = clean_and_encode(df)
    assert out["Type"].tolist() == [0, 1, 2]


def test_target_split():
    df = pd.DataFrame({
        "Torque [Nm]": [1, 2, 3],
        "Machine failure": [0, 1, 0],
    })
    X, y, names = get_feature_target(df)
    assert "Machine failure" not in X.columns
    assert y.tolist() == [0, 1, 0]
