"""
Data loading and preprocessing for machine failure prediction.
Robust to minor column-name differences in the AI4I dataset.
"""

import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split


ID_COLUMNS = ["udi", "product id", "product_id"]
TARGET_COLUMN = "machine failure"
TYPE_COLUMN = "type"

SENSOR_KEYWORDS = {
    "air_temp": ["air temperature", "air_temp", "airtemp"],
    "process_temp": ["process temperature", "process_temp", "processtemp"],
    "rpm": ["rotational speed", "rotational_speed", "rpm"],
    "torque": ["torque"],
    "tool_wear": ["tool wear", "tool_wear", "toolwear"],
}


def _find_column(df, candidates):
    """Case-insensitive fuzzy column finder."""
    lower_map = {c.lower(): c for c in df.columns}
    for cand in candidates:
        cand_l = cand.lower()
        if cand_l in lower_map:
            return lower_map[cand_l]
        for low, orig in lower_map.items():
            if cand_l in low:
                return orig
    return None


def load_dataset(path="data/ai4i2020.csv"):
    """Load CSV, raise clear error if missing."""
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Dataset not found at '{path}'.\n"
            "Please download the AI4I 2020 Predictive Maintenance Dataset "
            "and place it at data/ai4i2020.csv\n"
            "Kaggle: https://www.kaggle.com/datasets/stephanmatzka/"
            "predictive-maintenance-dataset-ai4i-2020"
        )
    df = pd.read_csv(path)
    df.columns = [c.strip() for c in df.columns]
    return df


def clean_and_encode(df):
    """Drop ID cols, encode Type, coerce numerics, handle missing."""
    drop_cols = []
    for c in df.columns:
        if c.lower() in ID_COLUMNS:
            drop_cols.append(c)
    
    # Drop failure sub-type columns to prevent data leakage
    failure_subtypes = ["twf", "hdf", "pwf", "osf", "rnf"]
    for c in df.columns:
        if c.lower() in failure_subtypes:
            drop_cols.append(c)
    
    df = df.drop(columns=drop_cols, errors="ignore")

    type_col = _find_column(df, [TYPE_COLUMN])
    if type_col is not None:
        mapping = {"L": 0, "M": 1, "H": 2}
        df[type_col] = (
            df[type_col]
            .astype(str)
            .str.strip()
            .str.upper()
            .map(mapping)
            .fillna(0)
            .astype(int)
        )

    for c in df.columns:
        if c.lower() == TARGET_COLUMN:
            continue
        df[c] = pd.to_numeric(df[c], errors="coerce")

    num_cols = df.select_dtypes(include=[np.number]).columns
    for c in num_cols:
        if df[c].isna().any():
            df[c] = df[c].fillna(df[c].median())

    return df


def get_feature_target(df):
    """Split into X (features) and y (target)."""
    target_col = _find_column(df, [TARGET_COLUMN])
    if target_col is None:
        raise ValueError("Target column 'Machine failure' not found.")

    y = df[target_col].astype(int).values
    X = df.drop(columns=[target_col])
    return X, y, list(X.columns)


def get_sensor_columns(feature_names):
    """Return mapping of canonical sensor -> actual column name."""
    mapping = {}
    for canon, keys in SENSOR_KEYWORDS.items():
        for key in keys:
            for f in feature_names:
                if key in f.lower():
                    mapping[canon] = f
                    break
            if canon in mapping:
                break
    return mapping


def split_data(X, y, test_size=0.2, random_state=42):
    """Stratified train/test split."""
    return train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )