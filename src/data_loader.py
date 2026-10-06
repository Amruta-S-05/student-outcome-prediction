"""Loading the data, removing prohibited columns, and splitting into train/test."""
import pandas as pd
from sklearn.model_selection import train_test_split

from config import (
    BINARY_COLUMNS, CATEGORICAL_COLUMNS, DATA_PATH, NUMERIC_COLUMNS,
    PROHIBITED_COLUMNS, RANDOM_STATE, TARGET, TEST_SIZE,
)


def load_data():
    """Read the semicolon-delimited CSV exactly as supplied (only header whitespace is stripped)."""
    df = pd.read_csv(DATA_PATH, sep=";")
    df.columns = df.columns.str.strip()
    return df


def get_features_and_target(df):
    """Separate the target (y) from the predictors (X) and drop the 12 prohibited columns."""
    y = df[TARGET]
    X = df.drop(columns=[TARGET] + PROHIBITED_COLUMNS)

    # Safety checks: no prohibited column left, and every column belongs to exactly one group
    assert not set(PROHIBITED_COLUMNS) & set(X.columns), "Prohibited column found in X!"
    expected = set(CATEGORICAL_COLUMNS + BINARY_COLUMNS + NUMERIC_COLUMNS)
    assert set(X.columns) == expected, "Feature groups in config.py do not match the data."
    return X, y


def get_train_test_split():
    """One stratified 80/20 split, always identical because of the fixed random seed."""
    df = load_data()
    X, y = get_features_and_target(df)
    return train_test_split(
        X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE
    )
