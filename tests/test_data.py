import pandas as pd

from credit_scoring.data import split_features_target


def test_split_features_target_minimal():
    df = pd.DataFrame({"a": [1, 2], "TARGET": [0, 1]})
    X, y = split_features_target(df, target_col="TARGET")
    assert "TARGET" not in X.columns
    assert list(y.values) == [0, 1]
