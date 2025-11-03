import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

from credit_scoring.features import build_preprocessor
from credit_scoring.model import TrainingConfig, build_training_pipeline


def test_training_pipeline_runs_on_synthetic_data():
    X, y = make_classification(
        n_samples=200,
        n_features=10,
        n_informative=5,
        random_state=42,
    )
    df = pd.DataFrame(X, columns=[f"f{i}" for i in range(10)])
    df["cat"] = np.where(df["f0"] > 0, "A", "B")

    pre = build_preprocessor(df)
    pipe = build_training_pipeline(pre, TrainingConfig(use_lightgbm=False, random_state=0))

    X_train, X_test, y_train, y_test = train_test_split(df, y, test_size=0.2, random_state=0)
    pipe.fit(X_train, y_train)

    proba = pipe.predict_proba(X_test)[:, 1]
    assert proba.shape[0] == X_test.shape[0]
