from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def load_dataset(path: str | Path, low_memory: bool = False) -> pd.DataFrame:
    """Load a CSV dataset.

    - Accepts absolute or relative paths.
    - Provides a clear error if the file is missing (common with DVC-managed data).
    """
    csv_path = Path(path)
    if not csv_path.exists():
        raise FileNotFoundError(
            f"Dataset not found at {csv_path}. If you use DVC, run `dvc pull` or provide --data path."
        )
    df = pd.read_csv(
        csv_path,
        low_memory=low_memory,
        encoding="ISO-8859-1",
    )
    return df


def split_features_target(df: pd.DataFrame, target_col: str = "TARGET"):
    if target_col not in df.columns:
        raise KeyError(
            f"Target column '{target_col}' not found in dataset. Columns: {list(df.columns)[:10]}..."
        )
    # Remplacement des infinis par NaN
    df = df.replace([np.inf, -np.inf], np.nan)
    X = df.drop(columns=[target_col])
    y = df[target_col]
    return X, y
