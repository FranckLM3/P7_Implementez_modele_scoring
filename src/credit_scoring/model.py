from __future__ import annotations

from dataclasses import dataclass

from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline


@dataclass
class TrainingConfig:
    use_lightgbm: bool = False
    random_state: int = 42
    max_iter: int = 200


def _make_selector(random_state: int = 42):
    """Try to create a feature selector using LightGBM if available, else LogisticRegression(L1).
    Using a simple linear selector keeps tests lightweight.
    """
    try:
        from lightgbm import LGBMClassifier  # type: ignore

        base = LGBMClassifier(n_estimators=200, random_state=random_state)
    except Exception:  # pragma: no cover - optional dep
        base = LogisticRegression(
            penalty="l1", solver="saga", max_iter=200, random_state=random_state
        )
    return SelectFromModel(base)


def build_training_pipeline(preprocessor, config: TrainingConfig | None = None) -> Pipeline:
    config = config or TrainingConfig()
    selector = _make_selector(config.random_state)

    if config.use_lightgbm:
        try:
            from lightgbm import LGBMClassifier  # type: ignore

            clf = LGBMClassifier(
                n_estimators=500,
                learning_rate=0.05,
                random_state=config.random_state,
            )
        except Exception:  # pragma: no cover - optional dep
            clf = LogisticRegression(max_iter=config.max_iter, random_state=config.random_state)
    else:
        clf = LogisticRegression(max_iter=config.max_iter, random_state=config.random_state)

    pipe = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("selector", selector),
            ("classifier", clf),
        ]
    )
    return pipe
