"""LightGBM expected value regressor."""

from __future__ import annotations

import os
import pickle
from pathlib import Path

import lightgbm as lgb
import numpy as np

from config import settings


class EVRegressor:
    FILENAME = "ev_regressor.pkl"

    def __init__(self) -> None:
        self.model: lgb.LGBMRegressor | None = None

    def train(self, X: np.ndarray, y: np.ndarray) -> None:
        import pandas as pd

        feature_names = (
            list(X.columns)
            if hasattr(X, "columns")
            else list(settings.FEATURE_COLUMNS)
        )
        X_fit = X if hasattr(X, "columns") else pd.DataFrame(X, columns=feature_names)
        self.model = lgb.LGBMRegressor(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            verbose=-1,
        )
        self.model.fit(X_fit, y)

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("EV regressor not trained")
        import pandas as pd

        if hasattr(X, "columns"):
            X_pred = X
        else:
            X_pred = pd.DataFrame(X, columns=settings.FEATURE_COLUMNS)
        return self.model.predict(X_pred)

    def save(self, directory: str | None = None) -> None:
        directory = directory or settings.MODEL_DIR
        os.makedirs(directory, exist_ok=True)
        path = Path(directory) / self.FILENAME
        with open(path, "wb") as f:
            pickle.dump(self.model, f)

    def load(self, directory: str | None = None) -> None:
        directory = directory or settings.MODEL_DIR
        path = Path(directory) / self.FILENAME
        if not path.exists():
            raise FileNotFoundError(f"EV regressor not found at {path}")
        with open(path, "rb") as f:
            self.model = pickle.load(f)
