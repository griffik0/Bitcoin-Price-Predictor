"""XGBoost binary bet/no-bet classifier."""

from __future__ import annotations

import os
import pickle
from pathlib import Path

import numpy as np
import xgboost as xgb
from sklearn.utils.class_weight import compute_class_weight

from config import settings


class BetClassifier:
    FILENAME = "bet_classifier.pkl"

    def __init__(self) -> None:
        self.model: xgb.XGBClassifier | None = None

    def train(self, X: np.ndarray, y: np.ndarray) -> None:
        classes = np.unique(y)
        weights = compute_class_weight(
            class_weight="balanced", classes=classes, y=y
        )
        weight_map = dict(zip(classes, weights))
        scale_pos = float(weight_map.get(1, 1.0) / weight_map.get(0, 1.0))

        self.model = xgb.XGBClassifier(
            n_estimators=200,
            max_depth=5,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            scale_pos_weight=scale_pos,
            eval_metric="logloss",
            random_state=42,
        )
        self.model.fit(X, y)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("Classifier not trained")
        proba = self.model.predict_proba(X)
        if proba.shape[1] == 1:
            return proba[:, 0]
        return proba[:, 1]

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
            raise FileNotFoundError(f"Classifier not found at {path}")
        with open(path, "rb") as f:
            self.model = pickle.load(f)
