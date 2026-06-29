"""Stacking meta-learner (logistic regression)."""

from __future__ import annotations

import os
import pickle
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from config import settings


class MetaLearner:
    FILENAME = "meta_learner.pkl"

    def __init__(self) -> None:
        self.model: LogisticRegression | None = None
        self.scaler: StandardScaler | None = None

    def train(self, X_meta: np.ndarray, y: np.ndarray) -> None:
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X_meta)
        self.model = LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            random_state=42,
        )
        self.model.fit(X_scaled, y)

    def predict_score(self, X_meta: np.ndarray) -> np.ndarray:
        if self.model is None or self.scaler is None:
            raise RuntimeError("Meta-learner not trained")
        X_scaled = self.scaler.transform(X_meta)
        return self.model.predict_proba(X_scaled)[:, 1]

    def save(self, directory: str | None = None) -> None:
        directory = directory or settings.MODEL_DIR
        os.makedirs(directory, exist_ok=True)
        path = Path(directory) / self.FILENAME
        with open(path, "wb") as f:
            pickle.dump({"model": self.model, "scaler": self.scaler}, f)

    def load(self, directory: str | None = None) -> None:
        directory = directory or settings.MODEL_DIR
        path = Path(directory) / self.FILENAME
        if not path.exists():
            raise FileNotFoundError(f"Meta-learner not found at {path}")
        with open(path, "rb") as f:
            data = pickle.load(f)
        self.model = data["model"]
        self.scaler = data["scaler"]
