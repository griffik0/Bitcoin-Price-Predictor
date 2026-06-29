"""Conformal prediction wrapper for uncertainty estimation."""

from __future__ import annotations

import os
import pickle
from pathlib import Path

import numpy as np
from sklearn.ensemble import GradientBoostingRegressor

from config import settings


class RiskScorer:
    FILENAME = "risk_scorer.pkl"
    ALPHA = 0.1

    def __init__(self) -> None:
        self.base_model: GradientBoostingRegressor | None = None
        self.calibration_scores: np.ndarray | None = None
        self.conformal_q: float = 1.0
        self.max_width: float = 1.0

    def train(self, X: np.ndarray, y: np.ndarray) -> None:
        self.base_model = GradientBoostingRegressor(
            n_estimators=100,
            max_depth=4,
            random_state=42,
        )
        self.base_model.fit(X, y)

    def fit(
        self,
        X: np.ndarray,
        y_true: np.ndarray,
        y_pred: np.ndarray | None = None,
    ) -> None:
        if self.base_model is None:
            raise RuntimeError("Call train() before fit()")
        if y_pred is None:
            y_pred = self.base_model.predict(X)
        residuals = np.abs(y_true - y_pred)
        self.calibration_scores = np.sort(residuals)
        n = len(self.calibration_scores)
        q_idx = int(np.ceil((n + 1) * (1 - self.ALPHA))) - 1
        q_idx = min(max(q_idx, 0), n - 1)
        self.conformal_q = float(
            self.calibration_scores[q_idx] if n > 0 else 1.0
        )
        self.max_width = float(
            max(np.max(self.calibration_scores), self.conformal_q, 1e-6)
            if n > 0
            else 1.0
        )

    def uncertainty(self, X: np.ndarray) -> np.ndarray:
        if self.base_model is None or self.calibration_scores is None:
            raise RuntimeError("Risk scorer not calibrated")
        preds = self.base_model.predict(X)
        local_dev = np.abs(preds - np.median(preds))
        interval_width = self.conformal_q + 0.25 * local_dev
        return np.clip(interval_width / (1.5 * self.max_width), 0, 1)

    def save(self, directory: str | None = None) -> None:
        directory = directory or settings.MODEL_DIR
        os.makedirs(directory, exist_ok=True)
        path = Path(directory) / self.FILENAME
        with open(path, "wb") as f:
            pickle.dump(
                {
                    "base_model": self.base_model,
                    "calibration_scores": self.calibration_scores,
                    "conformal_q": self.conformal_q,
                    "max_width": self.max_width,
                },
                f,
            )

    def load(self, directory: str | None = None) -> None:
        directory = directory or settings.MODEL_DIR
        path = Path(directory) / self.FILENAME
        if not path.exists():
            raise FileNotFoundError(f"Risk scorer not found at {path}")
        with open(path, "rb") as f:
            data = pickle.load(f)
        self.base_model = data["base_model"]
        self.calibration_scores = data["calibration_scores"]
        self.conformal_q = data.get("conformal_q", 1.0)
        self.max_width = data.get("max_width", 1.0)
