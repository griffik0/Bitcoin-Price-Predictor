"""Training pipeline with strict temporal splits."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, brier_score_loss, roc_auc_score

from config import settings
from config.settings import setup_logging
from engine.simulation import simulate_trades
from models.classifier import BetClassifier
from models.ev_regressor import EVRegressor
from models.meta_learner import MetaLearner
from models.risk_scorer import RiskScorer

logger = setup_logging()


def models_exist(directory: str | None = None) -> bool:
    directory = directory or settings.MODEL_DIR
    required = [
        BetClassifier.FILENAME,
        EVRegressor.FILENAME,
        RiskScorer.FILENAME,
        MetaLearner.FILENAME,
    ]
    return all((Path(directory) / f).exists() for f in required)


def temporal_split(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    df = df.sort_values("closes_at").reset_index(drop=True)
    n = len(df)
    train_end = int(n * settings.TRAIN_SPLIT)
    cal_end = int(n * (settings.TRAIN_SPLIT + settings.CALIBRATION_SPLIT))
    return df.iloc[:train_end], df.iloc[train_end:cal_end], df.iloc[cal_end:]


def build_labels(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    edge = out["edge"].astype(float)
    resolved_yes = out["resolved_yes"].astype(int)
    bet_yes_signal = edge > 0
    out["label_bet"] = (
        (bet_yes_signal & (resolved_yes == 1))
        | (~bet_yes_signal & (resolved_yes == 0))
    ).astype(int)
    entry = out["entry_price"].astype(float)
    resolved = out["resolved_price"].astype(float)
    direction_mult = np.where(edge > 0, 1, -1)
    out["label_ev"] = direction_mult * (resolved - entry)
    return out


class ModelTrainer:
    def __init__(self) -> None:
        self.classifier = BetClassifier()
        self.ev_regressor = EVRegressor()
        self.risk_scorer = RiskScorer()
        self.meta_learner = MetaLearner()

    def train_from_dataframe(self, df: pd.DataFrame) -> dict[str, float]:
        df = build_labels(df)
        train, cal, test = temporal_split(df)

        feature_cols = settings.FEATURE_COLUMNS
        X_train = train[feature_cols].values.astype(float)
        y_bet_train = train["label_bet"].values
        y_ev_train = train["label_ev"].values

        X_cal = cal[feature_cols].values.astype(float)
        y_bet_cal = cal["label_bet"].values
        y_ev_cal = cal["label_ev"].values

        X_test = test[feature_cols].values.astype(float)
        y_bet_test = test["label_bet"].values

        logger.info(
            "Training on %d rows, cal %d, test %d",
            len(train),
            len(cal),
            len(test),
        )

        self.classifier.train(X_train, y_bet_train)
        clf_cal_proba = self.classifier.predict_proba(X_cal)

        self.ev_regressor.train(
            pd.DataFrame(X_train, columns=feature_cols), y_ev_train
        )
        ev_cal_pred = self.ev_regressor.predict(
            pd.DataFrame(X_cal, columns=feature_cols)
        )

        self.risk_scorer.train(X_train, y_ev_train)
        self.risk_scorer.fit(X_cal, y_ev_cal, ev_cal_pred)
        uncertainty_cal = self.risk_scorer.uncertainty(X_cal)

        X_meta_train = self._build_meta_matrix(
            self.classifier.predict_proba(X_train),
            self.ev_regressor.predict(
                pd.DataFrame(X_train, columns=feature_cols)
            ),
            self.risk_scorer.uncertainty(X_train),
            train,
        )
        self.meta_learner.train(X_meta_train, y_bet_train)

        clf_test_proba = self.classifier.predict_proba(X_test)
        ev_test_pred = self.ev_regressor.predict(
            pd.DataFrame(X_test, columns=feature_cols)
        )
        uncertainty_test = self.risk_scorer.uncertainty(X_test)
        X_meta_test = self._build_meta_matrix(
            clf_test_proba,
            ev_test_pred,
            uncertainty_test,
            test,
        )
        meta_scores = self.meta_learner.predict_score(X_meta_test)
        meta_pred = (meta_scores >= settings.MIN_CONFIDENCE_FOR_TRADE).astype(int)

        metrics: dict[str, float] = {}
        if len(y_bet_test) > 0 and len(np.unique(y_bet_test)) > 1:
            metrics["accuracy"] = float(accuracy_score(y_bet_test, meta_pred))
            metrics["roc_auc"] = float(roc_auc_score(y_bet_test, meta_scores))
        metrics["brier"] = float(brier_score_loss(y_bet_test, meta_scores))

        sim = simulate_trades(test, meta_scores, uncertainty_test)
        metrics["simulated_pnl"] = float(sim["simulated_pnl"])
        metrics["test_bets"] = float(sim["total_bets"])
        metrics["test_win_rate"] = float(sim["win_rate"])
        metrics["test_return"] = float(sim["total_return"])

        self.classifier.save()
        self.ev_regressor.save()
        self.risk_scorer.save()
        self.meta_learner.save()

        return metrics

    def _build_meta_matrix(
        self,
        clf_proba: np.ndarray,
        ev_pred: np.ndarray,
        uncertainty: np.ndarray,
        df: pd.DataFrame,
    ) -> np.ndarray:
        return np.column_stack(
            [
                clf_proba,
                ev_pred,
                uncertainty,
                df["edge"].values.astype(float),
                df["minutes_to_close"].values.astype(float),
                df["log_volume"].values.astype(float),
            ]
        )
