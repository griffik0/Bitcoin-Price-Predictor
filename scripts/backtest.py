"""Historical backtest simulation on resolved BTC markets."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from config import settings
from config.settings import setup_logging
from data.storage.db import Database
from engine.simulation import simulate_trades
from models.classifier import BetClassifier
from models.ev_regressor import EVRegressor
from models.meta_learner import MetaLearner
from models.risk_scorer import RiskScorer
from models.trainer import build_labels, temporal_split
from scripts.train import load_training_dataframe

logger = setup_logging()


def main() -> None:
    df = load_training_dataframe()
    df = build_labels(df)
    _, _, test = temporal_split(df)

    if len(test) == 0:
        print("Not enough data for backtest.")
        return

    clf = BetClassifier()
    ev = EVRegressor()
    risk = RiskScorer()
    meta = MetaLearner()
    clf.load()
    ev.load()
    risk.load()
    meta.load()

    feature_cols = settings.FEATURE_COLUMNS
    X = test[feature_cols].values.astype(float)
    X_df = pd.DataFrame(X, columns=feature_cols)
    clf_proba = clf.predict_proba(X)
    ev_pred = ev.predict(X_df)
    uncertainty = risk.uncertainty(X)
    X_meta = np.column_stack(
        [
            clf_proba,
            ev_pred,
            uncertainty,
            test["edge"].values,
            test["minutes_to_close"].values,
            test["log_volume"].values,
        ]
    )
    scores = meta.predict_score(X_meta)

    sim = simulate_trades(test, scores, uncertainty)

    metrics = {
        "total_return": sim["total_return"],
        "sharpe_ratio": sim["sharpe_ratio"],
        "max_drawdown": 0.0,
        "win_rate": sim["win_rate"],
        "total_bets": int(sim["total_bets"]),
        "notes": f"Final bankroll: ${sim['final_bankroll']:.2f}",
    }

    db = Database()
    db.init_schema()
    db.save_backtest_run(metrics)

    print("\n=== Backtest Results ===")
    print(f"  Total bets: {metrics['total_bets']}")
    print(f"  Win rate: {metrics['win_rate']:.1%}")
    print(f"  Total return: {metrics['total_return']:.1%}")
    print(f"  Sharpe ratio: {metrics['sharpe_ratio']:.2f}")
    print(f"  Final bankroll: ${sim['final_bankroll']:.2f}")


if __name__ == "__main__":
    main()
