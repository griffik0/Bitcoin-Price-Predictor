"""Train all ML models on historical resolved BTC markets."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from config import settings
from config.settings import setup_logging
from data.storage.db import Database
from models.trainer import ModelTrainer

logger = setup_logging()


def load_training_dataframe() -> pd.DataFrame:
    db = Database()
    db.init_schema()
    rows = db.get_resolved_markets()
    if not rows:
        raise RuntimeError(
            "No training data. Run `python scripts/seed_db.py` first."
        )

    records = []
    for row in rows:
        feats = row.get("features") or {}
        if isinstance(feats, str):
            import json

            feats = json.loads(feats)
        rec = {**feats}
        rec["ticker"] = row["ticker"]
        rec["closes_at"] = row["closes_at"]
        rec["resolved_yes"] = row["resolved_yes"]
        rec["entry_price"] = row.get(
            "entry_price", feats.get("kalshi_prob", 0.5) * 100
        )
        rec["resolved_price"] = row.get("resolved_price", 0)
        rec["edge"] = rec.get("edge", row.get("edge", 0))
        records.append(rec)

    df = pd.DataFrame(records)
    for col in settings.FEATURE_COLUMNS:
        if col not in df.columns:
            df[col] = 0.0
    return df


def main() -> None:
    df = load_training_dataframe()
    logger.info("Loaded %d training rows", len(df))

    trainer = ModelTrainer()
    metrics = trainer.train_from_dataframe(df)

    print("\n=== Test Set Metrics ===")
    for name, value in metrics.items():
        print(f"  {name}: {value:.4f}")

    print(f"\nModels saved to {settings.MODEL_DIR}/")


if __name__ == "__main__":
    main()
