"""Shared backtest / paper-trade simulation logic."""

from __future__ import annotations

import numpy as np
import pandas as pd

from config import settings
from engine.entry_filters import reject_entry


def simulate_trades(
    test: pd.DataFrame,
    scores: np.ndarray,
    uncertainty: np.ndarray,
    *,
    bankroll: float | None = None,
) -> dict[str, float]:
    """
    Walk the test set with the same gates as live trading and backtest.py.

    Returns bankroll stats plus realized PnL components for metrics reporting.
    """
    bankroll = settings.BANKROLL if bankroll is None else bankroll
    start_bankroll = bankroll
    pnl_history: list[float] = []
    wins = 0
    bets = 0

    for idx in range(len(test)):
        row = test.iloc[idx]
        edge = float(row["edge"])
        direction = "YES" if edge > 0 else "NO"
        yes_price = float(row.get("kalshi_prob", 0.5)) * 100
        feats = {
            "model_prob": float(row["model_prob"]),
            "kalshi_prob": float(row["kalshi_prob"]),
            "edge": edge,
            "momentum_15m": float(row.get("momentum_15m", 0)),
            "ema_trend": float(row.get("ema_trend", 0)),
        }
        reason = reject_entry(
            feats=feats,
            direction=direction,
            yes_price_cents=yes_price,
            confidence=float(scores[idx]),
            uncertainty=float(uncertainty[idx]),
            spot=0.0,
            strike=None,
            strike_distance_pct=float(row.get("strike_distance_pct", 0)),
        )
        if reason:
            continue

        bets += 1
        realized = float(row["label_ev"])
        stake = min(
            bankroll * settings.MAX_STAKE_PCT_BANKROLL,
            bankroll * 0.05,
        )
        entry_price = float(row["entry_price"])
        if entry_price <= 0:
            continue
        contracts = max(1, int(stake / (entry_price / 100)))
        trade_pnl = realized * contracts / 100
        bankroll += trade_pnl
        pnl_history.append(trade_pnl)
        if realized > 0:
            wins += 1

    total_return = (bankroll - start_bankroll) / start_bankroll if start_bankroll else 0.0
    win_rate = wins / bets if bets else 0.0
    sharpe = 0.0
    if len(pnl_history) > 1 and np.std(pnl_history) > 0:
        sharpe = float(np.mean(pnl_history) / np.std(pnl_history) * np.sqrt(252))

    return {
        "total_return": total_return,
        "sharpe_ratio": sharpe,
        "win_rate": win_rate,
        "total_bets": float(bets),
        "final_bankroll": bankroll,
        "simulated_pnl": bankroll - start_bankroll,
    }
