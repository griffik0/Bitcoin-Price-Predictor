"""Shared engine types."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class BetOpportunity:
    market_ticker: str
    market_title: str
    window_label: str
    pick: str
    direction: str
    kalshi_price: float  # YES mid price (¢)
    contract_price: float  # entry limit price for direction (¢)
    market_implied_prob: float  # Kalshi implied P(win) for direction
    model_prob: float
    edge_pct: float
    confidence: float
    ev_per_contract: float
    uncertainty: float
    recommended_stake: float
    minutes_to_close: float
    closes_at: str
    floor_strike: float | None
    spot_price: float
    momentum_15m: float = 0.0
    ema_trend: float = 0.0


@dataclass
class ExitSignal:
    ticker: str
    side: str
    count: int
    reason: str
    entry_price_cents: int
    current_bid_cents: int
