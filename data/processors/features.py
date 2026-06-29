"""Feature engineering for BTC 15-minute Kalshi markets."""

from __future__ import annotations

import math
from typing import Any

from data.processors.cleaner import kalshi_price_to_prob
from data.processors.indicators import compute_indicators, estimate_yes_probability


def _kalshi_velocity_from_history(history: list[dict[str, Any]]) -> float:
    """Price change in Kalshi contract over last hour (cents)."""
    if len(history) < 2:
        return 0.0
    try:
        recent = history[-1]
        older = history[max(0, len(history) - 60)]
        p1 = float(recent.get("close", recent.get("price", 50)))
        p0 = float(older.get("close", older.get("price", 50)))
        return p1 - p0
    except (TypeError, ValueError, KeyError):
        return 0.0


def build_historical_seed_features(
    *,
    entry_price_cents: float,
    volume: int,
    candles_1m: list[dict[str, Any]],
    candles_15m: list[dict[str, Any]],
    candles_1h: list[dict[str, Any]],
    strike: float | None,
    minutes_to_close: float = 6.0,
    bid_ask_spread: float = 2.0,
    kalshi_history: list[dict[str, Any]] | None = None,
) -> dict[str, float] | None:
    """Build training features for a resolved market at a historical entry time."""
    spot = 0.0
    for series in (candles_1m, candles_15m, candles_1h):
        if series:
            spot = float(series[-1]["close"])
            break
    if spot <= 0:
        return None

    if strike is None or strike <= 0:
        strike = spot

    indicators = compute_indicators(
        candles_1m,
        candles_15m,
        candles_1h,
        spot,
        strike,
    )
    model_prob = estimate_yes_probability(indicators)
    kalshi_prob = entry_price_cents / 100
    edge = model_prob - kalshi_prob

    return {
        "kalshi_prob": kalshi_prob,
        "model_prob": model_prob,
        "edge": edge,
        "abs_edge": abs(edge),
        "minutes_to_close": float(minutes_to_close),
        "log_volume": math.log1p(volume),
        "bid_ask_spread": bid_ask_spread,
        "strike_distance_pct": indicators["strike_distance_pct"],
        "rsi_14": indicators["rsi_14"],
        "macd_histogram": indicators["macd_histogram"],
        "ema_trend": indicators["ema_trend"],
        "momentum_15m": indicators["momentum_15m"],
        "momentum_1h": indicators["momentum_1h"],
        "price_velocity_1h": _kalshi_velocity_from_history(kalshi_history or []),
        "volatility_atr_pct": indicators["volatility_atr_pct"],
    }


def build_features(
    market: dict[str, Any],
    spot_price: float,
    candles: dict[str, list[dict[str, Any]]],
    orderbook: dict[str, Any] | None,
    history: list[dict[str, Any]] | None,
) -> dict[str, float] | None:
    """
    Build feature vector for a BTC 15m market.

    Returns None if market is outside the entry window or missing critical data.
    """
    minutes_to_close = market.get("minutes_to_close")
    if minutes_to_close is None:
        return None

    from config import settings

    if minutes_to_close < settings.MIN_MINUTES_TO_CLOSE:
        return None
    if minutes_to_close > settings.MAX_MINUTES_TO_CLOSE:
        return None

    strike = market.get("floor_strike")
    if strike is None:
        strike = spot_price

    indicators = compute_indicators(
        candles.get("1m", []),
        candles.get("15m", []),
        candles.get("1h", []),
        spot_price,
        strike,
    )
    model_prob = estimate_yes_probability(indicators)

    kalshi_price = float(market.get("mid_price", 50))
    if orderbook and orderbook.get("yes_bid") and orderbook.get("yes_ask"):
        kalshi_price = (orderbook["yes_bid"] + orderbook["yes_ask"]) / 2

    kalshi_prob = kalshi_price_to_prob(kalshi_price)
    edge = model_prob - kalshi_prob

    spread = 2.0
    if orderbook and orderbook.get("spread") is not None:
        spread = float(orderbook["spread"])

    volume = int(market.get("volume", 0))

    return {
        "kalshi_prob": kalshi_prob,
        "model_prob": model_prob,
        "edge": edge,
        "abs_edge": abs(edge),
        "minutes_to_close": float(minutes_to_close),
        "log_volume": math.log1p(volume),
        "bid_ask_spread": spread,
        "strike_distance_pct": indicators["strike_distance_pct"],
        "rsi_14": indicators["rsi_14"],
        "macd_histogram": indicators["macd_histogram"],
        "ema_trend": indicators["ema_trend"],
        "momentum_15m": indicators["momentum_15m"],
        "momentum_1h": indicators["momentum_1h"],
        "price_velocity_1h": _kalshi_velocity_from_history(history or []),
        "volatility_atr_pct": indicators["volatility_atr_pct"],
    }
