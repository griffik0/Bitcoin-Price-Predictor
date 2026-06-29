"""Technical indicators computed from OHLCV candle data."""

from __future__ import annotations

import math
from typing import Any


def _closes(candles: list[dict[str, Any]]) -> list[float]:
    return [float(c["close"]) for c in candles]


def _ema(values: list[float], period: int) -> list[float]:
    if not values:
        return []
    k = 2 / (period + 1)
    result = [values[0]]
    for v in values[1:]:
        result.append(v * k + result[-1] * (1 - k))
    return result


def rsi(closes: list[float], period: int = 14) -> float:
    if len(closes) < period + 1:
        return 50.0
    gains, losses = [], []
    for i in range(1, len(closes)):
        delta = closes[i] - closes[i - 1]
        gains.append(max(delta, 0))
        losses.append(max(-delta, 0))
    avg_gain = sum(gains[-period:]) / period
    avg_loss = sum(losses[-period:]) / period
    if avg_loss == 0:
        return 100.0
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))


def macd(closes: list[float]) -> tuple[float, float, float]:
    if len(closes) < 26:
        return 0.0, 0.0, 0.0
    ema12 = _ema(closes, 12)
    ema26 = _ema(closes, 26)
    macd_line = [a - b for a, b in zip(ema12, ema26)]
    signal = _ema(macd_line, 9)
    hist = macd_line[-1] - signal[-1]
    return macd_line[-1], signal[-1], hist


def atr(candles: list[dict[str, Any]], period: int = 14) -> float:
    if len(candles) < period + 1:
        return 0.0
    trs = []
    for i in range(1, len(candles)):
        high = float(candles[i]["high"])
        low = float(candles[i]["low"])
        prev_close = float(candles[i - 1]["close"])
        tr = max(high - low, abs(high - prev_close), abs(low - prev_close))
        trs.append(tr)
    return sum(trs[-period:]) / period


def momentum(closes: list[float], lookback: int) -> float:
    if len(closes) <= lookback:
        return 0.0
    old = closes[-lookback - 1]
    if old == 0:
        return 0.0
    return (closes[-1] - old) / old


def ema_trend(closes: list[float]) -> int:
    """1 if short EMA above long EMA (bullish), else 0."""
    if len(closes) < 21:
        return 0
    ema9 = _ema(closes, 9)[-1]
    ema21 = _ema(closes, 21)[-1]
    return 1 if ema9 > ema21 else 0


def compute_indicators(
    candles_1m: list[dict[str, Any]],
    candles_15m: list[dict[str, Any]],
    candles_1h: list[dict[str, Any]],
    spot_price: float,
    strike: float | None,
) -> dict[str, float]:
    """Compute all technical indicators for prediction."""
    closes_1m = _closes(candles_1m)
    closes_15m = _closes(candles_15m)
    closes_1h = _closes(candles_1h)

    _, _, macd_hist = macd(closes_1m if len(closes_1m) >= 26 else closes_15m)
    rsi_val = rsi(closes_1m if len(closes_1m) >= 15 else closes_15m)
    atr_val = atr(candles_1m if len(candles_1m) >= 15 else candles_15m)
    atr_pct = (atr_val / spot_price * 100) if spot_price > 0 else 0.0

    strike_dist = 0.0
    if strike and strike > 0:
        strike_dist = (spot_price - strike) / strike * 100

    return {
        "rsi_14": rsi_val,
        "macd_histogram": macd_hist,
        "ema_trend": float(ema_trend(closes_1m if len(closes_1m) >= 21 else closes_15m)),
        "momentum_15m": momentum(closes_15m, 1) * 100,
        "momentum_1h": momentum(closes_1h, 1) * 100,
        "volatility_atr_pct": atr_pct,
        "strike_distance_pct": strike_dist,
    }


def estimate_yes_probability(indicators: dict[str, float]) -> float:
    """
    Heuristic model probability that BTC ends above strike (YES wins).

    Combines momentum, RSI, MACD, EMA trend, and strike distance.
    Returns value in [0.05, 0.95].
    """
    score = 0.5

    # Strike distance is the strongest signal for 15-min windows
    strike_dist = indicators.get("strike_distance_pct", 0.0)
    score += math.tanh(strike_dist / 0.15) * 0.25

    # Momentum
    mom_15 = indicators.get("momentum_15m", 0.0)
    score += math.tanh(mom_15 / 0.3) * 0.15

    mom_1h = indicators.get("momentum_1h", 0.0)
    score += math.tanh(mom_1h / 0.5) * 0.08

    # RSI mean-reversion adjustment (extreme RSI less likely to continue)
    rsi_val = indicators.get("rsi_14", 50.0)
    if rsi_val > 70:
        score -= 0.05
    elif rsi_val < 30:
        score += 0.05
    elif rsi_val > 55:
        score += (rsi_val - 50) / 200
    elif rsi_val < 45:
        score -= (50 - rsi_val) / 200

    # MACD histogram direction
    macd_hist = indicators.get("macd_histogram", 0.0)
    score += math.tanh(macd_hist / 50) * 0.1

    # EMA trend
    if indicators.get("ema_trend", 0) > 0:
        score += 0.05
    else:
        score -= 0.05

    return max(0.05, min(0.95, score))
