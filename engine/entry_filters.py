"""Quality gates for live entries — reduce low-edge and contrarian longshots."""

from __future__ import annotations

from config import settings
from data.processors.cleaner import implied_prob_for_direction, market_tradeable


def reject_entry(
    *,
    feats: dict,
    direction: str,
    yes_price_cents: float,
    confidence: float,
    uncertainty: float,
    spot: float,
    strike: float | None,
    strike_distance_pct: float | None = None,
    skip_momentum: bool = False,
) -> str | None:
    """
    Return a short reason to skip the trade, or None if all quality gates pass.
    """
    model_prob = float(feats["model_prob"])
    kalshi_prob = float(feats["kalshi_prob"])
    edge = float(feats["edge"])
    contract_price = (
        yes_price_cents
        if direction == "YES"
        else max(1.0, min(99.0, 100.0 - yes_price_cents))
    )
    market_impl = implied_prob_for_direction(yes_price_cents, direction)

    if confidence < settings.MIN_CONFIDENCE_FOR_TRADE:
        return f"confidence {confidence:.2f} < {settings.MIN_CONFIDENCE_FOR_TRADE:.2f}"
    if uncertainty > settings.MAX_UNCERTAINTY_FOR_TRADE:
        return f"uncertainty {uncertainty:.2f} > {settings.MAX_UNCERTAINTY_FOR_TRADE:.2f}"
    if abs(edge) < settings.MIN_EDGE_THRESHOLD:
        return f"edge {abs(edge)*100:.1f}% < {settings.MIN_EDGE_THRESHOLD*100:.1f}%"
    if not market_tradeable(
        yes_price_cents, direction, settings.MIN_MARKET_IMPLIED_PROB
    ):
        return (
            f"market implied {market_impl:.0%} for {direction} "
            f"< {settings.MIN_MARKET_IMPLIED_PROB:.0%}"
        )

    if direction == "YES":
        if model_prob < settings.MIN_MODEL_PROB_YES:
            return f"model {model_prob:.0%} < min YES {settings.MIN_MODEL_PROB_YES:.0%}"
        if kalshi_prob > settings.MAX_KALSHI_YES_TO_BET_YES:
            return (
                f"YES market {kalshi_prob:.0%} > "
                f"max {settings.MAX_KALSHI_YES_TO_BET_YES:.0%}"
            )
    else:
        if model_prob > settings.MAX_MODEL_PROB_NO:
            return (
                f"model still {model_prob:.0%} YES "
                f"(need <= {settings.MAX_MODEL_PROB_NO:.0%} to bet NO)"
            )
        if kalshi_prob < settings.MIN_KALSHI_YES_TO_BET_NO:
            return (
                f"YES market {kalshi_prob:.0%} < "
                f"min {settings.MIN_KALSHI_YES_TO_BET_NO:.0%} to bet NO"
            )

    if contract_price > settings.MAX_CONTRACT_PRICE_CENTS:
        return (
            f"contract price {contract_price:.0f}c > "
            f"max {settings.MAX_CONTRACT_PRICE_CENTS:.0f}c (poor upside)"
        )

    if settings.REQUIRE_STRIKE_ALIGN:
        dist = strike_distance_pct
        if dist is None and strike and strike > 0:
            dist = (spot - strike) / strike * 100
        if dist is not None:
            if direction == "YES" and dist < settings.MIN_STRIKE_DIST_FOR_YES:
                strike_note = f"strike ${strike:,.0f}" if strike else "strike"
                return f"spot below {strike_note} ({dist:+.2f}%)"
            if direction == "NO" and dist > settings.MAX_STRIKE_DIST_FOR_NO:
                strike_note = f"strike ${strike:,.0f}" if strike else "strike"
                return f"spot above {strike_note} ({dist:+.2f}%)"

    if settings.REQUIRE_MOMENTUM_ALIGN and not skip_momentum:
        mom = float(feats.get("momentum_15m", 0))
        trend = float(feats.get("ema_trend", 0))
        if direction == "YES" and mom <= 0 and trend <= 0:
            return "no bullish momentum (15m/EMA)"
        if direction == "NO" and mom >= 0 and trend >= 0:
            return "no bearish momentum (15m/EMA)"

    # Model must see meaningfully more edge than the market, not just a rounding blip.
    model_side_prob = model_prob if direction == "YES" else (1.0 - model_prob)
    if model_side_prob < market_impl + settings.MIN_EDGE_THRESHOLD:
        return (
            f"model side {model_side_prob:.0%} not above "
            f"market {market_impl:.0%} + edge buffer"
        )

    return None
