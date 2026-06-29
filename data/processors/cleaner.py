"""Price and probability helpers."""

from __future__ import annotations


def kalshi_price_to_prob(price_cents: float) -> float:
    """Convert Kalshi cents price to implied probability."""
    return max(0.01, min(0.99, price_cents / 100.0))


def contract_price_cents(yes_price_cents: float, direction: str) -> float:
    """Limit price in cents for the chosen side (YES or NO)."""
    yes = max(1.0, min(99.0, float(yes_price_cents)))
    if direction.upper() == "YES":
        return yes
    return max(1.0, min(99.0, 100.0 - yes))


def implied_prob_for_direction(yes_price_cents: float, direction: str) -> float:
    """Market-implied win probability for the bet direction."""
    yes_prob = kalshi_price_to_prob(yes_price_cents)
    if direction.upper() == "YES":
        return yes_prob
    return 1.0 - yes_prob


def market_tradeable(yes_price_cents: float, direction: str, min_implied_prob: float) -> bool:
    """False when the market heavily disagrees with the chosen side."""
    return implied_prob_for_direction(yes_price_cents, direction) >= min_implied_prob


def prob_to_logit(p: float) -> float:
    p = max(0.01, min(0.99, p))
    import math

    return math.log(p / (1 - p))
