"""Rank and dedupe BTC betting opportunities."""

from __future__ import annotations

from engine.types import BetOpportunity


def dedupe_one_per_window(
    opportunities: list[BetOpportunity],
) -> list[BetOpportunity]:
    """Keep only the best opportunity per event/window."""
    best: dict[str, BetOpportunity] = {}
    for opp in opportunities:
        key = opp.window_label or opp.market_ticker
        existing = best.get(key)
        if existing is None or opp.confidence > existing.confidence:
            best[key] = opp
    return list(best.values())


def rank_opportunities(
    opportunities: list[BetOpportunity],
    min_confidence: float | None = None,
    max_uncertainty: float | None = None,
) -> list[BetOpportunity]:
    from config import settings

    floor = (
        min_confidence
        if min_confidence is not None
        else settings.MIN_CONFIDENCE_THRESHOLD
    )
    unc_cap = (
        max_uncertainty
        if max_uncertainty is not None
        else settings.MAX_UNCERTAINTY_FOR_TRADE
    )
    filtered = [
        o
        for o in opportunities
        if o.confidence >= floor and o.uncertainty <= unc_cap
    ]
    return sorted(
        filtered,
        key=lambda o: (-o.confidence, -abs(o.edge_pct), o.minutes_to_close),
    )
