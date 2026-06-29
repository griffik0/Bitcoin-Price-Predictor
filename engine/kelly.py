"""Fractional Kelly stake sizing."""

from __future__ import annotations

from typing import TYPE_CHECKING

from config import settings

if TYPE_CHECKING:
    from engine.types import BetOpportunity


def fractional_kelly(
    prob_win: float,
    odds_decimal: float,
    bankroll: float,
    kelly_fraction: float | None = None,
    max_stake_pct: float | None = None,
) -> float:
    kelly_fraction = (
        kelly_fraction if kelly_fraction is not None else settings.KELLY_FRACTION
    )
    max_stake_pct = (
        max_stake_pct
        if max_stake_pct is not None
        else settings.MAX_STAKE_PCT_BANKROLL
    )

    if odds_decimal <= 1.0 or bankroll <= 0:
        return 0.0

    f = (prob_win * odds_decimal - 1.0) / (odds_decimal - 1.0)
    if f <= 0:
        return 0.0

    f *= kelly_fraction
    stake = f * bankroll
    cap = max_stake_pct * bankroll
    return float(min(max(stake, 0.0), cap))


def scale_portfolio_stakes(
    opportunities: list[BetOpportunity],
    bankroll: float,
    max_total_pct: float | None = None,
) -> list[BetOpportunity]:
    if not opportunities or bankroll <= 0:
        return opportunities

    max_total_pct = (
        max_total_pct
        if max_total_pct is not None
        else settings.MAX_TOTAL_EXPOSURE_PCT
    )
    cap_total = bankroll * max_total_pct
    total = sum(o.recommended_stake for o in opportunities)
    if total <= cap_total:
        return opportunities

    factor = cap_total / total
    for opp in opportunities:
        opp.recommended_stake = round(opp.recommended_stake * factor, 2)
    return opportunities
