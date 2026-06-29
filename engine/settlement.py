"""Resolve Kalshi BTC 15m market outcomes for paper settlement."""

from __future__ import annotations

import re
from datetime import datetime, timedelta, timezone
from typing import Any

from config.settings import setup_logging

logger = setup_logging()

_STRIKE_RE = re.compile(r"strike \$([\d,]+\.?\d*)", re.IGNORECASE)


def parse_strike_from_pick(pick: str | None) -> float | None:
    if not pick or "current price" in pick.lower():
        return None
    match = _STRIKE_RE.search(pick)
    if not match:
        return None
    try:
        return float(match.group(1).replace(",", ""))
    except ValueError:
        return None


def _parse_dt(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except (TypeError, ValueError):
        return None


def _float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def resolve_market_result(
    market: dict[str, Any],
    *,
    strike: float | None = None,
    entry_spot: float | None = None,
    coinbase: Any | None = None,
    now: datetime | None = None,
) -> str | None:
    """
    Return 'yes' or 'no' when the market outcome can be determined, else None.

    Uses Kalshi's official result first, then expiration_value vs strike,
    then Coinbase spot at expected expiration (for markets stuck in ``closed``).
    """
    result = str(market.get("result") or "").lower()
    if result in ("yes", "no"):
        return result

    raw = market.get("raw") or {}
    expiration_value = _float(raw.get("expiration_value") or market.get("expiration_value"))
    floor_strike = strike or _float(market.get("floor_strike")) or _float(
        raw.get("floor_strike")
    )

    if expiration_value is not None and floor_strike is not None:
        return "yes" if expiration_value > floor_strike else "no"

    status = str(market.get("status") or "").lower()
    if status not in ("closed", "finalized", "settled"):
        return None

    now = now or datetime.now(timezone.utc)
    exp_time = _parse_dt(
        raw.get("expected_expiration_time")
        or market.get("expected_expiration_time")
        or market.get("closes_at")
    )
    if exp_time is None or now < exp_time + timedelta(seconds=90):
        return None

    if coinbase is None or floor_strike is None:
        if floor_strike is None and entry_spot is not None:
            floor_strike = entry_spot
        else:
            return None

    try:
        candles = coinbase.get_candles_before(exp_time, granularity=60, limit=3)
    except Exception as exc:
        logger.warning("Coinbase settlement fetch failed for %s: %s", market.get("ticker"), exc)
        return None

    if not candles:
        return None

    settle_spot = float(candles[-1]["close"])
    resolved = "yes" if settle_spot > floor_strike else "no"
    logger.info(
        "Inferred settlement %s for %s (spot $%s vs strike $%s)",
        resolved.upper(),
        market.get("ticker"),
        f"{settle_spot:,.2f}",
        f"{floor_strike:,.2f}",
    )
    return resolved
