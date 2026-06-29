"""Seed SQLite with historical resolved KXBTC15M markets."""

from __future__ import annotations

import argparse
import math
import random
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from config import settings
from config.settings import setup_logging
from data.fetchers.coinbase import CoinbaseClient
from data.fetchers.kalshi import KalshiClient
from data.processors.features import build_historical_seed_features
from data.processors.indicators import estimate_yes_probability
from data.storage.db import Database

logger = setup_logging()

ENTRY_MINUTES_BEFORE_CLOSE = 6.0
COINBASE_DELAY_SECONDS = 0.12


class _CandleCache:
    def __init__(self, client: CoinbaseClient) -> None:
        self._client = client
        self._cache: dict[tuple[int, int], list[dict]] = {}

    def get_before(
        self, end_time: datetime, granularity: int, limit: int
    ) -> list[dict]:
        if end_time.tzinfo is None:
            end_time = end_time.replace(tzinfo=timezone.utc)
        key = (granularity, int(end_time.timestamp()))
        if key not in self._cache:
            self._cache[key] = self._client.get_candles_before(
                end_time, granularity, limit
            )
            time.sleep(COINBASE_DELAY_SECONDS)
        return self._cache[key]


def _parse_closes_at(value: str) -> datetime | None:
    try:
        return datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except (TypeError, ValueError):
        return None


def _entry_price_cents(market: dict) -> float:
    yes_bid = market.get("yes_bid")
    yes_ask = market.get("yes_ask")
    if yes_bid is not None and yes_ask is not None and yes_ask > 0:
        return float((yes_bid + yes_ask) / 2)
    return float(market.get("mid_price", market.get("last_price", 50)))


def _bid_ask_spread(market: dict) -> float:
    yes_bid = market.get("yes_bid")
    yes_ask = market.get("yes_ask")
    if yes_bid is not None and yes_ask is not None:
        return max(float(yes_ask - yes_bid), 1.0)
    return 2.0


def _synthetic_rows(n: int = 800) -> list[dict]:
    """Generate synthetic BTC 15m resolved markets for offline training."""
    rows = []
    base = datetime.now(timezone.utc) - timedelta(days=90)
    rng = random.Random(42)
    spot = 95000.0

    for i in range(n):
        closes = base + timedelta(minutes=15 * i)
        entry = rng.uniform(35, 65)
        strike_offset = rng.uniform(-0.005, 0.005)
        strike = spot * (1 + strike_offset)
        spot_at_close = strike * (1 + rng.uniform(-0.003, 0.003))
        resolved_yes = 1 if spot_at_close > strike else 0
        resolved_price = 100.0 if resolved_yes else 0.0

        indicators = {
            "rsi_14": rng.uniform(30, 70),
            "macd_histogram": rng.uniform(-100, 100),
            "ema_trend": float(rng.choice([0, 1])),
            "momentum_15m": rng.uniform(-0.5, 0.5),
            "momentum_1h": rng.uniform(-1.0, 1.0),
            "volatility_atr_pct": rng.uniform(0.1, 0.8),
            "strike_distance_pct": strike_offset * 100,
        }
        model_prob = estimate_yes_probability(indicators)
        edge = model_prob - entry / 100

        rows.append(
            {
                "ticker": f"SYN-BTC15M-{i:05d}",
                "title": f"BTC price up in 15 minutes? (window {i})",
                "closes_at": closes.isoformat(),
                "volume": rng.randint(100, 50000),
                "entry_price": entry,
                "resolved_price": resolved_price,
                "resolved_yes": resolved_yes,
                "model_prob": model_prob,
                "edge": edge,
                "floor_strike": strike,
                "features": {
                    "kalshi_prob": entry / 100,
                    "model_prob": model_prob,
                    "edge": edge,
                    "abs_edge": abs(edge),
                    "minutes_to_close": rng.uniform(2, 12),
                    "log_volume": math.log1p(rng.randint(100, 50000)),
                    "bid_ask_spread": rng.uniform(1, 4),
                    "strike_distance_pct": indicators["strike_distance_pct"],
                    "rsi_14": indicators["rsi_14"],
                    "macd_histogram": indicators["macd_histogram"],
                    "ema_trend": indicators["ema_trend"],
                    "momentum_15m": indicators["momentum_15m"] * 100,
                    "momentum_1h": indicators["momentum_1h"] * 100,
                    "price_velocity_1h": rng.uniform(-5, 5),
                    "volatility_atr_pct": indicators["volatility_atr_pct"],
                },
            }
        )
    return rows


def _enrich_market_row(
    market: dict,
    *,
    coinbase: CoinbaseClient,
    candle_cache: _CandleCache,
    kalshi: KalshiClient | None,
) -> dict | None:
    closes_at = _parse_closes_at(market.get("closes_at") or "")
    if closes_at is None:
        return None

    decision_time = closes_at - timedelta(minutes=ENTRY_MINUTES_BEFORE_CLOSE)
    candles_1m = candle_cache.get_before(decision_time, 60, 120)
    candles_15m = candle_cache.get_before(decision_time, 900, 48)
    candles_1h = candle_cache.get_before(decision_time, 3600, 24)

    entry = _entry_price_cents(market)
    kalshi_history = None
    if kalshi is not None:
        kalshi_history = kalshi.get_market_history(market["ticker"])

    features = build_historical_seed_features(
        entry_price_cents=entry,
        volume=int(market.get("volume", 0)),
        candles_1m=candles_1m,
        candles_15m=candles_15m,
        candles_1h=candles_1h,
        strike=market.get("floor_strike"),
        minutes_to_close=ENTRY_MINUTES_BEFORE_CLOSE,
        bid_ask_spread=_bid_ask_spread(market),
        kalshi_history=kalshi_history,
    )
    if features is None:
        return None

    resolved_yes = int(market.get("resolved_yes", 0))
    return {
        "ticker": market["ticker"],
        "title": market.get("title"),
        "closes_at": market["closes_at"],
        "volume": market.get("volume", 0),
        "entry_price": entry,
        "resolved_price": 100.0 if resolved_yes else 0.0,
        "resolved_yes": resolved_yes,
        "model_prob": features["model_prob"],
        "edge": features["edge"],
        "floor_strike": market.get("floor_strike"),
        "features": features,
    }


def _from_api(
    client: KalshiClient,
    coinbase: CoinbaseClient,
    *,
    limit: int,
    skip_kalshi_history: bool,
) -> list[dict]:
    markets = client.get_resolved_markets(limit=limit)
    candle_cache = _CandleCache(coinbase)
    kalshi = None if skip_kalshi_history else client
    rows: list[dict] = []
    skipped = 0

    for idx, market in enumerate(markets, start=1):
        row = _enrich_market_row(
            market,
            coinbase=coinbase,
            candle_cache=candle_cache,
            kalshi=kalshi,
        )
        if row is None:
            skipped += 1
            continue
        rows.append(row)
        if idx % 25 == 0:
            logger.info("Enriched %d / %d API markets", len(rows), idx)

    if skipped:
        logger.warning("Skipped %d markets (missing Coinbase history)", skipped)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Seed training data for btc-price-bot")
    parser.add_argument(
        "--fresh",
        action="store_true",
        help="Clear existing resolved markets before seeding",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=500,
        help="Maximum settled Kalshi markets to fetch (default: 500)",
    )
    parser.add_argument(
        "--fast",
        action="store_true",
        help="Skip per-market Kalshi candlestick fetches during enrichment",
    )
    args = parser.parse_args()

    db = Database()
    db.init_schema()

    if args.fresh:
        removed = db.clear_resolved_markets()
        logger.info("Cleared %d existing resolved markets", removed)

    rows: list[dict] = []
    if settings.kalshi_api_configured():
        try:
            kalshi = KalshiClient()
            coinbase = CoinbaseClient()
            rows = _from_api(
                kalshi,
                coinbase,
                limit=args.limit,
                skip_kalshi_history=args.fast,
            )
            logger.info(
                "Fetched and enriched %d markets from Kalshi + Coinbase",
                len(rows),
            )
        except Exception as exc:
            logger.warning("Kalshi API seed failed: %s", exc)
    else:
        logger.info("No KALSHI_API_KEY — using synthetic data only")

    if len(rows) < 100:
        logger.info("Supplementing with synthetic historical data")
        rows = _synthetic_rows(800)

    count = db.insert_resolved_batch(rows)
    logger.info("Seeded %d resolved markets into %s", count, settings.DB_PATH)
    print(f"Seeded {count} markets -> {settings.DB_PATH}")


if __name__ == "__main__":
    main()
