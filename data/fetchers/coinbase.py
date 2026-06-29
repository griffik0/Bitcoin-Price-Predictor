"""Coinbase Exchange public API for BTC price data (Kalshi settlement reference)."""

from __future__ import annotations

import time
from datetime import datetime, timezone
from typing import Any

import requests

from config import settings
from config.settings import setup_logging

logger = setup_logging()


class CoinbaseClient:
    """Fetches BTC-USD spot price and OHLCV candles from Coinbase Exchange."""

    def __init__(self) -> None:
        self.base_url = settings.COINBASE_API_BASE.rstrip("/")
        self.product = settings.BTC_PRODUCT
        self.session = requests.Session()
        self.session.verify = settings.REQUESTS_SSL_VERIFY

    def _request(self, path: str, params: dict[str, Any] | None = None) -> Any:
        url = f"{self.base_url}{path}"
        last_error: Exception | None = None
        for attempt in range(settings.MAX_RETRIES):
            try:
                response = self.session.get(
                    url,
                    params=params,
                    timeout=settings.REQUEST_TIMEOUT_SECONDS,
                )
                response.raise_for_status()
                return response.json()
            except requests.RequestException as exc:
                last_error = exc
                if attempt < settings.MAX_RETRIES - 1:
                    time.sleep(settings.BACKOFF_BASE_SECONDS * (2**attempt))
        raise RuntimeError(f"Coinbase API failed: {last_error}")

    def get_spot_price(self) -> float:
        """Current BTC-USD price."""
        data = self._request(f"/products/{self.product}/ticker")
        return float(data["price"])

    @staticmethod
    def _parse_candle_rows(raw: list, limit: int | None = None) -> list[dict[str, Any]]:
        rows = raw[-limit:] if limit else raw
        candles = []
        for row in rows:
            candles.append(
                {
                    "time": int(row[0]),
                    "low": float(row[1]),
                    "high": float(row[2]),
                    "open": float(row[3]),
                    "close": float(row[4]),
                    "volume": float(row[5]),
                }
            )
        candles.sort(key=lambda c: c["time"])
        return candles

    def get_candles(
        self,
        granularity: int = 60,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        """
        Fetch OHLCV candles.

        granularity: 60=1m, 300=5m, 900=15m, 3600=1h
        Returns list of {time, open, high, low, close, volume} sorted oldest-first.
        """
        raw = self._request(
            f"/products/{self.product}/candles",
            params={"granularity": granularity},
        )
        return self._parse_candle_rows(raw, limit)

    def get_candles_before(
        self,
        end_time: datetime,
        granularity: int,
        limit: int = 120,
    ) -> list[dict[str, Any]]:
        """Fetch OHLCV candles ending at or before ``end_time`` (UTC)."""
        if end_time.tzinfo is None:
            end_time = end_time.replace(tzinfo=timezone.utc)
        end_ts = int(end_time.timestamp())
        start_ts = end_ts - granularity * limit
        raw = self._request(
            f"/products/{self.product}/candles",
            params={
                "granularity": granularity,
                "start": start_ts,
                "end": end_ts,
            },
        )
        return self._parse_candle_rows(raw, limit)

    def get_multi_timeframe(self) -> dict[str, list[dict[str, Any]]]:
        """Fetch 1m, 5m, 15m, and 1h candles in one call batch."""
        return {
            "1m": self.get_candles(granularity=60, limit=120),
            "5m": self.get_candles(granularity=300, limit=60),
            "15m": self.get_candles(granularity=900, limit=48),
            "1h": self.get_candles(granularity=3600, limit=24),
        }
