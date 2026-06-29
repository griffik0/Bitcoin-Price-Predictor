"""Kalshi REST API client for KXBTC15M markets."""

from __future__ import annotations

import time
from datetime import datetime, timezone
from typing import Any
from zoneinfo import ZoneInfo

import requests

from config import settings
from config.settings import setup_logging

logger = setup_logging()

ET = ZoneInfo("America/New_York")


class KalshiNotFoundError(Exception):
    """Kalshi returned 404 for an optional resource."""


class KalshiClient:
    """Read-only Kalshi client for BTC 15-minute markets."""

    def __init__(self) -> None:
        self.base_url = settings.KALSHI_API_BASE.rstrip("/")
        self.api_key = settings.KALSHI_API_KEY
        self.session = requests.Session()
        self.session.verify = settings.REQUESTS_SSL_VERIFY
        if self.api_key and settings.kalshi_api_configured():
            self.session.headers.update(
                {"Authorization": f"Bearer {self.api_key}"}
            )

    def _rate_limit_wait(self, response: requests.Response, attempt: int) -> float:
        retry_after = response.headers.get("Retry-After")
        if retry_after:
            try:
                return max(float(retry_after), 1.0)
            except ValueError:
                pass
        return min(60.0, settings.BACKOFF_BASE_SECONDS * (2**attempt))

    def _request(
        self,
        method: str,
        path: str,
        params: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        url = f"{self.base_url}{path}"
        last_error: Exception | None = None
        rate_limit_attempts = 0

        for attempt in range(settings.MAX_RETRIES):
            try:
                while True:
                    response = self.session.request(
                        method,
                        url,
                        params=params,
                        timeout=settings.REQUEST_TIMEOUT_SECONDS,
                    )
                    if response.status_code == 429:
                        rate_limit_attempts += 1
                        if rate_limit_attempts > settings.KALSHI_RATE_LIMIT_MAX_RETRIES:
                            response.raise_for_status()
                        wait = self._rate_limit_wait(response, rate_limit_attempts)
                        logger.warning("Kalshi rate limit; sleeping %.1fs", wait)
                        time.sleep(wait)
                        continue
                    if response.status_code == 404:
                        raise KalshiNotFoundError(path)
                    response.raise_for_status()
                    return response.json()
            except KalshiNotFoundError:
                raise
            except requests.RequestException as exc:
                last_error = exc
                wait = settings.BACKOFF_BASE_SECONDS * (2**attempt)
                logger.warning(
                    "Kalshi request failed (attempt %d): %s", attempt + 1, exc
                )
                if attempt < settings.MAX_RETRIES - 1:
                    time.sleep(wait)

        raise RuntimeError(f"Kalshi API failed after retries: {last_error}")

    def _to_cents(self, val: Any) -> float | None:
        if val is None:
            return None
        v = float(val)
        if v <= 1.0:
            return v * 100
        return v

    def _parse_market(self, raw: dict[str, Any]) -> dict[str, Any]:
        close_time = raw.get("close_time") or raw.get("expiration_time")
        closes_at = None
        minutes_to_close = None
        if close_time:
            try:
                if isinstance(close_time, (int, float)):
                    dt = datetime.fromtimestamp(close_time, tz=timezone.utc)
                else:
                    dt = datetime.fromisoformat(
                        str(close_time).replace("Z", "+00:00")
                    )
                closes_at = dt.isoformat()
                minutes_to_close = max(
                    0.0,
                    (dt - datetime.now(timezone.utc)).total_seconds() / 60,
                )
            except (ValueError, TypeError):
                pass

        yes_bid = self._to_cents(raw.get("yes_bid") or raw.get("yes_bid_dollars"))
        yes_ask = self._to_cents(raw.get("yes_ask") or raw.get("yes_ask_dollars"))
        last_c = self._to_cents(
            raw.get("last_price") or raw.get("last_price_dollars")
        )

        if yes_bid is not None and yes_ask is not None and yes_ask > 0:
            mid = (yes_bid + yes_ask) / 2
        else:
            mid = last_c or 50.0

        vol = raw.get("volume")
        if vol is None or vol == 0:
            for key in ("volume_fp", "volume_24h_fp", "liquidity_dollars"):
                alt = raw.get(key)
                if alt is not None:
                    try:
                        vol = float(alt)
                        break
                    except (TypeError, ValueError):
                        pass
        volume = int(float(vol or 0))

        floor_strike = raw.get("floor_strike")
        if floor_strike is not None:
            try:
                floor_strike = float(floor_strike)
            except (TypeError, ValueError):
                floor_strike = None

        return {
            "ticker": raw.get("ticker", ""),
            "title": raw.get("title", ""),
            "subtitle": raw.get("subtitle", ""),
            "event_ticker": raw.get("event_ticker", ""),
            "series_ticker": raw.get("series_ticker", settings.KALSHI_BTC_SERIES),
            "status": raw.get("status", ""),
            "volume": volume,
            "yes_bid": yes_bid,
            "yes_ask": yes_ask,
            "last_price": last_c or mid,
            "mid_price": mid,
            "closes_at": closes_at,
            "minutes_to_close": minutes_to_close,
            "floor_strike": floor_strike,
            "raw": raw,
        }

    def _iter_market_pages(
        self,
        *,
        status: str,
        series_ticker: str | None = None,
        event_ticker: str | None = None,
        page_limit: int = 200,
        max_pages: int | None = None,
    ):
        cursor: str | None = None
        pages = 0
        cap = max_pages if max_pages is not None else settings.KALSHI_MAX_PAGES_PER_SERIES

        while pages < cap:
            params: dict[str, Any] = {"status": status, "limit": page_limit}
            if cursor:
                params["cursor"] = cursor
            if series_ticker:
                params["series_ticker"] = series_ticker
            if event_ticker:
                params["event_ticker"] = event_ticker

            data = self._request("GET", "/markets", params=params)
            batch = data.get("markets", data.get("data", []))
            if not batch:
                break

            for raw in batch:
                yield raw, self._parse_market(raw)

            cursor = data.get("cursor")
            pages += 1
            if not cursor:
                break
            time.sleep(settings.KALSHI_PAGE_DELAY_SECONDS)

    def _current_event_ticker(self) -> str:
        """Build KXBTC15M event ticker for the current 15-min ET window."""
        now_et = datetime.now(ET)
        minute_block = (now_et.minute // 15) * 15
        window_start = now_et.replace(minute=minute_block, second=0, microsecond=0)
        return window_start.strftime(f"{settings.KALSHI_BTC_SERIES}-%y%b%d%H%M").upper()

    def get_active_btc_markets(self) -> list[dict[str, Any]]:
        """Return open KXBTC15M markets for current and next windows."""
        markets: list[dict[str, Any]] = []
        seen: set[str] = set()

        # Try series-wide scan first
        try:
            for raw, parsed in self._iter_market_pages(
                status="open",
                series_ticker=settings.KALSHI_BTC_SERIES,
            ):
                if parsed["volume"] < settings.MIN_VOLUME:
                    continue
                if parsed.get("yes_ask") is not None and parsed["yes_ask"] <= 0:
                    continue
                ticker = parsed["ticker"]
                if ticker in seen:
                    continue
                seen.add(ticker)
                markets.append(parsed)
        except RuntimeError as exc:
            logger.warning("Kalshi series scan failed: %s", exc)

        # Also try current event ticker for freshest market
        if not markets:
            try:
                event = self._current_event_ticker()
                for raw, parsed in self._iter_market_pages(
                    status="open",
                    event_ticker=event,
                    max_pages=3,
                ):
                    ticker = parsed["ticker"]
                    if ticker not in seen:
                        seen.add(ticker)
                        markets.append(parsed)
            except RuntimeError as exc:
                logger.warning("Kalshi event scan failed: %s", exc)

        markets.sort(
            key=lambda m: m.get("minutes_to_close") or 999,
        )
        logger.info("Fetched %d open BTC 15m markets from Kalshi", len(markets))
        return markets

    def get_market_orderbook(self, market_ticker: str) -> dict[str, Any] | None:
        try:
            data = self._request("GET", f"/markets/{market_ticker}/orderbook")
        except (KalshiNotFoundError, RuntimeError):
            return None
        ob = data.get("orderbook", data)

        def best(levels: list | None, side: str) -> float | None:
            if not levels:
                return None
            if side == "bid":
                return float(levels[0][0]) if levels[0] else None
            return float(levels[-1][0]) if levels[-1] else None

        yes_bids = ob.get("yes") or ob.get("yes_dollars") or []
        no_bids = ob.get("no") or ob.get("no_dollars") or []

        yes_bid = best(yes_bids, "bid")
        yes_ask = None
        if no_bids:
            no_best = float(no_bids[-1][0]) if no_bids else None
            if no_best is not None:
                yes_ask = 100 - no_best if no_best <= 100 else 100 - no_best

        return {
            "ticker": market_ticker,
            "yes_bid": yes_bid,
            "yes_ask": yes_ask,
            "spread": (yes_ask - yes_bid) if yes_bid and yes_ask else None,
        }

    def get_resolved_markets(self, limit: int = 500) -> list[dict[str, Any]]:
        """Return settled KXBTC15M markets for training."""
        markets: list[dict[str, Any]] = []
        try:
            for raw, parsed in self._iter_market_pages(
                status="settled",
                series_ticker=settings.KALSHI_BTC_SERIES,
                max_pages=15,
            ):
                result = raw.get("result", "")
                parsed["result"] = result
                parsed["resolved_yes"] = 1 if result == "yes" else 0
                markets.append(parsed)
                if len(markets) >= limit:
                    break
        except RuntimeError as exc:
            logger.warning("Kalshi settled fetch failed: %s", exc)
        return markets[:limit]

    def get_market(self, market_ticker: str) -> dict[str, Any] | None:
        """Fetch a single market by ticker (includes settlement result when settled)."""
        try:
            data = self._request("GET", f"/markets/{market_ticker}")
        except (KalshiNotFoundError, RuntimeError):
            return None
        raw = data.get("market", data)
        parsed = self._parse_market(raw)
        result = raw.get("result", "")
        parsed["result"] = result
        if result == "yes":
            parsed["resolved_yes"] = 1
        elif result == "no":
            parsed["resolved_yes"] = 0
        else:
            parsed["resolved_yes"] = None
        parsed["status"] = raw.get("status", parsed.get("status", ""))
        exp_val = raw.get("expiration_value")
        if exp_val not in (None, ""):
            try:
                parsed["expiration_value"] = float(exp_val)
            except (TypeError, ValueError):
                parsed["expiration_value"] = None
        else:
            parsed["expiration_value"] = None
        parsed["expected_expiration_time"] = raw.get("expected_expiration_time")
        return parsed

    def get_market_history(self, market_ticker: str) -> list[dict[str, Any]]:
        try:
            data = self._request(
                "GET",
                f"/markets/{market_ticker}/candlesticks",
                params={"period_interval": 1},
            )
            return data.get("candlesticks", data.get("history", []))
        except (KalshiNotFoundError, RuntimeError):
            return []
