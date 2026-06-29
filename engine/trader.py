"""Place and exit Kalshi orders for BTC 15-minute markets."""

from __future__ import annotations

import json
import time
import uuid
from dataclasses import dataclass
from pathlib import Path

from config import settings
from config.settings import setup_logging
from data.fetchers.kalshi import KalshiClient
from data.fetchers.kalshi_auth import KalshiTradingClient
from engine.entry_filters import reject_entry
from engine.paper_ledger import PaperLedger
from engine.types import BetOpportunity, ExitSignal

logger = setup_logging()

_PLACED_LOG = Path(settings.PROJECT_ROOT) / "data" / "placed_orders.json"


@dataclass
class OrderResult:
    ticker: str
    pick: str
    action: str
    count: int
    side: str
    price_cents: int
    stake_usd: float
    live: bool
    success: bool
    order_id: str | None = None
    error: str | None = None


def stake_to_contract_count(stake_usd: float, price_cents: float) -> int:
    if stake_usd <= 0 or price_cents <= 0:
        return 0
    cost = price_cents / 100.0
    return max(1, int(stake_usd / cost))


def build_order_body(
    ticker: str,
    direction: str,
    count: int,
    price_cents: int,
    action: str = "buy",
) -> dict:
    body: dict = {
        "ticker": ticker,
        "action": action,
        "side": "yes" if direction == "YES" else "no",
        "type": "limit",
        "count": count,
        "client_order_id": str(uuid.uuid4()),
    }
    if body["side"] == "yes":
        body["yes_price"] = price_cents
    else:
        body["no_price"] = price_cents
    return body


def _load_placed() -> dict[str, dict[str, str]]:
    if not _PLACED_LOG.is_file():
        return {}
    try:
        raw = json.loads(_PLACED_LOG.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {}
    placed: dict[str, dict[str, str]] = {}
    for ticker, value in raw.items():
        if isinstance(value, dict):
            placed[ticker] = value
        else:
            placed[ticker] = {"order_id": str(value), "side": "yes"}
    return placed


def _save_placed(data: dict[str, dict[str, str]]) -> None:
    _PLACED_LOG.parent.mkdir(parents=True, exist_ok=True)
    _PLACED_LOG.write_text(json.dumps(data, indent=2), encoding="utf-8")


def already_placed_ticker(ticker: str, paper: PaperLedger | None = None) -> bool:
    if paper is not None and paper.has_traded(ticker):
        return True
    return ticker in _load_placed()


def mark_placed(
    ticker: str,
    order_id: str,
    *,
    side: str = "yes",
    entry_price_cents: int = 0,
    count: int = 1,
) -> None:
    data = _load_placed()
    data[ticker] = {
        "order_id": order_id,
        "side": side,
        "entry_price_cents": str(entry_price_cents),
        "count": str(count),
    }
    _save_placed(data)


def unmark_placed(ticker: str) -> None:
    data = _load_placed()
    data.pop(ticker, None)
    _save_placed(data)


class KalshiTrader:
    """Execute entries, evaluate exits, and manage positions."""

    def __init__(self, live: bool = False, bankroll: float | None = None) -> None:
        self.live = live
        self._client: KalshiTradingClient | None = None
        self._reader = KalshiClient()
        self.paper: PaperLedger | None = None
        if not live:
            self.paper = PaperLedger(bankroll or settings.BANKROLL)
        if live and settings.kalshi_trading_configured():
            self._client = KalshiTradingClient()

    def _open_position(self, ticker: str) -> dict | None:
        if self.paper:
            for trade in self.paper.db.get_open_paper_trades():
                if trade["ticker"] == ticker:
                    return trade
        placed = _load_placed().get(ticker)
        if not placed:
            return None
        return {
            "ticker": ticker,
            "side": placed.get("side", "yes"),
            "entry_price_cents": int(placed.get("entry_price_cents", 50)),
            "count": int(placed.get("count", 1)),
        }

    def _position_side(self, ticker: str) -> str:
        pos = self._open_position(ticker)
        return str(pos.get("side", "yes")) if pos else "yes"

    def evaluate_exits(
        self,
        opportunities: list[BetOpportunity],
    ) -> list[ExitSignal]:
        """Determine if open positions should be sold before settlement."""
        if settings.HOLD_TO_SETTLEMENT:
            return []
        if not settings.kalshi_api_configured():
            return []

        placed = _load_placed()
        if not placed and not (self.paper and self.paper.db.get_open_paper_trades()):
            return []

        open_tickers = set(placed.keys())
        if self.paper:
            open_tickers.update(
                t["ticker"] for t in self.paper.db.get_open_paper_trades()
            )
        opp_by_ticker = {o.market_ticker: o for o in opportunities}
        exits: list[ExitSignal] = []

        for ticker in list(open_tickers):
            position = self._open_position(ticker)
            if position is None:
                continue

            orderbook = self._reader.get_market_orderbook(ticker)
            if not orderbook:
                continue

            yes_bid = orderbook.get("yes_bid") or 0
            no_bid = 100 - (orderbook.get("yes_ask") or 50)
            side = str(position.get("side", "yes"))
            entry = int(position.get("entry_price_cents", 50))
            count = int(position.get("count", 1))
            bid = int(yes_bid if side == "yes" else no_bid)

            # Take profit only — never dump at a loss on noisy signal flips.
            profit_cents = bid - entry
            if profit_cents >= settings.EXIT_MIN_PROFIT_CENTS:
                exits.append(
                    ExitSignal(
                        ticker=ticker,
                        side=side,
                        count=count,
                        reason=f"take profit ({profit_cents}¢/contract)",
                        entry_price_cents=entry,
                        current_bid_cents=bid,
                    )
                )
                continue

            # Optional stop-loss when bid collapses.
            if profit_cents <= -settings.EXIT_MAX_LOSS_CENTS:
                opp = opp_by_ticker.get(ticker)
                if opp is not None and opp.uncertainty > settings.MAX_UNCERTAINTY_FOR_TRADE:
                    exits.append(
                        ExitSignal(
                            ticker=ticker,
                            side=side,
                            count=count,
                            reason=f"stop loss ({profit_cents}¢/contract)",
                            entry_price_cents=entry,
                            current_bid_cents=bid,
                        )
                    )

        return exits

    def execute_exits(self, exits: list[ExitSignal]) -> list[OrderResult]:
        results: list[OrderResult] = []
        if not exits:
            return results

        if self.live and not settings.AUTO_TRADE_LIVE:
            logger.error("Live exit blocked. Set AUTO_TRADE_LIVE=true.")
            return results

        for i, exit_sig in enumerate(exits):
            if i > 0 and self.live:
                time.sleep(settings.ORDER_DELAY_SECONDS)

            price = max(1, min(99, exit_sig.current_bid_cents))
            body = build_order_body(
                exit_sig.ticker,
                "YES" if exit_sig.side == "yes" else "NO",
                exit_sig.count,
                price,
                action="sell",
            )

            if not self.live:
                logger.info(
                    "[DRY-RUN] Would SELL: %s x%d @ %d¢ — %s",
                    exit_sig.ticker,
                    exit_sig.count,
                    price,
                    exit_sig.reason,
                )
                if self.paper:
                    self.paper.record_early_exit(
                        exit_sig.ticker,
                        exit_price_cents=price,
                        count=exit_sig.count,
                    )
                unmark_placed(exit_sig.ticker)
                results.append(
                    OrderResult(
                        ticker=exit_sig.ticker,
                        pick=exit_sig.reason,
                        action="sell",
                        count=exit_sig.count,
                        side=exit_sig.side,
                        price_cents=price,
                        stake_usd=0,
                        live=False,
                        success=True,
                        order_id="dry-run-exit",
                    )
                )
                continue

            try:
                assert self._client is not None
                resp = self._client.create_order(body)
                order_id = str(
                    resp.get("order", resp).get("order_id", resp.get("order_id", ""))
                )
                unmark_placed(exit_sig.ticker)
                logger.info(
                    "EXIT PLACED: %s x%d @ %d¢ — %s | id=%s",
                    exit_sig.ticker,
                    exit_sig.count,
                    price,
                    exit_sig.reason,
                    order_id,
                )
                results.append(
                    OrderResult(
                        ticker=exit_sig.ticker,
                        pick=exit_sig.reason,
                        action="sell",
                        count=exit_sig.count,
                        side=exit_sig.side,
                        price_cents=price,
                        stake_usd=0,
                        live=True,
                        success=True,
                        order_id=order_id,
                    )
                )
            except Exception as exc:
                logger.error("Exit failed for %s: %s", exit_sig.ticker, exc)

        return results

    def execute(
        self,
        opportunities: list[BetOpportunity],
        *,
        max_orders: int | None = None,
    ) -> list[OrderResult]:
        max_orders = max_orders or settings.MAX_ORDERS_PER_CYCLE
        results: list[OrderResult] = []

        if not settings.kalshi_api_configured():
            logger.warning("Auto-trade skipped: set KALSHI_API_KEY in .env")
            return results

        if self.live and not settings.kalshi_trading_configured():
            logger.warning(
                "Live trade skipped: set KALSHI_PRIVATE_KEY_PATH in .env"
            )
            return results

        if self.paper:
            self.paper.resolve_settled(self._reader)

        if self.live and not settings.AUTO_TRADE_LIVE:
            logger.error(
                "Live trading blocked. Set AUTO_TRADE_LIVE=true in .env to enable."
            )
            return results

        # Check exits first
        exits = self.evaluate_exits(opportunities)
        results.extend(self.execute_exits(exits))

        candidates: list[BetOpportunity] = []
        for opp in opportunities:
            if opp.recommended_stake < settings.MIN_ORDER_STAKE_USD:
                continue
            if already_placed_ticker(opp.market_ticker, self.paper):
                continue
            feats = {
                "model_prob": opp.model_prob,
                "kalshi_prob": opp.kalshi_price / 100,
                "edge": opp.edge_pct / 100,
                "momentum_15m": opp.momentum_15m,
                "ema_trend": opp.ema_trend,
            }
            reason = reject_entry(
                feats=feats,
                direction=opp.direction,
                yes_price_cents=opp.kalshi_price,
                confidence=opp.confidence,
                uncertainty=opp.uncertainty,
                spot=opp.spot_price,
                strike=opp.floor_strike,
            )
            if reason:
                logger.info("Skipping %s at execute: %s", opp.market_ticker, reason)
                continue
            candidates.append(opp)

        to_trade = candidates[:max_orders]
        if to_trade:
            logger.info(
                "Trading top %d of %d confident picks (max_orders=%d)",
                len(to_trade),
                len(candidates),
                max_orders,
            )

        for i, opp in enumerate(to_trade):
            if i > 0 and self.live:
                time.sleep(settings.ORDER_DELAY_SECONDS)

            count = stake_to_contract_count(opp.recommended_stake, opp.contract_price)
            if count <= 0:
                continue

            price_cents = int(max(1, min(99, round(opp.contract_price))))
            body = build_order_body(
                opp.market_ticker, opp.direction, count, price_cents
            )
            side = body["side"]

            if not self.live:
                logger.info(
                    "[DRY-RUN] Would BUY: %s | %s x%d @ %d¢ (~$%.2f) | mkt %s %.0f%%",
                    opp.pick,
                    side.upper(),
                    count,
                    price_cents,
                    count * price_cents / 100,
                    opp.direction,
                    opp.market_implied_prob * 100,
                )
                if self.paper:
                    self.paper.record_entry(
                        opp,
                        count=count,
                        side=side,
                        price_cents=price_cents,
                    )
                results.append(
                    OrderResult(
                        ticker=opp.market_ticker,
                        pick=opp.pick,
                        action="buy",
                        count=count,
                        side=side,
                        price_cents=price_cents,
                        stake_usd=opp.recommended_stake,
                        live=False,
                        success=True,
                        order_id="dry-run",
                    )
                )
                continue

            try:
                assert self._client is not None
                resp = self._client.create_order(body)
                order_id = str(
                    resp.get("order", resp).get("order_id", resp.get("order_id", ""))
                )
                mark_placed(
                    opp.market_ticker,
                    order_id,
                    side=side,
                    entry_price_cents=price_cents,
                    count=count,
                )
                logger.info(
                    "ORDER PLACED: %s | %s x%d @ %d¢ | id=%s",
                    opp.pick,
                    side.upper(),
                    count,
                    price_cents,
                    order_id,
                )
                results.append(
                    OrderResult(
                        ticker=opp.market_ticker,
                        pick=opp.pick,
                        action="buy",
                        count=count,
                        side=side,
                        price_cents=price_cents,
                        stake_usd=opp.recommended_stake,
                        live=True,
                        success=True,
                        order_id=order_id,
                    )
                )
            except Exception as exc:
                logger.error("Order failed for %s: %s", opp.market_ticker, exc)
                results.append(
                    OrderResult(
                        ticker=opp.market_ticker,
                        pick=opp.pick,
                        action="buy",
                        count=count,
                        side=side,
                        price_cents=price_cents,
                        stake_usd=opp.recommended_stake,
                        live=True,
                        success=False,
                        error=str(exc),
                    )
                )

        return results
