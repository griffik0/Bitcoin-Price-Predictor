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


def _cents_to_dollars_str(price_cents: int) -> str:
    return f"{max(1, min(99, price_cents)) / 100:.4f}"


def _complement_dollars_str(price_cents: int) -> str:
    clamped = max(1, min(99, price_cents))
    return f"{(100 - clamped) / 100:.4f}"


def _entry_price_from_fill(direction: str, resp: dict, fallback_cents: int) -> int:
    payload = _order_payload(resp)
    avg = payload.get("average_fill_price") or resp.get("average_fill_price")
    if avg is None:
        for key in ("yes_price_dollars", "no_price_dollars"):
            px = payload.get(key)
            if px is not None:
                side_cents = int(round(float(px) * 100))
                if key.startswith("yes"):
                    return side_cents if direction == "YES" else max(1, min(99, 100 - side_cents))
                return side_cents if direction == "NO" else max(1, min(99, 100 - side_cents))
        return fallback_cents
    yes_cents = int(round(float(avg) * 100))
    if direction == "YES":
        return yes_cents
    return max(1, min(99, 100 - yes_cents))


def _order_payload(resp: dict) -> dict:
    order = resp.get("order")
    return order if isinstance(order, dict) else resp


def _filled_contract_count(resp: dict, requested_count: int) -> int:
    payload = _order_payload(resp)
    for key in ("fill_count_fp", "fill_count"):
        raw = payload.get(key)
        if raw is not None:
            try:
                return max(0, int(float(raw)))
            except (TypeError, ValueError):
                pass
    return requested_count


def build_order_body(
    ticker: str,
    direction: str,
    count: int,
    price_cents: int,
    action: str = "buy",
    *,
    time_in_force: str = "good_till_canceled",
) -> dict:
    """Build Kalshi V2 order body for POST /portfolio/events/orders."""
    is_yes = direction == "YES"
    is_buy = action == "buy"
    if is_yes:
        book_side = "bid" if is_buy else "ask"
        price = _cents_to_dollars_str(price_cents)
    else:
        # V2 quotes from the YES book; NO trades use the complement price.
        book_side = "ask" if is_buy else "bid"
        price = _complement_dollars_str(price_cents)
    return {
        "ticker": ticker,
        "side": book_side,
        "count": f"{count:.2f}",
        "price": price,
        "client_order_id": str(uuid.uuid4()),
        "time_in_force": time_in_force,
        "self_trade_prevention_type": "taker_at_cross",
    }


def ask_price_cents(
    direction: str,
    *,
    yes_bid: float | None,
    yes_ask: float | None,
    fallback_cents: float,
) -> int:
    """Best ask (in the direction's own cents) that a buy must lift to fill now.

    Buying YES lifts the YES ask; buying NO lifts the NO ask (= 100 - yes_bid).
    Falls back to the model's contract price when the book is unavailable.
    """
    if direction == "YES":
        base = yes_ask if yes_ask else fallback_cents
    else:
        base = (100 - yes_bid) if yes_bid else fallback_cents
    return int(max(1, min(99, round(base))))


def marketable_limit_cents(ask_cents: int) -> int:
    """Limit price for a marketable buy: a few cents through the ask so the
    order crosses even if the book moves, capped at 99. Kalshi fills at the
    resting (maker) price, so the buffer does not raise the actual cost."""
    return int(min(99, ask_cents + settings.MARKETABLE_BUFFER_CENTS))


def entry_limit_cents(mid_contract_cents: float) -> int:
    """Resting buy limit: a few cents under mid for better upside on fill."""
    mid = int(max(1, min(99, round(mid_contract_cents))))
    discount = max(0, settings.ENTRY_LIMIT_DISCOUNT_CENTS)
    return int(max(1, min(99, mid - discount)))


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
        # Ledger tracks positions/PnL for the dashboard in both dry-run and
        # live modes (live records real fills instead of hypothetical entries).
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
                order_id = str(resp.get("order_id", resp.get("order", {}).get("order_id", "")))
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
            if self.live and self._client:
                self.paper.sync_live_orders(self._reader, self._client)
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

            side = "yes" if opp.direction == "YES" else "no"

            # Limit at the model's mid-based contract price. Rests until filled
            # or the market closes. Predictions already use this same mid.
            mid_cents = int(max(1, min(99, round(opp.contract_price))))
            price_cents = entry_limit_cents(opp.contract_price)
            count = stake_to_contract_count(opp.recommended_stake, price_cents)
            if count <= 0:
                continue

            body = build_order_body(
                opp.market_ticker,
                opp.direction,
                count,
                price_cents,
                action="buy",
                time_in_force="good_till_canceled",
            )

            if not self.live:
                logger.info(
                    "[DRY-RUN] Would BUY: %s | %s x%d @ %d¢ (mid %d¢) "
                    "GTC (~$%.2f) | mkt %s %.0f%%",
                    opp.pick,
                    side.upper(),
                    count,
                    price_cents,
                    mid_cents,
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
                logger.info(
                    "Placing LIVE buy: %s %s x%d limit %d¢ (mid) GTC",
                    opp.market_ticker,
                    side.upper(),
                    count,
                    price_cents,
                )
                resp = self._client.create_order(body)
                payload = _order_payload(resp)
                order_id = str(
                    payload.get("order_id", resp.get("order_id", ""))
                )
                filled_count = _filled_contract_count(resp, 0)
                mark_placed(
                    opp.market_ticker,
                    order_id,
                    side=side,
                    entry_price_cents=price_cents,
                    count=count if filled_count <= 0 else filled_count,
                )
                fill_price = price_cents
                if filled_count > 0:
                    fill_price = _entry_price_from_fill(
                        opp.direction, resp, price_cents
                    )
                if self.paper:
                    self.paper.record_order_placed(
                        opp,
                        count=count,
                        side=side,
                        limit_price_cents=fill_price,
                        mid_price_cents=mid_cents,
                        order_id=order_id,
                        filled_count=filled_count,
                        fill_price_cents=fill_price,
                    )
                if filled_count > 0:
                    logger.info(
                        "ORDER FILLED: %s | %s $%.2f -> %d @ %d¢ | id=%s",
                        opp.pick,
                        side.upper(),
                        opp.recommended_stake,
                        filled_count,
                        fill_price,
                        order_id,
                    )
                else:
                    logger.info(
                        "ORDER PLACED (resting): %s | %s x%d @ %d¢ | id=%s",
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
                        count=filled_count if filled_count > 0 else count,
                        side=side,
                        price_cents=fill_price,
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
