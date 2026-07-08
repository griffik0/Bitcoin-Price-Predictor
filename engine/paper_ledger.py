"""Paper-trading ledger for dry-run mode."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone

from config.settings import setup_logging
from data.fetchers.coinbase import CoinbaseClient
from data.fetchers.kalshi import KalshiClient
from engine.settlement import parse_strike_from_pick, resolve_market_result
from engine.types import BetOpportunity
from data.storage.db import Database

logger = setup_logging()


@dataclass
class PaperLedgerSummary:
    starting_bankroll: float
    realized_pnl: float
    open_stake: float
    current_bankroll: float
    resolved_bets: int
    wins: int
    losses: int
    win_rate: float
    open_positions: int

    def format_footer(self, live: bool = False) -> str:
        label = "Live" if live else "Paper"
        if self.resolved_bets == 0 and self.open_positions == 0:
            return f"{label}: no trades yet"
        parts = [
            f"{label} PnL: ${self.realized_pnl:+.2f}",
            f"Bankroll: ${self.current_bankroll:.2f}",
        ]
        if self.resolved_bets:
            parts.append(f"{self.wins}W-{self.losses}L ({self.win_rate:.0%})")
        if self.open_positions:
            parts.append(f"{self.open_positions} open (${self.open_stake:.2f} at risk)")
        return "  |  ".join(parts)


class PaperLedger:
    """Track hypothetical entries, exits, and settlement PnL during dry-run."""

    def __init__(self, starting_bankroll: float) -> None:
        self.db = Database()
        self.db.init_schema()
        self.starting_bankroll = starting_bankroll
        stored = self.db.get_paper_meta("starting_bankroll")
        if stored is None:
            self.db.set_paper_meta("starting_bankroll", str(starting_bankroll))
        else:
            self.starting_bankroll = float(stored)
        self._coinbase: CoinbaseClient | None = None

    def _coinbase_client(self) -> CoinbaseClient:
        if self._coinbase is None:
            self._coinbase = CoinbaseClient()
        return self._coinbase

    def has_open(self, ticker: str) -> bool:
        return self.db.paper_trade_exists(ticker, open_only=True)

    def has_traded(self, ticker: str) -> bool:
        """True if we already have an open or closed paper trade for this contract."""
        return self.db.paper_trade_exists(ticker)

    def record_entry(
        self,
        opp: BetOpportunity,
        *,
        count: int,
        side: str,
        price_cents: int,
    ) -> None:
        if self.has_traded(opp.market_ticker):
            return
        strike = opp.floor_strike if opp.floor_strike else opp.spot_price
        self.db.insert_paper_trade(
            {
                "ticker": opp.market_ticker,
                "pick": opp.pick,
                "direction": opp.direction,
                "side": side,
                "count": count,
                "entry_price_cents": price_cents,
                "stake_usd": count * price_cents / 100,
                "closes_at": opp.closes_at,
                "opened_at": datetime.now(timezone.utc).isoformat(),
                "floor_strike": strike,
                "entry_spot": opp.spot_price,
            }
        )
        logger.info(
            "[PAPER] Opened %s | %s x%d @ %dc (~$%.2f)",
            opp.market_ticker,
            side.upper(),
            count,
            price_cents,
            count * price_cents / 100,
        )

    def record_early_exit(
        self,
        ticker: str,
        *,
        exit_price_cents: int,
        count: int | None = None,
    ) -> float | None:
        open_trades = self.db.get_open_paper_trades()
        trade = next((t for t in open_trades if t["ticker"] == ticker), None)
        if trade is None:
            return None

        contracts = count or int(trade["count"])
        entry = int(trade["entry_price_cents"])
        pnl = contracts * (exit_price_cents - entry) / 100
        self.db.close_paper_trade(
            ticker,
            status="sold",
            exit_price_cents=exit_price_cents,
            pnl_usd=pnl,
            closed_at=datetime.now(timezone.utc).isoformat(),
        )
        logger.info(
            "[PAPER] Closed early %s @ %dc | PnL $%+.2f",
            ticker,
            exit_price_cents,
            pnl,
        )
        return pnl

    def _trade_strike(self, trade: dict, market: dict | None) -> float | None:
        for source in (
            trade.get("floor_strike"),
            market.get("floor_strike") if market else None,
            parse_strike_from_pick(trade.get("pick")),
        ):
            if source is not None:
                try:
                    return float(source)
                except (TypeError, ValueError):
                    continue
        entry_spot = trade.get("entry_spot")
        if entry_spot is not None:
            try:
                return float(entry_spot)
            except (TypeError, ValueError):
                pass
        return None

    def resolve_settled(self, kalshi: KalshiClient) -> int:
        """Check open positions against Kalshi settlement and close resolved ones."""
        open_trades = self.db.get_open_paper_trades()
        if not open_trades:
            return 0

        settled_map: dict[str, str] = {}
        try:
            for market in kalshi.get_resolved_markets(limit=80):
                ticker = market.get("ticker")
                result = str(market.get("result") or "").lower()
                if ticker and result in ("yes", "no"):
                    settled_map[ticker] = result
        except Exception as exc:
            logger.warning("Settled market scan failed: %s", exc)

        coinbase = self._coinbase_client()
        resolved_count = 0
        now = datetime.now(timezone.utc)

        for trade in open_trades:
            ticker = trade["ticker"]
            result = settled_map.get(ticker)

            market = kalshi.get_market(ticker)
            if result is None and market:
                strike = self._trade_strike(trade, market)
                entry_spot = trade.get("entry_spot")
                if entry_spot is not None:
                    try:
                        entry_spot = float(entry_spot)
                    except (TypeError, ValueError):
                        entry_spot = None
                result = resolve_market_result(
                    market,
                    strike=strike,
                    entry_spot=entry_spot,
                    coinbase=coinbase,
                    now=now,
                )

            if result not in ("yes", "no"):
                continue

            side = trade.get("side", "yes")
            won = (result == "yes") if side == "yes" else (result == "no")
            exit_cents = 100 if won else 0
            entry = int(trade["entry_price_cents"])
            count = int(trade["count"])
            pnl = count * (exit_cents - entry) / 100

            close_status = "won" if won else "lost"
            self.db.close_paper_trade(
                ticker,
                status=close_status,
                exit_price_cents=exit_cents,
                pnl_usd=pnl,
                closed_at=datetime.now(timezone.utc).isoformat(),
            )
            resolved_count += 1
            logger.info(
                "[PAPER] Settled %s -> %s | %s | PnL $%+.2f",
                ticker,
                result.upper(),
                close_status.upper(),
                pnl,
            )

        return resolved_count

    def summary(self) -> PaperLedgerSummary:
        stats = self.db.get_paper_trade_summary()
        realized = float(stats["realized_pnl"])
        return PaperLedgerSummary(
            starting_bankroll=self.starting_bankroll,
            realized_pnl=realized,
            open_stake=float(stats["open_stake"]),
            current_bankroll=self.starting_bankroll + realized,
            resolved_bets=int(stats["resolved_bets"]),
            wins=int(stats["wins"]),
            losses=int(stats["losses"]),
            win_rate=float(stats["win_rate"]),
            open_positions=int(stats["open_positions"]),
        )

    def cumulative_pnl_series(self) -> list[float]:
        """Cumulative realized PnL points for charting (starts at 0)."""
        curve = self.db.get_paper_pnl_curve()
        values = [0.0]
        for point in curve:
            values.append(float(point["cumulative_pnl"]))
        realized = self.summary().realized_pnl
        if not values or values[-1] != realized:
            values.append(realized)
        return values

    def reset(self) -> None:
        self.db.clear_paper_trades()
        self.db.set_paper_meta("starting_bankroll", str(self.starting_bankroll))
