"""Replay model confidence for historical live trades using Coinbase candles."""
from __future__ import annotations

import re
import sqlite3
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from config import settings
from data.fetchers.coinbase import CoinbaseClient
from data.processors.features import build_historical_seed_features
from models.classifier import BetClassifier
from models.ev_regressor import EVRegressor
from models.meta_learner import MetaLearner
from models.risk_scorer import RiskScorer

TS_RE = re.compile(r"^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})")
PLACE = re.compile(
    r"Placing LIVE buy: (KXBTC15M-\S+) (YES|NO) x(\d+) limit (\d+)"
)
SETTLE = re.compile(
    r"\[PAPER\] Settled (KXBTC15M-\S+) -> (YES|NO) \| (WON|LOST) \| PnL \$([+-]?[0-9.]+)"
)
UNFILL = re.compile(
    r"\[LIVE\] Unfilled (KXBTC15M-\S+) -> .* \| (would_win|would_lose) \| hypo PnL \$([+-]?[0-9.]+)",
    re.I,
)
PICK_SPOT = re.compile(r"spot \$([0-9,]+\.\d+)")
PICK_STRIKE = re.compile(r"strike \$([0-9,]+\.\d+)")


@dataclass
class Trade:
    ticker: str
    side: str
    placed_at: datetime
    limit_cents: int
    outcome: str | None = None
    pnl: float | None = None
    spot: float | None = None
    strike: float | None = None
    source: str = ""
    confidence: float | None = None
    edge: float | None = None


def parse_ts(line: str) -> datetime | None:
    m = TS_RE.match(line)
    if not m:
        return None
    return datetime.strptime(m.group(1), "%Y-%m-%d %H:%M:%S").replace(
        tzinfo=timezone.utc
    )


def minutes_to_close(ticker: str, placed_at: datetime) -> float:
    try:
        part = ticker.split("-")[1]
        hhmm = part[-4:]
        close_dt = placed_at.replace(
            hour=int(hhmm[:2]), minute=int(hhmm[2:]), second=0, microsecond=0
        )
        return max(0.5, (close_dt - placed_at).total_seconds() / 60.0)
    except Exception:
        return 8.0


def load_from_logs(paths: list[Path]) -> dict[str, Trade]:
    trades: dict[str, Trade] = {}
    for path in paths:
        if not path.is_file():
            continue
        pending: Trade | None = None
        for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
            ts = parse_ts(line)
            if ts is None:
                continue
            m = PLACE.search(line)
            if m:
                t = Trade(
                    ticker=m.group(1),
                    side=m.group(2),
                    placed_at=ts,
                    limit_cents=int(m.group(4)),
                    source=str(path),
                )
                trades[t.ticker] = t
                pending = t
                continue
            if pending and (
                "ORDER FILLED:" in line
                or "ORDER PLACED (resting):" in line
                or "BTC 15m:" in line
            ):
                ms = PICK_SPOT.search(line)
                mk = PICK_STRIKE.search(line)
                if ms and pending.spot is None:
                    pending.spot = float(ms.group(1).replace(",", ""))
                if mk and pending.strike is None:
                    pending.strike = float(mk.group(1).replace(",", ""))
            m = SETTLE.search(line)
            if m and m.group(1) in trades:
                trades[m.group(1)].outcome = m.group(3).lower()
                trades[m.group(1)].pnl = float(m.group(4))
            m = UNFILL.search(line)
            if m and m.group(1) in trades:
                trades[m.group(1)].outcome = m.group(2).lower()
                trades[m.group(1)].pnl = float(m.group(3))
    return trades


def enrich_from_db(trades: dict[str, Trade], db_paths: list[Path]) -> None:
    for db in db_paths:
        if not db.is_file():
            continue
        conn = sqlite3.connect(db)
        conn.row_factory = sqlite3.Row
        for row in conn.execute("SELECT * FROM paper_trades"):
            r = dict(row)
            ticker = r["ticker"]
            if ticker not in trades and not r.get("order_id"):
                continue
            if ticker not in trades:
                opened = r.get("opened_at") or ""
                try:
                    placed = datetime.fromisoformat(opened.replace("Z", "+00:00"))
                except Exception:
                    continue
                if placed.tzinfo is None:
                    placed = placed.replace(tzinfo=timezone.utc)
                side = (r.get("direction") or r.get("side") or "yes").upper()
                if side.startswith("Y"):
                    side = "YES"
                else:
                    side = "NO"
                trades[ticker] = Trade(
                    ticker=ticker,
                    side=side,
                    placed_at=placed,
                    limit_cents=int(r["entry_price_cents"]),
                    source=str(db),
                )
            t = trades[ticker]
            if t.spot is None and r.get("entry_spot") is not None:
                t.spot = float(r["entry_spot"])
            if t.strike is None and r.get("floor_strike") is not None:
                t.strike = float(r["floor_strike"])
            if t.outcome is None:
                st = (r.get("status") or "").lower()
                if st in ("won", "lost"):
                    t.outcome = st
                    t.pnl = float(r["pnl_usd"]) if r.get("pnl_usd") is not None else None
                elif st == "unfilled":
                    t.outcome = (r.get("hypothetical_outcome") or "").lower() or None
                    if r.get("hypothetical_pnl_usd") is not None:
                        t.pnl = float(r["hypothetical_pnl_usd"])
            if t.side in ("YES", "NO"):
                pass
            elif str(t.side).lower().startswith("y"):
                t.side = "YES"
            else:
                t.side = "NO"
        conn.close()


def score_trade(trade: Trade, coinbase: CoinbaseClient, models: tuple) -> bool:
    clf, ev, risk, meta = models
    if trade.strike is None:
        return False
    if trade.side == "YES":
        yes_mid = float(trade.limit_cents)
    else:
        yes_mid = float(100 - trade.limit_cents)

    end = trade.placed_at
    try:
        c1m = coinbase.get_candles_before(end, 60, 120)
        time.sleep(0.15)
        c5m = coinbase.get_candles_before(end, 300, 60)
        time.sleep(0.15)
        c15m = coinbase.get_candles_before(end, 900, 48)
        time.sleep(0.15)
        c1h = coinbase.get_candles_before(end, 3600, 24)
        time.sleep(0.15)
    except Exception as exc:
        print(f"  candle fail {trade.ticker}: {exc}")
        return False

    # Prefer trade spot for strike-distance consistency with the live decision
    if trade.spot and c1m:
        c1m = list(c1m)
        c1m[-1] = dict(c1m[-1])
        c1m[-1]["close"] = trade.spot

    feats = build_historical_seed_features(
        entry_price_cents=yes_mid,
        volume=1000,
        candles_1m=c1m,
        candles_15m=c15m,
        candles_1h=c1h,
        strike=trade.strike,
        minutes_to_close=minutes_to_close(trade.ticker, trade.placed_at),
    )
    if feats is None:
        return False

    X = np.array([[feats[c] for c in settings.FEATURE_COLUMNS]], dtype=float)
    Xdf = pd.DataFrame(X, columns=settings.FEATURE_COLUMNS)
    clf_p = clf.predict_proba(X)
    ev_p = ev.predict(Xdf)
    unc = risk.uncertainty(X)
    conf = float(
        meta.predict_score(
            np.column_stack(
                [
                    clf_p,
                    ev_p,
                    unc,
                    [feats["edge"]],
                    [feats["minutes_to_close"]],
                    [feats["log_volume"]],
                ]
            )
        )[0]
    )
    trade.confidence = conf
    trade.edge = float(feats["edge"])
    return True


def main() -> None:
    paths = [
        ROOT / "logs" / "app.log",
        ROOT.parent / "btc-price-bot-test2-2-dupe" / "logs" / "app.log",
    ]
    dbs = [
        ROOT / "data" / "btc_history.db",
        ROOT.parent / "btc-price-bot-test2-2-dupe" / "data" / "btc_history.db",
    ]
    print("Loading live trades...")
    trades = load_from_logs(paths)
    enrich_from_db(trades, dbs)
    with_outcome = [t for t in trades.values() if t.outcome]
    print(f"live trades={len(trades)} with outcome={len(with_outcome)}")

    print("Loading models...")
    clf, ev, risk, meta = BetClassifier(), EVRegressor(), RiskScorer(), MetaLearner()
    for m in (clf, ev, risk, meta):
        m.load()
    models = (clf, ev, risk, meta)
    coinbase = CoinbaseClient()

    scored: list[Trade] = []
    for i, t in enumerate(with_outcome):
        ok = score_trade(t, coinbase, models)
        if ok:
            scored.append(t)
        if (i + 1) % 10 == 0:
            print(f"  {i+1}/{len(with_outcome)} scored={len(scored)}")

    print(f"Done. scored={len(scored)}")
    if not scored:
        return

    top = sorted(scored, key=lambda t: t.confidence or 0, reverse=True)[:15]
    print("\n=== TOP 15 LIVE TRADES BY REPLAYED CONFIDENCE ===")
    print(
        "Exact live confidence was never logged on fills. Scores below are\n"
        "replayed with the same models + Coinbase candles at order time +\n"
        "the trade's spot/strike/limit mid. Close to live, not bit-identical.\n"
    )
    w = l = ww = wl = 0
    for i, t in enumerate(top, 1):
        src = "dupe" if "dupe" in t.source else "test2-2"
        out = t.outcome or "?"
        if out == "won":
            w += 1
        elif out == "lost":
            l += 1
        elif out == "would_win":
            ww += 1
        elif out == "would_lose":
            wl += 1
        pnl = f"{t.pnl:+.2f}" if t.pnl is not None else "n/a"
        print(
            f"{i:2d}. conf={t.confidence:.3f}  {out:10}  {t.side:3} "
            f"@{t.limit_cents:2d}c  pnl={pnl:>7}  edge={t.edge:+.1%}  "
            f"{t.ticker[-18:]}  [{src}]  {t.placed_at:%Y-%m-%d %H:%M} UTC"
        )
    print(
        f"\nAmong top 15: {w}W-{l}L filled | "
        f"{ww} would_win / {wl} would_lose unfilled"
    )
    filled = [t for t in top if t.outcome in ("won", "lost")]
    if filled:
        fw = sum(1 for t in filled if t.outcome == "won")
        print(f"Filled-only: {fw}/{len(filled)} = {fw/len(filled):.0%} win rate")


if __name__ == "__main__":
    main()
