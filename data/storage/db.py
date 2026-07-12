"""SQLite storage for historical BTC markets and backtest results."""

from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from typing import Any, Generator, Iterable

from config import settings


class Database:
    """SQLite helper for resolved BTC markets and backtest metrics."""

    def __init__(self, path: str | None = None) -> None:
        self.path = path or settings.DB_PATH

    @contextmanager
    def connect(self) -> Generator[sqlite3.Connection, None, None]:
        conn = sqlite3.connect(self.path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()

    def init_schema(self) -> None:
        with self.connect() as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS resolved_markets (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    ticker TEXT NOT NULL,
                    title TEXT,
                    closes_at TEXT NOT NULL,
                    volume INTEGER,
                    entry_price REAL,
                    resolved_price REAL,
                    resolved_yes INTEGER,
                    model_prob REAL,
                    edge REAL,
                    floor_strike REAL,
                    features_json TEXT,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP
                );

                CREATE INDEX IF NOT EXISTS idx_btc_resolved_closes
                    ON resolved_markets(closes_at);

                CREATE TABLE IF NOT EXISTS backtest_runs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    run_at TEXT NOT NULL,
                    total_return REAL,
                    sharpe_ratio REAL,
                    max_drawdown REAL,
                    win_rate REAL,
                    total_bets INTEGER,
                    notes TEXT
                );

                CREATE TABLE IF NOT EXISTS paper_trades (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    ticker TEXT NOT NULL,
                    pick TEXT,
                    direction TEXT,
                    side TEXT,
                    count INTEGER,
                    entry_price_cents INTEGER,
                    stake_usd REAL,
                    closes_at TEXT,
                    opened_at TEXT NOT NULL,
                    status TEXT NOT NULL DEFAULT 'open',
                    exit_price_cents INTEGER,
                    pnl_usd REAL,
                    closed_at TEXT
                );

                CREATE INDEX IF NOT EXISTS idx_paper_trades_ticker
                    ON paper_trades(ticker);

                CREATE INDEX IF NOT EXISTS idx_paper_trades_status
                    ON paper_trades(status);

                CREATE TABLE IF NOT EXISTS paper_ledger_meta (
                    key TEXT PRIMARY KEY,
                    value TEXT
                );
                """
            )
        self.dedupe_resolved_markets()
        with self.connect() as conn:
            conn.execute(
                """
                CREATE UNIQUE INDEX IF NOT EXISTS idx_btc_resolved_ticker_unique
                ON resolved_markets(ticker)
                """
            )
        self._migrate_paper_trades_table()
        self._migrate_paper_trade_columns()

    def _migrate_paper_trade_columns(self) -> None:
        with self.connect() as conn:
            cols = {
                row[1]
                for row in conn.execute("PRAGMA table_info(paper_trades)").fetchall()
            }
            migrations = {
                "floor_strike": "ALTER TABLE paper_trades ADD COLUMN floor_strike REAL",
                "entry_spot": "ALTER TABLE paper_trades ADD COLUMN entry_spot REAL",
                "order_id": "ALTER TABLE paper_trades ADD COLUMN order_id TEXT",
                "mid_price_cents": "ALTER TABLE paper_trades ADD COLUMN mid_price_cents INTEGER",
                "hypothetical_pnl_usd": (
                    "ALTER TABLE paper_trades ADD COLUMN hypothetical_pnl_usd REAL"
                ),
                "hypothetical_outcome": (
                    "ALTER TABLE paper_trades ADD COLUMN hypothetical_outcome TEXT"
                ),
            }
            for col, sql in migrations.items():
                if col not in cols:
                    conn.execute(sql)

    def _migrate_paper_trades_table(self) -> None:
        """Drop legacy UNIQUE(ticker) so closed trades don't block re-inserts."""
        with self.connect() as conn:
            row = conn.execute(
                "SELECT sql FROM sqlite_master WHERE type='table' AND name='paper_trades'"
            ).fetchone()
            if not row or not row["sql"]:
                return
            if "UNIQUE" not in str(row["sql"]).upper():
                return
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS paper_trades_new (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    ticker TEXT NOT NULL,
                    pick TEXT,
                    direction TEXT,
                    side TEXT,
                    count INTEGER,
                    entry_price_cents INTEGER,
                    stake_usd REAL,
                    closes_at TEXT,
                    opened_at TEXT NOT NULL,
                    status TEXT NOT NULL DEFAULT 'open',
                    exit_price_cents INTEGER,
                    pnl_usd REAL,
                    closed_at TEXT
                );
                INSERT INTO paper_trades_new
                SELECT * FROM paper_trades;
                DROP TABLE paper_trades;
                ALTER TABLE paper_trades_new RENAME TO paper_trades;
                CREATE INDEX IF NOT EXISTS idx_paper_trades_ticker
                    ON paper_trades(ticker);
                CREATE INDEX IF NOT EXISTS idx_paper_trades_status
                    ON paper_trades(status);
                """
            )

    def dedupe_resolved_markets(self) -> int:
        """Keep the newest row per ticker and remove historical duplicates."""
        with self.connect() as conn:
            cur = conn.execute(
                """
                DELETE FROM resolved_markets
                WHERE id NOT IN (
                    SELECT MAX(id) FROM resolved_markets GROUP BY ticker
                )
                """
            )
            return cur.rowcount

    def clear_resolved_markets(self) -> int:
        with self.connect() as conn:
            cur = conn.execute("DELETE FROM resolved_markets")
            return cur.rowcount

    def upsert_resolved_market(self, row: dict[str, Any]) -> None:
        with self.connect() as conn:
            conn.execute(
                """
                INSERT INTO resolved_markets (
                    ticker, title, closes_at, volume,
                    entry_price, resolved_price, resolved_yes,
                    model_prob, edge, floor_strike, features_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(ticker) DO UPDATE SET
                    title = excluded.title,
                    closes_at = excluded.closes_at,
                    volume = excluded.volume,
                    entry_price = excluded.entry_price,
                    resolved_price = excluded.resolved_price,
                    resolved_yes = excluded.resolved_yes,
                    model_prob = excluded.model_prob,
                    edge = excluded.edge,
                    floor_strike = excluded.floor_strike,
                    features_json = excluded.features_json
                """,
                (
                    row["ticker"],
                    row.get("title"),
                    row["closes_at"],
                    row.get("volume"),
                    row.get("entry_price"),
                    row.get("resolved_price"),
                    row.get("resolved_yes"),
                    row.get("model_prob"),
                    row.get("edge"),
                    row.get("floor_strike"),
                    json.dumps(row.get("features", {})),
                ),
            )

    def insert_resolved_batch(self, rows: Iterable[dict[str, Any]]) -> int:
        count = 0
        for row in rows:
            self.upsert_resolved_market(row)
            count += 1
        return count

    def get_resolved_markets(
        self, limit: int | None = None
    ) -> list[dict[str, Any]]:
        query = "SELECT * FROM resolved_markets ORDER BY closes_at ASC"
        if limit:
            query += f" LIMIT {int(limit)}"
        with self.connect() as conn:
            rows = conn.execute(query).fetchall()
        result = []
        for r in rows:
            d = dict(r)
            if d.get("features_json"):
                d["features"] = json.loads(d["features_json"])
            result.append(d)
        return result

    def count_resolved_markets(self) -> int:
        with self.connect() as conn:
            row = conn.execute(
                "SELECT COUNT(*) AS c FROM resolved_markets"
            ).fetchone()
        return int(row["c"]) if row else 0

    def save_backtest_run(self, metrics: dict[str, Any]) -> None:
        with self.connect() as conn:
            conn.execute(
                """
                INSERT INTO backtest_runs (
                    run_at, total_return, sharpe_ratio, max_drawdown,
                    win_rate, total_bets, notes
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    datetime.now(timezone.utc).isoformat(),
                    metrics.get("total_return"),
                    metrics.get("sharpe_ratio"),
                    metrics.get("max_drawdown"),
                    metrics.get("win_rate"),
                    metrics.get("total_bets"),
                    metrics.get("notes"),
                ),
            )

    def get_latest_backtest(self) -> dict[str, Any] | None:
        with self.connect() as conn:
            row = conn.execute(
                "SELECT * FROM backtest_runs ORDER BY id DESC LIMIT 1"
            ).fetchone()
        return dict(row) if row else None

    def get_paper_meta(self, key: str, default: str | None = None) -> str | None:
        with self.connect() as conn:
            row = conn.execute(
                "SELECT value FROM paper_ledger_meta WHERE key = ?", (key,)
            ).fetchone()
        return row["value"] if row else default

    def set_paper_meta(self, key: str, value: str) -> None:
        with self.connect() as conn:
            conn.execute(
                """
                INSERT INTO paper_ledger_meta (key, value) VALUES (?, ?)
                ON CONFLICT(key) DO UPDATE SET value = excluded.value
                """,
                (key, value),
            )

    def paper_trade_exists(self, ticker: str, *, open_only: bool = False) -> bool:
        query = "SELECT 1 FROM paper_trades WHERE ticker = ?"
        params: tuple = (ticker,)
        if open_only:
            query += " AND status = 'open'"
        query += " LIMIT 1"
        with self.connect() as conn:
            row = conn.execute(query, params).fetchone()
        return row is not None

    def insert_paper_trade(self, trade: dict[str, Any]) -> None:
        with self.connect() as conn:
            conn.execute(
                """
                INSERT INTO paper_trades (
                    ticker, pick, direction, side, count, entry_price_cents,
                    stake_usd, closes_at, opened_at, status, floor_strike,
                    entry_spot, order_id, mid_price_cents
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    trade["ticker"],
                    trade.get("pick"),
                    trade.get("direction"),
                    trade.get("side"),
                    trade.get("count"),
                    trade.get("entry_price_cents"),
                    trade.get("stake_usd"),
                    trade.get("closes_at"),
                    trade.get("opened_at"),
                    trade.get("status", "open"),
                    trade.get("floor_strike"),
                    trade.get("entry_spot"),
                    trade.get("order_id"),
                    trade.get("mid_price_cents"),
                ),
            )

    def get_resting_paper_trades(self) -> list[dict[str, Any]]:
        with self.connect() as conn:
            rows = conn.execute(
                """
                SELECT * FROM paper_trades
                WHERE status = 'resting'
                ORDER BY opened_at ASC
                """
            ).fetchall()
        return [dict(r) for r in rows]

    def promote_resting_paper_trade(
        self,
        ticker: str,
        *,
        count: int,
        entry_price_cents: int,
    ) -> None:
        stake = count * entry_price_cents / 100
        with self.connect() as conn:
            conn.execute(
                """
                UPDATE paper_trades
                SET status = 'open',
                    count = ?,
                    entry_price_cents = ?,
                    stake_usd = ?
                WHERE ticker = ? AND status = 'resting'
                """,
                (count, entry_price_cents, stake, ticker),
            )

    def finalize_unfilled_paper_trade(
        self,
        ticker: str,
        *,
        hypothetical_outcome: str,
        hypothetical_pnl_usd: float,
        closed_at: str,
    ) -> None:
        with self.connect() as conn:
            conn.execute(
                """
                UPDATE paper_trades
                SET status = 'unfilled',
                    hypothetical_outcome = ?,
                    hypothetical_pnl_usd = ?,
                    closed_at = ?
                WHERE ticker = ? AND status = 'resting'
                """,
                (hypothetical_outcome, hypothetical_pnl_usd, closed_at, ticker),
            )

    def get_open_paper_trades(self) -> list[dict[str, Any]]:
        with self.connect() as conn:
            rows = conn.execute(
                "SELECT * FROM paper_trades WHERE status = 'open' ORDER BY opened_at ASC"
            ).fetchall()
        return [dict(r) for r in rows]

    def close_paper_trade(
        self,
        ticker: str,
        *,
        status: str,
        exit_price_cents: int,
        pnl_usd: float,
        closed_at: str,
    ) -> None:
        with self.connect() as conn:
            conn.execute(
                """
                UPDATE paper_trades
                SET status = ?, exit_price_cents = ?, pnl_usd = ?, closed_at = ?
                WHERE ticker = ? AND status = 'open'
                """,
                (status, exit_price_cents, pnl_usd, closed_at, ticker),
            )

    def get_paper_trade_summary(self) -> dict[str, Any]:
        with self.connect() as conn:
            closed = conn.execute(
                """
                SELECT
                    COUNT(*) AS resolved,
                    SUM(CASE WHEN pnl_usd > 0 THEN 1 ELSE 0 END) AS wins,
                    SUM(CASE WHEN pnl_usd <= 0 THEN 1 ELSE 0 END) AS losses,
                    COALESCE(SUM(pnl_usd), 0) AS total_pnl
                FROM paper_trades
                WHERE status IN ('won', 'lost', 'sold')
                """
            ).fetchone()
            open_row = conn.execute(
                """
                SELECT COUNT(*) AS open_count,
                       COALESCE(SUM(stake_usd), 0) AS open_stake
                FROM paper_trades WHERE status = 'open'
                """
            ).fetchone()
            resting_row = conn.execute(
                "SELECT COUNT(*) AS c FROM paper_trades WHERE status = 'resting'"
            ).fetchone()
            unfilled_row = conn.execute(
                """
                SELECT
                    COUNT(*) AS c,
                    SUM(CASE WHEN hypothetical_outcome = 'would_win' THEN 1 ELSE 0 END)
                        AS would_wins,
                    SUM(CASE WHEN hypothetical_outcome = 'would_lose' THEN 1 ELSE 0 END)
                        AS would_losses,
                    COALESCE(SUM(hypothetical_pnl_usd), 0) AS hypo_pnl
                FROM paper_trades
                WHERE status = 'unfilled'
                """
            ).fetchone()
        resolved = int(closed["resolved"] or 0)
        wins = int(closed["wins"] or 0)
        return {
            "resolved_bets": resolved,
            "wins": wins,
            "losses": int(closed["losses"] or 0),
            "win_rate": wins / resolved if resolved else 0.0,
            "realized_pnl": float(closed["total_pnl"] or 0),
            "open_positions": int(open_row["open_count"] or 0),
            "open_stake": float(open_row["open_stake"] or 0),
            "resting_orders": int(resting_row["c"] or 0),
            "unfilled_orders": int(unfilled_row["c"] or 0),
            "would_wins": int(unfilled_row["would_wins"] or 0),
            "would_losses": int(unfilled_row["would_losses"] or 0),
            "hypothetical_pnl": float(unfilled_row["hypo_pnl"] or 0),
        }

    def get_recent_paper_trades(self, limit: int = 30) -> list[dict[str, Any]]:
        with self.connect() as conn:
            rows = conn.execute(
                """
                SELECT ticker, pick, direction, side, count, entry_price_cents,
                       mid_price_cents, stake_usd, status, pnl_usd,
                       hypothetical_pnl_usd, hypothetical_outcome,
                       order_id, opened_at, closed_at
                FROM paper_trades
                ORDER BY COALESCE(closed_at, opened_at) DESC, id DESC
                LIMIT ?
                """,
                (limit,),
            ).fetchall()
        return [dict(r) for r in rows]

    def get_paper_pnl_curve(self) -> list[dict[str, Any]]:
        """Closed paper trades in time order with running cumulative PnL."""
        with self.connect() as conn:
            rows = conn.execute(
                """
                SELECT closed_at, pnl_usd
                FROM paper_trades
                WHERE status != 'open'
                  AND closed_at IS NOT NULL
                  AND pnl_usd IS NOT NULL
                ORDER BY closed_at ASC, id ASC
                """
            ).fetchall()
        cumulative = 0.0
        curve: list[dict[str, Any]] = []
        for row in rows:
            cumulative += float(row["pnl_usd"])
            curve.append(
                {
                    "closed_at": row["closed_at"],
                    "pnl_usd": float(row["pnl_usd"]),
                    "cumulative_pnl": cumulative,
                }
            )
        return curve

    def clear_paper_trades(self) -> int:
        with self.connect() as conn:
            cur = conn.execute("DELETE FROM paper_trades")
            conn.execute("DELETE FROM paper_ledger_meta")
            return cur.rowcount
