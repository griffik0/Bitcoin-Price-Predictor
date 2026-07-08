"""Rich live terminal dashboard for BTC 15-minute betting."""

from __future__ import annotations

import json
import shutil
import time
from datetime import datetime, timezone

from rich.console import Console
from rich.layout import Layout
from rich.live import Live
from rich.panel import Panel
from rich.table import Table

from config import settings
from config.settings import setup_logging
from data.storage.db import Database
from engine.pipeline import InferencePipeline
from engine.trader import KalshiTrader
from ui.pnl_chart import format_uptime
from ui.web_server import WebDashboardServer
from ui.web_state import WebDashboardState

logger = setup_logging()


class Dashboard:
    def __init__(
        self,
        bankroll: float,
        refresh_interval: int | None = None,
        trader: KalshiTrader | None = None,
        max_orders: int | None = None,
        *,
        trade_mode: str = "off",
        enable_web: bool = True,
        web_port: int | None = None,
    ) -> None:
        self.bankroll = bankroll
        self.trader = trader
        self.max_orders = max_orders
        self.trade_mode = trade_mode
        self.refresh_interval = (
            refresh_interval
            if refresh_interval is not None
            else settings.REFRESH_INTERVAL_SECONDS
        )
        self.session_started_at = datetime.now(timezone.utc)
        term = shutil.get_terminal_size(fallback=(120, 36))
        self.console = Console(width=max(term.columns, 100))
        self.db = Database()
        self.web_state = WebDashboardState()
        self.web_server: WebDashboardServer | None = None
        if enable_web:
            port = web_port or settings.WEB_DASHBOARD_PORT
            self.web_server = WebDashboardServer(self.web_state, port)
            self.web_server.start()

    @property
    def web_url(self) -> str | None:
        return self.web_server.url if self.web_server else None

    def _refresh_console_width(self) -> None:
        term = shutil.get_terminal_size(fallback=(120, 36))
        self.console.width = max(term.columns, 100)

    def _uptime_seconds(self) -> float:
        return (
            datetime.now(timezone.utc) - self.session_started_at
        ).total_seconds()

    def _uptime_text(self) -> str:
        return format_uptime(self._uptime_seconds())

    def _serialize_opportunities(self, opportunities: list) -> list[dict]:
        return [
            {
                "market_ticker": o.market_ticker,
                "pick": o.pick,
                "direction": o.direction,
                "kalshi_price": o.kalshi_price,
                "contract_price": o.contract_price,
                "market_implied_prob": o.market_implied_prob,
                "model_prob": o.model_prob,
                "edge_pct": o.edge_pct,
                "confidence": o.confidence,
                "uncertainty": o.uncertainty,
                "recommended_stake": o.recommended_stake,
                "minutes_to_close": o.minutes_to_close,
            }
            for o in opportunities
        ]

    def _paper_web_payload(self) -> dict | None:
        if not self.trader or not self.trader.paper:
            return None
        paper = self.trader.paper
        summary = paper.summary()
        curve = paper.db.get_paper_pnl_curve()
        history = [{"at": "start", "cumulative": 0.0, "pnl": 0.0}]
        for point in curve:
            history.append(
                {
                    "at": point["closed_at"],
                    "pnl": point["pnl_usd"],
                    "cumulative": point["cumulative_pnl"],
                }
            )
        return {
            "starting_bankroll": summary.starting_bankroll,
            "realized_pnl": summary.realized_pnl,
            "current_bankroll": summary.current_bankroll,
            "resolved_bets": summary.resolved_bets,
            "wins": summary.wins,
            "losses": summary.losses,
            "win_rate": summary.win_rate,
            "open_positions": summary.open_positions,
            "open_stake": summary.open_stake,
            "pnl_history": history,
            "trades": paper.db.get_recent_paper_trades(),
        }

    def _push_web_state(
        self,
        opportunities: list,
        markets_scanned: int,
        spot_price: float,
        backtest: dict | None,
    ) -> None:
        if not self.web_server:
            return
        self.web_state.update(
            {
                "uptime_seconds": self._uptime_seconds(),
                "spot_price": spot_price,
                "bankroll": self.bankroll,
                "markets_scanned": markets_scanned,
                "opportunities": self._serialize_opportunities(opportunities),
                "paper": self._paper_web_payload(),
                "backtest": backtest,
                "trade_mode": self.trade_mode,
                "web_url": self.web_url,
            }
        )

    def _row_style(self, confidence: float, uncertainty: float) -> str:
        if confidence > 0.75 and uncertainty < 0.4:
            return "green"
        if confidence >= settings.MIN_CONFIDENCE_THRESHOLD:
            return "yellow"
        return "dim"

    @staticmethod
    def _short_pick(text: str, max_len: int = 52) -> str:
        if len(text) <= max_len:
            return text
        return text[: max_len - 3] + "..."

    def _build_table(
        self,
        opportunities: list,
        markets_scanned: int,
        spot_price: float,
        last_updated: str,
        backtest: dict | None,
        paper_summary: str | None = None,
    ) -> Layout:
        self._refresh_console_width()
        term = shutil.get_terminal_size(fallback=(120, 36))
        lines = max(term.lines, 20)
        header_h = 4
        footer_h = 3
        body_h = max(10, lines - header_h - footer_h)

        layout = Layout()
        header_text = (
            f"[bold]BTC 15-Min Kalshi Bot[/bold]  |  "
            f"Uptime: {self._uptime_text()}  |  "
            f"BTC: ${spot_price:,.2f}  |  "
            f"Updated: {last_updated}\n"
            f"Bankroll: ${self.bankroll:,.2f}  |  "
            f"Markets: {markets_scanned}  |  "
            f"Picks: {len(opportunities)}  |  "
            f"Entry: {settings.MIN_MINUTES_TO_CLOSE:.0f}-"
            f"{settings.MAX_MINUTES_TO_CLOSE:.0f}m  |  "
            f"Min conf: {settings.MIN_CONFIDENCE_THRESHOLD:.2f}"
        )
        if self.web_url:
            header_text += f"\n[dim]Web dashboard: {self.web_url}[/dim]"

        layout.split_column(
            Layout(Panel(header_text, style="bold cyan"), size=header_h),
            Layout(name="body", size=body_h),
            Layout(name="footer", size=footer_h),
        )

        table = Table(
            show_header=True,
            header_style="bold magenta",
            expand=True,
            show_lines=False,
        )
        table.add_column("Pick", ratio=2, no_wrap=True, overflow="ellipsis")
        table.add_column("Dir", width=4)
        table.add_column("Mkt", justify="right", width=7)
        table.add_column("Model", justify="right", width=7)
        table.add_column("Edge", justify="right", width=7)
        table.add_column("Conf", justify="right", width=5)
        table.add_column("Risk", justify="right", width=5)
        table.add_column("Stake", justify="right", width=7)
        table.add_column("Close", width=5)

        total_exposure = 0.0
        for opp in opportunities:
            style = self._row_style(opp.confidence, opp.uncertainty)
            total_exposure += opp.recommended_stake
            table.add_row(
                self._short_pick(opp.pick),
                opp.direction,
                f"{opp.contract_price:.0f}c",
                f"{opp.model_prob:.0%}",
                f"{opp.edge_pct:+.1f}%",
                f"{opp.confidence:.2f}",
                f"{opp.uncertainty:.2f}",
                f"${opp.recommended_stake:.2f}",
                f"{opp.minutes_to_close:.0f}m",
                style=style,
            )

        if not opportunities:
            table.add_row(
                "No picks this cycle - waiting for edge",
                "-", "-", "-", "-", "-", "-", "-", "-",
                style="dim",
            )

        layout["body"].update(
            Panel(table, title="BTC 15m Recommendations", expand=True)
        )

        exposure_pct = 100 * total_exposure / self.bankroll if self.bankroll else 0
        footer = (
            f"Exposure: ${total_exposure:.2f} ({exposure_pct:.1f}%)  |  "
            f"Refresh: {self.refresh_interval}s"
        )
        if backtest:
            footer += (
                f"  |  Backtest: {backtest.get('win_rate', 0):.1%} win, "
                f"{backtest.get('total_return', 0):.1%} return"
            )
        if paper_summary:
            footer += f"  |  {paper_summary}"
        if self.web_url:
            footer += f"  |  Charts: {self.web_url}"
        layout["footer"].update(Panel(footer, style="dim"))
        return layout

    def run(
        self,
        pipeline: InferencePipeline,
        no_dashboard: bool = False,
    ) -> None:
        backtest = self.db.get_latest_backtest()

        if no_dashboard:
            self._json_loop(pipeline)
            return

        with Live(
            console=self.console,
            refresh_per_second=2,
            screen=True,
        ) as live:
            while True:
                try:
                    opportunities = pipeline.run(self.bankroll)
                    if self.trader:
                        self.trader.execute(
                            opportunities, max_orders=self.max_orders
                        )
                    paper_summary = None
                    if self.trader and self.trader.paper:
                        paper_summary = self.trader.paper.summary().format_footer(
                            live=self.trade_mode == "live"
                        )
                    last_updated = datetime.now(timezone.utc).strftime(
                        "%H:%M:%S UTC"
                    )
                    self._push_web_state(
                        opportunities,
                        pipeline.markets_scanned,
                        pipeline.spot_price,
                        backtest,
                    )
                    layout = self._build_table(
                        opportunities,
                        pipeline.markets_scanned,
                        pipeline.spot_price,
                        last_updated,
                        backtest,
                        paper_summary,
                    )
                    live.update(layout)
                except FileNotFoundError as exc:
                    live.update(Panel(str(exc), style="bold red"))
                except Exception as exc:
                    logger.error("Dashboard cycle error: %s", exc)
                    live.update(Panel(f"Error (retrying): {exc}", style="yellow"))
                time.sleep(self.refresh_interval)

    def _json_loop(self, pipeline: InferencePipeline) -> None:
        while True:
            try:
                opportunities = pipeline.run(self.bankroll)
                if self.trader:
                    self.trader.execute(opportunities, max_orders=self.max_orders)
                self._push_web_state(
                    opportunities,
                    pipeline.markets_scanned,
                    pipeline.spot_price,
                    self.db.get_latest_backtest(),
                )
                payload = self.web_state.snapshot()
                print(json.dumps(payload, indent=2))
            except Exception as exc:
                logger.error("JSON cycle error: %s", exc)
            time.sleep(self.refresh_interval)
