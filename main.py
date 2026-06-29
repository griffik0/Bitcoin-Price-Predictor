"""Entry point: live BTC 15-minute Kalshi betting loop."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from config import settings
from config.settings import setup_logging
from engine.pipeline import InferencePipeline
from engine.trader import KalshiTrader
from ui.dashboard import Dashboard

logger = setup_logging()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Kalshi BTC 15-minute price betting bot"
    )
    parser.add_argument(
        "--bankroll",
        type=float,
        default=settings.BANKROLL,
        help=f"Bankroll in dollars (default: {settings.BANKROLL})",
    )
    parser.add_argument(
        "--refresh",
        type=int,
        default=settings.REFRESH_INTERVAL_SECONDS,
        help="Refresh interval in seconds",
    )
    parser.add_argument(
        "--no-dashboard",
        action="store_true",
        help="Print JSON recommendations instead of Rich UI",
    )
    parser.add_argument(
        "--trade",
        choices=("off", "dry-run", "live"),
        default="off",
        help="off=advice only; dry-run=log orders; live=place on Kalshi",
    )
    parser.add_argument(
        "--max-orders",
        type=int,
        default=None,
        help="Max orders per refresh cycle",
    )
    parser.add_argument(
        "--reset-paper",
        action="store_true",
        help="Clear dry-run paper trading history before starting",
    )
    parser.add_argument(
        "--web-port",
        type=int,
        default=settings.WEB_DASHBOARD_PORT,
        help=f"Web dashboard port (default: {settings.WEB_DASHBOARD_PORT})",
    )
    parser.add_argument(
        "--no-web",
        action="store_true",
        help="Disable the browser dashboard",
    )
    args = parser.parse_args()

    pipeline = InferencePipeline()
    trader = None
    if args.trade != "off":
        trader = KalshiTrader(live=args.trade == "live", bankroll=args.bankroll)
        if args.reset_paper and trader.paper:
            trader.paper.reset()
            logger.info("Paper trading history cleared")
        if args.trade == "live":
            if not settings.kalshi_trading_configured():
                logger.error(
                    "Live trade requires KALSHI_API_KEY and KALSHI_PRIVATE_KEY_PATH"
                )
                sys.exit(1)
            if not settings.AUTO_TRADE_LIVE:
                logger.error(
                    "Live trade blocked. Set AUTO_TRADE_LIVE=true in .env."
                )
                sys.exit(1)
            logger.warning("LIVE TRADING ENABLED — real money orders will be sent")

    dashboard = Dashboard(
        bankroll=args.bankroll,
        refresh_interval=args.refresh,
        trader=trader,
        max_orders=args.max_orders,
        trade_mode=args.trade,
        enable_web=not args.no_web,
        web_port=args.web_port,
    )
    dashboard.run(pipeline, no_dashboard=args.no_dashboard)


if __name__ == "__main__":
    main()
