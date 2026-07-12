"""Central configuration for the BTC 15-minute Kalshi betting bot."""

from __future__ import annotations

import logging
import os
from logging.handlers import RotatingFileHandler
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()


def _requests_ssl_verify() -> str | bool:
    if os.getenv("SSL_VERIFY", "true").lower() in ("0", "false", "no"):
        import urllib3

        urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
        for key in ("SSL_CERT_FILE", "REQUESTS_CA_BUNDLE", "CURL_CA_BUNDLE"):
            os.environ.pop(key, None)
        return False

    try:
        import certifi

        bundle = certifi.where()
        os.environ.setdefault("SSL_CERT_FILE", bundle)
        os.environ.setdefault("REQUESTS_CA_BUNDLE", bundle)
        return bundle
    except ImportError:
        return True


REQUESTS_SSL_VERIFY = _requests_ssl_verify()

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# API endpoints
KALSHI_API_BASE = "https://api.elections.kalshi.com/trade-api/v2"
COINBASE_API_BASE = "https://api.exchange.coinbase.com"

# Kalshi BTC 15-minute series
KALSHI_BTC_SERIES = "KXBTC15M"
BTC_PRODUCT = "BTC-USD"

# Credentials (can point to same keys as kalshi-sports-bot)
KALSHI_API_KEY = os.getenv("KALSHI_API_KEY", "").strip()
KALSHI_PRIVATE_KEY_PATH = os.getenv("KALSHI_PRIVATE_KEY_PATH", "").strip()
BANKROLL = float(os.getenv("BANKROLL", "10"))
AUTO_TRADE_LIVE = os.getenv("AUTO_TRADE_LIVE", "false").lower() in (
    "1",
    "true",
    "yes",
)

_KALSHI_KEY_PLACEHOLDERS = frozenset(
    {"", "your_kalshi_api_key_here", "changeme", "xxx"}
)


def kalshi_api_configured() -> bool:
    return KALSHI_API_KEY.lower() not in _KALSHI_KEY_PLACEHOLDERS


def kalshi_trading_configured() -> bool:
    if not kalshi_api_configured():
        return False
    if not KALSHI_PRIVATE_KEY_PATH:
        return False
    return Path(KALSHI_PRIVATE_KEY_PATH).is_file()


# Scheduling — faster refresh for 15-min windows
REFRESH_INTERVAL_SECONDS = int(os.getenv("REFRESH_INTERVAL_SECONDS", "45"))
MIN_EDGE_THRESHOLD = float(os.getenv("MIN_EDGE_THRESHOLD", "0.05"))
MIN_CONFIDENCE_THRESHOLD = float(os.getenv("MIN_CONFIDENCE_THRESHOLD", "0.68"))
MIN_CONFIDENCE_FOR_TRADE = float(
    os.getenv("MIN_CONFIDENCE_FOR_TRADE", os.getenv("MIN_CONFIDENCE_THRESHOLD", "0.68"))
)
MAX_UNCERTAINTY_FOR_TRADE = float(os.getenv("MAX_UNCERTAINTY_FOR_TRADE", "0.50"))
MIN_MARKET_IMPLIED_PROB = float(os.getenv("MIN_MARKET_IMPLIED_PROB", "0.12"))
MIN_MODEL_PROB_YES = float(os.getenv("MIN_MODEL_PROB_YES", "0.52"))
MAX_MODEL_PROB_NO = float(os.getenv("MAX_MODEL_PROB_NO", "0.48"))
MAX_KALSHI_YES_TO_BET_YES = float(os.getenv("MAX_KALSHI_YES_TO_BET_YES", "0.78"))
MIN_KALSHI_YES_TO_BET_NO = float(os.getenv("MIN_KALSHI_YES_TO_BET_NO", "0.22"))
MAX_CONTRACT_PRICE_CENTS = float(os.getenv("MAX_CONTRACT_PRICE_CENTS", "72"))
REQUIRE_STRIKE_ALIGN = os.getenv("REQUIRE_STRIKE_ALIGN", "true").lower() in (
    "1",
    "true",
    "yes",
)
REQUIRE_MOMENTUM_ALIGN = os.getenv("REQUIRE_MOMENTUM_ALIGN", "true").lower() in (
    "1",
    "true",
    "yes",
)
MIN_STRIKE_DIST_FOR_YES = float(os.getenv("MIN_STRIKE_DIST_FOR_YES", "0.0"))
MAX_STRIKE_DIST_FOR_NO = float(os.getenv("MAX_STRIKE_DIST_FOR_NO", "0.0"))
MIN_VOLUME = int(os.getenv("MIN_VOLUME", "100"))

# Entry window: minutes before close (avoid first/last minute noise)
MIN_MINUTES_TO_CLOSE = float(os.getenv("MIN_MINUTES_TO_CLOSE", "2.0"))
MAX_MINUTES_TO_CLOSE = float(os.getenv("MAX_MINUTES_TO_CLOSE", "14.5"))

# Kelly sizing
KELLY_FRACTION = 0.25
MAX_STAKE_PCT_BANKROLL = 0.05
MAX_TOTAL_EXPOSURE_PCT = 0.5
MAX_ORDERS_PER_CYCLE = int(os.getenv("MAX_ORDERS_PER_CYCLE", "1"))
MIN_ORDER_STAKE_USD = float(os.getenv("MIN_ORDER_STAKE_USD", "0.10"))
ORDER_DELAY_SECONDS = float(os.getenv("ORDER_DELAY_SECONDS", "2.0"))
# Cents below mid (opp.contract_price) for resting entry limits. Predictions
# and edge filters still use mid; only the live/dry-run order price is reduced.
ENTRY_LIMIT_DISCOUNT_CENTS = int(os.getenv("ENTRY_LIMIT_DISCOUNT_CENTS", "0"))
# Extra cents added to the ask when placing a marketable buy, so the order
# still crosses if the book ticks against us between the read and the order.
MARKETABLE_BUFFER_CENTS = int(os.getenv("MARKETABLE_BUFFER_CENTS", "3"))

# Exit rules — 15m binaries: hold to settlement avoids panic-selling at 1¢ bids
HOLD_TO_SETTLEMENT = os.getenv("HOLD_TO_SETTLEMENT", "true").lower() in (
    "1",
    "true",
    "yes",
)
EXIT_EDGE_REVERSAL = float(os.getenv("EXIT_EDGE_REVERSAL", "0.03"))
EXIT_MIN_PROFIT_CENTS = int(os.getenv("EXIT_MIN_PROFIT_CENTS", "5"))
EXIT_CONFIDENCE_DROP = float(os.getenv("EXIT_CONFIDENCE_DROP", "0.15"))
EXIT_MAX_LOSS_CENTS = int(os.getenv("EXIT_MAX_LOSS_CENTS", "25"))

WEB_DASHBOARD_PORT = int(os.getenv("WEB_DASHBOARD_PORT", "8765"))

# Paths
MODEL_DIR = str(PROJECT_ROOT / "models" / "saved")
DB_PATH = str(PROJECT_ROOT / "data" / "btc_history.db")
LOG_DIR = str(PROJECT_ROOT / "logs")
LOG_FILE = str(PROJECT_ROOT / "logs" / "app.log")

# HTTP
REQUEST_TIMEOUT_SECONDS = 10
MAX_RETRIES = 3
BACKOFF_BASE_SECONDS = 1.0
KALSHI_PAGE_DELAY_SECONDS = 0.35
KALSHI_RATE_LIMIT_MAX_RETRIES = 8
KALSHI_MAX_PAGES_PER_SERIES = 20

# Training
TRAIN_SPLIT = 0.6
CALIBRATION_SPLIT = 0.2

FEATURE_COLUMNS = [
    "kalshi_prob",
    "model_prob",
    "edge",
    "abs_edge",
    "minutes_to_close",
    "log_volume",
    "bid_ask_spread",
    "strike_distance_pct",
    "rsi_14",
    "macd_histogram",
    "ema_trend",
    "momentum_15m",
    "momentum_1h",
    "price_velocity_1h",
    "volatility_atr_pct",
]

META_FEATURE_COLUMNS = [
    "classifier_prob",
    "ev_pred",
    "uncertainty_score",
    "edge",
    "minutes_to_close",
    "log_volume",
]


def setup_logging() -> logging.Logger:
    import sys

    os.makedirs(LOG_DIR, exist_ok=True)
    logger = logging.getLogger("btc_bot")
    if logger.handlers:
        return logger

    logger.setLevel(logging.INFO)
    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
    )

    file_handler = RotatingFileHandler(
        LOG_FILE, maxBytes=10 * 1024 * 1024, backupCount=3, encoding="utf-8"
    )
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    console = logging.StreamHandler(sys.stdout)
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        except (OSError, ValueError):
            pass
    console.setFormatter(formatter)
    logger.addHandler(console)

    return logger
