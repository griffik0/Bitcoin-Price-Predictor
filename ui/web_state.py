"""Thread-safe snapshot for the web dashboard API."""

from __future__ import annotations

import threading
from datetime import datetime, timezone
from typing import Any


class WebDashboardState:
    """Latest bot metrics exposed to the browser via /api/state."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._data: dict[str, Any] = {
            "updated_at": None,
            "uptime_seconds": 0,
            "spot_price": 0.0,
            "bankroll": 0.0,
            "markets_scanned": 0,
            "opportunities": [],
            "paper": None,
            "backtest": None,
            "trade_mode": "off",
        }

    def update(self, payload: dict[str, Any]) -> None:
        with self._lock:
            self._data.update(payload)
            self._data["updated_at"] = datetime.now(timezone.utc).isoformat()

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            return dict(self._data)
