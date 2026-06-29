"""Kalshi RSA-PSS request signing for authenticated API calls."""

from __future__ import annotations

import base64
import time
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import requests
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import padding

from config import settings
from config.settings import setup_logging

logger = setup_logging()

_ORDERS_PATH = "/portfolio/orders"
_POSITIONS_PATH = "/portfolio/positions"


def load_private_key(path: str | Path):
    key_path = Path(path)
    if not key_path.is_file():
        raise FileNotFoundError(f"Kalshi private key not found: {key_path}")
    pem = key_path.read_bytes()
    return serialization.load_pem_private_key(pem, password=None)


def sign_request(private_key, timestamp: str, method: str, path: str) -> str:
    message = f"{timestamp}{method.upper()}{path}".encode("utf-8")
    signature = private_key.sign(
        message,
        padding.PSS(
            mgf=padding.MGF1(hashes.SHA256()),
            salt_length=padding.PSS.DIGEST_LENGTH,
        ),
        hashes.SHA256(),
    )
    return base64.b64encode(signature).decode("utf-8")


def auth_headers(private_key, api_key_id: str, method: str, path: str) -> dict[str, str]:
    timestamp = str(int(time.time() * 1000))
    signature = sign_request(private_key, timestamp, method, path)
    return {
        "KALSHI-ACCESS-KEY": api_key_id,
        "KALSHI-ACCESS-SIGNATURE": signature,
        "KALSHI-ACCESS-TIMESTAMP": timestamp,
        "Content-Type": "application/json",
    }


def signed_path(base_url: str, relative_path: str) -> str:
    return urlparse(f"{base_url.rstrip('/')}{relative_path}").path


class KalshiTradingClient:
    """Authenticated Kalshi client for portfolio, orders, and positions."""

    def __init__(self) -> None:
        if not settings.kalshi_trading_configured():
            raise RuntimeError(
                "Kalshi trading not configured. Set KALSHI_API_KEY and "
                "KALSHI_PRIVATE_KEY_PATH in .env."
            )
        self.base_url = settings.KALSHI_API_BASE.rstrip("/")
        self.api_key_id = settings.KALSHI_API_KEY
        self.private_key = load_private_key(settings.KALSHI_PRIVATE_KEY_PATH)
        self.session = requests.Session()
        self.session.verify = settings.REQUESTS_SSL_VERIFY

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
        relative_path: str,
        *,
        json_body: dict[str, Any] | None = None,
        params: dict[str, Any] | None = None,
    ) -> requests.Response:
        path = signed_path(self.base_url, relative_path.split("?")[0])
        headers = auth_headers(self.private_key, self.api_key_id, method, path)
        url = f"{self.base_url}{relative_path}"
        return self.session.request(
            method,
            url,
            headers=headers,
            json=json_body,
            params=params,
            timeout=settings.REQUEST_TIMEOUT_SECONDS,
        )

    def get_balance(self) -> dict[str, Any]:
        response = self._request("GET", "/portfolio/balance")
        response.raise_for_status()
        return response.json()

    def get_positions(self, ticker: str | None = None) -> list[dict[str, Any]]:
        params: dict[str, Any] = {}
        if ticker:
            params["ticker"] = ticker
        response = self._request("GET", _POSITIONS_PATH, params=params)
        response.raise_for_status()
        data = response.json()
        return data.get("market_positions", data.get("positions", []))

    def create_order(self, body: dict[str, Any]) -> dict[str, Any]:
        last_error: Exception | None = None
        for attempt in range(settings.KALSHI_RATE_LIMIT_MAX_RETRIES):
            response = self._request("POST", _ORDERS_PATH, json_body=body)
            if response.status_code in (200, 201):
                return response.json()
            if response.status_code == 429:
                wait = self._rate_limit_wait(response, attempt + 1)
                logger.warning("Order rate limit; sleeping %.1fs", wait)
                time.sleep(wait)
                continue
            last_error = RuntimeError(
                f"Kalshi order failed ({response.status_code}): {response.text}"
            )
            break
        raise last_error or RuntimeError("Kalshi order failed after retries")
