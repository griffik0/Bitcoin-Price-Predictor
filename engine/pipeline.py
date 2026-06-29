"""End-to-end inference pipeline for BTC 15-minute markets."""

from __future__ import annotations

import numpy as np
import pandas as pd

from config import settings
from config.settings import setup_logging
from data.fetchers.coinbase import CoinbaseClient
from data.fetchers.kalshi import KalshiClient
from data.processors.cleaner import (
    contract_price_cents,
    implied_prob_for_direction,
    kalshi_price_to_prob,
)
from data.processors.features import build_features
from engine.entry_filters import reject_entry
from engine.kelly import fractional_kelly, scale_portfolio_stakes
from engine.scorer import dedupe_one_per_window, rank_opportunities
from engine.types import BetOpportunity
from models.classifier import BetClassifier
from models.ev_regressor import EVRegressor
from models.meta_learner import MetaLearner
from models.risk_scorer import RiskScorer
from models.trainer import models_exist

logger = setup_logging()


class InferencePipeline:
    """Fetch BTC data → compute trend → predict → size bets."""

    def __init__(self) -> None:
        self.kalshi = KalshiClient()
        self.coinbase = CoinbaseClient()
        self.classifier = BetClassifier()
        self.ev_regressor = EVRegressor()
        self.risk_scorer = RiskScorer()
        self.meta_learner = MetaLearner()
        self._models_loaded = False
        self.markets_scanned = 0
        self.spot_price = 0.0

    def _ensure_models(self) -> None:
        if self._models_loaded:
            return
        if not models_exist():
            raise FileNotFoundError(
                "Models not trained. Run `python scripts/train.py` first."
            )
        self.classifier.load()
        self.ev_regressor.load()
        self.risk_scorer.load()
        self.meta_learner.load()
        self._models_loaded = True

    def _format_pick(
        self,
        market: dict,
        direction: str,
        spot: float,
        strike: float | None,
    ) -> str:
        strike_str = f"${strike:,.2f}" if strike else "current price"
        mins = market.get("minutes_to_close", 0)
        return (
            f"BTC 15m: {direction} — spot ${spot:,.2f} vs strike {strike_str} "
            f"({mins:.0f}m to close)"
        )

    def _decimal_odds_from_contract(self, contract_price: float) -> float:
        prob = kalshi_price_to_prob(contract_price)
        if prob <= 0 or prob >= 1:
            return 2.0
        return 1.0 / prob

    def _build_opportunity(
        self,
        market: dict,
        feats: dict,
        bankroll: float,
        spot: float,
        *,
        confidence: float,
        ev_per_contract: float,
        uncertainty: float,
    ) -> BetOpportunity:
        edge = float(feats["edge"])
        direction = "YES" if edge > 0 else "NO"
        yes_price = float(market.get("mid_price", 50))
        contract_px = contract_price_cents(yes_price, direction)
        market_impl = implied_prob_for_direction(yes_price, direction)
        model_prob = float(feats["model_prob"])
        prob_win = model_prob if direction == "YES" else (1.0 - model_prob)
        odds_dec = self._decimal_odds_from_contract(contract_px)
        stake = fractional_kelly(
            prob_win=prob_win,
            odds_decimal=odds_dec,
            bankroll=bankroll,
        )
        strike = market.get("floor_strike")
        return BetOpportunity(
            market_ticker=market["ticker"],
            market_title=market.get("title", ""),
            window_label=market.get("event_ticker", market["ticker"]),
            pick=self._format_pick(market, direction, spot, strike),
            direction=direction,
            kalshi_price=yes_price,
            contract_price=contract_px,
            market_implied_prob=market_impl,
            model_prob=model_prob,
            edge_pct=edge * 100,
            confidence=confidence,
            ev_per_contract=ev_per_contract,
            uncertainty=uncertainty,
            recommended_stake=stake,
            minutes_to_close=float(feats["minutes_to_close"]),
            closes_at=market.get("closes_at") or "",
            floor_strike=strike,
            spot_price=spot,
            momentum_15m=float(feats.get("momentum_15m", 0)),
            ema_trend=float(feats.get("ema_trend", 0)),
        )

    def run(self, bankroll: float) -> list[BetOpportunity]:
        try:
            self._ensure_models()
        except FileNotFoundError as exc:
            logger.error(str(exc))
            raise

        try:
            spot = self.coinbase.get_spot_price()
            self.spot_price = spot
            candles = self.coinbase.get_multi_timeframe()
        except Exception as exc:
            logger.error("Coinbase fetch failed: %s", exc)
            return []

        try:
            markets = self.kalshi.get_active_btc_markets()
        except Exception as exc:
            logger.error("Kalshi fetch failed: %s", exc)
            return []

        self.markets_scanned = len(markets)

        feature_rows: list[dict] = []
        market_meta: list[dict] = []

        for market in markets:
            if market.get("volume", 0) < settings.MIN_VOLUME:
                logger.info(
                    "Skipping %s: volume %s < %s",
                    market.get("ticker"),
                    market.get("volume"),
                    settings.MIN_VOLUME,
                )
                continue

            orderbook = None
            history: list = []
            try:
                orderbook = self.kalshi.get_market_orderbook(market["ticker"])
                history = self.kalshi.get_market_history(market["ticker"])
            except Exception:
                pass

            feats = build_features(market, spot, candles, orderbook, history)
            if feats is None:
                mins = market.get("minutes_to_close")
                if mins is not None:
                    logger.info(
                        "Skipping %s: %.1fm to close (entry window %.0f-%.0fm)",
                        market.get("ticker"),
                        mins,
                        settings.MIN_MINUTES_TO_CLOSE,
                        settings.MAX_MINUTES_TO_CLOSE,
                    )
                continue

            if feats["abs_edge"] < settings.MIN_EDGE_THRESHOLD:
                logger.info(
                    "Skipping %s: edge %.1f%% < %.1f%%",
                    market.get("ticker"),
                    feats["abs_edge"] * 100,
                    settings.MIN_EDGE_THRESHOLD * 100,
                )
                continue

            feature_rows.append(feats)
            market_meta.append({"market": market, "feats": feats})

        if not feature_rows:
            logger.info(
                "No markets passed entry/edge filters (window %.0f-%.0fm, edge >= %.1f%%)",
                settings.MIN_MINUTES_TO_CLOSE,
                settings.MAX_MINUTES_TO_CLOSE,
                settings.MIN_EDGE_THRESHOLD * 100,
            )
            return []

        X = np.array(
            [[f[c] for c in settings.FEATURE_COLUMNS] for f in feature_rows],
            dtype=float,
        )
        X_df = pd.DataFrame(X, columns=settings.FEATURE_COLUMNS)
        clf_proba = self.classifier.predict_proba(X)
        ev_pred = self.ev_regressor.predict(X_df)
        uncertainty = self.risk_scorer.uncertainty(X)
        X_meta = np.column_stack(
            [
                clf_proba,
                ev_pred,
                uncertainty,
                [f["edge"] for f in feature_rows],
                [f["minutes_to_close"] for f in feature_rows],
                [f["log_volume"] for f in feature_rows],
            ]
        )
        scores = self.meta_learner.predict_score(X_meta)

        candidates: list[BetOpportunity] = []
        for i, meta in enumerate(market_meta):
            market = meta["market"]
            feats = meta["feats"]
            ticker = market.get("ticker", "")
            yes_price = float(market.get("mid_price", 50))
            direction = "YES" if float(feats["edge"]) > 0 else "NO"
            conf = float(scores[i])
            unc = float(uncertainty[i])
            reason = reject_entry(
                feats=feats,
                direction=direction,
                yes_price_cents=yes_price,
                confidence=conf,
                uncertainty=unc,
                spot=spot,
                strike=market.get("floor_strike"),
            )
            if reason:
                logger.info("Skipping %s: %s", ticker, reason)
                continue
            candidates.append(
                self._build_opportunity(
                    market,
                    feats,
                    bankroll,
                    spot,
                    confidence=conf,
                    ev_per_contract=float(ev_pred[i]),
                    uncertainty=unc,
                )
            )

        opportunities = dedupe_one_per_window(candidates)
        scale_portfolio_stakes(opportunities, bankroll)
        ranked = rank_opportunities(opportunities)
        if candidates and not ranked:
            best = max(candidates, key=lambda o: o.confidence)
            logger.info(
                "Best candidate %s filtered: conf %.2f (need %.2f), unc %.2f (max %.2f)",
                best.market_ticker,
                best.confidence,
                settings.MIN_CONFIDENCE_THRESHOLD,
                best.uncertainty,
                settings.MAX_UNCERTAINTY_FOR_TRADE,
            )
        return ranked
