"""One-shot live pipeline diagnostic (why picks are 0)."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from config import settings
from data.fetchers.coinbase import CoinbaseClient
from data.fetchers.kalshi import KalshiClient
from data.processors.features import build_features
from engine.entry_filters import reject_entry
from models.classifier import BetClassifier
from models.ev_regressor import EVRegressor
from models.meta_learner import MetaLearner
from models.risk_scorer import RiskScorer


def main() -> None:
    coinbase, kalshi = CoinbaseClient(), KalshiClient()
    spot = coinbase.get_spot_price()
    candles = coinbase.get_multi_timeframe()
    markets = kalshi.get_active_btc_markets()
    print(f"spot=${spot:,.2f}  markets={len(markets)}")

    clf, ev, risk, meta = BetClassifier(), EVRegressor(), RiskScorer(), MetaLearner()
    for m in (clf, ev, risk, meta):
        m.load()

    for market in markets:
        ticker = market["ticker"]
        mins = market.get("minutes_to_close")
        mid = float(market.get("mid_price", 50))
        strike = market.get("floor_strike")
        print(f"\n{ticker}  {mins:.1f}m left  YES={mid:.0f}c  strike={strike}")
        if mins is None or mins < settings.MIN_MINUTES_TO_CLOSE or mins > settings.MAX_MINUTES_TO_CLOSE:
            print("  -> entry window")
            continue
        ob = kalshi.get_market_orderbook(ticker)
        hist = kalshi.get_market_history(ticker)
        feats = build_features(market, spot, candles, ob, hist)
        if feats is None:
            print("  -> no features")
            continue
        if feats["abs_edge"] < settings.MIN_EDGE_THRESHOLD:
            print(f"  -> edge {feats['abs_edge']*100:.1f}% < {settings.MIN_EDGE_THRESHOLD*100:.1f}%")
            continue
        X = np.array([[feats[c] for c in settings.FEATURE_COLUMNS]], dtype=float)
        Xdf = pd.DataFrame(X, columns=settings.FEATURE_COLUMNS)
        clf_p = clf.predict_proba(X)
        ev_p = ev.predict(Xdf)
        unc = risk.uncertainty(X)
        score = float(
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
        direction = "YES" if feats["edge"] > 0 else "NO"
        reason = reject_entry(
            feats=feats,
            direction=direction,
            yes_price_cents=mid,
            confidence=score,
            uncertainty=float(unc[0]),
            spot=spot,
            strike=strike,
        )
        print(
            f"  dir={direction} edge={feats['edge']*100:+.1f}% "
            f"model={feats['model_prob']*100:.0f}% conf={score:.2f} unc={unc[0]:.2f}"
        )
        print(f"  reject_entry: {reason or 'PASS'}")
        if reason is None and score < settings.MIN_CONFIDENCE_THRESHOLD:
            print(f"  rank_opportunities: conf {score:.2f} < {settings.MIN_CONFIDENCE_THRESHOLD}")
        elif reason is None:
            print("  -> WOULD TRADE")


if __name__ == "__main__":
    main()
