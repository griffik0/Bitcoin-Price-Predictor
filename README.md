# BTC Price Bot

Autonomous betting bot for Kalshi's **BTC 15 min** markets (`KXBTC15M`). Every 15 minutes a new binary contract opens: will Bitcoin's price be **higher** at the end of the window than at the start?

The bot mirrors the architecture of `kalshi-sports-bot`: fetch historical Kalshi data, train an ML ensemble, score live opportunities, and optionally place trades — but uses **Coinbase Exchange** price data (the same reference feed Kalshi settles against) and technical indicators instead of sportsbook odds.

## How it works

1. **Price data** — Fetches BTC-USD spot price and 1m/5m/15m/1h candles from Coinbase Exchange (free, no API key).
2. **Trend analysis** — Computes RSI, MACD, EMA trend, momentum, ATR volatility, and distance from the Kalshi strike price.
3. **Model probability** — Estimates P(BTC ends above strike) and compares to Kalshi's implied probability to find **edge**.
4. **ML ensemble** — XGBoost classifier + LightGBM EV regressor + conformal risk scorer + logistic meta-learner (same stack as sports bot).
5. **Risk gates** — Only bets when confidence is high, uncertainty is low, and edge exceeds threshold. Skips windows with too much risk.
6. **Entry timing** — Enters only when 2–12 minutes remain before close (configurable), avoiding noisy open/close periods.
7. **Exits** — Sells positions when edge reverses, confidence drops, or uncertainty spikes.

## Quick start

```bash
cd btc-price-bot
python -m venv .venv
.venv\Scripts\activate        # Windows
pip install -r requirements.txt
copy .env.example .env        # Add your Kalshi API keys (same as sports bot)
python scripts/seed_db.py       # Fetch historical KXBTC15M markets from Kalshi
python scripts/train.py         # Train ML models
python scripts/backtest.py      # Simulate on historical data
python main.py --bankroll 100   # Live recommendations dashboard
```

### Auto-trading (dry run first)

```bash
python main.py --trade dry-run --bankroll 100
python main.py --trade live --bankroll 100   # Requires AUTO_TRADE_LIVE=true in .env
```

## Configuration (.env)

| Variable | Default | Purpose |
|----------|---------|---------|
| `KALSHI_API_KEY` | — | Kalshi API Key ID |
| `KALSHI_PRIVATE_KEY_PATH` | — | Path to RSA private key file |
| `AUTO_TRADE_LIVE` | false | Must be `true` for real orders |
| `BANKROLL` | 10 | Starting bankroll for Kelly sizing |
| `MIN_CONFIDENCE_FOR_TRADE` | 0.65 | Minimum ML confidence to place a bet |
| `MAX_UNCERTAINTY_FOR_TRADE` | 0.55 | Maximum risk score to place a bet |
| `MIN_EDGE_THRESHOLD` | 0.04 | Minimum model vs Kalshi edge (4%) |
| `REFRESH_INTERVAL_SECONDS` | 45 | How often to scan markets |
| `MIN_MINUTES_TO_CLOSE` | 2.0 | Don't enter with < 2 min left |
| `MAX_MINUTES_TO_CLOSE` | 12.0 | Don't enter with > 12 min left |

## Project structure

```
btc-price-bot/
├── main.py                    # Entry point
├── config/settings.py         # All tunables
├── data/
│   ├── fetchers/
│   │   ├── kalshi.py          # KXBTC15M market discovery
│   │   ├── kalshi_auth.py     # RSA-signed trading
│   │   └── coinbase.py        # BTC price + candles
│   ├── processors/
│   │   ├── indicators.py      # RSI, MACD, EMA, momentum
│   │   └── features.py        # Feature engineering
│   └── storage/db.py          # SQLite history
├── models/                    # ML ensemble (classifier, EV, risk, meta)
├── engine/
│   ├── pipeline.py            # Inference orchestrator
│   ├── trader.py              # Buy/sell with risk gates
│   └── kelly.py               # Position sizing
├── scripts/
│   ├── seed_db.py             # Populate training data
│   ├── train.py               # Train models
│   └── backtest.py            # Historical simulation
└── ui/dashboard.py            # Rich terminal UI
```

## Reusing kalshi-sports-bot credentials

Copy the same `KALSHI_API_KEY` and `KALSHI_PRIVATE_KEY_PATH` from your sports bot `.env`. The trading client uses identical RSA-PSS signing.

## Disclaimer

This is experimental software for educational purposes. Prediction markets involve real financial risk. Past performance does not guarantee future results. Start with dry-run mode and small bankrolls.
