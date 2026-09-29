# VPA Trading Strategy - Anna Coulling Volume Price Analysis

A Python implementation of Anna Coulling's Volume Price Analysis (VPA) anomaly detection for trading. Includes full backtest engine and live market scanner.

## What It Does

- **Detects VPA Anomalies** on any stock/ETF using free Yahoo Finance data
- **Backtests** the strategy with configurable parameters
- **Scans** your watchlist for real-time anomaly signals
- **No API key required** - uses yfinance (free)

## VPA Anomaly Types

| Anomaly | Description | Signal |
|---------|-------------|--------|
| Fake Up | Wide spread UP candle + LOW volume | Bearish (reversal) |
| Fake Down | Wide spread DOWN candle + LOW volume | Bullish (reversal) |
| Absorb Up | Narrow spread UP candle + HIGH volume | Bearish (absorption) |
| Absorb Down | Narrow spread DOWN candle + HIGH volume | Bullish (absorption) |
| Confirm Up | Wide spread UP + HIGH volume | Trend continuation |
| Confirm Down | Wide spread DOWN + HIGH volume | Trend continuation |

## ⚠️ Validation status (read before trading any of this)

The VPA anomaly signal below was walk-forward tested and does **not** show
a durable edge — see [`VPA_DIAGNOSIS.md`](./VPA_DIAGNOSIS.md) for the full
trade-level audit (2,700+ trades, out-of-sample validation). Short-side
signals are net-negative and should not be traded. `vpa_coulling.py` and
`vpa_etf_daily.py` are still useful as anomaly *scanners* for awareness, but
not as mechanical trading signals.

**`vpa_trend_timing.py`** (SMA-200 trend timing + a hard 5% stop-loss, cash
at T-bill yield when flat) is the validated risk-managed alternative — it
trails buy-and-hold on raw CAGR in the tested 2022–2026 bull-market window
but matches or beats it on many symbols individually with meaningfully
lower drawdown, and the stop caps worst-case single-trade loss at -5%
(down from -14% without it). See `VPA_DIAGNOSIS.md` §5 and §7 for numbers.
A Chaikin Money Flow "accumulation" signal was also tested (§6) and
rejected — too much whipsaw, no net benefit.

### What gets posted / traded right now

Only long ("buy") signals are surfaced anywhere in this repo — no short
signals are printed or traded by any script, since the short side showed no
real edge (see `VPA_DIAGNOSIS.md` §1–2). On top of that,
`vpa_trend_timing.scan_buy_signals_only()` further restricts to a whitelist
of symbols (`BEATS_BH_WHITELIST`) that were both **profitable and ahead of
buy-and-hold** out-of-sample — it only prints a symbol when it's currently
in a BUY state *and* on that whitelist. Nothing else gets posted: no short
alerts, no "sitting in cash" calls, no unvalidated symbols. Regenerate the
whitelist periodically with `get_beats_bh_whitelist()` — an OOS edge can
decay over time.

## Quick Start

```bash
# Clone the repo
git clone https://github.com/marketclick360-ux/vpa-trading-strategy.git
cd vpa-trading-strategy

# Install dependencies
pip install -r requirements.txt

# Run the strategy
python vpa_coulling.py
```

## Configuration

Edit the CONFIG section in `vpa_coulling.py`:

```python
SYMBOL = 'SPY'           # Symbol to backtest
START_DATE = '2010-01-01' # Backtest start date
LOOKBACK_WINDOW = 20      # Rolling window for percentiles
HOLD_BARS = 5             # Hold position for N bars
COST_PER_TRADE = 0.001    # Transaction cost (0.1%)
```

## Output

- Full backtest metrics (CAGR, Sharpe, Max Drawdown, etc.)
- Anomaly signal counts
- Live scanner results for your watchlist
- CSV export of equity curve and signals

## Next Steps

- [ ] Connect to Schwab Trader API for live execution
- [ ] Add parameter optimization sweep
- [ ] Add multi-timeframe confirmation
- [ ] Add visualization with matplotlib

## Based On

Anna Coulling's Volume Price Analysis methodology.
