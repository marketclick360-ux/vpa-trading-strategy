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
