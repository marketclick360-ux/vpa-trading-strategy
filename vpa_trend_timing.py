"""
SMA-200 trend-timing strategy, validated out-of-sample against buy-and-hold.

Why this file exists: the original VPA anomaly signals (vpa_coulling.py,
vpa_etf_daily.py) were walk-forward tested and did NOT show a durable edge
-- see VPA_DIAGNOSIS.md for the full trade-level audit. This is the
risk-adjusted alternative that survived out-of-sample testing: be long a
symbol only while it's above its 200-day SMA, and park cash at the prevailing
T-bill yield the rest of the time. It does not beat buy-and-hold on raw CAGR
in the strong-bull 2022-2026 test window (median 4.11% vs 6.38%), but it
matches or beats CAGR on ~47% of symbols individually and cuts max drawdown
on ~80% of them (median -21% vs -28%). See VPA_DIAGNOSIS.md for the numbers.
"""
import pandas as pd
import numpy as np
from vpa_etf_daily import get_daily_data, ALL_ETFS, COST_PER_TRADE, INITIAL_EQUITY

# =========================
# CONFIG
# =========================
START_DATE = '2010-01-01'
SMA_LEN = 200
CASH_YIELD_ANNUAL = 0.045   # approx. T-bill yield; cash is not actually zero-return
CASH_DAILY = (1 + CASH_YIELD_ANNUAL) ** (1 / 252) - 1


def backtest_trend_timing(df, sma_len=SMA_LEN, cost=COST_PER_TRADE,
                           cash_yield_daily=CASH_DAILY, initial_equity=INITIAL_EQUITY):
    d = df.copy()
    d['SMA'] = d['Close'].rolling(sma_len).mean()
    d['Uptrend'] = d['Close'] > d['SMA']
    d = d.dropna(subset=['SMA'])
    d['Ret'] = d['Close'].pct_change().fillna(0.0)
    d['Pos'] = d['Uptrend'].shift(1).fillna(False).astype(int)  # act on yesterday's signal
    d['PosChange'] = d['Pos'].diff().abs().fillna(0)
    d['PnL'] = np.where(d['Pos'] == 1, d['Ret'], cash_yield_daily) - d['PosChange'] * cost
    d['Equity'] = (1 + d['PnL']).cumprod() * initial_equity
    return d


def cagr_of(series, n):
    total = series.iloc[-1] / series.iloc[0] - 1.0
    return (1 + total) ** (252.0 / n) - 1.0 if n > 0 else 0.0


def maxdd_of(eq):
    return (eq / eq.cummax() - 1.0).min()


def sharpe_of(rets):
    vol = rets.std() * np.sqrt(252)
    ann_ret = rets.mean() * 252
    return ann_ret / vol if vol > 0 else 0.0


def evaluate(symbols, start=START_DATE, metric_start=None, label=""):
    """metric_start: if set, indicators are computed on the full history from
    `start`, but CAGR/DD/Sharpe are measured only from metric_start onward
    (true out-of-sample window, with proper lookback for the SMA)."""
    rows = []
    for sym in symbols:
        try:
            df = get_daily_data(sym, start)
            if len(df) < SMA_LEN + 30:
                continue
            d = backtest_trend_timing(df)
            if metric_start is not None:
                d = d[d.index >= metric_start]
                if len(d) < 30:
                    continue
                d = d.copy()
                d['Equity'] = (1 + d['PnL']).cumprod() * INITIAL_EQUITY
                d['Close'] = d['Close'] / d['Close'].iloc[0] * INITIAL_EQUITY
            n = len(d)
            strat_cagr = cagr_of(d['Equity'], n)
            strat_dd = maxdd_of(d['Equity'])
            strat_sharpe = sharpe_of(d['PnL'])
            bh_cagr = cagr_of(d['Close'], n)
            bh_dd = maxdd_of(d['Close'])
            rows.append({
                'symbol': sym, 'n_days': n,
                'strat_cagr': round(strat_cagr * 100, 2),
                'strat_dd': round(strat_dd * 100, 2),
                'strat_sharpe': round(strat_sharpe, 2),
                'bh_cagr': round(bh_cagr * 100, 2),
                'bh_dd': round(bh_dd * 100, 2),
                'beats_bh_cagr': strat_cagr > bh_cagr,
                'beats_bh_dd': strat_dd > bh_dd,  # less negative = smaller drawdown
            })
        except Exception as e:
            print(f"  {sym}: ERROR - {e}")
    res = pd.DataFrame(rows)
    if len(res) == 0:
        print(f"No results for {label}")
        return res

    print(f"\n{'=' * 70}")
    print(f"  TREND-TIMING vs BUY & HOLD  {label}  ({len(res)} symbols)")
    print(f"{'=' * 70}")
    print(f"  Median CAGR:    strat {res.strat_cagr.median():6.2f}%   B&H {res.bh_cagr.median():6.2f}%")
    print(f"  Mean CAGR:      strat {res.strat_cagr.mean():6.2f}%   B&H {res.bh_cagr.mean():6.2f}%")
    print(f"  Median MaxDD:   strat {res.strat_dd.median():6.2f}%   B&H {res.bh_dd.median():6.2f}%")
    print(f"  Beats B&H CAGR: {res.beats_bh_cagr.sum()}/{len(res)}  ({res.beats_bh_cagr.mean()*100:.1f}%)")
    print(f"  Beats B&H DD:   {res.beats_bh_dd.sum()}/{len(res)}  ({res.beats_bh_dd.mean()*100:.1f}%)")
    return res


def scan_today(symbols):
    print(f"\n{'=' * 60}")
    print(f"  TREND-TIMING SCANNER - {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}")
    print(f"{'=' * 60}")
    for sym in symbols:
        try:
            df = get_daily_data(sym, '2023-01-01')
            if len(df) < SMA_LEN + 5:
                print(f"  {sym:6s} | insufficient history for {SMA_LEN}d SMA")
                continue
            d = df.copy()
            d['SMA'] = d['Close'].rolling(SMA_LEN).mean()
            last = d.iloc[-1]
            state = "LONG (above SMA200)" if last['Close'] > last['SMA'] else "CASH (below SMA200)"
            print(f"  {sym:6s} | ${last['Close']:.2f} | SMA200=${last['SMA']:.2f} | {state}")
        except Exception as e:
            print(f"  {sym:6s} | ERROR: {e}")
    print(f"{'=' * 60}\n")


def main():
    print("SMA-200 Trend-Timing Strategy (validated OOS alternative to VPA anomaly signals)")
    print(f"See VPA_DIAGNOSIS.md for why this replaced the anomaly-based entries.\n")

    print("--- Full-history backtest (in-sample, all data) ---")
    evaluate(ALL_ETFS, start=START_DATE, label="(full history)")

    print("\n--- Out-of-sample test (metrics measured 2022-01-01 onward only) ---")
    evaluate(ALL_ETFS, start=START_DATE, metric_start='2022-01-01', label="(OOS)")

    scan_today(['SPY', 'QQQ', 'IWM', 'EFA', 'EEM', 'IEF', 'TLT', 'GLD', 'AAPL', 'MSFT', 'NVDA'])


if __name__ == '__main__':
    main()
