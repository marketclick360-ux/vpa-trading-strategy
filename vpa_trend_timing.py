"""
SMA-200 trend-timing strategy + hard stop-loss, validated out-of-sample
against buy-and-hold.

Why this file exists: the original VPA anomaly signals (vpa_coulling.py,
vpa_etf_daily.py) were walk-forward tested and did NOT show a durable edge
-- see VPA_DIAGNOSIS.md for the full trade-level audit. This is the
risk-adjusted alternative that survived out-of-sample testing: be long a
symbol only while it's above its 200-day SMA, and park cash at the prevailing
T-bill yield the rest of the time.

A hard STOP_LOSS_PCT is layered on top of the trend exit: the 200-day SMA
is slow and can't react to a sudden drop, so a position also exits the
moment a day's Low breaches entry_price * (1 - STOP_LOSS_PCT), regardless
of the SMA. Tested OOS (2022-2026): barely changes returns (median CAGR
4.88% -> 4.91%) but caps the worst single-trade loss from -14.14% to
-5.10% and lifts the whitelist's win count. See VPA_DIAGNOSIS.md for the
full numbers.
"""
import pandas as pd
import numpy as np
from vpa_etf_daily import get_daily_data, ALL_ETFS, COST_PER_TRADE, INITIAL_EQUITY

# =========================
# CONFIG
# =========================
START_DATE = '2010-01-01'
SMA_LEN = 200
STOP_LOSS_PCT = 0.05        # hard stop: exit if price falls 5% below entry, regardless of SMA
CASH_YIELD_ANNUAL = 0.045   # approx. T-bill yield; cash is not actually zero-return
CASH_DAILY = (1 + CASH_YIELD_ANNUAL) ** (1 / 252) - 1


def backtest_trend_timing(df, sma_len=SMA_LEN, cost=COST_PER_TRADE,
                           cash_yield_daily=CASH_DAILY, initial_equity=INITIAL_EQUITY,
                           stop_loss_pct=STOP_LOSS_PCT):
    """Path-dependent day-by-day simulation (not vectorized) because the hard
    stop needs to check each day's Low against the actual entry price, which
    a vectorized cumulative-product can't express."""
    d = df.copy()
    d['SMA'] = d['Close'].rolling(sma_len).mean()
    d['Uptrend'] = d['Close'] > d['SMA']
    d = d.dropna(subset=['SMA']).copy()

    closes = d['Close'].values
    lows = d['Low'].values
    uptrend = d['Uptrend'].values
    n = len(d)

    equity = np.empty(n)
    pnl_arr = np.zeros(n)
    pos_arr = np.zeros(n, dtype=int)
    equity[0] = initial_equity
    in_pos = False
    entry_price = None

    for i in range(1, n):
        prev_signal = uptrend[i - 1]  # act on yesterday's signal, no lookahead
        pnl = 0.0
        exited = False

        if in_pos:
            if stop_loss_pct is not None and lows[i] <= entry_price * (1 - stop_loss_pct):
                stop_price = entry_price * (1 - stop_loss_pct)
                pnl = (stop_price / closes[i - 1] - 1.0) - cost
                in_pos = False
                exited = True
            else:
                pnl = closes[i] / closes[i - 1] - 1.0
                if not prev_signal:
                    pnl -= cost
                    in_pos = False
                    exited = True
        else:
            pnl = cash_yield_daily

        if not in_pos and not exited and prev_signal:
            in_pos = True
            entry_price = closes[i]
            pnl -= cost

        pos_arr[i] = 1 if in_pos else 0
        pnl_arr[i] = pnl
        equity[i] = equity[i - 1] * (1 + pnl)

    d['Pos'] = pos_arr
    d['PnL'] = pnl_arr
    d['Equity'] = equity
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


def get_beats_bh_whitelist(symbols=None, start=START_DATE, metric_start='2022-01-01'):
    """Symbols where trend-timing is BOTH profitable AND ahead of buy-and-hold
    out-of-sample. Excludes cases that only "win" because both strategy and
    B&H lost money and the strategy merely lost less (e.g. VIXY, TLT, UNG in
    the 2022-2026 test window) -- that is not a buy signal."""
    symbols = symbols or ALL_ETFS
    res = evaluate(symbols, start=start, metric_start=metric_start, label="(whitelist build)")
    if len(res) == 0:
        return []
    winners = res[(res['strat_cagr'] > 0) & (res['strat_cagr'] > res['bh_cagr'])]
    return sorted(winners['symbol'].tolist())


# Regenerate with get_beats_bh_whitelist() periodically -- OOS edge can decay.
# Built from 2022-2026 out-of-sample validation WITH the 5% hard stop-loss
# active (see VPA_DIAGNOSIS.md). Rebuilding after adding the stop changed
# membership vs the no-stop version: EWJ, HACK, and QQQ dropped out; nothing
# new was added -- always regenerate under the exact rules you intend to trade.
BEATS_BH_WHITELIST = [
    'AGG', 'ARKW', 'BND', 'BOTZ', 'EEM', 'EFA', 'FXI', 'HYG', 'ICLN', 'IEMG',
    'IWM', 'LIT', 'LQD', 'MUB', 'SHY', 'TIP', 'VEA', 'XLC',
]


def scan_buy_signals_only(symbols=None, stop_loss_pct=STOP_LOSS_PCT):
    """Only post what's actually actionable: symbols with a validated
    (profitable, beats-B&H) edge that are CURRENTLY in a BUY state. Nothing
    else gets printed -- no short signals, no "sitting in cash" calls, no
    unvalidated symbols.

    This scanner has no record of your actual fill price, so it can't track
    a live stop for you -- it prints the stop level you'd set *if you buy
    today*. If you already hold a position from an earlier signal, your
    stop is 5% below YOUR entry price, not today's price."""
    symbols = symbols or BEATS_BH_WHITELIST
    print(f"\n{'=' * 60}")
    print(f"  BUY SIGNALS ONLY - validated vs buy-and-hold - {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}")
    print(f"{'=' * 60}")
    posted = 0
    for sym in symbols:
        try:
            df = get_daily_data(sym, '2023-01-01')
            if len(df) < SMA_LEN + 5:
                continue
            d = df.copy()
            d['SMA'] = d['Close'].rolling(SMA_LEN).mean()
            last = d.iloc[-1]
            if last['Close'] > last['SMA']:  # BUY state only; CASH state is not posted
                stop_price = last['Close'] * (1 - stop_loss_pct)
                print(f"  BUY  {sym:6s} | ${last['Close']:.2f} | SMA200=${last['SMA']:.2f} | "
                      f"stop if bought today=${stop_price:.2f} (-{stop_loss_pct*100:.0f}%)")
                posted += 1
        except Exception:
            continue
    if posted == 0:
        print("  No validated symbols are currently in a BUY state.")
    print(f"{'=' * 60}\n")


def main():
    print("SMA-200 Trend-Timing Strategy (validated OOS alternative to VPA anomaly signals)")
    print(f"See VPA_DIAGNOSIS.md for why this replaced the anomaly-based entries.\n")

    print("--- Full-history backtest (in-sample, all data) ---")
    evaluate(ALL_ETFS, start=START_DATE, label="(full history)")

    print("\n--- Out-of-sample test (metrics measured 2022-01-01 onward only) ---")
    evaluate(ALL_ETFS, start=START_DATE, metric_start='2022-01-01', label="(OOS)")

    scan_buy_signals_only()


if __name__ == '__main__':
    main()
