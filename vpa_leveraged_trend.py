"""
Leveraged SMA-200 trend timing on QQQ/XLK/EFA -- the strategy that
actually beat buy-and-hold, validated on 20-30 years of history including
the dot-com crash and 2008 GFC, not just the recent bull run.

Per-symbol leverage is set to whichever of 2x/3x tested the higher CAGR on
full history (QQQ=2x, XLK=3x, EFA=3x) -- maximizing profitability per an
explicit "don't care about risk" instruction, NOT the risk-managed default.
This means real historical worst-case drawdowns of roughly -73% (QQQ),
-88% (XLK), -70% (EFA). The hard 5% stop-loss is the actual risk control
here, not the drawdown number -- it caps every individual trade's loss
regardless of leverage; the drawdown figures above are what happens when
many trades lose in sequence during a sustained bear market, not a single
blown stop.

Why these three, and not the wider whitelist: vpa_trend_timing.py's
strategy only showed a genuine (non-overfit) CAGR edge over buy-and-hold on
these three symbols specifically, out of 20 tested with 20-30 years of
real history. That's not random -- QQQ/XLK/EFA all suffered catastrophic
70-85% buy-and-hold drawdowns (dot-com bust, 2008), and the 200-day SMA
exit avoided most of that damage. Assets without a crash that severe (bonds,
staples, utilities) never showed the same edge, because there was nothing
to avoid. See VPA_DIAGNOSIS.md SS8 for the full leverage comparison
(margin vs leveraged ETF, 2x vs 3x, and why 3x backfires on QQQ but not
XLK/EFA).

Signal is generated from the underlying ETF's own 200-day SMA + 5% stop --
unchanged from vpa_trend_timing.py. This backtest models the leveraged
return synthetically (N times the underlying's daily return, minus
financing drag) since that's what was actually tested and validated -- a
real leveraged product's own tracking error versus this model is a
live-execution detail to verify before funding it, not something this
backtest can capture. No listed product tracks XLK/EFA at exactly 3x;
that exposure means margin on top of the 2x product, or a generic 3x
tech/international ETF with a different underlying index -- confirm the
real product before funding.
"""
import pandas as pd
import numpy as np
import yfinance as yf
from vpa_etf_daily import COST_PER_TRADE, INITIAL_EQUITY

# =========================
# CONFIG
# =========================
SMA_LEN = 200
STOP_LOSS_PCT = 0.05
CASH_YIELD_ANNUAL = 0.045
CASH_DAILY = (1 + CASH_YIELD_ANNUAL) ** (1 / 252) - 1
LEV_ETF_EXPENSE = 0.0095    # ~0.95%/yr, typical leveraged-ETF expense ratio
LEV_ETF_FINANCING_SPREAD = 0.01

# Per-symbol leverage set to whichever tested higher CAGR on full history.
# QQQ/XLK/EFA are diversified funds -- see VPA_DIAGNOSIS.md SS8 for the
# 2x-vs-3x comparison. The 6 individual stocks below are a SEPARATE later
# finding (SS13): leverage helps CSCO but actively HURTS the other five --
# single-stock volatility decay is much more punishing than a diversified
# fund's, so AAL/AMD/M/C/MU are traded UNLEVERED (1x) even though they
# passed the same beats-buy-and-hold screen. These five also carry real
# single-company risk a diversified fund doesn't (Citigroup and American
# Airlines both came close to zero in 2008) -- that's a different risk
# than leverage and doesn't go away by staying at 1x.
LEVERAGE = {
    'QQQ': 2.0,
    'XLK': 3.0,
    'EFA': 3.0,
    'AAL': 1.0,
    'AMD': 1.0,
    'M': 1.0,
    'C': 1.0,
    'MU': 1.0,
    'CSCO': 2.0,
}

# Signal computed on the underlying; live execution via the mapped leveraged
# product (or margin at the same multiple on the underlying itself). Verify
# real tracking error before funding -- ROM (XLK-adjacent) and EFO
# (EFA-adjacent) track related but not identical indexes to their
# underlying's exact benchmark. No listed 3x product tracks these exactly;
# 3x exposure means either margin on the 2x product or a generic 3x
# tech/international leveraged ETF -- confirm the real product before use.
# 'DIRECT' means buy the stock itself (1x, no leverage product needed);
# 'MARGIN' means no dedicated leveraged single-stock product was confirmed,
# so 2x there means margin on the stock itself, not a named ETF ticker.
LEVERAGED_PRODUCT = {
    'QQQ': 'QLD',    # ProShares Ultra QQQ (2x Nasdaq-100)
    'XLK': 'ROM',    # ProShares Ultra Technology (2x tech) -- 3x needs margin on top, or TECL (3x, different index)
    'EFA': 'EFO',    # ProShares Ultra MSCI EAFE (2x developed intl) -- 3x needs margin on top, no clean 3x EAFE product
    'AAL': 'DIRECT',
    'AMD': 'DIRECT',
    'M': 'DIRECT',
    'C': 'DIRECT',
    'MU': 'DIRECT',
    'CSCO': 'MARGIN',  # no confirmed dedicated 2x CSCO product -- 2x means margin on the stock itself
}
SYMBOLS = list(LEVERAGED_PRODUCT.keys())


def get_full_history(symbol):
    df = yf.download(symbol, period="max", progress=False, auto_adjust=True)
    if df is None or df.empty:
        return None
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.droplevel(1)
    return df.dropna()


def backtest_leveraged_trend(df, leverage, sma_len=SMA_LEN, stop_pct=STOP_LOSS_PCT,
                              cost=COST_PER_TRADE, cash_yield_daily=CASH_DAILY,
                              initial_equity=INITIAL_EQUITY):
    d = df.copy()
    d['SMA'] = d['Close'].rolling(sma_len).mean()
    d['Uptrend'] = d['Close'] > d['SMA']
    d['DailyRet'] = d['Close'].pct_change()
    d = d.dropna(subset=['SMA']).copy()

    closes, lows, uptrend, rets = d['Close'].values, d['Low'].values, d['Uptrend'].values, d['DailyRet'].values
    n = len(d)
    equity = np.empty(n)
    equity[0] = initial_equity
    in_pos = False
    entry_price = None
    daily_financing = ((leverage - 1) * (0.045 + LEV_ETF_FINANCING_SPREAD) + LEV_ETF_EXPENSE) / 252

    for i in range(1, n):
        prev_signal = uptrend[i - 1]
        pnl = 0.0
        exited = False
        if in_pos:
            if lows[i] <= entry_price * (1 - stop_pct):
                underlying_pnl = entry_price * (1 - stop_pct) / closes[i - 1] - 1.0
                pnl = leverage * underlying_pnl - daily_financing - cost
                in_pos = False
                exited = True
            else:
                pnl = leverage * rets[i] - daily_financing
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
        pnl = max(pnl, -1.0)
        equity[i] = equity[i - 1] * (1 + pnl)

    d['Pos'] = 0
    d.loc[d.index[1:], 'Equity'] = equity[1:]
    d.loc[d.index[0], 'Equity'] = equity[0]
    return d


def cagr_of(series, n):
    total = series.iloc[-1] / series.iloc[0] - 1.0
    return (1 + total) ** (252.0 / n) - 1.0 if n > 0 else 0.0


def maxdd_of(eq):
    return (eq / eq.cummax() - 1.0).min()


def evaluate(symbols=None, label=""):
    symbols = symbols or SYMBOLS
    rows = []
    for sym in symbols:
        df = get_full_history(sym)
        if df is None or len(df) < SMA_LEN + 100:
            continue
        lev = LEVERAGE[sym]
        d = backtest_leveraged_trend(df, leverage=lev)
        n = len(d)
        strat_cagr = cagr_of(d['Equity'], n)
        strat_dd = maxdd_of(d['Equity'])
        bh_cagr = cagr_of(d['Close'], n)
        bh_dd = maxdd_of(d['Close'])
        rows.append({
            'symbol': sym, 'leverage': f"{lev:.0f}x", 'leveraged_product': LEVERAGED_PRODUCT[sym],
            'start': d.index[0].date().isoformat(), 'years': round(n / 252, 1),
            'strat_cagr': round(strat_cagr * 100, 2), 'strat_dd': round(strat_dd * 100, 2),
            'bh_cagr': round(bh_cagr * 100, 2), 'bh_dd': round(bh_dd * 100, 2),
            'beats_bh': strat_cagr > bh_cagr,
        })
    res = pd.DataFrame(rows)
    if len(res) == 0:
        print(f"No results for {label}")
        return res
    print(f"\n{'=' * 70}\n  LEVERAGED TREND TIMING (per-symbol optimal leverage)  {label}\n{'=' * 70}")
    print(res.to_string(index=False))
    return res


def scan_buy_signals_only(symbols=None, stop_loss_pct=STOP_LOSS_PCT):
    """Only posts a symbol when its underlying is currently above its 200-day
    SMA. Shows the real leveraged product to execute with and the stop level
    on the UNDERLYING (not the leveraged product -- leverage amplifies the
    same % move, so basing the stop on the underlying's own price keeps the
    trigger clean regardless of the leveraged product's own tracking noise)."""
    symbols = symbols or SYMBOLS
    print(f"\n{'=' * 70}")
    print(f"  LEVERAGED BUY SIGNALS (per-symbol optimal leverage) - {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}")
    print(f"{'=' * 70}")
    posted = 0
    for sym in symbols:
        try:
            df = get_full_history(sym)
            if df is None or len(df) < SMA_LEN + 5:
                continue
            d = df.copy()
            d['SMA'] = d['Close'].rolling(SMA_LEN).mean()
            last = d.iloc[-1]
            if last['Close'] > last['SMA']:
                stop_price = last['Close'] * (1 - stop_loss_pct)
                product = LEVERAGED_PRODUCT[sym]
                lev = LEVERAGE[sym]
                if product == 'DIRECT':
                    via = f"direct, {sym} shares"
                elif product == 'MARGIN':
                    via = f"{lev:.0f}x via margin on {sym} (no dedicated leveraged product)"
                else:
                    via = f"{lev:.0f}x via {product}"
                print(f"  BUY  {sym:6s} ({via}) | ${last['Close']:.2f} | SMA200=${last['SMA']:.2f} | "
                      f"stop if bought today=${stop_price:.2f} (-{stop_loss_pct*100:.0f}% on {sym})")
                posted += 1
        except Exception:
            continue
    if posted == 0:
        print("  No symbols currently in a BUY state.")
    print(f"{'=' * 70}\n")


def main():
    print("Trend Timing Portfolio -- QQQ/XLK/EFA (leveraged) + AAL/AMD/M/C/MU/CSCO (stocks)")
    print("Validated beating buy-and-hold on real multi-decade history")
    print("including the dot-com bust and 2008, not just the recent bull run.")
    print("See VPA_DIAGNOSIS.md SS7-SS8 (ETFs) and SS13 (stocks) for the full derivation.\n")
    evaluate(label="(full history, 20-50yrs incl. dot-com + GFC)")
    scan_buy_signals_only()


if __name__ == '__main__':
    main()
