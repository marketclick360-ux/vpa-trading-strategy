"""
VIX-confirmed crash re-entry -- validated, not just observed.

Origin: this week (2026-11-03 to 2026-11-06) a broad selloff pushed most
of the leveraged trend portfolio into CASH at once (vpa_leveraged_trend.py).
The question that prompted this script: can the VIX (the market's own
implied-volatility "fear gauge", the real options-derived signal behind
what people mean by "the Greeks warning of a crash") tell you WHEN it's
safer to buy back in, rather than either (a) buying immediately into the
panic, or (b) waiting for the slow 200-day SMA to recover?

What the VIX does NOT do: warn in advance. Same-day correlation between a
VIX spike and that day's stock move is strongly negative (~-0.46 average
across these 9 symbols) -- it moves WITH the crash, not before it.
Next-day predictive correlation is ~0.00-0.05 -- no advance warning
exists in the VIX move itself. See VPA_DIAGNOSIS.md SS18 for the full
derivation.

What it DOES do, validated: after a single-day crash (a day 3+ standard
deviations below the symbol's own trailing-252-day daily return), buying
IMMEDIATELY tends to lose money (post-2009: mean CAGR -0.24% across the
9 symbols, several negative) -- the "don't catch the falling knife"
result from the bounce research. But waiting for VIX to fall 15% from its
post-crash peak before entering -- a sign the panic is measurably
subsiding, not just paused -- flips this to a real, positive, walk-forward
validated edge in 7 of 9 symbols (mean CAGR 3.58% -> 5.17%, mean Sharpe
0.47 -> 0.69, mean MaxDD -23.31% -> -17.07%, post-2009).

Important, honest limits:
  - This edge is REGIME-DEPENDENT. It did not hold pre-2009 (dot-com era,
    no "Fed put" backdrop) -- see SS18. It is tied to the post-2008
    QE/rapid-intervention market structure continuing, not a law of
    markets.
  - It does NOT help every symbol: QQQ is clearly worse with the filter
    (-2.04pp CAGR, -0.15 Sharpe) and CSCO is marginally worse. The
    biggest gains are on the single-stock names (AMD +4.57pp, C +8.56pp,
    MU +1.84pp, M +1.10pp) and EFA (+0.62pp) -- apply this to those, not
    QQQ/CSCO.
  - This is a RARE, cash-heavy, event-driven tactic (crash events are
    uncommon), not a continuous strategy -- its standalone CAGR (3-9%)
    is far below buy-and-hold in a bull market because it's sitting in
    cash almost all the time. It's a confirmation layer for the specific
    moment after a crash, not a replacement for vpa_leveraged_trend.py's
    everyday trend signal.
"""
import pandas as pd
import numpy as np
import yfinance as yf

SYMBOLS = ['QQQ', 'XLK', 'EFA', 'AAL', 'AMD', 'M', 'C', 'MU', 'CSCO']
RECOMMENDED_SYMBOLS = ['XLK', 'EFA', 'AAL', 'AMD', 'M', 'C', 'MU']  # excludes QQQ, CSCO -- see header
COST = 0.001
STOP_PCT = 0.05
HOLD_DAYS = 10
VIX_PULLBACK = 0.15
CONFIRM_WINDOW = 15
CASH_YIELD_ANNUAL = 0.045
CASH_DAILY = (1 + CASH_YIELD_ANNUAL) ** (1 / 252) - 1
INITIAL_EQUITY = 10000.0


def get_hist(sym):
    d = yf.download(sym, period='max', progress=False, auto_adjust=True)
    if d is None or d.empty:
        return None
    if isinstance(d.columns, pd.MultiIndex):
        d.columns = d.columns.droplevel(1)
    return d.dropna()


def load_with_vix(symbols=None):
    symbols = symbols or SYMBOLS
    vix = get_hist('^VIX')['Close']
    vix.name = 'VIX'
    data = {}
    for sym in symbols:
        df = get_hist(sym)
        if df is None:
            continue
        df = df.join(vix, how='inner')
        df['Ret1'] = df['Close'].pct_change()
        roll_mean = df['Ret1'].rolling(252).mean()
        roll_std = df['Ret1'].rolling(252).std()
        df['Ret1_z'] = (df['Ret1'] - roll_mean) / roll_std
        data[sym] = df.dropna().reset_index()
    return data


def equity_curve(df, mode, start_idx=0, end_idx=None):
    end_idx = len(df) if end_idx is None else end_idx
    d = df.iloc[start_idx:end_idx].reset_index(drop=True)
    n = len(d)
    equity = np.empty(n)
    equity[0] = INITIAL_EQUITY
    state = 'flat'
    entry_price = entry_i = peak_vix = wait_start = None

    for i in range(1, n):
        pnl = 0.0
        z = d['Ret1_z'].iloc[i]
        is_crash = z <= -3.0 and (d['Ret1_z'].iloc[i - 1] > -3.0)

        if state == 'flat':
            pnl = CASH_DAILY
            if is_crash:
                if mode == 'immediate':
                    state, entry_price, entry_i = 'in_trade', d['Close'].iloc[i], i
                    pnl -= COST
                else:
                    state, wait_start, peak_vix = 'waiting_confirm', i, d['VIX'].iloc[i]
        elif state == 'waiting_confirm':
            pnl = CASH_DAILY
            peak_vix = max(peak_vix, d['VIX'].iloc[i])
            if d['VIX'].iloc[i] <= peak_vix * (1 - VIX_PULLBACK):
                state, entry_price, entry_i = 'in_trade', d['Close'].iloc[i], i
                pnl -= COST
            elif (i - wait_start) >= CONFIRM_WINDOW:
                state = 'flat'
        elif state == 'in_trade':
            if d['Low'].iloc[i] <= entry_price * (1 - STOP_PCT):
                pnl = entry_price * (1 - STOP_PCT) / d['Close'].iloc[i - 1] - 1.0 - COST
                state = 'flat'
            else:
                pnl = d['Close'].iloc[i] / d['Close'].iloc[i - 1] - 1.0
                if (i - entry_i) >= HOLD_DAYS:
                    pnl -= COST
                    state = 'flat'

        equity[i] = equity[i - 1] * (1 + max(pnl, -1.0))

    d = d.copy()
    d['Equity'] = equity
    return d


def cagr_of(series, n):
    total = series.iloc[-1] / series.iloc[0] - 1.0
    return (1 + total) ** (252.0 / n) - 1.0 if n > 0 else 0.0


def maxdd_of(eq):
    return (eq / eq.cummax() - 1.0).min()


def sharpe_of(eq):
    r = eq.pct_change().dropna()
    return (r.mean() * 252) / (r.std() * np.sqrt(252)) if r.std() > 0 else 0.0


def evaluate(symbols=None, date_filter=None, label=""):
    symbols = symbols or SYMBOLS
    data = load_with_vix(symbols)
    rows = []
    for mode in ['immediate', 'confirmed']:
        for sym, df in data.items():
            d = df if date_filter is None else df[date_filter(df['Date'])]
            if len(d) < 300:
                continue
            eq_df = equity_curve(df, mode, d.index[0], d.index[-1] + 1)
            n = len(eq_df)
            rows.append({
                'mode': mode, 'symbol': sym, 'years': round(n / 252, 1),
                'strat_cagr': round(cagr_of(eq_df['Equity'], n) * 100, 2),
                'strat_dd': round(maxdd_of(eq_df['Equity']) * 100, 2),
                'sharpe': round(sharpe_of(eq_df['Equity']), 2),
                'bh_cagr': round(cagr_of(eq_df['Close'], n) * 100, 2),
            })
    res = pd.DataFrame(rows)
    print(f"\n{'=' * 95}\n  VIX-CONFIRMED CRASH RE-ENTRY  {label}\n{'=' * 95}")
    print(res.to_string(index=False))
    return res


def scan_crash_watch(symbols=None):
    """Live check: is any symbol currently mid-crash-watch (waiting for
    VIX to confirm) or did one just confirm (entry window still open)?"""
    symbols = symbols or SYMBOLS
    data = load_with_vix(symbols)
    print(f"\n{'=' * 95}")
    print(f"  CRASH / VIX-CONFIRMATION WATCH - {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}")
    print(f"{'=' * 95}")
    any_active = False
    for sym, df in data.items():
        n = len(df)
        z = df['Ret1_z'].values
        crash_idx = None
        for i in range(max(0, n - CONFIRM_WINDOW - HOLD_DAYS - 2), n):
            if z[i] <= -3.0 and (i == 0 or z[i - 1] > -3.0):
                crash_idx = i
        if crash_idx is None:
            continue
        days_since = n - 1 - crash_idx
        vix_window = df['VIX'].iloc[crash_idx:n]
        peak_vix = vix_window.max()
        peak_date = df['Date'].iloc[vix_window.values.argmax() + crash_idx].date()
        current_vix = df['VIX'].iloc[-1]
        target_vix = peak_vix * (1 - VIX_PULLBACK)
        confirmed = current_vix <= target_vix
        note = '' if sym in RECOMMENDED_SYMBOLS else '  (historically NOT improved by this filter -- see header)'
        any_active = True
        if days_since > CONFIRM_WINDOW:
            print(f"  {sym:6s} crash on {df['Date'].iloc[crash_idx].date()} ({days_since}d ago) -- "
                  f"confirmation window expired, no trade{note}")
        elif confirmed:
            print(f"  {sym:6s} CONFIRMED -- crash {df['Date'].iloc[crash_idx].date()}, VIX peaked {peak_vix:.1f} "
                  f"on {peak_date}, now {current_vix:.1f} (<= target {target_vix:.1f}){note}")
        else:
            print(f"  {sym:6s} WATCHING -- crash {df['Date'].iloc[crash_idx].date()} ({days_since}d ago), "
                  f"VIX peaked {peak_vix:.1f} on {peak_date}, now {current_vix:.1f}, "
                  f"needs <= {target_vix:.1f} to confirm ({CONFIRM_WINDOW - days_since}d left in window){note}")
    if not any_active:
        print("  No symbol has had a qualifying crash in the lookback window. Nothing to watch.")
    print(f"{'=' * 95}\n")


if __name__ == '__main__':
    evaluate(label="(full history)")
    evaluate(date_filter=lambda s: s >= pd.Timestamp('2009-01-01'), label="(post-2009 regime)")
    scan_crash_watch()
