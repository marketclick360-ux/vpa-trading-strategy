import pandas as pd
import numpy as np
import yfinance as yf
import math
from datetime import datetime

# =========================
# CONFIG
# =========================
START_DATE      = '2017-01-01'   # ~7 years of daily data
INITIAL_EQUITY  = 10000.0
COST_PER_TRADE  = 0.001
LOOKBACK_WINDOW = 20
HOLD_BARS       = 5

# =========================
# FULL ETF UNIVERSE
# =========================
ETF_UNIVERSE = {
    'Broad Market': ['SPY','QQQ','IWM','DIA','VTI','VOO','IVV','MDY','IJR'],
    'Sector':       ['XLK','XLF','XLE','XLV','XLI','XLC','XLP','XLU','XLB','XLRE'],
    'International':['EFA','EEM','VEA','VWO','EWJ','FXI','IEMG','ACWI'],
    'Bonds':        ['TLT','IEF','SHY','AGG','BND','HYG','LQD','TIP','MUB'],
    'Commodities':  ['GLD','SLV','GDX','USO','UNG','DBC','PDBC','IAU','CPER'],
    'Real Estate':  ['VNQ','IYR','SCHH','REM'],
    'Volatility':   ['VIXY','UVXY'],
    'Thematic':     ['ARK','ARKK','ARKG','ARKW','ICLN','LIT','HACK','BOTZ'],
}

# Flatten to one list
ALL_ETFS = [sym for group in ETF_UNIVERSE.values() for sym in group]

# =========================
# DATA
# =========================
def get_daily_data(symbol, start):
    df = yf.download(symbol, start=start, auto_adjust=True, progress=False)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.droplevel(1)
    df = df.dropna()
    return df

# =========================
# VPA ANOMALY DETECTION
# =========================
def detect_vpa_anomalies(df, lookback=LOOKBACK_WINDOW):
    d = df.copy()
    d['Spread']       = (d['High'] - d['Low']).abs()
    d['Body']         = d['Close'] - d['Open']
    d['IsUp']         = d['Body'] > 0
    d['IsDown']       = d['Body'] < 0
    d['Spread_p75']   = d['Spread'].rolling(lookback).quantile(0.75)
    d['Spread_p25']   = d['Spread'].rolling(lookback).quantile(0.25)
    d['Vol_p_low']    = d['Volume'].rolling(lookback).quantile(0.25)
    d['Vol_p_high']   = d['Volume'].rolling(lookback).quantile(0.75)
    d['WideSpread']   = d['Spread'] > d['Spread_p75']
    d['NarrowSpread'] = d['Spread'] < d['Spread_p25']
    d['LowVol']       = d['Volume'] < d['Vol_p_low']
    d['HighVol']      = d['Volume'] > d['Vol_p_high']
    # Anomalies
    d['Anomaly_FakeUp']     = d['IsUp']   & d['WideSpread']   & d['LowVol']
    d['Anomaly_FakeDown']   = d['IsDown'] & d['WideSpread']   & d['LowVol']
    d['Anomaly_AbsorbUp']   = d['IsUp']   & d['NarrowSpread'] & d['HighVol']
    d['Anomaly_AbsorbDown'] = d['IsDown'] & d['NarrowSpread'] & d['HighVol']
    d['Confirm_Up']         = d['IsUp']   & d['WideSpread']   & d['HighVol']
    d['Confirm_Down']       = d['IsDown'] & d['WideSpread']   & d['HighVol']
    d['Signal_Long']  = d['Anomaly_FakeDown'] | d['Anomaly_AbsorbDown']
    d['Signal_Short'] = d['Anomaly_FakeUp']   | d['Anomaly_AbsorbUp']
    return d

# =========================
# BACKTEST
# =========================
def backtest_vpa(df, hold_bars=HOLD_BARS, cost=COST_PER_TRADE,
                 initial_equity=INITIAL_EQUITY, mode='long_only'):
    data = df.copy()
    data['Return'] = data['Close'].pct_change().fillna(0.0)
    equity = [initial_equity]
    position, bars_held, entry_date = 0, 0, None
    trades = []
    for i in range(1, len(data)):
        row      = data.iloc[i]
        prev_row = data.iloc[i - 1]
        pnl = position * row['Return'] if position != 0 else 0.0
        if position != 0:
            bars_held += 1
        if position != 0 and bars_held >= hold_bars:
            pnl -= cost
            trades.append({'entry_date': entry_date, 'exit_date': row.name,
                           'direction': 'LONG' if position == 1 else 'SHORT',
                           'bars_held': bars_held})
            position, bars_held = 0, 0
        if position == 0:
            if prev_row.get('Signal_Long', False) and mode in ('long_only','long_short'):
                position, bars_held, entry_date = 1, 0, row.name
                pnl -= cost
            elif prev_row.get('Signal_Short', False) and mode in ('short_only','long_short'):
                position, bars_held, entry_date = -1, 0, row.name
                pnl -= cost
        equity.append(equity[-1] * (1.0 + pnl))
    data = data.iloc[:len(equity)]
    data['Equity'] = equity
    data['Strategy_Return'] = pd.Series(equity).pct_change().fillna(0.0).values[:len(data)]
    return data, trades

# =========================
# METRICS
# =========================
def calc_metrics(data, trades, symbol, mode):
    eq   = data['Equity']
    rets = data['Strategy_Return']
    n    = len(rets)
    total_ret = eq.iloc[-1] / eq.iloc[0] - 1.0
    cagr      = (1.0 + total_ret) ** (252.0 / n) - 1.0 if n > 0 else 0
    vol       = rets.std() * math.sqrt(252)
    sharpe    = cagr / vol if vol > 0 else 0
    max_dd    = (eq / eq.cummax() - 1.0).min()
    bh_ret    = data['Close'].iloc[-1] / data['Close'].iloc[0] - 1.0
    bh_cagr   = (1.0 + bh_ret) ** (252.0 / n) - 1.0 if n > 0 else 0
    return {
        'Symbol':    symbol,
        'Mode':      mode,
        'Trades':    len(trades),
        'TotalRet':  round(total_ret * 100, 2),
        'CAGR':      round(cagr * 100, 2),
        'Sharpe':    round(sharpe, 2),
        'MaxDD':     round(max_dd * 100, 2),
        'BH_Ret':    round(bh_ret * 100, 2),
        'BH_CAGR':   round(bh_cagr * 100, 2),
    }

# =========================
# TODAY'S SCANNER
# =========================
def scan_today(symbols):
    print(f"\n{'='*70}")
    print(f"  VPA DAILY ETF SCANNER  |  {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print(f"{'='*70}")
    alerts = []
    clean  = []
    for sym in symbols:
        try:
            df = get_daily_data(sym, '2025-06-01')
            if len(df) < LOOKBACK_WINDOW + 2:
                continue
            df = detect_vpa_anomalies(df)
            last = df.iloc[-1]
            sigs = []
            if last['Anomaly_FakeUp']:     sigs.append('FAKE UP   (bearish reversal)')
            if last['Anomaly_FakeDown']:   sigs.append('FAKE DOWN (bullish reversal)')
            if last['Anomaly_AbsorbUp']:   sigs.append('ABSORB UP   (bearish absorption)')
            if last['Anomaly_AbsorbDown']: sigs.append('ABSORB DOWN (bullish absorption)')
            if last['Confirm_Up']:         sigs.append('CONFIRM UP   (trend continuation)')
            if last['Confirm_Down']:       sigs.append('CONFIRM DOWN (trend continuation)')
            price = last['Close']
            vol   = int(last['Volume'])
            if sigs:
                for s in sigs:
                    alerts.append({'Symbol': sym, 'Price': price, 'Volume': vol, 'Signal': s})
            else:
                clean.append(sym)
        except Exception as e:
            print(f"  {sym}: ERROR - {e}")
    if alerts:
        print(f"\n  *** ANOMALIES DETECTED ({len(alerts)}) ***")
        print(f"  {'Symbol':<8} {'Price':>8}  {'Volume':>12}  Signal")
        print(f"  {'-'*60}")
        for a in alerts:
            print(f"  {a['Symbol']:<8} ${a['Price']:>7.2f}  {a['Volume']:>12,}  {a['Signal']}")
    else:
        print("\n  No anomalies detected today.")
    print(f"\n  Clean (no anomaly): {', '.join(clean)}")
    print(f"{'='*70}\n")
    return alerts

# =========================
# BACKTEST ALL ETFs
# =========================
def backtest_all(symbols):
    results = []
    print(f"\nRunning daily backtest on {len(symbols)} ETFs from {START_DATE}...")
    for sym in symbols:
        try:
            df = get_daily_data(sym, START_DATE)
            if len(df) < LOOKBACK_WINDOW + 10:
                continue
            df = detect_vpa_anomalies(df)
            for mode in ('long_only', 'long_short'):
                data, trades = backtest_vpa(df, mode=mode)
                r = calc_metrics(data, trades, sym, mode)
                results.append(r)
        except Exception as e:
            print(f"  {sym}: ERROR - {e}")
    return pd.DataFrame(results)

# =========================
# MAIN
# =========================
def main():
    print("\nAnna Coulling VPA - Daily ETF Scanner & Backtest")
    print(f"Universe: {len(ALL_ETFS)} ETFs | Start: {START_DATE}\n")

    # 1. TODAY'S LIVE SCAN
    alerts = scan_today(ALL_ETFS)

    # 2. BACKTEST ALL ETFs
    results_df = backtest_all(ALL_ETFS)

    # 3. PRINT BACKTEST SUMMARY
    print("\n" + "="*90)
    print("  BACKTEST SUMMARY (Daily | Long-Only) - Sorted by CAGR")
    print("="*90)
    lo = results_df[results_df['Mode'] == 'long_only'].sort_values('CAGR', ascending=False)
    print(lo[['Symbol','Trades','TotalRet','CAGR','Sharpe','MaxDD','BH_Ret','BH_CAGR']].to_string(index=False))

    print("\n" + "="*90)
    print("  BACKTEST SUMMARY (Daily | Long-Short) - Sorted by CAGR")
    print("="*90)
    ls = results_df[results_df['Mode'] == 'long_short'].sort_values('CAGR', ascending=False)
    print(ls[['Symbol','Trades','TotalRet','CAGR','Sharpe','MaxDD','BH_Ret','BH_CAGR']].to_string(index=False))

    # 4. SAVE
    results_df.to_csv('vpa_etf_backtest.csv', index=False)
    print("\nSaved results to vpa_etf_backtest.csv")

if __name__ == '__main__':
    main()
