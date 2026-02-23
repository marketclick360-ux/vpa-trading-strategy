import pandas as pd
import numpy as np
import yfinance as yf
import math

# =========================
# CONFIG
# =========================
SYMBOL = 'SPY'
START_DATE = '2010-01-01'
INITIAL_EQUITY = 10000.0
COST_PER_TRADE = 0.001

SPREAD_PERCENTILE = 75
VOLUME_PERCENTILE_LOW = 25
VOLUME_PERCENTILE_HIGH = 75
LOOKBACK_WINDOW = 20
HOLD_BARS = 5


# =========================
# DATA (FREE - yfinance)
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
    data = df.copy()
    data['Spread'] = (data['High'] - data['Low']).abs()
    data['Body'] = data['Close'] - data['Open']
    data['IsUp'] = data['Body'] > 0
    data['IsDown'] = data['Body'] < 0

    data['Spread_p75'] = data['Spread'].rolling(lookback).quantile(0.75)
    data['Spread_p25'] = data['Spread'].rolling(lookback).quantile(0.25)
    data['Vol_p_low'] = data['Volume'].rolling(lookback).quantile(0.25)
    data['Vol_p_high'] = data['Volume'].rolling(lookback).quantile(0.75)

    data['WideSpread'] = data['Spread'] > data['Spread_p75']
    data['NarrowSpread'] = data['Spread'] < data['Spread_p25']
    data['LowVol'] = data['Volume'] < data['Vol_p_low']
    data['HighVol'] = data['Volume'] > data['Vol_p_high']

    # Anomaly Type 1: Wide spread + Low volume (fake move)
    data['Anomaly_FakeUp'] = data['IsUp'] & data['WideSpread'] & data['LowVol']
    data['Anomaly_FakeDown'] = data['IsDown'] & data['WideSpread'] & data['LowVol']

    # Anomaly Type 2: Narrow spread + High volume (absorption)
    data['Anomaly_AbsorbUp'] = data['IsUp'] & data['NarrowSpread'] & data['HighVol']
    data['Anomaly_AbsorbDown'] = data['IsDown'] & data['NarrowSpread'] & data['HighVol']

    # Confirmation (legit moves)
    data['Confirm_Up'] = data['IsUp'] & data['WideSpread'] & data['HighVol']
    data['Confirm_Down'] = data['IsDown'] & data['WideSpread'] & data['HighVol']

    # Combined trade signals
    data['Signal_Long'] = data['Anomaly_FakeDown'] | data['Anomaly_AbsorbDown']
    data['Signal_Short'] = data['Anomaly_FakeUp'] | data['Anomaly_AbsorbUp']

    return data


# =========================
# BACKTEST
# =========================
def backtest_vpa(df, hold_bars=HOLD_BARS, cost=COST_PER_TRADE,
                 initial_equity=INITIAL_EQUITY, mode='long_only'):
    data = df.copy()
    data['Return'] = data['Close'].pct_change().fillna(0.0)

    equity = [initial_equity]
    position = 0
    bars_held = 0
    entry_date = None
    trades = []

    for i in range(1, len(data)):
        row = data.iloc[i]
        prev_row = data.iloc[i - 1]
        daily_ret = row['Return']

        pnl = position * daily_ret if position != 0 else 0.0

        if position != 0:
            bars_held += 1

        # Exit after hold period
        if position != 0 and bars_held >= hold_bars:
            pnl -= cost
            trades.append({
                'entry_date': entry_date,
                'exit_date': row.name,
                'direction': 'LONG' if position == 1 else 'SHORT',
                'bars_held': bars_held,
            })
            position = 0
            bars_held = 0

        # Entry (only if flat, using previous bar signal)
        if position == 0:
            if prev_row.get('Signal_Long', False) and mode in ('long_only', 'long_short'):
                position = 1
                bars_held = 0
                entry_date = row.name
                pnl -= cost
            elif prev_row.get('Signal_Short', False) and mode in ('short_only', 'long_short'):
                position = -1
                bars_held = 0
                entry_date = row.name
                pnl -= cost

        new_eq = equity[-1] * (1.0 + pnl)
        equity.append(new_eq)

    data = data.iloc[:len(equity)]
    data['Equity'] = equity
    data['Strategy_Return'] = pd.Series(equity).pct_change().fillna(0.0).values[:len(data)]
    return data, trades


# =========================
# METRICS
# =========================
def calc_metrics(data, trades, label="VPA Strategy"):
    eq = data['Equity']
    rets = data['Strategy_Return']

    total_ret = eq.iloc[-1] / eq.iloc[0] - 1.0
    n_days = len(rets)
    cagr = (1.0 + total_ret) ** (252.0 / n_days) - 1.0 if n_days > 0 else 0
    vol = rets.std() * math.sqrt(252)
    sharpe = cagr / vol if vol > 0 else 0

    running_max = eq.cummax()
    dd = eq / running_max - 1.0
    max_dd = dd.min()

    total_trades = len(trades)

    n_fake_up = data.get('Anomaly_FakeUp', pd.Series(dtype=bool)).sum()
    n_fake_down = data.get('Anomaly_FakeDown', pd.Series(dtype=bool)).sum()
    n_absorb_up = data.get('Anomaly_AbsorbUp', pd.Series(dtype=bool)).sum()
    n_absorb_down = data.get('Anomaly_AbsorbDown', pd.Series(dtype=bool)).sum()

    print(f"\n{'='*50}")
    print(f"  {label}")
    print(f"{'='*50}")
    print(f"  Total Return:      {total_ret*100:.2f}%")
    print(f"  CAGR:              {cagr*100:.2f}%")
    print(f"  Ann. Volatility:   {vol*100:.2f}%")
    print(f"  Sharpe Ratio:      {sharpe:.2f}")
    print(f"  Max Drawdown:      {max_dd*100:.2f}%")
    print(f"  Total Trades:      {total_trades}")
    print(f"  ---")
    print(f"  Anomalies Detected:")
    print(f"    Fake Up (bearish):     {n_fake_up}")
    print(f"    Fake Down (bullish):   {n_fake_down}")
    print(f"    Absorb Up (bearish):   {n_absorb_up}")
    print(f"    Absorb Down (bullish): {n_absorb_down}")
    print(f"{'='*50}\n")


# =========================
# LIVE SCANNER
# =========================
def scan_for_anomalies_today(symbols):
    print(f"\n{'='*60}")
    print(f"  VPA ANOMALY SCANNER - {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}")
    print(f"{'='*60}")

    for sym in symbols:
        try:
            df = get_daily_data(sym, '2025-06-01')
            df = detect_vpa_anomalies(df)
            last = df.iloc[-1]

            signals = []
            if last['Anomaly_FakeUp']:
                signals.append("FAKE UP (bearish)")
            if last['Anomaly_FakeDown']:
                signals.append("FAKE DOWN (bullish)")
            if last['Anomaly_AbsorbUp']:
                signals.append("ABSORB UP (bearish)")
            if last['Anomaly_AbsorbDown']:
                signals.append("ABSORB DOWN (bullish)")
            if last['Confirm_Up']:
                signals.append("CONFIRMED UP")
            if last['Confirm_Down']:
                signals.append("CONFIRMED DOWN")

            status = ', '.join(signals) if signals else "-- no anomaly"
            print(f"  {sym:6s} | ${last['Close']:.2f} | {status}")
        except Exception as e:
            print(f"  {sym:6s} | ERROR: {e}")

    print(f"{'='*60}\n")


# =========================
# MAIN
# =========================
def main():
    print("Downloading data (free via yfinance)...\n")
    df = get_daily_data(SYMBOL, START_DATE)
    print(f"Data: {df.index[0].date()} to {df.index[-1].date()}, {len(df)} bars\n")

    df = detect_vpa_anomalies(df)

    # Backtest long-only
    data_long, trades_long = backtest_vpa(df, mode='long_only')
    calc_metrics(data_long, trades_long, label=f"VPA Long-Only ({SYMBOL})")

    # Backtest long-short
    data_ls, trades_ls = backtest_vpa(df, mode='long_short')
    calc_metrics(data_ls, trades_ls, label=f"VPA Long-Short ({SYMBOL})")

    # Buy and hold comparison
    bh_ret = df['Close'].iloc[-1] / df['Close'].iloc[0] - 1.0
    bh_cagr = (1.0 + bh_ret) ** (252.0 / len(df)) - 1.0
    print(f"  {SYMBOL} Buy & Hold: Total={bh_ret*100:.2f}%  CAGR={bh_cagr*100:.2f}%\n")

    # SCAN watchlist RIGHT NOW
    watchlist = [
        'SPY', 'QQQ', 'IWM', 'EFA', 'EEM',
        'IEF', 'TLT', 'GLD', 'AAPL', 'MSFT',
        'NVDA', 'TSLA', 'AMZN', 'META', 'GOOGL'
    ]
    scan_for_anomalies_today(watchlist)

    # Save equity curve to CSV
    data_long[['Close', 'Volume', 'Equity', 'Anomaly_FakeUp',
               'Anomaly_FakeDown', 'Anomaly_AbsorbUp',
               'Anomaly_AbsorbDown']].to_csv('vpa_backtest.csv')
    print("Saved backtest results to vpa_backtest.csv")


if __name__ == "__main__":
    main()
