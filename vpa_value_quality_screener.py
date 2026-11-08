"""
Live value + quality screener (Greenblatt "Magic Formula" style) --
INFORMATIONAL ONLY, NOT BACKTESTED.

Why this isn't validated like everything else in this repo: every other
strategy here (vpa_trend_timing.py, vpa_leveraged_trend.py) was tested on
20-50 years of real price/volume history, which yfinance provides free.
A value/quality factor strategy needs historical FUNDAMENTALS (P/E, ROE,
debt) as they actually looked years ago, not restated today -- that kind
of point-in-time fundamental data isn't available for free (it requires a
paid provider like Compustat, Sharadar, or SimFin). yfinance only exposes
CURRENT fundamentals, so there is no way to backtest this the way the
price-based strategies were backtested.

What this script actually does: ranks today's stock universe by the
classic, well-established Magic Formula (Joel Greenblatt, "The Little
Book That Beats the Market", 2005) -- combined rank of earnings yield
(cheapness) and return on equity (quality). This is a live snapshot for
awareness, same status as vpa_coulling.py's anomaly scanner: not a
mechanical trading signal, not proven to beat buy-and-hold, no stop-loss
or exit rule defined. See VPA_DIAGNOSIS.md SS14.
"""
import pandas as pd
import yfinance as yf

# A diversified large-cap universe across sectors. Not the S&P 500 (no
# free, reliable live constituent list fetched here) -- a representative
# sample instead.
UNIVERSE = [
    'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'NVDA', 'TSLA', 'BRK-B',
    'JPM', 'V', 'MA', 'UNH', 'JNJ', 'PG', 'HD', 'XOM', 'CVX', 'KO', 'PEP',
    'WMT', 'DIS', 'NFLX', 'ADBE', 'CRM', 'ORCL', 'INTC', 'CSCO', 'IBM',
    'QCOM', 'TXN', 'AVGO', 'COST', 'MCD', 'NKE', 'SBUX', 'BA', 'CAT', 'GE',
    'HON', 'LMT', 'UPS', 'FDX', 'GS', 'MS', 'BAC', 'WFC', 'C', 'AXP',
    'BLK', 'AMD', 'MU', 'T', 'VZ', 'ABT', 'PFE', 'MRK', 'LLY', 'TMO', 'DHR',
]


def get_metrics(symbol):
    try:
        info = yf.Ticker(symbol).info
    except Exception:
        return None
    pe = info.get('trailingPE')
    roe = info.get('returnOnEquity')
    price = info.get('currentPrice') or info.get('regularMarketPrice')
    market_cap = info.get('marketCap')
    if pe is None or roe is None or pe <= 0 or price is None:
        return None
    earnings_yield = 1.0 / pe  # cheapness proxy (classic Magic Formula uses EBIT/EV; this is a simpler, common substitute)
    return {
        'symbol': symbol, 'price': price, 'market_cap': market_cap,
        'trailing_pe': pe, 'earnings_yield': earnings_yield,
        'roe': roe, 'debt_to_equity': info.get('debtToEquity'),
        'profit_margin': info.get('profitMargins'),
    }


def screen(universe=None, top_n=15):
    universe = universe or UNIVERSE
    rows = []
    for sym in universe:
        m = get_metrics(sym)
        if m is not None:
            rows.append(m)
    df = pd.DataFrame(rows)
    if len(df) == 0:
        print("No data retrieved.")
        return df

    # Magic Formula: rank by earnings yield (higher=cheaper=better) and by
    # ROE (higher=better quality), combine ranks, lower combined = better.
    df['ey_rank'] = df['earnings_yield'].rank(ascending=False)
    df['roe_rank'] = df['roe'].rank(ascending=False)
    df['combined_rank'] = df['ey_rank'] + df['roe_rank']
    df = df.sort_values('combined_rank')

    print(f"\n{'=' * 90}")
    print(f"  VALUE + QUALITY SCREEN (Magic Formula style) -- INFORMATIONAL, NOT BACKTESTED")
    print(f"  {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}  |  {len(df)}/{len(universe)} symbols scored")
    print(f"{'=' * 90}")
    top = df.head(top_n)
    print(f"{'Symbol':<8}{'Price':>10}{'P/E':>8}{'Earn.Yield':>12}{'ROE':>8}{'D/E':>8}{'Margin':>9}")
    for _, r in top.iterrows():
        de = f"{r['debt_to_equity']:.0f}" if pd.notna(r['debt_to_equity']) else "n/a"
        margin = f"{r['profit_margin']*100:.1f}%" if pd.notna(r['profit_margin']) else "n/a"
        print(f"{r['symbol']:<8}${r['price']:>8.2f}{r['trailing_pe']:>7.1f}x{r['earnings_yield']*100:>10.1f}%"
              f"{r['roe']*100:>7.1f}%{de:>8}{margin:>9}")
    print(f"{'=' * 90}")
    print("NOT a trading signal. No stop-loss, no exit rule, no backtest. See VPA_DIAGNOSIS.md SS14.\n")
    return df


if __name__ == '__main__':
    screen()
