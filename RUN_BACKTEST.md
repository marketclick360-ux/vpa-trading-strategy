# How to Run the VPA ETF Backtest (and get real numbers)

The backtest (`vpa_etf_daily.py`) downloads daily price data from Yahoo Finance
on every run and computes results live — it does **not** ship with numbers. To
see results you have to run it where Yahoo Finance is reachable.

---

## Step 1 — Open the network (one-time, before starting the session)

In **claude.com/code → the `vpa-trading-strategy` environment → Settings →
Network access**, switch off the restricted default. Either:

- pick **Trusted / open network access**, or
- use a **custom allowlist** with these hosts:

```
query1.finance.yahoo.com
query2.finance.yahoo.com
fc.yahoo.com
finance.yahoo.com
```

Save. The policy only applies to **new** containers, so do this *before*
starting the session. (Docs: https://code.claude.com/docs/en/claude-code-on-the-web)

> Running locally instead? You can skip this — Yahoo is reachable from your own
> machine. See "Run locally" below.

---

## Step 2 — Start a fresh session on this repo and paste this prompt

```
Check out branch claude/rule-based-stock-scanner-0ajq8m. Install deps
(pip install -r requirements.txt), then run the VPA ETF backtest with
caching on:

    python3 vpa_etf_daily.py --cache

Show me the full output — the per-ETF backtest table (Trades, CAGR,
Sharpe, MaxDD) for BOTH long-only and long-short, each compared to
buy-and-hold (BH_Ret / BH_CAGR). Then give me an honest verdict: does
the VPA anomaly signal actually beat just holding the ETF, or not?
Count how many of the 59 ETFs the strategy beats buy-and-hold on, and
don't sugarcoat it if the answer is "no edge."
```

---

## Run locally (no network policy needed)

From `~/Documents/GitHub/vpa-trading-strategy`:

```bash
git checkout claude/rule-based-stock-scanner-0ajq8m
pip install -r requirements.txt

# Online run, saving data to data_cache/ for reuse:
python3 vpa_etf_daily.py --cache

# Later / offline (uses only the cached CSVs, never hits the network):
python3 vpa_etf_daily.py --offline
```

Results are also written to `vpa_etf_backtest.csv`.

---

## Cache flags

| Flag | Effect |
|------|--------|
| *(none)* | Online, no caching (default). |
| `--cache` | Save downloads to `data_cache/<SYMBOL>.csv` and reuse them. Falls back to cache automatically if the network fails. |
| `--offline` | Use only cached data, never hit the network. Implies `--cache`. Run once online with `--cache` first to populate it. |
| `--cache-dir DIR` | Use a different cache directory (default: `data_cache`). |

> `data_cache/` is gitignored — market data is never committed.

---

## What the backtest measures

For each of 59 ETFs it flags Anna Coulling VPA anomalies (fake moves / absorption
based on spread + volume), enters **long** for a fixed 5-bar hold, and reports
Trades, TotalRet, CAGR, Sharpe, and MaxDD — **each compared against simply
buying and holding the ETF** (`BH_Ret` / `BH_CAGR`). Beating buy-and-hold across
the universe is the real test of whether the signal has an edge.

---

## Heads-up on scope

This branch contains only the VPA backtest (`vpa_etf_daily.py`,
`vpa_coulling.py`). It does **not** contain `tactical_signal_radar.py` or any
"rule-based stock scanner" / PR #13 work referenced in earlier chats — that code
is not in this repository as it currently stands.
