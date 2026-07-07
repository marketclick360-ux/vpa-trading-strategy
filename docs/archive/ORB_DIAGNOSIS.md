# ORB Research — Diagnosis: why every variant is "STATISTICALLY INCONCLUSIVE"

**TL;DR — this is a data problem, not a filter-tuning problem.** Opening-Range
Breakout (ORB) is an *intraday* strategy, but the only multi-year price history
available here is **daily** bars. On daily data the "opening range" does not
exist, and the only intraday source on hand (Yahoo 5-minute) is capped at ~60
days — far too little to ever reach a statistically meaningful 100+ trades. No
amount of tuning the Gann/VWAP filters changes that ceiling.

> Note: `orb_research.py` itself was not pushed to the repo, so this is a
> data-and-method audit, not a line-by-line code review. The binding constraint
> below holds regardless of the script's internals. Push the file and I'll add a
> line-level pass.

## Evidence

1. **The 15-year cache is daily.** `schwab-backtest/etf_data/*.json` candles are
   spaced exactly 24h apart (verified). A daily bar has only O/H/L/C — there is
   no intraday sequence from which to build an "opening range." ORB is undefined
   on it.

2. **The project's own framework says so.** The scanner's
   `opening_range.py` setup documents: *"Runs on INTRADAY frames (15m or 1h)
   only … if only daily data is available this family is not testable and must
   be disclosed as such."* The table's `STATISTICALLY INCONCLUSIVE` flag is that
   disclosure firing correctly.

3. **Intraday history is severely capped.** `fetch_real_data.py` can pull 5m
   bars (`--interval 5m --range 60d`) but warns: *"Yahoo only serves intraday
   history for a limited window per interval."* Yahoo's 5m feed is ~60 calendar
   days; Schwab's MINUTE feed is similarly bounded. So the realistic intraday
   window is weeks, not years.

4. **The sample sizes match the data ceiling, not a real edge.** The "best"
   rows are **8–35 trades**:
   - IWM "rank 1" = 8 trades = **5 wins / 3 losses**. A 1.80 profit factor on
     that sample is indistinguishable from a coin flip.
   - ~60 days of intraday history × a heavily-filtered ORB on 3 symbols yields
     exactly this order of magnitude. You cannot reach 100+ trades from a 60-day
     window without years of intraday data.

5. **VWAP on daily bars is degenerate.** The `*_vwap_*` variants need an
   intraday volume profile to mean anything. Computed from a single daily bar,
   VWAP collapses to roughly the typical price `(H+L+C)/3`, so those variants
   aren't contributing the intraday information their names imply — unless run on
   the 5m data.

## Why the ranking is misleading

Sorting *inconclusive* samples by profit factor surfaces the **luckiest small
sample**, not the best strategy. IWM landing at rank 1 on 8 trades is
survivorship inside noise. The Status column is the real signal here, and it is
telling you not to trust the rank.

## What it would actually take to test ORB

| Requirement | Why | Status |
|---|---|---|
| Multi-year **intraday** bars (1–5m), per symbol | Define the opening range; reach 100+ trades | ❌ not available (only ~60d) |
| A paid intraday source (Polygon/Databento/IBKR) **or** forward-accumulating 5m bars over time | Yahoo/Schwab caps make a long history impossible to backfill | ❌ |
| Walk-forward across many sessions | Same OOS rigor we applied to the Radar | blocked by data |

Until that data exists, ORB stays in the *"not testable — disclose as such"*
bucket. Treat the current table as a wiring/plumbing check that the harness runs
end to end — **not** as evidence of an edge.

## Recommendation

- **Do not allocate** (paper or live) based on this table. 8–35 trades is noise.
- If ORB is a priority, the next step is **data**, not parameters: stand up an
  intraday source and start logging 5m bars forward so a real sample accumulates.
- For finding a *validated* edge now, the **Tactical Signal Radar**
  (`trading_system`) is the better-supported path — it has ~1,500 out-of-sample
  trades and the edge survived walk-forward. ORB has 8.
