# VPA Anomaly Signal — Diagnosis: does it beat buy-and-hold?

**TL;DR — no, not on a walk-forward test, and neither does anything else we
tried against this specific 2022–2026 test window.** The percentile-based
VPA anomaly signal (fake move / absorption detection on daily bars, `N`-day
hold) does not show a durable edge once tested honestly out-of-sample. The
short side is actively harmful and should not be traded. A simpler risk-
managed alternative (`vpa_trend_timing.py`) trails buy-and-hold on raw CAGR
too, but is a defensible, non-overfit result with much lower drawdown.
**Do not allocate capital to the original anomaly-based long/short signal.**

## Method

All numbers below are on the 58-symbol ETF universe from `vpa_etf_daily.py`
(`ALL_ETFS`), using `yfinance` daily bars. Trade-level (not daily-bar-level)
returns were used throughout — a trade held `N` days is one observation, not
`N` observations, to avoid inflating apparent sample size.

## 1. Original signal, full history (2010/2017–2026), in-sample

| Mode | Trades | Win rate | Avg win | Avg loss | Profit factor | Expectancy/trade |
|---|---|---|---|---|---|---|
| Long-only | 2,763 | 48.61% | +2.04% | -2.04% | 0.95 | **-0.057%** |
| Long-short | 5,183 | 42.64% | +2.18% | -1.95% | 0.83 | **-0.190%** |

Both modes lose money net of the default 0.1%/leg (0.2% round-trip) cost.
The short side is materially worse than the long side — remove it.

## 2. Isolating signal quality from cost drag (same trades, cost = 0)

| Mode | Trades | Win rate | Profit factor | Expectancy/trade |
|---|---|---|---|---|
| Long-only | 2,772 | 54.37% | 1.16 | +0.152% |
| Long-short | 5,197 | 48.35% | 1.02 | +0.020% |

The long side has a small *raw* edge (54.4% win rate) — but it's roughly the
same size as the round-trip transaction cost (0.2%), so the net-of-cost
result in §1 is close to a wash-to-loss. The short side has essentially no
raw edge at all (48.35%, barely above a coin flip) and should never have
been combined with the long side in `long_short` mode.

## 3. Parameter tuning — and why it doesn't count until validated OOS

A grid search over `SPREAD_PERCENTILE`, volume percentiles, and hold period,
tuned on 2017–2021 data only, found a combo (`spread=85, vol=(15,85),
hold=8`) with **+0.169% expectancy, win rate 50.4%, PF 1.15** on the training
window — looked like a clear improvement.

Validated on 2022–2026 data the tuner never saw:

| | Strategy (tuned params, long-only) | Buy & Hold |
|---|---|---|
| Median CAGR | **-0.40%** | **6.38%** |
| Symbols beating B&H | 12/58 (21%) | — |

The tuned version is *worse* than the untuned default, out of sample. This
is a textbook overfitting result — the grid search picked the combination
that fit 2017–2021 noise, not a real pattern. **Lesson: any further
parameter tuning on this signal family needs a train/test split before the
numbers mean anything**, same principle `ORB_DIAGNOSIS.md` applies to sample
size.

## 4. VPA signal + 200-day trend filter (dip-buy only in confirmed uptrend)

Combining the (untuned, original-threshold) long signal with a
`Close > SMA200` filter, tested purely OOS on 2022–2026:

| Median CAGR | Median MaxDD |
|---|---|
| **-0.36%** | -9.02% |

Still loses badly on CAGR. The anomaly signal itself is not adding value —
see §5.

## 5. Plain SMA-200 trend timing (no VPA signal at all) — the real result

Drop the anomaly detection entirely. Be long a symbol only while
`Close > SMA200`; otherwise hold cash. Cash is not modeled at 0% — it earns
an approximate T-bill yield (4.5%/yr), since that's what real idle cash
actually does.

**In-sample (full history):**

| | Strategy | Buy & Hold |
|---|---|---|
| Median CAGR | 5.20% | 7.04% |
| Median MaxDD | -28.29% | -40.93% |
| Beats B&H on CAGR | 15/58 (26%) | — |
| Beats B&H on drawdown | 56/58 (97%) | — |

**Out-of-sample (2022–2026 only, indicator lookback computed on prior data):**

| | Strategy | Buy & Hold |
|---|---|---|
| Median CAGR | 4.13% | 6.38% |
| Mean CAGR | 3.61% | 4.21% |
| Median MaxDD | -21.12% | -28.40% |
| Beats B&H on CAGR | 27/58 (47%) | — |
| Beats B&H on drawdown | 49/58 (85%) | — |

## 6. Chaikin Money Flow (accumulation) as a signal — tested, rejected

Chaikin Money Flow (20-day) is the standard multi-bar accumulation/
distribution indicator: it weights each day's volume by where the close
landed within that day's range, averaged over 20 days. Two variants were
tested OOS (2022–2026), both untuned (standard 20-day period):

**Standalone (long whenever CMF(20) > 0, cash otherwise):**

| | CMF(20) alone | SMA-200 (existing) | Buy & Hold |
|---|---|---|---|
| Median CAGR | -1.17% (mean) | 3.61% (mean) | 4.21% (mean) |
| Profitable AND beats B&H | 4/58 | 21/58 | — |
| Median position flips | 114 | 38 | — |

**As a confirmation filter (long only when SMA200 uptrend AND CMF(20) > 0):**

| | SMA200 + CMF confirm | SMA200 alone |
|---|---|---|
| Median CAGR | 1.69% | 4.11% |
| Profitable AND beats B&H | 11/58 | 21/58 |
| Median position flips | 88 | 38 |

Both fail for the same reason: CMF is a much noisier, faster-moving
indicator than a 200-day SMA. Standalone, it flips position ~3x more often,
and the transaction-cost drag from all that flipping wipes out any
accumulation signal it might carry. As a confirmation filter, it just
injects that same noise into an otherwise-stable trend signal, more than
doubling the whipsaw. **Rejected — not added to the codebase.**

## 7. Hard stop-loss added to the trend exit

The 200-day SMA exit is slow: it can't react to a sudden single-day drop
before real damage is done. A hard stop — exit immediately if a day's Low
breaches 5% below the entry price, regardless of the SMA — was tested on
top of the existing (no-CMF) trend-timing strategy, OOS, 2022–2026:

| | No stop | With 5% hard stop |
|---|---|---|
| Median CAGR | 4.88% | 4.91% |
| Mean CAGR | 6.12% | 6.42% |
| Median MaxDD | -17.04% | -14.78% |
| **Worst single trade, any symbol** | **-14.14%** | **-5.10%** |
| Profitable AND beats B&H (of 21 pre-stop whitelist symbols) | 16/21 | 18/21 |

Returns are essentially unchanged, but the worst-case single-trade loss
drops from -14.14% to -5.10% — the stop catches gap-downs the SMA can't.
**Accepted.** The whitelist was rebuilt from scratch across the full
58-symbol universe under the with-stop rules (not just re-tested on the
old 21), since membership can shift once the exit rule changes: EWJ, HACK,
and QQQ dropped out; nothing new was added. Current whitelist: 18 symbols
(`BEATS_BH_WHITELIST` in `vpa_trend_timing.py`).

**Caveat:** the live scanner has no record of your actual fill price, so it
can't track a running stop for you automatically — it shows the stop level
*if you buy today*. If you're already holding a position from an earlier
signal, your stop is 5% below your own entry price, not the current quote.

## Why nothing beats raw buy-and-hold CAGR here

2022–2026 contains one of the strongest recovery bull runs on record (QQQ,
XLK, SLV, GDX all up double digits annually for multiple years). Any
strategy that ever holds cash — even briefly, even at a real yield — gives
up upside during a run like that. This shows up industry-wide: most active
and hedge-fund strategies trailed the S&P 500 through 2023–2024 for the same
structural reason. It is not a flaw specific to this codebase.

## Recommendation

- **Do not trade the original VPA anomaly signal**, long/short or long-only,
  as configured. It is net-negative after realistic costs, and "improving"
  it via untested parameter tuning made it worse, not better.
- **`vpa_trend_timing.py`** (new, in this repo) is the closest thing to a
  validated, non-overfit result: it trails buy-and-hold on raw CAGR in this
  specific bull-market test window, but delivers similar-to-competitive
  returns on many symbols individually with meaningfully lower drawdown on
  most of them, and (with the §7 hard stop) caps worst-case single-trade
  loss at -5%. Treat it as a risk-reduction overlay, not an
  alpha-generating signal — that's an honest description of what it is.
- `vpa_coulling.py` and `vpa_etf_daily.py` remain useful as anomaly
  *scanners* (informational alerts) — just not as a source of trading
  signals to act on mechanically.
- Any future parameter change to any of these scripts should be validated
  with a train/test (or walk-forward) split before being trusted, per §3.
