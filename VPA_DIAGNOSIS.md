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
  returns on ~47% of symbols individually with meaningfully lower drawdown
  on ~85% of them. Treat it as a risk-reduction overlay, not an
  alpha-generating signal — that's an honest description of what it is.
- `vpa_coulling.py` and `vpa_etf_daily.py` remain useful as anomaly
  *scanners* (informational alerts) — just not as a source of trading
  signals to act on mechanically.
- Any future parameter change to any of these scripts should be validated
  with a train/test (or walk-forward) split before being trusted, per §3.
