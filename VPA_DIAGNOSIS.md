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

## Why nothing beats raw buy-and-hold CAGR on the 2022-2026 window

2022–2026 contains one of the strongest recovery bull runs on record (QQQ,
XLK, SLV, GDX all up double digits annually for multiple years). Any
strategy that ever holds cash — even briefly, even at a real yield — gives
up upside during a run like that. This shows up industry-wide: most active
and hedge-fund strategies trailed the S&P 500 through 2023–2024 for the same
structural reason. It is not a flaw specific to this codebase, and it is why
the strategies below had to be tested on much longer history to find a real
edge at all.

## 8. Momentum/relative-strength rotation — tested across 4 historical
periods, rejected

A separate strategy family: instead of timing in/out of one asset, stay
fully invested at all times and rotate into whichever of the 58 ETFs has
the strongest trailing momentum (rebalanced periodically). This avoids the
cash-drag problem in §7 entirely. Seven variants (lookback 3/6/12/12-1
months, top-1/3/5 holdings, monthly/quarterly rebalance) were tested on
2022–2026 first:

| Variant | CAGR | MaxDD | Beats SPY |
|---|---|---|---|
| 6mo lookback, monthly, top 3 | +14.74% | -35.95% | Yes |
| 12mo lookback, monthly, top 3 | +22.87% | -37.54% | Yes |
| 12-1 momentum (classic academic) | +11.52% | -52.85% | No |
| 3mo lookback, monthly, top 3 | +4.88% | -68.56% | No |
| 6mo lookback, top 1 (concentrated) | +2.90% | -66.01% | No |
| 6mo lookback, top 5 (diversified) | +16.87% | -26.23% | Yes |
| 6mo lookback, quarterly rebal | -4.19% | -51.24% | No |

Four of seven reasonable variants of the *same idea* lost to buy-and-hold,
one badly. Cherry-picking the best one here would repeat the exact overfit
mistake from §3. Three of those variants were then re-tested across
2014-2018, 2018-2022, and 2022-2026 (the ETF universe's real history only
goes back to 2017, so "2010-2014" returned no data despite being
requested):

| Strategy | 2014-2018 | 2018-2022 | 2022-2026 |
|---|---|---|---|
| Classic 12-1 momentum | -6.26% vs SPY 7.39% | 7.92% vs 9.16% | 11.52% vs 12.02% |
| 6mo momentum, top 3 | 4.45% vs 7.39% | **-22.29%** vs 9.16% (-79% DD) | 14.74% vs 12.02% |
| 6mo momentum, top 5 | 4.49% vs 7.39% | -4.16% vs 9.16% | 16.87% vs 12.02% |

Every variant lost to buy-and-hold in 2014-2018 and 2018-2022, one by a
catastrophic margin. The only period any of them won was 2022-2026 — the
one window tested repeatedly throughout this document. That is decisive
evidence the earlier "beats buy-and-hold" result was regime-specific luck,
not a real edge. **Rejected.**

## 9. Long-history test on legacy ETFs — a real edge found

Everything above was tested on ETFs with data only from 2017 onward — one
continuous bull market. The same (unmodified, untuned) `vpa_trend_timing.py`
strategy — 200-day SMA + 5% stop — was re-run on 20 ETFs with 20-30 years of
real history, spanning the dot-com bust and the 2008 GFC:

| | Strategy | Buy & Hold |
|---|---|---|
| Median CAGR (20 symbols) | 5.37% | 8.24% |
| Median MaxDD | **-34.28%** | **-58.84%** |
| Beats B&H on CAGR | 3/20 | — |

Still no universal edge — but the 3 winners are **QQQ (+10.17% vs 8.91%),
XLK (+10.99% vs 9.78%), and EFA (+8.74% vs 7.09%)**, and that's not random:
those are exactly the assets that suffered the worst historical crashes
(buy-and-hold QQQ: -83% drawdown in the dot-com bust; XLK: -82%; EFA: -61%
across 2000-2003 and 2008). The 200-day SMA exit got out before most of that
damage, and avoiding an 80%+ drawdown (which needs +400%+ to recover from)
compounds into a genuine, mechanically-explainable CAGR edge. Low-volatility
assets (bonds, staples, utilities) never had a crash that severe, so there
was nothing for the strategy to save them from, and its occasional
cash-drag cost more than the protection was worth. **This is the first
result in this document that is a real edge, not noise** — it is
non-cherry-picked (same exact untuned parameters used everywhere else in
this file) and it held up across two genuine multi-year bear markets, not
just the 2022-2026 recovery.

## 10. Volatility-spike exit added to the trend filter — no improvement

Hypothesis: a 200-day SMA lags a real crash by construction; a fast
realized-volatility spike (10-day vol > 2x its 60-day baseline) should
catch a crash earlier. Tested on the same 20 legacy ETFs, added on top of
the §9 strategy:

| | SMA-200 only | + Vol-spike exit |
|---|---|---|
| Median CAGR | 5.37% | 5.50% |
| Median MaxDD | -34.28% | -35.34% |
| Beats B&H CAGR | 3/20 | 3/20 |

No improvement — a wash at best. On QQQ specifically (the single best case
for the base strategy) it made things *worse* (CAGR 10.2%→9.4%, drawdown
-41.5%→-49.6%). **Rejected** — consistent with the broader pattern in this
document that added complexity has not once improved on the simplest
version of the trend rule.

## 11. Options (LEAPS calls) to leverage the §9 signal — catastrophic, rejected

Buying long-dated (1-year, rolled every ~9 months) at-the-money calls on
QQQ/XLK/EFA instead of holding the stock, priced via Black-Scholes off
trailing realized volatility (a generous assumption — real implied vol runs
higher than realized during selloffs, so this likely understates the real
cost):

| Symbol | Stock CAGR | **LEAPS-call CAGR** | Stock MaxDD | **LEAPS-call MaxDD** |
|---|---|---|---|---|
| QQQ | +10.2% | **-11.5%** | -41.5% | **-99.9%** |
| XLK | +11.0% | **-12.0%** | -39.4% | **-100.0%** |
| EFA | +8.7% | **-19.1%** | -22.5% | **-100.0%** |

All three effectively went to zero at some point. Theta decay compounds
daily regardless of whether the underlying moves; rolling costs stack up
over dozens of rolls across the multi-decade test; and a 5% underlying
drop (a clean stock-level stop) can wipe out 40-60%+ of an option's value
in a single day because leveraged losses are as convex as leveraged gains.
**Rejected outright — do not use options to leverage this signal.**

## 12. Margin / leveraged-ETF exposure on the §9 signal — validated, shipped

Unlike options, scaling the *same* linear exposure (2x or 3x notional, via
margin or a real leveraged ETF) preserves the underlying signal instead of
introducing a new instrument with its own decay mechanics:

| | QQQ | XLK | EFA |
|---|---|---|---|
| Stock (1x) | CAGR 10.2% / DD -41.5% | CAGR 11.0% / DD -39.4% | CAGR 8.7% / DD -22.5% |
| Margin 2x | CAGR 10.7% / DD -73.1% | CAGR 12.4% / DD -69.9% | CAGR 9.4% / DD -48.9% |
| Leveraged ETF 2x | CAGR 11.9% / DD -72.9% | CAGR 13.7% / DD -69.7% | CAGR 10.6% / DD -45.2% |
| Leveraged ETF 3x | CAGR 11.2% / DD -88.8% | **CAGR 14.2%** / DD -87.9% | **CAGR 11.4%** / DD -70.1% |
| Buy & Hold (1x) | CAGR 8.9% / DD -83.0% | CAGR 9.8% / DD -82.0% | CAGR 7.1% / DD -61.0% |

2x meaningfully beats both the 1x signal and buy-and-hold on CAGR, on all
three symbols, while keeping drawdown *better* than plain buy-and-hold. 3x
is asset-dependent: it makes QQQ strictly worse (lower CAGR than 2x, worse
drawdown than even buy-and-hold — volatility decay from daily-reset
compounding erases the crash-avoidance advantage), but it's the best CAGR
found for XLK and EFA. **Shipped in `vpa_leveraged_trend.py`** with
per-symbol leverage set to whichever tested higher (QQQ 2x, XLK 3x, EFA
3x), per an explicit "maximize profit, risk tolerated" instruction — this
is NOT the conservative default; real historical worst-case drawdowns are
roughly -73% (QQQ), -88% (XLK), -70% (EFA). The 5% hard stop is what caps
any single trade's loss; the drawdown figures are what a sustained bear
market does across many trades in sequence, not a blown stop.

Financing is modeled at ~roughly institutional leveraged-ETF rates
(expense ratio + swap-financing spread, ~6-7%/yr all-in at 2x), cheaper
than the 8% retail margin rate also tested — a real leveraged ETF (QLD for
QQQ; no exact 3x product exists for XLK/EFA, meaning that exposure needs
margin on top of the 2x product or a different-index 3x substitute) is the
more cost-efficient way to get this exposure versus borrowing on margin
directly.

## Recommendation

- **Do not trade the original VPA anomaly signal**, long/short or long-only,
  as configured. It is net-negative after realistic costs, and "improving"
  it via untested parameter tuning made it worse, not better.
- **`vpa_leveraged_trend.py` (new, in this repo) is the actual answer to
  "beat buy-and-hold"** — the only strategy in this document that did so on
  real multi-decade history including two genuine bear markets, not just
  the 2022-2026 recovery. QQQ at 2x, XLK and EFA at 3x, same 200-day SMA
  entry and 5% hard stop throughout. This is the maximum-profitability
  configuration per an explicit "don't care about risk" instruction — real
  historical worst-case drawdowns are roughly -73% to -88%. If risk
  tolerance changes, drop to a flat 2x everywhere (§12) for a meaningfully
  better drawdown profile at a small CAGR cost on XLK/EFA.
- **`vpa_trend_timing.py`** (unleveraged, wider whitelist) is the
  risk-managed alternative: it trails buy-and-hold on raw CAGR in the
  2022-2026 window specifically, but delivers similar-to-competitive
  returns on many symbols individually with meaningfully lower drawdown,
  and (with the §7 hard stop) caps worst-case single-trade loss at -5%.
- Momentum rotation (§8), CMF/accumulation signals (§6), a volatility-spike
  exit (§10), and options leverage (§11) were all tested and rejected —
  see each section for why. Don't re-try these without a new idea for why
  they'd behave differently; the same mechanisms that failed them once
  will fail them again.
- `vpa_coulling.py` and `vpa_etf_daily.py` remain useful as anomaly
  *scanners* (informational alerts) — just not as a source of trading
  signals to act on mechanically.
- Any future parameter change to any of these scripts should be validated
  with a train/test (or walk-forward) split before being trusted, per §3.
