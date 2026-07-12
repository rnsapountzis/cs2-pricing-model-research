# Forward Validation & Closing-Line Benchmark — Plan (in progress)

> Status: **design / work-in-progress.** This document describes an intended
> validation step and the data-collection scaffolding for it. The
> model-probability join and the CLV/ROI evaluation described under *Planned work*
> are **not yet implemented in this repository** — this is a plan, not a result.

## Why this step is needed

The locked-test diagnostics in this repo carry a stated limitation: the historical
test set was used to diagnose multiple candidates, so it is no longer a clean
promotion gate. An honest promotion would require a **fresh forward hold-out** the
model has never seen.

There is also a metric gap. Proper scores (log loss, AUC, Brier, ECE) answer "is it
well-calibrated?", not "does it beat the market?". For a pricing model the relevant
yardstick is **Closing-Line Value (CLV)** and realized ROI against the actual
pre-match line — which requires the market price *at the moment a bet would be
placed*, with a trustworthy timestamp.

An exploration of historical bookmaker odds showed they cannot support this:

- the public source stores only a single line per finished match, with **no
  timestamp** and no movement history;
- coverage collapses for older seasons (≈0% for 2023–2024 in a per-match backfill;
  usable pairs concentrated in the most recent season);
- many stored lines were degraded/near-settled (odds ≈ 1.001, overround ≈ +99%),
  i.e. not clean pre-match prices.

So historical odds can measure *market availability*, but not CLV. The only rigorous
route is to capture the pre-match line **forward, with timestamps**.

## What exists today (data-collection scaffolding)

A separate (private) capture utility snapshots the upcoming match line at fixed
offsets before kickoff and writes timestamped rows. Each row carries:

- `captured_at`, `start_date`, a `pre_match` flag, and a heuristic `veto_state`
  (`pre_veto` / `maybe_post_veto` / `in_play_post_veto`), so post-veto or in-play
  captures can be identified and excluded — the mechanism that keeps the eventual
  evaluation set leakage-safe by construction;
- the market series line (and series markets) and the data provider's own AI
  prediction fields.

This is deliberately a **collection** step. It accumulates a forward, out-of-sample
panel; it does **not** by itself decide anything.

## Planned work (not yet built)

1. **Model-probability join.** Attach the V8/V12 series-win probability to each
   captured pre-match snapshot at the matching timestamp. *(Not implemented here.)*
2. **De-vigged market probability.** Convert the two-sided line to an implied
   probability with the overround removed. *(Not implemented here.)*
3. **CLV / ROI evaluation.** On the accumulated forward window, measure where the
   model disagreed with the market, whether those positions beat the closing line,
   and the realized ROI — reported on data collected *after* all model selection,
   i.e. a genuine forward hold-out. *(Not implemented here.)*
4. **Incremental-value check.** Whether the model adds anything over the provider's
   free AI prediction, or merely tracks it.

## Honest scope

- The public odds feed exposes a **single featured bookmaker** and **series-level**
  markets only (no per-map / post-veto line), so this would benchmark one book, not
  the full market consensus.
- The forward panel needs several weeks of accumulation before any CLV/ROI statement
  is defensible. Consistent with the rest of this project, a positive point estimate
  would not be treated as promotion until it clears reliability and forward-stability
  checks.

The purpose of this stage is not to declare an edge — it is to define, and start
collecting, the only kind of evidence that could honestly justify one.
