# Analysis log

Chronological record of what was built, what broke, and what changed as a result.
This is not a success story with the failures removed; it is the order in which the
evidence arrived.

The project was built by a subagent working to a fixed charter, then audited by the
lead. The audit changed two things: the bootstrap unit (rows → SKUs) and the
comparator claims. Both are recorded below rather than silently patched.

---

## Session 1 — dataset and integrity

The charter fixed UCI Online Retail II. Downloaded the published ZIP, verified the
ZIP hash and the extracted workbook hash, and wrote `data/raw/manifest.json` with
retrieval time, byte size and sheet audit.

First surprise: the workbook has **two yearly sheets that overlap** (2010-12-01 →
2010-12-09 appears in both). A naive concatenation double-counts that week.
Decision: de-duplicate before any aggregation (D2), and audit the overlap
explicitly instead of trusting the row count.

Second: cancellations are real rows with negative quantities. Netting them into
demand would mix returns with sales and make the target uninterpretable. Decision:
separate them, never net (D3).

Raw rows: **1,067,371**.

---

## Session 2 — cleaning and the frozen sample

Built the clean panel step by step, with a reconciliation after each step:

1. de-duplicate sheet overlap;
2. drop non-product stock codes with an explicit list plus a strict regex;
3. separate cancellations (19,104 rows, −476,819 units);
4. drop non-positive prices;
5. build a complete ISO-week spine (104 weeks);
6. restrict to SKUs with at least 8 positive weeks (panel) and at least 12 before
   the first origin (model sample);
7. reconcile the row audit.

Step 7 initially did not reconcile. The missing piece was that dropping the two
partial calendar weeks (2009-11-30 and 2011-12-05) removes rows that earlier steps
had counted. Once the partial weeks were made explicit, the audit closed:
**1,067,371 → 972,068 rows (91.0712%)**, 4,675 SKUs after cleaning, **3,995** in the
universe, **3,186** in the model sample.

The model sample is frozen on pre-origin information. No SKU enters or leaves
because of what happens in the test window.

Zero-demand share in the model sample: **46.93%**. Top 1% of invoices: **20.46%** of
units. The data are sparse and bulk-driven, which matters for every metric later.

---

## Session 3 — baselines, and the first uncomfortable result

Fitted the four simple baselines first, on the same rolling origins.

At h=1, MASE:

| Model | MASE |
|---|---:|
| SES | **0.819** |
| historical mean | 1.319 |
| Croston/SBA | 1.569 |
| seasonal naive | 1.723 |

Two things were already clear:

- **Croston/SBA loses to the historical mean.** The intermittent-demand standard
  does not fit this retailer's inter-arrival pattern. This is a finding, not a bug;
  it is reported as such.
- **Seasonal naive is unusable.** Two years give one seasonal cycle, and the two
  Decembers differ sharply. It is kept as a baseline, not a candidate.

No model tuning was done on the test window. The fit window ends 13 weeks before
each origin.

---

## Session 4 — point models and the aggregate/individual tension

Added the Poisson GLM and the two LightGBM objectives (Tweedie mean, quantile-0.5
median).

LightGBM median won the per-SKU MASE contest (**0.686**). It also had the worst
pooled WAPE (**40.05%**), because a median forecast is below the mean for a
right-skewed, zero-inflated target. The Poisson GLM behaved the same way at a
smaller scale (MASE 1.219, WAPE 26.01%).

This is the central tension of the project. It is not a metric bug: MASE rewards
accuracy in the middle of a distribution with a large zero atom; the aggregate and
the newsvendor punish systematic under-ordering. The docs report both and do not
choose the metric that makes the preferred model look best.

---

## Session 5 — the uncertainty layer failed the first time

The first attempt modelled uncertainty as a negative binomial with variance
`μ + αμ²`, estimated per SKU. It **collapsed the median** and destroyed interval
coverage: the fitted α absorbed the overdispersion in a way that pulled the central
forecast down.

Replaced it with a variance-stabilising scale:

```
z = (y − μ) / sqrt(μ + 0.25)
```

then empirical quantiles of `z` on the 13-week pre-origin window **separately per
demand class** (Mondrian conformal), mapped back to the count scale. The `+0.25`
keeps the scale finite at μ = 0.

Result at h=1: 80% interval coverage **0.778–0.806** across models; strict
`P(y < q10)` **0.084–0.135**; `P(y ≤ q90)` **0.888–0.914**. Not perfect, but
calibrated enough to use, and the failures are in the tails where the zero atom
lives.

The episode is recorded because it is the kind of thing that would otherwise be
quietly deleted: the first probabilistic layer did not work.

---

## Session 6 — decision model, and a bootstrap bug

Built the newsvendor sweep: lead times 2/4/8, cost ratios 1/3/9/19, holding cost 1%
of unit price per week, underage cost = ratio × holding cost. Everything is swept,
never presented as "the" cost.

The first bootstrap resampled **rows**. That treats the 9,558 SKU-week observations
as independent, which they are not: the same SKU appears at three origins and at
multiple horizons. The audit changed it to resample **SKUs** with replacement, using
one index matrix per lead time so every model is compared on the same resample.

Consequence: the CIs widened. The central SES-quantile saving versus SES point is
**23.88% [21.91, 26.67]** after the fix. The earlier row bootstrap was too narrow.
This is one of the two changes the lead made to the subagent's work.

---

## Session 7 — the comparator was too weak, and the claim had to shrink

The original headline compared the quantile policy with the **best simple point
rule**, SES. That is a defensible comparator but not the strongest one: at the
central case, LightGBM-Tweedie **point** is cheaper than SES point.

Added a second comparator to every decision row:
`best_point_model_all`, `best_point_cost_all`, `savings_vs_best_point_all_pct`.
Against it, the central SES-quantile saving is **18.53%**, not 23.88%. At
`Cu/Co = 1`, the SES quantile policy **loses** to the strongest point model at every
lead time: **−7.89%** (L=2), **−11.20%** (L=4), **−1.81%** (L=8). At ratio 3 it is
roughly break-even.

The README was rewritten around that. The 23.9% number is still there, but it is
labelled "versus SES point", and the negative low-asymmetry result is stated in the
first section rather than buried.

The symmetric-cost claim also had to shrink. SES gains at most 4.22% at ratio 1, but
LightGBM median reaches **14.21%** at L=4, ratio 1 — so "the gain is small at
symmetric costs" is model-specific, not universal. The docs now say exactly that,
and note that the LightGBM median gain collapses to 1.6% at ratio 3.

---

## Session 8 — figures and the second audit

Visual inspection of all seven figures found two problems:

- `fig4` plotted only savings versus SES point. A reader would not see the stronger
  comparator. Added a second row with savings versus the strongest point model.
  The bottom row has **no CI**: the bootstrap is defined against SES, so drawing a
  CI for the other difference would be fabricated.
- `fig6` had overlapping model labels around the quantile cluster. Replaced fixed
  offsets with a deterministic greedy placement that tries a ring of offsets and
  keeps the first non-overlapping one.

The pipeline was rerun from cache after the changes: 118 s, 13 tests passed,
`results.json` regenerated, no metric moved.

---

## Session 9 — reproducibility

Two full runs:

- `python src/run_all.py`: download, clean from the workbook, backtest, decision,
  figures, tests. About 240 s on the first build.
- `python src/run_all.py --from-cache`: reuses the raw parquet. About 118–150 s.

The second run reproduced every accuracy, calibration and decision number exactly.
`results.json` records the commit, the seed (`20261024`), package versions and stage
timings. LightGBM is fitted with `deterministic=True`.

---

## What I would do differently

1. **Decide the comparator before the first result.** The strongest point model
   should have been in the decision table from day one, not added after an audit.
2. **Resample the unit of independence from the start.** The row bootstrap looked
   fine and was wrong.
3. **Model the zero atom explicitly.** A hurdle or censored model would fit the
   data-generating process better than a single Tweedie/quantile objective.
4. **Reserve more origins for time uncertainty.** Three adjacent origins cannot
   support a claim about regime change, and no amount of bootstrapping fixes that.
5. **Build the decision-side bulk trim at the same time as the accuracy-side one.**
   Reporting one and not the other invites the objection that the decision result
   is driven by the tail that was trimmed elsewhere.

---

## Honest summary

The project set out to test whether calibrated probabilistic forecasts reduce
inventory cost. They do — at high cost asymmetry, and by less against the strongest
point model than against the weakest defensible one. At low asymmetry they can lose.
The most accurate per-SKU point model is the worst ordering rule, which is the
result worth remembering.
