# Analysis log

Chronological record of what was built, what broke, and what changed as a result.
This is not a success story with the failures removed; it is the order in which the
evidence arrived.

The forecasting engine underwent two rounds of formal code and statistical review. The first review updated the bootstrap resampling unit from row-level to SKU cluster resampling and adjusted benchmark comparator baselines. The second review resolved four technical defects: cumulative conformal window definitions, a bootstrap index alignment, price vector look-ahead leakage, and figure labeling.

---

## Session 1 - dataset and integrity

Selected UCI Online Retail II as the evaluation benchmark. Downloaded the published ZIP, verified the
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

## Session 2 - cleaning and the frozen sample

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

## Session 3 - baselines, and the first uncomfortable result

Fitted the four simple baselines first, on the same rolling origins.

At h=1, MASE:

| Model | MASE |
|---|---:|
| SES | **0.819** |
| historical mean | 1.319 |
| Croston/SBA | 1.692 |
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

## Session 4 - point models and the aggregate/individual tension

Added the Poisson GLM and the two LightGBM objectives (Tweedie mean, quantile-0.5
median).

LightGBM median won the per-SKU MASE contest (**0.686**). Its pooled WAPE was
**40.05%**, the second-worst in the table after seasonal naive (71.09%), because a
median forecast is below the mean for a right-skewed, zero-inflated target. The
Poisson GLM behaved the same way at a smaller scale (MASE 1.219, WAPE 26.01%).

This is the central tension of the project. It is not a metric bug: MASE rewards
accuracy in the middle of a distribution with a large zero atom; the aggregate and
the newsvendor punish systematic under-ordering. The docs report both and do not
choose the metric that makes the preferred model look best.

---

## Session 5 - the uncertainty layer failed the first time

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

## Session 6 - decision model, and a bootstrap bug

Built the newsvendor sweep: lead times 2/4/8, cost ratios 1/3/9/19, holding cost 1%
of unit price per week, underage cost = ratio × holding cost. Everything is swept,
never presented as "the" cost.

The first bootstrap resampled **rows**. That treats the 9,558 SKU-week observations
as independent, which they are not: the same SKU appears at three origins and at
multiple horizons. The audit changed it to resample **SKUs** with replacement, using
one index matrix per lead time so every model is compared on the same resample.

Consequence: the CIs widened. The central SES-quantile saving versus SES point is
**23.82% [21.20, 26.18]** after the fix. The earlier row bootstrap was too narrow.
This adjustment ensures confidence intervals accurately reflect SKU-level clustering.

---

## Session 7 - the comparator was too weak, and the claim had to shrink

The original headline compared the quantile policy with the **best simple point
rule**, SES. That is a defensible comparator but not the cheapest point rule
available: at the central case, LightGBM-Tweedie **point** is cheaper than SES point.

Added a second comparator to every decision row:
`best_point_model_all`, `best_point_cost_all`, `savings_vs_best_point_all_pct`.
Against it, the central SES-quantile saving is **18.62%**, not 23.82%. At
`Cu/Co = 1`, the SES quantile policy **loses** to the cheapest point rule at every
lead time: **−7.70%** (L=2), **−10.76%** (L=4), **−1.54%** (L=8). At ratio 3 it is
roughly break-even.

The README was rewritten around that. The 23.82% number is still there, but it is
labelled "versus SES point", and the negative low-asymmetry result is stated in the
first section rather than buried.

The symmetric-cost claim also had to shrink. SES gains at most 4.23% at ratio 1, but
LightGBM median reaches **13.38%** at L=4, ratio 1 - so "the gain is small at
symmetric costs" is model-specific, not universal. The docs now say exactly that,
and note that the LightGBM-median quantile gain falls to +0.54% at ratio 3 (its point
rule falls to −18.86%).

---

## Session 8 - figures and the second audit

Visual inspection of all seven figures found two problems:

- `fig4` plotted only savings versus SES point. A reader would not see the stronger
  comparator. Added a second row with savings versus the cheapest point rule in each
  cell. The bottom row has **no CI**: the bootstrap is defined against SES, so drawing
  a CI for the other difference would be fabricated.
- `fig6` had overlapping model labels around the quantile cluster. Replaced fixed
  offsets with a deterministic greedy placement that tries a ring of offsets and
  keeps the first non-overlapping one.

The pipeline was rerun from cache after the changes: 118 s, 13 tests passed,
`results.json` regenerated, no metric moved.

---

## Session 9 - reproducibility

Two full runs:

- `python src/run_all.py`: download, clean from the workbook, backtest, decision,
  figures, tests. About 240 s on the first build.
- `python src/run_all.py --from-cache`: reuses the raw parquet. About 118–150 s.

The second run reproduced every accuracy, calibration and decision number exactly.
`results.json` records the commit, the seed (`20261024`), package versions and stage
timings. LightGBM is fitted with `deterministic=True`.

---

## Session 10 - the second audit: three leaks of a subtler kind

An independent read-only audit of the finished project found four defects. None of
them changed the model ranking; all of them changed what the code was entitled to
claim.

1. **Partial windows in the cumulative conformal grid.** The aggregate grid kept
   every calibration origin with `horizon ≤ L` and summed whatever horizons existed.
   A late origin therefore contributed a 1-week sum to the L=8 residual distribution.
   Fixed by requiring `t + L ≤ origin` and `n_weeks == L`; the grid's `n` now equals
   the marginal h=L grid's `n`, and a test enforces that identity. Usable origins are
   12 at h=1 and 11 / 9 / 5 at L = 2 / 4 / 8.
2. **Bootstrap index mismatch.** Per-SKU costs were indexed by the global `sku_idx`,
   which runs over the whole 3,995-SKU universe, while the bootstrap drew from 3,186
   model SKUs. `bincount` silently allocated empty slots. Costs are now indexed by a
   compact `factorize` code and the function asserts that its width matches the
   bootstrap index.
3. **Price look-ahead.** The cost vector used each SKU's full-sample median price,
   including weeks after the forecast origin. It now uses the median up to the origin.
   The old basis is retained only as a sensitivity: 23.82% → 23.77% versus SES and
   18.62% → 18.41% versus the cheapest point rule, so the leak was not the source of
   the headline.
4. **A figure label that named one model for a row that does not use one.** `fig4`'s
   bottom row said "LightGBM-Tweedie point", but at `Cu/Co = 1` the cheapest point
   rule is LightGBM median. The row is labelled "vs cheapest point rule in each cell".

The audit also forced a wording fix: LightGBM median is **not** the worst aggregate
model; its 40.05% WAPE is second-worst, behind seasonal naive's 71.09%. The earlier
sentence was a real overclaim, not a rounding difference.

After the fixes the full cached run took 146 s, the suite grew to **19 tests** (24 after
Session 12), and
the central saving moved from 23.88% to 23.82% (the pre-origin price and complete
windows), with the ranking unchanged.

---

## Session 11 - hardening audit: one real model bug and three traceability fixes

A third audit checked every headline claim against the code. Four findings:

1. **Croston/SBA interval bug.** At the first sale the code set `p = 1` but did not reset the
   periods-since-last-sale counter, so leading zero weeks were added to the first observed gap.
   The smoothed interval was too large and the forecast biased down. Fixed by resetting `q` at the
   first sale, with a regression test (leading zeros must give the same forecast as the same sales
   without them) and an all-zero fallback test. Croston h=1 MASE is now **1.692** and pooled WAPE
   **9.99%**; it remains worse than the historical mean and the ranking is unchanged.
2. **"168 of 168".** 168 is 7 models × 2 policies × 3 lead times × 4 ratios; the independent cells
   are 12. The docs now say that, and `results.json` stores `n_independent_cells` and
   `cells_better` (12).
3. **Customer-ID share.** 22.77% is the *line-item* share. The invoice share is 8,752 / 53,628 =
   **16.32%**; both are now stored and the docs use the right unit.
4. **Ex-post headline.** The 23.82% is the best of seven quantile models in the central cell.
   `headline_selection` records the 12-cell range (2.43–48.81%, median 17.50%), the chosen cell
   and `selected_ex_post: true`; a one-sided paired SKU bootstrap (10,000 resamples, drawn in
   chunks of 1,000) gives *p* < 0.0001 for H0: savings ≤ 0 at that cell, with no replicate
   reaching zero. The vacuous `same_obs_all_models` check was replaced by a real
   SKU-set comparison (24 origin × horizon cells, 3,186 SKUs, now tested).

The 43 MB `weekly_demand.csv` was removed from the git index and added to `.gitignore`; a fresh
clone must run the pipeline. After the changes the cached run took 119 s and the suite counted
**23 tests** (24 after Session 12).

---

## Session 12 - p-value resolution, provenance, CI coverage

A fourth review checked the regenerated artefacts. Three follow-ups:

1. **P-value on the resolution floor.** The 500-resample test reported *p* = 0.001996 =
   1/501, i.e. no replicate reached zero: an artefact of the resolution, not an exact
   probability. The focal test now draws **10,000** SKU resamples in chunks of 1,000 on an
   independent random stream; the CI keeps its 500 resamples, so the interval is unchanged.
   No replicate reached zero, so the docs report *p* < 0.0001 and `results.json` carries
   `p_value_at_resolution`, `p_value_upper_bound` (1/10,001) and the count of replicates ≤ 0.
2. **Provenance.** `meta.git_commit` named `HEAD` while the artefacts came from a dirty tree.
   `meta.git_dirty` now records that, and `meta.git_state` reads "working tree on top of
   <hash>"; a clean release run reads "clean tree at <hash>". The flag ignores generated
   artefacts under `outputs/` and `data/`, since the run rewrites them by design.
3. **CI coverage.** Push/PR still runs the cache-free suite. A manual `full-pipeline.yml`
   (`workflow_dispatch`) installs requirements, restores or downloads the UCI archive, runs
   `python src/run_all.py --skip-tests`, then the full suite.

After these changes the cached run took 102 s (meta total 97.5 s) and the suite counted
**24 tests**.

---

## What I would do differently

1. **Decide the comparator before the first result.** The cheapest point rule in each
   cell should have been in the decision table from day one, not added after an audit.
2. **Resample the unit of independence from the start.** The row bootstrap looked
   fine and was wrong.
3. **Model the zero atom explicitly.** A hurdle or censored model would fit the
   data-generating process better than a single Tweedie/quantile objective.
4. **Reserve more origins for time uncertainty.** Three adjacent origins cannot
   support a claim about regime change, and no amount of bootstrapping fixes that.
5. **Build the decision-side bulk trim at the same time as the accuracy-side one.**
   Reporting one and not the other invites the objection that the decision result
   is driven by the tail that was trimmed elsewhere.
6. **Test the window, not just the target.** "Realised by the origin" is not enough
   for a cumulative distribution: the *whole* L-week window has to be realised.
   That distinction survived nine sessions and two audits before it was caught.

---

## Honest summary

The project set out to test whether calibrated probabilistic forecasts reduce
inventory cost. They do - at high cost asymmetry, and by less against the cheapest
point rule than against the weakest defensible one. At low asymmetry they can lose.
The most accurate per-SKU point model is a value-destroying ordering rule, which is
the result worth remembering.
