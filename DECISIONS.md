# Decisions

A decision log, not a narrative. Each entry states the context, the choice, the alternatives that
were rejected, and the consequence. Entries are numbered for reference from `METHOD.md` and
`LIMITATIONS.md`. "Assumption" means a value chosen without data support and swept where possible.

---

## D1 — Use UCI Online Retail II, verified by hash, with a local manifest

**Context.** The charter fixes the dataset: two yearly sheets of a UK online retailer,
2009-12-01 → 2011-12-09.

**Decision.** Download the published zip from
`https://archive.ics.uci.edu/static/public/502/online+retail+ii.zip`, verify the zip sha256
(`572e3627…e67bfb`) and the extracted workbook sha256 (`bcbe73b3…f2e980`), and write
`data/raw/manifest.json` with retrieval time, byte size, sheet names and row counts. If the
workbook is already present and matches, reuse it and refresh the manifest without re-reading
1.07 M rows.

**Alternatives.** A CSV mirror (no canonical hash); pandas `read_excel` directly from the URL
(no integrity check, slow, not reproducible offline); committing the 45 MB workbook (unnecessary
repository weight).

**Consequence.** A changed upstream file fails loudly. The workbook is git-ignored; the manifest is
committed. A fresh clone needs network access or a manually placed workbook.

---

## D2 — De-duplicate the sheet overlap before any aggregation

**Context.** The sheets overlap on 2010-12-01 → 2010-12-09.

**Decision.** Drop exact duplicate rows across the concatenated sheets **before** cleaning or
aggregating. 34,335 rows are removed. The audit records the units (+198,713) and net revenue
(+431,716.87) removed.

**Alternatives.** Keep the later sheet and drop the earlier one for the overlap (loses rows that
appear only once); de-duplicate on `(Invoice, StockCode, InvoiceDate)` (may merge legitimate repeat
lines); do nothing (double-counts nine days).

**Consequence.** The overlap is counted once. If a genuinely repeated identical line existed on the
same timestamp with identical fields, it would also be removed; given the identical timestamps and
descriptions, duplication is the overwhelmingly likely explanation. The audit makes the assumption
visible.

---

## D3 — Cancellations are separated, never netted

**Context.** `Invoice` values starting with `C` are returns. 19,104 rows carry −476,819 units and
−1,462,050.61 revenue.

**Decision.** Exclude cancellations from demand and report them separately in
`data.cancellations` and `DATA.md`. Do not subtract them from the week of the original sale, and do
not create negative demand.

**Alternatives.** Net returns in the same week (a returned unit becomes negative demand and can
make a SKU-week negative); net against the original invoice (requires a reliable link that the
data does not provide); ignore them silently (hides 1.85% of rows and a material revenue stream).

**Consequence.** Demand is **positive units shipped**, not net demand. Return-heavy SKUs are
overstated. This is a stated limitation, not a data error, and the audit quantifies the excluded
volume. A net-demand variant is cheap to add because the cancellation block is already isolated.

---

## D4 — Drop non-product codes with an explicit list and a strict regex

**Context.** Postage, manual adjustments, samples, gifts and bank charges appear as stock codes and
would otherwise become "SKUs".

**Decision.** Drop an explicit list (`POST`, `D`, `M`, `C2`, `DOT`, `S`, `B`, `BANK CHARGES`,
`PAD`, `PADS`, `CRUK`, `AMAZONFEE`, `ADJUST`, `ADJUST2`, `SP1002`), prefixes (`TEST`, `GIFT`,
`DCGS`, `BANK`), any code not matching `^[0-9]{4,6}[A-Z]{0,2}$`, and rows with a blank description.
9,002 rows (674,892.16 revenue) are removed.

**Alternatives.** A description keyword filter only (misses codes with plausible descriptions);
keep everything and let the ≥8-positive-week filter remove them (they can survive and pollute the
panel); a hand-maintained exclusion list only (brittle to new codes).

**Consequence.** The product universe is 4,675 codes after cleaning. The regex is deliberately
strict; a legitimate product code outside the pattern would be dropped, which is why the step is
logged with rows, units and revenue rather than applied silently.

---

## D5 — Keep only complete ISO weeks

**Context.** The data starts on Tuesday 2009-12-01 and ends on Friday 2011-12-09. The first and last
calendar weeks are partial.

**Decision.** Aggregate to ISO weeks (Monday–Sunday) and drop the week starting **2009-11-30** and
the week starting **2011-12-05**. Retain **104 complete weeks**, 2009-12-07 → 2011-11-28. Log this
as the seventh audit step (31,146 rows, 389,633 units, 744,489.76 revenue).

**Alternatives.** Keep partial weeks and flag them (they bias any weekly total and any lag/rolling
feature at the boundary); scale them up (invents demand); start at the first complete week but keep
the last partial (asymmetric and still biased).

**Consequence.** The panel is balanced: 3,995 SKUs × 104 weeks. The seventh audit step was added
after the first version of the audit failed to reconcile by exactly the partial-week rows; the
`test_cleaning_audit_adds_up` test now walks the full chain.

---

## D6 — Two nested samples: ≥8 positive weeks for the panel, ≥12 pre-origin for modelling

**Context.** The charter asks to test N = 8, 12, 26. A single threshold cannot serve both "a useful
shipped panel" and "enough history to fit a model".

**Decision.**
- **Universe** in `data/processed/weekly_demand.csv`: every SKU with **≥8 positive weeks** in the
  full 104-week sample → **3,995 SKUs**. This excludes one-off codes but keeps the panel reusable.
- **Model sample**: every SKU with **≥12 positive weeks in weeks 0–76 only** (ending 2011-05-23,
  before the first origin) → **3,186 SKUs**, frozen in `data/processed/sku_selection.json`.
  `features.load_panel()` recomputes the count and asserts it matches the JSON.

**Alternatives.** Select on the full sample (survivor bias: a SKU that only sells after the origin
would be included); use ≥26 weeks (2,286 pre-origin SKUs, loses too much of the cold-start
population); use ≥8 for modelling (more cold SKUs, but too little history for per-SKU baselines and
conformal classes).

**Consequence.** 296 SKUs with 8–11 pre-origin weeks are in the panel but not in the model sample;
no metric describes them. Cold-start analysis covers the 12–25-week group (880 SKUs at the first
origin). The threshold trade-off is explicit, not hidden in an average.

---

## D7 — Three rolling origins inside the last 26 weeks, not a random split

**Context.** The charter requires a rolling-origin backtest, at least three origins, and no random
split.

**Decision.** Origins at panel indices **77, 85, 93** = **2011-05-30**, **2011-07-25**,
**2011-09-19**. The test window is the last 26 weeks (indices 78–103). Every model is fitted
separately at each origin on weeks ≤ `origin − 13`; the 13 weeks before the origin are the
conformal calibration window. No target after the origin is used in fitting or calibration.

**Alternatives.** A single 26-week holdout (no rolling evidence, one path); random K-fold (leaks
future into past and violates the charter); weekly origins (more compute, highly correlated
overlapping targets); origins at 78/91/104 (the charter's illustrative numbers do not leave a full
h=8 target window for the last origin).

**Consequence.** Each origin contributes 3,186 SKU forecasts per horizon → 9,558 observations per
horizon. The origins are adjacent 8-week blocks and are **not independent**; the bootstrap is over
SKUs, not time. This is stated in LIMITATIONS §2.1.

---

## D8 — Forecast h = 1…8; report accuracy at 1, 2, 4; decide at L = 2, 4, 8

**Context.** The charter asks for accuracy at 1/2/4 weeks and a decision sweep with lead times
2/4/8.

**Decision.** Generate forecasts for every h = 1…8. Report MAE/RMSE/MASE/pinball/coverage at
h = 1, 2, 4. Run the newsvendor at cumulative L = 2, 4, 8, using a separate conformal grid for
each L. The central decision case is L = 4, `Cu/Co = 9`.

**Alternatives.** Only h = 1, 2, 4 and approximate L = 8 by summing marginal quantiles (ignores
within-window dependence and would understate the upper tail); only L = 2, 4 (drops the charter's
most asymmetric cell); a single horizon for both accuracy and decisions (loses the distinction
between forecast skill and decision value).

**Consequence.** The L = 8 result (the largest saving) relies on h = 5–8, which are not in the
accuracy tables. That asymmetry is flagged in LIMITATIONS §2.6; the central L = 4 claim does not
depend on it.

---

## D9 — Four simple baselines, each with a stated configuration

**Context.** The charter names historical mean, seasonal naive, SES and Croston/SBA.

**Decision.**
- Mean: `mean(y_0…y_O)`, constant across horizons.
- Seasonal naive: `y_{O+h−52}` if the index exists, else the historical mean.
- SES: per-SKU α ∈ {0.05, 0.1, 0.2, 0.3, 0.5}, selected by in-sample SSE, one final pass at the
  chosen α.
- Croston/SBA: α = 0.1, Syntetos–Boylan factor `(1 − α/2)`, demand size and interval smoothed only
  when a sale occurs, fallback = historical mean.

**Alternatives.** A per-SKU MLE/optimised SES (slower, another overfitting channel on 65–93
observations); a single global α (ignores heterogeneity); Croston without SBA (known positive
bias); Theta/ETS (not in the charter, and a larger baseline family would need its own tuning).

**Consequence.** SES is deliberately a small grid, not an MLE fit; the configuration is stated so
the comparison is reproducible. Croston/SBA is worse than the historical mean on this data
(MASE 1.692 vs 1.319), which is a legitimate finding about a retailer without a stable
inter-arrival rate, not a bug. The first sale resets the interval clock, so leading zero weeks
cannot inflate the first smoothed interval (regression-tested).

---

## D10 — Poisson GLM with log1p count features, not a negative-binomial GLM

**Context.** The charter suggests "negative binomial GLM … or a Poisson GLM with overdispersion".
scikit-learn has no negative-binomial GLM; statsmodels would add a dependency and a much slower
per-origin fit.

**Decision.** `PoissonRegressor(alpha=0.5, max_iter=400, tol=1e-5)` on standardised features, with
all count-like features (`lags`, rolling means/std, prices) transformed by `log1p`. The L2 penalty
is chosen a priori, not tuned on the test window.

**Alternatives.** A statsmodels NB2 GLM (slow, and the single global α is the problem below); a
quasi-Poisson with estimated dispersion (still a mean model; the conformal layer supplies the
distribution); raw counts in the linear predictor (four orders of magnitude dominate the
coefficient scale).

**Consequence.** The GLM is a mean model; uncertainty comes from the shared conformal layer, not
from the Poisson variance. It is biased low at h=1 (mean forecast 20.09 vs actual 27.16) and has a
pooled WAPE of 26.01%, so it is not a candidate for the decision headline.

**Why not NB2.** An earlier version used a negative-binomial variance function
`Var = μ + αμ²` with a single global α estimated from residuals. The estimate was dominated by
the bulk tail; the implied scale collapsed the P50 toward zero and destroyed coverage. The
variance-stabilising `sqrt(μ + 0.25)` scale (D12) replaced it. A per-class or per-SKU α would be a
different, larger model and was not attempted.

---

## D11 — Two LightGBM objectives: Tweedie for the mean, quantile-0.5 for the median

**Context.** The charter suggests "LightGBM quantile regression (P10/P50/P90) or LightGBM with
Tweedie/Poisson objective, plus conformal calibration".

**Decision.** Fit two global models on identical rows and features:
- `lgbm_tweedie`: Tweedie objective, variance power 1.3 (between Poisson at 1 and gamma at 2).
- `lgbm_quantile`: quantile objective at α = 0.5. Its prediction is a **median**, not a mean.

Both use 400 estimators, learning rate 0.04, 31 leaves, `min_child_samples=50`, subsample 0.8
(frequency 1), colsample 0.8, L2 1.0, `random_state=20261024`, `deterministic=True`,
`force_col_wise=True`. No SKU identity is used as a raw categorical.

**Alternatives.** Three separate quantile LightGBM models at 0.1/0.5/0.9 (triple the training
cost, quantile crossing is possible, and the conformal layer already supplies calibrated
quantiles); a Poisson objective (Tweedie 1.3 handles the overdispersion better in practice); a
single model with SKU embeddings (the charter calls it optional; no evidence it earns its keep at
this history length).

**Consequence.** The median model has the best MASE (0.686) and the second-worst aggregate WAPE
(40.05%; only seasonal naive is worse at 71.09%). That tension is the point of the project, not an
accident. The Tweedie mean has a worse MASE (0.961) but tracks volume better (27.10% WAPE) and is
the stronger decision model among the two (22.31% vs 17.58% at the central case). Neither is
presented as universally better.

---

## D12 — Variance-stabilising scale `sqrt(μ + 0.25)` plus Mondrian conformal

**Context.** All models need P10/P50/P90 and a cumulative distribution for the newsvendor. The
models have different scales and different biases, so a model-specific parametric distribution
would confound the comparison.

**Decision.** One shared, model-agnostic layer:
1. Standardise residuals as `z = (y − μ) / sqrt(μ + 0.25)`. The `+0.25` keeps the scale finite and
   non-degenerate at μ = 0.
2. On the 13-week calibration window, take empirical quantiles of `z` on a 199-point grid
   (`q = 0.005 … 0.995`), **separately for each demand class** (Mondrian conformal). If a class has
   fewer than 30 rows at a horizon, fall back to the pooled grid.
3. Predictive quantiles are `μ + sqrt(μ + 0.25) · Q_z(q)`, clipped at zero.

The same layer is applied to every model, so the accuracy and decision comparisons isolate the
quality of the conditional mean/median.

**Alternatives.**
- NB2 variance `μ + αμ²` with global α: rejected — see D10; collapsed the median and destroyed
  coverage.
- A parametric Poisson/NB predictive distribution: would impose a variance-mean relationship that
  the data rejects (overdispersion varies by class).
- Residual bootstrap: slower, and the empirical quantile grid is effectively a smoothed bootstrap
  on 199 points.
- Direct LightGBM quantile models at 0.1/0.5/0.9: no finite-sample coverage guarantee, possible
  crossing, and no easy cumulative-L distribution.
- Pooled conformal without the class split: marginal coverage hides systematic under-coverage in
  the intermittent class.

**Consequence.** Coverage at h=1 is 0.778–0.806 for an 80% target; strict `P(y < q10)` is
0.084–0.135 and `P(y ≤ q90)` is 0.888–0.914. The zero atom makes `P(y ≤ q10)` look inflated
(0.521–0.581) when `q10 = 0`; the meaningful lower-tail check is strict `<`. This is documented in
README §4. The layer is marginal within class, not SKU-conditional (LIMITATIONS §2.5).

---

## D13 — 13-week calibration window, realised targets only

**Context.** The conformal window must be long enough to estimate 199 quantiles per class/horizon
but short enough to stay local and avoid using post-origin information.

**Decision.** `CALIB_WEEKS = 13`; candidate calibration origins are `origin − 12 … origin`; only
rows with `target_idx ≤ origin` are kept, so every calibration target is realised by the forecast
origin. For h=1 that leaves 12 usable origins, because the origin itself has no realised next-week
target; the cumulative L-week grids use 11 / 9 / 5 origins at L = 2 / 4 / 8. The model itself is
fitted on weeks ≤ `origin − 13`.

**Alternatives.** 8 weeks (too few rows for the tail and for small classes); 26 weeks (more stable,
but spans a longer regime and overlaps more of the test window); a fixed expanding window (not
local); using targets after the origin (leakage).

**Consequence.** Each origin has 216,648 calibration rows before class/horizon splitting. The
window is short, so tail quantiles are noisy; the class fallback at <30 rows prevents empty grids
but can make a small class borrow the pooled shape. ACI or a longer window is the next step
(LIMITATIONS §2.5).

---

## D14 — Separate conformal grid for cumulative L-week demand

**Context.** The newsvendor needs `F_L`, the distribution of `Σ_{h=1..L} y_h`, not the sum of L
marginal quantiles (which assumes perfect dependence and ignores the correlation of residuals
within the window).

**Decision.** For each calibration origin, SKU and L, compute `y_L = Σ y_h` and `μ_L = Σ μ_h`, then
`z_L = (y_L − μ_L) / sqrt(μ_L + 0.25)` and empirical quantiles per demand class on the same
199-point grid. The order is `μ_L + sqrt(μ_L + 0.25) · Q_{z_L}(CR)`. Grids are stored with
`kind = "aggregate"`. A calibration origin enters only when its **whole** L-week window is realised
by the forecast origin (`t + L ≤ origin`); partial windows are excluded rather than summed over
fewer than L weeks. The test suite asserts that each aggregate grid has the same `n` as the
corresponding marginal h=L grid.

**Alternatives.** Sum marginal quantiles (wrong dependence); simulate paths from a copula (more
assumptions, no clear gain at this sample size); a Dirichlet/compound model (does not fit the
zero-inflated marginals); a single grid pooled over L (ignores that the L-week sum is much less
zero-inflated than a single week).

**Consequence.** The cumulative grid captures the empirical within-window correlation, but it is
still a **single-period** order: no carry-over, no intermediate replenishment. That is the main
decision-model limitation (LIMITATIONS §3.1).

---

## D15 — Cost model: 1% of unit price per week, ratios {1,3,9,19}, lead times {2,4,8}

**Context.** The data has no costs. The charter requires a newsvendor with underage/overage costs
and an explicit sweep.

**Decision.**
```
Co = 0.01 × unit_price × L
Cu = (Cu/Co) × Co,  Cu/Co ∈ {1, 3, 9, 19}
CR = Cu / (Cu + Co) ∈ {0.5, 0.75, 0.9, 0.95}
```
`unit_price` is the SKU's median observed price **up to each forecast origin**, winsorised at
p99.5; the full-sample median is used only for the price-basis sensitivity in `results.json`. It
is also the code's fallback for a SKU with no positive pre-origin revenue, a case excluded by the
model-sample rule (≥12 positive pre-origin weeks), and `run_decision` asserts that it never binds
on an evaluated SKU. All values live in `config.py` and the derived assumptions are written to
`results.json.decision.assumptions`.

**Alternatives.** A single cost ratio (hides the dependence of the conclusion on asymmetry);
revenue-based margin (not observed); a fixed currency holding cost (ignores that a £1 item and a
£50 item do not cost the same to hold); estimating `Co` from the data (impossible).

**Consequence.** Absolute costs scale with price and L and are not a P&L claim. For the SES
quantile policy, at `Cu/Co = 1` the critical ratio is 0.5 and the saving is at most 4.23%; the
grid's maximum at ratio 1 is 13.38% (LightGBM-median quantile, L=4), but that model's quantile rule
falls to +0.54% at ratio 3. The sweep, not one cell, is the evidence. The central case (L=4,
ratio 9) is a reporting convention, not an estimate of the retailer's true costs.

---

## D16 - Savings versus the best simple point rule and the cheapest point rule in each cell

**Context.** "Savings versus what?" determines the whole headline.

**Decision.** For each L and cost ratio, compare every model/policy with the **best point rule
among the four simple baselines** (mean, seasonal naive, SES, Croston/SBA) on the same realised
demand. Report the saving and a 500-rep bootstrap CI over SKUs. There are 168 model-policy-L-ratio
rows but only **12 independent L × ratio cells** (7 models × 2 policies = 14 rows per cell); the
best baseline is **SES point** in all 168 rows, and the best quantile policy beats it in all 12
cells (`headline.decision.baseline_comparison`). The oracle orders realised demand and has zero
cost by construction; it is reported as a bound, not a policy.

**Alternatives.** A fixed baseline (e.g. historical mean) — makes the result depend on an arbitrary
weak comparator; the company's actual policy — not in the data; the best *probabilistic* baseline —
would conflate the value of the distribution with the value of the model.

**Consequence.** Choosing the best baseline on the evaluation cells makes the comparison
conservative (the baseline is as strong as it can be on this sample), but it uses evaluation
information to select the comparator. It is not a pre-registered incumbent. The number to quote is
"versus SES point on these cells", not "versus the company's current policy". The oracle is
labelled clairvoyant and its zero cost is not an achievable result. A second, stronger comparator
is also recorded: the **cheapest point rule in each cell**. At the central case that is LightGBM
Tweedie point, and the SES quantile saves **18.62%** against it
(`headline.decision.central_case_quantile.savings_vs_best_point_all_pct`). The identity of that
rule changes with the cell — LightGBM median at `Cu/Co = 1`, LightGBM Tweedie above it — which is
why the docs call it "the cheapest point rule in the cell", not "the strongest model".

The 23.82% headline is itself the best of the seven quantile models in the central cell, selected
after seeing the table (`headline.decision.headline_selection.selected_ex_post = true`). Across the
12 cells the best quantile model saves 2.43–48.81% (median 17.50%), so the central value is not a
pre-registered point estimate; the bootstrap CI is conditional on the selected cell.

---

## D17 — Bootstrap over SKUs, 500 reps, fixed seed

**Context.** The saving is a mean over heterogeneous SKUs; a CI is needed, but the origins are
correlated and there are only three of them.

**Decision.** Resample SKUs with replacement 500 times (`numpy.default_rng(20261024)`), recompute
the difference in mean cost, and report the 2.5/97.5 percentiles. The same index matrix is used
for every model and cell within an L, so the comparison is paired. Per-SKU cost vectors are
indexed by a compact `factorize` code (0…n−1); the function asserts that its width equals the
bootstrap index width, because the global `sku_idx` runs over the whole universe and would
silently leave empty slots or draw the wrong SKUs.

**Alternatives.** A bootstrap over origins (only three, too few); a block bootstrap over time (the
test window is short and the origins overlap); a parametric CI on the mean (ignores the heavy
tail); no CI (hides sampling uncertainty).

**Consequence.** The CI describes cross-sectional heterogeneity only. It does not cover temporal
uncertainty, model specification, or cost assumptions; those are addressed by the sweep (costs) or
left as limitations (time/model). A CI that excludes zero is not evidence of future savings
(LIMITATIONS §3.3). A separate, higher-resolution bootstrap (10,000 SKU resamples, drawn in
chunks of 1,000) gives a one-sided paired test of H0: savings ≤ 0 at the selected central cell:
*p* < 0.0001, with no replicate reaching zero (3,186 SKUs; `p_value_at_resolution: true`,
bound 1/10,001). It is a cost-bootstrap test, not a Diebold–Mariano test on forecast errors.

---

## D18 — Bulk sensitivity on the accuracy side only

**Context.** Bulk orders are 20.46% of units from the top 1% of invoices and dominate absolute
errors. The charter asks whether excluding them changes conclusions.

**Decision.** For accuracy, exclude an evaluation week if its demand exceeds the SKU's **training**
p99.5 at that origin (floor 1 unit), then recompute MASE for every model with the same MASE scale
exclusion (`scale ≥ 0.01`). Report the excluded observation share (1.34%) and demand share
(14.86%). **Do not** re-estimate the decision table on a trimmed panel in this version.

**Alternatives.** Trim the decision side too (would remove whole L-week SKU-decisions and change
the cost base; a like-for-like saving is not available without a separate run); use a fixed
absolute threshold (not comparable across SKUs); winsorise instead of exclude (changes the target,
not just the evaluation).

**Consequence.** The accuracy ranking is unchanged under trimming (median 0.474, SES 0.624,
Tweedie 0.773, GLM 1.023, mean 1.113, Croston 1.493, seasonal naive 1.535). The decision magnitude
is not proven robust to trimming; that is an explicit limitation, not a hidden result. A
decision-side rerun is the first item of next work.

---

## D19 — Cold-start split at <26 vs ≥26 positive weeks before each origin

**Context.** The charter asks for cold-start performance and says not to hide it in the average.
The model sample requires ≥12 positive weeks, so the cold bucket cannot include 0–11.

**Decision.** At each origin, split SKUs by the number of positive weeks in their history:
`<26` vs `≥26`. Report h=1 MASE pooled over origins for each bucket, plus a per-origin table
(`outputs/tables/cold_start.csv`). At the first origin, 880 of 3,186 SKUs are cold (12–25 weeks).

**Alternatives.** A fixed calendar age (new SKUs cannot be identified reliably before their first
sale); a <52-week split (almost everyone is cold at the first origin); no split (the charter
explicitly forbids hiding them).

**Consequence.** Cold-start SKUs are reported separately, not averaged away. LightGBM median is
strongest there (MASE 0.558) and Croston is worst (3.689); SES is essentially unchanged from the
warm group (0.818 vs 0.819). The decision cost of cold-start SKUs is **not** measured separately;
that is a limitation.

---

## D20 — Exclude near-deterministic SKUs from MASE, identically everywhere

**Context.** MASE divides by the SKU's in-sample naive MAE. For a deterministic or almost constant
series the denominator approaches zero and the ratio explodes.

**Decision.** Exclude observations with `scale < 0.01` from MASE and MASE(P50), report
`n_mase_excluded`, and apply the **same** exclusion in the bulk sensitivity. MAE, RMSE, pinball and
coverage still use all observations.

**Alternatives.** Add a floor to the denominator (arbitrary and still unstable); winsorise the
ratio (hides which observations are affected); drop MASE (loses the scale-free comparison the
charter requires).

**Consequence.** Two h=1 observations are excluded. The earlier bulk table did not apply the same
exclusion, which produced MASE values around 1e3–1e4; that was an artefact and was fixed. The
accuracy and bulk tables are now comparable.

---

## D21 — Figure choices: representative SKUs and a clipped cost axis

**Context.** The charter requires at least five figures, including forecast-vs-actual for
fast/medium/slow SKUs, calibration, cost vs cost ratio, and error by demand class.

**Decision.**
- `fig1`: intermittency and invoice concentration.
- `fig2`: rolling-origin h=1…8 forecast paths for three SKUs selected by **median test-window
  demand** within the fast/medium/slow classes, not by the largest spike.
- `fig3`: calibration curves (predicted vs observed).
- `fig4`: cost saving vs `Cu/Co`; the y-axis is clipped at **−65%** with an off-scale annotation
  for seasonal naive (which reaches −73% to −136%).
- `fig5`: MASE by demand class.
- `fig6`: service–inventory frontier.
- `fig7`: all **20** worst SKUs with demand histories, not just the top few.

**Alternatives.** Pick the most extreme SKUs (dramatic but unrepresentative); plot all models on
an unclipped axis (seasonal naive compresses every other line); show only 5 worst SKUs (the
charter asks for 20).

**Consequence.** `fig2` and `fig7` use test-window information for **display** selection; no
reported metric depends on them. The clipping is annotated so the seasonal-naive failure is not
hidden. Figures are descriptive, not inferential.

---

## D22 — Tests assert invariants and the contract, not the headline numbers

**Context.** A test suite that hard-codes today's MASE would fail on every legitimate model change
and would not catch the failures that matter (leakage, broken spine, bad accounting).

**Decision.** 24 tests:
1. no negative demand and a plausible zero share;
2. complete 104-week calendar spine, one row per SKU-week;
3. cleaning audit reconciles row by row;
4. model selection is frozen before the first origin and matches the recomputed count;
5. no feature changes when **every** week after the origin is perturbed;
6. training targets stay inside the fit window;
7. quantiles are monotone and non-negative;
8. h=1 coverage and lower/upper-tail frequencies are in a broad ballpark;
9. newsvendor accounting identities (`total = holding + stockout`, fill/service in [0,1]);
10. the decision sweep covers the charter grid;
11. `results.json` has the contract keys, 7 models, 171 decision rows, 3,186 SKUs;
12. the headline best-MASE model matches the table and the central case is L=4, ratio 9;
13. docs, `requirements.txt` and ≥5 non-empty figures exist;
14. every aggregate conformal grid has the same `n` as the marginal h=L grid (complete windows);
15. the bootstrap rejects a non-compact SKU index and is relabel-invariant;
16. `price_vector(origin)` is unchanged when every post-origin week is perturbed;
17. the headline numbers match `decision.csv` and `results.json` (central and best cells);
18. the price-basis sensitivity is stored and does not reverse the central result;
19. coverage is below target at h=1 and above it at h=4, as reported;
20. Croston/SBA ignores leading zero weeks in the first interval and matches the same sales
    without those zeros;
21. an all-zero Croston series returns the mean fallback;
22. the SKU-set quality gate detects a model that misses a SKU in one origin × horizon cell;
23. the headline selection block is traceable to the 12-cell grid (168 rows, paired-test fields);
24. the focal p-value uses ≥10,000 resamples and is consistent with its resolution flag, and the
    chunked test bootstrap matches a one-shot count of replicates.

**Alternatives.** A golden-master regression on every number (brittle, and would freeze bugs); no
tests (fails the charter); only smoke tests (would not catch leakage).

**Consequence.** The suite catches structural breakage, not small metric drift. The README numbers
are checked against `results.json` manually at release. The leakage test is stronger than the
charter's "assert max feature date < forecast origin": it perturbs all post-origin weeks and
compares every feature, with `rtol=1e-9, atol=1e-9` to tolerate floating-point summation order.

---

## D23 — One orchestrator, `results.json` as the single source of truth

**Context.** The charter requires one command to rebuild everything and every README number to be
pipeline-generated.

**Decision.** `src/run_all.py` runs each stage in a separate process (`download`, `clean`,
`backtest`, `decision`, `figures`), streams logs, then assembles `outputs/results.json` from the
stage artefacts and runs the tests. Flags: `--from-cache` (reuse the transaction parquet),
`--force-download` (ignore the cached zip and fetch it again), `--skip-tests`. The default is a
full clean from the workbook. The file records `meta.versions`, `meta.stage_seconds`,
`meta.git_commit`, `meta.git_dirty` and a human-readable `meta.git_state` (when git is available),
the data audit, accuracy, calibration, cold start, decision rows, headline fields, figure inventory
and quality checks.

**Alternatives.** A notebook (not one command, hidden state); a Makefile (less portable on
Windows); separate scripts with no assembly (docs drift from artefacts); committing a hand-written
results file (violates the traceability gate).

**Consequence.** The docs quote fields of `results.json`, not numbers typed from a console. A
failed test makes the build exit non-zero. `--from-cache` is for iteration; the release run uses
the default so the audit is rebuilt from the verified workbook. `meta.git_commit` is `HEAD` at
run time and `meta.git_dirty` says whether the working tree had uncommitted changes. When it did,
`meta.git_state` reads `working tree on top of <hash>`: the artefacts were produced by code that
is not fully contained in `git_commit`, and the field says so instead of implying otherwise. A
release run should be made from a clean tree, where `git_state` reads `clean tree at <hash>`.

---

## D24 — Deliberately not built

These are scope decisions, not oversights:

- hierarchical reconciliation from SKU to category/total (the aggregate bias of the median model is
  reported instead);
- promotions, price elasticity, competitor, weather, holidays beyond month/week-of-year;
- stockout/latent-demand modelling (no inventory data);
- supplier lead-time uncertainty (L is fixed and known);
- multi-echelon inventory, MOQ, shelf life, capacity, quantity discounts, salvage;
- SKU embeddings or raw SKU categorical in the trees;
- a decision-side bulk-order rerun;
- a dynamic multi-period inventory simulation;
- CRPS and formal paired forecast tests;
- drift monitoring, retraining schedule, model registry, deployment.

Each appears in LIMITATIONS.md with its consequence and the cheapest credible fix.

---

## D25 — Resolved blockers and open items

**Resolved during the build.**
- **Audit did not reconcile.** The first six filters left 1,003,214 rows, but the panel started at
  1,003,214 minus the partial weeks. The partial-week drop was not an audit step. Added
  `partial_calendar_weeks` as step 7; `test_cleaning_audit_adds_up` now walks the full chain.
- **MASE explosion in the bulk table.** The bulk sensitivity did not apply the `scale ≥ 0.01`
  exclusion used by `accuracy.csv`, producing values around 1e3–1e4. Fixed; the bulk table now
  matches the main table's scale exclusion.
- **Leakage test too strict.** Comparing perturbed features with exact equality failed on
  floating-point summation order (differences ~1e-13). Relaxed to `rtol=1e-9, atol=1e-9`; a real
  leak moves features by orders of magnitude.
- **`origin_week` missing from `forecasts.parquet`.** The test derived it from the parquet; fixed
  by deriving it from the panel (`p.weeks[fc["origin"]]`).
- **Headline float mismatch.** `results.json` stores table records via 15-digit JSON; the test now
  compares with `1e-9` rather than exact equality.
- **Zero-atom confusion.** `P(y ≤ q10)` is 0.52–0.58 because 47% of demands are zero and `q10` is
  often zero. The correct lower-tail check is `P(y < q10)`; both are reported and explained.
- **NB2 scale.** Rejected in favour of `sqrt(μ + 0.25)`; see D10/D12.
- **Parquet dtype.** `StockCode` must be read as `str` (leading zeros) and `Description` kept as
  NaN before caching; otherwise the panel loses codes or crashes on mixed types.
- **Pandas constructor.** `pd.DataFrame(list_of_dfs)` fails on this pandas build; use
  `pd.concat(..., ignore_index=True)`.

**Found by the external audit of the finished project (all fixed).**
- **Partial cumulative windows.** The aggregate conformal grid summed `horizon ≤ L` over every
  calibration origin, so late origins contributed 1- or 2-week sums to an L-week distribution.
  Only origins with `t + L ≤ origin` now enter, and a test checks `n` against the marginal h=L grid.
- **Bootstrap index mismatch.** Per-SKU cost vectors were indexed by the global `sku_idx` (up to
  ~4,000) while the bootstrap index had 3,186 columns, so `bincount` allocated empty slots. Costs
  are now indexed by a compact factorized code with an assertion on the width.
- **Price look-ahead.** The cost vector used the full-sample median price, including weeks after
  the forecast origin. It now uses the pre-origin median; the full-sample version is retained only
  as a sensitivity (23.82 → 23.77% vs SES).
- **Figure label.** `fig4`'s bottom row named LightGBM-Tweedie, but at `Cu/Co = 1` the cheapest
  point rule is LightGBM median. The row is labelled "vs cheapest point rule in each cell".

**Open.**
- `meta.git_commit` is `null` and `meta.git_state` says "not a git checkout" in runs made before
  the repository is initialised. The release run records the commit once git exists, and should be
  made from a clean tree so `meta.git_dirty` is false.
- The decision side is not trimmed for bulk orders; documented, not fixed.
- No CRPS; the paired test is a SKU bootstrap on newsvendor costs at the selected cell, not a
  Diebold–Mariano test on forecast errors.
- The repository is Windows-tested only; `requirements.txt` pins versions but not the OS/BLAS.

---

## Decision summary

| # | Decision | Main alternative rejected | Why |
|---:|---|---|---|
| D1 | UCI workbook, sha256 + manifest | CSV mirror | Integrity and reproducibility |
| D2 | De-duplicate sheet overlap first | Keep one sheet | Loses unique rows |
| D3 | Cancellations separated, not netted | Net returns | Avoids negative demand |
| D4 | Explicit non-product list + regex | Description filter only | Misses plausible codes |
| D5 | 104 complete ISO weeks | Keep partial weeks | Boundary bias |
| D6 | ≥8 panel, ≥12 pre-origin model | ≥26 model sample | Preserves cold start |
| D7 | Origins 77/85/93 | Random split | Charter, no leakage |
| D8 | h=1…8; report 1/2/4; decide 2/4/8 | Sum marginal quantiles | Within-window dependence |
| D9 | Four fixed baselines | Tuned per-SKU models | Reproducibility, overfitting |
| D10 | Poisson GLM, log1p counts | NB2 GLM | NB2 α collapsed coverage |
| D11 | Tweedie + median LightGBM | Three quantile models | Cost, crossing, shared layer |
| D12 | `sqrt(μ+0.25)` + Mondrian conformal | NB2 / direct quantiles | Coverage, model-agnostic |
| D13 | 13-week realised-only window | 26 weeks / expanding | Locality, no leakage |
| D14 | Separate cumulative-L grid | Sum marginal quantiles | Dependence |
| D15 | 1%/week holding, ratios 1–19 | Single cost ratio | Honest sweep |
| D16 | Save vs best simple point rule; also vs cheapest point rule in each cell | Fixed weak baseline | Conservative comparators, both reported |
| D17 | 500-rep SKU bootstrap | Origin bootstrap | Only 3 origins |
| D18 | Bulk trim accuracy only | Trim decisions too | Not like-for-like |
| D19 | Cold split <26 vs ≥26 | No split | Charter |
| D20 | MASE scale ≥0.01, everywhere | Floor the denominator | Transparency |
| D21 | Representative SKUs, clipped axis | Extreme SKUs | Honest display |
| D22 | Invariant tests | Golden master | Catches structural bugs |
| D23 | One orchestrator + `results.json` | Notebook / Makefile | One command, traceability |
| D24 | Explicit non-goals | Silent omission | Boundary of the claim |
| D25 | Log resolved blockers | Hide them | Audit trail |
