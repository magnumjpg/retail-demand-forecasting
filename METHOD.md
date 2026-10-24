# Method

This document describes exactly what is estimated, how it is evaluated, and which choices are
assumptions. The code is the final authority: `src/features.py`, `src/models.py`,
`src/backtest.py`, `src/decision.py`.

## 1. Target and forecast design

- **Unit of analysis:** SKU × ISO week.
- **Target:** units shipped in the week (sum of positive `Quantity`), never revenue.
- **Forecast origins:** 2011-05-30, 2011-07-25, 2011-09-19 (panel indices 77, 85, 93).
- **Horizons:** 1–8 weeks ahead for every origin. Accuracy is reported at h = 1, 2, 4 (charter);
  the decision analysis uses cumulative windows L = 2, 4, 8.
- **Training window:** weeks 0…`origin − 13`. The 13-week block immediately before the origin is
  reserved for conformal calibration. No target after the origin is used in fitting or
  calibration.
- **Evaluation:** last 26 weeks (indices 78–103). All three origins lie in this window. With
  h = 1, the origins contribute targets at weeks 78, 86 and 94; longer horizons fill the gaps.
  This is a rolling-origin backtest, not a single 26-week path.

Every model is fitted separately at each origin. There are no random splits and no shuffled
rows. The model sample is frozen on pre-origin data (DATA.md §4).

## 2. Baselines

All baselines are computed from the history `y_0 … y_O` available at the origin and produce one
forecast per horizon. They are re-computed at every calibration origin as well, so they get the
same conformal treatment.

| Model | Definition |
|---|---|
| Historical mean | `mean(y_0…y_O)` for every horizon |
| Seasonal naive | `y_{O+h−52}` if that index exists, otherwise the historical mean |
| SES | per-SKU simple exponential smoothing; α ∈ {0.05, 0.1, 0.2, 0.3, 0.5} selected by in-sample SSE, one final pass at the chosen α |
| Croston/SBA | Croston with α = 0.1 and the Syntetos–Boylan factor `(1 − α/2)`; demand size and inter-arrival interval are smoothed only when a sale occurs; fallback = historical mean |

SES is deliberately a small grid, not an MLE fit. A per-SKU optimiser would be slower and would
add another source of overfitting on 65–93 observations; the grid is stated so the comparison is
reproducible.

## 3. Machine-learning models

Both ML models are **global** (one model for all SKUs) and are trained on the same rows.

### 3.1 Training frame

For a fit end `F`, training origins run from week 16 to `F − 8` in steps of 2. For each origin and
each horizon 1…8, one row per SKU is created with features computed from weeks ≤ origin and target
`y` at `origin + h`. This produces 535,248 / 637,200 / 739,152 rows at the three origins
(`n_train_rows` in `backtest_origins.csv`). The assertion `target_idx ≤ F` is part of the code.

### 3.2 Features

All features are lagged or calendar features. No feature at origin `O` reads a week > `O`.

- **Lags:** `lag_1 … lag_8`.
- **Rolling:** means over 4, 8, 13, 26 weeks; standard deviations over 8 and 13; zero shares over
  13 and 26; `cv_13 = std_13 / mean_13` (clipped 0–10).
- **Intermittency:** `weeks_since_sale` (capped at 26), `pos_freq` = share of positive weeks,
  `trend_ratio = mean_4 / (mean_26 + 0.1)` (clipped 0–10).
- **Lifecycle:** `log_sku_age = log1p(weeks since first sale)`; levels that grow with calendar
  time (`n_pos_weeks`, raw `sku_age`) are deliberately not used because trees cannot extrapolate.
- **Price / mix:** `price_lag1`, `price_mean_13`, `log_price_lag1`, `uk_share_13`,
  `log_invoices_13` (log1p of mean invoices).
- **Calendar:** `woy_sin`, `woy_cos` (week-of-year), `month`, `is_december`.
- **Horizon:** the forecast horizon itself, so one model can serve all 1–8 week targets.

`assert_no_leakage()` checks that `target_idx > origin_idx`, that every feature is finite, and
`tests/test_pipeline.py` perturbs **all** weeks after the origin and verifies that no feature
changes. This is a stronger test than checking a single date.

### 3.3 Poisson GLM

`sklearn.linear_model.PoissonRegressor(alpha=0.5, max_iter=400, tol=1e-5)`, with all count-like
features (`lags`, rolling means/std, prices) transformed by `log1p` and all features standardised.
A Poisson mean is multiplicative; without the log transform the raw counts (four orders of
magnitude) dominate the linear predictor. `alpha=0.5` is an L2 penalty, chosen a priori, not tuned
on the test window.

### 3.4 LightGBM

Two global models on the same 400-tree configuration:

- `lgbm_tweedie`: Tweedie objective, variance power 1.3 (a continuous interpolation between
  Poisson at 1 and gamma at 2, appropriate for non-negative, zero-inflated demand).
- `lgbm_quantile`: quantile objective at α = 0.5. Its prediction is a **median**, not a mean; this
  matters for aggregate volume and for the point-order policy.

Shared parameters: 400 estimators, learning rate 0.04, 31 leaves, `min_child_samples=50`,
subsample 0.8 with frequency 1, colsample 0.8, L2 1.0, `random_state=20261024`,
`deterministic=True`. No SKU identity is used as a raw categorical.

The median model is included because it is the natural LightGBM quantile-regression baseline and
because it lets us show the difference between “best MASE” and “best order quantity”. It is not
presented as a mean forecast.

## 4. Uncertainty layer (shared by all models)

The point models differ; the uncertainty layer is identical, so the comparison isolates the
quality of the conditional mean (or median). It has two steps.

### 4.1 Variance-stabilising scale

For a count with mean μ, the standard deviation grows roughly with `sqrt(μ)`. Residuals are
standardised as

```
z = (y − μ) / sqrt(μ + 0.25)
```

The `+0.25` keeps the scale finite and non-degenerate when μ = 0 (a zero point forecast still
needs a positive upper quantile). This replaced an earlier negative-binomial variance function
`μ + αμ²`; a single global α collapsed the median to zero and destroyed coverage (DECISIONS.md).

### 4.2 Mondrian split conformal

For each origin `O` and each model:

1. The model is fitted on weeks ≤ `O − 13`.
2. On each calibration origin `t ∈ [O − 12, O]`, forecasts are made for h = 1…8. Only targets
   with `t + h ≤ O` are kept, so the calibration set contains only realised values.
3. `z` is computed and its empirical quantile function `Q_z` is estimated on a grid of 199
   probabilities (`0.005 … 0.995`), **separately for each demand class** (Mondrian conformal).
   If a class has <30 rows at a horizon, the pooled grid is used.
4. Predictive quantiles are `μ + sqrt(μ + 0.25) · Q_z(q)`, clipped at zero.

This yields P10/P50/P90 per origin, model, horizon and class. Because `Q_z` is an empirical
quantile function, the intervals are conformal by construction; the class split makes coverage
conditional on the demand segment rather than only marginal.

Calibration is applied **per horizon** (`kind = "horizon"`), not with one grid for all horizons.

### 4.3 Cumulative demand for the newsvendor

The newsvendor needs the distribution of cumulative demand over L weeks, not the sum of L marginal
quantiles (which would ignore dependence). A separate conformal grid is therefore built
(`kind = "aggregate"`):

- for each calibration origin and SKU, `y_L = Σ_{h=1..L} y_h` and `μ_L = Σ μ_h`;
- `z_L = (y_L − μ_L) / sqrt(μ_L + 0.25)`;
- empirical quantiles per demand class, same 199-point grid;
- the order is `μ_L + sqrt(μ_L + 0.25) · Q_{z_L}(CR)`.

This captures the empirical within-window correlation of the residuals, but it does **not** model
dynamic inventory: it is a single order covering L weeks, with no carry-over between weeks and no
intermediate replenishment. That is a deliberate simplification, stated in LIMITATIONS.md.

## 5. Accuracy metrics

All metrics use the same 9,558 observations per horizon (3,186 SKUs × 3 origins).

| Metric | Definition |
|---|---|
| MAE | `mean(|y − μ|)` |
| RMSE | `sqrt(mean((y − μ)²))` |
| MASE | `mean(|y − μ| / scale)`; `scale` = in-sample mean absolute change, or mean `|y_t − y_{t−52}|` if ≥52 weeks |
| MASE(P50) | same denominator, numerator uses the conformal P50 |
| Pinball(q) | `mean(max(q(y − q̂), (q − 1)(y − q̂)))` for q = 0.10/0.50/0.90 |
| Coverage80 | share of `y` in `[P10, P90]` |
| Aggregate WAPE | `|Σ_pred − Σ_actual| / Σ_actual` per origin; pooled = `Σ|Σ_pred − Σ_actual| / Σ Σ_actual` |

**MASE denominator.** If a SKU's scale is <0.01 (a deterministic or almost constant series), the
ratio explodes; two h=1 observations are excluded, and the count is reported
(`n_mase_excluded`). The exclusion is applied identically in the bulk sensitivity.

**Why MASE and not MAE alone.** Intermittent demand has a large zero atom; a model can win MAE by
predicting near zero. MASE scales by each SKU's own naive error, and the decision analysis checks
whether the ordering actually saves money. The aggregate WAPE is reported separately because
“sum over SKUs” answers a different question from “average SKU error”.

**Calibration curve.** `outputs/tables/calibration.csv` reports `q_obs = P(y ≤ q̂)` for a grid of
`q̂`. With a 47% zero atom, `P(y ≤ q10)` is inflated whenever `q10 = 0`; the meaningful lower-tail
check is `P(y < q10)`, which is reported in `results.json` and in README §4.

## 6. Newsvendor decision model

### 6.1 Costs

For lead time L (weeks):

```
Co = 0.01 × unit_price × L          # holding/obsolescence per unit over the L-week cover
Cu = (Cu/Co ratio) × Co             # lost margin per unit short
CR = Cu / (Cu + Co)                 # critical ratio ∈ {0.5, 0.75, 0.9, 0.95}
```

`unit_price` is the SKU's median observed price, winsorised at the 99.5th percentile. The four
ratios are 1, 3, 9, 19. `Co` and `Cu` are **assumptions**; absolute costs are price-scaled and are
not a P&L claim. Only comparisons within the same cost grid are meaningful.

### 6.2 Policies

For each origin, SKU and L:

- **point:** order `Q = μ_L`;
- **quantile:** order `Q = F_L⁻¹(CR)` from the cumulative conformal grid;
- **oracle:** order `Q = y_L` (clairvoyant, zero cost by construction; a bound, not achievable).

Costs: `holding = Co × max(Q − y_L, 0)`, `stockout = Cu × max(y_L − Q, 0)`,
`total = holding + stockout`. There is no fixed ordering cost, no salvage, no capacity limit and
no weekly carry-over.

Reported diagnostics: fill rate `1 − Σ under / Σ demand`; cycle service level `P(y_L ≤ Q)`;
average inventory `mean(max(Q − y_L, 0))`; weeks of cover = average inventory / average weekly
demand. The fill rate is against recorded sales, not latent demand (no stockout data).

### 6.3 Savings and uncertainty

Savings are always relative to the **best simple point rule** among historical mean, seasonal
naive, SES and Croston/SBA, at the same L and cost ratio. In every one of the 168
model × policy × L × ratio cells that baseline is SES (point).

A second comparator is recorded for every cell: the **strongest point model overall** (the cheapest
point policy in that cell; LightGBM Tweedie at the central case). The saving against it is reported
without a CI. The bootstrap below is defined against SES point, so attaching that interval to a
different difference would be a fabricated CI. The two benchmarks answer different questions — "does
the distribution beat a simple rule?" and "does it beat the best point model I have?" — and the
README reports both, including the cells where the second answer is negative.

The 95% CI is a bootstrap over SKUs: 500 resamples with replacement of the per-SKU cost vectors,
recomputing the difference in mean cost. It reflects cross-sectional heterogeneity, **not**
uncertainty over time, model specification or cost assumptions. The cost assumptions are swept
instead of bootstrapped.

### 6.4 Worst SKUs

At the focal cell (L = 4, Cu/Co = 9, LightGBM-median quantile policy), per-SKU costs are summed
across origins; the top 20 by cost are exported with description, class, price, demand and order.
The 20 SKUs account for 16.85% of total focal cost.

## 7. Sensitivities

### 7.1 Bulk orders

For the accuracy side, an evaluation week is excluded if its demand exceeds the SKU's training
p99.5 at that origin (with a floor of 1 unit). This removes 1.34% of h=1 observations and 14.86%
of evaluated units. MASE is recomputed for every model with the same scale exclusion. The ranking
is unchanged. The decision side is not re-estimated; trimming a cumulative L-week decision changes
the evaluated SKU set and the cost base, so a like-for-like saving is not available from the
existing run. This is an explicit limitation, not a silently omitted result.

### 7.2 Cold start

At each origin, SKUs are split by the number of positive weeks in their history: `<26` vs `≥26`.
h=1 MASE is pooled over origins for each bucket. The modelling sample requires ≥12 positive
pre-origin weeks, so the cold bucket is 12–25 weeks; SKUs below 12 are outside the sample and are
not hidden inside an average. At the first origin, 880 of 3,186 model SKUs are in the cold bucket.

### 7.3 Demand classes

Classes are assigned pre-origin from zero share and median demand (DATA.md §4). They are used for
Mondrian calibration, for the accuracy table, and for the error-by-class figure. The thresholds
are conventions, not estimated.

## 8. Reproducibility and determinism

- Seed `20261024` is used for LightGBM and the bootstrap.
- LightGBM runs with `deterministic=True`; the same inputs reproduce the same forecasts.
- No random train/test split exists. Origins are calendar positions.
- `src/run_all.py` runs each stage in a separate process, writes `outputs/results.json`, and then
  runs the tests. A failed test makes the build exit non-zero.
- The environment is recorded in `results.json.meta.versions` (Python 3.11.9, pandas 2.3.2,
  scikit-learn 1.7.2, LightGBM 4.7.0 in the reported run).

## 9. What is deliberately not modelled

- hierarchical reconciliation from SKU to category/total;
- promotions, price elasticity, competitor stock, weather, holidays beyond month/week-of-year;
- stockouts and latent demand;
- supplier lead-time uncertainty (L is fixed and known);
- multi-echelon inventory, MOQ, shelf life, capacity;
- SKU-level embedding or raw SKU categorical in the trees;
- a decision-side bulk-order rerun.

These are not oversights; they define the boundary of the claim. See LIMITATIONS.md.
