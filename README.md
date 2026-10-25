# Forecasting sparse demand

**Probabilistic SKU-week forecasts and the cost of getting inventory wrong**  
UCI Online Retail II · 104 ISO weeks (2009-12-07 → 2011-11-28) · 3,186 model SKUs · 3 rolling origins

**At a glance.** *Question:* do calibrated probabilistic SKU forecasts reduce inventory cost
relative to point forecasts? *Data:* 972,068 cleaned UCI Online Retail II line items, aggregated
to 3,995 SKUs × 104 ISO weeks. *Sample:* 3,186 SKUs with ≥12 positive pre-origin weeks, three
rolling origins, 26-week test window. *Method:* seven point models plus a shared Mondrian
conformal layer, evaluated in a 3 lead-time × 4 cost-ratio newsvendor grid. *Headline:* at the
central cell (L=4, `Cu/Co=9`) the SES quantile rule saves **23.82%** versus SES point (95%
bootstrap CI **21.20–26.18%**, conditional on that cell). *Caveat:* the reported model is
selected ex post inside the central cell (a reporting convention, not the best cell); the best
quantile model per cell ranges from **2.43%** to **48.81%**, and at `Cu/Co=1` the SES quantile
rule loses to the cheapest point rule.

This is a decision memo, not a leaderboard entry. It asks whether calibrated probabilistic
forecasts reduce inventory cost under a newsvendor policy, and where the approach breaks.
Every number below is a field of [`outputs/results.json`](outputs/results.json); the file is
assembled by `src/run_all.py` from the pipeline artefacts, never typed into the docs by hand.

---

## 1. The answer in one page

1. **Probabilistic ordering is worth real money at high cost ratios, and the SES policy's value is
   concentrated there.** At a 4-week lead time and `Cu/Co = 9` (critical ratio 0.90), ordering at
   the 90th percentile of an SES forecast cuts total cost by **23.82%** versus the best simple
   point rule (SES; 95% bootstrap CI over SKUs: **21.20% to 26.18%**, one-sided paired bootstrap
   *p* < **0.0001** from 10,000 resamples for H0: savings ≤ 0), and by **18.62%** versus the
   cheapest point rule in that cell (LightGBM-Tweedie point). **This is an ex-post selection:** SES
   is the best of the seven quantile models inside this cell, and the CI is conditional on that
   choice. Across the 12 lead-time × cost-ratio cells the best quantile model ranges from
   **2.43%** to **48.81%** (median **17.50%**) and is positive in all 12. At the most asymmetric grid point
   (`L = 8`, `Cu/Co = 19`) the saving is **48.81%** versus SES point (CI **46.36% to 51.18%**) and
   **46.96%** versus the cheapest point rule. For SES, at `Cu/Co = 1` the gain over SES point
   is **4.23%** or less (L=2: 4.23%, L=4: 3.83%, L=8: 1.67%) and at ratio 3 it is at most **5.12%**.
   The grid's largest ratio-1 saving is **13.38%** (LightGBM-median quantile, L=4), but that
   model's quantile rule falls to **+0.54%** at ratio 3, so it is not a stable symmetric-cost
   result; the SES pattern is the robust one. Against the cheapest point rule the picture is
   harsher: at `Cu/Co = 1` the SES quantile policy **loses** at every lead time (L=2 **−7.70%**,
   L=4 **−10.76%**, L=8 **−1.54%**), and at ratio 3 it is roughly break-even (+0.18%, −1.47%,
   +1.82%).

2. **The simplest calibrated model wins the decision, not the most complex one.**
   At the central case, SES + conformal quantiles saves **23.82%** versus SES point and **18.62%**
   versus the cheapest point rule; LightGBM-Tweedie + conformal saves **22.31%** and **17.01%**;
   LightGBM-median + conformal **17.58%** and **11.95%**. The quantile policy beats the best simple
   point forecast (SES in every row) in all **12** lead-time × cost-ratio cells, over **168**
   model-policy-lead-ratio rows.

3. **Accuracy and decision value are not the same thing.**
   LightGBM median has the best per-SKU MASE at h=1 (**0.686**, versus SES **0.819** and
   seasonal naive **1.723**), yet its *aggregate* volume error is the **second-worst** in the
   table: pooled WAPE **40.1%**, better only than seasonal naive (71.1%), because a median
   forecast under-orders by construction. SES tracks aggregate volume to **7.6%** WAPE and wins
   the newsvendor. The distribution layer matters more than the point model: the same
   LightGBM-Tweedie mean moves from **6.39%** savings as a point rule to **22.31%** as a quantile
   rule.

4. **The data is dominated by zeros and a few bulk orders.**
   **46.9%** of model-sample SKU-weeks have zero demand; the top **1%** of invoices carry
   **20.5%** of units. Dropping the top 0.5% tail of evaluation weeks removes **1.34%** of
   observations but **14.9%** of demand, and improves every model's MASE without changing the
   ranking. Conclusions are directionally robust, but absolute errors are not.

5. **What fails is predictable.** Seasonal naive loses badly (MASE **1.723**; −129.8% decision
   value at the central case). The first fit window is 65 weeks (to 2011-02-28): one seasonal
   cycle, not two, and the two Decembers differ sharply, so a single lag-52 rule is fragile.
   Croston/SBA is worse than the historical mean on this data (**1.692** vs **1.319**). SKUs
   with a bulk-order week account for most of the worst-cost list: the top 20 SKUs are **16.15%**
   of total newsvendor cost.

---

## 2. Data at a glance

| Item | Value | Source in `results.json` |
|---|---:|---|
| Raw rows (both sheets) | 1,067,371 | `data.raw_rows_two_sheets` |
| Rows after cleaning | 972,068 (**91.07%**) | `data.rows_after_cleaning` |
| Complete ISO weeks | 104 (2009-12-07 → 2011-11-28) | `data.weeks_total`, `data.weeks_first/last` |
| Cancellation rows (separated, never netted) | 19,104 rows, −476,819 units | `data.cancellations` |
| SKUs after cleaning / universe / model sample | 4,675 / 3,995 / 3,186 | `data.skus_after_cleaning`, `n_model_skus` |
| Zero-demand SKU-weeks (model sample) | **46.93%** | `data.zero_demand_share_model_skus` |
| Top 1% invoices' share of units | **20.46%** | `data.bulk_orders.top_1pct_invoice_share_units` |
| Largest single invoice | 87,167 units | `data.bulk_orders.largest_invoice_units` |

The two partial calendar weeks (2009-11-30 and 2011-12-05) are dropped. The modelling sample
requires **≥12 positive weeks before the first origin** (3,186 of 3,995 universe SKUs); the
selection is frozen on pre-origin data. See [DATA.md](DATA.md) for the full row/revenue audit.

Backtest: origins **2011-05-30**, **2011-07-25**, **2011-09-19**; 26-week test window;
3,186 SKUs × 3 origins = **9,558** SKU-week observations per horizon. Models are fitted only up
to `origin − 13 weeks`; the 13-week conformal window uses only targets realised by the origin.
For a cumulative `L`-week grid, a calibration origin enters only when its whole window is
realised (`t + L ≤ origin`), so 12 calibration origins are usable at h=1 and 11 / 9 / 5 at
`L = 2 / 4 / 8`; partial windows are never mixed into a cumulative residual grid.

---

## 3. Point accuracy at h = 1

Same 9,558 SKU-weeks for every model. MASE is scaled by each SKU's in-sample seasonal-naive
MAE; two observations with scale < 0.01 are excluded (deterministic series), because a
near-zero denominator otherwise explodes MASE.

| Model | MAE | RMSE | MASE ↓ | Pinball P50 ↓ | 80% coverage | Mean forecast |
|---|---:|---:|---:|---:|---:|---:|
| **LightGBM median** | **17.482** | 77.061 | **0.686** | **8.723** | 0.789 | 16.28 |
| SES | 19.910 | 75.960 | 0.819 | 9.521 | 0.804 | 26.03 |
| LightGBM Tweedie | 20.810 | 76.998 | 0.961 | 9.656 | 0.778 | 28.07 |
| Poisson GLM | 19.736 | 77.796 | 1.219 | 9.207 | 0.801 | 20.09 |
| Historical mean | 25.720 | 79.473 | 1.319 | 11.250 | 0.806 | 29.03 |
| Croston/SBA | 25.934 | 86.762 | 1.692 | 11.746 | 0.798 | 29.87 |
| Seasonal naive | 44.467 | 220.920 | 1.723 | 21.250 | 0.782 | 44.57 |

Actual mean demand is **27.16** units/SKU-week. The two median-style models (LightGBM median
and Poisson GLM) are biased low: their mean forecasts are **16.28** and **20.09**. MASE rewards
them because most observations are zero and the absolute error is lower in the middle of the
distribution; the aggregate and decision metrics expose the cost of that bias.

**Aggregate error is a different question.** Pooled WAPE over the three origins at h=1:

| Model | Pooled WAPE |
|---|---:|
| Historical mean | **6.91%** |
| SES | 7.58% |
| Croston/SBA | 9.99% |
| Poisson GLM | 26.01% |
| LightGBM Tweedie | 27.10% |
| LightGBM median | 40.05% |
| Seasonal naive | 71.09% |

This is the central tension of the project: the model with the best per-SKU error has the
second-worst aggregate WAPE (only seasonal naive is worse), and the newsvendor punishes that.
A production system would reconcile SKU forecasts to a volume plan; we do not, and we say so
in [LIMITATIONS.md](LIMITATIONS.md).

### By demand class (MASE, h=1)

| Class | SES | LGBM Tweedie | LGBM median | Croston/SBA |
|---|---:|---:|---:|---:|
| fast | 0.621 | 0.695 | **0.565** | 0.671 |
| medium | 0.759 | 0.877 | **0.685** | 0.931 |
| slow | 0.692 | 0.808 | **0.551** | 0.874 |
| intermittent | 0.978 | 1.173 | **0.786** | 2.628 |

LightGBM median wins every class. LightGBM Tweedie loses to SES in every class. Croston/SBA is
the worst model in the intermittent class — the case it is designed for — which is a legitimate
finding, not a bug: SBA smooths towards a stable inter-arrival rate that this retailer does not
have. Cold-start SKUs (12–25 positive weeks at the first origin, **880** SKUs) are shown
separately: LightGBM median **0.558** vs SES **0.818** vs Croston **3.689**; SES is essentially
unchanged from the warm group (**0.819**).

---

## 4. Calibration

Intervals are produced by one shared layer: standardise residuals as
`z = (y − μ) / sqrt(μ + 0.25)`, take empirical quantiles of `z` on the 13-week pre-origin
window **separately per demand class** (Mondrian conformal), then map back. The `+0.25` keeps
the scale finite at μ = 0.

| Check (h=1, all models) | Result | Target |
|---|---:|---:|
| Empirical coverage of P10–P90 | **0.778 – 0.806** | 0.80 |
| `P(y < q10)` | **0.084 – 0.135** | 0.10 |
| `P(y ≤ q90)` | **0.888 – 0.914** | 0.90 |
| `P(y ≤ q10)` (descriptive only) | 0.521 – 0.581 | — |

The last row is not a failure: **46.9%** of demands are exactly zero, so the CDF has a large
atom. If `q10` is also zero, then `P(y ≤ q10)` includes the whole atom; the meaningful lower-tail
check is strict `<`. At h=1, coverage sits at or just below the 80% target (0.778–0.806); at
longer horizons it becomes over-covered (h=2: 0.792–0.874; h=4: 0.798–0.914), because the marginal
conformal grid widens with horizon while the zero atom does not move. Figure
`fig3_calibration.png` plots central-interval coverage curves.

---

## 5. Decision value

Newsvendor order for a lead time `L`: `Q = F⁻¹(CR)` with `CR = Cu / (Cu + Co)`,
`Co = 1% × unit price × L` per week held, `Cu = (Cu/Co) × Co`. Unit price is the median
observed price per SKU over weeks up to each forecast origin (no evaluation-window price
information), winsorised at p99.5. **These are assumptions, swept explicitly; the absolute cost
level is not a claim about the company.** Two comparators are reported, because they answer
different questions:

- **vs best simple point rule** — SES in every cell; this is what the 500-rep bootstrap over
  SKUs estimates, so it carries a CI.
- **vs cheapest point rule in each cell** — the lowest-cost point policy among all seven models.
  That rule is LightGBM-Tweedie at the central case and LightGBM-median at `Cu/Co = 1`, so the
  label does not name one model for the whole grid. No CI is claimed for this difference; the
  bootstrap is defined against SES.

### Central case: L = 4 weeks, Cu/Co = 9 (critical ratio 0.90)

| Policy | vs SES point | 95% CI | vs cheapest point rule | Fill rate | Avg inventory | Weeks of cover |
|---|---:|---:|---:|---:|---:|---:|
| **SES + conformal quantile** | **23.82%** | 21.20 – 26.18 | **18.62%** | 0.846 | 76.6 | 2.93 |
| LightGBM Tweedie + conformal | 22.31% | 19.10 – 25.29 | 17.01% | 0.850 | 84.2 | 3.22 |
| LightGBM median + conformal | 17.58% | 13.88 – 20.99 | 11.95% | 0.854 | 93.5 | 3.58 |
| Poisson GLM + conformal | 10.50% | 7.10 – 13.60 | 4.39% | 0.811 | 80.4 | 3.08 |
| LightGBM Tweedie point | 6.39% | 3.41 – 9.38 | 0.00% | 0.731 | 32.4 | 1.24 |
| Croston/SBA + conformal | 4.36% | 0.23 – 8.31 | −2.17% | 0.801 | 92.8 | 3.55 |
| Historical mean + conformal | −1.88% | −7.03 – 2.92 | −8.84% | 0.780 | 95.8 | 3.67 |
| SES point (baseline) | 0.00% | — | −6.82% | 0.710 | 29.9 | 1.15 |
| LightGBM median point | −39.01% | −43.67 – −35.07 | −48.49% | 0.547 | 8.0 | 0.31 |
| Seasonal naive + conformal | −129.84% | −151.48 – −111.72 | −145.53% | 0.845 | 408.0 | 15.62 |

At the central case the quantile policy raises fill from **71.0%** (SES point) to **84.6%** and
cuts stockouts, at the cost of more inventory. Holding cost is 66,435.28 and stockout cost
92,918.61 of the SES-quantile total 159,353.89; under this cost ratio the stockout side still
dominates. Against the cheapest point rule in the cell — LightGBM Tweedie point — the SES
quantile still saves **18.62%**. Note that ordering the **median** (LightGBM median point) has a
fill rate of only **54.7%** and destroys value: a good MASE is not a good ordering rule.

### Cost-ratio and lead-time sweep

Saving of the SES + conformal quantile policy, % versus SES point (95% CI in `results.json`):

| Lead time | Cu/Co = 1 | 3 | 9 | 19 |
|---|---:|---:|---:|---:|
| 2 weeks | 4.23 | 2.14 | 21.61 | 39.31 |
| 4 weeks | 3.83 | 2.37 | 23.82 | 43.12 |
| 8 weeks | 1.67 | 5.12 | 29.66 | 48.81 |

The value is driven by asymmetry for the SES policy: at `Cu/Co = 1` the critical ratio
is 0.50 and an SES point forecast is already close to optimal (4.23% or less versus SES point).
The LightGBM median is the exception in that corner (13.38% at L=4, ratio 1) but its quantile
rule falls to +0.54% at ratio 3, so its gain is not a stable asymmetry result.

**Versus the cheapest point rule, the low-asymmetry case is not a win at all.** The SES quantile
policy saves **−7.70%** (L=2), **−10.76%** (L=4) and **−1.54%** (L=8) at `Cu/Co = 1`, and
**+0.18%**, **−1.47%**, **+1.82%** at ratio 3. The 23.82% headline is a high-asymmetry result, not
a general one. `fig4_cost_vs_ratio.png` shows both comparators: the top row is versus SES point
with the bootstrap CI, the bottom row is versus the cheapest point rule in each cell as point
estimates only.

Best cell per lead time (all at `Cu/Co = 19`, versus SES point / versus cheapest point rule):
L=2 **39.31%** [36.35, 42.56] / **33.99%**, L=4 **43.12%** [39.89, 45.97] / **38.66%**, L=8
**48.81%** [46.36, 51.18] / **46.96%**. Higher lead times need more cover, so a distribution matters
more; they also carry more inventory and a larger absolute cost.

---

## 6. Where it fails

**Bulk orders.** Evaluation weeks above each SKU's training p99.5 account for **14.86%** of
evaluated units while removing only **1.34%** of rows. Excluding them
improves every model's MASE (SES 0.819 → 0.624; LightGBM median 0.686 → 0.474; Croston 1.692 →
1.493) and leaves the ranking unchanged. Absolute MAE/RMSE and the worst-SKU list are still
dominated by these weeks. The decision-side counterpart was **not** re-run: dropping decisions
that contain a bulk week changes the evaluated SKU set and the cost base, so a like-for-like
saving is not available without a separate study (documented in LIMITATIONS.md).

**Worst SKUs.** The 20 worst SKUs by newsvendor cost (L=4, Cu/Co=9, LightGBM-median quantile
policy) are **16.15%** of total cost. The worst is `84879` (ASSORTED COLOUR BIRD ORNAMENT), 2,639.85
cost units or **1.53%**, with 10,442 units of demand and an 8,223.81-unit order; its history is
dominated by one extreme week. `fig7_worst_skus.png` plots all 20 histories; `outputs/tables/worst_skus.csv`
has descriptions, class and price. The pattern is consistent: fast movers with one extreme
week, not slow movers.

**Seasonal naive.** MASE 1.723 and a negative decision value at every cost ratio; it is
included as a baseline, not as a candidate. The first model fit ends 2011-02-28 and has no
complete seasonal cycle; the second December in-sample overlaps the first sheet. Two years is
not enough for a seasonal model, and we do not claim otherwise.

**Cold start.** SKUs with fewer than 12 positive pre-origin weeks are excluded from the model
sample (3,482 SKUs pass ≥8, 3,186 pass ≥12 at the first origin). Among those retained, 880 have
12–25 positive weeks. LightGBM median is strongest there (MASE 0.558), but this is also where
the median's under-forecasting is most dangerous for ordering; the backtest does not measure
the cost of cold-start SKUs separately, which is a limitation.

---

## 7. What this does not prove

- **Not the company's costs.** Holding cost = 1% of unit price per week and the four `Cu/Co`
  ratios are assumptions. The absolute costs and the optimal policy change with them; only the
  sweep is evidence. No real margin, stockout penalty, salvage value, MOQ or capacity constraint
  is observed.
- **No stockout data.** Zeros are observed no-sales, not necessarily zero demand. Demand is
  censored; fill rate is measured against recorded sales. A true lost-sales model needs stock
  levels we do not have.
- **No promotions, price changes, competitor actions or weather.** Price is used only as a
  lagged feature and a cost proxy, not as a demand driver. The 22.8% of line items (16.3% of
  invoices) without a customer ID are not linked across time.
- **The cost price is a proxy, and its basis matters little here.** The primary cost vector is
  the median price observed before each origin. Costing the same central-case orders with the
  full-sample median price moves the saving from 23.82% to 23.77% (vs SES) and from 18.62% to
  18.41% (vs the cheapest point rule), so the no-look-ahead choice is not producing the result.
  It is still an observed selling price, not a purchase cost.
- **Conformal windows shrink with the horizon.** A cumulative `L`-week grid keeps only
  calibration origins whose whole window is realised: 11 / 9 / 5 origins at L = 2 / 4 / 8,
  against 12 at h=1. The L=8 tails therefore rest on five overlapping windows, and that is one
  reason to treat the L=8 saving as the least stable in the grid.
- **Short history and three origins.** 104 weeks, three rolling origins in adjacent 8-week
  blocks, one dataset. The bootstrap CI is over SKUs, not over time; it does not quantify
  temporal uncertainty or regime change. Results are not evidence of out-of-sample performance
  on another retailer or another year.
- **No reconciliation.** SKU forecasts are not reconciled to a category or total-volume plan.
  The aggregate bias of LightGBM median (40.1% WAPE) is a symptom; a production system would
  add hierarchical reconciliation.
- **No decision-side bulk trimming.** The accuracy-side sensitivity is reported; the decision
  table is not re-estimated on a trimmed demand set.
- **Not production.** No drift monitoring, retraining schedule, model registry, approval
  workflow, or integration with an ERP/inventory system. This is a reproducible analytical
  prototype, not a deployed policy.
- **MASE denominators.** MASE is undefined for deterministic series; two observations are
  excluded. Results for very smooth SKUs should not be read as if the denominator were stable.
- **The "oracle" is clairvoyant.** It orders realised demand and has zero cost by construction;
  it is a bound, not an achievable policy.

---

## Literature and positioning

The project sits between intermittent-demand forecasting and inventory theory. These are the
works that inform the design; it does not claim to extend them.

- **Croston (1972), "Forecasting and Stock Control for Intermittent Demands", *Operational
  Research Quarterly* 23(3).** The original size/interval decomposition that `croston_sba`
  implements; our result is that it does not help on this retailer's demand.
- **Syntetos & Boylan (2005), "The Accuracy of Intermittent Demand Estimates",
  *International Journal of Forecasting* 21(2).** Motivates the Syntetos–Boylan bias correction
  `(1 − α/2)` and the use of intermittent-demand baselines rather than a straw man.
- **Hyndman, Koehler, Snyder & Grose (2002), "A State Space Framework for Automatic Forecasting
  Using Exponential Smoothing Methods", *International Journal of Forecasting* 18(3).**
  Underpins the SES/ETS family and the MASE scale used to compare point forecasts across SKUs.
- **Makridakis, Spiliotis & Assimakopoulos (2020), "The M4 Competition: 100,000 Time Series and
  61 Forecasting Methods", *International Journal of Forecasting* 36(1).** The benchmark showing
  that simple statistical methods remain hard to beat; consistent with SES winning the decision
  comparison here.
- **Ke et al. (2017), "LightGBM: A Highly Efficient Gradient Boosting Decision Tree", NeurIPS.**
  The implementation behind the Tweedie-mean and quantile-0.5 point models; included to test
  whether a stronger point model improves the ordering decision (it does not).
- **Romano, Patterson & Candès (2019), "Conformalized Quantile Regression", NeurIPS.** The
  conformal framing that lets one uncertainty layer be applied to every model; this project uses
  split conformal with a variance-stabilising scale rather than CQR itself.
- **Petropoulos et al. (2022), "Forecasting: Theory and Practice", *International Journal of
  Forecasting* 38(3).** A scope reference for what a forecasting study should report and which
  uncertainties (model, temporal, cost) a backtest does not cover.
- **Silver, Pyke & Thomas (2016), *Inventory and Production Management in Supply Chains*
  (4th ed., CRC Press).** The single-period newsvendor logic and the critical-ratio rule used in
  the decision layer.

---

## 8. Reproduce

```powershell
python -m pip install -r requirements.txt
python src/run_all.py
```

`run_all.py` verifies/downloads the workbook (sha256), cleans, runs the rolling-origin backtest,
the newsvendor analysis and all figures, assembles `outputs/results.json`, and runs the tests.
To iterate without re-reading the 45 MB workbook:

```powershell
python src/run_all.py --from-cache          # reuse data/processed/transactions_raw.parquet
python src/run_all.py --skip-tests
python -m pytest -q tests/test_pipeline.py  # tests alone
```

A manual GitHub Actions workflow (`.github/workflows/full-pipeline.yml`, `workflow_dispatch` only)
downloads the UCI archive, rebuilds every cache and runs the full suite; the light `ci.yml` on
push/PR runs the cache-free subset of the tests.

Individual stages: `python src/download_data.py`, `clean.py`, `backtest.py`, `decision.py`,
`figures.py`. The full run is deterministic given the seed (`20261024`); LightGBM is fitted with
`deterministic=True`.

**Quality gates:** rolling-origin only (no random split); all models evaluated on the same
SKU-weeks; no future information in features (tested by perturbing every post-origin week);
complete calendar spine; monotone quantiles; cumulative conformal grids built only from complete
L-week windows; cost vector priced from pre-origin data; compact 0…n−1 bootstrap index; cost
accounting identities; `results.json` contract; 7 figures; the realised SKU set is compared
across models in all 24 origin × horizon cells (3,186 SKUs per cell); **24 tests**.

---

## 9. Repository layout

```
03_online_retail_forecasting/
  README.md                 this memo
  DATA.md                   source, hashes, cleaning audit, panel schema
  METHOD.md                 models, features, conformal layer, metrics
  LIMITATIONS.md            what the design cannot answer
  DECISIONS.md              decisions, alternatives, blockers
  ANALYSIS_LOG.md           chronological record, including the failures
  RED_TEAM.md               hostile objections, answered or conceded
  requirements.txt
  src/
    config.py               paths, filters, origins, cost assumptions
    download_data.py        download + sha256 verification + manifest
    clean.py                audit, cancellations, spine, SKU selection
    features.py             lag/rolling/calendar features + leakage assertions
    models.py               baselines, Poisson GLM, LightGBM, conformal layer
    backtest.py             rolling origins, metrics, calibration, sensitivities
    decision.py             newsvendor sweep, bootstrap, worst SKUs
    figures.py              7 figures
    run_all.py              one-command rebuild + results.json
  data/
    raw/                    workbook + manifest (workbook git-ignored)
    processed/
      weekly_demand.csv     3,995 SKUs x 104 weeks, explicit zeros
      clean_stats.json      row/revenue audit
      sku_selection.json    frozen pre-origin model sample
      forecasts.parquet     mu, P10/P50/P90 per origin/model/horizon/SKU
      residual_quantiles.csv
      _backtest.json, _decision.json
  outputs/
    results.json            single source of truth for every README number
    tables/                 accuracy, by class, aggregate, calibration, cold start,
                            decision, worst SKUs, origins
    figures/                fig1..fig7
  tests/test_pipeline.py
```

Figures: intermittency and concentration (`fig1`); rolling forecast paths for a fast, medium and
slow SKU (`fig2`); calibration (`fig3`); decision value vs cost ratio (`fig4`); MASE by demand
class (`fig5`); service–inventory frontier (`fig6`); worst 20 SKUs (`fig7`).

---

## 10. Assumptions register

| Assumption | Value | Where |
|---|---|---|
| Holding cost | 1% of unit price per week | `config.HOLD_COST_WEEK` |
| Underage/overage ratios | 1, 3, 9, 19 | `config.COST_RATIOS` |
| Lead times | 2, 4, 8 weeks | `config.LEAD_TIMES` |
| Critical ratio | `Cu / (Cu + Co)` | `decision.py` |
| Unit price | median observed per SKU up to each origin, winsorised p99.5 | `decision.price_vector` |
| Conformal window | 13 candidate weeks, realised targets only; complete L-week windows for cumulative grids | `config.CALIB_WEEKS`, `backtest.aggregate_grids` |
| Calibration | Mondrian by demand class, `sqrt(mu + 0.25)` scale | `models.py` |
| Model sample | ≥12 positive weeks before first origin | `clean.py` |
| Evaluation | 26-week test window, 3 origins, h=1/2/4 accuracy | `config.py` |
| Bootstrap | 500 reps over SKUs, seed 20261024 | `config.py` |

Dataset: Chen, D. (2012), *Online Retail II*, UCI Machine Learning Repository,
<https://archive.ics.uci.edu/dataset/502/online+retail+ii>. Workbook sha256
`bcbe73b3…f2e980` (verified by `download_data.py` and recorded in `manifest.json`).
See [DATA.md](DATA.md).

---

## Resumen en español

Este proyecto pregunta si unos pronósticos probabilísticos bien calibrados reducen el coste de
inventario frente a reglas de pedido puntuales. Sobre 3.186 SKU y tres orígenes móviles, la regla
de pedir el cuantil 0,90 de un pronóstico SES reduce el coste total un **23,82%** frente a la mejor
regla puntual simple (SES) en el caso central (L=4, Cu/Co=9), con un intervalo bootstrap del
**21,20%–26,18%**; frente a la regla puntual más barata de esa celda (LightGBM-Tweedie) la
reducción sigue siendo del **18,62%**. En el punto más asimétrico de la malla (L=8, Cu/Co=19) el
ahorro es del **48,81%** frente a SES puntual y del **46,96%** frente a la regla puntual más barata.
El 23,82% es el mejor de los siete modelos de cuantiles dentro de la celda central, elegido ex post;
entre las 12 celdas el mejor modelo de cuantiles ahorra entre **2,43%** y **48,81%** (mediana
**17,50%**), y el intervalo es condicional a la celda elegida.

El resultado no es general. Con Cu/Co=1 la política por cuantiles **pierde** frente a la regla
puntual más barata en las tres duraciones (−7,70%, −10,76% y −1,54%) y con Cu/Co=3 está
prácticamente empatada. El valor de la distribución aparece cuando el coste de quedarse corto
domina al de almacenar, no antes.

La tensión central del proyecto es que precisión y valor de decisión no coinciden: LightGBM mediana
tiene el mejor MASE por SKU (0,686) pero el segundo peor WAPE agregado (40,1%; solo la naive
estacional es peor) y destruye valor como regla de pedido, porque una mediana pide por debajo de la
media por construcción. SES, el modelo más simple, gana la comparación de decisión. Los costes (1%
del precio por semana en almacén, ratios 1/3/9/19) y los plazos (2/4/8 semanas) son supuestos
barridos, no datos de la empresa; no hay información de roturas de stock, por lo que la demanda
está censurada. Es un prototipo analítico reproducible, no una política desplegada.

