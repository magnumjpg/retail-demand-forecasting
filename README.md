# Forecasting sparse demand

**Probabilistic SKU-week forecasts and the cost of getting inventory wrong**  
UCI Online Retail II · 104 ISO weeks (2009-12-07 → 2011-11-28) · 3,186 model SKUs · 3 rolling origins

This is a decision memo, not a leaderboard entry. It asks whether calibrated probabilistic
forecasts reduce inventory cost under a newsvendor policy, and where the approach breaks.
Every number below is a field of [`outputs/results.json`](outputs/results.json); the file is
assembled by `src/run_all.py` from the pipeline artefacts, never typed into the docs by hand.

---

## 1. The answer in one page

1. **Probabilistic ordering is worth real money at high cost ratios, and the SES policy's value is
   concentrated there.** At a 4-week lead time and `Cu/Co = 9` (critical ratio 0.90), ordering at
   the 90th percentile of an SES forecast cuts total cost by **23.9%** versus the best simple
   point rule (SES; 95% bootstrap CI over SKUs: **21.9% to 26.7%**), and by **18.5%** versus the
   strongest point model overall (LightGBM-Tweedie point). At the most asymmetric grid point
   (`L = 8`, `Cu/Co = 19`) the saving is **47.8%** versus SES point (CI **45.7% to 50.7%**) and
   **45.9%** versus the strongest point model. For SES, at `Cu/Co = 1` the gain over SES point
   is **4.2%** or less (L=2: 4.22%, L=4: 3.71%, L=8: 1.65%) and at ratio 3 it is at most **4.74%**.
   The grid's largest ratio-1 saving is **14.21%** (LightGBM median, L=4), but that model falls to
   **1.6%** at ratio 3, so it is not a stable symmetric-cost result; the SES pattern is the robust
   one. Against the strongest point model the picture is harsher: at `Cu/Co = 1` the SES quantile
   policy **loses** to it at every lead time (L=2 **−7.9%**, L=4 **−11.2%**, L=8 **−1.8%**), and at
   ratio 3 it is roughly break-even (+0.1%, −1.4%, +1.4%).

2. **The simplest calibrated model wins the decision, not the most complex one.**
   At the central case, SES + conformal quantiles saves **23.9%** versus SES point and **18.5%**
   versus the strongest point model; LightGBM-Tweedie + conformal saves **22.4%** and **16.9%**;
   LightGBM-median + conformal **18.6%** and **12.9%**. The best simple point baseline is SES in
   **168 of 168** model × policy × lead-time × cost-ratio cells.

3. **Accuracy and decision value are not the same thing.**
   LightGBM median has the best per-SKU MASE at h=1 (**0.686**, versus SES **0.819** and
   seasonal naive **1.723**), yet it forecasts the *aggregate* volume worst: pooled WAPE
   **40.1%**, because a median forecast under-orders by construction. SES tracks aggregate
   volume to **7.6%** WAPE and wins the newsvendor. The distribution layer matters more than
   the point model: the same LightGBM-Tweedie mean moves from **6.6%** savings as a point rule
   to **22.4%** as a quantile rule.

4. **The data is dominated by zeros and a few bulk orders.**
   **46.9%** of model-sample SKU-weeks have zero demand; the top **1%** of invoices carry
   **20.5%** of units. Dropping the top 0.5% tail of evaluation weeks removes **1.34%** of
   observations but **14.9%** of demand, and improves every model's MASE without changing the
   ranking. Conclusions are directionally robust, but absolute errors are not.

5. **What fails is predictable.** Seasonal naive loses badly (MASE **1.723**; −112% decision
   value at the central case). The first fit window is 65 weeks (to 2011-02-28): one seasonal
   cycle, not two, and the two Decembers differ sharply, so a single lag-52 rule is fragile.
   Croston/SBA is worse than the historical mean on this data (**1.569** vs **1.319**). SKUs
   with a bulk-order week account for most of the worst-cost list: the top 20 SKUs are **16.9%**
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
| Croston/SBA | 25.727 | 86.714 | 1.569 | 11.759 | 0.800 | 29.13 |
| Seasonal naive | 44.467 | 220.920 | 1.723 | 21.250 | 0.782 | 44.57 |

Actual mean demand is **27.16** units/SKU-week. The two median-style models (LightGBM median
and Poisson GLM) are biased low: their mean forecasts are **16.28** and **20.09**. MASE rewards
them because most observations are zero and the absolute error is lower in the middle of the
distribution; the aggregate and decision metrics expose the cost of that bias.

**Aggregate error is a different question.** Pooled WAPE over the three origins at h=1:

| Model | Pooled WAPE |
|---|---:|
| Historical mean | **6.91%** |
| Croston/SBA | 7.25% |
| SES | 7.58% |
| Poisson GLM | 26.01% |
| LightGBM Tweedie | 27.10% |
| LightGBM median | 40.05% |
| Seasonal naive | 71.09% |

This is the central tension of the project: the model with the best per-SKU error is the worst
at total volume, and the newsvendor punishes that. A production system would reconcile SKU
forecasts to a volume plan; we do not, and we say so in [LIMITATIONS.md](LIMITATIONS.md).

### By demand class (MASE, h=1)

| Class | SES | LGBM Tweedie | LGBM median | Croston/SBA |
|---|---:|---:|---:|---:|
| fast | 0.621 | 0.695 | **0.565** | 0.671 |
| medium | 0.759 | 0.877 | **0.685** | 0.930 |
| slow | 0.692 | 0.808 | **0.551** | 0.874 |
| intermittent | 0.978 | 1.173 | **0.786** | 2.382 |

LightGBM median wins every class. LightGBM Tweedie loses to SES in every class. Croston/SBA is
the worst model in the intermittent class — the case it is designed for — which is a legitimate
finding, not a bug: SBA smooths towards a stable inter-arrival rate that this retailer does not
have. Cold-start SKUs (12–25 positive weeks at the first origin, **880** SKUs) are shown
separately: LightGBM median **0.558** vs SES **0.818** vs Croston **3.193**; SES is essentially
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
check is strict `<`. Coverage is slightly conservative at longer horizons (h=2: 0.788–0.870;
h=4: 0.806–0.908) because the marginal conformal grid widens with horizon while the zero atom
does not move. Figure `fig3_calibration.png` plots central-interval coverage curves.

---

## 5. Decision value

Newsvendor order for a lead time `L`: `Q = F⁻¹(CR)` with `CR = Cu / (Cu + Co)`,
`Co = 1% × unit price × L` per week held, `Cu = (Cu/Co) × Co`. Unit price is the median
observed price per SKU, winsorised at p99.5. **These are assumptions, swept explicitly; the
absolute cost level is not a claim about the company.** Two comparators are reported, because
they answer different questions:

- **vs best simple point rule** — SES in every cell; this is what the 500-rep bootstrap over
  SKUs estimates, so it carries a CI.
- **vs strongest point model overall** — LightGBM-Tweedie point, the cheapest point policy in
  the central case. No CI is claimed for this difference; the bootstrap is defined against SES.

### Central case: L = 4 weeks, Cu/Co = 9 (critical ratio 0.90)

| Policy | vs SES point | 95% CI | vs strongest point model | Fill rate | Avg inventory | Weeks of cover |
|---|---:|---:|---:|---:|---:|---:|
| **SES + conformal quantile** | **23.88%** | 21.91 – 26.67 | **18.53%** | 0.843 | 74.3 | 2.85 |
| LightGBM Tweedie + conformal | 22.38% | 19.62 – 25.29 | 16.94% | 0.847 | 81.8 | 3.13 |
| LightGBM median + conformal | 18.64% | 15.59 – 22.56 | 12.93% | 0.844 | 85.6 | 3.28 |
| Poisson GLM + conformal | 10.23% | 9.41 – 15.39 | 3.93% | 0.805 | 76.5 | 2.93 |
| LightGBM Tweedie point | 6.56% | 2.83 – 8.78 | 0.00% | 0.731 | 32.4 | 1.24 |
| Croston/SBA + conformal | 3.06% | 0.71 – 8.76 | −3.75% | 0.799 | 93.4 | 3.58 |
| Historical mean + conformal | −1.51% | −5.48 – 4.16 | −8.64% | 0.778 | 93.4 | 3.58 |
| SES point (baseline) | 0.00% | — | −7.02% | 0.710 | 29.9 | 1.15 |
| LightGBM median point | −38.94% | −44.28 – −35.22 | −48.70% | 0.547 | 8.0 | 0.31 |
| Seasonal naive + conformal | −112.29% | −113.37 – −80.78 | −127.19% | 0.838 | 365.0 | 13.97 |

At the central case the quantile policy raises fill from **71.0%** (SES point) to **84.3%** and
cuts stockouts, at the cost of more inventory. Holding cost is 64,177.60 and stockout cost
93,426.39 of the SES-quantile total 157,603.99; under this cost ratio the stockout side still
dominates. Against the strongest point model anywhere in the table — LightGBM Tweedie point — the
SES quantile still saves **18.53%**. Note that ordering the **median** (LightGBM median point)
has a fill rate of only **54.7%** and destroys value: a good MASE is not a good ordering rule.

### Cost-ratio and lead-time sweep

Saving of the SES + conformal quantile policy, % versus SES point (95% CI in `results.json`):

| Lead time | Cu/Co = 1 | 3 | 9 | 19 |
|---|---:|---:|---:|---:|
| 2 weeks | 4.22 | 2.19 | 21.63 | 39.25 |
| 4 weeks | 3.71 | 2.60 | 23.88 | 43.00 |
| 8 weeks | 1.65 | 4.74 | 28.40 | 47.80 |

The value is driven by asymmetry for the SES policy: at `Cu/Co = 1` the critical ratio
is 0.50 and an SES point forecast is already close to optimal (4.22% or less versus SES point).
The LightGBM median is the exception in that corner (14.21% at L=4, ratio 1) but falls to 1.6% at
ratio 3, so its gain is not a stable asymmetry result.

**Versus the strongest point model, the low-asymmetry case is not a win at all.** The SES quantile
policy saves **−7.89%** (L=2), **−11.20%** (L=4) and **−1.81%** (L=8) at `Cu/Co = 1`, and
**+0.13%**, **−1.41%**, **+1.40%** at ratio 3. The 23.9% headline is a high-asymmetry result, not a
general one. `fig4_cost_vs_ratio.png` now shows both comparators: the top row is versus SES point
with the bootstrap CI, the bottom row is versus the strongest point model as point estimates only.

Best cell per lead time (all at `Cu/Co = 19`, versus SES point / versus strongest point model):
L=2 **39.25%** [36.74, 42.69] / **33.85%**, L=4 **43.00%** [40.86, 46.66] / **38.41%**, L=8
**47.80%** [45.67, 50.71] / **45.90%**. Higher lead times need more cover, so a distribution matters
more; they also carry more inventory and a larger absolute cost.

---

## 6. Where it fails

**Bulk orders.** Evaluation weeks above each SKU's training p99.5 account for **14.86%** of
evaluated units while removing only **1.34%** of rows. Excluding them
improves every model's MASE (SES 0.819 → 0.624; LightGBM median 0.686 → 0.474; Croston 1.569 →
1.368) and leaves the ranking unchanged. Absolute MAE/RMSE and the worst-SKU list are still
dominated by these weeks. The decision-side counterpart was **not** re-run: dropping decisions
that contain a bulk week changes the evaluated SKU set and the cost base, so a like-for-like
saving is not available without a separate study (documented in LIMITATIONS.md).

**Worst SKUs.** The 20 worst SKUs by newsvendor cost (L=4, Cu/Co=9, LightGBM-median policy)
are **16.85%** of total cost. The worst is `84879` (ASSORTED COLOUR BIRD ORNAMENT), 2,639.31
cost units or **1.57%**, with 10,442 units of demand and an 8,007-unit order; its history is
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
  lagged feature and a cost proxy, not as a demand driver. The 22.8% of orders without a
  customer ID are not linked across time.
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

Individual stages: `python src/download_data.py`, `clean.py`, `backtest.py`, `decision.py`,
`figures.py`. The full run is deterministic given the seed (`20261024`); LightGBM is fitted with
`deterministic=True`.

**Quality gates:** rolling-origin only (no random split); all models evaluated on the same
SKU-weeks; no future information in features (tested by perturbing every post-origin week);
complete calendar spine; monotone quantiles; cost accounting identities; `results.json`
contract; 7 figures; **13 tests**.

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
| Unit price | median observed per SKU, winsorised p99.5 | `decision.price_vector` |
| Conformal window | 13 weeks, realised targets only | `config.CALIB_WEEKS` |
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
de pedir el cuantil 0,90 de un pronóstico SES reduce el coste total un **23,9%** frente a la mejor
regla puntual simple (SES) en el caso central (L=4, Cu/Co=9), con un intervalo bootstrap del
**21,9%–26,7%**; frente al mejor modelo puntual de toda la tabla (LightGBM-Tweedie) la reducción
sigue siendo del **18,5%**. En el punto más asimétrico de la malla (L=8, Cu/Co=19) el ahorro es del
**47,8%** frente a SES puntual y del **45,9%** frente al mejor puntual.

El resultado no es general. Con Cu/Co=1 la política por cuantiles **pierde** frente al mejor
modelo puntual en las tres duraciones (−7,9%, −11,2% y −1,8%) y con Cu/Co=3 está prácticamente
empatada. El valor de la distribución aparece cuando el coste de quedarse corto domina al de
almacenar, no antes.

La tensión central del proyecto es que precisión y valor de decisión no coinciden: LightGBM mediana
tiene el mejor MASE por SKU (0,686) pero el peor WAPE agregado (40,1%) y destruye valor como regla
de pedido, porque una mediana pide por debajo de la media por construcción. SES, el modelo más
simple, gana la comparación de decisión. Los costes (1% del precio por semana en almacén, ratios
1/3/9/19) y los plazos (2/4/8 semanas) son supuestos barridos, no datos de la empresa; no hay
información de roturas de stock, por lo que la demanda está censurada. Es un prototipo analítico
reproducible, no una política desplegada.

