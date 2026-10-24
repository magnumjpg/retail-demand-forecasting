# Limitations

This file states what the analysis **cannot** establish. It is deliberately specific: each
limitation is paired with its consequence for the reported numbers and with the change that would
remove it. The README's "What this does not prove" is the short version; this is the audit trail.

The project is a reproducible analytical prototype on one public dataset. It is not a deployed
inventory policy, and none of the cost numbers are a claim about the retailer's accounts.

---

## 1. Data limitations

### 1.1 Demand is censored — there is no stockout data

The dataset records **sales**, not demand. A zero SKU-week may mean no customer, no stock, a
delisted item, or a data gap. We cannot separate them.

**Consequence.** Every model is fitted to observed sales, so a stockout looks like low demand and
the model learns to under-forecast exactly when inventory was constrained. The fill rate in
`decision.csv` is `1 − under / recorded demand`, not a true service level. The newsvendor
stockout term is computed against recorded sales, so it is a **lower bound** on the cost of being
short. The reported savings could move in either direction with true lost-sales data: a better
model of latent demand would raise the value of holding cover, but it would also change which SKUs
look intermittent.

**What would fix it.** Inventory snapshots or stockout flags at SKU-week level, or a censored-demand
model (e.g. a Tobit/latent-demand likelihood). Neither is in Online Retail II.

### 1.2 Costs, margins and lead times are assumptions

`Co = 1% × unit price × L` per unit held over the cover, `Cu = (Cu/Co) × Co` with ratios
`{1, 3, 9, 19}`, and lead times `{2, 4, 8}` weeks are **invented for the sweep**. Unit price is the
SKU's median observed price, winsorised at p99.5, used as a proxy for value — not the purchase cost,
margin, or salvage value.

**Consequence.** Absolute costs (`157,603.99` at the central SES-quantile cell, for example) are
not a P&L statement and should not be quoted as money. Only the *ordering* of policies at the same
`L` and ratio is meaningful. For the SES quantile policy, a symmetric cost ratio (`Cu/Co = 1`,
critical ratio 0.5) produces savings of at most **4.22%** (and **1.65%** at L=8). The grid's
largest ratio-1 saving is **14.21%** (LightGBM median, L=4), but that model is only **1.6%** at
ratio 3, so the symmetric-cost result is model-specific, not a general gain. Real margin, markdown,
salvage, fixed ordering cost, MOQ and capacity are absent.

**What would fix it.** Finance-owned unit costs and margins, observed lead-time distributions, and
a cost model with fixed order cost and salvage. The sweep in `decision.csv` is the honest
substitute until then.

### 1.3 Two years is one seasonal cycle, and the Decembers differ

After dropping the two partial weeks, the panel is **104 weeks** (2009-12-07 → 2011-11-28). The
first model fit ends **2011-02-28** (65 weeks), so it contains one full seasonal cycle, not two.
Seasonal naive has to reuse a single lag-52 observation; the two Decembers in the sample differ.
There are only three forecast origins.

**Consequence.** Seasonal naive is a weak baseline by construction (MASE **1.723** at h=1), and no
claim about annual seasonality is identified. A model that needs several years of history — or a
neural seasonal model — cannot be fairly evaluated here. Regime change (the 2008–09 recession is
outside the window, but the retailer's assortment changes inside it) is not modelled.

**What would fix it.** More years of the same SKU universe, or a hierarchical seasonal model with
partial pooling. The current data cannot supply it.

### 1.4 Bulk orders dominate the tails

The top **1%** of invoices carry **20.46%** of units; the largest single invoice is **87,167**
units. In evaluation, weeks above each SKU's training p99.5 are **1.34%** of h=1 observations but
**14.86%** of evaluated units.

**Consequence.** MAE, RMSE and the worst-SKU list are dominated by a handful of bulk weeks. The
accuracy-side sensitivity removes them and every model improves (SES 0.819 → 0.624; LightGBM
median 0.686 → 0.474), with the ranking unchanged. But the **decision side was not re-estimated**:
trimming a cumulative L-week decision removes whole SKU-decisions and changes the cost base, so a
like-for-like saving is not available from this run. The 20 worst SKUs are **16.85%** of focal
newsvendor cost; that concentration is a finding about the data, not a stable SKU list.

**What would fix it.** A decision-side rerun on a trimmed demand set, and a bulk-order indicator or
a mixture model with a separate heavy-tail component. Both are next-iteration work, not claims
made here.

### 1.5 No promotions, price elasticity, competitor or weather data

Price appears only as a lagged feature and a cost proxy. There are no promotion flags, no
competitor prices, no marketing calendar, no weather, and no holiday indicator beyond month and
week-of-year. **22.77%** of raw orders have no customer ID, and customer identity is not used at
all.

**Consequence.** The models estimate an average seasonal/autoregressive pattern, not a causal
response. A promotion week looks like a demand spike and is treated as such; the forecast cannot
anticipate the next promotion. Omitted variables can bias both accuracy and the decision
comparison, although the comparison is between models on the same information set, so it is still
internally valid.

**What would fix it.** Promotion and price-history feeds, and a causal design. The public dataset
does not contain them.

### 1.6 Cleaning choices change the demand definition

Cancellations (19,104 rows, −476,819 units) are excluded, never netted. Non-product codes (9,002
rows), negative non-cancellation quantities (755 rows), zero/negative prices (961 rows) and the two
partial calendar weeks (31,146 rows) are dropped. The resulting demand is **positive units
shipped**, not net demand.

**Consequence.** A returned unit does not reduce the demand of the original week, and a cancelled
order is invisible to the model. If returns are concentrated in particular SKUs, the demand series
is overstated for those SKUs. The cleaning is fully audited and reconciles to 972,068 rows, but it
is one defensible definition among several.

**What would fix it.** A net-demand variant, and a sensitivity run on the return treatment. The
audit in `DATA.md` makes such a variant cheap to add.

---

## 2. Sample and evaluation limitations

### 2.1 Three adjacent origins are not three independent samples

The origins are **2011-05-30**, **2011-07-25** and **2011-09-19** — adjacent 8-week blocks inside
the last 26 weeks. At h=1 they contribute target weeks 78, 86 and 94, not a continuous 26-week
path; longer horizons fill some gaps. The same SKUs appear at all origins, and the origins share
history.

**Consequence.** The 500-rep bootstrap resamples **SKUs**, not time. Confidence intervals describe
cross-sectional heterogeneity only. They do not cover temporal uncertainty, regime change, or the
fact that the three origins are correlated. No Diebold–Mariano or other paired forecast test is
run, so "model A beats model B" is a point-estimate statement, not a significance claim. The
project also does not claim out-of-sample performance on another retailer or another year.

**What would fix it.** More origins (e.g. monthly rolling with a longer history), a block bootstrap
over time, and paired loss-difference tests.

### 2.2 The model sample excludes 296 low-history SKUs

The panel keeps SKUs with **≥8** positive weeks (**3,995**). The model sample requires **≥12**
positive weeks **before the first origin** (**3,186**). SKUs with 8–11 positive pre-origin weeks are
in the panel but no metric describes them. The cold-start split covers the retained SKUs with
**12–25** positive weeks (**880** at the first origin); SKUs below 12 are outside it.

**Consequence.** The headline accuracy and savings do not apply to the newest or rarest SKUs — the
population where a cold-start policy is most needed. Excluding them is not hidden inside an
average, but it does mean the reported result is conditional on a minimum history. Selection uses
only pre-origin data, so it avoids survivor bias from future sales, but it is still a selection
rule.

**What would fix it.** A separate cold-start model (hierarchical, SKU-embedding or category-level)
evaluated on the 8–11-week group, with its own decision analysis.

### 2.3 MASE is unstable for near-deterministic SKUs

MASE divides by the SKU's in-sample naive MAE. If that scale is below **0.01**, the ratio explodes.
Two h=1 observations are excluded, and the same exclusion is applied to the bulk sensitivity. The
exclusion count is reported (`n_mase_excluded`).

**Consequence.** MASE is not defined for perfectly smooth series, and the ranking of models on very
low-volume SKUs is sensitive to the denominator. This is why MAE, RMSE, pinball loss, coverage and
aggregate WAPE are reported alongside it, and why the decision analysis is the primary test.

**What would fix it.** A different scaled error (e.g. RMSSE with a pooled denominator) or an
explicit hurdle model for deterministic series. The exclusion is a transparent compromise.

### 2.4 Hyperparameters were fixed a priori, not tuned

LightGBM uses one configuration (400 trees, learning rate 0.04, 31 leaves,
`min_child_samples=50`, subsample/colsample 0.8, L2 1.0). The Poisson GLM uses `alpha=0.5`. SES
searches a five-point α grid; Croston uses α=0.1. No hyperparameter is tuned on the test window,
and there is no nested cross-validation.

**Consequence.** The ML models may be under- or over-regularised for this dataset. The comparison
is honest — every model gets the same origins and rows — but it is a comparison of **these
configurations**, not of model classes at their best. LightGBM's loss to SES at the central
decision cell could shrink with tuning; it could also worsen. The project does not claim
"LightGBM cannot beat SES".

**What would fix it.** Nested rolling-origin tuning, with the inner loop strictly before each
origin. That is a larger compute budget and a separate study.

### 2.5 Calibration is marginal and window-local

The conformal layer uses a **13-week** window immediately before each origin, split by demand class
(Mondrian). If a class has fewer than 30 calibration rows, it falls back to the pooled grid. The
grid is per horizon (and separately per cumulative L for decisions), not a full joint predictive
distribution over all horizons.

**Consequence.** Coverage is conditional on the demand class, not on the individual SKU, price,
season or inventory state. Exchangeability is assumed within the 13-week window; a structural
break inside that window invalidates the guarantee. The window is short, so the empirical
quantiles are noisy, especially in the tail (q=0.95) and for small classes. Coverage at longer
horizons is slightly conservative (h=2: 0.788–0.870; h=4: 0.806–0.908), consistent with a marginal
grid widening while the zero atom stays fixed.

**What would fix it.** Adaptive conformal inference (ACI) or a rolling recalibration with a longer
window; SKU-level partial pooling; a copula or bootstrap over the joint path for multi-horizon
decisions.

### 2.6 Accuracy is reported at h = 1, 2, 4; decisions use L = 2, 4, 8

The charter asks for 1, 2 and 4 weeks. The decision analysis adds L=8, whose cumulative quantiles
rely on h=5–8 forecasts that are **not** in the accuracy tables. The cumulative conformal grid is
built for L=2, 4, 8 and is not the sum of marginal quantiles.

**Consequence.** The L=8 decision result (the largest saving, **47.80%**) rests on forecast
horizons with less accuracy evidence than the central case. It is a legitimate grid point, but it
should not be read as equally well validated as L=4.

**What would fix it.** Report accuracy at h=8, or restrict the decision claim to L≤4 until the
longer horizons are evaluated.

### 2.7 No CRPS, no formal model-comparison test

The charter says "CRPS if feasible". It is not computed. The project reports pinball loss at
P10/P50/P90 and 80% coverage instead, plus a calibration curve. There is no paired test of
predictive accuracy and no correction for multiple comparisons across seven models and three
horizons.

**Consequence.** Probabilistic skill is summarised by a few quantiles, not by a proper score over
the whole distribution. Differences of a few percent in MASE or pinball should not be treated as
significant.

**What would fix it.** A CRPS implementation (or `properscoring`) and a Diebold–Mariano test on
per-observation losses.

---

## 3. Decision-model limitations

### 3.1 One order for L weeks, no dynamic inventory

The newsvendor orders once to cover cumulative demand over `L` weeks. There is no weekly
carry-over, no intermediate replenishment, no reorder point, no safety stock review, no capacity
constraint, no MOQ, no shelf life, no quantity discount, no salvage, and no supply uncertainty.
Lead time is fixed and known.

**Consequence.** This is a **single-period** approximation of a multi-period inventory problem. It
isolates the value of the predictive distribution, which is the research question, but it
overstates the simplicity of a real replenishment policy. A weekly (s, S) or base-stock simulation
with carry-over would give different absolute costs and possibly a different ranking.

**What would fix it.** A discrete-event or dynamic-programming inventory simulation over the same
forecasts, with a stated review period and holding/stockout accounting. That is the natural next
iteration.

### 3.2 The savings baseline is chosen on the evaluation cells

Savings are measured against the **best simple point rule** among historical mean, seasonal naive,
SES and Croston/SBA, at the same `L` and ratio. In all **168** model × policy × L × ratio cells that
baseline is SES point. Choosing the best baseline on the same cells makes the comparison
conservative for the probabilistic policy (the baseline is as strong as it can be), but it also
uses evaluation information to select the baseline. The baseline is not a pre-registered incumbent
policy.

**Consequence.** "23.88% versus the best simple point rule (SES)" is not "23.88% versus the company's current
policy". A real incumbent could be better or worse. A stronger post-hoc comparator is recorded as
well: the best point model across all seven models at the central case is LightGBM Tweedie point,
and the SES quantile saves **18.53%** against it. The oracle (order realised demand, zero cost) is a
bound, not an achievable policy, and is labelled as such.

**What would fix it.** Pre-register one incumbent (e.g. SES point) or evaluate against the actual
replenishment policy, which the dataset does not contain.

### 3.3 Bootstrap intervals are cross-sectional only

The 95% CI is a 500-rep bootstrap over SKUs of the per-SKU cost difference. It answers: "if we
resampled this SKU population, how stable is the mean saving?" It does **not** answer: "how stable
is the saving across time, cost assumptions or model specifications?"

**Consequence.** The intervals are narrower than a full uncertainty statement. Cost assumptions are
swept rather than bootstrapped, and the model is fixed. A CI that excludes zero is not evidence
that the policy will save money in a new year.

**What would fix it.** A block bootstrap over origins/time, a Bayesian or bootstrap distribution
over costs, and model averaging.

### 3.4 The focal worst-SKU analysis uses the median model

`worst_skus.csv` and `fig7` use the **LightGBM-median quantile** policy at `L=4`, `Cu/Co=9`. That
is not the best policy (SES quantile is), and the median model is deliberately biased low. The list
therefore illustrates failure modes of a plausible but suboptimal policy; it is not a list of SKUs
that the recommended policy handles badly.

**Consequence.** The 16.85% cost concentration and the individual SKU costs are descriptive of that
cell. They should not be generalised to the SES policy without recomputation.

**What would fix it.** Produce the worst-SKU table for the recommended policy as well, and compare.

### 3.5 The decision table is not trimmed for bulk orders

The accuracy sensitivity removes extreme weeks; the decision table does not. A single bulk week
inside an L-week window can dominate that SKU-decision's cost and therefore the saving.

**Consequence.** The headline savings may be sensitive to a small number of bulk decisions. The
direction (quantiles help when `Cu/Co` is high) is supported by the accuracy-side robustness and by
the monotone pattern across the cost-ratio grid, but the magnitude is not robust to trimming.

**What would fix it.** Rerun `decision.py` on a trimmed demand panel and report both. This is
explicitly listed as next work, not silently omitted.

### 3.6 Service metrics are in-sample realisations, not guarantees

Fill rate and cycle service level are computed on the same 9,558 SKU-decisions used to compare
costs. They are descriptive. `cycle_service_level = P(y_L ≤ Q)` is an empirical frequency, not a
model-based guarantee, and it is not the same as the critical ratio.

**Consequence.** A policy can show a high fill rate and still be expensive, or a low fill rate and
cheap, depending on the cost ratio. The cost is the decision metric; service is a diagnostic.

**What would fix it.** A service-constrained optimisation (minimise cost subject to fill ≥ target)
and out-of-sample service monitoring.

---

## 4. Engineering and reproducibility limitations

- **One platform.** The reported run is Windows 10, Python 3.11.9. `requirements.txt` pins the
  libraries, but LightGBM and BLAS may produce last-bit differences across CPUs. The pipeline is
  deterministic *on the same stack*; it is not bit-reproducible across arbitrary hardware.
- **External download.** `download_data.py` depends on the UCI URL remaining available. The zip and
  workbook sha256 are hard-coded and verified, so a changed file fails loudly rather than silently
  training on different data. The raw workbook is git-ignored; a fresh clone needs network access
  (or the workbook placed in `data/raw/`).
- **Runtime.** A full clean + backtest + decision + figures run takes roughly two minutes on the
  development machine. That is short enough for a portfolio, but it is not a production
  retraining pipeline.
- **Tests are invariants, not a numerical golden master.** The 13 tests check the calendar spine,
  the audit identity, leakage by perturbing every post-origin week, monotone quantiles, accounting
  identities, the `results.json` contract and the presence of docs/figures. They do **not** assert
  every headline number, so a model change can move the metrics without failing the suite. The
  README numbers are checked against `results.json` by hand at release.
- **Figures use test-window information for display.** `fig2` selects representative fast/medium/
  slow SKUs by median test-window demand, and `fig7` selects the worst SKUs by realised cost. These
  are descriptive plots, not ex-ante selection rules; no reported metric depends on them.
- **No monitoring or deployment.** There is no drift detection, retraining schedule, model
  registry, approval workflow, API, or ERP integration. This is an analytical prototype.

---

## 5. What would change the conclusion

The result is conditional, and it is worth stating the conditions under which it would fail:

1. **Costs near symmetric.** For the SES quantile policy at `Cu/Co = 1` (critical ratio 0.5) the
   saving is at most **4.22%** and falls to **1.65%** at L=8, so the recommended policy's case is
   weak near symmetric costs. The grid's largest ratio-1 saving is **14.21%** (LightGBM median,
   L=4), but that model is only **1.6%** at ratio 3; the symmetric-cost gain is not stable. If the
   true ratio is near 1, the evidence for a robust probabilistic gain is weak.
2. **True lost sales.** If stockouts are common and demand is heavily censored, observed zeros
   understate demand and the evaluated stockout cost understates the true cost. The ranking of
   policies could change once latent demand is modelled.
3. **A strong incumbent.** The baseline is the best of four simple point rules *on these cells*. A
   tuned production policy, or one with a service-level constraint, could close the gap.
4. **Aggregate-volume planning.** LightGBM median has the best MASE but a pooled WAPE of **40.05%**;
   a business that plans total volume would reject it. With hierarchical reconciliation, the
   decision ranking could change.
5. **Longer horizons.** The largest saving (L=8, **47.80%**) relies on h=5–8, which are not in the
   accuracy tables. Restricting to L≤4 leaves the central **23.88%** result intact but removes the
   headline maximum.
6. **A different year or retailer.** Three origins in one 26-week window are not evidence of
   generalisation. A replication on a second dataset is the only real test.

---

## 6. Summary

| Limitation | Consequence for the reported result | Cheapest credible fix |
|---|---|---|
| No stockout data | Zeros are sales, not demand; stockout cost is a lower bound | Inventory snapshots / censored-demand model |
| Costs and lead times assumed | Absolute costs are not a P&L; only the sweep is evidence | Finance-owned costs, lead-time distribution |
| 104 weeks, one seasonal cycle | Seasonal naive is weak; seasonality not identified | More years of history |
| Bulk orders | Absolute errors and worst-SKU list dominated; decision side untrimmed | Decision-side trimmed rerun; heavy-tail mixture |
| Three adjacent origins | CIs are cross-sectional, not temporal; no significance test | More origins, block bootstrap, DM test |
| ≥12 positive weeks required | Result excludes the newest/rarest SKUs | Cold-start model for 8–11-week SKUs |
| Fixed hyperparameters | Comparison is of configurations, not model classes at their best | Nested rolling-origin tuning |
| 13-week marginal conformal | Coverage is class-conditional, not SKU-conditional; noisy tails | ACI / longer window / partial pooling |
| L=8 not in accuracy tables | The 47.80% headline has weaker accuracy support than L=4 | Report h=8 accuracy |
| One order per L, no carry-over | Single-period approximation of a multi-period problem | Inventory simulation |
| Baseline chosen on evaluation cells | Not a comparison with the real incumbent policy | Pre-register one incumbent |
| Bootstrap over SKUs only | CIs narrower than full uncertainty | Block bootstrap over time |
| Focal worst list uses median model | Illustrates a suboptimal policy, not the recommended one | Recompute for SES quantile |
| No decision-side trim | Magnitude may be bulk-sensitive | Trimmed decision rerun |
| No monitoring/deployment | Prototype, not a production policy | Monitoring + retraining + integration |

These are the boundaries of the claim. Within them, the result is reproducible from
`python src/run_all.py`, and every number is traceable to `outputs/results.json`.
