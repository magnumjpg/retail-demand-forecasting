# Red team

The project attacked as if by a hostile referee. Each objection, the strongest
version of it, what the project does about it, and what remains unresolved.

---

## "Zeros are not demand, so the whole target is wrong."

**Conceded, with consequences.** The data record sales, not demand. A zero SKU-week
may be no customer, no stock, a delisted item or a data gap, and there is no
inventory file to separate them.

Every model is therefore fitted to observed sales, and a stockout looks like low
demand. The fill rate in `decision.csv` is `1 − under / recorded demand`, not a true
service level. The stockout cost is computed against recorded sales, so it is a
**lower bound** on the cost of being short if stockouts existed.

**What would fix it:** inventory or stockout records, or a latent-demand model with
an observation mechanism. Neither is in the public dataset. This is stated in
README §7 and LIMITATIONS §1.1; it is not a footnote.

---

## "One retailer, two years, three adjacent origins. This is a case study, not evidence."

**Correct.** There are 104 complete weeks. The three origins are in adjacent 8-week
blocks, so they are not independent draws from different regimes. The bootstrap is
over SKUs, not time, and there is no Diebold–Mariano test.

The claim is therefore about **this retailer in this window**, not about retail. The
README says "a reproducible analytical prototype on one public dataset", and the
limitations section says results are not evidence of out-of-sample performance on
another retailer or another year.

**What would fix it:** more history, more origins, a block bootstrap over time, and
paired forecast tests. None of that turns two years into a business cycle.

---

## "The cost model is invented, so the savings number is invented."

**Half conceded.** Holding cost = 1% of unit price per week and `Cu/Co ∈ {1,3,9,19}`
are assumptions. The absolute cost level and the optimal policy depend on them.

The response is not to defend one number but to **sweep the assumptions**. Every
cell of the 3 lead times × 4 ratios grid is in `decision.csv`. The honest statement
is conditional: *if* holding costs are around 1% per week and the underage/overage
ratio is 9, then the SES quantile policy saves 23.82% versus SES point and 18.62%
versus the cheapest point rule in that cell. Change the ratio to 1 and the same policy loses.

No margin, stockout penalty, salvage value, MOQ, capacity constraint or lead-time
variability is observed. A real implementation would change the answer.

---

## "You chose the central case after seeing the results."

**Fair challenge, partly answerable.** The grid was fixed before estimation (lead
times 2/4/8; ratios 1/3/9/19), and `L=4, Cu/Co=9` is the middle of the grid, not the
best cell. The best cell is `L=8, Cu/Co=19`, and it is reported as such.

But the choice to *foreground* the central case is a presentation judgement, and
the ratio-9 cell happens to be favourable. Worse, **23.82% is the best of the seven
quantile models inside that cell**, chosen after seeing the table. `results.json`
now says so explicitly (`headline_selection.selected_ex_post = true`) and reports
the 12-cell range: 2.43–48.81%, median 17.50%. The honest defence is that the full
grid is shown, including the cells where the policy loses. A reader who wants the
unfavourable corner can see it in the same table.

**Unresolved:** no pre-registration. The charter fixed the question, not a single
primary specification. The 95% CI is conditional on the selected cell and does not
cover model or cell selection.

---

## "Your MASE winner contradicts your decision claim. Which metric is right?"

**Both, for different questions.** LightGBM median has the best per-SKU MASE
(0.686) and the second-worst pooled WAPE (40.05%, better only than seasonal naive's
71.09%), and it destroys value as an ordering rule (−39.01% versus SES point at the
central case).

MASE is computed SKU by SKU and rewards getting the middle of a zero-inflated
distribution right. Pooled WAPE and the newsvendor care about total volume, and a
median forecast under-orders by construction. Reporting only MASE would hide the
bias; reporting only WAPE would hide the per-SKU accuracy. Both are in the README.

**The lesson is not "MASE is bad".** It is that a point-accuracy metric is not a
decision metric, and a portfolio that stops at MASE would recommend the wrong
policy.

---

## "SES wins because it is smooth and the aggregate is smooth. That is an artefact."

**Partly true, and it is the interesting part.** SES tracks aggregate volume to
7.58% WAPE; the ML models are better per SKU and worse in aggregate. If the business
cares about total units and total cost — which the newsvendor does — SES is the
better *ordering* input even though it is the worse *forecasting* model.

This is not a claim that SES is generally superior. It is a claim about this data,
this cost model and this evaluation. A production system with hierarchical
reconciliation might recover the ML advantage, which is exactly the next step in
LIMITATIONS §5, item 4.

---

## "The 18.62% versus the cheapest point rule is itself selected on the same cells."

**Conceded.** `best_point_model_all` is chosen by looking at the cheapest point
policy in the evaluation table. That makes it a conservative comparator on this
sample, but it is not a pre-registered incumbent, and the selection uses test
information. Its identity is not stable: LightGBM median at `Cu/Co = 1`,
LightGBM Tweedie above it, which is why the docs do not name a single model.

The same caveat applies to the SES baseline: "best simple point rule" is chosen on
the evaluation cells. The correct reading is "versus the best point rule on these
cells", not "versus the policy the company actually uses".

---

## "Bulk orders drive everything."

**Largely true, and quantified.** Evaluation weeks above each SKU's training p99.5
are 1.34% of rows but 14.86% of units. Excluding them improves every model's MASE
(SES 0.819 → 0.624; LightGBM median 0.686 → 0.474) and leaves the ranking unchanged.

But the **decision side was not re-estimated**. Dropping decisions that contain a
bulk week changes the evaluated SKU set and the cost base, so a like-for-like
saving is not available without a separate study. This is a real gap: the cost
result may be more tail-dependent than the accuracy result.

**What would fix it:** a decision-side trim and a multi-period inventory simulation
with carry-over. Both are listed as future work, not as done.

---

## "L=8 decisions use horizons you never evaluate for accuracy."

**Correct.** Accuracy is reported at h=1, 2, 4. Decisions at L=8 use cumulative
demand over h=1..8, and h=5–8 are not in the accuracy tables. The conformal grid
for cumulative demand is built separately (D14), so the decision is internally
consistent, but the reader cannot see the h=8 forecast error.

This is a reporting gap, not a computational error. The L=8 result is the largest
saving in the grid (48.81% / 46.96%), so the gap matters. Adding h=8 accuracy is the
first thing to do next.

---

## "A quantile policy would not survive MOQ, capacity, or lead-time variability."

**Correct, and outside the model.** The newsvendor assumes a single order, no
minimum order quantity, unlimited capacity, deterministic lead time and no
substitution. Real replenishment has all five.

The project does not claim deployability. It claims that, *within this decision
model*, the distribution has value at high asymmetry and can lose at low asymmetry.
A production evaluation needs a multi-period simulation with those constraints;
that is LIMITATIONS §3.1.

---

## "You are reporting a 23.82% saving from a model that cannot even beat SES on aggregate error."

**The premise is wrong: the quantile policy uses SES.** The 23.82% is SES **plus the
conformal distribution layer**, compared with SES point. The point model is held
constant; the only change is ordering at a quantile instead of the mean. That is
precisely the project's claim: the distribution layer, not a fancier point model,
is what creates the value.

The same LightGBM-Tweedie mean moves from 6.39% as a point rule to 22.31% as a
quantile rule. If the result were driven by model sophistication, that gap would
not exist.

---

## "This is a decision memo with a negative result dressed up as a positive one."

**No: it has both, and it labels them.** The positive result is conditional on high
cost asymmetry. The negative results are explicit: at ratio 1 the policy loses to
the cheapest point rule; seasonal naive and Croston fail; the median model wins
MASE and loses money; the decision side is not bulk-trimmed; five horizons are not
evaluated.

A reader who takes only "23.82%" from this project has ignored half the document.
That is why the first section states the negative case in the same paragraph as the
headline.
