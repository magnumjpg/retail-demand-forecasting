# Data

## 1. Source and integrity

| Field | Value |
|---|---|
| Dataset | *Online Retail II* (Chen, 2012), UCI Machine Learning Repository |
| Landing page | <https://archive.ics.uci.edu/dataset/502/online+retail+ii> |
| Direct download | <https://archive.ics.uci.edu/static/public/502/online+retail+ii.zip> |
| Workbook | `online_retail_II.xlsx`, 45,622,278 bytes |
| Workbook sha256 | `bcbe73b35f5b7babf197fb0cb983a11f5d9ff929078d4aa53d171b1f2df2e980` |
| Expected zip sha256 | `572e36277c2390fbfde10664750731e0a86f55e33470d91919085f0408e67bfb` |
| Sheets | `Year 2009-2010` (525,461 rows), `Year 2010-2011` (541,910 rows) |
| Raw rows | **1,067,371** |
| Raw date range | 2009-12-01 → 2011-12-09 |

`src/download_data.py` downloads the zip if needed, verifies both hashes, extracts the workbook,
and writes `data/raw/manifest.json` (retrieval time, byte size, sheet names and row counts). The
workbook is git-ignored; the manifest is committed. If the workbook is present and its hash
matches, the script verifies it and reuses the manifest without re-reading 1.07 M rows.

The dataset has no documented missing-data convention beyond blank fields; all cleaning rules
below are explicit and auditable. The raw units are shipped units, not orders, reservations or
revenue.

## 2. Cleaning audit

Every filter is logged in `data/processed/clean_stats.json` with rows, units and net revenue
removed. `tests/test_pipeline.py::test_cleaning_audit_adds_up` walks the chain and checks it
reconciles to `rows_after_cleaning`.

| # | Step | Rows before | Rows dropped | % of current | Units dropped | Net revenue dropped |
|---:|---|---:|---:|---:|---:|---:|
| 1 | Duplicate rows across the two sheets | 1,067,371 | 34,335 | 3.2168% | +198,713 | +431,716.87 |
| 2 | Missing `StockCode` or `InvoiceDate` | 1,033,036 | 0 | 0% | 0 | 0.00 |
| 3 | Cancellations (`Invoice` starts with `C`, kept aside) | 1,033,036 | 19,104 | 1.8493% | −476,819 | −1,462,050.61 |
| 4 | Non-product codes | 1,013,932 | 9,002 | 0.8878% | −56,668 | +674,892.16 |
| 5 | `Quantity <= 0` on non-cancellation invoices | 1,004,930 | 755 | 0.0751% | −313,696 | 0.00 |
| 6 | `Price <= 0` | 1,004,175 | 961 | 0.0957% | +69,188 | 0.00 |
| 7 | Partial first/last calendar weeks | 1,003,214 | 31,146 | 3.1046% | +389,633 | +744,489.76 |
| | **Final clean rows** | | | | | |
| | **972,068 (91.0712% of raw)** | | | | | |

Notes:

- **Revenue is net in this table.** Cancelled rows carry negative revenue, so the column cannot
  be summed to the difference between gross and clean positive revenue. Gross positive revenue
  in the raw file is 20,972,968.14; clean positive revenue is 18,898,202.39.
- **Cancellations are never netted.** 19,104 cancellation rows (−476,819 units,
  −1,462,050.61 revenue) are excluded from demand and reported separately. Netting returns into
  sales would make a returned unit look like negative demand and would distort the week of the
  original sale.
- **Non-product codes.** Explicit list (`POST`, `D`, `M`, `C2`, `DOT`, `S`, `B`,
  `BANK CHARGES`, `PAD(S)`, `CRUK`, `AMAZONFEE`, `ADJUST`, `SP1002`), prefixes (`TEST`, `GIFT`,
  `DCGS`, `BANK`), a strict product regex `^[0-9]{4,6}[A-Z]{0,2}$`, and blank descriptions.
  9,002 rows (674,892.16 revenue) are postage, manual adjustments, samples and gifts.
- **Negative quantities on non-C invoices** (755 rows) are manual adjustments, not demand.
- **Zero/negative prices** (961 rows, 69,188 units) carry no usable revenue and are dropped.
- **Duplicate rows.** The two yearly sheets overlap 2010-12-01 → 2010-12-09; 34,335 identical
  rows are removed before any aggregation.
- **Missing identifiers** are counted after de-duplication; none remain. In the raw file,
  customer ID is missing for 22.77% of orders, but customer ID is not used as a feature.

## 3. Weekly panel

Weeks are ISO weeks (Monday–Sunday). Two partial weeks are dropped because the data starts on
Tuesday 2009-12-01 and ends on Friday 2011-12-09:

- dropped: week starting **2009-11-30** and week starting **2011-12-05**;
- retained: **104 complete weeks**, **2009-12-07 → 2011-11-28**.

The panel is a complete Cartesian product of selected SKUs and these 104 weeks, so every SKU has
exactly 104 rows and every missing week is an explicit zero. This is the hard part of
intermittent-demand modelling: `NaN` is not “no demand”, it is “no row”, and both are different
from a zero.

`data/processed/weekly_demand.csv` is 43.1 MB; 415,480 rows = 3,995 SKUs × 104 weeks.

| Column | Meaning |
|---|---|
| `StockCode` | normalised product code (string; leading zeros preserved) |
| `week` | Monday of the ISO week (`YYYY-MM-DD`) |
| `demand` | units shipped in that week, sum of positive quantities (never negative) |
| `revenue` | positive revenue from those units |
| `n_invoices` | distinct invoices in the week |
| `n_customers` | distinct customer IDs in the week |
| `uk_revenue` | revenue from `Country == "United Kingdom"` |
| `price_mean` | revenue / demand, NaN in zero weeks |
| `n_pos_weeks` | positive weeks in the full sample (constant per SKU) |
| `total_demand`, `total_revenue` | full-sample totals (constant per SKU) |
| `median_price` | median weekly unit price in positive weeks, used as cost proxy |
| `description` | modal non-blank description |

Unit price is *derived* (`revenue / demand`) and is used only as a lagged feature and as the
newsvendor cost proxy, never as a target. It is winsorised at the 99.5th percentile before the
decision analysis.

## 4. Sample construction

Two different samples are used, and the difference is deliberate:

| Filter | Full sample (panel) | Pre-origin (model sample) |
|---|---:|---:|
| ≥ 8 positive weeks | **3,995** | 3,482 |
| ≥ 12 positive weeks | 3,645 | **3,186** |
| ≥ 26 positive weeks | 2,753 | 2,286 |
| Zero-demand share | 54.52% | **46.93%** |
| Median positive weeks per SKU | 41 | — |

- The **universe** written to `weekly_demand.csv` is every SKU with ≥8 positive weeks in the full
  104-week sample. This keeps the shipped panel useful for alternative splits while excluding
  one-off codes.
- The **model sample** is every SKU with ≥12 positive weeks in the **first 77 weeks only**
  (weeks 0–76, ending 2011-05-23), frozen in `data/processed/sku_selection.json`. Selecting on
  pre-origin data prevents survivor bias from future sales; `features.load_panel()` re-computes
  the count and asserts it matches the JSON.
- SKUs with 8–11 positive pre-origin weeks (296) are in the panel but not in the model sample.
  No metric in this project describes them. Cold-start analysis covers the 880 retained SKUs
  with 12–25 positive weeks at the first origin.

Demand classes are assigned from pre-origin history only (first 77 weeks):

- **intermittent**: >50% zero weeks;
- **fast**: median weekly demand >4 units;
- **medium**: 1 < median ≤4;
- **slow**: median ≤1.

These thresholds are conventions, not estimated parameters; they are used for stratification
and for Mondrian conformal calibration.

## 5. Evaluation origins

Three rolling origins, all inside the last 26-week test window (weeks 78–103):

| Origin week | Origin index | Fit ends | Fit end week | Training rows | Conformal rows |
|---|---:|---|---|---:|---:|
| 2011-05-30 | 77 | 64 | 2011-02-28 | 535,248 | 216,648 |
| 2011-07-25 | 85 | 72 | 2011-04-25 | 637,200 | 216,648 |
| 2011-09-19 | 93 | 80 | 2011-06-20 | 739,152 | 216,648 |

- The model is fitted on weeks ≤ `origin − 13`; the 13-week conformal window is
  `(origin − 13, origin]`, and only targets with `t + h ≤ origin` are used.
- Accuracy is reported at h = 1, 2, 4; forecasts are generated for h = 1…8 so the decision
  analysis can cover lead times 2, 4 and 8.
- Each origin contributes 3,186 SKU forecasts per horizon, hence 9,558 observations per horizon.
- Origins are adjacent 8-week blocks, not independent samples. The test window is covered by
  non-overlapping h=1 targets at weeks 78, 86 and 94 plus longer horizons; it is a rolling-origin
  backtest, not a single 26-week forecast path.

## 6. Structural facts that drive the modelling

| Fact | Value |
|---|---:|
| Zero-demand SKU-weeks (model sample) | 46.93% |
| Zero-demand SKU-weeks (universe) | 54.52% |
| Top 1% of invoices' share of units | 20.46% |
| Top 0.5% of invoices' share of units | 15.63% |
| Top 1% of invoices' share of revenue | 14.22% |
| Largest invoice | 87,167 units |
| Top 1% share including cancellations | 26.01% |
| Orders without customer ID (raw) | 22.77% |
| Invoices in clean data | 38,459 |

The largest invoice is a bulk order, not a data error. It is why MASE, MAE and RMSE tell
different stories, and why the bulk sensitivity in `_backtest.json` matters. No stockout or
inventory level is recorded, so a zero cannot be distinguished from “no stock available”.

## 7. Known data-quality caveats

1. **Demand is censored.** We observe sales, not demand. A zero may mean no customer, no stock,
   or no listing. The retailer's stock levels are not in the dataset.
2. **No returns netting.** Cancellations are separated, not subtracted. A returned unit does not
   reduce the demand of the original week.
3. **Bulk orders are rare and huge.** They inflate absolute errors and can dominate a SKU's
   history. The accuracy-side sensitivity excludes evaluation weeks above the SKU's training
   p99.5; the decision side is not re-estimated (see LIMITATIONS.md).
4. **Two years, two Decembers.** The first model fit ends 2011-02-28 (65 weeks): one full
   seasonal cycle, not two. Seasonal naive has to extrapolate a single lag-52 observation, and
   the two Decembers in the data differ sharply. It is therefore a weak baseline, and we say so.
5. **Sheet overlap.** The 2010-12-01 → 2010-12-09 overlap is de-duplicated by exact row match.
   If the same invoice line were legitimately repeated on the same day with identical fields, it
   would be removed too; given the identical timestamps and descriptions, duplication is the
   overwhelmingly likely explanation.
6. **No customer linkage.** 22.77% of raw orders have no customer ID; customer-level features are
   not used, so this does not bias the model sample, but it limits any claim about customer
   behaviour.
7. **Prices are not constant.** `median_price` is a SKU-level proxy, winsorised at p99.5. Cost
   results scale with it; they are not accounting profit.

## 8. Rebuild the data

```powershell
python src/download_data.py          # verify/download; writes manifest.json
python src/clean.py --force          # re-read the workbook, rebuild the panel + audit
python src/clean.py                  # reuse data/processed/transactions_raw.parquet
```

The intermediate `transactions_raw.parquet` (7.3 MB) is a local cache, git-ignored. The audited
panel, stats and frozen SKU selection are the deliverables.
