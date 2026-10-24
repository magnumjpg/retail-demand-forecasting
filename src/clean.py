"""Clean raw transactions and build a complete SKU x ISO-week demand panel.

    python src/clean.py [--force]

Outputs
-------
data/processed/weekly_demand.csv   SKU x week panel with an explicit calendar
                                   spine (missing weeks are zero demand)
data/processed/clean_stats.json    row/revenue audit trail of every filter
data/processed/sku_selection.json  modelling sample, chosen on pre-origin data

Cleaning rules follow the charter (§3.1). Cancellations are *separated*, never
netted. Every step is logged with rows, units and revenue discarded.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import (DATA_PROC, MIN_POS_WEEKS_MODEL, MIN_POS_WEEKS_UNIVERSE,
                    NON_PRODUCT_CODES, NON_PRODUCT_PREFIXES, ORIGIN_OFFSETS,
                    POS_WEEK_SENSITIVITY, PRODUCT_RE, RAW_XLSX, SHEETS,
                    TEST_WEEKS, WEEKLY)

CACHE_RAW = DATA_PROC / "transactions_raw.parquet"
STATS = DATA_PROC / "clean_stats.json"
SELECTION = DATA_PROC / "sku_selection.json"


# --------------------------------------------------------------------- load
def load_raw(force: bool = False) -> pd.DataFrame:
    if CACHE_RAW.exists() and not force:
        df = pd.read_parquet(CACHE_RAW)
        print(f"[clean] raw cache {CACHE_RAW.name}: {len(df):,} rows")
        return df
    if not RAW_XLSX.exists():
        raise FileNotFoundError(
            f"{RAW_XLSX} not found - run `python src/download_data.py` first")
    parts = []
    for sheet in SHEETS:
        part = pd.read_excel(RAW_XLSX, sheet_name=sheet)
        part["source_sheet"] = sheet
        parts.append(part)
        print(f"[clean] read {sheet}: {len(part):,} rows")
    df = pd.concat(parts, ignore_index=True)
    # mixed int/str id columns break parquet casting; normalise to text here
    df["Invoice"] = df["Invoice"].astype(str)
    df["StockCode"] = df["StockCode"].astype(str)
    df["Description"] = df["Description"].where(
        df["Description"].isna(), df["Description"].astype(str))
    df.to_parquet(CACHE_RAW, index=False)
    return df


def normalise_code(s: pd.Series) -> pd.Series:
    out = s.astype(str).str.strip().str.upper()
    # numeric codes read back as floats ("85048.0") or with trailing spaces
    out = out.str.replace(r"\.0$", "", regex=True)
    return out.replace({"NAN": np.nan, "NONE": np.nan, "": np.nan})


# --------------------------------------------------------------------- main
def clean(force: bool = False) -> dict:
    raw = load_raw(force=force)
    audit: list[dict] = []

    def log(step: str, before: pd.DataFrame, after: pd.DataFrame, note: str = ""):
        dropped = before.index.difference(after.index)
        audit.append({
            "step": step,
            "rows_before": int(len(before)),
            "rows_dropped": int(len(dropped)),
            "pct_rows_dropped": float(len(dropped) / len(before) * 100),
            "units_dropped": float(before.loc[dropped, "Quantity"].sum()),
            "revenue_dropped": float(before.loc[dropped, "_rev"].sum()),
            "note": note,
        })

    raw["_rev"] = raw["Quantity"] * raw["Price"]
    raw["StockCode"] = normalise_code(raw["StockCode"])
    raw["Invoice"] = raw["Invoice"].astype(str).str.strip()
    gross_positive = float(raw.loc[raw["_rev"] > 0, "_rev"].sum())

    cur = raw
    # 1. duplicate rows: the two sheets overlap 2010-12-01..2010-12-09
    dedup = cur.drop_duplicates(subset=[c for c in raw.columns
                                        if c not in ("_rev", "source_sheet")])
    log("duplicate_rows_sheet_overlap", cur, dedup,
        "identical rows present in both yearly sheets")
    cur = dedup

    # 2. missing identifiers
    nxt = cur.dropna(subset=["StockCode", "InvoiceDate"])
    log("missing_stockcode_or_invoicedate", cur, nxt)
    cur = nxt

    # 3. cancellations ("C..." invoices) - separated, not netted
    is_cancel = cur["Invoice"].str.startswith("C")
    cancel = cur.loc[is_cancel]
    nxt = cur.loc[~is_cancel]
    log("cancellations_C_invoice", cur, nxt,
        "returns kept aside; see cancellation_summary")
    cur = nxt

    # 4. non-product stock codes
    code = cur["StockCode"]
    is_nonprod = (
        code.isin(NON_PRODUCT_CODES)
        | code.str.startswith(tuple(NON_PRODUCT_PREFIXES))
        | ~code.str.match(PRODUCT_RE)
        | cur["Description"].isna()
        | (cur["Description"].astype(str).str.strip() == "")
    )
    nxt = cur.loc[~is_nonprod]
    log("non_product_codes", cur, nxt,
        "postage/manual/adjustment/DCGS/GIFT/TEST + codes outside ^[0-9]{4,6}[A-Z]{0,2}$")
    cur = nxt

    # 5. non-positive quantity
    nxt = cur.loc[cur["Quantity"] > 0]
    log("quantity_le_zero", cur, nxt,
        "negative lines on non-C invoices are manual adjustments")
    cur = nxt

    # 6. non-positive price
    nxt = cur.loc[cur["Price"] > 0]
    log("price_le_zero", cur, nxt,
        "zero/negative price lines carry no revenue information")
    cur = nxt

    # ------------------------------------------------------------- weekly
    df = cur.copy()
    # Weeks are ISO weeks (Monday start). The first and the last calendar week
    # are partial (data starts Tue 2009-12-01, ends Fri 2011-12-09), so both are
    # dropped rather than modelled as if they were full weeks.
    df["week"] = df["InvoiceDate"].dt.to_period("W-SUN").dt.start_time
    weeks_all = pd.date_range(df["week"].min(), df["week"].max(), freq="7D")
    partial = [weeks_all[0], weeks_all[-1]]
    before_partial = df
    df = df.loc[~df["week"].isin(partial)].copy()
    log("partial_calendar_weeks", before_partial, df,
        "first and last calendar weeks are incomplete and excluded")
    weeks = pd.DatetimeIndex(
        pd.date_range(df["week"].min(), df["week"].max(), freq="7D"))
    assert len(weeks) == len(weeks_all) - 2

    for a in audit:
        a["pct_gross_revenue_dropped"] = round(a["revenue_dropped"] / gross_positive * 100, 4)

    df["is_uk"] = df["Country"].eq("United Kingdom")
    df["_rev_uk"] = df["_rev"].where(df["is_uk"], 0.0)
    g = df.groupby(["StockCode", "week"], observed=True)
    agg = pd.DataFrame({
        "demand": g["Quantity"].sum(),
        "revenue": g["_rev"].sum(),
        "n_invoices": g["Invoice"].nunique(),
        "n_customers": g["Customer ID"].nunique(),
        "uk_revenue": g["_rev_uk"].sum(),
    }).reset_index()

    desc = (df.dropna(subset=["Description"])
              .assign(Description=lambda x: x["Description"].astype(str).str.strip())
              .groupby("StockCode")["Description"]
              .agg(lambda s: s.mode().iat[0] if len(s.mode()) else s.iat[0]))
    pos_weeks = (agg.loc[agg["demand"] > 0]
                   .groupby("StockCode").size().rename("n_pos_weeks"))
    totals = agg.groupby("StockCode").agg(total_demand=("demand", "sum"),
                                          total_revenue=("revenue", "sum"))
    med_price = (agg.loc[agg["demand"] > 0]
                    .assign(p=lambda x: x["revenue"] / x["demand"])
                    .groupby("StockCode")["p"].median().rename("median_price"))
    meta = (pd.concat([pos_weeks, totals, med_price], axis=1)
              .join(desc.rename("description")))
    meta["n_pos_weeks"] = meta["n_pos_weeks"].fillna(0).astype(int)

    universe = meta.index[meta["n_pos_weeks"] >= MIN_POS_WEEKS_UNIVERSE]
    spine = pd.MultiIndex.from_product([universe, weeks],
                                       names=["StockCode", "week"])
    panel = (agg.set_index(["StockCode", "week"])
                .reindex(spine)
                .reset_index())
    for col in ("demand", "revenue", "n_invoices", "n_customers", "uk_revenue"):
        panel[col] = panel[col].fillna(0.0)
    panel["price_mean"] = np.where(panel["demand"] > 0,
                                   panel["revenue"] / panel["demand"].replace(0, np.nan),
                                   np.nan)
    panel = panel.merge(meta.reset_index(), on="StockCode", how="left")
    panel = panel.sort_values(["StockCode", "week"]).reset_index(drop=True)

    # -------------------------------------------------- structural facts
    clean_inv = df.groupby("Invoice").agg(units=("Quantity", "sum"),
                                          revenue=("_rev", "sum"))
    clean_inv = clean_inv.sort_values("units", ascending=False)
    k1 = max(1, int(np.ceil(len(clean_inv) * 0.01)))
    k05 = max(1, int(np.ceil(len(clean_inv) * 0.005)))
    all_inv = raw.groupby("Invoice")["Quantity"].sum().sort_values(ascending=False)

    neg_noncancel = int(audit[4]["rows_dropped"])  # Quantity<=0 after cancellation split

    n_weeks = len(weeks)
    origin0 = n_weeks + ORIGIN_OFFSETS[0]
    wide = (panel.pivot(index="StockCode", columns="week", values="demand")
                 .reindex(columns=weeks).fillna(0.0))
    pos_full = (wide > 0).sum(axis=1)
    pos_train = (wide.iloc[:, :origin0] > 0).sum(axis=1)
    model_skus = pos_train.index[pos_train >= MIN_POS_WEEKS_MODEL].tolist()

    facts = {
        "raw_rows_two_sheets": int(len(raw)),
        "raw_sheets": SHEETS,
        "duplicate_rows_removed": int(audit[0]["rows_dropped"]),
        "rows_after_cleaning": int(len(df)),
        "pct_rows_kept": round(len(df) / len(raw) * 100, 4),
        "gross_positive_revenue_raw": round(gross_positive, 2),
        "revenue_after_cleaning": round(float(df["_rev"].sum()), 2),
        "units_after_cleaning": int(df["Quantity"].sum()),
        "skus_after_cleaning": int(df["StockCode"].nunique()),
        "cancellations": {
            "rows": int(is_cancel.sum()),
            "pct_rows": round(float(is_cancel.mean() * 100), 4),
            "units": float(cancel["Quantity"].sum()),
            "revenue": round(float(cancel["_rev"].sum()), 2),
        },
        "negative_quantity_non_cancellation_rows": neg_noncancel,
        "weeks_total": int(len(weeks)),
        "weeks_partial_dropped": [str(p.date()) for p in partial],
        "weeks_first": str(weeks[0].date()),
        "weeks_last": str(weeks[-1].date()),
        "test_weeks": TEST_WEEKS,
        "first_origin_week_index": int(origin0),
        "first_origin_week": str(weeks[origin0].date()),
        "skus_by_positive_weeks_full_sample": {
            str(n): int((pos_full >= n).sum()) for n in POS_WEEK_SENSITIVITY},
        "skus_by_positive_weeks_pre_origin": {
            str(n): int((pos_train >= n).sum()) for n in POS_WEEK_SENSITIVITY},
        "universe_skus_min_pos_weeks": MIN_POS_WEEKS_UNIVERSE,
        "model_skus_min_pos_weeks": MIN_POS_WEEKS_MODEL,
        "n_model_skus": len(model_skus),
        "panel_rows": int(len(panel)),
        "zero_demand_share_universe": round(float((panel["demand"] == 0).mean()), 4),
        "zero_demand_share_model_skus": round(
            float((panel.loc[panel["StockCode"].isin(model_skus), "demand"] == 0).mean()), 4),
        "median_positive_weeks_per_sku": float(meta.loc[universe, "n_pos_weeks"].median()),
        "bulk_orders": {
            "n_invoices_clean": int(len(clean_inv)),
            "top_1pct_invoice_share_units": round(float(clean_inv["units"].head(k1).sum() / clean_inv["units"].sum()), 4),
            "top_0_5pct_invoice_share_units": round(float(clean_inv["units"].head(k05).sum() / clean_inv["units"].sum()), 4),
            "top_1pct_invoice_share_revenue": round(float(clean_inv["revenue"].head(k1).sum() / clean_inv["revenue"].sum()), 4),
            "largest_invoice_units": float(clean_inv["units"].iat[0]),
            "top_1pct_invoice_share_units_incl_cancellations": round(float(all_inv.head(max(1, int(np.ceil(len(all_inv) * 0.01)))).sum() / all_inv.sum()), 4),
        },
        "share_orders_without_customer_id": round(float(raw["Customer ID"].isna().mean()), 4),
        "audit": audit,
        "assumptions": [
            "Weeks are ISO weeks (Mon-Sun). The first and last calendar weeks are partial and dropped.",
            "Cancellations are returns, not negative demand; they are excluded, never netted.",
            "Demand is units shipped (positive Quantity), not revenue.",
            "Zeros in the spine are observed no-sales; with no stockout data they may hide lost demand.",
        ],
    }

    panel.to_csv(WEEKLY, index=False, date_format="%Y-%m-%d")
    STATS.write_text(json.dumps(facts, indent=2), encoding="utf-8")
    SELECTION.write_text(json.dumps({
        "min_pos_weeks_model": MIN_POS_WEEKS_MODEL,
        "window_weeks": [0, origin0 - 1],
        "window_dates": [str(weeks[0].date()), str(weeks[origin0 - 1].date())],
        "skus": model_skus,
    }, indent=2), encoding="utf-8")

    print(f"[clean] {facts['raw_rows_two_sheets']:,} raw rows -> "
          f"{facts['rows_after_cleaning']:,} clean rows "
          f"({facts['pct_rows_kept']}%)")
    print(f"[clean] {facts['weeks_total']} weeks, "
          f"{facts['skus_after_cleaning']} SKUs, universe "
          f"{len(universe)} SKUs, model sample {len(model_skus)} SKUs")
    print(f"[clean] zero-demand share {facts['zero_demand_share_model_skus']:.1%} "
          f"(model sample); top 1% invoices = "
          f"{facts['bulk_orders']['top_1pct_invoice_share_units']:.1%} of units")
    print(f"[clean] -> {WEEKLY}")
    return facts


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true", help="re-read the workbook")
    clean(force=ap.parse_args().force)
