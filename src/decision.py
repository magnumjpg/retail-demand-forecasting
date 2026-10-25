"""Newsvendor inventory decisions from the probabilistic forecasts.

    python src/decision.py

For a lead time L (weeks) the order must cover cumulative demand over
``origin+1 .. origin+L``.  Two policies are compared on identical realised
demand:

* ``point``     : order = the model's mean forecast (what you do without a
                  distribution);
* ``quantile``  : order = F^-1(critical ratio), critical ratio = Cu/(Cu+Co).

Cost assumptions (swept, never presented as "the" answer):

* Co = 1% of unit price per week held x L  (holding / obsolescence)
* Cu = (Cu/Co) x Co                        (lost margin)
* unit price = median observed price per SKU

Everything is evaluated out-of-sample on the same three rolling origins.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config as C
from features import load_panel, model_mask
from models import BASELINES, MODELS, QGRID

DECISION_JSON = C.DATA_PROC / "_decision.json"
SIMPLE = list(BASELINES)


def price_vector(p, origin: int | None = None) -> np.ndarray:
    """Unit-value proxy: median observed price per SKU, winsorised at p99.5.

    With ``origin`` set, only weeks up to that origin are used, so the cost
    vector contains no information from the evaluation window. ``origin=None``
    uses the full sample and exists only for the sensitivity stored in
    ``results.json``. The full-sample median is also the fallback for a SKU with
    no positive revenue before ``origin``; that case cannot occur on the model
    sample, whose selection requires at least 12 positive pre-origin weeks
    (asserted in ``run_decision``).
    """
    fallback = p.meta["median_price"].to_numpy(dtype=float)
    if origin is None:
        pr = fallback.copy()
    else:
        d = p.D[:, : origin + 1]
        r = p.R[:, : origin + 1]
        with np.errstate(invalid="ignore", divide="ignore"):
            px = np.where(d > 0, r / np.maximum(d, 1e-9), np.nan)
        pr = pd.DataFrame(px).median(axis=1, skipna=True).to_numpy(dtype=float)
        pr = np.where(np.isfinite(pr), pr, fallback)
    pos = pr[pr > 0]
    cap = np.quantile(pos, 0.995) if len(pos) else 1.0
    pr = np.clip(np.where(np.isfinite(pr), pr, np.median(pos)), 0.01, cap)
    return pr


def cumulative_targets(fc: pd.DataFrame, L: int) -> pd.DataFrame:
    """Cumulative demand and forecasts over weeks 1..L for each (origin, sku)."""
    d = fc[fc["horizon"] <= L]
    agg = {"y": ("y", "sum")}
    for m in MODELS:
        agg[f"mu__{m}"] = (f"mu__{m}", "sum")
    g = (d.groupby(["origin", "sku_idx", "StockCode"], as_index=False)
           .agg(**agg))
    g["demand_class"] = (fc.drop_duplicates("StockCode")
                           .set_index("StockCode")["demand_class"]
                           .reindex(g["StockCode"]).to_numpy())
    return g


def order_quantile(mu, zgrid, q):
    z = np.interp(q, QGRID, zgrid)
    return np.clip(mu + np.sqrt(np.maximum(mu, 0) + 0.25) * z, 0, None)


def evaluate(g: pd.DataFrame, Q: np.ndarray, L: int, price_rows: np.ndarray,
             ratio: float) -> dict:
    y = g["y"].to_numpy()
    co = C.HOLD_COST_WEEK * price_rows * L
    cu = ratio * co
    over = np.maximum(Q - y, 0)
    under = np.maximum(y - Q, 0)
    hold = float(np.sum(co * over))
    stock = float(np.sum(cu * under))
    total = hold + stock
    n = len(g)
    weekly = float(y.sum()) / (n * L)
    inv = float(np.mean(over))
    return {
        "total_cost": round(total, 2),
        "holding_cost": round(hold, 2),
        "stockout_cost": round(stock, 2),
        "cost_per_sku_decision": round(total / n, 3),
        "fill_rate": round(float(1 - under.sum() / max(y.sum(), 1e-9)), 4),
        "cycle_service_level": round(float(np.mean(y <= Q)), 4),
        "avg_inventory_units": round(inv, 3),
        "weeks_of_cover": round(inv / weekly, 3) if weekly > 0 else None,
        "avg_order": round(float(np.mean(Q)), 3),
        "total_demand": round(float(y.sum()), 1),
    }


def bootstrap_savings(cost_model: np.ndarray, cost_base: np.ndarray,
                      codes: np.ndarray, idx: np.ndarray) -> tuple[float, float]:
    """Percentile CI from a bootstrap that resamples *SKUs*, not rows.

    Each SKU appears at three origins. Resampling rows would treat those three
    correlated observations as independent and understate the interval. Costs
    are therefore summed per SKU first, and whole SKUs are drawn with
    replacement.
    """
    n_sku = int(codes.max()) + 1
    cm = np.bincount(codes, weights=cost_model, minlength=n_sku)
    cb = np.bincount(codes, weights=cost_base, minlength=n_sku)
    assert n_sku == idx.shape[1], (
        f"bootstrap index has {idx.shape[1]} SKUs but cost arrays have {n_sku}; "
        "codes must be factorised to 0..n_sku-1")
    s = 1 - cm[idx].sum(axis=1) / np.maximum(cb[idx].sum(axis=1), 1e-9)
    return float(np.percentile(s, 2.5) * 100), float(np.percentile(s, 97.5) * 100)


def run_decision(save: bool = True) -> dict:
    p = load_panel()
    fc = pd.read_parquet(C.FORECASTS)
    grids = pd.read_csv(C.RESIDUALS)
    mask = model_mask(p)
    origin_list = sorted(fc["origin"].unique().tolist())
    # Every model SKU has >= 12 positive weeks before the first origin, so its
    # pre-origin price is observed and the full-sample fallback never binds.
    for o in origin_list:
        assert (p.D[mask, : int(o) + 1] > 0).any(axis=1).all(), (
            f"a model SKU has no pre-origin price at origin {o}")
    prices = {int(o): price_vector(p, int(o)) for o in origin_list}
    price_full = price_vector(p)          # sensitivity only
    rng = np.random.default_rng(C.BOOTSTRAP_SEED)

    rows = []
    boot_rows = []
    worst = None
    focal = None

    for L in C.LEAD_TIMES:
        g = cumulative_targets(fc, L)
        g = g[g["StockCode"].isin(p.model_skus)].reset_index(drop=True)
        y = g["y"].to_numpy()
        n_sku = int(g["sku_idx"].nunique())
        # factorise to a compact 0..n_sku-1 index: the global sku_idx runs to
        # ~4,000 because it covers the whole universe, so bincount on it would
        # allocate empty slots and the bootstrap draw would miss model SKUs
        codes = pd.factorize(g["sku_idx"], sort=True)[0]
        price_rows = np.empty(len(g), dtype=float)
        for o, pv in prices.items():
            sel = (g["origin"] == o).to_numpy()
            price_rows[sel] = pv[g.loc[sel, "sku_idx"].to_numpy()]
        price_rows_full = np.empty(len(g), dtype=float)
        for o in origin_list:
            sel = (g["origin"] == o).to_numpy()
            price_rows_full[sel] = price_full[g.loc[sel, "sku_idx"].to_numpy()]
        per_sku = {}
        eval_rows = {}

        for model in MODELS:
            gg = grids[(grids["kind"] == "aggregate") & (grids["horizon"] == L) &
                       (grids["model"] == model)]
            mu = g[f"mu__{model}"].to_numpy()
            for policy in ("point", "quantile"):
                for ratio in C.COST_RATIOS:
                    if policy == "point":
                        Q = mu
                    else:
                        cr = ratio / (1 + ratio)
                        z = np.zeros(len(g))
                        for c in C.CLASSES:
                            for o in g["origin"].unique():
                                sel = ((g["demand_class"] == c) &
                                       (g["origin"] == o)).to_numpy()
                                if not sel.any():
                                    continue
                                gz = gg[(gg["demand_class"] == c) &
                                        (gg["origin"] == o)].sort_values("q")
                                if len(gz) != len(QGRID):
                                    gz = gg[(gg["demand_class"] == C.CLASSES[0]) &
                                            (gg["origin"] == o)].sort_values("q")
                                z[sel] = np.interp(cr, QGRID, gz["z"].to_numpy())
                        Q = np.clip(mu + np.sqrt(np.maximum(mu, 0) + 0.25) * z, 0, None)
                    key = (model, policy, ratio)
                    res = evaluate(g, Q, L, price_rows, ratio)
                    res.update({"lead_time": L, "model": model, "policy": policy,
                                "cost_ratio": ratio})
                    rows.append(res)
                    eval_rows[key] = (Q, res["total_cost"])
                    if L == 4 and model == "lgbm_quantile" and policy == "quantile" \
                            and ratio == 9:
                        focal_Q = Q
                    co = C.HOLD_COST_WEEK * price_rows * L
                    cu = ratio * co
                    per_sku[key] = (co * np.maximum(Q - y, 0) + cu * np.maximum(y - Q, 0))

        # oracle: order exactly the realised L-week demand
        res = evaluate(g, g["y"].to_numpy(), L, price_rows, 1.0)
        res.update({"lead_time": L, "model": "oracle", "policy": "clairvoyant",
                    "cost_ratio": np.nan})
        rows.append(res)

        # Price-basis sensitivity at the central cell: the same orders costed
        # with the full-sample median price instead of the pre-origin one. The
        # primary path is pre-origin; this only quantifies the look-ahead.
        if L == 4:
            q_ses = eval_rows[("ses", "quantile", 9)][0]
            mu_tw = g["mu__lgbm_tweedie"].to_numpy()
            mu_ses = g["mu__ses"].to_numpy()
            c_ses_pre = eval_rows[("ses", "quantile", 9)][1]
            c_ses_full = evaluate(g, q_ses, 4, price_rows_full, 9)["total_cost"]
            c_base_pre = eval_rows[("ses", "point", 9)][1]
            c_base_full = evaluate(g, mu_ses, 4, price_rows_full, 9)["total_cost"]
            c_tw_pre = evaluate(g, mu_tw, 4, price_rows, 9)["total_cost"]
            c_tw_full = evaluate(g, mu_tw, 4, price_rows_full, 9)["total_cost"]
            price_sensitivity = {
                "note": ("same orders, costed with the pre-origin median price "
                         "(primary) versus the full-sample median price"),
                "lead_time": 4, "cost_ratio": 9,
                "ses_quantile_vs_ses_point_pct": {
                    "pre_origin_price": round((1 - c_ses_pre / c_base_pre) * 100, 2),
                    "full_sample_price": round((1 - c_ses_full / c_base_full) * 100, 2),
                },
                "ses_quantile_vs_tweedie_point_pct": {
                    "pre_origin_price": round((1 - c_ses_pre / c_tw_pre) * 100, 2),
                    "full_sample_price": round((1 - c_ses_full / c_tw_full) * 100, 2),
                },
            }

        # ---- savings vs the best *point* rule among the simple baselines
        idx = rng.integers(0, n_sku, size=(C.N_BOOTSTRAP, n_sku))
        for ratio in C.COST_RATIOS:
            base_keys = [(m, "point", ratio) for m in SIMPLE]
            best_base = min(base_keys, key=lambda k: eval_rows[k][1])
            base_cost = per_sku[best_base]
            for model in MODELS:
                for policy in ("point", "quantile"):
                    key = (model, policy, ratio)
                    if key not in per_sku:
                        continue
                    own = per_sku[key]
                    sav = 1 - own.sum() / max(base_cost.sum(), 1e-9)
                    lo, hi = bootstrap_savings(own, base_cost, codes, idx)
                    boot_rows.append({
                        "lead_time": L, "cost_ratio": ratio, "model": model,
                        "policy": policy,
                        "savings_vs_best_simple_point_pct": round(float(sav) * 100, 2),
                        "ci_low_pct": round(lo, 2), "ci_high_pct": round(hi, 2),
                        "best_baseline": best_base[0],
                        "best_baseline_cost": round(float(base_cost.sum()), 2),
                        "own_cost": round(float(own.sum()), 2),
                    })

        if L == 4:
            focal = (g, per_sku[("lgbm_quantile", "quantile", 9)], focal_Q)

    table = pd.DataFrame(rows)
    boot = pd.DataFrame(boot_rows)
    # attach savings to the main table
    table = table.merge(
        boot[["lead_time", "cost_ratio", "model", "policy",
              "savings_vs_best_simple_point_pct", "ci_low_pct", "ci_high_pct",
              "best_baseline"]],
        on=["lead_time", "cost_ratio", "model", "policy"], how="left")

    # Honest benchmark: the cheapest *point* policy across all models, not just
    # the simple baselines. Without this column the savings look larger than
    # they are, because LightGBM-Tweedie point beats SES point in most cells.
    best_all = (table[table["policy"] == "point"]
                .sort_values("total_cost")
                .drop_duplicates(["lead_time", "cost_ratio"])
                .rename(columns={"model": "best_point_model_all",
                                 "total_cost": "best_point_cost_all"})
                [["lead_time", "cost_ratio", "best_point_model_all",
                  "best_point_cost_all"]])
    table = table.merge(best_all, on=["lead_time", "cost_ratio"], how="left")
    table["savings_vs_best_point_all_pct"] = (
        (1 - table["total_cost"] / table["best_point_cost_all"]) * 100
    ).round(2)

    # ------------------------------------------------------- worst SKUs
    g, cost_sku, Q_focal = focal
    sku_cost = (pd.DataFrame({"StockCode": g["StockCode"], "sku_idx": g["sku_idx"],
                              "c": cost_sku, "y": g["y"], "order": Q_focal})
                  .groupby(["StockCode", "sku_idx"], as_index=False)
                  .agg(cost=("c", "sum"), demand=("y", "sum"), order=("order", "sum")))
    meta = p.meta.reset_index()[["StockCode", "description", "class", "median_price"]]
    sku_cost = sku_cost.merge(meta, on="StockCode", how="left")
    sku_cost = sku_cost.sort_values("cost", ascending=False)
    sku_cost["share_of_total_cost_pct"] = (sku_cost["cost"] /
                                           sku_cost["cost"].sum() * 100).round(2)
    worst = sku_cost.head(20).round(2)

    table.to_csv(C.OUT_TABLES / "decision.csv", index=False)
    worst.to_csv(C.OUT_TABLES / "worst_skus.csv", index=False)

    summary = {
        "assumptions": {
            "holding_cost_per_week_of_unit_price": C.HOLD_COST_WEEK,
            "unit_price": ("median observed price per SKU up to each forecast "
                           "origin, winsorised at p99.5 (no look-ahead)"),
            "price_basis_sensitivity": price_sensitivity,
            "cost_ratios_swept": list(C.COST_RATIOS),
            "lead_times_weeks": list(C.LEAD_TIMES),
            "critical_ratio": {str(r): round(r / (1 + r), 4) for r in C.COST_RATIOS},
            "note": "absolute costs scale with the ratio and L; only the comparison is meaningful",
        },
        "n_sku_decisions_per_cell": int(len(g)),
        "oracle_cost": 0.0,
        "best_point_overall": (
            table[table["policy"] == "point"]
            .sort_values("total_cost")
            .drop_duplicates(["lead_time", "cost_ratio"])
            [["lead_time", "cost_ratio", "model", "total_cost",
              "savings_vs_best_simple_point_pct"]]
            .rename(columns={"model": "best_point_model"})
            .to_dict(orient="records")
        ),
        "worst_skus_top20": worst.to_dict(orient="records"),
        "cost_share_top20_skus_pct": round(float(worst["cost"].sum() /
                                                 sku_cost["cost"].sum() * 100), 2),
    }
    if save:
        DECISION_JSON.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(f"[decision] -> decision.csv ({len(table)} rows), worst_skus.csv")
    print(table[table.policy == "quantile"]
          .pivot_table(index="model", columns=["lead_time", "cost_ratio"],
                       values="savings_vs_best_simple_point_pct")
          .round(1).to_string())
    return summary


if __name__ == "__main__":
    run_decision()
