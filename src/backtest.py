"""Rolling-origin backtest, accuracy metrics and probabilistic calibration.

    python src/backtest.py

For each forecast origin O (three of them, inside the last 26 weeks):
  * the models are fitted on weeks <= O - CALIB_WEEKS,
  * they are conformalised on weeks (O - CALIB_WEEKS, O] using only targets
    that are realised by O (t + h <= O) -- no look-ahead,
  * they are evaluated on target weeks O+1 .. O+8.

Outputs
-------
data/processed/forecasts.parquet     mu + P10/P50/P90 per origin/model/h/SKU
data/processed/residual_quantiles.csv  conformal residual grids
outputs/tables/accuracy*.csv         MAE/RMSE/MASE/pinball/coverage
outputs/tables/aggregate_error.csv   aggregate (sum over SKUs) error
outputs/tables/calibration.csv       predicted vs empirical quantile
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config as C
from features import assign_classes, load_panel, model_mask
from models import (BASELINES, ML_MODELS, MODELS, QGRID, MLForecaster,
                    baseline_mu)

BACKTEST_JSON = C.DATA_PROC / "_backtest.json"
H_POS = {h: i for i, h in enumerate(C.HORIZONS)}
EVAL_H = C.HORIZONS_ACC


# ---------------------------------------------------------------- utilities
def scale_naive(panel, origin: int) -> np.ndarray:
    """In-sample seasonal-naive MAE per SKU, used as the MASE denominator."""
    hist = panel.D[:, : origin + 1]
    if hist.shape[1] <= 52:
        return np.maximum(np.abs(np.diff(hist, axis=1)).mean(axis=1), 1e-6)
    return np.maximum(np.abs(hist[:, 52:] - hist[:, :-52]).mean(axis=1), 1e-6)


def align_baselines(p, origin, frame, tag=""):
    """Expand baseline forecasts (n_sku, n_h) onto the rows of ``frame``."""
    base = baseline_mu(p, origin, C.HORIZONS)
    out = {}
    for name, arr in base.items():
        v = np.zeros(len(frame))
        for h in C.HORIZONS:
            sel = (frame["horizon"] == h).to_numpy()
            v[sel] = arr[frame.loc[sel, "sku_idx"].to_numpy(), H_POS[h]]
        out[name] = v
    return out


def conformal_grids(y, mu, cls, horizon, origin, model, store, kind="horizon"):
    """Mondrian (per demand class) conformal residual grid on the sqrt scale."""
    s = np.sqrt(np.maximum(mu, 0) + 0.25)
    z = (y - mu) / s
    for c in C.CLASSES:
        sel = cls == c
        if sel.sum() < 30:                      # too few rows: fall back to pooled
            sel = np.ones_like(sel, dtype=bool)
        store.append(pd.DataFrame({
            "origin": origin, "model": model, "kind": kind, "horizon": horizon,
            "demand_class": c, "q": QGRID, "z": np.quantile(z[sel], QGRID),
            "n": int(sel.sum()),
        }))


def aggregate_grids(cal, mu_cal, model, cls, origin, store):
    """Conformal grid for L-week *cumulative* demand (newsvendor horizon).

    Uses the same calibration origins, summing weeks 1..L of each origin; only
    cumulative windows fully realised by the forecast origin are kept.
    """
    for L in C.LEAD_TIMES:
        sub = cal[cal["horizon"] <= L].copy()
        sub["mu"] = mu_cal[model][sub.index.to_numpy()]
        g = (sub.groupby(["origin_idx", "sku_idx"], as_index=False)
                .agg(yL=("y", "sum"), muL=("mu", "sum"), sku=("sku_idx", "first")))
        if len(g) < 50:
            continue
        klass = cls[g["sku"].to_numpy()]
        z = (g["yL"].to_numpy() - g["muL"].to_numpy()) / np.sqrt(
            np.maximum(g["muL"].to_numpy(), 0) + 0.25)
        for c in C.CLASSES:
            sel = klass == c
            if sel.sum() < 30:
                sel = np.ones_like(sel, dtype=bool)
            store.append(pd.DataFrame({
                "origin": origin, "model": model, "kind": "aggregate",
                "horizon": L, "demand_class": c, "q": QGRID,
                "z": np.quantile(z[sel], QGRID), "n": int(sel.sum()),
            }))


def q_from_grid(mu, grid: pd.DataFrame, q: float) -> np.ndarray:
    z = np.interp(q, grid["q"].to_numpy(), grid["z"].to_numpy())
    return np.clip(mu + np.sqrt(np.maximum(mu, 0) + 0.25) * z, 0, None)


def pinball(y, q_pred, q):
    d = y - q_pred
    return float(np.mean(np.maximum(q * d, (q - 1) * d)))


# --------------------------------------------------------------------- main
def run_backtest(save: bool = True) -> dict:
    t0 = time.time()
    p = load_panel()
    mask = model_mask(p)
    cls_all = p.meta["class"].reindex(p.skus).to_numpy()
    origins = [p.n_weeks + off for off in C.ORIGIN_OFFSETS]
    print(f"[backtest] panel {p.n_sku} SKUs x {p.n_weeks} weeks, "
          f"model sample {mask.sum()}, origins {origins}")

    forecast_rows, grid_rows = [], []
    train_log = []

    for origin in origins:
        t1 = time.time()
        fc = MLForecaster(p, origin)
        cal_origins = list(range(max(0, origin - C.CALIB_WEEKS + 1), origin + 1))
        cal = p.frame(cal_origins, C.HORIZONS)
        cal = cal[cal["StockCode"].isin(p.model_skus)]
        cal = cal[cal["target_idx"] <= origin].reset_index(drop=True)  # only realised targets
        tgt = p.frame(origin, C.HORIZONS)
        tgt = tgt[tgt["StockCode"].isin(p.model_skus)].reset_index(drop=True)

        mu_cal = {}
        # baselines are recomputed at every calibration origin
        for name in BASELINES:
            mu_cal[name] = np.zeros(len(cal))
        for t in cal_origins:
            sel = (cal["origin_idx"] == t).to_numpy()
            if not sel.any():
                continue
            b = baseline_mu(p, t, C.HORIZONS)
            for name in BASELINES:
                for h in C.HORIZONS:
                    sh = sel & (cal["horizon"] == h).to_numpy()
                    if sh.any():
                        mu_cal[name][sh] = b[name][cal.loc[sh, "sku_idx"].to_numpy(),
                                                    H_POS[h]]
        mu_cal.update(fc.predict(cal))

        y_cal = cal["y"].to_numpy()
        cls_cal = cls_all[cal["sku_idx"].to_numpy()]
        for model in MODELS:
            for h in C.HORIZONS:
                sel = (cal["horizon"] == h).to_numpy()
                conformal_grids(y_cal[sel], mu_cal[model][sel], cls_cal[sel],
                                h, origin, model, grid_rows)
            aggregate_grids(cal, mu_cal, model, cls_all, origin, grid_rows)

        mu_tgt = align_baselines(p, origin, tgt)
        mu_tgt.update(fc.predict(tgt))
        y_tgt = tgt["y"].to_numpy()
        cls_tgt = cls_all[tgt["sku_idx"].to_numpy()]
        grids = pd.concat(grid_rows, ignore_index=True)
        grids_o = grids[grids["origin"] == origin]

        block = {"origin": origin, "horizon": tgt["horizon"].to_numpy(),
                 "sku_idx": tgt["sku_idx"].to_numpy(),
                 "StockCode": tgt["StockCode"].to_numpy(),
                 "target_week": tgt["target_week"].to_numpy(),
                 "y": y_tgt}
        for model in MODELS:
            block[f"mu__{model}"] = mu_tgt[model]
            g_m = grids_o[grids_o["model"] == model]
            for q, col in ((0.1, "q10"), (0.5, "q50"), (0.9, "q90")):
                v = np.zeros(len(tgt))
                for h in C.HORIZONS:
                    sh = (tgt["horizon"] == h).to_numpy()
                    for c in C.CLASSES:
                        sc = sh & (cls_tgt == c)
                        if not sc.any():
                            continue
                        gg = g_m[(g_m["horizon"] == h) & (g_m["demand_class"] == c)]
                        v[sc] = q_from_grid(mu_tgt[model][sc], gg, q)
                block[f"{col}__{model}"] = v
        forecast_rows.append(pd.DataFrame(block))
        train_log.append({"origin": int(origin),
                          "origin_week": str(p.weeks[origin].date()),
                          "fit_end": int(fc.fit_end),
                          "fit_end_week": str(p.weeks[fc.fit_end].date()),
                          "n_train_rows": int(fc.n_train),
                          "n_calib_rows": int(len(cal)),
                          "seconds": round(time.time() - t1, 1)})
        print(f"[backtest] origin {origin} ({p.weeks[origin].date()}) "
              f"fit<={p.weeks[fc.fit_end].date()} train={fc.n_train:,} "
              f"calib={len(cal):,} target={len(tgt):,} "
              f"[{time.time()-t1:.0f}s]")

    fc_df = pd.concat(forecast_rows, ignore_index=True)
    grids = pd.concat(grid_rows, ignore_index=True)

    # ------------------------------------------------------------- metrics
    cls_by_sku = p.meta["class"].reindex(p.skus)
    fc_df["demand_class"] = cls_by_sku.reindex(fc_df["StockCode"]).to_numpy()
    scales = {o: scale_naive(p, o) for o in origins}
    fc_df["scale"] = np.nan
    for o in origins:
        sel = (fc_df["origin"] == o).to_numpy()
        fc_df.loc[sel, "scale"] = scales[o][fc_df.loc[sel, "sku_idx"].to_numpy()]

    acc_rows, class_rows, agg_rows, calib_rows = [], [], [], []
    for model in MODELS:
        for h in EVAL_H:
            d = fc_df[fc_df["horizon"] == h]
            y = d["y"].to_numpy()
            mu = d[f"mu__{model}"].to_numpy()
            q10, q50, q90 = (d[f"{c}__{model}"].to_numpy() for c in ("q10", "q50", "q90"))
            sc = d["scale"].to_numpy()
            ok = sc >= 0.01              # MASE undefined for deterministic series
            row = {
                "horizon": h, "model": model, "n_obs": int(len(d)),
                "mae": float(np.mean(np.abs(y - mu))),
                "rmse": float(np.sqrt(np.mean((y - mu) ** 2))),
                "mase": float(np.mean(np.abs(y - mu)[ok] / sc[ok])),
                "mase_p50": float(np.mean(np.abs(y - q50)[ok] / sc[ok])),
                "n_mase_excluded": int((~ok).sum()),
                "mae_p50": float(np.mean(np.abs(y - q50))),
                "bias": float(np.mean(mu - y)),
                "pinball_p10": pinball(y, q10, 0.1),
                "pinball_p50": pinball(y, q50, 0.5),
                "pinball_p90": pinball(y, q90, 0.9),
                "coverage80": float(np.mean((y >= q10) & (y <= q90))),
                "mean_pred": float(mu.mean()), "mean_actual": float(y.mean()),
            }
            acc_rows.append(row)
            for c in C.CLASSES:
                sel = (d["demand_class"] == c).to_numpy()
                if sel.sum() < 10:
                    continue
                okc = ok & sel
                class_rows.append({
                    "horizon": h, "model": model, "demand_class": c,
                    "n_obs": int(sel.sum()),
                    "mae": float(np.mean(np.abs(y[sel] - mu[sel]))),
                    "mase": float(np.mean(np.abs(y[okc] - mu[okc]) / sc[okc])),
                    "mae_p50": float(np.mean(np.abs(y[sel] - q50[sel]))),
                    "pinball_p50": pinball(y[sel], q50[sel], 0.5),
                    "coverage80": float(np.mean((y[sel] >= q10[sel]) & (y[sel] <= q90[sel]))),
                    "mean_pred": float(mu[sel].mean()),
                    "mean_actual": float(y[sel].mean()),
                })
            # aggregate error: sum over SKUs, then compare
            for o, g in d.groupby("origin"):
                ya, yp = g["y"].sum(), g[f"mu__{model}"].sum()
                agg_rows.append({"horizon": h, "model": model, "origin": int(o),
                                 "sum_actual": float(ya), "sum_pred": float(yp),
                                 "wape": float(abs(yp - ya) / ya),
                                 "bias_pct": float((yp - ya) / ya)})
            # calibration curve (all origins, this horizon)
            for q in np.arange(0.05, 0.96, 0.05):
                qp = np.zeros(len(d))
                for o in origins:
                    gg = grids[(grids["origin"] == o) & (grids["model"] == model) &
                               (grids["horizon"] == h)]
                    so = (d["origin"] == o).to_numpy()
                    so_idx = d[so].index
                    mu_o = d.loc[so_idx, f"mu__{model}"].to_numpy()
                    cls_o = d.loc[so_idx, "demand_class"].to_numpy()
                    v = np.zeros(len(mu_o))
                    for c in C.CLASSES:
                        sc_ = cls_o == c
                        if sc_.any():
                            v[sc_] = q_from_grid(mu_o[sc_], gg[gg["demand_class"] == c], q)
                    qp[so] = v
                calib_rows.append({"horizon": h, "model": model, "q_pred": round(float(q), 2),
                                   "q_obs": float(np.mean(y <= qp))})

    accuracy = pd.DataFrame(acc_rows)
    by_class = pd.DataFrame(class_rows)
    aggregate = pd.DataFrame(agg_rows)
    calibration = pd.DataFrame(calib_rows)

    # ---------------------------------------------------- bulk sensitivity
    bulk = bulk_sensitivity(p, fc_df, origins, scales)
    cold = cold_start_sensitivity(p, fc_df, origins)

    accuracy.to_csv(C.OUT_TABLES / "accuracy.csv", index=False)
    by_class.to_csv(C.OUT_TABLES / "accuracy_by_class.csv", index=False)
    aggregate.to_csv(C.OUT_TABLES / "aggregate_error.csv", index=False)
    calibration.to_csv(C.OUT_TABLES / "calibration.csv", index=False)
    pd.DataFrame(train_log).to_csv(C.OUT_TABLES / "backtest_origins.csv", index=False)
    cold["table"].to_csv(C.OUT_TABLES / "cold_start.csv", index=False)

    if save:
        fc_df.to_parquet(C.FORECASTS, index=False)
        grids.to_csv(C.RESIDUALS, index=False)
        print(f"[backtest] -> {C.FORECASTS.name} ({len(fc_df):,} rows), "
              f"{C.RESIDUALS.name} ({len(grids):,} rows)")

    summary = {
        "origins": train_log,
        "origins_weeks": [str(p.weeks[o].date()) for o in origins],
        "test_weeks": int(C.TEST_WEEKS),
        "horizons_eval": list(EVAL_H),
        "n_obs_per_horizon": int(len(fc_df[fc_df["horizon"] == 1])),
        "n_skus": int(mask.sum()),
        "seconds": round(time.time() - t0, 1),
        "bulk_sensitivity": bulk,
        "cold_start": cold["summary"],
        "as_of": pd.Timestamp.now().isoformat(timespec="seconds"),
    }
    BACKTEST_JSON.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"[backtest] done in {time.time()-t0:.0f}s")
    return summary


def bulk_sensitivity(p, fc_df, origins, scales) -> dict:
    """Recompute accuracy excluding evaluation weeks with extreme demand.

    A week is extreme if its demand exceeds the SKU's 99.5th percentile of
    *training* demand at that origin.  This tests whether a handful of bulk
    orders drive the ranking; the decision-side counterpart would need a full
    re-run (documented in LIMITATIONS.md).
    """
    keep = np.ones(len(fc_df), dtype=bool)
    for o in origins:
        hist = p.D[:, :o]
        thr = np.quantile(hist, 0.995, axis=1)
        sel = (fc_df["origin"] == o).to_numpy()
        y = fc_df.loc[sel, "y"].to_numpy()
        sku = fc_df.loc[sel, "sku_idx"].to_numpy()
        keep[sel] = y <= np.maximum(thr[sku], 1.0)
    h1 = fc_df[fc_df["horizon"] == 1]
    keep_h1 = keep[(fc_df["horizon"] == 1).to_numpy()]
    d_all = h1[keep_h1]
    total = h1["y"].sum()
    # the MASE denominator is undefined for (near-)deterministic series; use
    # the same exclusion as accuracy.csv so the two tables are comparable
    ok_all = h1["scale"].to_numpy() >= 0.01
    d = d_all[d_all["scale"].to_numpy() >= 0.01]
    out = {"excluded_obs_share_h1": round(float(1 - len(d_all) / len(h1)), 4),
           "excluded_demand_share_h1": round(float(1 - d_all["y"].sum() / total), 4),
           "n_mase_excluded_scale": int((~ok_all).sum()),
           "models": {}}
    for model in MODELS:
        y = d["y"].to_numpy(); mu = d[f"mu__{model}"].to_numpy()
        sc = d["scale"].to_numpy()
        y_all = h1.loc[ok_all, "y"].to_numpy()
        mu_all = h1.loc[ok_all, f"mu__{model}"].to_numpy()
        sc_all = h1.loc[ok_all, "scale"].to_numpy()
        out["models"][model] = {
            "mase": round(float(np.mean(np.abs(y - mu) / sc)), 4),
            "mase_all_weeks": round(float(np.mean(
                np.abs(y_all - mu_all) / sc_all)), 4),
        }
    return out


def cold_start_sensitivity(p, fc_df, origins) -> dict:
    """h=1 accuracy for SKUs with <26 vs >=26 positive weeks before the origin.

    The charter asks explicitly for cold-start performance.  The modelling
    sample requires >=12 positive pre-origin weeks, so the cold bucket contains
    SKUs with 12-25 positive weeks; SKUs with fewer are outside the sample and
    are not silently averaged in.
    """
    rows, pooled = [], {}
    n_skus = {}
    for o in origins:
        npos = (p.D[:, : o + 1] > 0).sum(axis=1)
        d = fc_df[(fc_df["origin"] == o) & (fc_df["horizon"] == 1)].copy()
        d["hist_pos_weeks"] = npos[d["sku_idx"].to_numpy()]
        d["history_bucket"] = np.where(
            d["hist_pos_weeks"] < 26, "lt26_positive_weeks", "ge26_positive_weeks")
        if o == origins[0]:
            n_skus = {b: int(d.loc[d["history_bucket"] == b, "sku_idx"].nunique())
                      for b in d["history_bucket"].unique()}
        for bucket in ("lt26_positive_weeks", "ge26_positive_weeks"):
            b = d[(d["history_bucket"] == bucket) & (d["scale"] >= 0.01)]
            if len(b) < 10:
                continue
            y = b["y"].to_numpy()
            for model in MODELS:
                mu = b[f"mu__{model}"].to_numpy()
                sc = b["scale"].to_numpy()
                ratio = np.abs(y - mu) / sc
                pooled.setdefault((bucket, model), []).append(ratio)
                rows.append({
                    "origin": int(o), "history_bucket": bucket, "model": model,
                    "n_obs": int(len(b)), "n_skus": int(b["sku_idx"].nunique()),
                    "mase": float(np.mean(ratio)),
                    "mae": float(np.mean(np.abs(y - mu))),
                })
    table = pd.DataFrame(rows)
    summary = {"n_skus_by_bucket": n_skus, "models": {}}
    for bucket in ("lt26_positive_weeks", "ge26_positive_weeks"):
        summary["models"][bucket] = {}
        for model in MODELS:
            arr = pooled.get((bucket, model), [])
            if not arr:
                continue
            flat = np.concatenate(arr)
            summary["models"][bucket][model] = {
                "n_obs": int(flat.size),
                "mase": round(float(flat.mean()), 4),
            }
    return {"table": table, "summary": summary}


if __name__ == "__main__":
    run_backtest()

