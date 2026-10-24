"""Contract tests for the forecasting pipeline.

Run either way:

    python -m pytest -q tests/test_pipeline.py
    python tests/test_pipeline.py

The tests are deliberately about *invariants* (no negative demand, complete
calendar spine, no future information, monotone quantiles, accounting
identities), not about re-asserting the headline numbers.  Point-in-time
metrics are allowed to change when a model is changed; a leaked feature or a
broken spine is never allowed.
"""
from __future__ import annotations

import dataclasses
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import config as C  # noqa: E402
from features import FEATURES, load_panel  # noqa: E402

_PANEL = None


def panel():
    global _PANEL
    if _PANEL is None:
        _PANEL = load_panel()
    return _PANEL


def _panel():
    return panel()


# --------------------------------------------------------------- the demand
def test_no_negative_demand():
    d = pd.read_csv(C.WEEKLY, dtype={"StockCode": str}, parse_dates=["week"])
    assert d["demand"].notna().all(), "missing demand"
    assert (d["demand"] >= 0).all(), "negative demand in the panel"
    assert (d["revenue"] >= 0).all(), "negative revenue in the panel"
    assert (d["demand"] == 0).mean() > 0.2, "suspiciously few zero weeks"


def test_calendar_spine_is_complete():
    d = pd.read_csv(C.WEEKLY, dtype={"StockCode": str}, parse_dates=["week"])
    weeks = pd.DatetimeIndex(sorted(d["week"].unique()))
    assert len(weeks) == 104, f"expected 104 complete weeks, got {len(weeks)}"
    assert (weeks.to_series().diff().dropna() == pd.Timedelta(days=7)).all()
    assert not d.duplicated(["StockCode", "week"]).any(), "duplicate SKU-week"
    sizes = d.groupby("StockCode")["week"].agg(["size", "nunique"])
    assert (sizes["size"] == len(weeks)).all(), "SKU with a missing week"
    assert (sizes["nunique"] == len(weeks)).all()
    # every SKU has 104 distinct weeks drawn from the same 104-week calendar
    assert set(d["week"].unique()) == set(weeks)


def test_cleaning_audit_adds_up():
    stats = json.loads((C.DATA_PROC / "clean_stats.json").read_text(encoding="utf-8"))
    rows = stats["raw_rows_two_sheets"]
    for step in stats["audit"]:
        assert step["rows_before"] == rows, f"audit chain broken at {step['step']}"
        rows -= step["rows_dropped"]
    assert rows == stats["rows_after_cleaning"]
    assert stats["cancellations"]["units"] < 0, "cancellations should net negative"
    assert stats["bulk_orders"]["top_1pct_invoice_share_units"] > 0
    assert 0 < stats["pct_rows_kept"] <= 100


def test_model_selection_is_frozen_before_the_first_origin():
    p = _panel()
    sel = json.loads((C.DATA_PROC / "sku_selection.json").read_text(encoding="utf-8"))
    assert int(sel["window_weeks"][1]) == p.origin0 - 1
    assert len(sel["skus"]) == len(p.model_skus) > 0
    pos = (p.D[:, : p.origin0] > 0).sum(axis=1)
    assert set(p.skus[pos >= C.MIN_POS_WEEKS_MODEL]) == set(p.model_skus)


# ------------------------------------------------------------- no leakage
def test_features_ignore_everything_after_the_origin():
    """Perturb all future weeks; features at the origin must not move."""
    p = _panel()
    o = p.origin0
    f = p.frame(o, C.HORIZONS)
    assert (f["target_idx"] == f["origin_idx"] + f["horizon"]).all()
    assert (f["target_idx"] > o).all()
    assert (f["target_week"] > f["origin_week"]).all()
    assert f["origin_week"].nunique() == 1

    rng = np.random.default_rng(0)
    cut = o + 1
    n, w = p.D.shape
    D2 = p.D.copy(); R2 = p.R.copy(); P2 = p.P.copy(); Pff2 = p.P_ff.copy()
    N2 = p.N.copy(); UK = p.UK.copy()
    D2[:, cut:] = rng.uniform(1e5, 2e5, size=(n, w - cut))
    R2[:, cut:] = rng.uniform(1e7, 2e7, size=(n, w - cut))
    P2[:, cut:] = rng.uniform(100, 200, size=(n, w - cut))
    Pff2[:, cut:] = rng.uniform(100, 200, size=(n, w - cut))
    N2[:, cut:] = rng.uniform(1e3, 2e3, size=(n, w - cut))
    UK[:, cut:] = rng.uniform(1e7, 2e7, size=(n, w - cut))
    p2 = dataclasses.replace(p, D=D2, R=R2, P=P2, P_ff=Pff2, N=N2, UK=UK)
    f2 = p2.frame(o, C.HORIZONS)
    for col in FEATURES:
        # floating-point summation is not bit-identical across array layouts;
        # a real leak moves features by orders of magnitude, not by 1e-13
        if not np.allclose(f[col].to_numpy(dtype=float),
                           f2[col].to_numpy(dtype=float),
                           rtol=1e-9, atol=1e-9, equal_nan=False):
            raise AssertionError(f"feature {col!r} uses future weeks")


def test_training_targets_stay_inside_the_fit_window():
    p = _panel()
    fit_end = p.origin0 - C.CALIB_WEEKS
    tr = p.training_frame(fit_end, (1, 2, 4), step=8)
    assert tr["target_idx"].max() <= fit_end
    assert tr["origin_idx"].max() <= fit_end - 4


# ------------------------------------------------------------- forecasts
def test_quantiles_are_monotone_and_non_negative():
    fc = pd.read_parquet(C.FORECASTS)
    models = ["mean", "seasonal_naive", "ses", "croston_sba", "glm_poisson",
              "lgbm_tweedie", "lgbm_quantile"]
    for m in models:
        q10, q50, q90 = (fc[f"{q}__{m}"].to_numpy() for q in ("q10", "q50", "q90"))
        assert np.isfinite(q10).all() and np.isfinite(q90).all(), m
        assert (q10 >= -1e-9).all() and (q50 >= -1e-9).all() and (q90 >= -1e-9).all()
        assert (q10 <= q50 + 1e-6).all(), f"{m}: P10 > P50"
        assert (q50 <= q90 + 1e-6).all(), f"{m}: P50 > P90"
    p = _panel()
    origin_week = p.weeks[fc["origin"].to_numpy()]
    assert (fc["target_week"].to_numpy() > origin_week).all()


def test_calibration_is_in_the_right_ballpark():
    acc = pd.read_csv(C.OUT_TABLES / "accuracy.csv")
    h1 = acc[acc["horizon"] == 1]
    assert h1["coverage80"].between(0.72, 0.88).all(), h1[["model", "coverage80"]]
    fc = pd.read_parquet(C.FORECASTS)
    d = fc[fc["horizon"] == 1]
    for m in h1["model"]:
        below = float((d["y"] < d[f"q10__{m}"]).mean())
        below90 = float((d["y"] <= d[f"q90__{m}"]).mean())
        assert 0.04 <= below <= 0.18, f"{m}: P(y<q10)={below:.3f}"
        assert 0.84 <= below90 <= 0.95, f"{m}: P(y<=q90)={below90:.3f}"


# -------------------------------------------------------------- decisions
def test_newsvendor_accounting_identities():
    dec = pd.read_csv(C.OUT_TABLES / "decision.csv")
    assert len(dec) == 3 * 4 * 7 * 2 + 3, "unexpected decision table shape"
    real = dec[dec["model"] != "oracle"]
    assert np.allclose(real["total_cost"],
                       real["holding_cost"] + real["stockout_cost"], atol=0.02)
    assert real["fill_rate"].between(0, 1).all()
    assert real["cycle_service_level"].between(0, 1).all()
    assert (real["total_cost"] >= 0).all()
    oracle = dec[dec["model"] == "oracle"]
    assert len(oracle) == 3
    assert np.allclose(oracle["total_cost"], 0.0)
    ses_point = dec[(dec["model"] == "ses") & (dec["policy"] == "point")]
    assert np.allclose(ses_point["savings_vs_best_simple_point_pct"], 0.0)


def test_decision_sweep_covers_the_charter_grid():
    dec = pd.read_csv(C.OUT_TABLES / "decision.csv")
    assert set(dec["lead_time"].dropna().astype(int)) == set(C.LEAD_TIMES)
    assert set(dec["cost_ratio"].dropna()) == set(C.COST_RATIOS)
    q = dec[dec["policy"] == "quantile"]
    assert (q["avg_inventory_units"] >= 0).all()
    assert q["weeks_of_cover"].notna().all()


# ---------------------------------------------------------------- results
def test_results_json_has_the_contract_keys():
    assert C.RESULTS.exists(), "run src/run_all.py first"
    res = json.loads(C.RESULTS.read_text(encoding="utf-8"))
    for key in ("schema_version", "meta", "data", "backtest", "origins",
                "accuracy", "accuracy_by_class", "aggregate_error",
                "calibration", "cold_start", "decision", "headline",
                "figures", "quality"):
        assert key in res, f"results.json missing {key!r}"
    assert len(res["figures"]) >= 5
    assert res["quality"]["same_obs_all_models"] is True
    assert set(res["quality"]["models"]) == {
        "mean", "seasonal_naive", "ses", "croston_sba", "glm_poisson",
        "lgbm_tweedie", "lgbm_quantile"}
    assert len(res["decision"]["rows"]) == 171
    assert res["data"]["n_model_skus"] == 3186


def test_headline_is_consistent_with_the_tables():
    res = json.loads(C.RESULTS.read_text(encoding="utf-8"))
    h1 = [r for r in res["accuracy"] if r["horizon"] == 1]
    best = min(h1, key=lambda r: r["mase"])
    assert best["model"] == res["headline"]["accuracy_h1"]["best_mase"]["model"]
    # results.json stores table records via 15-digit JSON, so compare loosely
    assert abs(best["mase"] - res["headline"]["accuracy_h1"]["best_mase"]["mase"]) < 1e-9
    central = res["headline"]["decision"]["central_case_quantile"]
    assert central["lead_time"] == 4 and central["cost_ratio"] == 9
    assert central["model"] in res["quality"]["models"]
    assert central["ci_low_pct"] <= central["savings_vs_best_simple_point_pct"] \
        <= central["ci_high_pct"]


def test_docs_and_figures_are_present():
    for name in ("README.md", "DATA.md", "METHOD.md", "LIMITATIONS.md",
                 "DECISIONS.md", "requirements.txt"):
        assert (ROOT / name).exists(), f"missing {name}"
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    assert "What this does not prove" in readme
    assert "results.json" in readme
    figs = sorted(C.OUT_FIGURES.glob("*.png"))
    assert len(figs) >= 5
    assert min(f.stat().st_size for f in figs) > 5_000, "empty figure"


# ------------------------------------------------------------------- runner
if __name__ == "__main__":
    tests = [(n, f) for n, f in sorted(globals().items())
             if n.startswith("test_") and callable(f)]
    failures = 0
    for name, fn in tests:
        try:
            fn()
        except Exception as exc:  # noqa: BLE001 - report and continue
            failures += 1
            print(f"FAIL {name}: {type(exc).__name__}: {exc}")
        else:
            print(f"ok   {name}")
    print(f"\n{len(tests) - failures} passed, {failures} failed")
    raise SystemExit(1 if failures else 0)
