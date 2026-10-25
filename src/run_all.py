"""One-command rebuild of the whole project.

    python src/run_all.py                 # verify/download, clean, backtest, decide, figures, tests
    python src/run_all.py --from-cache    # reuse data/processed/transactions_raw.parquet
    python src/run_all.py --skip-tests    # pipeline only
    python src/run_all.py --force-download

The script runs each stage in a separate Python process so a failure stops the
build with the failing stage's log.  Finally it assembles ``outputs/results.json``
from the stage artefacts.  Every number quoted in README.md is a field of that
file; no metric is typed by hand.
"""
from __future__ import annotations

import argparse
import importlib.metadata as md
import json
import platform
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config as C

SRC = Path(__file__).resolve().parent
ROOT = SRC.parent
TEST_LOG = C.OUT / "test_report.txt"
STAGES = ("download", "clean", "backtest", "decision", "figures")


# --------------------------------------------------------------------- utils
SCRIPTS = {"download": "download_data.py", "clean": "clean.py",
           "backtest": "backtest.py", "decision": "decision.py",
           "figures": "figures.py"}


def run_stage(name: str, args: list[str]) -> float:
    """Run one script as a subprocess; return elapsed seconds."""
    cmd = [sys.executable, str(SRC / SCRIPTS[name]), *args]
    print(f"\n=== {name} ===\n$ {' '.join(cmd)}", flush=True)
    t0 = time.time()
    subprocess.run(cmd, cwd=ROOT, check=True)
    dt = time.time() - t0
    print(f"=== {name}: ok in {dt:.1f}s ===", flush=True)
    return dt


def records(path: Path) -> list[dict]:
    """DataFrame -> JSON-native records (NaN becomes null)."""
    return json.loads(pd.read_csv(path).to_json(orient="records"))


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def versions() -> dict:
    out = {"python": platform.python_version(), "platform": platform.platform()}
    for pkg in ("pandas", "numpy", "scikit-learn", "lightgbm", "matplotlib",
                "requests", "pyarrow", "openpyxl", "pytest"):
        try:
            out[pkg] = md.version(pkg)
        except md.PackageNotFoundError:
            out[pkg] = None
    return out


def git_commit() -> str | None:
    try:
        r = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, check=True,
                           capture_output=True, text=True)
        return r.stdout.strip() or None
    except Exception:
        return None


def _f(x) -> float:
    return float(x)


def _native(v):
    """pandas/numpy scalar -> strict JSON type (None for missing)."""
    if v is None or (not isinstance(v, (list, dict)) and pd.isna(v)):
        return None
    if isinstance(v, np.integer):
        return int(v)
    if isinstance(v, np.floating):
        return float(v)
    return v


def build_headline(acc: pd.DataFrame, agg: pd.DataFrame, cal: pd.DataFrame,
                   dec: pd.DataFrame, fc: pd.DataFrame) -> dict:
    """Numbers the README quotes, derived from the tables (no hardcoding)."""
    h1 = acc[acc["horizon"] == 1].copy()
    h1 = h1.sort_values("mase")
    ses_mase = _f(h1.loc[h1["model"] == "ses", "mase"].iat[0])
    cro_mase = _f(h1.loc[h1["model"] == "croston_sba", "mase"].iat[0])
    ranked = [{
        "model": r["model"], "mae": _f(r["mae"]), "rmse": _f(r["rmse"]),
        "mase": _f(r["mase"]), "mase_p50": _f(r["mase_p50"]),
        "pinball_p50": _f(r["pinball_p50"]),
        "coverage80": _f(r["coverage80"]),
    } for _, r in h1.iterrows()]
    best_mase = h1.iloc[0]
    best_mae = h1.loc[h1["mae"].idxmin()]

    cov = h1.set_index("model")["coverage80"]
    h1_cal = cal[(cal["horizon"] == 1) & (cal["q_pred"].isin([0.1, 0.9]))]
    q10 = h1_cal[h1_cal["q_pred"] == 0.1]["q_obs"]
    q90 = h1_cal[h1_cal["q_pred"] == 0.9]["q_obs"]
    f1 = fc[fc["horizon"] == 1]
    strict10 = {m: _f((f1["y"] < f1[f"q10__{m}"]).mean()) for m in h1["model"]}
    below90 = {m: _f((f1["y"] <= f1[f"q90__{m}"]).mean()) for m in h1["model"]}

    a1 = agg[agg["horizon"] == 1]
    pooled = {}
    for m, g in a1.groupby("model"):
        pooled[m] = _f(g["sum_pred"].sub(g["sum_actual"]).abs().sum()
                       / g["sum_actual"].sum())

    q = dec[dec["policy"] == "quantile"]
    central = q[(q["lead_time"] == 4) & (q["cost_ratio"] == 9)]
    central = central.sort_values("savings_vs_best_simple_point_pct").iloc[-1]
    best_cell = q.loc[q["savings_vs_best_simple_point_pct"].idxmax()]
    per_l = []
    for L in C.LEAD_TIMES:
        s = q[q["lead_time"] == L]
        r = s.loc[s["savings_vs_best_simple_point_pct"].idxmax()]
        per_l.append({k: _native(r[k])
                      for k in ("lead_time", "cost_ratio", "model",
                                "savings_vs_best_simple_point_pct",
                                "ci_low_pct", "ci_high_pct", "fill_rate",
                                "avg_inventory_units", "weeks_of_cover",
                                "best_baseline")})

    point_central = dec[(dec["policy"] == "point") & (dec["lead_time"] == 4)
                        & (dec["cost_ratio"] == 9)].copy()
    point_central = point_central.sort_values("total_cost")

    def slim(r, cols):
        return {k: _native(r[k]) for k in cols}

    dec_cols = ("lead_time", "cost_ratio", "model", "policy",
                "total_cost", "holding_cost", "stockout_cost", "fill_rate",
                "avg_inventory_units", "weeks_of_cover",
                "savings_vs_best_simple_point_pct", "ci_low_pct", "ci_high_pct",
                "best_baseline", "best_point_model_all", "best_point_cost_all",
                "savings_vs_best_point_all_pct")
    central_best_point = (dec[(dec["policy"] == "point")
                              & (dec["lead_time"] == 4)
                              & (dec["cost_ratio"] == 9)]
                          .sort_values("total_cost").iloc[0])
    return {
        "accuracy_h1": {
            "ranked_by_mase": ranked,
            "best_mase": slim(best_mase, ("model", "mase", "mae", "rmse",
                                          "pinball_p50", "coverage80")),
            "best_mae": slim(best_mae, ("model", "mase", "mae", "rmse",
                                        "pinball_p50", "coverage80")),
            "ses_mase": ses_mase,
            "croston_sba_mase": cro_mase,
            "models_beating_ses_mase": [r["model"] for r in ranked
                                        if r["mase"] < ses_mase],
            "models_beating_croston_mase": [r["model"] for r in ranked
                                            if r["mase"] < cro_mase],
            "coverage80_min": _f(cov.min()),
            "coverage80_max": _f(cov.max()),
        },
        "calibration_h1": {
            "observed_p_at_q10_min": _f(q10.min()),
            "observed_p_at_q10_max": _f(q10.max()),
            "observed_p_at_q90_min": _f(q90.min()),
            "observed_p_at_q90_max": _f(q90.max()),
            "strict_below_q10_min": min(strict10.values()),
            "strict_below_q10_max": max(strict10.values()),
            "at_or_below_q90_min": min(below90.values()),
            "at_or_below_q90_max": max(below90.values()),
            "per_model": {m: {"p_y_lt_q10": strict10[m],
                              "p_y_le_q90": below90[m]} for m in strict10},
            "note": "atom at zero: P(y<=q10) is inflated; P(y<q10) and P(y<=q90) are the meaningful checks",
        },
        "aggregate_h1_pooled_wape": pooled,
        "decision": {
            "central_case_quantile": slim(central, dec_cols),
            "central_case_best_point": slim(central_best_point, dec_cols),
            "best_quantile_cell": slim(best_cell, dec_cols),
            "best_quantile_per_lead_time": per_l,
            "central_case_point_rules": [slim(r, dec_cols)
                                         for _, r in point_central.iterrows()],
            "best_baseline_counts": {
                str(k): int(v) for k, v in
                dec["best_baseline"].value_counts().items()},
        },
    }


# ------------------------------------------------------------------ assembly
def assemble(stage_seconds: dict) -> dict:
    stats = read_json(C.DATA_PROC / "clean_stats.json")
    bt = read_json(C.DATA_PROC / "_backtest.json")
    dc = read_json(C.DATA_PROC / "_decision.json")
    acc = pd.read_csv(C.OUT_TABLES / "accuracy.csv")
    by_class = pd.read_csv(C.OUT_TABLES / "accuracy_by_class.csv")
    agg = pd.read_csv(C.OUT_TABLES / "aggregate_error.csv")
    cal = pd.read_csv(C.OUT_TABLES / "calibration.csv")
    dec = pd.read_csv(C.OUT_TABLES / "decision.csv")
    origins = pd.read_csv(C.OUT_TABLES / "backtest_origins.csv")
    fc = pd.read_parquet(C.FORECASTS)

    manifest_path = C.DATA_RAW / "manifest.json"
    manifest = read_json(manifest_path) if manifest_path.exists() else None
    fig_files = sorted(C.OUT_FIGURES.glob("*.png"))
    artifact_paths = [
        C.WEEKLY, C.FORECASTS, C.RESIDUALS,
        *sorted(C.OUT_TABLES.glob("*.csv")), *fig_files, C.RESULTS,
    ]
    artifacts = {str(p.relative_to(ROOT)).replace("\\", "/"): p.stat().st_size
                 for p in artifact_paths if p.exists()}

    results = {
        "schema_version": 1,
        "meta": {
            "project": "03_online_retail_forecasting",
            "title": "Forecasting sparse demand: probabilistic SKU forecasts and the cost of getting inventory wrong",
            "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "command": "python src/run_all.py",
            "git_commit": git_commit(),
            "seed": C.SEED,
            "versions": versions(),
            "stage_seconds": {k: round(v, 1) for k, v in stage_seconds.items()},
            "total_seconds": round(sum(stage_seconds.values()), 1),
        },
        "data": {**stats, "raw_manifest": manifest,
                 "panel_file": str(C.WEEKLY.relative_to(ROOT)).replace("\\", "/")},
        "backtest": bt,
        "origins": records(C.OUT_TABLES / "backtest_origins.csv"),
        "accuracy": records(C.OUT_TABLES / "accuracy.csv"),
        "accuracy_by_class": records(C.OUT_TABLES / "accuracy_by_class.csv"),
        "aggregate_error": records(C.OUT_TABLES / "aggregate_error.csv"),
        "calibration": records(C.OUT_TABLES / "calibration.csv"),
        "cold_start": records(C.OUT_TABLES / "cold_start.csv"),
        "decision": {
            "assumptions": dc["assumptions"],
            "n_sku_decisions_per_cell": dc["n_sku_decisions_per_cell"],
            "oracle_cost": dc["oracle_cost"],
            "cost_share_top20_skus_pct": dc["cost_share_top20_skus_pct"],
            "rows": records(C.OUT_TABLES / "decision.csv"),
            "worst_skus_top20": records(C.OUT_TABLES / "worst_skus.csv"),
        },
        "headline": build_headline(acc, agg, cal, dec, fc),
        "figures": [p.name for p in fig_files],
        "artifacts_bytes": artifacts,
        "quality": {
            "origins_count": len(origins),
            "models": sorted(acc["model"].unique().tolist()),
            "horizons_evaluated": sorted(int(h) for h in acc["horizon"].unique()),
            "same_obs_all_models": bool(
                (acc.groupby(["horizon", "model"])["n_obs"].nunique() == 1).all()),
            "tests": {"status": "not_run"},
        },
    }
    return results


def write_results(results: dict) -> None:
    C.RESULTS.write_text(json.dumps(results, indent=2, allow_nan=False),
                         encoding="utf-8")
    print(f"[run_all] results -> {C.RESULTS} "
          f"({C.RESULTS.stat().st_size / 1024:.1f} KB)")


# ---------------------------------------------------------------------- tests
def run_tests() -> dict:
    print("\n=== tests ===", flush=True)
    cmd = [sys.executable, "-m", "pytest", "-q", "tests/test_pipeline.py"]
    t0 = time.time()
    r = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True,
                       encoding="utf-8", errors="replace")
    out = (r.stdout or "") + (r.stderr or "")
    TEST_LOG.write_text(f"$ {' '.join(cmd)}\n\n{out}", encoding="utf-8")
    print(out.strip(), flush=True)
    import re
    m = re.search(r"(\d+) passed", out)
    failed = re.search(r"(\d+) failed", out)
    status = "passed" if r.returncode == 0 else "failed"
    return {
        "status": status,
        "command": "python -m pytest -q tests/test_pipeline.py",
        "n_passed": int(m.group(1)) if m else 0,
        "n_failed": int(failed.group(1)) if failed else 0,
        "seconds": round(time.time() - t0, 1),
        "log": str(TEST_LOG.relative_to(ROOT)).replace("\\", "/"),
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--from-cache", action="store_true",
                    help="reuse data/processed/transactions_raw.parquet instead of re-reading the workbook")
    ap.add_argument("--force-download", action="store_true")
    ap.add_argument("--skip-tests", action="store_true")
    args = ap.parse_args(argv)

    seconds: dict[str, float] = {}
    t0 = time.time()
    seconds["download"] = run_stage("download",
                                    ["--force"] if args.force_download else [])
    seconds["clean"] = run_stage("clean", [] if args.from_cache else ["--force"])
    seconds["backtest"] = run_stage("backtest", [])
    seconds["decision"] = run_stage("decision", [])
    seconds["figures"] = run_stage("figures", [])

    results = assemble(seconds)
    write_results(results)

    if not args.skip_tests:
        results["quality"]["tests"] = run_tests()
        write_results(results)
        if results["quality"]["tests"]["status"] != "passed":
            print("[run_all] TESTS FAILED - results.json written, build not clean",
                  file=sys.stderr)
            return 1

    print(f"\n[run_all] done in {time.time() - t0:.0f}s")
    h = results["headline"]["accuracy_h1"]
    print(f"[run_all] h=1 MASE winner: {h['best_mase']['model']} "
          f"({h['best_mase']['mase']:.3f}); SES {h['ses_mase']:.3f}; "
          f"Croston/SBA {h['croston_sba_mase']:.3f}")
    c = results["headline"]["decision"]["central_case_quantile"]
    print(f"[run_all] central decision (L={c['lead_time']}, Cu/Co={c['cost_ratio']}): "
          f"{c['model']} quantile saves {c['savings_vs_best_simple_point_pct']:.1f}% "
          f"[{c['ci_low_pct']:.1f}, {c['ci_high_pct']:.1f}] vs {c['best_baseline']} point; "
          f"{c['savings_vs_best_point_all_pct']:.1f}% vs the cheapest point rule in that cell "
          f"({c['best_point_model_all']})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
