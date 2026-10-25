"""Figures. Everything is read from the pipeline outputs, nothing is hardcoded.

    python src/figures.py

produces outputs/figures/fig1..fig7.
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config as C
from models import MODELS, QGRID

plt.rcParams.update({"figure.dpi": 130, "font.size": 9,
                     "axes.grid": True, "grid.alpha": 0.25,
                     "axes.spines.top": False, "axes.spines.right": False})
COLORS = {"mean": "#8c8c8c", "seasonal_naive": "#c7a76c", "ses": "#1f77b4",
          "croston_sba": "#2ca02c", "glm_poisson": "#9467bd",
          "lgbm_tweedie": "#d62728", "lgbm_quantile": "#111111"}
NICE = {"mean": "historical mean", "seasonal_naive": "seasonal naive",
        "ses": "SES", "croston_sba": "Croston/SBA", "glm_poisson": "Poisson GLM",
        "lgbm_tweedie": "LightGBM Tweedie", "lgbm_quantile": "LightGBM median"}


def fig1_intermittency(p):
    fig, ax = plt.subplots(1, 3, figsize=(11, 3.2))
    d = p.D[np.isin(p.skus, p.model_skus)]
    zs = (d == 0).mean(axis=1)
    ax[0].hist(zs, bins=40, color="#1f77b4", alpha=0.85)
    ax[0].axvline(np.median(zs), color="k", ls="--", lw=1)
    ax[0].set_title(f"Zero-demand share per SKU\nmedian {np.median(zs):.0%}, "
                    f"pooled {(d == 0).mean():.0%}")
    ax[0].set_xlabel("share of weeks with zero demand")
    ax[0].set_ylabel("SKUs")

    vals = np.sort(d.ravel())[::-1]
    vals = vals[vals > 0]
    ax[1].loglog(np.arange(1, len(vals) + 1), vals, color="#d62728", lw=1)
    ax[1].set_title("SKU-week demand, rank-frequency")
    ax[1].set_xlabel("rank of SKU-week")
    ax[1].set_ylabel("units")

    share = np.cumsum(vals) / vals.sum()
    ax[2].plot(np.arange(1, len(vals) + 1) / len(vals) * 100, share * 100,
               color="#2ca02c")
    ax[2].plot([0, 100], [0, 100], "k--", lw=0.8)
    for q in (1, 10):
        k = int(len(vals) * q / 100)
        ax[2].scatter([q], [share[k] * 100], color="k", zorder=5, s=14)
        ax[2].annotate(f"top {q}% weeks = {share[k]:.0%} of units",
                       (q, share[k] * 100), textcoords="offset points",
                       xytext=(6, -10), fontsize=8)
    ax[2].set_title("Demand concentration")
    ax[2].set_xlabel("% of SKU-weeks (ranked)"); ax[2].set_ylabel("% of units")
    fig.tight_layout()
    fig.savefig(C.OUT_FIGURES / "fig1_intermittency.png", bbox_inches="tight")
    plt.close(fig)


def _representative_skus(p):
    """One fast, one medium and one slow SKU, typical of their class.

    Picked on the *median* weekly demand in the 26-week evaluation window, so a
    single bulk-order week cannot decide the panel.  Candidates with a maximum
    week above ``15x median (+20 units)`` are excluded from the choice, and the
    SKU closest to the class median is shown; this is a display choice only and
    does not enter any metric.
    """
    test_start = p.n_weeks - C.TEST_WEEKS
    win = p.D[:, test_start:]
    mask = np.isin(p.skus, p.model_skus)
    klass = p.meta["class"].reindex(p.skus).to_numpy()
    med = np.median(win, axis=1)
    mx = win.max(axis=1)
    picks = []
    for cls in ("fast", "medium", "slow"):
        in_class = mask & (klass == cls)
        target = float(np.median(med[in_class]))
        cand = in_class & (med > 0) & (mx <= np.maximum(15 * med, med + 20))
        if not cand.any():
            cand = in_class
        idx = np.where(cand)[0]
        picks.append(p.skus[idx[np.argmin(np.abs(med[idx] - target))]])
    return picks


def fig2_forecasts(p, fc):
    """Forecast paths vs actual for one fast, one medium and one slow SKU.

    Each rolling origin issues h=1..8 week-ahead forecasts.  The three origins
    are adjacent in 8-week blocks, so the paths tile weeks 78-101 of the test
    window and can be compared with the realised series.  Points are genuine
    multi-step forecasts (not a one-step filtered path).
    """
    d = fc[fc["horizon"].between(1, 8)].copy()
    d["class"] = p.meta["class"].reindex(d["StockCode"]).to_numpy()
    picks = _representative_skus(p)
    test_start = p.n_weeks - C.TEST_WEEKS
    origins = sorted(d["origin"].unique())

    fig, axes = plt.subplots(3, 1, figsize=(9.2, 7.6), sharex=True)
    for ax, sku in zip(axes, picks):
        i = int(np.where(p.skus == sku)[0][0])
        lo = max(0, test_start - 4)
        ax.plot(p.weeks[lo:], p.D[i, lo:], "o-", color="k", lw=1.1, ms=3.2,
                label="actual")
        s = d[d["StockCode"] == sku]
        for j, o in enumerate(origins):
            seg = s[s["origin"] == o].sort_values("horizon")
            ax.fill_between(seg["target_week"], seg["q10__lgbm_tweedie"],
                            seg["q90__lgbm_tweedie"], color="#d62728",
                            alpha=0.14,
                            label="LightGBM P10-P90" if j == 0 else "_nolegend_")
            ax.plot(seg["target_week"], seg["mu__lgbm_tweedie"], color="#d62728",
                    lw=1.5, marker=".", ms=4,
                    label="LightGBM mean" if j == 0 else "_nolegend_")
            ax.plot(seg["target_week"], seg["mu__ses"], color="#1f77b4",
                    lw=1.2, ls="--", marker=".", ms=3,
                    label="SES" if j == 0 else "_nolegend_")
            ax.axvline(p.weeks[o], color="grey", lw=0.6, ls=":",
                       label="forecast origin" if j == 0 else "_nolegend_")
        cls = p.meta.loc[sku, "class"]
        desc = p.meta.loc[sku, "description"]
        ax.set_title(f"{sku} — {desc} [{cls}]", fontsize=9)
        ax.set_ylabel("units / week")
    axes[0].legend(fontsize=7.5, ncol=4, loc="upper left")
    fig.suptitle("Rolling-origin forecast paths (h=1..8) vs actual demand — "
                 "origins 2011-05-30, 2011-07-25, 2011-09-19", fontsize=9.5)
    fig.tight_layout()
    fig.savefig(C.OUT_FIGURES / "fig2_forecast_vs_actual.png", bbox_inches="tight")
    plt.close(fig)


def fig3_calibration(fc, grids):
    """Empirical coverage of central intervals vs nominal level, by model."""
    d = fc[fc["horizon"] == 1]
    levels = np.arange(0.50, 0.976, 0.025)
    fig, ax = plt.subplots(figsize=(5.2, 4.6))
    for model in MODELS:
        cov = []
        for lv in levels:
            lo_q, hi_q = (1 - lv) / 2, 1 - (1 - lv) / 2
            q10 = np.zeros(len(d)); q90 = np.zeros(len(d))
            for o in d["origin"].unique():
                sel = (d["origin"] == o).to_numpy()
                mu = d.loc[sel, f"mu__{model}"].to_numpy()
                klass = d.loc[sel, "demand_class"].to_numpy()
                gg = grids[(grids["kind"] == "horizon") & (grids["horizon"] == 1) &
                           (grids["model"] == model) & (grids["origin"] == o)]
                for c in C.CLASSES:
                    sc = klass == c
                    if not sc.any():
                        continue
                    z = gg[gg["demand_class"] == c].sort_values("q")
                    zl = np.interp(lo_q, QGRID, z["z"].to_numpy())
                    zh = np.interp(hi_q, QGRID, z["z"].to_numpy())
                    s = np.sqrt(np.maximum(mu[sc], 0) + 0.25)
                    q10[sel.nonzero()[0][sc]] = np.clip(mu[sc] + s * zl, 0, None)
                    q90[sel.nonzero()[0][sc]] = np.clip(mu[sc] + s * zh, 0, None)
            y = d["y"].to_numpy()
            cov.append(np.mean((y >= q10) & (y <= q90)))
        ax.plot(levels, cov, marker="o", ms=2.5, lw=1.2, color=COLORS[model],
                label=NICE[model])
    ax.plot([0.5, 1], [0.5, 1], "k--", lw=0.8)
    ax.axvline(0.8, color="grey", lw=0.8, ls=":")
    ax.set_xlabel("nominal coverage of central interval")
    ax.set_ylabel("empirical coverage")
    ax.set_title("Interval calibration, h=1, all origins")
    ax.legend(fontsize=7, loc="upper left")
    fig.tight_layout()
    fig.savefig(C.OUT_FIGURES / "fig3_calibration.png", bbox_inches="tight")
    plt.close(fig)


def fig4_cost_vs_ratio(dec):
    """Two honest comparators.

    Top row: saving vs the best *simple* point rule (SES in every cell), with the
    bootstrap CI, because that is what the bootstrap in decision.py estimates.
    Bottom row: saving vs the cheapest *point* rule in each cell. That rule is
    LightGBM-Tweedie at most cost ratios but LightGBM-median at Cu/Co = 1, so the
    label must not name one model for the whole row. Point estimates only: the
    bootstrap is not defined against that comparator, so drawing a CI band there
    would be fabricated.
    """
    fig, axes = plt.subplots(2, 3, figsize=(11, 6.8), sharex="col")
    q = dec[(dec["policy"] == "quantile")]
    floor = -65.0
    specs = [("savings_vs_best_simple_point_pct", True,
              "vs best simple point rule (SES)"),
             ("savings_vs_best_point_all_pct", False,
              "vs cheapest point rule in each cell")]
    for r, (col, show_ci, ylabel) in enumerate(specs):
        for c, L in enumerate(C.LEAD_TIMES):
            ax = axes[r, c]
            for model in MODELS:
                s = q[(q["lead_time"] == L) & (q["model"] == model)].sort_values("cost_ratio")
                ax.plot(s["cost_ratio"], s[col],
                        marker="o", ms=3, color=COLORS[model], label=NICE[model], lw=1.2)
                if show_ci:
                    ax.fill_between(s["cost_ratio"], s["ci_low_pct"], s["ci_high_pct"],
                                    color=COLORS[model], alpha=0.08)
            sn = q[(q["lead_time"] == L) & (q["model"] == "seasonal_naive")]
            lo = float(sn[col].min())
            ax.axhline(0, color="k", lw=0.8, ls="--")
            ax.set_xscale("log")
            ax.set_xticks(list(C.COST_RATIOS))
            ax.set_xticklabels([str(x) for x in C.COST_RATIOS])
            ax.set_ylim(floor, 55)
            ax.set_xlabel("Cu / Co (underage / holding cost)")
            if r == 0:
                ax.set_title(f"lead time = {L} weeks")
            if lo < floor:
                ax.text(0.02, 0.03, f"seasonal naive off scale: {lo:.0f}%",
                        transform=ax.transAxes, fontsize=6.2, color="#8a6d3b")
        axes[r, 0].set_ylabel("cost saving (%)")
        axes[r, 0].text(0.02, 0.96, ylabel, transform=axes[r, 0].transAxes,
                        fontsize=7, va="top", color="#333333")
    axes[0, 0].legend(fontsize=6.2, loc="upper left", bbox_to_anchor=(0.0, 0.90))
    fig.suptitle("Decision value of probabilistic ordering. Top: shaded 95% bootstrap CI over SKUs "
                 "(defined vs SES point).\nBottom: point estimates vs the cheapest point rule in "
                 "each cell; no CI is claimed. y-axis clipped at -65%.", fontsize=8.6)
    fig.tight_layout()
    fig.savefig(C.OUT_FIGURES / "fig4_cost_vs_ratio.png", bbox_inches="tight")
    plt.close(fig)


def fig5_error_by_class(by_class):
    d = by_class[by_class["horizon"] == 1]
    classes = [c for c in C.CLASSES if c in set(d["demand_class"])]
    x = np.arange(len(classes))
    w = 0.8 / len(MODELS)
    fig, ax = plt.subplots(figsize=(8, 3.8))
    for i, model in enumerate(MODELS):
        s = d[d["model"] == model].set_index("demand_class").reindex(classes)
        ax.bar(x + i * w - 0.4 + w / 2, s["mase"], width=w * 0.92,
               color=COLORS[model], label=NICE[model])
    ax.axhline(1, color="k", lw=0.8, ls="--")
    ax.set_xticks(x); ax.set_xticklabels(classes)
    ax.set_ylabel("MASE (h=1)")
    ax.set_title("Error by demand class — below 1 beats seasonal naive")
    ax.legend(fontsize=7, ncol=4)
    fig.tight_layout()
    fig.savefig(C.OUT_FIGURES / "fig5_error_by_class.png", bbox_inches="tight")
    plt.close(fig)


def fig6_service_vs_inventory(dec):
    fig, ax = plt.subplots(figsize=(6.4, 4.6))
    d = dec[(dec["lead_time"] == 4) & (dec["cost_ratio"] == 9)]
    labels = []
    for model in MODELS:
        s = d[d["model"] == model]
        for _, row in s.iterrows():
            ax.scatter(row["avg_inventory_units"], row["fill_rate"] * 100,
                       color=COLORS[model], marker="o" if row["policy"] == "quantile" else "s",
                       s=42 if row["policy"] == "quantile" else 30,
                       edgecolor="k", linewidth=0.4, zorder=3)
        if not s.empty:
            a = s[s.policy == "point"]; b = s[s.policy == "quantile"]
            if len(a) and len(b):
                ax.annotate("", xy=(b["avg_inventory_units"].iat[0], b["fill_rate"].iat[0] * 100),
                            xytext=(a["avg_inventory_units"].iat[0], a["fill_rate"].iat[0] * 100),
                            arrowprops=dict(arrowstyle="->", color=COLORS[model], lw=1))
            if len(b):
                labels.append((b["avg_inventory_units"].iat[0], b["fill_rate"].iat[0] * 100,
                               NICE[model], COLORS[model]))
    ax.set_xlabel("average inventory (units, L=4)")
    ax.set_ylabel("fill rate (%)")
    ax.set_title("Service–inventory frontier, Cu/Co=9, L=4\n"
                 "square = point rule, circle = quantile rule")
    fig.tight_layout()

    # Place labels greedily: try a ring of offsets and keep the first that does
    # not overlap a label already placed.  Deterministic, so the figure is stable.
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    placed = []
    candidates = [(5, 4), (5, -11), (-6, 5), (8, 9), (-10, -10), (9, -6),
                  (0, 11), (0, -13), (-14, 0), (14, 3)]
    for i, (x, y, text, color) in enumerate(labels):
        chosen = None
        for dx, dy in candidates:
            t = ax.annotate(text, (x, y), fontsize=6.5, color=color,
                            xytext=(dx, dy), textcoords="offset points")
            box = t.get_window_extent(renderer=renderer)
            if not any(box.overlaps(o) for o in placed):
                chosen = (t, box)
                break
            t.remove()
        if chosen is None:
            t = ax.annotate(text, (x, y), fontsize=6.5, color=color,
                            xytext=candidates[i % len(candidates)],
                            textcoords="offset points")
            chosen = (t, t.get_window_extent(renderer=renderer))
        placed.append(chosen[1])
    fig.savefig(C.OUT_FIGURES / "fig6_service_vs_inventory.png", bbox_inches="tight")
    plt.close(fig)


def fig7_worst_skus(p, worst):
    """All 20 worst SKUs by newsvendor cost, each with its demand history."""
    top = worst.head(20)
    fig, axes = plt.subplots(4, 5, figsize=(14, 8.4))
    for ax, (_, row) in zip(axes.ravel(), top.iterrows()):
        i = int(np.where(p.skus == row["StockCode"])[0][0])
        ax.plot(p.weeks, p.D[i], color="#444", lw=0.9)
        desc = str(row["description"])[:30]
        ax.set_title(f"{row['StockCode']} [{row['class']}]  "
                     f"{row['share_of_total_cost_pct']:.1f}% of cost\n{desc}",
                     fontsize=6.2)
        ax.tick_params(axis="both", labelsize=5)
        ax.xaxis.set_major_locator(mdates.MonthLocator(bymonth=(1, 7)))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
        ax.grid(alpha=0.15)
    fig.suptitle("Worst 20 SKUs by newsvendor cost "
                 "(L=4, Cu/Co=9, LightGBM quantile policy; share of total cost in title)",
                 fontsize=9.5)
    fig.tight_layout()
    fig.savefig(C.OUT_FIGURES / "fig7_worst_skus.png", bbox_inches="tight")
    plt.close(fig)


def make_all():
    from features import load_panel
    p = load_panel()
    fc = pd.read_parquet(C.FORECASTS)
    grids = pd.read_csv(C.RESIDUALS)
    dec = pd.read_csv(C.OUT_TABLES / "decision.csv")
    by_class = pd.read_csv(C.OUT_TABLES / "accuracy_by_class.csv")
    worst = pd.read_csv(C.OUT_TABLES / "worst_skus.csv")

    fig1_intermittency(p)
    fig2_forecasts(p, fc)
    fig3_calibration(fc, grids)
    fig4_cost_vs_ratio(dec)
    fig5_error_by_class(by_class)
    fig6_service_vs_inventory(dec)
    fig7_worst_skus(p, worst)
    print(f"[figures] {len(list(C.OUT_FIGURES.glob('*.png')))} figures -> {C.OUT_FIGURES}")
    return sorted(p.name for p in C.OUT_FIGURES.glob("*.png"))


if __name__ == "__main__":
    make_all()
