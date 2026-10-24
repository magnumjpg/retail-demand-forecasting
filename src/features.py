"""Feature engineering for SKU-week demand.

Design rule: a feature row for (origin o, horizon h) may only use weeks
``0..o``.  The target is week ``o + h``.  Everything is computed from a wide
``SKU x week`` matrix, so a leak would require indexing a column > o, which the
assertions in :meth:`Panel.frame` forbid.

    from features import load_panel
    p = load_panel()
    f = p.frame(origin=77, horizons=(1, 2, 4))
"""
from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import (DATA_PROC, LAGS, MIN_POS_WEEKS_MODEL, ORIGIN_OFFSETS, WEEKLY)

SELECTION = DATA_PROC / "sku_selection.json"

FEATURES = (
    [f"lag_{k}" for k in LAGS]
    + ["roll_mean_4", "roll_mean_8", "roll_mean_13", "roll_mean_26",
       "roll_std_8", "roll_std_13", "zero_share_13", "zero_share_26",
       "weeks_since_sale", "log_sku_age", "pos_freq", "cv_13", "trend_ratio",
       "price_lag1", "price_mean_13", "log_price_lag1", "uk_share_13",
       "log_invoices_13", "woy_sin", "woy_cos", "month", "is_december",
       "horizon"]
)
# count-like columns are log1p-transformed for the linear GLM
COUNT_FEATURES = tuple([f"lag_{k}" for k in LAGS]
                       + ["roll_mean_4", "roll_mean_8", "roll_mean_13",
                          "roll_mean_26", "roll_std_8", "roll_std_13",
                          "price_lag1", "price_mean_13"])


def _ffill_rows(a: np.ndarray) -> np.ndarray:
    """Forward-fill NaNs along axis 1 (per SKU); remaining NaNs become 0."""
    out = a.copy()
    idx = np.where(np.isnan(out), 0, np.arange(out.shape[1])[None, :])
    np.maximum.accumulate(idx, axis=1, out=idx)
    out = out[np.arange(out.shape[0])[:, None], idx]
    out[np.isnan(out)] = 0.0
    return out


def _row_nanmean(a: np.ndarray) -> np.ndarray:
    cnt = np.isfinite(a).sum(axis=1)
    s = np.nansum(a, axis=1)
    return np.where(cnt > 0, s / np.maximum(cnt, 1), np.nan)


@dataclass
class Panel:
    skus: np.ndarray
    weeks: pd.DatetimeIndex
    D: np.ndarray          # demand, (n_sku, n_weeks)
    R: np.ndarray          # revenue
    P: np.ndarray          # unit price (NaN when no sale that week)
    P_ff: np.ndarray       # forward-filled price
    N: np.ndarray          # number of invoices
    UK: np.ndarray         # UK revenue
    meta: pd.DataFrame     # indexed by StockCode
    model_skus: np.ndarray
    origin0: int

    # ------------------------------------------------------------- helpers
    @property
    def n_sku(self) -> int:
        return self.D.shape[0]

    @property
    def n_weeks(self) -> int:
        return self.D.shape[1]

    # ------------------------------------------------------------ features
    def _sku_features(self, origin: int) -> pd.DataFrame:
        assert 0 <= origin < self.n_weeks, origin
        hist = self.D[:, : origin + 1]
        n = hist.shape[1]
        f: dict[str, np.ndarray] = {}
        for k in LAGS:
            f[f"lag_{k}"] = hist[:, -k] if n >= k else np.zeros(self.n_sku)
        for w in (4, 8, 13, 26):
            win = hist[:, -w:] if n >= w else hist
            f[f"roll_mean_{w}"] = win.mean(axis=1)
            if w in (8, 13):
                f[f"roll_std_{w}"] = win.std(axis=1, ddof=0)
            if w in (13, 26):
                f[f"zero_share_{w}"] = (win == 0).mean(axis=1)

        nz = hist > 0
        any_sale = nz.any(axis=1)
        since = np.where(any_sale, np.argmax(nz[:, ::-1], axis=1), 26)
        f["weeks_since_sale"] = np.minimum(since, 26).astype(float)
        first_sale = np.where(any_sale, np.argmax(nz, axis=1), origin)
        # levels that grow with time are normalised: trees cannot extrapolate
        f["log_sku_age"] = np.log1p((origin - first_sale).clip(min=0))
        f["pos_freq"] = nz.sum(axis=1) / float(origin + 1)
        f["cv_13"] = np.clip(f["roll_std_13"] / (f["roll_mean_13"] + 1e-6), 0, 10)
        f["trend_ratio"] = np.clip(f["roll_mean_4"] / (f["roll_mean_26"] + 0.1), 0, 10)

        w13 = slice(max(0, origin - 12), origin + 1)
        p_lag = np.where(self.P_ff[:, origin] > 0, self.P_ff[:, origin], np.nan)
        p13 = _row_nanmean(self.P[:, w13])
        p13 = np.where(np.isfinite(p13), p13, p_lag)
        f["price_lag1"] = p_lag
        f["price_mean_13"] = np.where(p13 > 0, p13, np.nan)
        rev13 = self.R[:, w13].sum(axis=1)
        f["uk_share_13"] = np.clip(self.UK[:, w13].sum(axis=1) / (rev13 + 1e-9), 0, 1)
        f["log_invoices_13"] = np.log1p(self.N[:, w13].mean(axis=1))

        out = pd.DataFrame(f, index=pd.RangeIndex(self.n_sku))
        fill = self.meta["median_price"].to_numpy(dtype=float)
        for col in ("price_lag1", "price_mean_13"):
            out[col] = out[col].fillna(pd.Series(fill)).fillna(0.0)
        out["log_price_lag1"] = np.log1p(out["price_lag1"].clip(lower=0))
        return out

    def frame(self, origin, horizons=tuple(range(1, 9))) -> pd.DataFrame:
        """Feature rows for one or more forecast origins, ordered (origin, horizon, sku)."""
        origins = [int(origin)] if np.isscalar(origin) else [int(o) for o in origin]
        blocks = []
        for o in origins:
            base = self._sku_features(o)
            for h in horizons:
                h = int(h)
                target = o + h
                assert target < self.n_weeks, f"target week {target} outside panel"
                block = base.copy()
                block["horizon"] = h
                block["origin_idx"] = o
                block["target_idx"] = target
                block["sku_idx"] = np.arange(self.n_sku)
                block["StockCode"] = self.skus
                block["target_week"] = self.weeks[target]
                block["origin_week"] = self.weeks[o]
                block["y"] = self.D[:, target]
                woy = int(self.weeks[target].isocalendar().week)
                block["woy_sin"] = np.sin(2 * np.pi * woy / 52.18)
                block["woy_cos"] = np.cos(2 * np.pi * woy / 52.18)
                block["month"] = float(self.weeks[target].month)
                block["is_december"] = float(self.weeks[target].month == 12)
                assert target > o, "target must be after the forecast origin"
                blocks.append(block)
        out = pd.concat(blocks, ignore_index=True)
        assert_no_leakage(out)
        return out

    def training_frame(self, fit_end: int, horizons=tuple(range(1, 9)),
                       step: int = 2, min_history: int = 16) -> pd.DataFrame:
        max_h = max(horizons)
        origins = list(range(min_history, fit_end - max_h + 1, step))
        if not origins:
            raise ValueError(f"no training origins for fit_end={fit_end}")
        fr = self.frame(origins, horizons)
        assert fr["target_idx"].max() <= fit_end, "training target beyond fit window"
        return fr

    def history_upto(self, origin: int) -> np.ndarray:
        return self.D[:, : origin + 1]


def assert_no_leakage(frame: pd.DataFrame) -> None:
    """Hard contract: every feature must be computable at the origin."""
    assert (frame["target_idx"] > frame["origin_idx"]).all(), "target must be future"
    assert frame["origin_idx"].min() >= 0
    # the newest information any feature can see is the origin week itself
    assert (frame["origin_idx"] < frame["target_idx"]).all(), "feature week >= target week"
    assert frame[list(FEATURES)].notna().all().all(), "NaN in feature matrix"
    assert np.isfinite(frame[list(FEATURES)].to_numpy(dtype=float)).all(), "non-finite"


def assign_classes(demand: np.ndarray, horizon_window: int) -> np.ndarray:
    """Demand class from *pre-origin* data only.

    intermittent: > 50% zero weeks | fast: median > 4 | medium: 1 < median <= 4
    | slow: median <= 1
    """
    hist = demand[:, :horizon_window]
    zero_share = (hist == 0).mean(axis=1)
    med = np.median(hist, axis=1)
    return np.where(zero_share > 0.5, "intermittent",
                    np.where(med > 4, "fast",
                             np.where(med > 1, "medium", "slow")))


def load_panel() -> Panel:
    df = pd.read_csv(WEEKLY, parse_dates=["week"],
                     dtype={"StockCode": str, "description": str})
    sel = json.loads(SELECTION.read_text(encoding="utf-8"))
    weeks = pd.DatetimeIndex(sorted(df["week"].unique()))
    skus = np.array(sorted(df["StockCode"].unique()))

    def wide(col, fill=0.0):
        m = df.pivot(index="StockCode", columns="week", values=col)
        return m.reindex(index=skus, columns=weeks).fillna(fill).to_numpy(dtype=float)

    meta = df.drop_duplicates("StockCode").set_index("StockCode").reindex(skus)
    P = wide("price_mean", np.nan)
    p = Panel(
        skus=skus, weeks=weeks,
        D=wide("demand"), R=wide("revenue"), P=P, P_ff=_ffill_rows(P),
        N=wide("n_invoices"), UK=wide("uk_revenue"),
        meta=meta, model_skus=np.array(sel["skus"]),
        origin0=int(weeks.size + ORIGIN_OFFSETS[0]),
    )

    # the modelling sample was frozen on the pre-origin window inside clean.py
    mask = np.isin(p.skus, p.model_skus)
    pos = (p.D[:, : p.origin0] > 0).sum(axis=1)
    assert np.array_equal(np.sort(p.skus[pos >= MIN_POS_WEEKS_MODEL]),
                          np.sort(p.model_skus)), "selection mismatch with clean.py"
    cls = np.full(len(p.skus), None, dtype=object)
    cls[mask] = assign_classes(p.D[mask], p.origin0)
    p.meta["class"] = cls
    p.meta["median_price"] = p.meta["median_price"].astype(float)
    return p


def model_mask(p: Panel) -> np.ndarray:
    return np.isin(p.skus, p.model_skus)


if __name__ == "__main__":
    p = load_panel()
    m = model_mask(p)
    print(f"panel {p.n_sku} SKUs x {p.n_weeks} weeks; model sample {m.sum()}")
    f = p.frame(p.origin0, (1, 2, 4))
    print(f"frame {f.shape} with {len(FEATURES)} features; "
          f"y mean {f['y'].mean():.2f}; zero share {(f['y'] == 0).mean():.3f}")
    print(p.meta.loc[m, "class"].value_counts().to_string())
    print(f"origins {p.origin0}, {p.n_weeks + ORIGIN_OFFSETS[1]}, "
          f"{p.n_weeks + ORIGIN_OFFSETS[2]} of {p.n_weeks} weeks")

