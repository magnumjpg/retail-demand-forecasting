"""Forecasters: baselines, count GLM and LightGBM, plus the calibration layer.

All models produce a *one-week-ahead* (or h-week-ahead) demand mean for each
SKU.  Uncertainty is added by a shared, model-agnostic layer, so that the
newsvendor comparison isolates the quality of the conditional mean:

1. variance-stabilising scaling ``s(mu) = sqrt(mu + 0.25)``.  For counts the
   standard deviation grows roughly with ``sqrt(mu)``; the ``+0.25`` keeps the
   scale finite when the mean forecast is zero.  (An earlier NB2 variance
   function ``mu + alpha * mu^2`` was tried and rejected: a single global
   ``alpha`` collapsed the P50 to zero and destroyed coverage - see
   DECISIONS.md.)
2. split-conformal recalibration of the standardised residual
   ``z = (y - mu) / s(mu)`` on the ``CALIB_WEEKS`` weeks immediately before
   the forecast origin (the model itself is fitted only up to
   ``origin - CALIB_WEEKS``), separately per demand class (Mondrian
   conformal).

Predictive quantiles are ``mu + s(mu) * Q_z(q)``, clipped at zero.  Because
``Q_z`` is an empirical quantile function, the intervals are conformal by
construction, and the same machinery is applied to every model.  For the
newsvendor a separate grid is built on *cumulative* L-week demand.
"""
from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config as C
from features import COUNT_FEATURES, FEATURES, Panel

BASELINES = ("mean", "seasonal_naive", "ses", "croston_sba")
ML_MODELS = ("glm_poisson", "lgbm_tweedie", "lgbm_quantile")
MODELS = BASELINES + ML_MODELS
QGRID = np.linspace(0.005, 0.995, C.N_QUANTILE_GRID)


# ------------------------------------------------------------------ helpers
def _glm_design(frame: pd.DataFrame) -> np.ndarray:
    """Log1p on count-like columns: a Poisson GLM is multiplicative, and the
    raw unit counts span four orders of magnitude."""
    X = frame[list(FEATURES)].to_numpy(dtype=np.float64)
    cols = [list(FEATURES).index(c) for c in COUNT_FEATURES]
    X[:, cols] = np.log1p(np.clip(X[:, cols], 0, None))
    return X


def ses_forecast(hist: np.ndarray, alphas=(0.05, 0.1, 0.2, 0.3, 0.5)) -> np.ndarray:
    """Per-SKU SES with alpha chosen by in-sample SSE (vectorised over SKUs)."""
    n, T = hist.shape
    a = np.asarray(alphas, dtype=float)
    levels = np.repeat(hist[:, :1], len(a), axis=1)        # (n, k)
    sse = np.zeros((n, len(a)))
    for t in range(1, T):
        y = hist[:, t]
        sse += (y[:, None] - levels) ** 2
        levels = a[None, :] * y[:, None] + (1 - a[None, :]) * levels
    best = a[np.argmin(sse, axis=1)]
    level = hist[:, 0].copy()
    for t in range(1, T):
        level = best * hist[:, t] + (1 - best) * level
    return np.clip(level, 0, None)


def croston_sba(hist: np.ndarray, alpha: float = 0.1) -> np.ndarray:
    """Croston with the Syntetos-Boylan approximation, vectorised over SKUs."""
    n, T = hist.shape
    z = np.zeros(n)          # smoothed demand size
    p = np.zeros(n)          # smoothed inter-arrival interval
    q = np.zeros(n)          # periods since last sale
    started = np.zeros(n, dtype=bool)
    for t in range(T):
        y = hist[:, t]
        sale = y > 0
        first = sale & ~started
        z[first] = y[first]
        p[first] = 1.0
        started |= first
        upd = sale & started & ~first
        if upd.any():
            q[upd] += 1.0
            z[upd] += alpha * (y[upd] - z[upd])
            p[upd] += alpha * (q[upd] - p[upd])
            q[upd] = 0.0
        q[~sale] += 1.0
    fallback = hist.mean(axis=1)
    out = np.where(p > 0, (1 - alpha / 2) * z / np.maximum(p, 1e-9), fallback)
    return np.clip(out, 0, None)


def baseline_mu(panel: Panel, origin: int, horizons) -> dict[str, np.ndarray]:
    """Baseline forecasts, shape (n_sku, n_horizons)."""
    hist = panel.D[:, : origin + 1]
    n, T = hist.shape
    m_mean = hist.mean(axis=1)
    ses = ses_forecast(hist)
    crs = croston_sba(hist)
    out = {}
    for h_pos, h in enumerate(horizons):
        sn_idx = origin + h - 52
        sn = hist[:, sn_idx] if sn_idx >= 0 else m_mean.copy()
        out.setdefault("mean", []).append(m_mean)
        out.setdefault("seasonal_naive", []).append(sn)
        out.setdefault("ses", []).append(ses)
        out.setdefault("croston_sba", []).append(crs)
    return {k: np.column_stack(v) for k, v in out.items()}


# ------------------------------------------------------------------- sklearn
@dataclass
class MLForecaster:
    panel: Panel
    origin: int
    calib_weeks: int = C.CALIB_WEEKS
    step: int = 2
    models: dict = field(default_factory=dict)
    scaler: object = None

    def __post_init__(self):
        self.fit_end = self.origin - self.calib_weeks
        tr = self.panel.training_frame(self.fit_end, C.HORIZONS, step=self.step)
        tr = tr[tr["StockCode"].isin(self.panel.model_skus)]
        self.n_train = len(tr)
        X = tr[list(FEATURES)].to_numpy(dtype=np.float32)
        y = tr["y"].to_numpy(dtype=np.float64)
        self._fit_glm(_glm_design(tr), y)
        self._fit_lgbm(X, y)

    # ------------------------------------------------------------------ glm
    def _fit_glm(self, X, y):
        from sklearn.linear_model import PoissonRegressor
        from sklearn.preprocessing import StandardScaler
        self.scaler = StandardScaler().fit(X)
        glm = PoissonRegressor(alpha=0.5, max_iter=400, tol=1e-5)
        glm.fit(self.scaler.transform(X), np.clip(y, 0, None))
        self.models["glm_poisson"] = glm

    # ----------------------------------------------------------------- lgbm
    def _fit_lgbm(self, X, y):
        import lightgbm as lgb
        base = dict(n_estimators=400, learning_rate=0.04, num_leaves=31,
                    min_child_samples=50, subsample=0.8, subsample_freq=1,
                    colsample_bytree=0.8, reg_lambda=1.0, n_jobs=-1,
                    random_state=C.SEED, verbose=-1, deterministic=True,
                    force_col_wise=True)
        tw = lgb.LGBMRegressor(objective="tweedie", tweedie_variance_power=1.3, **base)
        tw.fit(X, np.clip(y, 0, None))
        self.models["lgbm_tweedie"] = tw
        q50 = lgb.LGBMRegressor(objective="quantile", alpha=0.5, **base)
        q50.fit(X, np.clip(y, 0, None))
        self.models["lgbm_quantile"] = q50

    # -------------------------------------------------------------- predict
    def predict(self, frame: pd.DataFrame) -> dict[str, np.ndarray]:
        X = frame[list(FEATURES)].to_numpy(dtype=np.float32)
        out = {}
        out["glm_poisson"] = self.models["glm_poisson"].predict(
            self.scaler.transform(_glm_design(frame)))
        for name in ("lgbm_tweedie", "lgbm_quantile"):
            out[name] = self.models[name].predict(X)
        return {k: np.clip(v, 0, None) for k, v in out.items()}


# ------------------------------------------------------- calibration layer
# Variance-stabilising scaling: for count data the standard deviation grows
# roughly with sqrt(mu), so residuals are standardised as
#     z = (y - mu) / sqrt(mu + 0.25)
# The 0.25 keeps the scale finite at mu = 0 (a zero forecast still needs a
# non-degenerate upper quantile).  The empirical quantiles of z are taken on
# the conformal window, separately per demand class (Mondrian conformal), so
# the intervals are calibrated conditional on the segment.
def standardise(y: np.ndarray, mu: np.ndarray) -> np.ndarray:
    s = np.sqrt(np.maximum(mu, 0) + 0.25)
    return (y - mu) / s


def predictive_quantiles(mu: np.ndarray, zgrid: np.ndarray, qs) -> np.ndarray:
    s = np.sqrt(np.maximum(mu, 0) + 0.25)
    out = []
    for q in np.atleast_1d(qs):
        z = np.interp(q, QGRID, zgrid)
        out.append(np.clip(mu + s * z, 0, None))
    return np.column_stack(out)


def calibrate(y: np.ndarray, mu: np.ndarray) -> np.ndarray:
    return np.quantile(standardise(y, mu), QGRID)


if __name__ == "__main__":
    from features import load_panel, model_mask
    p = load_panel()
    m = model_mask(p)
    o = p.origin0
    b = baseline_mu(p, o, (1, 2, 4))
    for k, v in b.items():
        print(k, v.shape, np.round(v[m, 0].mean(), 2))
    f = p.frame(o, (1, 2, 4))
    fc = MLForecaster(p, o)
    print("train rows", fc.n_train, "fit_end", fc.fit_end)
    ml = fc.predict(f)
    for k, v in ml.items():
        print(k, f"mean {v.mean():.2f}  (actual mean {f['y'].mean():.2f})")

