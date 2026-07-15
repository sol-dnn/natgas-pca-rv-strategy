"""Feature preprocessor for walk-forward ML — 0DTE protocol.

Single official ML preprocessor. It does NOT create features and does NOT
compute rolling/expanding or cross-sectional transformations. Those belong in
feature engineering.

At each walk-forward refit, all statistics are fitted on the TRAINING window
only, then frozen for train/test transforms:

  1. sanitize      : ±inf → NaN
  2. imputation    : dummies/cs → 0, ts/raw → train median, all-NaN → 0
  3. winsorisation : train quantile bounds for explicit winsor_cols only
  4. z-score       : ts_cols only, using train mean/std
  5. fallback      : any residual non-finite → 0, assert finite output

Column policy:
  dummy_cols : binary / one-hot indicators; fill 0; no winsor; no z-score
  cs_cols    : already CS-normalised (*_cs, *_timeslot_z); fill 0; optional winsor; no z-score
  ts_cols    : market-level / lagged / time-series features; fill train median; winsor; z-score
  raw_cols   : snapshot continuous features; fill train median; winsor; no z-score

No lookahead: fit() must receive training-window rows only.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np

log = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class PreprocessorConfig:
    """
    Column-group configuration for FeaturePreprocessor.

    Build from feature_catalog.categorize_features():

        from code.features.feature_catalog import categorize_features
        cat = categorize_features(feature_cols)
        cfg = PreprocessorConfig(
            dummy_cols  = cat["dummies"],
            cs_cols     = cat["cs"],
            ts_cols     = cat["ts"],
            raw_cols    = cat["raw"],
            winsor_cols = cat["ts"] + cat["raw"] + cat["cs"],
        )
    """

    dummy_cols: list[str] = field(default_factory=list)
    """Binary / one-hot indicators. Fill=0, no winsor, no z-score."""

    cs_cols: list[str] = field(default_factory=list)
    """Already CS-normalised (*_cs, *_timeslot_z). Fill=0, optional winsor, no z-score."""

    ts_cols: list[str] = field(default_factory=list)
    """Time-varying / lagged market features. Fill=train-median, winsor, z-score."""

    raw_cols: list[str] = field(default_factory=list)
    """Snapshot continuous features. Fill=train-median, winsor, no z-score."""

    winsor_cols: list[str] = field(default_factory=list)
    """Explicit whitelist of columns to winsorise. Dummies excluded regardless."""

    winsor_quantile: float = 0.025
    """Symmetric winsorisation quantile. Default 2.5%–97.5%."""

    impute: bool = True
    """Enable imputation stage. Set False only for debugging."""

    auto_raw_uncategorized: bool = True
    """
    Treat columns not in any category list as raw continuous features (with warning).
    Set False to raise ValueError on uncategorized columns.
    """

    error_on_overlap: bool = True
    """
    Raise ValueError if a column appears in more than one category list.
    Set False to use priority dummy > cs > ts > raw silently.
    """

    def to_dict(self) -> dict[str, Any]:
        """Serialise to a JSON-safe dict (for split_log metadata)."""
        return {
            "dummy_cols":             self.dummy_cols,
            "cs_cols":                self.cs_cols,
            "ts_cols":                self.ts_cols,
            "raw_cols":               self.raw_cols,
            "winsor_cols":            self.winsor_cols,
            "winsor_quantile":        self.winsor_quantile,
            "impute":                 self.impute,
            "auto_raw_uncategorized": self.auto_raw_uncategorized,
            "error_on_overlap":       self.error_on_overlap,
            "n_dummy":                len(self.dummy_cols),
            "n_cs":                   len(self.cs_cols),
            "n_ts":                   len(self.ts_cols),
            "n_raw":                  len(self.raw_cols),
        }


# ─────────────────────────────────────────────────────────────────────────────
# Preprocessor
# ─────────────────────────────────────────────────────────────────────────────

class FeaturePreprocessor:
    """
    Five-stage feature preprocessor for walk-forward ML.

    All statistics are fit on training data only and frozen for subsequent
    prediction steps — no lookahead.

    Usage
    -----
        pp = FeaturePreprocessor(cfg)
        X_tr_pp = pp.fit_transform(X_tr_raw, feature_cols)  # fit on train
        X_te_pp = pp.transform(X_te_raw)                    # apply to test

    After fit_transform(), `pp.fit_log` contains diagnostics.
    `pp.to_dict()` returns a JSON-safe summary for logging.
    """

    def __init__(self, config: PreprocessorConfig) -> None:
        self.config = config
        self._fitted = False

        # Column indices (set during fit)
        self._col_names: list[str] = []
        self._n_cols: int = 0
        self._dummy_idx:  np.ndarray = np.array([], dtype=int)
        self._cs_idx:     np.ndarray = np.array([], dtype=int)
        self._ts_idx:     np.ndarray = np.array([], dtype=int)
        self._raw_idx:    np.ndarray = np.array([], dtype=int)
        self._winsor_idx: np.ndarray = np.array([], dtype=int)

        # Fitted statistics (frozen after fit)
        self._fill:    np.ndarray | None = None
        self._lower:   np.ndarray | None = None
        self._upper:   np.ndarray | None = None
        self._ts_mean: np.ndarray | None = None
        self._ts_std:  np.ndarray | None = None

        # Diagnostics
        self.fit_log: dict[str, Any] = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(self, X: np.ndarray, col_names: list[str]) -> "FeaturePreprocessor":
        """Fit all preprocessing statistics on TRAINING data only."""
        self._validate_input(X, col_names)
        self._validate_quantile()

        self._col_names = list(col_names)
        self._n_cols    = X.shape[1]
        col_idx = {c: j for j, c in enumerate(col_names)}
        col_set = set(col_names)

        # Validate config lists reference existing columns
        self._validate_category_names(col_set)

        # Build disjoint groups (raises on overlap if error_on_overlap=True)
        groups = self._make_disjoint_groups(col_names)

        self._dummy_idx  = self._idx(groups["dummy"],  col_idx)
        self._cs_idx     = self._idx(groups["cs"],     col_idx)
        self._ts_idx     = self._idx(groups["ts"],     col_idx)
        self._raw_idx    = self._idx(groups["raw"],    col_idx)

        # Dummies are never winsorised regardless of winsor_cols
        dummy_set = set(groups["dummy"])
        self._winsor_idx = self._idx(
            [c for c in self.config.winsor_cols if c in col_set and c not in dummy_set],
            col_idx,
        )

        # ── Stage 0: sanitise (±inf → NaN) ───────────────────────────────────
        X_s = _inf_to_nan(X)
        n_nan_before = int(np.isnan(X_s).sum())
        n_inf_before = int(np.isinf(X).sum())

        # ── Stage 1: compute fill values (train median for ts/raw; 0 for dummy/cs) ─
        fill = np.zeros(self._n_cols, dtype=float)
        if self.config.impute:
            for j in np.concatenate([self._ts_idx, self._raw_idx]):
                fill[j] = _finite_median_or_zero(X_s[:, j])
        else:
            log.warning(
                "PreprocessorConfig.impute=False: NaN are not imputed before winsor/z-score. "
                "Use only for debugging."
            )
        self._fill = fill

        X_imp = _impute(X_s, fill) if self.config.impute else X_s.copy()
        n_imputed = int(np.isnan(X_s).sum()) - int(np.isnan(X_imp).sum())

        # ── Stage 2: winsor bounds (on imputed train data) ───────────────────
        self._lower = np.full(self._n_cols, -np.inf, dtype=float)
        self._upper = np.full(self._n_cols,  np.inf, dtype=float)
        if len(self._winsor_idx) > 0:
            # Compute bounds on X_s (pre-imputation) using nanquantile so the
            # bounds reflect the TRUE data distribution, not the imputed one.
            # Numerically identical to computing on X_imp (median injections never
            # shift the 2.5/97.5 tails), but cleaner intent.
            X_w = X_s[:, self._winsor_idx]
            lo  = np.nanquantile(X_w, self.config.winsor_quantile, axis=0)
            hi  = np.nanquantile(X_w, 1.0 - self.config.winsor_quantile, axis=0)
            # Guard: all-NaN column → no clipping
            lo = np.where(np.isfinite(lo), lo, -np.inf)
            hi = np.where(np.isfinite(hi), hi,  np.inf)
            self._lower[self._winsor_idx] = lo
            self._upper[self._winsor_idx] = hi

        X_win = np.clip(X_imp, self._lower, self._upper)

        # ── Stage 3: z-score stats for ts_cols only ───────────────────────────
        self._ts_mean = np.zeros(self._n_cols, dtype=float)
        self._ts_std  = np.ones(self._n_cols,  dtype=float)
        if len(self._ts_idx) > 0:
            X_ts = X_win[:, self._ts_idx]
            mu   = np.nanmean(X_ts, axis=0)
            sig  = np.nanstd(X_ts, axis=0)
            # Guard: NaN mean/std for all-NaN columns → neutral (0, 1)
            mu  = np.where(np.isfinite(mu),  mu,  0.0)
            sig = np.where(np.isfinite(sig) & (sig >= 1e-12), sig, 1.0)
            self._ts_mean[self._ts_idx] = mu
            self._ts_std[self._ts_idx]  = sig

        # ── Validate training output ──────────────────────────────────────────
        X_pp = self._apply_transform(X_s)
        n_nonfinite = int((~np.isfinite(X_pp)).sum())
        if n_nonfinite > 0:
            raise RuntimeError(
                f"Preprocessor produced {n_nonfinite} non-finite values on training data. "
                "Check for all-NaN columns or impute=False with NaN-only features."
            )

        self._validate_dummy_values(X_pp, col_names)
        self._validate_cs_not_ts(col_names)

        # ── Fit log ───────────────────────────────────────────────────────────
        n_train = X.shape[0]
        self.fit_log = {
            "n_train_rows":         n_train,
            "n_features":           self._n_cols,
            "n_dummy":              len(self._dummy_idx),
            "n_cs":                 len(self._cs_idx),
            "n_ts":                 len(self._ts_idx),
            "n_raw":                len(self._raw_idx),
            "n_winsor":             len(self._winsor_idx),
            "n_nan_before":         n_nan_before,
            "n_inf_before":         n_inf_before,
            "n_imputed":            n_imputed,
            "nan_rate_before":      round(n_nan_before / max(1, n_train * self._n_cols), 6),
            "n_nonfinite_after":    n_nonfinite,
            "winsor_quantile":      self.config.winsor_quantile,
            "uncategorized_as_raw": groups["uncategorized_as_raw"],
        }
        log.info(
            "Preprocessor fit: %d rows, %d features | NaN before=%.2f%% | "
            "imputed=%d | finite output ✓",
            n_train, self._n_cols,
            100.0 * self.fit_log["nan_rate_before"],
            n_imputed,
        )

        self._fitted = True
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Apply frozen train statistics to any data (train or test).
        Guaranteed finite output — raises RuntimeError if not.
        """
        if not self._fitted:
            raise RuntimeError("Call fit() before transform().")
        self._validate_input(X, self._col_names)
        out = self._apply_transform(_inf_to_nan(X))
        n_bad = int((~np.isfinite(out)).sum())
        if n_bad > 0:
            raise RuntimeError(f"transform() produced {n_bad} non-finite values — preprocessor bug.")
        return out

    def fit_transform(self, X: np.ndarray, col_names: list[str]) -> np.ndarray:
        """Fit on X and return the preprocessed version of X."""
        return self.fit(X, col_names).transform(X)

    def to_dict(self) -> dict[str, Any]:
        """JSON-safe summary of fitted statistics for logging and reproducibility."""
        if not self._fitted:
            raise RuntimeError("Call fit() before to_dict().")

        def _json_arr(a: np.ndarray | None) -> list[Any]:
            if a is None:
                return []
            return [None if not np.isfinite(v) else float(v) for v in a]

        return {
            "config":   self.config.to_dict(),
            "fit_log":  self.fit_log,
            "col_names": self._col_names,
            "fill":     _json_arr(self._fill),
            "lower":    _json_arr(self._lower),
            "upper":    _json_arr(self._upper),
            "ts_mean":  _json_arr(self._ts_mean),
            "ts_std":   _json_arr(self._ts_std),
        }

    # ------------------------------------------------------------------
    # Internal methods
    # ------------------------------------------------------------------

    def _apply_transform(self, X_nan: np.ndarray) -> np.ndarray:
        """Apply all five stages to a sanitised (inf→NaN) array."""
        if (self._fill is None or self._lower is None
                or self._upper is None or self._ts_mean is None or self._ts_std is None):
            raise RuntimeError("Preprocessor statistics not initialised — call fit() first.")

        # Stage 1 — impute
        out = _impute(X_nan, self._fill) if self.config.impute else X_nan.copy()

        # Stage 2 — winsorise
        out = np.clip(out, self._lower, self._upper)

        # Stage 3 — z-score ts_cols only
        if len(self._ts_idx) > 0:
            out = out.copy()
            out[:, self._ts_idx] = (
                (out[:, self._ts_idx] - self._ts_mean[self._ts_idx])
                / self._ts_std[self._ts_idx]
            )

        # Stage 4 — final fallback: residual NaN/inf → 0
        bad = ~np.isfinite(out)
        if bad.any():
            out = out.copy()
            out[bad] = 0.0

        return out

    def _make_disjoint_groups(self, col_names: list[str]) -> dict[str, list]:
        """Assign each column to exactly one group with priority dummy > cs > ts > raw."""
        cfg = self.config
        memberships: dict[str, list[str]] = {c: [] for c in col_names}
        for group_name, cols in {
            "dummy": cfg.dummy_cols, "cs": cfg.cs_cols,
            "ts":    cfg.ts_cols,    "raw": cfg.raw_cols,
        }.items():
            for c in cols:
                if c in memberships:
                    memberships[c].append(group_name)

        overlaps = {c: gs for c, gs in memberships.items() if len(gs) > 1}
        if overlaps and cfg.error_on_overlap:
            sample = dict(list(overlaps.items())[:10])
            raise ValueError(f"Columns appear in multiple preprocessing groups: {sample}")
        elif overlaps:
            log.warning(
                "%d columns appear in multiple groups; using priority dummy>cs>ts>raw.",
                len(overlaps),
            )

        dummy = [c for c in col_names if "dummy" in memberships[c]]
        cs    = [c for c in col_names
                 if "dummy" not in memberships[c] and "cs" in memberships[c]]
        ts    = [c for c in col_names
                 if not ({"dummy","cs"} & set(memberships[c])) and "ts" in memberships[c]]
        raw   = [c for c in col_names
                 if not ({"dummy","cs","ts"} & set(memberships[c])) and "raw" in memberships[c]]

        categorized = set(dummy) | set(cs) | set(ts) | set(raw)
        uncategorized = [c for c in col_names if c not in categorized]

        if uncategorized:
            if cfg.auto_raw_uncategorized:
                raw.extend(uncategorized)
                log.warning(
                    "%d uncategorized columns treated as raw: %s%s",
                    len(uncategorized),
                    uncategorized[:10],
                    " ..." if len(uncategorized) > 10 else "",
                )
            else:
                raise ValueError(
                    f"Uncategorized feature columns (set auto_raw_uncategorized=True to treat as raw): "
                    f"{uncategorized[:20]}" + (" ..." if len(uncategorized) > 20 else "")
                )

        return {"dummy": dummy, "cs": cs, "ts": ts, "raw": raw,
                "uncategorized_as_raw": uncategorized if cfg.auto_raw_uncategorized else []}

    def _validate_category_names(self, col_set: set[str]) -> None:
        """Raise if any config list references columns absent from feature_cols."""
        cfg = self.config
        for name, cols in {
            "dummy_cols":  cfg.dummy_cols,
            "cs_cols":     cfg.cs_cols,
            "ts_cols":     cfg.ts_cols,
            "raw_cols":    cfg.raw_cols,
            "winsor_cols": cfg.winsor_cols,
        }.items():
            unknown = sorted(set(cols) - col_set)
            if unknown:
                raise ValueError(
                    f"{name} contains columns not in feature_cols: {unknown[:20]}"
                    + (" ..." if len(unknown) > 20 else "")
                )

    def _validate_quantile(self) -> None:
        q = self.config.winsor_quantile
        if not (0.0 <= q < 0.5):
            raise ValueError(f"winsor_quantile must be in [0, 0.5), got {q}.")

    def _validate_dummy_values(self, X_pp: np.ndarray, col_names: list[str]) -> None:
        if len(self._dummy_idx) == 0:
            return
        vals = X_pp[:, self._dummy_idx]
        finite = vals[np.isfinite(vals)]
        if finite.size == 0:
            return
        mn, mx = float(np.min(finite)), float(np.max(finite))
        if mn < -1e-9 or mx > 1.0 + 1e-9:
            names = [col_names[j] for j in self._dummy_idx]
            log.warning(
                "Dummy columns are not all in [0,1] after preprocessing "
                "(min=%.3f, max=%.3f). Affected: %s. Check dummy_cols config.",
                mn, mx, names[:10],
            )

    def _validate_cs_not_ts(self, col_names: list[str]) -> None:
        overlap_idx = sorted(set(self._cs_idx.tolist()) & set(self._ts_idx.tolist()))
        if overlap_idx:
            names = [col_names[j] for j in overlap_idx]
            raise ValueError(
                f"cs_cols and ts_cols overlap — these columns would be z-scored "
                f"despite being already CS-normalised: {names}"
            )

    @staticmethod
    def _idx(cols: list[str], col_idx: dict[str, int]) -> np.ndarray:
        return np.array([col_idx[c] for c in cols], dtype=int)

    @staticmethod
    def _validate_input(X: np.ndarray, col_names: list[str]) -> None:
        if X.ndim != 2:
            raise ValueError(f"X must be 2-D, got shape {X.shape}")
        if len(col_names) != X.shape[1]:
            raise ValueError(
                f"col_names has {len(col_names)} entries but X has {X.shape[1]} columns"
            )


# ─────────────────────────────────────────────────────────────────────────────
# Private helpers
# ─────────────────────────────────────────────────────────────────────────────

def _inf_to_nan(X: np.ndarray) -> np.ndarray:
    """Return a float copy with ±inf replaced by NaN."""
    out = X.astype(float, copy=True)
    out[np.isinf(out)] = np.nan
    return out


def _finite_median_or_zero(x: np.ndarray) -> float:
    """Median of finite values, or 0.0 if all NaN."""
    finite = x[np.isfinite(x)]
    return float(np.median(finite)) if finite.size > 0 else 0.0


def _impute(X: np.ndarray, fill: np.ndarray) -> np.ndarray:
    """Fill NaN cells with per-column fill values."""
    out = X.copy()
    nan_rows, nan_cols = np.where(np.isnan(out))
    if len(nan_rows) > 0:
        out[nan_rows, nan_cols] = fill[nan_cols]
    return out
