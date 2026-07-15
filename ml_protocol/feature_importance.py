"""Feature importance analysis for walk-forward ML — 0DTE protocol.

Computes importances across all walk-forward refit folds, then aggregates
with mean ± std and a cross-fold stability t-stat.

Three importance types:
  coef          — logistic regression coefficients (signed), with
                  within-fold t-statistics from the Hessian (Fisher information)
  mdi           — tree MDI (mean decrease in impurity), always non-negative
  perm          — permutation importance on the TEST fold (honest OOS estimate)

Permutation importance is always computed on the TEST data, not training data.
This is the most conservative and honest importance estimate.

Optional:
  shap          — SHAP values (requires `pip install shap`); subsampled for speed

Stability t-stat across folds:
  stability_t = mean_importance / (std_importance / sqrt(n_folds))
  Under H0: importance = 0, this tests whether the mean importance is
  consistently non-zero across refits. High stability_t = robust feature.
"""
from __future__ import annotations

import logging
import warnings
from dataclasses import dataclass, field
from typing import Literal

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.inspection import permutation_importance

log = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Result container
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class FeatureImportanceSummary:
    """
    Aggregated feature importances from a walk-forward experiment.

    Attributes
    ----------
    summary      : main table — one row per feature, sorted by |perm_mean|.
    fold_details : per-fold importances (long format) for diagnostic plots.
    model_type   : detected model family ("logistic", "tree", "other").
    n_folds      : number of refit steps used.
    model_id     : model identifier passed to collect_feature_importances().
    """
    summary:      pd.DataFrame
    fold_details: pd.DataFrame
    model_type:   Literal["logistic", "tree", "other"]
    n_folds:      int
    model_id:     str

    def top(self, n: int = 30, by: str = "perm_mean") -> pd.DataFrame:
        """Return top-n features by absolute importance."""
        _priority = ["perm_mean", "coef_mean", "mdi_mean", "shap_mean_abs"]
        col = by if by in self.summary.columns else next(
            (c for c in _priority if c in self.summary.columns), None
        )
        if col is None:
            return self.summary.head(n)
        return (
            self.summary
            .assign(_abs=self.summary[col].abs())
            .sort_values("_abs", ascending=False)
            .drop(columns="_abs")
            .head(n)
        )

    def print_top(self, n: int = 30, by: str = "perm_mean") -> None:
        """Print top-n feature importance table."""
        df = self.top(n=n, by=by)
        # Select columns that exist
        cols = [c for c in [
            "feature", "perm_mean", "perm_std", "stability_t", "sig",
            "coef_mean", "coef_t_hessian", "mdi_mean",
        ] if c in df.columns]
        print(f"\n{'─'*70}")
        print(f"  Feature importance  ({self.model_id}, {self.model_type}, "
              f"{self.n_folds} folds, top {n} by |{by}|)")
        print(f"{'─'*70}")
        print(df[cols].to_string(index=False, float_format=lambda x: f"{x:+.4f}"))
        print(f"{'─'*70}")
        print("  sig: *** p<0.001  ** p<0.01  * p<0.05  . p<0.10  (cross-fold stability t-stat)")


# ─────────────────────────────────────────────────────────────────────────────
# Main entry point
# ─────────────────────────────────────────────────────────────────────────────

def collect_feature_importances(
    data:          pd.DataFrame,
    feature_cols:  list[str],
    target_col:    str,
    model_id:      str,
    model_spec,                  # ModelSpec
    config,                      # WalkForwardConfig
    *,
    n_permutations: int  = 30,
    perm_scoring:   str  = "roc_auc",
    compute_shap:   bool = False,
    shap_subsample: int  = 500,
    random_state:   int  = 0,
) -> FeatureImportanceSummary:
    """
    Re-run the walk-forward loop for one model, collecting importances per fold.

    Parameters
    ----------
    data, feature_cols, target_col, model_spec, config
        Same as walk_forward().  Only one model is analysed (pass the winner).
    n_permutations
        Number of permutation repeats per fold (more → lower variance, slower).
        30 is a good balance for research.
    perm_scoring
        sklearn scoring string for permutation importance.
        "roc_auc" for binary classification (default).
    compute_shap
        Compute SHAP values (requires shap package).  Slow — subsampled.
    shap_subsample
        Max rows of test data per fold to use for SHAP (for speed).
    random_state
        Seed for permutation importance reproducibility.

    Returns
    -------
    FeatureImportanceSummary with summary table and per-fold details.
    """
    from .preprocessor import FeaturePreprocessor
    from .splitter import TimeSeriesSplitter

    cfg       = config
    pp_cfg    = cfg.preprocessor_config
    splitter  = TimeSeriesSplitter(cfg.split_config)
    date_col  = cfg.date_col

    # ── Sort data, build date → row mapping ──────────────────────────────────
    sort_cols = [date_col] + ([cfg.group_col] if cfg.group_col in data.columns else [])
    sort_cols += [c for c in cfg.sort_extra_cols if c in data.columns and c not in sort_cols]
    data = data.sort_values(sort_cols).reset_index(drop=True)

    date_groups   = data.groupby(date_col, sort=True, observed=True).size()
    dates_unique  = date_groups.index.to_numpy()
    counts        = date_groups.to_numpy(dtype=int)
    date_stops    = np.cumsum(counts)
    date_starts   = np.r_[0, date_stops[:-1]]

    X_all   = data[feature_cols].to_numpy(dtype=float)
    y_all   = data[target_col].to_numpy(dtype=float)
    y_bin_all = np.where(np.isfinite(y_all),
                         (y_all > cfg.binary_threshold).astype(float), np.nan)

    model_type = _detect_model_type(model_spec.estimator)
    log.info("Collecting importances: model=%s (%s), perm_n=%d, shap=%s",
             model_id, model_type, n_permutations, compute_shap)

    # ── Per-fold storage (one entry per completed refit cycle) ──────────────────
    # A "fold" = one refit cycle = [refit_date, next_refit_date).
    # coef / coef_t / mdi : model-level, set at fit time (training data only).
    # perm / shap         : data-level, computed on ALL test days in the cycle.
    fold_coef:        list[np.ndarray] = []
    fold_coef_t:      list[np.ndarray] = []
    fold_mdi:         list[np.ndarray] = []
    fold_perm:        list[np.ndarray] = []
    fold_shap:        list[np.ndarray] = []
    fold_refit_dates: list             = []

    nan_vec = np.full(len(feature_cols), np.nan)

    # ── Current cycle state — reset at each refit step ──────────────────────────
    # Model quantities (set at fit time, read when cycle is committed).
    _cur_estimator:       object | None     = None
    _cur_pp:              object | None     = None
    _cur_X_tr_pp:         np.ndarray | None = None   # needed for SHAP background
    _cur_refit_date:      object | None     = None
    _cur_coef:            np.ndarray        = nan_vec.copy()
    _cur_coef_t:          np.ndarray        = nan_vec.copy()
    _cur_mdi:             np.ndarray        = nan_vec.copy()
    # Accumulated test data for the current cycle (grows each non-refit step).
    _cur_X_te: list[np.ndarray] = []
    _cur_y_te: list[np.ndarray] = []

    def _commit_cycle() -> None:
        """
        Flush the current cycle: compute perm/shap on ALL accumulated test data,
        then append coef/mdi/perm/shap to the fold lists.

        Called at the START of every new refit step (before the new model is fit)
        and once more after the loop ends to capture the last cycle.
        Uses Python closure to read the current cycle's state variables.
        Reads only — never reassigns outer names (no nonlocal needed).
        """
        if _cur_estimator is None or not _cur_X_te:
            return

        X_te = np.vstack(_cur_X_te)        # shape: (all_cycle_rows, n_features)
        y_te = np.concatenate(_cur_y_te)   # shape: (all_cycle_rows,)
        valid = np.isfinite(y_te)

        # Permutation importance — requires ≥ 3 rows and both classes present
        if valid.sum() >= 3 and len(np.unique(y_te[valid])) >= 2:
            perm = _permutation_imp(
                _cur_estimator, X_te[valid], y_te[valid].astype(int),
                scoring=perm_scoring, n_repeats=n_permutations,
                random_state=random_state,
            )
        else:
            log.warning(
                "Cycle %s: insufficient test data for perm importance "
                "(%d valid rows, classes=%s) — filling NaN.",
                _cur_refit_date, int(valid.sum()),
                list(np.unique(y_te[valid])) if valid.any() else [],
            )
            perm = nan_vec.copy()

        # SHAP — optional, computed on full cycle test data subsampled to shap_subsample
        if compute_shap:
            shap_vals = _shap_values(
                _cur_estimator, X_te, _cur_X_tr_pp, model_type,
                max_rows=shap_subsample, random_state=random_state,
            )
            shap_imp = (np.abs(shap_vals).mean(axis=0)
                        if shap_vals is not None else nan_vec.copy())
        else:
            shap_imp = nan_vec.copy()

        fold_coef.append(_cur_coef.copy())
        fold_coef_t.append(_cur_coef_t.copy())
        fold_mdi.append(_cur_mdi.copy())
        fold_perm.append(perm)
        fold_shap.append(shap_imp)
        fold_refit_dates.append(_cur_refit_date)

    for train_iloc, test_iloc, is_refit in splitter.split(dates_unique):
        te_start = int(date_starts[test_iloc[0]])
        te_stop  = int(date_stops[test_iloc[-1]])

        if is_refit:
            _commit_cycle()   # commit PREVIOUS cycle before fitting the new model

            tr_start = int(date_starts[train_iloc[0]])
            tr_stop  = int(date_stops[train_iloc[-1]])
            X_tr_raw       = X_all[tr_start:tr_stop]
            y_tr           = y_all[tr_start:tr_stop]
            y_tr_bin       = y_bin_all[tr_start:tr_stop]
            valid_tr       = np.isfinite(y_tr)
            X_tr_clean     = X_tr_raw[valid_tr]
            y_tr_bin_clean = y_tr_bin[valid_tr]

            if len(X_tr_clean) < max(10, cfg.split_config.min_train_days // 10):
                _cur_estimator = None   # mark cycle invalid → next _commit_cycle is no-op
                _cur_X_te      = []
                _cur_y_te      = []
                continue

            pp      = FeaturePreprocessor(pp_cfg)
            X_tr_pp = pp.fit_transform(X_tr_clean, feature_cols)

            try:
                estimator = model_spec.clone_estimator()
                estimator.fit(X_tr_pp, y_tr_bin_clean.astype(int))
            except Exception as exc:
                log.warning("Fit failed at %s: %s", dates_unique[test_iloc[0]], exc)
                _cur_estimator = None
                _cur_X_te      = []
                _cur_y_te      = []
                continue

            # ── Model-level importances (training data only, computed once per cycle) ──
            _cur_coef   = (_extract_coef(estimator)
                           if model_type == "logistic" else nan_vec.copy())
            _cur_coef_t = (_hessian_tstat(estimator, X_tr_pp, y_tr_bin_clean)
                           if model_type == "logistic" else nan_vec.copy())
            _cur_mdi    = (_extract_mdi(estimator, len(feature_cols))
                           if model_type == "tree" else nan_vec.copy())

            # Reset cycle
            _cur_estimator  = estimator
            _cur_pp         = pp
            _cur_X_tr_pp    = X_tr_pp
            _cur_refit_date = dates_unique[test_iloc[0]]
            _cur_X_te       = []
            _cur_y_te       = []

        # ── Accumulate test data for the current cycle (every step, not just refits) ──
        if _cur_pp is not None and _cur_estimator is not None:
            X_te_pp = _cur_pp.transform(X_all[te_start:te_stop])
            _cur_X_te.append(X_te_pp)
            _cur_y_te.append(y_bin_all[te_start:te_stop])

    _commit_cycle()   # flush the last cycle

    n_folds = len(fold_refit_dates)
    if n_folds == 0:
        raise RuntimeError("No refit folds found — check config.split_config dates.")

    log.info("Collected %d refit folds for importance analysis.", n_folds)

    # ── Aggregate across folds ─────────────────────────────────────────────────
    summary = _build_summary(
        feature_cols = feature_cols,
        fold_coef    = fold_coef,
        fold_coef_t  = fold_coef_t,
        fold_mdi     = fold_mdi,
        fold_perm    = fold_perm,
        fold_shap    = fold_shap,
        model_type   = model_type,
    )
    fold_details = _build_fold_details(
        feature_cols     = feature_cols,
        fold_perm        = fold_perm,
        fold_coef        = fold_coef,
        fold_coef_t      = fold_coef_t,
        fold_mdi         = fold_mdi,
        fold_shap        = fold_shap,
        fold_refit_dates = fold_refit_dates,
        model_type       = model_type,
    )
    return FeatureImportanceSummary(
        summary      = summary,
        fold_details = fold_details,
        model_type   = model_type,
        n_folds      = n_folds,
        model_id     = model_id,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────────────────────────────────────

def _detect_model_type(estimator) -> Literal["logistic", "tree", "other"]:
    cls = type(estimator).__name__
    if "Logistic" in cls or "Ridge" in cls or "ElasticNet" in cls:
        return "logistic"
    if any(k in cls for k in ("Forest", "Tree", "Boost", "Gradient", "LGBM", "XGB", "Extra")):
        return "tree"
    return "other"


def _extract_coef(estimator) -> np.ndarray:
    if hasattr(estimator, "coef_"):
        return np.asarray(estimator.coef_).ravel()
    return np.array([])


def _hessian_tstat(estimator, X_tr: np.ndarray, y_tr: np.ndarray) -> np.ndarray:
    """
    Within-fold t-statistics for logistic regression via Fisher information.

    H = X^T diag(p*(1-p)) X  + I/C   (Hessian with L2 regularisation)
    Cov(β) = H^{-1}
    t_j = β_j / sqrt(Cov(β)_{jj})

    Returns zeros if computation fails (singular Hessian, non-logistic model).
    """
    if not hasattr(estimator, "coef_") or not hasattr(estimator, "C"):
        return np.zeros(X_tr.shape[1])
    coef = _extract_coef(estimator)
    if len(coef) != X_tr.shape[1]:
        return np.zeros(X_tr.shape[1])
    try:
        p   = estimator.predict_proba(X_tr)[:, 1]
        W   = p * (1.0 - p)
        # Weighted cross-product: efficient form X.T @ (W[:, None] * X)
        XtWX = X_tr.T @ (W[:, np.newaxis] * X_tr)
        C    = float(estimator.C)
        # Add L2 regularisation to Hessian diagonal
        XtWX += np.eye(len(coef)) / C
        cov  = np.linalg.pinv(XtWX)
        se   = np.sqrt(np.maximum(np.diag(cov), 0.0))
        return np.where(se > 1e-14, coef / se, 0.0)
    except Exception as exc:
        log.debug("Hessian t-stat failed: %s", exc)
        return np.zeros(len(coef))


def _extract_mdi(estimator, n_features: int) -> np.ndarray:
    if hasattr(estimator, "feature_importances_"):
        fi = np.asarray(estimator.feature_importances_)
        if len(fi) == n_features:
            return fi
    return np.zeros(n_features)


def _permutation_imp(
    estimator,
    X_te: np.ndarray,
    y_te: np.ndarray,
    scoring: str,
    n_repeats: int,
    random_state: int,
) -> np.ndarray:
    """sklearn permutation_importance on test data, returns importances_mean."""
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = permutation_importance(
                estimator, X_te, y_te,
                scoring=scoring,
                n_repeats=n_repeats,
                random_state=random_state,
                n_jobs=1,   # avoid nested parallelism
            )
        return result.importances_mean
    except Exception as exc:
        log.debug("Permutation importance failed: %s", exc)
        return np.zeros(X_te.shape[1])


def _shap_values(
    estimator,
    X_te: np.ndarray,
    X_tr: np.ndarray,
    model_type: str,
    max_rows: int,
    random_state: int,
) -> np.ndarray | None:
    try:
        import shap  # type: ignore[import]
    except ImportError:
        log.debug("shap not installed — skipping SHAP values.")
        return None
    try:
        rng = np.random.default_rng(random_state)
        # Subsample test rows to explain
        te_idx = rng.choice(len(X_te), min(max_rows, len(X_te)), replace=False)
        X_sub = X_te[te_idx]
        if model_type == "logistic":
            # Background must be training data: LinearExplainer estimates the feature
            # covariance matrix from this set. Using test data would make the background
            # depend on the same rows being explained.
            tr_idx = rng.choice(len(X_tr), min(max_rows, len(X_tr)), replace=False)
            X_bg = X_tr[tr_idx]
            explainer = shap.LinearExplainer(estimator, X_bg, feature_perturbation="correlation_dependent")
            vals = explainer.shap_values(X_sub)
        else:
            explainer = shap.TreeExplainer(estimator)
            vals = explainer.shap_values(X_sub)
            if isinstance(vals, list):
                vals = vals[1]  # take positive class for binary trees
        return np.asarray(vals)
    except Exception as exc:
        log.debug("SHAP failed: %s", exc)
        return None


def _stability_tstat(values: np.ndarray) -> tuple[float, float]:
    """Cross-fold stability: t = mean / (std / sqrt(n)), p two-tailed."""
    from scipy.stats import t as tdist  # type: ignore[import]
    n = len(values)
    if n < 2:
        return float(values[0]) if n == 1 else 0.0, 1.0
    m, s = float(np.mean(values)), float(np.std(values, ddof=1))
    if s < 1e-14:
        return 0.0, 1.0
    t_stat = m / (s / np.sqrt(n))
    p_val  = float(2.0 * tdist.sf(abs(t_stat), df=n - 1))
    return float(t_stat), p_val


def _sig_stars(p: float) -> str:
    if not np.isfinite(p): return ""
    if p < 0.001: return "***"
    if p < 0.01:  return "**"
    if p < 0.05:  return "*"
    if p < 0.10:  return "."
    return ""


def _build_summary(
    feature_cols: list[str],
    fold_coef:    list[np.ndarray],
    fold_coef_t:  list[np.ndarray],
    fold_mdi:     list[np.ndarray],
    fold_perm:    list[np.ndarray],
    fold_shap:    list[np.ndarray],
    model_type:   str,
) -> pd.DataFrame:
    """Aggregate per-fold importances into one summary row per feature."""
    p = len(feature_cols)
    rows = []

    for j, feat in enumerate(feature_cols):
        row: dict = {"feature": feat}

        # All fold lists are length n_folds by construction — no guards needed.
        coefs = np.array([fc[j] for fc in fold_coef])
        ts_h  = np.array([ft[j] for ft in fold_coef_t])
        mdis  = np.array([fm[j] for fm in fold_mdi])
        perms = np.array([fp[j] for fp in fold_perm])
        shaps = np.array([fs[j] for fs in fold_shap])

        # ── Coefficients (logistic; NaN filled for tree/other) ────────────────
        if model_type == "logistic":
            row["coef_mean"] = float(np.nanmean(coefs))
            row["coef_std"]  = float(np.nanstd(coefs, ddof=1))
            # coef_t_hessian: average within-fold Hessian t-stat.
            # Diagnostic for ranking only — L2 regularisation inflates SEs slightly.
            row["coef_t_hessian"] = float(np.nanmean(ts_h))

        # ── MDI (tree; NaN filled for logistic/other) ─────────────────────────
        if model_type == "tree":
            row["mdi_mean"] = float(np.nanmean(mdis))
            row["mdi_std"]  = float(np.nanstd(mdis, ddof=1))

        # ── Permutation importance (all models, computed on OOS test folds) ───
        row["perm_mean"] = float(np.nanmean(perms))
        row["perm_std"]  = float(np.nanstd(perms, ddof=1))
        finite_perms = perms[np.isfinite(perms)]
        if len(finite_perms) > 1:
            t_stab, p_stab = _stability_tstat(finite_perms)
        else:
            t_stab, p_stab = np.nan, np.nan
        row["stability_t"] = t_stab
        row["stability_p"] = p_stab
        row["sig"]         = _sig_stars(p_stab)

        # ── SHAP — only show column when at least one fold computed it ─────────
        if np.isfinite(shaps).any():
            row["shap_mean_abs"] = float(np.nanmean(shaps))

        rows.append(row)

    df = pd.DataFrame(rows)

    # Sort by |perm_mean| if available, else |coef_mean|, else |mdi_mean|
    for sort_col in ["perm_mean", "coef_mean", "mdi_mean"]:
        if sort_col in df.columns:
            df = df.assign(_abs=df[sort_col].abs()).sort_values("_abs", ascending=False).drop(columns="_abs")
            break

    return df.reset_index(drop=True)


def _build_fold_details(
    feature_cols:     list[str],
    fold_perm:        list[np.ndarray],
    fold_coef:        list[np.ndarray],
    fold_coef_t:      list[np.ndarray],
    fold_mdi:         list[np.ndarray],
    fold_shap:        list[np.ndarray],
    fold_refit_dates: list,
    model_type:       str,
) -> pd.DataFrame:
    """Long-format table of per-fold importances for diagnostic plots.

    All fold lists are guaranteed to be aligned (same length as fold_refit_dates)
    by construction in collect_feature_importances.
    """
    n = len(fold_refit_dates)
    assert len(fold_perm)   == n, f"fold_perm misaligned: {len(fold_perm)} != {n}"
    assert len(fold_coef)   == n, f"fold_coef misaligned: {len(fold_coef)} != {n}"
    assert len(fold_coef_t) == n, f"fold_coef_t misaligned: {len(fold_coef_t)} != {n}"
    assert len(fold_mdi)    == n, f"fold_mdi misaligned: {len(fold_mdi)} != {n}"
    assert len(fold_shap)   == n, f"fold_shap misaligned: {len(fold_shap)} != {n}"

    records = []
    for k, date in enumerate(fold_refit_dates):
        for j, feat in enumerate(feature_cols):
            rec: dict = {
                "refit_date": date,
                "feature":    feat,
                "perm":       float(fold_perm[k][j]),
                "coef":       float(fold_coef[k][j]),
                "coef_t":     float(fold_coef_t[k][j]),
                "mdi":        float(fold_mdi[k][j]),
                "shap_abs":   float(fold_shap[k][j]),
            }
            records.append(rec)
    return pd.DataFrame(records)
