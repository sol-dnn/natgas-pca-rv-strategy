"""Walk-forward model runner.

Core loop that produces strictly out-of-sample predictions. Supports two
modes:
  panel        : one model trained on all strategy observations jointly
                 (strategy dummies expected in feature_cols).
  per_strategy : separate model per strategy group (no dummies needed).
"""
from __future__ import annotations

import time
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed

# Suppress cosmetic warnings that flood logs in production runs
warnings.filterwarnings("ignore", message=".*does not have valid feature names.*")
warnings.filterwarnings("ignore", message=".*An input array is constant.*")
from dataclasses import dataclass, field
from typing import Literal

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator

from .preprocessor import FeaturePreprocessor, PreprocessorConfig
from .splitter import SplitConfig, TimeSeriesSplitter
from .zoo import ModelSpec


# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class WalkForwardConfig:
    """Full configuration for a walk-forward experiment."""

    split_config: SplitConfig = field(default_factory=SplitConfig)
    preprocessor_config: PreprocessorConfig = field(default_factory=PreprocessorConfig)

    task: Literal["binary", "regression"] = "binary"
    """Only binary classification is supported. Regression requires separate calibration."""

    binary_threshold: float = 0.0
    """Return level that separates positive from negative class."""

    mode: Literal["panel", "per_strategy"] = "panel"
    """
    panel        : one model for all strategies; strategy dummies in features.
    per_strategy : separate model per strategy; no dummies needed.
    """

    group_col: str = "option_type"
    """Column identifying the strategy group. Used in per_strategy mode."""

    date_col: str = "quote_date"
    """Column containing the trading date."""

    compute_is_metrics: bool = True
    """
    Compute in-sample IC on the last is_eval_days of each training window.
    Used to compute the IS–OOS IC gap (overfitting diagnostic).
    Adds one predict() call per model per refit step.
    """

    is_eval_days: int = 63
    """Number of recent training days used for IS IC (only if compute_is_metrics=True)."""

    id_cols: list[str] = field(default_factory=list)
    """
    Extra columns from data to carry through into predictions (e.g. ["mnes", "quote_time"]).
    Useful for downstream portfolio construction without a separate merge.
    Columns not present in data are silently ignored.
    """

    sort_extra_cols: list[str] = field(default_factory=list)
    """
    Additional sort keys within each (date, group) block for deterministic row order.
    Required when data has multiple intraday rows per (date, strategy) — e.g. different
    quote_time or moneyness levels. Without this, row order within a block is arbitrary
    and results differ if the input DataFrame arrives in a different order.
    Example: ["quote_time", "mnes"]
    Columns absent from data are silently ignored.
    """

    ic_time_col: str | None = None
    """
    Column containing the intraday time slot (e.g. "quote_time").
    When set and the training data contains multiple time slots per date,
    IS IC is computed as ic_series_intraday()-style: IC per (date, time) slot
    averaged to daily, then mean over days.  This makes IS IC and OOS IC
    directly comparable, giving a clean ic_gap diagnostic.

    Leave None (default) for single-time models or when exact IS/OOS
    consistency is not required (ic_gap remains approximate but directional).
    """

    filter_liquid: bool = True
    """
    If True (default), restrict data to all_liquid_int == 1 before training
    and prediction — i.e. only strategies where every leg has a valid bid.
    Set False for robustness checks on the full (including illiquid) universe.
    No-op if the column is absent from data.
    """

    n_jobs_models: int = 1
    """
    Number of models to fit in parallel at each refit step.
      1   : sequential (default, safe for debugging)
      -1  : all models simultaneously (recommended for production)
      k>1 : k models at a time

    Uses Python threads (shared memory, no pickling overhead).
    sklearn/XGBoost/LightGBM release the GIL so genuine CPU parallelism is achieved.
    When n_jobs_models != 1, each model's own n_jobs is forced to 1 to avoid
    CPU over-subscription (each model gets one dedicated core).
    Typical speedup: 5-7x on an 8-core machine.
    """


# ─────────────────────────────────────────────────────────────────────────────
# Result containers
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class WalkForwardResult:
    """Output of walk_forward()."""

    predictions: pd.DataFrame
    """
    One row per (date × strategy × model_id).

    Columns:
      quote_date   — test date
      <group_col>  — strategy identifier (NaN in panel mode if not present)
      model_id     — model key from model_zoo
      y            — realised return (continuous)
      y_bin        — binary target (1 if y > binary_threshold)
      y_hat        — model output: predicted prob (binary) or predicted return (regression)
      p_hat        — probability of positive return (clipped to [1e-6, 1-1e-6])
      sign         — directional signal: +1 (p_hat ≥ 0.5) or -1 (p_hat < 0.5); NaN if features missing
      weight       — same as sign (no-trade band belongs in backtest module)
      refit_date   — first prediction date of the refit cycle (model trained on data up to train_end)
    """

    split_log: pd.DataFrame
    """
    One row per refit event per model.

    Columns:
      refit_date      — first prediction date of the refit cycle (= test_date, kept for backward compat)
      test_date       — same as refit_date; explicit alias for unambiguous merges
      train_start     — first date in training window
      train_end       — last date in training window (inclusive)
      n_train_days    — number of unique training dates
      n_train_obs     — number of training rows (dates × strategies)
      model_id
      is_ic           — mean daily IC on last is_eval_days of training data (NaN if disabled)
      train_pos_rate  — fraction of positive labels in clean training window (class balance check)
      purge_days      — purge gap used (from SplitConfig)
      window_type     — rolling or expanding (from SplitConfig)
      rolling_window  — training window size in days (from SplitConfig)
      refit_every     — refit cadence in trading days (from SplitConfig)

    Note: refit_date is the FIRST PREDICTION DATE of the refit cycle, not the training date.
    The model is trained on data up to train_end; refit_date = train_end + purge_days + 1.
    """


# ─────────────────────────────────────────────────────────────────────────────
# Internal helpers — no lookahead, all operate on pre-sliced arrays
# ─────────────────────────────────────────────────────────────────────────────

def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -50.0, 50.0)))


def _get_proba(estimator: BaseEstimator, X: np.ndarray) -> np.ndarray:
    """Return P(return > 0) for each row (binary classifiers only)."""
    if hasattr(estimator, "predict_proba"):
        p = np.asarray(estimator.predict_proba(X), dtype=float)
        return np.clip(p[:, 1] if p.ndim == 2 else p.reshape(-1), 1e-6, 1 - 1e-6)
    # Fallback: decision_function → sigmoid
    scores = np.asarray(estimator.decision_function(X), dtype=float).reshape(-1)
    return np.clip(_sigmoid(scores), 1e-6, 1 - 1e-6)


def _get_predict(estimator: BaseEstimator, X: np.ndarray) -> np.ndarray:
    """Return continuous predictions for regression estimators."""
    return np.asarray(estimator.predict(X), dtype=float).reshape(-1)


def _sign_weight(p_hat: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Directional signal from probabilities. NaN input → NaN output.

    sign/weight = +1 when p_hat ≥ 0.5, -1 otherwise.
    No-trade band and continuous sizing belong in the backtest module.
    """
    p = np.asarray(p_hat, dtype=float)
    sign = np.where(np.isfinite(p), np.where(p >= 0.5, 1.0, -1.0), np.nan)
    return sign, sign.copy()


def _sign_weight_reg(y_hat: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Directional signal from regression predictions. sign = +1 if y_hat > 0."""
    y = np.asarray(y_hat, dtype=float)
    sign = np.where(np.isfinite(y), np.where(y > 0.0, 1.0, -1.0), np.nan)
    return sign, y.copy()  # weight proportional to predicted magnitude


def _daily_mean_ic(
    p_hat: np.ndarray,
    y: np.ndarray,
    dates: np.ndarray,
    times: np.ndarray | None = None,
) -> float:
    """
    Mean cross-sectional Spearman IC.

    If times is None (or all dates have a single time slot): IC per date,
    then mean over dates.  This is the standard pooled daily IC.

    If times is provided and a date has multiple time slots: IC per
    (date, time) slot, averaged to a single daily value, then mean over dates.
    This matches ic_series_intraday() in evaluator.py, making IS IC and OOS IC
    directly comparable in the ic_gap diagnostic.
    """
    from scipy.stats import spearmanr  # type: ignore[import]
    daily_ics: list[float] = []
    for d in np.unique(dates):
        d_mask = dates == d
        if times is not None and np.unique(times[d_mask]).size > 1:
            # Multi-time slot: IC per slot, averaged
            slot_ics: list[float] = []
            for t in np.unique(times[d_mask]):
                mask = d_mask & (times == t) & np.isfinite(p_hat) & np.isfinite(y)
                if mask.sum() < 3 or np.unique(y[mask]).size < 2:
                    continue
                r, _ = spearmanr(p_hat[mask], y[mask])
                if np.isfinite(r):
                    slot_ics.append(float(r))
            if slot_ics:
                daily_ics.append(float(np.mean(slot_ics)))
        else:
            # Single time (or no times array): IC pooled over cross-section
            mask = d_mask & np.isfinite(p_hat) & np.isfinite(y)
            if mask.sum() < 3 or np.unique(y[mask]).size < 2:
                continue
            r, _ = spearmanr(p_hat[mask], y[mask])
            if np.isfinite(r):
                daily_ics.append(float(r))
    return float(np.mean(daily_ics)) if daily_ics else np.nan


# ─────────────────────────────────────────────────────────────────────────────
# Per-model fit helper (module-level so ThreadPoolExecutor can pickle it)
# ─────────────────────────────────────────────────────────────────────────────

def _fit_one_model(
    model_id: str,
    spec: "ModelSpec",
    X_tr_pp: np.ndarray,
    y_tr_bin_clean: np.ndarray,
    y_tr_clean: np.ndarray,
    compute_is_metrics: bool,
    is_ic_rows: int | None,
    dates_tr_clean: np.ndarray,
    times_is: np.ndarray | None,
    test_date_str: str,
    force_njobs_1: bool,
    task: str = "binary",
) -> tuple[str, "BaseEstimator | None", float]:
    """
    Fit one model and compute IS IC.  Safe for concurrent execution via threads.

    Returns (model_id, fitted_estimator_or_None, is_ic).
    Arrays are read-only shared memory when called from ThreadPoolExecutor —
    no copies are made.
    """
    # Use continuous target for regression, binary (0/1) for classification
    y_fit = y_tr_clean if task == "regression" else y_tr_bin_clean.astype(int)

    try:
        fitted = spec.clone_estimator()
        if force_njobs_1 and hasattr(fitted, "n_jobs"):
            fitted.set_params(n_jobs=1)
        fitted.fit(X_tr_pp, y_fit)
    except Exception as exc:
        warnings.warn(
            f"Model fit failed: model={model_id}, date={test_date_str}. Error: {exc}",
            RuntimeWarning,
        )
        fitted = None

    is_ic = np.nan
    if compute_is_metrics and fitted is not None and is_ic_rows:
        X_is = X_tr_pp[-is_ic_rows:]
        y_is = y_tr_clean[-is_ic_rows:]
        dates_is = dates_tr_clean[-is_ic_rows:]
        try:
            if task == "regression":
                p_is = _get_predict(fitted, X_is)
            else:
                p_is = _get_proba(fitted, X_is)
            is_ic = _daily_mean_ic(p_is, y_is, dates_is, times_is)
        except Exception:
            pass

    return model_id, fitted, is_ic


# ─────────────────────────────────────────────────────────────────────────────
# Core panel walk-forward
# ─────────────────────────────────────────────────────────────────────────────

def _walk_forward_panel(
    data: pd.DataFrame,
    feature_cols: list[str],
    target_col: str,
    model_zoo: dict[str, ModelSpec],
    config: WalkForwardConfig,
) -> WalkForwardResult:
    """
    Walk-forward loop for panel data (all strategies or single strategy).

    Lookahead audit
    ---------------
    1. Preprocessor.fit() is called ONLY with training rows (X_train).
       Its bounds/statistics are then frozen for the lifetime of the refit
       step. → No test data enters the scaling statistics.

    2. Estimator.fit() receives ONLY preprocessed training rows.
       clone() is called before each fit to discard any state from prior
       refit steps. → No cross-contamination between windows.

    3. IS IC is computed on the last is_eval_days rows of the training
       window using the FITTED model. This is purely a reporting metric
       (not used for model selection or early stopping).
       → No feedback from IS evaluation to the fitted model.

    4. date_starts / date_stops map each unique date index to its row range.
       Slicing X_all[tr_start_row:tr_stop_row] gives exactly the training
       rows. Test rows X_all[te_start_row:te_stop_row] are disjoint.
       → Row slicing is date-aligned, no partial days.
    """
    if config.task not in ("binary", "regression"):
        raise ValueError(f"task must be 'binary' or 'regression', got {config.task!r}")

    # Guard: feature_cols must not contain the target or any other target column
    # (target columns start with "target_" by convention)
    bad_features = [c for c in feature_cols
                    if c == target_col or c.startswith("target_")]
    if bad_features:
        raise ValueError(
            f"feature_cols contains target/forbidden columns: {bad_features}. "
            "Remove them before calling walk_forward()."
        )

    cfg = config
    splitter = TimeSeriesSplitter(cfg.split_config)
    date_col = cfg.date_col
    group_col = cfg.group_col

    # ── Sort data: date → group → any extra intraday keys ─────────────────
    sort_cols = [date_col] + ([group_col] if group_col in data.columns else [])
    sort_cols += [c for c in cfg.sort_extra_cols if c in data.columns and c not in sort_cols]
    data = data.sort_values(sort_cols).reset_index(drop=True)

    # ── Date → row-range mapping ───────────────────────────────────────────
    date_groups = data.groupby(date_col, sort=True, observed=True).size()
    dates_unique = date_groups.index.to_numpy()
    counts = date_groups.to_numpy(dtype=int)
    date_stops_row = np.cumsum(counts)
    date_starts_row = np.r_[0, date_stops_row[:-1]]

    # ── Extract raw arrays (avoids repeated DataFrame indexing) ──────────
    # Handle NaN: rows with NaN in features or target are dropped inside the
    # walk-forward loop (not here) so that date slicing remains correct.
    X_all = data[feature_cols].to_numpy(dtype=float)
    y_all = data[target_col].to_numpy(dtype=float)
    # NaN targets stay NaN — don't silently convert them to 0 (negative class)
    y_bin_all = np.where(np.isfinite(y_all), (y_all > cfg.binary_threshold).astype(float), np.nan)
    dates_all = data[date_col].to_numpy()
    groups_all = data[group_col].to_numpy(dtype=str) if group_col in data.columns else None
    # IS IC time array — only extracted if ic_time_col is set and present
    times_all = (
        data[cfg.ic_time_col].to_numpy(dtype=str)
        if cfg.ic_time_col and cfg.ic_time_col in data.columns
        else None
    )
    # Extra identifier columns passed through to predictions as-is
    active_id_cols = [c for c in cfg.id_cols if c in data.columns and c != date_col and c != group_col]
    id_arrays: dict[str, np.ndarray] = {c: data[c].to_numpy() for c in active_id_cols}

    if cfg.split_config.verbose:
        _W = 72
        dummy_cols = [c for c in feature_cols
                      if any(c.startswith(p) for p in ("strat_", "is_", "dummy_", "type_"))]
        non_dummy = [c for c in feature_cols if c not in dummy_cols]
        print("═" * _W)
        print(f"  walk_forward — {cfg.mode}")
        print(f"  Dataset     : {len(data):,} rows  |  {len(dates_unique)} trading days  "
              f"({pd.Timestamp(dates_unique[0]).date()} → {pd.Timestamp(dates_unique[-1]).date()})")
        print(f"  Target      : {target_col}  (threshold={cfg.binary_threshold}, "
              f"overall pos_rate={float(np.nanmean(y_bin_all)):.3f})")
        if dummy_cols:
            print(f"  Dummies     : {dummy_cols}")
        else:
            print(f"  Dummies     : (none detected — strat_/is_/dummy_/type_ prefix)")
        print(f"  Features    : {non_dummy}")
        print(f"  Models      : {list(model_zoo.keys())}")
        print(f"  IS eval     : last {cfg.is_eval_days}d  |  compute_is_metrics={cfg.compute_is_metrics}")
        print(f"  id_cols     : {active_id_cols or '(none)'}")
        print("  Evaluation  : IC/AUC/comparison_table → call evaluator.comparison_table() after run")
        print("═" * _W)
        _refit_counter = 0

    # ── State: cached model + preprocessor (reused between refit steps) ──
    cached_models: dict[str, BaseEstimator | None] = {mid: None for mid in model_zoo}
    cached_preprocessor: FeaturePreprocessor | None = None
    last_refit_date: dict[str, np.datetime64 | None] = {mid: None for mid in model_zoo}

    pred_rows: list[dict] = []
    log_rows: list[dict] = []

    for train_iloc, test_iloc, is_refit in splitter.split(dates_unique):
        # ── Row slices for training and test ─────────────────────────────
        if len(train_iloc) == 0 or len(test_iloc) != 1:
            raise RuntimeError(
                f"Splitter returned unexpected shapes: "
                f"train_iloc={len(train_iloc)}, test_iloc={len(test_iloc)}"
            )
        if not (dates_unique[train_iloc[-1]] < dates_unique[test_iloc[0]]):
            raise RuntimeError("Splitter violation: last train date >= test date (no purge gap).")

        tr_start_row = int(date_starts_row[train_iloc[0]])
        tr_stop_row = int(date_stops_row[train_iloc[-1]])
        te_start_row = int(date_starts_row[test_iloc[0]])
        te_stop_row = int(date_stops_row[test_iloc[-1]])
        if tr_stop_row > te_start_row:
            raise RuntimeError(
                f"Row overlap: train rows end at {tr_stop_row}, test rows start at {te_start_row}."
            )
        test_date = dates_unique[test_iloc[0]]

        X_tr_raw = X_all[tr_start_row:tr_stop_row]
        X_te_raw = X_all[te_start_row:te_stop_row]
        y_tr = y_all[tr_start_row:tr_stop_row]
        y_tr_bin = y_bin_all[tr_start_row:tr_stop_row]

        # Drop only rows with NaN TARGET from training.
        # NaN features are kept and will be imputed by the preprocessor.
        valid_tr = np.isfinite(y_tr)
        X_tr_clean = X_tr_raw[valid_tr]
        y_tr_clean = y_tr[valid_tr]
        y_tr_bin_clean = y_tr_bin[valid_tr]

        if len(X_tr_clean) < max(10, cfg.split_config.min_train_days // 10):
            continue  # insufficient training data after dropping target-NaN rows

        # ── Refit step ────────────────────────────────────────────────────
        if is_refit:
            n_nan_feat_tr = int((~np.isfinite(X_tr_clean)).any(axis=1).sum())
            # Preprocessor: fit ONLY on clean training data
            pp = FeaturePreprocessor(cfg.preprocessor_config)
            X_tr_pp = pp.fit_transform(X_tr_clean, feature_cols)
            cached_preprocessor = pp

            # Date for each clean training row (used by IS IC)
            dates_tr_clean = dates_all[tr_start_row:tr_stop_row][valid_tr]
            train_pos_rate = float(np.nanmean(y_tr_bin_clean))

            # IS eval window: last is_eval_days trading days of training data
            is_ic_rows: int | None = None
            if cfg.compute_is_metrics and cfg.is_eval_days > 0:
                unique_tr_dates = np.unique(dates_tr_clean)
                if len(unique_tr_dates) > cfg.is_eval_days:
                    cutoff = unique_tr_dates[-cfg.is_eval_days]
                    is_ic_rows = int(np.sum(dates_tr_clean >= cutoff))
                else:
                    is_ic_rows = len(X_tr_pp)

            if cfg.split_config.verbose:
                _refit_counter += 1
                print(f"  Refit {_refit_counter:>3}  │  {pd.Timestamp(test_date).date()}"
                      f"  │  train_obs={int(valid_tr.sum()):,}"
                      f"  │  nan_feat_rows={n_nan_feat_tr:,}"
                      f"  │  pos_rate={train_pos_rate:.3f}")

            # Pre-compute times_is slice (shared read-only across threads)
            times_is_slice: np.ndarray | None = None
            if times_all is not None and is_ic_rows:
                times_is_slice = times_all[tr_start_row:tr_stop_row][valid_tr][-is_ic_rows:]

            _n_models = len(model_zoo)
            test_date_str = str(pd.Timestamp(test_date).date())
            parallel = cfg.n_jobs_models != 1
            n_workers = _n_models if cfg.n_jobs_models == -1 else max(1, cfg.n_jobs_models)

            if cfg.split_config.verbose:
                mode_str = f"parallel({n_workers})" if parallel else "sequential"
                print(f"    fitting {_n_models} models [{mode_str}]", flush=True)

            if parallel:
                # Submit all models to thread pool; each gets n_jobs=1 to avoid
                # CPU over-subscription (models share cores, not fight for all of them).
                with ThreadPoolExecutor(max_workers=n_workers) as pool:
                    future_to_mid = {
                        pool.submit(
                            _fit_one_model,
                            mid, spec, X_tr_pp, y_tr_bin_clean, y_tr_clean,
                            cfg.compute_is_metrics, is_ic_rows,
                            dates_tr_clean, times_is_slice,
                            test_date_str, True, cfg.task,
                        ): mid
                        for mid, spec in model_zoo.items()
                    }
                    fit_results: dict[str, tuple] = {}
                    _fit_t0 = time.time()
                    _done = 0
                    for future in as_completed(future_to_mid):
                        mid, fitted, is_ic = future.result()
                        fit_results[mid] = (fitted, is_ic)
                        _done += 1
                        if cfg.split_config.verbose:
                            status = "✓" if fitted is not None else "✗"
                            elapsed = time.time() - _fit_t0
                            print(f"      {status} {mid:<22s}  [{_done}/{_n_models}]  {elapsed:.0f}s", flush=True)
            else:
                fit_results = {}
                for _mi, (mid, spec) in enumerate(model_zoo.items()):
                    if cfg.split_config.verbose:
                        print(f"\r    [{_mi + 1}/{_n_models}] {mid:<22s}", end="", flush=True)
                    _, fitted, is_ic = _fit_one_model(
                        mid, spec, X_tr_pp, y_tr_bin_clean, y_tr_clean,
                        cfg.compute_is_metrics, is_ic_rows,
                        dates_tr_clean, times_is_slice,
                        test_date_str, False, cfg.task,
                    )
                    fit_results[mid] = (fitted, is_ic)
                if cfg.split_config.verbose:
                    print()

            # Collect results and build log rows (order-stable via model_zoo key order)
            for model_id, (fitted, is_ic) in {
                mid: fit_results[mid] for mid in model_zoo
            }.items():
                cached_models[model_id] = fitted
                last_refit_date[model_id] = test_date
                log_rows.append({
                    "refit_date": pd.Timestamp(test_date),
                    "test_date": pd.Timestamp(test_date),
                    "train_start": pd.Timestamp(dates_unique[train_iloc[0]]),
                    "train_end": pd.Timestamp(dates_unique[train_iloc[-1]]),
                    "n_train_days": len(train_iloc),
                    "n_train_obs": int(valid_tr.sum()),
                    "n_nan_feat_rows_train": n_nan_feat_tr,
                    "model_id": model_id,
                    "is_ic": is_ic,
                    "train_pos_rate": train_pos_rate,
                    "purge_days": cfg.split_config.purge_days,
                    "window_type": cfg.split_config.window_type,
                    "rolling_window": cfg.split_config.rolling_window,
                    "refit_every": cfg.split_config.refit_every,
                    "pp_nan_rate_before": pp.fit_log.get("nan_rate_before"),
                    "pp_n_imputed":       pp.fit_log.get("n_imputed"),
                    "pp_winsor_q":        pp.fit_log.get("winsor_quantile"),
                    "pp_n_ts":            pp.fit_log.get("n_ts"),
                    "pp_n_raw":           pp.fit_log.get("n_raw"),
                })

        # ── Prediction step ───────────────────────────────────────────────
        if cached_preprocessor is None:
            continue  # first refit not yet reached (should not happen with correct oos_start_date)

        # Transform test data with FROZEN training preprocessor (no refit).
        # Preprocessor guarantees fully finite output — no NaN/inf filtering needed.
        X_te_pp = cached_preprocessor.transform(X_te_raw)

        y_te = y_all[te_start_row:te_stop_row]
        y_te_bin = y_bin_all[te_start_row:te_stop_row]
        groups_te = groups_all[te_start_row:te_stop_row] if groups_all is not None else None

        for model_id, fitted in cached_models.items():
            if fitted is None:
                continue

            try:
                if config.task == "regression":
                    y_hat_full = _get_predict(fitted, X_te_pp)
                    sign, weight = _sign_weight_reg(y_hat_full)
                    p_hat_full = y_hat_full  # y_hat used as ranking signal
                else:
                    p_hat_full = _get_proba(fitted, X_te_pp)
                    y_hat_full = p_hat_full
                    sign, weight = _sign_weight(p_hat_full)

                abs_k = te_start_row  # absolute row index into data
                for k in range(len(X_te_pp)):
                    row: dict = {
                        date_col: pd.Timestamp(test_date),
                        "model_id": model_id,
                        "y": float(y_te[k]),
                        "y_bin": float(y_te_bin[k]),  # NaN when target is missing
                        "y_hat": float(y_hat_full[k]),
                        "p_hat": float(p_hat_full[k]),
                        "sign": float(sign[k]),
                        "weight": float(weight[k]),
                        "refit_date": pd.Timestamp(last_refit_date[model_id]),
                    }
                    if groups_te is not None:
                        row[group_col] = groups_te[k]
                    for col, arr in id_arrays.items():
                        row[col] = arr[abs_k + k]
                    pred_rows.append(row)

            except Exception as exc:
                warnings.warn(
                    f"Prediction failed: model={model_id}, date={pd.Timestamp(test_date).date()}. "
                    f"Error: {exc}",
                    RuntimeWarning,
                )

    predictions = pd.DataFrame(pred_rows)
    split_log = pd.DataFrame(log_rows)
    return WalkForwardResult(predictions=predictions, split_log=split_log)


# ─────────────────────────────────────────────────────────────────────────────
# Per-strategy wrapper
# ─────────────────────────────────────────────────────────────────────────────

def _walk_forward_per_strategy(
    data: pd.DataFrame,
    feature_cols: list[str],
    target_col: str,
    model_zoo: dict[str, ModelSpec],
    config: WalkForwardConfig,
) -> WalkForwardResult:
    """
    Run a separate panel walk-forward for each strategy group.

    Feature note: in per-strategy mode, strategy dummy columns are NOT used
    (each sub-dataset has only one strategy). Dummies should not be in
    feature_cols, or they will be constant columns, harmless but wasteful.
    """
    group_col = config.group_col
    if group_col not in data.columns:
        raise ValueError(f"group_col={group_col!r} not found in data columns.")

    groups = sorted(data[group_col].unique())
    all_preds: list[pd.DataFrame] = []
    all_logs: list[pd.DataFrame] = []

    for group_id in groups:
        subset = data[data[group_col] == group_id].copy().reset_index(drop=True)
        # Build a panel config for single-group processing
        panel_cfg = WalkForwardConfig(
            split_config=config.split_config,
            preprocessor_config=config.preprocessor_config,
            task=config.task,
            binary_threshold=config.binary_threshold,
            mode="panel",
            group_col=config.group_col,
            date_col=config.date_col,
            compute_is_metrics=config.compute_is_metrics,
            is_eval_days=config.is_eval_days,
            id_cols=config.id_cols,
            sort_extra_cols=config.sort_extra_cols,
            ic_time_col=config.ic_time_col,
        )
        result = _walk_forward_panel(subset, feature_cols, target_col, model_zoo, panel_cfg)
        all_preds.append(result.predictions)
        # Tag split_log with the group so per-group audit is possible
        log = result.split_log.copy()
        log[group_col] = group_id
        all_logs.append(log)

    return WalkForwardResult(
        predictions=pd.concat(all_preds, ignore_index=True) if all_preds else pd.DataFrame(),
        split_log=pd.concat(all_logs, ignore_index=True) if all_logs else pd.DataFrame(),
    )


# ─────────────────────────────────────────────────────────────────────────────
# Public entry point
# ─────────────────────────────────────────────────────────────────────────────

def walk_forward(
    data: pd.DataFrame,
    feature_cols: list[str],
    target_col: str,
    model_zoo: dict[str, ModelSpec],
    config: WalkForwardConfig | None = None,
) -> WalkForwardResult:
    """
    Run a walk-forward OOS experiment.

    Parameters
    ----------
    data
        Panel DataFrame sorted (or sortable) by date. Must contain:
        - config.date_col  : trading date
        - config.group_col : strategy identifier (required for per_strategy mode)
        - feature_cols     : model inputs (precomputed, including any CS z-scores)
        - target_col       : continuous return (y_bin is derived internally)
    feature_cols
        List of column names to use as model inputs. For panel mode, include
        strategy dummy columns (e.g., strat_strangle, strat_iron_condor).
        For per_strategy mode, omit dummies.
    target_col
        Continuous return column. Binary target is derived as
        (target > config.binary_threshold).
    model_zoo
        Dict mapping model_id → ModelSpec. Typically from build_default_zoo().
    config
        WalkForwardConfig. Uses defaults if None.

    Returns
    -------
    WalkForwardResult with predictions and split_log DataFrames.
    """
    if config is None:
        config = WalkForwardConfig()

    if config.filter_liquid and "all_liquid_int" in data.columns:
        data = data[data["all_liquid_int"] == 1].copy()

    missing = [c for c in feature_cols + [target_col, config.date_col] if c not in data.columns]
    if missing:
        raise ValueError(f"Columns missing from data: {missing}")

    if config.mode == "per_strategy":
        return _walk_forward_per_strategy(data, feature_cols, target_col, model_zoo, config)
    else:
        return _walk_forward_panel(data, feature_cols, target_col, model_zoo, config)
