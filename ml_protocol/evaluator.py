"""Evaluation metrics for walk-forward ML experiments.

All metrics here are purely predictive quality — no cost assumptions,
no portfolio construction.  Economic / backtest metrics belong in a
separate module (backtest/).

IC (Information Coefficient) is the primary metric for ranking models in
a cross-sectional setting (multiple strategies per date). ICIR is its
risk-adjusted version (analogous to Sharpe ratio of the IC series).
IC t-stat answers: is the mean IC statistically different from zero?

IC gap = IS IC − OOS IC is the primary overfitting diagnostic: a large
positive gap indicates the model memorises training patterns that do not
generalise.
"""
from __future__ import annotations

from typing import Any, Literal

import numpy as np
import pandas as pd


# ─────────────────────────────────────────────────────────────────────────────
# Multiple-testing correction helpers
# ─────────────────────────────────────────────────────────────────────────────

def _pvalues_from_tstats(
    t_stats: np.ndarray,
    n_days: np.ndarray,
) -> np.ndarray:
    """Two-tailed p-values from IC t-statistics via the t-distribution.

    With n_days ≥ 30 the t-distribution → normal; both are correct.
    We use the t-distribution with df = n_days − 1 for accuracy.
    """
    from scipy.stats import t as tdist  # type: ignore[import]
    df = np.maximum(n_days - 1, 1).astype(float)
    return 2.0 * tdist.sf(np.abs(t_stats), df=df)


def _bhy_correction(
    pvalues: np.ndarray,
    alpha: float = 0.05,
) -> tuple[np.ndarray, np.ndarray]:
    """Benjamini–Hochberg–Yekutieli (2001) FDR correction.

    Controls the False Discovery Rate under arbitrary dependence between
    tests (unlike plain BH which requires PRDS). Since our 14 models are
    all trained on the same data, their IC series are highly correlated,
    so BHY is the right choice.

    Parameters
    ----------
    pvalues : raw p-values, shape (k,).  NaN entries are treated as p=1.
    alpha   : target FDR level (default 5%).

    Returns
    -------
    rejected   : bool array (True = reject H0 = signal is significant)
    adj_pvalue : BHY-adjusted p-values (for reporting).
                 adj_p_i = min(1, k * c(k) * p_i / rank_i) with monotone
                 enforcement — a direct analogue of BH-adjusted p-values.
    """
    k = len(pvalues)
    if k == 0:
        return np.array([], dtype=bool), np.array([], dtype=float)

    # Replace NaN with 1 (non-significant)
    p = np.where(np.isfinite(pvalues), pvalues, 1.0)

    # Harmonic series correction c(k) = Σ 1/i
    c_k = float(np.sum(1.0 / np.arange(1, k + 1)))

    # Sort ascending
    order = np.argsort(p)
    sorted_p = p[order]

    # BHY threshold for rank i (1-indexed): alpha * i / (k * c(k))
    thresholds = alpha * np.arange(1, k + 1) / (k * c_k)
    reject_sorted = sorted_p <= thresholds

    # Step-up: once the highest rank that rejects is found, all lower ranks reject
    if reject_sorted.any():
        max_rej = int(np.max(np.where(reject_sorted)[0]))
        reject_sorted[: max_rej + 1] = True

    # BHY-adjusted p-value: p_adj_i = min(1, k*c(k)*p_i/rank_i),
    # enforced monotone from the right so adj_p[i] ≤ adj_p[i+1]
    adj_sorted = np.minimum(1.0, k * c_k * sorted_p / np.arange(1, k + 1))
    # Monotone enforcement (cummin from right)
    for i in range(k - 2, -1, -1):
        adj_sorted[i] = min(adj_sorted[i], adj_sorted[i + 1])

    # Map back to original order
    rejected   = np.empty(k, dtype=bool)
    adj_pvalue = np.empty(k, dtype=float)
    rejected[order]   = reject_sorted
    adj_pvalue[order] = adj_sorted
    return rejected, adj_pvalue


def _sig_stars(p: float | np.floating) -> str:
    """Significance stars from a p-value."""
    if not np.isfinite(p):
        return ""
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    if p < 0.10:
        return "."
    return ""


# ─────────────────────────────────────────────────────────────────────────────
# IC utilities
# ─────────────────────────────────────────────────────────────────────────────

def ic_series(
    pred: pd.DataFrame,
    *,
    date_col: str = "quote_date",
    signal_col: str = "p_hat",
    return_col: str = "y",
    min_cross_section: int = 3,
) -> pd.Series:
    """
    Daily cross-sectional IC (Spearman rank correlation).

    For each date, computes the Spearman correlation between the model's
    ranking signal (p_hat) and realised returns across all strategies.

    IC ≈ 0.02–0.05 is economically meaningful for daily equity strategies.
    """
    from scipy.stats import spearmanr  # type: ignore[import]

    def _daily_ic(group: pd.DataFrame) -> float:
        x = group[signal_col].to_numpy(dtype=float)
        y = group[return_col].to_numpy(dtype=float)
        mask = np.isfinite(x) & np.isfinite(y)
        if mask.sum() < min_cross_section:
            return np.nan
        r, _ = spearmanr(x[mask], y[mask])
        return float(r)

    g = pred.groupby(date_col, observed=True, sort=True)
    try:
        out = g.apply(_daily_ic, include_groups=False)
    except TypeError:
        out = g.apply(_daily_ic)
    return out.rename("ic")


def ic_series_intraday(
    pred: pd.DataFrame,
    *,
    date_col: str = "quote_date",
    time_col: str = "quote_time",
    signal_col: str = "p_hat",
    return_col: str = "y",
    min_cross_section: int = 3,
) -> pd.Series:
    """
    Daily IC computed per (date, time) slot then averaged by date.

    Preferred over ic_series() when predictions span multiple intraday time
    slots (multi-time panel).  Pooling strategies from different times in a
    single Spearman correlation mixes returns from incompatible horizons
    (e.g. PnL@10:00→close vs PnL@15:00→close have different distributions).
    This function evaluates ranking within each (date, time) slot — the
    correct cross-section for each decision moment — then averages per day.

    When predictions contain only one time slot per date, this function gives
    an identical result to ic_series().

    In ml_metrics() and comparison_table(), pass time_col="quote_time" to
    trigger this function automatically.  ic_by_time() uses the same logic
    but reports summary statistics per time slot rather than per day.

    Returns
    -------
    pd.Series indexed by date, one IC value per trading day.
    """
    from scipy.stats import spearmanr  # type: ignore[import]

    if time_col not in pred.columns:
        raise ValueError(f"time_col={time_col!r} not found in predictions — use ic_series() instead.")

    daily_ics: dict = {}
    for date, grp_d in pred.groupby(date_col, sort=True, observed=True):
        slot_ics: list[float] = []
        for _, grp_t in grp_d.groupby(time_col, sort=True, observed=True):
            x = grp_t[signal_col].to_numpy(dtype=float)
            y = grp_t[return_col].to_numpy(dtype=float)
            mask = np.isfinite(x) & np.isfinite(y)
            if mask.sum() < min_cross_section:
                continue
            r, _ = spearmanr(x[mask], y[mask])
            if np.isfinite(r):   # skip constant-prediction slots (ConstantInputWarning → NaN)
                slot_ics.append(float(r))
        daily_ics[date] = float(np.mean(slot_ics)) if slot_ics else np.nan

    return pd.Series(daily_ics, name="ic")


def icir(ic: pd.Series, annualise: bool = True, fill_nan_zero: bool = True) -> float:
    """
    Information Coefficient Information Ratio.

    ICIR = mean(IC) / std(IC) × √252 (annualised).
    Analogous to the Sharpe ratio of the IC series.

    fill_nan_zero=True (default): NaN IC days are treated as IC=0 before
    computing mean/std.  This is the unbiased estimator — a model that cannot
    rank strategies on some days contributes zero alpha on those days, and its
    ICIR is penalised accordingly.  Using dropna() (fill_nan_zero=False) inflates
    ICIR for models with many dead days (e.g. over-regularised Lasso).
    """
    if fill_nan_zero:
        # Keep the full length (including dead days as 0) for mean and std
        s = ic.fillna(0.0)
    else:
        s = ic.dropna()
    if len(s) < 2:
        return np.nan
    mu, sd = float(s.mean()), float(s.std())
    if sd < 1e-12:
        return np.nan
    return (mu / sd) * (np.sqrt(252.0) if annualise else 1.0)


def ic_tstat(ic: pd.Series, newey_west: bool = True, max_lag: int | None = None) -> float:
    """
    IC t-statistic with optional Newey-West (HAC) standard error.

    Answers: "Is the mean IC statistically different from zero?"
    t > 1.96 → significant at 5%.

    newey_west=True (default): uses HAC variance that accounts for serial
    correlation in the IC series. Daily IC is autocorrelated when features
    have rolling lookbacks (e.g. pnl_mean5_l1), causing the naive t-stat
    to be inflated by 20–30%. NW corrects this, giving a more conservative
    and honest t-stat.

    max_lag: number of lags in the NW kernel. Defaults to the Andrews (1991)
    data-driven rule: floor(4 × (N/100)^(2/9)). For N=500 days this gives
    lag≈5, matching one week of autocorrelation. Set explicitly to override.

    newey_west=False: falls back to the naive mean/std × √N formula (i.i.d.
    assumption). Use only for comparison or very short IC series (N < 20).
    """
    s = ic.dropna()
    n = len(s)
    if n < 2:
        return np.nan
    mu = float(s.mean())

    if not newey_west:
        sd = float(s.std())
        if sd < 1e-12:
            return np.nan
        return mu / sd * np.sqrt(n)

    # Newey-West HAC variance of the IC series
    # V_NW = γ_0 + 2 Σ_{l=1}^{L} w_l γ_l   with w_l = 1 − l/(L+1) (Bartlett kernel)
    # SE_NW = sqrt(V_NW / N),  t_NW = mean(IC) / SE_NW
    x = s.to_numpy(dtype=float) - mu          # demeaned
    L = max_lag if max_lag is not None else int(np.floor(4.0 * (n / 100.0) ** (2.0 / 9.0)))
    L = max(1, min(L, n - 1))

    gamma0 = float(np.dot(x, x)) / n          # lag-0 autocovariance = variance
    nw_var = gamma0
    for lag in range(1, L + 1):
        gamma_l = float(np.dot(x[lag:], x[:-lag])) / n
        nw_var += 2.0 * (1.0 - lag / (L + 1.0)) * gamma_l

    if nw_var <= 1e-24:
        return np.nan

    se_nw = np.sqrt(nw_var / n)
    return float(mu / se_nw)


# ─────────────────────────────────────────────────────────────────────────────
# Calibration
# ─────────────────────────────────────────────────────────────────────────────

def _calibration(
    y_true: np.ndarray,
    p_hat: np.ndarray,
) -> tuple[float, float]:
    """
    Platt calibration via logistic regression on logit(p_hat).

    Returns (slope, intercept).
      slope ≈ 1, intercept ≈ 0  →  well-calibrated
      slope < 1                 →  over-confident (predictions too extreme)
      intercept > 0             →  systematic bullish bias

    Returns (NaN, NaN) if statsmodels is not installed or fitting fails.
    No linear fallback — the linear interpretation differs from Platt/logit.
    """
    if len(y_true) < 30:
        return np.nan, np.nan
    p = np.clip(p_hat, 1e-6, 1 - 1e-6)
    logit_p = np.log(p / (1 - p))
    try:
        import statsmodels.api as sm  # type: ignore[import]
        X = sm.add_constant(logit_p)
        mod = sm.Logit(y_true.astype(float), X).fit(disp=0)
        return float(mod.params[1]), float(mod.params[0])
    except Exception:
        return np.nan, np.nan


# ─────────────────────────────────────────────────────────────────────────────
# ML metrics
# ─────────────────────────────────────────────────────────────────────────────

def ml_metrics(
    pred: pd.DataFrame,
    *,
    task: Literal["binary", "regression"] = "binary",
    binary_threshold: float = 0.0,
    date_col: str = "quote_date",
    time_col: str | None = None,
    signal_col: str = "p_hat",
    return_col: str = "y",
    bin_col: str = "y_bin",
    sign_col: str = "sign",
) -> dict[str, float]:
    """
    Compute predictive quality metrics (no cost assumptions).

    Parameters
    ----------
    pred
        OOS prediction DataFrame (one model, one protocol).
    binary_threshold
        Return level used to define the positive class (must match runner config).
    time_col
        If provided and the column exists with >1 unique value per date, IC is
        computed per (date, time) slot then averaged per day via
        ic_series_intraday().  Use this when predictions contain multiple intraday
        time slots (multi-time panel).  If None or single time per date, falls back
        to ic_series() — same result as before.

    Returns
    -------
    dict with keys:
      ic_oos        — mean daily cross-sectional IC on OOS predictions
      icir          — annualised ICIR
      ic_tstat      — IC t-statistic (significance of mean IC, Newey-West HAC)
      ic_mode       — "intraday" if ic_series_intraday() was used, "daily" otherwise
      hit_rate      — fraction of correctly signed predictions (NaN rows excluded)
      auc           — ROC-AUC (NaN if only one class present)
      brier         — Brier score; lower = better
      brier_skill   — 1 − brier/brier_naive; > 0 beats constant-probability baseline
      log_loss      — cross-entropy; lower = better
      calib_slope   — Platt calibration slope (1.0 = well-calibrated)
      calib_intercept — Platt calibration intercept (0.0 = no directional bias)
      coverage      — fraction of rows with a valid (non-NaN) prediction
      long_share    — fraction of predictions ≥ 0.5 (model symmetry check)
      p_mean        — mean predicted probability
      p_std         — std of predicted probabilities
      n_obs         — total row count
      n_valid_pred  — rows with finite p_hat
      n_days        — number of unique test dates
    """
    from sklearn.metrics import log_loss as sk_log_loss, roc_auc_score, balanced_accuracy_score  # type: ignore[import]

    if pred.empty:
        return {}

    y        = pred[return_col].to_numpy(dtype=float)
    y_bin    = pred[bin_col].to_numpy(dtype=float)   # float — may contain NaN
    p_hat    = pred[signal_col].to_numpy(dtype=float)
    sign_pred = pred[sign_col].to_numpy(dtype=float)
    sign_true = np.where(y > binary_threshold, 1.0, -1.0)

    n_obs = len(pred)

    # ── IC metrics ────────────────────────────────────────────────────────────
    # Use intraday IC when time_col is present and multiple times exist per date.
    # Rationale: pooling multiple time slots in a single Spearman correlation
    # mixes returns from incompatible horizons (e.g. PnL@10:00→close vs
    # PnL@15:00→close).  ic_series_intraday() evaluates ranking within each
    # (date, time) slot and averages, which is the correct cross-section.
    _use_intraday = (
        time_col is not None
        and time_col in pred.columns
        and pred.groupby(date_col, observed=True)[time_col].nunique().max() > 1
    )
    if _use_intraday:
        ics = ic_series_intraday(
            pred, date_col=date_col, time_col=time_col,
            signal_col=signal_col, return_col=return_col,
        )
        ic_mode_used = "intraday"
    else:
        ics = ic_series(pred, date_col=date_col, signal_col=signal_col, return_col=return_col)
        ic_mode_used = "daily"

    ic_dead_days = int(ics.isna().sum())
    ic_oos    = float(ics.fillna(0.0).mean()) if len(ics) > 0 else np.nan
    icir_val  = icir(ics)     # fill_nan_zero=True by default
    ictstat   = ic_tstat(ics)

    # ── Hit rate (NaN signal → skip row) ─────────────────────────────────────
    valid_hit = np.isfinite(sign_pred) & np.isfinite(y)
    hit = float(np.mean(sign_pred[valid_hit] == sign_true[valid_hit])) if valid_hit.any() else np.nan

    # ── Binary metrics (classification only) ─────────────────────────────────
    valid_bin = np.isfinite(y_bin) & np.isfinite(p_hat)
    bal_acc: float = np.nan
    auc = ll = brier = brier_skill = calib_slope = calib_intercept = np.nan

    if task == "binary" and valid_bin.any():
        yb = y_bin[valid_bin].astype(int)
        ph = np.clip(p_hat[valid_bin], 1e-7, 1 - 1e-7)

        if len(np.unique(yb)) >= 2:
            y_pred_bin = (ph >= 0.5).astype(int)
            bal_acc = float(balanced_accuracy_score(yb, y_pred_bin))
            try:
                auc = float(roc_auc_score(yb, ph))
            except Exception:
                pass
            try:
                ll = float(sk_log_loss(yb, ph))
            except Exception:
                pass

        brier = float(np.mean((ph - yb) ** 2))
        base_rate = float(np.mean(yb))
        brier_naive = float(np.mean((base_rate - yb) ** 2))
        brier_skill = (1.0 - brier / brier_naive) if brier_naive > 1e-12 else np.nan
        calib_slope, calib_intercept = _calibration(yb, ph)

    # ── Regression metrics ────────────────────────────────────────────────────
    mae = r2 = dir_acc = np.nan
    if task == "regression":
        valid_reg = np.isfinite(p_hat) & np.isfinite(y)
        if valid_reg.any():
            yv, yh = y[valid_reg], p_hat[valid_reg]
            mae = float(np.mean(np.abs(yh - yv)))
            ss_res = float(np.sum((yv - yh) ** 2))
            ss_tot = float(np.sum((yv - yv.mean()) ** 2))
            r2  = (1.0 - ss_res / ss_tot) if ss_tot > 1e-15 else np.nan
            # directional accuracy: sign(y_hat) == sign(y_actual)
            dir_acc = float(np.mean(np.sign(yh) == np.sign(yv)))

    # ── Prediction coverage & distribution ───────────────────────────────────
    valid_p      = np.isfinite(p_hat)
    n_valid_pred = int(valid_p.sum())
    coverage     = float(n_valid_pred / n_obs) if n_obs > 0 else np.nan
    p_mean       = float(np.nanmean(p_hat)) if valid_p.any() else np.nan
    p_std        = float(np.nanstd(p_hat))  if valid_p.any() else np.nan
    long_share   = float(np.nanmean(p_hat[valid_p] >= 0.5)) if valid_p.any() else np.nan

    return {
        "ic_oos":          ic_oos,
        "icir":            icir_val,
        "ic_tstat":        ictstat,
        "ic_mode":         ic_mode_used,
        "ic_dead_days":    ic_dead_days,
        "hit_rate":        hit,
        # classification
        "bal_acc":         bal_acc,
        "auc":             auc,
        "brier":           brier,
        "brier_skill":     brier_skill,
        "log_loss":        ll,
        "calib_slope":     calib_slope,
        "calib_intercept": calib_intercept,
        # regression
        "mae":             mae,
        "r2":              r2,
        "dir_acc":         dir_acc,
        # prediction distribution
        "coverage":        coverage,
        "long_share":      long_share,
        "p_mean":          p_mean,
        "p_std":           p_std,
        "n_obs":           n_obs,
        "n_valid_pred":    n_valid_pred,
        "n_days":          int(pred[date_col].nunique()),
    }


# ─────────────────────────────────────────────────────────────────────────────
# IC gap — overfitting diagnostic
# ─────────────────────────────────────────────────────────────────────────────

def ic_gap(
    split_log: pd.DataFrame,
    oos_predictions: pd.DataFrame,
    *,
    date_col: str = "quote_date",
    time_col: str | None = None,
    signal_col: str = "p_hat",
    return_col: str = "y",
) -> dict[str, dict]:
    """
    IS–OOS IC gap per model.

    IS IC : mean IC stored in split_log (last is_eval_days of each training window).
            Computed by the runner using pooled daily IC (diagnostic only).
    OOS IC: mean of the IC series on test predictions, using the same IC method
            as ml_metrics() — intraday if time_col is provided and multiple time
            slots are present, daily otherwise.
    Gap   : IS IC − OOS IC.  Large positive → overfitting.

    Note: IS IC in split_log matches OOS IC method when WalkForwardConfig.ic_time_col
    is set — both use intraday slot averaging for multi-time panels.  If ic_time_col
    is None (default), both use pooled-daily IC.  The gap is exact in both cases.

    Parameters
    ----------
    time_col : passed through to select ic_series_intraday() for OOS IC when
               predictions contain multiple time slots per date.

    Returns dict[model_id → {'is_ic', 'oos_ic', 'ic_gap'}]
    """
    results: dict[str, dict] = {}
    model_ids = split_log["model_id"].unique() if not split_log.empty else []

    for model_id in model_ids:
        log_m     = split_log[split_log["model_id"] == model_id]
        is_ic_val = float(log_m["is_ic"].dropna().mean()) if "is_ic" in log_m.columns else np.nan

        oos_ic_val = np.nan
        if not oos_predictions.empty and "model_id" in oos_predictions.columns:
            pred_m = oos_predictions[oos_predictions["model_id"] == model_id]
            _use_intraday = (
                time_col is not None
                and time_col in pred_m.columns
                and pred_m.groupby(date_col, observed=True)[time_col].nunique().max() > 1
            )
            if _use_intraday:
                ics = ic_series_intraday(
                    pred_m, date_col=date_col, time_col=time_col,
                    signal_col=signal_col, return_col=return_col,
                )
            else:
                ics = ic_series(pred_m, date_col=date_col, signal_col=signal_col, return_col=return_col)
            oos_ic_val = float(ics.fillna(0.0).mean()) if len(ics) > 0 else np.nan

        results[str(model_id)] = {
            "is_ic":  is_ic_val,
            "oos_ic": oos_ic_val,
            "ic_gap": is_ic_val - oos_ic_val if np.isfinite(is_ic_val + oos_ic_val) else np.nan,
        }

    return results


# ─────────────────────────────────────────────────────────────────────────────
# Model comparison table
# ─────────────────────────────────────────────────────────────────────────────

def comparison_table(  # noqa: C901
    result_predictions: pd.DataFrame,
    result_split_log: pd.DataFrame,
    model_zoo_labels: dict[str, str] | None = None,
    *,
    task: Literal["binary", "regression"] = "binary",
    binary_threshold: float = 0.0,
    date_col: str = "quote_date",
    time_col: str | None = None,
    signal_col: str = "p_hat",
    return_col: str = "y",
) -> pd.DataFrame:
    """
    Build a model comparison table sorted by OOS ICIR descending.

    All metrics are purely predictive quality — no economic / backtest metrics.

    Parameters
    ----------
    time_col
        Passed through to ml_metrics(). Set to "quote_time" when predictions
        contain multiple intraday time slots (multi-time panel) so that IC is
        computed per (date, time) slot — see ml_metrics() for full rationale.
    return_col
        Column used for IC computation.  Default "y" (binary training target).
        Set to "target_y_long_net" for continuous-PnL IC as defined in Vilkov
        (2026): IC = Spearman(p_hat, y_raw) is a stricter test than binary IC.

    Columns:
      model_id, label,
      ic_oos, icir, ic_tstat, ic_mode, ← ranking signal + stats + IC method used
      ic_is, ic_gap,                   ← overfitting diagnostic
      hit_rate, auc, brier_skill,      ← direction + classification
      calib_slope, calib_intercept,    ← calibration
      coverage, long_share,            ← prediction diagnostics
      n_obs, n_valid_pred, n_days

    Model selection heuristic (Stage 1):
      1. ic_oos > 0  and  ic_tstat > 1.96
      2. ic_gap not too large  (< 0.05 is comfortable)
      3. auc > 0.50  and  brier_skill > 0
      4. calib_slope ≈ 1.0  and  calib_intercept ≈ 0
      5. coverage ≈ 1.0  and  long_share ≈ 0.5

    Only Stage 2 result can be reported as honest OOS in a paper.
    """
    if result_predictions.empty:
        return pd.DataFrame()

    model_ids = sorted(result_predictions["model_id"].unique())

    # ic_gap must compare IS IC vs OOS IC on the SAME target (binary).
    # IS IC in split_log is always computed with the binary training target.
    # Using return_col (e.g. "target_y_long_net") here would mix binary IS
    # vs continuous OOS → inflated gap (~0.11 artefact, not real overfitting).
    # We use the binary column "y" (= y_bin) for gap, and separately compute
    # ic_oos_bin to make the comparison explicit in the output table.
    _bin_col = "y" if "y" in result_predictions.columns else return_col
    gap_dict  = ic_gap(result_split_log, result_predictions,
                       date_col=date_col, time_col=time_col,
                       signal_col=signal_col, return_col=_bin_col)

    # Binary OOS IC per model (for ic_gap validation and cross-reference)
    _bin_oos: dict[str, float] = {}
    for mid in model_ids:
        pred_m = result_predictions[result_predictions["model_id"] == mid]
        _use_intraday = (
            time_col is not None
            and time_col in pred_m.columns
            and pred_m.groupby(date_col, observed=True)[time_col].nunique().max() > 1
        )
        if _use_intraday:
            _s = ic_series_intraday(pred_m, date_col=date_col, time_col=time_col,
                                    signal_col=signal_col, return_col=_bin_col)
        else:
            _s = ic_series(pred_m, date_col=date_col, signal_col=signal_col, return_col=_bin_col)
        _bin_oos[str(mid)] = float(_s.fillna(0.0).mean()) if len(_s) > 0 else np.nan

    rows = []
    for mid in model_ids:
        pred_m = result_predictions[result_predictions["model_id"] == mid]
        ml     = ml_metrics(pred_m, task=task, binary_threshold=binary_threshold,
                            date_col=date_col, time_col=time_col,
                            signal_col=signal_col, return_col=return_col)
        gap    = gap_dict.get(str(mid), {})
        label  = (model_zoo_labels or {}).get(str(mid), str(mid))

        rows.append({
            "model_id":        mid,
            "label":           label,
            # IC — main (continuous return_col)
            "ic_oos":          ml.get("ic_oos"),
            "icir":            ml.get("icir"),
            "ic_tstat":        ml.get("ic_tstat"),
            "ic_mode":         ml.get("ic_mode"),
            "ic_dead_days":    ml.get("ic_dead_days"),
            # Overfitting: both IS and OOS on binary y — apples-to-apples
            "ic_is":           gap.get("is_ic"),
            "ic_oos_bin":      _bin_oos.get(str(mid)),
            "ic_gap":          gap.get("ic_gap"),
            # Classification
            "hit_rate":        ml.get("hit_rate"),
            "bal_acc":         ml.get("bal_acc"),
            "auc":             ml.get("auc"),
            "brier_skill":     ml.get("brier_skill"),
            "log_loss":        ml.get("log_loss"),
            # Calibration
            "brier":           ml.get("brier"),
            "calib_slope":     ml.get("calib_slope"),
            "calib_intercept": ml.get("calib_intercept"),
            # Regression
            "mae":             ml.get("mae"),
            "r2":              ml.get("r2"),
            "dir_acc":         ml.get("dir_acc"),
            # Diagnostics
            "coverage":        ml.get("coverage"),
            "long_share":      ml.get("long_share"),
            "p_mean":          ml.get("p_mean"),
            "p_std":           ml.get("p_std"),
            # Counts
            "n_obs":           ml.get("n_obs"),
            "n_valid_pred":    ml.get("n_valid_pred"),
            "n_days":          ml.get("n_days"),
        })

    df = pd.DataFrame(rows)
    if df.empty:
        return df

    # ── Multiple-testing correction (BHY) on IC t-statistics ─────────────────
    # Each model is one "hypothesis": H0 = IC = 0.
    # Selecting the best of k models based on raw t-stat inflates Type I error.
    # BHY controls the False Discovery Rate (FDR) under arbitrary dependence —
    # appropriate here because all models are trained on the same data.
    t_stats = df["ic_tstat"].to_numpy(dtype=float)
    n_days  = df["n_days"].fillna(1).to_numpy(dtype=float)
    p_raw   = _pvalues_from_tstats(t_stats, n_days)
    rejected, p_bhy = _bhy_correction(p_raw)

    df["p_ic_raw"] = p_raw
    df["p_ic_bhy"] = p_bhy
    df["sig_raw"]  = [_sig_stars(p) for p in p_raw]
    df["sig_bhy"]  = np.where(rejected, df["sig_raw"], "")

    if "icir" in df.columns:
        df = df.sort_values("icir", ascending=False, na_position="last").reset_index(drop=True)
    return df


def comparison_table_by_group(
    result_predictions: pd.DataFrame,
    model_zoo_labels: dict[str, str] | None = None,
    *,
    group_col: str = "strategy_family",
    task: Literal["binary"] = "binary",
    binary_threshold: float = 0.0,
    date_col: str = "quote_date",
    time_col: str | None = None,
    signal_col: str = "p_hat",
    return_col: str = "y",
) -> pd.DataFrame:
    """
    Per-group breakdown of ML metrics.

    Same metrics as comparison_table() but computed separately for each
    value of group_col (e.g. strategy_family or option_type).  Useful for
    checking whether a model's aggregate IC is driven by one family only.

    No IS IC / ic_gap — the split_log is not broken down by group.
    Returns a DataFrame with (model_id, <group_col>) as the key columns.
    Groups absent from predictions are silently skipped.

    Parameters
    ----------
    time_col   : passed through to ml_metrics() — use "quote_time" for multi-time panel.
    return_col : column used for IC (Spearman).  Default "y" (binary training target).
                 Use "target_y_long_net" for continuous-PnL IC (Vilkov 2026 definition).
    """
    if result_predictions.empty or group_col not in result_predictions.columns:
        return pd.DataFrame()

    groups    = sorted(result_predictions[group_col].dropna().unique())
    model_ids = sorted(result_predictions["model_id"].unique())
    rows: list[dict] = []

    for mid in model_ids:
        label = (model_zoo_labels or {}).get(str(mid), str(mid))
        for grp in groups:
            mask   = (result_predictions["model_id"] == mid) & (result_predictions[group_col] == grp)
            pred_g = result_predictions[mask]
            if pred_g.empty:
                continue
            ml = ml_metrics(pred_g, task=task, binary_threshold=binary_threshold,
                            date_col=date_col, time_col=time_col,
                            signal_col=signal_col, return_col=return_col)
            rows.append({
                "model_id":    mid,
                "label":       label,
                group_col:     grp,
                "ic_oos":      ml.get("ic_oos"),
                "icir":        ml.get("icir"),
                "ic_tstat":    ml.get("ic_tstat"),
                "ic_mode":     ml.get("ic_mode"),
                "hit_rate":    ml.get("hit_rate"),
                "bal_acc":     ml.get("bal_acc"),
                "auc":         ml.get("auc"),
                "brier_skill": ml.get("brier_skill"),
                "coverage":    ml.get("coverage"),
                "long_share":  ml.get("long_share"),
                "n_obs":       ml.get("n_obs"),
                "n_days":      ml.get("n_days"),
            })

    df = pd.DataFrame(rows)
    if not df.empty and "icir" in df.columns:
        df = df.sort_values(["model_id", "icir"], ascending=[True, False]).reset_index(drop=True)
    return df


def print_comparison_summary(table: pd.DataFrame, top_n: int = 10) -> None:
    """Print a readable summary of the comparison table (top N models by ICIR)."""
    cols = [c for c in [
        "model_id", "label",
        "ic_oos", "ic_tstat", "icir", "ic_gap", "ic_mode",
        "auc", "brier_skill", "coverage", "long_share",
        "n_days",
    ] if c in table.columns]
    print(table[cols].head(top_n).to_string(index=False, float_format=lambda x: f"{x:.4f}"))


def ic_by_time(
    pred: pd.DataFrame,
    *,
    date_col: str = "quote_date",
    time_col: str = "quote_time",
    signal_col: str = "p_hat",
    return_col: str = "y",
    min_cross_section: int = 3,
) -> pd.DataFrame:
    """
    IC broken down by intraday time slot.

    For each quote_time, computes the daily IC series (Spearman across
    strategies/mnes on that date × time), then summarises as mean IC and
    ICIR per time slot.

    Use this AFTER model selection (Stage 2 or Stage 1 best model) to
    diagnose whether predictability is concentrated at specific times of day
    (e.g. open vs close) rather than uniform across the session.

    Returns
    -------
    DataFrame indexed by quote_time with columns:
      ic_mean, icir, ic_tstat, n_days
    Sorted by quote_time ascending.
    """
    from scipy.stats import spearmanr  # type: ignore[import]

    if time_col not in pred.columns:
        raise ValueError(f"time_col={time_col!r} not found in predictions.")

    records = []
    for t, grp_t in pred.groupby(time_col, sort=True, observed=True):
        daily_ics: list[float] = []
        for _, grp_d in grp_t.groupby(date_col, sort=True, observed=True):
            x = grp_d[signal_col].to_numpy(dtype=float)
            y = grp_d[return_col].to_numpy(dtype=float)
            mask = np.isfinite(x) & np.isfinite(y)
            if mask.sum() < min_cross_section:
                continue
            r, _ = spearmanr(x[mask], y[mask])
            if np.isfinite(r):
                daily_ics.append(float(r))

        s = pd.Series(daily_ics, dtype=float)
        records.append({
            time_col:   t,
            "ic_mean":  float(s.mean())      if len(s) >= 1 else np.nan,
            "icir":     icir(s)              if len(s) >= 2 else np.nan,
            "ic_tstat": ic_tstat(s)          if len(s) >= 2 else np.nan,
            "n_days":   len(s),
        })

    return pd.DataFrame(records).set_index(time_col)


def save_stage1_winner(
    table: pd.DataFrame,
    zoo: dict,
    config,
    feature_cols: list[str],
    target_col: str,
    path: str,
    *,
    best_model_id: str | None = None,
    dataset_info: dict | None = None,
    stage1_end_date: str | None = None,
    notes: str | None = None,
) -> dict:
    """
    Save the Stage 1 winner configuration as a JSON artifact.

    Call this AFTER inspecting comparison_table() and choosing the winner,
    BEFORE running Stage 2.  The saved JSON is the single source of truth
    for reproducing Stage 2 — it contains every choice made in Stage 1.

    Stage 2 rule: once this file is written, NO further changes are allowed.

    Parameters
    ----------
    table           : output of comparison_table()
    zoo             : model zoo dict used in Stage 1
    config          : WalkForwardConfig used in Stage 1
    feature_cols    : feature columns passed to walk_forward()
    target_col      : target column passed to walk_forward()
    path            : where to write the JSON (e.g. "output/models/stage1_winner.json")
    best_model_id   : override winner; if None uses top ICIR row
    dataset_info    : optional dict e.g. {"name": "massive_strategy_5min", "path": "..."}
    stage1_end_date : last date of Stage 1 period (e.g. "2025-06-30")
    notes           : free-text note about why this model was chosen

    Returns
    -------
    The dict that was saved.
    """
    import dataclasses, json
    from pathlib import Path

    if table.empty:
        raise ValueError("comparison_table is empty — run Stage 1 first.")

    mid = best_model_id if best_model_id is not None else str(table.iloc[0]["model_id"])
    if mid not in zoo:
        raise KeyError(f"model_id={mid!r} not found in zoo.")

    row  = table[table["model_id"] == mid].iloc[0]
    spec = zoo[mid]

    def _cfg_dict(obj) -> dict:
        if dataclasses.is_dataclass(obj):
            return {k: v for k, v in dataclasses.asdict(obj).items() if not callable(v)}
        return str(obj)

    def _f(key: str) -> float | None:
        v = row.get(key)
        return float(v) if pd.notna(v) else None

    artifact = {
        "best_model_id":    mid,
        "best_model_label": str(row.get("label", mid)),
        "selection_ts":     pd.Timestamp.now().isoformat(),
        "notes":            notes,

        # ── Stage 1 metrics (record only — do NOT tune on these) ────────────
        "stage1_metrics": {
            "ic_oos":      _f("ic_oos"),
            "icir":        _f("icir"),
            "ic_tstat":    _f("ic_tstat"),
            "ic_is":       _f("ic_is"),
            "ic_gap":      _f("ic_gap"),
            "auc":         _f("auc"),
            "brier_skill": _f("brier_skill"),
            "n_days":      int(row["n_days"]) if pd.notna(row.get("n_days")) else None,
        },

        # ── Experiment config (everything needed to reproduce Stage 2) ──────
        "experiment": {
            "task":              config.task,
            "mode":              config.mode,
            "group_col":         config.group_col,
            "date_col":          config.date_col,
            "id_cols":           list(config.id_cols),
            "target_col":        target_col,
            "binary_threshold":  float(config.binary_threshold),
            "compute_is_metrics": bool(config.compute_is_metrics),
            "is_eval_days":      int(config.is_eval_days),
        },

        "stage1_period": {
            "oos_start_date": str(config.split_config.oos_start_date),
            "oos_end_date":   stage1_end_date,
        },

        "feature_cols":        list(feature_cols),
        "split_config":        _cfg_dict(config.split_config),
        "preprocessor_config": _cfg_dict(config.preprocessor_config),

        # ── Model ────────────────────────────────────────────────────────────
        "model_class":  spec.estimator.__class__.__name__,
        "model_module": spec.estimator.__class__.__module__,
        "model_params": spec.estimator.get_params(),
        "model_label":  spec.label,

        "dataset_info": dataset_info or {},
    }

    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(artifact, f, indent=2, default=str)

    def _fmt(x: object) -> str:
        try:
            return "nan" if x is None or not np.isfinite(float(x)) else f"{float(x):.4f}"
        except (TypeError, ValueError):
            return "nan"

    m = artifact["stage1_metrics"]
    print(f"Stage 1 winner saved → {out}")
    print(f"  model   : {mid}  ({artifact['best_model_label']})")
    print(f"  ic_oos  : {_fmt(m['ic_oos'])}  |  icir: {_fmt(m['icir'])}  |  ic_gap: {_fmt(m['ic_gap'])}")
    print(f"  mode    : {config.mode}  |  group_col: {config.group_col}")
    print(f"  features: {len(feature_cols)}  |  target: {target_col}")
    if notes:
        print(f"  notes   : {notes}")
    return artifact


def annualised_sharpe(series: pd.Series) -> float:
    """Annualised Sharpe ratio from a daily return series."""
    s = series.dropna()
    if len(s) < 2:
        return np.nan
    std = float(s.std())
    if std < 1e-12:
        return np.nan
    return float(s.mean() / std * np.sqrt(252.0))

'''
# ─────────────────────────────────────────────────────────────────────────────
# Best-model JSON summary
# ─────────────────────────────────────────────────────────────────────────────

def best_model_json(
    table: pd.DataFrame,
    split_log: pd.DataFrame,
    model_zoo: "dict",
    preprocessor_config: Any | None = None,
    *,
    rank_by: str = "icir",
) -> dict:
    """
    Return a JSON-safe dict summarising the best model and its configuration.

    Includes:
      - all metrics from comparison_table() for the winning model
      - model hyperparameters from model_zoo (ModelSpec.to_dict())
      - preprocessor configuration (PreprocessorConfig.to_dict())
      - split_log averages (mean IS IC, mean n_train_obs, etc.)

    Parameters
    ----------
    table              : output of comparison_table()
    split_log          : output of WalkForwardResult.split_log
    model_zoo          : dict[model_id → ModelSpec]
    preprocessor_config: PreprocessorConfig (optional)
    rank_by            : metric to rank by (default "icir")
    """
    import json as _json
    if table.empty:
        return {}

    col = rank_by if rank_by in table.columns else "icir"
    best_row = table.dropna(subset=[col]).sort_values(col, ascending=False).iloc[0]
    best_id  = str(best_row["model_id"])

    # Model hyperparameters
    spec  = model_zoo.get(best_id)
    model_params = spec.to_dict() if spec is not None and hasattr(spec, "to_dict") else {}

    # Preprocessor config
    pp_params = {}
    if preprocessor_config is not None and hasattr(preprocessor_config, "to_dict"):
        pp_params = preprocessor_config.to_dict()

    # Split log averages for this model
    log_m = split_log[split_log["model_id"] == best_id] if not split_log.empty else pd.DataFrame()
    pp_cols = [c for c in log_m.columns if c.startswith("pp_")]
    split_summary: dict = {}
    for col_name in ["n_train_obs", "n_train_days", "train_pos_rate",
                     "is_ic", "n_nan_feat_rows_train"] + pp_cols:
        if col_name in log_m.columns:
            split_summary[f"mean_{col_name}"] = float(log_m[col_name].mean())

    result = {
        "best_model_id":    best_id,
        "rank_by":          rank_by,
        "metrics":          {k: (None if pd.isna(v) else v)
                             for k, v in best_row.to_dict().items()},
        "model_params":     model_params,
        "preprocessor":     pp_params,
        "split_summary":    split_summary,
    }
    # Validate JSON-serialisable
    try:
        _json.dumps(result)
    except TypeError:
        # Convert non-serialisable values to strings as fallback
        result = _json.loads(_json.dumps(result, default=str))
    return result
'''