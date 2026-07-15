"""Feature catalog for 0DTE conditional panel.

Categorises features from massive_build_conditional_features.py into
four preprocessing groups (see get_feature_cols("all") for current count):

  dummies   — binary / one-hot indicators: pass through unchanged
  cs        — already cross-sectionally normalised: winsorise lightly, no z-score
  ts        — market-level / lagged: winsorise + z-score (training window only)
  raw       — raw prices / Greeks / ratios: winsorise only

Also provides:
  categorize_features(cols)      — assign any column list to the four groups
  add_cs_features(df, ...)       — compute cross-sectional z-scores by group
  add_sqtr_features(df, cols)    — signed square-root transform to reduce skewness
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# ─────────────────────────────────────────────────────────────────────────────
# Detection rules
# ─────────────────────────────────────────────────────────────────────────────

# Prefixes → strategy-specific raw features (winsorise only, no z-score).
# Checked BEFORE dummies so that strat_vol_* / strat_txn_* are not caught
# by the broader "strat_" dummy prefix.
RAW_OVERRIDE_PREFIXES: tuple[str, ...] = (
    "strat_vol_",    # strategy OHLCV lags — strategy-specific, varies per (strategy,mnes)
    "strat_txn_",    # strategy transaction lags — strategy-specific
    "pnl_lag",           # per-strategy long-side PnL lags
    "pnl_mean",          # per-strategy long-side rolling PnL mean
    "pnl_std",           # per-strategy long-side rolling PnL std
    "pnl_short_lag",     # per-strategy short-side PnL lags
    "pnl_short_mean",    # per-strategy short-side rolling PnL mean
    "pnl_short_std",     # per-strategy short-side rolling PnL std
    "vilkov_v",      # per-strategy volume flow (lagged)
    "vilkov_fD",     # per-strategy delta flow (lagged)
    "vilkov_fG",     # per-strategy gamma flow (lagged)
    "vilkov_fV",     # per-strategy vega flow (lagged)
    "vilkov_txn",    # per-strategy transaction count (lagged)
)

# Prefixes/exact names → dummies (pass through, no scaling)
DUMMY_PREFIXES: tuple[str, ...] = (
    "strat_",       # one-hot strategy type (strat_strangle, strat_iron_condor, ...)
                    # strat_vol_/strat_txn_ are caught by RAW_OVERRIDE_PREFIXES first
    "is_",          # event dummies (is_cpi_day, is_fomc_day, ...)
    "post_",        # post_fomc_time
    "pre_",         # pre_fomc_time
    "cpi_or_",      # cpi_or_nfp_day
    "event_",       # event_CPI, event_NFP, event_FOMC
)
DUMMY_SUFFIXES: tuple[str, ...] = ("_dummy",)  # first_30min_dummy, last_30min_dummy, etc.
DUMMY_EXACT: frozenset[str] = frozenset({
    "illiquid_dummy",
    "all_liquid_int",
    "flow_all_legs_matched",     # binary coverage flag (0/1)
    "vilkov_all_legs_matched",   # binary coverage flag (0/1)
    # bar_index removed: it is 0..71 (ordinal), goes to raw
})

# Suffixes → already CS-normalised (light winsorise only, no z-score)
CS_SUFFIXES: tuple[str, ...] = (
    "_cs",
    "_cs_all",
    "_cs_family",
    "_cs_strategy",
    "_timeslot_z",
)

# Prefixes → columns that typically have large skewness → candidates for sqtr transform
# sign(x) * sqrt(|x|) compresses heavy tails while preserving sign and direction.
SQTR_PREFIXES: tuple[str, ...] = (
    "abs_",             # absolute values — always ≥ 0, right-skewed
    "log_",             # log-transformed but may still have outliers
    "cost_",            # cost ratios — right-skewed
    "vilkov_v",         # Greek-weighted volume — heavy right tail
    "vilkov_fD",
    "vilkov_fG",
    "vilkov_fV",
    "vilkov_txn",       # transaction counts — heavy right tail
    "ps_max_profit",    # payoff shape features — bounded below by 0
    "ps_max_loss",
    "ps_dist_be",       # breakeven distances — bounded below by 0
    "depth_per_",       # depth ratios
    "gp_per_",
    "strat_vol_",       # strategy-level OHLCV — volume-like, right-skewed
)
SQTR_EXACT: frozenset[str] = frozenset({
    "d",                # depth — right-skewed
    "n_contracts",
})

# Suffix added by add_sqtr_features
SQTR_SUFFIX: str = "_sqtr"

# Prefixes/substrings → time-series features (winsorise + z-score)
#
# STRICT DESIGN RULE: ts = MARKET-LEVEL ONLY
#
# A feature is ts (winsorise + z-score in training window) if and only if it has
# the SAME value for all 64 strategies at a given (date, time).  These are
# market-wide state variables: implied vol, SPX intraday, macro, AFT surface.
#
# Strategy-specific features (different value per strategy at the same date×time)
# go in raw (winsorise only) regardless of whether they are time-varying.
# Their cross-sectional normalization is handled by the _cs suffix columns.
#
# Examples of correct assignment:
#   MARKET-LEVEL → ts : ivar_0dte (same for all strategies), spx_rv_since_1000,
#                        vrp_lag1_same_time (SPX VRP, same for all), aft_atm_iv
#   STRATEGY-SPECIFIC → raw : pnl_lag1_same_time (each strategy has its own PnL),
#                              vilkov_v (each strategy has its own volume flow),
#                              strat_vol_lag5 (each strategy has its own OHLCV)
#
# For the _sqtr columns: they inherit the category of their source via prefix
# matching (e.g. ivar_0dte_sqtr → ts via "ivar_"; vilkov_v_sqtr → raw via no
# ts-prefix match).  This is automatic and correct.
TS_PREFIXES: tuple[str, ...] = (
    # ── Implied vol / surface state — market-level ────────────────────────────
    "atm_iv", "ivar_", "slope_", "expected_move_", "opt_iv_",
    # ── AFT tail features — Andersen-Fusari-Todorov (2017), market-level ──────
    "aft_",
    # ── Lagged realized — SPX market-level (same for all strategies) ──────────
    # vrp, rv, rv_skew, rv_up, rv_dn, ret_to_close: all computed from SPX moments,
    # grouped by quote_time and shifted +1 trading date.  Same value for all 64
    # strategies at a given (date, time) → market-level → ts.
    "vrp_", "rv_lag", "rv_up_lag", "rv_dn_lag", "rv_skew_lag",
    "ret_to_close_lag",
    # ── Macro continuous — market-level ───────────────────────────────────────
    "vix_lag", "sofr_lag",
    # ── IV/RV ratio — market-level (SPX implied vs SPX realised intraday) ─────
    "exp_move_to_rv",
    # ── SPX intraday state — market-level (single SPX price series) ───────────
    "spx_ret", "spx_simple_ret", "spx_abs_ret", "spx_lret", "spx_lrv",
    "spx_rv", "spx_realized_vol", "spx_range", "spx_dist", "spx_trend",
    "spx_high", "spx_low", "spx_reversal",
    # ── Note: strategy-specific lagged features are raw (not listed here) ──────
    # pnl_lag*, pnl_mean*, pnl_std* → raw  (per-strategy PnL, different per row)
    # vilkov_v*, vilkov_f*, vilkov_txn* → raw  (per-strategy option flow)
    # strat_vol_*, strat_txn_* → raw  (per-strategy OHLCV)
)


# ─────────────────────────────────────────────────────────────────────────────
# Categorisation
# ─────────────────────────────────────────────────────────────────────────────

def categorize_features(cols: list[str]) -> dict[str, list[str]]:
    """
    Assign feature columns to preprocessing groups.

    Parameters
    ----------
    cols
        All feature column names (excluding target and id columns).

    Returns
    -------
    dict with keys 'dummies', 'cs', 'ts', 'raw'.
    Every column appears in exactly one group.

    Usage with PreprocessorConfig
    ------------------------------
        cat = categorize_features(feature_cols)
        pp_cfg = PreprocessorConfig(
            winsor_cols = cat["ts"] + cat["raw"],
            ts_cols     = cat["ts"],
        )
    """
    dummies: list[str] = []
    cs:      list[str] = []
    ts:      list[str] = []
    raw:     list[str] = []

    for c in cols:
        # Priority order (first match wins):
        # 1. CS_SUFFIXES  — already normalised (_cs, _timeslot_z): light winsor, no z-score
        # 2. RAW_OVERRIDE — strategy-specific lags (pnl_lag*, vilkov_v*, strat_vol_*):
        #                   must come before TS and DUMMY to avoid misclassification
        # 3. TS_PREFIXES  — market-level continuous: winsor + z-score (training window)
        # 4. DUMMY        — binary indicators: pass-through, no scaling
        # 5. raw          — everything else: winsor only
        if any(c.endswith(s) for s in CS_SUFFIXES):
            cs.append(c)
        elif any(c.startswith(p) for p in RAW_OVERRIDE_PREFIXES):
            raw.append(c)
        elif any(c.startswith(p) for p in TS_PREFIXES):
            ts.append(c)
        elif (c in DUMMY_EXACT
              or any(c.startswith(p) for p in DUMMY_PREFIXES)
              or any(c.endswith(s) for s in DUMMY_SUFFIXES)):
            dummies.append(c)
        else:
            raw.append(c)

    return {"dummies": dummies, "cs": cs, "ts": ts, "raw": raw}


# ─────────────────────────────────────────────────────────────────────────────
# Cross-sectional scaling (feature engineering step — not in preprocessor)
# ─────────────────────────────────────────────────────────────────────────────

def add_cs_features(
    df: pd.DataFrame,
    cols: list[str],
    group_cols: list[str],
    suffix: str,
    min_group_size: int = 3,
) -> pd.DataFrame:
    """
    Add cross-sectional z-score features.

    For each column in `cols`, computes:
        X_cs = (X - mean_group) / std_group

    where the group is defined by `group_cols` (e.g. ["quote_date", "quote_time"]).
    Groups with fewer than `min_group_size` rows get NaN (z-score is meaningless).

    This belongs in feature engineering (before the walk-forward loop), NOT in
    the preprocessor, because it requires date/time/strategy context that the
    preprocessor does not have.

    Parameters
    ----------
    df          : panel DataFrame
    cols        : columns to scale (raw features, not already-scaled ones)
    group_cols  : grouping keys, e.g. ["quote_date", "quote_time"]
    suffix      : appended to each column name, e.g. "_cs_all" → "mid_cs_all"
    min_group_size : groups smaller than this get NaN

    Returns
    -------
    df with new `{col}{suffix}` columns added (original cols unchanged).

    Typical calls
    -------------
    Panel mode — across all strategies at the same time:
        df = add_cs_features(df, raw_cols,
                             group_cols=["quote_date", "quote_time"],
                             suffix="_cs_all")

    Per-family mode — within economic family:
        df = add_cs_features(df, raw_cols,
                             group_cols=["quote_date", "quote_time", "strategy_family"],
                             suffix="_cs_family")

    Per-strategy mode — within option_type configs:
        df = add_cs_features(df, raw_cols,
                             group_cols=["quote_date", "quote_time", "option_type"],
                             suffix="_cs_strategy")
    """
    out = df.copy()
    g = out.groupby(group_cols, observed=True, sort=False)
    # "size" counts all rows regardless of NaN; "count" would undercount if cols[0] has NaN
    sizes = g[group_cols[0]].transform("size")

    for c in cols:
        if c not in out.columns:
            continue
        mean = g[c].transform("mean")
        std  = g[c].transform("std").replace(0.0, np.nan)
        z    = (out[c] - mean) / std
        z[sizes < min_group_size] = np.nan
        out[f"{c}{suffix}"] = z

    return out


# ─────────────────────────────────────────────────────────────────────────────
# Signed square-root transform (skewness reduction)
# ─────────────────────────────────────────────────────────────────────────────

def sqtr(x: np.ndarray) -> np.ndarray:
    """Signed square-root: sign(x) * sqrt(|x|). Vectorised, NaN-safe."""
    return np.sign(x) * np.sqrt(np.abs(x))


def sqtr_candidates(cols: list[str]) -> list[str]:
    """
    Return the subset of cols that are candidates for the sqtr transform,
    based on SQTR_PREFIXES and SQTR_EXACT.

    Use this to build the cols argument for add_sqtr_features().
    """
    return [
        c for c in cols
        if c in SQTR_EXACT or any(c.startswith(p) for p in SQTR_PREFIXES)
    ]


def add_sqtr_features(
    df: pd.DataFrame,
    cols: list[str] | None = None,
    suffix: str = SQTR_SUFFIX,
) -> pd.DataFrame:
    """
    Add signed square-root transformed columns to reduce skewness.

    Transform: X_sqtr = sign(X) * sqrt(|X|)

    Compresses heavy tails while preserving sign and direction.
    Good for volume-like, depth, cost, and absolute-value features
    that have strong right-skewness.

    Both the original column and the _sqtr column are kept — select
    which to use in your feature list at model-selection time.

    Parameters
    ----------
    df     : panel DataFrame
    cols   : columns to transform.  If None, auto-detects via sqtr_candidates().
    suffix : suffix for the new columns (default "_sqtr")

    Returns
    -------
    df with new `{col}{suffix}` columns added (originals unchanged).

    Example
    -------
        df = add_sqtr_features(df)          # auto-detect
        df = add_sqtr_features(df, ["d", "abs_delta", "ps_max_profit"])
    """
    if cols is None:
        cols = sqtr_candidates([c for c in df.columns])

    out = df.copy()
    for c in cols:
        if c not in out.columns:
            continue
        out[f"{c}{suffix}"] = sqtr(out[c].to_numpy(dtype=float))
    return out
