#!/usr/bin/env python3
"""
massive_build_conditional_features.py
--------------------------------------
Build a conditional-feature dataset for 0DTE strategy modelling.

Each output row: (quote_date, quote_time, option_type, mnes)

Feature families
----------------
  strategy_base   – strategy snapshot: price, Greeks, cost, geometry, ratios
  liquidity       – spread, depth, tightness (Vilkov h / d / rho / T)
  vilkov_flow     – Greek-weighted volume flow (LAGGED 1 bar to avoid lookahead)
  vilkov_gex      – gamma balance / flow-pressure without OI (OI all NaN)
  payoff_shape    – max profit/loss, breakeven distances, expected-move ratio
  option_flow     – per-strategy lagged OHLCV: lag5, lag30, since-open
  spx_intraday    – SPX returns, RV, semivariance, range, timing dummies
  implied         – 0DTE implied variance, skew, slope, expected move
  lagged_realized – prev-day same-time VRP, RV, skew, return-to-close
  lagged_pnl      – prev-day same-time PnL lags (Vilkov Eq 26)
  macro           – daily VIX, SOFR, event dummies (FOMC/CPI/NFP)
  timeslot_norm   – expanding z-score within (strategy, mnes, quote_time)
  cs_scaled       – cross-sectional z-score within (date, quote_time)
  strat_dummies   – one-hot strategy type indicators

Targets (all named target_*)
------
  target_y_long_net, target_y_short_net,
  target_y_long_gross, target_y_short_gross,
  target_y_best_side_net, target_y_long_minus_short_net,
  target_y_long_profitable, target_y_short_profitable

No-lookahead guarantee
----------------------
  – All OHLCV / Greek-flow columns shifted by 1 intraday bar before use.
  – Same-day realized moments shifted by 1 TRADING DATE (same bar_time).
  – PnL lags shifted by 1 trading date (grouped by quote_time).
  – Macro VIX/SOFR are prior-close values (already lagged in source file).
  – No forward-looking columns from FORWARD_LOOKING_COLS appear in features.

Usage
-----
  # full run, all 72 bars, 11 strategies
  python massive_build_conditional_features.py

  # quick test on 5 dates
  python massive_build_conditional_features.py --sample 5

  # 10:00 only
  python massive_build_conditional_features.py --times 10:00

  # validate existing output
  python massive_build_conditional_features.py --validate-only
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

# ── repo-relative import ───────────────────────────────────────────────────────
_HERE = Path(__file__).resolve().parent
_ROOT = _HERE
while _ROOT != _ROOT.parent:
    if (_ROOT / "data").exists():
        break
    _ROOT = _ROOT.parent

if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

_ML_PROTOCOL = _ROOT / "code" / "ml_protocol"
if str(_ML_PROTOCOL.parent) not in sys.path:
    sys.path.insert(0, str(_ML_PROTOCOL.parent))

from ml_protocol.strategy_groups import STRATEGY_FAMILY_MAP

from feature_utils import (
    EPS, SQRT252, BAR_ORDER, BAR_INDEX, BAR_MINUTES, MINUTES_TO_CLOSE,
    ALL_STRATEGIES, FORWARD_LOOKING_COLS,
    get_legs, parse_mnes_levels, safe_div, safe_log,
    add_prev_trading_date, payoff_shape_vectorised,
)
from feature_catalog import add_sqtr_features, sqtr_candidates

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")

# ══════════════════════════════════════════════════════════════════════════════
# CONFIG
# ══════════════════════════════════════════════════════════════════════════════
DEFAULT_CONFIG: dict = {
    # --- I/O ---
    "ticker": "SPX",
    "out_parquet": str(_ROOT / "data" / "processed" / "conditional_features_5min.parquet"),
    "out_metadata": str(_ROOT / "data" / "processed" / "conditional_feature_metadata.json"),

    # --- Strategy / entry-time filter ---
    "entry_times": None,          # None = all 72 bars; pass list like ["10:00"]
    "strategies": ALL_STRATEGIES,  # None = all 11

    # --- Payoff-shape grid ---
    "payoff_grid_lo": 0.85,
    "payoff_grid_hi": 1.15,
    "payoff_grid_n":  501,

    # --- Timeslot normalization ---
    "timeslot_norm_cols": [
        "mid", "h", "rho", "d", "gross_entry_premium",
        "ivar_0dte", "atm_iv", "slope_dn", "spx_rv_since_1000",
    ],
    "timeslot_norm_min_obs": 30,   # minimum past obs before computing z-score

    # --- Cross-sectional scaling ---
    # Strategy-specific features: z-scored within (date, time) cross-section.
    # Adds a {col}_cs column for each.  These go into cs_cols in the preprocessor
    # (light winsor only, no further z-score — already normalised here).
    #
    # Includes pnl_lag* and vilkov flow so that logistic regression has a proper
    # normalised version.  The raw versions remain in raw_cols (winsor only).
    "cs_scale_cols": [
        # Price / Greeks / liquidity (strategy-level snapshots)
        "mid", "h", "rho", "d", "gross_entry_premium",
        "delta", "abs_delta", "gamma", "abs_gamma", "vega", "abs_vega",
        "theta", "abs_theta", "M", "tv",
        # Vilkov snapshot (liquidity / GEX)
        "vilkov_h", "vilkov_d", "vilkov_rho", "vilkov_T",
        "vilkov_gGamma_n", "vilkov_gGamma_a",
        # Vilkov flow — strategy-specific lagged volume (raw also kept as raw_cols)
        "vilkov_v", "vilkov_fDelta", "vilkov_fGamma", "vilkov_fVega",
        # Strategy OHLCV lags — strategy-specific
        "strat_vol_lag5", "strat_txn_lag5",
        # PnL lags — strategy-specific (each (strategy,mnes) has its own PnL history)
        "pnl_lag1_same_time", "pnl_lag2_same_time", "pnl_lag3_same_time",
        "pnl_mean5_same_time", "pnl_std5_same_time",
        # Short-side PnL lags — for short-side target models
        "pnl_short_lag1_same_time", "pnl_short_lag2_same_time", "pnl_short_lag3_same_time",
        "pnl_short_mean5_same_time", "pnl_short_std5_same_time",
    ],

    # --- Data quality filters (applied before feature computation) ---
    "require_all_liquid": False,
    "max_snap_error": None,   # e.g. 0.01 to require snap_error_max <= 0.01

    # --- Sample mode ---
    "sample_dates": None,     # int: use first N trading dates
}


# ══════════════════════════════════════════════════════════════════════════════
# 1. DATA LOADING
# ══════════════════════════════════════════════════════════════════════════════
def _paths(cfg: dict) -> dict[str, Path]:
    d = _ROOT / "data" / "derived" / cfg["ticker"]
    return {
        "strategy":   d / "massive_strategy_5min.parquet",
        "option":     d / "massive_opt_5min.parquet",
        "vix":        d / "vix_5min.parquet",
        "slopes":     d / "slopes_5min.parquet",
        "realized":   d / "realized_moments_5min.parquet",
        "macro":      _ROOT / "data" / "external" / "macro" / "macro_daily_controls.csv",
    }


def load_inputs(cfg: dict, paths: dict[str, Path]) -> dict[str, pd.DataFrame]:
    t0 = time.time()
    out = {}

    # ── strategy parquet (all rows, full intraday panel needed for RV / shift) ──
    strat_cols = [
        "quote_date", "quote_time", "option_type", "mnes",
        "mid", "tv", "intrinsic", "gross_entry_premium",
        "bas", "h", "fee", "rho", "d",
        "delta", "gamma", "vega", "theta", "iv_mean",
        "S", "M", "n_legs", "n_contracts", "all_liquid", "turnover",
        "snap_error_max", "snap_error_mean",
        "reth_und", "reth_und_net",
        "reth_und_opp", "reth_und_opp_net",
        "legs_json",
        "ohlcv_volume_sum", "ohlcv_transactions_sum",
    ]
    avail = pd.read_parquet(paths["strategy"], columns=None).columns.tolist()
    strat_cols = [c for c in strat_cols if c in avail]
    log.info("Loading strategy parquet …")
    df_strat = pd.read_parquet(paths["strategy"], columns=strat_cols)
    df_strat["quote_date"] = pd.to_datetime(df_strat["quote_date"])
    df_strat["mnes"] = df_strat["mnes"].astype(str)
    if cfg.get("strategies"):
        df_strat = df_strat[df_strat["option_type"].isin(cfg["strategies"])].copy()
    if cfg.get("require_all_liquid"):
        df_strat = df_strat[df_strat["all_liquid"]].copy()
    if cfg.get("max_snap_error") is not None:
        df_strat = df_strat[df_strat["snap_error_max"] <= cfg["max_snap_error"]].copy()
    if cfg.get("sample_dates"):
        sample_n = int(cfg["sample_dates"])
        all_dates = sorted(df_strat["quote_date"].unique())[:sample_n]
        df_strat = df_strat[df_strat["quote_date"].isin(all_dates)].copy()
    log.info(f"  strategy: {len(df_strat):,} rows, {df_strat['quote_date'].nunique()} dates")
    out["strategy"] = df_strat

    # ── option parquet (only needed columns) ──────────────────────────────────
    opt_cols = [
        "quote_date", "bar_time", "option_type", "mnes",
        "strike", "active_underlying_price",
        "mid", "bas", "iv",
        "delta", "gamma", "vega",
        "bid_size", "ask_size",
        "ohlcv_volume", "ohlcv_transactions",
        "trade_volume_delta", "trade_volume_gamma", "trade_volume_vega",
        "trade_volume_delta_usd", "trade_volume_gamma_usd", "trade_volume_vega_usd",
        "liquid",
    ]
    opt_avail = pd.read_parquet(paths["option"], columns=None).columns.tolist()
    opt_cols = [c for c in opt_cols if c in opt_avail]
    log.info("Loading option parquet (selected cols) …")
    df_opt = pd.read_parquet(paths["option"], columns=opt_cols)
    df_opt["quote_date"] = pd.to_datetime(df_opt["quote_date"])
    df_opt["mnes_int"] = pd.to_numeric(df_opt["mnes"], errors="coerce").round().astype("Int64")
    df_opt = df_opt.dropna(subset=["mnes_int"]).copy()
    df_opt["mnes_int"] = df_opt["mnes_int"].astype(int)
    if cfg.get("sample_dates"):
        df_opt = df_opt[df_opt["quote_date"].isin(all_dates)].copy()
    log.info(f"  option:   {len(df_opt):,} rows")
    out["option"] = df_opt

    # ── vix_5min ──────────────────────────────────────────────────────────────
    log.info("Loading vix_5min …")
    df_vix = pd.read_parquet(paths["vix"])
    df_vix["quote_date"] = pd.to_datetime(df_vix["quote_date"])
    if cfg.get("sample_dates"):
        df_vix = df_vix[df_vix["quote_date"].isin(all_dates)].copy()
    out["vix"] = df_vix

    # ── slopes_5min ───────────────────────────────────────────────────────────
    log.info("Loading slopes_5min …")
    df_sl = pd.read_parquet(paths["slopes"])
    df_sl["quote_date"] = pd.to_datetime(df_sl["quote_date"])
    if cfg.get("sample_dates"):
        df_sl = df_sl[df_sl["quote_date"].isin(all_dates)].copy()
    out["slopes"] = df_sl

    # ── realized_moments_5min ─────────────────────────────────────────────────
    log.info("Loading realized_moments_5min …")
    df_rm = pd.read_parquet(paths["realized"])
    df_rm["quote_date"] = pd.to_datetime(df_rm["quote_date"])
    if cfg.get("sample_dates"):
        df_rm = df_rm[df_rm["quote_date"].isin(all_dates)].copy()
    out["realized"] = df_rm

    # ── macro CSV ─────────────────────────────────────────────────────────────
    log.info("Loading macro controls …")
    df_macro = pd.read_csv(paths["macro"])
    df_macro["quote_date"] = pd.to_datetime(df_macro["quote_date"])
    out["macro"] = df_macro

    log.info(f"All inputs loaded in {time.time()-t0:.1f}s")
    return out


# ══════════════════════════════════════════════════════════════════════════════
# 2. STRATEGY BASE FEATURES  (all from strategy parquet, safe at t)
# ══════════════════════════════════════════════════════════════════════════════
def build_strategy_base_features(strat: pd.DataFrame) -> pd.DataFrame:
    """
    Price, Greek, geometry, and ratio features directly from the strategy parquet.
    All quantities are spot-normalised (divided by S) as stored in the dataset.
    """
    df = strat[[
        "quote_date", "quote_time", "option_type", "mnes",
        "S", "M", "mid", "tv", "intrinsic", "gross_entry_premium",
        "bas", "h", "fee", "rho", "d", "all_liquid",
        "delta", "gamma", "vega", "theta", "iv_mean",
        "n_legs", "n_contracts", "turnover",
        "snap_error_max", "snap_error_mean",
    ]].copy()

    # ── price / premium ────────────────────────────────────────────────────────
    df["abs_mid"]             = df["mid"].abs()
    df["premium_abs"]         = df["mid"].abs()
    df["net_premium"]         = df["mid"]
    df["half_spread_bps"]     = df["h"] * 1e4
    df["full_spread_bps"]     = df["bas"] * 1e4
    df["fee_bps"]             = df["fee"] * 1e4

    # ── Greeks ────────────────────────────────────────────────────────────────
    # delta is dimensionless (∂V/∂S ∈ [-1,1]) — no scaling needed.
    # gamma (1/S_USD), vega (USD/1%vol), theta (USD/calendar_day) are raw and NOT
    # normalised by S, so we create spot-normalised versions for ratio computation.
    df["abs_delta"]           = df["delta"].abs()
    df["abs_gamma"]           = df["gamma"].abs()
    df["abs_vega"]            = df["vega"].abs()
    df["abs_theta"]           = df["theta"].abs()
    # Spot-normalised Greeks — consistent scale with mid (which is already /S)
    df["gamma_S"]             = df["gamma"] * df["S"]          # dimensionless
    df["vega_S"]              = df["vega"]  / df["S"]          # per %vol / S
    df["theta_S"]             = df["theta"] / df["S"]          # daily fraction of S
    df["abs_gamma_S"]         = df["gamma_S"].abs()
    df["abs_vega_S"]          = df["vega_S"].abs()
    df["abs_theta_S"]         = df["theta_S"].abs()

    # ── geometry: parse mnes string ───────────────────────────────────────────
    levels_s = df["mnes"].apply(parse_mnes_levels)
    df["mnes_min"]            = levels_s.apply(lambda x: min(x) if x else np.nan)
    df["mnes_max"]            = levels_s.apply(lambda x: max(x) if x else np.nan)
    df["mnes_mean"]           = levels_s.apply(lambda x: np.mean(x) if x else np.nan)
    df["mnes_width"]          = df["mnes_max"] - df["mnes_min"]
    df["mnes_abs_dist_atm"]   = (df["mnes_mean"] - 1.0).abs()
    df["all_liquid_int"]      = df["all_liquid"].astype(int)
    df["illiquid_dummy"]      = 1 - df["all_liquid_int"]
    df["depth_per_contract"]  = safe_div(df["d"], df["n_contracts"])

    # ── ratios ────────────────────────────────────────────────────────────────
    df["cost_to_premium"]       = safe_div(df["h"], df["mid"].abs())
    df["cost_to_gross_premium"] = safe_div(df["h"], df["gross_entry_premium"])
    df["cost_to_tv"]            = safe_div(df["h"], df["tv"].abs())
    df["tv_to_premium"]         = safe_div(df["tv"], df["mid"].abs())
    df["gp_per_contract"]       = safe_div(df["gross_entry_premium"], df["n_contracts"])
    # All ratios use S-normalised Greeks so numerator and denominator share units
    df["abs_gamma_to_abs_theta"]= safe_div(df["abs_gamma_S"], df["abs_theta_S"])
    df["abs_vega_to_abs_theta"] = safe_div(df["abs_vega_S"],  df["abs_theta_S"])
    df["abs_delta_to_premium"]  = safe_div(df["delta"].abs(), df["mid"].abs())
    df["abs_gamma_to_premium"]  = safe_div(df["abs_gamma_S"], df["mid"].abs())
    df["theta_to_premium"]      = safe_div(df["theta_S"],     df["mid"].abs())
    df["cost_per_depth"]        = safe_div(df["h"], df["d"] + 1.0)
    df["log_abs_mid"]           = np.log1p(df["mid"].abs())
    df["log_gamma"]             = np.log1p(df["gamma"].abs())
    df["log_d"]                 = np.log1p(df["d"])

    # ── strategy identifier ───────────────────────────────────────────────────
    df["strategy_id"] = df["option_type"].astype(str) + "__" + df["mnes"].astype(str)

    # ── one-hot encode option_type (11 dummies) ──────────────────────────────
    dummies = pd.get_dummies(df["option_type"].astype(str), prefix="strat", dtype=float)
    df = pd.concat([df, dummies], axis=1)

    # ── strategy family + one-hot (6 families) ────────────────────────────────
    df["strategy_family"] = df["option_type"].map(STRATEGY_FAMILY_MAP)
    family_dummies = pd.get_dummies(
        df["strategy_family"].astype(str), prefix="strat_family", dtype=float
    )
    df = pd.concat([df, family_dummies], axis=1)

    return df.drop(columns=["all_liquid"])


# ══════════════════════════════════════════════════════════════════════════════
# 3. PAYOFF SHAPE FEATURES
# ══════════════════════════════════════════════════════════════════════════════
def build_payoff_shape_features(strat: pd.DataFrame, vix_df: pd.DataFrame,
                                 cfg: dict) -> pd.DataFrame:
    """
    Compute payoff-shape features for each strategy row.

    Normalised payoff (in units of S) on a grid around spot, then compute
    max profit/loss, breakeven distances, expected-move ratio.

    Uses K ≈ mnes_target × S (approximation; snap_error typically < 1%).
    No lookahead: payoff grid uses only entry price and option structure,
    not settlement.
    """
    lo, hi, n = cfg["payoff_grid_lo"], cfg["payoff_grid_hi"], cfg["payoff_grid_n"]
    grid_rel = np.linspace(lo, hi, n)

    # pre-compute normalised payoff per (option_type, mnes) pair
    pairs = strat[["option_type", "mnes"]].drop_duplicates()
    shape_cache = payoff_shape_vectorised(
        [(r.option_type, r.mnes) for r in pairs.itertuples(index=False)],
        grid_rel,
    )

    # merge vix so we can compute expected-move ratio
    vix_key = (
        vix_df[["quote_date", "quote_time", "vix"]]
        .rename(columns={"quote_time": "qt"})  # avoid clash
    )

    records = []
    for row in strat[["quote_date", "quote_time", "option_type", "mnes",
                       "mid", "S"]].itertuples(index=False):
        key = (row.option_type, row.mnes)
        payoff_rel = shape_cache.get(key)
        if payoff_rel is None or np.isnan(row.mid) or np.isnan(row.S):
            records.append({
                "quote_date": row.quote_date, "quote_time": row.quote_time,
                "option_type": row.option_type, "mnes": row.mnes,
            })
            continue

        mid = row.mid  # spot-relative
        pnl = payoff_rel - mid  # spot-relative PnL at each grid point

        max_payoff  = float(payoff_rel.max())
        min_payoff  = float(payoff_rel.min())
        max_profit  = float(pnl.max())
        max_loss    = float(-pnl.min())

        # payoff at current spot (grid index closest to 1.0)
        idx_spot = int(np.argmin(np.abs(grid_rel - 1.0)))
        payoff_at_spot = float(payoff_rel[idx_spot])

        # breakeven: zero crossings of pnl
        signs = np.sign(pnl)
        crosses = np.where(np.diff(signs) != 0)[0]  # indices where sign changes
        be_x_rel = []
        for ci in crosses:
            if abs(pnl[ci + 1] - pnl[ci]) > 1e-15:
                x_be = grid_rel[ci] - pnl[ci] * (
                    grid_rel[ci + 1] - grid_rel[ci]
                ) / (pnl[ci + 1] - pnl[ci])
                be_x_rel.append(x_be)
        # also count cases where pnl exactly = 0
        for ci in np.where(pnl == 0.0)[0]:
            be_x_rel.append(grid_rel[ci])

        be_dists = [abs(x - 1.0) for x in be_x_rel]
        dist_be   = float(min(be_dists)) if be_dists else np.nan
        lower_bes = [abs(x - 1.0) for x in be_x_rel if x < 1.0]
        upper_bes = [abs(x - 1.0) for x in be_x_rel if x > 1.0]
        dist_lower_be = float(min(lower_bes)) if lower_bes else np.nan
        dist_upper_be = float(min(upper_bes)) if upper_bes else np.nan

        profit_to_loss = float(max_profit / (max_loss + EPS))
        premium_received = float(max(-mid, 0.0))
        premium_paid     = float(max(mid, 0.0))
        mnes_w = float(
            np.ptp([ml / 1000.0 for _, ml, _ in get_legs(row.option_type, row.mnes)] or [0.0])
        )
        pr_to_width = float(premium_received / (mnes_w + EPS))

        rec: dict = {
            "quote_date": row.quote_date, "quote_time": row.quote_time,
            "option_type": row.option_type, "mnes": row.mnes,
            "ps_max_payoff":    max_payoff,
            "ps_min_payoff":    min_payoff,
            "ps_max_profit":    max_profit,
            "ps_max_loss":      max_loss,
            "ps_payoff_at_spot":payoff_at_spot,
            "ps_dist_be":       dist_be,
            "ps_dist_lower_be": dist_lower_be,
            "ps_dist_upper_be": dist_upper_be,
            "ps_profit_to_loss":profit_to_loss,
            "ps_premium_received": premium_received,
            "ps_premium_paid":     premium_paid,
            "ps_pr_to_width":      pr_to_width,
        }
        records.append(rec)

    df_shape = pd.DataFrame(records)

    # merge expected move from vix (safe at t)
    vix_small = (
        vix_df[["quote_date", "quote_time", "vix"]]
        .rename(columns={"vix": "_vix"})
    )
    df_shape = df_shape.merge(vix_small, on=["quote_date", "quote_time"], how="left")

    exp_move = np.sqrt(np.maximum(df_shape["_vix"].fillna(0.0), 0.0))
    df_shape["ps_exp_move_pct"] = exp_move
    df_shape["ps_be_to_exp_move"] = safe_div(df_shape["ps_dist_be"], exp_move)
    df_shape.drop(columns=["_vix"], inplace=True, errors="ignore")

    return df_shape


# ══════════════════════════════════════════════════════════════════════════════
# 4. VILKOV-STYLE LEG FEATURES (from option snapshot at t, no flow)
# ══════════════════════════════════════════════════════════════════════════════
def _build_leg_table(strat: pd.DataFrame) -> pd.DataFrame:
    """
    Expand strategy rows into a leg table using get_legs().
    Columns: quote_date, quote_time, option_type (strategy), mnes,
             leg_opt_type, leg_mnes_int, qty
    """
    records = []
    for row in strat[["quote_date", "quote_time", "option_type", "mnes"]].itertuples(index=False):
        legs = get_legs(str(row.option_type), str(row.mnes))
        for leg_type, leg_mnes_int, qty in legs:
            records.append({
                "quote_date":    row.quote_date,
                "quote_time":    row.quote_time,
                "option_type":   row.option_type,
                "mnes":          row.mnes,
                "leg_opt_type":  leg_type,
                "leg_mnes_int":  leg_mnes_int,
                "qty":           float(qty),
            })
    if not records:
        return pd.DataFrame(columns=[
            "quote_date", "quote_time", "option_type", "mnes",
            "leg_opt_type", "leg_mnes_int", "qty",
        ])
    df = pd.DataFrame(records)
    df["quote_date"] = pd.to_datetime(df["quote_date"])
    return df


def _opt_snapshot_lookup(opt: pd.DataFrame) -> pd.DataFrame:
    """
    Build a lookup table: (quote_date, bar_time, option_type, mnes_int) → agg fields.

    Prices/Greeks: mean (multiple strikes may map to same mnes_int after rounding).
    Sizes: sum.

    Scale convention: the strategy parquet stores all price/spread quantities
    spot-normalised (divided by S).  Here we normalise mid/bas by
    active_underlying_price before aggregating so that vilkov_h, vilkov_rho are
    in the same units as h and rho in the strategy dataset.
    """
    num_cols = ["mid", "bas", "mid_scaled", "bas_scaled", "iv", "delta", "gamma", "vega",
                "active_underlying_price", "bid_size", "ask_size"]
    for c in num_cols:
        if c not in opt.columns:
            opt[c] = np.nan
        else:
            opt[c] = pd.to_numeric(opt[c], errors="coerce")

    S = opt["active_underlying_price"].replace(0, np.nan)

    # Prefer pre-scaled columns; fall back to raw/S normalisation
    if "mid_scaled" in opt.columns and opt["mid_scaled"].notna().mean() > 0.5:
        opt["mid_norm"] = opt["mid_scaled"]
        opt["bas_norm"] = opt["bas_scaled"]
    else:
        opt["mid_norm"] = opt["mid"] / S
        opt["bas_norm"] = opt["bas"] / S

    agg_dict = {
        "mid_norm": "mean", "bas_norm": "mean", "iv": "mean",
        "delta": "mean", "gamma": "mean", "vega": "mean",
        "active_underlying_price": "mean",
        "bid_size": "sum", "ask_size": "sum",
    }
    agg_dict = {k: v for k, v in agg_dict.items() if k in opt.columns}

    lookup = (
        opt
        .groupby(["quote_date", "bar_time", "option_type", "mnes_int"], as_index=False)
        .agg(agg_dict)
        .rename(columns={
            "bar_time":    "quote_time",
            "option_type": "leg_opt_type",
            "mnes_int":    "leg_mnes_int",
        })
    )
    return lookup


def build_vilkov_leg_features(strat: pd.DataFrame, opt: pd.DataFrame) -> pd.DataFrame:
    """
    Vilkov-style leg-level aggregated features from the option NBBO snapshot.

    Scaling note: bas_scaled in opt_5min = (ask−bid)/(2S) — the HALF-spread.
    This matches the download convention: bas = (ask−bid)/2 in the raw NBBO data.
    So the paper formula h_i = Σ|q_l|·bas_l/2 (where bas_l = full spread/S)
    equals Σ|q_l|·bas_scaled (no extra /2 needed here).

    Liquidity:
      vilkov_h  = Σ |q_l| × bas_scaled_l          (half-spread cost, spot-norm)
      vilkov_d  = Σ |q_l| × (bid_sz_l + ask_sz_l) (depth)
      vilkov_rho= Σ |q_l| × bas_l / |mid_l|        (rel spread)
      vilkov_T  = vilkov_h / (vilkov_d + 1)         (tightness)

    GEX without OI:
      vilkov_gGamma_n = Σ q_l × Gamma_l           (net gamma, signed)
      vilkov_gGamma_a = Σ |q_l| × Gamma_l         (abs gamma, always ≥ 0)
      vilkov_gamma_balance = gGamma_n / (gGamma_a + eps)

    NOTE: flow features (v, fDelta, fGamma, fVega) use LAGGED volume and are
    built separately in build_vilkov_flow_features().
    """
    leg_table = _build_leg_table(strat)
    if leg_table.empty:
        log.warning("Leg table empty; returning empty Vilkov features.")
        return strat[["quote_date", "quote_time", "option_type", "mnes"]].copy()

    snapshot = _opt_snapshot_lookup(opt)

    legs = leg_table.merge(
        snapshot,
        on=["quote_date", "quote_time", "leg_opt_type", "leg_mnes_int"],
        how="left",
    )
    legs["leg_found"] = legs["mid_norm"].notna().astype(float)
    legs["_n_legs_expected"] = 1.0  # one row per leg — sum gives total expected legs

    q    = legs["qty"]
    qabs = q.abs()
    mid  = legs["mid_norm"].fillna(0.0)   # spot-normalised, consistent with strategy dataset
    bas  = legs["bas_norm"].fillna(0.0).clip(lower=0.0)
    bid_sz = legs["bid_size"].fillna(0.0).clip(lower=0.0)
    ask_sz = legs["ask_size"].fillna(0.0).clip(lower=0.0)
    gamma  = legs["gamma"].fillna(0.0)
    delta  = legs["delta"].fillna(0.0)
    vega   = legs["vega"].fillna(0.0)
    iv     = legs["iv"].fillna(0.0)

    legs["_h"]        = qabs * bas          # bas_scaled already = (ask−bid)/(2S) = half-spread
    legs["_d"]        = qabs * (bid_sz + ask_sz)
    legs["_rho"]      = qabs * safe_div(bas, mid.abs())
    legs["_gGamma_n"] = q * gamma          # signed (q can be -1)
    legs["_gGamma_a"] = qabs * gamma       # abs (gamma always ≥ 0 in BS)
    legs["_iv_w"]     = qabs * iv
    legs["_iv_wt"]    = qabs              # weight denominator for mean IV

    key4 = ["quote_date", "quote_time", "option_type", "mnes"]
    agg_cols = ["_h", "_d", "_rho", "_gGamma_n", "_gGamma_a",
                "_iv_w", "_iv_wt", "leg_found", "_n_legs_expected"]
    out = legs.groupby(key4, as_index=False)[agg_cols].sum()

    out.rename(columns={
        "_h":        "vilkov_h",
        "_d":        "vilkov_d",
        "_rho":      "vilkov_rho",
        "_gGamma_n": "vilkov_gGamma_n",
        "_gGamma_a": "vilkov_gGamma_a",
    }, inplace=True)

    out["vilkov_T"] = safe_div(out["vilkov_h"], out["vilkov_d"] + 1.0)
    out["vilkov_gamma_balance"] = safe_div(out["vilkov_gGamma_n"],
                                           out["vilkov_gGamma_a"].abs() + EPS)
    out["vilkov_iv_leg_mean"] = safe_div(out["_iv_w"], out["_iv_wt"])

    # NaN out rows with incomplete leg coverage (ratio check, not absolute count)
    out["vilkov_leg_coverage"] = safe_div(out["leg_found"], out["_n_legs_expected"])
    feat_cols = [c for c in out.columns
                 if c not in set(key4) | {"leg_found", "_n_legs_expected", "_iv_w", "_iv_wt",
                                          "vilkov_leg_coverage"}]
    out.loc[out["vilkov_leg_coverage"] < 0.999, feat_cols] = np.nan

    return out.drop(columns=["leg_found", "_n_legs_expected", "_iv_w", "_iv_wt"])


# ══════════════════════════════════════════════════════════════════════════════
# 5. VILKOV FLOW FEATURES  (option volume LAGGED 1 bar)
# ══════════════════════════════════════════════════════════════════════════════
def build_opt_lagged_flow(opt: pd.DataFrame) -> pd.DataFrame:
    """
    Build a per-option shifted-volume lookup table.

    OHLCV bar at bar_time t covers [t, t+5min), observable at t+5min.
    → shift(1) within (option_ticker or key, quote_date) to get
      the value available at the NEXT bar (decision time t+5min).
    → at decision time t, use the shifted value (= bar [t-5, t) volume).

    Returns a table keyed by (quote_date, bar_time, option_type, mnes_int)
    with columns: lag5_vol, lag5_txn, lag30_vol, lag30_txn,
                  cum_vol_since_open, cum_txn_since_open,
                  lag5_flow_delta_usd, lag5_flow_gamma_usd, lag5_flow_vega_usd,
                  lag30_flow_delta_usd, lag30_flow_gamma_usd, lag30_flow_vega_usd.
    """
    vol_cols = ["ohlcv_volume", "ohlcv_transactions",
                "trade_volume_delta_usd", "trade_volume_gamma_usd", "trade_volume_vega_usd"]
    vol_cols = [c for c in vol_cols if c in opt.columns]
    if not vol_cols:
        log.warning("No OHLCV flow cols in opt parquet; returning empty flow table.")
        return pd.DataFrame()

    # Aggregate to (quote_date, bar_time, option_type, mnes_int) to reduce row count
    agg_dict = {}
    for c in ["ohlcv_volume", "ohlcv_transactions"]:
        if c in vol_cols:
            agg_dict[c] = "sum"
    for c in ["trade_volume_delta_usd", "trade_volume_gamma_usd", "trade_volume_vega_usd"]:
        if c in vol_cols:
            agg_dict[c] = "sum"

    df = (
        opt[["quote_date", "bar_time", "option_type", "mnes_int"] + vol_cols]
        .copy()
    )
    for c in vol_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0)

    df = (
        df.groupby(["quote_date", "bar_time", "option_type", "mnes_int"], as_index=False)
        .agg(agg_dict)
    )

    # Sort for shift
    df = df.sort_values(["option_type", "mnes_int", "quote_date", "bar_time"]).reset_index(drop=True)
    key3 = ["option_type", "mnes_int", "quote_date"]

    # ── shift by 1 bar (within same date) ─────────────────────────────────────
    raw_vol = "ohlcv_volume"  if "ohlcv_volume"  in df.columns else None
    raw_txn = "ohlcv_transactions" if "ohlcv_transactions" in df.columns else None

    def _shift(col: str, n: int = 1) -> pd.Series:
        return df.groupby(key3)[col].shift(n)

    if raw_vol:
        df["lag5_vol"] = _shift(raw_vol)
        df["lag30_vol"] = (
            df.groupby(key3)["lag5_vol"]
            .transform(lambda x: x.rolling(6, min_periods=1).sum())
        )
        df["cum_vol_since_open"] = (
            df.groupby(key3)["lag5_vol"]
            .transform(lambda x: x.fillna(0.0).cumsum())
        )
    if raw_txn:
        df["lag5_txn"] = _shift(raw_txn)
        df["lag30_txn"] = (
            df.groupby(key3)["lag5_txn"]
            .transform(lambda x: x.rolling(6, min_periods=1).sum())
        )
        df["cum_txn_since_open"] = (
            df.groupby(key3)["lag5_txn"]
            .transform(lambda x: x.fillna(0.0).cumsum())
        )

    for fcol, name in [
        ("trade_volume_delta_usd", "delta"),
        ("trade_volume_gamma_usd", "gamma"),
        ("trade_volume_vega_usd", "vega"),
    ]:
        if fcol in df.columns:
            lag5 = _shift(fcol)
            df[f"lag5_flow_{name}_usd"] = lag5
            df[f"lag30_flow_{name}_usd"] = (
                df.groupby(key3)[f"lag5_flow_{name}_usd"]
                .transform(lambda x: x.rolling(6, min_periods=1).sum())
            )

    # drop original raw cols before returning
    raw_cols = [c for c in [raw_vol, raw_txn,
                             "trade_volume_delta_usd",
                             "trade_volume_gamma_usd",
                             "trade_volume_vega_usd"] if c and c in df.columns]
    df.drop(columns=raw_cols, inplace=True, errors="ignore")
    df.rename(columns={"bar_time": "quote_time", "option_type": "leg_opt_type",
                       "mnes_int": "leg_mnes_int"}, inplace=True)
    return df


def build_vilkov_flow_features(strat: pd.DataFrame, opt_flow: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate LAGGED option-level volume flow to strategy level via leg table.

    vilkov_v          = Σ |q_l| × lag5_vol_l
    vilkov_fDelta_usd = Σ |q_l| × lag5_flow_delta_usd_l
    vilkov_fGamma_usd = Σ |q_l| × lag5_flow_gamma_usd_l
    vilkov_fVega_usd  = Σ |q_l| × lag5_flow_vega_usd_l
    signed versions (using q_l instead of |q_l|) for directionality.
    """
    if opt_flow.empty:
        return strat[["quote_date", "quote_time", "option_type", "mnes"]].copy()

    leg_table = _build_leg_table(strat)
    if leg_table.empty:
        return strat[["quote_date", "quote_time", "option_type", "mnes"]].copy()

    legs = leg_table.merge(
        opt_flow,
        on=["quote_date", "quote_time", "leg_opt_type", "leg_mnes_int"],
        how="left",
    )
    legs["flow_leg_found"] = (
        legs["lag5_vol"].notna().astype(float) if "lag5_vol" in legs.columns else 0.0
    )
    legs["_n_legs_expected"] = 1.0  # one row per leg; sum → total expected legs

    q    = legs["qty"]
    qabs = q.abs()

    flow_result = {
        "quote_date":         legs["quote_date"],
        "quote_time":         legs["quote_time"],
        "option_type":        legs["option_type"],
        "mnes":               legs["mnes"],
        "flow_leg_found":     legs["flow_leg_found"],
        "_n_legs_expected":   legs["_n_legs_expected"],
    }

    for fcol, outname in [
        ("lag5_vol",              "vilkov_v"),
        ("lag5_flow_delta_usd",   "vilkov_fDelta"),
        ("lag5_flow_gamma_usd",   "vilkov_fGamma"),
        ("lag5_flow_vega_usd",    "vilkov_fVega"),
        ("lag30_vol",             "vilkov_v_lag30"),
        ("lag30_flow_delta_usd",  "vilkov_fDelta_lag30"),
        ("lag30_flow_gamma_usd",  "vilkov_fGamma_lag30"),
        ("lag30_flow_vega_usd",   "vilkov_fVega_lag30"),
        ("cum_vol_since_open",    "vilkov_v_cum"),
        ("cum_txn_since_open",    "vilkov_txn_cum"),
        ("lag5_txn",              "vilkov_txn"),
    ]:
        if fcol in legs.columns:
            # main column = absolute weighting (|q|), no _abs suffix to match cs_scale_cols
            flow_result[outname]                = (qabs * legs[fcol].fillna(0.0)).values
            flow_result[f"{outname}_signed"]    = (q    * legs[fcol].fillna(0.0)).values

    tmp = pd.DataFrame(flow_result)
    key4 = ["quote_date", "quote_time", "option_type", "mnes"]
    agg_cols = [c for c in tmp.columns if c not in key4]
    out = tmp.groupby(key4, as_index=False)[agg_cols].sum()

    # coverage: matched legs / expected legs (ratio, not raw count)
    out["flow_leg_coverage_ratio"] = safe_div(out["flow_leg_found"], out["_n_legs_expected"])
    out["flow_all_legs_matched"] = (out["flow_leg_coverage_ratio"] >= 0.999).astype(int)

    feat_cols = [c for c in out.columns if c not in set(key4) | {
        "flow_leg_found", "_n_legs_expected", "flow_leg_coverage_ratio", "flow_all_legs_matched",
    }]
    out.loc[out["flow_all_legs_matched"] == 0, feat_cols] = np.nan
    out.drop(columns=["flow_leg_found", "_n_legs_expected"], inplace=True, errors="ignore")

    return out


# ══════════════════════════════════════════════════════════════════════════════
# 6. SPX INTRADAY FEATURES
# ══════════════════════════════════════════════════════════════════════════════
def build_spx_intraday_features(strat: pd.DataFrame) -> pd.DataFrame:
    """
    SPX spot-based intraday features at each bar t:
      returns since open, RV (completed bars only), range, timing.

    No lookahead:
      - ret_since_open uses S_t and S_{10:00}, both known at t.
      - rv_since_open uses squared returns of COMPLETED bars [10:05, t]:
        At t we know r_{10:05}, r_{10:10}, ..., r_t (= log(S_t/S_{t-5})).
        Note: r_t IS observable at t because S_t is the current snapshot.
      - Running H/L uses S_u for u ≤ t.
    """
    # One SPX spot per (date, time) – median across strategies
    spot = (
        strat.groupby(["quote_date", "quote_time"])["S"]
        .median()
        .reset_index()
        .rename(columns={"S": "spot"})
        .sort_values(["quote_date", "quote_time"])
    )

    s_open = (
        spot[spot["quote_time"] == "10:00"][["quote_date", "spot"]]
        .rename(columns={"spot": "spot_open"})
    )
    spot = spot.merge(s_open, on="quote_date", how="left")

    # Timing
    spot["minutes_since_1000"] = spot["quote_time"].map(BAR_MINUTES).fillna(0).astype(int)
    spot["minutes_to_close"]   = spot["quote_time"].map(MINUTES_TO_CLOSE).fillna(0).astype(int)
    spot["bar_index"]          = spot["quote_time"].map(BAR_INDEX).fillna(0).astype(int)
    spot["time_frac"]          = spot["minutes_since_1000"] / 360.0
    spot["hour"]               = spot["quote_time"].str[:2].astype(int)
    spot["minute"]             = spot["quote_time"].str[3:5].astype(int)
    spot["first_30min_dummy"]  = (spot["quote_time"] <= "10:30").astype(int)
    spot["last_30min_dummy"]   = (spot["quote_time"] >= "15:30").astype(int)
    spot["lunchtime_dummy"]    = (
        (spot["quote_time"] >= "12:00") & (spot["quote_time"] <= "13:30")
    ).astype(int)
    # Time to maturity (approx): remaining minutes / (252 × 390 min/year)
    spot["T_years_approx"] = spot["minutes_to_close"] / (252.0 * 390.0)


    # 5-min log returns  r_t = log(S_t / S_{t-5})
    # r_t IS observable at t (we know current S_t)
    spot["r5min"] = (
        np.log(spot["spot"])
        - np.log(spot.groupby("quote_date")["spot"].shift(1))
    )

    # RV since open: sum of r^2 for bars [10:05..t] (inclusive of current bar t)
    spot["r5min_safe"] = spot["r5min"].fillna(0.0)
    spot["r2"]   = spot["r5min_safe"] ** 2
    spot["rup2"] = np.where(spot["r5min_safe"] > 0, spot["r2"], 0.0)
    spot["rdn2"] = np.where(spot["r5min_safe"] < 0, spot["r2"], 0.0)

    g = spot.groupby("quote_date")
    # At 10:00, r5min is NaN (no prior bar), r2=0 → rv_since_1000 = 0 at 10:00 ✓
    spot["spx_rv_since_1000"]    = g["r2"].cumsum()   * 1e5
    spot["spx_rv_up_since_1000"] = g["rup2"].cumsum() * 1e5
    spot["spx_rv_dn_since_1000"] = g["rdn2"].cumsum() * 1e5
    spot["spx_rv_skew_since_1000"] = (
        spot["spx_rv_up_since_1000"] - spot["spx_rv_dn_since_1000"]
    )
    spot["spx_realized_vol_since_1000"] = np.sqrt(
        np.maximum(spot["spx_rv_since_1000"] / 1e5, 0.0)
    )

    # Returns since open
    spot["spx_ret_since_1000"]      = safe_log(spot["spot"] / spot["spot_open"])
    spot["spx_simple_ret_since_1000"] = spot["spot"] / spot["spot_open"] - 1.0
    spot["spx_abs_ret_since_1000"]  = spot["spx_ret_since_1000"].abs()

    # Short-horizon returns
    for lag_bars, name in [(1, "5m"), (3, "15m"), (6, "30m")]:
        spot[f"spx_ret_last_{name}"] = (
            np.log(spot["spot"])
            - np.log(g["spot"].shift(lag_bars))
        )
    spot["spx_reversal_30m"] = -spot["spx_ret_last_30m"]

    # Trend strength: ret_since_open / realized_vol
    spot["spx_trend_strength"] = safe_div(
        spot["spx_ret_since_1000"],
        spot["spx_realized_vol_since_1000"],
    )

    # Running H/L
    spot["spx_high_since_1000"]  = g["spot"].cummax()
    spot["spx_low_since_1000"]   = g["spot"].cummin()
    spot["spx_range_since_1000"] = (
        spot["spx_high_since_1000"] / spot["spx_low_since_1000"] - 1.0
    )
    spot["spx_dist_from_high"]   = spot["spot"] / spot["spx_high_since_1000"] - 1.0
    spot["spx_dist_from_low"]    = spot["spot"] / spot["spx_low_since_1000"]  - 1.0

    keep = [
        "quote_date", "quote_time",
        "minutes_since_1000", "minutes_to_close", "bar_index", "time_frac",
        "hour", "minute", "T_years_approx",
        "first_30min_dummy", "last_30min_dummy", "lunchtime_dummy",
        "spx_ret_since_1000", "spx_simple_ret_since_1000", "spx_abs_ret_since_1000",
        "spx_rv_since_1000", "spx_rv_up_since_1000", "spx_rv_dn_since_1000",
        "spx_rv_skew_since_1000", "spx_realized_vol_since_1000",
        "spx_ret_last_5m", "spx_ret_last_15m", "spx_ret_last_30m",
        "spx_reversal_30m", "spx_trend_strength",
        "spx_high_since_1000", "spx_low_since_1000",
        "spx_range_since_1000", "spx_dist_from_high", "spx_dist_from_low",
    ]
    return spot[[c for c in keep if c in spot.columns]]


# ══════════════════════════════════════════════════════════════════════════════
# 6b. IV SURFACE FROM OPTION CHAIN  (direct from opt_5min, cross-check vs slopes)
# ══════════════════════════════════════════════════════════════════════════════
def build_iv_surface_from_opt(opt: pd.DataFrame, quote_times: list[str]) -> pd.DataFrame:
    """
    Compute ATM IV, put-skew, and call-skew directly from the option NBBO chain.
    Adapted from feat_iv_surface() in the previous compute_conditional_features script.

    This is complementary to slopes_5min (which fits a linear model); here we
    take medians at specific mnes buckets.  Both are safe at t.
    """
    df = opt[opt["bar_time"].isin(quote_times)].copy()
    df = df[df["iv"].notna() & (df["iv"] > 0) & (df["iv"] < 5.0)]
    if df.empty:
        return pd.DataFrame(columns=["quote_date", "quote_time"])

    def _med(mask: pd.Series, col: str = "iv") -> pd.Series:
        return (
            df[mask]
            .groupby(["quote_date", "bar_time"])[col]
            .median()
        )

    atm    = _med(df["mnes_int"].between(998, 1002)).rename("opt_iv_atm")
    iv990  = _med(df["mnes_int"].between(988, 992)).rename("opt_iv_990")
    iv1010 = _med(df["mnes_int"].between(1008, 1012)).rename("opt_iv_1010")
    iv980  = _med(df["mnes_int"].between(978, 982)).rename("opt_iv_980")
    iv1020 = _med(df["mnes_int"].between(1018, 1022)).rename("opt_iv_1020")
    atm_c  = _med(df["mnes_int"].between(998, 1002) & (df["option_type"] == "C")).rename("opt_iv_atm_call")
    atm_p  = _med(df["mnes_int"].between(998, 1002) & (df["option_type"] == "P")).rename("opt_iv_atm_put")

    surf = (
        pd.concat([atm, iv990, iv1010, iv980, iv1020, atm_c, atm_p], axis=1)
        .reset_index()
        .rename(columns={"bar_time": "quote_time"})
    )
    surf["opt_iv_skew_990_1010"] = surf["opt_iv_990"]  - surf["opt_iv_1010"]
    surf["opt_iv_slope_980_1020"] = surf["opt_iv_980"] - surf["opt_iv_1020"]
    surf["opt_iv_cp_diff"]        = surf["opt_iv_atm_call"] - surf["opt_iv_atm_put"]
    surf["opt_iv_put_skew"]       = surf["opt_iv_990"]  - surf["opt_iv_atm"]
    surf["opt_iv_call_skew"]      = surf["opt_iv_atm"]  - surf["opt_iv_1010"]

    keep = [
        "quote_date", "quote_time",
        "opt_iv_atm", "opt_iv_990", "opt_iv_1010", "opt_iv_980", "opt_iv_1020",
        "opt_iv_atm_call", "opt_iv_atm_put",
        "opt_iv_skew_990_1010", "opt_iv_slope_980_1020",
        "opt_iv_cp_diff", "opt_iv_put_skew", "opt_iv_call_skew",
    ]
    return surf[[c for c in keep if c in surf.columns]]


# ══════════════════════════════════════════════════════════════════════════════
# 6c. AFT TAIL FEATURES — Andersen-Fusari-Todorov (2017)
# ══════════════════════════════════════════════════════════════════════════════

def _aft_tau(bar_time: str) -> float:
    """Minutes remaining to 16:00 ET as calendar-year fraction (tau)."""
    try:
        h, m = int(str(bar_time)[:2]), int(str(bar_time)[3:5])
        remaining = max(0, (16 - h) * 60 - m)
        return remaining / (365 * 24 * 60)
    except Exception:
        return np.nan


def build_aft_features(opt: pd.DataFrame, quote_times: list[str]) -> pd.DataFrame:
    """
    Andersen-Fusari-Todorov (2017)-inspired tail features from the 0DTE option chain.

    Reference
    ---------
    Andersen, T.G., Fusari, N., Todorov, V. (2017). Short-term market risks implied
    by weekly options. Journal of Finance, 72(3), 1335–1386.

    Method
    ------
    For each (quote_date, quote_time) option chain snapshot:
      tau    = remaining minutes to 16:00 ET / (365 × 24 × 60)   [calendar-year fraction]
      ATM_IV = mean IV of near-ATM calls and puts (mnes_int ∈ [997, 1003])
      F_t    ≈ S_t  (forward ≈ spot for 0DTE: interest-rate effect negligible at τ < 1 day)
      m_std  = log(K / F_t) / (ATM_IV × √τ)   [standardised moneyness — units: std-devs]

    At 0DTE, m_std compresses near-ATM strikes into the [-3, +3] range because τ is tiny.
    For example at 10:00 with ATM_IV ≈ 15% ann.: the put at K=0.995F gives m_std ≈ −1.3.
    This is the AFT key insight: "tail" means 1–2 normalised std-devs, not 3–5%.

    OTM puts (K < F, m_std < 0) and OTM calls (K > F, m_std > 0) are interpolated
    linearly in (m_std, IV) space at targets m = −1, −2 (puts) and +1, +2 (calls).
    No extrapolation: NaN if target is outside the observed m_std range.
    Requires ≥ 2 valid OTM options per side.

    Features (market-level: merged on quote_date × quote_time, same for all strategies)
    --------
    aft_tau                   — remaining time in calendar-year fractions
    aft_atm_iv                — ATM implied volatility (annualised)
    aft_put_iv_m1             — OTM put IV at m_std = −1
    aft_put_iv_m2             — OTM put IV at m_std = −2
    aft_call_iv_m1            — OTM call IV at m_std = +1
    aft_call_iv_m2            — OTM call IV at m_std = +2
    aft_left_tail_richness    = put_iv_m1 − ATM_IV   (left skew premium, vol-adjusted)
    aft_right_tail_richness   = call_iv_m1 − ATM_IV  (right skew premium)
    aft_tail_imbalance        = left_richness − right_richness
    aft_put_tail_slope        = log(put_iv_m2) − log(put_iv_m1)  (left tail convexity)
    aft_call_tail_slope       = log(call_iv_m2) − log(call_iv_m1)
    aft_tail_slope_imbalance  = put_tail_slope − call_tail_slope

    Δ30-min variations (shift 6 bars within same date — backward-looking, no lookahead):
    aft_d30_atm_iv            — 30-min change in ATM IV
    aft_d30_left_tail_richness
    aft_d30_right_tail_richness
    aft_d30_tail_imbalance
    aft_isolated_left_tail_shift  = aft_d30_left_tail_richness − aft_d30_atm_iv
    aft_isolated_right_tail_shift = aft_d30_right_tail_richness − aft_d30_atm_iv

    No-lookahead guarantee
    ----------------------
    Uses only options observable at (quote_date, bar_time = quote_time).
    The Δ30 shift(6) is computed within each quote_date group: at bar t we subtract
    the value from bar t−6 (30 min earlier on the same day). At bars 10:00–10:25 the
    Δ30 is NaN (no 30-min history yet). No future data enters.
    """
    col_time = "bar_time" if "bar_time" in opt.columns else "quote_time"
    df = opt[opt[col_time].isin(quote_times)].copy()

    # ── Coerce numerics ────────────────────────────────────────────────────────
    for col in ["mid", "bas", "iv", "active_underlying_price", "mnes_int"]:
        if col not in df.columns:
            df[col] = np.nan
        else:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # mnes_rel = K / S (decimal form, e.g. 0.995)
    if "mnes_rel" in df.columns:
        df["_mnes_rel"] = pd.to_numeric(df["mnes_rel"], errors="coerce")
    else:
        df["_mnes_rel"] = df["mnes_int"] / 1000.0

    # Spot-normalised mid: prefer pre-scaled column
    S = df["active_underlying_price"].replace(0.0, np.nan)
    if "mid_scaled" in df.columns and df["mid_scaled"].notna().mean() > 0.5:
        df["_mid_norm"] = pd.to_numeric(df["mid_scaled"], errors="coerce")
    else:
        df["_mid_norm"] = df["mid"] / S

    # ── Quality filter ─────────────────────────────────────────────────────────
    df = df[
        (df["_mid_norm"] > 0) &
        (df["iv"] > 0) & (df["iv"] < 5.0) &
        (S > 0) &
        (df["_mnes_rel"].notna())
    ].copy()

    if df.empty:
        log.warning("build_aft_features: no valid option quotes after quality filter.")
        return pd.DataFrame(columns=["quote_date", "quote_time"])

    # ── ATM IV (vectorised) ────────────────────────────────────────────────────
    # Near-ATM strikes: mnes_int ∈ [997, 1003] ≈ ±0.3% from spot
    atm_mask = (df["mnes_int"] >= 997) & (df["mnes_int"] <= 1003) & df["iv"].notna()
    atm_iv_df = (
        df[atm_mask]
        .groupby(["quote_date", col_time])["iv"]
        .mean()
        .rename("_atm_iv")
        .reset_index()
    )
    df = df.merge(atm_iv_df, on=["quote_date", col_time], how="left")

    # ── Tau and standardised moneyness (vectorised) ────────────────────────────
    df["_tau"] = df[col_time].astype(str).map(_aft_tau)
    valid_tau_iv = df["_atm_iv"] > 0
    df["_m_std"] = np.where(
        valid_tau_iv & (df["_tau"] > 0),
        np.log(df["_mnes_rel"].clip(lower=1e-6))
        / (df["_atm_iv"] * np.sqrt(df["_tau"].clip(lower=1e-12))),
        np.nan,
    )

    # ── Per (date, time) interpolation ────────────────────────────────────────
    PUT_TARGETS  = np.array([-1.0, -2.0])
    CALL_TARGETS = np.array([ 1.0,  2.0])
    MIN_PTS = 2

    records: list[dict] = []
    for (date, bar_t), grp in df.groupby(["quote_date", col_time], sort=True):
        atm_iv = grp["_atm_iv"].iloc[0]
        tau    = grp["_tau"].iloc[0]
        if not (np.isfinite(atm_iv) and atm_iv > 0 and np.isfinite(tau) and tau > 0):
            continue

        rec: dict = {
            "quote_date": date,
            "quote_time": str(bar_t),
            "aft_tau":    float(tau),
            "aft_atm_iv": float(atm_iv),
        }

        # OTM puts: m_std < 0, sorted ascending (most OTM → near ATM)
        puts = (
            grp[(grp["option_type"] == "P") & (grp["_m_std"] < -1e-6)]
            .dropna(subset=["_m_std", "iv"])
            .sort_values("_m_std")           # ascending: −3 < −2 < −1 < 0
        )
        if len(puts) >= MIN_PTS:
            xp, yp = puts["_m_std"].values, puts["iv"].values
            for i, tgt in enumerate(PUT_TARGETS, 1):
                if xp[0] <= tgt <= xp[-1]:  # no extrapolation
                    iv_val = float(np.interp(tgt, xp, yp))
                    if iv_val > 0:
                        rec[f"aft_put_iv_m{i}"] = iv_val

        # OTM calls: m_std > 0, sorted ascending (near ATM → most OTM)
        calls = (
            grp[(grp["option_type"] == "C") & (grp["_m_std"] > 1e-6)]
            .dropna(subset=["_m_std", "iv"])
            .sort_values("_m_std")           # ascending: 0 < 1 < 2 < 3
        )
        if len(calls) >= MIN_PTS:
            xc, yc = calls["_m_std"].values, calls["iv"].values
            for i, tgt in enumerate(CALL_TARGETS, 1):
                if xc[0] <= tgt <= xc[-1]:
                    iv_val = float(np.interp(tgt, xc, yc))
                    if iv_val > 0:
                        rec[f"aft_call_iv_m{i}"] = iv_val

        # ── Derived features ──────────────────────────────────────────────────
        if "aft_put_iv_m1" in rec:
            rec["aft_left_tail_richness"]  = rec["aft_put_iv_m1"]  - atm_iv
        if "aft_call_iv_m1" in rec:
            rec["aft_right_tail_richness"] = rec["aft_call_iv_m1"] - atm_iv
        if "aft_left_tail_richness" in rec and "aft_right_tail_richness" in rec:
            rec["aft_tail_imbalance"] = (
                rec["aft_left_tail_richness"] - rec["aft_right_tail_richness"]
            )
        if "aft_put_iv_m1" in rec and "aft_put_iv_m2" in rec:
            rec["aft_put_tail_slope"] = (
                np.log(max(rec["aft_put_iv_m2"], 1e-9))
                - np.log(max(rec["aft_put_iv_m1"], 1e-9))
            )
        if "aft_call_iv_m1" in rec and "aft_call_iv_m2" in rec:
            rec["aft_call_tail_slope"] = (
                np.log(max(rec["aft_call_iv_m2"], 1e-9))
                - np.log(max(rec["aft_call_iv_m1"], 1e-9))
            )
        if "aft_put_tail_slope" in rec and "aft_call_tail_slope" in rec:
            rec["aft_tail_slope_imbalance"] = (
                rec["aft_put_tail_slope"] - rec["aft_call_tail_slope"]
            )
        records.append(rec)

    if not records:
        log.warning("build_aft_features: no valid (date, time) groups produced output.")
        return pd.DataFrame(columns=["quote_date", "quote_time"])

    result = (
        pd.DataFrame(records)
        .sort_values(["quote_date", "quote_time"])
        .reset_index(drop=True)
    )

    # ── Δ30-min variations: shift(6 bars) within quote_date ───────────────────
    # At bar t, the Δ30 = value_t − value_{t−6}.  shift(6) within groupby(date)
    # ensures we never cross day boundaries. NaN for first 6 bars of each day.
    d30_bases = [
        "aft_atm_iv", "aft_left_tail_richness",
        "aft_right_tail_richness", "aft_tail_imbalance",
    ]
    for col in d30_bases:
        if col not in result.columns:
            continue
        shifted = result.groupby("quote_date")[col].shift(6)
        result[f"aft_d30_{col[4:]}"] = result[col] - shifted  # strip "aft_" prefix

    if ("aft_d30_left_tail_richness" in result.columns
            and "aft_d30_atm_iv" in result.columns):
        result["aft_isolated_left_tail_shift"] = (
            result["aft_d30_left_tail_richness"] - result["aft_d30_atm_iv"]
        )
    if ("aft_d30_right_tail_richness" in result.columns
            and "aft_d30_atm_iv" in result.columns):
        result["aft_isolated_right_tail_shift"] = (
            result["aft_d30_right_tail_richness"] - result["aft_d30_atm_iv"]
        )

    n_ok = int(result["aft_atm_iv"].notna().sum())
    log.info("build_aft_features: %d valid (date×time) rows  |  %d AFT columns",
             n_ok, sum(c.startswith("aft_") for c in result.columns))
    return result


# ══════════════════════════════════════════════════════════════════════════════
# 7. IMPLIED-VOLATILITY FEATURES  (vix_5min + slopes_5min, safe at t)
# ══════════════════════════════════════════════════════════════════════════════
def build_implied_features(vix_df: pd.DataFrame, slopes_df: pd.DataFrame) -> pd.DataFrame:
    """
    Merge vix_5min and slopes_5min at (quote_date, quote_time).
    All variables are cross-sectionally available at decision time t.
    """
    vix = vix_df.copy()
    sl  = slopes_df.copy()

    # Rename slopes bar_time → quote_time if needed
    if "bar_time" in sl.columns and "quote_time" not in sl.columns:
        sl = sl.rename(columns={"bar_time": "quote_time"})
    if "bar_time" in vix.columns and "quote_time" not in vix.columns:
        vix = vix.rename(columns={"bar_time": "quote_time"})

    imp = vix.merge(sl, on=["quote_date", "quote_time"], how="outer", suffixes=("_vix", "_sl"))

    imp["ivar_0dte"]       = imp["vix"] * 1e5
    imp["ivar_up_0dte"]    = imp["vixup"] * 1e5
    imp["ivar_dn_0dte"]    = imp["vixdn"] * 1e5
    imp["ivar_skew_0dte"]  = (imp["vixup"] - imp["vixdn"]) * 1e5
    imp["ivar_asymmetry"]  = safe_div(imp["vixdn"] - imp["vixup"], imp["vix"])
    imp["ivar_up_share"]   = safe_div(imp["vixup"], imp["vix"])
    imp["ivar_dn_share"]   = safe_div(imp["vixdn"], imp["vix"])

    imp["atm_iv"]     = imp["iv_atm"] if "iv_atm" in imp.columns else np.nan
    imp["slope_up"]   = imp["slope_up"] if "slope_up" in imp.columns else np.nan
    imp["slope_dn"]   = imp["slope_dn"] if "slope_dn" in imp.columns else np.nan
    if "slope_dn" in imp.columns and "slope_up" in imp.columns:
        imp["slope_skew"] = imp["slope_dn"] - imp["slope_up"]
        imp["slope_sum"]  = imp["slope_dn"] + imp["slope_up"]

    imp["expected_move_pct"] = np.sqrt(np.maximum(imp["vix"].fillna(0.0), 0.0))

    keep = [
        "quote_date", "quote_time",
        "ivar_0dte", "ivar_up_0dte", "ivar_dn_0dte", "ivar_skew_0dte",
        "ivar_asymmetry", "ivar_up_share", "ivar_dn_share",
        "atm_iv", "slope_up", "slope_dn", "slope_skew", "slope_sum",
        "expected_move_pct", "minutes_to_mat", "n_strikes",
    ]
    return imp[[c for c in keep if c in imp.columns]]


# ══════════════════════════════════════════════════════════════════════════════
# 8. LAGGED REALIZED FEATURES  (prev trading-day, same bar_time)
# ══════════════════════════════════════════════════════════════════════════════
def build_lagged_realized_features(vix_df: pd.DataFrame,
                                    realized_df: pd.DataFrame) -> pd.DataFrame:
    """
    VRP and realized moments shifted by 1 TRADING DATE within each quote_time.

    VRP(t−1) = vix_5min.vix(prev_date, same_time) × 1e5
               − realized_moments.SPX_lrv(prev_date, same_time)

    All forward-looking realized columns are shifted so they represent
    yesterday's ex-post outcome, not today's future.
    """
    # normalise column names
    vix = vix_df.copy()
    if "bar_time" in vix.columns and "quote_time" not in vix.columns:
        vix = vix.rename(columns={"bar_time": "quote_time"})

    rm = realized_df.copy()
    if "bar_time" in rm.columns and "quote_time" not in rm.columns:
        rm = rm.rename(columns={"bar_time": "quote_time"})

    # Merge vix and realized on (date, time)
    merged = vix[["quote_date", "quote_time", "vix"]].merge(
        rm[["quote_date", "quote_time",
            "SPX_lrv", "SPX_lrv_skew", "SPX_lret",
            "SPX_lrvup", "SPX_lrvdn"]],
        on=["quote_date", "quote_time"], how="outer",
    ).sort_values(["quote_time", "quote_date"])

    merged["_vrp_same_day"] = merged["vix"] * 1e5 - merged["SPX_lrv"]

    # Shift by 1 trading date within each quote_time group
    def _lag(col: str, n: int = 1) -> pd.Series:
        return merged.groupby("quote_time")[col].shift(n)

    lag = pd.DataFrame({
        "quote_date":               merged["quote_date"],
        "quote_time":               merged["quote_time"],
        "vrp_lag1_same_time":       _lag("_vrp_same_day"),
        "rv_lag1_same_time":        _lag("SPX_lrv"),
        "rv_skew_lag1_same_time":   _lag("SPX_lrv_skew"),
        "ret_to_close_lag1_same_time": _lag("SPX_lret"),
        "rv_up_lag1_same_time":     _lag("SPX_lrvup"),
        "rv_dn_lag1_same_time":     _lag("SPX_lrvdn"),
    })
    return lag.reset_index(drop=True)


# ══════════════════════════════════════════════════════════════════════════════
# 9. LAGGED PnL FEATURES  (Vilkov Eq 26)
# ══════════════════════════════════════════════════════════════════════════════
def build_lagged_pnl_features(strat: pd.DataFrame) -> pd.DataFrame:
    """
    pnl_lag1_same_time   = reth_und_net(prev trading date, same quote_time)
    pnl_mean5_same_time  = mean of last 5 prev-date same-time PnL
    pnl_std5_same_time   = std  of last 5 prev-date same-time PnL

    Grouped by (option_type, mnes, quote_time), shifted by 1 date.
    No same-day future PnL is used.
    """
    needed = ["reth_und_net", "reth_und_opp_net"]
    for c in needed:
        if c not in strat.columns:
            strat = strat.copy()
            strat[c] = np.nan

    pnl = strat[["quote_date", "quote_time", "option_type", "mnes"] + needed].copy()
    pnl = pnl.sort_values(["option_type", "mnes", "quote_time", "quote_date"])

    key = ["option_type", "mnes", "quote_time"]

    # ── Long-side PnL lags (reth_und_net) ─────────────────────────────────────
    g_long = pnl.groupby(key)["reth_und_net"]
    pnl["pnl_lag1_same_time"] = g_long.shift(1)
    pnl["pnl_lag2_same_time"] = g_long.shift(2)
    pnl["pnl_lag3_same_time"] = g_long.shift(3)
    pnl["pnl_mean5_same_time"] = (
        pnl.groupby(key)["pnl_lag1_same_time"]
        .transform(lambda x: x.rolling(5, min_periods=2).mean())
    )
    pnl["pnl_std5_same_time"] = (
        pnl.groupby(key)["pnl_lag1_same_time"]
        .transform(lambda x: x.rolling(5, min_periods=2).std())
    )

    # ── Short-side PnL lags (reth_und_opp_net) ────────────────────────────────
    # Include if modelling short-side targets (target_y_short_*).
    g_short = pnl.groupby(key)["reth_und_opp_net"]
    pnl["pnl_short_lag1_same_time"] = g_short.shift(1)
    pnl["pnl_short_lag2_same_time"] = g_short.shift(2)
    pnl["pnl_short_lag3_same_time"] = g_short.shift(3)
    pnl["pnl_short_mean5_same_time"] = (
        pnl.groupby(key)["pnl_short_lag1_same_time"]
        .transform(lambda x: x.rolling(5, min_periods=2).mean())
    )
    pnl["pnl_short_std5_same_time"] = (
        pnl.groupby(key)["pnl_short_lag1_same_time"]
        .transform(lambda x: x.rolling(5, min_periods=2).std())
    )

    return pnl.drop(columns=needed)


# ══════════════════════════════════════════════════════════════════════════════
# 10. MACRO FEATURES  (daily, safe at t)
# ══════════════════════════════════════════════════════════════════════════════
def build_macro_features(macro_df: pd.DataFrame) -> pd.DataFrame:
    """
    Daily macro controls to merge on quote_date.

    Lookahead audit (macro_daily_controls.csv):
      vix        = end-of-day VIX close  → NOT known at 10:00am → EXCLUDED
      sofr       = same-day SOFR         → NOT known at 10:00am → EXCLUDED
      vix_chg_1d = vix(t) - vix(t-1)    → requires today's close → EXCLUDED
      vix_chg_5d = vix(t) - vix(t-5)    → requires today's close → EXCLUDED
      is_high_vix = (vix > threshold)   → uses today's close → EXCLUDED (replaced below)
      vix_lag1   = prior-day VIX close   → safe ✓
      sofr_lag1  = prior-day SOFR        → safe ✓
      event dummies (FOMC/CPI/NFP)       → calendar-announced before open → safe ✓

    is_high_vix from the raw CSV is EXCLUDED because it is based on today's VIX
    close, which is not known at any intraday decision time.  We compute a safe
    replacement — is_high_vix_lag1 = (vix_lag1 > 25) — using only the prior day's
    VIX close.  The threshold 25 is a conventional round number for "elevated vol";
    adjust in feature_sets.py if needed.
    """
    m = macro_df.copy()

    m["cpi_or_nfp_day"] = (
        (m.get("is_cpi_day", 0) | m.get("is_nfp_day", 0)).astype(int)
    )

    # Event one-hots from event_name
    if "event_name" in m.columns:
        m["event_CPI"]  = (m["event_name"] == "CPI").astype(int)
        m["event_NFP"]  = (m["event_name"] == "NFP").astype(int)
        m["event_FOMC"] = (m["event_name"] == "FOMC").astype(int)
    else:
        m["event_CPI"] = m.get("is_cpi_day", 0)
        m["event_NFP"] = m.get("is_nfp_day", 0)
        m["event_FOMC"] = m.get("is_fomc_day", 0)

    # Safe is_high_vix replacement — prior-day close only
    if "vix_lag1" in m.columns:
        m["is_high_vix_lag1"] = (m["vix_lag1"] > 25.0).fillna(0).astype(int)

    # vix/sofr/vix_chg_*/is_high_vix are today's close values → forward-looking, excluded
    keep = [
        "quote_date",
        "vix_lag1", "sofr_lag1",
        "is_high_vix_lag1",   # safe replacement (vix_lag1 > 25); is_high_vix dropped
        "is_cpi_day", "is_nfp_day", "is_fomc_day", "is_macro_day",
        "cpi_or_nfp_day", "event_CPI", "event_NFP", "event_FOMC",
    ]
    return m[[c for c in keep if c in m.columns]]


def add_fomc_time_dummies(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add post_fomc_time and pre_fomc_time dummies using quote_time.
    Must be called AFTER merging macro features.
    """
    if "is_fomc_day" not in df.columns:
        return df
    df["post_fomc_time"] = (
        (df["is_fomc_day"] == 1) & (df["quote_time"] >= "14:00")
    ).astype(int)
    df["pre_fomc_time"] = (
        (df["is_fomc_day"] == 1) & (df["quote_time"] < "14:00")
    ).astype(int)
    return df


# ══════════════════════════════════════════════════════════════════════════════
# 11. OPTION OHLCV AGGREGATED TO STRATEGY (strategy-level lagged OHLCV)
# ══════════════════════════════════════════════════════════════════════════════
def build_strategy_ohlcv_features(strat: pd.DataFrame) -> pd.DataFrame:
    """
    Strategy-level lagged OHLCV from the strategy parquet's pre-aggregated
    ohlcv_volume_sum and ohlcv_transactions_sum.

    These cover bar [t, t+5), so shift(1) within (option_type, mnes, quote_date).
    """
    vcols = [c for c in ["ohlcv_volume_sum", "ohlcv_transactions_sum"]
             if c in strat.columns]
    if not vcols:
        log.info("No strategy-level OHLCV columns; skipping.")
        return strat[["quote_date", "quote_time", "option_type", "mnes"]].copy()

    df = strat[["quote_date", "quote_time", "option_type", "mnes"] + vcols].copy()
    df = df.sort_values(["option_type", "mnes", "quote_date", "quote_time"])
    key = ["option_type", "mnes", "quote_date"]

    for raw, prefix in [("ohlcv_volume_sum", "vol"), ("ohlcv_transactions_sum", "txn")]:
        if raw not in df.columns:
            continue
        g = df.groupby(key)[raw]
        df[f"strat_{prefix}_lag5"]   = g.shift(1)
        df[f"strat_{prefix}_lag30"]  = (
            df.groupby(key)[f"strat_{prefix}_lag5"]
            .transform(lambda x: x.rolling(6, min_periods=1).sum())
        )
        df[f"strat_{prefix}_since_open"] = (
            df.groupby(key)[f"strat_{prefix}_lag5"]
            .transform(lambda x: x.fillna(0.0).cumsum())
        )

    return df.drop(columns=vcols)


# ══════════════════════════════════════════════════════════════════════════════
# 12. TIMESLOT NORMALIZATION  (expanding z-score, no lookahead)
# ══════════════════════════════════════════════════════════════════════════════
def build_timeslot_normalized_features(df: pd.DataFrame,
                                        cols: list[str],
                                        min_obs: int = 30) -> pd.DataFrame:
    """
    For each column in cols, compute an expanding z-score within
    (option_type, mnes, quote_time), using only previous dates.

    z = (x - expanding_mean_shifted) / (expanding_std_shifted + eps)

    Output columns: {col}_timeslot_z
    """
    available = [c for c in cols if c in df.columns]
    if not available:
        return df[["quote_date", "quote_time", "option_type", "mnes"]].copy()

    work = df[["quote_date", "quote_time", "option_type", "mnes"] + available].copy()
    work = work.sort_values(["option_type", "mnes", "quote_time", "quote_date"])
    key = ["option_type", "mnes", "quote_time"]
    result_cols: dict[str, pd.Series] = {}

    for col in available:
        g = work.groupby(key)[col]
        # expanding mean/std from all PREVIOUS rows → shift(1)
        exp_mean = g.transform(lambda x: x.expanding().mean().shift(1))
        exp_std  = g.transform(lambda x: x.expanding().std().shift(1))
        n_obs    = g.transform(lambda x: x.expanding().count().shift(1))
        z = safe_div(work[col] - exp_mean, exp_std)
        # mask until sufficient history
        z = z.where(n_obs >= min_obs, other=np.nan)
        result_cols[f"{col}_timeslot_z"] = z

    out = pd.DataFrame(result_cols, index=work.index)
    out.insert(0, "mnes",        work["mnes"])
    out.insert(0, "option_type", work["option_type"])
    out.insert(0, "quote_time",  work["quote_time"])
    out.insert(0, "quote_date",  work["quote_date"])
    return out


# ══════════════════════════════════════════════════════════════════════════════
# 13. CROSS-SECTIONAL SCALING  (within date × quote_time)
# ══════════════════════════════════════════════════════════════════════════════
def build_cross_sectional_scaled_features(df: pd.DataFrame,
                                           cols: list[str]) -> pd.DataFrame:
    """
    z-score within each (quote_date, quote_time) cross-section.
    Allowed because all strategies at the same (date, time) are simultaneously
    observable; targets are excluded.

    Output columns: {col}_cs
    """
    available = [c for c in cols if c in df.columns]
    if not available:
        return df[["quote_date", "quote_time", "option_type", "mnes"]].copy()

    work = df[["quote_date", "quote_time", "option_type", "mnes"] + available].copy()
    key2 = ["quote_date", "quote_time"]
    result_cols: dict[str, pd.Series] = {}

    # Use "size" (not "count") so group_sizes is independent of NaN in any column
    group_sizes = work.groupby(key2)["quote_date"].transform("size")
    for col in available:
        g = work.groupby(key2)[col]
        cs_mean = g.transform("mean")
        cs_std  = g.transform("std").replace(0.0, np.nan)
        z = safe_div(work[col] - cs_mean, cs_std)
        z[group_sizes < 3] = np.nan   # CS z-score meaningless with < 3 obs
        result_cols[f"{col}_cs"] = z

    out = pd.DataFrame(result_cols, index=work.index)
    out.insert(0, "mnes",        work["mnes"])
    out.insert(0, "option_type", work["option_type"])
    out.insert(0, "quote_time",  work["quote_time"])
    out.insert(0, "quote_date",  work["quote_date"])
    return out


# ══════════════════════════════════════════════════════════════════════════════
# 14. COMBINED IMPLIED + INTRADAY RATIO FEATURES
# ══════════════════════════════════════════════════════════════════════════════
def build_combined_iv_rv_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Features comparing implied volatility to realised so far today.
    Both quantities are observable at t (no lookahead).
    """
    out = df[["quote_date", "quote_time", "option_type", "mnes"]].copy()

    if "ivar_0dte" in df.columns and "spx_rv_since_1000" in df.columns:
        out["ivar_to_rv_so_far"] = safe_div(df["ivar_0dte"],
                                            df["spx_rv_since_1000"].clip(lower=EPS))
    if "atm_iv" in df.columns and "spx_realized_vol_since_1000" in df.columns:
        out["atm_iv_to_rv_so_far"] = safe_div(
            df["atm_iv"],
            df["spx_realized_vol_since_1000"].clip(lower=EPS),
        )
    if "expected_move_pct" in df.columns and "spx_realized_vol_since_1000" in df.columns:
        out["exp_move_to_rv_so_far"] = safe_div(
            df["expected_move_pct"],
            df["spx_realized_vol_since_1000"].clip(lower=EPS),
        )
    return out


# ══════════════════════════════════════════════════════════════════════════════
# 15. TARGETS
# ══════════════════════════════════════════════════════════════════════════════
def build_targets(strat: pd.DataFrame) -> pd.DataFrame:
    """
    All target columns, prefixed with 'target_'.

    Absolute targets (sign depends on market direction):
      target_y_long_net / short_net   — continuous net P&L
      target_y_long_profitable        — 1{long_net > 0}   PRIMARY binary
      target_y_short_profitable       — 1{short_net > 0}
      target_y_best_direction         — +1 if long > short else -1 (direction choice)
      target_y_long_minus_short_net   — long - short (direction magnitude, continuous)
      target_y_best_side_net          — max(long, short); NOT TRADABLE (kept for research)

    Relative-value targets (rank within date × time cross-section):
      target_y_long_cs_rank           — percentile rank of long_net in cross-section [0,1]
      target_y_long_above_median      — 1{long_net > cross-sectional median}
      target_y_long_above_cs_mean     — 1{long_net > cross-sectional mean}
      target_y_short_cs_rank          — percentile rank of short_net in cross-section
      target_y_short_above_median     — 1{short_net > cross-sectional median}

    Relative-value targets remove market-level effects: if all strategies lose today,
    the model still selects the "least bad" one. More coherent with IC evaluation
    (Spearman rank correlation = ranking quality within cross-section).
    """
    needed = ["reth_und_net", "reth_und_opp_net", "reth_und", "reth_und_opp"]
    for c in needed:
        if c not in strat.columns:
            strat = strat.copy()
            strat[c] = np.nan

    df = strat[["quote_date", "quote_time", "option_type", "mnes"] + needed].copy()

    # ── Continuous ────────────────────────────────────────────────────────────
    df["target_y_long_net"]              = df["reth_und_net"]
    df["target_y_short_net"]             = df["reth_und_opp_net"]
    df["target_y_long_gross"]            = df["reth_und"]
    df["target_y_short_gross"]           = df["reth_und_opp"]
    df["target_y_long_minus_short_net"]  = df["reth_und_net"] - df["reth_und_opp_net"]
    df["target_y_best_side_net"]         = df[["reth_und_net", "reth_und_opp_net"]].max(axis=1)

    # ── Binary absolute ───────────────────────────────────────────────────────
    df["target_y_long_profitable"]  = (df["reth_und_net"]     > 0).astype(int)
    df["target_y_short_profitable"] = (df["reth_und_opp_net"] > 0).astype(int)
    # Direction choice: which side beats the other? (+1=long, -1=short)
    df["target_y_best_direction"]   = np.where(
        df["reth_und_net"] > df["reth_und_opp_net"], 1.0, -1.0
    )

    # ── Relative-value (cross-sectional rank within date × time) ──────────────
    cs_key = ["quote_date", "quote_time"]

    # Percentile rank [0, 1] — directly comparable across dates/strategies
    df["target_y_long_cs_rank"]  = (
        df.groupby(cs_key)["reth_und_net"].rank(pct=True)
    )
    df["target_y_short_cs_rank"] = (
        df.groupby(cs_key)["reth_und_opp_net"].rank(pct=True)
    )

    # Binary: above cross-sectional median (= rank > 0.5)
    df["target_y_long_above_median"]  = (df["target_y_long_cs_rank"]  > 0.5).astype(int)
    df["target_y_short_above_median"] = (df["target_y_short_cs_rank"] > 0.5).astype(int)

    # Binary: above cross-sectional mean
    cs_long_mean  = df.groupby(cs_key)["reth_und_net"].transform("mean")
    cs_short_mean = df.groupby(cs_key)["reth_und_opp_net"].transform("mean")
    df["target_y_long_above_cs_mean"]  = (df["reth_und_net"]     > cs_long_mean).astype(int)
    df["target_y_short_above_cs_mean"] = (df["reth_und_opp_net"] > cs_short_mean).astype(int)

    target_cols = [c for c in df.columns if c.startswith("target_")]
    return df[["quote_date", "quote_time", "option_type", "mnes"] + target_cols]


# ══════════════════════════════════════════════════════════════════════════════
# 16. VALIDATION
# ══════════════════════════════════════════════════════════════════════════════
def validate_no_lookahead(df: pd.DataFrame,
                           feature_cols: list[str],
                           target_cols: list[str]) -> list[str]:
    """
    Assert that no banned forward-looking columns appear in feature_cols.
    Returns list of violation names (empty = all clear).
    """
    violations = []
    for c in feature_cols:
        if c in FORWARD_LOOKING_COLS:
            violations.append(c)
        # catch accidental inclusion of target-like columns as features
        for raw in ["reth_und", "reth_und_net", "payoff", "S_settle",
                    "reth_und_opp", "reth_und_opp_net"]:
            if c == raw:
                violations.append(c)
    return list(set(violations))


def run_validation_report(df: pd.DataFrame, feature_cols: list[str],
                           target_cols: list[str], strat_raw: pd.DataFrame) -> dict:
    """
    Run all validation checks and return a report dict.
    Prints results to log.
    """
    W = 70
    sep = "=" * W
    print(sep)
    print("VALIDATION REPORT")
    print(sep)

    report: dict = {}

    # ── 1. Shape ───────────────────────────────────────────────────────────────
    print(f"\n[SHAPE]")
    print(f"  Input strategy rows : {len(strat_raw):,}")
    print(f"  Output rows         : {len(df):,}")
    print(f"  Feature columns     : {len(feature_cols)}")
    print(f"  Target columns      : {len(target_cols)}")
    report["n_rows_output"] = len(df)
    report["n_features"]    = len(feature_cols)
    report["n_targets"]     = len(target_cols)

    # ── 2. Key uniqueness ──────────────────────────────────────────────────────
    key4 = ["quote_date", "quote_time", "option_type", "mnes"]
    dup = df.duplicated(subset=key4).sum()
    status = "PASS ✓" if dup == 0 else f"FAIL ✗ ({dup:,} duplicates)"
    print(f"\n[KEY UNIQUENESS]  {status}")
    report["duplicate_keys"] = int(dup)

    # ── 3. No-lookahead ────────────────────────────────────────────────────────
    violations = validate_no_lookahead(df, feature_cols, target_cols)
    la_status = "PASS ✓" if not violations else f"FAIL ✗  {violations}"
    print(f"\n[LOOKAHEAD SAFETY]  {la_status}")
    report["lookahead_violations"] = violations

    # ── 4. OHLCV shift sanity (vol at 10:00 should be NaN or 0) ───────────────
    for col in ["strat_vol_lag5", "strat_txn_lag5"]:
        if col in df.columns:
            at_open = df[df["quote_time"] == "10:00"][col]
            n_nonzero = (at_open.dropna() > 0).sum()
            status = "PASS ✓" if n_nonzero == 0 else f"WARN  {n_nonzero:,} non-zero at 10:00"
            print(f"\n[OHLCV SHIFT {col}]  {status}")
            report[f"ohlcv_nonzero_at_open_{col}"] = int(n_nonzero)

    # ── 5. Lagged PnL sanity (pnl_lag1 at 10:05 → 10:00 same strategy) ────────
    if "pnl_lag1_same_time" in df.columns:
        # spot check: for one (option_type, mnes, quote_time) series, lag should
        # match the previous-date reth_und_net
        sample_type = df["option_type"].iloc[0] if len(df) > 0 else None
        if sample_type:
            sub = df[(df["option_type"] == sample_type) &
                     (df["quote_time"] == "10:00")].sort_values("quote_date")
            if len(sub) > 1:
                lag_check = sub["pnl_lag1_same_time"].iloc[1]
                src_check = strat_raw[
                    (strat_raw["option_type"] == sub["option_type"].iloc[1]) &
                    (strat_raw["mnes"]        == sub["mnes"].iloc[1]) &
                    (pd.to_datetime(strat_raw["quote_date"]) == sub["quote_date"].iloc[0]) &
                    (strat_raw["quote_time"]  == "10:00")
                ]["reth_und_net"]
                if len(src_check) > 0 and not np.isnan(lag_check):
                    diff = abs(lag_check - src_check.iloc[0])
                    ok = diff < 1e-9
                    print(f"\n[PnL LAG SANITY]  {'PASS ✓' if ok else 'FAIL ✗'}  diff={diff:.2e}")
                    report["pnl_lag_sanity_diff"] = float(diff)
                else:
                    print("\n[PnL LAG SANITY]  SKIP (NaN in sample)")

    # ── 6. Flow lag5 sanity (vilkov_v at 10:00 should be NaN — no bar before open) ─
    for col in ["vilkov_v", "strat_vol_lag5"]:
        if col in df.columns:
            at_open = df[df["quote_time"] == "10:00"][col]
            n_nonnan_nonzero = (at_open.dropna() > 0).sum()
            ok = n_nonnan_nonzero == 0
            print(f"\n[FLOW LAG SANITY {col}@10:00]  {'PASS ✓' if ok else f'WARN  {n_nonnan_nonzero:,} positive values at open bar'}")
            report[f"flow_lag_nonzero_at_open_{col}"] = int(n_nonnan_nonzero)

    # ── 7. Realized lag sanity ────────────────────────────────────────────────
    # vrp_lag1_same_time at date d must equal yesterday's VRP (= vix*1e5 - SPX_lrv)
    # We check that the first value (where lag exists) is finite and date is > min_date.
    if "vrp_lag1_same_time" in df.columns:
        col_check = df[df["quote_time"] == "10:00"][["quote_date", "vrp_lag1_same_time"]].dropna()
        n_finite = col_check["vrp_lag1_same_time"].notna().sum()
        has_first_row_nan = df[df["quote_time"] == "10:00"].sort_values("quote_date")["vrp_lag1_same_time"].iloc[0] if len(df) > 0 else np.nan
        first_is_nan = pd.isna(has_first_row_nan)
        ok = first_is_nan  # first date should have NaN lag (no previous day)
        print(f"\n[REALIZED LAG vrp_lag1 SANITY]  {'PASS ✓' if ok else 'WARN  first row not NaN — possible shift error'}"
              f"  ({n_finite:,} finite values)")
        report["vrp_lag1_first_is_nan"] = bool(first_is_nan)
        report["vrp_lag1_n_finite"] = int(n_finite)

    # ── 8. Inf values ─────────────────────────────────────────────────────────
    print(f"\n[INF VALUES]")
    n_inf = np.isinf(df[feature_cols].select_dtypes("number").values).sum()
    print(f"  Replacing {n_inf:,} inf/-inf with NaN")
    report["n_inf_replaced"] = int(n_inf)

    # ── 7. Missingness (top 30) ────────────────────────────────────────────────
    print(f"\n[MISSINGNESS — top 30 cols]")
    miss = df[feature_cols].isna().mean().sort_values(ascending=False).head(30)
    for col, pct in miss[miss > 0.01].items():
        print(f"  {col:55s}  {pct:.1%}")
    report["top_missing_features"] = {k: float(v) for k, v in miss.items()}

    print(f"\n{sep}")
    return report


# ══════════════════════════════════════════════════════════════════════════════
# 17. METADATA + OUTPUT
# ══════════════════════════════════════════════════════════════════════════════
def save_outputs(df: pd.DataFrame, feature_cols: list[str], target_cols: list[str],
                 report: dict, cfg: dict) -> None:
    out_path = Path(cfg["out_parquet"])
    meta_path = Path(cfg["out_metadata"])
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Replace inf → NaN
    num_cols = df.select_dtypes("number").columns.tolist()
    df[num_cols] = df[num_cols].replace([np.inf, -np.inf], np.nan)

    df.to_parquet(out_path, index=False)
    log.info(f"Saved features → {out_path}  ({len(df):,} rows × {df.shape[1]} cols)")

    metadata = {
        "created_at":    datetime.utcnow().isoformat() + "Z",
        "n_rows":        len(df),
        "n_columns":     df.shape[1],
        "feature_cols":  feature_cols,
        "target_cols":   target_cols,
        "n_features":    len(feature_cols),
        "n_targets":     len(target_cols),
        "date_range": {
            "min": str(df["quote_date"].min()),
            "max": str(df["quote_date"].max()),
        },
        "strategies": df["option_type"].unique().tolist(),
        "quote_times": sorted(df["quote_time"].unique().tolist()),
        "excluded_lookahead_cols": sorted(FORWARD_LOOKING_COLS),
        "excluded_oi_cols": [
            "open_interest", "oi_gamma", "oi_gamma_abs",
            "oi_gamma_usd", "oi_gamma_abs_usd",
            "gex_oi_net_usd", "gex_oi_abs_usd",
            "gex_balance_oi", "flow_gamma_to_oi",
        ],
        "no_lookahead_rules": [
            "OHLCV at bar t covers [t,t+5min): shifted +1 bar before use",
            "trade_volume_* in opt_5min: shifted +1 bar before use",
            "Same-day SPX realized moments (SPX_lrv etc.) shifted +1 TRADING DATE per quote_time",
            "PnL lags grouped by (option_type,mnes,quote_time), shifted +1 trading date",
            "Macro VIX/SOFR are prior-close values (pre-lagged in source file)",
            "Timeslot z-scores use expanding mean/std shifted +1 date (strict past-only)",
            "Cross-sectional scaling uses same-date/time cross-section (allowed)",
        ],
        "validation_report": report,
        "config": {k: str(v) if not isinstance(v, (int, float, bool, type(None))) else v
                   for k, v in cfg.items()},
    }

    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2, default=str)
    log.info(f"Saved metadata → {meta_path}")


# ══════════════════════════════════════════════════════════════════════════════
# 18. MAIN PIPELINE
# ══════════════════════════════════════════════════════════════════════════════
def build_pipeline(cfg: dict) -> pd.DataFrame:
    t_start = time.time()
    paths = _paths(cfg)
    for k, p in paths.items():
        if not p.exists():
            raise FileNotFoundError(f"Missing input: {p}")

    # ── load ──────────────────────────────────────────────────────────────────
    data = load_inputs(cfg, paths)
    strat_full = data["strategy"]
    opt        = data["option"]
    vix_df     = data["vix"]
    slopes_df  = data["slopes"]
    realized_df = data["realized"]
    macro_df   = data["macro"]

    n_input_rows = len(strat_full)

    # ── entry-time filter ──────────────────────────────────────────────────────
    entry_times = cfg.get("entry_times") or BAR_ORDER
    strat_full = strat_full[strat_full["quote_time"].isin(set(BAR_ORDER))].copy()

    # ── Feature blocks ────────────────────────────────────────────────────────
    log.info("─── Build: strategy base features")
    base = build_strategy_base_features(strat_full)

    log.info("─── Build: targets")
    targets = build_targets(strat_full)

    log.info("─── Build: strategy OHLCV features (shifted)")
    strat_ohlcv = build_strategy_ohlcv_features(strat_full)

    log.info("─── Build: Vilkov leg snapshot features (opt join)")
    vil_snap = build_vilkov_leg_features(strat_full, opt)

    log.info("─── Build: option lagged flow table")
    opt_flow = build_opt_lagged_flow(opt)

    log.info("─── Build: Vilkov flow features (lagged)")
    vil_flow = build_vilkov_flow_features(strat_full, opt_flow)

    log.info("─── Build: SPX intraday features")
    spx = build_spx_intraday_features(strat_full)

    log.info("─── Build: implied volatility features (vix_5min + slopes_5min)")
    impl = build_implied_features(vix_df, slopes_df)

    log.info("─── Build: IV surface from option chain (opt_5min direct)")
    iv_surf = build_iv_surface_from_opt(opt, entry_times)

    log.info("─── Build: AFT tail features (Andersen-Fusari-Todorov 2017)")
    aft = build_aft_features(opt, entry_times)

    log.info("─── Build: lagged realized features (VRP, RV, skew)")
    lag_real = build_lagged_realized_features(vix_df, realized_df)

    log.info("─── Build: lagged PnL features")
    lag_pnl = build_lagged_pnl_features(strat_full)

    log.info("─── Build: macro features")
    mac = build_macro_features(macro_df)

    log.info("─── Build: payoff shape features")
    pshape = build_payoff_shape_features(strat_full, vix_df, cfg)

    # ── Assemble base dataframe ───────────────────────────────────────────────
    log.info("─── Merging all feature blocks …")
    key4 = ["quote_date", "quote_time", "option_type", "mnes"]
    key2 = ["quote_date", "quote_time"]
    keyd = ["quote_date"]

    df = base.copy()

    # Remove raw forward-looking columns that made it into base via strat cols
    for bad in FORWARD_LOOKING_COLS:
        if bad in df.columns:
            df.drop(columns=[bad], inplace=True)

    # Merges — all left so we keep the strategy universe intact
    df = df.merge(strat_ohlcv.drop(columns=["ohlcv_volume_sum","ohlcv_transactions_sum"],
                                   errors="ignore"),
                  on=key4, how="left")
    df = df.merge(vil_snap,  on=key4, how="left")
    df = df.merge(vil_flow,  on=key4, how="left")
    df = df.merge(spx,       on=key2, how="left")
    df = df.merge(impl,      on=key2, how="left")
    df = df.merge(iv_surf,   on=key2, how="left")
    df = df.merge(aft,       on=key2, how="left")
    df = df.merge(lag_real,  on=key2, how="left")
    df = df.merge(lag_pnl,   on=key4, how="left")
    df = df.merge(mac,       on=keyd, how="left")
    df = df.merge(pshape,    on=key4, how="left")
    df = df.merge(targets,   on=key4, how="left")

    # FOMC intraday dummies (need quote_time, added after macro merge)
    df = add_fomc_time_dummies(df)

    # Combined IV/RV ratio features
    iv_rv = build_combined_iv_rv_features(df)
    df = df.merge(iv_rv, on=key4, how="left")

    # ── Add quote_dt ──────────────────────────────────────────────────────────
    df.insert(4, "quote_dt",
              pd.to_datetime(df["quote_date"].astype(str) + " " + df["quote_time"]))

    # ── Filter to requested entry times ───────────────────────────────────────
    df = df[df["quote_time"].isin(entry_times)].copy()

    log.info(f"After entry-time filter: {len(df):,} rows")

    # ── Identify feature / target columns ─────────────────────────────────────
    non_feat = {"quote_date", "quote_time", "quote_dt", "option_type",
                "mnes", "strategy_id", "strategy_family", "S"}
    target_cols = [c for c in df.columns if c.startswith("target_")]
    feature_cols = [c for c in df.columns
                    if c not in non_feat and not c.startswith("target_")]

    # Guard: remove any forward-looking columns that snuck in
    feature_cols = [c for c in feature_cols if c not in FORWARD_LOOKING_COLS]
    drop_fl = [c for c in df.columns
               if c in FORWARD_LOOKING_COLS and c not in target_cols]
    if drop_fl:
        log.info(f"Dropping forward-looking columns from features: {drop_fl}")
        df.drop(columns=drop_fl, inplace=True, errors="ignore")
        feature_cols = [c for c in feature_cols if c not in drop_fl]

    # ── Timeslot normalisation ────────────────────────────────────────────────
    ts_cols = [c for c in cfg.get("timeslot_norm_cols", []) if c in df.columns]
    if ts_cols:
        log.info(f"─── Build: timeslot z-scores for {len(ts_cols)} cols")
        ts_norm = build_timeslot_normalized_features(
            df, ts_cols, min_obs=cfg.get("timeslot_norm_min_obs", 30)
        )
        df = df.merge(ts_norm, on=key4, how="left")
        feature_cols += [c for c in ts_norm.columns if c not in key4 and c not in feature_cols]

    # ── Cross-sectional scaling ───────────────────────────────────────────────
    cs_cols = [c for c in cfg.get("cs_scale_cols", []) if c in df.columns]
    if cs_cols:
        log.info(f"─── Build: cross-sectional z-scores for {len(cs_cols)} cols")
        cs_scaled = build_cross_sectional_scaled_features(df, cs_cols)
        df = df.merge(cs_scaled, on=key4, how="left")
        feature_cols += [c for c in cs_scaled.columns if c not in key4 and c not in feature_cols]

    # refresh after merges
    target_cols = [c for c in df.columns if c.startswith("target_")]
    feature_cols = sorted(set(feature_cols) - set(target_cols) - non_feat)

    # ── Sqtr transform (signed square-root, reduces right-skew of volume/depth features) ─
    sqtr_c = sqtr_candidates(feature_cols)
    if sqtr_c:
        log.info(f"─── Build: sqtr transform for {len(sqtr_c)} cols")
        df = add_sqtr_features(df, sqtr_c)
        new_sqtr = [f"{c}_sqtr" for c in sqtr_c if f"{c}_sqtr" not in feature_cols]
        feature_cols += new_sqtr
        log.info(f"  Added {len(new_sqtr)} _sqtr columns")

    # refresh after sqtr
    target_cols = [c for c in df.columns if c.startswith("target_")]
    feature_cols = sorted(set(feature_cols) - set(target_cols) - non_feat)

    # ── Validation ────────────────────────────────────────────────────────────
    report = run_validation_report(df, feature_cols, target_cols, strat_full)

    # ── Save ──────────────────────────────────────────────────────────────────
    save_outputs(df, feature_cols, target_cols, report, cfg)

    log.info(f"Pipeline complete in {time.time()-t_start:.1f}s")
    log.info(f"  Output: {len(df):,} rows × {df.shape[1]} columns")
    log.info(f"  Features: {len(feature_cols)}  Targets: {len(target_cols)}")
    return df


# ══════════════════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════════════════
def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Build conditional feature dataset from Massive 0DTE panel.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--ticker", default="SPX")
    p.add_argument("--times", nargs="+", metavar="HH:MM", default=None,
                   help="Entry times (default: all 72 bars)")
    p.add_argument("--sample", type=int, default=None, metavar="N",
                   help="Use first N trading dates only (quick test)")
    p.add_argument("--strategies", nargs="+", default=None,
                   help="Subset of strategy names")
    p.add_argument("--require-liquid", action="store_true")
    p.add_argument("--out", default=None,
                   help="Override output parquet path")
    p.add_argument("--validate-only", action="store_true",
                   help="Load existing output and re-run validation")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()

    cfg = DEFAULT_CONFIG.copy()
    cfg["ticker"] = args.ticker
    if args.times:
        cfg["entry_times"] = args.times
    if args.sample:
        cfg["sample_dates"] = args.sample
    if args.strategies:
        cfg["strategies"] = args.strategies
    if args.require_liquid:
        cfg["require_all_liquid"] = True
    if args.out:
        cfg["out_parquet"] = args.out

    if args.validate_only:
        out_path = Path(cfg["out_parquet"])
        if not out_path.exists():
            print(f"ERROR: {out_path} not found. Run without --validate-only first.")
            sys.exit(1)
        df = pd.read_parquet(out_path)
        target_cols = [c for c in df.columns if c.startswith("target_")]
        feature_cols = [c for c in df.columns
                        if c not in {"quote_date","quote_time","quote_dt",
                                     "option_type","mnes","strategy_id","S"}
                        and not c.startswith("target_")]
        strat_raw = pd.read_parquet(
            _paths(cfg)["strategy"],
            columns=["quote_date","quote_time","option_type","mnes","reth_und_net"],
        )
        strat_raw["quote_date"] = pd.to_datetime(strat_raw["quote_date"])
        run_validation_report(df, feature_cols, target_cols, strat_raw)
        sys.exit(0)

    build_pipeline(cfg)
