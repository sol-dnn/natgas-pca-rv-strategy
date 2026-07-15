"""
feature_sets.py
---------------
Feature groups for 0DTE conditional model experiments.

Usage
-----
    from features.feature_sets import get_feature_cols, FAMILIES

    # Vilkov replication baseline
    cols = get_feature_cols(["vilkov_baseline", "dummies"])

    # Full specification
    cols = get_feature_cols("all")

    # Custom: comment out families you don't want
    cols = get_feature_cols([
        "vilkov_baseline",
        "liquidity",
        # "flow",           # uncomment to add flow features
        "spx_intraday",
        "implied",
        "lagged_realized",
        "pnl_lags",
        "macro",
        # "payoff_shape",   # uncomment to add payoff shape features
        "dummies",
        "cs_scaled",        # cross-sectional z-scores of the above
        # "timeslot_z",     # uncomment for expanding timeslot z-scores
        # "sqtr",           # uncomment for signed sqrt transforms
    ])

Workflow
--------
Build the full parquet once (massive_build_conditional_features.py), then
experiment with feature subsets here for model selection on Stage 1.
Never rebuild the parquet per experiment — just change which cols you pass
to walk_forward().

Column naming conventions
-------------------------
  raw column            : mid, gamma, h, ivar_0dte, ...
  _cs suffix            : cross-sectional z-score within (date, time)
  _timeslot_z suffix    : expanding z-score within (strategy, mnes, time)
  _sqtr suffix          : sign(x)·sqrt(|x|) transform (right-skew reduction)
  strat_* prefix        : one-hot strategy type dummy
  is_* / post_* prefix  : event / time dummies
"""
from __future__ import annotations

# ─────────────────────────────────────────────────────────────────────────────
# FAMILY DEFINITIONS
# Each list contains raw column names as they appear in the output parquet.
# Include both the raw version AND the _cs / _sqtr version if you want both.
# ─────────────────────────────────────────────────────────────────────────────

# ── Vilkov (2026) replication baseline — 10 features ─────────────────────────
VILKOV_BASELINE: list[str] = [
    # implied variance state (= iv × 1e5, = vix-style)
    "ivar_0dte",
    "ivar_skew_0dte",          # vixup - vixdn (upside - downside IV)
    # vol surface slopes
    "slope_up",
    "slope_dn",
    # lagged realized (prev trading-day, same quote_time)
    "rv_lag1_same_time",
    "rv_skew_lag1_same_time",
    "ret_to_close_lag1_same_time",
    # lagged PnL (prev trading-day, same strategy × mnes × quote_time)
    "pnl_lag1_same_time",
    "pnl_mean5_same_time",
    "pnl_std5_same_time",
]

# ── Liquidity / execution cost ────────────────────────────────────────────────
LIQUIDITY: list[str] = [
    # strategy-level (from strategy parquet, spot-normalised)
    "h",                       # half-spread cost Σ|q_l|·bas_l/2S
    #"rho",                     # relative spread  Σ|q_l|·bas_l/|mid_l|
    #"d",                       # displayed depth  Σ|q_l|·(bid_sz+ask_sz)
    "cost_to_premium",         # h / |mid|
    "cost_to_tv",              # h / tv
    #"depth_per_contract",      # d / n_contracts
    "cost_per_depth",          # h / (d+1)
    "tv_to_premium",           # tv / |mid|
    "all_liquid_int",          # 1 if all legs have bid>0
    #"illiquid_dummy",          # 1 - all_liquid_int
    # Vilkov-leg version (from option NBBO, more granular)
    #"vilkov_h",
    #"vilkov_d",
    #"vilkov_rho",
    #"vilkov_T",                # tightness = vilkov_h / (vilkov_d+1)
]

# ── Strategy price / premium ──────────────────────────────────────────────────
STRATEGY_PRICE: list[str] = [
    #"mid",                     # net strategy premium (spot-relative, signed)
    #"abs_mid",                 # |mid|
    #"tv",                      # time value (spot-relative)
    #"gross_entry_premium",     # total premium paid (unsigned)
    #"gp_per_contract",         # gross_entry_premium / n_contracts
    #"n_contracts",             # number of option contracts in the strategy
    "M",                       # effective moneyness (mean strike / S)
    #"mnes_mean",               # mean of mnes levels
    #"mnes_min",                # minimum mnes level (lowest strike / S)
    #"mnes_max",                # maximum mnes level (highest strike / S)
    #"mnes_width",              # max_mnes - min_mnes (strike width)
    #"mnes_abs_dist_atm",       # |mean_mnes - 1.0| (how far from ATM)
    #"log_abs_mid",             # log1p(|mid|)
    #"log_d",                   # log1p(depth) — right-skewed depth transform
    "cost_to_gross_premium",   # h / gross_entry_premium (execution cost ratio)
]

# ── Greeks ────────────────────────────────────────────────────────────────────
GREEKS: list[str] = [
    # Spot-normalised Greeks (preferred — comparable across strategy types and dates)
    #"delta",
    #"abs_delta",
    "gamma_S",                 # gamma × S  (dimensionless)
    #"abs_gamma_S",
    "vega_S",                  # vega / S   (per %vol / S)
    #"abs_vega_S",
    "theta_S",                 # theta / S  (daily fraction of S)
    #"abs_theta_S",
    # Raw Greeks (not spot-normalised) — use _S versions above for models
    # Kept here so the parquet column is documented; exclude from feature_cols
    # if _S version already included (to avoid redundancy).
    # "gamma",                 # raw gamma (1/S units) — prefer gamma_S
    # "vega",                  # raw vega ($/1%vol) — prefer vega_S
    # "theta",                 # raw theta ($/day) — prefer theta_S
    # Ratios (all use spot-normalised Greeks)
    "abs_gamma_to_abs_theta",
    #"abs_vega_to_abs_theta",
    "abs_delta_to_premium",
    "abs_gamma_to_premium",
    "theta_to_premium",
    #"iv_mean",                 # mean BS IV across legs (from strategy parquet)
    "vilkov_iv_leg_mean",      # |q|-weighted mean IV from option chain
]

# ── Volume flow (LAGGED 1 bar — no lookahead) ─────────────────────────────────
FLOW: list[str] = [
    # lag-5min (previous bar)
    #"vilkov_v",                # Σ|q_l|·lag5_vol_l  (volume)
    #"vilkov_fDelta",           # Σ|q_l|·lag5_flow_delta_usd_l
    #"vilkov_fGamma",
    #"vilkov_fVega",
    #"vilkov_v_signed",         # signed versions (q_l, not |q_l|)
    #"vilkov_fDelta_signed",
    #"vilkov_fGamma_signed",
    #"vilkov_fVega_signed",
    # lag-30min (rolling 6 bars)
    #"vilkov_v_lag30",
    #"vilkov_fDelta_lag30",
    #"vilkov_fGamma_lag30",
    #"vilkov_fVega_lag30",
    # cumulative since 10:00
    #"vilkov_v_cum",
    #"vilkov_txn_cum",
    #"vilkov_txn",              # lag5 transaction count
    # strategy-level OHLCV (lag5, lag30, cum)
    #"strat_vol_lag5",
    "strat_vol_lag30",
    #"strat_vol_since_open",
    #"strat_txn_lag5",
    #"strat_txn_lag30",
    #"strat_txn_since_open",
]

# ── GEX without OI (gamma exposure, structural) ───────────────────────────────
GEX: list[str] = [
    "vilkov_gGamma_n",         # Σ q_l · gamma_l  (net, signed)
    "vilkov_gGamma_a",         # Σ|q_l|· gamma_l  (absolute)
    "vilkov_gamma_balance",    # gGamma_n / (gGamma_a + eps)  ∈ [-1, 1]
    "log_gamma",               # log1p(|gamma|)
]

# NOTE: OI-based GEX (g_i^{OI,n}, g_i^{OI,a}, B_i^Γ, R_i^Γ from Vilkov Table A3)
# are NOT available — open interest data is absent from the Massive dataset.

# ── SPX intraday (spot returns, RV, timing) ───────────────────────────────────
SPX_INTRADAY: list[str] = [
    # timing (raw integers / fractions, go into raw_cols in preprocessor)
    #"minutes_since_1000",
    #"minutes_to_close",
    #"time_frac",               # minutes_since_1000 / 360
    "T_years_approx",          # remaining time in calendar-year fraction
    # returns since open
    "spx_ret_since_1000",
    #"spx_simple_ret_since_1000",
    #"spx_abs_ret_since_1000",
    # short-horizon returns
    #"spx_ret_last_5m",
    #"spx_ret_last_15m",
    #"spx_ret_last_30m",
    #"spx_reversal_30m",        # -ret_last_30m (mean-reversion signal)
    # intraday RV since 10:00
    #"spx_rv_since_1000",
    "spx_rv_up_since_1000",
    "spx_rv_dn_since_1000",
    "spx_rv_skew_since_1000",  # rv_up - rv_dn
    #"spx_realized_vol_since_1000",
    # trend / range
    "spx_trend_strength",      # ret / realized_vol
    #"spx_range_since_1000",    # high/low - 1
    #"spx_dist_from_high",
    #"spx_dist_from_low",
    # High / low levels (raw) — use range/dist above for models; these are raw levels
    # "spx_high_since_1000",   # running high since 10:00 (raw spot level)
    # "spx_low_since_1000",    # running low since 10:00 (raw spot level)
]

# ── Implied variance / surface ────────────────────────────────────────────────
IMPLIED: list[str] = [
    # VIX-style 0DTE (from vix_5min)
    "ivar_0dte",               # total implied variance × 1e5
    #"ivar_up_0dte",            # upside semi-variance × 1e5
    #"ivar_dn_0dte",            # downside semi-variance × 1e5
    "ivar_skew_0dte",          # ivar_up - ivar_dn
    "ivar_asymmetry",          # (ivar_dn - ivar_up) / ivar_0dte
    "ivar_up_share",           # ivar_up / ivar_0dte
    #"ivar_dn_share",           # ivar_dn / ivar_0dte
    # surface slope (from slopes_5min)
    "atm_iv",
    #"slope_up",
    #"slope_dn",
    "slope_skew",              # slope_dn - slope_up
    #"slope_sum",
    #"expected_move_pct",       # sqrt(vix) ≈ expected 1-day move
    # From option chain directly (complementary, less model-dependent)
    #"opt_iv_atm",
    #"opt_iv_990",
    #"opt_iv_1010",
    "opt_iv_980",              # IV at 0.98 moneyness (deeper OTM put wing)
    "opt_iv_1020",             # IV at 1.02 moneyness (deeper OTM call wing)
    #"opt_iv_atm_call",         # ATM call IV
    #"opt_iv_atm_put",          # ATM put IV
    #"opt_iv_skew_990_1010",    # IV(0.99) - IV(1.01)
    #"opt_iv_slope_980_1020",   # IV(0.98) - IV(1.02)  (wider slope measure)
    "opt_iv_put_skew",
    "opt_iv_call_skew",
    "opt_iv_cp_diff",          # ATM call IV - ATM put IV
    # From slopes_5min (may be NaN if slopes parquet has missing bars)
    # "minutes_to_mat",        # time to maturity from slopes fitting (proxy)
    # "n_strikes",             # number of valid strikes used in slope fit
]

# ── Lagged realized (prev day, same quote_time) ───────────────────────────────
LAGGED_REALIZED: list[str] = [
    "vrp_lag1_same_time",      # IV_t-1 - RV_t-1  (yesterday's VRP same time)
    "rv_lag1_same_time",
    "rv_skew_lag1_same_time",
    #"rv_up_lag1_same_time",
    #"rv_dn_lag1_same_time",
    "ret_to_close_lag1_same_time",
]

# ── Lagged PnL (Vilkov Eq.26 — prev day, same strategy × mnes × time) ─────────
PNL_LAGS: list[str] = [
    # Long-side PnL lags (from reth_und_net) — use with target_y_long_*
    "pnl_lag1_same_time",
    #"pnl_lag2_same_time",
    #"pnl_lag3_same_time",
    "pnl_mean5_same_time",
    "pnl_std5_same_time",
    # Short-side PnL lags (from reth_und_opp_net) — use with target_y_short_*
    #"pnl_short_lag1_same_time",
    #"pnl_short_lag2_same_time",
    #"pnl_short_lag3_same_time",
    #"pnl_short_mean5_same_time",
    #"pnl_short_std5_same_time",
]

# ── Payoff shape ──────────────────────────────────────────────────────────────
PAYOFF_SHAPE: list[str] = [
    # Risk/reward ratios (main signals)
    #"ps_max_profit",           # max(payoff - entry_cost) on theoretical spot grid
    #"ps_max_loss",             # max loss on grid
    "ps_profit_to_loss",       # max_profit / max_loss
    "ps_pr_to_width",          # premium_received / strike_width
    # Breakeven distances (spot-relative)
    "ps_dist_be",              # min distance to nearest breakeven
    "ps_dist_lower_be",        # lower breakeven distance
    "ps_dist_upper_be",        # upper breakeven distance
    "ps_be_to_exp_move",       # dist_be normalised by expected move = sqrt(VIX)
    # Payoff levels (less used — raw payoff shape properties)
    # "ps_max_payoff",         # theoretical max payoff (before subtracting entry cost)
    # "ps_min_payoff",         # theoretical min payoff (= -max_loss for most strategies)
    # "ps_payoff_at_spot",     # payoff if spot stays exactly at current price
    # "ps_premium_received",   # credit received for credit strategies (0 for debit)
    # "ps_premium_paid",       # debit paid for debit strategies (0 for credit)
    #"ps_exp_move_pct",         # sqrt(VIX) expected 1-day move — contextual duplicate of expected_move_pct
]

# ── Macro / event ─────────────────────────────────────────────────────────────
MACRO: list[str] = [
    #"vix_lag1",                # prior-day VIX close (safe at any intraday time)
    "sofr_lag1",               # prior-day SOFR
    "is_high_vix_lag1",        # (vix_lag1 > 25) — safe version; is_high_vix EXCLUDED (uses today's close)
    "is_cpi_day",
    "is_nfp_day",
    "is_fomc_day",
    #"is_macro_day",
    #"cpi_or_nfp_day",
    #"event_CPI",
    #"event_NFP",
    #"event_FOMC",
    #"post_fomc_time",          # FOMC day AND quote_time >= 14:00
    #"pre_fomc_time",           # FOMC day AND quote_time < 14:00
]

# ── AFT tail features — Andersen-Fusari-Todorov (2017) ───────────────────────
# Reference: Andersen, Fusari, Todorov (2017), "Short-term market risks implied
# by weekly options", Journal of Finance 72(3), 1335-1386.
#
# All market-level (same for all strategies at a given date × time).
# Preprocessor category: ts (winsorised + z-scored) via "aft_" prefix in TS_PREFIXES.
#
# Standardised moneyness m_std = log(K/F) / (ATM_IV × √τ) adjusts for the
# vol regime: at m=-1 we are exactly 1 vol-adjusted std-dev OTM, regardless of
# the level of ATM_IV or remaining time.  This makes features comparable
# across dates and time-of-day — key advantage over fixed-strike IV features.
AFT: list[str] = [
    # Base (per timestamp)
    #"aft_tau",                    # remaining time in calendar-year fractions
    #"aft_atm_iv",                 # ATM implied volatility (annualised)
    #"aft_put_iv_m1",              # OTM put IV at standardised moneyness m = -1
    #"aft_put_iv_m2",              # OTM put IV at m = -2
    #"aft_call_iv_m1",             # OTM call IV at m = +1
    #"aft_call_iv_m2",             # OTM call IV at m = +2
    "aft_left_tail_richness",     # put_iv_m1 - ATM_IV  (left skew, vol-adjusted)
    #"aft_right_tail_richness",    # call_iv_m1 - ATM_IV (right skew)
    "aft_tail_imbalance",         # left_richness - right_richness
    #"aft_put_tail_slope",         # log(put_iv_m2) - log(put_iv_m1)  (left convexity)
    #"aft_call_tail_slope",        # log(call_iv_m2) - log(call_iv_m1)
    "aft_tail_slope_imbalance",   # put_slope - call_slope
    # Δ30-min variations (30-min backward change, NaN at first 6 bars of day)
    "aft_d30_atm_iv",
    #"aft_d30_left_tail_richness",
    #"aft_d30_right_tail_richness",
    "aft_d30_tail_imbalance",
    #"aft_isolated_left_tail_shift",   # aft_d30_left_tail_richness - aft_d30_atm_iv
    #"aft_isolated_right_tail_shift",  # aft_d30_right_tail_richness - aft_d30_atm_iv
]

# ── IV/RV ratio (combined implied and intraday realized) ──────────────────────
IV_RV_RATIO: list[str] = [
    #"ivar_to_rv_so_far",        # ivar_0dte / spx_rv_since_1000  (NaN at 10:00, rv=0)
    "atm_iv_to_rv_so_far",      # atm_iv / spx_realized_vol_since_1000 (→ ts via "atm_iv")
    #"exp_move_to_rv_so_far",    # expected_move_pct / spx_realized_vol_since_1000
]

# ── Strategy type one-hots (dummies — no scaling) ─────────────────────────────
# 11 strategy-type dummies + 6 family dummies + time-of-day dummies
# Use for:
#   Panel model          : include all dummies so model sees which strategy type it is
#   Per-strategy model   : filter df[df["option_type"]=="strangle"] before walk_forward()
#   Per-time model       : filter df[df["quote_time"]=="13:00"] before walk_forward()
DUMMIES: list[str] = [
    # Strategy family (6 one-hots) — USE THIS by default, more regularised than 11 strat_*
    # Family is deterministic from option_type → family dummies OR strat_* dummies, not both.
    "strat_family_long_vol",           # straddle, strangle
    "strat_family_short_vol_bounded",  # iron_butterfly, iron_condor
    "strat_family_directional_spread", # bull_call_spread, bear_put_spread
    "strat_family_ratio_spread",       # call_ratio_spread, put_ratio_spread
    "strat_family_butterfly",          # call_butterfly, put_butterfly
    "strat_family_skew_directional",   # risk_reversal
    # Strategy type (11 one-hots) — finer grained, use INSTEAD of family for per-strategy effects
    # "strat_strangle",
    # "strat_iron_condor",
    # "strat_risk_reversal",
    # "strat_bull_call_spread",
    # "strat_bear_put_spread",
    # "strat_call_ratio_spread",
    # "strat_put_ratio_spread",
    # "strat_straddle",
    # "strat_iron_butterfly",
    # "strat_call_butterfly",
    # "strat_put_butterfly",
    # Time-of-day dummies
    #"first_30min_dummy",   # quote_time <= 10:30
    "last_30min_dummy",    # quote_time >= 15:30
    #"lunchtime_dummy",     # 12:00 <= quote_time <= 13:30
]

# ── Time encoding — bar_index as ordinal ─────────────────────────────────────
# bar_index (0..71) is sufficient for tree models to identify any bar exactly.
# Add to SPX_INTRADAY subset or use directly as raw_col.
# For per-time models, filter df before walk_forward() — no bar encoding needed.
TIME_ORDINAL: list[str] = [
    "bar_index",           # 0 = 10:00, 71 = 15:55 — trees split on this perfectly
]

# ── Cross-sectional z-scores (_cs) — within (date, time) ─────────────────────
# Computed at BUILD TIME by build_cross_sectional_scaled_features().
# Preprocessor treats these as cs_cols: light winsor only, NO z-score.
#
# Two types of features get _cs versions:
#   1. Strategy snapshot (price, Greeks, liquidity, GEX) → compare strategies at same t
#   2. Strategy-specific lags (flow, OHLCV, PnL) → normalise within cross-section
#      because the raw versions are in raw_cols (no pooled ts z-score, wrong for
#      strategy-specific features with different PnL/volume scales per strategy type)
CS_SCALED: list[str] = [
    # Price / Greeks / liquidity snapshot
    #"mid_cs",
    "h_cs",
    "rho_cs",
    "d_cs",
    "gross_entry_premium_cs",
    "delta_cs",
    #"abs_delta_cs",
    "gamma_cs",
    #"abs_gamma_cs",
    #"vega_cs",
    #"abs_vega_cs",
    "theta_cs",
    #"abs_theta_cs",
    #"M_cs",
    "tv_cs",
    # Vilkov snapshot (liquidity / GEX)
    #"vilkov_h_cs",
    #"vilkov_d_cs",
    #"vilkov_rho_cs",
    #"vilkov_T_cs",
    #"vilkov_gGamma_n_cs",
    #"vilkov_gGamma_a_cs",
    # Vilkov flow — strategy-specific lagged volume (primary normalised version)
    "vilkov_v_cs",
    "vilkov_fDelta_cs",
    "vilkov_fGamma_cs",
    #"vilkov_fVega_cs",
    # Strategy OHLCV lags — strategy-specific
    "strat_vol_lag5_cs",
    "strat_txn_lag5_cs",
    # PnL lags — strategy-specific (primary normalised version for logistic regression)
    "pnl_lag1_same_time_cs",
    #"pnl_lag2_same_time_cs",
    #"pnl_lag3_same_time_cs",
    "pnl_mean5_same_time_cs",
    "pnl_std5_same_time_cs",
]

# ── Timeslot z-scores (_timeslot_z) — expanding within (strategy, mnes, time) ─
# These use all PAST observations (shift(1)) at that specific time slot.
# Require min 30 past obs before computing (NaN otherwise).
TIMESLOT_Z: list[str] = [
    "mid_timeslot_z",
    "h_timeslot_z",
    "rho_timeslot_z",
    "d_timeslot_z",
    "gross_entry_premium_timeslot_z",
    #"ivar_0dte_timeslot_z",
    "atm_iv_timeslot_z",
    "slope_dn_timeslot_z",
    #"spx_rv_since_1000_timeslot_z",
]

# ── Signed sqrt transform (_sqtr) — skewness reduction ───────────────────────
# sign(x)·sqrt(|x|) — compresses right-skewed heavy tails while preserving sign.
# Generated by build_pipeline() → add_sqtr_features().  Inherit same preprocessing
# category as their source: vilkov_v_sqtr → raw (like vilkov_v), d_sqtr → raw, etc.
# Complete list of actually-generated _sqtr columns (from sqtr_candidates()):
SQTR: list[str] = [
    # Vilkov flow — strategy-specific lagged volume (raw category)
    "vilkov_v_sqtr",
    "vilkov_fDelta_sqtr",
    "vilkov_fGamma_sqtr",
    "vilkov_fVega_sqtr",
    "vilkov_v_signed_sqtr",
    "vilkov_fDelta_signed_sqtr",
    "vilkov_fGamma_signed_sqtr",
    "vilkov_fVega_signed_sqtr",
    #"vilkov_v_lag30_sqtr",
    #"vilkov_fDelta_lag30_sqtr",
    #"vilkov_fGamma_lag30_sqtr",
    #"vilkov_fVega_lag30_sqtr",
    "vilkov_v_cum_sqtr",
    #"vilkov_txn_sqtr",
    #"vilkov_txn_cum_sqtr",
    # Strategy OHLCV (raw)
    #"strat_vol_lag5_sqtr",
    #"strat_vol_lag30_sqtr",
    #"strat_vol_since_open_sqtr",
    # Depth / granular liquidity (raw)
    "d_sqtr",
    "depth_per_contract_sqtr",
    "gp_per_contract_sqtr",
    # Cost ratios (raw)
    #"cost_to_premium_sqtr",
    #"cost_to_tv_sqtr",
    #"cost_per_depth_sqtr",
    # Greeks absolute values (raw)
    #"abs_delta_sqtr",
    #"abs_gamma_sqtr",
    #"abs_gamma_S_sqtr",
    #"abs_vega_S_sqtr",
    #"abs_theta_S_sqtr",
    # Payoff shape (raw) — NB: ps_dist_be_sqtr now generated (bug fix: SQTR_PREFIXES)
    "ps_max_profit_sqtr",
    "ps_max_loss_sqtr",
    #"ps_dist_be_sqtr",
    # Log transforms (raw)
    #"log_abs_mid_sqtr",
    #"log_gamma_sqtr",
    #"log_d_sqtr",
]

# ── Features that are always 0 or NaN at the 10:00 bar ───────────────────────
# If running a single-time model at 10:00, these features are constant (all 0 or
# all NaN) and carry no information — exclude them from 10:00-only experiments.
# For multi-time models (10:00 + 13:00 + 15:00), keep them — they are informative
# at later bars.
ZERO_AT_OPEN: list[str] = [
    # SPX intraday accumulated — 0 at 10:00 by definition (open IS 10:00)
    "spx_ret_since_1000",
    #"spx_simple_ret_since_1000",
    #"spx_abs_ret_since_1000",
    #"spx_rv_since_1000",
    "spx_rv_up_since_1000",
    "spx_rv_dn_since_1000",
    "spx_rv_skew_since_1000",
    #"spx_realized_vol_since_1000",
    #"spx_dist_from_high",
    #"spx_dist_from_low",
    #"spx_range_since_1000",
    "spx_trend_strength",
    # Short-horizon returns — NaN (no previous bar within same date)
    #"spx_ret_last_5m",
    #"spx_ret_last_15m",
    #"spx_ret_last_30m",
    #"spx_reversal_30m",
    # IV/RV ratios — NaN (rv=0 at 10:00)
    "ivar_to_rv_so_far",
    #"exp_move_to_rv_so_far",
    # Flow lag-5 — NaN (no bar before 10:00 within same date)
    #"vilkov_v",
    #"strat_vol_lag5",
    #"strat_txn_lag5",
    # Timeslot z-scores — 0 at 10:00 (rv_since_1000=0 → z=undefined → 0 after imputation)
    #"spx_rv_since_1000_timeslot_z",
]


# ─────────────────────────────────────────────────────────────────────────────
# COLUMNS AVAILABLE IN PARQUET — NOT INCLUDED AS FEATURES BY DEFAULT
# ─────────────────────────────────────────────────────────────────────────────
# These columns exist in the output parquet but are excluded from FAMILIES
# for one of these reasons (noted per column):
#   [dup]  Duplicate of another column already in a family
#   [meta] Data-quality metadata, not a signal
#   [id]   Identifier / administrative column
#   [fwd]  Forward-looking → never use as feature
#   [raw]  Available but redundant with a better version already included

AVAILABLE_NOT_IN_FAMILIES: dict[str, str] = {
    # ── Administration / identifiers ─────────────────────────────────────────
    "strategy_family":      "[id] string label; use strat_family_* dummies instead",
    "strategy_id":          "[id] string concat of option_type + mnes",
    "S":                    "[id] spot price; used for normalisation, not as signal",
    "bar_index":            "[raw] integer 0..71 ordinal; use time_frac instead",
    "hour":                 "[raw] integer hour; use time_frac or timing dummies instead",
    "minute":               "[raw] integer minute; use time_frac instead",
    # ── Data quality ─────────────────────────────────────────────────────────
    "snap_error_max":       "[meta] max snapshot error across legs; filter criterion",
    "snap_error_mean":      "[meta] mean snapshot error",
    # ── Duplicates of existing features ──────────────────────────────────────
    "premium_abs":          "[dup] = abs_mid (already in STRATEGY_PRICE)",
    "net_premium":          "[dup] = mid (already in STRATEGY_PRICE)",
    "half_spread_bps":      "[dup] = h × 1e4 (already have h in LIQUIDITY)",
    "full_spread_bps":      "[dup] = bas × 1e4 (already have bas implicitly via vilkov)",
    "fee_bps":              "[dup] = fee × 1e4 (fee not in default features; add if needed)",
    "bas":                  "[dup] full bid-ask spread; h = bas/2 already in LIQUIDITY",
    "intrinsic":            "[dup] max(S-K, 0)/S; tv = mid - intrinsic, both available",
    "turnover":             "[dup] = gross_entry_premium × n_contracts; redundant",
    "fee":                  "[raw] option fee in spot-relative units; fee_bps is cleaner",
    "n_legs":               "[raw] number of legs (2 or 4); covered by strat dummies",
    "abs_vega":             "[raw] raw vega (not spot-normalised); abs_vega_S preferred",
    "abs_theta":            "[raw] raw theta; abs_theta_S preferred",
    "abs_gamma":            "[raw] raw gamma; abs_gamma_S preferred",
    # ── Strategy type one-hots (commented out in DUMMIES, family dummies preferred) ─
    "strat_strangle":              "[dup] covered by strat_family_long_vol; uncomment strat_* in DUMMIES to use fine-grained",
    "strat_straddle":              "[dup] covered by strat_family_long_vol",
    "strat_iron_condor":           "[dup] covered by strat_family_short_vol_bounded",
    "strat_iron_butterfly":        "[dup] covered by strat_family_short_vol_bounded",
    "strat_bull_call_spread":      "[dup] covered by strat_family_directional_spread",
    "strat_bear_put_spread":       "[dup] covered by strat_family_directional_spread",
    "strat_call_ratio_spread":     "[dup] covered by strat_family_ratio_spread",
    "strat_put_ratio_spread":      "[dup] covered by strat_family_ratio_spread",
    "strat_call_butterfly":        "[dup] covered by strat_family_butterfly",
    "strat_put_butterfly":         "[dup] covered by strat_family_butterfly",
    "strat_risk_reversal":         "[dup] covered by strat_family_skew_directional",
    # ── Documented in families but commented out ──────────────────────────────
    "gamma":               "[dup/raw] raw gamma (1/S units); gamma_S preferred",
    "vega":                "[dup/raw] raw vega ($/1%vol); vega_S preferred",
    "theta":               "[dup/raw] raw theta ($/day); theta_S preferred",
    "spx_high_since_1000": "[raw] running high SPX since 10:00; spx_dist_from_high captures relative version",
    "spx_low_since_1000":  "[raw] running low SPX since 10:00; spx_dist_from_low captures relative version",
    "ps_max_payoff":       "[raw] theoretical max payoff (before subtracting entry cost)",
    "ps_min_payoff":       "[raw] theoretical min payoff on grid",
    "ps_payoff_at_spot":   "[raw] payoff if spot stays at current level",
    "ps_premium_received": "[raw] credit received for credit strategies",
    "ps_premium_paid":     "[raw] debit paid for debit strategies",
    "minutes_to_mat":      "[raw] from slopes_5min (may be NaN if slopes unavailable)",
    "n_strikes":           "[raw] strike count used in slope fit; from slopes_5min",
    # ── Timeslot z-scores not in TIMESLOT_Z (available, add to config if useful) ─
    "spx_rv_since_1000_timeslot_z": "[available] in TIMESLOT_Z family",
    # ── Flow coverage flags ──────────────────────────────────────────────────
    "flow_all_legs_matched":  "[meta] binary; all flow legs matched (quality filter)",
    "flow_leg_coverage_ratio":"[meta] fraction of legs with flow data",
    "vilkov_leg_coverage":    "[meta] fraction of legs with snapshot data",
}

# ─────────────────────────────────────────────────────────────────────────────
# TARGET DEFINITIONS
# ─────────────────────────────────────────────────────────────────────────────
#
# All targets are prefixed "target_" in the parquet.
# "long" = buying the strategy at entry price (reth_und_net = forward P&L net of costs)
# "short" = selling the strategy (reth_und_opp_net = -reth_und_net - 2×spread)
#
# ┌────────────────────────────────┬────────────────────────────────────────────┐
# │ Target                         │ Description                                │
# ├────────────────────────────────┼────────────────────────────────────────────┤
# │ target_y_long_net              │ Continuous net P&L — go LONG               │
# │ target_y_short_net             │ Continuous net P&L — go SHORT               │
# │ target_y_long_gross            │ Gross P&L long (no transaction costs)       │
# │ target_y_short_gross           │ Gross P&L short (no transaction costs)      │
# │ target_y_long_profitable       │ 1{long net > 0}  ← PRIMARY BINARY TARGET   │
# │ target_y_short_profitable      │ 1{short net > 0}                            │
# │ target_y_best_side_net         │ max(long_net, short_net) — NOT TRADABLE ✗  │
# │ target_y_long_minus_short_net  │ Spread long - short (direction strength)    │
# ├────────────────────────────────┼────────────────────────────────────────────┤
# │ target_y_best_direction        │ NEW +1 if long>short, -1 if short>long      │
# │ (target_y_long_profitable is the alias for "trade or not long")            │
# │ (target_y_short_profitable is the alias for "trade or not short")           │
# └────────────────────────────────┴────────────────────────────────────────────┘
#
# PRIMARY RECOMMENDED TARGETS:
#   Classification : target_y_long_profitable    → binary, most common
#                    target_y_short_profitable   → binary, short side
#                    target_y_best_direction     → binary +1/-1, predict long vs short
#   Regression     : target_y_long_net           → continuous, IC on actual P&L
#                    target_y_long_minus_short_net → direction magnitude
#
# NOT RECOMMENDED as primary target:
#   target_y_best_side_net — max(long, short) is not tradable because you'd need
#   to know ex-ante which side to take. Use target_y_best_direction instead.
#
# HOW target_y_best_direction should be built (add to build_targets() if needed):
#   df["target_y_best_direction"] = np.where(
#       df["reth_und_net"] > df["reth_und_opp_net"], 1.0, -1.0
#   )
# This tells the model: "should I go long or short this strategy today?"
# Tradable: yes — you commit to a direction before observing the outcome.

PRIMARY_TARGETS: list[str] = [
    # ── Absolute (direction vs zero) ─────────────────────────────────────────
    "target_y_long_profitable",    # 1{long_net>0}  most common binary target
    "target_y_short_profitable",   # 1{short_net>0}
    "target_y_best_direction",     # +1 if long>short else -1 (which side to trade)
    "target_y_long_net",           # continuous — IC on actual P&L; use for regression
    # ── Relative-value (rank within cross-section) ────────────────────────────
    "target_y_long_above_median",  # 1{long_net > median(date,time)} — coherent with IC
    "target_y_short_above_median",
    "target_y_long_cs_rank",       # percentile rank [0,1] — use as regression target
]

SECONDARY_TARGETS: list[str] = [
    "target_y_long_above_cs_mean",
    "target_y_short_above_cs_mean",
    "target_y_short_cs_rank",
    "target_y_long_minus_short_net",
    "target_y_short_net",
    "target_y_long_gross",
    "target_y_short_gross",
]



# ─────────────────────────────────────────────────────────────────────────────
# NAMED SETS — convenience presets for experiments
# ─────────────────────────────────────────────────────────────────────────────

FAMILIES: dict[str, list[str]] = {
    "vilkov_baseline":  VILKOV_BASELINE,
    "liquidity":        LIQUIDITY,
    "strategy_price":   STRATEGY_PRICE,
    "greeks":           GREEKS,
    "flow":             FLOW,
    "gex":              GEX,
    "spx_intraday":     SPX_INTRADAY,
    "implied":          IMPLIED,
    "aft":              AFT,          # Andersen-Fusari-Todorov tail surface
    "lagged_realized":  LAGGED_REALIZED,
    "pnl_lags":         PNL_LAGS,
    "payoff_shape":     PAYOFF_SHAPE,
    "macro":            MACRO,
    "iv_rv_ratio":      IV_RV_RATIO,
    "dummies":          DUMMIES,
    "cs_scaled":        CS_SCALED,
    "timeslot_z":       TIMESLOT_Z,
    "sqtr":             SQTR,
}

ALL_FAMILIES: list[str] = list(FAMILIES.keys())


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def get_feature_cols(
    families: list[str] | str = "all",
    *,
    extra: list[str] | None = None,
    exclude: list[str] | None = None,
    available: list[str] | None = None,
) -> list[str]:
    """
    Build a deduplicated feature column list from named families.

    Parameters
    ----------
    families
        List of family names (keys of FAMILIES dict), or "all" for everything.
        Order matters: determines the order in the output list (first wins on dedup).
    extra
        Additional columns to append (after dedup + exclude).
    exclude
        Columns to remove from the final list.
    available
        If provided, silently drops columns not present in this list.
        Pass df.columns.tolist() to guard against missing columns in the parquet.

    Returns
    -------
    Deduplicated list of feature column names.

    Examples
    --------
    # Vilkov replication
    cols = get_feature_cols(["vilkov_baseline", "dummies"])

    # Full model
    cols = get_feature_cols("all")

    # No-flow, no-sqtr, filter to available cols in parquet
    cols = get_feature_cols(
        ["vilkov_baseline", "liquidity", "greeks", "spx_intraday",
         "implied", "lagged_realized", "pnl_lags", "macro", "dummies", "cs_scaled"],
        available=df.columns.tolist(),
    )
    """
    if families == "all":
        families = ALL_FAMILIES

    cols: list[str] = []
    seen: set[str] = set()
    for fam in families:
        if fam not in FAMILIES:
            raise ValueError(
                f"Unknown family {fam!r}. Available: {sorted(FAMILIES.keys())}"
            )
        for c in FAMILIES[fam]:
            if c not in seen:
                cols.append(c)
                seen.add(c)

    if extra:
        for c in extra:
            if c not in seen:
                cols.append(c)
                seen.add(c)

    if exclude:
        excl = set(exclude)
        cols = [c for c in cols if c not in excl]

    if available is not None:
        avail_set = set(available)
        missing = [c for c in cols if c not in avail_set]
        if missing:
            import warnings
            warnings.warn(
                f"get_feature_cols: {len(missing)} columns not in parquet (dropped): "
                f"{missing[:20]}" + (" ..." if len(missing) > 20 else ""),
                UserWarning,
                stacklevel=2,
            )
        cols = [c for c in cols if c in avail_set]

    return cols


def describe_families(families: list[str] | str = "all") -> None:
    """Print a summary of family sizes."""
    if families == "all":
        families = ALL_FAMILIES
    total = 0
    print(f"{'Family':<20}  {'N cols':>6}")
    print("-" * 30)
    for fam in families:
        n = len(FAMILIES[fam])
        total += n
        print(f"  {fam:<18}  {n:>6}")
    print("-" * 30)
    print(f"  {'TOTAL (with dups)':<18}  {total:>6}")
    # deduplicated
    dedup = get_feature_cols(families)
    print(f"  {'TOTAL (dedup)':<18}  {len(dedup):>6}")
