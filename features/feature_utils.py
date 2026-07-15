"""
feature_utils.py
----------------
Shared constants, helpers, and strategy leg definitions for the
conditional feature pipeline.
"""
from __future__ import annotations

import json
import logging
from typing import List, Tuple

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

# ── Numeric constants ──────────────────────────────────────────────────────────
EPS = 1e-12
SQRT252 = np.sqrt(252)
SPX_MULTIPLIER = 100  # one SPX contract covers 100 index units

# ── Intraday bar grid 10:00–15:55 ET, 5-min spacing ───────────────────────────
BAR_ORDER: list[str] = [
    f"{h:02d}:{m:02d}"
    for h in range(10, 16)
    for m in range(0, 60, 5)
    if f"{h:02d}:{m:02d}" <= "15:55"
]  # 72 bars
BAR_INDEX: dict[str, int] = {t: i for i, t in enumerate(BAR_ORDER)}
BAR_MINUTES: dict[str, int] = {
    t: (int(t[:2]) - 10) * 60 + int(t[3:5]) for t in BAR_ORDER
}
MINUTES_TO_CLOSE: dict[str, int] = {t: 360 - m for t, m in BAR_MINUTES.items()}

# ── Strategy universe ─────────────────────────────────────────────────────────
ALL_STRATEGIES: list[str] = [
    "strangle", "iron_condor", "risk_reversal",
    "bull_call_spread", "bear_put_spread",
    "call_ratio_spread", "put_ratio_spread",
    "straddle", "iron_butterfly", "call_butterfly", "put_butterfly",
]

# Columns that are strictly forward-looking — never use as features
FORWARD_LOOKING_COLS: frozenset[str] = frozenset({
    # strategy parquet
    "S_settle", "settlement_proxy", "payoff",
    "reth_und", "reth", "reth_und_net_spread", "reth_und_net",
    "reth_und_opp", "reth_und_opp_net_spread", "reth_und_opp_net",
    # option parquet
    "settlement_time_used", "reth_und_pct", "reth_pct", "sret", "log_sret",
    # realized moments (same-day)
    "SPX_lrv", "SPX_srv", "SPX_lrvup", "SPX_lrvdn",
    "SPX_srvup", "SPX_srvdn", "SPX_lrv_skew", "SPX_srv_skew",
    "SPX_lret", "SPX_sret",
})


# ── Leg construction: all 11 strategies ──────────────────────────────────────
def parse_mnes_levels(mnes_str: str) -> list[float]:
    """'0.995/1.005' → [0.995, 1.005]; '1/1' → [1.0, 1.0]."""
    try:
        return [float(x) for x in str(mnes_str).split("/")]
    except Exception:
        return []


def _mnes_int(v: float) -> int:
    return int(round(v * 1000))


def get_legs(strategy: str, mnes_str: str) -> list[tuple[str, int, int]]:
    """
    Return [(leg_opt_type, mnes_int, signed_qty), ...] for a strategy config.

    mnes_int is round(target_mnes × 1000) matching the convention in opt_5min.
    Covers all 11 strategies in the massive dataset.
    """
    levels = parse_mnes_levels(mnes_str)
    if not levels:
        return []
    lo, hi = levels[0], levels[-1]

    if strategy == "straddle":          # 1/1
        return [("P", _mnes_int(levels[0]), 1), ("C", _mnes_int(levels[1]), 1)]
    if strategy == "strangle":          # 0.995/1.005
        return [("P", _mnes_int(lo), 1), ("C", _mnes_int(hi), 1)]
    if strategy == "risk_reversal":     # 0.995/1.005
        return [("P", _mnes_int(lo), -1), ("C", _mnes_int(hi), 1)]
    if strategy == "bull_call_spread":  # 1/1.005
        return [("C", _mnes_int(lo), 1), ("C", _mnes_int(hi), -1)]
    if strategy == "bear_put_spread":   # 0.995/1
        return [("P", _mnes_int(lo), -1), ("P", _mnes_int(hi), 1)]
    if strategy == "call_ratio_spread": # 1/1.005  → +1C/-2C
        return [("C", _mnes_int(lo), 1), ("C", _mnes_int(hi), -2)]
    if strategy == "put_ratio_spread":  # 0.995/1  → -2P/+1P
        return [("P", _mnes_int(lo), -2), ("P", _mnes_int(hi), 1)]
    if strategy == "iron_condor":       # 0.995/0.997/1.003/1.005
        if len(levels) == 4:
            ml, mh = levels[1], levels[2]
            return [("P", _mnes_int(lo), 1), ("P", _mnes_int(ml), -1),
                    ("C", _mnes_int(mh), -1), ("C", _mnes_int(hi), 1)]
        if len(levels) == 3:
            m = levels[1]
            return [("P", _mnes_int(lo), 1), ("P", _mnes_int(m), -1),
                    ("C", _mnes_int(m), -1), ("C", _mnes_int(hi), 1)]
    if strategy == "iron_butterfly":    # 0.995/1/1.005
        if len(levels) >= 3:
            m = levels[1]
            return [("P", _mnes_int(lo), 1), ("P", _mnes_int(m), -1),
                    ("C", _mnes_int(m), -1), ("C", _mnes_int(hi), 1)]
    if strategy == "call_butterfly":    # 0.99/1/1.01
        if len(levels) >= 3:
            m = levels[1]
            return [("C", _mnes_int(lo), 1), ("C", _mnes_int(m), -2),
                    ("C", _mnes_int(hi), 1)]
    if strategy == "put_butterfly":     # 0.98/0.99/1
        if len(levels) >= 3:
            m = levels[1]
            return [("P", _mnes_int(lo), 1), ("P", _mnes_int(m), -2),
                    ("P", _mnes_int(hi), 1)]
    log.warning(f"Unknown strategy: {strategy!r}, mnes={mnes_str!r}")
    return []


# ── Safe arithmetic helpers ───────────────────────────────────────────────────
def safe_div(
    num: "pd.Series | np.ndarray",
    denom: "pd.Series | np.ndarray",
    eps: float = EPS,
) -> "pd.Series | np.ndarray":
    """num / denom, returns NaN where |denom| <= eps."""
    if isinstance(denom, pd.Series):
        d = denom.where(denom.abs() > eps, other=np.nan)
        return num / d
    else:
        d = np.where(np.abs(denom) > eps, denom, np.nan)
        return num / d


def safe_log(x: "pd.Series | np.ndarray", eps: float = EPS) -> "pd.Series | np.ndarray":
    """log(x), returning NaN for x <= 0."""
    if isinstance(x, pd.Series):
        return np.log(x.clip(lower=eps))
    return np.log(np.clip(x, eps, None))


# ── Trading-date shift helper ─────────────────────────────────────────────────
def add_prev_trading_date(df: pd.DataFrame, date_col: str = "quote_date") -> pd.DataFrame:
    """
    Add a 'prev_trading_date' column = the immediately preceding trading date
    in the dataset's own calendar (not a fixed calendar day lag).
    """
    dates_sorted = sorted(df[date_col].unique())
    prev_map = {d: dates_sorted[i - 1] if i > 0 else pd.NaT
                for i, d in enumerate(dates_sorted)}
    df = df.copy()
    df["prev_trading_date"] = df[date_col].map(prev_map)
    return df


# ── Payoff shape on a normalised grid ────────────────────────────────────────
def payoff_shape_vectorised(
    strategies: "list[tuple[str,str]]",   # (option_type, mnes_str) pairs
    grid_rel: np.ndarray,                  # x/S grid, e.g. linspace(0.85, 1.15, 501)
) -> dict[tuple[str, str], np.ndarray]:
    """
    Pre-compute spot-normalised payoff (in units of S) on grid_rel for each
    (option_type, mnes_str) pair.  Entry cost is NOT subtracted here.

    Returns dict mapping (option_type, mnes_str) → 1-D np.ndarray of length len(grid_rel).
    """
    cache: dict[tuple[str, str], np.ndarray] = {}
    for opt_type, mnes_str in strategies:
        legs = get_legs(opt_type, mnes_str)
        if not legs:
            continue
        payoff = np.zeros(len(grid_rel))
        for leg_type, mnes_int, q in legs:
            K_rel = mnes_int / 1000.0
            if leg_type == "C":
                payoff += q * np.maximum(grid_rel - K_rel, 0.0)
            else:
                payoff += q * np.maximum(K_rel - grid_rel, 0.0)
        cache[(opt_type, mnes_str)] = payoff
    return cache


# ── Winsorisation ─────────────────────────────────────────────────────────────
def winsorize(s: pd.Series, lo: float = 0.005, hi: float = 0.995) -> pd.Series:
    """Clip series to [lo, hi] empirical quantiles (ignores NaN)."""
    lo_val = s.quantile(lo)
    hi_val = s.quantile(hi)
    return s.clip(lo_val, hi_val)
