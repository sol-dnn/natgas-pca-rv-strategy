"""Strategy family groupings for per-family walk-forward mode.

Used when mode="per_strategy" with group_col="strategy_family".
Add a strategy_family column to the feature dataset at build time:

    from ml_protocol.strategy_groups import STRATEGY_FAMILY_MAP
    data["strategy_family"] = data["strategy_type"].map(STRATEGY_FAMILY_MAP)
"""
from __future__ import annotations

STRATEGY_FAMILY_MAP: dict[str, str] = {
    "straddle":           "long_vol",
    "strangle":           "long_vol",

    "iron_butterfly":     "short_vol_bounded",
    "iron_condor":        "short_vol_bounded",

    "call_butterfly":     "butterfly",
    "put_butterfly":      "butterfly",

    "bull_call_spread":   "directional_spread",
    "bear_put_spread":    "directional_spread",

    "call_ratio_spread":  "ratio_spread",
    "put_ratio_spread":   "ratio_spread",

    "risk_reversal":      "skew_directional",
}

STRATEGY_FAMILIES: list[str] = sorted(set(STRATEGY_FAMILY_MAP.values()))
