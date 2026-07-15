# Conditional Feature Pipeline — 0DTE Massive Panel

## Overview

`massive_build_conditional_features.py` builds a strategy-level panel dataset with 232 features and 8 targets, one row per `(quote_date, quote_time, option_type, mnes)`.

Coverage: 992 dates × 72 bars (10:00–15:55 ET) × 11 strategy types.

## How to Run

```bash
# Quick test on 5 dates
python code/model/massive_build_conditional_features.py --sample 5

# Full run (all 992 dates, all 72 bars, ~4.5M rows) — takes ~15 min
python code/model/massive_build_conditional_features.py

# 10:00 entry only
python code/model/massive_build_conditional_features.py --times 10:00

# Validate existing output
python code/model/massive_build_conditional_features.py --validate-only
```

## Output

| File | Description |
|------|-------------|
| `data/processed/conditional_features_5min.parquet` | Feature + target panel |
| `data/processed/conditional_feature_metadata.json` | Feature list, target list, no-lookahead rules, validation report |

## Feature Families (232 total)

| Prefix | Count | Source | Description |
|--------|-------|--------|-------------|
| `mid`, `bas`, `h`, `rho`, `d`, `fee`, ... | ~30 | strategy_5min | Strategy snapshot: price, spread, depth |
| `delta`, `gamma`, `vega`, `theta`, `iv_mean` | ~10 | strategy_5min | Strategy-level Greeks |
| `abs_*`, `log_*`, `cost_*`, `tv_*`, ratio features | ~20 | derived | Price/Greek ratios |
| `mnes_min/max/width`, `n_legs`, `n_contracts`, ... | ~8 | strategy_5min | Geometry |
| `ps_*` (17 cols) | 17 | legs + vix_5min | Payoff shape: max profit/loss, breakeven distances, expected-move ratio |
| `vilkov_h/d/rho/T/gGamma_n/a/gamma_balance` | ~10 | opt_5min snapshot | Vilkov liquidity & GEX (no OI) |
| `vilkov_v/fDelta/fGamma/fVega` + `_lag30/_cum` | ~27 | opt_5min lagged | Vilkov flow features (SHIFTED 1 bar) |
| `strat_vol_lag5/lag30/since_open` | ~6 | strategy_5min shifted | Strategy OHLCV lags |
| `spx_*` (19 cols) | 19 | strategy_5min (S) | SPX returns, RV, range, timing, dummies |
| `ivar_*`, `atm_iv`, `slope_*`, `expected_move_*` | ~15 | vix_5min + slopes_5min | 0DTE implied variance & surface |
| `opt_iv_atm/990/1010/skew/slope/cp_diff` | 12 | opt_5min direct | IV from option chain (cross-check) |
| `vrp_lag1_same_time`, `rv_lag1_*`, `ret_to_close_lag1_*` | 6 | vix_5min + realized_5min | Lagged realized features (prev day, same time) |
| `pnl_lag1/2/3_same_time`, `pnl_mean5/std5_same_time` | 5 | strategy_5min | Vilkov Eq (26) PnL lags |
| `macro_vix_prev_close`, `macro_sofr`, VIX changes | ~5 | macro_daily_controls | Macro controls |
| `is_cpi/nfp/fomc_day`, `post/pre_fomc_time`, `cpi_or_nfp_day` | ~7 | macro_daily_controls | Event dummies |
| `*_timeslot_z` (9 cols) | 9 | derived | Expanding z-score within (strategy, mnes, bar_time) — needs ≥30 obs |
| `*_cs` (15 cols) | 15 | derived | Cross-sectional z-score within (date, time) |
| `strat_*` dummies | 11 | option_type | One-hot strategy type indicators |

## Targets (8)

All prefixed `target_`:

| Column | Definition |
|--------|-----------|
| `target_y_long_net` | `reth_und_net` (long, net of spread + fee) |
| `target_y_short_net` | `reth_und_opp_net` (short, net) |
| `target_y_long_gross` | `reth_und` (long, no costs) |
| `target_y_short_gross` | `reth_und_opp` (short, no costs) |
| `target_y_best_side_net` | `max(long_net, short_net)` |
| `target_y_long_minus_short_net` | `long_net − short_net` |
| `target_y_long_profitable` | `1{long_net > 0}` |
| `target_y_short_profitable` | `1{short_net > 0}` |

## No-Lookahead Rules

1. **OHLCV** (strategy-level `ohlcv_volume_sum`, opt-level `ohlcv_volume`) covers bar `[t, t+5)` → shifted +1 bar before use. At 10:00, lag is NaN.
2. **Option flow** (`trade_volume_delta/gamma/vega_usd`) is bar `[t, t+5)` → shifted +1 bar.
3. **Realized moments** (`SPX_lrv`, `SPX_lrv_skew`, `SPX_lret`) are forward-looking from `t+1` to 16:00. Used ONLY with +1 trading-date lag at same `quote_time`.
4. **VRP** = `vix × 1e5 − SPX_lrv` from *previous* trading date.
5. **PnL lags** grouped by `(option_type, mnes, quote_time)`, shifted by +1 date.
6. **Macro VIX/SOFR** are prior-close values (pre-lagged in `macro_daily_controls.csv`).
7. **Timeslot z-scores** use expanding mean/std shifted +1 date (strict past-only).

## Assumptions

1. Strategy Greek NaNs (~40% of rows for some strategies) come from the source `massive_strategy_5min.parquet`, not the pipeline.
2. Timeslot z-scores need ≥30 prior observations; with 5-date test the field is 100% NaN.
3. Open interest (`oi_gamma`, `open_interest`, etc.) is all NaN in the source and is not used.
4. Payoff shape uses target moneyness (`mnes_target × S`) to approximate strikes; snap error is typically < 1%.
5. OI-based Vilkov features (`B^Gamma`, `R^Gamma`) are not computed since OI is unavailable.
6. `09:30–10:00` opening data is unavailable; opening gap features are omitted.
