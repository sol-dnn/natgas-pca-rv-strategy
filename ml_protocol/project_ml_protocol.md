---
name: project-ml-protocol
description: ML protocol module (code/ml_protocol/) — design decisions, current state, what's done
metadata:
  type: project
---

**Why:** Build a walk-forward OOS ML protocol for 0DTE conditional strategies.
**How to apply:** Don't re-engineer this — it's been fully audited (2026-06-05). Use it directly.

## Architecture decisions (fixed)

- **Cross-section**: (strategy_type × mnes) per (date, time) slot — ~45 obs per slot
- **Target**: net_pnl > 0 from entry at time t to 16:00 close; continuous net_pnl for IC
- **Timing strategy**: run walk_forward() per time slot (10:00, 13:00, 15:00) separately — cleaner econometrics, homogeneous target. Multi-time panel is supported but IC must use ic_series_intraday(), not ic_series().
- **Panel vs per-strategy**: both supported. Panel recommended (more data, strategy dummies in features). Per-strategy runs 7 separate calls, follows Vilkov.
- **S1/S2 separation**: strictly enforced by oos_end_date / oos_start_date in SplitConfig. Running S1 never sees S2 data. S2 is a single separate walk_forward() call with the S1 winner only.

## Files

### preprocessor.py — FeaturePreprocessor + PreprocessorConfig
- 5-stage pipeline: sanitize → impute → winsor → z-score → fallback
- All stats fit on train only; output guaranteed finite
- Column groups: dummy_cols (fill 0, no winsor, no zscore), cs_cols (fill 0, winsor, no zscore), ts_cols (fill median, winsor, zscore), raw_cols (fill median, winsor, no zscore)
- Z-score pooled across all rows in training window (correct: market-level ts_cols have same value per (date,time) across strategies; strategy-specific features go in cs_cols already normalized)
- Vilkov uses plain StandardScaler, no winsorization, no column-type distinction — our preprocessor is strictly superior
- `auto_raw_uncategorized=True`: uncategorized columns treated as raw with warning
- `error_on_overlap=True`: raises if a col appears in >1 category
- Winsor bounds computed from pre-imputation data via nanquantile
- `pp.fit_log` has diagnostics; `pp.to_dict()` is JSON-safe

### runner.py — walk_forward()
- `valid_tr = np.isfinite(y_tr)` — only drops NaN TARGETS; NaN features imputed by preprocessor
- Preprocessor guarantees finite test output — no valid_te filter needed
- split_log includes pp diagnostics (nan_rate, n_imputed, etc.)
- IS IC computed on last is_eval_days of training data (diagnostic only, no feedback to model)
- sort_extra_cols=["quote_time","mnes"] ensures deterministic row order within (date,strategy)

### splitter.py — TimeSeriesSplitter + SplitConfig
- Operates on unique trading dates (never splits within a day)
- purge_days=1: last train date = test_date - 2, purged date = test_date - 1, test = test_date
- rolling or expanding window
- oos_start_date / oos_end_date enforce strict S1/S2 boundary — no data contamination possible

### zoo.py — build_default_zoo()
- 14 binary models: 3 logit (C=0.05/0.35/1.0), 2 RF, 1 ET, 1 HGB, 3 XGB, 4 LGBM
- All heavily regularized for low-SNR daily financial panel (IC ≈ 0.02–0.05)
- 5 regression models: ridge, EN, RF, LGBM, XGB (for robustness checks)
- ModelSpec.clone_estimator() → fresh sklearn.base.clone at each refit step

### evaluator.py
- ml_metrics: IC, ICIR, IC t-stat (Newey-West HAC), hit rate, AUC, Brier, calibration, coverage
- comparison_table: ranks all models by ICIR, BHY multiple-testing correction on t-stats
- ic_series(): pd.Series(date→IC) — use per model after filtering by model_id
- ic_series_intraday(): IC per (date,time) slot averaged to one value per day — use for multi-time panel
- ic_by_time(): IC breakdown by quote_time — diagnose where predictability concentrates
- ic_gap(): IS IC vs OOS IC per model — main overfitting diagnostic
- save_stage1_winner: saves JSON artifact with full config for Stage 2 reproducibility

### feature_importance.py — collect_feature_importances()
- Called SEPARATELY after walk_forward(), for ONE model (the winner)
- Re-runs the full walk-forward loop to collect importances per refit cycle
- Per-cycle: coef/mdi computed at fit time (train data), perm/shap on FULL CYCLE test data (refit_every × ~45 obs ≈ 945 obs/fold) — NOT just 1 day
- _commit_cycle() flushes accumulated test data at each new refit and at end of loop
- Summary: mean ± std across folds + stability t-stat (cross-fold consistency)

### strategy_groups.py
- Simple dict: STRATEGY_FAMILY_MAP maps 11 strategy types → 5 families
- Used manually: data["strategy_family"] = data["strategy_type"].map(STRATEGY_FAMILY_MAP)
- Only needed for mode="per_strategy" with group_col="strategy_family"

## Usage pattern — single time slot (recommended baseline)

```python
# Filter to one entry time
data_t = full_data[full_data["quote_time"] == pd.Timestamp("13:00:00").time()].copy()

# Configure
pp_cfg = PreprocessorConfig(
    dummy_cols=cat["dummies"], cs_cols=cat["cs"],
    ts_cols=cat["ts"], raw_cols=cat["raw"],
    winsor_cols=cat["ts"] + cat["raw"] + cat["cs"],
)
wf_cfg = WalkForwardConfig(
    split_config=SplitConfig(
        oos_start_date=pd.Timestamp("2023-07-01"),
        oos_end_date=pd.Timestamp("2025-06-30"),   # S1 ends here
        min_train_days=252, refit_every=21, purge_days=1,
    ),
    preprocessor_config=pp_cfg,
    sort_extra_cols=["mnes"],
)

# Stage 1 — all models
result_s1 = walk_forward(data_t, feature_cols, "target", build_default_zoo(), wf_cfg)
table = comparison_table(result_s1.predictions, result_s1.split_log)
save_stage1_winner(table, zoo, wf_cfg, feature_cols, "target", "output/stage1_13h.json")

# Feature importance (winner only)
fi = collect_feature_importances(data_t, feature_cols, "target",
                                  model_id="lgbm_mid", model_spec=zoo["lgbm_mid"],
                                  config=wf_cfg)

# Stage 2 — winner only, untouched holdout
cfg_s2 = replace(wf_cfg, split_config=replace(wf_cfg.split_config,
    oos_start_date=pd.Timestamp("2025-07-01"), oos_end_date=None))
result_s2 = walk_forward(data_t, feature_cols, "target",
                          {"lgbm_mid": zoo["lgbm_mid"]}, cfg_s2)
```

## Bugs fixed (history)
- pnl_mean5_same_time rolling across group boundaries → fixed (grouped rolling)
- flow coverage ratio wrong for multi-leg strategies → fixed (_n_legs_expected ratio)
- macro vix/sofr lookahead (vix = today's close, not prior) → removed from features
- vilkov_h divided by 2 twice (bas is already half-spread) → fixed
- theta/vega/gamma not spot-normalized → gamma_S=gamma*S, theta_S=theta/S added for ratios
- feature_catalog CS_SUFFIXES checked after TS_PREFIXES → fixed (CS first)
- feature_importance perm/shap on 1 day of test only → fixed: accumulated over full refit cycle
