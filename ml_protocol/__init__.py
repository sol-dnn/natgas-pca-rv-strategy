"""ml_protocol — walk-forward ML evaluation framework for 0DTE strategies.

Modules
-------
splitter         : TimeSeriesSplitter, SplitConfig
preprocessor     : FeaturePreprocessor, PreprocessorConfig
zoo              : ModelSpec, build_default_zoo
runner           : walk_forward, WalkForwardConfig, WalkForwardResult
evaluator        : ml_metrics, ic_series, icir, ic_gap, comparison_table
strategy_groups  : STRATEGY_FAMILY_MAP, STRATEGY_FAMILIES

Note: economic / backtest metrics (SR, drawdown, turnover) are intentionally
not included here.  They will live in a separate backtest/ module, applied
after model selection.

Typical usage
-------------
    from ml_protocol import (
        SplitConfig, WalkForwardConfig, PreprocessorConfig,
        build_default_zoo, walk_forward, comparison_table,
    )

    config = WalkForwardConfig(
        split_config=SplitConfig(
            min_train_days=252,
            window_type="expanding",
            refit_every=21,
            oos_start_date=pd.Timestamp("2023-01-01"),
        ),
        preprocessor_config=PreprocessorConfig(
            ts_cols=["iv", "isk", "slope_up", "slope_dn", "SPX_lrv", "pnl_l1"],
            winsor_quantile=0.025,
        ),
        task="binary",
        mode="panel",
    )

    zoo = build_default_zoo(task="binary")
    result = walk_forward(data, feature_cols, "pnl_net", zoo, config)
    table = comparison_table(result.predictions, result.split_log)
"""
from .strategy_groups import STRATEGY_FAMILIES, STRATEGY_FAMILY_MAP
from .evaluator import (
    annualised_sharpe,
    comparison_table,
    comparison_table_by_group,
    ic_by_time,
    ic_gap,
    ic_series,
    ic_series_intraday,
    ic_tstat,
    icir,
    ml_metrics,
    print_comparison_summary,
    save_stage1_winner,
)
from .preprocessor import FeaturePreprocessor, PreprocessorConfig
from .runner import WalkForwardConfig, WalkForwardResult, walk_forward
from .splitter import SplitConfig, TimeSeriesSplitter
from .zoo import ModelSpec, build_default_zoo
from .feature_importance import collect_feature_importances, FeatureImportanceSummary

__all__ = [
    # splitter
    "SplitConfig",
    "TimeSeriesSplitter",
    # preprocessor
    "PreprocessorConfig",
    "FeaturePreprocessor",
    # zoo
    "ModelSpec",
    "build_default_zoo",
    # runner
    "WalkForwardConfig",
    "WalkForwardResult",
    "walk_forward",
    # strategy groups
    "STRATEGY_FAMILY_MAP",
    "STRATEGY_FAMILIES",
    # evaluator
    "ml_metrics",
    "ic_series",
    "icir",
    "ic_by_time",
    "ic_series_intraday",
    "ic_gap",
    "ic_tstat",
    "comparison_table",
    "comparison_table_by_group",
    "print_comparison_summary",
    "save_stage1_winner",
    "annualised_sharpe",
    "best_model_json",
    "collect_feature_importances",
    "FeatureImportanceSummary",
]
