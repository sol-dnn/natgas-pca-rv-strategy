"""Model zoo for 0DTE conditional prediction (binary classification).

ModelSpec wraps a sklearn-compatible estimator.  All hyperparameters are
tuned for low-SNR daily financial panel data:
  - small training windows (~252 days × 64 strategies)
  - IC ≈ 0.02–0.05, high noise
  - over-regularise rather than under-regularise

Zoo composition (14 models, binary classifier):
  logit_tight / mid / loose          — L2 logistic, 3 regularisation levels
  rf_deep / rf_reg                   — Random Forest, 2 depth/leaf variants
  et                                 — Extra Trees
  hgb                                — sklearn HistGradientBoosting
  xgb_tight / mid / loose            — XGBoost, 3 lambda variants
  lgbm_tight / mid / loose / bagged  — LightGBM, 4 variants

LightGBM and XGBoost are required dependencies (requirements-modelzoo.txt).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from lightgbm import LGBMClassifier, LGBMRegressor          # type: ignore[import]
from xgboost import XGBClassifier, XGBRegressor              # type: ignore[import]
from sklearn.base import BaseEstimator, clone
from sklearn.ensemble import (
    ExtraTreesClassifier,
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
)
from sklearn.linear_model import ElasticNetCV, RidgeCV
from sklearn.model_selection import TimeSeriesSplit


@dataclass
class ModelSpec:
    """Specification for a single model entry in the zoo."""

    estimator: BaseEstimator
    """Sklearn-compatible estimator (classifier or regressor)."""

    label: str = ""
    """Human-readable model name for tables and figures."""

    param_grid: dict[str, list] | None = None
    """
    Reserved for future nested-CV use.  All zoo entries currently use
    explicit hyperparameter variants instead of per-refit grid search.
    """

    def clone_estimator(self) -> BaseEstimator:
        """Return a fresh unfitted copy via sklearn.base.clone."""
        return clone(self.estimator)

    def to_dict(self) -> dict:
        """Return a JSON-safe dict of model hyperparameters."""
        return {
            "label":      self.label,
            "class":      type(self.estimator).__name__,
            "params":     {
                k: (v if isinstance(v, (int, float, str, bool, type(None))) else str(v))
                for k, v in self.estimator.get_params(deep=False).items()
            },
        }


def build_default_zoo(
    task: Literal["binary", "regression"] = "binary",
    random_state: int = 0,
) -> dict[str, ModelSpec]:
    """
    Build the default model zoo.

    Binary zoo (14 models):
      - 3 Logistic Regression (L2, tight / mid / loose)
      - 2 Random Forest (deep / more-regularised)
      - 1 Extra Trees
      - 1 HistGradientBoosting
      - 3 XGBoost (reg_lambda: tight / mid / loose)
      - 4 LightGBM (num_leaves: tight / mid / loose / bagged)

    All tree models use conservative depth and high min-leaf counts.
    Stage 1 runs all 14 variants OOF; comparison_table() picks the winner.
    Stage 2 runs the winner only.

    Parameters
    ----------
    task         : "binary" (default) or "regression"
    random_state : seed for all stochastic models
    """
    if task == "binary":
        return _binary_zoo(random_state)
    else:
        return _regression_zoo(random_state)


# ─────────────────────────────────────────────────────────────────────────────
# Binary zoo
# ─────────────────────────────────────────────────────────────────────────────

def _binary_zoo(rs: int) -> dict[str, ModelSpec]:
    return {
        # ── LightGBM: 3 regularisation levels ────────────────────────────
        "lgbm_ultra": ModelSpec(
            estimator=LGBMClassifier(
                n_estimators=200,
                learning_rate=0.02,
                num_leaves=8,
                max_depth=3,
                min_child_samples=200,
                subsample=0.7,
                colsample_bytree=0.7,
                subsample_freq=1,
                reg_lambda=15.0,
                reg_alpha=0.5,
                objective="binary",
                random_state=rs,
                verbose=-1,
                n_jobs=-1,
            ),
            label="LGBM ultra-tight",
        ),
        "lgbm_tight": ModelSpec(
            estimator=LGBMClassifier(
                n_estimators=150,
                learning_rate=0.02,
                num_leaves=15,
                max_depth=4,
                min_child_samples=150,
                subsample=0.75,
                colsample_bytree=0.75,
                subsample_freq=1,
                reg_lambda=5.0,
                reg_alpha=0.2,
                objective="binary",
                random_state=rs,
                verbose=-1,
                n_jobs=-1,
            ),
            label="LGBM tight",
        ),
        "lgbm_mid": ModelSpec(
            estimator=LGBMClassifier(
                n_estimators=300,
                learning_rate=0.02,
                num_leaves=31,
                max_depth=5,
                min_child_samples=100,
                subsample=0.8,
                colsample_bytree=0.8,
                subsample_freq=1,
                reg_lambda=2.0,
                reg_alpha=0.1,
                objective="binary",
                random_state=rs,
                verbose=-1,
                n_jobs=-1,
            ),
            label="LGBM mid",
        ),

        # ── HistGradientBoosting: 3 regularisation levels ─────────────────
        "hgb_tight": ModelSpec(
            estimator=HistGradientBoostingClassifier(
                learning_rate=0.02,
                max_leaf_nodes=15,
                max_depth=3,
                min_samples_leaf=150,
                l2_regularization=2.0,
                max_iter=100,
                random_state=rs,
            ),
            label="HGB tight",
        ),
        "hgb_reg": ModelSpec(
            estimator=HistGradientBoostingClassifier(
                learning_rate=0.025,
                max_leaf_nodes=15,
                max_depth=3,
                min_samples_leaf=100,
                l2_regularization=1.0,
                max_iter=100,
                random_state=rs,
            ),
            label="HGB regularized",
        ),
        "hgb_loose": ModelSpec(
            estimator=HistGradientBoostingClassifier(
                learning_rate=0.025,
                max_leaf_nodes=31,
                max_depth=4,
                min_samples_leaf=60,
                l2_regularization=0.3,
                max_iter=150,
                random_state=rs,
            ),
            label="HGB loose",
        ),

        # ── XGBoost: 2 regularisation levels ──────────────────────────────
        "xgb_tight": ModelSpec(
            estimator=XGBClassifier(
                n_estimators=200,
                learning_rate=0.025,
                max_depth=3,
                min_child_weight=8,
                subsample=0.75,
                colsample_bytree=0.75,
                reg_lambda=8.0,
                reg_alpha=0.2,
                objective="binary:logistic",
                eval_metric="logloss",
                random_state=rs,
                n_jobs=-1,
                verbosity=0,
            ),
            label="XGB tight",
        ),
        "xgb_mid": ModelSpec(
            estimator=XGBClassifier(
                n_estimators=300,
                learning_rate=0.025,
                max_depth=3,
                min_child_weight=4,
                subsample=0.8,
                colsample_bytree=0.8,
                reg_lambda=3.0,
                reg_alpha=0.1,
                objective="binary:logistic",
                eval_metric="logloss",
                random_state=rs,
                n_jobs=-1,
                verbosity=0,
            ),
            label="XGB mid",
        ),
    }
# ─────────────────────────────────────────────────────────────────────────────
# Regression zoo (minimal — extend when needed)
# ─────────────────────────────────────────────────────────────────────────────

def _regression_zoo(rs: int) -> dict[str, ModelSpec]:
    # v1 zoo (8 models): ridge, hgb×3, lgbm×2, xgb×2
    #   Results: ridge ICIR=2.03 (ic_gap=0.031), hgb_reg ICIR=1.97 (ic_gap=0.028).
    #   LGBM/XGB massively overfit (ic_gap 0.17–0.41). New v2 zoo drops loose variants,
    #   adds ultra-tight HGB/LGBM/XGB with 3–10× stronger regularisation.
    ridge_alphas = [1e-4, 1e-3, 1e-2, 0.1, 1.0, 10.0, 100.0]
    return {
        # ── Linear baselines ─────────────────────────────────────────────────
        # fit_intercept=True (default): captures unconditional mean PnL per strategy.
        # fit_intercept=False: forces model to explain everything cross-sectionally;
        #   appropriate if features are already CS-standardised (mean 0 per bar).
        # Both variants provided — comparison shows whether unconditional mean helps.
        "ridge": ModelSpec(
            estimator=RidgeCV(alphas=ridge_alphas, cv=TimeSeriesSplit(n_splits=3),
                              fit_intercept=True),
            label="Ridge CV",
        ),
        "ridge_no_int": ModelSpec(
            estimator=RidgeCV(alphas=ridge_alphas, cv=TimeSeriesSplit(n_splits=3),
                              fit_intercept=False),
            label="Ridge (no intercept)",
        ),
        # ElasticNet (L1+L2): sparse selection without full zeroing.
        # L2 term prevents all-zero coefficients (the Lasso failure mode).
        # l1_ratio grid: [0.5, 0.7, 0.9] — 0.9 is near-Lasso, 0.5 is balanced.
        "elasticnet": ModelSpec(
            estimator=ElasticNetCV(
                l1_ratio=[0.5, 0.7, 0.9],
                cv=TimeSeriesSplit(n_splits=3), n_alphas=10,
                max_iter=2000, tol=1e-3, fit_intercept=True, random_state=rs,
            ),
            label="ElasticNet CV",
        ),
        "elasticnet_no_int": ModelSpec(
            estimator=ElasticNetCV(
                l1_ratio=[0.5, 0.7, 0.9],
                cv=TimeSeriesSplit(n_splits=3), n_alphas=10,
                max_iter=2000, tol=1e-3, fit_intercept=False, random_state=rs,
            ),
            label="ElasticNet (no intercept)",
        ),
        # ── HGB regressors: 4 regularisation levels ───────────────────────────
        # hgb_reg is the v1 winner (ic_gap=0.028). ultra/super explore tighter.
        "hgb_ultra": ModelSpec(
            estimator=HistGradientBoostingRegressor(
                max_iter=200, learning_rate=0.02, max_depth=2,
                min_samples_leaf=200, l2_regularization=20.0,
                random_state=rs,
            ),
            label="HGB ultra-tight",
        ),
        "hgb_super": ModelSpec(
            estimator=HistGradientBoostingRegressor(
                max_iter=200, learning_rate=0.02, max_depth=3,
                min_samples_leaf=150, l2_regularization=10.0,
                random_state=rs,
            ),
            label="HGB super-tight",
        ),
        "hgb_reg": ModelSpec(
            estimator=HistGradientBoostingRegressor(
                max_iter=300, learning_rate=0.03, max_depth=3,
                min_samples_leaf=80, l2_regularization=5.0,
                random_state=rs,
            ),
            label="HGB regularised",
        ),
        "hgb_tight": ModelSpec(
            estimator=HistGradientBoostingRegressor(
                max_iter=300, learning_rate=0.05, max_depth=4,
                min_samples_leaf=60, l2_regularization=1.0,
                random_state=rs,
            ),
            label="HGB tight",
        ),
        # ── LightGBM: ultra-tight only (v1 versions overfit severely) ─────────
        "lgbm_ultra": ModelSpec(
            estimator=LGBMRegressor(
                n_estimators=200, learning_rate=0.02, num_leaves=8,
                min_child_samples=300, reg_lambda=30.0, reg_alpha=1.0,
                subsample=0.7, colsample_bytree=0.7, subsample_freq=1,
                objective="regression", random_state=rs, verbose=-1, n_jobs=-1,
            ),
            label="LGBM ultra-tight",
        ),
        "lgbm_super": ModelSpec(
            estimator=LGBMRegressor(
                n_estimators=200, learning_rate=0.02, num_leaves=15,
                min_child_samples=200, reg_lambda=15.0, reg_alpha=0.5,
                subsample=0.7, colsample_bytree=0.7, subsample_freq=1,
                objective="regression", random_state=rs, verbose=-1, n_jobs=-1,
            ),
            label="LGBM super-tight",
        ),
        # ── XGBoost: ultra-tight only ─────────────────────────────────────────
        "xgb_ultra": ModelSpec(
            estimator=XGBRegressor(
                n_estimators=200, learning_rate=0.02, max_depth=2,
                min_child_weight=20, reg_lambda=30.0, reg_alpha=1.0,
                subsample=0.7, colsample_bytree=0.7,
                objective="reg:squarederror", random_state=rs,
                n_jobs=-1, verbosity=0,
            ),
            label="XGB ultra-tight",
        ),
        "xgb_super": ModelSpec(
            estimator=XGBRegressor(
                n_estimators=200, learning_rate=0.02, max_depth=3,
                min_child_weight=15, reg_lambda=15.0, reg_alpha=0.5,
                subsample=0.7, colsample_bytree=0.7,
                objective="reg:squarederror", random_state=rs,
                n_jobs=-1, verbosity=0,
            ),
            label="XGB super-tight",
        ),
    }
