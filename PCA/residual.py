"""
Residual-based features from the rolling two-stage PCA, known after the close at time t.


Input: df_residuals  — DataFrame (date × maturity) of daily residual returns ε_{i,t}

Features built per (date, maturity):
  eps                  — current residual ε_{i,t}
  eps_lag_1            — ε_{i,t-1}
  eps_lag_5            — ε_{i,t-5}
  eps_rolling_mean_5   — mean(ε_{i,t-4:t})
  eps_rolling_mean_20  — mean(ε_{i,t-19:t})
  eps_rolling_std_20   — std(ε_{i,t-19:t})
  eps_zscore_20        — ε / rolling_std(ε, 20)
  eps_zscore_63        — ε / rolling_std(ε, 63)   [longer horizon normalisation]
  eps_momentum_5       — rolling_mean_5 / rolling_std_20  [sign × magnitude]
  eps_momentum_20      — rolling_mean_20 / rolling_std_20
  cum_dislocation_5    — Σ_{k=0}^{4} ε_{t-k}   [5-day accumulated rich/cheap]
  cum_dislocation_20   — Σ_{k=0}^{19} ε_{t-k}
  idio_vol             — σ_{i,t} from two-stage PCA (stage-1 idiosyncratic vol proxy)
  eps_idio_zscore      — ε_{i,t} / σ_{i,t}   [PCA-normalised residual]

  
  Cross-sectional residual-shape features:
  eps_vs_curve         — ε_{i,t} − mean_i(ε_{i,t})
  eps_local_curvature  — ε_{i+1,t} − 2ε_{i,t} + ε_{i-1,t}
                         with one-sided second difference at M1 and M12
  eps_curve_std        — std_i(ε_{i,t}) across maturities at date t

  
Family A:
  all features are returned per (date, maturity).
  eps_curve_std is date-level in meaning, but broadcast across maturities here.
"""

import numpy as np
import pandas as pd
from typing import Optional


_MATURITIES = [f'M{i}' for i in range(1, 13)]

# Map between price-column convention (hh_1, hh_2, …) and maturity labels (M1, M2, …)
_PRICE_TO_MAT = {f'hh_{i}': f'M{i}' for i in range(1, 13)}


FAMILY_A_COLS = [
    "eps", "eps_lag_1", "eps_lag_5",
    "eps_rolling_mean_5", "eps_rolling_mean_20",
    "eps_rolling_std_20", "eps_rolling_std_63",
    "eps_zscore_20", "eps_zscore_63",
    "eps_momentum_5", "eps_momentum_20",
    "cum_dislocation_5", "cum_dislocation_20",
    "eps_cum5_zscore",
    "eps_vs_curve", "eps_local_curvature", "eps_curve_std",
]


def build_residual_features(
    df_residuals: pd.DataFrame,
    maturities: Optional[list] = None,
) -> pd.DataFrame:
    """
    Build residual-based features from the rolling factor model outputs.

    Parameters
    ----------
    df_residuals : wide DataFrame indexed by date, columns hh_1…hh_12 or M1…M12
    maturities   : subset of ['M1', …, 'M12'] to build, default all 12

    Returns
    -------
    DataFrame with MultiIndex (date, maturity), columns = feature names.

    Timing
    ------
    All features at date t use only residuals available up to and including t.
    Therefore they are known after the close at t and can be used for a signal
    traded from t+1.
    """
    maturities = maturities or _MATURITIES

    # Normalise column names to M1…M12
    eps = df_residuals.rename(columns=_PRICE_TO_MAT).copy()
    eps = eps[[m for m in maturities if m in eps.columns]]
    eps = eps.sort_index()

    # Cross-sectional quantities at date t
    eps_curve_mean = eps.mean(axis=1)
    eps_curve_std = eps.std(axis=1)

    parts = []

    for mat in eps.columns:
        m_num = int(mat[1:])
        s = eps[mat]

        feat = pd.DataFrame(index=s.index)

        # Current and time-series residual features
        feat["eps"] = s
        feat["eps_lag_1"] = s.shift(1)
        feat["eps_lag_5"] = s.shift(5)

        feat["eps_rolling_mean_5"] = s.rolling(5, min_periods=3).mean()
        feat["eps_rolling_mean_20"] = s.rolling(20, min_periods=10).mean()

        feat["eps_rolling_std_20"] = s.rolling(20, min_periods=10).std()
        feat["eps_rolling_std_63"] = s.rolling(63, min_periods=30).std()

        vol20 = feat["eps_rolling_std_20"].replace(0, np.nan)
        vol63 = feat["eps_rolling_std_63"].replace(0, np.nan)

        feat["eps_zscore_20"] = s / vol20
        feat["eps_zscore_63"] = s / vol63

        feat["eps_momentum_5"] = feat["eps_rolling_mean_5"] / vol20
        feat["eps_momentum_20"] = feat["eps_rolling_mean_20"] / vol20

        feat["cum_dislocation_5"] = s.rolling(5, min_periods=3).sum()
        feat["cum_dislocation_20"] = s.rolling(20, min_periods=10).sum()

        feat["eps_cum5_zscore"] = feat["cum_dislocation_5"] / (np.sqrt(5) * vol63)

        # Cross-sectional residual-shape features
        feat["eps_vs_curve"] = s - eps_curve_mean
        feat["eps_curve_std"] = eps_curve_std

        prev_mat = f"M{m_num - 1}"
        next_mat = f"M{m_num + 1}"

        if prev_mat in eps.columns and next_mat in eps.columns:
            feat["eps_local_curvature"] = (
                eps[next_mat] - 2.0 * eps[mat] + eps[prev_mat]
            )

        elif next_mat in eps.columns:
            # M1 boundary: one-sided second difference
            next2_mat = f"M{m_num + 2}"
            if next2_mat in eps.columns:
                feat["eps_local_curvature"] = (
                    eps[next2_mat] - 2.0 * eps[next_mat] + eps[mat]
                )
            else:
                feat["eps_local_curvature"] = np.nan

        elif prev_mat in eps.columns:
            # M12 boundary: one-sided second difference
            prev2_mat = f"M{m_num - 2}"
            if prev2_mat in eps.columns:
                feat["eps_local_curvature"] = (
                    eps[mat] - 2.0 * eps[prev_mat] + eps[prev2_mat]
                )
            else:
                feat["eps_local_curvature"] = np.nan

        else:
            feat["eps_local_curvature"] = np.nan

        feat["maturity"] = mat
        feat.index.name = "date"
        parts.append(feat.reset_index())

    df = (
        pd.concat(parts, ignore_index=True)
        .set_index(["date", "maturity"])
        .sort_index()
    )

    return df


