"""
Fundamental features for the Henry Hub RV strategy.

Weather demand shocks and EIA storage surprises, adapted from
Chen, Hartley & Lan (2023) to daily data.

All features at date t use only data available at the close of t.
They can be used as signals for a trade entered at t+1.

Pipeline (called via build_fundamental_features):
  1. prepare_base              — HDD/CDD levels, squares, increments
  2. add_weather_rolling_features — rolling sums, means, variances
  3. add_historical_normals    — climate normals (no-lookahead, rolling)
  4. add_expected_unexpected   — E/U/Z weather via walk-forward ARMAX
  5. add_forecast_weather_shock — WHDD/WCDD: 5-day-ahead expected deviation
  6. build_storage_announcement_table — weekly EIA release dates + ΔStorage
  7. add_storage_window_weather_features — weather aggregated within each week
  8. add_expected_unexpected_storage   — E/U/Z ΔStorage (rolling Ridge)
  9. merge_storage_features_back_to_daily — daily panel with timing dummies

All outputs are date-indexed → Family B (broadcast to all maturities in builder).



Storage update day is detected from the forward-filled daily storage series.
In this dataset, updates are dated on Friday, so D_storage_0 means the first
trading day on which the new storage value is available in the dataset.
This is conservative relative to the official EIA Thursday release and avoids lookahead.
"""


import numpy as np
import pandas as pd
from typing import Optional
from sklearn.linear_model import Ridge


# ─────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────────────────────────────────────

def _fourier(dates: pd.DatetimeIndex, order: int = 2) -> pd.DataFrame:
    """Fourier seasonal terms: sin_k, cos_k for k=1..order."""
    doy = np.array([d.timetuple().tm_yday for d in dates], dtype=float)
    cols = {}
    for k in range(1, order + 1):
        cols[f'sin{k}'] = np.sin(2 * np.pi * k * doy / 365.25)
        cols[f'cos{k}'] = np.cos(2 * np.pi * k * doy / 365.25)
    return pd.DataFrame(cols, index=dates)


def _historical_normal(
    dates: pd.DatetimeIndex,
    values: np.ndarray,
    lookback_years: int = 4,
    day_window: int = 3,
    min_obs: int = 10,
) -> np.ndarray:
    """
    For each date t compute the historical normal:
        mean( X_s  for s < t, year(s) in [year(t)-lookback_years, year(t)-1],
              |doy(s) - doy(t)| <= day_window  modulo 365 )

    Strictly no-lookahead: uses only years strictly before year(t).
    """
    dates  = pd.DatetimeIndex(dates)
    vals   = np.asarray(values, dtype=float)
    years  = np.array([d.year for d in dates])
    doys   = np.array([d.timetuple().tm_yday for d in dates])
    n      = len(dates)
    result = np.full(n, np.nan)

    for i in range(n):
        yr        = years[i]
        doy       = doys[i]
        yr_mask   = (years >= yr - lookback_years) & (years <= yr - 1)
        doy_diff  = np.abs(doys - doy)
        doy_diff  = np.minimum(doy_diff, 365 - doy_diff)   # wrap at year boundary
        doy_mask  = doy_diff <= day_window
        mask      = yr_mask & doy_mask & np.isfinite(vals)
        if mask.sum() >= min_obs:
            result[i] = vals[mask].mean()

    return result


def _make_armax_X(
    df: pd.DataFrame,
    col: str,
    normal_col: Optional[str],
    ar_lags: list,
    fourier_order: int = 2,
) -> pd.DataFrame:
    """
    Build the ARMAX feature matrix for column `col`:
        const + Fourier(order) + normal (optional) + AR lags
    Returns a DataFrame with the same index as df.
    """
    X = _fourier(df.index, order=fourier_order)
    X['const'] = 1.0
    if normal_col and normal_col in df.columns:
        X['normal'] = df[normal_col].values
    for lag in sorted(ar_lags):
        X[f'ar_{lag}'] = df[col].shift(lag).values
    return X


# ─────────────────────────────────────────────────────────────────────────────
# Step 1 — Base features
# ─────────────────────────────────────────────────────────────────────────────

def prepare_base(df: pd.DataFrame, lag_weather: bool = False) -> pd.DataFrame:
    """
    Compute base weather features.

    Parameters
    ----------
    df          : DatetimeIndex DataFrame with columns 'HDD', 'CDD', 'Storage'
    lag_weather : if True, shift HDD/CDD by 1 day before computing
                  (use when HDD/CDD is only known at close of t+1)
    """
    df = df.copy().sort_index()
    assert isinstance(df.index, pd.DatetimeIndex), "Index must be DatetimeIndex"

    hdd = df['HDD'].shift(1 if lag_weather else 0)
    cdd = df['CDD'].shift(1 if lag_weather else 0)

    df['HDD']      = hdd
    df['CDD']      = cdd
    df['HDD2']     = hdd ** 2
    df['CDD2']     = cdd ** 2
    delta_hdd = hdd.diff()
    delta_cdd = cdd.diff()
    df['HDD_inc'] = delta_hdd.clip(lower=0)
    df['HDD_dec'] = (-delta_hdd).clip(lower=0)
    df['CDD_inc'] = delta_cdd.clip(lower=0)
    df['CDD_dec'] = (-delta_cdd).clip(lower=0)

    return df


# ─────────────────────────────────────────────────────────────────────────────
# Step 2 — Rolling weather aggregates
# ─────────────────────────────────────────────────────────────────────────────

def add_weather_rolling_features(df: pd.DataFrame) -> pd.DataFrame:
    """Rolling sums, means, variances for HDD and CDD (5-day and 10-day windows)."""
    df = df.copy()
    for c in ['HDD', 'CDD']:
        base = df[c]
        df[f'{c}_sum_5d']  = base.rolling(5,  min_periods=3).sum()
        df[f'{c}_mean_5d'] = base.rolling(5,  min_periods=3).mean()
        df[f'{c}_var_5d']  = base.rolling(5,  min_periods=2).var()
        df[f'{c}_sum_10d'] = base.rolling(10, min_periods=5).sum()
        df[f'{c}_var_10d'] = base.rolling(10, min_periods=5).var()
    return df


# ─────────────────────────────────────────────────────────────────────────────
# Step 3 — Historical climate normals
# ─────────────────────────────────────────────────────────────────────────────

def add_historical_normals(
    df: pd.DataFrame,
    cols: list,
    lookback_years: int = 4,
    day_window: int = 3,
    min_obs: int = 10,
    with_anom: bool = True,
) -> pd.DataFrame:
    """
    For each column in cols, add:
      {col}_normal_{k}y_pm{w}d   — historical mean (rolling, no-lookahead)
      {col}_anom_normal          — X_t - normal_t  (only if with_anom=True)
    """
    df  = df.copy()
    sfx = f'normal_{lookback_years}y_pm{day_window}d'

    for col in cols:
        if col not in df.columns:
            continue
        norm_col = f'{col}_{sfx}'
        df[norm_col] = _historical_normal(
            df.index, df[col].values, lookback_years, day_window, min_obs
        )
        if with_anom:
            df[f'{col}_anom_normal'] = df[col] - df[norm_col]

    return df


# ─────────────────────────────────────────────────────────────────────────────
# Step 4 — Expected / unexpected weather (walk-forward ARMAX)
# ─────────────────────────────────────────────────────────────────────────────

def add_expected_unexpected_weather_features(
    df: pd.DataFrame,
    weather_cols: list,
    normal_suffix: str = 'normal_4y_pm3d',
    ar_lags: list = None,
    train_min_obs: int = 500,
    refit_frequency: int = 20,
    rolling_sigma_window: int = 60,
    rolling_train_window: int = 756,
    fourier_order: int = 2,
) -> pd.DataFrame:
    """
    For each col in weather_cols, fit rolling-window Ridge regression:

        X_t = μ + β X_normal_t + Σ φ_p X_{t-p} + Σ γ_k sin/cos(k·doy/365) + ε_t

    Refit every `refit_frequency` steps. Produces:
        E_{col}_t  — predicted value (pre-observation forecast)
        U_{col}_t  — surprise = X_t - E_{col}_t
        Z_{col}_t  — U_{col}_t / σ_t   (σ = rolling std of past residuals)
    """
    if ar_lags is None:
        ar_lags = [1, 2, 3, 5, 10]
    df = df.copy()
    n  = len(df)

    for col in weather_cols:
        if col not in df.columns:
            continue

        normal_col = f'{col}_{normal_suffix}'
        X          = _make_armax_X(df, col, normal_col, ar_lags, fourier_order)
        X_arr      = X.values.astype(float)
        y          = df[col].values.astype(float)

        E_arr    = np.full(n, np.nan)
        U_arr    = np.full(n, np.nan)
        res_arr  = np.full(n, np.nan)
        model    = None
        last_fit = -(refit_frequency + 1)

        for t in range(train_min_obs, n):
            if t - last_fit >= refit_frequency:
                start = max(0, t - rolling_train_window)

                X_train = X_arr[start:t]
                y_train = y[start:t]

                ok = np.isfinite(X_train).all(1) & np.isfinite(y_train)

                if ok.sum() >= train_min_obs:
                    model = Ridge(alpha=1.0).fit(X_train[ok], y_train[ok])
                    last_fit = t

            if model is None:
                continue

            if np.isfinite(X_arr[t]).all() and np.isfinite(y[t]):
                pred = float(model.predict(X_arr[t:t+1])[0])
                E_arr[t] = pred
                U_arr[t] = y[t] - pred
                res_arr[t] = U_arr[t]


        sigma = (
            pd.Series(res_arr, index=df.index)
            .shift(1)
            .rolling(rolling_sigma_window, min_periods=rolling_sigma_window // 2)
            .std()
            .replace(0, np.nan)
        )

        df[f'E_{col}'] = E_arr
        df[f'U_{col}'] = U_arr
        df[f'Z_{col}'] = U_arr / sigma.values

    return df


# ─────────────────────────────────────────────────────────────────────────────
# Step 5 — Forecast weather shock at horizon H
# ─────────────────────────────────────────────────────────────────────────────

def add_forecast_weather_shock_features(
    df: pd.DataFrame,
    cols: list = None,
    horizon: int = 5,
    normal_suffix: str = 'normal_4y_pm3d',
    ar_lags: list = None,
    train_min_obs: int = 500,
    refit_frequency: int = 20,
    rolling_train_window: int = 756,
    fourier_order: int = 2,
) -> pd.DataFrame:
    """
    At each date t, generate a recursive H-step-ahead forecast using an
    ARMAX/Ridge model fitted on a rolling window ending at t-1.

        W{col}_{H}d_t = (1/H) Σ_{m=1}^{H} [ Ê_t[X_{t+m}] − normal(X_{t+m}) ]

    No lookahead:
    - model coefficients are fitted using observations from max(0, t-rolling_train_window) to t-1
    - recursive forecasts use observed X_t as the latest known value
    - future HDD/CDD realized values are never used
    """
    if cols is None:
        cols = ['HDD', 'CDD']
    if ar_lags is None:
        ar_lags = [1, 2, 3, 5, 10]

    df = df.copy()
    n = len(df)
    max_lag = max(ar_lags)
    fourier_all = _fourier(df.index, order=fourier_order)

    for col in cols:
        if col not in df.columns:
            continue

        normal_col = f'{col}_{normal_suffix}'
        X_full = _make_armax_X(df, col, normal_col, ar_lags, fourier_order)
        feat_cols = list(X_full.columns)
        X_arr = X_full.values.astype(float)
        y = df[col].values.astype(float)
        norms = df[normal_col].values.astype(float) if normal_col in df.columns else np.full(n, np.nan)

        model = None
        last_fit = -(refit_frequency + 1)
        W_arr = np.full(n, np.nan)

        for t in range(train_min_obs, n - horizon):
            if t - last_fit >= refit_frequency:
                start = max(0, t - rolling_train_window)

                X_train = X_arr[start:t]
                y_train = y[start:t]

                ok = np.isfinite(X_train).all(1) & np.isfinite(y_train)

                if ok.sum() >= train_min_obs:
                    model = Ridge(alpha=1.0).fit(X_train[ok], y_train[ok])
                    last_fit = t

            if model is None:
                continue

            # Known history up to t. For future dates, we add recursive predictions.
            hist = {i: float(y[i]) for i in range(max(0, t - max_lag), t + 1)}

            shock_sum = 0.0
            all_valid = True

            for m in range(1, horizon + 1):
                fi = t + m

                if fi >= n:
                    all_valid = False
                    break

                row = {}

                for fc in feat_cols:
                    if fc.startswith('sin') or fc.startswith('cos'):
                        row[fc] = float(fourier_all.iloc[fi][fc])

                    elif fc == 'const':
                        row[fc] = 1.0

                    elif fc == 'normal':
                        row[fc] = float(norms[fi])

                    elif fc.startswith('ar_'):
                        lag = int(fc.split('_')[1])
                        row[fc] = hist.get(fi - lag, np.nan)

                row_arr = np.array([row.get(c, np.nan) for c in feat_cols], dtype=float)

                if not np.isfinite(row_arr).all() or not np.isfinite(norms[fi]):
                    all_valid = False
                    break

                pred_m = float(model.predict(row_arr.reshape(1, -1))[0])
                hist[fi] = pred_m
                shock_sum += pred_m - norms[fi]

            if all_valid:
                W_arr[t] = shock_sum / horizon

        h = f'{horizon}d'
        df[f'W{col}_{h}'] = W_arr
        df[f'W{col}2_{h}'] = W_arr ** 2

    return df

# ─────────────────────────────────────────────────────────────────────────────
# Step 6 — Storage announcement table
# ─────────────────────────────────────────────────────────────────────────────

def build_storage_announcement_table(df: pd.DataFrame) -> pd.DataFrame:
    """
    Extract EIA weekly storage releases from the forward-filled daily series.

    Each announcement a has:
      storage_bcf          — level at announcement
      delta_storage_bcf    — change from prior announcement (bcf)
      ann_prev             — date of prior announcement
      window_days          — calendar days between announcements
    """
    storage   = df['Storage'].sort_index()
    delta     = storage.diff()
    ann_dates = delta[delta.notna() & (delta != 0)].index

    ann = pd.DataFrame({
        'storage_bcf':       storage[ann_dates],
        'delta_storage_bcf': delta[ann_dates],
    })
    ann.index.name  = 'ann_date'
    ann['ann_prev']     = ann.index.to_series().shift(1).values
    ann['window_days']  = ann.index.to_series().diff().dt.days.values

    return ann.dropna(subset=['ann_prev'])


# ─────────────────────────────────────────────────────────────────────────────
# Step 7 — Weather aggregated over each storage window
# ─────────────────────────────────────────────────────────────────────────────

def add_storage_window_weather_features(
    df: pd.DataFrame,
    ann: pd.DataFrame,
) -> pd.DataFrame:
    """
    For each announcement a, aggregate weather over (a_prev, a]:
      {col}_storage_window       — sum of col over the window
      {col}_var_storage_window   — variance of col over the window
    """
    ann      = ann.copy()
    wx_cols  = [c for c in ['HDD', 'HDD2', 'CDD', 'CDD2'] if c in df.columns]

    agg_sum  = {c: [] for c in wx_cols}
    agg_var  = {c: [] for c in wx_cols}

    for ann_date, row in ann.iterrows():
        window = df.loc[(df.index > row['ann_prev']) & (df.index <= ann_date)]
        for c in wx_cols:
            agg_sum[c].append(window[c].sum() if c in window.columns else np.nan)
            agg_var[c].append(window[c].var() if (c in window.columns and len(window) > 1) else np.nan)

    for c in wx_cols:
        ann[f'{c}_storage_window']     = agg_sum[c]
        ann[f'{c}_var_storage_window'] = agg_var[c]

    return ann


# ─────────────────────────────────────────────────────────────────────────────
# Step 8 — Expected / unexpected storage change (walk-forward Ridge)
# ─────────────────────────────────────────────────────────────────────────────

def add_expected_unexpected_storage_features(
    ann: pd.DataFrame,
    lookback_years: int = 5,
    day_window: int = 7,
    normal_min_obs: int = 5,
    train_min_obs: int = 104,
    refit_frequency: int = 4,
    rolling_train_window: int = 156,
    rolling_sigma_window: int = 26,
) -> pd.DataFrame:
    """
    Rolling-window Ridge regression on announcement-frequency data.

        ΔStorage_a = μ
                   + β1 StorageChange_normal_a
                   + Σ β_k weather_window_features_a
                   + Σ φ_l ΔStorage_{a-l}
                   + Σ month_m dummies
                   + ε_a

    At announcement a, the model is fitted only on the previous
    rolling_train_window announcements, ending at a-1.

    Produces:
        E_DeltaStorage = model forecast
        U_DeltaStorage = actual - expected
        Z_DeltaStorage = standardized surprise
    """
    ann = ann.copy()
    n = len(ann)

    # Seasonal normal of ΔStorage, no lookahead
    ann["StorageChange_normal"] = _historical_normal(
        ann.index,
        ann["delta_storage_bcf"].values,
        lookback_years=lookback_years,
        day_window=day_window,
        min_obs=normal_min_obs,
    )

    # AR lags, weekly
    for lag in [1, 2, 3, 4]:
        ann[f"delta_storage_ar_{lag}"] = ann["delta_storage_bcf"].shift(lag)

    # Month dummies
    for m in range(1, 13):
        ann[f"month_{m}"] = (ann.index.month == m).astype(float)

    wx_win_cols = [c for c in ann.columns if c.endswith("_storage_window")]

    feat_cols = (
        ["StorageChange_normal"]
        + wx_win_cols
        + [f"delta_storage_ar_{l}" for l in [1, 2, 3, 4]]
        + [f"month_{m}" for m in range(1, 13)]
    )
    feat_cols = [c for c in feat_cols if c in ann.columns]

    y = ann["delta_storage_bcf"].values.astype(float)
    X_arr = ann[feat_cols].values.astype(float)

    E_arr = np.full(n, np.nan)
    U_arr = np.full(n, np.nan)
    res_arr = np.full(n, np.nan)

    model = None
    last_fit = -(refit_frequency + 1)
    col_means = np.zeros(X_arr.shape[1])

    for t in range(train_min_obs, n):
        if t - last_fit >= refit_frequency:
            start = max(0, t - rolling_train_window)

            Xtr = X_arr[start:t].copy()
            ytr = y[start:t].copy()

            # Impute NaNs using training-window means only
            current_col_means = np.zeros(Xtr.shape[1])

            for j in range(Xtr.shape[1]):
                m = np.nanmean(Xtr[:, j])
                current_col_means[j] = m if np.isfinite(m) else 0.0
                Xtr[np.isnan(Xtr[:, j]), j] = current_col_means[j]

            ok = np.isfinite(Xtr).all(axis=1) & np.isfinite(ytr)

            if ok.sum() >= train_min_obs:
                model = Ridge(alpha=1.0).fit(Xtr[ok], ytr[ok])
                col_means = current_col_means
                last_fit = t

        if model is None:
            continue

        if np.isfinite(y[t]):
            row = X_arr[t:t+1].copy()

            nan_mask = np.isnan(row[0])
            row[0, nan_mask] = col_means[nan_mask]

            if np.isfinite(row).all():
                pred = float(model.predict(row)[0])
                E_arr[t] = pred
                U_arr[t] = y[t] - pred
                res_arr[t] = U_arr[t]

    sigma = (
        pd.Series(res_arr, index=ann.index)
        .shift(1)
        .rolling(rolling_sigma_window, min_periods=rolling_sigma_window // 2)
        .std()
        .replace(0, np.nan)
    )

    ann["E_DeltaStorage"] = E_arr
    ann["U_DeltaStorage"] = U_arr
    ann["Z_DeltaStorage"] = U_arr / sigma.values

    return ann

# ─────────────────────────────────────────────────────────────────────────────
# Step 9 — Merge announcement features back to the daily panel
# ─────────────────────────────────────────────────────────────────────────────

def merge_storage_features_back_to_daily(
    df: pd.DataFrame,
    ann: pd.DataFrame,
) -> pd.DataFrame:
    """
    Forward-fill storage announcement features into the daily panel.

    Adds:
      is_storage_announcement_day        — 1 on EIA release day
      days_since_storage_announcement    — business days since last release
      DeltaStorage_last, E/U/Z_DeltaStorage_last
      D_storage_0..3, D_storage_4plus    — timing dummies (0=release day)
      UStorage_x_D0..3, EStorage_x_D0..3 — surprise × timing interactions
    """
    df  = df.copy().sort_index()
    ann = ann.sort_index()

    # Business-day index for days_since computation
    bdates     = pd.bdate_range(df.index.min(), df.index.max())
    bday_num   = pd.Series(range(len(bdates)), index=bdates, dtype=float)

    ann_bday_ffill = (
        pd.Series({d: bday_num.get(d, np.nan) for d in ann.index}, dtype=float)
        .reindex(df.index)
        .ffill()
    )
    df_bday = pd.Series({d: bday_num.get(d, np.nan) for d in df.index}, dtype=float)

    df['is_storage_announcement_day']  = df.index.isin(ann.index).astype(int)
    df['days_since_storage_announcement'] = (df_bday - ann_bday_ffill).values

    # Forward-fill announcement-level features
    for src_col, dst_col in [
        ('delta_storage_bcf', 'DeltaStorage_last'),
        ('E_DeltaStorage',    'E_DeltaStorage_last'),
        ('U_DeltaStorage',    'U_DeltaStorage_last'),
        ('Z_DeltaStorage',    'Z_DeltaStorage_last'),
    ]:
        if src_col not in ann.columns:
            continue
        s = pd.Series(np.nan, index=df.index, dtype=float)
        shared = ann.index.intersection(df.index)
        s.loc[shared] = ann.loc[shared, src_col].values
        df[dst_col] = s.ffill()

    # Timing dummies
    ds = df['days_since_storage_announcement']
    for i in range(4):
        df[f'D_storage_{i}'] = (ds == i).astype(float)
    df['D_storage_4plus'] = (ds >= 4).astype(float)

    # Surprise × timing interactions
    u = df.get('U_DeltaStorage_last', pd.Series(np.nan, index=df.index))
    e = df.get('E_DeltaStorage_last', pd.Series(np.nan, index=df.index))
    for i in range(4):
        df[f'UStorage_x_D{i}'] = u.values * df[f'D_storage_{i}'].values
        df[f'EStorage_x_D{i}'] = e.values * df[f'D_storage_{i}'].values

    return df


# ─────────────────────────────────────────────────────────────────────────────
# Main builder
# ─────────────────────────────────────────────────────────────────────────────


def build_fundamental_features(
    df_weather: pd.DataFrame,
    df_storage: pd.DataFrame,
    lag_weather: bool = False,
    lookback_years: int = 4,
    day_window: int = 3,
    train_min_obs: int = 500,
    refit_frequency: int = 20,
    ar_lags: list = None,
    rolling_sigma_window: int = 60,
    weather_rolling_train_window: int = 756,
    storage_train_min_obs: int = 104,
    storage_rolling_train_window: int = 156,
    horizon: int = 5,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    
    """
    Full pipeline: weather + storage fundamental features.

    Parameters
    ----------
    df_weather  : DatetimeIndex DataFrame with 'HDD' and 'CDD' columns
    df_storage  : DatetimeIndex DataFrame with 'gas_storage_bcf' column
    lag_weather : shift HDD/CDD by 1 if only known next day

    Returns
    -------
    (df_daily, ann_table)
      df_daily   : daily feature panel indexed by date
      ann_table  : weekly announcement table (useful for diagnostics)
    """
    if ar_lags is None:
        ar_lags = [1, 2, 3, 5, 10]

    df = (
        df_weather[['HDD', 'CDD']].copy()
        .join(df_storage.rename(columns={'gas_storage_bcf': 'Storage'}), how='outer')
        .sort_index()
    )
    df.index = pd.to_datetime(df.index)
    df.index.name = 'date'

    print('[1/7] Base features...')
    df = prepare_base(df, lag_weather=lag_weather)

    print('[2/7] Rolling weather features...')
    df = add_weather_rolling_features(df)

    print('[3/7] Historical climate normals (rolling, no-lookahead)...')
    # HDD/CDD/HDD2/CDD2: normals + anomalies go into the feature panel
    df = add_historical_normals(df, ['HDD', 'CDD', 'HDD2', 'CDD2'],
                                lookback_years=lookback_years, day_window=day_window)

    suffix = f'normal_{lookback_years}y_pm{day_window}d'

    print('[4/7] Expected/unexpected weather (walk-forward ARMAX)...')
    exp_cols = ['HDD', 'CDD', 'HDD2', 'CDD2']

    df = add_expected_unexpected_weather_features(
        df,
        exp_cols,
        normal_suffix=suffix,
        ar_lags=ar_lags,
        train_min_obs=train_min_obs,
        refit_frequency=refit_frequency,
        rolling_sigma_window=rolling_sigma_window,
        rolling_train_window=weather_rolling_train_window,
    )


    print('[5/7] 5-day-ahead forecast weather shocks...')
    df = add_forecast_weather_shock_features(
        df,
        cols=['HDD', 'CDD'],
        horizon=horizon,
        normal_suffix=suffix,
        ar_lags=ar_lags,
        train_min_obs=train_min_obs,
        refit_frequency=refit_frequency,
        rolling_train_window=weather_rolling_train_window,
    )

    print('[6/7] Storage announcement table + window weather features...')
    ann = build_storage_announcement_table(df)
    ann = add_storage_window_weather_features(df, ann)
    ann = add_expected_unexpected_storage_features(
        ann,
        train_min_obs=storage_train_min_obs,
        rolling_train_window=storage_rolling_train_window,
    )

    print('[7/7] Merge storage features to daily panel...')
    df = merge_storage_features_back_to_daily(df, ann)

    print(f'Done — daily panel: {df.shape[0]:,} rows × {df.shape[1]} features')
    return df, ann


# ─────────────────────────────────────────────────────────────────────────────
# Family B column list (all date-level, broadcast to all maturities)
# ─────────────────────────────────────────────────────────────────────────────

FAMILY_B_COLS = [
    # Observed weather
    'HDD', 'CDD', 'HDD2', 'CDD2', 'HDD_inc', 'HDD_dec', 'CDD_inc', 'CDD_dec',
    # Rolling
    'HDD_sum_5d', 'CDD_sum_5d', 'HDD_mean_5d', 'CDD_mean_5d',
    'HDD_var_5d', 'CDD_var_5d', 'HDD_sum_10d', 'CDD_sum_10d', 'HDD_var_10d', 'CDD_var_10d',
    # Climate normals
    'HDD_normal_4y_pm3d', 'CDD_normal_4y_pm3d',
    'HDD2_normal_4y_pm3d', 'CDD2_normal_4y_pm3d',
    'HDD_anom_normal', 'CDD_anom_normal', 'HDD2_anom_normal', 'CDD2_anom_normal',

    # E/U/Z weather
    'E_HDD', 'U_HDD', 'Z_HDD',
    'E_CDD', 'U_CDD', 'Z_CDD',
    'E_HDD2', 'U_HDD2', 'Z_HDD2',
    'E_CDD2', 'U_CDD2', 'Z_CDD2',

    # Forecast weather shocks
    'WHDD_5d', 'WCDD_5d', 'WHDD2_5d', 'WCDD2_5d',

    # Storage
    'Storage', 'DeltaStorage_last',
    'is_storage_announcement_day', 'days_since_storage_announcement',
    'E_DeltaStorage_last', 'U_DeltaStorage_last', 'Z_DeltaStorage_last',
    'UStorage_x_D0', 'UStorage_x_D1', 'UStorage_x_D2', 'UStorage_x_D3',
    'EStorage_x_D0', 'EStorage_x_D1', 'EStorage_x_D2', 'EStorage_x_D3',
]
