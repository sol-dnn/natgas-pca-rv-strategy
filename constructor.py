"""
PortfolioConstructor: converts ML signal (y_pred) into factor-neutral RV weights.

Signal date t:
  - rank maturities by y_pred_t
  - go long top_k (predicted cheapest) / short bottom_k (predicted richest)
  - normalize to gross_exposure
  - enforce dollar neutrality (sum w = 0)
  - project out PCA factor exposures so B.T @ w ≈ 0
  - clip max_abs_weight per position

Factor neutralization via orthogonal projection (López de Prado, 2018):
    w_neut = w - B @ inv(B.T @ B + ridge * I) @ (B.T @ w)
where B is the (n_assets × n_factors) loading matrix (columns = factor eigenvectors).

Tu utilises le signal tous les jours.
Tu tiens chaque signal 5 jours.
Le portefeuille réel est la moyenne des 5 derniers signaux.
Tu trades à t+1 (entrée au close t+1) grâce au shift(2) dans le backtester.

"""

import numpy as np
import pandas as pd
from typing import List, Optional


class PortfolioConstructor:
    """
    Construct long/short RV weights from cross-sectional ML signal.

    Parameters
    ----------
    top_k, bottom_k
        Number of maturities to go long / short each day.
    gross_exposure
        Target sum of |weights| before factor neutralization (default 1.0).
    max_abs_weight
        Per-position cap applied after factor neutralization (default 0.25).
    exclude_maturities
        Maturities excluded from portfolio (default ["M1", "M12"] — no alpha on
        front; M12 dropped due to unadjustable roll contamination).
    neutralize_factors
        Whether to project out PCA factor exposures.
    n_neutral_factors
        Number of leading PCA factors to neutralize (default 3).
    factor_neutrality_method
        "projection" (default) — closed-form orthogonal projection.
    ridge
        Ridge regularization for the factor neutralization inverse (default 1e-6).
    turnover_smoothing
        EWM decay on the raw signal before ranking, applied across time.
        0.0 = no smoothing (default).
    """

    def __init__(
        self,
        top_k: int = 4,
        bottom_k: int = 4,
        gross_exposure: float = 1.0,
        max_abs_weight: float = 0.25,
        exclude_maturities: Optional[List[str]] = None,
        neutralize_factors: bool = True,
        n_neutral_factors: int = 3,
        factor_neutrality_method: str = "projection",
        ridge: float = 1e-6,
        turnover_smoothing: float = 0.0,
    ):
        self.top_k = top_k
        self.bottom_k = bottom_k
        self.gross_exposure = gross_exposure
        self.max_abs_weight = max_abs_weight
        self.exclude_maturities = exclude_maturities if exclude_maturities is not None else ["M1", "M12"]
        self.neutralize_factors = neutralize_factors
        self.n_neutral_factors = n_neutral_factors
        self.factor_neutrality_method = factor_neutrality_method
        self.ridge = ridge
        self.turnover_smoothing = turnover_smoothing

    # ── Public API ─────────────────────────────────────────────────────────────

    def construct_weights_panel(
        self,
        predictions: pd.DataFrame,
        loadings: Optional[pd.DataFrame] = None,
    ) -> pd.DataFrame:
        """
        Build weights for every signal date in predictions.

        Parameters
        ----------
        predictions
            DataFrame with MultiIndex (date, maturity) and column 'y_pred'.
        loadings
            DataFrame with MultiIndex (date, maturity) and columns
            ['loading_f1', 'loading_f2', ..., 'loading_f{n}'].
            Required when neutralize_factors=True.

        Returns
        -------
        DataFrame with MultiIndex (date, maturity), column 'weight'.
        """
        pred_wide = predictions['y_pred'].unstack(level='maturity')

        if self.turnover_smoothing > 0.0:
            pred_wide = pred_wide.ewm(alpha=self.turnover_smoothing).mean()

        # Loading matrix: (date, maturity, factor) → wide (maturity × factors) per date
        if self.neutralize_factors:
            if loadings is None:
                raise ValueError("loadings must be provided when neutralize_factors=True")
            factor_cols = [c for c in loadings.columns if c.startswith('loading_f')]
            factor_cols = sorted(factor_cols)[: self.n_neutral_factors]
            loadings_wide = loadings[factor_cols].unstack(level='maturity')  # (date, f × maturity)

        rows = []
        for date, signal_row in pred_wide.iterrows():
            signal = signal_row.dropna()
            # Exclude maturities
            signal = signal[~signal.index.isin(self.exclude_maturities)]
            if len(signal) < self.top_k + self.bottom_k:
                continue

            # Factor loadings for this date
            B = None
            if self.neutralize_factors:
                if date in loadings_wide.index:
                    B_raw = loadings_wide.loc[date].unstack(level=0)  # maturity × factors
                    B_raw = B_raw.reindex(signal.index).dropna(how='any')
                    signal = signal.reindex(B_raw.index).dropna()
                    B = B_raw.reindex(signal.index).values
                    if B.shape[0] < self.top_k + self.bottom_k:
                        continue

            w = self._construct_weights_for_date(signal, B)

            if w is None:
                # Return zero weights for this signal date to avoid accidental forward-fill
                for mat in signal.index:
                    rows.append({"date": date, "maturity": mat, "weight": 0.0})
                continue

            for mat, wt in w.items():
                rows.append({"date": date, "maturity": mat, "weight": wt})

        if not rows:
            return pd.DataFrame(columns=['date', 'maturity', 'weight']).set_index(['date', 'maturity'])

        df_w = pd.DataFrame(rows).set_index(['date', 'maturity'])
        return df_w

    def construct_weights_for_date(
        self,
        signal: pd.Series,
        B: Optional[np.ndarray] = None,
    ) -> Optional[pd.Series]:
        """Single-date weight construction (public wrapper for diagnostics)."""
        return self._construct_weights_for_date(signal, B)

    # ── Internal ───────────────────────────────────────────────────────────────

    def _net_neutralize(self, w: pd.Series) -> pd.Series:
        """Force sum_i w_i = 0."""
        return w - w.mean()


    def _normalize_gross(self, w: pd.Series) -> Optional[pd.Series]:
        """Force sum_i |w_i| = gross_exposure."""
        gross = w.abs().sum()
        if gross < 1e-10:
            return None
        return w * (self.gross_exposure / gross)


    def _construct_weights_for_date(
        self,
        signal: pd.Series,
        B: Optional[np.ndarray] = None,
    ) -> Optional[pd.Series]:
        """
        1. Rank signal
        2. Long top_k / short bottom_k
        3. Dollar neutral
        4. Gross normalize
        5. Factor neutralize
        6. Re-normalize
        7. Clip weights
        8. Re-neutralize / re-normalize
        """
        if signal.isna().all() or len(signal) < self.top_k + self.bottom_k:
            return None

        signal = signal.dropna()

        # 1) Rank signal
        ranked = signal.rank(ascending=True)

        long_idx = ranked.nlargest(self.top_k).index
        short_idx = ranked.nsmallest(self.bottom_k).index

        # 2) Raw equal-weight long/short
        w = pd.Series(0.0, index=signal.index)
        w.loc[long_idx] = 1.0 / self.top_k
        w.loc[short_idx] = -1.0 / self.bottom_k

        # 3) Dollar neutrality
        w = self._net_neutralize(w)

        # 4) Gross normalization
        w = self._normalize_gross(w)
        if w is None:
            return None

        # 5) Factor neutrality + dollar neutrality
        if self.neutralize_factors and B is not None:
            w_vals = self._project_out_factors(w.values.astype(float), B)
            w = pd.Series(w_vals, index=signal.index)
        else:
            w = self._net_neutralize(w)

        # 6) Normalize again after projection
        w = self._normalize_gross(w)
        if w is None:
            return None

        # 7) Clip max position
        w = w.clip(-self.max_abs_weight, self.max_abs_weight)

        # 8) Re-apply dollar/factor neutrality after clipping
        if self.neutralize_factors and B is not None:
            w_vals = self._project_out_factors(w.values.astype(float), B)
            w = pd.Series(w_vals, index=signal.index)
        else:
            w = self._net_neutralize(w)

        # 9) Final gross normalization
        w = self._normalize_gross(w)
        if w is None:
            return None

        # 10) Final safety: if projection caused a huge weight, reject date
        if w.abs().max() > self.max_abs_weight * 1.25:
            return None

        return w

    def _project_out_factors(self, w: np.ndarray, B: np.ndarray) -> np.ndarray:
        """
        Orthogonal projection removing PCA factor exposures AND enforcing dollar
        neutrality simultaneously.

        Augment B with a normalized ones column so the single projection step:
            w_neut = w - B_aug @ inv(B_aug.T @ B_aug + ridge * I) @ (B_aug.T @ w)
        ensures both B.T @ w_neut ≈ 0 and 1.T @ w_neut ≈ 0.
        """
        n = len(w)
        ones_col = np.ones((n, 1)) / np.sqrt(n)  # normalized so ridge has consistent scale
        B_aug = np.hstack([B, ones_col])           # (n × k+1)
        k_aug = B_aug.shape[1]
        BtB = B_aug.T @ B_aug + self.ridge * np.eye(k_aug)
        Btw = B_aug.T @ w
        try:
            projection = B_aug @ np.linalg.solve(BtB, Btw)
        except np.linalg.LinAlgError:
            return w
        return w - projection
