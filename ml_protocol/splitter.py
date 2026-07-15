"""Walk-forward time-series splitter for financial ML protocols.

═══════════════════════════════════════════════════════════════════════════════
TRAINING PROTOCOL — THREE PHASES
═══════════════════════════════════════════════════════════════════════════════

Phase 0 — Burn-in  (e.g. Apr 2022 → Dec 2022)
───────────────────────────────────────────────
No predictions are made. The period exists only to accumulate enough history
for the first rolling training window (min_train_days = rolling_window = 252).

Phase 1 — Stage 1: OOF model-selection period  (e.g. Jan 2023 → Jun 2025)
───────────────────────────────────────────────────────────────────────────
Walk-forward out-of-fold predictions are collected day by day.
All model variants in the zoo run simultaneously.
At each refit step (every refit_every=21 trading days):
  - the model is fitted on the rolling window [t-252, t-2]
  - day t-1 is skipped (purge_days=1 — not in train, not in test)
  - the fitted model predicts days t, t+1, ..., t+20 unchanged
After Stage 1 is complete, use comparison_table() on the concatenated
OOF predictions to select:
  - the best model class and hyperparameters
  - the best feature set
  - the best decision threshold
  - the best portfolio rule (long-only, long-short, etc.)

Phase 2 — Stage 2: final untouched evaluation  (e.g. Jul 2025 → end)
──────────────────────────────────────────────────────────────────────
Run walk_forward() again with oos_start_date = stage2_start, passing ONLY
the winning model. No further tuning of any kind. This is the only number
that can be reported as an honest OOS performance in a paper or thesis.

Usage pattern
─────────────
    # Stage 1 — run all variants, pick winner
    cfg_s1 = SplitConfig(oos_start_date=pd.Timestamp("2023-01-02"), ...)
    result_s1 = walk_forward(data_s1, features, target, full_zoo, cfg_s1)
    table = comparison_table(result_s1.predictions, result_s1.split_log)
    # → inspect table, choose best_model_id

    # Stage 2 — final honest evaluation
    cfg_s2 = SplitConfig(oos_start_date=pd.Timestamp("2025-07-01"), ...)
    result_s2 = walk_forward(data, features, target,
                             {best_model_id: zoo[best_model_id]}, cfg_s2)

Splitter behaviour
──────────────────
Operates exclusively on unique trading dates. Never splits within a day:
features are observed at 10 AM, target is realised at 4 PM on the same day.
The purge gap (purge_days) is expressed in trading days and is applied both
in the main walk-forward loop (between train end and test date) and in the
nested_cv_splits helper (unused when all ModelSpecs have param_grid=None).
═══════════════════════════════════════════════════════════════════════════════
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator, Literal

import numpy as np
import pandas as pd


@dataclass
class SplitConfig:
    """Configuration for walk-forward time-series splitting."""

    min_train_days: int = 252
    """Burn-in period: minimum trading days before the first OOS prediction."""

    window_type: Literal["expanding", "rolling"] = "rolling"
    """
    expanding: training window grows monotonically from the first date.
    rolling  : training window is a fixed-length window ending the day
               before the test date.
    """

    rolling_window: int = 252
    """Size of the rolling training window (trading days). Unused for expanding."""

    purge_days: int = 1
    """
    Trading days removed at the boundary between inner train and val in nested CV.

    Rationale: lagged features such as pnl_mean5_l1 (mean of last 5 days' PNL)
    carry information from the cv_train tail into the first cv_val observations.
    Discarding purge_days observations on both sides of the split boundary
    severs this dependence.

    purge_days: number of trading days skipped between the end of the training window
    and the prediction date. In the main walk-forward split, this acts as an embargo.
    """

    val_fraction: float = 0.26
    """Fraction of the training window held out for validation in nested CV."""

    refit_every: int = 21
    """
    Refit the model every N OOS trading days (default: ~monthly).

    The first OOS step always refits. Subsequent refits occur at steps
    refit_every, 2*refit_every, ... measured from the first OOS date.

    Relationship with rolling_window: these are independent. rolling_window
    controls how much data goes into training; refit_every controls how
    often the model is re-estimated. With rolling_window=252 and
    refit_every=21, each consecutive refit shifts the training window by
    21 days (8.3% turnover in training data).
    """

    oos_start_date: pd.Timestamp | None = None
    """
    First test date (= first day of Stage 1 or Stage 2).

    Stage 1: set to the first day of your model-selection window.
    Stage 2: set to the first day of your final evaluation window.
    If None, OOS starts after min_train_days + purge_days are satisfied.
    Always set this explicitly for reproducible IS/OOS boundaries.
    """

    oos_end_date: pd.Timestamp | None = None
    """
    Last test date (inclusive). Predictions stop after this date.

    Stage 1: set to the last day of your model-selection window (e.g. day
    before stage2_start). Prevents accidentally including Stage 2 dates in
    the comparison_table() and contaminating model selection.
    Stage 2: leave None (predict through end of dataset).
    """

    verbose: bool = False
    """
    Print a summary header and one line per refit step when iterating split().
    Also prints the cv_train/cv_val boundary when nested_cv_splits() is called.
    Safe to leave False in production; useful for sanity-checking a new dataset.
    """


class TimeSeriesSplitter:
    """
    Walk-forward splitter that operates on sorted, unique trading dates.

    Implements the three-phase protocol documented in this module's docstring.
    Call split() to drive the outer walk-forward loop (Phases 1 & 2).
    Call nested_cv_splits() only if a ModelSpec has a param_grid (unused by
    default — hyperparameter variants are explicit zoo entries instead).

    Guarantees
    ----------
    1. Date-level boundaries: the training set is always a contiguous block
       of complete trading days ending strictly before the test date.
    2. Purge gap: purge_days trading days are skipped between the last
       training day and the test date (applied in both split() and
       nested_cv_splits()).
    3. Deterministic refit schedule: first OOS step always refits; subsequent
       refits fire every refit_every steps from oos_start.
    4. Rolling window: with window_type="rolling", the training window always
       contains exactly rolling_window days — it slides forward, never grows.
    """

    def __init__(self, config: SplitConfig) -> None:
        self.config = config

    def split(
        self,
        dates: np.ndarray,
    ) -> Iterator[tuple[np.ndarray, np.ndarray, bool]]:
        """
        Iterate over walk-forward OOS steps.

        Parameters
        ----------
        dates
            Sorted 1-D array of unique trading dates. Shape (D,).

        Yields
        ------
        train_iloc : int array — positions into *dates* for the training window
        test_iloc  : int array of length 1 — position of the test date
        is_refit   : bool — True when the model must be re-trained this step
        """
        cfg = self.config
        dates = np.asarray(dates)
        D = len(dates)
        purge = max(0, cfg.purge_days)

        if D == 0:
            raise ValueError("dates cannot be empty.")
        if len(np.unique(dates)) != D:
            raise ValueError(
                "dates must contain unique trading dates. "
                "Pass dates_unique (one entry per day), not row-level data."
            )
        if not np.all(dates[:-1] <= dates[1:]):
            raise ValueError("dates must be sorted in ascending order.")
        if cfg.window_type == "rolling" and cfg.rolling_window < cfg.min_train_days:
            raise ValueError(
                f"rolling_window ({cfg.rolling_window}) must be >= "
                f"min_train_days ({cfg.min_train_days})."
            )

        # First OOS index: need min_train_days of training data even after the purge gap
        oos_start = cfg.min_train_days + purge
        if cfg.oos_start_date is not None:
            oos_start = max(
                cfg.min_train_days + purge,
                int(np.searchsorted(dates, np.datetime64(cfg.oos_start_date), side="left")),
            )

        if oos_start >= D:
            raise ValueError(
                f"No OOS dates available: oos_start index {oos_start} >= D={D}. "
                "Extend the dataset or move oos_start_date earlier."
            )

        # Hard ceiling on OOS predictions — enforces Stage 1 / Stage 2 boundary
        if cfg.oos_end_date is not None:
            oos_end = int(np.searchsorted(
                dates, np.datetime64(cfg.oos_end_date), side="right"
            ))
            D = min(D, oos_end)
            if oos_start >= D:
                raise ValueError(
                    f"No OOS dates available after applying oos_end_date: "
                    f"oos_start={oos_start}, oos_end_index={D}."
                )

        if cfg.verbose:
            _W = 72
            _win = (f"{cfg.rolling_window} days (rolling, constant)"
                    if cfg.window_type == "rolling" else "expanding")
            print("═" * _W)
            print("  TimeSeriesSplitter")
            print(f"  Dataset     : {pd.Timestamp(dates[0]).date()} → "
                  f"{pd.Timestamp(dates[-1]).date()}  ({D} trading days)")
            print(f"  OOS start   : {pd.Timestamp(dates[oos_start]).date()}  "
                  f"(index {oos_start}, burn-in = {oos_start} days)")
            print(f"  Train window: {_win}")
            print(f"  Purge gap   : {purge} trading day(s)")
            print(f"  Refit every : {cfg.refit_every} trading days")
            print("═" * _W)
            refit_count = 0

        for i in range(oos_start, D):
            # train_end: last exclusive index of the training window.
            # purge_days dates immediately before the test date are skipped.
            train_end = i - purge

            if cfg.window_type == "expanding":
                train_start = 0
            else:  # rolling: exactly rolling_window training days
                train_start = max(0, train_end - cfg.rolling_window)

            train_iloc = np.arange(train_start, train_end)
            test_iloc = np.array([i])

            steps_since_oos_start = i - oos_start
            is_refit = (steps_since_oos_start % max(1, cfg.refit_every)) == 0

            if cfg.verbose and is_refit:
                refit_count += 1
                t0 = pd.Timestamp(dates[train_start]).date()
                t1 = pd.Timestamp(dates[train_end - 1]).date()
                purge_str = (pd.Timestamp(dates[i - purge]).date()
                             if purge else "—")
                test_str = pd.Timestamp(dates[i]).date()
                print(f"  Refit {refit_count:>3}  │  "
                      f"train {t0} → {t1}  ({len(train_iloc)}d)  │  "
                      f"purge {purge_str}  │  test from {test_str}")

            yield train_iloc, test_iloc, is_refit

        if cfg.verbose:
            oos_days = D - oos_start
            print("─" * _W)
            print(f"  {refit_count} refits total  │  {oos_days} OOS days")
            print("═" * _W)


    '''def nested_cv_splits(
        self,
        train_dates: np.ndarray,
    ) -> list[tuple[np.ndarray, np.ndarray]]:
        """
        Single chronological train/val split in trading-date space.

        Operates entirely in day units — purge_days and val_fraction are
        both expressed in trading days, never in rows.  The caller maps the
        returned date arrays back to row indices via np.isin().

        Layout (days):
        [cv_train] [purge_days gap] [cv_val  (val_fraction × D days)]

        Parameters
        ----------
        train_dates
            Sorted array of unique trading dates in the training window.

        Returns
        -------
        List with a single (cv_train_dates, cv_val_dates) tuple.
        Returns [] if the window is too small for a useful split.
        """
        cfg = self.config
        dates = np.unique(train_dates)
        D = len(dates)

        val_days = max(1, int(D * cfg.val_fraction))
        gap = cfg.purge_days

        cv_train_end = D - val_days - gap
        if cv_train_end < max(20, val_days):
            if cfg.verbose:
                print(f"  nested_cv_splits : window too small ({D}d) — skipped")
            return []

        cv_train_dates = dates[:cv_train_end]
        cv_val_dates = dates[D - val_days:]

        if cfg.verbose:
            print(f"  nested_cv_splits : "
                  f"cv_train {len(cv_train_dates)}d "
                  f"({pd.Timestamp(cv_train_dates[0]).date()} → "
                  f"{pd.Timestamp(cv_train_dates[-1]).date()})  │  "
                  f"purge {gap}d  │  "
                  f"cv_val {len(cv_val_dates)}d "
                  f"({pd.Timestamp(cv_val_dates[0]).date()} → "
                  f"{pd.Timestamp(cv_val_dates[-1]).date()})")

        return [(cv_train_dates, cv_val_dates)]'''