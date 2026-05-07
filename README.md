# Statistical Factor Relative Value in Natural Gas Futures

**Oxford Alpha Fund — Team 1**  
*Project Manager: Solal Danan · Analyst: Shenghao Sun*

---

## Competition Result

**Winner — Oxford Alpha Fund Quant Pitch Competition (May 2026)**  
Placed 1st out of 4 teams. Judged by:
- **Dr. Eric Huang** — Portfolio Manager, Point72
- **Siobhan Cooke** — Quantitative Risk Manager, Citi

---

## Overview

A quantitative relative-value strategy on the NYMEX Henry Hub natural gas forward curve. The goal is **not** to forecast outright gas prices — it is to identify which maturities (M1–M12) are temporarily rich or cheap relative to the equilibrium curve, and whether those dislocations mean-revert conditional on storage and weather regimes.

## Alpha Hypothesis

Intra-curve dislocations become predictably mean-reverting under inventory stress, driven by the non-linear relationship between storage constraints and forward pricing. A two-stage rolling PCA isolates idiosyncratic residuals; ML predicts the 5-day forward residual return from storage, weather and curve-state features; the portfolio is dollar- and factor-neutral.

---

## Methodology (Summary)

### 1. Statistical Factor Model (Two-Stage PCA)

The forward curve is decomposed into three latent factors — **level**, **slope**, and **curvature** — using a rolling two-stage PCA. Each contract return is decomposed as:

```
r_t = B_t · f_t + ε_t
```

where `ε_t` is the idiosyncratic residual: the maturity-specific dislocation that forms the tradable signal.

### 2. Target Variable

The prediction target is the standardized 5-day cumulative residual return:

```
y_z(i,t) = Σ ε(i, t+k) / (√5 · σ_ε(i,t,63))   for k = 1..5
```

### 3. Feature Engineering

~70 features across 7 signal families: storage, weather, curve shape, residual momentum, calendar, macro, and storage×weather interaction terms.

### 4. ML Pipeline

XGBoost trained on rolling 3-year windows with purged walk-forward cross-validation (quarterly refits, 6-day purge gap). In-sample period 2013–2022 used for all model selection; out-of-sample 2022–2026 for final evaluation only.

### 5. Portfolio Construction

Long top 4 / short bottom 4 maturities by predicted signal. Dollar-neutral and neutralized to the first 3 PCA factors. 5 overlapping sleeves, 2 bps one-way transaction costs.

---

## Slides

The pitch deck is included in this repository: [`Stat Factor Model.pptx`](./Stat%20Factor%20Model.pptx)

---

## Full Implementation

The complete codebase (data pipeline, PCA model, feature engineering, ML training, backtester) is in a **private repository**.

Feel free to reach out if you'd like to discuss the strategy or the implementation: **solal.danan@gmail.com**

---

## References

1. G. Paleologo (2025). *The Elements of Quantitative Investing* — Ch. 7: Statistical Factor Model
2. Y. Chen (2023). Temperature, Storage and Natural Gas Futures Prices. *Journal of Futures Markets*, Wiley
3. M. Boons & M. Porras Prado (2019). Basis-Momentum. *Journal of Finance*
4. M. Lopez de Prado (2018). *Advances in Financial Machine Learning*. Wiley Finance Series
