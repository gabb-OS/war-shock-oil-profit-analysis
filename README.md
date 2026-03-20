# Margin Call: Multi-Model Analysis of Italian Fuel Distributors' Profits During Geopolitical Crises

> **Human Data Science Project — University of Bologna, A.Y. 2025–2026**  

---

## Overview

This project investigates whether Italian fuel distributors earned **anomalous profits** in the wake of three geopolitical shocks that affected oil supply chains. Using daily pump-price and wholesale-cost data, we compute gross margins and apply a **multi-model counterfactual approach** to detect price mark-ups beyond what raw input costs would justify.

The three events under study are:

| # | Event | Date |
|---|-------|------|
| 1 | Russian invasion of Ukraine | 24 February 2022 |
| 2 | Iran–Israel conflict | 13–24 June 2025 |
| 3 | Strait of Hormuz crisis | 28 February 2026 |

### Key findings

| Event | Fuel | Estimated extra profit | H₀ verdict |
|-------|------|------------------------|------------|
| Ukraine (Feb 2022) | Petrol | +26 ÷ +44 M€ | **Rejected** |
| Ukraine (Feb 2022) | Diesel | −38 ÷ −76 M€ | Not rejected |
| Iran–Israel (Jun 2025) | Petrol | +22 ÷ +45 M€ | **Rejected** |
| Iran–Israel (Jun 2025) | Diesel | +99 ÷ +110 M€ | **Rejected** |
| Hormuz (Feb 2026) | Petrol | −3 ÷ −18 M€ | Not rejected |
| Hormuz (Feb 2026) | Diesel | −65 ÷ −126 M€ | Not rejected |

---

## Repository Structure

```
war-shock-oil-profit-analysis/
├── src/
│   └── data/
│       ├── processed/
│       │   ├── daily_fuel_prices_all.csv        # All stations, daily average prices
│       │   └── daily_fuel_prices_stradale.csv   # Self-service pump prices
│       ├── Futures/                             # Eurobob & Gas Oil futures (EUR/L)
│       └── plots/
│           ├── its/
│           │   ├── detected/
│           │   │   ├── margin/compare/          # Cross-model margin comparisons
│           │   │   └── price/compare/
│           │   └── fixed/
│           │       ├── margin/                  # Per-model margin plots & residuals
│           │       └── price/
│           └── triangulation/                   # Fisher & Stouffer combination results
├── requirements.txt
└── README.md
```

---

## Data Sources

| Source | Content | URL |
|--------|---------|-----|
| **MIMIT Open Data Carburanti** | Daily self-service pump prices per station (€/L), 2015–2026 | https://opendatacarburanti.mise.gov.it |
| **SISEN-MASE** | Weekly fiscal breakdown (excise duties + VAT); monthly petroleum consumption | https://sisen.mase.gov.it |
| **TradingView** | Eurobob B7H1 futures — petrol wholesale proxy (USD/t) | https://tradingview.com |
| **Investing.com** | London Gas Oil ICE LGOk6 — diesel wholesale proxy (USD/t) | https://investing.com |
| **Yahoo Finance** | EUR/USD daily exchange rate | https://finance.yahoo.com |

Wholesale futures on refined products (rather than Brent crude) are used as the cost proxy, following Meyler (2009), because they more faithfully capture distributors' marginal replacement cost.

---

## Methodology

### 1. Data Preparation

- **Unit conversion** — futures originally in USD/t are converted to EUR/L using fuel-specific densities (petrol: 0.74 kg/L → 1 351 L/t; diesel: 0.84 kg/L → 1 190 L/t) and the daily EUR/USD rate.
- **Net price** — excise duties and VAT are stripped from the pump price to isolate the net retail revenue:

$$P_{\text{netto}} = P_{\text{pompa}} - \text{Accise} - \text{IVA}$$

- **Gross margin** — the daily spread between net retail price and converted wholesale cost:

$$M = P_{\text{netto}} - P_{\text{futures}}$$

- **Daily consumption** — monthly immission-to-consumption volumes (SISEN/MASE, in thousands of tonnes) are converted to litres and distributed uniformly across days, enabling conversion of €/L margins into cumulative M€ extra profits.

### 2. Event Windows

Each shock is analysed over a **±40-day window** centred on the event date. The pre-break period serves as the baseline; the post-break period is where anomalies are measured. This window is wide enough to satisfy the Central Limit Theorem and to cover the typical wholesale-to-pump pass-through lag documented in the literature.

### 3. Multi-Model Counterfactual Estimation

Four models — each with different assumptions — are run in parallel to guard against model-specification uncertainty:

| Model | Temporal structure | Error distribution | Paradigm | Key strength |
|-------|-------------------|--------------------|----------|--------------|
| **OLS (LinReg)** | Ignored | Gaussian | Frequentist | Simple, interpretable baseline |
| **ARIMA** | Explicit AR/MA | Gaussian white noise | Frequentist | Captures serial memory & inertia |
| **Theil-Sen** | Order only | None (non-parametric) | Frequentist / NP | Robust to outliers (breakdown ≈ 29%) |
| **PyMC (AR(1) Student-t)** | Explicit AR(1) | Student-t | Bayesian | Heavy-tail robustness + asymmetric HDI |

The ARIMA block selects among Auto-ARIMA AIC, Holt-Winters ETS, and OLS-trend projection based on minimum RMSE on the pre-break period.

The PyMC model uses weakly informative priors:
- α ~ Normal(μ_pre, 3σ_pre)
- β ~ Normal(0, 0.005)
- σ ~ HalfNormal(σ_pre)
- ρ ~ Uniform(−0.95, 0.95)
- ν ~ Exponential(1/30) + 2

Posterior sampling is performed with the NUTS sampler.

### 4. Anomaly Threshold

An observation is flagged as anomalous when the absolute residual exceeds twice the pre-shock standard deviation:

$$|\hat{\varepsilon}_t| > 2 \cdot \sigma_{\text{pre}}$$

This conservative threshold limits false positives and classifies as anomalies only deviations significantly above the natural pre-shock variability.

### 5. Statistical Triangulation

For each model, a one-sided **Wilcoxon signed-rank test** (H₁: extra > 0) is applied to post-break residuals. The effect size is reported via the **Hodges-Lehmann estimator** (median of Walsh averages), which is robust to outliers and retains direct economic interpretation (€/L).

The four p-values are then combined using two complementary meta-analytic methods:

$$\chi^2_{\text{Fisher}} = -2\sum \ln(p_i) \sim \chi^2_{2k}$$

$$Z_{\text{Stouffer}} = \sum \frac{\Phi^{-1}(1-p_i)}{\sqrt{k}} \sim \mathcal{N}(0,1)$$

The **final verdict** is based on the more conservative of the two combined p-values, with α = 0.05 as the rejection threshold.

---

## Results Summary

The formal triangulation (Table III of the report) confirms:

- **Ukraine / Petrol**: Fisher p = 0.0007, Stouffer p = 0.0002 → **H₀ rejected**
- **Ukraine / Diesel**: Fisher p = 0.97, Stouffer p = 0.91 → H₀ not rejected *(margins compressed, consistent with delayed pass-through)*
- **Iran–Israel / Petrol**: all four models return p ≈ 0 (numerical overflow) → **H₀ rejected** (HL = +0.0147 €/L)
- **Iran–Israel / Diesel**: Fisher p ≈ 0, Stouffer p ≈ 0 → **H₀ rejected**
- **Hormuz / Petrol**: Fisher p = 0.98, Stouffer p = 0.94 → H₀ not rejected
- **Hormuz / Diesel**: Fisher p ≈ 1, Stouffer p ≈ 1 → H₀ not rejected

---

## Limitations

- **Data availability** — no public data on operational costs means the computed margin is *gross* (not net of logistical and commercial costs); estimates should be interpreted as an upper bound on net speculative profit.
- **Conservative lower bound** — the 2σ threshold and fixed break-date choice (markets often anticipate shocks) make all estimates conservative.
- **Causal inference** — the interrupted time-series design documents association, not strict causality; confounders cannot be fully ruled out.
- **Single-country scope** — results may not generalise beyond the Italian retail fuel market.

---

## Possible Extensions

- **Dynamic change-point detection** to replace fixed event dates with data-driven structural breaks.
- **Window sensitivity analysis** (30 / 60 / 90 days) to test result robustness.
- **Longitudinal analysis** covering 2020–2026 with a pre-pandemic baseline.
- **International comparison** and difference-in-differences designs for stronger causal identification.

---

## Requirements

Install dependencies with:

```bash
pip install -r requirements.txt
```

Core libraries: `pandas`, `numpy`, `scipy`, `statsmodels`, `pmdarima`, `pymc`, `matplotlib`, `seaborn`.

---
