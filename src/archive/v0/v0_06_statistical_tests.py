"""
06_statistical_tests.py
========================
Test statistici aggiuntivi per rafforzare il rifiuto di H0.

Tests implementati:
  1. Kolmogorov-Smirnov       — distribuzione prezzi pre vs post shock
  2. ANOVA a un fattore        — varianza tra 3 periodi (pre / shock / post)
  3. Chow Test                 — structural break formale sulla regressione
  4. Cross-Correlation (CCF)   — lag ottimale Brent → pompa
  5. Rolling Correlation       — correlazione mobile nel tempo
  6. Confidence Intervals      — su changepoint e pendenze (bootstrap)
  7. RMSE / MAE                — bonta del fit della regressione piecewise
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from scipy import stats
from scipy.ndimage import uniform_filter1d
import warnings
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────
# Carica dati
# ─────────────────────────────────────────
merged = pd.read_csv("data/dataset_merged.csv", index_col=0, parse_dates=True)
merged.dropna(inplace=True)
print(f"Dataset: {len(merged)} settimane | {merged.index[0].date()} → {merged.index[-1].date()}\n")

# Eventi di guerra con finestre temporali
EVENTS = {
    "Ucraina (Feb 2022)": {
        "shock":      pd.Timestamp("2022-02-24"),
        "pre_start":  pd.Timestamp("2021-09-01"),
        "post_end":   pd.Timestamp("2022-08-31"),
    },
    "Hormuz (Feb 2026)": {
        "shock":      pd.Timestamp("2026-02-28"),
        "pre_start":  pd.Timestamp("2025-09-01"),
        "post_end":   pd.Timestamp("2026-03-17"),
    },
}

FUELS = {
    "Benzina": "benzina_4w",
    "Diesel":  "diesel_4w",
}

ALPHA = 0.05

# ─────────────────────────────────────────────────────────────────────────────
# 1. KOLMOGOROV-SMIRNOV TEST
#    H0: la distribuzione dei prezzi pre-shock = distribuzione post-shock
#    Rifiuto H0 → i prezzi post-shock appartengono a una distribuzione diversa
# ─────────────────────────────────────────────────────────────────────────────
print("=" * 65)
print("1. KOLMOGOROV-SMIRNOV TEST")
print("   H0: distribuzione pre-shock = distribuzione post-shock")
print("=" * 65)

ks_results = []

for event_name, cfg in EVENTS.items():
    shock = cfg["shock"]
    for fuel_name, fuel_col in FUELS.items():
        if fuel_col not in merged.columns:
            continue

        pre  = merged.loc[cfg["pre_start"]:shock, fuel_col].dropna()
        post = merged.loc[shock:cfg["post_end"],   fuel_col].dropna()

        if len(pre) < 4 or len(post) < 4:
            continue

        ks_stat, ks_p = stats.ks_2samp(pre.values, post.values)
        result = "RIFIUTATA" if ks_p < ALPHA else "non rifiutata"

        ks_results.append({
            "Evento":      event_name,
            "Carburante":  fuel_name,
            "n_pre":       len(pre),
            "n_post":      len(post),
            "KS_stat":     round(ks_stat, 4),
            "p_value":     round(ks_p, 6),
            "H0":          result,
        })

        print(f"  {event_name} | {fuel_name}: KS={ks_stat:.4f}, p={ks_p:.6f} → H0 {result}")

pd.DataFrame(ks_results).to_csv("data/ks_results.csv", index=False)
print()


# ─────────────────────────────────────────────────────────────────────────────
# 2. ANOVA A UN FATTORE
#    Confronta la varianza dei prezzi in 3 periodi:
#      - Periodo A: 6 mesi pre-shock
#      - Periodo B: 0-6 settimane post-shock (shock acuto)
#      - Periodo C: 6 settimane - 6 mesi post-shock (normalizzazione)
#    H0: le medie dei 3 periodi sono uguali
# ─────────────────────────────────────────────────────────────────────────────
print("=" * 65)
print("2. ANOVA A UN FATTORE (3 periodi: pre / shock acuto / post)")
print("   H0: media prezzi uguale nei 3 periodi")
print("=" * 65)

anova_results = []

for event_name, cfg in EVENTS.items():
    shock = cfg["shock"]
    for fuel_name, fuel_col in FUELS.items():
        if fuel_col not in merged.columns:
            continue

        period_A = merged.loc[cfg["pre_start"]:shock,                       fuel_col].dropna()
        period_B = merged.loc[shock:shock + pd.Timedelta(weeks=6),          fuel_col].dropna()
        period_C = merged.loc[shock + pd.Timedelta(weeks=6):cfg["post_end"],fuel_col].dropna()

        if len(period_A) < 3 or len(period_B) < 3 or len(period_C) < 3:
            continue

        f_stat, anova_p = stats.f_oneway(period_A, period_B, period_C)
        result = "RIFIUTATA" if anova_p < ALPHA else "non rifiutata"

        anova_results.append({
            "Evento":     event_name,
            "Carburante": fuel_name,
            "F_stat":     round(f_stat, 4),
            "p_value":    round(anova_p, 6),
            "mean_A":     round(period_A.mean(), 4),
            "mean_B":     round(period_B.mean(), 4),
            "mean_C":     round(period_C.mean(), 4),
            "H0":         result,
        })

        print(f"  {event_name} | {fuel_name}: F={f_stat:.4f}, p={anova_p:.6f} → H0 {result}")
        print(f"    medie: pre={period_A.mean():.4f} | shock={period_B.mean():.4f} | post={period_C.mean():.4f}")

pd.DataFrame(anova_results).to_csv("data/anova_results.csv", index=False)
print()


# ─────────────────────────────────────────────────────────────────────────────
# 3. CHOW TEST
#    Test formale di structural break su una regressione lineare.
#    Divide la serie in due segmenti al punto di break e testa se
#    i coefficienti sono significativamente diversi.
#    H0: nessun structural break (coefficienti uguali prima e dopo)
# ─────────────────────────────────────────────────────────────────────────────
print("=" * 65)
print("3. CHOW TEST (structural break)")
print("   H0: nessuna rottura strutturale nella regressione")
print("=" * 65)

def chow_test(y: np.ndarray, breakpoint: int):
    """
    Chow test for structural break at index `breakpoint`.
    F = [(RSS_r - RSS_u) / k] / [RSS_u / (n - 2k)]
    dove:
      RSS_r = residui regressione sull'intero campione
      RSS_u = RSS_1 + RSS_2 (regressioni separate)
      k = numero di parametri (2: intercetta + slope)
    """
    n = len(y)
    x = np.arange(n)
    k = 2  # intercetta + slope

    # Regressione sull'intero campione (restricted)
    X_full = np.column_stack([np.ones(n), x])
    beta_r, _, _, _ = np.linalg.lstsq(X_full, y, rcond=None)
    rss_r = np.sum((y - X_full @ beta_r) ** 2)

    # Regressioni separate (unrestricted)
    def ols_rss(xv, yv):
        if len(xv) < k + 1:
            return 0.0
        Xv = np.column_stack([np.ones(len(xv)), xv])
        b, _, _, _ = np.linalg.lstsq(Xv, yv, rcond=None)
        return np.sum((yv - Xv @ b) ** 2)

    rss1 = ols_rss(x[:breakpoint],  y[:breakpoint])
    rss2 = ols_rss(x[breakpoint:],  y[breakpoint:])
    rss_u = rss1 + rss2

    if rss_u < 1e-12:
        return np.nan, np.nan

    f_stat = ((rss_r - rss_u) / k) / (rss_u / (n - 2 * k))
    p_val  = 1 - stats.f.cdf(f_stat, dfn=k, dfd=n - 2 * k)
    return f_stat, p_val


chow_results = []

for event_name, cfg in EVENTS.items():
    shock = cfg["shock"]
    for fuel_name, fuel_col in FUELS.items():
        if fuel_col not in merged.columns:
            continue

        series = merged.loc[cfg["pre_start"]:cfg["post_end"], fuel_col].dropna()
        if len(series) < 10:
            continue

        # Trova l'indice corrispondente alla data dello shock
        shock_idx = series.index.searchsorted(shock)
        shock_idx = max(3, min(shock_idx, len(series) - 3))

        f_stat, p_val = chow_test(series.values, shock_idx)

        if np.isnan(f_stat):
            continue

        result = "RIFIUTATA" if p_val < ALPHA else "non rifiutata"
        chow_results.append({
            "Evento":     event_name,
            "Carburante": fuel_name,
            "Break_date": shock.date(),
            "F_stat":     round(f_stat, 4),
            "p_value":    round(p_val, 6),
            "H0":         result,
        })

        print(f"  {event_name} | {fuel_name}: F={f_stat:.4f}, p={p_val:.6f} → H0 {result}")

pd.DataFrame(chow_results).to_csv("data/chow_results.csv", index=False)
print()


# ─────────────────────────────────────────────────────────────────────────────
# 4. CROSS-CORRELATION FUNCTION (CCF)
#    Calcola la correlazione tra Brent e prezzi pompa per lag da 0 a 12 sett.
#    Il lag al massimo della CCF = lag ottimale di trasmissione.
#    Se lag_ottimale < 4 sett (< 30gg) → H0 rifiutata.
# ─────────────────────────────────────────────────────────────────────────────
print("=" * 65)
print("4. CROSS-CORRELATION FUNCTION (CCF)")
print("   Lag al picco massimo = velocita di trasmissione stimata")
print("=" * 65)

MAX_LAG_CCF = 12  # settimane

ccf_results = {}
d_brent = merged["log_brent"].diff().dropna()

for fuel_name, fuel_col in FUELS.items():
    log_col = "log_benzina" if fuel_name == "Benzina" else "log_diesel"
    if log_col not in merged.columns:
        continue

    d_fuel = merged[log_col].diff().dropna()
    common = d_brent.index.intersection(d_fuel.index)
    x = d_brent.loc[common].values
    y = d_fuel.loc[common].values

    # Calcola cross-correlazione per lag 0..MAX_LAG_CCF
    ccf_vals = []
    for lag in range(0, MAX_LAG_CCF + 1):
        if lag == 0:
            r, _ = stats.pearsonr(x, y)
        else:
            r, _ = stats.pearsonr(x[:-lag], y[lag:])
        ccf_vals.append(r)

    ccf_results[fuel_name] = ccf_vals
    best_lag = np.argmax(np.abs(ccf_vals))
    best_r   = ccf_vals[best_lag]

    print(f"  {fuel_name}: lag ottimale = {best_lag} settimane ({best_lag*7} giorni), "
          f"r = {best_r:.4f} → "
          f"{'H0 RIFIUTATA (lag < 4 sett)' if best_lag < 4 else 'compatibile con logistica'}")

print()


# ─────────────────────────────────────────────────────────────────────────────
# 5. ROLLING CORRELATION (finestra mobile 12 settimane)
#    Mostra come la correlazione Brent-pompa cambia nel tempo.
#    Durante le guerre la correlazione dovrebbe aumentare bruscamente.
# ─────────────────────────────────────────────────────────────────────────────
print("=" * 65)
print("5. ROLLING CORRELATION (finestra 12 settimane)")
print("=" * 65)

ROLL_WIN = 12

rolling_corr = {}
for fuel_name, fuel_col in FUELS.items():
    if fuel_col not in merged.columns:
        continue
    rc = merged["brent_7d_eur"].rolling(ROLL_WIN).corr(merged[fuel_col])
    rolling_corr[fuel_name] = rc
    print(f"  {fuel_name}: corr media = {rc.mean():.4f} | "
          f"corr durante Ucraina (mar 2022) = "
          f"{rc.loc['2022-03-01':'2022-04-30'].mean():.4f}")

print()


# ─────────────────────────────────────────────────────────────────────────────
# 6. BOOTSTRAP CONFIDENCE INTERVALS SUL LAG D
#    Stima non parametrica dell'incertezza sul lag changepoint.
#    Ricampiona con replacement la serie e ricalcola il changepoint N volte.
# ─────────────────────────────────────────────────────────────────────────────
print("=" * 65)
print("6. BOOTSTRAP CONFIDENCE INTERVALS (95%) sul lag D")
print("=" * 65)

import ruptures as rpt

N_BOOTSTRAP = 500

def detect_changepoint(series_values):
    signal = series_values.reshape(-1, 1)
    try:
        algo = rpt.Binseg(model="l2").fit(signal)
        cp = algo.predict(n_bkps=1)[0]
    except Exception:
        cp = len(series_values) // 2
    return min(cp, len(series_values) - 2)


bootstrap_results = []

for event_name, cfg in EVENTS.items():
    shock = cfg["shock"]

    # Serie Brent nella finestra
    brent_series = merged.loc[cfg["pre_start"]:cfg["post_end"], "log_brent"].dropna()
    if len(brent_series) < 10:
        continue

    cp_brent_idx  = detect_changepoint(brent_series.values)
    tau_crude_base = brent_series.index[cp_brent_idx]

    for fuel_name, fuel_col in FUELS.items():
        log_col = "log_benzina" if fuel_name == "Benzina" else "log_diesel"
        if log_col not in merged.columns:
            continue

        fuel_series = merged.loc[cfg["pre_start"]:cfg["post_end"], log_col].dropna()
        if len(fuel_series) < 10:
            continue

        np.random.seed(42)
        lag_samples = []

        for _ in range(N_BOOTSTRAP):
            # Ricampiona con replacement (block bootstrap con blocchi da 4 sett.)
            block_size = 4
            n = len(fuel_series)
            n_blocks = n // block_size + 1
            idx = np.concatenate([
                np.arange(i, min(i + block_size, n))
                for i in np.random.choice(range(0, n - block_size + 1), n_blocks)
            ])[:n]
            sample = fuel_series.values[idx]
            cp_idx = detect_changepoint(sample)
            tau_retail_boot = fuel_series.index[min(cp_idx, len(fuel_series) - 1)]
            lag_samples.append((tau_retail_boot - tau_crude_base).days)

        lag_samples = np.array(lag_samples)
        ci_low  = np.percentile(lag_samples, 2.5)
        ci_high = np.percentile(lag_samples, 97.5)
        lag_mean = np.mean(lag_samples)

        bootstrap_results.append({
            "Evento":     event_name,
            "Carburante": fuel_name,
            "Lag_mean":   round(lag_mean, 1),
            "CI_95_low":  round(ci_low, 1),
            "CI_95_high": round(ci_high, 1),
            "H0 (CI < 30gg)": "RIFIUTATA" if ci_high < 30 else "non rifiutata",
        })

        print(f"  {event_name} | {fuel_name}: "
              f"D = {lag_mean:.1f}gg [95% CI: {ci_low:.1f} – {ci_high:.1f}] → "
              f"{'H0 RIFIUTATA' if ci_high < 30 else 'non rifiutata'}")

pd.DataFrame(bootstrap_results).to_csv("data/bootstrap_ci.csv", index=False)
print()


# ─────────────────────────────────────────────────────────────────────────────
# 7. RMSE / MAE — BONTA DEL FIT DELLA REGRESSIONE PIECEWISE
#    Misura quanto bene la regressione piecewise (script 02) fitta i dati.
#    RMSE e MAE bassi = modello a due tratti descrive bene la dinamica reale.
# ─────────────────────────────────────────────────────────────────────────────
print("=" * 65)
print("7. RMSE / MAE — bonta del fit piecewise vs regressione semplice")
print("=" * 65)

def fit_and_score(y, breakpoint_idx=None):
    """
    Fitta regressione semplice e piecewise, restituisce RMSE e MAE per entrambe.
    """
    x = np.arange(len(y))

    # Regressione lineare semplice
    slope, intercept, *_ = stats.linregress(x, y)
    y_hat_simple = intercept + slope * x
    rmse_simple = np.sqrt(np.mean((y - y_hat_simple) ** 2))
    mae_simple  = np.mean(np.abs(y - y_hat_simple))

    if breakpoint_idx is None or breakpoint_idx <= 1 or breakpoint_idx >= len(y) - 1:
        return rmse_simple, mae_simple, np.nan, np.nan

    # Regressione piecewise
    cp = breakpoint_idx
    def ols(xv, yv):
        Xv = np.column_stack([np.ones(len(xv)), xv])
        b, _, _, _ = np.linalg.lstsq(Xv, yv, rcond=None)
        return b
    b1 = ols(x[:cp], y[:cp])
    b2 = ols(x[cp:], y[cp:])
    y_hat_pw = np.concatenate([
        b1[0] + b1[1] * x[:cp],
        b2[0] + b2[1] * x[cp:],
    ])
    rmse_pw = np.sqrt(np.mean((y - y_hat_pw) ** 2))
    mae_pw  = np.mean(np.abs(y - y_hat_pw))

    return rmse_simple, mae_simple, rmse_pw, mae_pw


fit_results = []

for event_name, cfg in EVENTS.items():
    shock = cfg["shock"]
    for fuel_name, fuel_col in FUELS.items():
        log_col = "log_benzina" if fuel_name == "Benzina" else "log_diesel"
        if log_col not in merged.columns:
            continue

        series = merged.loc[cfg["pre_start"]:cfg["post_end"], log_col].dropna()
        if len(series) < 10:
            continue

        shock_idx = series.index.searchsorted(shock)
        shock_idx = max(2, min(shock_idx, len(series) - 2))

        rs, ms, rpw, mpw = fit_and_score(series.values, shock_idx)

        fit_results.append({
            "Evento":        event_name,
            "Carburante":    fuel_name,
            "RMSE_semplice": round(rs, 6),
            "MAE_semplice":  round(ms, 6),
            "RMSE_piecewise": round(rpw, 6) if not np.isnan(rpw) else "N/A",
            "MAE_piecewise":  round(mpw, 6) if not np.isnan(mpw) else "N/A",
            "Miglioramento_%": round((1 - rpw / rs) * 100, 1) if not np.isnan(rpw) and rs > 0 else "N/A",
        })

        if not np.isnan(rpw):
            print(f"  {event_name} | {fuel_name}: "
                  f"RMSE semplice={rs:.5f} | RMSE piecewise={rpw:.5f} "
                  f"(miglioramento: {(1 - rpw/rs)*100:.1f}%)")

pd.DataFrame(fit_results).to_csv("data/fit_quality.csv", index=False)
print()


# ─────────────────────────────────────────────────────────────────────────────
# 8. TEST PER SELEZIONE DEL TIPO DI REGRESSIONE
#    Risponde alla domanda: quale modello è più appropriato?
#    Test implementati:
#      8a. Breusch-Pagan (LM test)   → eteroschedasticità formale
#      8b. Ljung-Box                 → autocorrelazione dei residui (AR test)
#      8c. AIC/BIC OLS vs GLSAR      → quale modello fitta meglio?
#      8d. Confronto SE: OLS vs HAC vs GLSAR  → quanto distorcono gli SE classici?
#      8e. Decisione finale          → raccomandazione motivata per ogni serie
#
#    OUTPUT: data/regression_selection.csv  +  plots/08_regression_selection.png
# ─────────────────────────────────────────────────────────────────────────────
print("=" * 65)
print("8. TEST SELEZIONE TIPO DI REGRESSIONE")
print("   OLS standard vs HAC Newey-West vs GLSAR AR(1) vs Bayesian StudentT+AR(1)")
print("=" * 65)

from statsmodels.regression.linear_model import OLS, GLSAR
from statsmodels.stats.sandwich_covariance import cov_hac
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.stats.stattools import durbin_watson

def breusch_pagan_lm(residuals, fitted):
    """Breusch-Pagan LM test: H0 = omoschedasticità."""
    n  = len(residuals)
    e2 = residuals ** 2
    X_ = np.column_stack([np.ones(n), fitted])
    b_, _, _, _ = np.linalg.lstsq(X_, e2, rcond=None)
    e2h = X_ @ b_
    ss_res = np.sum((e2 - e2h)**2)
    ss_tot = np.sum((e2 - e2.mean())**2)
    r2 = 1 - ss_res/ss_tot if ss_tot > 0 else 0.
    lm = n * r2
    return lm, 1 - stats.chi2.cdf(lm, df=1)


# Serie da testare: serie in livelli log sulle finestre evento
SERIES_LOG = {
    "log_brent":    "Brent",
    "log_benzina":  "Benzina",
    "log_diesel":   "Diesel",
}

reg_sel_rows = []

for event_name, cfg in EVENTS.items():
    shock = cfg["shock"]
    window_data = merged.loc[cfg["pre_start"]:cfg["post_end"]].dropna()
    n_obs = len(window_data)
    if n_obs < 15:
        continue

    x_arr = np.arange(n_obs, dtype=float)
    X_mat = np.column_stack([np.ones(n_obs), x_arr])

    for log_col, series_label in SERIES_LOG.items():
        if log_col not in window_data.columns:
            continue

        y_arr = window_data[log_col].values

        # ── 8a. OLS baseline
        ols_res  = OLS(y_arr, X_mat).fit()
        resid_ols = ols_res.resid
        fitted_ols = ols_res.fittedvalues

        # ── 8b. Breusch-Pagan (eteroschedasticità)
        bp_lm, bp_p = breusch_pagan_lm(resid_ols, fitted_ols)
        bp_verdict  = "eteroschedasticità" if bp_p < ALPHA else "omoschedasticità"

        # ── 8c. Ljung-Box (autocorrelazione residui, lag 1–4)
        lb_lags = min(4, n_obs // 4)
        lb_res  = acorr_ljungbox(resid_ols, lags=lb_lags, return_df=True)
        lb_p_min = float(lb_res["lb_pvalue"].min())   # p più basso tra i lag
        lb_verdict = "autocorrelazione" if lb_p_min < ALPHA else "no autocorrelazione"

        # ── 8d. Durbin-Watson
        dw_val = durbin_watson(resid_ols)
        if dw_val < 1.5:
            dw_verdict = "autocorrelazione positiva"
        elif dw_val > 2.5:
            dw_verdict = "autocorrelazione negativa"
        else:
            dw_verdict = "assenza autocorrelazione"

        # ── 8e. GLSAR AR(1)
        try:
            gl_res    = GLSAR(y_arr, X_mat, rho=1).iterative_fit(maxiter=10)
            rho_ar    = float(gl_res.rho)
            aic_ols   = ols_res.aic
            aic_glsar = gl_res.aic
            bic_ols   = ols_res.bic
            bic_glsar = gl_res.bic
            glsar_ok  = True
        except Exception:
            rho_ar = np.nan
            aic_ols = ols_res.aic
            aic_glsar = np.nan
            bic_ols = ols_res.bic
            bic_glsar = np.nan
            glsar_ok = False

        # ── 8f. Confronto SE: OLS classici vs HAC vs GLSAR
        se_ols_slope  = float(ols_res.bse[1])
        cov_nw_mat    = cov_hac(ols_res, nlags=4)
        se_hac_slope  = float(np.sqrt(cov_nw_mat[1, 1]))
        if glsar_ok:
            cov_gl  = cov_hac(gl_res, nlags=4)
            se_gl_slope = float(np.sqrt(cov_gl[1, 1]))
        else:
            se_gl_slope = np.nan

        # ── 8g. Decisione finale basata sui test
        issues = []
        if bp_p < ALPHA:
            issues.append("eteroschedasticità")
        if lb_p_min < ALPHA or dw_val < 1.5:
            issues.append("AR(1)")
        if stats.shapiro(resid_ols)[1] < ALPHA:
            issues.append("non-normalità")

        if not issues:
            recommendation = "OLS standard"
            rec_reason     = "tutte le ipotesi soddisfatte"
        elif "AR(1)" in issues and "non-normalità" in issues:
            recommendation = "Bayesian StudentT + AR(1)"
            rec_reason     = "autocorrelazione + non-normalità simultanee"
        elif "AR(1)" in issues:
            recommendation = "GLSAR AR(1) + HAC"
            rec_reason     = "autocorrelazione nei residui"
        elif "eteroschedasticità" in issues:
            recommendation = "OLS + HAC Newey-West"
            rec_reason     = "eteroschedasticità (no autocorrelazione)"
        else:
            recommendation = "OLS + HAC Newey-West"
            rec_reason     = "violazione minore"

        row = {
            "Evento":         event_name,
            "Serie":          series_label,
            "n_obs":          n_obs,
            "BP_LM":          round(bp_lm, 3),
            "BP_p":           round(bp_p, 4),
            "BP_verdict":     bp_verdict,
            "LjungBox_p_min": round(lb_p_min, 4),
            "LB_verdict":     lb_verdict,
            "DW":             round(dw_val, 4),
            "DW_verdict":     dw_verdict,
            "rho_AR1":        round(rho_ar, 3) if not np.isnan(rho_ar) else "N/A",
            "AIC_OLS":        round(aic_ols, 2),
            "AIC_GLSAR":      round(aic_glsar, 2) if not np.isnan(aic_glsar) else "N/A",
            "BIC_OLS":        round(bic_ols, 2),
            "BIC_GLSAR":      round(bic_glsar, 2) if not np.isnan(bic_glsar) else "N/A",
            "SE_OLS_slope":   round(se_ols_slope, 6),
            "SE_HAC_slope":   round(se_hac_slope, 6),
            "SE_GLSAR_slope": round(se_gl_slope, 6) if not np.isnan(se_gl_slope) else "N/A",
            "SE_inflation_%": round((se_hac_slope / se_ols_slope - 1) * 100, 1) if se_ols_slope > 0 else "N/A",
            "Problemi":       " + ".join(issues) if issues else "nessuno",
            "Raccomandazione": recommendation,
            "Motivazione":    rec_reason,
        }
        reg_sel_rows.append(row)

        inflate_str = f"{row['SE_inflation_%']}%" if row['SE_inflation_%'] != "N/A" else "N/A"
        aic_delta   = round(aic_ols - aic_glsar, 1) if not np.isnan(aic_glsar) else "N/A"
        print(f"\n  {event_name} | {series_label}:")
        print(f"    BP p={bp_p:.4f} ({bp_verdict}) | LB p_min={lb_p_min:.4f} ({lb_verdict}) | DW={dw_val:.3f} ({dw_verdict})")
        print(f"    ρ AR(1)={row['rho_AR1']}  |  ΔAIC(OLS-GLSAR)={aic_delta}")
        print(f"    SE OLS={se_ols_slope:.5f} → SE HAC={se_hac_slope:.5f} (+{inflate_str})")
        print(f"    ✦ RACCOMANDAZIONE: {recommendation}  [{rec_reason}]")

pd.DataFrame(reg_sel_rows).to_csv("data/regression_selection.csv", index=False)
print(f"\n  Salvato: data/regression_selection.csv")
print()


# ─────────────────────────────────────────────────────────────────────────────
# PLOT 8: Confronto SE per metodo + Mappa decisioni
# ─────────────────────────────────────────────────────────────────────────────
if reg_sel_rows:
    df_sel = pd.DataFrame(reg_sel_rows)

    fig8, axes8 = plt.subplots(1, 2, figsize=(15, 6))
    fig8.suptitle(
        "Selezione Tipo di Regressione — Test Diagnostici\n"
        "Confronto SE: OLS classico vs HAC Newey-West vs GLSAR AR(1)",
        fontsize=12, fontweight="bold"
    )

    # ── Pannello sinistro: SE inflation (% di distorsione OLS→HAC)
    ax8l = axes8[0]
    df_sel_num = df_sel[df_sel["SE_inflation_%"] != "N/A"].copy()
    df_sel_num["SE_inflation_%"] = df_sel_num["SE_inflation_%"].astype(float)
    labels_l = [f"{r['Evento'].split('(')[0].strip()}\n{r['Serie']}"
                for _, r in df_sel_num.iterrows()]
    colors_l = ["#e74c3c" if v > 50 else "#e67e22" if v > 20 else "#27ae60"
                for v in df_sel_num["SE_inflation_%"]]
    bars_l = ax8l.barh(range(len(labels_l)), df_sel_num["SE_inflation_%"].values,
                       color=colors_l, alpha=0.80, edgecolor="black", lw=0.5)
    for i, (_, row) in enumerate(df_sel_num.iterrows()):
        ax8l.text(df_sel_num["SE_inflation_%"].values[i] + 0.5, i,
                  f"{row['SE_inflation_%']:.0f}%",
                  va="center", fontsize=8)
    ax8l.axvline(0, color="black", lw=0.8)
    ax8l.axvline(20, color="#e67e22", lw=1.4, linestyle="--",
                 label="soglia warning (20%)")
    ax8l.axvline(50, color="#e74c3c", lw=1.4, linestyle="--",
                 label="soglia critica (50%)")
    ax8l.set_yticks(range(len(labels_l)))
    ax8l.set_yticklabels(labels_l, fontsize=8)
    ax8l.set_xlabel("SE distorsione OLS→HAC (%)", fontsize=10)
    ax8l.set_title("Distorsione SE: OLS classico vs HAC\n(rosso = OLS molto distorto)", fontsize=10)
    ax8l.legend(fontsize=8)
    ax8l.grid(alpha=0.3, axis="x")

    # ── Pannello destro: heatmap decisioni (tabella colorata)
    ax8r = axes8[1]
    ax8r.axis("off")

    rec_colors = {
        "OLS standard":               "#27ae60",
        "OLS + HAC Newey-West":        "#f39c12",
        "GLSAR AR(1) + HAC":           "#e67e22",
        "Bayesian StudentT + AR(1)":   "#e74c3c",
    }
    col_headers = ["Evento", "Serie", "BP", "LB", "DW", "ρ AR(1)", "Raccomandazione"]
    rows_tab    = []
    for _, row in df_sel.iterrows():
        rows_tab.append([
            row["Evento"].split("(")[0].strip(),
            row["Serie"],
            f"p={row['BP_p']:.3f}",
            f"p={row['LjungBox_p_min']:.3f}",
            f"{row['DW']:.2f}",
            str(row["rho_AR1"]),
            row["Raccomandazione"],
        ])

    tbl = ax8r.table(
        cellText=rows_tab,
        colLabels=col_headers,
        cellLoc="center", loc="center",
        bbox=[0, 0, 1, 1]
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(7.5)
    for (r_idx, c_idx), cell in tbl.get_celld().items():
        if r_idx == 0:
            cell.set_facecolor("#2c3e50")
            cell.set_text_props(color="white", fontweight="bold")
        elif c_idx == 6:  # colonna Raccomandazione
            rec_text = rows_tab[r_idx - 1][6] if r_idx > 0 else ""
            cell.set_facecolor(rec_colors.get(rec_text, "#ecf0f1"))
            cell.set_text_props(fontweight="bold", fontsize=6.5)
        else:
            cell.set_facecolor("#f8f9fa" if r_idx % 2 == 0 else "white")
    ax8r.set_title("Matrice di Decisione: Tipo di Regressione", fontsize=10, fontweight="bold", pad=10)

    fig8.tight_layout(pad=1.5)
    fig8.savefig("plots/08_regression_selection.png", dpi=180, bbox_inches="tight")
    plt.close(fig8)
    print("  Plot salvato: plots/08_regression_selection.png")
    print()


# ─────────────────────────────────────────────────────────────────────────────
# PLOTS
# ─────────────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(15, 11))
fig.suptitle("Test statistici aggiuntivi — Speculazione carburanti Italia",
             fontsize=13, fontweight="bold")

war_dates = {
    "Ucraina": pd.Timestamp("2022-02-24"),
    "Hormuz":  pd.Timestamp("2026-02-28"),
}

# --- Plot 1: CCF ---
ax = axes[0, 0]
lags_x = list(range(0, MAX_LAG_CCF + 1))
colors  = ["#e74c3c", "#3498db"]
for (fuel_name, ccf_vals), color in zip(ccf_results.items(), colors):
    ax.plot(lags_x, ccf_vals, marker="o", color=color, lw=2, label=fuel_name)
ax.axvline(4, color="orange", lw=2, linestyle="--", label="Soglia 30gg (4 sett.)")
ax.axhline(0, color="black", lw=0.5)
ax.set_xlabel("Lag (settimane)")
ax.set_ylabel("Correlazione")
ax.set_title("Cross-Correlation: Brent → Prezzi Pompa")
ax.legend(fontsize=9)
ax.grid(alpha=0.3)
ax.set_xticks(lags_x)

# --- Plot 2: Rolling Correlation ---
ax = axes[0, 1]
colors_rc = ["#e74c3c", "#27ae60"]
for (fuel_name, rc), color in zip(rolling_corr.items(), colors_rc):
    ax.plot(rc.index, rc.values, color=color, lw=1.5, label=fuel_name)
for label, date in war_dates.items():
    if merged.index[0] <= date <= merged.index[-1]:
        ax.axvline(date, color="gray", lw=1.5, linestyle="--")
        ax.text(date, 0.05, label, rotation=90, fontsize=7, color="gray", va="bottom")
ax.set_ylabel("Correlazione (rolling 12 sett.)")
ax.set_title(f"Correlazione mobile Brent-Carburante ({ROLL_WIN} sett.)")
ax.legend(fontsize=9)
ax.grid(alpha=0.3)
ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %y"))
ax.xaxis.set_major_locator(mdates.MonthLocator(interval=6))
plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)

# --- Plot 3: KS test — distribuzioni pre vs post ---
ax = axes[1, 0]
event_name_plot = "Ucraina (Feb 2022)"
cfg_plot = EVENTS[event_name_plot]
shock_plot = cfg_plot["shock"]

for (fuel_name, fuel_col), color in zip(FUELS.items(), ["#e74c3c", "#3498db"]):
    if fuel_col not in merged.columns:
        continue
    pre  = merged.loc[cfg_plot["pre_start"]:shock_plot, fuel_col].dropna()
    post = merged.loc[shock_plot:cfg_plot["post_end"],   fuel_col].dropna()
    # ECDF
    pre_sorted  = np.sort(pre.values)
    post_sorted = np.sort(post.values)
    ax.step(pre_sorted,  np.linspace(0, 1, len(pre_sorted)),
            color=color, lw=2, linestyle="--", label=f"{fuel_name} pre")
    ax.step(post_sorted, np.linspace(0, 1, len(post_sorted)),
            color=color, lw=2, label=f"{fuel_name} post")

ax.set_xlabel("Prezzo (EUR/litro)")
ax.set_ylabel("ECDF")
ax.set_title(f"KS Test: ECDF pre vs post\n({event_name_plot})")
ax.legend(fontsize=8)
ax.grid(alpha=0.3)

# --- Plot 4: Bootstrap CI sul lag D ---
ax = axes[1, 1]
if bootstrap_results:
    df_boot = pd.DataFrame(bootstrap_results)
    labels  = [f"{r['Evento'].split('(')[0].strip()}\n{r['Carburante']}"
               for _, r in df_boot.iterrows()]
    y_pos   = np.arange(len(labels))
    means   = df_boot["Lag_mean"].values
    ci_low  = df_boot["CI_95_low"].values
    ci_high = df_boot["CI_95_high"].values

    bar_colors = ["#e74c3c" if h < 30 else "#3498db" for h in ci_high]
    ax.barh(y_pos, means, color=bar_colors, alpha=0.7, edgecolor="black", lw=0.5)
    for i, (l, h) in enumerate(zip(ci_low, ci_high)):
        ax.errorbar(means[i], y_pos[i], xerr=[[means[i]-l], [h-means[i]]],
                    fmt="none", color="black", capsize=5, lw=2)

    ax.axvline(30, color="orange", lw=2, linestyle="--", label="Soglia H0 (30gg)")
    ax.axvline(0,  color="black",  lw=0.5)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlabel("Lag D (giorni)")
    ax.set_title("Bootstrap 95% CI sul lag D\n(rosso = H0 rifiutata)")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3, axis="x")

plt.tight_layout()
plt.savefig("plots/06_statistical_tests.png", dpi=150, bbox_inches="tight")
plt.close()
print("Plot salvato: plots/06_statistical_tests.png")


# ─────────────────────────────────────────────────────────────────────────────
# SOMMARIO FINALE
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 65)
print("SOMMARIO — Tutti i test statistici")
print("=" * 65)
print(f"  {'Test':<35} {'H0 rifiutata?':<15} {'File output'}")
print(f"  {'-'*65}")
print(f"  {'Kolmogorov-Smirnov':<35} {'vedi tabella':<15} data/ks_results.csv")
print(f"  {'ANOVA (3 periodi)':<35} {'vedi tabella':<15} data/anova_results.csv")
print(f"  {'Chow Test':<35} {'vedi tabella':<15} data/chow_results.csv")
print(f"  {'CCF (lag ottimale)':<35} {'vedi output':<15} —")
print(f"  {'Rolling Correlation':<35} {'visualizzazione':<15} plots/06_...")
print(f"  {'Bootstrap CI (95%)':<35} {'vedi tabella':<15} data/bootstrap_ci.csv")
print(f"  {'RMSE / MAE':<35} {'vedi tabella':<15} data/fit_quality.csv")
print(f"  {'Selezione tipo regressione':<35} {'vedi tabella':<15} data/regression_selection.csv")
print()
print("  Script 02: Bayesian changepoint — StudentT + AR(1) MCMC")
print("  Script 03: Granger causality (ADF + F-test)")
print("  Script 04: Rockets & Feathers (GLSAR AR(1) + HAC Newey-West)")
print("  Script 06: KS, ANOVA, Chow, CCF, Rolling Corr, Bootstrap, RMSE/MAE")
print("  Script 06 §8: Test selezione regressione (BP, LB, DW, AIC/BIC, SE comparison)")
print()
print("Tutti i risultati sono salvati in data/ e plots/")
print("Tutto il codice e i risultati sono riproducibili.")