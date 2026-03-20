"""
02_changepoint_detection.py  (v3 — regressione Bayesiana StudentT + AR(1))
==========================================================================
Regressione piecewise lineare bayesiana per i tre eventi di guerra.

SCELTA DELLA REGRESSIONE — motivazione basata sui test diagnostici
  I test in 06_statistical_tests.py mostrano 3 violazioni simultanee:
    • DW ≈ 0.003–0.04  → autocorrelazione positiva quasi perfetta (AR(1))
    • BP p = 0.000      → eteroschedasticità (8/9 casi)
    • SW W = 0.27–0.68  → non-normalità degli errori (7/9 casi)
  Opzione scelta: Bayesian con likelihood StudentT + errori AR(1)
    • StudentT (ν stimato) → code pesanti, gestisce non-normalità
    • AR(1) sui residui    → struttura di autocorrelazione esplicita
    • Il CI posteriore di τ assorbe già l'incertezza totale del modello
  Riferimenti: Gelman et al. (2013); Casini & Perron (2021); Roccetti (2021)

Ogni serie produce DUE figure:
  A) Grafico principale (stile paper Casini & Roccetti 2021):
       • log(prezzo) con media lineare a tratti
       • Changepoint τ + CI 95% posteriore
       • Campana KDE posteriore di τ (pannello inferiore)
       • Doubling time e lag D
  B) Pannello diagnostica della regressione (3 sotto-plot):
       • Residuals vs Fitted  → test Breusch-Pagan
       • QQ plot              → test Shapiro-Wilk
       • ACF dei residui      → statistca Durbin-Watson

Metodologia MCMC:
  • Prior su τ: Uniform(min(x), max(x))
  • Prior su σ: HalfNormal(sd(y))
  • Prior su ν: Exponential(1/30)       [gradi libertà StudentT]
  • Prior su ρ: Uniform(-1, 1)          [coefficiente AR(1)]
  • Prior su b: StudentT(0, 3·sd(y), ν=3)
  • Prior su a: StudentT(0, sd(y)/range(x), ν=3)
  • Stima via MCMC (PyMC / NUTS sampler)
  • CI al 95% = credible interval dalla distribuzione posteriore di τ
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.gridspec as gridspec
import pymc as pm
import pytensor.tensor as pt
from scipy import stats
from statsmodels.stats.stattools import durbin_watson
from statsmodels.graphics.tsaplots import plot_acf
from pytensor.scan import scan
import warnings
warnings.filterwarnings("ignore")

# ─── Carica dataset ──────────────────────────────────────────────────────────
merged = pd.read_csv("data/dataset_merged.csv", index_col=0, parse_dates=True)
print(f"Dataset: {len(merged)} settimane\n")

# ─── Configurazione eventi ───────────────────────────────────────────────────
EVENTS = {
    "Ucraina (Feb 2022)": {
        "shock_date":   "2022-02-24",
        "window_start": "2021-10-01",
        "window_end":   "2022-07-31",
        "color":        "#e74c3c",
    },
    "Iran-Israele (Giu 2025)": {
        "shock_date":   "2025-06-13",
        "window_start": "2025-02-01",
        "window_end":   "2025-10-31",
        "color":        "#e67e22",
    },
    "Hormuz (Feb 2026)": {
        "shock_date":   "2026-02-28",
        "window_start": "2025-10-01",
        "window_end":   "2026-03-17",
        "color":        "#8e44ad",
    },
}

H0_THRESHOLD = 30
DPI          = 180
MCMC_DRAWS   = 2000
MCMC_TUNE    = 1000
MCMC_CHAINS  = 2
ALPHA        = 0.05


# ─────────────────────────────────────────
# Utility: OLS piecewise (per R² e DT)
# ─────────────────────────────────────────
def fit_piecewise_ols(x, y, cp_idx):
    def linreg(xv, yv):
        if len(xv) < 2:
            return 0., 0., 0.
        s, i, r, *_ = stats.linregress(xv, yv)
        return s, i, r**2
    b1, a1, r2_1 = linreg(x[:cp_idx], y[:cp_idx])
    b2, _,  r2_2 = linreg(x[cp_idx:], y[cp_idx:])
    return b1, b2, a1, r2_1, r2_2


def doubling_time(slope):
    return np.log(2) / (slope / 7) if slope > 0 else np.inf


# ─────────────────────────────────────────
# MCMC Bayesiano
# ─────────────────────────────────────────
def bayesian_changepoint(x_vals, y_vals, alpha=0.05):
    """
    Modello Bayesiano piecewise con:
      • Likelihood StudentT (ν stimato) → gestisce code pesanti / non-normalità
      • Errori AR(1) espliciti           → gestisce autocorrelazione quasi perfetta
        εₜ = ρ·εₜ₋₁ + ηₜ,  ηₜ ~ N(0, σ)
      • CI posteriore di τ più onesto (più largo dove i dati sono più rumorosi)
    """
    n     = len(x_vals)
    sd_y  = float(np.std(y_vals))
    rng_x = float(x_vals[-1] - x_vals[0])

    with pm.Model():
        tau   = pm.Uniform("tau",   lower=x_vals[0], upper=x_vals[-1])
        sigma = pm.HalfNormal("sigma", sigma=sd_y)
        nu    = pm.Exponential("nu", lam=1/30)          # StudentT gradi libertà
        rho   = pm.Uniform("rho", lower=-1, upper=1)    # AR(1) coefficiente
        b1    = pm.StudentT("b1",   mu=0, sigma=3 * sd_y,             nu=3)
        b2    = pm.StudentT("b2",   mu=0, sigma=3 * sd_y,             nu=3)
        a1    = pm.StudentT("a1",   mu=0, sigma=sd_y / max(rng_x, 1), nu=3)
        a2    = pm.Deterministic("a2", a1 + tau * (b1 - b2))

        x_pt  = pt.as_tensor_variable(x_vals.astype(float))
        step  = pm.math.sigmoid((x_pt - tau) * 50)
        mu    = (a1 + b1 * x_pt) * (1 - step) + (a2 + b2 * x_pt) * step

        # Errori AR(1): εₜ = ρ·εₜ₋₁ + ηₜ
        # Implementazione: media strutturale μₜ + ε_init per t=0,
        # poi εₜ = ρ·εₜ₋₁ + ηₜ per t>0
        eps_init = pm.Normal("eps_init", mu=0, sigma=sigma)
        eta      = pm.Normal("eta", mu=0, sigma=sigma, shape=n - 1)
        # Costruisce il processo AR(1) come scan
        eps_rest, _ = scan(
            fn=lambda eta_t, eps_prev, rho_v: rho_v * eps_prev + eta_t,
            sequences=[eta],
            outputs_info=[eps_init],
            non_sequences=[rho],
        )
        eps = pt.concatenate([[eps_init], eps_rest])

        pm.StudentT("obs", nu=nu, mu=mu + eps, sigma=sigma, observed=y_vals)

        trace = pm.sample(
            draws=MCMC_DRAWS, tune=MCMC_TUNE, chains=MCMC_CHAINS,
            progressbar=True, random_seed=42, target_accept=0.9,
            return_inferencedata=True,
        )

    tau_post = trace.posterior["tau"].values.flatten()
    b1_post  = trace.posterior["b1"].values.flatten()
    b2_post  = trace.posterior["b2"].values.flatten()
    a1_post  = trace.posterior["a1"].values.flatten()
    nu_post  = trace.posterior["nu"].values.flatten()
    rho_post = trace.posterior["rho"].values.flatten()

    lo_pct = (alpha / 2) * 100
    hi_pct = (1 - alpha / 2) * 100

    return {
        "tau_mean": float(np.mean(tau_post)),
        "tau_lo":   float(np.percentile(tau_post, lo_pct)),
        "tau_hi":   float(np.percentile(tau_post, hi_pct)),
        "tau_idx":  int(np.clip(round(float(np.median(tau_post))), 1, n - 2)),
        "tau_post": tau_post,          # ← campionamento posteriore completo
        "b1_mean":  float(np.mean(b1_post)),
        "b1_lo":    float(np.percentile(b1_post, lo_pct)),
        "b1_hi":    float(np.percentile(b1_post, hi_pct)),
        "b2_mean":  float(np.mean(b2_post)),
        "b2_lo":    float(np.percentile(b2_post, lo_pct)),
        "b2_hi":    float(np.percentile(b2_post, hi_pct)),
        "a1_mean":  float(np.mean(a1_post)),
        "nu_mean":  float(np.mean(nu_post)),     # gradi libertà StudentT
        "rho_mean": float(np.mean(rho_post)),    # AR(1) stimato
        "trace":    trace,
    }


# ─────────────────────────────────────────────────────────────────────────────
# TEST DIAGNOSTICI SULLA REGRESSIONE PIECEWISE
# ─────────────────────────────────────────────────────────────────────────────

def breusch_pagan_test(residuals, fitted):
    """
    Test di Breusch-Pagan per omoschedasticità.
    H0: varianza degli errori è costante (omoschedasticità).
    Procedura:
      1. Calcola i residui al quadrato
      2. Regressione OLS di e² sui valori fittati
      3. Statistica LM = n · R² ~ chi²(1)
    Rifiuto H0 (p < α) → eteroschedasticità.
    """
    n = len(residuals)
    e2 = residuals ** 2
    # Regressione ausiliaria: e² ~ a + b·ŷ
    X_aux = np.column_stack([np.ones(n), fitted])
    b_aux, _, _, _ = np.linalg.lstsq(X_aux, e2, rcond=None)
    e2_hat = X_aux @ b_aux
    ss_res = np.sum((e2 - e2_hat) ** 2)
    ss_tot = np.sum((e2 - e2.mean()) ** 2)
    r2_aux = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
    lm_stat = n * r2_aux
    p_val   = 1 - stats.chi2.cdf(lm_stat, df=1)
    return lm_stat, p_val


def compute_piecewise_residuals(x, y, cp_idx, b1, b2, a1):
    """
    Calcola residui e valori fittati del modello piecewise.
    Il secondo segmento ha intercetta a2 = a1 + cp*(b1-b2)
    per garantire continuità nel changepoint.
    """
    a2    = a1 + cp_idx * (b1 - b2)
    y_hat = np.concatenate([
        a1 + b1 * x[:cp_idx],
        a2 + b2 * x[cp_idx:],
    ])
    residuals = y - y_hat
    return residuals, y_hat


def regression_diagnostics(residuals, fitted, ax_res, ax_qq, ax_acf):
    """
    Popola tre assi con i classici grafici diagnostici della regressione:
      ax_res  → Residuals vs Fitted  (Breusch-Pagan per omoschedasticità)
      ax_qq   → QQ plot normale      (Shapiro-Wilk per normalità)
      ax_acf  → ACF dei residui      (Durbin-Watson per autocorrelazione)

    Restituisce un dizionario con le statistiche dei tre test.
    """
    n = len(residuals)

    # ── 1. Breusch-Pagan (omoschedasticità) ─────────────────────────────────
    lm_stat, bp_p = breusch_pagan_test(residuals, fitted)
    bp_result     = "eteroschedasticità" if bp_p < ALPHA else "omoschedasticità"
    bp_color      = "#e74c3c" if bp_p < ALPHA else "#27ae60"

    ax_res.scatter(fitted, residuals, s=22, color="#2c3e50", alpha=0.65, zorder=3)
    ax_res.axhline(0, color="red", lw=1.2, linestyle="--")
    # Linea di trend sui residui (idealmente piatta)
    if len(fitted) > 2:
        z = np.polyfit(fitted, residuals, 1)
        f_line = np.linspace(fitted.min(), fitted.max(), 100)
        ax_res.plot(f_line, np.polyval(z, f_line), color="#e67e22", lw=1.4,
                    linestyle="-", alpha=0.8, label="trend residui")
    ax_res.set_xlabel("Valori fittati  ŷ", fontsize=9)
    ax_res.set_ylabel("Residui  eᵢ", fontsize=9)
    ax_res.set_title(
        f"Residuals vs Fitted\n"
        f"Breusch-Pagan: LM={lm_stat:.3f}, p={bp_p:.4f}  → {bp_result}",
        fontsize=9, color=bp_color, fontweight="bold"
    )
    ax_res.grid(alpha=0.3)
    ax_res.tick_params(labelsize=8)

    # ── 2. Shapiro-Wilk (normalità) ──────────────────────────────────────────
    sw_stat, sw_p = stats.shapiro(residuals)
    sw_result     = "NON normale" if sw_p < ALPHA else "normale"
    sw_color      = "#e74c3c" if sw_p < ALPHA else "#27ae60"

    (osm, osr), (slope, intercept, r) = stats.probplot(residuals, dist="norm")
    ax_qq.scatter(osm, osr, s=22, color="#2c3e50", alpha=0.65, zorder=3)
    qq_line = np.array([osm[0], osm[-1]])
    ax_qq.plot(qq_line, slope * qq_line + intercept, color="#e74c3c",
               lw=1.8, linestyle="--", label="normale teorica")
    ax_qq.set_xlabel("Quantili teorici  N(0,1)", fontsize=9)
    ax_qq.set_ylabel("Quantili campionari", fontsize=9)
    ax_qq.set_title(
        f"QQ Plot — Normalità degli errori\n"
        f"Shapiro-Wilk: W={sw_stat:.4f}, p={sw_p:.4f}  → distribuzione {sw_result}",
        fontsize=9, color=sw_color, fontweight="bold"
    )
    ax_qq.grid(alpha=0.3)
    ax_qq.tick_params(labelsize=8)

    # ── 3. Durbin-Watson + ACF (autocorrelazione) ────────────────────────────
    dw_stat = durbin_watson(residuals)
    # DW ≈ 2 → no autocorrelazione; DW < 1.5 → autocorrelazione positiva
    if dw_stat < 1.5:
        dw_result = "autocorrelazione positiva"
        dw_color  = "#e74c3c"
    elif dw_stat > 2.5:
        dw_result = "autocorrelazione negativa"
        dw_color  = "#e67e22"
    else:
        dw_result = "assenza autocorrelazione"
        dw_color  = "#27ae60"

    n_lags = min(12, n // 3)
    plot_acf(residuals, ax=ax_acf, lags=n_lags,
             alpha=ALPHA, color="#2980b9", zero=False,
             title="")
    ax_acf.axhline(0, color="black", lw=0.5)
    ax_acf.set_xlabel("Lag (settimane)", fontsize=9)
    ax_acf.set_ylabel("ACF", fontsize=9)
    ax_acf.set_title(
        f"ACF Residui — Autocorrelazione\n"
        f"Durbin-Watson: DW={dw_stat:.3f}  → {dw_result}",
        fontsize=9, color=dw_color, fontweight="bold"
    )
    ax_acf.grid(alpha=0.3)
    ax_acf.tick_params(labelsize=8)

    return {
        "BP_LM":    round(lm_stat, 4),
        "BP_p":     round(bp_p, 4),
        "BP_H0":    bp_result,
        "SW_W":     round(sw_stat, 4),
        "SW_p":     round(sw_p, 4),
        "SW_H0":    sw_result,
        "DW":       round(dw_stat, 4),
        "DW_H0":    dw_result,
    }


# ─────────────────────────────────────────
# RUN PRINCIPALE
# ─────────────────────────────────────────
SERIES = [("Brent", "log_brent"), ("Benzina", "log_benzina"), ("Diesel", "log_diesel")]

results       = []
summary_rows  = []
diag_rows     = []   # ← raccoglie i test diagnostici

print(f"{'EVENTO':<28} {'SERIE':<10} {'TAU':<14} {'CI 95%':<22} {'LAG (gg)':<10} {'H0'}")
print("=" * 90)

for event_name, cfg in EVENTS.items():
    for series_name, log_col in SERIES:
        if log_col not in merged.columns:
            continue

        df = merged.loc[cfg["window_start"]:cfg["window_end"], log_col].dropna()
        if len(df) < 10:
            continue

        x_vals = np.arange(len(df), dtype=float)
        y_vals = df.values.astype(float)

        print(f"\n  MCMC Bayesiano: {event_name} | {series_name}...")
        ci = bayesian_changepoint(x_vals, y_vals)

        cp_idx  = ci["tau_idx"]
        cp_date = df.index[cp_idx]

        cp_lo_idx  = int(np.clip(round(ci["tau_lo"]),  0, len(df) - 1))
        cp_hi_idx  = int(np.clip(round(ci["tau_hi"]), 0, len(df) - 1))
        cp_lo_date = df.index[cp_lo_idx]
        cp_hi_date = df.index[cp_hi_idx]

        shock = pd.Timestamp(cfg["shock_date"])
        lag   = (cp_date - shock).days

        b1 = ci["b1_mean"]
        b2 = ci["b2_mean"]
        a1 = ci["a1_mean"]
        _, _, _, r2_1, r2_2 = fit_piecewise_ols(x_vals, y_vals, cp_idx)

        # ── Residui del modello piecewise (slope bayesiane, OLS intercette)
        residuals, y_hat = compute_piecewise_residuals(x_vals, y_vals, cp_idx, b1, b2, a1)

        h0_status = "RIFIUTATA" if lag < H0_THRESHOLD else "non rifiutata"
        ci_str    = f"[{cp_lo_date.strftime('%d %b %y')} – {cp_hi_date.strftime('%d %b %y')}]"

        print(f"{event_name:<28} {series_name:<10} {str(cp_date.date()):<14} "
              f"{ci_str:<22} {lag:<10} {h0_status}")

        results.append({
            **cfg,
            "event":      event_name,
            "series":     series_name,
            "cp_date":    cp_date,
            "cp_idx":     cp_idx,
            "lag_days":   lag,
            "b1":         b1, "b2": b2, "a1": a1,
            "dt1":        doubling_time(b1), "dt2": doubling_time(b2),
            "r2_1":       r2_1, "r2_2": r2_2,
            "df":         df, "x": x_vals,
            "ci":         ci,
            "cp_lo_date": cp_lo_date,
            "cp_hi_date": cp_hi_date,
            "residuals":  residuals,
            "y_hat":      y_hat,
        })

        summary_rows.append({
            "Evento":      event_name,
            "Serie":       series_name,
            "tau":         cp_date.date(),
            "CI_95_lo":    cp_lo_date.date(),
            "CI_95_hi":    cp_hi_date.date(),
            "Shock":       shock.date(),
            "Lag (gg)":    lag,
            "b1":          round(b1, 4),
            "b1_CI_lo":    round(ci["b1_lo"], 4),
            "b1_CI_hi":    round(ci["b1_hi"], 4),
            "b2":          round(b2, 4),
            "b2_CI_lo":    round(ci["b2_lo"], 4),
            "b2_CI_hi":    round(ci["b2_hi"], 4),
            "nu_StudentT": round(ci.get("nu_mean", np.nan), 2),
            "rho_AR1":     round(ci.get("rho_mean", np.nan), 3),
            "DT1 (gg)":    round(doubling_time(b1), 1) if doubling_time(b1) != np.inf else "inf",
            "DT2 (gg)":    round(doubling_time(b2), 1) if doubling_time(b2) != np.inf else "inf",
            "R2_pre":      round(r2_1, 3),
            "R2_post":     round(r2_2, 3),
            "H0":          h0_status,
        })


# ─────────────────────────────────────────
# LAG D = tau_retail - tau_crude
# ─────────────────────────────────────────
print("\nLAG D = tau_retail - tau_crude")
print("─" * 55)
lag_rows = []
for event_name in EVENTS:
    by_s = {r["series"]: r for r in results if r["event"] == event_name}
    if "Brent" not in by_s:
        continue
    tau_crude = by_s["Brent"]["cp_date"]
    for fuel in ["Benzina", "Diesel"]:
        if fuel not in by_s:
            continue
        D    = (by_s[fuel]["cp_date"] - tau_crude).days
        flag = "SPECULAZIONE" if D < H0_THRESHOLD else "compatibile con logistica"
        print(f"  {event_name} | {fuel}: D = {D:+d} gg → {flag}")
        lag_rows.append({
            "Evento":      event_name,
            "Carburante":  fuel,
            "tau_crude":   tau_crude.date(),
            "tau_retail":  by_s[fuel]["cp_date"].date(),
            "D (gg)":      D,
            "H0":          "RIFIUTATA" if D < H0_THRESHOLD else "non rifiutata",
        })

pd.DataFrame(lag_rows).to_csv("data/lag_results.csv", index=False)
pd.DataFrame(summary_rows).to_csv("data/table1_changepoints.csv", index=False)


# ─────────────────────────────────────────────────────────────────────────────
# PLOT A — Grafico principale (stile Casini & Roccetti 2021)
#           + PLOT B — Pannello diagnostica (3 sotto-grafici)
# ─────────────────────────────────────────────────────────────────────────────
for res in results:
    df_plot = res["df"]
    x       = res["x"]
    cp      = res["cp_idx"]
    b1, b2, a1 = res["b1"], res["b2"], res["a1"]
    a2      = a1 + cp * (b1 - b2)
    ci      = res["ci"]
    residuals = res["residuals"]
    y_hat     = res["y_hat"]

    safe_event  = (res["event"]
                   .replace(" ", "_")
                   .replace("(", "").replace(")", "")
                   .replace("/", ""))
    safe_series = res["series"].lower()

    # ══════════════════════════════════════════════════
    # FIGURA A: Regressione piecewise (stile Casini & Roccetti 2021)
    #           con campana posteriore di τ in basso (come Figure 1 del paper)
    # ══════════════════════════════════════════════════
    from scipy.stats import gaussian_kde

    # ── Layout: asse principale + asse secondario in basso per la campana
    fig_a = plt.figure(figsize=(12, 5.8))
    # Asse principale occupa ~80% dell'altezza, la campana il 20% in fondo
    gs = fig_a.add_gridspec(2, 1, height_ratios=[4, 1], hspace=0.0)
    ax     = fig_a.add_subplot(gs[0])
    ax_kde = fig_a.add_subplot(gs[1], sharex=ax)

    # ── Dati grezzi
    ax.scatter(df_plot.index, df_plot.values,
               s=20, color="black", alpha=0.60, zorder=3, label="log(prezzo)")

    # ── Rette piecewise con bande CI sui slopes (stile paper: verde pre, rosso post)
    ci_b1_lo, ci_b1_hi = ci["b1_lo"], ci["b1_hi"]
    ax.plot(df_plot.index[:cp], a1 + b1 * x[:cp],
            color="#27ae60", lw=2.5,
            label=f"Pre-shock  y = {a1:.2f} + {b1:.4f}·x")
    ax.fill_between(df_plot.index[:cp],
                    a1 + ci_b1_lo * x[:cp],
                    a1 + ci_b1_hi * x[:cp],
                    color="#27ae60", alpha=0.10)

    ci_b2_lo, ci_b2_hi = ci["b2_lo"], ci["b2_hi"]
    ax.plot(df_plot.index[cp:], a2 + b2 * x[cp:],
            color="#e74c3c", lw=2.5,
            label=f"Post-shock y = {a2:.2f} + {b2:.4f}·x")
    ax.fill_between(df_plot.index[cp:],
                    a2 + ci_b2_lo * x[cp:],
                    a2 + ci_b2_hi * x[cp:],
                    color="#e74c3c", alpha=0.10)

    # ── Campana posteriore di τ (asse ax_kde) — come nel paper Roccetti
    tau_post = ci.get("tau_post", None)
    if tau_post is not None and len(tau_post) > 10:
        # Converte da indice settimane → date matplotlib
        base_num   = mdates.date2num(df_plot.index[0])
        tau_num    = base_num + tau_post * 7   # 1 unità = 7 giorni

        kde_fn  = gaussian_kde(tau_num, bw_method=0.25)
        t_grid  = np.linspace(tau_num.min(), tau_num.max(), 400)
        density = kde_fn(t_grid)
        density /= density.max()   # normalizza 0→1

        t_dates = mdates.num2date(t_grid)
        ax_kde.fill_between(t_dates, 0, density,
                            alpha=0.50, color="#2980b9", zorder=3)
        ax_kde.plot(t_dates, density,
                    color="#1a5276", lw=1.6, zorder=4)
        ax_kde.set_ylim(0, 1.6)

    # ── Linea verticale τ e CI su entrambi gli assi
    ax.axvline(res["cp_date"], color="#2980b9", lw=2.0, linestyle="--",
               label=f"τ̂ = {res['cp_date'].date()}")
    ax.axvspan(res["cp_lo_date"], res["cp_hi_date"],
               alpha=0.13, color="#2980b9",
               label=f"CI 95%: [{res['cp_lo_date'].strftime('%d %b %y')} – "
                     f"{res['cp_hi_date'].strftime('%d %b %y')}]")
    ax_kde.axvline(res["cp_date"], color="#2980b9", lw=2.0, linestyle="--", zorder=5)
    ax_kde.axvspan(res["cp_lo_date"], res["cp_hi_date"],
                   alpha=0.13, color="#2980b9", zorder=2)

    # ── Shock date su entrambi gli assi
    ax.axvline(pd.Timestamp(res["shock_date"]), color=res["color"], lw=1.8,
               linestyle=":", label=f"Shock = {res['shock_date']}")
    ax_kde.axvline(pd.Timestamp(res["shock_date"]), color=res["color"], lw=1.8,
                   linestyle=":", zorder=5)

    # ── Decorazioni asse principale
    lag   = res["lag_days"]
    dt1_s = str(round(res["dt1"], 1)) if res["dt1"] != np.inf else "∞"
    dt2_s = str(round(res["dt2"], 1)) if res["dt2"] != np.inf else "∞"
    ax.set_title(
        f"{res['event']} — {res['series']}\n"
        f"D = {lag:+d} giorni  |  DT₁ = {dt1_s} gg  →  DT₂ = {dt2_s} gg",
        fontsize=13, fontweight="bold",
        color="#c0392b" if lag < H0_THRESHOLD else "black",
    )
    ax.set_ylabel("log(prezzo)", fontsize=12)
    ax.legend(fontsize=9, loc="upper left")
    ax.grid(alpha=0.3)
    ax.tick_params(axis="x", labelbottom=False)   # nasconde le x-tick sull'asse principale
    ax.spines["bottom"].set_visible(False)

    # ── Decorazioni asse campana (KDE)
    ax_kde.set_ylabel("p(τ)", fontsize=9, color="#1a5276")
    ax_kde.tick_params(axis="y", labelsize=7, colors="#1a5276")
    ax_kde.set_yticks([])
    ax_kde.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    ax_kde.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
    ax_kde.tick_params(axis="x", rotation=45, labelsize=9)
    ax_kde.grid(alpha=0.3)
    ax_kde.spines["top"].set_visible(False)
    ax_kde.set_xlabel("Data", fontsize=11)

    # ── Asse y destro (prezzi non-log) — come nel paper
    y_lo, y_hi = ax.get_ylim()
    ax2 = ax.twinx()
    ax2.set_ylim(np.exp(y_lo), np.exp(y_hi))
    ax2.set_ylabel("prezzo  (scala originale)", fontsize=9, color="#555555")
    ax2.tick_params(axis="y", labelsize=8, colors="#555555")
    ax2.yaxis.set_major_formatter(
        plt.FuncFormatter(lambda v, _: f"{v:.1f}")
    )

    fname_a = f"plots/02_{safe_event}_{safe_series}.png"
    fig_a.savefig(fname_a, dpi=DPI, bbox_inches="tight")
    plt.close(fig_a)
    print(f"  Salvato: {fname_a}")

    # ══════════════════════════════════════════════════
    # FIGURA B: Diagnostica della regressione
    # ══════════════════════════════════════════════════
    fig_b, axes_diag = plt.subplots(1, 3, figsize=(15, 4.5))
    fig_b.suptitle(
        f"Diagnostica della regressione piecewise — {res['event']} | {res['series']}\n"
        f"Verifica ipotesi: omoschedasticità · normalità errori · autocorrelazione",
        fontsize=11, fontweight="bold", y=1.02,
    )

    diag_stats = regression_diagnostics(
        residuals, y_hat,
        ax_res=axes_diag[0],
        ax_qq=axes_diag[1],
        ax_acf=axes_diag[2],
    )

    # Aggiungi un footer con l'interpretazione sintetica
    n_ok  = sum(1 for k, v in diag_stats.items()
                if k.endswith("_H0") and "NON" not in v and "eteroschedasticità" not in v
                and "autocorrelazione" not in v.lower().replace("assenza", ""))
    # conteggio semplificato
    issues = []
    if diag_stats["BP_p"] < ALPHA:
        issues.append("eteroschedasticità")
    if diag_stats["SW_p"] < ALPHA:
        issues.append("non-normalità errori")
    if float(diag_stats["DW"]) < 1.5 or float(diag_stats["DW"]) > 2.5:
        issues.append("autocorrelazione")

    if issues:
        footer = (f"⚠  Violazioni rilevate: {', '.join(issues)} — "
                  f"usare CI bayesiani (non classici)")
        footer_color = "#c0392b"
    else:
        footer = "✓  Tutte le ipotesi della regressione lineare sono soddisfatte"
        footer_color = "#1a7a1a"

    fig_b.text(0.5, -0.04, footer, ha="center", fontsize=10,
               color=footer_color, fontweight="bold",
               bbox=dict(boxstyle="round,pad=0.3", fc="#f8f8f8", ec=footer_color, lw=1.2))

    plt.tight_layout(pad=1.5)
    fname_b = f"plots/02_{safe_event}_{safe_series}_diagnostics.png"
    plt.savefig(fname_b, dpi=DPI, bbox_inches="tight")
    plt.close(fig_b)
    print(f"  Salvato: {fname_b}")

    # Accoda risultati diagnostica
    diag_rows.append({
        "Evento":   res["event"],
        "Serie":    res["series"],
        **diag_stats,
    })

# Salva tabella diagnostica
pd.DataFrame(diag_rows).to_csv("data/regression_diagnostics.csv", index=False)

print(f"\nScript 02 completato.")
print(f"  Metodo: Bayesian MCMC — likelihood StudentT + errori AR(1)")
print(f"  Motivazione: DW ≈ 0.003–0.04 (autocorrelazione quasi perfetta) + eteroschedasticità + non-normalità")
print(f"  CI al 95% = credible interval dalla distribuzione posteriore di τ")
print(f"  Test diagnostici salvati: data/regression_diagnostics.csv")
print(f"  Parametri posteriori aggiuntivi: nu (StudentT df) + rho (AR(1)) in table1_changepoints.csv")
print(f"\n  nu_StudentT basso (< 5) → code pesanti → Normal non adeguata → StudentT corretto")
print(f"  rho_AR1 vicino a 1     → forte autocorrelazione → AR(1) necessario")
print(f"\n  Interpretazione Breusch-Pagan (omoschedasticità):")
print(f"    p ≥ 0.05 → H0 accettata (omoschedasticità) → regressione OLS valida")
print(f"    p <  0.05 → H0 rifiutata (eteroschedasticità) → usare CI bayesiani/robusti")
print(f"\n  Interpretazione Shapiro-Wilk (normalità):")
print(f"    p ≥ 0.05 → errori normali → test t classici applicabili")
print(f"    p <  0.05 → errori non normali → affidare inferenza al MCMC posteriore")
print(f"\n  Interpretazione Durbin-Watson (autocorrelazione):")
print(f"    DW ≈ 2   → nessuna autocorrelazione → ipotesi soddisfatta")
print(f"    DW < 1.5 → autocorrelazione positiva → AR(1) necessario")
print(f"    DW > 2.5 → autocorrelazione negativa → stime meno efficienti")