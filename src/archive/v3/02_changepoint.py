"""
02_changepoint.py
==================
Quando si è rotta la dinamica dei prezzi?

Prima di chiederci se i distributori hanno aumentato i margini, vogliamo
capire QUANDO il prezzo si è mosso rispetto allo shock — e se quel movimento
fosse già iniziato prima dell'evento geopolitico (anticipazione dai mercati
futures) o solo dopo (trasmissione con ritardo).

Usiamo un modello piecewise-lineare bayesiano sui log-prezzi: due rette
raccordate al changepoint tau, con likelihood StudentT per gestire le code
pesanti che i diagnostici OLS mostrano sistematicamente su queste serie.
I diagnostici (DW, Shapiro-Wilk, Breusch-Pagan) sono prodotti come output
strutturato: saranno letti da 03_margin_hypothesis.py per scegliere i test
appropriati sul margine.

Il lag D = tau - shock_date misura l'anticipo (D < 0) o ritardo (D > 0).
Un lag fortemente negativo (prezzi già in salita settimane prima dello shock)
è consistente con mercati forward-looking che scontano aspettative geopolitiche,
non con speculazione post-shock.

Nota sul modello MCMC:
  - tau ~ Beta(2,2) su [x_min, x_max]: prior leggermente lontano dai bordi
    rispetto a Uniform, geometria più stabile per NUTS.
  - nu ~ Gamma(2, 0.1): E[nu]=20, evita nu->1 (Cauchy) e nu->inf (Normale).
  - AR(1) esplicitamente rimosso: la dipendenza sequenziale crea funnel
    geometry incompatibile con NUTS. L'autocorrelazione e' riportata come
    diagnostico (DW) ma non modellata nel changepoint — il CI posteriore
    di tau gia assorbe l'incertezza residua.

Input:
  data/dataset_merged.csv

Output:
  data/table1_changepoints.csv
  data/regression_diagnostics.csv   <- letto da script 03
  plots/02_{evento}_{serie}.png
  plots/02_{evento}_{serie}_diag.png
"""

import os, warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pymc as pm
import pytensor.tensor as pt
import arviz as az
from scipy import stats
from scipy.stats import gaussian_kde
from statsmodels.stats.stattools import durbin_watson
from statsmodels.stats.diagnostic import het_breuschpagan
from statsmodels.graphics.tsaplots import plot_acf
import statsmodels.api as sm

warnings.filterwarnings("ignore")
os.makedirs("data",  exist_ok=True)
os.makedirs("plots", exist_ok=True)

# ── Carica τ_margin da table2 se disponibile (prodotto da script 03)
# Script 02 gira prima di 03, ma se la pipeline è rieseguita table2 esiste già.
TAU_MARGIN_MAP = {}   # (ev_name, fuel_name) -> pd.Timestamp
_t2_path = "data/table2_margin_anomaly.csv"
if os.path.exists(_t2_path):
    _df_t2 = pd.read_csv(_t2_path)
    for _, _r in _df_t2.iterrows():
        if pd.notna(_r.get("tau_margin")) and str(_r["tau_margin"]) not in ("N/A","nan",""):
            TAU_MARGIN_MAP[(_r["Evento"], _r["Carburante"])] = pd.Timestamp(_r["tau_margin"])
    print(f"τ_margin caricati da table2 (run precedente): {len(TAU_MARGIN_MAP)} entry")
else:
    print("table2_margin_anomaly.csv non trovato — τ_margin non disponibile nei plot 02")
    print("(sarà aggiunto ai plot alla prossima esecuzione della pipeline completa)")

DPI         = 180
ALPHA       = 0.05
LAG_THRESH  = 30   # giorni: |D| < 30gg = trasmissione rapida
EDGE_FRAC   = 0.15 # changepoint troppo vicino ai bordi = inaffidabile

EVENTS = {
    "Ucraina (Feb 2022)": {
        "shock":   "2022-02-24",
        "win_start": "2021-10-01",
        "win_end":   "2022-07-31",
        "color":   "#e74c3c",
    },
    "Iran-Israele (Giu 2025)": {
        "shock":   "2025-06-13",
        "win_start": "2025-02-01",
        "win_end":   "2025-10-31",
        "color":   "#e67e22",
    },
    "Hormuz (Feb 2026)": {
        "shock":   "2026-02-28",
        "win_start": "2025-10-01",
        "win_end":   "2026-04-01",
        "color":   "#8e44ad",
    },
}

SERIES = {
    "Brent":   ("log_brent",   "#2166ac"),
    "Benzina": ("log_benzina", "#d6604d"),
    "Diesel":  ("log_diesel",  "#31a354"),
}

# MCMC: configurazioni per scenario (override su _default dove necessario)
_MCMC_DEFAULT = dict(draws=2000, tune=2000, chains=4,
                     target_accept=0.95, init="adapt_diag", max_treedepth=15)
MCMC_OVERRIDES = {
    # Ucraina/Brent: Rhat lievemente elevato nel run di riferimento
    "Ucraina (Feb 2022)__Brent": dict(
        tune=6000, target_accept=0.99, init="adapt_full", max_treedepth=20
    ),
}


def _mcmc_cfg(key):
    cfg = dict(_MCMC_DEFAULT)
    cfg.update(MCMC_OVERRIDES.get(key, {}))
    return cfg


# ─────────────────────────────────────────────────────────────────────────────
# Modello bayesiano piecewise-lineare
# ─────────────────────────────────────────────────────────────────────────────
def bayesian_changepoint(x: np.ndarray, y: np.ndarray, cfg: dict) -> dict:
    n      = len(x)
    sd_y   = float(np.std(y))
    mean_y = float(np.mean(y))
    x_lo, x_hi = float(x[0]), float(x[-1])
    x_rng  = max(x_hi - x_lo, 1.0)

    with pm.Model():
        tau_raw = pm.Beta("tau_raw", alpha=2, beta=2)
        tau     = pm.Deterministic("tau", x_lo + tau_raw * x_rng)

        sigma = pm.HalfNormal("sigma", sigma=sd_y)
        nu    = pm.Gamma("nu", alpha=2, beta=0.1)

        b1 = pm.Normal("b1", mu=0.0, sigma=3.0 * sd_y)
        b2 = pm.Normal("b2", mu=0.0, sigma=3.0 * sd_y)
        a1 = pm.Normal("a1", mu=mean_y, sigma=sd_y)
        a2 = pm.Deterministic("a2", a1 + tau * (b1 - b2))

        x_t  = pt.as_tensor_variable(x.astype(float))
        step = pm.math.sigmoid((x_t - tau) * 50)
        mu   = (a1 + b1 * x_t) * (1.0 - step) + (a2 + b2 * x_t) * step

        pm.StudentT("obs", nu=nu, mu=mu, sigma=sigma, observed=y)

        trace = pm.sample(
            draws=cfg["draws"], tune=cfg["tune"], chains=cfg["chains"],
            progressbar=True, random_seed=list(range(42, 42 + cfg["chains"])),
            target_accept=cfg["target_accept"],
            nuts_sampler_kwargs={"max_treedepth": cfg["max_treedepth"]},
            init=cfg["init"], return_inferencedata=True,
        )

    try:
        rhat_max = float(np.nanmax(az.rhat(trace).to_array().values.flatten()))
        ess_vals = az.ess(trace, method="bulk").to_array().values.flatten()
        ess_min  = float(np.min(ess_vals[np.isfinite(ess_vals) & (ess_vals > 0)]))
    except Exception:
        rhat_max, ess_min = np.nan, np.nan

    if np.isnan(rhat_max):
        print("  Diagnostica convergenza non disponibile")
    elif rhat_max > 1.05:
        print(f"  CONVERGENZA DUBBIA: Rhat={rhat_max:.3f} > 1.05")
    elif rhat_max > 1.01:
        print(f"  Rhat={rhat_max:.3f} (lieve non-convergenza)")
    else:
        print(f"  Convergenza ok: Rhat={rhat_max:.3f}  ESS={ess_min:.0f}")

    tau_post = trace.posterior["tau"].values.flatten()
    lo, hi   = ALPHA / 2 * 100, (1 - ALPHA / 2) * 100

    return {
        "tau_mean": float(np.mean(tau_post)),
        "tau_lo":   float(np.percentile(tau_post, lo)),
        "tau_hi":   float(np.percentile(tau_post, hi)),
        "tau_idx":  int(np.clip(round(float(np.median(tau_post))), 1, n-2)),
        "tau_post": tau_post,
        "b1_mean":  float(np.mean(trace.posterior["b1"].values.flatten())),
        "b1_lo":    float(np.percentile(trace.posterior["b1"].values.flatten(), lo)),
        "b1_hi":    float(np.percentile(trace.posterior["b1"].values.flatten(), hi)),
        "b2_mean":  float(np.mean(trace.posterior["b2"].values.flatten())),
        "b2_lo":    float(np.percentile(trace.posterior["b2"].values.flatten(), lo)),
        "b2_hi":    float(np.percentile(trace.posterior["b2"].values.flatten(), hi)),
        "a1_mean":  float(np.mean(trace.posterior["a1"].values.flatten())),
        "nu_mean":  float(np.mean(trace.posterior["nu"].values.flatten())),
        "rhat_max": rhat_max,
        "ess_min":  ess_min,
    }


# ─────────────────────────────────────────────────────────────────────────────
# [FIX critic.1] Confronto AIC: Student-T vs Skew-Normal sui residui OLS
# ─────────────────────────────────────────────────────────────────────────────
def distribution_aic_comparison(resid: np.ndarray) -> dict:
    """
    Confronta via AIC la distribuzione Student-T e la Skew-Normal MLE
    sui residui OLS piecewise.

    Motivazione: la critica metodologica (27-apr-2026) suggerisce di
    verificare se la Skewed-T si adatti meglio della Student-T al changepoint
    MCMC. Prima di passare a una likelihood più complessa (che aumenta il
    rischio di non-convergenza), si valuta l'asimmetria dei residui empirici
    tramite MLE sulla distribuzione marginalizzata.

    Se ΔAIC = AIC(SkewNorm) − AIC(StudentT) < −2 → SkewNormal preferita:
    raccomandare cambio likelihood nel modello MCMC.

    Student-T:   3 parametri liberi (df, loc, scale)
    SkewNormal:  3 parametri liberi (alpha=skewness, loc, scale)
    → stesso numero di parametri → AIC = BIC a meno della costante log(n).

    Rif: Akaike (1974) IEEE Trans. Autom. Control;
         Burnham & Anderson (2002) Model Selection and Multimodel Inference.
    """
    r = np.asarray(resid, dtype=float)
    r = r[np.isfinite(r)]
    n = len(r)

    # ── Student-t MLE ─────────────────────────────────────────────────────
    aic_t = np.nan
    try:
        params_t = stats.t.fit(r)           # (df, loc, scale)
        ll_t     = float(np.sum(stats.t.logpdf(r, *params_t)))
        aic_t    = -2.0 * ll_t + 2.0 * 3   # k = 3
    except Exception:
        pass

    # ── SkewNormal MLE ────────────────────────────────────────────────────
    aic_sn = np.nan
    alpha_sn = np.nan
    try:
        params_sn = stats.skewnorm.fit(r)   # (a, loc, scale)
        ll_sn     = float(np.sum(stats.skewnorm.logpdf(r, *params_sn)))
        aic_sn    = -2.0 * ll_sn + 2.0 * 3
        alpha_sn  = float(params_sn[0])     # skewness parameter
    except Exception:
        pass

    delta_aic = float(aic_sn - aic_t) if (np.isfinite(aic_sn) and np.isfinite(aic_t)) else np.nan

    # ── Skewness empirica (test di Fisher) ────────────────────────────────
    skew_emp, skew_p = np.nan, np.nan
    try:
        skew_emp = float(stats.skew(r))
        if n >= 8:
            skew_p = float(stats.skewtest(r).pvalue)
    except Exception:
        pass

    # ── Raccomandazione ───────────────────────────────────────────────────
    use_skewed = False
    if np.isfinite(delta_aic):
        if delta_aic < -2.0:
            rec = (f"⚠  ΔAIC={delta_aic:.1f}: SkewNormal migliore di {abs(delta_aic):.1f} AIC. "
                   f"α_skew={alpha_sn:.3f}. Considerare Skewed-T nel MCMC.")
            use_skewed = True
        elif delta_aic > 2.0:
            rec = f"✓  ΔAIC={delta_aic:.1f}: Student-T adeguato (SkewNormal peggiore)."
        else:
            rec = f"~  ΔAIC={delta_aic:.1f}: modelli equivalenti (|ΔAIC| ≤ 2). Student-T sufficiente."
    else:
        rec = "ΔAIC non disponibile (fit MLE fallito)"

    return {
        "AIC_StudentT":         round(float(aic_t), 2) if np.isfinite(aic_t) else None,
        "AIC_SkewNormal":       round(float(aic_sn), 2) if np.isfinite(aic_sn) else None,
        "delta_AIC_Skew_T":     round(delta_aic, 2) if np.isfinite(delta_aic) else None,
        "alpha_SkewNorm":       round(alpha_sn, 3) if np.isfinite(alpha_sn) else None,
        "skewness_empirica":    round(skew_emp, 3) if np.isfinite(skew_emp) else None,
        "skewtest_p":           round(skew_p, 4) if np.isfinite(skew_p) else None,
        "use_skewed_t":         use_skewed,
        "raccomandazione_dist": rec,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Diagnostici OLS piecewise (motivano la scelta dei test in script 03)
# ─────────────────────────────────────────────────────────────────────────────
def ols_diagnostics(x: np.ndarray, y: np.ndarray, cp_idx: int) -> dict:
    """
    Calcola Breusch-Pagan, Shapiro-Wilk, Durbin-Watson e confronto AIC
    distribuzionale (StudentT vs SkewNormal) sui residui OLS piecewise.
    Il risultato viene salvato in regression_diagnostics.csv
    e letto da 03_margin_hypothesis.py per scegliere i test appropriati.
    """
    cp   = cp_idx
    s1, i1, _, *_ = stats.linregress(x[:cp], y[:cp])
    s2, _,  _, *_ = stats.linregress(x[cp:], y[cp:])
    i2   = i1 + cp * (s1 - s2)
    y_hat = np.concatenate([i1 + s1 * x[:cp], i2 + s2 * x[cp:]])
    resid = y - y_hat

    try:
        _, bp_p, _, _ = het_breuschpagan(resid, sm.add_constant(y_hat))
    except Exception:
        bp_p = np.nan
    try:
        _, sw_p = stats.shapiro(resid[:5000])
    except Exception:
        sw_p = np.nan
    dw = durbin_watson(resid)

    # [FIX critic.1] Confronto AIC distribuzionale
    aic_info = distribution_aic_comparison(resid)

    return {"BP_p": bp_p, "SW_p": sw_p, "DW": dw,
            "resid": resid, "y_hat": y_hat, **aic_info}


# ─────────────────────────────────────────────────────────────────────────────
# Carica dati e lancia
# ─────────────────────────────────────────────────────────────────────────────
merged = pd.read_csv("data/dataset_merged.csv", index_col=0, parse_dates=True)
print(f"Dataset: {len(merged)} settimane | "
      f"{merged.index[0].date()} – {merged.index[-1].date()}\n")

table1_rows = []
diag_rows   = []

plt.rcParams.update({
    "font.family": "serif", "font.serif": ["Times New Roman","DejaVu Serif"],
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": "#e8e8e8", "grid.linewidth": 0.6,
})

for ev_name, cfg in EVENTS.items():
    shock    = pd.Timestamp(cfg["shock"])
    ev_color = cfg["color"]
    safe_ev  = ev_name.replace(" ","_").replace("(","").replace(")","").replace("/","").replace(",","")

    for ser_name, (log_col, ser_color) in SERIES.items():
        if log_col not in merged.columns:
            continue

        df_ev = merged.loc[cfg["win_start"]:cfg["win_end"], log_col].dropna()
        if len(df_ev) < 8:
            print(f"  SKIP {ev_name}|{ser_name}: <8 osservazioni")
            continue

        print(f"\nMCMC -> {ev_name} | {ser_name} ({len(df_ev)} punti)")
        x = np.arange(len(df_ev), dtype=float)
        y = df_ev.values

        ci = bayesian_changepoint(x, y, _mcmc_cfg(f"{ev_name}__{ser_name}"))

        cp       = ci["tau_idx"]
        cp_date  = df_ev.index[0] + pd.Timedelta(weeks=ci["tau_idx"])
        cp_lo    = df_ev.index[0] + pd.Timedelta(weeks=max(0, round(ci["tau_lo"])))
        cp_hi    = df_ev.index[0] + pd.Timedelta(weeks=min(len(df_ev)-1, round(ci["tau_hi"])))
        lag_days = (cp_date - shock).days

        # OLS piecewise per R² e slope leggibili
        def _lr(xv, yv):
            if len(xv) < 2: return 0.,0.,0.
            s,i,r,*_ = stats.linregress(xv,yv); return s,i,r**2
        b1_ols, a1_ols, r2_pre  = _lr(x[:cp], y[:cp])
        b2_ols, _,      r2_post = _lr(x[cp:], y[cp:])

        dt1 = np.log(2)/(b1_ols/7) if b1_ols > 0 else np.inf
        dt2 = np.log(2)/(b2_ols/7) if b2_ols > 0 else np.inf

        # Diagnostici OLS (output strutturato per script 03)
        diag = ols_diagnostics(x, y, cp)
        diag_rows.append({
            "Evento": ev_name, "Serie": ser_name,
            "BP_p": round(float(diag["BP_p"]),4) if not np.isnan(diag["BP_p"]) else None,
            "SW_p": round(float(diag["SW_p"]),4) if not np.isnan(diag["SW_p"]) else None,
            "DW":   round(diag["DW"],3),
            # [FIX critic.1] AIC confronto distribuzionale
            "AIC_StudentT":       diag.get("AIC_StudentT"),
            "AIC_SkewNormal":     diag.get("AIC_SkewNormal"),
            "delta_AIC_Skew_T":   diag.get("delta_AIC_Skew_T"),
            "alpha_SkewNorm":     diag.get("alpha_SkewNorm"),
            "skewness_empirica":  diag.get("skewness_empirica"),
            "skewtest_p":         diag.get("skewtest_p"),
            "use_skewed_t":       diag.get("use_skewed_t", False),
            "dist_raccomandazione": diag.get("raccomandazione_dist", ""),
        })
        # Stampa raccomandazione distribuzionale
        print(f"  Distr. AIC: {diag.get('raccomandazione_dist', 'N/A')}")

        rhat_flag = ("CONVERGENZA DUBBIA" if ci["rhat_max"] > 1.05
                     else "ok" if not np.isnan(ci["rhat_max"]) else "N/A")

        table1_rows.append({
            "Evento":      ev_name,
            "Serie":       ser_name,
            "tau":         cp_date.date(),
            "CI_95_lo":    cp_lo.date(),
            "CI_95_hi":    cp_hi.date(),
            "Lag (gg)":    lag_days,
            "H0_rif":      "SI" if abs(lag_days) < LAG_THRESH else "NO",
            "b1_OLS":      round(b1_ols,5),
            "b2_OLS":      round(b2_ols,5),
            "DT1 (gg)":    round(dt1,1) if dt1 != np.inf else "inf",
            "DT2 (gg)":    round(dt2,1) if dt2 != np.inf else "inf",
            "R2_pre":      round(r2_pre,4),
            "R2_post":     round(r2_post,4),
            "nu_StudentT": round(ci["nu_mean"],2),
            "rhat_max":    round(ci["rhat_max"],3) if not np.isnan(ci["rhat_max"]) else None,
            "ess_min":     round(ci["ess_min"],0)  if not np.isnan(ci["ess_min"])  else None,
            "rhat_flag":   rhat_flag,
            "BP_p":        diag_rows[-1]["BP_p"],
            "SW_p":        diag_rows[-1]["SW_p"],
            "DW":          diag_rows[-1]["DW"],
        })

        print(f"  tau={cp_date.date()}  lag={lag_days:+d}gg  "
              f"DW={diag['DW']:.2f}  BP_p={diag['BP_p']:.4f}  "
              f"SW_p={diag['SW_p']:.4f}")

        # ── Plot principale con KDE posteriore ───────────────────────────
        a2_ols = a1_ols + cp * (b1_ols - b2_ols)

        fig = plt.figure(figsize=(12, 5.8))
        gs  = fig.add_gridspec(2, 1, height_ratios=[4,1], hspace=0.0)
        ax  = fig.add_subplot(gs[0])
        axk = fig.add_subplot(gs[1], sharex=ax)

        ax.scatter(df_ev.index, y, s=18, color="black", alpha=0.55, zorder=3,
                   label="log(prezzo)")
        ax.plot(df_ev.index[:cp], a1_ols + ci["b1_mean"]*x[:cp],
                color="#27ae60", lw=2.5, label=f"Pre-shock β={ci['b1_mean']:.4f}")
        ax.fill_between(df_ev.index[:cp],
                        a1_ols + ci["b1_lo"]*x[:cp], a1_ols + ci["b1_hi"]*x[:cp],
                        color="#27ae60", alpha=0.10)
        ax.plot(df_ev.index[cp:], a2_ols + ci["b2_mean"]*x[cp:],
                color="#e74c3c", lw=2.5, label=f"Post-shock β={ci['b2_mean']:.4f}")
        ax.fill_between(df_ev.index[cp:],
                        a2_ols + ci["b2_lo"]*x[cp:], a2_ols + ci["b2_hi"]*x[cp:],
                        color="#e74c3c", alpha=0.10)
        ax.axvline(cp_date, color="#2980b9", lw=2.0, ls="--",
                   label=f"tau={cp_date.date()}")
        ax.axvspan(cp_lo, cp_hi, alpha=0.13, color="#2980b9",
                   label=f"CI 95%: {cp_lo.strftime('%d %b %y')} – {cp_hi.strftime('%d %b %y')}")
        ax.axvline(shock, color=ev_color, lw=1.8, ls=":",
                   label=f"Shock {cfg['shock']}")

        # τ_margin (da script 03, run precedente) — se disponibile
        _tm = TAU_MARGIN_MAP.get((ev_name, ser_name))
        if _tm is not None and df_ev.index[0] <= _tm <= df_ev.index[-1]:
            ax.axvline(_tm, color=ev_color, lw=1.6, ls=(0,(3,1,1,1)),
                       alpha=0.75, label=f"τ_margin={_tm.strftime('%d %b %y')}")
            axk.axvline(_tm, color=ev_color, lw=1.6, ls=(0,(3,1,1,1)), alpha=0.75)

        ax.set_title(
            f"{ev_name} — {ser_name}   |   D={lag_days:+d}gg   "
            f"DT1={round(dt1,0) if dt1!=np.inf else 'inf'}gg -> "
            f"DT2={round(dt2,0) if dt2!=np.inf else 'inf'}gg   "
            f"nu={ci['nu_mean']:.1f}",
            fontsize=11, fontweight="bold",
            color="#c0392b" if abs(lag_days) < LAG_THRESH else "black",
        )
        ax.set_ylabel("log(prezzo)", fontsize=10)
        ax.legend(fontsize=8, loc="upper left")
        ax.tick_params(axis="x", labelbottom=False)

        # KDE posteriore tau
        base_num  = mdates.date2num(df_ev.index[0])
        tau_num   = base_num + ci["tau_post"] * 7
        kde_fn    = gaussian_kde(tau_num, bw_method=0.25)
        t_grid    = np.linspace(tau_num.min(), tau_num.max(), 400)
        density   = kde_fn(t_grid) / kde_fn(t_grid).max()
        axk.fill_between(mdates.num2date(t_grid), 0, density,
                         alpha=0.5, color="#2980b9")
        axk.axvline(cp_date, color="#2980b9", lw=2.0, ls="--")
        axk.axvspan(cp_lo, cp_hi, alpha=0.13, color="#2980b9")
        axk.axvline(shock, color=ev_color, lw=1.8, ls=":")
        axk.set_ylim(0, 1.6); axk.set_yticks([]); axk.set_ylabel("p(tau)", fontsize=9)
        axk.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
        axk.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
        axk.tick_params(axis="x", rotation=45, labelsize=9)

        safe_s = ser_name.lower()
        fpath  = f"plots/02_{safe_ev}_{safe_s}.png"
        fig.tight_layout(pad=1.5)
        fig.savefig(fpath, dpi=DPI, bbox_inches="tight")
        plt.close(fig)
        print(f"  Salvato: {fpath}")

        # ── Plot diagnostica ─────────────────────────────────────────────
        fig_d, axes_d = plt.subplots(1, 3, figsize=(14, 4))
        fig_d.suptitle(
            f"Diagnostica OLS piecewise — {ev_name} | {ser_name}\n"
            "(motivano la scelta dei test in 03_margin_hypothesis.py)",
            fontsize=10, fontweight="bold")

        # Residui vs fitted
        axes_d[0].scatter(diag["y_hat"], diag["resid"], s=14, alpha=0.5, color="#3498db")
        axes_d[0].axhline(0, color="red", lw=1.5, ls="--")
        bp_s = f"BP p={diag['BP_p']:.4f} ({'eterosch.' if not np.isnan(diag['BP_p']) and diag['BP_p']<ALPHA else 'ok'})"
        axes_d[0].set_title(f"Residui vs Fitted\n{bp_s}", fontsize=9)

        # QQ plot
        stats.probplot(diag["resid"], dist="norm", plot=axes_d[1])
        sw_s = f"SW p={diag['SW_p']:.4f} ({'non norm.' if not np.isnan(diag['SW_p']) and diag['SW_p']<ALPHA else 'ok'})"
        axes_d[1].set_title(f"QQ Plot\n{sw_s}", fontsize=9)

        # ACF
        plot_acf(diag["resid"], lags=min(20,len(diag["resid"])//2-1),
                 ax=axes_d[2], zero=False)
        dw_s = f"DW={diag['DW']:.3f} ({'autocorr.' if diag['DW']<1.5 else 'ok'})"
        axes_d[2].set_title(f"ACF Residui\n{dw_s}", fontsize=9)

        issues = []
        if not np.isnan(diag["BP_p"]) and diag["BP_p"] < ALPHA:
            issues.append("eteroschedasticita")
        if not np.isnan(diag["SW_p"]) and diag["SW_p"] < ALPHA:
            issues.append("non-normalita")
        if diag["DW"] < 1.5:
            issues.append(f"autocorrelazione (DW={diag['DW']:.2f})")
        footer = (f"Violazioni: {', '.join(issues)} -> test non parametrici + HAC in script 03"
                  if issues else "Ipotesi OLS soddisfatte")
        fig_d.text(0.5, -0.02, footer, ha="center", fontsize=9,
                   color="#c0392b" if issues else "#1a7a1a",
                   bbox=dict(boxstyle="round,pad=0.3", fc="#f8f8f8",
                             ec="#c0392b" if issues else "#1a7a1a", lw=1.2))
        fig_d.tight_layout(pad=1.5)
        fig_d.savefig(f"plots/02_{safe_ev}_{safe_s}_diag.png", dpi=DPI, bbox_inches="tight")
        plt.close(fig_d)


# ─────────────────────────────────────────────────────────────────────────────
# Salva output
# ─────────────────────────────────────────────────────────────────────────────
pd.DataFrame(table1_rows).to_csv("data/table1_changepoints.csv", index=False)
pd.DataFrame(diag_rows).to_csv("data/regression_diagnostics.csv", index=False)

print(f"\nSalvato: data/table1_changepoints.csv ({len(table1_rows)} righe)")
print(f"Salvato: data/regression_diagnostics.csv ({len(diag_rows)} righe)")
print("\nRiepilogo lag D (changepoint vs shock):")
for r in table1_rows:
    print(f"  {r['Evento'][:30]} | {r['Serie']:7}: D={r['Lag (gg)']:+4d}gg  "
          f"DW={r['DW']}  SW_p={r['SW_p']}  -> H0 {r['H0_rif']}")

print("\nScript 02 completato.")
plt.rcParams.update(plt.rcParamsDefault)