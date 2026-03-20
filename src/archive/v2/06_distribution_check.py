"""
06_distribution_check.py
=========================
Verifica se la StudentT è la distribuzione corretta per i residui del modello
bayesiano piecewise (script 02) e per i crack spread (script 03).

LOGICA DEL TEST
----------------
Per ogni scenario (evento × serie) confrontiamo quattro distribuzioni
usando Maximum Likelihood + AIC (= 2k − 2·logL, più basso = migliore):

  1. Normale      — nessuna coda pesante, nessuna asimmetria
  2. StudentT     — code pesanti (scelta attuale in script 02)
  3. Skew-Normal  — asimmetria, code leggere
  4. Skewed-T     — asimmetria + code pesanti (più generale)
     Implementata come Azzalini skew-t via scipy.stats.t + trasformazione
     oppure con nct (non-central t) come approssimazione computazionale.

REGOLA DI DECISIONE (AIC differenze vs StudentT):
  - ΔAIC(skewnorm - t)  < -2  AND  ΔAIC(skewt - t) ≥ 0  → Skew-Normal
  - ΔAIC(skewt   - t)   < -2  AND  ΔAIC(skewnorm - t) ≥ 0 → Skewed-T
  - ΔAIC(skewt   - t)   < -2  AND  ΔAIC(skewnorm - t) < -2 → Skewed-T
    (le code pesanti dominano; la skew-t generalizza la skew-normal)
  - Altrimenti: StudentT attuale è adeguata (o la Normale se è lei la migliore)

NOTA SULLA SCELTA Skew-Normal vs Skewed-T
-------------------------------------------
Se i residui mostrano sia asimmetria (|skewness| > 0.3) sia code pesanti
(kurtosi excess > 1), la Skewed-T è preferibile perché generalizza entrambe:
  - α = 0 → StudentT simmetrica
  - ν → ∞ → Skew-Normal

In pratica per i margini petroliferi la Skewed-T è quasi sempre la scelta
conservativa corretta, ma costa un parametro extra e può essere
indistinguibile dalla StudentT con pochi dati (< 50 osservazioni).

Input:
  data/dataset_merged.csv                     (log-prezzi, per residui OLS)
  data/dataset_merged_with_futures.csv        (crack spread, per margini)
  data/table1_changepoints.csv               (changepoint tau per ogni scenario)

Output:
  data/distribution_check.csv               (AIC, parametri, raccomandazione)
  plots/06_distrib_{evento}_{serie}.png      (QQ-plot × 4 distribuzioni)
  plots/06_distrib_summary.png               (heatmap ΔAIC per tutti gli scenari)
"""

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy import stats
from scipy.stats import (
    norm, t as t_dist, skewnorm, johnsonsu,
    shapiro, skewtest, kurtosistest,
)
from scipy.optimize import minimize

warnings.filterwarnings("ignore")
os.makedirs("data",  exist_ok=True)
os.makedirs("plots", exist_ok=True)

DPI   = 180
ALPHA = 0.05

# ─────────────────────────────────────────────────────────────────────────────
# Implementazione Skewed-T (Azzalini & Capitanio 2003)
# ─────────────────────────────────────────────────────────────────────────────
# Scipy non include la skewed-t di Azzalini direttamente.
# Usiamo la rappresentazione: X = μ + σ · Z  dove
#   Z | U>0 ~ t_ν   con U ~ Skew-Normal(α)
# Approssimata computazionalmente con la distribuzione nct (non-central t)
# che condivide la stessa forma funzionale.
#
# Per il fit usiamo invece la Fernandez-Steel (1998) che è implementabile
# con MLE esplicito e ha un parametro di asimmetria γ > 0:
#   f(x;ν,γ) = 2/(γ+1/γ) · [t_ν(γx) se x<0, t_ν(x/γ) se x≥0]
# γ=1 → StudentT simmetrica, γ>1 → destra pesante.

def skewt_logpdf(x: np.ndarray, nu: float, gamma: float,
                  mu: float, sigma: float) -> np.ndarray:
    """Fernandez-Steel skewed-t log-pdf. gamma>0, nu>2."""
    z = (x - mu) / sigma
    c = 2.0 / (gamma + 1.0 / gamma)
    lp = np.where(
        z < 0,
        np.log(c) + t_dist.logpdf(gamma * z, df=nu),
        np.log(c) + t_dist.logpdf(z / gamma, df=nu),
    ) - np.log(sigma)
    return lp


def skewt_negloglik(params, x):
    nu, gamma, mu, sigma = params
    if nu <= 2 or gamma <= 0 or sigma <= 0:
        return 1e10
    ll = skewt_logpdf(x, nu, gamma, mu, sigma).sum()
    return -ll if np.isfinite(ll) else 1e10


def fit_skewt(x: np.ndarray):
    """Fit Fernandez-Steel skewed-t via MLE. Ritorna (params, loglik, AIC)."""
    x = np.asarray(x, dtype=float)
    # Initial guess: usa momentI
    mu0, sigma0 = np.mean(x), np.std(x, ddof=1)
    nu0    = 10.0
    gamma0 = 1.0 + 0.5 * stats.skew(x)  # asimmetria empirica come prior
    gamma0 = max(0.3, min(gamma0, 5.0))

    res = minimize(
        skewt_negloglik,
        x0=[nu0, gamma0, mu0, sigma0],
        args=(x,),
        method="Nelder-Mead",
        options={"maxiter": 5000, "xatol": 1e-6, "fatol": 1e-6},
    )
    nu, gamma, mu, sigma = res.x
    if nu <= 2 or sigma <= 0 or gamma <= 0:
        return None, -np.inf, np.inf

    ll  = -res.fun
    aic = 2 * 4 - 2 * ll   # k=4 parametri
    return {"nu": nu, "gamma": gamma, "mu": mu, "sigma": sigma}, ll, aic


# ─────────────────────────────────────────────────────────────────────────────
# Fit generico per le distribuzioni scipy
# ─────────────────────────────────────────────────────────────────────────────
def fit_scipy(dist, x: np.ndarray):
    """Fit MLE con scipy, ritorna (params_dict, loglik, AIC)."""
    try:
        params = dist.fit(x)
        ll     = dist.logpdf(x, *params).sum()
        k      = len(params)
        aic    = 2 * k - 2 * ll
        return params, ll, aic
    except Exception:
        return None, -np.inf, np.inf


# ─────────────────────────────────────────────────────────────────────────────
# Analisi di una serie di residui
# ─────────────────────────────────────────────────────────────────────────────
DIST_REGISTRY = {
    "Normale":    norm,
    "StudentT":   t_dist,
    "Skew-Normal": skewnorm,
}


def analyse_residuals(resid: np.ndarray, label: str, plot_path: str) -> dict:
    """
    Fit quattro distribuzioni, produce QQ-plot comparativi,
    ritorna dizionario con AIC e raccomandazione.
    """
    resid = np.asarray(resid, dtype=float)
    resid = resid[np.isfinite(resid)]
    n     = len(resid)

    if n < 12:
        return {"label": label, "n": n, "warning": "troppo pochi dati", "raccomandazione": "N/A"}

    # ── Test descrittivi ──────────────────────────────────────────────────────
    sk   = float(stats.skew(resid))
    kurt = float(stats.kurtosis(resid))      # excess kurtosis (Gaussiana = 0)
    _, sw_p = shapiro(resid[:5000])
    sk_p    = skewtest(resid).pvalue  if n >= 8  else np.nan
    ku_p    = kurtosistest(resid).pvalue if n >= 20 else np.nan

    # ── Fit distribuzioni ─────────────────────────────────────────────────────
    results_fit = {}
    for name, dist in DIST_REGISTRY.items():
        params, ll, aic = fit_scipy(dist, resid)
        results_fit[name] = {"params": params, "ll": ll, "aic": aic}

    # Skewed-T separata (implementazione custom)
    st_params, st_ll, st_aic = fit_skewt(resid)
    results_fit["Skewed-T"] = {"params": st_params, "ll": st_ll, "aic": st_aic}

    # ── AIC comparison ────────────────────────────────────────────────────────
    aic_vals = {k: v["aic"] for k, v in results_fit.items()}
    best     = min(aic_vals, key=aic_vals.get)
    aic_t    = aic_vals.get("StudentT", np.inf)

    delta_skn = aic_vals.get("Skew-Normal", np.inf) - aic_t
    delta_skt = aic_vals.get("Skewed-T",    np.inf) - aic_t
    delta_n   = aic_vals.get("Normale",     np.inf) - aic_t

    # Regola di decisione
    if delta_skt < -2 and delta_skn < -2:
        raccomandazione = "Skewed-T"          # asimmetria + code pesanti
        motivazione     = (f"Skewed-T migliore di StudentT di ΔAIC={-delta_skt:.1f} "
                           f"e di Skew-Normal di ΔAIC={delta_skn - delta_skt:.1f}. "
                           f"Residui: sk={sk:.2f} kurt={kurt:.2f}")
    elif delta_skt < -2 and delta_skn >= -2:
        raccomandazione = "Skewed-T"          # code pesanti asimmetriche dominano
        motivazione     = (f"Code pesanti asimmetriche: Skewed-T ΔAIC={-delta_skt:.1f} "
                           f"su StudentT. Skew-Normal non sufficiente (ΔAIC={delta_skn:.1f}). "
                           f"sk={sk:.2f} kurt={kurt:.2f}")
    elif delta_skn < -2 and delta_skt >= -2:
        raccomandazione = "Skew-Normal"       # asimmetria ma code leggere
        motivazione     = (f"Asimmetria dominante, code non pesanti: "
                           f"Skew-Normal ΔAIC={-delta_skn:.1f} su StudentT. "
                           f"sk={sk:.2f} kurt={kurt:.2f}")
    elif delta_n < -2:
        raccomandazione = "Normale"           # StudentT inutilmente flessibile
        motivazione     = (f"Normale sufficiente (ΔAIC={-delta_n:.1f} su StudentT). "
                           f"sk={sk:.2f} kurt={kurt:.2f} ≈ 0")
    else:
        raccomandazione = "StudentT (OK)"     # scelta attuale confermata
        motivazione     = (f"StudentT non dominata da alternative "
                           f"(ΔAIC skewnorm={delta_skn:.1f}, skewt={delta_skt:.1f}). "
                           f"sk={sk:.2f} kurt={kurt:.2f}")

    # ── Plot QQ × 4 + istogramma ─────────────────────────────────────────────
    fig = plt.figure(figsize=(16, 10))
    fig.suptitle(
        f"Distributional check — {label}\n"
        f"n={n}  skewness={sk:.3f}  excess_kurt={kurt:.3f}  "
        f"SW_p={sw_p:.4f}  →  Raccomandazione: {raccomandazione}",
        fontsize=11, fontweight="bold",
    )
    gs_main = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.35)

    # Istogramma con fit
    ax_hist = fig.add_subplot(gs_main[0, :])
    ax_hist.hist(resid, bins=30, density=True, alpha=0.5, color="#95a5a6",
                 edgecolor="white", label="Residui empirici")
    x_grid = np.linspace(resid.min() - 0.5 * resid.std(),
                         resid.max() + 0.5 * resid.std(), 400)

    colors = {"Normale": "#3498db", "StudentT": "#e74c3c",
              "Skew-Normal": "#27ae60", "Skewed-T": "#e67e22"}
    lw_map = {k: (3.0 if k == best else 1.5) for k in colors}

    for name, color in colors.items():
        rf = results_fit[name]
        if rf["aic"] == np.inf:
            continue
        if name == "Skewed-T" and rf["params"] is not None:
            p = rf["params"]
            pdf_vals = np.exp(skewt_logpdf(x_grid, p["nu"], p["gamma"],
                                           p["mu"], p["sigma"]))
        else:
            pdf_vals = DIST_REGISTRY[name].pdf(x_grid, *rf["params"])
        aic_label = f"{name}  AIC={rf['aic']:.1f}"
        if name == best:
            aic_label += " ★"
        ax_hist.plot(x_grid, pdf_vals, color=color, lw=lw_map[name],
                     label=aic_label)

    ax_hist.set_title("Istogramma residui + PDF fittate  (★ = AIC minimo)", fontsize=10)
    ax_hist.legend(fontsize=8, ncol=2)
    ax_hist.set_xlabel("Residuo")
    ax_hist.set_ylabel("Densità")

    # QQ-plot per le prime 3 distribuzioni scipy (la skewed-t la facciamo
    # con i quantili teorici calcolati tramite CDF numerica)
    qq_dists = [
        ("Normale",    norm,       "#3498db", results_fit["Normale"]["params"]),
        ("StudentT",   t_dist,     "#e74c3c", results_fit["StudentT"]["params"]),
        ("Skew-Normal", skewnorm,  "#27ae60", results_fit["Skew-Normal"]["params"]),
    ]

    resid_sorted = np.sort(resid)
    probs = (np.arange(1, n + 1) - 0.5) / n   # plotting positions

    for col_idx, (name, dist_obj, color, params) in enumerate(qq_dists):
        ax_qq = fig.add_subplot(gs_main[1, col_idx])
        if params is None:
            ax_qq.set_title(f"QQ — {name}\n(fit fallito)", fontsize=9)
            continue
        q_theor = dist_obj.ppf(probs, *params)
        ax_qq.scatter(q_theor, resid_sorted, s=12, alpha=0.6, color=color)
        lo_q = min(q_theor[np.isfinite(q_theor)].min(), resid_sorted.min())
        hi_q = max(q_theor[np.isfinite(q_theor)].max(), resid_sorted.max())
        ax_qq.plot([lo_q, hi_q], [lo_q, hi_q], "k--", lw=1.2, alpha=0.7)
        star = " ★" if name == best else ""
        aic_str = f"AIC={results_fit[name]['aic']:.1f}"
        border_color = "#c0392b" if name == best else "#888"
        ax_qq.set_title(f"QQ — {name}{star}\n{aic_str}",
                        fontsize=9, color=border_color,
                        fontweight="bold" if name == best else "normal")
        ax_qq.set_xlabel("Quantili teorici", fontsize=8)
        ax_qq.set_ylabel("Quantili empirici", fontsize=8)
        ax_qq.grid(alpha=0.3)
        for spine in ax_qq.spines.values():
            spine.set_edgecolor(border_color)
            spine.set_linewidth(2.0 if name == best else 0.8)

    fig.text(0.5, 0.01,
             f"Motivazione: {motivazione}",
             ha="center", fontsize=8.5, color="#2c3e50",
             bbox=dict(boxstyle="round,pad=0.4", fc="#fdfcf0",
                       ec="#e67e22" if raccomandazione != "StudentT (OK)" else "#27ae60",
                       lw=1.2))

    fig.savefig(plot_path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  Salvato: {plot_path}")

    return {
        "label":             label,
        "n":                 n,
        "skewness":          round(sk, 4),
        "excess_kurtosis":   round(kurt, 4),
        "SW_p":              round(sw_p, 4),
        "skewtest_p":        round(sk_p, 4) if not np.isnan(sk_p) else None,
        "kurttest_p":        round(ku_p, 4) if not np.isnan(ku_p) else None,
        "AIC_Normale":       round(aic_vals["Normale"],    2),
        "AIC_StudentT":      round(aic_vals["StudentT"],   2),
        "AIC_SkewNormal":    round(aic_vals["Skew-Normal"],2),
        "AIC_SkewedT":       round(aic_vals["Skewed-T"],   2),
        "ΔAIC_skewnorm_vs_t":round(delta_skn, 2),
        "ΔAIC_skewt_vs_t":   round(delta_skt, 2),
        "best_distribution": best,
        "raccomandazione":   raccomandazione,
        "motivazione":       motivazione,
    }


# ─────────────────────────────────────────────────────────────────────────────
# 1.  Residui piecewise OLS sui log-prezzi (contesto script 02)
# ─────────────────────────────────────────────────────────────────────────────
print("=" * 70)
print("PARTE 1 — Residui OLS piecewise sui log-prezzi (contesto script 02)")
print("=" * 70)

EVENTS_02 = {
    "Ucraina_Feb2022":    {"shock": "2022-02-24", "win_start": "2021-10-01", "win_end": "2022-07-31"},
    "Iran-Israele_Giu25": {"shock": "2025-06-13", "win_start": "2025-02-01", "win_end": "2025-10-31"},
    "Hormuz_Feb2026":     {"shock": "2026-02-28", "win_start": "2025-10-01", "win_end": "2026-04-01"},
}
SERIES_02 = {
    "Brent":   "log_brent",
    "Benzina": "log_benzina",
    "Diesel":  "log_diesel",
}

all_check_rows = []

merged = None
if os.path.exists("data/dataset_merged.csv"):
    merged = pd.read_csv("data/dataset_merged.csv", index_col=0, parse_dates=True)
    print(f"  Dataset: {len(merged)} settimane\n")
else:
    print("  data/dataset_merged.csv non trovato — parte 1 saltata\n")

# Carica changepoint tau da table1 (se disponibile)
tau_map = {}   # (ev_key, ser_key) -> cp_idx (float settimane)
if os.path.exists("data/table1_changepoints.csv"):
    df_t1 = pd.read_csv("data/table1_changepoints.csv")
    for _, row in df_t1.iterrows():
        ev_raw = row["Evento"]
        # Mappa da nome lungo a chiave breve usata in EVENTS_02
        for key in EVENTS_02:
            # match flessibile
            if key.split("_")[0].lower() in ev_raw.lower():
                ser = row["Serie"]
                if ser in SERIES_02:
                    tau_map[(key, ser)] = pd.Timestamp(str(row["tau"]))
    print(f"  Changepoint tau caricati da table1: {len(tau_map)} entry")

if merged is not None:
    for ev_key, ev_cfg in EVENTS_02.items():
        for ser_name, log_col in SERIES_02.items():
            if log_col not in merged.columns:
                continue

            df_ev = merged.loc[ev_cfg["win_start"]:ev_cfg["win_end"], log_col].dropna()
            if len(df_ev) < 12:
                print(f"  SKIP {ev_key}|{ser_name}: <12 obs")
                continue

            x = np.arange(len(df_ev), dtype=float)
            y = df_ev.values

            # Changepoint: usa tau da table1 se disponibile, altrimenti metà
            tau_ts = tau_map.get((ev_key, ser_name))
            if tau_ts is not None and df_ev.index[0] <= tau_ts <= df_ev.index[-1]:
                idx_list = df_ev.index.get_indexer([tau_ts], method="nearest")
                cp = int(idx_list[0])
            else:
                cp = len(x) // 2

            # Residui OLS piecewise
            from scipy.stats import linregress
            s1, i1, *_ = linregress(x[:cp], y[:cp])
            s2, _,  *_ = linregress(x[cp:], y[cp:])
            i2  = i1 + cp * (s1 - s2)
            y_hat = np.concatenate([i1 + s1 * x[:cp], i2 + s2 * x[cp:]])
            resid = y - y_hat

            label    = f"{ev_key} | {ser_name}"
            safe_key = (ev_key + "_" + ser_name).replace(" ", "_").replace("/", "")
            plot_path = f"plots/06_distrib_{safe_key}.png"

            print(f"\n  → {label}  (cp={cp}, n={len(resid)})")
            row_out = analyse_residuals(resid, label, plot_path)
            row_out["tipo"] = "log_prezzo"
            row_out["evento"] = ev_key
            row_out["serie"]  = ser_name
            all_check_rows.append(row_out)


# ─────────────────────────────────────────────────────────────────────────────
# 2.  Crack spread post-shock (contesto script 03)
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("PARTE 2 — Distribuzione crack spread post-shock (contesto script 03)")
print("=" * 70)

EVENTS_03 = {
    "Ucraina_Feb2022":    {"shock": pd.Timestamp("2022-02-24"), "post_end": pd.Timestamp("2022-08-31")},
    "Iran-Israele_Giu25": {"shock": pd.Timestamp("2025-06-13"), "post_end": pd.Timestamp("2025-10-31")},
    "Hormuz_Feb2026":     {"shock": pd.Timestamp("2026-02-28"), "post_end": pd.Timestamp("2026-04-27")},
}
MARGIN_COLS = {
    "Benzina": "margine_benz_crack",
    "Diesel":  "margine_dies_crack",
}

futures = None
for fname in ["data/dataset_merged_with_futures.csv", "data/dataset_merged.csv"]:
    if os.path.exists(fname):
        futures = pd.read_csv(fname, index_col=0, parse_dates=True)
        print(f"  Dataset futures: {fname}  ({len(futures)} righe)\n")
        break

if futures is not None:
    # Baseline 2019 per ogni carburante
    baseline_2019 = {}
    for fuel, col in MARGIN_COLS.items():
        if col not in futures.columns:
            continue
        bl = futures.loc["2019-01-01":"2019-12-31", col].dropna()
        if len(bl) > 0:
            baseline_2019[fuel] = bl.values

    for ev_key, ev_cfg in EVENTS_03.items():
        for fuel, col in MARGIN_COLS.items():
            if col not in futures.columns:
                continue

            post_data = futures.loc[ev_cfg["shock"]:ev_cfg["post_end"], col].dropna()
            if len(post_data) < 6:
                print(f"  SKIP {ev_key}|{fuel}: <6 obs post-shock")
                continue

            label     = f"{ev_key} | {fuel} [crack spread post-shock]"
            safe_key  = (ev_key + "_" + fuel + "_crack").replace(" ", "_").replace("/", "")
            plot_path = f"plots/06_distrib_{safe_key}.png"

            print(f"\n  → {label}  (n={len(post_data)})")
            row_out = analyse_residuals(post_data.values, label, plot_path)
            row_out["tipo"]   = "crack_spread_post"
            row_out["evento"] = ev_key
            row_out["serie"]  = fuel
            all_check_rows.append(row_out)


# ─────────────────────────────────────────────────────────────────────────────
# 3. Salva risultati
# ─────────────────────────────────────────────────────────────────────────────
if all_check_rows:
    df_out = pd.DataFrame(all_check_rows)
    df_out.to_csv("data/distribution_check.csv", index=False)
    print(f"\n\nSalvato: data/distribution_check.csv  ({len(df_out)} scenari)")
else:
    print("\nNessun risultato da salvare.")
    df_out = pd.DataFrame()


# ─────────────────────────────────────────────────────────────────────────────
# 4. Heatmap ΔAIC
# ─────────────────────────────────────────────────────────────────────────────
if not df_out.empty and "ΔAIC_skewnorm_vs_t" in df_out.columns:
    labels_h  = df_out["label"].values
    delta_skn = df_out["ΔAIC_skewnorm_vs_t"].astype(float).values
    delta_skt = df_out["ΔAIC_skewt_vs_t"].astype(float).values
    raccs     = df_out["raccomandazione"].values

    fig_h, ax_h = plt.subplots(figsize=(10, max(4, len(labels_h) * 0.55 + 1.5)))
    x_pos = np.arange(len(labels_h))
    w     = 0.35

    bars_skn = ax_h.barh(x_pos + w/2, delta_skn, height=w, color="#27ae60",
                         alpha=0.8, label="ΔAIC Skew-Normal vs StudentT")
    bars_skt = ax_h.barh(x_pos - w/2, delta_skt, height=w, color="#e67e22",
                         alpha=0.8, label="ΔAIC Skewed-T vs StudentT")

    ax_h.axvline(-2, color="black", lw=1.5, ls="--", alpha=0.7,
                 label="Soglia decisione ΔAIC = -2")
    ax_h.axvline(0, color="black", lw=0.8)

    # Annotazione raccomandazione
    for i, racc in enumerate(raccs):
        color_r = ("#c0392b" if "Skewed" in racc
                   else "#2980b9" if "Normale" == racc
                   else "#888")
        ax_h.text(max(delta_skn[i], delta_skt[i], 0) + 0.3, i,
                  f"  → {racc}", va="center", fontsize=7.5, color=color_r,
                  fontweight="bold")

    ax_h.set_yticks(x_pos)
    ax_h.set_yticklabels(labels_h, fontsize=8)
    ax_h.set_xlabel("ΔAIC rispetto a StudentT  (< −2 = alternativa preferita)", fontsize=10)
    ax_h.set_title(
        "Confronto distribuzioni: ΔAIC rispetto alla StudentT attuale\n"
        "Barre a sinistra di −2 indicano che l'alternativa è preferibile",
        fontsize=10, fontweight="bold",
    )
    ax_h.legend(fontsize=9, loc="lower right")
    ax_h.grid(alpha=0.25, axis="x")
    plt.tight_layout()
    fig_h.savefig("plots/06_distrib_summary.png", dpi=DPI, bbox_inches="tight")
    plt.close(fig_h)
    print("Salvato: plots/06_distrib_summary.png")


# ─────────────────────────────────────────────────────────────────────────────
# 5. Sommario e guida operativa
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("SOMMARIO RACCOMANDAZIONI")
print("=" * 70)

if not df_out.empty:
    groups = df_out.groupby("raccomandazione")["label"].apply(list)
    for racc, labels in groups.items():
        print(f"\n  [{racc}]")
        for lbl in labels:
            print(f"    • {lbl}")

print("""
GUIDA OPERATIVA — come modificare script 02 se la Skewed-T è raccomandata
---------------------------------------------------------------------------
Se la maggioranza degli scenari suggerisce Skewed-T, nel modello PyMC
(script 02, funzione bayesian_changepoint) sostituire:

  # ATTUALE:
  pm.StudentT("obs", nu=nu, mu=mu, sigma=sigma, observed=y)

  # CON Fernandez-Steel Skewed-T (richiede custom potential):
  alpha_skew = pm.HalfNormal("alpha_skew", sigma=1.0)  # asimmetria
  # La likelihood FS è implementata come mixture condizionale:
  # P(obs | x<mu) ∝ StudentT(gamma * z)
  # P(obs | x>=mu) ∝ StudentT(z / gamma)
  # → usare pm.Potential con la logpdf custom definita sopra.

Se invece la raccomandazione è Skew-Normal:
  # Skew-Normal disponibile direttamente in PyMC:
  alpha_skew = pm.Normal("alpha_skew", mu=0, sigma=2)
  pm.SkewNormal("obs", mu=mu, sigma=sigma, alpha=alpha_skew, observed=y)

NOTA: cambiare la likelihood NON invalida i test di script 03 (che sono
non-parametrici o HAC). Impatta solo la stima del changepoint tau e
dell'incertezza posteriore. Valuta se ΔAIC > 4 prima di complicare il modello.
""")

print("Script 06 completato.")