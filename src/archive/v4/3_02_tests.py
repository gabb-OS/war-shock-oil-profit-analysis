"""
3_02_tests.py  — Famiglie A e B: test statistici + grafici + tabelle (v3)
==========================================================================
Testa H₀(i) (Family B) + analisi descrittiva strutturale (Family A).

MAPPA H₀ DICHIARATE → FAMIGLIE
───────────────────────────────
  H₀(i)  dichiarata: "gli shock geopolitici non determinano un aumento
          significativo dei margini rispetto al periodo immediatamente
          precedente" → FAMIGLIA B (confronto pre→post)

  H₀(ii) dichiarata: "le variazioni italiane non risultano superiori a
          quelle dei paesi EU di riferimento" → FAMIGLIA C (DiD, in 3_03)

FAMIGLIA A  — Analisi descrittiva del livello strutturale (COMPLEMENTARE)
  NON è un test diretto di H₀(i) come dichiarata.
  Risponde a: "i margini post-shock sono strutturalmente più alti del
  periodo pre-crisi (2019)?" — utile per descrivere la persistenza del
  livello ma non identifica la causalità dello shock.
  H₀: μ_post_real ≤ μ_2019_real   (one-sided upper)
  H₁: μ_post_real > μ_2019_real
  Test: HAC_t one-sample + Mann-Whitney

FAMIGLIA B  — H₀(i): salto pre→post shock (test causale dichiarato)
  H₀: μ_post ≤ μ_pre   (one-sided upper)
  H₁: μ_post > μ_pre
  Test: HAC_t two-sample + Mann-Whitney
  Metrica: margine reale HICP-deflato (col_real) — coerente con Family A
  Logica: lo shock geopolitico causa un salto significativo dei margini
  rispetto al periodo immediatamente precedente?
  ⚠ Se δ_local < 0 (es. Iran-Israele), H₀(i) NON è rifiutata anche se
    il livello assoluto rimane alto rispetto al 2019 (Family A).
  ⚠ τ pre/post: usa il changepoint del prezzo wholesale (Benzina/Diesel),
    NON del crack spread → evita circolarità logica (τ esogeno).

METODI STATISTICI
─────────────────
  HAC_t: OLS con Newey-West, bandwidth Andrews (1991) plug-in AR(1).
    Corregge per autocorrelazione seriale (tipicamente ρ̂ ≈ 0.7–0.9
    nelle serie settimanali di prezzi carburanti).
  n_eff = n·(1−ρ̂)/(1+ρ̂): dimensione campionaria effettiva.
    ρ̂ stimato su serie detrended (evita sovrastima da trend lineare).
    ⚠ CAUTELA se n_eff < 5 (test poco informativo).
  Mann-Whitney: non-parametrico, robusto a non-normalità.

Input:  data/3_dataset.csv
Output:
  data/3_AB.csv              — 16 test (8 famiglia A + 8 famiglia B)
  data/3_neff_report.csv     — diagnostica n_eff e ρ̂ per ogni test
  data/3_annual_margins.csv  — analisi annuale dei margini
  plots/3_02a_margins.png    — crack spread nel tempo con bande di riferimento
  plots/3_02b_delta.png      — confronto δ pre→post per famiglia A e B
  plots/3_02c_annual.png     — margini medi annuali (nominale + reale)
  plots/3_02d_neff.png       — diagnostica autocorrelazione e n_eff
"""

import os
import warnings
import math
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
from scipy.stats import mannwhitneyu
import statsmodels.api as sm

warnings.filterwarnings("ignore")
os.makedirs("data",  exist_ok=True)
os.makedirs("plots", exist_ok=True)

# ── Costanti ──────────────────────────────────────────────────────────────────
ALPHA          = 0.05
BASELINE_START = "2019-01-01"
BASELINE_END   = "2019-12-31"
DPI            = 160
NEFF_WARN      = 5    # soglia per flag CAUTELA

EVENTS = {
    "Ucraina (Feb 2022)": {
        "shock":     pd.Timestamp("2022-02-24"),
        "pre_start": pd.Timestamp("2021-09-01"),
        "post_end":  pd.Timestamp("2022-08-31"),
        "color":     "#e74c3c",
        "label":     "Russia-Ucraina\n(24 feb 2022)",
    },
    "Iran-Israele (Giu 2025)": {
        "shock":     pd.Timestamp("2025-06-13"),
        "pre_start": pd.Timestamp("2025-01-01"),
        "post_end":  pd.Timestamp("2025-10-31"),
        "color":     "#e67e22",
        "label":     "Iran-Israele\n(13 giu 2025)",
    },
    "Hormuz (Feb 2026)": {
        "shock":     pd.Timestamp("2026-02-28"),
        "pre_start": pd.Timestamp("2025-08-01"),   # ~6 mesi pre τ (τ benzina ≈ 2026-02-23)
        "post_end":  pd.Timestamp("2026-04-30"),   # limite dati disponibili
        "color":     "#8e44ad",
        "label":     "Stretto di Hormuz\n(28 feb 2026)",
    },
}

FUELS = {
    "Benzina": {"col_real": "margine_benz_real", "col_nom": "margine_benz_crack",
                "color": "#d6604d"},
    "Diesel":  {"col_real": "margine_dies_real", "col_nom": "margine_dies_crack",
                "color": "#4393c3"},
}

# Mappa fuel → nome serie in 3_cp.csv
# ⚠ Usiamo la serie del prezzo wholesale (Benzina/Diesel) e NON quella del
#   crack spread (Crack_Benz/Crack_Dies): il changepoint del WHOLESALE è
#   esogeno rispetto al margine e risolve la circolarità logica (usare il τ
#   del crack spread per testare un salto del crack spread sarebbe endogeno).
_FUEL_TO_SERIE = {"Benzina": "Benzina", "Diesel": "Diesel"}


# ════════════════════════════════════════════════════════════════════════════
# AGGIORNA date di shock con τ MCMC (da 3_05_changepoint.py)
# ════════════════════════════════════════════════════════════════════════════
# Razionale scientifico:
#   La data dell'evento geopolitico (es. 24 feb 2022) è il "trigger" esterno,
#   ma il mercato può anticiparla (futures forward-looking) o reagire in
#   ritardo.  Il changepoint τ rilevato dai dati ci dice QUANDO la struttura
#   del margine si è effettivamente rotta.  Usare τ come cutoff pre/post:
#     • Family B: split tra pre e post (più preciso del giorno dell'evento)
#     • Family A: inizio della finestra post-shock (stessa logica)
#   Le date hardcoded rimangono come fallback se 3_05 non è ancora girato.
_CP_PATH = "data/3_cp.csv"
print(f"\nCerco tau MCMC in {_CP_PATH}...")
if os.path.exists(_CP_PATH):
    _df_cp   = pd.read_csv(_CP_PATH)
    _ev_col  = next((c for c in _df_cp.columns if c.lower() == "evento"),  None)
    _ser_col = next((c for c in _df_cp.columns if c.lower() == "serie"),   None)
    _tau_col = next((c for c in _df_cp.columns if c.lower() == "tau"),     None)

    if _ev_col and _ser_col and _tau_col:
        for evento in EVENTS:
            # Raccogli τ per Benzina e Diesel (prezzo wholesale, esogeno) separatamente
            tau_per_fuel: dict[str, pd.Timestamp] = {}
            for fuel, serie_key in _FUEL_TO_SERIE.items():
                row = _df_cp[
                    _df_cp[_ev_col].str.contains(evento[:10], na=False, regex=False) &
                    (_df_cp[_ser_col] == serie_key)
                ]
                if not row.empty:
                    tau_per_fuel[fuel] = pd.Timestamp(row[_tau_col].iloc[0])

            # Salva tau per-carburante nel dict EVENTS (usato nei loop test)
            default_shock = EVENTS[evento]["shock"]
            if tau_per_fuel:
                # τ medio per aggiornare la finestra grafica e Family A
                tau_avg = pd.Timestamp(
                    int(np.mean([t.value for t in tau_per_fuel.values()]))
                )
                lag_avg = (tau_avg - default_shock).days
                if abs(lag_avg) <= 180:
                    # Cutoff medio per grafici e Family A (level vs 2019)
                    EVENTS[evento]["shock"]          = tau_avg
                    EVENTS[evento]["shock_geopolit"] = default_shock
                    # Cutoff per-fuel per Family B (pre vs post split)
                    EVENTS[evento]["shock_benz"] = tau_per_fuel.get("Benzina", tau_avg)
                    EVENTS[evento]["shock_dies"] = tau_per_fuel.get("Diesel",  tau_avg)
                    old_lbl = EVENTS[evento]["label"]
                    EVENTS[evento]["label"] = (
                        old_lbl.split("\n")[0]
                        + f"\nτ={tau_avg.strftime('%d %b %Y')}"
                    )
                    print(f"   ✓ {evento}: shock → τ_avg={tau_avg.date()} "
                          f"(lag={lag_avg:+d}gg)  "
                          f"[benz={tau_per_fuel.get('Benzina','–')}  "
                          f"dies={tau_per_fuel.get('Diesel','–')}]")
                else:
                    print(f"   ⚠ {evento}: τ_avg={tau_avg.date()} troppo distante "
                          f"({lag_avg:+d}gg) — mantengo data hardcoded")
            else:
                print(f"   – {evento}: nessun τ crack trovato — uso data hardcoded")
    else:
        print(f"   ⚠ {_CP_PATH}: colonne Evento/Serie/tau non trovate — uso hardcoded")
else:
    print(f"   {_CP_PATH} non trovato.")
    print("     ATTENZIONE: 3_05_changepoint.py deve girare prima di 3_02.")
    print("     Uso date hardcoded (meno precise del τ MCMC).")


def _shock_for(ecfg: dict, fuel: str) -> pd.Timestamp:
    """Restituisce il τ per-carburante se disponibile, altrimenti shock evento."""
    if fuel == "Benzina":
        return ecfg.get("shock_benz", ecfg["shock"])
    return ecfg.get("shock_dies", ecfg["shock"])


# ════════════════════════════════════════════════════════════════════════════
# FUNZIONI STATISTICHE
# ════════════════════════════════════════════════════════════════════════════

def compute_neff(series: np.ndarray):
    """n_eff = n·(1−ρ̂)/(1+ρ̂) con ρ̂ su serie detrended."""
    x = np.asarray(series, dtype=float)
    x = x[~np.isnan(x)]
    n = len(x)
    if n < 4:
        return max(1.0, float(n)), 0.0
    t     = np.arange(n, dtype=float)
    coefs = np.polyfit(t, x, 1)
    x_det = x - np.polyval(coefs, t)
    rho   = float(np.corrcoef(x_det[:-1], x_det[1:])[0, 1])
    rho   = np.clip(rho, -0.9999, 0.9999)
    neff  = float(n) * (1.0 - rho) / (1.0 + rho)
    return max(1.0, neff), rho


def andrews_bw(rho: float, n: int) -> int:
    """Bandwidth ottimale Andrews (1991), AR(1) plug-in."""
    if abs(rho) < 1e-6:
        return 1
    alpha_val = 4.0 * rho**2 / (1.0 - rho**2)**2
    bw = max(1, math.ceil(1.1447 * (alpha_val * n) ** (1.0 / 3.0)))
    return min(bw, n // 2)


def hac_t_onesample(post: np.ndarray, mu0: float) -> dict:
    """HAC t-test one-sample (H₀: E[post]=mu0, H₁: >mu0)."""
    y = np.asarray(post, dtype=float)
    y = y[~np.isnan(y)]
    n = len(y)
    if n < 4:
        return {"t_stat": np.nan, "p_value": np.nan, "n": n,
                "n_eff": np.nan, "rho_hat": np.nan, "cautela": True}
    neff, rho = compute_neff(y)
    bw = andrews_bw(rho, n)
    y_c = y - mu0
    X   = sm.add_constant(np.ones(n))
    mod = sm.OLS(y_c, X).fit(cov_type="HAC", cov_kwds={"maxlags": bw, "use_correction": True})
    t   = float(mod.tvalues[0])
    from scipy.stats import t as t_dist
    p   = float(t_dist.sf(t, df=max(1, neff - 1)))
    return {"t_stat": round(t, 4), "p_value": round(p, 6), "n": n,
            "n_eff": round(neff, 1), "rho_hat": round(rho, 3), "bw": bw,
            "mean_post": round(float(y.mean()), 5), "cautela": bool(neff < NEFF_WARN)}


def hac_t_twosample(post: np.ndarray, pre: np.ndarray) -> dict:
    """HAC t-test two-sample (H₀: E[post]=E[pre], H₁: post>pre)."""
    post = np.asarray(post, dtype=float)
    post = post[~np.isnan(post)]
    pre  = np.asarray(pre,  dtype=float)
    pre  = pre[~np.isnan(pre)]
    mu_pre = float(pre.mean()) if len(pre) > 0 else 0.0
    res = hac_t_onesample(post, mu0=mu_pre)
    res["mean_pre"] = round(mu_pre, 5)
    res["delta"]    = round(res.get("mean_post", np.nan) - mu_pre, 5)
    return res


def mw_onesided(a: np.ndarray, b: np.ndarray) -> float:
    """Mann-Whitney U one-sided (H₁: a > b)."""
    a = a[~np.isnan(a)]
    b = b[~np.isnan(b)]
    if len(a) < 3 or len(b) < 3:
        return np.nan
    _, p = mannwhitneyu(a, b, alternative="greater")
    return float(p)


# ════════════════════════════════════════════════════════════════════════════
# CARICA DATI
# ════════════════════════════════════════════════════════════════════════════
print("Carico data/3_dataset.csv...")
df = pd.read_csv("data/3_dataset.csv", index_col=0, parse_dates=True).sort_index()
baseline = df.loc[BASELINE_START:BASELINE_END]
print(f"   Settimane totali: {len(df)}  |  Baseline 2019: {len(baseline)} settimane")

rows_tests = []   # risultati test (→ 3_AB.csv)
rows_neff  = []   # diagnostica n_eff (→ 3_neff_report.csv)


# ════════════════════════════════════════════════════════════════════════════
# FAMIGLIA A — H₀(i): livello reale vs 2019
# ════════════════════════════════════════════════════════════════════════════
print("\n" + "="*62)
print("FAMIGLIA A — Analisi descrittiva: livello reale vs baseline 2019")
print("  (Non è il test causale di H₀(i); cfr. Family B)")
print("="*62)

for evento, ecfg in EVENTS.items():
    for fuel, fcfg in FUELS.items():
        col_real = fcfg["col_real"]
        if col_real not in df.columns:
            continue

        # Family A usa il τ medio (ecfg["shock"] già aggiornato con τ MCMC se disponibile)
        post_ser  = df.loc[ecfg["shock"]:ecfg["post_end"], col_real].dropna().values
        bl_ser    = baseline[col_real].dropna().values
        mu_2019   = float(bl_ser.mean())

        # n_eff diagnostica
        neff_post, rho_post = compute_neff(post_ser)
        rows_neff.append({"evento": evento, "carburante": fuel, "famiglia": "A",
                          "n_post": len(post_ser), "n_eff_post": round(neff_post, 1),
                          "rho_post": round(rho_post, 3),
                          "cautela": bool(neff_post < NEFF_WARN)})

        # HAC_t
        hac = hac_t_onesample(post_ser, mu0=mu_2019)
        rows_tests.append({
            "famiglia": "A", "ipotesi": "desc_strutturale",
            "evento": evento, "carburante": fuel, "test": "HAC_t",
            "fonte": f"HAC_t_{evento}_{fuel}",
            "n_post": hac["n"], "n_pre": len(bl_ser),
            "n_eff": hac.get("n_eff"), "rho_hat": hac.get("rho_hat"),
            "mu_2019": round(mu_2019, 5),
            "mean_post": hac.get("mean_post"),
            "delta": round(hac.get("mean_post", np.nan) - mu_2019, 5),
            "t_stat": hac["t_stat"], "p_value": hac["p_value"],
            "cautela": hac["cautela"],
        })

        # Mann-Whitney
        p_mw = mw_onesided(post_ser, bl_ser)
        rows_tests.append({
            "famiglia": "A", "ipotesi": "desc_strutturale",
            "evento": evento, "carburante": fuel, "test": "MannWhitney",
            "fonte": f"MannWhitney_{evento}_{fuel}",
            "n_post": len(post_ser), "n_pre": len(bl_ser),
            "n_eff": None, "rho_hat": None,
            "mu_2019": round(mu_2019, 5),
            "mean_post": round(float(post_ser.mean()), 5) if len(post_ser) else np.nan,
            "delta": round(float(post_ser.mean()) - mu_2019, 5) if len(post_ser) else np.nan,
            "t_stat": None, "p_value": round(p_mw, 6) if not np.isnan(p_mw) else np.nan,
            "cautela": False,
        })

        rej = not np.isnan(hac["p_value"]) and hac["p_value"] < ALPHA
        print(f"  {'✓ RIFIUTA H₀' if rej else '  non rifiuta'}  "
              f"{evento[:18]} | {fuel}  δ={rows_tests[-2]['delta']:+.4f}  "
              f"p_HAC={hac['p_value']:.4f}  n_eff={hac.get('n_eff','-')}"
              + ("  ⚠CAUTELA" if hac["cautela"] else ""))


# ════════════════════════════════════════════════════════════════════════════
# FAMIGLIA B — H₀(ii): salto pre→post
# ════════════════════════════════════════════════════════════════════════════
print("\n" + "="*62)
print("FAMIGLIA B — H₀(i): salto reale (HICP-deflato) pre→post shock (test causale)")
print("  δ < 0 → H₀(i) NON rifiutata anche se livello strutturale alto (Family A)")
print("  Metrica: margine reale (HICP), coerente con Family A")
print("="*62)

for evento, ecfg in EVENTS.items():
    for fuel, fcfg in FUELS.items():
        col_real = fcfg["col_real"]
        if col_real not in df.columns:
            continue

        # Family B: usa il τ per-carburante come data di taglio pre/post
        # (più preciso della data geopolitica hardcoded)
        shock_b = _shock_for(ecfg, fuel)
        pre_ser  = df.loc[ecfg["pre_start"]:shock_b - pd.Timedelta(days=1),
                          col_real].dropna().values
        post_ser = df.loc[shock_b:ecfg["post_end"],
                          col_real].dropna().values

        neff_post, rho_post = compute_neff(post_ser)
        rows_neff.append({"evento": evento, "carburante": fuel, "famiglia": "B",
                          "n_post": len(post_ser), "n_eff_post": round(neff_post, 1),
                          "rho_post": round(rho_post, 3),
                          "cautela": bool(neff_post < NEFF_WARN)})

        # HAC_t two-sample
        hac = hac_t_twosample(post_ser, pre_ser)
        rows_tests.append({
            "famiglia": "B", "ipotesi": "H0_i",
            "evento": evento, "carburante": fuel, "test": "HAC_t",
            "fonte": f"HAC_t_{evento}_{fuel}_jump",
            "n_post": len(post_ser), "n_pre": len(pre_ser),
            "n_eff": hac.get("n_eff"), "rho_hat": hac.get("rho_hat"),
            "mu_2019": None,
            "mean_post": hac.get("mean_post"),
            "delta": hac.get("delta"),
            "t_stat": hac["t_stat"], "p_value": hac["p_value"],
            "cautela": hac["cautela"],
        })

        # Mann-Whitney
        p_mw = mw_onesided(post_ser, pre_ser)
        rows_tests.append({
            "famiglia": "B", "ipotesi": "H0_i",
            "evento": evento, "carburante": fuel, "test": "MannWhitney",
            "fonte": f"MannWhitney_{evento}_{fuel}_jump",
            "n_post": len(post_ser), "n_pre": len(pre_ser),
            "n_eff": None, "rho_hat": None, "mu_2019": None,
            "mean_post": round(float(post_ser.mean()), 5) if len(post_ser) else np.nan,
            "delta": round(float(post_ser.mean()) - float(pre_ser.mean()), 5)
                     if len(post_ser) and len(pre_ser) else np.nan,
            "t_stat": None, "p_value": round(p_mw, 6) if not np.isnan(p_mw) else np.nan,
            "cautela": False,
        })

        rej = not np.isnan(hac["p_value"]) and hac["p_value"] < ALPHA
        shock_src = "τ_MCMC" if "shock_benz" in ecfg or "shock_dies" in ecfg else "hardcoded"
        print(f"  {'✓ RIFIUTA H₀' if rej else '  non rifiuta'}  "
              f"{evento[:18]} | {fuel}  δ={hac.get('delta',np.nan):+.4f}  "
              f"p_HAC={hac['p_value']:.4f}  n_post={len(post_ser)}  n_eff={hac.get('n_eff','-')}"
              f"  cutoff={shock_b.date()}({shock_src})  [metrica: reale HICP]"
              + ("  ⚠CAUTELA" if hac["cautela"] else ""))


# ════════════════════════════════════════════════════════════════════════════
# SALVA CSV
# ════════════════════════════════════════════════════════════════════════════
df_tests = pd.DataFrame(rows_tests)
df_tests.to_csv("data/3_AB.csv", index=False)
print(f"\n✓ data/3_AB.csv  ({len(df_tests)} test: "
      f"{len(df_tests[df_tests['famiglia']=='A'])} A + "
      f"{len(df_tests[df_tests['famiglia']=='B'])} B)")

df_neff = pd.DataFrame(rows_neff)
df_neff.to_csv("data/3_neff_report.csv", index=False)

# Analisi annuale margini
annual_rows = []
for year in range(2019, df.index.year.max() + 1):
    yr = df[df.index.year == year]
    for fuel, fcfg in FUELS.items():
        for col_type, col in [("nominale", fcfg["col_nom"]), ("reale", fcfg["col_real"])]:
            if col not in yr.columns:
                continue
            s = yr[col].dropna()
            if len(s) == 0:
                continue
            annual_rows.append({
                "anno": year, "carburante": fuel, "tipo": col_type,
                "media": round(float(s.mean()), 5), "std": round(float(s.std()), 5),
                "min": round(float(s.min()), 5), "max": round(float(s.max()), 5),
                "n": len(s),
            })
df_annual = pd.DataFrame(annual_rows)
df_annual.to_csv("data/3_annual_margins.csv", index=False)


# ════════════════════════════════════════════════════════════════════════════
# GRAFICI
# ════════════════════════════════════════════════════════════════════════════

# ── Fig A: Crack spread nel tempo + bande di riferimento ─────────────────
fig, axes = plt.subplots(2, 1, figsize=(15, 9), sharex=True)

for ax, fuel, fcfg in [(axes[0], "Benzina", FUELS["Benzina"]),
                        (axes[1], "Diesel",  FUELS["Diesel"])]:
    col_nom  = fcfg["col_nom"]
    col_real = fcfg["col_real"]
    col_fc   = fcfg["color"]
    if col_nom not in df.columns:
        continue

    s  = df[col_nom].dropna()
    bl = baseline[col_nom].dropna()

    # Serie nominale e reale
    ax.plot(s.index, s.values, color=col_fc, lw=1.8, alpha=0.9, label="Nominale")
    if col_real in df.columns:
        ax.plot(df.index, df[col_real], color=col_fc, lw=1.2, ls="--",
                alpha=0.55, label="Reale (HICP)")

    # Banda baseline 2019
    if len(bl) >= 4:
        mu2019 = bl.mean()
        sd2019 = bl.std()
        ax.axhline(mu2019, color="#666", lw=1.0, ls="-.")
        ax.axhspan(mu2019 - 2*sd2019, mu2019 + 2*sd2019,
                   alpha=0.10, color="#888888", label="Baseline 2019 ±2σ")

    # Finestre pre/post per ogni evento
    for evento, ecfg in EVENTS.items():
        ax.axvspan(ecfg["pre_start"], ecfg["shock"], alpha=0.06, color=ecfg["color"])
        ax.axvspan(ecfg["shock"], ecfg["post_end"], alpha=0.10, color=ecfg["color"])
        ax.axvline(ecfg["shock"], color=ecfg["color"], lw=1.8, ls="--", alpha=0.85)
        ax.text(ecfg["shock"] + pd.Timedelta(days=6), s.max() * 0.97,
                ecfg["label"], rotation=90, fontsize=7, color=ecfg["color"], va="top")

    ax.set_ylabel("Margine lordo (EUR/L)", fontsize=10)
    ax.set_title(f"Crack spread — {fuel}  (banda = baseline 2019 ±2σ)", fontsize=10)
    ax.legend(fontsize=8, loc="upper left")
    ax.grid(alpha=0.25)

axes[1].xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
axes[1].xaxis.set_major_locator(mdates.MonthLocator(interval=3))
plt.xticks(rotation=35, fontsize=8)
fig.suptitle("Famiglia A (livello strutturale vs 2019) e Famiglia B (H₀(i): salto pre→post)\n"
             "Ombra chiara = finestra pre-shock  |  ombra scura = finestra post-shock",
             fontsize=11, fontweight="bold")
plt.tight_layout()
fig.savefig("plots/3_02a_margins.png", dpi=DPI, bbox_inches="tight")
plt.close(fig)
print("✓ plots/3_02a_margins.png")


# ── Fig B: Confronto δ (pre→post e vs 2019) per ogni caso ────────────────
# Mostra μ_pre, μ_post e μ_2019 affiancati per ogni combinazione evento×fuel

fig, axes = plt.subplots(1, 2, figsize=(13, 5))
x_labels = [f"{ev[:14]}\n{fuel}" for ev in EVENTS for fuel in FUELS]
x_pos = np.arange(len(x_labels))
width = 0.25

for ai, (col_type, col_key, title_suffix) in enumerate([
    ("nominale", "col_nom", "nominale"),
    ("reale",    "col_real", "reale HICP-deflato"),
]):
    ax = axes[ai]
    bars_2019, bars_pre, bars_post = [], [], []
    for ev, ecfg in EVENTS.items():
        for fuel, fcfg in FUELS.items():
            col = fcfg[col_key]
            if col not in df.columns:
                bars_2019.append(0); bars_pre.append(0); bars_post.append(0)
                continue
            mu_2019 = float(baseline[col].dropna().mean())
            mu_pre  = float(df.loc[ecfg["pre_start"]:ecfg["shock"] - pd.Timedelta(days=1),
                                   col].dropna().mean())
            mu_post = float(df.loc[ecfg["shock"]:ecfg["post_end"], col].dropna().mean())
            bars_2019.append(mu_2019)
            bars_pre.append(mu_pre)
            bars_post.append(mu_post)

    ax.bar(x_pos - width, bars_2019, width, label="Media 2019", color="#b0bec5", alpha=0.85)
    ax.bar(x_pos,         bars_pre,  width, label="Pre-shock",  color="#78909c", alpha=0.85)
    ax.bar(x_pos + width, bars_post, width, label="Post-shock", color="#d32f2f", alpha=0.85)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(x_labels, fontsize=8)
    ax.set_ylabel("EUR/L")
    ax.set_title(f"Confronto medie — {title_suffix}", fontsize=10, fontweight="bold")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.25, axis="y")

fig.suptitle("Famiglia A: post vs 2019 (livello strutturale)  |  Famiglia B: post vs pre-shock (H₀(i) causale)\n"
             "Barre rosse (post) sopra grigie (pre/2019) → δ positivo", fontsize=11)
plt.tight_layout()
fig.savefig("plots/3_02b_delta.png", dpi=DPI, bbox_inches="tight")
plt.close(fig)
print("✓ plots/3_02b_delta.png")


# ── Fig C: Margini medi annuali ───────────────────────────────────────────
if not df_annual.empty:
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for ai, col_type in enumerate(["nominale", "reale"]):
        ax = axes[ai]
        sub = df_annual[df_annual["tipo"] == col_type]
        for fuel, fcfg in FUELS.items():
            fs = sub[sub["carburante"] == fuel].sort_values("anno")
            if fs.empty:
                continue
            ax.plot(fs["anno"], fs["media"], marker="o", lw=2, ms=7,
                    color=fcfg["color"], label=fuel)
            ax.fill_between(fs["anno"],
                            fs["media"] - fs["std"],
                            fs["media"] + fs["std"],
                            alpha=0.12, color=fcfg["color"])
        # Linea 2019
        bl_2019 = sub[sub["anno"] == 2019]
        if not bl_2019.empty:
            mu19 = float(bl_2019.groupby("carburante")["media"].mean().mean())
            ax.axhline(mu19, color="#888", lw=1.0, ls="--", label="Media 2019")
        for ev, ecfg in EVENTS.items():
            shock_yr = ecfg["shock"].year
            ax.axvline(shock_yr, color=ecfg["color"], lw=1.5, ls="--", alpha=0.7)
        ax.set_xlabel("Anno")
        ax.set_ylabel("EUR/L")
        ax.set_title(f"Margine medio annuale — {col_type}", fontweight="bold")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.25)
    fig.suptitle("Analisi annuale margini lordi  (nastro = ±1σ settimanale)",
                 fontsize=11, fontweight="bold")
    plt.tight_layout()
    fig.savefig("plots/3_02c_annual.png", dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print("✓ plots/3_02c_annual.png")


# ── Fig D: Diagnostica n_eff e ρ̂ ─────────────────────────────────────────
if not df_neff.empty:
    labels_n  = [f"{r['evento'].split('(')[0].strip()} | {r['carburante']} | Fam.{r['famiglia']}"
                 for _, r in df_neff.iterrows()]
    neff_vals = df_neff["n_eff_post"].values
    rho_vals  = df_neff["rho_post"].values

    fig, (ax_n, ax_r) = plt.subplots(1, 2,
                                      figsize=(13, max(4, len(labels_n) * 0.55 + 2)))
    colors_n = ["#c0392b" if v < NEFF_WARN else "#e67e22" if v < 10 else "#27ae60"
                for v in neff_vals]
    ax_n.barh(range(len(labels_n)), neff_vals, color=colors_n, alpha=0.85,
              edgecolor="black", lw=0.5)
    ax_n.axvline(NEFF_WARN, color="#c0392b", lw=2, ls="--",
                 label=f"Soglia CAUTELA (n_eff={NEFF_WARN})")
    ax_n.axvline(10, color="#e67e22", lw=1.5, ls=":",
                 label="Soglia ATTENZIONE (n_eff=10)")
    ax_n.set_yticks(range(len(labels_n)))
    ax_n.set_yticklabels(labels_n, fontsize=9)
    ax_n.set_xlabel("n_eff (osservazioni indipendenti effettive)", fontsize=9)
    ax_n.set_title("Potenza dei test: n_eff\nRosso = test poco informativo",
                   fontsize=9, fontweight="bold")
    ax_n.legend(fontsize=8)
    ax_n.grid(alpha=0.25, axis="x")

    colors_r = ["#c0392b" if v > 0.85 else "#e67e22" if v > 0.6 else "#27ae60"
                for v in rho_vals]
    ax_r.barh(range(len(labels_n)), rho_vals, color=colors_r, alpha=0.85,
              edgecolor="black", lw=0.5)
    ax_r.axvline(0.85, color="#c0392b", lw=2, ls="--", label="ρ̂=0.85")
    ax_r.set_yticks(range(len(labels_n)))
    ax_r.set_yticklabels([], fontsize=9)
    ax_r.set_xlabel("ρ̂ AR(1) stimato (su serie detrended)", fontsize=9)
    ax_r.set_title("Autocorrelazione AR(1)\nAlta ρ → HAC molto conservativo",
                   fontsize=9, fontweight="bold")
    ax_r.legend(fontsize=8)
    ax_r.grid(alpha=0.25, axis="x")

    fig.suptitle("Diagnostica autocorrelazione e potenza\n"
                 "n_eff = n·(1−ρ̂)/(1+ρ̂)  |  Bandwidth: Andrews (1991) plug-in",
                 fontsize=10, fontweight="bold")
    plt.tight_layout()
    fig.savefig("plots/3_02d_neff.png", dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print("✓ plots/3_02d_neff.png")

print("\nScript 3_02 completato.")