"""
02_core_analysis.py
====================
Analisi principale per il test di H0:

  H0: Il margine lordo sui carburanti italiani non aumenta in modo
      statisticamente significativo rispetto al baseline pre-crisi 2019
      durante le tre crisi energetiche analizzate.

Questo script:
  A) Stima i changepoint bayesiani sui LOG-PREZZI (Table 1 del paper)
     → individua QUANDO è avvenuta la rottura strutturale nei prezzi
  B) Calcola i margini lordi con CRACK SPREAD da futures wholesale europei
       - Benzina: prezzi_pompa − Eurobob (ARA, EUR/litro)
       - Diesel:  prezzi_pompa − Gas Oil (London ICE, EUR/litro)
       [dati: Investing.com, file CSV hardcoded]
     LIMITAZIONE: i distributori italiani possono usare contratti forward
     o prezzi CIF-Genova anziché spot ARA/ICE; il crack spread misurato
     è una proxy del margine reale, non il margine reale stesso.
  C) Testa l'anomalia del margine post-shock (Table 2, solo Ucraina e Iran):
       t-test Welch (test PRIMARIO UNICO), KS (ausiliario-diagnostico),
       bootstrap 95% CI (ausiliario-diagnostico)
       + MCMC Bayesiano sul margine (StudentT + AR(1), 4 catene)
       + Benjamini-Hochberg FDR correction su t_p (test primario)
     NOTE METODOLOGICHE SUL TEST PRIMARIO:
       - Solo Welch t-test governa la regola di rigetto (stat_sig = t_p < α).
         KS e CI bootstrap sono riportati come diagnostici ma NON entrano
         nella decisione, per coerenza con la BH correction applicata su t_p.
         Un AND(t, KS) creerebbe un test composito con α nominale non
         controllato dalla BH sul solo t_p.
       - Classificazioni usano nomenclatura descrittiva del pattern statistico,
         NON causale. "MARGINE ANOMALO POSITIVO" ≠ "speculazione"; è
         consistente anche con effetti FIFO/LIFO, risk premium razionale,
         cost-push wholesale non catturato da ARA/ICE.
     NOTA: Hormuz (Feb 2026) escluso da Table 2 — dati post-shock
     insufficienti (≤4 settimane al momento dell'analisi).
  D) Plot: prezzi con changepoints, margini nel tempo, Δmargini

Input:
  data/dataset_merged.csv
  data/Eurobob Futures Historical Data.csv          (Investing.com)
  data/London Gas Oil Futures Historical Data.csv   (Investing.com)

Output:
  data/table1_changepoints.csv
  data/table2_margin_anomaly.csv
  data/baseline_sensitivity.csv
  plots/02_{event}_{series}.png      (changepoint + KDE posteriore)
  plots/02_{event}_{series}_diag.png (diagnostica regressione)
  plots/07_margins_{fuel}.png        (margine nel tempo)
  plots/07_delta_summary.png         (Δmargine per evento)

NOTA METODOLOGICA — scelta del modello:
  Test diagnostici in script 03 mostrano DW ≈ 0.003–0.04 (autocorrelazione
  quasi perfetta), BP p=0.000 (eteroschedasticità), SW p=0.000 (non-normalità).
  → Likelihood StudentT(ν stimato) gestisce code pesanti e autocorrelazione
    residua implicita (ν piccolo ↔ distribuzioni più pesanti alle code).
  → AR(1) esplicito RIMOSSO: creava geometria sequenzialmente dipendente
    incompatibile con NUTS (max_treedepth sistematico, Rhat > 1.01).
    L'autocorrelazione è riportata come diagnostico (DW test in script 03)
    ma non modellata nel passo di changepoint.
  → tau ~ Beta(2,2) su [0,1]: migliore geometria rispetto a Uniform.
  Rif: Gelman et al. (2013); Betancourt (2017); Casini & Perron (2021).

NOTA BASELINE — 2019 full year (primaria):
  Baseline: 2019-01-01 → 2019-12-31.
  Sensitivity analysis con H1-2021 e Full-2021: data/baseline_sensitivity.csv.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
import pymc as pm
import pytensor.tensor as pt
from scipy import stats
from scipy.stats import gaussian_kde
from statsmodels.stats.stattools import durbin_watson
from statsmodels.graphics.tsaplots import plot_acf
import warnings
warnings.filterwarnings("ignore")

# ─── Configurazione globale ──────────────────────────────────────────────────
DPI          = 180
ALPHA        = 0.05
H0_THRESHOLD = 30       # giorni: lag > 30gg → trasmissione lenta (H0 non rifiutata)
EDGE_FRAC    = 0.15     # changepoint troppo vicino ai bordi → break non affidabile

# ─── Configurazione MCMC per scenario ────────────────────────────────────────
# Ogni chiave corrisponde a un pattern "Evento__Serie" (Table 1) oppure
# "Evento__Carburante__Metodo" (Table 2 margini). La chiave "_default" fornisce
# i valori di fallback usati da tutti gli scenari non elencati.
#
# Come scegliere i valori per scenari problematici:
#   • Rhat 1.01–1.05 (lieve non-convergenza)
#     → aumentare `tune` (es. 3000–4000) e `target_accept` (es. 0.95–0.97)
#   • Divergences > 0
#     → aumentare `target_accept` (es. 0.97–0.99); cambiare `init` in "adapt_full"
#       se il numero rimane alto dopo target_accept = 0.97
#   • ESS < 100 per catena → raddoppiare `draws`
#   • Rhat > 1.05 (convergenza dubbia) → aumentare tutti i parametri; considerare
#       reparametrizzazione della funzione bayesian_changepoint
#
# Scenari che hanno prodotto warning nel run di riferimento (log output):
#   "Ucraina (Feb 2022)__Brent"
#       Rhat max=1.030 → tune insufficiente per la geometria Brent
#   "Ucraina (Feb 2022)__Diesel__Crack Gas Oil"
#       Rhat max=1.012 + 6 divergences → target_accept troppo basso
#   "Iran-Israele (Giu 2025)__Diesel__Crack Gas Oil"
#       Rhat max=1.015, sampling molto lento (67s) → geometria complessa

MCMC_CONFIG: dict[str, dict] = {

    # ── Default: applicato a tutti gli scenari non elencati ──────────────────
    "_default": {
        "draws":         2000,
        "tune":          2000,
        "chains":        4,
        "target_accept": 0.95,
        "init":          "adapt_diag",
        "max_treedepth": 15,
    },

    # ── Table 1: Changepoint sui log-prezzi ───────────────────────────────────
    # Formato chiave: "Evento__Serie"  (Serie ∈ {Brent, Benzina, Diesel})

    # Ucraina / Brent → Rhat 1.030 nel run di riferimento
    "Ucraina (Feb 2022)__Brent": {
        "draws":         3000,        # più campioni → ESS più alto
        "tune":          6000,        # warm‑up più lungo → migliore adattamento della metrica
        "target_accept": 0.99,        # passo più piccolo → meno divergenze e mixing più stabile
        "init":          "adapt_full", # matrice di massa densa (covarianza completa) anziché diagonale
        "max_treedepth": 20,          # profondità albero aumentata per geometrie strette
    },

    # ── Table 2: Changepoint sui margini lordi ────────────────────────────────
    # Formato chiave: "Evento__Carburante__Metodo"

    # Ucraina / Diesel / Crack Gas Oil → Rhat 1.012 + 6 divergences
    "Ucraina (Feb 2022)__Diesel__Crack Gas Oil": {
        "tune":          4000,
        "target_accept": 0.98,
        "init":          "adapt_full",   # metrica di massa completa: meglio per
                                         # geometrie con forti correlazioni
    },

    # Iran-Israele / Diesel / Crack Gas Oil → Rhat 1.015, sampling lento (67s)
    "Iran-Israele (Giu 2025)__Diesel__Crack Gas Oil": {
        "draws":         3000,       # più campioni per ESS più alto
        "tune":          5000,       # adattamento molto più lungo
        "target_accept": 0.99,       # passo più piccolo → meno divergences
        "init":          "adapt_full",
        "max_treedepth": 18,         # albero più profondo per geometrie strette
    },
}


def _get_mcmc_cfg(scenario_key: str) -> dict:
    """
    Restituisce la configurazione MCMC per lo scenario dato.
    Merge: default + override specifico (i campi del default non presenti
    nell'override vengono mantenuti, così ogni entry di MCMC_CONFIG può
    sovrascrivere solo i campi necessari).

    Parametri
    ---------
    scenario_key : str
        Chiave nel formato "Evento__Serie" (Table 1) o
        "Evento__Carburante__Metodo" (Table 2).

    Esempi
    ------
    >>> _get_mcmc_cfg("Ucraina (Feb 2022)__Brent")
    {'draws': 2000, 'tune': 4000, 'chains': 4, 'target_accept': 0.97, ...}
    >>> _get_mcmc_cfg("Hormuz (Feb 2026)__Benzina")
    {'draws': 2000, 'tune': 2000, 'chains': 4, 'target_accept': 0.95, ...}
    """
    base = dict(MCMC_CONFIG["_default"])
    override = MCMC_CONFIG.get(scenario_key, {})
    base.update(override)
    return base


# Manteniamo alias per retrocompatibilità con eventuali riferimenti esterni
MCMC_DRAWS  = MCMC_CONFIG["_default"]["draws"]
MCMC_TUNE   = MCMC_CONFIG["_default"]["tune"]
MCMC_CHAINS = MCMC_CONFIG["_default"]["chains"]

# Conversioni unità per crack spread wholesale → EUR/litro
DENSITY_BENZ_KG_L  = 0.74
DENSITY_DIES_KG_L  = 0.84
L_PER_TONNE_BENZ   = 1000.0 / DENSITY_BENZ_KG_L   # ≈ 1351 L/t
L_PER_TONNE_DIES   = 1000.0 / DENSITY_DIES_KG_L   # ≈ 1190 L/t
# Eurobob e Gas Oil sono il costo wholesale reale per i distributori.

# Baseline: 2019 full year (gennaio–dicembre).
# MOTIVAZIONE: 2019 è l'unico anno genuinamente pre-crisi disponibile.
#   - 2020: COVID-19, WTI negativo in aprile, domanda collassata ~25%.
#     σ artificialmente alta e non stazionaria — baseline non difendibile.
#   - 2021 H1: post-COVID recovery, prezzi Brent ancora in rimbalzo.
#     Margini potenzialmente compressi; non rappresentativo di condizioni normali.
#   - 2019: mercato maturo, nessuno shock, Brent stabile 60-70 $/bbl.
#     Margini distributivi italiani strutturalmente non disturbati.
# Sensitivity analysis con H1-2021 e Full-2021: vedi data/baseline_sensitivity.csv.
BASELINE_START = "2019-01-01"
BASELINE_END   = "2019-12-31"

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
        "window_end":   "2026-04-01",
        "color":        "#8e44ad",
    },
}

PRICE_SERIES = {
    "Brent":   ("log_brent",   "brent_7d_eur", "log(EUR/barile)", "EUR/barile", "#2166ac"),
    "Benzina": ("log_benzina", "benzina_4w",   "log(EUR/litro)",  "EUR/litro",  "#d6604d"),
    "Diesel":  ("log_diesel",  "diesel_4w",    "log(EUR/litro)",  "EUR/litro",  "#31a354"),
}

CLAS_COLOR = {
    # NOTA METODOLOGICA: i label descrivono il pattern statistico osservato,
    # NON la causa economica. "MARGINE ANOMALO POSITIVO" indica che il margine
    # lordo post-shock supera la soglia 2σ ed è statisticamente significativo,
    # ma NON esclude spiegazioni alternative alla speculazione:
    # effetti FIFO/LIFO su inventario, risk premium razionale, cost-push
    # wholesale non catturato da ARA/ICE, riduzione temporanea della concorrenza.
    "MARGINE ANOMALO POSITIVO":                 "#c0392b",   # ex SPECULAZIONE
    "COMPRESSIONE MARGINE":                     "#2980b9",
    "NEUTRO / TRASMISSIONE ATTESA":             "#27ae60",   # ex ANTICIPAZIONE RAZIONALE / NEUTRO
    "VARIAZIONE STATISTICA":      "#e67e22",
    "INCONCLUSIVO":                             "#95a5a6",
}

# ─────────────────────────────────────────────────────────────────────────────
# SEZIONE A: Utility condivise
# ─────────────────────────────────────────────────────────────────────────────

def bayesian_changepoint(x_vals: np.ndarray, y_vals: np.ndarray, alpha: float = 0.05,
                         mcmc_cfg: dict | None = None):
    """
    Modello Bayesiano piecewise-lineare per changepoint detection.

    CAMBIAMENTI rispetto alla versione precedente (motivazione convergenza):

    1. AR(1) RIMOSSO.
       Il termine scan(rho*eps_prev + eta) crea dipendenze sequenziali n-deep
       che rendono la geometria della posterior difficile per NUTS:
       ogni eps_t dipende da eps_{t-1} → funnel geometry → max_treedepth
       sistematico, Rhat > 1.01, ESS < 100.
       L'autocorrelazione nei residui è gestita implicitamente dalla
       StudentT(ν) con ν stimato, ed è riportata come diagnostico (DW test)
       ma non modellata esplicitamente nel passo di changepoint.
       Rif: Betancourt (2017) "A Conceptual Introduction to HMC", §5.

    2. tau ~ Beta(2,2) su [0,1] trasformato a [x_lo, x_hi].
       Uniform aveva boundary effects (gradiente nullo ai bordi).
       Beta(2,2) concentra il prior leggermente lontano dai bordi,
       coerente con EDGE_FRAC=0.15, e ha derivata finita ovunque.

    3. nu ~ Gamma(2, 0.1)  [E=20, SD=14].
       Exponential(1/30) metteva troppa massa su ν piccoli (Cauchy-like)
       e aveva coda pesante verso ∞ (normale). Gamma(2,0.1) è più
       informativa sull'intervallo ν ∈ [5, 50] tipico di serie finanziarie.

    4. b1, b2 ~ Normal (era StudentT ν=3).
       Prior Normal riduce il numero di parametri latenti e la complessità
       del grafo computazionale. La likelihood ha già code pesanti.

    5. a1 centrato su mean(y) invece di 0: inizializzazione più realistica
       per serie con mean non nulla (prezzi in log).

    6. init="adapt_diag": inizializzazione adattiva della metrica di massa.

    7. max_treedepth=15 (era 20): sufficiente senza AR(1).

    Rif: Gelman et al. (2013); Betancourt (2017); Casini & Perron (2021).
    """
    import arviz as az

    # Usa la config specifica dello scenario (o il default se None)
    cfg    = mcmc_cfg if mcmc_cfg is not None else MCMC_CONFIG["_default"]
    _draws  = cfg["draws"]
    _tune   = cfg["tune"]
    _chains = cfg["chains"]
    _ta     = cfg["target_accept"]
    _init   = cfg["init"]
    _mtd    = cfg["max_treedepth"]

    n      = len(x_vals)
    sd_y   = float(np.std(y_vals))
    mean_y = float(np.mean(y_vals))
    x_lo   = float(x_vals[0])
    x_hi   = float(x_vals[-1])
    x_rng  = max(x_hi - x_lo, 1.0)

    with pm.Model():
        # ── tau: reparametrizzato su [0,1] via Beta(2,2) ─────────────────
        tau_raw = pm.Beta("tau_raw", alpha=2, beta=2)
        tau     = pm.Deterministic("tau", x_lo + tau_raw * x_rng)

        # ── Parametri likelihood ─────────────────────────────────────────
        sigma = pm.HalfNormal("sigma", sigma=sd_y)
        nu    = pm.Gamma("nu", alpha=2, beta=0.1)   # E[ν]=20, evita ν→1 e ν→∞

        # ── Piecewise linear: prior Normal (geometria più semplice) ──────
        b1 = pm.Normal("b1", mu=0.0, sigma=3.0 * sd_y)
        b2 = pm.Normal("b2", mu=0.0, sigma=3.0 * sd_y)
        a1 = pm.Normal("a1", mu=mean_y, sigma=sd_y)     # centrato su mean(y)
        a2 = pm.Deterministic("a2", a1 + tau * (b1 - b2))  # continuità in τ

        # ── Mean piecewise ───────────────────────────────────────────────
        x_pt = pt.as_tensor_variable(x_vals.astype(float))
        step = pm.math.sigmoid((x_pt - tau) * 50)          # transizione hard
        mu   = (a1 + b1 * x_pt) * (1.0 - step) + (a2 + b2 * x_pt) * step

        pm.StudentT("obs", nu=nu, mu=mu, sigma=sigma, observed=y_vals)

        trace = pm.sample(
            draws=_draws,
            tune=_tune,
            chains=_chains,
            progressbar=True,
            random_seed=list(range(42, 42 + _chains)),
            target_accept=_ta,
            nuts_sampler_kwargs={"max_treedepth": _mtd},
            init=_init,
            return_inferencedata=True,
        )

    # ── Diagnostica convergenza ──────────────────────────────────────────
    try:
        rhat_vals = az.rhat(trace).to_array().values.flatten()
        ess_vals  = az.ess(trace, method="bulk").to_array().values.flatten()
        rhat_max  = float(np.nanmax(rhat_vals))
        ess_pos   = ess_vals[np.isfinite(ess_vals) & (ess_vals > 0)]
        ess_min   = float(np.min(ess_pos)) if len(ess_pos) > 0 else 0.0
    except Exception:
        rhat_max, ess_min = np.nan, np.nan

    if np.isnan(rhat_max):
        print("  ⚠ Diagnostica convergenza non disponibile")
    elif rhat_max > 1.05:
        print(f"  ⚠ CONVERGENZA DUBBIA: Rhat max={rhat_max:.3f} > 1.05 "
              f"— interpretare con cautela")
    elif rhat_max > 1.01:
        print(f"  ⚠ Rhat max={rhat_max:.3f} (1.01–1.05) — lieve non-convergenza")
    else:
        print(f"  ✓ Convergenza ok: Rhat max={rhat_max:.3f}  ESS min={ess_min:.0f}")

    lo_pct   = (alpha / 2) * 100
    hi_pct   = (1 - alpha / 2) * 100
    tau_post = trace.posterior["tau"].values.flatten()
    b1_post  = trace.posterior["b1"].values.flatten()
    b2_post  = trace.posterior["b2"].values.flatten()
    a1_post  = trace.posterior["a1"].values.flatten()

    return {
        "tau_mean": float(np.mean(tau_post)),
        "tau_lo":   float(np.percentile(tau_post, lo_pct)),
        "tau_hi":   float(np.percentile(tau_post, hi_pct)),
        "tau_idx":  int(np.clip(round(float(np.median(tau_post))), 1, n - 2)),
        "tau_post": tau_post,
        "b1_mean":  float(np.mean(b1_post)),
        "b1_lo":    float(np.percentile(b1_post, lo_pct)),
        "b1_hi":    float(np.percentile(b1_post, hi_pct)),
        "b2_mean":  float(np.mean(b2_post)),
        "b2_lo":    float(np.percentile(b2_post, lo_pct)),
        "b2_hi":    float(np.percentile(b2_post, hi_pct)),
        "a1_mean":  float(np.mean(a1_post)),
        "nu_mean":  float(np.mean(trace.posterior["nu"].values.flatten())),
        # AR(1) rimosso — rho_mean sostituito da diagnostici convergenza
        "rhat_max": rhat_max,
        "ess_min":  ess_min,
    }


def regression_diagnostics(residuals, y_hat, ax_res, ax_qq, ax_acf):
    """Test Breusch-Pagan, Shapiro-Wilk, Durbin-Watson + plot."""
    from statsmodels.stats.diagnostic import het_breuschpagan
    import statsmodels.api as sm

    # Breusch-Pagan
    try:
        bp_stat, bp_p, _, _ = het_breuschpagan(residuals, sm.add_constant(y_hat))
    except Exception:
        bp_stat, bp_p = np.nan, np.nan

    # Shapiro-Wilk
    try:
        sw_stat, sw_p = stats.shapiro(residuals[:5000])
    except Exception:
        sw_stat, sw_p = np.nan, np.nan

    # Durbin-Watson
    dw = durbin_watson(residuals)

    # Plot residuals vs fitted
    ax_res.scatter(y_hat, residuals, alpha=0.5, s=15, color="#3498db", edgecolors="none")
    ax_res.axhline(0, color="red", lw=1.5, linestyle="--")
    ax_res.set_xlabel("Valori adattati", fontsize=9)
    ax_res.set_ylabel("Residui", fontsize=9)
    bp_verdict = f"p={bp_p:.3f} {'✗ eterosc.' if bp_p < ALPHA else '✓ omog.'}"
    ax_res.set_title(f"Residui vs Fitted\nBreusch-Pagan: {bp_verdict}", fontsize=9)

    # QQ plot
    stats.probplot(residuals, dist="norm", plot=ax_qq)
    sw_verdict = f"p={sw_p:.3f} {'✗ non-norm.' if sw_p < ALPHA else '✓ norm.'}"
    ax_qq.set_title(f"QQ Plot\nShapiro-Wilk: {sw_verdict}", fontsize=9)

    # ACF residui
    plot_acf(residuals, lags=min(20, len(residuals) // 2 - 1), ax=ax_acf, zero=False)
    dw_verdict = f"{'✗ autocorr.' if dw < 1.5 or dw > 2.5 else '✓ ok'}"
    ax_acf.set_title(f"ACF Residui\nDurbin-Watson: {dw:.3f} {dw_verdict}", fontsize=9)

    return {"BP_p": bp_p, "SW_p": sw_p, "DW": dw}


def bootstrap_delta(pre_vals, post_vals, n_boot=2000, seed=42):
    """Bootstrap 95% CI su Δ = mean(post) - mean(pre)."""
    rng = np.random.default_rng(seed)
    deltas = np.array([
        rng.choice(post_vals, len(post_vals), replace=True).mean() -
        rng.choice(pre_vals,  len(pre_vals),  replace=True).mean()
        for _ in range(n_boot)
    ])
    return float(np.mean(deltas)), float(np.percentile(deltas, 2.5)), float(np.percentile(deltas, 97.5))


def bh_correction(p_values, alpha=0.05):
    """Benjamini-Hochberg FDR correction. Restituisce array booleano di rigetti."""
    p = np.array(p_values, dtype=float)
    n = len(p)
    order = np.argsort(p)
    ranked = np.empty(n, dtype=float)
    ranked[order] = np.arange(1, n + 1)
    threshold = ranked / n * alpha
    reject = p <= threshold
    # Ensure monotonicity: se rang k è rigettato, tutti i ranghi < k lo sono
    cum_reject = np.zeros(n, dtype=bool)
    for i in range(n - 1, -1, -1):
        if reject[order[i]]:
            cum_reject[order[:i + 1]] = True
    return cum_reject


# ─────────────────────────────────────────────────────────────────────────────
# Carica dataset
# ─────────────────────────────────────────────────────────────────────────────
merged = pd.read_csv("data/dataset_merged.csv", index_col=0, parse_dates=True)
print(f"Dataset: {len(merged)} settimane | "
      f"{merged.index[0].date()} → {merged.index[-1].date()}")


# ─────────────────────────────────────────────────────────────────────────────
# SEZIONE B: Changepoint bayesiano sui LOG-PREZZI (Table 1)
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "="*70)
print("SEZIONE B — Changepoint bayesiano sui log-prezzi (Table 1)")
print("="*70)

table1_rows = []
diag_rows   = []

plt.rcParams.update({
    "font.family": "serif",
    "font.serif":  ["Times New Roman", "DejaVu Serif"],
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "axes.grid":   True,
    "grid.color":  "#e8e8e8",
    "grid.linewidth": 0.6,
    "figure.dpi":  DPI,
})

for event_name, cfg in EVENTS.items():
    shock   = pd.Timestamp(cfg["shock_date"])
    ev_col  = cfg["color"]
    safe_e  = (event_name.replace(" ", "_").replace("(", "").replace(")", "")
               .replace("/", "").replace(",", ""))

    for series_name, (log_col, raw_col, log_ylabel, raw_ylabel, fc) in PRICE_SERIES.items():
        if log_col not in merged.columns:
            print(f"  SKIP {event_name} | {series_name}: colonna {log_col} mancante")
            continue

        df_ev = merged.loc[cfg["window_start"]:cfg["window_end"], log_col].dropna()
        if len(df_ev) < 8:
            print(f"  SKIP {event_name} | {series_name}: <8 osservazioni")
            continue

        print(f"\n  MCMC → {event_name} | {series_name} ({len(df_ev)} punti)")
        x = np.arange(len(df_ev), dtype=float)
        y = df_ev.values

        _scenario_key = f"{event_name}__{series_name}"
        _mcmc_cfg     = _get_mcmc_cfg(_scenario_key)
        if _scenario_key in MCMC_CONFIG:
            print(f"  [cfg] override attivo: {MCMC_CONFIG[_scenario_key]}")
        ci = bayesian_changepoint(x, y, mcmc_cfg=_mcmc_cfg)

        # OLS piecewise per R² e doubling time
        cp = ci["tau_idx"]
        def _linreg(xv, yv):
            if len(xv) < 2:
                return 0., 0., 0.
            s, i, r, *_ = stats.linregress(xv, yv)
            return s, i, r ** 2

        b1_ols, a1_ols, r2_1 = _linreg(x[:cp], y[:cp])
        b2_ols, _,     r2_2  = _linreg(x[cp:], y[cp:])
        a2_ols = a1_ols + cp * (b1_ols - b2_ols)

        # Lag D (giorni tra shock e τ̂)
        base_date  = df_ev.index[0]
        cp_date    = base_date + pd.Timedelta(weeks=ci["tau_idx"])
        cp_lo_date = base_date + pd.Timedelta(weeks=max(0, round(ci["tau_lo"])))
        cp_hi_date = base_date + pd.Timedelta(weeks=min(len(df_ev) - 1, round(ci["tau_hi"])))
        lag_days   = (cp_date - shock).days

        doubling_t1 = np.log(2) / (b1_ols / 7) if b1_ols > 0 else np.inf
        doubling_t2 = np.log(2) / (b2_ols / 7) if b2_ols > 0 else np.inf

        table1_rows.append({
            "Evento":    event_name,
            "Serie":     series_name,
            "tau":       cp_date.date(),
            "CI_95_lo":  cp_lo_date.date(),
            "CI_95_hi":  cp_hi_date.date(),
            "Lag (gg)":  lag_days,
            "H0_rif":    "SÌ" if abs(lag_days) < H0_THRESHOLD else "NO",
            "b1_OLS":    round(b1_ols, 5),
            "b2_OLS":    round(b2_ols, 5),
            "DT1 (gg)":  round(doubling_t1, 1) if doubling_t1 != np.inf else "∞",
            "DT2 (gg)":  round(doubling_t2, 1) if doubling_t2 != np.inf else "∞",
            "R2_pre":      round(r2_1, 4),
            "R2_post":     round(r2_2, 4),
            "nu_StudentT": round(ci["nu_mean"], 2),
            "rhat_max":    round(ci["rhat_max"], 3),
            "ess_min":     round(ci["ess_min"],  0),
        })

        # ─── Plot A: changepoint con KDE posteriore ───────────────────────
        fig_a = plt.figure(figsize=(12, 5.8))
        gs    = fig_a.add_gridspec(2, 1, height_ratios=[4, 1], hspace=0.0)
        ax    = fig_a.add_subplot(gs[0])
        ax_k  = fig_a.add_subplot(gs[1], sharex=ax)

        ax.scatter(df_ev.index, y, s=20, color="black", alpha=0.60, zorder=3,
                   label="log(prezzo)")
        # Rette piecewise con CI slopes
        ax.plot(df_ev.index[:cp], a1_ols + ci["b1_mean"] * x[:cp],
                color="#27ae60", lw=2.5,
                label=f"Pre-shock  β={ci['b1_mean']:.4f}")
        ax.fill_between(df_ev.index[:cp],
                        a1_ols + ci["b1_lo"] * x[:cp],
                        a1_ols + ci["b1_hi"] * x[:cp],
                        color="#27ae60", alpha=0.10)
        ax.plot(df_ev.index[cp:], a2_ols + ci["b2_mean"] * x[cp:],
                color="#e74c3c", lw=2.5,
                label=f"Post-shock β={ci['b2_mean']:.4f}")
        ax.fill_between(df_ev.index[cp:],
                        a2_ols + ci["b2_lo"] * x[cp:],
                        a2_ols + ci["b2_hi"] * x[cp:],
                        color="#e74c3c", alpha=0.10)

        # τ̂ e CI
        ax.axvline(cp_date, color="#2980b9", lw=2.0, ls="--",
                   label=f"τ̂ = {cp_date.date()}")
        ax.axvspan(cp_lo_date, cp_hi_date, alpha=0.13, color="#2980b9",
                   label=f"CI 95%: {cp_lo_date.strftime('%d %b %y')} – "
                         f"{cp_hi_date.strftime('%d %b %y')}")
        ax.axvline(shock, color=ev_col, lw=1.8, ls=":",
                   label=f"Shock = {cfg['shock_date']}")

        lag_s = f"D = {lag_days:+d} gg"
        dt1_s = f"{doubling_t1:.0f}" if doubling_t1 != np.inf else "∞"
        dt2_s = f"{doubling_t2:.0f}" if doubling_t2 != np.inf else "∞"
        convergence_label = (
            "✓ converged" if ci["rhat_max"] <= 1.01
            else f"⚠ Rhat={ci['rhat_max']:.3f}"
        )
        ax.set_title(
            f"{event_name} — {series_name}\n"
            f"{lag_s}  |  DT₁={dt1_s}gg → DT₂={dt2_s}gg  |  "
            f"ν={ci['nu_mean']:.1f}  {convergence_label}",
            fontsize=12, fontweight="bold",
            color="#c0392b" if abs(lag_days) < H0_THRESHOLD else "black",
        )
        ax.set_ylabel(log_ylabel, fontsize=11)
        ax.legend(fontsize=8.5, loc="upper left")
        ax.tick_params(axis="x", labelbottom=False)

        # KDE posteriore τ
        tau_num  = mdates.date2num(df_ev.index[0]) + ci["tau_post"] * 7
        kde_fn   = gaussian_kde(tau_num, bw_method=0.25)
        t_grid   = np.linspace(tau_num.min(), tau_num.max(), 400)
        density  = kde_fn(t_grid) / kde_fn(t_grid).max()
        ax_k.fill_between(mdates.num2date(t_grid), 0, density, alpha=0.5, color="#2980b9")
        ax_k.plot(mdates.num2date(t_grid), density, color="#1a5276", lw=1.6)
        ax_k.axvline(cp_date, color="#2980b9", lw=2.0, ls="--", zorder=5)
        ax_k.axvspan(cp_lo_date, cp_hi_date, alpha=0.13, color="#2980b9")
        ax_k.axvline(shock, color=ev_col, lw=1.8, ls=":", zorder=5)
        ax_k.set_ylim(0, 1.6)
        ax_k.set_ylabel("p(τ)", fontsize=9, color="#1a5276")
        ax_k.set_yticks([])
        ax_k.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
        ax_k.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
        ax_k.tick_params(axis="x", rotation=45, labelsize=9)

        safe_s = series_name.lower()
        fname_a = f"plots/02_{safe_e}_{safe_s}.png"
        fig_a.tight_layout(pad=1.5)
        fig_a.savefig(fname_a, dpi=DPI, bbox_inches="tight")
        plt.close(fig_a)
        print(f"    Salvato: {fname_a}  |  lag={lag_days:+d}gg")

        # ─── Plot B: Diagnostica regressione ─────────────────────────────
        # Residui OLS piecewise
        y_hat = np.where(np.arange(len(x)) < cp,
                         a1_ols + b1_ols * x,
                         a2_ols + b2_ols * x)
        residuals = y - y_hat

        fig_b, axes_d = plt.subplots(1, 3, figsize=(14, 4))
        fig_b.suptitle(
            f"Diagnostica regressione piecewise — {event_name} | {series_name}",
            fontsize=11, fontweight="bold",
        )
        diag_stats = regression_diagnostics(residuals, y_hat, *axes_d)

        issues = (["eterosch." if diag_stats["BP_p"] < ALPHA else ""] +
                  ["non-norm." if diag_stats["SW_p"] < ALPHA else ""] +
                  ["autocorr." if diag_stats["DW"] < 1.5 or diag_stats["DW"] > 2.5 else ""])
        issues = [i for i in issues if i]
        footer_txt = (f"⚠ Violazioni: {', '.join(issues)} → CI bayesiani più affidabili"
                      if issues else "✓ Ipotesi OLS soddisfatte")
        fig_b.text(0.5, -0.03, footer_txt, ha="center", fontsize=9,
                   color="#c0392b" if issues else "#1a7a1a",
                   bbox=dict(boxstyle="round,pad=0.3", fc="#f8f8f8",
                             ec="#c0392b" if issues else "#1a7a1a", lw=1.2))
        fig_b.tight_layout(pad=1.5)
        fname_b = f"plots/02_{safe_e}_{safe_s}_diag.png"
        fig_b.savefig(fname_b, dpi=DPI, bbox_inches="tight")
        plt.close(fig_b)

        diag_rows.append({
            "Evento": event_name, "Serie": series_name, **diag_stats,
        })

pd.DataFrame(table1_rows).to_csv("data/table1_changepoints.csv", index=False)
pd.DataFrame(diag_rows).to_csv("data/regression_diagnostics.csv", index=False)
print(f"\n  Salvato: data/table1_changepoints.csv ({len(table1_rows)} righe)")

# ─────────────────────────────────────────────────────────────────────────────
# SEZIONE C: Costruzione margini lordi
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "="*70)
print("SEZIONE C — Costruzione margini lordi")
print("="*70)

# Normalizzazione unità pompa
pump_b_med = merged["benzina_4w"].dropna().median()
pump_d_med = merged["diesel_4w"].dropna().median()
uf_b = 1000.0 if pump_b_med > 10 else 1.0
uf_d = 1000.0 if pump_d_med > 10 else 1.0
merged["benzina_eur_l"] = merged["benzina_4w"] / uf_b
merged["diesel_eur_l"]  = merged["diesel_4w"]  / uf_d
if "eurusd" not in merged.columns or merged["eurusd"].dropna().mean() < 0.5:
    merged["eurusd"] = 1.08
    print("  ATTENZIONE: EUR/USD non trovato → usando fallback 1.08")
merged["eurusd"] = merged["eurusd"].ffill().bfill()


def _load_investing_csv(filepath, price_col="Price"):
    """Carica CSV da Investing.com (formato MM/DD/YYYY, virgole migliaia)."""
    df = pd.read_csv(filepath, thousands=",")
    df.columns = [c.lstrip("\ufeff") for c in df.columns]
    df["Date"] = pd.to_datetime(df["Date"], format="%m/%d/%Y", errors="coerce")
    df = df.dropna(subset=["Date"]).set_index("Date").sort_index()
    if price_col in df.columns:
        df[price_col] = pd.to_numeric(
            df[price_col].astype(str).str.replace(",", ""), errors="coerce")
    return df


# Carica futures CSV (solo Eurobob e Gas Oil — metodo Yield rimosso)
futures_ok = {"eurobob": False, "gasoil": False}

try:
    eurobob_csv = _load_investing_csv("data/Eurobob Futures Historical Data.csv")
    eurobob_csv.rename(columns={"Price": "eurobob_usd_tonne"}, inplace=True)
    eb_w = eurobob_csv["eurobob_usd_tonne"].resample("W-MON").mean().ffill()
    merged = merged.join(eb_w.rename("eurobob_usd_tonne"), how="left")
    merged["eurobob_eur_l"] = (merged["eurobob_usd_tonne"] / merged["eurusd"]
                                / L_PER_TONNE_BENZ)
    futures_ok["eurobob"] = True
    print(f"  Eurobob futures: {len(eurobob_csv)} righe")
except Exception as e:
    print(f"  Eurobob futures: ERRORE ({e}) → crack spread benzina non disponibile")

try:
    gasoil_csv = _load_investing_csv("data/London Gas Oil Futures Historical Data.csv")
    gasoil_csv.rename(columns={"Price": "gasoil_usd_tonne"}, inplace=True)
    go_w = gasoil_csv["gasoil_usd_tonne"].resample("W-MON").mean().ffill()
    merged = merged.join(go_w.rename("gasoil_usd_tonne"), how="left")
    merged["gasoil_eur_l"] = (merged["gasoil_usd_tonne"] / merged["eurusd"]
                               / L_PER_TONNE_DIES)
    futures_ok["gasoil"] = True
    print(f"  Gas Oil futures: {len(gasoil_csv)} righe")
except Exception as e:
    print(f"  Gas Oil futures: ERRORE ({e}) → crack spread diesel non disponibile")

# Calcolo margini (solo crack spread wholesale europeo)
if futures_ok["eurobob"]:
    merged["margine_benz_crack"] = merged["benzina_eur_l"] - merged["eurobob_eur_l"]

if futures_ok["gasoil"]:
    merged["margine_dies_crack"] = merged["diesel_eur_l"]  - merged["gasoil_eur_l"]

merged.to_csv("data/dataset_merged_with_futures.csv")

# ─── Baseline thresholds (2σ da 2019) ────────────────────────────────────────
baseline = merged.loc[BASELINE_START:BASELINE_END]
thresholds = {}
for col in ["margine_benz_crack", "margine_dies_crack"]:
    if col in baseline.columns:
        vals = baseline[col].dropna()
        thresholds[col] = float(2 * vals.std()) if len(vals) >= 4 else 0.03
        print(f"  Soglia 2σ baseline 2019 | {col}: {thresholds[col]:.5f} EUR/L")

# ─────────────────────────────────────────────────────────────────────────────
# Plot: margini nel tempo
# ─────────────────────────────────────────────────────────────────────────────
WAR_EVENTS_CFG = {ev: (cfg["shock_date"], cfg["color"]) for ev, cfg in EVENTS.items()}

def _margin_plot(col, fuel_name, color, title):
    if col not in merged.columns:
        return
    series = merged[col].dropna()
    if len(series) < 5:
        return
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(series.index, series.values, color=color, lw=2.0, label=fuel_name)
    # Baseline band (mean ± 2σ dal 2019)
    bl_vals = baseline[col].dropna()
    if len(bl_vals) >= 4:
        ax.axhspan(bl_vals.mean() - 2*bl_vals.std(),
                   bl_vals.mean() + 2*bl_vals.std(),
                   alpha=0.12, color="#888888", label="Baseline ±2σ (2019)")
        ax.axhline(bl_vals.mean(), color="#888888", lw=1.0, ls="--")
    for label, (date, c) in WAR_EVENTS_CFG.items():
        ts = pd.Timestamp(date)
        if series.index[0] <= ts <= series.index[-1]:
            ax.axvline(ts, color=c, lw=1.8, ls="--", alpha=0.9)
            ax.text(ts + pd.Timedelta(days=5), series.max() * 0.97,
                    label.split(" ")[0], rotation=90, fontsize=9, color=c, va="top")
    ax.set_ylabel("Margine lordo (EUR/litro)", fontsize=12)
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.legend(fontsize=10, loc="upper left")
    ax.grid(alpha=0.3)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    plt.xticks(rotation=45)
    plt.tight_layout()
    fname = f"plots/07_margins_{col}.png"
    fig.savefig(fname, dpi=DPI, bbox_inches="tight")
    plt.close()
    print(f"  Salvato: {fname}")


_margin_plot("margine_benz_crack", "Benzina (crack spread Eurobob)", "#e67e22",
             "Margine lordo Benzina — metodo Crack Spread (Eurobob)")
_margin_plot("margine_dies_crack", "Diesel (crack)", "#8e44ad",
             "Margine lordo Diesel — metodo Crack Spread (Gas Oil)")


# ─────────────────────────────────────────────────────────────────────────────
# SEZIONE D: Test anomalia margine + BH correction (Table 2)
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "="*70)
print("SEZIONE D — Test anomalia margine (Table 2) + Benjamini-Hochberg FDR")
print("="*70)

# Definisci serie da testare — solo crack spread wholesale europeo
SERIES_TO_TEST = []
if futures_ok["eurobob"]:
    SERIES_TO_TEST += [("Benzina", "margine_benz_crack", "Crack Eurobob")]
if futures_ok["gasoil"]:
    SERIES_TO_TEST += [("Diesel",  "margine_dies_crack", "Crack Gas Oil")]

# Hormuz escluso dalla Table 2: dati post-shock insufficienti (≤4 settimane).
# L'evento è troppo recente per un test affidabile del margine.
EVENTS_TABLE2 = {
    k: v for k, v in EVENTS.items() if "Hormuz" not in k
}
print("  Nota: Hormuz (Feb 2026) escluso da Table 2 — post-shock < 5 settimane.")

if not SERIES_TO_TEST:
    print("  ATTENZIONE: nessun futures disponibile. Sezione D saltata.")
else:
    results_d = []

    for event_name, cfg in EVENTS_TABLE2.items():
        shock = pd.Timestamp(cfg["shock_date"])

        for fuel_name, margine_col, method_name in SERIES_TO_TEST:
            if margine_col not in merged.columns:
                continue

            df_ev = merged.loc[cfg["window_start"]:cfg["window_end"]].dropna(subset=[margine_col])
            if len(df_ev) < 10:
                print(f"  SKIP {event_name}|{fuel_name}|{method_name}: <10 obs")
                continue

            shock_idx = int(np.clip(df_ev.index.searchsorted(shock), 2, len(df_ev) - 2))
            pre_m  = df_ev.iloc[:shock_idx][margine_col].dropna()
            post_m = df_ev.iloc[shock_idx:][margine_col].dropna()

            if len(pre_m) < 3 or len(post_m) < 3:
                continue

            # Test frequentisti
            t_stat, t_p   = stats.ttest_ind(post_m.values, pre_m.values, equal_var=False)
            ks_stat, ks_p = stats.ks_2samp(pre_m.values, post_m.values)
            delta_mean    = float(post_m.mean() - pre_m.mean())
            boot_mean, boot_lo, boot_hi = bootstrap_delta(pre_m.values, post_m.values)

            # MCMC sul margine
            print(f"\n  MCMC margine → {event_name} | {fuel_name} | {method_name}")
            x_vals = np.arange(len(df_ev), dtype=float)
            y_vals = df_ev[margine_col].values.astype(float)
            _scenario_key_m = f"{event_name}__{fuel_name}__{method_name}"
            _mcmc_cfg_m     = _get_mcmc_cfg(_scenario_key_m)
            if _scenario_key_m in MCMC_CONFIG:
                print(f"  [cfg] override attivo: {MCMC_CONFIG[_scenario_key_m]}")
            ci_m   = bayesian_changepoint(x_vals, y_vals, mcmc_cfg=_mcmc_cfg_m)

            cp_idx     = ci_m["tau_idx"]
            cp_date    = df_ev.index[cp_idx]
            no_break   = (cp_idx < int(len(df_ev) * EDGE_FRAC) or
                          cp_idx > int(len(df_ev) * (1 - EDGE_FRAC)))
            lag_vs_sh  = (cp_date - shock).days

            # Classificazione (p-value corretti DOPO raccolta di tutti i test)
            soglia    = thresholds.get(margine_col, 0.03)
            anomalo   = abs(delta_mean) > soglia
            ci_non0   = (boot_lo > 0) or (boot_hi < 0)
            # Decisione statistica — test primario UNICO: Welch t-test.
            # KS e bootstrap CI sono diagnostici ausiliari (riportati ma non
            # usati nella regola di rigetto), per coerenza con la BH correction
            # che è applicata solo su t_p.
            # Usare AND(t, KS) come gate creerebbe un test composito non standard
            # con alpha nominale non controllato dalla BH sul solo t_p.
            # Rif: Benjamini & Hochberg (1995); Holm (1979).
            stat_sig  = (t_p < ALPHA)   # solo Welch t come test primario

            if stat_sig and anomalo and delta_mean > 0:
                clas = "MARGINE ANOMALO POSITIVO"
            elif stat_sig and anomalo and delta_mean < 0:
                clas = "COMPRESSIONE MARGINE"
            elif not anomalo:
                clas = "NEUTRO / TRASMISSIONE ATTESA"
            elif stat_sig and not anomalo:
                clas = "VARIAZIONE STATISTICA"
            else:
                clas = "INCONCLUSIVO"

            print(f"    Δ={delta_mean:+.5f} EUR/L  [CI: {boot_lo:+.5f},{boot_hi:+.5f}]")
            print(f"    soglia 2σ={soglia:.5f} | t_p={t_p:.4f} | ks_p={ks_p:.4f} | "
                  f"τ={cp_date.date()} (lag {lag_vs_sh:+d}gg) | → {clas}")

            results_d.append({
                "Evento":              event_name,
                "Serie":               fuel_name,
                "Metodo":              method_name,
                "n_pre":               len(pre_m),
                "n_post":              len(post_m),
                "delta_margine_eur":   round(delta_mean, 5),
                "boot_CI_lo":          round(boot_lo, 5),
                "boot_CI_hi":          round(boot_hi, 5),
                "soglia_2sigma":       round(soglia, 5),
                "delta_anomalo":       anomalo,
                "t_p":                 round(float(t_p), 4),
                "ks_p":                round(float(ks_p), 4),
                "tau_margine":         cp_date.date(),
                "lag_tau_vs_shock":    lag_vs_sh,
                "break_strutturale":   not no_break,
                "nu_StudentT":         round(ci_m["nu_mean"], 2),
                "rhat_max":            round(ci_m["rhat_max"], 3),
                "ess_min":             round(ci_m["ess_min"],  0),
                "classificazione":     clas,
            })

    # ── Benjamini-Hochberg FDR correction ────────────────────────────────────
    if results_d:
        df_res = pd.DataFrame(results_d)
        # BH correction su t_p (Welch t-test = test primario).
        # ks_p è test ausiliario — riportato ma non usato per FDR.
        # Usare min(t_p, ks_p) deflaziona artificialmente i p-value
        # (equivale a scegliere il test più favorevole dopo aver guardato).
        p_combined = df_res["t_p"].values.astype(float)
        bh_reject  = bh_correction(p_combined, alpha=ALPHA)
        df_res["BH_reject_FDR5%"] = bh_reject

        # Rivaluta classificazione con BH correction
        def _reclassify_bh(row):
            if not row["BH_reject_FDR5%"]:
                return ("NEUTRO / TRASMISSIONE ATTESA"
                        if row["delta_anomalo"] is False
                        else "VARIAZIONE STATISTICA")
            return row["classificazione"]

        df_res["classificazione_BH"] = df_res.apply(_reclassify_bh, axis=1)

        df_res.to_csv("data/table2_margin_anomaly.csv", index=False)
        print(f"\n  Salvato: data/table2_margin_anomaly.csv ({len(df_res)} righe)")
        print(f"  BH correction: {bh_reject.sum()} / {len(bh_reject)} test rigettati a FDR 5%")

        # ── Sensitivity analysis baseline ─────────────────────────────────
        # Verifica robustezza soglia 2σ rispetto alla scelta di baseline:
        #   A) 2019 full  (PRIMARIA — pre-COVID, pre-crisi, mercato maturo)
        #   B) Full 2021  (sensitivity check — post-COVID recovery, prezzi in rimbalzo)
        # H1-2021 rimosso: era un fallback temporaneo usato quando il dataset
        # partiva da 2021; ora che 2019 è disponibile come baseline primaria
        # non aggiunge informazione rispetto a Full-2021.
        # Rif: Benjamini & Hochberg (1995); Tukey (1991).
        sens_rows = []
        for bl_label, bl_start, bl_end in [
            ("2019_full",  "2019-01-01", "2019-12-31"),   # PRIMARY baseline
            ("Full_2021",  "2021-01-01", "2021-12-31"),   # sensitivity check
        ]:
            for col in ["margine_benz_crack", "margine_dies_crack"]:
                if col not in merged.columns:
                    continue
                bl_vals = merged.loc[bl_start:bl_end, col].dropna()
                if len(bl_vals) < 4:
                    continue
                sens_rows.append({
                    "baseline":   bl_label,
                    "serie":      col,
                    "n_weeks":    len(bl_vals),
                    "mean":       round(float(bl_vals.mean()), 5),
                    "std":        round(float(bl_vals.std()),  5),
                    "soglia_2sigma": round(float(2 * bl_vals.std()), 5),
                })
        if sens_rows:
            pd.DataFrame(sens_rows).to_csv("data/baseline_sensitivity.csv", index=False)
            print("  Salvato: data/baseline_sensitivity.csv (sensitivity baseline)")
            # Stampa confronto rapido
            df_sens = pd.DataFrame(sens_rows)
            for _, r in df_sens.iterrows():
                print(f"    {r['baseline']:12} | {r['serie']:22} | "
                      f"σ={r['std']:.4f} | soglia 2σ={r['soglia_2sigma']:.4f}")

        # ── Plot Δmargine riassuntivo ─────────────────────────────────────
        fig_s, ax_s = plt.subplots(figsize=(14, max(5, len(df_res) * 0.85)))
        labels  = [f"{r['Evento'].split('(')[0].strip()}\n{r['Serie']} — {r['Metodo']}"
                   for _, r in df_res.iterrows()]
        deltas  = df_res["delta_margine_eur"].values
        ci_lo   = df_res["boot_CI_lo"].values
        ci_hi   = df_res["boot_CI_hi"].values
        colors  = [CLAS_COLOR.get(c, "#555555") for c in df_res["classificazione_BH"]]

        ax_s.barh(range(len(df_res)), deltas, color=colors,
                  alpha=0.78, edgecolor="black", lw=0.7)
        for i in range(len(df_res)):
            ax_s.errorbar(deltas[i], i,
                          xerr=[[deltas[i] - ci_lo[i]], [ci_hi[i] - deltas[i]]],
                          fmt="none", color="black", capsize=5, lw=1.8)
            ax_s.text(max(ci_hi[i], deltas[i]) + 0.003, i,
                      df_res.iloc[i]["classificazione_BH"][:28], va="center", fontsize=8)

        ax_s.axvline(0, color="black", lw=0.8)
        ax_s.set_yticks(range(len(df_res)))
        ax_s.set_yticklabels(labels, fontsize=8.5)
        ax_s.set_xlabel("Δ margine lordo post-shock (EUR/litro)", fontsize=11)
        ax_s.set_title("Variazione margine lordo — Test anomalia margine\n"
                       "(classificazione con Benjamini-Hochberg FDR 5% — etichette descrittive, non causali)",
                       fontsize=12, fontweight="bold")
        ax_s.legend(handles=[
            mpatches.Patch(color=c, label=k) for k, c in CLAS_COLOR.items()
        ], fontsize=8, loc="lower right")
        ax_s.grid(alpha=0.3, axis="x")
        plt.tight_layout(pad=1.5)
        plt.savefig("plots/07_delta_summary.png", dpi=DPI, bbox_inches="tight")
        plt.close()
        print("  Salvato: plots/07_delta_summary.png")

        # Metodo Yield rimosso → nessun confronto metodi necessario.

print("\n\nScript 02 completato.")
print(f"  Table 1 (changepoints):    data/table1_changepoints.csv")
print(f"  Table 2 (margini, no Hor): data/table2_margin_anomaly.csv")
print(f"  BH correction FDR 5%:      su t_p (Welch, test primario unico)")
print(f"  KS / bootstrap CI:         diagnostici ausiliari (riportati, non nel gate)")
print(f"  Nomenclatura:              MARGINE ANOMALO POSITIVO (non 'speculazione')")
print(f"  Baseline 2019:             data/baseline_sensitivity.csv")
print(f"  Hormuz:                    escluso — dati post-shock insufficienti")
print(f"  Plot changepoints:         plots/02_*.png")
print(f"  Plot margini crack:        plots/07_margins_margine_*_crack.png")
print(f"  Plot delta summary:        plots/07_delta_summary.png")

plt.rcParams.update(plt.rcParamsDefault)