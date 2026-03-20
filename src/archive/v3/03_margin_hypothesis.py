"""
03_margin_hypothesis_v2.py  — Redesign critico
================================================
DIFFERENZE RISPETTO A v1 (difetti corretti):

  [FIX 1 — CRITICO] τ_margin rimosso dalla famiglia BH confirmatory.
    Era data-snooping: argmax|t| su tutti i split poi test sullo stesso split.
    → τ_margin è ora SOLO descrittivo (timing relativo a τ_price).

  [FIX 2 — CRITICO] Famiglia BH raccoglie SOLTANTO test su H₀: μ_post = μ_2019
    (livello assoluto vs baseline). Block permutation e HAC testano H₀_locale
    (salto pre→post) — domanda diversa — e vanno nell'esplorativa separata.

  [FIX 3 — MEDIO] HAC bandwidth: Andrews (1991) plug-in AR(1) invece di
    maxlags fisso=4. Per ρ̂≈0.85 il bandwidth ottimale è molto più alto;
    se supera n//2 viene flaggato come "test essenzialmente non informativo".

  [FIX 4 — MEDIO] n_eff = n·(1−ρ̂)/(1+ρ̂) calcolato e riportato accanto
    ad ogni test. Se n_eff < 5 viene apposta nota "CAUTELA".

ARCHITETTURA DEI LIVELLI EPISTEMICI
-------------------------------------
LIVELLO 1 — CONFIRMATORY  (→ confirmatory_pvalues_v2.csv → BH globale)
  H₀: μ_post = μ_2019  (one-sided upper)
  Split esogeni: shock_hard + τ_price  (entrambi indipendenti dal margine)
  Test: Welch 1-sample t + Mann-Whitney U (post vs distribuzione 2019)
  Famiglia: 2 eventi × 2 carburanti × 2 split × 2 test = 16 test

LIVELLO 2 — EXPLORATORY   (→ exploratory_results.csv, mai in BH)
  τ_margin: descrittivo (Δ locale prima/dopo, timing vs τ_price)
  Block permutation: H₀_locale salto pre→post
  HAC Andrews: H₀_locale salto pre→post
  n_eff, ρ̂: diagnostica della potenza

LIVELLO 3 — DIAGNOSTICA
  n_eff esplicitato per ogni test
  Flag automatico se n_eff < 5

Input:
  data/dataset_merged_with_futures.csv
  data/regression_diagnostics.csv
  data/table1_changepoints.csv

Output:
  data/all_test_results_v2.csv          ← master con colonna 'livello'
  data/confirmatory_pvalues_v2.csv      ← 16 righe (pulite, per BH globale)
  data/exploratory_results.csv          ← block perm, HAC, τ_margin (non in BH)
  data/neff_report.csv                  ← n_eff e ρ̂ per ogni serie×evento
  data/table2_margin_anomaly_v2.csv     ← compatibilità (split shock_hard)
  data/table2_multi_split_v2.csv        ← tutti gli split con livello epistemico
  data/baseline_sensitivity.csv
  data/annual_margin_analysis.csv
  plots/03_margins_v2.png
  plots/03_delta_summary_v2.png
  plots/03_split_comparison_v2.png
  plots/03_tau_lag_v2.png
  plots/03_annual_margins_v2.png
  plots/03_neff_report.png
"""

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
import statsmodels.api as sm
from scipy import stats
from scipy.stats import mannwhitneyu

warnings.filterwarnings("ignore")
os.makedirs("data",  exist_ok=True)
os.makedirs("plots", exist_ok=True)

# ── Costanti ──────────────────────────────────────────────────────────────────
ALPHA          = 0.05
N_PERM         = 10_000
SEED           = 42
BLOCK_SIZE     = 4
DPI            = 180
NEFF_WARN      = 5      # soglia per flag "CAUTELA: n_eff insufficiente"
BASELINE_START = "2019-01-01"
BASELINE_END   = "2019-12-31"

EVENTS = {
    "Ucraina (Feb 2022)": {
        "shock":       pd.Timestamp("2022-02-24"),
        "pre_start":   pd.Timestamp("2021-09-01"),
        "post_end":    pd.Timestamp("2022-08-31"),
        "preliminare": False,
    },
    "Iran-Israele (Giu 2025)": {
        "shock":       pd.Timestamp("2025-06-13"),
        "pre_start":   pd.Timestamp("2025-01-01"),
        "post_end":    pd.Timestamp("2025-10-31"),
        "preliminare": False,
    },
    "Hormuz (Feb 2026)": {
        "shock":       pd.Timestamp("2026-02-28"),
        "pre_start":   pd.Timestamp("2025-10-01"),
        "post_end":    pd.Timestamp("2026-04-27"),
        "preliminare": True,
    },
}

MARGIN_COLS = {
    "Benzina": "margine_benz_crack",
    "Diesel":  "margine_dies_crack",
}

WAR_DATES = {
    "Ucraina":  (pd.Timestamp("2022-02-24"), "#e74c3c"),
    "Iran":     (pd.Timestamp("2025-06-13"), "#e67e22"),
    "Hormuz":   (pd.Timestamp("2026-02-28"), "#8e44ad"),
}

SPLIT_LABELS = {
    "shock_hard": "Data shock (esogena)",
    "tau_price":  "τ_price (changepoint log-prezzo, MCMC)",
    "tau_margin": "τ_margin [DESCRITTIVO — non in BH]",
}

CLAS_COLOR = {
    "Margine anomalo — confermato":            "#c0392b",
    "Variazione significativa, non anomala":   "#e67e22",
    "Anomalia descrittiva, non confermata":    "#f39c12",
    "Neutro / trasmissione attesa":            "#27ae60",
    "Inconclusivo (n_eff insufficiente)":      "#bdc3c7",
}

# ── Proxy volumi settimanali (litri/settimana, 2022 — vedi nota limiti) ──
_VOL_L_WEEK = {"Benzina": 182_000_000, "Diesel": 596_000_000}


# =============================================================================
# UTILITY — n_eff, Andrews HAC, test
# =============================================================================

def compute_neff(series: np.ndarray):
    """
    Numero effettivo di osservazioni indipendenti via AR(1).
    n_eff = n · (1−ρ̂) / (1+ρ̂)   [Kish 1965]

    [FIX critic.2 — 27-apr-2026] Detrend lineare prima di stimare ρ̂.
    Senza detrend, un trend crescente/decrescente nella serie gonfia
    artificialmente ρ̂ (la differenza consecutive sembra correlata anche
    se il processo è quasi-iid attorno al trend). Il detrend rimuove
    la componente deterministica prima di calcolare la dipendenza seriale,
    allineandosi alla pratica standard (Andrews 1991, Newey-West 1987).

    Implementazione: np.polyfit(t, x, 1) → residui → lag-1 corrcoef.
    Equivalente a OLS(x ~ 1 + t) → ρ̂(resid).

    Restituisce (neff, rho_hat).
    """
    x = np.asarray(series, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) < 4:
        return float(len(x)), 0.0
    # Detrend: rimuovi trend lineare prima di stimare ρ̂
    t = np.arange(len(x), dtype=float)
    coefs = np.polyfit(t, x, 1)
    x_det = x - np.polyval(coefs, t)
    rho = float(np.corrcoef(x_det[:-1], x_det[1:])[0, 1])
    rho = np.clip(rho, -0.9999, 0.9999)
    neff = float(len(x)) * (1.0 - rho) / (1.0 + rho)
    return max(1.0, neff), rho


def andrews_bw_ar1(rho: float, n: int) -> int:
    """
    Andrews (1991) plug-in bandwidth per Newey-West con processo AR(1).
    Formula: m = ceil(1.1447 · (α̂·n)^{1/3})
    dove α̂ = 4ρ²/(1−ρ)⁴ per AR(1).

    Floored a 4 per non essere meno conservativo di v1.
    Capped a n//2 (oltre questo il test non è informativo — segnalato in flag).
    """
    rho = np.clip(rho, -0.9999, 0.9999)
    if abs(rho) < 0.05:
        return 4   # praticamente iid
    try:
        alpha_hat = 4 * rho**2 / (1 - rho)**4
        bw = 1.1447 * (alpha_hat * n) ** (1.0 / 3.0)
        return int(np.clip(np.ceil(bw), 4, n // 2))
    except (ZeroDivisionError, FloatingPointError):
        return n // 2


def neff_flag(neff: float) -> str:
    if neff < NEFF_WARN:
        return f"CAUTELA: n_eff={neff:.1f} < {NEFF_WARN} — test non informativo"
    if neff < 10:
        return f"ATTENZIONE: n_eff={neff:.1f} (interpreta con cautela)"
    return "ok"


def bh_correction(p_values: np.ndarray, alpha: float = ALPHA):
    """Benjamini-Hochberg (1995) con monotonicity enforcement."""
    p = np.array(p_values, dtype=float)
    n = len(p)
    if n == 0:
        return np.array([], dtype=bool), np.array([])
    order   = np.argsort(p)
    ranked  = np.empty(n, dtype=float)
    ranked[order] = np.arange(1, n + 1)
    p_adj   = np.minimum(1.0, p * n / ranked)
    p_adj_m = np.minimum.accumulate(p_adj[order][::-1])[::-1]
    p_out   = np.empty(n)
    p_out[order] = p_adj_m
    return p_out <= alpha, p_out


def _stars(p: float) -> str:
    if np.isnan(p):
        return ""
    return "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "n.s."


# =============================================================================
# LIVELLO 1 — test confirmatory (H₀: μ_post = μ_2019)
# =============================================================================

def hac_1sample_test(post: np.ndarray, mu_0: float, rho_hat: float) -> tuple:
    """
    Test one-sample H₀: E[post] = mu_0 con long-run variance (Newey-West HAC).

    Motivazione: con dati autocorrelati (ρ̂ ≈ 0.7–0.9), il Welch ttest_1samp
    standard assume osservazioni iid e sovrastima il t-statistic di un fattore
    ≈ sqrt(n/n_eff) ≈ 3–5×. Il test corretto usa la varianza di lungo periodo
    (long-run variance, LRV) stimata con Newey-West Andrews BW.

    Implementazione: OLS di (post − mu_0) su costante con errori HAC.
    Il t-stat risultante è t = (x̄ − mu_0) / sqrt(LRV/n), con LRV stimato
    via Newey-West (Andrews 1991 plug-in bandwidth).
    Il p-value è one-sided upper: p_one = p_two/2 se t > 0, else 1.

    Rif: Andrews (1991) Econometrica; Kiefer & Vogelsang (2005) Econometrica.

    Returns: (t_stat, p_one_sided)
    """
    n   = len(post)
    bw  = andrews_bw_ar1(rho_hat, n)
    x_c = post - mu_0   # centrato attorno all'ipotesi nulla
    try:
        res = sm.OLS(x_c, np.ones(n)).fit(
            cov_type="HAC",
            cov_kwds={"maxlags": bw, "use_correction": True}
        )
        t_stat  = float(res.tvalues[0])
        p_two   = float(res.pvalues[0])
        p_one   = p_two / 2.0 if t_stat > 0 else 1.0
    except Exception:
        # fallback iid se HAC fallisce (dati degeneri)
        t_stat, p_two = float(stats.ttest_1samp(post, popmean=mu_0)[0]), \
                        float(stats.ttest_1samp(post, popmean=mu_0)[1])
        p_one = p_two / 2.0 if t_stat > 0 else 1.0
    return t_stat, p_one


def run_confirmatory(post: np.ndarray, baseline_vals: np.ndarray,
                     mu_2019: float, soglia: float):
    """
    Famiglia confirmatory — H₀: μ_post = μ_2019 (livello assoluto, one-sided).

    [v2.1 FIX — DeepSeek review] Il test sul livello è ora HAC one-sample
    invece di Welch iid. Con ρ̂ ≈ 0.7–0.9, il Welch classico gonfia il
    t-stat di un fattore ≈ 3–5× (n/n_eff). Il test HAC usa la long-run
    variance (Newey-West, Andrews 1991 bandwidth) per la SE della media.
    Mann-Whitney rimane invariato (non-parametrico, già robusto).

    Parameters
    ----------
    post           : valori del margine nel periodo post-split
    baseline_vals  : distribuzione del margine nel 2019
    mu_2019        : media baseline 2019
    soglia         : 2σ_2019 (soglia anomalia descrittiva)

    Returns
    -------
    dict con t_p (HAC), mw_p, delta_vs_baseline, anomalo, neff, rho, neff_flag_str
    """
    if len(post) < 4 or len(baseline_vals) < 4:
        return None

    # n_eff e ρ̂ — necessari per il bandwidth Andrews
    neff, rho = compute_neff(post)
    flag      = neff_flag(neff)

    # HAC one-sample t-test (PRIMARIO — long-run variance, Andrews BW)
    t_stat, t_p = hac_1sample_test(post, mu_2019, rho)

    # Welch classico (RIFERIMENTO — riportato per confronto con letteratura)
    t_stat_welch, t_p_welch_2s = stats.ttest_1samp(post, popmean=mu_2019)
    t_p_welch = float(t_p_welch_2s) / 2.0 if t_stat_welch > 0 else 1.0

    # Mann-Whitney (post vs distribuzione 2019, upper — non parametrico)
    _, mw_p = mannwhitneyu(post, baseline_vals, alternative="greater")

    delta_vs_bl = float(post.mean() - mu_2019)
    anomalo     = delta_vs_bl > soglia

    return {
        "t_stat":            round(float(t_stat), 4),
        "t_p":               round(float(t_p), 4),         # HAC — primario
        "t_p_welch_iid":     round(float(t_p_welch), 4),   # Welch iid — riferimento
        "mw_p":              round(float(mw_p), 4),
        "delta_vs_baseline": round(delta_vs_bl, 5),
        "anomalo_2sigma":    anomalo,
        "neff_post":         round(neff, 1),
        "rho_post":          round(rho, 3),
        "neff_flag":         flag,
    }


# =============================================================================
# LIVELLO 2 — esplorativa (H₀_locale: salto pre→post)
# =============================================================================

def run_exploratory(pre: np.ndarray, post: np.ndarray):
    """
    Esplorativa — H₀_locale: μ_post = μ_pre (salto locale pre→post).
    DIVERSA da H₀ confirmatory. NON deve entrare nella BH globale.

    Restituisce block permutation, HAC Andrews, n_eff, ρ̂.
    """
    if len(pre) < 4 or len(post) < 4:
        return None

    combined = np.concatenate([pre, post])
    neff_combined, rho_combined = compute_neff(combined)
    maxlags  = andrews_bw_ar1(rho_combined, len(combined))
    flag_combined = neff_flag(neff_combined)

    # Avviso se il bandwidth supera n//2 (test non informativo)
    bw_exceeds_n = maxlags >= len(combined) // 2
    hac_note = ("CAUTELA: HAC bandwidth ≥ n/2, test non informativo"
                if bw_exceeds_n else "ok")

    # Block permutation
    rng = np.random.default_rng(SEED)
    n_po = len(post)
    obs_perm = float(np.median(post) - np.median(pre))

    def _block_perm(arr, rng):
        n = len(arr)
        n_bl = int(np.ceil(n / BLOCK_SIZE))
        blocks = [arr[i*BLOCK_SIZE:min((i+1)*BLOCK_SIZE, n)] for i in range(n_bl)]
        rng.shuffle(blocks)
        perm = np.concatenate(blocks)
        return float(np.median(perm[-n_po:]) - np.median(perm[:-n_po]))

    nulls = [_block_perm(combined, rng) for _ in range(N_PERM)]
    perm_p = float(np.mean(np.array(nulls) >= obs_perm))

    # HAC (Newey-West, Andrews bandwidth)
    y = combined
    d = np.concatenate([np.zeros(len(pre)), np.ones(len(post))])
    X = sm.add_constant(d)
    try:
        res = sm.OLS(y, X).fit(
            cov_type="HAC",
            cov_kwds={"maxlags": maxlags, "use_correction": True}
        )
        hac_delta = float(res.params[1])
        hac_p     = float(res.pvalues[1])
    except Exception:
        hac_delta, hac_p = np.nan, np.nan

    # Bootstrap CI sul salto locale
    rng2  = np.random.default_rng(SEED + 1)
    deltas = [rng2.choice(post, len(post), replace=True).mean() -
              rng2.choice(pre,  len(pre),  replace=True).mean()
              for _ in range(2000)]
    arr = np.array(deltas)
    boot_delta = float(arr.mean())
    boot_ci_lo = float(np.percentile(arr, 2.5))
    boot_ci_hi = float(np.percentile(arr, 97.5))

    return {
        # Salto locale
        "delta_local":  round(float(post.mean() - pre.mean()), 5),
        "boot_delta":   round(boot_delta, 5),
        "boot_CI_lo":   round(boot_ci_lo, 5),
        "boot_CI_hi":   round(boot_ci_hi, 5),
        # Block permutation (H₀_locale)
        "perm_delta":   round(obs_perm, 5),
        "perm_p":       round(perm_p, 4),
        # HAC Andrews (H₀_locale)
        "hac_delta":    round(hac_delta, 5) if not np.isnan(hac_delta) else np.nan,
        "hac_p":        round(hac_p, 4)     if not np.isnan(hac_p)     else np.nan,
        "hac_maxlags_used": maxlags,
        "hac_note":     hac_note,
        # n_eff diagnostica
        "neff_combined":    round(neff_combined, 1),
        "rho_combined":     round(rho_combined, 3),
        "neff_flag_combined": flag_combined,
    }


# =============================================================================
# τ_margin — SOLO DESCRITTIVO (Bai-Perron)
# =============================================================================

def compute_tau_margin_descriptive(margin_series: pd.Series,
                                   tau_p_date,
                                   mu_2019: float,
                                   min_frac: float = 0.25):
    """
    Stima τ_margin con Bai-Perron (argmax |t_Welch|).
    ATTENZIONE: endogeno alla serie del margine → NON usare il suo p-value
    per inferenza. Restituisce solo il timing e le statistiche descrittive
    delle due finestre (prima/dopo τ_margin).

    Il campo 'tau_margin_t_stat' è riportato come indicatore della
    nitidezza della rottura, NON come statistica di test.
    """
    n = len(margin_series)
    if n < 8:
        return None
    vals = margin_series.values
    idx  = margin_series.index
    min_n = max(4, int(n * min_frac))

    best_t, best_k = 0.0, n // 2
    for k in range(min_n, n - min_n):
        pre_k, post_k = vals[:k], vals[k:]
        if len(pre_k) < 4 or len(post_k) < 4:
            continue
        t_k, _ = stats.ttest_ind(pre_k, post_k, equal_var=False)
        if abs(t_k) > abs(best_t):
            best_t, best_k = t_k, k

    tau_m_date = idx[best_k]
    pre_vals   = vals[:best_k]
    post_vals  = vals[best_k:]

    lag_vs_price = int((tau_m_date - tau_p_date).days) if tau_p_date is not None else None
    if lag_vs_price is not None:
        if lag_vs_price < -7:
            timing_interp = f"ANTICIPATORIO ({lag_vs_price:+d}gg): τ_margin precede τ_price"
        elif lag_vs_price <= 14:
            timing_interp = f"SINCRONO ({lag_vs_price:+d}gg)"
        else:
            timing_interp = f"REATTIVO ({lag_vs_price:+d}gg): τ_margin segue τ_price"
    else:
        timing_interp = "N/A (τ_price non disponibile)"

    return {
        "tau_margin":                str(tau_m_date.date()),
        "tau_margin_t_stat_bplike":  round(float(best_t), 3),
        "tau_margin_NOTE":           "DESCRITTIVO — endogeno al margine, non usare per inferenza",
        "lag_tau_margin_vs_price_gg": lag_vs_price,
        "tau_lag_interpretation":    timing_interp,
        "delta_before_tau_margin":   round(float(pre_vals.mean() - mu_2019), 5),
        "delta_after_tau_margin":    round(float(post_vals.mean() - mu_2019), 5),
        "n_before":                  len(pre_vals),
        "n_after":                   len(post_vals),
    }


# =============================================================================
# CLASSIFICAZIONE (basata solo su Lv.1 confirmatory)
# =============================================================================

def classify(delta_vs_bl: float, soglia: float,
             bh_reject: bool, neff: float) -> str:
    """
    Classificazione finale basata SOLO sulla famiglia confirmatory (BH_reject)
    e sull'anomalia descrittiva 2σ.
    n_eff < NEFF_WARN → "Inconclusivo (n_eff insufficiente)" anche se BH_reject.
    """
    anomalo = delta_vs_bl > soglia
    if neff < NEFF_WARN:
        return "Inconclusivo (n_eff insufficiente)"
    if bh_reject and anomalo:
        return "Margine anomalo — confermato"
    if bh_reject and not anomalo:
        return "Variazione significativa, non anomala"
    if not bh_reject and anomalo:
        return "Anomalia descrittiva, non confermata"
    return "Neutro / trasmissione attesa"


# =============================================================================
# CARICA DATI
# =============================================================================

print("=" * 72)
print("Script 03 v2 — Test H0 margine lordo (architettura epistemica separata)")
print("=" * 72)

diag_path = "data/regression_diagnostics.csv"
if os.path.exists(diag_path):
    df_diag = pd.read_csv(diag_path)
    print("\nDiagnostici OLS (da script 02):")
    for _, r in df_diag.iterrows():
        print(f"  {str(r.get('Evento','?'))[:28]:28} | {str(r.get('Serie','?')):7}: "
              f"DW={r.get('DW', 'N/A'):.2f}  SW_p={r.get('SW_p', 'N/A')}")

tau_price_map = {}
t1_path = "data/table1_changepoints.csv"
if os.path.exists(t1_path):
    df_t1 = pd.read_csv(t1_path)
    for _, r in df_t1.iterrows():
        if str(r["Serie"]) in ("Benzina", "Diesel"):
            # Flag Rhat per τ_price inaffidabili
            rhat = r.get("rhat_max", 1.0)
            tau_price_map[(r["Evento"], r["Serie"])] = {
                "date": pd.Timestamp(str(r["tau"])),
                "rhat_flag": str(r.get("rhat_flag", "ok")),
                "reliable": float(rhat) <= 1.05 if not pd.isna(rhat) else False,
            }
    print(f"\nτ_price da table1: {len(tau_price_map)} entry")
    for k, v in tau_price_map.items():
        if not v["reliable"]:
            print(f"  ⚠ τ_price per {k}: Rhat={v['rhat_flag']} — non affidabile, escluso")

merged = pd.read_csv("data/dataset_merged_with_futures.csv",
                     index_col=0, parse_dates=True)
print(f"\nDataset: {len(merged)} settimane | "
      f"{merged.index[0].date()} – {merged.index[-1].date()}")

# Safety check unità
for col in ["benzina_eur_l", "diesel_eur_l"]:
    if col in merged.columns and merged[col].dropna().median() > 10:
        merged[col] = merged[col] / 1000.0

# Baseline 2019
baseline = merged.loc[BASELINE_START:BASELINE_END]
thresholds   = {}
baseline_mu  = {}
baseline_vals = {}
for fuel, col in MARGIN_COLS.items():
    if col in baseline.columns:
        v = baseline[col].dropna()
        thresholds[fuel]    = float(2 * v.std()) if len(v) >= 4 else 0.030
        baseline_mu[fuel]   = float(v.mean())    if len(v) >= 4 else 0.0
        baseline_vals[fuel] = v.values
        print(f"Baseline 2019 | {fuel}: μ={baseline_mu[fuel]:.5f}  "
              f"2σ={thresholds[fuel]:.5f} EUR/L")

# Sensitivity baseline
sens_rows = []
for bl_label, bl_start, bl_end in [("2019_full", "2019-01-01", "2019-12-31"),
                                     ("2021_full", "2021-01-01", "2021-12-31")]:
    for fuel, col in MARGIN_COLS.items():
        if col not in merged.columns:
            continue
        b = merged.loc[bl_start:bl_end, col].dropna()
        if len(b) >= 4:
            sens_rows.append({
                "baseline": bl_label, "serie": col,
                "n_weeks": len(b), "mean": round(float(b.mean()), 5),
                "std": round(float(b.std()), 5),
                "soglia_2sigma": round(float(2*b.std()), 5),
            })
if sens_rows:
    pd.DataFrame(sens_rows).to_csv("data/baseline_sensitivity.csv", index=False)


# =============================================================================
# CICLO PRINCIPALE
# =============================================================================

print("\n" + "=" * 72)
print("LIVELLO 1 — CONFIRMATORY (H₀: μ_post = μ_2019, split esogeni)")
print("LIVELLO 2 — EXPLORATORY  (H₀_locale: salto pre→post, block perm + HAC)")
print("LIVELLO 2b — τ_margin DESCRITTIVO (timing only, NON in BH)")
print("=" * 72)

rng_perm      = np.random.default_rng(SEED)
conf_rows     = []    # → confirmatory_pvalues_v2.csv
explo_rows    = []    # → exploratory_results.csv
neff_rows     = []    # → neff_report.csv
all_rows      = []    # → all_test_results_v2.csv
tau_margin_store = {} # (ev, fuel) → dict descrittivo
annual_rows   = []

for ev_name, cfg in EVENTS.items():
    shock  = cfg["shock"]
    prelim = cfg.get("preliminare", False)
    prelim_lbl = " [PREL.]" if prelim else ""

    print(f"\n{'─'*72}")
    print(f"  {ev_name}{prelim_lbl}  |  shock = {shock.date()}")
    print(f"{'─'*72}")

    for fuel, margin_col in MARGIN_COLS.items():
        if margin_col not in merged.columns:
            print(f"  {fuel}: {margin_col} non trovata — skip")
            continue

        df_ev = merged.loc[cfg["pre_start"]:cfg["post_end"]].dropna(subset=[margin_col])
        if len(df_ev) < 8:
            print(f"  {fuel}: meno di 8 osservazioni — skip")
            continue

        margin_series = df_ev[margin_col]
        mu_2019       = baseline_mu.get(fuel, 0.0)
        soglia        = thresholds.get(fuel, 0.030)
        bl_vals       = baseline_vals.get(fuel, np.array([]))

        # τ_price (esogeno al margine, da MCMC su log-prezzo)
        tau_p_info = tau_price_map.get((ev_name, fuel))
        tau_p_date = (tau_p_info["date"]
                      if tau_p_info and tau_p_info["reliable"]
                      and cfg["pre_start"] < tau_p_info["date"] < cfg["post_end"]
                      else None)
        tau_p_reliable = tau_p_info["reliable"] if tau_p_info else False

        # τ_margin (endogeno, solo descrittivo)
        tau_m_desc = compute_tau_margin_descriptive(
            margin_series, tau_p_date, mu_2019
        )
        if tau_m_desc:
            tau_margin_store[(ev_name, fuel)] = tau_m_desc
            print(f"\n  {fuel}  |  τ_margin [DESCRITTIVO]: "
                  f"{tau_m_desc['tau_margin']}  "
                  f"{tau_m_desc['tau_lag_interpretation']}")

        # ── SPLITS da testare: esogeni soltanto ──────────────────────────
        EXOG_SPLITS = {"shock_hard": shock}
        if tau_p_date is not None:
            EXOG_SPLITS["tau_price"] = tau_p_date

        print(f"\n  {fuel}  μ_2019={mu_2019:+.5f}  soglia 2σ={soglia:.5f}")

        for split_type, split_date in EXOG_SPLITS.items():
            # Trova indice split nella serie evento
            idx = int(np.clip(
                margin_series.index.searchsorted(split_date),
                2, len(margin_series) - 2
            ))
            pre  = margin_series.iloc[:idx].values
            post = margin_series.iloc[idx:].values

            if len(pre) < 4 or len(post) < 4:
                print(f"    [{split_type}] campione insufficiente — skip")
                continue

            # ── LIVELLO 1: confirmatory ───────────────────────────────────
            conf = run_confirmatory(post, bl_vals, mu_2019, soglia)
            if conf is None:
                continue

            # ── LIVELLO 2: esplorativa ────────────────────────────────────
            explo = run_exploratory(pre, post)

            # Pre vs baseline (era già anomalo?)
            _, t_p_pre = stats.ttest_1samp(pre, popmean=mu_2019, alternative="greater")
            delta_pre_bl = float(pre.mean() - mu_2019)
            pre_anomalo  = delta_pre_bl > soglia
            neff_pre, rho_pre = compute_neff(pre)

            lag_vs_shock = int((split_date - shock).days)

            # Stampa riassunto
            hac_p_str = (f"{explo['hac_p']:.4f}" if explo and not np.isnan(explo["hac_p"])
                         else "N/A")
            print(f"    [{split_type:12}] split={split_date.date()}"
                  f"  δ_bl={conf['delta_vs_baseline']:+.4f}"
                  f"  t_p={conf['t_p']:.4f} {_stars(conf['t_p'])}"
                  f"  mw_p={conf['mw_p']:.4f}"
                  f"  perm_p={explo['perm_p'] if explo else 'N/A':.4f}"
                  f"  HAC_p={hac_p_str}"
                  f"  n_eff={conf['neff_post']:.1f}"
                  f"  {'ANOM' if conf['anomalo_2sigma'] else '    '}")
            if conf["neff_flag"] != "ok":
                print(f"      ⚠  {conf['neff_flag']}")

            row_base = {
                "livello_epistemico": "CONFIRMATORY",
                "evento":             ev_name,
                "carburante":         fuel,
                "split_type":         split_type,
                "split_label":        SPLIT_LABELS[split_type],
                "preliminare":        prelim,
                "shock_date":         str(shock.date()),
                "split_date":         str(split_date.date()),
                "lag_split_vs_shock_gg": lag_vs_shock,
                "tau_price":          str(tau_p_date.date()) if tau_p_date else "N/A",
                "tau_price_reliable": tau_p_reliable,
                "tau_margin":         tau_m_desc["tau_margin"] if tau_m_desc else "N/A",
                "tau_margin_NOTE":    "[DESCRITTIVO — non in BH]",
                "mu_baseline_2019":   round(mu_2019, 5),
                "soglia_2sigma":      round(soglia, 5),
                "n_pre":              len(pre),
                "n_post":             len(post),
                # Confirmatory (H₀: livello vs baseline)
                "t_stat":             conf["t_stat"],
                "t_p":                conf["t_p"],
                "mw_p":               conf["mw_p"],
                "delta_vs_baseline":  conf["delta_vs_baseline"],
                "anomalo_2sigma":     conf["anomalo_2sigma"],
                "neff_post":          conf["neff_post"],
                "rho_post":           conf["rho_post"],
                "neff_flag":          conf["neff_flag"],
                # Pre-split
                "t_p_pre":            round(float(t_p_pre), 4),
                "delta_pre_vs_bl":    round(delta_pre_bl, 5),
                "pre_anomalo_2sigma": pre_anomalo,
                "neff_pre":           round(neff_pre, 1),
                # BH (riempito dopo il loop)
                "BH_reject":          False,
                "t_p_BH_adj":         np.nan,
                "mw_p_BH_adj":        np.nan,
                "classificazione":    "",
            }

            # Esplorativi aggiunti come colonne
            if explo:
                row_base.update({
                    "livello_esplorativo": "block_perm+HAC  [H0_locale — NON in BH]",
                    "delta_local":         explo["delta_local"],
                    "boot_delta":          explo["boot_delta"],
                    "boot_CI_lo":          explo["boot_CI_lo"],
                    "boot_CI_hi":          explo["boot_CI_hi"],
                    "perm_p":              explo["perm_p"],
                    "hac_delta":           explo["hac_delta"],
                    "hac_p":               explo["hac_p"],
                    "hac_maxlags_used":    explo["hac_maxlags_used"],
                    "hac_note":            explo["hac_note"],
                    "neff_combined":       explo["neff_combined"],
                    "rho_combined":        explo["rho_combined"],
                    "neff_flag_combined":  explo["neff_flag_combined"],
                })

            all_rows.append(row_base)

            # n_eff report
            neff_rows.append({
                "evento": ev_name, "carburante": fuel, "split_type": split_type,
                "neff_post": conf["neff_post"], "rho_post": conf["rho_post"],
                "neff_pre": round(neff_pre, 1), "rho_pre": round(rho_pre, 3),
                "neff_combined": explo["neff_combined"] if explo else np.nan,
                "rho_combined":  explo["rho_combined"]  if explo else np.nan,
                "hac_maxlags_andrews": explo["hac_maxlags_used"] if explo else np.nan,
                "neff_flag": conf["neff_flag"],
            })

            # Confirmatory pvalues — HAC_t (primario, long-run variance) + MW
            # [v2.1] Welch_t sostituito da HAC_t per gestire autocorrelazione.
            # conf["t_p"] è ora il p-value del test HAC one-sample (Andrews BW).
            # conf["t_p_welch_iid"] conservato come riferimento (non in BH).
            if not prelim:
                for test_name, p_val in [("HAC_t", conf["t_p"]),
                                          ("MannWhitney", conf["mw_p"])]:
                    conf_rows.append({
                        "fonte":       f"{test_name}_{ev_name}_{fuel}_{split_type}",
                        "tipo":        "confirmatory",
                        "test":        test_name,
                        "descrizione": f"{ev_name} | {fuel} | split={split_type}",
                        "p_value":     p_val,
                        "split_type":  split_type,
                        "evento":      ev_name,
                        "carburante":  fuel,
                    })

            # Esplorativa separata (block perm + HAC — H₀_locale)
            if explo and not prelim:
                for test_name, p_val in [("BlockPerm", explo["perm_p"]),
                                          ("HAC_Andrews", explo["hac_p"])]:
                    if p_val is not None and not np.isnan(float(p_val)):
                        explo_rows.append({
                            "fonte":         f"{test_name}_{ev_name}_{fuel}_{split_type}",
                            "tipo":          "exploratory_locale",
                            "test":          test_name,
                            "H0":            "H0_locale: delta_post = delta_pre",
                            "descrizione":   f"{ev_name} | {fuel} | split={split_type}",
                            "p_value":       float(p_val),
                            "split_type":    split_type,
                            "evento":        ev_name,
                            "carburante":    fuel,
                            "hac_maxlags":   explo["hac_maxlags_used"]
                                             if test_name == "HAC_Andrews" else np.nan,
                            "neff_combined": explo["neff_combined"],
                            "neff_flag":     explo["neff_flag_combined"],
                            "NOTE":          "[NON in BH globale — H0 diversa da confirmatory]",
                        })


# =============================================================================
# BENJAMINI-HOCHBERG FDR — SOLO sulla famiglia confirmatory (Livello 1)
# =============================================================================

print("\n" + "=" * 72)
print("BH CORRECTION — FDR 5% su famiglia confirmatory pulita")
print(f"  Famiglia: solo Welch 1-sample + MW, split esogeni (shock_hard + τ_price)")
print(f"  Dimensione attesa: 2 eventi × 2 carburanti × ≤2 split × 2 test ≤ 16 test")
print("=" * 72)

df_all = pd.DataFrame(all_rows)
df_conf = pd.DataFrame(conf_rows)

if not df_conf.empty:
    # BH separata per Welch e per MW (stessa H₀, test diversi — BH congiunta è ok)
    p_arr  = df_conf["p_value"].values
    rej, adj = bh_correction(p_arr, alpha=ALPHA)
    df_conf["BH_reject"]   = rej
    df_conf["p_BH_adj"]    = adj

    n_rej  = int(rej.sum())
    n_tot  = len(df_conf)
    n_hac  = (df_conf["test"] == "HAC_t").sum()
    n_mw   = (df_conf["test"] == "MannWhitney").sum()
    print(f"\n  Famiglia: {n_tot} test ({n_hac} HAC_t + {n_mw} MW)"
          f"  |  Rigettati a FDR 5%: {n_rej}")
    print(f"  [v2.1: HAC_t usa long-run variance Andrews BW — non più Welch iid]")

    for st in ["shock_hard", "tau_price"]:
        sub = df_conf[df_conf["split_type"] == st]
        print(f"    {st:12}: {int(sub['BH_reject'].sum())}/{len(sub)} rigettati")

    # Propaga BH in df_all
    bh_lookup = {
        row["fonte"]: (row["BH_reject"], row["p_BH_adj"])
        for _, row in df_conf.iterrows()
    }
    for i, row in df_all.iterrows():
        ev, fuel, st = row["evento"], row["carburante"], row["split_type"]
        # HAC_t (sostituisce Welch_t in v2.1)
        key_w = f"HAC_t_{ev}_{fuel}_{st}"
        if key_w in bh_lookup:
            df_all.at[i, "BH_reject"]   = bh_lookup[key_w][0]
            df_all.at[i, "t_p_BH_adj"]  = bh_lookup[key_w][1]
        # MW
        key_m = f"MannWhitney_{ev}_{fuel}_{st}"
        if key_m in bh_lookup:
            df_all.at[i, "mw_p_BH_adj"] = bh_lookup[key_m][1]

    # Classificazione finale
    def _classify_row(r):
        neff_v = float(r.get("neff_post", NEFF_WARN + 1))
        return classify(
            float(r["delta_vs_baseline"]),
            float(r["soglia_2sigma"]),
            bool(r["BH_reject"]),
            neff_v,
        )
    df_all["classificazione"] = df_all.apply(_classify_row, axis=1)

    # Stampa classificazioni finali
    print(f"\n  Classificazioni finali (Livello 1 — confirmatory + BH):")
    for _, r in df_all[~df_all["preliminare"]].sort_values(
            ["evento", "carburante", "split_type"]).iterrows():
        bh = "BH✓" if r["BH_reject"] else "   "
        nf = "⚠" if r["neff_flag"] != "ok" else " "
        print(f"  {bh} {nf}  {r['evento'][:26]:26} | {r['carburante']:7} | "
              f"{r['split_type']:12}: "
              f"δ={r['delta_vs_baseline']:+.4f}  "
              f"t_p={r['t_p']:.4f} {_stars(r['t_p'])}"
              f"  n_eff={r['neff_post']:.1f}"
              f"  → {r['classificazione']}")


# =============================================================================
# ANALISI ANNUALE
# =============================================================================

print("\n" + "=" * 72)
print("ANALISI ANNUALE (confronto ogni anno vs baseline 2019)")
print("=" * 72)

for fuel, margin_col in MARGIN_COLS.items():
    if margin_col not in merged.columns:
        continue
    mu_2019  = baseline_mu.get(fuel, 0.0)
    soglia   = thresholds.get(fuel, 0.030)
    vol_week = _VOL_L_WEEK.get(fuel, 0)
    bl_vals  = baseline_vals.get(fuel, np.array([]))

    for yr in sorted(merged.index.year.unique()):
        yr_data = merged.loc[str(yr), margin_col].dropna()
        if len(yr_data) < 4:
            continue
        yr_vals = yr_data.values
        _, mw_p = mannwhitneyu(yr_vals, bl_vals, alternative="greater") \
                  if len(bl_vals) > 0 else (0, 1.0)
        delta_yr = float(yr_vals.mean()) - mu_2019
        anomalo  = delta_yr > soglia
        wf_net   = float((yr_data - mu_2019).sum()) * vol_week / 1e6
        neff_yr, rho_yr = compute_neff(yr_vals)

        annual_rows.append({
            "anno": yr, "carburante": fuel, "n_settimane": len(yr_vals),
            "media_eur_l": round(float(yr_vals.mean()), 5),
            "delta_vs_2019": round(delta_yr, 5),
            "anomalo_2sigma": anomalo,
            "mw_p_vs_2019": round(float(mw_p), 4),
            "windfall_net_meur": round(wf_net, 1),
            "neff_yr": round(neff_yr, 1),
            "rho_yr":  round(rho_yr, 3),
            "NOTE_windfall": "volumi proxy fissi 2022 — sovrastimato per 2025+",
        })


# =============================================================================
# SALVA OUTPUT
# =============================================================================

print("\n" + "=" * 72)
print("SALVA OUTPUT")
print("=" * 72)

# 1. MASTER CSV
if not df_all.empty:
    df_all.to_csv("data/all_test_results_v2.csv", index=False)
    print(f"  ✓ data/all_test_results_v2.csv  ({len(df_all)} righe)")

# 2. Confirmatory pvalues (puliti, 16 righe)
if not df_conf.empty:
    df_conf.to_csv("data/confirmatory_pvalues_v2.csv", index=False)
    print(f"  ✓ data/confirmatory_pvalues_v2.csv  ({len(df_conf)} test — "
          f"dimensione famiglia: {len(df_conf)})")

# 3. Exploratory results (H₀_locale, NON in BH)
if explo_rows:
    df_explo = pd.DataFrame(explo_rows)
    df_explo.to_csv("data/exploratory_results.csv", index=False)
    print(f"  ✓ data/exploratory_results.csv  ({len(df_explo)} test — H₀_locale, non in BH)")

# 4. n_eff report
if neff_rows:
    df_neff = pd.DataFrame(neff_rows)
    df_neff.to_csv("data/neff_report.csv", index=False)
    print(f"  ✓ data/neff_report.csv")

# 5. table2_margin_anomaly_v2 (solo shock_hard per compatibilità)
df_sh = df_all[df_all["split_type"] == "shock_hard"].copy()
df_sh.rename(columns={
    "evento": "Evento", "carburante": "Carburante",
    "delta_vs_baseline": "delta_margine_eur",
}, inplace=True)
df_sh.to_csv("data/table2_margin_anomaly_v2.csv", index=False)
print(f"  ✓ data/table2_margin_anomaly_v2.csv  ({len(df_sh)} righe, split shock_hard)")

# 6. table2_multi_split
df_all.to_csv("data/table2_multi_split_v2.csv", index=False)
print(f"  ✓ data/table2_multi_split_v2.csv  ({len(df_all)} righe)")

# 7. Annual
if annual_rows:
    pd.DataFrame(annual_rows).to_csv("data/annual_margin_analysis.csv", index=False)
    print(f"  ✓ data/annual_margin_analysis.csv")


# =============================================================================
# FIGURE
# =============================================================================

def _war_lines(ax, s_max):
    for lbl, (dt, c) in WAR_DATES.items():
        ax.axvline(dt, color=c, lw=1.8, ls="--", alpha=0.85)
        ax.text(dt + pd.Timedelta(days=5), s_max * 0.97,
                lbl, rotation=90, fontsize=8, color=c, va="top")

# ── Fig 1: Margini nel tempo ────────────────────────────────────────────────
if not df_all.empty:
    fig_m, axes_m = plt.subplots(2, 1, figsize=(15, 9), sharex=True)
    EV_COLOR = {
        "Ucraina (Feb 2022)":       "#e74c3c",
        "Iran-Israele (Giu 2025)":  "#e67e22",
        "Hormuz (Feb 2026)":        "#8e44ad",
    }

    for ax, (fuel, col), fc in zip(
            axes_m,
            [("Benzina","margine_benz_crack"),("Diesel","margine_dies_crack")],
            ["#e67e22","#8e44ad"]):
        if col not in merged.columns:
            continue
        s  = merged[col].dropna()
        bl = baseline.get(col, pd.Series(dtype=float)).dropna()
        ax.plot(s.index, s.values, color=fc, lw=1.8, alpha=0.85, label=fuel)
        if len(bl) >= 4:
            ax.axhspan(bl.mean()-2*bl.std(), bl.mean()+2*bl.std(),
                       alpha=0.11, color="#888", label="Baseline ±2σ (2019)")
            ax.axhline(bl.mean(), color="#888", lw=1.0, ls="--")

        _war_lines(ax, s.max())

        for ev_full, cfg in EVENTS.items():
            ec = EV_COLOR.get(ev_full, "#555")
            tp_info = tau_price_map.get((ev_full, fuel))
            tm_desc = tau_margin_store.get((ev_full, fuel))

            if (tp_info and tp_info["reliable"] and
                    s.index[0] <= tp_info["date"] <= s.index[-1]):
                ax.axvline(tp_info["date"], color=ec, lw=1.4, ls="-.", alpha=0.8)

            if tm_desc:
                tm_ts = pd.Timestamp(tm_desc["tau_margin"])
                if s.index[0] <= tm_ts <= s.index[-1]:
                    ax.axvline(tm_ts, color=ec, lw=1.2, ls=":", alpha=0.6)
                    ax.text(tm_ts + pd.Timedelta(days=3), s.quantile(0.82),
                            f"τm\n[desc]", rotation=90, fontsize=6,
                            color=ec, va="top", alpha=0.7)

        ax.set_ylabel("Margine lordo (EUR/litro)", fontsize=10)
        ax.set_title(f"Crack spread — {fuel}  (─── shock  ·─· τ_price  ··· τ_margin[desc])",
                     fontsize=10)
        ax.legend(handles=[
            mpatches.Patch(color=fc, label=fuel),
            mpatches.Patch(color="#888", alpha=0.4, label="Baseline ±2σ 2019"),
            Line2D([0],[0], color="#555", lw=1.4, ls="-.", label="τ_price (MCMC, esogeno)"),
            Line2D([0],[0], color="#555", lw=1.2, ls=":", alpha=0.6,
                   label="τ_margin [DESCRITTIVO — non in BH]"),
        ], fontsize=8, loc="upper left")
        ax.grid(alpha=0.25)

    axes_m[-1].xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    axes_m[-1].xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    plt.xticks(rotation=40, fontsize=9)
    fig_m.suptitle(
        "Margine lordo crack spread — split esogeni nella famiglia confirmatory\n"
        "H₁: μ_post > μ₂₀₁₉  |  τ_margin in grigio = DESCRITTIVO, non testato",
        fontsize=11, fontweight="bold",
    )
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    fig_m.savefig("plots/03_margins_v2.png", dpi=DPI, bbox_inches="tight")
    plt.close(fig_m)
    print("\n  ✓ plots/03_margins_v2.png")

# ── Fig 2: n_eff report ─────────────────────────────────────────────────────
if neff_rows:
    df_neff_p = pd.DataFrame(neff_rows)
    labels_n  = [f"{r['evento'].split('(')[0].strip()} | {r['carburante']} | {r['split_type']}"
                 for _, r in df_neff_p.iterrows()]
    neff_vals = df_neff_p["neff_post"].values
    rho_vals  = df_neff_p["rho_post"].values

    fig_n, (ax_n, ax_r) = plt.subplots(1, 2, figsize=(14, max(4, len(labels_n)*0.55+2)))

    colors_n = ["#c0392b" if v < NEFF_WARN else "#e67e22" if v < 10 else "#27ae60"
                for v in neff_vals]
    ax_n.barh(range(len(labels_n)), neff_vals, color=colors_n, alpha=0.85,
              edgecolor="black", lw=0.6)
    ax_n.axvline(NEFF_WARN, color="#c0392b", lw=2.0, ls="--",
                 label=f"Soglia CAUTELA n_eff={NEFF_WARN}")
    ax_n.axvline(10, color="#e67e22", lw=1.5, ls=":",
                 label="Soglia ATTENZIONE n_eff=10")
    ax_n.set_yticks(range(len(labels_n)))
    ax_n.set_yticklabels(labels_n, fontsize=8)
    ax_n.set_xlabel("n_eff (obs indipendenti effettive nella finestra post)", fontsize=9)
    ax_n.set_title("Effective sample size — potenza dei test\n"
                   "Rosso=test non informativo | Arancio=cautela", fontsize=9, fontweight="bold")
    ax_n.legend(fontsize=8)
    ax_n.grid(alpha=0.25, axis="x")

    colors_r = ["#c0392b" if v > 0.7 else "#e67e22" if v > 0.5 else "#27ae60"
                for v in rho_vals]
    ax_r.barh(range(len(labels_n)), rho_vals, color=colors_r, alpha=0.85,
              edgecolor="black", lw=0.6)
    ax_r.axvline(0.85, color="#c0392b", lw=2.0, ls="--",
                 label="ρ̂=0.85 (n_eff≈2 per n=27)")
    ax_r.set_yticks(range(len(labels_n)))
    ax_r.set_yticklabels([], fontsize=8)
    ax_r.set_xlabel("ρ̂ AR(1) stimato", fontsize=9)
    ax_r.set_title("Autocorrelazione AR(1) stimata\nRosso=ρ>0.7 → HAC molto conservativo",
                   fontsize=9, fontweight="bold")
    ax_r.legend(fontsize=8)
    ax_r.grid(alpha=0.25, axis="x")

    fig_n.suptitle("Diagnostica autocorrelazione e potenza dei test\n"
                   "n_eff = n · (1−ρ̂) / (1+ρ̂)  —  HAC bandwidth: Andrews (1991) plug-in",
                   fontsize=10, fontweight="bold")
    plt.tight_layout(pad=1.5)
    fig_n.savefig("plots/03_neff_report.png", dpi=DPI, bbox_inches="tight")
    plt.close(fig_n)
    print("  ✓ plots/03_neff_report.png")

print("""
═══════════════════════════════════════════════════════════════════════════════
SOMMARIO — Script 03 v2 completato

  FAMIGLIA BH CONFIRMATORY (questo script → 05_global_corrections_v2.py):
    • H₀: μ_post = μ_2019  (one-sided upper)
    • Test: Welch 1-sample + Mann-Whitney (entrambi vs distribuzione 2019)
    • Split: shock_hard + τ_price (esogeni — da MCMC su log-PREZZO, non margine)
    • Dimensione: ≤16 test (2 eventi × 2 carburanti × 2 split × 2 test)
    • File: data/confirmatory_pvalues_v2.csv

  ESPLORATIVA (NON in BH):
    • Block permutation + HAC Andrews  →  H₀_locale: salto pre→post
    • τ_margin: solo timing (Δ giorni vs τ_price), nessun p-value
    • File: data/exploratory_results.csv

  DIAGNOSTICA:
    • n_eff e ρ̂ per ogni combinazione evento×carburante×split
    • File: data/neff_report.csv
    • Flag automatico se n_eff < 5 ("CAUTELA: test non informativo")
═══════════════════════════════════════════════════════════════════════════════
""")