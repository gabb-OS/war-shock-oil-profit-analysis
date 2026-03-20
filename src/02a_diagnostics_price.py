#!/usr/bin/env python3
"""
02b_diagnostics_margin.py  (v2 — test statisticamente rigorosi + H₀)
======================================================================

DIAGNOSTICHE (finestra per-evento × carburante):
  1. Stazionarietà     — ADF + KPSS dual-test sulla serie nella finestra
                         (nota: potenza limitata su finestre brevi)
  2. Autocorrelazione  — AR(p) con selezione AIC → Ljung-Box sui RESIDUI
                         (non sulla serie grezza: renderebbe il test un non-test)
  3. Normalità         — Jarque-Bera (primario, asintoticamente valido)
                         + D'Agostino K² (secondario, migliore per n moderato)
                         + Shapiro-Wilk ausiliario SOLO se n ≤ 200
                         — applicati ai residui AR, non alle prime differenze
  4. Omoschedasticità  — Fligner-Killeen (primario, robusto a non-normalità)
                         + Levene (secondario, informativo)
                         FLAG: se ARCH rilevato → Levene è inaffidabile
  5. Effetti ARCH      — Engle ARCH-LM applicato ai RESIDUI AR(p)
                         (non alla serie grezza o alle differenze prime)
  Ordine di esecuzione: ARCH e autocorrelazione prima, i loro risultati
  contestualizzano normalità e omoschedasticità.

H₀ FINALE (per evento × carburante + sintesi aggregata):
  ─────────────────────────────────────────────────────────────────────
  Ipotesi Nulla (H₀):
    In prossimità temporale di shock geopolitici che coinvolgono Paesi
    fornitori di petrolio greggio o semilavorati verso l'Italia,
    i distributori italiani di carburante NON generano profitti anomali:
    gli aumenti dei prezzi risultano coerenti con la crescita dei costi
    di approvvigionamento e delle materie prime.

  Test principale: Mann-Whitney U one-sided (H₁: margine_post > margine_pre)
    — non parametrico, non assume normalità, robusto a outlier
    — pre = H0_PRE_DAYS giorni precedenti lo shock
    — post = H0_POST_DAYS giorni successivi allo shock
  Effect size: Cliff's δ  (interpretazione: negligible/small/medium/large)
  Sintesi multidimensionale (Fisher combination):
    • Per evento  — combina p-value di benzina + gasolio
    • Per carburante — combina p-value dei 3 eventi
    • Globale     — combina tutti i (evento × carburante) con dati sufficienti
  ─────────────────────────────────────────────────────────────────────

Output in data/plots/diagnostics/margin/:
  00_margine_serie_storica.png
  0N_test__evento__fuel.png   (grafici diagnostici, uno per test × evento × fuel)
  h0_verdict_matrix.png       — matrice evento × carburante con verdetto H₀
  h0_distributions.png        — violin plot margini pre/post per ogni cella
  risultati_riepilogo.csv
  risultati_stazionarieta.csv
  risultati_autocorrelazione.csv
  risultati_normalita.csv
  risultati_omoschedasticita.csv
  risultati_arch.csv
  h0_per_cella.csv
  h0_sintesi.csv
"""

from __future__ import annotations

import warnings
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import chi2 as chi2_dist
from statsmodels.stats.diagnostic import acorr_ljungbox, het_arch
from statsmodels.tsa.ar_model import AutoReg
from statsmodels.tsa.stattools import adfuller, kpss

import sys
sys.path.insert(0, str(Path(__file__).parent / "utils"))
from conversions import GAS_OIL, EUROBOB as EUROBOB_HC, load_eurusd, usd_ton_to_eur_liter

# ── Configurazione ────────────────────────────────────────────────────────────
BASE_DIR    = Path(__file__).parent
DAILY_CSV   = BASE_DIR / "data" / "processed" / "daily_fuel_prices_all.csv"
GASOIL_CSV  = BASE_DIR / "data" / "Futures" / "London Gas Oil Futures Historical Data.csv"
EUROBOB_CSV = BASE_DIR / "data" / "Futures" / "Eurobob_B7H1_date.csv"
EURUSD_CSV  = BASE_DIR / "data" / "raw" / "eurusd.csv"
OUT_DIR     = BASE_DIR / "data" / "plots" / "diagnostics" / "margin"
OUT_DIR.mkdir(parents=True, exist_ok=True)

ALPHA        = 0.05
HALF_WIN     = 45          # Alzato da 30 a 45: n_eff ≈ 24 con φ=0.3 (più stabile)
LB_LAGS      = [5, 10, 20]
ARCH_LAGS    = 10
AR_MAX_P     = 12          # Ordine massimo AR per selezione AIC
H0_PRE_DAYS  = 60          # Giorni pre-shock per test H₀ (dalla serie completa)
H0_POST_DAYS = 45          # Giorni post-shock per test H₀

FUELS = {
    "benzina": ("#E63946", EUROBOB_HC),
    "gasolio": ("#1D3557", GAS_OIL),
}

EVENTS: dict[str, dict] = {
    "Ucraina (Feb 2022)": {
        "shock":     pd.Timestamp("2022-02-24"),
        "pre_start": pd.Timestamp("2021-11-15"),   # ≈3 mesi pre
        "post_end":  pd.Timestamp("2022-05-15"),   # ≈3 mesi post
        "color":     "#e74c3c",
        "label":     "Russia-Ucraina\n(24 feb 2022)",
    },
    "Iran-Israele (Giu 2025)": {
        "shock":     pd.Timestamp("2025-06-13"),
        "pre_start": pd.Timestamp("2025-03-13"),
        "post_end":  pd.Timestamp("2025-09-13"),
        "color":     "#e67e22",
        "label":     "Iran-Israele\n(13 giu 2025)",
    },
    "Hormuz (Feb 2026)": {
        "shock":     pd.Timestamp("2026-02-28"),
        "pre_start": pd.Timestamp("2025-11-28"),
        "post_end":  pd.Timestamp("2026-05-15"),
        "color":     "#8e44ad",
        "label":     "Stretto di Hormuz\n(28 feb 2026)",
    },
}

_OK   = "#2ecc71"
_WARN = "#f39c12"
_FAIL = "#e74c3c"


# ══════════════════════════════════════════════════════════════════════════════
# Caricamento dati  (identico alla v1)
# ══════════════════════════════════════════════════════════════════════════════

def load_futures_eurl(path: Path, hc, eurusd: pd.Series) -> pd.Series:
    df = pd.read_csv(path, encoding="utf-8-sig", dtype=str)
    df["date"] = pd.to_datetime(df["Date"], format="%m/%d/%Y", errors="coerce")
    df["price_usd_ton"] = (df["Price"].str.replace(",", "", regex=False)
                           .pipe(pd.to_numeric, errors="coerce"))
    df = df.dropna(subset=["date", "price_usd_ton"]).sort_values("date").set_index("date")
    return usd_ton_to_eur_liter(df["price_usd_ton"], eurusd, hc)


def load_futures_b7h1(path: Path, hc, eurusd: pd.Series) -> pd.Series:
    df = pd.read_csv(path, encoding="utf-8-sig", dtype=str)
    if "timestamp" in df.columns:
        ts = pd.to_numeric(df["timestamp"], errors="coerce")
        df["date"] = pd.to_datetime(ts, unit="s", utc=True).dt.tz_localize(None).dt.normalize()
    else:
        _IT = {"gen":"Jan","feb":"Feb","mar":"Mar","apr":"Apr","mag":"May","giu":"Jun",
               "lug":"Jul","ago":"Aug","set":"Sep","ott":"Oct","nov":"Nov","dic":"Dec"}
        def _parse(s: str):
            for it, en in _IT.items(): s = s.replace(it, en)
            return pd.to_datetime(s, dayfirst=True, errors="coerce")
        df["date"] = df["data"].astype(str).apply(_parse)
    df["price_usd_ton"] = pd.to_numeric(df["chiusura"], errors="coerce")
    df = df.dropna(subset=["date","price_usd_ton"]).sort_values("date").set_index("date")
    df = df[~df.index.duplicated(keep="first")]
    n = len(df)
    print(f"  B7H1: {n} righe  ({df.index.min().date()} → {df.index.max().date()})")
    return usd_ton_to_eur_liter(df["price_usd_ton"], eurusd, hc)


def build_margin(daily: pd.DataFrame,
                 gasoil_eurl: pd.Series,
                 eurobob_eurl: pd.Series | None) -> pd.DataFrame:
    df = daily[["benzina_net", "gasolio_net"]].copy()
    ws_gas = gasoil_eurl.reindex(df.index, method="ffill")
    df["margin_gasolio"] = df["gasolio_net"] - ws_gas
    if eurobob_eurl is not None:
        ws_benz = eurobob_eurl.reindex(df.index, method="ffill")
        df["margin_benzina"] = df["benzina_net"] - ws_benz
    else:
        df["margin_benzina"] = np.nan
    return df


# ══════════════════════════════════════════════════════════════════════════════
# Helper: fit AR(p) con selezione AIC
# ══════════════════════════════════════════════════════════════════════════════

def fit_ar_model(series: pd.Series, max_p: int = AR_MAX_P) -> tuple:
    """
    Seleziona l'ordine p di un AR(p) per AIC su `series`.
    Restituisce (residuals: np.ndarray, p_selected: int, aic: float).

    Nota: se il fit fallisce per tutti gli ordini, usa le prime differenze
    come proxy dei residui e segnala p=1 (fallback conservativo).
    """
    s = series.dropna().values
    n = len(s)
    # Limita max_p: almeno 4 obs per parametro
    effective_max = min(max_p, n // 4)
    if effective_max < 1:
        # Serie troppo corta — usa differenze prime come fallback
        return np.diff(s), 1, np.nan, True

    best_aic, best_p, best_resid = np.inf, 1, None
    for p in range(1, effective_max + 1):
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                fit = AutoReg(s, lags=p, old_names=False).fit()
            if fit.aic < best_aic:
                best_aic, best_p, best_resid = fit.aic, p, fit.resid
        except Exception:
            continue

    if best_resid is None:
        return np.diff(s), 1, np.nan, True

    return best_resid, best_p, best_aic, False


# ══════════════════════════════════════════════════════════════════════════════
# Test 1 — Stazionarietà  (ADF + KPSS dual-test)
# ══════════════════════════════════════════════════════════════════════════════

def test_stationarity(series: pd.Series) -> dict:
    """
    ADF (H₀: radice unitaria) + KPSS (H₀: stazionario).
    Due lati: ADF rifiuta UR → stazionario; KPSS non rifiuta → stazionario.
    Potenza limitata su n < 60 — il risultato è orientativo sulla finestra evento.
    """
    s = series.dropna()
    n = len(s)

    adf_stat, adf_p = adfuller(s, autolag="AIC")[:2]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        kpss_stat, kpss_p, *_ = kpss(s, regression="c", nlags="auto")

    adf_ok  = adf_p < ALPHA        # rifiuta H₀-UR → segnale stazionarietà
    kpss_ok = kpss_p > ALPHA       # non rifiuta H₀-staz

    power_note = ""
    if n < 60:
        power_note = f" [ATTENZIONE: n={n}<60, potenza ridotta]"

    if adf_ok and kpss_ok:
        status = "OK"
    elif adf_ok or kpss_ok:
        status = "⚠ WARN"
    else:
        status = "✗ FAIL"

    return {
        "adf_stat": adf_stat, "adf_p": adf_p, "adf_ok": adf_ok,
        "kpss_stat": kpss_stat, "kpss_p": kpss_p, "kpss_ok": kpss_ok,
        "n": n, "power_note": power_note, "status": status,
        "detail": (f"ADF p={adf_p:.3f} {'✓' if adf_ok else '✗'} | "
                   f"KPSS p={kpss_p:.3f} {'✓' if kpss_ok else '✗'}"
                   f"{power_note}"),
    }


# ══════════════════════════════════════════════════════════════════════════════
# Test 2 — Autocorrelazione  (AR(p) → Ljung-Box sui RESIDUI)
# ══════════════════════════════════════════════════════════════════════════════

def test_autocorrelation(series: pd.Series) -> dict:
    """
    Fit AR(p) con AIC, poi Ljung-Box sui RESIDUI.
    Applicarlo alla serie grezza dimostrerebbe solo che i prezzi sono
    autocorrelati (cosa nota a priori) — non è informativo.
    L'LB sui residui testa se la struttura AR scelta è sufficiente.
    """
    s = series.dropna()
    residuals, p_sel, aic, fallback = fit_ar_model(s)

    lb = acorr_ljungbox(residuals, lags=LB_LAGS, return_df=True)
    lb_fail = (lb["lb_pvalue"] < ALPHA).any()
    lb_summary = {int(lag): round(float(row["lb_pvalue"]), 3)
                  for lag, row in lb.iterrows()}

    phi = float(s.autocorr(lag=1))
    phi_c = float(np.clip(phi, -0.999, 0.999))
    n = len(s)
    n_eff = max(1, int(n * (1 - phi_c) / (1 + phi_c)))

    fallback_note = " [FALLBACK: differenze prime]" if fallback else ""
    status = "✗ FAIL" if lb_fail else "OK"

    return {
        "ar_order": p_sel, "aic": aic, "phi_ar1": phi,
        "n": n, "n_eff": n_eff,
        "lb": lb_summary, "lb_fail": lb_fail,
        "residuals": residuals,   # passato ai test normalità/ARCH
        "fallback": fallback, "status": status,
        "detail": (f"AR({p_sel}) AIC={f'{aic:.1f}' if not np.isnan(aic) else 'n.d.'} | "
                   f"φ={phi:.3f}  n_eff={n_eff}/{n} | "
                   f"LB-residui p({LB_LAGS})={list(lb_summary.values())}"
                   f"{fallback_note}"),
    }


# ══════════════════════════════════════════════════════════════════════════════
# Test 3 — Normalità  (JB + D'Agostino K² + SW ausiliario)
# ══════════════════════════════════════════════════════════════════════════════

def test_normality(ar_residuals: np.ndarray) -> dict:
    """
    Test applicati ai RESIDUI AR (non alle differenze prime):
      • Jarque-Bera     — primario: asintoticamente χ²(2), valido per n grande.
                          Basato su skewness + kurtosis.
      • D'Agostino K²   — secondario: migliore potenza per n moderato (30-300).
                          scipy.stats.normaltest.
      • Shapiro-Wilk    — ausiliario SOLO se n ≤ 200: tende a rifiutare
                          sistematicamente per n > 200 anche per deviazioni
                          praticamente trascurabili.
    Nota: la non-normalità non invalida l'ITS se n_eff ≥ 30 (CLT),
    ma informa sulla scelta del metodo (es. v6 GLM Gamma).
    """
    data = np.asarray(ar_residuals, dtype=float)
    data = data[np.isfinite(data)]
    n = len(data)

    jb_stat, jb_p   = stats.jarque_bera(data)
    dag_stat, dag_p = stats.normaltest(data)

    if n <= 200:
        sw_stat, sw_p = stats.shapiro(data)
        sw_note = ""
    else:
        sw_stat, sw_p = np.nan, np.nan
        sw_note = f" [SW disabilitato: n={n}>200, perde controllo dimensione]"

    skewness  = float(stats.skew(data))
    kurt_exc  = float(stats.kurtosis(data))   # eccesso rispetto a normale

    # Verdetto: JB e D'Agostino concordano su H₀?
    jb_ok  = jb_p  > ALPHA
    dag_ok = dag_p > ALPHA
    sw_ok  = (sw_p > ALPHA) if not np.isnan(sw_p) else True  # non voto se n.d.

    if jb_ok and dag_ok:
        status = "OK"
    elif jb_ok or dag_ok:
        status = "⚠ WARN"
    else:
        status = "✗ FAIL"

    return {
        "n_residuals": n,
        "jb_stat": jb_stat, "jb_p": jb_p, "jb_ok": jb_ok,
        "dag_stat": dag_stat, "dag_p": dag_p, "dag_ok": dag_ok,
        "sw_stat": sw_stat, "sw_p": sw_p, "sw_ok": sw_ok,
        "skewness": skewness, "kurt_excess": kurt_exc,
        "sw_note": sw_note, "status": status,
        "detail": (f"JB p={jb_p:.3f} {'✓' if jb_ok else '✗'} | "
                   f"D'Agostino p={dag_p:.3f} {'✓' if dag_ok else '✗'} | "
                   f"skew={skewness:.2f}  kurt_exc={kurt_exc:.2f}"
                   f"{sw_note}"),
    }


# ══════════════════════════════════════════════════════════════════════════════
# Test 4 — Effetti ARCH  (Engle ARCH-LM sui RESIDUI AR)
# ══════════════════════════════════════════════════════════════════════════════

def test_arch(ar_residuals: np.ndarray) -> dict:
    """
    Engle ARCH-LM applicato ai RESIDUI AR (non alla serie grezza).
    Se ARCH è rilevato → Levene in test_homoscedasticity è inaffidabile.
    """
    data = np.asarray(ar_residuals, dtype=float)
    data = data[np.isfinite(data)]
    try:
        lm_stat, lm_p, f_stat, f_p = het_arch(data, nlags=ARCH_LAGS)
        arch_detected = lm_p < ALPHA
        status = "✗ FAIL (volatility clustering)" if arch_detected else "OK"
        detail = f"ARCH-LM p={lm_p:.3f}  F={f_stat:.2f}"
    except Exception as e:
        lm_stat = lm_p = f_stat = f_p = np.nan
        arch_detected = False
        status, detail = "⚠ WARN", f"Test fallito: {e}"
    return {
        "lm_stat": lm_stat, "lm_p": lm_p,
        "f_stat": f_stat, "f_p": f_p,
        "arch_detected": arch_detected,
        "status": status, "detail": detail,
    }


# ══════════════════════════════════════════════════════════════════════════════
# Test 5 — Omoschedasticità  (Fligner-Killeen + Levene, contestualizzati con ARCH)
# ══════════════════════════════════════════════════════════════════════════════

def test_homoscedasticity(series: pd.Series,
                          shock: pd.Timestamp,
                          arch_detected: bool = False) -> dict:
    """
    Confronta varianza pre vs post con due test:
      • Fligner-Killeen — PRIMARIO: non parametrico, robusto a non-normalità
                          e a eteroschedasticità condizionale lieve.
      • Levene          — SECONDARIO: informativo, ma INAFFIDABILE se ARCH
                          è presente (l'ARCH-LM lo segnala).

    Usa la finestra completa (fino a HALF_WIN da ogni lato dello shock).

    Se arch_detected=True: il risultato Levene viene flaggato come unreliable;
    il verdetto primario si basa su Fligner-Killeen.
    """
    pre  = series[series.index < shock].tail(HALF_WIN).dropna()
    post = series[series.index >= shock].head(HALF_WIN).dropna()

    if len(pre) < 5 or len(post) < 5:
        return {
            "status": "⚠ WARN", "detail": f"Campioni insufficienti (pre={len(pre)}, post={len(post)})",
            "fligner_stat": np.nan, "fligner_p": np.nan,
            "lev_stat": np.nan, "lev_p": np.nan,
            "std_pre": np.nan, "std_post": np.nan,
            "ratio": np.nan, "arch_warning": arch_detected,
        }

    fligner_stat, fligner_p = stats.fligner(pre, post)
    lev_stat,     lev_p     = stats.levene(pre, post)
    std_pre, std_post = float(pre.std()), float(post.std())
    ratio = std_post / std_pre if std_pre > 0 else np.nan

    # Verdetto primario su Fligner-Killeen
    fligner_het = fligner_p < ALPHA
    status = "✗ FAIL" if fligner_het else "OK"

    arch_note = ""
    if arch_detected:
        arch_note = " [Levene unreliable: ARCH presente → usa Fligner]"

    return {
        "fligner_stat": fligner_stat, "fligner_p": fligner_p,
        "fligner_het": fligner_het,
        "lev_stat": lev_stat, "lev_p": lev_p,
        "std_pre": std_pre, "std_post": std_post, "ratio": ratio,
        "arch_warning": arch_detected,
        "n_pre": len(pre), "n_post": len(post),
        "status": status,
        "detail": (f"Fligner p={fligner_p:.3f} {'✗HET' if fligner_het else '✓'}  "
                   f"Levene p={lev_p:.3f}  "
                   f"σ_pre={std_pre:.4f}  σ_post={std_post:.4f}  ratio={ratio:.2f}×"
                   f"{arch_note}"),
    }


# ══════════════════════════════════════════════════════════════════════════════
# Orchestratore: esegue tutti i test nell'ordine corretto
# ══════════════════════════════════════════════════════════════════════════════

def run_diagnostics(series: pd.Series, shock: pd.Timestamp, label: str) -> dict:
    """
    Esegue i 5 test nell'ordine di dipendenza corretto:
      1. fit_ar_model (ARCH e autocorrelazione condividono i residui)
      2. test_autocorrelation
      3. test_arch (sui residui AR)
      4. test_normality (sui residui AR)
      5. test_homoscedasticity (ARCH informa il verdetto)
      6. test_stationarity (indipendente, sulla serie grezza)
    """
    s = series.dropna()
    print(f"  [{label.upper()}]  n={len(s)}")

    # Step 1+2: Autocorrelazione → produce AR residuals
    ac_res = test_autocorrelation(s)
    ar_residuals = ac_res["residuals"]

    # Step 3: ARCH sui residui AR
    arch_res = test_arch(ar_residuals)

    # Step 4: Normalità sui residui AR
    norm_res = test_normality(ar_residuals)

    # Step 5: Omoschedasticità, informata dall'ARCH
    homo_res = test_homoscedasticity(s, shock, arch_detected=arch_res["arch_detected"])

    # Step 6: Stazionarietà (sulla serie grezza, livelli)
    stat_res = test_stationarity(s)

    results = {
        "stationarity":    stat_res,
        "autocorrelation": ac_res,
        "normality":       norm_res,
        "homoscedasticity":homo_res,
        "arch":            arch_res,
    }

    for name, res in results.items():
        icon = {"OK": "✓", "⚠ WARN": "⚠", "✗ FAIL": "✗",
                "✗ FAIL (volatility clustering)": "✗"}.get(res["status"], "?")
        print(f"    {icon} {name:<18}  {res['status']:<35}  {res['detail']}")
    return results


# ══════════════════════════════════════════════════════════════════════════════
# H₀ — Test per singola cella (evento × carburante)
# ══════════════════════════════════════════════════════════════════════════════

def _cliff_delta_interp(d: float) -> str:
    a = abs(d)
    if   a < 0.147: return "negligible"
    elif a < 0.330: return "small"
    elif a < 0.474: return "medium"
    else:           return "large"


def test_h0_speculative(
    full_margin: pd.Series,
    shock: pd.Timestamp,
    pre_days: int = H0_PRE_DAYS,
    post_days: int = H0_POST_DAYS,
) -> dict:
    """
    Test H₀: il margine post-shock non supera il margine pre-shock.

    Usa la SERIE COMPLETA (non la finestra di diagnostica) per avere
    il numero di osservazioni pre/post più ampio possibile.

    Test: Mann-Whitney U one-sided
      H₀: F(post) ≤ F(pre)   [i.e. post non stocasticamente dominante su pre]
      H₁: F(post) > F(pre)   [profitto anomalo — speculazione]
    Effect size: Cliff's δ = (U/(n₁·n₂))·2 − 1  ∈ [−1, 1]
      δ > 0 → la distribuzione post tende a valori più alti di pre
      δ > 0.474 → effetto "large" (soglia Cohen)
    """
    s = full_margin.dropna()
    pre_start = shock - pd.Timedelta(days=pre_days)
    post_end  = shock + pd.Timedelta(days=post_days)

    pre  = s[(s.index >= pre_start) & (s.index < shock)]
    post = s[(s.index >= shock) & (s.index <= post_end)]

    if len(pre) < 10 or len(post) < 10:
        return {
            "status": "insufficient_data",
            "n_pre": len(pre), "n_post": len(post),
            "mw_p_onesided": np.nan, "cliff_delta": np.nan,
            "reject_h0": False,
            "mean_pre": np.nan, "mean_post": np.nan, "delta_mean": np.nan,
            "detail": f"Dati insufficienti: n_pre={len(pre)}, n_post={len(post)} (min 10)",
        }

    # One-sided: H₁ post > pre
    u_stat, mw_p = stats.mannwhitneyu(post, pre, alternative="greater")
    # Two-sided per confronto
    _, mw_p2 = stats.mannwhitneyu(post, pre, alternative="two-sided")

    # Cliff's delta: (2U / n1*n2) - 1  (dove U da H₁: post > pre)
    n1, n2 = len(post), len(pre)
    cliff_d = (2.0 * float(u_stat) / (n1 * n2)) - 1.0
    cliff_interp = _cliff_delta_interp(cliff_d)

    mean_pre  = float(pre.mean())
    mean_post = float(post.mean())
    delta_mean = mean_post - mean_pre

    reject_h0 = (mw_p < ALPHA) and (cliff_d > 0)
    status = "RIFIUTO H₀" if reject_h0 else "NON RIFIUTO H₀"

    return {
        "status": status,
        "n_pre": n1, "n_post": n2,
        "mean_pre": mean_pre, "mean_post": mean_post, "delta_mean": delta_mean,
        "u_stat": float(u_stat),
        "mw_p_onesided": mw_p, "mw_p_twosided": mw_p2,
        "cliff_delta": cliff_d, "cliff_interp": cliff_interp,
        "reject_h0": reject_h0,
        "detail": (f"Δmean={delta_mean:+.4f} €/L | "
                   f"MW p(one-sided)={mw_p:.3f} | "
                   f"Cliff δ={cliff_d:.3f} [{cliff_interp}] | "
                   f"{status}"),
    }


# ══════════════════════════════════════════════════════════════════════════════
# H₀ — Sintesi per evento, carburante, globale  (Fisher combination)
# ══════════════════════════════════════════════════════════════════════════════

def _fisher_combine(p_values: list[float]) -> tuple[float, float]:
    """
    Fisher's method: χ² = −2·Σln(pᵢ), df = 2k.
    Restituisce (chi2_stat, p_combined).
    Filtra automaticamente NaN e p = 0 (sostituisce con 1e-15).
    """
    ps = [max(float(p), 1e-15) for p in p_values if not np.isnan(p)]
    if not ps:
        return np.nan, np.nan
    chi2 = -2.0 * sum(np.log(ps))
    df   = 2 * len(ps)
    p_comb = 1.0 - chi2_dist.cdf(chi2, df=df)
    return chi2, p_comb


def synthesize_h0(h0_grid: dict) -> dict:
    """
    h0_grid: {ev_name: {fuel_key: risultato_test_h0}}

    Produce tre livelli di sintesi con Fisher combination:
      • per_evento[ev_name]    — combina benzina + gasolio per quell'evento
      • per_carburante[fuel]   — combina i 3 eventi per quel carburante
      • globale                — combina tutte le celle disponibili
    """
    ev_names   = list(EVENTS.keys())
    fuel_names = list(FUELS.keys())

    per_evento: dict = {}
    for ev in ev_names:
        ps = [h0_grid.get(ev, {}).get(f, {}).get("mw_p_onesided", np.nan)
              for f in fuel_names]
        ps_valid = [p for p in ps if not np.isnan(p)]
        chi2, p_c = _fisher_combine(ps_valid) if ps_valid else (np.nan, np.nan)
        per_evento[ev] = {
            "chi2": chi2, "p_combined": p_c,
            "k": len(ps_valid),
            "reject_h0": (p_c < ALPHA) if not np.isnan(p_c) else False,
        }

    per_carburante: dict = {}
    for fuel in fuel_names:
        ps = [h0_grid.get(ev, {}).get(fuel, {}).get("mw_p_onesided", np.nan)
              for ev in ev_names]
        ps_valid = [p for p in ps if not np.isnan(p)]
        chi2, p_c = _fisher_combine(ps_valid) if ps_valid else (np.nan, np.nan)
        per_carburante[fuel] = {
            "chi2": chi2, "p_combined": p_c,
            "k": len(ps_valid),
            "reject_h0": (p_c < ALPHA) if not np.isnan(p_c) else False,
        }

    # Globale
    all_ps = [h0_grid.get(ev, {}).get(f, {}).get("mw_p_onesided", np.nan)
              for ev in ev_names for f in fuel_names]
    all_valid = [p for p in all_ps if not np.isnan(p)]
    g_chi2, g_p = _fisher_combine(all_valid) if all_valid else (np.nan, np.nan)
    globale = {
        "chi2": g_chi2, "p_combined": g_p,
        "k": len(all_valid),
        "reject_h0": (g_p < ALPHA) if not np.isnan(g_p) else False,
    }

    return {"per_evento": per_evento, "per_carburante": per_carburante, "globale": globale}


# ══════════════════════════════════════════════════════════════════════════════
# Plot — Serie storica margine
# ══════════════════════════════════════════════════════════════════════════════

def plot_margin_series(margin: pd.DataFrame) -> plt.Figure:
    fig, axes = plt.subplots(2, 1, figsize=(13, 7), sharex=True)
    fig.suptitle("Margine industriale+distribuzione+retailer (€/L)\n"
                 "= Prezzo netto ex-tasse  −  Wholesale futures convertiti",
                 fontsize=12, fontweight="bold")
    configs = [
        ("margin_gasolio", "#1D3557", "Gasolio  (netto SISEN − Gas Oil futures €/L)", "Gasolio"),
        ("margin_benzina", "#E63946", "Benzina  (netto SISEN − Eurobob B7H1 futures €/L)", "Benzina"),
    ]
    for ax, (col, color, desc, title) in zip(axes, configs):
        s = margin[col].dropna()
        if s.empty:
            ax.text(0.5, 0.5, "Dati non disponibili", ha="center", va="center",
                    transform=ax.transAxes, fontsize=12, color="grey")
        else:
            ax.fill_between(s.index, s.values, alpha=0.15, color=color)
            ax.plot(s.index, s.values, color=color, lw=1.0, label=desc)
            ax.axhline(s.mean(), color=color, lw=1, ls=":", alpha=0.8,
                       label=f"media = {s.mean():.3f} €/L")
            ax.axhline(0, color="black", lw=0.6)
            for ev_name, ev in EVENTS.items():
                if ev["shock"] >= s.index.min() and ev["shock"] <= s.index.max():
                    ax.axvline(ev["shock"], color=ev["color"], lw=1.5, ls="--", alpha=0.8)
                    ax.text(ev["shock"], s.max(),
                            f" {ev['shock'].strftime('%b %Y')}",
                            color=ev["color"], fontsize=7, va="top")
        ax.set_ylabel("€/L", fontsize=9)
        ax.set_title(title, fontsize=10, fontweight="bold")
        ax.legend(fontsize=8, loc="upper left")
        ax.grid(axis="y", alpha=0.2)
        ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    axes[-1].xaxis.set_major_locator(mdates.YearLocator())
    fig.tight_layout()
    return fig


# ══════════════════════════════════════════════════════════════════════════════
# Plot — Grafici diagnostici (uno per test × evento × fuel)
# ══════════════════════════════════════════════════════════════════════════════

def _badge(ax, status: str) -> None:
    _COLORS = {"OK": _OK, "⚠ WARN": _WARN, "✗ FAIL": _FAIL,
               "✗ FAIL (volatility clustering)": _FAIL}
    col   = _COLORS.get(status, "#aaa")
    label = status if status.startswith(("✓","⚠","✗")) else status
    ax.text(0.98, 0.97, label, transform=ax.transAxes,
            ha="right", va="top", fontsize=10, fontweight="bold", color="white",
            bbox=dict(boxstyle="round,pad=0.35", facecolor=col, edgecolor="none"), zorder=10)


def _base_title(ev: dict, fuel_key: str, test_label: str) -> str:
    return f"{test_label}  —  {ev['label']}  ·  {fuel_key.capitalize()}  [margine]"


def plot_one_stationarity(win, ev, fuel_key, fcolor, res):
    fig, ax = plt.subplots(figsize=(9, 4))
    shock = ev["shock"]
    ax.fill_between(win.index, win.values, alpha=0.15, color=fcolor)
    ax.plot(win.index, win.values, color=fcolor, lw=1.2)
    ax.axhline(0, color="black", lw=0.6)
    ax.axvline(shock, color=ev["color"], lw=2, ls="--")
    ax.axvspan(win.index.min(), shock, alpha=0.04, color="steelblue")
    ax.axvspan(shock, win.index.max(), alpha=0.04, color="tomato")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    ax.set_ylabel("Margine (€/L)", fontsize=10)
    ax.grid(axis="y", alpha=0.25)
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    adf_ok  = "✓" if res["adf_p"]  < ALPHA else "✗"
    kpss_ok = "✓" if res["kpss_p"] > ALPHA else "✗"
    pnote = res.get("power_note", "")
    ax.text(0.02, 0.05,
            f"ADF   p = {res['adf_p']:.3f}  {adf_ok}   (rif. H₀ = staz.)\n"
            f"KPSS  p = {res['kpss_p']:.3f}  {kpss_ok}   (non rif. H₀ = staz.)\n"
            f"n = {res['n']}{pnote}",
            transform=ax.transAxes, fontsize=8.5, va="bottom", family="monospace",
            bbox=dict(boxstyle="round,pad=0.4", facecolor="white", alpha=0.9, edgecolor="none"))
    ax.set_title(_base_title(ev, fuel_key, "Stazionarietà — ADF + KPSS"), fontsize=11, fontweight="bold")
    _badge(ax, res["status"])
    fig.tight_layout()
    return fig


def plot_one_autocorrelation(win, ev, fuel_key, fcolor, res):
    from statsmodels.graphics.tsaplots import plot_acf
    residuals = res.get("residuals", win.diff().dropna().values)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # Panel 1: ACF della serie grezza
    try:
        from statsmodels.graphics.tsaplots import plot_acf as _pacf
        plot_acf(win.dropna(), lags=25, ax=axes[0], color="grey", alpha=0.05,
                 zero=False, title="")
        axes[0].set_title("ACF serie grezza (solo visivo)", fontsize=9)
    except Exception:
        axes[0].text(0.5, 0.5, "n.d.", ha="center", va="center", transform=axes[0].transAxes)

    # Panel 2: ACF residui AR — quello che conta
    try:
        plot_acf(residuals, lags=min(25, len(residuals)//2 - 1), ax=axes[1],
                 color=fcolor, alpha=0.05, zero=False, title="")
        axes[1].set_title(f"ACF residui AR({res['ar_order']}) — Ljung-Box applicato qui",
                          fontsize=9, fontweight="bold")
    except Exception:
        axes[1].text(0.5, 0.5, "n.d.", ha="center", va="center", transform=axes[1].transAxes)

    for ax in axes:
        ax.set_xlabel("Lag (giorni)", fontsize=9)
        ax.set_ylabel("ACF", fontsize=9)
        ax.grid(axis="y", alpha=0.25)
        ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)

    lb_vals = list(res["lb"].values())
    fallback = " [FALLBACK: diff. prime]" if res.get("fallback") else ""
    _aic_str760 = f"{res['aic']:.1f}" if not np.isnan(res["aic"]) else "n.d."
    axes[1].text(0.02, 0.05,
                 f"φ AR(1) = {res['phi_ar1']:.3f}  n_eff={res['n_eff']}/{res['n']}\n"
                 f"AR({res['ar_order']}) AIC={_aic_str760}\n"
                 f"LB p (lag 5/10/20) = {lb_vals[0]}  {lb_vals[1]}  {lb_vals[2]}{fallback}",
                 transform=axes[1].transAxes, fontsize=8.5, va="bottom", family="monospace",
                 bbox=dict(boxstyle="round,pad=0.4", facecolor="white", alpha=0.9, edgecolor="none"))

    fig.suptitle(_base_title(ev, fuel_key, "Autocorrelazione — AR(p) → Ljung-Box sui residui"),
                 fontsize=11, fontweight="bold")
    _badge(axes[1], res["status"])
    fig.tight_layout()
    return fig


def plot_one_normality(win, ev, fuel_key, fcolor, res):
    residuals_key = "residuals"  # from autocorrelation result
    ar_residuals = res.get("_ar_residuals", win.diff().dropna().values)

    fig, axes = plt.subplots(1, 2, figsize=(11, 5))

    # QQ plot
    ax = axes[0]
    data = ar_residuals[np.isfinite(ar_residuals)]
    (osm, osr), (slope, intercept, _) = stats.probplot(data, dist="norm")
    n  = len(data)
    pi = stats.norm.cdf(osm)
    phi_i = np.where(stats.norm.pdf(osm) < 1e-10, 1e-10, stats.norm.pdf(osm))
    se = np.sqrt(pi * (1 - pi) / n) / phi_i
    ci_lo = slope * osm + intercept - 1.96 * abs(slope) * se
    ci_hi = slope * osm + intercept + 1.96 * abs(slope) * se
    ax.fill_between(osm, ci_lo, ci_hi, color="red", alpha=0.12, label="IC 95%")
    ax.scatter(osm, osr, s=10, color=fcolor, alpha=0.65, rasterized=True, zorder=3)
    xlim = np.array([min(osm), max(osm)])
    ax.plot(xlim, slope * xlim + intercept, color="red", lw=1.5, ls="--")
    ax.set_xlabel("Quantili N(0,1)", fontsize=9)
    ax.set_ylabel("Quantili residui AR", fontsize=9)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.25)
    ax.set_title("QQ plot  (residui AR)", fontsize=10)
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)

    # Istogramma residui + curva normale teorica
    ax2 = axes[1]
    ax2.hist(data, bins=min(30, n//3 + 1), density=True, color=fcolor, alpha=0.5, label="residui AR")
    xg = np.linspace(data.min(), data.max(), 200)
    ax2.plot(xg, stats.norm.pdf(xg, data.mean(), data.std()), "r--", lw=1.5, label="N(μ,σ)")
    ax2.set_xlabel("Residuo AR", fontsize=9)
    ax2.set_ylabel("Densità", fontsize=9)
    ax2.legend(fontsize=8)
    ax2.grid(alpha=0.25)
    ax2.set_title("Istogramma residui AR", fontsize=10)
    ax2.spines["top"].set_visible(False); ax2.spines["right"].set_visible(False)

    sw_str = f"SW    p = {res['sw_p']:.3f}" if not np.isnan(res.get("sw_p", np.nan)) else f"SW    {res.get('sw_note','n.d.')}"
    ax.text(0.02, 0.97,
            f"JB    p = {res['jb_p']:.3f}  {'✓' if res['jb_ok'] else '✗'}\n"
            f"D'Ag  p = {res['dag_p']:.3f}  {'✓' if res['dag_ok'] else '✗'}\n"
            f"{sw_str}\n"
            f"skew={res['skewness']:.2f}  kurt_exc={res['kurt_excess']:.2f}",
            transform=ax.transAxes, fontsize=8.5, va="top", family="monospace",
            bbox=dict(boxstyle="round,pad=0.4", facecolor="white", alpha=0.9, edgecolor="none"))

    fig.suptitle(_base_title(ev, fuel_key, "Normalità residui AR — JB + D'Agostino K²"), fontsize=11, fontweight="bold")
    _badge(axes[0], res["status"])
    fig.tight_layout()
    return fig


def plot_one_homoscedasticity(win, ev, fuel_key, fcolor, res):
    fig, ax = plt.subplots(figsize=(9, 4))
    shock = ev["shock"]
    roll_std = win.rolling(14, min_periods=7).std()
    ax.fill_between(roll_std.index, roll_std.values, alpha=0.2, color=fcolor)
    ax.plot(roll_std.index, roll_std.values, color=fcolor, lw=1.5)
    ax.axvline(shock, color=ev["color"], lw=2, ls="--")
    pre_idx  = win.index[win.index < shock]
    post_idx = win.index[win.index >= shock]
    if len(pre_idx) and len(post_idx) and not np.isnan(res["std_pre"]):
        ax.hlines(res["std_pre"],  pre_idx.min(), pre_idx.max(),
                  colors="steelblue", lw=1.5, ls=":", label=f"σ pre = {res['std_pre']:.4f}")
        ax.hlines(res["std_post"], post_idx.min(), post_idx.max(),
                  colors="tomato",   lw=1.5, ls=":", label=f"σ post = {res['std_post']:.4f}")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    ax.set_ylabel("σ rolling 14gg  (€/L)", fontsize=10)
    ax.legend(fontsize=9, loc="upper left")
    ax.grid(axis="y", alpha=0.25)
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    arch_note = "⚠ Levene unreliable (ARCH)" if res.get("arch_warning") else ""
    ax.text(0.98, 0.05,
            f"Fligner-Killeen  p = {res['fligner_p']:.3f}  [PRIMARIO]\n"
            f"Levene           p = {res['lev_p']:.3f}  {arch_note}\n"
            f"ratio σ post/pre = {res.get('ratio', float('nan')):.2f}×",
            transform=ax.transAxes, fontsize=8.5, va="bottom", ha="right", family="monospace",
            bbox=dict(boxstyle="round,pad=0.4", facecolor="white", alpha=0.9, edgecolor="none"))
    ax.set_title(_base_title(ev, fuel_key, "Omoschedasticità — Fligner-Killeen (primario) + Levene"), fontsize=11, fontweight="bold")
    _badge(ax, res["status"])
    fig.tight_layout()
    return fig


def plot_one_arch(win, ev, fuel_key, fcolor, res):
    fig, ax = plt.subplots(figsize=(9, 4))
    shock = ev["shock"]
    diff2 = win.diff().dropna() ** 2
    ax.fill_between(diff2.index, diff2.values, alpha=0.35, color=fcolor)
    ax.plot(diff2.index, diff2.values, color=fcolor, lw=0.8)
    ax.axvline(shock, color=ev["color"], lw=2, ls="--")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    ax.set_ylabel("(Δmargine)²  [proxy varianza]", fontsize=10)
    ax.grid(axis="y", alpha=0.25)
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    ax.text(0.02, 0.97,
            f"ARCH-LM  p = {res['lm_p']:.3f}\n"
            f"F-stat     = {res['f_stat']:.2f}\n"
            f"Applicato ai RESIDUI AR (non alla serie grezza)",
            transform=ax.transAxes, fontsize=8.5, va="top", family="monospace",
            bbox=dict(boxstyle="round,pad=0.4", facecolor="white", alpha=0.9, edgecolor="none"))
    ax.set_title(_base_title(ev, fuel_key, "Effetti ARCH — residui AR(p)"), fontsize=11, fontweight="bold")
    _badge(ax, res["status"])
    fig.tight_layout()
    return fig


# ══════════════════════════════════════════════════════════════════════════════
# Plot H₀ — Matrice verdetto (evento × carburante)
# ══════════════════════════════════════════════════════════════════════════════

def plot_h0_verdict_matrix(h0_grid: dict, synthesis: dict) -> plt.Figure:
    """
    Heatmap evento × carburante con colore = rifiuto/non rifiuto H₀.
    Annotazioni: Cliff's δ, p-value one-sided.
    Riga/colonna marginale: Fisher combination.
    """
    ev_names   = list(EVENTS.keys())
    fuel_names = list(FUELS.keys())
    ev_labels  = [EVENTS[e]["label"].replace("\n", " ") for e in ev_names]

    n_ev   = len(ev_names)
    n_fuel = len(fuel_names)

    # Costruisci matrici numeriche per la heatmap
    p_matrix     = np.full((n_ev, n_fuel), np.nan)
    cliff_matrix = np.full((n_ev, n_fuel), np.nan)
    reject_matrix= np.zeros((n_ev, n_fuel), dtype=bool)

    for i, ev in enumerate(ev_names):
        for j, fuel in enumerate(fuel_names):
            cell = h0_grid.get(ev, {}).get(fuel, {})
            if cell.get("status") != "insufficient_data":
                p_matrix[i, j]      = cell.get("mw_p_onesided", np.nan)
                cliff_matrix[i, j]  = cell.get("cliff_delta", np.nan)
                reject_matrix[i, j] = cell.get("reject_h0", False)

    # Figura con extra colonna/riga per sintesi Fisher
    fig, ax = plt.subplots(figsize=(max(8, n_fuel * 2.5 + 3), n_ev * 1.8 + 2.5))
    ax.set_xlim(-0.5, n_fuel + 0.5)
    ax.set_ylim(-0.5, n_ev + 0.5)
    ax.set_aspect("equal")
    ax.invert_yaxis()

    # Celle per cella
    for i, ev in enumerate(ev_names):
        for j, fuel in enumerate(fuel_names):
            cell = h0_grid.get(ev, {}).get(fuel, {})
            if cell.get("status") == "insufficient_data":
                color = "#eeeeee"
                txt   = "n.d."
            else:
                reject = cell.get("reject_h0", False)
                color  = "#e74c3c" if reject else "#2ecc71"
                p      = cell.get("mw_p_onesided", np.nan)
                d      = cell.get("cliff_delta", np.nan)
                dm     = cell.get("delta_mean", np.nan)
                interp = cell.get("cliff_interp", "")
                verdict = "RIFIUTO H₀\n(profitto anomalo)" if reject else "NON RIFIUTO H₀"
                txt = (f"{verdict}\n"
                       f"Δmean={dm:+.4f} €/L\n"
                       f"p(one)={p:.3f}  δ={d:.3f}\n"
                       f"[{interp}]")

            rect = plt.Rectangle((j - 0.45, i - 0.45), 0.9, 0.9,
                                  facecolor=color, edgecolor="white", lw=2)
            ax.add_patch(rect)
            ax.text(j, i, txt, ha="center", va="center", fontsize=7.5,
                    color="white" if cell.get("reject_h0") else "black", fontweight="bold")

    # Riga sintesi Fisher per-evento
    for i, ev in enumerate(ev_names):
        s = synthesis["per_evento"].get(ev, {})
        p_c = s.get("p_combined", np.nan)
        rej = s.get("reject_h0", False)
        color = "#c0392b" if rej else "#27ae60"
        rect = plt.Rectangle((n_fuel - 0.45, i - 0.45), 0.9, 0.9,
                              facecolor=color, edgecolor="white", lw=2, alpha=0.7)
        ax.add_patch(rect)
        verdict = "RIFIUTO H₀" if rej else "NON RIFIUTO H₀"
        txt = f"Fisher\n{verdict}\np={p_c:.3f}" if not np.isnan(p_c) else "Fisher\nn.d."
        ax.text(n_fuel, i, txt, ha="center", va="center", fontsize=7, color="white", fontweight="bold")

    # Colonna sintesi Fisher per-carburante
    for j, fuel in enumerate(fuel_names):
        s = synthesis["per_carburante"].get(fuel, {})
        p_c = s.get("p_combined", np.nan)
        rej = s.get("reject_h0", False)
        color = "#c0392b" if rej else "#27ae60"
        rect = plt.Rectangle((j - 0.45, n_ev - 0.45), 0.9, 0.9,
                              facecolor=color, edgecolor="white", lw=2, alpha=0.7)
        ax.add_patch(rect)
        verdict = "RIFIUTO H₀" if rej else "NON RIFIUTO H₀"
        txt = f"Fisher\n{verdict}\np={p_c:.3f}" if not np.isnan(p_c) else "Fisher\nn.d."
        ax.text(j, n_ev, txt, ha="center", va="center", fontsize=7, color="white", fontweight="bold")

    # Cella globale
    sg = synthesis["globale"]
    p_g = sg.get("p_combined", np.nan)
    rej_g = sg.get("reject_h0", False)
    color_g = "#922b21" if rej_g else "#1e8449"
    rect = plt.Rectangle((n_fuel - 0.45, n_ev - 0.45), 0.9, 0.9,
                          facecolor=color_g, edgecolor="white", lw=2)
    ax.add_patch(rect)
    txt_g = f"GLOBALE\n{'RIFIUTO' if rej_g else 'NON RIF.'} H₀\np={p_g:.3f}" if not np.isnan(p_g) else "GLOBALE\nn.d."
    ax.text(n_fuel, n_ev, txt_g, ha="center", va="center", fontsize=7, color="white", fontweight="bold")

    # Etichette assi
    ax.set_xticks(range(n_fuel + 1))
    ax.set_xticklabels([f.capitalize() for f in fuel_names] + ["SINTESI\nFisher"], fontsize=9)
    ax.set_yticks(range(n_ev + 1))
    ax.set_yticklabels(ev_labels + ["SINTESI\nFisher"], fontsize=8.5)

    ax.set_title(
        "Verdetto H₀: i distributori italiani generano profitti anomali\n"
        "in prossimità di shock geopolitici?\n"
        "Test: Mann-Whitney U one-sided (H₁: margine_post > margine_pre) + Fisher combination",
        fontsize=11, fontweight="bold", pad=15,
    )

    # Legenda
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor="#e74c3c", label="RIFIUTO H₀ — profitto anomalo rilevato"),
        Patch(facecolor="#2ecc71", label="NON RIFIUTO H₀ — margine coerente"),
        Patch(facecolor="#eeeeee", label="Dati insufficienti"),
    ]
    ax.legend(handles=legend_elements, loc="lower center",
              bbox_to_anchor=(0.5, -0.12), ncol=3, fontsize=8)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ax.tick_params(left=False, bottom=False)
    fig.tight_layout()
    return fig


def plot_h0_distributions(full_margin: pd.DataFrame, h0_grid: dict) -> plt.Figure:
    """
    Violin plot dei margini pre e post shock per ogni (evento × carburante).
    Mostra visivamente l'entità e la direzione degli shift.
    """
    fuel_cols  = {"benzina": "margin_benzina", "gasolio": "margin_gasolio"}
    fuel_colors = {"benzina": "#E63946", "gasolio": "#1D3557"}
    ev_names   = list(EVENTS.keys())
    fuel_names = list(FUELS.keys())

    n_ev   = len(ev_names)
    n_fuel = len(fuel_names)
    fig, axes = plt.subplots(n_ev, n_fuel,
                             figsize=(4.5 * n_fuel, 3.5 * n_ev),
                             squeeze=False)
    fig.suptitle(
        "H₀ — Distribuzione margine pre vs post shock\n"
        "(pre = 60gg, post = 45gg dalla data shock)",
        fontsize=12, fontweight="bold",
    )

    for i, ev_name in enumerate(ev_names):
        ev = EVENTS[ev_name]
        shock = ev["shock"]
        for j, fuel_key in enumerate(fuel_names):
            ax = axes[i][j]
            col = fuel_cols[fuel_key]
            fcolor = fuel_colors[fuel_key]
            s = full_margin[col].dropna()

            pre  = s[(s.index >= shock - pd.Timedelta(days=H0_PRE_DAYS)) & (s.index < shock)]
            post = s[(s.index >= shock) & (s.index <= shock + pd.Timedelta(days=H0_POST_DAYS))]

            if len(pre) < 5 or len(post) < 5:
                ax.text(0.5, 0.5, "Dati n.d.", ha="center", va="center",
                        transform=ax.transAxes, color="grey")
                ax.set_title(f"{fuel_key} — {ev['label'].replace(chr(10),' ')}", fontsize=8)
                continue

            data_plot = [pre.values, post.values]
            parts = ax.violinplot(data_plot, positions=[0, 1],
                                  showmedians=True, showextrema=True)
            for pc in parts["bodies"]:
                pc.set_facecolor(fcolor)
                pc.set_alpha(0.5)
            for part in ("cmedians", "cbars", "cmins", "cmaxes"):
                if part in parts:
                    parts[part].set_edgecolor(fcolor)

            # Aggiungi punti gitter
            for k, d in enumerate([pre.values, post.values]):
                jitter = np.random.default_rng(42).uniform(-0.05, 0.05, len(d))
                ax.scatter(jitter + k, d, s=8, color=fcolor, alpha=0.4, zorder=3)

            cell = h0_grid.get(ev_name, {}).get(fuel_key, {})
            p_v  = cell.get("mw_p_onesided", np.nan)
            cliff = cell.get("cliff_delta", np.nan)
            verdict = cell.get("status", "n.d.")
            vcolor  = "#e74c3c" if cell.get("reject_h0") else "#27ae60"

            ax.set_xticks([0, 1])
            ax.set_xticklabels([f"Pre\n(n={len(pre)})", f"Post\n(n={len(post)})"], fontsize=8)
            ax.set_ylabel("Margine (€/L)", fontsize=8)
            ax.grid(axis="y", alpha=0.2)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)

            ax.set_title(
                f"{fuel_key.capitalize()} — {ev['label'].replace(chr(10),' ')}",
                fontsize=8.5, fontweight="bold",
            )
            ax.text(0.5, 0.98,
                    f"p={p_v:.3f}  δ={cliff:.3f}\n{verdict}",
                    transform=ax.transAxes, ha="center", va="top", fontsize=8,
                    color="white", fontweight="bold",
                    bbox=dict(boxstyle="round,pad=0.3", facecolor=vcolor, edgecolor="none"))

    fig.tight_layout()
    return fig


# ══════════════════════════════════════════════════════════════════════════════
# Export CSV
# ══════════════════════════════════════════════════════════════════════════════

def export_diagnostic_csv(all_results: dict) -> None:
    """Esporta CSV diagnostici (stazionarietà, autocorrelazione, …)."""
    rows_summary = []
    rows = {k: [] for k in ("stationarity","autocorrelation","normality","homoscedasticity","arch")}

    for ev_name, ev in EVENTS.items():
        for fuel_key, res_all in all_results.get(ev_name, {}).items():
            base = {"evento": ev_name, "shock": ev["shock"].date(), "carburante": fuel_key}

            for test_it, test_key in [
                ("Stazionarietà",     "stationarity"),
                ("Autocorrelazione",  "autocorrelation"),
                ("Normalità",         "normality"),
                ("Omoschedasticità",  "homoscedasticity"),
                ("Effetti ARCH",      "arch"),
            ]:
                res = res_all[test_key]
                rows_summary.append({**base, "test": test_it,
                                     "status": res["status"], "dettaglio": res["detail"]})

            r = res_all["stationarity"]
            rows["stationarity"].append({**base, "status": r["status"],
                "adf_p": round(r["adf_p"],4), "adf_ok": r["adf_ok"],
                "kpss_stat": round(r["kpss_stat"],4), "kpss_p": round(r["kpss_p"],4),
                "kpss_ok": r["kpss_ok"], "n": r["n"]})

            r = res_all["autocorrelation"]
            rows["autocorrelation"].append({**base, "status": r["status"],
                "ar_order": r["ar_order"], "aic": round(r["aic"],2) if not np.isnan(r["aic"]) else "",
                "phi_ar1": round(r["phi_ar1"],4), "n": r["n"], "n_eff": r["n_eff"],
                **{f"lb_p_lag{k}": v for k,v in r["lb"].items()},
                "lb_fail": r["lb_fail"], "fallback": r["fallback"]})

            r = res_all["normality"]
            rows["normality"].append({**base, "status": r["status"],
                "jb_stat": round(r["jb_stat"],4), "jb_p": round(r["jb_p"],4), "jb_ok": r["jb_ok"],
                "dag_stat": round(r["dag_stat"],4), "dag_p": round(r["dag_p"],4), "dag_ok": r["dag_ok"],
                "sw_stat": round(r["sw_stat"],4) if not np.isnan(r.get("sw_stat",np.nan)) else "",
                "sw_p": round(r["sw_p"],4) if not np.isnan(r.get("sw_p",np.nan)) else "",
                "skewness": round(r["skewness"],4), "kurtosis_excess": round(r["kurt_excess"],4),
                "n_residuals": r["n_residuals"]})

            r = res_all["homoscedasticity"]
            rows["homoscedasticity"].append({**base, "status": r["status"],
                "fligner_stat": round(r["fligner_stat"],4) if not np.isnan(r["fligner_stat"]) else "",
                "fligner_p": round(r["fligner_p"],4) if not np.isnan(r["fligner_p"]) else "",
                "fligner_het": r.get("fligner_het",""),
                "lev_stat": round(r["lev_stat"],4) if not np.isnan(r["lev_stat"]) else "",
                "lev_p": round(r["lev_p"],4) if not np.isnan(r["lev_p"]) else "",
                "sigma_pre": round(r["std_pre"],6) if not np.isnan(r["std_pre"]) else "",
                "sigma_post": round(r["std_post"],6) if not np.isnan(r["std_post"]) else "",
                "ratio_post_pre": round(r.get("ratio",np.nan),3) if not np.isnan(r.get("ratio",np.nan)) else "",
                "arch_warning": r["arch_warning"]})

            r = res_all["arch"]
            rows["arch"].append({**base, "status": r["status"],
                "lm_stat": round(r["lm_stat"],4) if not np.isnan(r["lm_stat"]) else "",
                "lm_p": round(r["lm_p"],4) if not np.isnan(r["lm_p"]) else "",
                "f_stat": round(r["f_stat"],4) if not np.isnan(r["f_stat"]) else "",
                "arch_detected": r["arch_detected"]})

    pd.DataFrame(rows_summary).to_csv(OUT_DIR / "risultati_riepilogo.csv", index=False, encoding="utf-8-sig")
    print("  CSV: risultati_riepilogo.csv")
    for key, fname in [
        ("stationarity",     "risultati_stazionarieta.csv"),
        ("autocorrelation",  "risultati_autocorrelazione.csv"),
        ("normality",        "risultati_normalita.csv"),
        ("homoscedasticity", "risultati_omoschedasticita.csv"),
        ("arch",             "risultati_arch.csv"),
    ]:
        pd.DataFrame(rows[key]).to_csv(OUT_DIR / fname, index=False, encoding="utf-8-sig")
        print(f"  CSV: {fname}")


def export_h0_csv(h0_grid: dict, synthesis: dict) -> None:
    """Esporta CSV per H₀ — per cella e sintesi Fisher."""
    ev_names   = list(EVENTS.keys())
    fuel_names = list(FUELS.keys())

    # Per cella
    rows_cell = []
    for ev in ev_names:
        for fuel in fuel_names:
            cell = h0_grid.get(ev, {}).get(fuel, {})
            rows_cell.append({
                "evento": ev,
                "shock": EVENTS[ev]["shock"].date(),
                "carburante": fuel,
                "n_pre": cell.get("n_pre", ""),
                "n_post": cell.get("n_post", ""),
                "mean_pre": round(cell.get("mean_pre", np.nan), 5) if not np.isnan(cell.get("mean_pre", np.nan)) else "",
                "mean_post": round(cell.get("mean_post", np.nan), 5) if not np.isnan(cell.get("mean_post", np.nan)) else "",
                "delta_mean": round(cell.get("delta_mean", np.nan), 5) if not np.isnan(cell.get("delta_mean", np.nan)) else "",
                "mw_p_onesided": round(cell.get("mw_p_onesided", np.nan), 4) if not np.isnan(cell.get("mw_p_onesided", np.nan)) else "",
                "mw_p_twosided": round(cell.get("mw_p_twosided", np.nan), 4) if not np.isnan(cell.get("mw_p_twosided", np.nan)) else "",
                "cliff_delta": round(cell.get("cliff_delta", np.nan), 4) if not np.isnan(cell.get("cliff_delta", np.nan)) else "",
                "cliff_interp": cell.get("cliff_interp", ""),
                "reject_h0": cell.get("reject_h0", ""),
                "status": cell.get("status", ""),
            })
    pd.DataFrame(rows_cell).to_csv(OUT_DIR / "h0_per_cella.csv", index=False, encoding="utf-8-sig")
    print("  CSV: h0_per_cella.csv")

    # Sintesi
    rows_synth = []
    for ev in ev_names:
        s = synthesis["per_evento"].get(ev, {})
        rows_synth.append({
            "dimensione": "evento", "categoria": ev,
            "chi2_fisher": round(s.get("chi2", np.nan), 3) if not np.isnan(s.get("chi2", np.nan)) else "",
            "p_combined": round(s.get("p_combined", np.nan), 4) if not np.isnan(s.get("p_combined", np.nan)) else "",
            "k_tests": s.get("k", ""),
            "reject_h0": s.get("reject_h0", ""),
        })
    for fuel in fuel_names:
        s = synthesis["per_carburante"].get(fuel, {})
        rows_synth.append({
            "dimensione": "carburante", "categoria": fuel,
            "chi2_fisher": round(s.get("chi2", np.nan), 3) if not np.isnan(s.get("chi2", np.nan)) else "",
            "p_combined": round(s.get("p_combined", np.nan), 4) if not np.isnan(s.get("p_combined", np.nan)) else "",
            "k_tests": s.get("k", ""),
            "reject_h0": s.get("reject_h0", ""),
        })
    sg = synthesis["globale"]
    rows_synth.append({
        "dimensione": "globale", "categoria": "tutti",
        "chi2_fisher": round(sg.get("chi2", np.nan), 3) if not np.isnan(sg.get("chi2", np.nan)) else "",
        "p_combined": round(sg.get("p_combined", np.nan), 4) if not np.isnan(sg.get("p_combined", np.nan)) else "",
        "k_tests": sg.get("k", ""),
        "reject_h0": sg.get("reject_h0", ""),
    })
    pd.DataFrame(rows_synth).to_csv(OUT_DIR / "h0_sintesi.csv", index=False, encoding="utf-8-sig")
    print("  CSV: h0_sintesi.csv")


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    print("Carico dati...")
    daily = (pd.read_csv(DAILY_CSV, parse_dates=["date"])
               .sort_values("date").set_index("date"))

    eurusd = load_eurusd(
        csv_path=EURUSD_CSV if EURUSD_CSV.exists() else None,
        start="2015-01-01", end="2026-12-31",
    )

    gasoil_eurl  = load_futures_eurl(GASOIL_CSV, GAS_OIL, eurusd)
    eurobob_eurl = load_futures_b7h1(EUROBOB_CSV, EUROBOB_HC, eurusd) \
                   if EUROBOB_CSV.exists() else None

    margin = build_margin(daily, gasoil_eurl, eurobob_eurl)

    benz_avail = margin["margin_benzina"].dropna()
    if not benz_avail.empty:
        print(f"  Margine benzina: {benz_avail.index.min().date()} → {benz_avail.index.max().date()}")
    else:
        print("  Margine benzina: Eurobob non disponibile")
    print(f"  Margine gasolio: {margin['margin_gasolio'].dropna().index.min().date()} → "
          f"{margin['margin_gasolio'].dropna().index.max().date()}")

    # ── Grafico 0: serie storica ──────────────────────────────────────────────
    print("\nGrafico 0: serie storica margine...")
    fig0 = plot_margin_series(margin)
    out0 = OUT_DIR / "00_margine_serie_storica.png"
    fig0.savefig(out0, dpi=150, bbox_inches="tight")
    plt.close(fig0)
    print(f"  ✓ {out0.name}")

    # ── Diagnostiche per evento × carburante ──────────────────────────────────
    fuel_margin_col = {"gasolio": "margin_gasolio", "benzina": "margin_benzina"}
    fuel_color_map  = {"gasolio": "#1D3557", "benzina": "#E63946"}

    all_results: dict = {}
    for ev_name, ev in EVENTS.items():
        shock = ev["shock"]
        print(f"\n{'═'*72}")
        print(f"  {ev_name}  (shock={shock.date()})")
        print(f"{'═'*72}")
        all_results[ev_name] = {}

        for fuel_key, margin_col in fuel_margin_col.items():
            win = margin[margin_col][
                (margin.index >= ev["pre_start"]) &
                (margin.index <= ev["post_end"])
            ].dropna()
            if len(win) < 20:
                print(f"  [{fuel_key}] dati insufficienti (n={len(win)}) — salto.")
                continue
            diag = run_diagnostics(win, shock, fuel_key)
            # Attacca i residui AR al risultato normalità per il plot
            diag["normality"]["_ar_residuals"] = diag["autocorrelation"]["residuals"]
            all_results[ev_name][fuel_key] = diag

    # ── 30 grafici diagnostici ────────────────────────────────────────────────
    TEST_PLOT_FNS = [
        ("01_stazionarieta",    "stationarity",    plot_one_stationarity),
        ("02_autocorrelazione", "autocorrelation", plot_one_autocorrelation),
        ("03_normalita",        "normality",        plot_one_normality),
        ("04_omoschedasticita", "homoscedasticity", plot_one_homoscedasticity),
        ("05_arch",             "arch",             plot_one_arch),
    ]

    print(f"\n{'═'*72}")
    print("Generazione grafici diagnostici...")
    count = 0
    for ev_name, ev in EVENTS.items():
        ev_slug = (ev_name.lower()
                   .replace(" ", "_").replace("(","").replace(")","")
                   .replace("/","").replace("-","_"))
        for fuel_key in ("gasolio", "benzina"):
            if fuel_key not in all_results.get(ev_name, {}):
                continue
            res_all = all_results[ev_name][fuel_key]
            fcolor  = fuel_color_map[fuel_key]
            win = margin[fuel_margin_col[fuel_key]][
                (margin.index >= ev["pre_start"]) &
                (margin.index <= ev["post_end"])
            ].dropna()

            for prefix, test_key, fn in TEST_PLOT_FNS:
                fig = fn(win, ev, fuel_key, fcolor, res_all[test_key])
                out = OUT_DIR / f"{prefix}__{ev_slug}__{fuel_key}.png"
                fig.savefig(out, dpi=150, bbox_inches="tight")
                plt.close(fig)
                print(f"  ✓ {out.name}")
                count += 1

    # ════════════════════════════════════════════════════════════════════════════
    # H₀ — TEST FINALE: i distributori italiani generano profitti anomali?
    # ════════════════════════════════════════════════════════════════════════════
    print(f"\n{'═'*72}")
    print("  H₀ FINALE — Profitti anomali in prossimità di shock geopolitici?")
    print(f"  H₀: il margine post-shock NON supera quello pre-shock")
    print(f"  Test: Mann-Whitney U one-sided + Cliff's δ")
    print(f"  Finestra: {H0_PRE_DAYS}gg pre / {H0_POST_DAYS}gg post shock (dalla serie completa)")
    print(f"{'═'*72}")

    h0_grid: dict = {}
    for ev_name, ev in EVENTS.items():
        shock = ev["shock"]
        h0_grid[ev_name] = {}
        print(f"\n  ▶ {ev_name}  (shock={shock.date()})")
        for fuel_key, margin_col in fuel_margin_col.items():
            full_margin_series = margin[margin_col]
            result = test_h0_speculative(full_margin_series, shock)
            h0_grid[ev_name][fuel_key] = result
            icon = "✗" if result.get("reject_h0") else "✓"
            print(f"    {icon} {fuel_key:<10}  {result['detail']}")

    # Sintesi Fisher
    print(f"\n  Sintesi Fisher combination:")
    synthesis = synthesize_h0(h0_grid)

    print(f"\n  Per evento:")
    for ev, s in synthesis["per_evento"].items():
        r = "RIFIUTO H₀" if s["reject_h0"] else "NON RIFIUTO H₀"
        p = s.get("p_combined", np.nan)
        print(f"    {'✗' if s['reject_h0'] else '✓'} {ev:<30}  Fisher χ²(2k={s['k']*2})  "
              f"p={p:.4f}  → {r}")

    print(f"\n  Per carburante:")
    for fuel, s in synthesis["per_carburante"].items():
        r = "RIFIUTO H₀" if s["reject_h0"] else "NON RIFIUTO H₀"
        p = s.get("p_combined", np.nan)
        print(f"    {'✗' if s['reject_h0'] else '✓'} {fuel:<12}  Fisher χ²(2k={s['k']*2})  "
              f"p={p:.4f}  → {r}")

    sg = synthesis["globale"]
    p_g = sg.get("p_combined", np.nan)
    print(f"\n  ┌─────────────────────────────────────────────────────────┐")
    print(f"  │  SINTESI GLOBALE  Fisher χ²(2k={sg['k']*2})  p={p_g:.4f}       │")
    if sg["reject_h0"]:
        print(f"  │  → RIFIUTO H₀: evidenza di profitti anomali           │")
    else:
        print(f"  │  → NON RIFIUTO H₀: margini coerenti con i costi       │")
    print(f"  └─────────────────────────────────────────────────────────┘")

    # Grafici H₀
    print(f"\n{'═'*72}")
    print("Generazione grafici H₀...")

    fig_matrix = plot_h0_verdict_matrix(h0_grid, synthesis)
    out_matrix = OUT_DIR / "h0_verdict_matrix.png"
    fig_matrix.savefig(out_matrix, dpi=150, bbox_inches="tight")
    plt.close(fig_matrix)
    print(f"  ✓ {out_matrix.name}")

    fig_dist = plot_h0_distributions(margin, h0_grid)
    out_dist = OUT_DIR / "h0_distributions.png"
    fig_dist.savefig(out_dist, dpi=150, bbox_inches="tight")
    plt.close(fig_dist)
    print(f"  ✓ {out_dist.name}")

    # CSV
    print(f"\n{'═'*72}")
    print("Export CSV...")
    export_diagnostic_csv(all_results)
    export_h0_csv(h0_grid, synthesis)

    print(f"\nDone. {count} grafici diagnostici + 2 grafici H₀ + CSV in: {OUT_DIR}")


if __name__ == "__main__":
    main()