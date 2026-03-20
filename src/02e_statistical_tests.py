#!/usr/bin/env python3
"""
02e_statistical_tests.py  —  Batteria completa di test statistici
===================================================================
Testa le 4 serie principali della pipeline crack-spread:
  • benzina_net    (prezzo pompa netto IVA/accise)
  • gasolio_net    (prezzo pompa netto IVA/accise)
  • margin_benzina (prezzo netto − futures Eurobob €/L)
  • margin_gasolio (prezzo netto − futures Gas Oil €/L)

Struttura dei test
──────────────────
A. INTERA SERIE STORICA (per ogni serie)
   A1. Statistiche descrittive  (media, std, skew, curtosi, IQR, CV, Hurst)
   A2. Stazionarietà             ADF (AIC-lag), KPSS (trend e level)
   A3. Normalità                 Jarque-Bera, D'Agostino K², Shapiro-Wilk (n≤200)
   A4. Autocorrelazione          Ljung-Box (lag 5/10/20), Durbin-Watson
   A5. Effetti ARCH              Engle ARCH-LM (lag 5/10)
   A6. Lunga memoria             Esponente di Hurst (metodo R/S e DFA)

B. FINESTRE EVENTO (pre 40 gg vs post 40 gg × 3 eventi × 4 serie)
   B1. Statistiche descrittive pre e post separati
   B2. Uguaglianza medie        Welch t (n_eff-corretto), Mann-Whitney U, Wilcoxon
   B3. Uguaglianza varianze     Levene, Bartlett, Fligner-Killeen, F-ratio, Brown-Forsythe
   B4. Uguaglianza distribuzione KS 2-campioni, Anderson-Darling 2-campioni
   B5. Effect size              Cohen d, Cliff δ, Hedge g
   B6. Normalità intra-finestra Shapiro-Wilk (pre e post)
   B7. Stazionarietà intra-finestra ADF (pre e post)
   B8. Correlazione seriale intra-finestra Ljung-Box (lag 5)

C. CONFRONTO CROSS-EVENTO (3 eventi insieme, per ogni serie)
   C1. Kruskal-Wallis           medie k-campioni
   C2. ANOVA a una via          (se normalità)
   C3. Fligner-Killeen k-campioni varianze
   C4. Bartlett k-campioni      varianze (assume normalità)

Output
──────
  data/plots/stat_tests/
    stat_tests_global.csv          — risultati A (serie completa)
    stat_tests_windows.csv         — risultati B (finestre pre/post)
    stat_tests_crossevent.csv      — risultati C (confronto cross-evento)
    plot_windows_{serie}.png       — strip + box plot per ogni serie
    plot_hurst_rolling.png         — Hurst esponente rolling

Uso
───
  python3 02e_statistical_tests.py
  python3 02e_statistical_tests.py --pre-win 60 --post-win 60
  python3 02e_statistical_tests.py --alpha 0.01
  python3 02e_statistical_tests.py --series margin_benzina margin_gasolio
"""

from __future__ import annotations

import argparse
import warnings
from pathlib import Path
import sys

import matplotlib.dates as mdates
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import (
    chi2 as chi2_dist,
    fligner,
    levene,
    bartlett,
    mannwhitneyu,
    ks_2samp,
    wilcoxon,
    shapiro,
    jarque_bera,
    f_oneway,
    kruskal,
)

# statsmodels — opzionale ma molto consigliato
try:
    from statsmodels.stats.diagnostic import acorr_ljungbox, het_arch
    from statsmodels.tsa.stattools import adfuller, kpss as kpss_test
    HAS_SM = True
except ImportError:
    HAS_SM = False
    warnings.warn(
        "statsmodels non trovato — ADF, KPSS, Ljung-Box e ARCH-LM disabilitati.\n"
        "  Installa con: pip install statsmodels"
    )

# Anderson-Darling 2-campioni (scipy >= 1.7)
try:
    from scipy.stats import anderson_ksamp
    HAS_ANDERSON_KSAMP = True
except ImportError:
    HAS_ANDERSON_KSAMP = False

sys.path.insert(0, str(Path(__file__).parent / "utils"))
from conversions import GAS_OIL, EUROBOB as EUROBOB_HC, load_eurusd, usd_ton_to_eur_liter

# ══════════════════════════════════════════════════════════════════════════════
# Configurazione
# ══════════════════════════════════════════════════════════════════════════════
BASE_DIR     = Path(__file__).parent
DAILY_CSV    = BASE_DIR / "data" / "processed" / "daily_fuel_prices_all.csv"
GASOIL_CSV   = BASE_DIR / "data" / "Futures" / "London Gas Oil Futures Historical Data.csv"
EUROBOB_CSV  = BASE_DIR / "data" / "Futures" / "Eurobob_B7H1_date.csv"
EURUSD_CSV   = BASE_DIR / "data" / "raw" / "eurusd.csv"
OUT_DIR      = BASE_DIR / "data" / "plots" / "stat_tests"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Finestre pre/post (default, sovrascrivibili da CLI)
PRE_WIN_DEFAULT  = 40
POST_WIN_DEFAULT = 40

ALPHA_DEFAULT = 0.05

EVENTS: dict[str, dict] = {
    "Ucraina (Feb 2022)": {
        "shock": pd.Timestamp("2022-02-24"),
        "color": "#e74c3c",
    },
    "Iran-Israele (Giu 2025)": {
        "shock": pd.Timestamp("2025-06-13"),
        "color": "#e67e22",
    },
    "Hormuz (Feb 2026)": {
        "shock": pd.Timestamp("2026-02-28"),
        "color": "#8e44ad",
    },
}

# Etichette brevi per i grafici
EVENT_SHORT = {
    "Ucraina (Feb 2022)":       "Ucraina\n2022",
    "Iran-Israele (Giu 2025)":  "Iran-Israele\n2025",
    "Hormuz (Feb 2026)":        "Hormuz\n2026",
}

ALL_SERIES = ["benzina_net", "gasolio_net", "margin_benzina", "margin_gasolio"]

SEP = "═" * 72


# ══════════════════════════════════════════════════════════════════════════════
# Caricamento dati (identico a 02d_v1_naive.py)
# ══════════════════════════════════════════════════════════════════════════════

def _load_gasoil_futures(eurusd: pd.Series) -> pd.Series:
    df = pd.read_csv(GASOIL_CSV, encoding="utf-8-sig", dtype=str)
    df["date"]  = pd.to_datetime(df["Date"], format="%m/%d/%Y", errors="coerce")
    df["price"] = (df["Price"].str.replace(",", "", regex=False)
                   .pipe(pd.to_numeric, errors="coerce"))
    df = df.dropna(subset=["date", "price"]).sort_values("date").set_index("date")
    return usd_ton_to_eur_liter(df["price"], eurusd, GAS_OIL)


def _load_eurobob_futures(eurusd: pd.Series) -> pd.Series | None:
    if not EUROBOB_CSV.exists():
        return None
    df = pd.read_csv(EUROBOB_CSV, encoding="utf-8-sig", dtype=str)
    _IT = {"gen": "Jan", "feb": "Feb", "mar": "Mar", "apr": "Apr",
           "mag": "May", "giu": "Jun", "lug": "Jul", "ago": "Aug",
           "set": "Sep", "ott": "Oct", "nov": "Nov", "dic": "Dec"}
    if "timestamp" in df.columns:
        ts = pd.to_numeric(df["timestamp"], errors="coerce")
        df["date"] = (pd.to_datetime(ts, unit="s", utc=True)
                      .dt.tz_localize(None).dt.normalize())
    else:
        def _parse(s):
            for it, en in _IT.items():
                s = s.replace(it, en)
            return pd.to_datetime(s, dayfirst=True, errors="coerce")
        df["date"] = df["data"].astype(str).apply(_parse)
    df["price"] = pd.to_numeric(df["chiusura"], errors="coerce")
    df = df.dropna(subset=["date", "price"]).sort_values("date").set_index("date")
    df = df[~df.index.duplicated(keep="first")]
    return usd_ton_to_eur_liter(df["price"], eurusd, EUROBOB_HC)


def load_data() -> pd.DataFrame:
    """Carica e assembla tutte e 4 le serie principali."""
    daily = (pd.read_csv(DAILY_CSV, parse_dates=["date"])
               .sort_values("date").set_index("date"))
    eurusd  = load_eurusd(
        csv_path=EURUSD_CSV if EURUSD_CSV.exists() else None,
        start="2015-01-01", end="2026-12-31",
    )
    gasoil  = _load_gasoil_futures(eurusd)
    eurobob = _load_eurobob_futures(eurusd)

    df = daily[["benzina_net", "gasolio_net"]].copy()
    df["margin_gasolio"] = df["gasolio_net"] - gasoil.reindex(df.index, method="ffill")
    if eurobob is not None:
        df["margin_benzina"] = df["benzina_net"] - eurobob.reindex(df.index, method="ffill")
    else:
        df["margin_benzina"] = np.nan
        warnings.warn("Eurobob futures non trovato — margin_benzina sarà NaN.")
    return df


# ══════════════════════════════════════════════════════════════════════════════
# Helpers statistici
# ══════════════════════════════════════════════════════════════════════════════

def _phi_ar1(x: np.ndarray) -> float:
    """Autocorrelazione lag-1 (stima AR(1)), clampata a (−0.99, 0.99)."""
    x = np.asarray(x, float)
    if len(x) < 3:
        return 0.0
    xc = x - x.mean()
    r  = float(np.corrcoef(xc[:-1], xc[1:])[0, 1])
    return float(np.clip(r, -0.99, 0.99))


def _n_eff(x: np.ndarray) -> float:
    """Dimensione effettiva campione corretta per autocorrelazione AR(1)."""
    phi = _phi_ar1(x)
    return max(2.0, len(x) * (1 - phi) / (1 + phi))


def _cohens_d(a: np.ndarray, b: np.ndarray) -> float:
    """Cohen d (pooled std, unequal n)."""
    na, nb = len(a), len(b)
    if na < 2 or nb < 2:
        return np.nan
    sp = np.sqrt(((na - 1) * a.var(ddof=1) + (nb - 1) * b.var(ddof=1)) / (na + nb - 2))
    if sp == 0:
        return np.nan
    return float((b.mean() - a.mean()) / sp)


def _hedges_g(a: np.ndarray, b: np.ndarray) -> float:
    """Hedge g (Cohen d con fattore di correzione per piccoli campioni)."""
    d = _cohens_d(a, b)
    if np.isnan(d):
        return np.nan
    n = len(a) + len(b)
    # fattore di correzione J (approssimazione)
    j = 1 - (3 / (4 * (n - 2) - 1))
    return float(d * j)


def _cliffs_delta(a: np.ndarray, b: np.ndarray) -> float:
    """
    Cliff δ: proporzione (post > pre) − proporzione (post < pre).
    Range [−1, 1]; interpretazione: |δ|<0.147 negligible, <0.33 small,
    <0.474 medium, ≥0.474 large.
    """
    a, b = np.asarray(a, float), np.asarray(b, float)
    dominance = np.sum(b[:, None] > a[None, :]) - np.sum(b[:, None] < a[None, :])
    return float(dominance / (len(a) * len(b)))


def _interpret_cliffs(delta: float) -> str:
    ad = abs(delta)
    if ad < 0.147:
        return "negligible"
    if ad < 0.330:
        return "small"
    if ad < 0.474:
        return "medium"
    return "large"


def _interpret_cohens_d(d: float) -> str:
    if np.isnan(d):
        return "N/A"
    ad = abs(d)
    if ad < 0.2:
        return "negligible"
    if ad < 0.5:
        return "small"
    if ad < 0.8:
        return "medium"
    return "large"


def _hurst_rs(ts: np.ndarray, min_chunk: int = 20) -> float:
    """
    Esponente di Hurst con metodo R/S (Rescaled Range).
    H ≈ 0.5  → random walk (no memoria)
    H > 0.5  → trend persistente (memoria lunga)
    H < 0.5  → mean-reverting
    """
    ts = np.asarray(ts, float)
    ts = ts[~np.isnan(ts)]
    n  = len(ts)
    if n < min_chunk * 2:
        return np.nan
    lags  = []
    rs_list = []
    for size in np.unique(np.logspace(np.log10(min_chunk), np.log10(n // 2), 20).astype(int)):
        chunks = [ts[i: i + size] for i in range(0, n - size + 1, size)]
        if len(chunks) < 2:
            continue
        rs_chunk = []
        for chunk in chunks:
            mean  = chunk.mean()
            dev   = chunk - mean
            cumdev = np.cumsum(dev)
            R     = cumdev.max() - cumdev.min()
            S     = chunk.std(ddof=1)
            if S > 0:
                rs_chunk.append(R / S)
        if rs_chunk:
            lags.append(size)
            rs_list.append(np.mean(rs_chunk))
    if len(lags) < 4:
        return np.nan
    log_lags = np.log(lags)
    log_rs   = np.log(rs_list)
    slope, *_ = np.polyfit(log_lags, log_rs, 1)
    return float(slope)


def _durbin_watson(resid: np.ndarray) -> float:
    """Statistica Durbin-Watson (valore atteso ≈ 2 se no autocorrelazione)."""
    resid = np.asarray(resid, float)
    resid = resid[~np.isnan(resid)]
    if len(resid) < 3:
        return np.nan
    diff = np.diff(resid)
    return float(np.sum(diff**2) / np.sum(resid**2))


def _dagostino_k2(x: np.ndarray):
    """D'Agostino K² test di normalità. Restituisce (stat, p)."""
    x = np.asarray(x, float)
    x = x[~np.isnan(x)]
    if len(x) < 8:
        return np.nan, np.nan
    try:
        s, p = stats.normaltest(x)
        return float(s), float(p)
    except Exception:
        return np.nan, np.nan


def _f_ratio_test(a: np.ndarray, b: np.ndarray):
    """
    F-test per uguaglianza varianze (assume normalità).
    H0: var(a) = var(b). Restituisce (F, p, df1, df2).
    """
    a, b = np.asarray(a, float), np.asarray(b, float)
    a, b = a[~np.isnan(a)], b[~np.isnan(b)]
    if len(a) < 2 or len(b) < 2:
        return np.nan, np.nan, np.nan, np.nan
    va, vb = a.var(ddof=1), b.var(ddof=1)
    if vb == 0:
        return np.nan, np.nan, np.nan, np.nan
    F    = va / vb
    df1  = len(a) - 1
    df2  = len(b) - 1
    # p bilaterale
    p    = 2 * min(stats.f.cdf(F, df1, df2), stats.f.sf(F, df1, df2))
    return float(F), float(p), int(df1), int(df2)


def _brown_forsythe(a: np.ndarray, b: np.ndarray):
    """
    Brown-Forsythe test (variante Levene con mediana invece della media).
    """
    a, b = np.asarray(a, float), np.asarray(b, float)
    a, b = a[~np.isnan(a)], b[~np.isnan(b)]
    if len(a) < 2 or len(b) < 2:
        return np.nan, np.nan
    try:
        stat, p = levene(a, b, center="median")
        return float(stat), float(p)
    except Exception:
        return np.nan, np.nan


# ══════════════════════════════════════════════════════════════════════════════
# A. Test sull'intera serie storica
# ══════════════════════════════════════════════════════════════════════════════

def test_global_series(series: pd.Series, name: str, alpha: float) -> dict:
    """
    Applica la batteria completa di test A all'intera serie storica.
    Restituisce un dizionario di risultati.
    """
    x = series.dropna().values
    n = len(x)

    row: dict = {
        "serie": name,
        "n_obs": n,
    }

    if n < 4:
        print(f"  [{name}] Troppo pochi dati ({n}) — salto.")
        return row

    # ── A1. Statistiche descrittive ──────────────────────────────────────────
    row.update({
        "mean":       float(np.mean(x)),
        "median":     float(np.median(x)),
        "std":        float(np.std(x, ddof=1)),
        "cv_pct":     float(np.std(x, ddof=1) / np.abs(np.mean(x)) * 100) if np.mean(x) != 0 else np.nan,
        "skewness":   float(stats.skew(x)),
        "kurtosis":   float(stats.kurtosis(x)),   # excess kurtosis
        "iqr":        float(np.percentile(x, 75) - np.percentile(x, 25)),
        "min":        float(np.min(x)),
        "max":        float(np.max(x)),
        "p5":         float(np.percentile(x, 5)),
        "p95":        float(np.percentile(x, 95)),
        "phi_ar1":    float(_phi_ar1(x)),
        "n_eff":      float(_n_eff(x)),
    })

    # ── A6. Hurst (qui già calcolabile) ─────────────────────────────────────
    h_rs = _hurst_rs(x)
    row["hurst_rs"]    = round(h_rs, 4) if not np.isnan(h_rs) else np.nan
    row["hurst_interp"] = (
        "random_walk" if (not np.isnan(h_rs) and 0.45 <= h_rs <= 0.55)
        else "trending/persistent" if (not np.isnan(h_rs) and h_rs > 0.55)
        else "mean_reverting" if (not np.isnan(h_rs) and h_rs < 0.45)
        else "N/A"
    )

    # ── A3. Normalità ────────────────────────────────────────────────────────
    jb_stat, jb_p = jarque_bera(x)
    row["jb_stat"]  = round(float(jb_stat), 4)
    row["jb_p"]     = round(float(jb_p), 4)
    row["jb_reject_h0"] = bool(jb_p < alpha)

    dag_stat, dag_p = _dagostino_k2(x)
    row["dag_stat"] = round(dag_stat, 4) if not np.isnan(dag_stat) else np.nan
    row["dag_p"]    = round(dag_p, 4)    if not np.isnan(dag_p)    else np.nan
    row["dag_reject_h0"] = bool(dag_p < alpha) if not np.isnan(dag_p) else None

    if n <= 200:
        sw_stat, sw_p = shapiro(x)
        row["sw_stat"] = round(float(sw_stat), 4)
        row["sw_p"]    = round(float(sw_p), 4)
        row["sw_reject_h0"] = bool(sw_p < alpha)
    else:
        row["sw_stat"]      = np.nan
        row["sw_p"]         = np.nan
        row["sw_reject_h0"] = None  # n troppo grande

    # ── A2. Stazionarietà ────────────────────────────────────────────────────
    if HAS_SM:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                adf_stat, adf_p, adf_lags, *_ = adfuller(x, autolag="AIC")
                row["adf_stat"]      = round(float(adf_stat), 4)
                row["adf_p"]         = round(float(adf_p), 4)
                row["adf_lags"]      = int(adf_lags)
                # ADF: H0 = radice unitaria (non stazionaria)
                # p < alpha → rifiuto H0 → serie stazionaria
                row["adf_stationary"] = bool(adf_p < alpha)
            except Exception as e:
                row["adf_stat"] = row["adf_p"] = row["adf_lags"] = np.nan
                row["adf_stationary"] = None

            for kpss_reg in ("c", "ct"):
                try:
                    kstat, kp, klags, _ = kpss_test(x, regression=kpss_reg, nlags="auto")
                    k = "level" if kpss_reg == "c" else "trend"
                    row[f"kpss_{k}_stat"]  = round(float(kstat), 4)
                    row[f"kpss_{k}_p"]     = round(float(kp), 4)
                    # KPSS: H0 = stazionarietà
                    # p < alpha → rifiuto H0 → serie NON stazionaria
                    row[f"kpss_{k}_nonstationary"] = bool(kp < alpha)
                except Exception:
                    k = "level" if kpss_reg == "c" else "trend"
                    row[f"kpss_{k}_stat"] = row[f"kpss_{k}_p"] = np.nan
                    row[f"kpss_{k}_nonstationary"] = None

        # Dual ADF-KPSS verdict
        adf_stat_bool  = row.get("adf_stationary")
        kpss_stat_bool = not row.get("kpss_level_nonstationary", True)  # True = stazionaria
        if adf_stat_bool is not None and kpss_stat_bool is not None:
            if adf_stat_bool and kpss_stat_bool:
                row["dual_verdict"] = "STAZIONARIA"
            elif not adf_stat_bool and not kpss_stat_bool:
                row["dual_verdict"] = "NON_STAZIONARIA"
            else:
                row["dual_verdict"] = "INCERTO"
        else:
            row["dual_verdict"] = "N/A"
    else:
        for k in ["adf_stat", "adf_p", "adf_lags", "adf_stationary",
                  "kpss_level_stat", "kpss_level_p", "kpss_level_nonstationary",
                  "kpss_trend_stat", "kpss_trend_p", "kpss_trend_nonstationary",
                  "dual_verdict"]:
            row[k] = np.nan

    # ── A4. Autocorrelazione ─────────────────────────────────────────────────
    # Durbin-Watson sui residui AR(1)
    x_dm  = x - x.mean()
    row["durbin_watson"] = round(_durbin_watson(x_dm), 4)

    if HAS_SM:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            for lag in [5, 10, 20]:
                if n <= lag + 5:
                    row[f"lb_lag{lag}_stat"] = np.nan
                    row[f"lb_lag{lag}_p"]    = np.nan
                    continue
                try:
                    lb = acorr_ljungbox(x, lags=[lag], return_df=True)
                    row[f"lb_lag{lag}_stat"] = round(float(lb["lb_stat"].iloc[-1]), 4)
                    row[f"lb_lag{lag}_p"]    = round(float(lb["lb_pvalue"].iloc[-1]), 4)
                except Exception:
                    row[f"lb_lag{lag}_stat"] = np.nan
                    row[f"lb_lag{lag}_p"]    = np.nan

    # ── A5. ARCH ─────────────────────────────────────────────────────────────
    if HAS_SM:
        for arch_lag in [5, 10]:
            if n > arch_lag + 5:
                try:
                    lm, lm_p, _, _ = het_arch(x - x.mean(), nlags=arch_lag)
                    row[f"arch_lag{arch_lag}_lm"]  = round(float(lm), 4)
                    row[f"arch_lag{arch_lag}_p"]   = round(float(lm_p), 4)
                    row[f"arch_lag{arch_lag}_reject"] = bool(lm_p < alpha)
                except Exception:
                    row[f"arch_lag{arch_lag}_lm"]    = np.nan
                    row[f"arch_lag{arch_lag}_p"]     = np.nan
                    row[f"arch_lag{arch_lag}_reject"] = None
            else:
                row[f"arch_lag{arch_lag}_lm"]    = np.nan
                row[f"arch_lag{arch_lag}_p"]     = np.nan
                row[f"arch_lag{arch_lag}_reject"] = None

    return row


# ══════════════════════════════════════════════════════════════════════════════
# B. Test sulle finestre evento (pre vs post)
# ══════════════════════════════════════════════════════════════════════════════

def test_window(
    series: pd.Series,
    series_name: str,
    event_name: str,
    shock: pd.Timestamp,
    pre_win: int,
    post_win: int,
    alpha: float,
) -> dict:
    """
    Confronto statistico pre-shock vs post-shock per una singola
    serie × evento. Restituisce un dizionario di risultati.
    """
    pre_data = series[
        (series.index >= shock - pd.Timedelta(days=pre_win)) &
        (series.index <  shock)
    ].dropna()
    post_data = series[
        (series.index >= shock) &
        (series.index <  shock + pd.Timedelta(days=post_win))
    ].dropna()

    row: dict = {
        "serie":      series_name,
        "evento":     event_name,
        "shock":      shock.date(),
        "pre_win":    pre_win,
        "post_win":   post_win,
        "n_pre":      len(pre_data),
        "n_post":     len(post_data),
        "pre_start":  pre_data.index.min().date() if len(pre_data) > 0 else None,
        "pre_end":    pre_data.index.max().date() if len(pre_data) > 0 else None,
        "post_start": post_data.index.min().date() if len(post_data) > 0 else None,
        "post_end":   post_data.index.max().date() if len(post_data) > 0 else None,
    }

    a = pre_data.values.astype(float)
    b = post_data.values.astype(float)

    if len(a) < 5 or len(b) < 5:
        row["skip_reason"] = "dati_insufficienti"
        return row

    # ── B1. Statistiche descrittive ──────────────────────────────────────────
    for tag, arr in [("pre", a), ("post", b)]:
        row[f"{tag}_mean"]   = round(float(arr.mean()), 5)
        row[f"{tag}_median"] = round(float(np.median(arr)), 5)
        row[f"{tag}_std"]    = round(float(arr.std(ddof=1)), 5)
        row[f"{tag}_skew"]   = round(float(stats.skew(arr)), 4)
        row[f"{tag}_kurt"]   = round(float(stats.kurtosis(arr)), 4)
        row[f"{tag}_phi_ar1"] = round(_phi_ar1(arr), 4)
        row[f"{tag}_n_eff"]   = round(_n_eff(arr), 1)

    row["delta_mean"]   = round(float(b.mean() - a.mean()), 5)
    row["delta_pct"]    = round(float((b.mean() - a.mean()) / abs(a.mean()) * 100)
                                if a.mean() != 0 else np.nan, 2)
    row["ratio_std"]    = round(float(b.std(ddof=1) / a.std(ddof=1))
                                if a.std(ddof=1) != 0 else np.nan, 4)

    # ── B2. Uguaglianza medie ────────────────────────────────────────────────
    # Welch t-test con n_eff (robusto ad autocorrelazione)
    na_eff, nb_eff = _n_eff(a), _n_eff(b)
    va, vb = a.var(ddof=1) / na_eff, b.var(ddof=1) / nb_eff
    se = np.sqrt(va + vb)
    if se > 0:
        t_neff = float((b.mean() - a.mean()) / se)
        df_neff = (va + vb)**2 / (va**2 / (na_eff - 1) + vb**2 / (nb_eff - 1))
        p_welch_neff = float(2 * stats.t.sf(abs(t_neff), df_neff))
        p_welch_neff_1s = float(stats.t.sf(t_neff, df_neff))  # H1: post > pre
    else:
        t_neff = p_welch_neff = p_welch_neff_1s = np.nan

    row["welch_neff_t"]      = round(t_neff, 4) if not np.isnan(t_neff) else np.nan
    row["welch_neff_p_2s"]   = round(p_welch_neff, 4) if not np.isnan(p_welch_neff) else np.nan
    row["welch_neff_p_1s"]   = round(p_welch_neff_1s, 4) if not np.isnan(p_welch_neff_1s) else np.nan
    row["welch_neff_reject"]  = bool(p_welch_neff_1s < alpha) if not np.isnan(p_welch_neff_1s) else None

    # Mann-Whitney U (one-sided: H1 = post > pre, alternativa "greater")
    try:
        mw_stat, mw_p = mannwhitneyu(b, a, alternative="greater")
        row["mw_stat"]   = round(float(mw_stat), 2)
        row["mw_p_1s"]   = round(float(mw_p), 4)
        row["mw_reject"] = bool(mw_p < alpha)
    except Exception:
        row["mw_stat"] = row["mw_p_1s"] = np.nan
        row["mw_reject"] = None

    # Wilcoxon signed-rank (richiede n uguale o diffs; usiamo interpolazione min)
    min_n = min(len(a), len(b))
    if min_n >= 10:
        try:
            # Allinea le due serie per lunghezza uguale (prendi le ultime min_n di a)
            a_trim = a[-min_n:]
            b_trim = b[:min_n]
            wx_stat, wx_p = wilcoxon(b_trim - a_trim, alternative="greater")
            row["wx_stat"]   = round(float(wx_stat), 2)
            row["wx_p_1s"]   = round(float(wx_p), 4)
            row["wx_reject"] = bool(wx_p < alpha)
        except Exception:
            row["wx_stat"] = row["wx_p_1s"] = np.nan
            row["wx_reject"] = None
    else:
        row["wx_stat"] = row["wx_p_1s"] = np.nan
        row["wx_reject"] = None

    # ── B3. Uguaglianza varianze ─────────────────────────────────────────────
    try:
        lev_stat, lev_p = levene(a, b)
        row["levene_stat"]   = round(float(lev_stat), 4)
        row["levene_p"]      = round(float(lev_p), 4)
        row["levene_reject"] = bool(lev_p < alpha)
    except Exception:
        row["levene_stat"] = row["levene_p"] = np.nan
        row["levene_reject"] = None

    try:
        bart_stat, bart_p = bartlett(a, b)
        row["bartlett_stat"]   = round(float(bart_stat), 4)
        row["bartlett_p"]      = round(float(bart_p), 4)
        row["bartlett_reject"] = bool(bart_p < alpha)
    except Exception:
        row["bartlett_stat"] = row["bartlett_p"] = np.nan
        row["bartlett_reject"] = None

    try:
        fk_stat, fk_p = fligner(a, b)
        row["fligner_stat"]   = round(float(fk_stat), 4)
        row["fligner_p"]      = round(float(fk_p), 4)
        row["fligner_reject"] = bool(fk_p < alpha)
    except Exception:
        row["fligner_stat"] = row["fligner_p"] = np.nan
        row["fligner_reject"] = None

    F_ratio, F_p, F_df1, F_df2 = _f_ratio_test(a, b)
    row["f_ratio"]         = round(F_ratio, 4) if not np.isnan(F_ratio) else np.nan
    row["f_ratio_p"]       = round(F_p, 4)     if not np.isnan(F_p)     else np.nan
    row["f_ratio_reject"]  = bool(F_p < alpha)  if not np.isnan(F_p)     else None

    bf_stat, bf_p = _brown_forsythe(a, b)
    row["bf_stat"]   = round(bf_stat, 4) if not np.isnan(bf_stat) else np.nan
    row["bf_p"]      = round(bf_p, 4)    if not np.isnan(bf_p)    else np.nan
    row["bf_reject"] = bool(bf_p < alpha) if not np.isnan(bf_p)   else None

    # ── B4. Uguaglianza distribuzione ────────────────────────────────────────
    try:
        ks_stat, ks_p = ks_2samp(a, b)
        row["ks_stat"]   = round(float(ks_stat), 4)
        row["ks_p"]      = round(float(ks_p), 4)
        row["ks_reject"] = bool(ks_p < alpha)
    except Exception:
        row["ks_stat"] = row["ks_p"] = np.nan
        row["ks_reject"] = None

    if HAS_ANDERSON_KSAMP:
        try:
            ad_res = anderson_ksamp([a, b], variant="midrank")
            row["ad_ksamp_stat"]   = round(float(ad_res.statistic), 4)
            row["ad_ksamp_p"]      = round(float(ad_res.pvalue), 4)
            row["ad_ksamp_reject"] = bool(ad_res.pvalue < alpha)
        except Exception:
            row["ad_ksamp_stat"] = row["ad_ksamp_p"] = np.nan
            row["ad_ksamp_reject"] = None

    # ── B5. Effect size ──────────────────────────────────────────────────────
    d  = _cohens_d(a, b)
    g  = _hedges_g(a, b)
    cd = _cliffs_delta(a, b)
    row["cohens_d"]      = round(d, 4)  if not np.isnan(d)  else np.nan
    row["cohens_d_interp"] = _interpret_cohens_d(d)
    row["hedges_g"]      = round(g, 4)  if not np.isnan(g)  else np.nan
    row["cliffs_delta"]  = round(cd, 4) if not np.isnan(cd) else np.nan
    row["cliffs_interp"] = _interpret_cliffs(cd) if not np.isnan(cd) else "N/A"

    # ── B6. Normalità intra-finestra ─────────────────────────────────────────
    for tag, arr in [("pre", a), ("post", b)]:
        if len(arr) >= 3:
            sw_s, sw_p = shapiro(arr)
            row[f"{tag}_sw_stat"] = round(float(sw_s), 4)
            row[f"{tag}_sw_p"]    = round(float(sw_p), 4)
            row[f"{tag}_normal"]  = bool(sw_p >= alpha)
        else:
            row[f"{tag}_sw_stat"] = row[f"{tag}_sw_p"] = np.nan
            row[f"{tag}_normal"]  = None
        jbs, jbp = jarque_bera(arr)
        row[f"{tag}_jb_p"] = round(float(jbp), 4)

    # ── B7. Stazionarietà intra-finestra ─────────────────────────────────────
    if HAS_SM:
        for tag, arr in [("pre", a), ("post", b)]:
            if len(arr) >= 10:
                try:
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        adf_s, adf_p, *_ = adfuller(arr, autolag="AIC")
                    row[f"{tag}_adf_stat"]       = round(float(adf_s), 4)
                    row[f"{tag}_adf_p"]          = round(float(adf_p), 4)
                    row[f"{tag}_adf_stationary"] = bool(adf_p < alpha)
                except Exception:
                    row[f"{tag}_adf_stat"] = row[f"{tag}_adf_p"] = np.nan
                    row[f"{tag}_adf_stationary"] = None
            else:
                row[f"{tag}_adf_stat"] = row[f"{tag}_adf_p"] = np.nan
                row[f"{tag}_adf_stationary"] = None

    # ── B8. Autocorrelazione intra-finestra ──────────────────────────────────
    if HAS_SM:
        for tag, arr in [("pre", a), ("post", b)]:
            if len(arr) >= 10:
                try:
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        lb = acorr_ljungbox(arr, lags=[5], return_df=True)
                    row[f"{tag}_lb5_stat"] = round(float(lb["lb_stat"].iloc[-1]), 4)
                    row[f"{tag}_lb5_p"]    = round(float(lb["lb_pvalue"].iloc[-1]), 4)
                    row[f"{tag}_autocorr"] = bool(lb["lb_pvalue"].iloc[-1] < alpha)
                except Exception:
                    row[f"{tag}_lb5_stat"] = row[f"{tag}_lb5_p"] = np.nan
                    row[f"{tag}_autocorr"] = None
            else:
                row[f"{tag}_lb5_stat"] = row[f"{tag}_lb5_p"] = np.nan
                row[f"{tag}_autocorr"] = None

    # ── Sintesi finale ───────────────────────────────────────────────────────
    # Verdetto "profitto anomalo": almeno 2 test su 3 rifiutano H0 (media post > pre)
    votes = [
        row.get("welch_neff_reject"),
        row.get("mw_reject"),
        row.get("wx_reject"),
    ]
    valid_votes = [v for v in votes if v is not None]
    row["n_mean_tests_reject"] = sum(valid_votes)
    row["n_mean_tests_valid"]  = len(valid_votes)
    row["verdict_mean_shift"]  = (
        "ANOMALO" if sum(valid_votes) >= 2
        else "NON_ANOMALO" if len(valid_votes) >= 2
        else "INDETERMINATO"
    )

    var_votes = [
        row.get("levene_reject"),
        row.get("fligner_reject"),
        row.get("bf_reject"),
    ]
    valid_var = [v for v in var_votes if v is not None]
    row["verdict_var_shift"] = (
        "VOLATILITA_AUMENTATA" if (sum(valid_var) >= 2 and b.std(ddof=1) > a.std(ddof=1))
        else "VOLATILITA_RIDOTTA"   if (sum(valid_var) >= 2 and b.std(ddof=1) < a.std(ddof=1))
        else "VOLATILITA_STABILE"   if len(valid_var) >= 2
        else "INDETERMINATO"
    )

    return row


# ══════════════════════════════════════════════════════════════════════════════
# C. Test cross-evento (3 eventi comparati insieme)
# ══════════════════════════════════════════════════════════════════════════════

def test_cross_event(
    series: pd.Series,
    series_name: str,
    pre_win: int,
    post_win: int,
    alpha: float,
    phase: str = "post",  # "pre" | "post" | "delta"
) -> dict:
    """
    Confronto k=3 campioni (uno per evento) sulla fase indicata.
    'delta' = post − media_pre (per controllare il livello assoluto).
    """
    groups = {}
    for ev_name, ev in EVENTS.items():
        shock = ev["shock"]
        pre = series[
            (series.index >= shock - pd.Timedelta(days=pre_win)) &
            (series.index <  shock)
        ].dropna().values

        post = series[
            (series.index >= shock) &
            (series.index <  shock + pd.Timedelta(days=post_win))
        ].dropna().values

        if phase == "pre":
            groups[ev_name] = pre
        elif phase == "post":
            groups[ev_name] = post
        else:  # delta
            if len(pre) > 0 and len(post) > 0:
                groups[ev_name] = post - np.mean(pre)
            else:
                groups[ev_name] = np.array([])

    # Filtra gruppi con almeno 5 obs
    valid = {k: v for k, v in groups.items() if len(v) >= 5}

    row: dict = {
        "serie":    series_name,
        "fase":     phase,
        "n_groups": len(valid),
        "events":   list(valid.keys()),
    }

    if len(valid) < 2:
        row["skip_reason"] = "meno_di_2_gruppi_validi"
        return row

    arrs = list(valid.values())

    # C1. Kruskal-Wallis
    try:
        kw_stat, kw_p = kruskal(*arrs)
        row["kw_stat"]   = round(float(kw_stat), 4)
        row["kw_p"]      = round(float(kw_p), 4)
        row["kw_reject"] = bool(kw_p < alpha)
    except Exception:
        row["kw_stat"] = row["kw_p"] = np.nan
        row["kw_reject"] = None

    # C2. ANOVA a una via
    try:
        f_stat, f_p = f_oneway(*arrs)
        row["anova_f"]      = round(float(f_stat), 4)
        row["anova_p"]      = round(float(f_p), 4)
        row["anova_reject"] = bool(f_p < alpha)
    except Exception:
        row["anova_f"] = row["anova_p"] = np.nan
        row["anova_reject"] = None

    # C3. Fligner-Killeen (varianze k-campioni)
    try:
        fk_stat, fk_p = fligner(*arrs)
        row["fligner_k_stat"]   = round(float(fk_stat), 4)
        row["fligner_k_p"]      = round(float(fk_p), 4)
        row["fligner_k_reject"] = bool(fk_p < alpha)
    except Exception:
        row["fligner_k_stat"] = row["fligner_k_p"] = np.nan
        row["fligner_k_reject"] = None

    # C4. Bartlett k-campioni
    try:
        bt_stat, bt_p = bartlett(*arrs)
        row["bartlett_k_stat"]   = round(float(bt_stat), 4)
        row["bartlett_k_p"]      = round(float(bt_p), 4)
        row["bartlett_k_reject"] = bool(bt_p < alpha)
    except Exception:
        row["bartlett_k_stat"] = row["bartlett_k_p"] = np.nan
        row["bartlett_k_reject"] = None

    return row


# ══════════════════════════════════════════════════════════════════════════════
# Plotting
# ══════════════════════════════════════════════════════════════════════════════

def _significance_stars(p: float | None) -> str:
    if p is None or (isinstance(p, float) and np.isnan(p)):
        return ""
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return "ns"


def plot_windows(
    data: pd.DataFrame,
    series_name: str,
    window_rows: list[dict],
    pre_win: int,
    post_win: int,
) -> None:
    """
    Per ogni serie: panel 3×2 con strip plot + box plot dei 3 eventi.
    """
    series = data[series_name].dropna()
    ev_names = list(EVENTS.keys())

    fig = plt.figure(figsize=(18, 5 * len(ev_names)), constrained_layout=True)
    fig.suptitle(
        f"Finestre pre/post shock  ·  {series_name}\n"
        f"(pre={pre_win}gg, post={post_win}gg)",
        fontsize=13, fontweight="bold",
    )

    gs_outer = gridspec.GridSpec(len(ev_names), 2, figure=fig, wspace=0.35, hspace=0.45)

    for i, ev_name in enumerate(ev_names):
        shock = EVENTS[ev_name]["shock"]
        color = EVENTS[ev_name]["color"]

        pre = series[
            (series.index >= shock - pd.Timedelta(days=pre_win)) &
            (series.index < shock)
        ].dropna()
        post = series[
            (series.index >= shock) &
            (series.index < shock + pd.Timedelta(days=post_win))
        ].dropna()

        if len(pre) == 0 or len(post) == 0:
            continue

        # Recupera p-value Mann-Whitney dal row corrispondente
        match_rows = [r for r in window_rows
                      if r.get("serie") == series_name and r.get("evento") == ev_name]
        mw_p = match_rows[0].get("mw_p_1s") if match_rows else None
        cliff = match_rows[0].get("cliffs_delta") if match_rows else None
        verdict = match_rows[0].get("verdict_mean_shift", "") if match_rows else ""

        # ── Pannello sinistra: serie temporale con shading ──────────────────
        ax_ts = fig.add_subplot(gs_outer[i, 0])
        context_start = shock - pd.Timedelta(days=pre_win + 10)
        context_end   = shock + pd.Timedelta(days=post_win + 10)
        ctx = series[(series.index >= context_start) & (series.index <= context_end)]

        ax_ts.plot(ctx.index, ctx.values, color="steelblue", lw=1.2, alpha=0.9)
        ax_ts.axvspan(pre.index.min(), pre.index.max(), alpha=0.12, color="#2196F3", label="pre")
        ax_ts.axvspan(post.index.min(), post.index.max(), alpha=0.12, color="#F44336", label="post")
        ax_ts.axvline(shock, color=color, lw=2, ls="--", label="shock")
        ax_ts.set_title(f"{EVENT_SHORT[ev_name]}  ·  {series_name}", fontsize=9, fontweight="bold")
        ax_ts.xaxis.set_major_formatter(mdates.DateFormatter("%b\n%Y"))
        ax_ts.xaxis.set_major_locator(mdates.MonthLocator())
        ax_ts.legend(fontsize=7, loc="upper left")
        ax_ts.set_ylabel("€/L", fontsize=8)
        ax_ts.tick_params(labelsize=7)

        # ── Pannello destra: box + strip plot ───────────────────────────────
        ax_bx = fig.add_subplot(gs_outer[i, 1])

        # Box plot
        bp = ax_bx.boxplot(
            [pre.values, post.values],
            positions=[0, 1],
            widths=0.35,
            patch_artist=True,
            medianprops=dict(color="black", lw=2),
            flierprops=dict(marker=".", markersize=3, alpha=0.5),
        )
        bp["boxes"][0].set_facecolor("#2196F3"); bp["boxes"][0].set_alpha(0.5)
        bp["boxes"][1].set_facecolor("#F44336"); bp["boxes"][1].set_alpha(0.5)

        # Strip jitter
        rng = np.random.default_rng(42)
        for pos, arr, c in [(0, pre.values, "#1565C0"), (1, post.values, "#B71C1C")]:
            jitter = rng.uniform(-0.08, 0.08, size=len(arr))
            ax_bx.scatter(pos + jitter, arr, s=8, alpha=0.4, color=c, zorder=3)

        # Annotazione stars
        y_max = max(np.nanmax(pre.values), np.nanmax(post.values))
        y_min = min(np.nanmin(pre.values), np.nanmin(post.values))
        y_range = y_max - y_min
        ax_bx.plot([0, 1], [y_max + 0.02 * y_range] * 2, "k-", lw=1)
        ax_bx.text(
            0.5, y_max + 0.04 * y_range,
            f"MW p={mw_p:.3f}" if mw_p is not None else "",
            ha="center", va="bottom", fontsize=9, fontweight="bold"
        )
        cd_str = f"  Cliff δ={cliff:+.3f} ({match_rows[0].get('cliffs_interp', '')})" if cliff is not None else ""
        ax_bx.set_title(
            f"Pre vs Post  ·  {verdict}{cd_str}",
            fontsize=8, fontweight="bold",
        )
        ax_bx.set_xticks([0, 1])
        ax_bx.set_xticklabels(["Pre", "Post"], fontsize=9)
        ax_bx.set_ylabel("€/L", fontsize=8)
        ax_bx.tick_params(labelsize=7)

    out = OUT_DIR / f"plot_windows_{series_name}.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  → Plot: {out}")


def plot_hurst_rolling(data: pd.DataFrame, series_to_plot: list[str], window: int = 250) -> None:
    """
    Rolling Hurst exponent (R/S) con finestra mobile di `window` giorni.
    """
    fig, axes = plt.subplots(len(series_to_plot), 1,
                             figsize=(14, 3.5 * len(series_to_plot)),
                             sharex=True, constrained_layout=True)
    if len(series_to_plot) == 1:
        axes = [axes]

    colors = ["#E63946", "#1D3557", "#2a9d8f", "#e9c46a"]

    for ax, sname, c in zip(axes, series_to_plot, colors):
        s = data[sname].dropna()
        dates_h, hursts = [], []
        step = 20  # calcola ogni 20 giorni (velocità)
        idx_arr = s.index.to_numpy()
        val_arr = s.values

        for i in range(window, len(val_arr), step):
            chunk = val_arr[i - window: i]
            h = _hurst_rs(chunk, min_chunk=20)
            if not np.isnan(h):
                dates_h.append(idx_arr[i])
                hursts.append(h)

        if not hursts:
            ax.set_title(f"{sname} — Hurst N/D")
            continue

        ax.plot(dates_h, hursts, color=c, lw=1.5, alpha=0.85, label=sname)
        ax.axhline(0.5, color="grey", ls="--", lw=1, alpha=0.6, label="H=0.5 (random walk)")
        ax.axhline(0.55, color="#F4A261", ls=":", lw=1, alpha=0.7)
        ax.axhline(0.45, color="#457B9D", ls=":", lw=1, alpha=0.7)
        ax.fill_between(dates_h, 0.45, 0.55, alpha=0.07, color="grey")

        for ev_name, ev in EVENTS.items():
            ax.axvline(ev["shock"], color=ev["color"], lw=1.5, ls="--", alpha=0.7)
            ax.text(ev["shock"], ax.get_ylim()[1] if ax.get_ylim()[1] != 1 else 0.95,
                    EVENT_SHORT[ev_name].replace("\n", " "), rotation=90,
                    fontsize=6, va="top", ha="right", color=ev["color"])

        ax.set_ylabel("Hurst H", fontsize=8)
        ax.set_ylim(0.25, 0.85)
        ax.legend(fontsize=8, loc="upper left")
        ax.tick_params(labelsize=7)
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
        ax.xaxis.set_major_locator(mdates.YearLocator())

    axes[0].set_title(
        f"Hurst Esponente (R/S)  ·  rolling window={window}gg\n"
        f"H>0.55 trend persistente  |  H<0.45 mean-reverting  |  H≈0.5 random walk",
        fontsize=10, fontweight="bold",
    )
    out = OUT_DIR / "plot_hurst_rolling.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  → Plot Hurst: {out}")


# ══════════════════════════════════════════════════════════════════════════════
# Console summary
# ══════════════════════════════════════════════════════════════════════════════

def _pstar(p) -> str:
    """p-value formattato con stars."""
    if p is None or (isinstance(p, float) and np.isnan(p)):
        return "N/D   "
    stars = _significance_stars(p)
    return f"{p:.4f}{stars:<3}"


def print_global_summary(rows: list[dict], alpha: float) -> None:
    print(f"\n{SEP}")
    print("  A. TEST INTERA SERIE STORICA")
    print(f"{SEP}")
    headers = ["Serie", "N", "Media", "Std", "Skew", "Kurt", "Hurst",
               "ADF-p", "KPSS-p", "JB-p", "LB10-p", "ARCH5-p"]
    fmt = "  {:<18} {:>6} {:>7} {:>6} {:>6} {:>6} {:>6}  {:>8} {:>8} {:>8} {:>8} {:>8}"
    print(fmt.format(*headers))
    print("  " + "─" * 100)
    for r in rows:
        print(fmt.format(
            r.get("serie", "?"),
            int(r.get("n_obs", 0)),
            f"{r.get('mean', np.nan):.4f}",
            f"{r.get('std', np.nan):.4f}",
            f"{r.get('skewness', np.nan):+.2f}",
            f"{r.get('kurtosis', np.nan):+.2f}",
            f"{r.get('hurst_rs', np.nan):.3f}" if not _is_nan(r.get("hurst_rs")) else "N/D",
            _pstar(r.get("adf_p")),
            _pstar(r.get("kpss_level_p")),
            _pstar(r.get("jb_p")),
            _pstar(r.get("lb_lag10_p")),
            _pstar(r.get("arch_lag5_p")),
        ))
    print(f"\n  α={alpha}  |  *** p<0.001  ** p<0.01  * p<0.05  ns p≥0.05")
    print(f"  ADF: p<α → stazionaria  |  KPSS: p<α → NON stazionaria")


def print_window_summary(rows: list[dict], alpha: float) -> None:
    print(f"\n{SEP}")
    print("  B. TEST FINESTRE PRE/POST EVENTO")
    print(f"{SEP}")
    fmt = "  {:<22} {:<26} {:>7} {:>7} {:>8} {:>8} {:>8} {:>7} {:>12} {:>15}"
    print(fmt.format(
        "Serie", "Evento", "Δmedia", "Δ%",
        "Welch-p", "MW-p", "KS-p",
        "Cliff-δ", "Cliff-interp", "Verdict"
    ))
    print("  " + "─" * 120)
    for r in rows:
        if "skip_reason" in r:
            continue
        print(fmt.format(
            str(r.get("serie", "?"))[:22],
            str(r.get("evento", "?"))[:26],
            f"{r.get('delta_mean', np.nan):+.4f}",
            f"{r.get('delta_pct', np.nan):+.1f}",
            _pstar(r.get("welch_neff_p_1s")),
            _pstar(r.get("mw_p_1s")),
            _pstar(r.get("ks_p")),
            f"{r.get('cliffs_delta', np.nan):+.3f}" if not _is_nan(r.get("cliffs_delta")) else "N/D",
            str(r.get("cliffs_interp", "?"))[:12],
            str(r.get("verdict_mean_shift", "?"))[:15],
        ))


def print_variance_summary(rows: list[dict]) -> None:
    print(f"\n{SEP}")
    print("  B3. UGUAGLIANZA VARIANZE (sintesi)")
    print(f"{SEP}")
    fmt  = "  {:<22} {:<26} {:>6} {:>8} {:>8} {:>8} {:>8} {:>20}"
    print(fmt.format("Serie", "Evento", "σ-ratio", "Levene-p", "Fligner-p", "BF-p", "F-ratio-p", "Verdict-var"))
    print("  " + "─" * 110)
    for r in rows:
        if "skip_reason" in r:
            continue
        print(fmt.format(
            str(r.get("serie", "?"))[:22],
            str(r.get("evento", "?"))[:26],
            f"{r.get('ratio_std', np.nan):.3f}" if not _is_nan(r.get("ratio_std")) else "N/D",
            _pstar(r.get("levene_p")),
            _pstar(r.get("fligner_p")),
            _pstar(r.get("bf_p")),
            _pstar(r.get("f_ratio_p")),
            str(r.get("verdict_var_shift", "?"))[:20],
        ))


def _is_nan(v) -> bool:
    try:
        return v is None or np.isnan(float(v))
    except (TypeError, ValueError):
        return True


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Batteria test statistici — serie storiche carburanti IT",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Esempi:
  python3 02e_statistical_tests.py
  python3 02e_statistical_tests.py --pre-win 60 --post-win 60
  python3 02e_statistical_tests.py --alpha 0.01
  python3 02e_statistical_tests.py --series margin_benzina margin_gasolio
  python3 02e_statistical_tests.py --no-plots
""",
    )
    p.add_argument("--pre-win",  type=int, default=PRE_WIN_DEFAULT,
                   help=f"Giorni finestra pre-shock [default={PRE_WIN_DEFAULT}]")
    p.add_argument("--post-win", type=int, default=POST_WIN_DEFAULT,
                   help=f"Giorni finestra post-shock [default={POST_WIN_DEFAULT}]")
    p.add_argument("--alpha",    type=float, default=ALPHA_DEFAULT,
                   help=f"Livello α [default={ALPHA_DEFAULT}]")
    p.add_argument("--series",   nargs="+", default=ALL_SERIES,
                   choices=ALL_SERIES,
                   help="Serie da testare (default: tutte e 4)")
    p.add_argument("--no-plots", action="store_true",
                   help="Salta la generazione dei plot")
    p.add_argument("--hurst-window", type=int, default=250,
                   help="Finestra rolling per Hurst [default=250]")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    pre_win  = args.pre_win
    post_win = args.post_win
    alpha    = args.alpha
    series_list = args.series

    print(f"\n{SEP}")
    print("  02e_statistical_tests.py  —  Batteria test statistici")
    print(f"  Finestra: pre={pre_win}gg / post={post_win}gg  |  α={alpha}")
    print(f"  Serie: {series_list}")
    print(f"  Output: {OUT_DIR}")
    print(f"  statsmodels: {'OK' if HAS_SM else '✗ mancante'}")
    print(f"{SEP}\n")

    # ── Caricamento dati ──────────────────────────────────────────────────────
    print("Caricamento dati...")
    data = load_data()
    print(f"  Serie caricate: {list(data.columns)}")
    print(f"  Range: {data.index.min().date()} → {data.index.max().date()}")
    print(f"  N osservazioni: {len(data)}\n")

    # ══════════════════════════════════════════════════════════════════════════
    # A. Test globali
    # ══════════════════════════════════════════════════════════════════════════
    print(f"{SEP}")
    print("  A. TEST INTERA SERIE STORICA")
    print(f"{SEP}")

    global_rows: list[dict] = []
    for sname in series_list:
        if sname not in data.columns:
            print(f"  ⚠ Serie '{sname}' non trovata.")
            continue
        print(f"  Testing {sname}...")
        row = test_global_series(data[sname], sname, alpha)
        global_rows.append(row)

    print_global_summary(global_rows, alpha)

    df_global = pd.DataFrame(global_rows)
    out_global = OUT_DIR / "stat_tests_global.csv"
    df_global.to_csv(out_global, index=False)
    print(f"\n  → CSV globale: {out_global}")

    # ══════════════════════════════════════════════════════════════════════════
    # B. Test finestre
    # ══════════════════════════════════════════════════════════════════════════
    print(f"\n{SEP}")
    print("  B. TEST FINESTRE PRE/POST EVENTO")
    print(f"{SEP}")

    window_rows: list[dict] = []
    for sname in series_list:
        if sname not in data.columns:
            continue
        for ev_name, ev in EVENTS.items():
            print(f"  Testing {sname} × {ev_name}...")
            row = test_window(
                data[sname], sname, ev_name, ev["shock"],
                pre_win, post_win, alpha,
            )
            window_rows.append(row)

    print_window_summary(window_rows, alpha)
    print_variance_summary(window_rows)

    df_windows = pd.DataFrame(window_rows)
    out_windows = OUT_DIR / "stat_tests_windows.csv"
    df_windows.to_csv(out_windows, index=False)
    print(f"\n  → CSV finestre: {out_windows}")

    # ══════════════════════════════════════════════════════════════════════════
    # C. Test cross-evento
    # ══════════════════════════════════════════════════════════════════════════
    print(f"\n{SEP}")
    print("  C. CONFRONTO CROSS-EVENTO (k=3 eventi)")
    print(f"{SEP}")

    cross_rows: list[dict] = []
    for sname in series_list:
        if sname not in data.columns:
            continue
        for phase in ["pre", "post", "delta"]:
            row = test_cross_event(data[sname], sname, pre_win, post_win, alpha, phase)
            cross_rows.append(row)

    # Stampa sintesi cross-evento
    fmt = "  {:<22} {:<6} {:>10} {:>10} {:>14} {:>14}"
    print(fmt.format("Serie", "Fase", "KW-p", "ANOVA-p", "Fligner-k-p", "Bartlett-k-p"))
    print("  " + "─" * 80)
    for r in cross_rows:
        if "skip_reason" in r:
            continue
        print(fmt.format(
            r.get("serie", "?"),
            r.get("fase", "?"),
            _pstar(r.get("kw_p")),
            _pstar(r.get("anova_p")),
            _pstar(r.get("fligner_k_p")),
            _pstar(r.get("bartlett_k_p")),
        ))

    df_cross = pd.DataFrame(cross_rows)
    out_cross = OUT_DIR / "stat_tests_crossevent.csv"
    df_cross.to_csv(out_cross, index=False)
    print(f"\n  → CSV cross-evento: {out_cross}")

    # ══════════════════════════════════════════════════════════════════════════
    # Plot
    # ══════════════════════════════════════════════════════════════════════════
    if not args.no_plots:
        print(f"\n{SEP}")
        print("  GENERAZIONE PLOT")
        print(f"{SEP}")
        for sname in series_list:
            if sname not in data.columns:
                continue
            print(f"  Plot finestre: {sname}...")
            plot_windows(data, sname, window_rows, pre_win, post_win)

        print("  Plot Hurst rolling...")
        plot_hurst_rolling(data, [s for s in series_list if s in data.columns],
                           window=args.hurst_window)

    # ══════════════════════════════════════════════════════════════════════════
    # Riepilogo finale
    # ══════════════════════════════════════════════════════════════════════════
    print(f"\n{SEP}")
    print("  RIEPILOGO VERDETTI FINALI")
    print(f"{SEP}")

    for r in window_rows:
        if "skip_reason" in r:
            continue
        vm  = r.get("verdict_mean_shift", "?")
        vv  = r.get("verdict_var_shift", "?")
        cd  = r.get("cliffs_delta")
        cd_s = f"{cd:+.3f}" if not _is_nan(cd) else "N/D"
        icon_m = "🔴" if vm == "ANOMALO" else ("🟡" if vm == "INDETERMINATO" else "🟢")
        icon_v = "⚠" if "AUMENTATA" in str(vv) else "✓"
        print(
            f"  {icon_m} {r['serie']:<22} {r['evento']:<28}"
            f"  Media: {vm:<14}  Var: {vv:<22}  Cliff δ={cd_s}  ({r.get('cliffs_interp','?')})"
        )

    print(f"\n  Output → {OUT_DIR}")
    print(f"    stat_tests_global.csv     — test serie complete")
    print(f"    stat_tests_windows.csv    — test finestre pre/post")
    print(f"    stat_tests_crossevent.csv — confronto k=3 eventi")
    print(f"    plot_windows_*.png        — strip + box per ogni serie")
    print(f"    plot_hurst_rolling.png    — Hurst rolling")
    print(f"{SEP}\n")


if __name__ == "__main__":
    main()