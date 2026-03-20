#!/usr/bin/env python3
"""
02d_v3_arima.py  ─  Metodo 3: ITS / ARIMA (come il paper)
=============================================================
Implementazione Python dell'approccio Interrupted Time Series (ITS) con
modello ARIMA, generalizzando il metodo di Masena & Shongwe (2024)
dalla stima delle perdite alla stima dei guadagni speculativi.

Pipeline:
  i.   Identifica T0 (inizio intervento)
  ii.  Stima il modello controfattuale sulla serie pre-intervento
  iii. Produce previsioni controfattuali per il periodo post-intervento
       con 3 varianti, tutte pre-only (zero look-ahead):
         A  Auto-ARIMA AIC      — ARIMA(p,d,q) min. AIC sulla pre
         B  Holt-Winters ETS    — trend addit. estrapolato dalla pre
         C  OLS trend lineare   — regressione y=a+b·t sulla pre, estesa
  iv.  Calcola l'effetto intervento: margine_effettivo − controfattuale
  v.   Seleziona il miglior approccio per RMSE

Modalità (--mode):
  fixed     : T0 = data dello shock hardcodata [default]
  detected  : T0 rilevato via PELT (ruptures, modello RBF)
              È il metodo più sofisticato di detection: PELT (Pruned Exact
              Linear Time) trova il numero ottimale di break con penalità
              bayesiana — più potente del naive (v1) e del t-test (v2).
              Tra i break nel range ±SEARCH, viene scelto il PRIMO
              in ordine temporale (inizio ottimale della rottura).

Richiede: statsmodels>=0.14  ruptures>=1.1.8  scipy>=1.7

Parametro --detect (solo quando --mode detected):
  margin  : detection sul margine distributore           [default]
  price   : detection sul prezzo alla pompa netto (€/L)

Output:
  data/plots/its/{mode}/v3_arima/              (se mode=fixed)
  data/plots/its/detected/{detect}/v3_arima/   (se mode=detected)
    plot_{evento}_{carburante}.png
    v3_arima_results.csv
"""

from __future__ import annotations
from pathlib import Path
import argparse
import itertools
import warnings
import sys

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).parent / "utils"))
from conversions import GAS_OIL, EUROBOB as EUROBOB_HC, load_eurusd, usd_ton_to_eur_liter
from diagnostics import (
    run_diagnostic_tests,
    fit_sarima_benchmark,
    plot_residual_diagnostics,
    plot_sarima_diagnostics,
)
from theta_loader import load_theta
from utils.forecast_consumi import load_daily_consumption   # <-- nuova importazione

try:
    from statsmodels.tsa.arima.model import ARIMA
    from statsmodels.tsa.statespace.sarimax import SARIMAX
    from statsmodels.tsa.holtwinters import ExponentialSmoothing
    from statsmodels.stats.stattools import durbin_watson
    HAS_SM = True
except ImportError:
    HAS_SM = False
    warnings.warn("statsmodels non installato. pip install statsmodels")

try:
    import ruptures as rpt
    HAS_RPT = True
except ImportError:
    HAS_RPT = False
    warnings.warn("ruptures non installato (pip install ruptures). PELT non disponibile.")

# ── Configurazione ─────────────────────────────────────────────────────────────
BASE_DIR    = Path(__file__).parent
DAILY_CSV   = BASE_DIR / "data" / "processed" / "daily_fuel_prices_all.csv"
GASOIL_CSV  = BASE_DIR / "data" / "Futures" / "London Gas Oil Futures Historical Data.csv"
EUROBOB_CSV = BASE_DIR / "data" / "Futures" / "Eurobob_B7H1_date.csv"
EURUSD_CSV  = BASE_DIR / "data" / "raw" / "eurusd.csv"
_OUT_BASE   = BASE_DIR / "data" / "plots" / "its"
# v3 usa PELT (ruptures RBF) come detection autonoma — non legge theta_results.csv

PRE_WIN   = 40    # giorni pre-T0 per il fit ARIMA
POST_WIN  = 40    # giorni post-T0 per il counterfactual
CI_ALPHA  = 0.05  # CI al 90%
SEARCH    = 30    # ricerca PELT ±SEARCH giorni dallo shock (mode=detected)

# Griglia auto-ARIMA
ARIMA_P  = [0, 1, 2]
ARIMA_D  = [0, 1]
ARIMA_Q  = [0, 1, 2]
MAX_ITER = 200

# DAILY_CONSUMPTION_L rimosso – ora i consumi sono letti dal file CSV giornaliero

EVENTS: dict[str, dict] = {
    "Ucraina (Feb 2022)": {
        "shock": pd.Timestamp("2022-02-24"),
        "color": "#e74c3c",
        "label": "Russia-Ucraina (24 feb 2022)",
    },
    "Iran-Israele (Giu 2025)": {
        "shock": pd.Timestamp("2025-06-13"),
        "color": "#e67e22",
        "label": "Iran-Israele (13 giu 2025)",
    },
    "Hormuz (Feb 2026)": {
        "shock": pd.Timestamp("2026-02-28"),
        "color": "#8e44ad",
        "label": "Stretto di Hormuz (28 feb 2026)",
    },
}

FUELS: dict[str, tuple[str, str]] = {
    "benzina": ("margin_benzina", "#E63946"),
    "gasolio": ("margin_gasolio", "#1D3557"),
}


# ══════════════════════════════════════════════════════════════════════════════
# Caricamento dati
# ══════════════════════════════════════════════════════════════════════════════

def _load_gasoil_futures(eurusd: pd.Series) -> pd.Series:
    df = pd.read_csv(GASOIL_CSV, encoding="utf-8-sig", dtype=str)
    df["date"]  = pd.to_datetime(df["Date"], format="%m/%d/%Y", errors="coerce")
    df["price"] = (df["Price"].str.replace(",", "", regex=False)
                   .pipe(pd.to_numeric, errors="coerce"))
    return (df.dropna(subset=["date","price"]).sort_values("date")
              .set_index("date")
              .pipe(lambda d: usd_ton_to_eur_liter(d["price"], eurusd, GAS_OIL)))


def _load_eurobob_futures(eurusd: pd.Series) -> pd.Series | None:
    if not EUROBOB_CSV.exists():
        return None
    df = pd.read_csv(EUROBOB_CSV, encoding="utf-8-sig", dtype=str)
    _IT = {"gen":"Jan","feb":"Feb","mar":"Mar","apr":"Apr","mag":"May","giu":"Jun",
           "lug":"Jul","ago":"Aug","set":"Sep","ott":"Oct","nov":"Nov","dic":"Dec"}
    if "timestamp" in df.columns:
        ts = pd.to_numeric(df["timestamp"], errors="coerce")
        df["date"] = pd.to_datetime(ts, unit="s", utc=True).dt.tz_localize(None).dt.normalize()
    else:
        def _parse(s):
            for it, en in _IT.items():
                s = s.replace(it, en)
            return pd.to_datetime(s, dayfirst=True, errors="coerce")
        df["date"] = df["data"].astype(str).apply(_parse)
    df["price"] = pd.to_numeric(df["chiusura"], errors="coerce")
    df = df.dropna(subset=["date","price"]).sort_values("date").set_index("date")
    df = df[~df.index.duplicated(keep="first")]
    return usd_ton_to_eur_liter(df["price"], eurusd, EUROBOB_HC)


def load_margin_data() -> pd.DataFrame:
    daily  = (pd.read_csv(DAILY_CSV, parse_dates=["date"])
               .sort_values("date").set_index("date"))
    eurusd = load_eurusd(csv_path=EURUSD_CSV if EURUSD_CSV.exists() else None,
                         start="2015-01-01", end="2026-12-31")
    gasoil  = _load_gasoil_futures(eurusd)
    eurobob = _load_eurobob_futures(eurusd)
    df = daily[["benzina_net","gasolio_net"]].copy()
    df["margin_gasolio"] = df["gasolio_net"] - gasoil.reindex(df.index, method="ffill")
    df["margin_benzina"] = (df["benzina_net"] - eurobob.reindex(df.index, method="ffill")
                            if eurobob is not None else np.nan)
    return df


# ══════════════════════════════════════════════════════════════════════════════
# Break point detection — PELT (mode=detected)
# ══════════════════════════════════════════════════════════════════════════════

def detect_breakpoint_pelt(series: pd.Series, shock: pd.Timestamp) -> dict:
    """
    PELT (Pruned Exact Linear Time) con modello RBF — il metodo più sofisticato
    dei 3 implementati nella pipeline.

    PELT trova il numero e le posizioni ottimali dei break minimizzando un
    criterio di costo con penalità (qui: pen=2.0 su modello RBF).  È globalmente
    ottimale a differenza dei metodi sliding window (v1 naive, v2 t-test).

    Tra tutti i break trovati nel range [shock−SEARCH, shock+SEARCH], seleziona
    il PRIMO in ordine temporale (il più antico / più a sinistra), che
    rappresenta l'inizio ottimale della rottura strutturale.

    Fallback → shock date se ruptures non è installato o PELT non trova break.
    """
    if not HAS_RPT:
        warnings.warn("ruptures non installato. Fallback alla shock date.")
        return {"tau": shock, "method": "pelt_fallback_no_ruptures",
                "n_breaks_total": 0, "_window_series": pd.Series(dtype=float),
                "_all_breaks": [], "_filtered_breaks": []}

    # Usa la finestra allargata per dare contesto a PELT
    search_start = shock - pd.Timedelta(days=SEARCH)
    search_end   = shock + pd.Timedelta(days=SEARCH)
    window_series = series[
        (series.index >= search_start - pd.Timedelta(days=20)) &
        (series.index <= search_end   + pd.Timedelta(days=20))
    ].dropna()

    if len(window_series) < 20:
        return {"tau": shock, "method": "pelt_insufficient_data", "n_breaks_total": 0,
                "_window_series": window_series, "_all_breaks": [], "_filtered_breaks": []}

    try:
        arr  = window_series.values.astype(float).reshape(-1, 1)
        algo = rpt.Pelt(model="rbf").fit(arr)
        bps  = algo.predict(pen=2.0)   # pen=2: bilanciamento sensitività/FP
        # bps è lista di indici 1-based; l'ultimo è sempre len(arr)
        bp_indices  = [bp - 1 for bp in bps[:-1]]
        break_dates = [window_series.index[i] for i in bp_indices
                       if 0 <= i < len(window_series)]
    except Exception as e:
        warnings.warn(f"PELT fallito: {e}. Fallback shock date.")
        return {"tau": shock, "method": "pelt_error_fallback", "n_breaks_total": 0,
                "_window_series": window_series, "_all_breaks": [], "_filtered_breaks": []}

    # Filtra nel range di ricerca
    filtered = [d for d in break_dates if search_start <= d <= search_end]
    n_total  = len(break_dates)

    if not filtered:
        return {"tau": shock, "method": "pelt_nofound_fallback",
                "n_breaks_total": n_total,
                "_window_series": window_series,
                "_all_breaks": break_dates,
                "_filtered_breaks": []}

    # Scegli il PRIMO break in ordine temporale tra quelli nel range
    best = min(filtered)
    return {
        "tau":              best,
        "method":           "pelt_rbf",
        "n_breaks_total":   n_total,
        "all_filtered":     [str(d.date()) for d in filtered],
        "_window_series":   window_series,   # serie usata da PELT (per il plot)
        "_all_breaks":      break_dates,     # tutti i break trovati
        "_filtered_breaks": filtered,        # break nel range [shock±SEARCH]
    }


# ══════════════════════════════════════════════════════════════════════════════
# Auto-ARIMA — varianti pre-only (nessuna fuga di dati futuri)
# ══════════════════════════════════════════════════════════════════════════════

def _fit_arima_criterion(series: pd.Series, criterion: str = "aic",
                         force_d: int | None = None) -> tuple[tuple, object] | tuple[None, None]:
    """
    Seleziona ARIMA(p,d,q) su `series` minimizzando `criterion` (aic o bic).
    Se force_d è impostato, fissa d a quel valore (es. force_d=1 → sempre I(1)).
    Tutti i fit usano solo i dati passati come `series` — zero look-ahead.
    """
    if not HAS_SM:
        return None, None

    best_val   = np.inf
    best_order = None
    best_fit   = None
    d_values   = [force_d] if force_d is not None else ARIMA_D

    for p, d, q in itertools.product(ARIMA_P, d_values, ARIMA_Q):
        if p + q == 0:
            continue
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                m = ARIMA(series, order=(p, d, q))
                f = m.fit(method_kwargs={"maxiter": MAX_ITER, "warn_convergence": False})
            val = f.aic if criterion == "aic" else f.bic
            if val < best_val:
                best_val, best_order, best_fit = val, (p, d, q), f
        except Exception:
            continue

    return best_order, best_fit


def auto_arima(series: pd.Series) -> tuple[tuple, object] | tuple[None, None]:
    """Auto-ARIMA AIC — usato per il fit diagnostico pre-periodo."""
    return _fit_arima_criterion(series, criterion="aic")


def _arima_metrics(actual: np.ndarray, forecast: np.ndarray) -> dict:
    resid = actual - forecast
    rmse  = float(np.sqrt(np.mean(resid**2)))
    mask  = actual != 0
    mape  = float(np.mean(np.abs(resid[mask] / actual[mask])) * 100) if mask.any() else np.nan
    return {"rmse": rmse, "mape": mape}


def _arima_forecast(pre_series: pd.Series, order: tuple,
                    n_steps: int, alpha: float = CI_ALPHA
                    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Fit ARIMA(order) su pre_series, proietta n_steps in avanti.
    Vero controfattuale out-of-sample: il modello non vede mai i dati post.
    """
    if not HAS_SM:
        m   = float(pre_series.mean())
        std = float(pre_series.std())
        z   = stats.norm.ppf(1 - alpha / 2)
        return (np.full(n_steps, m),
                np.full(n_steps, m - z * std),
                np.full(n_steps, m + z * std))

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fit = ARIMA(pre_series, order=order).fit(
            method_kwargs={"maxiter": MAX_ITER, "warn_convergence": False}
        )
        fc  = fit.get_forecast(steps=n_steps)

    mean = fc.predicted_mean.values
    conf = fc.conf_int(alpha=alpha).values
    return mean, conf[:, 0], conf[:, 1]


def _make_approach(label: str, pre: pd.Series, post: pd.Series,
                   order: tuple, cons: pd.Series, aic: float, bic: float) -> dict:
    """Helper comune: fit pre-only, forecast post, calcola metriche con consumo giornaliero."""
    fc, ci_lo, ci_hi = _arima_forecast(pre, order, len(post))
    extra = post.values - fc
    # cons è Serie con stesso index di post (o allineabile)
    gain = float((extra * cons.values).sum() / 1e6)
    m = _arima_metrics(post.values, fc)
    return {
        "label": label,
        "forecast": fc, "ci_lo": ci_lo, "ci_hi": ci_hi, "extra": extra,
        "gain_meur": gain,
        "ci_gain_lo": float(((post.values - ci_hi) * cons.values).sum() / 1e6),
        "ci_gain_hi": float(((post.values - ci_lo) * cons.values).sum() / 1e6),
        "omega": np.nan, "aic": aic, "bic": bic, **m,
    }


# ══════════════════════════════════════════════════════════════════════════════
# I 3 approcci — tutti pre-only, zero look-ahead, visivamente distinti
# ══════════════════════════════════════════════════════════════════════════════
#
#  A  Auto-ARIMA AIC      →  ARIMA(p,d,q) min. AIC sul pre
#                             Ipotesi: serie stazionaria/AR a bassa memoria.
#                             Forecast che converge alla media pre-periodo.
#
#  B  Holt-Winters ETS    →  ExponentialSmoothing con trend additivo sul pre.
#                             Ipotesi: la serie aveva un trend determinato nel
#                             pre-periodo; lo estrapoliamo nel post.
#                             Produce un controfattuale inclinato/curvo.
#
#  C  OLS trend lineare   →  y = a + b·t fit sul pre, esteso al post.
#                             Ipotesi: trend lineare deterministico stabile.
#                             Produce una retta di proiezione forward.
#
# Tutti e tre fittano solo su `pre` e producono previsioni out-of-sample
# per il periodo post — nessun dato futuro usato in stima.
# ══════════════════════════════════════════════════════════════════════════════

def approach_a_pure_arima(pre: pd.Series, post: pd.Series,
                          order: tuple, cons: pd.Series) -> dict:
    """A: Auto-ARIMA AIC — selezione automatica (p,d,q) su pre."""
    if not HAS_SM:
        order_a = order or (1, 0, 1)
        aic = bic = np.nan
    else:
        order_a, fit_a = _fit_arima_criterion(pre, criterion="aic")
        if order_a is None:
            order_a = order or (1, 0, 1)
            aic = bic = np.nan
        else:
            aic, bic = float(fit_a.aic), float(fit_a.bic)
    return _make_approach("A: Auto-ARIMA AIC (pre-only forecast)",
                          pre, post, order_a, cons, aic, bic)


def approach_b_holt_winters(pre: pd.Series, post: pd.Series, cons: pd.Series) -> dict:
    """B: Holt-Winters ETS con trend additivo — estrapolazione del trend pre."""
    n = len(post)
    if HAS_SM:
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                hw = ExponentialSmoothing(
                    pre.values.astype(float),
                    trend="add",
                    initialization_method="estimated",
                ).fit(optimized=True, remove_bias=False)
            fc = hw.forecast(n)
            resid_std = float(np.std(hw.resid, ddof=1)) if len(hw.resid) > 1 else 0.0
            z = stats.norm.ppf(1 - CI_ALPHA / 2)
            h_arr = np.arange(1, n + 1)
            margin = z * resid_std * np.sqrt(h_arr)
            ci_lo = fc - margin
            ci_hi = fc + margin
            aic = float(hw.aic)
            bic = float(hw.bic)
        except Exception:
            slope = float((pre.iloc[-1] - pre.iloc[0]) / max(len(pre) - 1, 1))
            last_v = float(pre.iloc[-1])
            fc = np.array([last_v + slope * (i + 1) for i in range(n)])
            resid_std = float(pre.std())
            z = stats.norm.ppf(1 - CI_ALPHA / 2)
            ci_lo = fc - z * resid_std
            ci_hi = fc + z * resid_std
            aic = bic = np.nan
    else:
        slope = float((pre.iloc[-1] - pre.iloc[0]) / max(len(pre) - 1, 1))
        last_v = float(pre.iloc[-1])
        fc = np.array([last_v + slope * (i + 1) for i in range(n)])
        resid_std = float(pre.std())
        z = stats.norm.ppf(1 - CI_ALPHA / 2)
        ci_lo = fc - z * resid_std
        ci_hi = fc + z * resid_std
        aic = bic = np.nan

    extra = post.values - fc
    gain = float((extra * cons.values).sum() / 1e6)
    m = _arima_metrics(post.values, fc)
    return {
        "label": "B: Holt-Winters ETS trend (pre-only forecast)",
        "forecast": fc, "ci_lo": ci_lo, "ci_hi": ci_hi, "extra": extra,
        "gain_meur": gain,
        "ci_gain_lo": float(((post.values - ci_hi) * cons.values).sum() / 1e6),
        "ci_gain_hi": float(((post.values - ci_lo) * cons.values).sum() / 1e6),
        "omega": np.nan, "aic": aic, "bic": bic,
        **m,
    }


def approach_c_ols_trend(pre: pd.Series, post: pd.Series, cons: pd.Series) -> dict:
    """C: OLS trend lineare — regressione y=a+b·t sulla pre, estesa al post."""
    n = len(post)
    z = stats.norm.ppf(1 - CI_ALPHA / 2)

    t_pre = np.arange(len(pre), dtype=float)
    y_pre = pre.values.astype(float)
    X_pre = np.column_stack([np.ones(len(pre)), t_pre])
    try:
        coef, *_ = np.linalg.lstsq(X_pre, y_pre, rcond=None)
        a, b = float(coef[0]), float(coef[1])
    except Exception:
        a = float(pre.mean())
        b = 0.0

    t_post = np.arange(len(pre), len(pre) + n, dtype=float)
    fc = a + b * t_post

    fitted_pre = a + b * t_pre
    pre_resid = y_pre - fitted_pre
    resid_std = float(np.std(pre_resid, ddof=2)) if len(pre_resid) > 2 else float(pre.std())
    ci_lo = fc - z * resid_std
    ci_hi = fc + z * resid_std

    rss = float(np.sum(pre_resid**2))
    k = 2
    nn = len(pre)
    sigma2 = rss / nn
    ll = -nn / 2 * np.log(2 * np.pi * sigma2) - rss / (2 * sigma2)
    ols_aic = -2 * ll + 2 * k
    ols_bic = -2 * ll + k * np.log(nn)

    extra = post.values - fc
    gain = float((extra * cons.values).sum() / 1e6)
    m = _arima_metrics(post.values, fc)
    return {
        "label": "C: OLS trend lineare (pre-only forecast)",
        "forecast": fc, "ci_lo": ci_lo, "ci_hi": ci_hi, "extra": extra,
        "gain_meur": gain,
        "ci_gain_lo": float(((post.values - ci_hi) * cons.values).sum() / 1e6),
        "ci_gain_hi": float(((post.values - ci_lo) * cons.values).sum() / 1e6),
        "omega": np.nan, "aic": float(ols_aic), "bic": float(ols_bic),
        **m,
    }


def select_best_approach(results: dict) -> str:
    valid = {k: v for k, v in results.items() if not np.isnan(v["rmse"])}
    if not valid:
        return list(results.keys())[0]
    return min(valid, key=lambda k: (valid[k]["rmse"],
                                     valid[k]["aic"] if not np.isnan(valid[k]["aic"]) else 1e9))


# ══════════════════════════════════════════════════════════════════════════════
# Plot
# ══════════════════════════════════════════════════════════════════════════════

def _fmt_ax_v3(ax: plt.Axes) -> None:
    ax.grid(axis="y", alpha=0.2)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%d %b %y"))
    ax.xaxis.set_major_locator(mdates.WeekdayLocator(byweekday=0, interval=2))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=35, ha="right", fontsize=7)


def _plot_its(
    ev_name, ev, series, fuel_key, fuel_color,
    pre, post, order, results, best_key, t0, shock, mode, fig, row_start,
    cons,               # serie consumi giornalieri
) -> None:
    """
    3 pannelli per ogni fuel:
      (a) Serie + previsione CF best approccio (linea nera tratteggiata)
          + area verde dove actual > CF, area rossa dove actual < CF
      (b) Differenziale  actual − CF  con le 3 stime del gap
          + area verde dove gap > 0, area rossa dove gap < 0
      (c) Guadagno cumulato
    """
    colors = {"A": "#2980b9", "B": "#e67e22", "C": "#27ae60"}
    base   = row_start * 3

    ax_ser  = fig.add_subplot(len(FUELS), 3, base + 1)
    ax_diff = fig.add_subplot(len(FUELS), 3, base + 2)
    ax_gain = fig.add_subplot(len(FUELS), 3, base + 3)

    win = series[
        (series.index >= shock - pd.Timedelta(days=PRE_WIN)) &
        (series.index <= shock + pd.Timedelta(days=POST_WIN))
    ].dropna()

    # ── Pannello A: serie + CF best ──────────────────────────────────────────
    ax_ser.plot(win.index, win.values, color=fuel_color, lw=1.5, zorder=5,
                label=f"{fuel_key.capitalize()} effettivo")
    ax_ser.axvline(shock, color=ev["color"], lw=1.6, ls="--",
                   label=f"Shock ({shock.date()})")
    if mode == "detected" and t0 != shock:
        ax_ser.axvline(t0, color="black", lw=1.2, ls=":",
                       label=f"T0 PELT ({t0.date()})")

    best_fc = pd.Series(results[best_key]["forecast"], index=post.index)

    # Area verde dove actual > CF, area rossa dove actual < CF (solo nel periodo post)
    actual_post = pd.Series(post.values, index=post.index)
    ax_ser.fill_between(
        post.index,
        actual_post, best_fc,
        where=(actual_post >= best_fc),
        alpha=0.20, color="green",
        label="Actual > CF (margine sopra atteso)",
        zorder=2,
    )
    ax_ser.fill_between(
        post.index,
        actual_post, best_fc,
        where=(actual_post < best_fc),
        alpha=0.20, color="red",
        label="Actual < CF (margine sotto atteso)",
        zorder=2,
    )

    ax_ser.plot(best_fc.index, best_fc.values, color="black", lw=1.4, ls="--",
                label=f"CF {best_key} ★  RMSE={results[best_key]['rmse']:.4f}", zorder=4)
    ax_ser.fill_between(post.index,
                        results[best_key]["ci_lo"], results[best_key]["ci_hi"],
                        alpha=0.10, color="black", zorder=1)

    approach_labels = {"A": "ARIMA-AIC", "B": "Holt-Winters", "C": "OLS-trend"}
    ax_ser.set_title(
        f"[V3-ARIMA / mode={mode}]  {fuel_key.capitalize()} – {ev_name}\n"
        f"T0={t0.date()}  ARIMA{order}  Best: {best_key} ({approach_labels.get(best_key, best_key)})",
        fontsize=8, fontweight="bold"
    )
    ax_ser.set_ylabel("Margine (€/L)", fontsize=8)
    ax_ser.legend(fontsize=5.5, loc="upper left")
    _fmt_ax_v3(ax_ser)

    # ── Pannello B: differenziale  actual − CF  ──────────────────────────────
    for key, res in results.items():
        lc   = colors.get(key, "grey")
        ls   = "-"  if key == best_key else ":"
        lw   = 1.8 if key == best_key else 1.0
        gap  = post.values - res["forecast"]
        lbl  = (f"{key} ({approach_labels.get(key, key)})  RMSE={res['rmse']:.4f}"
                + (" ★" if key == best_key else ""))
        ax_diff.plot(post.index, gap, color=lc, lw=lw, ls=ls, label=lbl, zorder=3)

    # Area verde dove gap > 0 (margine sopra CF), area rossa dove gap < 0
    raw_gap = pd.Series(post.values - results[best_key]["forecast"], index=post.index)
    ax_diff.fill_between(
        post.index,
        raw_gap, 0,
        where=(raw_gap >= 0),
        alpha=0.20, color="green",
        label="extra > 0 (margine sopra CF)",
        zorder=2,
    )
    ax_diff.fill_between(
        post.index,
        raw_gap, 0,
        where=(raw_gap < 0),
        alpha=0.20, color="red",
        label="extra < 0 (margine sotto CF)",
        zorder=2,
    )

    ax_diff.axhline(0, color="grey", lw=0.8, ls="--")
    ax_diff.axvline(shock, color=ev["color"], lw=1.2, ls="--")
    ax_diff.set_title(
        f"Effetto stimato  (actual − CF)\n"
        f"se → 0: nessun effetto; se > 0: margine sopra atteso",
        fontsize=7.5
    )
    ax_diff.set_ylabel("Δ Margine (€/L)", fontsize=8)
    ax_diff.legend(fontsize=5.5, loc="upper left")
    _fmt_ax_v3(ax_diff)

    # ── Pannello C: guadagno cumulato (basato su consumi giornalieri) ────────
    for key, res in results.items():
        lc  = colors.get(key, "grey")
        ls  = "-"  if key == best_key else ":"
        lw  = 1.3 if key == best_key else 0.9
        cum = pd.Series(np.cumsum(res["extra"] * cons.values) / 1e6, index=post.index)
        ax_gain.plot(cum.index, cum.values, color=lc, lw=lw, ls=ls,
                     label=f"{key}: {res['gain_meur']:+.0f} M€")

    ax_gain.axhline(0, color="grey", lw=0.7, ls="--")
    ax_gain.set_title(
        f"Guadagno extra cumulato  [{fuel_key.capitalize()}]\n"
        f"Best ({best_key}) → {results[best_key]['gain_meur']:+.0f} M€",
        fontsize=7
    )
    ax_gain.set_ylabel("M€ cumulati", fontsize=8)
    ax_gain.legend(fontsize=6, loc="upper left")
    _fmt_ax_v3(ax_gain)


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    parser = argparse.ArgumentParser(description="V3 ARIMA/ITS – ITS pipeline")
    parser.add_argument("--mode", choices=["fixed", "detected"], default="fixed",
                        help="fixed = usa shock date; detected = θ da theta_results.csv (02c), fallback nativo se CSV assente")
    parser.add_argument("--detect", choices=["margin", "price"], default="margin",
                        help="(solo mode=detected) serie di detection: "
                             "margin = PELT sul margine [default], "
                             "price  = PELT sul prezzo pompa netto")
    args, _ = parser.parse_known_args()
    mode          = args.mode
    detect_target = args.detect

    if mode == "detected":
        OUT_DIR = _OUT_BASE / "detected" / detect_target / "v3_arima"
    else:
        OUT_DIR = _OUT_BASE / mode / "v3_arima"
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("═"*70)
    print(f"  02d_v3_arima.py  –  Metodo 3: ITS/ARIMA  [mode={mode}]")
    if mode == "fixed":
        print("  T0 = shock date hardcodata (nessuna detection)")
    else:
        print(f"  T0 = PELT (ruptures RBF, pen=2.0, ricerca ±{SEARCH}gg)")
        print(f"  Detection su: {'MARGINE distributore' if detect_target == 'margin' else 'PREZZO POMPA NETTO'}")
        if not HAS_RPT:
            print("  ⚠ ruptures non installato → fallback shock date")
    print(f"  statsmodels: {'OK' if HAS_SM else 'MANCANTE'}")
    print(f"  ruptures:    {'OK' if HAS_RPT else 'MANCANTE'}")
    print("  Consumi:      letti da data/consumi/consumi_giornalieri.csv (via forecast_consumi)")
    print(f"  Output: {OUT_DIR}")
    print("═"*70)

    # Carica dati margine
    data = load_margin_data()
    rows: list[dict] = []

    for ev_name, ev in EVENTS.items():
        shock = ev["shock"]

        fig = plt.figure(figsize=(22, 6 * len(FUELS)))
        fig.suptitle(
            f"[Metodo 3 – ITS/ARIMA / mode={mode}]  {ev_name}\n{ev['label']}",
            fontsize=11, fontweight="bold"
        )

        for row_idx, (fuel_key, (col_name, fuel_color)) in enumerate(FUELS.items()):
            series = data[col_name].dropna()

            # ── Determina T0 — θ da 02c_change_point_detection.py ───────────
            if mode == "detected":
                theta = load_theta(ev_name, fuel_key, detect_target,
                                   base_dir=BASE_DIR)
                if theta is not None:
                    t0           = theta
                    break_method = "glm_poisson_02c"
                else:
                    print(f"  ⚠ [{fuel_key}] θ non trovato — uso shock come fallback.")
                    t0           = shock
                    break_method = "fallback_shock"
                n_breaks = 0
            else:
                t0           = shock
                break_method = "fixed_at_shock"
                n_breaks     = 0

            pre = series[
                (series.index >= t0 - pd.Timedelta(days=PRE_WIN)) &
                (series.index < t0)
            ].dropna()
            post = series[
                (series.index >= t0) &
                (series.index < shock + pd.Timedelta(days=POST_WIN))
            ].dropna()

            if len(pre) < 15 or len(post) < 5:
                print(f"  [{fuel_key}] dati insufficienti – salto.")
                continue

            # ── Carica consumi giornalieri per il periodo post ───────────────
            cons = load_daily_consumption(post.index, fuel_key)   # Serie litri/giorno

            print(f"\n  {ev_name}  [{fuel_key.upper()}]")
            print(f"    T0 ({break_method}) = {t0.date()}  (shock={shock.date()})")
            if mode == "detected" and t0 != shock:
                print(f"    Anticipo/ritardo  = {(t0 - shock).days:+d} giorni  "
                      f"(PELT trovati {n_breaks} break totali)")

            # ── Auto-ARIMA (pre-only, per diagnostica residui) ────────────────
            print("    Auto-ARIMA AIC (pre)... ", end="", flush=True)
            order, pre_fit = auto_arima(pre)
            if order is None:
                order = (1, 0, 1)
            print(f"ARIMA{order}  AIC={pre_fit.aic:.1f}" if pre_fit else f"ARIMA{order} (fallback)")

            # Tutti e tre gli approcci usano SOLO dati pre — nessun look-ahead
            res_a = approach_a_pure_arima(pre, post, order, cons)
            res_b = approach_b_holt_winters(pre, post, cons)
            res_c = approach_c_ols_trend(pre, post, cons)

            results  = {"A": res_a, "B": res_b, "C": res_c}
            best_key = select_best_approach(results)
            best     = results[best_key]

            print(f"    {'Approccio':<8} {'Gain M€':>10} {'RMSE':>10} "
                  f"{'MAPE%':>8} {'AIC':>10} {'ω':>10}")
            for k, r in results.items():
                star = " ★" if k == best_key else ""
                print(f"    {k}{star:<7} {r['gain_meur']:>+10.1f} {r['rmse']:>10.5f} "
                      f"{r['mape']:>8.2f} {r['aic']:>10.1f} {r['omega']:>10.4f}")

            # ── Diagnostics residui ARIMA pre-periodo ─────────────────────────
            pre_resid = np.asarray(pre_fit.resid) if pre_fit is not None else np.array([])
            diag = run_diagnostic_tests(pre_resid, x_for_bg=None, n_lags=None)

            safe_ev = ev_name.replace(" ","_").replace("/","").replace("(","").replace(")","")
            diag_plot_out = OUT_DIR / f"diag_{safe_ev}_{fuel_key}.png"
            plot_residual_diagnostics(
                resid=pre_resid,
                dates=pre.index,
                title=(f"[V3-ARIMA] Diagnostica residui ARIMA{order} pre-periodo\n"
                       f"{ev_name} · {fuel_key.capitalize()}  (T0={t0.date()})"),
                out_path=diag_plot_out,
                diag_stats=diag,
            )
            if not np.isnan(diag.get("sw_p", np.nan)):
                print(f"    SW (normalità)    = W={diag['sw_stat']:.3f}  "
                      f"p={diag['sw_p']:.3f}  "
                      f"{'OK' if diag['sw_p'] > 0.05 else '⚠'}")
            if not np.isnan(diag.get("lb_p", np.nan)):
                print(f"    LB({diag['n_lags']}) (autocorr.) = "
                      f"Q={diag['lb_stat']:.2f}  p={diag['lb_p']:.3f}  "
                      f"{'OK' if diag['lb_p'] > 0.05 else '⚠'}")

            # ── SARIMA(0,1,1)(0,1,0)_12 benchmark (usa stessi consumi) ───────
            sarima = fit_sarima_benchmark(pre, n_steps=len(post), s=7)
            sarima_res_s: dict | None = None
            if sarima is not None:
                sarima_extra = post.values - sarima["forecast"]
                sarima_gain  = float((sarima_extra * cons.values).sum() / 1e6)
                sarima_rmse  = float(np.sqrt(np.mean(sarima_extra**2)))
                sarima_mape_mask = post.values != 0
                sarima_mape = (float(np.mean(np.abs(sarima_extra[sarima_mape_mask]
                                                    / post.values[sarima_mape_mask])) * 100)
                               if sarima_mape_mask.any() else np.nan)
                sarima_diag = run_diagnostic_tests(sarima["resid"], x_for_bg=None)
                sarima_diag_out = OUT_DIR / f"sarima_diag_{safe_ev}_{fuel_key}.png"
                plot_sarima_diagnostics(
                    resid=sarima["resid"],
                    dates=pre.index,
                    title=(f"[V3-ARIMA / SARIMA{sarima['order']}]  "
                           f"{ev_name} · {fuel_key.capitalize()}"),
                    out_path=sarima_diag_out,
                    diag_stats=sarima_diag,
                )
                print(f"    SARIMA bench      = {sarima_gain:+.0f} M€  "
                      f"RMSE={sarima_rmse:.5f}  AIC={sarima['aic']:.1f}")
                sarima_res_s = {
                    "label":     sarima["order"],
                    "forecast":  sarima["forecast"],
                    "ci_lo":     sarima["ci_lo"],
                    "ci_hi":     sarima["ci_hi"],
                    "extra":     sarima_extra,
                    "gain_meur": sarima_gain,
                    "ci_gain_lo": float(((post.values - sarima["ci_hi"]) * cons.values).sum() / 1e6),
                    "ci_gain_hi": float(((post.values - sarima["ci_lo"]) * cons.values).sum() / 1e6),
                    "omega": np.nan, "aic": sarima["aic"], "bic": sarima["bic"],
                    "rmse": sarima_rmse, "mape": sarima_mape,
                    "_diag": sarima_diag,
                }

            _plot_its(ev_name, ev, series, fuel_key, fuel_color,
                      pre, post, order, results, best_key,
                      t0, shock, mode, fig, row_idx, cons)

            # ── Export residui pre/post BEST approach (standard per nonparam) ─
            _safe_ev = (ev_name.replace(" ", "_").replace("/", "")
                               .replace("(", "").replace(")", ""))
            _best_res = results[best_key]
            _pre_resid_arr = (np.asarray(pre_fit.resid, float)
                              if pre_fit is not None else np.array([]))
            _resid_rows = []
            # Pre: residui ARIMA stimati sulla finestra pre-break
            _pre_dates = pre.index[:len(_pre_resid_arr)]
            for _d, _r in zip(_pre_dates, _pre_resid_arr):
                _resid_rows.append({
                    "date": str(_d.date()), "residual": float(_r), "phase": "pre",
                    "metodo": "v3_arima", "evento": ev_name,
                    "carburante": fuel_key, "break_date": str(t0.date()),
                })
            # Post: actual − baseline ARIMA (= extra-profitto giornaliero)
            _post_extra = _best_res.get("extra", np.array([]))
            for _d, _r in zip(post.index[:len(_post_extra)], _post_extra):
                _resid_rows.append({
                    "date": str(_d.date()), "residual": float(_r), "phase": "post",
                    "metodo": "v3_arima", "evento": ev_name,
                    "carburante": fuel_key, "break_date": str(t0.date()),
                })
            pd.DataFrame(_resid_rows).to_csv(
                OUT_DIR / f"residuals_{_safe_ev}_{fuel_key}.csv", index=False
            )

            def _row_base():
                return {
                    "metodo":        "v3_arima",
                    "mode":          mode,
                    "break_method":  break_method,
                    "evento":        ev_name,
                    "carburante":    fuel_key,
                    "shock":         shock.date(),
                    "break_date":    t0.date(),
                    "pre_win_days":  PRE_WIN,
                    "post_win_days": POST_WIN,
                    "n_pre":         len(pre),
                    "n_post":        len(post),
                    "arima_order":   str(order),
                    # Primary ARIMA residual diagnostics (same for all approaches)
                    "sw_stat":       round(diag.get("sw_stat", np.nan), 4),
                    "sw_p":          round(diag.get("sw_p", np.nan), 4),
                    "lb_stat":       round(diag.get("lb_stat", np.nan), 3),
                    "lb_p":          round(diag.get("lb_p", np.nan), 4),
                    "diag_n_lags":   diag.get("n_lags", np.nan),
                    # SARIMA benchmark summary
                    "sarima_bench_gain_meur": (round(sarima_res_s["gain_meur"], 1)
                                               if sarima_res_s else np.nan),
                    "sarima_bench_rmse":      (round(sarima_res_s["rmse"], 5)
                                               if sarima_res_s else np.nan),
                    "sarima_bench_aic":       (round(sarima_res_s["aic"], 2)
                                               if sarima_res_s else np.nan),
                }

            for approach_key, res in results.items():
                row = _row_base()
                row.update({
                    "approccio":         approach_key,
                    "extra_mean_eurl":   round(float(np.mean(res["extra"])), 5),
                    "extra_sum_eurl":    round(float(np.sum(res["extra"])), 4),
                    "gain_total_meur":   round(res["gain_meur"], 1),
                    "gain_ci_low_meur":  round(res["ci_gain_lo"], 1),
                    "gain_ci_high_meur": round(res["ci_gain_hi"], 1),
                    "omega":             round(res["omega"], 5) if not np.isnan(res["omega"]) else np.nan,
                    "aic":               round(res["aic"], 2) if not np.isnan(res["aic"]) else np.nan,
                    "bic":               round(res["bic"], 2) if not np.isnan(res["bic"]) else np.nan,
                    "rmse":              round(res["rmse"], 5),
                    "mape_pct":          round(res["mape"], 3),
                    "best_approach":     best_key,
                    "is_best":           approach_key == best_key,
                    "transfer_fn":       res["label"],
                    "note": f"ITS/ARIMA, mode={mode}, break={break_method}",
                })
                rows.append(row)

            # SARIMA benchmark approach row
            if sarima_res_s is not None:
                s_diag = sarima_res_s.get("_diag", {})
                row_s = _row_base()
                row_s.update({
                    "approccio":         "S",
                    "extra_mean_eurl":   round(float(np.mean(sarima_res_s["extra"])), 5),
                    "extra_sum_eurl":    round(float(np.sum(sarima_res_s["extra"])), 4),
                    "gain_total_meur":   round(sarima_res_s["gain_meur"], 1),
                    "gain_ci_low_meur":  round(sarima_res_s["ci_gain_lo"], 1),
                    "gain_ci_high_meur": round(sarima_res_s["ci_gain_hi"], 1),
                    "omega":             np.nan,
                    "aic":               round(sarima_res_s["aic"], 2),
                    "bic":               round(sarima_res_s["bic"], 2),
                    "rmse":              round(sarima_res_s["rmse"], 5),
                    "mape_pct":          round(sarima_res_s["mape"], 3) if not np.isnan(sarima_res_s["mape"]) else np.nan,
                    "best_approach":     best_key,
                    "is_best":           False,
                    "transfer_fn":       sarima_res_s["label"],
                    # SARIMA's own residual diagnostics
                    "sarima_sw_stat":    round(s_diag.get("sw_stat", np.nan), 4),
                    "sarima_sw_p":       round(s_diag.get("sw_p", np.nan), 4),
                    "sarima_lb_stat":    round(s_diag.get("lb_stat", np.nan), 3),
                    "sarima_lb_p":       round(s_diag.get("lb_p", np.nan), 4),
                    "note": f"SARIMA(0,1,1)(0,1,0)_12 benchmark, mode={mode}",
                })
                rows.append(row_s)

        fig.tight_layout()
        safe = ev_name.replace(" ","_").replace("/","").replace("(","").replace(")","")
        out  = OUT_DIR / f"plot_{safe}.png"
        fig.savefig(out, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"\n  → Salvato: {out}")

    if rows:
        df = pd.DataFrame(rows)
        csv_out = OUT_DIR / "v3_arima_results.csv"
        df.to_csv(csv_out, index=False)
        print(f"\n  → CSV: {csv_out}")
        best_df = df[df["is_best"]]
        print("\nRIEPILOGO (best approach per ogni caso):")
        cols = ["evento","carburante","break_date","best_approach","gain_total_meur","rmse"]
        print(best_df[cols].to_string(index=False))
    else:
        print("\n  ⚠ Nessun risultato prodotto.")


if __name__ == "__main__":
    main()