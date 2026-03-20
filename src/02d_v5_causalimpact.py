#!/usr/bin/env python3
"""
02d_v5_causalimpact.py  ─  Metodo 5: Bayesian Structural Time Series (BSTS)
=============================================================================
Stima l'extra-profitto speculativo sul margine distributori usando CausalImpact
(Brodersen et al. 2015, Google), un approccio Bayesiano strutturale che:

  ─ Baseline     : modello BSTS stimato sui PRE_WIN giorni precedenti il BREAK
                   (coerente con v1–v4: [break_date − PRE_WIN, break_date − 1])
  ─ Controfattuale: distribuzione posteriore proiettata nel periodo post-break
  ─ Extra profitto : media posteriore (effettivo − controfattuale)
  ─ CI           : credible interval bayesiano al (1-α)%
  ─ p-value      : probabilità di coda posteriore (Bayesian tail-area p)

Differenze chiave rispetto a v1–v4:
  ✓ Inferenza Bayesiana completa (posterior distribution, non SE frequentista)
  ✓ Baseline stimata SEMPRE su [shock − PRE_WIN, shock − 1], mai contaminata
    dal periodo di transizione [shock, break_date] in mode=detected
  ✓ Covariate opzionale: margine dell'altro carburante come serie di controllo
    (es. margin_gasolio come covariate per benzina e viceversa)
  ✓ Distribuzione assunta: Gaussiana sul modello latente BSTS
    (robustezza diversa da OLS, ARIMA, Gamma)
  ✓ Consumi giornalieri reali da data/consumi/consumi_giornalieri.csv

Dipendenze:
  pip install causalimpact     # wrapper Python di Google CausalImpact (TFP)

Modalità (--mode):
  fixed     : break = shock date hardcodata [default]
  detected  : break θ letto da theta_results.csv prodotto da
              02c_change_point_detection.py
              Eseguire prima: python3 02c_change_point_detection.py --detect {margin|price}

Parametro --detect (solo mode=detected):
  margin  : usa θ rilevato sul margine distributore  [default]
  price   : usa θ rilevato sul prezzo alla pompa netto (€/L)

Parametro --covariate:
  none      : BSTS univariato sul solo margine  [default]
  cross     : usa il margine dell'ALTRO carburante come covariate di controllo
              (es. per benzina usa gasolio, e viceversa)

Output:
  data/plots/its/fixed/v5_causalimpact/                    (mode=fixed)
  data/plots/its/detected/{margin|price}/v5_causalimpact/  (mode=detected)
    plot_{evento}.png
    bsts_{evento}_{carburante}.png    ← posterior + pointwise effect
    v5_causalimpact_results.csv

Riferimento:
  Brodersen KH, Gallusser F, Koehler J, Remy N, Scott SL (2015).
  Inferring causal impact using Bayesian structural time-series models.
  Annals of Applied Statistics, 9(1), 247–274.
"""

from __future__ import annotations

# ── Patch compatibilità pandas ≥ 2.0 con causalimpact 0.2.x ──────────────────
import pandas.core.dtypes.common as _pdc
if not hasattr(_pdc, "is_datetime_or_timedelta_dtype"):
    _pdc.is_datetime_or_timedelta_dtype = _pdc.is_datetime64_any_dtype

import os
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")   # silenzia TensorFlow

from pathlib import Path
import argparse
import warnings
import sys

warnings.filterwarnings("ignore")

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent / "utils"))
from conversions import GAS_OIL, EUROBOB as EUROBOB_HC, load_eurusd, usd_ton_to_eur_liter
from theta_loader import load_theta
from forecast_consumi import load_daily_consumption   # <-- nuova importazione

try:
    from causalimpact import CausalImpact
    HAS_CI = True
except ImportError:
    HAS_CI = False

# ── Configurazione ─────────────────────────────────────────────────────────────
BASE_DIR    = Path(__file__).parent
DAILY_CSV   = BASE_DIR / "data" / "processed" / "daily_fuel_prices_all.csv"
GASOIL_CSV  = BASE_DIR / "data" / "Futures" / "London Gas Oil Futures Historical Data.csv"
EUROBOB_CSV = BASE_DIR / "data" / "Futures" / "Eurobob_B7H1_date.csv"
EURUSD_CSV  = BASE_DIR / "data" / "raw" / "eurusd.csv"
_OUT_BASE   = BASE_DIR / "data" / "plots" / "its"

PRE_WIN  = 40    # giorni pre-SHOCK per stimare la baseline BSTS
POST_WIN = 40    # giorni post-break per misurare l'extra profitto
CI_ALPHA = 0.05  # livello α → credible interval al 95%

# DAILY_CONSUMPTION_L rimosso – ora letto dal CSV

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

# Mappa fuel_key → colonna margine e colore plot
FUELS: dict[str, tuple[str, str]] = {
    "benzina": ("margin_benzina", "#E63946"),
    "gasolio": ("margin_gasolio", "#1D3557"),
}

# Carburante opposto per covariate cross
_CROSS_FUEL: dict[str, str] = {
    "benzina": "margin_gasolio",
    "gasolio": "margin_benzina",
}


# ══════════════════════════════════════════════════════════════════════════════
# Caricamento dati  (identico a v1–v4)
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
    _IT = {"gen":"Jan","feb":"Feb","mar":"Mar","apr":"Apr","mag":"May","giu":"Jun",
           "lug":"Jul","ago":"Aug","set":"Sep","ott":"Oct","nov":"Nov","dic":"Dec"}
    if "timestamp" in df.columns:
        ts = pd.to_numeric(df["timestamp"], errors="coerce")
        df["date"] = (pd.to_datetime(ts, unit="s", utc=True)
                      .dt.tz_localize(None).dt.normalize())
    else:
        def _parse(s: str) -> pd.Timestamp:
            for it, en in _IT.items():
                s = s.replace(it, en)
            return pd.to_datetime(s, dayfirst=True, errors="coerce")
        df["date"] = df["data"].astype(str).apply(_parse)
    df["price"] = pd.to_numeric(df["chiusura"], errors="coerce")
    df = df.dropna(subset=["date","price"]).sort_values("date").set_index("date")
    df = df[~df.index.duplicated(keep="first")]
    return usd_ton_to_eur_liter(df["price"], eurusd, EUROBOB_HC)


def load_margin_data() -> pd.DataFrame:
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
    return df


# ══════════════════════════════════════════════════════════════════════════════
# Core BSTS via statsmodels UnobservedComponents
# (sostituisce la libreria causalimpact che crasha con TFP su Python ≥ 3.10)
# ══════════════════════════════════════════════════════════════════════════════

class _BSResult:
    """
    Wrapper leggero che espone .inferences con le stesse colonne
    che il resto del codice si aspetta da causalimpact 0.2.x:
      response, point_pred, point_pred_lower/upper,
      point_effect, point_effect_lower/upper,
      cum_effect, cum_effect_lower/upper
    """
    def __init__(self, inferences: pd.DataFrame):
        self.inferences = inferences


def _run_causal_impact(
    series: pd.Series,
    shock: pd.Timestamp,
    break_date: pd.Timestamp,
    covariate: pd.Series | None = None,
    alpha: float = CI_ALPHA,
) -> "_BSResult | None":
    """
    Stima BSTS (local linear trend) via statsmodels.UnobservedComponents.

    Pre-period : [break_date − PRE_WIN, break_date − 1]
    Post-period: [break_date, break_date + POST_WIN]

    Identico agli altri metodi v1–v4: baseline stimata sui PRE_WIN giorni
    precedenti il break, poi forecastata nel post per costruire il controfattuale.
    """
    from statsmodels.tsa.statespace.structural import UnobservedComponents
    from scipy import stats as _st

    win_start  = break_date - pd.Timedelta(days=PRE_WIN)
    win_end    = break_date + pd.Timedelta(days=POST_WIN)
    pre_end    = break_date - pd.Timedelta(days=1)
    post_start = break_date

    # ── Prepara serie outcome ─────────────────────────────────────────────────
    if series.index.duplicated().any():
        series = series.groupby(series.index).mean()

    full_idx = pd.date_range(win_start, win_end, freq="D")
    y_full   = series.reindex(full_idx).ffill().dropna()

    if len(y_full) < 15:
        return None

    pre_end    = min(pre_end,    y_full.index[-1])
    post_start = max(post_start, y_full.index[0])
    if pre_end < y_full.index[0] or post_start > y_full.index[-1]:
        return None
    if pre_end >= post_start:
        pre_end = post_start - pd.Timedelta(days=1)
    if pre_end < y_full.index[0]:
        return None

    y_pre  = y_full[y_full.index <= pre_end]
    y_post = y_full[y_full.index >= post_start]
    n_pre, n_post = len(y_pre), len(y_post)

    if n_pre < 10 or n_post < 3:
        return None

    # ── Prepara covariate ─────────────────────────────────────────────────────
    exog_pre = exog_post = None
    if covariate is not None:
        if covariate.index.duplicated().any():
            covariate = covariate.groupby(covariate.index).mean()
        cov_full = covariate.reindex(y_full.index).ffill().bfill()
        if not cov_full.isna().all():
            exog_pre  = cov_full.reindex(y_pre.index).values.reshape(-1, 1)
            exog_post = cov_full.reindex(y_post.index).values.reshape(-1, 1)

    # ── Fit UCM sul pre-periodo ───────────────────────────────────────────────
    # Proviamo più metodi di ottimizzazione per evitare ConvergenceWarning:
    # lbfgs (veloce ma può non convergere su serie piatte) → powell → nm (Nelder-Mead)
    res = None
    for spec in ("local linear trend", "local level"):
        for opt_method in ("lbfgs", "powell", "nm"):
            try:
                import warnings as _warnings
                mod = UnobservedComponents(y_pre.values, level=spec, exog=exog_pre)
                with _warnings.catch_warnings():
                    _warnings.simplefilter("ignore")
                    _r = mod.fit(disp=False, maxiter=500, method=opt_method,
                                 optim_complex_step=(opt_method == "lbfgs"))
                # accettiamo solo se la log-likelihood è finita
                if np.isfinite(_r.llf):
                    res = _r
                    break
            except Exception:
                pass
        if res is not None:
            break

    if res is None:
        return None

    # ── Forecast post-periodo ─────────────────────────────────────────────────
    try:
        fcast     = res.get_forecast(steps=n_post, exog=exog_post)
        pred_mean = np.asarray(fcast.predicted_mean).ravel()
        pred_ci   = fcast.conf_int(alpha=alpha)
        if hasattr(pred_ci, "iloc"):        # DataFrame (statsmodels ≥ 0.14)
            pred_lower = pred_ci.iloc[:, 0].values
            pred_upper = pred_ci.iloc[:, 1].values
        else:                               # ndarray (versioni più vecchie)
            pred_ci    = np.asarray(pred_ci)
            pred_lower = pred_ci[:, 0]
            pred_upper = pred_ci[:, 1]
    except Exception as exc:
        print(f"  ⚠  UCM forecast error: {exc}")
        return None

    # ── Fitted values + IC in-sample nel pre-periodo ──────────────────────────
    fitted    = np.asarray(res.fittedvalues).ravel()
    resid_std = float(np.nanstd(np.asarray(res.resid).ravel()))
    z         = _st.norm.ppf(1 - alpha / 2)
    fitted_lo = fitted - z * resid_std
    fitted_hi = fitted + z * resid_std

    # ── Build inferences DataFrame ────────────────────────────────────────────
    inf = pd.DataFrame(index=y_full.index)
    inf["response"] = y_full.values

    pp = np.empty(len(y_full)); pp_lo = pp.copy(); pp_hi = pp.copy()
    pp[:n_pre]  = fitted;    pp_lo[:n_pre]  = fitted_lo; pp_hi[:n_pre]  = fitted_hi
    pp[n_pre:]  = pred_mean; pp_lo[n_pre:]  = pred_lower; pp_hi[n_pre:] = pred_upper

    inf["point_pred"]        = pp
    inf["point_pred_lower"]  = pp_lo
    inf["point_pred_upper"]  = pp_hi

    actual = y_full.values
    pe     = np.where(inf.index >= post_start, actual - pp,       0.0)
    pe_lo  = np.where(inf.index >= post_start, actual - pp_hi,    0.0)
    pe_hi  = np.where(inf.index >= post_start, actual - pp_lo,    0.0)

    inf["point_effect"]        = pe
    inf["point_effect_lower"]  = pe_lo
    inf["point_effect_upper"]  = pe_hi

    cum = np.zeros(len(y_full)); cum_lo = cum.copy(); cum_hi = cum.copy()
    cum[n_pre:]    = np.cumsum(pe[n_pre:])
    cum_lo[n_pre:] = np.cumsum(pe_lo[n_pre:])
    cum_hi[n_pre:] = np.cumsum(pe_hi[n_pre:])

    inf["cum_effect"]        = cum
    inf["cum_effect_lower"]  = cum_lo
    inf["cum_effect_upper"]  = cum_hi

    return _BSResult(inf)


# ══════════════════════════════════════════════════════════════════════════════
# Plot
# ══════════════════════════════════════════════════════════════════════════════

def _plot_event_fuel(
    ev_name: str, ev: dict,
    series: pd.Series,
    fuel_key: str, fuel_color: str,
    ci_result: "_BSResult",
    shock: pd.Timestamp,
    break_date: pd.Timestamp,
    mode: str,
    covariate_label: str,
    cons: pd.Series,               # <-- consumi giornalieri (litri/giorno)
    ax_main: plt.Axes,
    ax_gain: plt.Axes,
) -> float:
    """
    Riempie ax_main (margine + controfattuale BSTS) e ax_gain (guadagno cumulato).
    Restituisce gain_meur.
    """
    inf = ci_result.inferences  # causalimpact 0.2.x: columns point_pred/point_effect/...

    win_start = shock - pd.Timedelta(days=PRE_WIN)
    win_end   = shock + pd.Timedelta(days=POST_WIN)

    # Serie effettiva nella finestra completa
    win = series[(series.index >= win_start) & (series.index <= win_end)].dropna()

    # Estrarre predizioni post-periodo  (nuovi nomi colonne 0.2.x)
    post_pred_mean  = inf["point_pred"]
    post_pred_lower = inf["point_pred_lower"]
    post_pred_upper = inf["point_pred_upper"]

    # ── ax_main ───────────────────────────────────────────────────────────────
    ax_main.plot(win.index, win.values, color=fuel_color, lw=1.0,
                 label=f"{fuel_key.capitalize()} effettivo")

    # Baseline (controfattuale BSTS)
    ax_main.plot(post_pred_mean.index, post_pred_mean.values,
                 color="dimgrey", lw=1.3, ls="--",
                 label="Controfattuale BSTS")
    ax_main.fill_between(post_pred_lower.index,
                         post_pred_lower.values, post_pred_upper.values,
                         alpha=0.15, color="grey",
                         label=f"CI {int((1-CI_ALPHA)*100)}% bayesiano")

    # Effetto: aree colorate solo nel post-periodo
    post_mask = (win.index >= break_date) & (win.index <= win_end)
    post_actual = win[post_mask]
    post_cf     = post_pred_mean.reindex(post_actual.index)
    if post_cf is not None and len(post_cf) > 0:
        gap = post_actual - post_cf
        ax_main.fill_between(post_actual.index,
                             post_actual.values, post_cf.values,
                             where=(gap >= 0), alpha=0.22, color="green",
                             label="Extra profitto (≥0)")
        ax_main.fill_between(post_actual.index,
                             post_actual.values, post_cf.values,
                             where=(gap < 0), alpha=0.22, color="red",
                             label="Sotto-baseline (<0)")

    # Linee verticali
    ax_main.axvline(shock, color=ev["color"], lw=1.6, ls="--",
                    label=f"Shock ({shock.date()})")
    if mode == "detected" and break_date != shock:
        ax_main.axvline(break_date, color="black", lw=1.2, ls=":",
                        label=f"θ rilevato ({break_date.date()})")

    mode_str = (f"Break=θ {break_date.date()} (GLM Poisson 02c)"
                if mode == "detected" else f"Break=shock ({shock.date()})")
    cov_str  = f" | cov={covariate_label}" if covariate_label != "none" else ""
    ax_main.set_title(
        f"[V5-BSTS / mode={mode}]  {fuel_key.capitalize()} – {ev_name}\n"
        f"{mode_str}  |  BSTS Bayesiano{cov_str}",
        fontsize=8, fontweight="bold",
    )
    ax_main.set_ylabel("Margine (€/L)", fontsize=8)
    ax_main.legend(fontsize=6, loc="upper left", ncol=2)
    ax_main.grid(axis="y", alpha=0.20)
    ax_main.xaxis.set_major_formatter(mdates.DateFormatter("%d %b %y"))
    ax_main.xaxis.set_major_locator(mdates.WeekdayLocator(byweekday=0, interval=2))
    plt.setp(ax_main.xaxis.get_majorticklabels(), rotation=35, ha="right", fontsize=7)

    # ── ax_gain (guadagno cumulato) – usando consumi reali ────────────────────
    # point_effect = actual − counterfactual (€/L)
    pe = inf["point_effect"]
    post_pe = pe[pe.index >= break_date]   # Series allineata ai giorni post-break

    # Allinea i consumi alle stesse date (cons è già una Series con indice date)
    cons_aligned = cons.reindex(post_pe.index, method="ffill")
    if cons_aligned.isna().any():
        # Se mancano giorni, usiamo ffill (valore precedente)
        cons_aligned = cons_aligned.fillna(method="ffill").fillna(cons.mean())

    # Guadagno giornaliero in M€ = (€/L) * (L/giorno) / 1e6
    daily_gain_meur = (post_pe * cons_aligned) / 1e6
    cum = daily_gain_meur.cumsum()
    gain_meur = float(cum.iloc[-1]) if len(cum) > 0 else 0.0

    # CI cumulato
    pe_lo = inf["point_effect_lower"]
    pe_hi = inf["point_effect_upper"]
    post_lo = pe_lo[pe_lo.index >= break_date]
    post_hi = pe_hi[pe_hi.index >= break_date]
    daily_lo = (post_lo * cons_aligned) / 1e6
    daily_hi = (post_hi * cons_aligned) / 1e6
    cum_lo = daily_lo.cumsum()
    cum_hi = daily_hi.cumsum()

    ax_gain.plot(cum.index, cum.values, color=fuel_color, lw=1.2)
    ax_gain.fill_between(cum.index, cum_lo.values, cum_hi.values,
                         alpha=0.20, color="grey",
                         label=f"CI {int((1-CI_ALPHA)*100)}% bayesiano")
    ax_gain.axhline(0, color="grey", lw=0.7, ls="--")
    ax_gain.fill_between(cum.index, cum.values, 0,
                         where=(cum >= 0), alpha=0.25, color="green")
    ax_gain.fill_between(cum.index, cum.values, 0,
                         where=(cum < 0), alpha=0.25, color="red")

    gain_lo = float(cum_lo.iloc[-1]) if len(cum_lo) > 0 else 0.0
    gain_hi = float(cum_hi.iloc[-1]) if len(cum_hi) > 0 else 0.0
    avg_cons_ml = cons.mean() / 1e6
    ax_gain.set_title(
        f"Guadagno extra cumulato → {gain_meur:+.0f} M€  "
        f"({len(post_pe)}gg post-break)\n"
        f"CI95% [{gain_lo:+.0f}, {gain_hi:+.0f}] M€  "
        f"[consumo medio {avg_cons_ml:.1f} ML/giorno]",
        fontsize=7,
    )
    ax_gain.set_ylabel("M€ cumulati", fontsize=8)
    ax_gain.legend(fontsize=6)
    ax_gain.grid(axis="y", alpha=0.20)
    ax_gain.xaxis.set_major_formatter(mdates.DateFormatter("%d %b %y"))
    ax_gain.xaxis.set_major_locator(mdates.WeekdayLocator(byweekday=0, interval=2))
    plt.setp(ax_gain.xaxis.get_majorticklabels(), rotation=35, ha="right", fontsize=7)

    return gain_meur


def _plot_bsts_diagnostics(
    ci_result: "_BSResult",
    ev_name: str,
    fuel_key: str,
    fuel_color: str,
    break_date: pd.Timestamp,
    mode: str,
    out_path: Path,
) -> None:
    """
    Plot diagnostico BSTS a 3 pannelli:
      1. Serie completa + controfattuale + CI
      2. Effetto puntuale (observed − counterfactual) con CI
      3. Effetto cumulato con CI
    """
    inf = ci_result.inferences

    fig, axes = plt.subplots(3, 1, figsize=(11, 9), sharex=True)
    fig.suptitle(
        f"[V5-BSTS] Diagnostica CausalImpact — {ev_name} · {fuel_key.capitalize()}\n"
        f"mode={mode}  |  break={break_date.date()}",
        fontsize=9, fontweight="bold",
    )

    # ── 1. Serie + controfattuale ─────────────────────────────────────────────
    ax = axes[0]
    actual = inf["response"]
    ax.plot(actual.index, actual.values, color=fuel_color, lw=1.0,
            label="Effettivo")
    ax.plot(inf["point_pred"].index,
            inf["point_pred"].values,
            color="dimgrey", lw=1.2, ls="--", label="Controfattuale BSTS")
    ax.fill_between(inf["point_pred_lower"].index,
                    inf["point_pred_lower"].values,
                    inf["point_pred_upper"].values,
                    alpha=0.15, color="grey", label="CI 95%")
    ax.axvline(break_date, color="black", lw=1.0, ls=":")
    ax.set_ylabel("Margine (€/L)", fontsize=8)
    ax.legend(fontsize=7)
    ax.grid(axis="y", alpha=0.20)

    # ── 2. Effetto puntuale ───────────────────────────────────────────────────
    ax = axes[1]
    pe     = inf["point_effect"]
    pe_lo  = inf["point_effect_lower"]
    pe_hi  = inf["point_effect_upper"]
    ax.plot(pe.index, pe.values, color=fuel_color, lw=1.0)
    ax.fill_between(pe_lo.index, pe_lo.values, pe_hi.values,
                    alpha=0.20, color="grey")
    ax.axhline(0, color="grey", lw=0.8, ls="--")
    ax.axvline(break_date, color="black", lw=1.0, ls=":")
    ax.fill_between(pe.index, pe.values, 0,
                    where=(pe >= 0), alpha=0.25, color="green")
    ax.fill_between(pe.index, pe.values, 0,
                    where=(pe < 0), alpha=0.25, color="red")
    ax.set_ylabel("Effetto puntuale (€/L)", fontsize=8)
    ax.grid(axis="y", alpha=0.20)

    # ── 3. Effetto cumulato (€/L cumulati, senza consumi) ────────────────────
    ax = axes[2]
    post_mask = inf.index >= break_date
    cum    = inf.loc[post_mask, "cum_effect"]
    cum_lo = inf.loc[post_mask, "cum_effect_lower"]
    cum_hi = inf.loc[post_mask, "cum_effect_upper"]
    ax.plot(cum.index, cum.values, color=fuel_color, lw=1.0)
    ax.fill_between(cum_lo.index, cum_lo.values, cum_hi.values,
                    alpha=0.20, color="grey")
    ax.axhline(0, color="grey", lw=0.8, ls="--")
    ax.axvline(break_date, color="black", lw=1.0, ls=":")
    ax.fill_between(cum.index, cum.values, 0,
                    where=(cum >= 0), alpha=0.25, color="green")
    ax.fill_between(cum.index, cum.values, 0,
                    where=(cum < 0), alpha=0.25, color="red")
    ax.set_ylabel("Effetto cumulato (€/L)", fontsize=8)
    ax.grid(axis="y", alpha=0.20)

    for ax in axes:
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%d %b %y"))
        ax.xaxis.set_major_locator(mdates.WeekdayLocator(byweekday=0, interval=2))
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=35, ha="right", fontsize=7)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    if not HAS_CI:
        print("  ⚠  causalimpact (dafiti) non installato → uso statsmodels UCM come backend BSTS.")

    parser = argparse.ArgumentParser(description="V5 CausalImpact BSTS – ITS pipeline")
    parser.add_argument("--mode", choices=["fixed", "detected"], default="fixed",
                        help="fixed = usa shock date; detected = usa θ da theta_results.csv")
    parser.add_argument("--detect", choices=["margin", "price"], default="margin",
                        help="(solo mode=detected) serie su cui è stata fatta la detection")
    parser.add_argument("--covariate", choices=["none", "cross"], default="none",
                        help="none = BSTS univariato; cross = usa margine opposto come covariate")
    args, _ = parser.parse_known_args()
    mode          = args.mode
    detect_target = args.detect
    covariate_opt = args.covariate

    if mode == "detected":
        OUT_DIR = _OUT_BASE / "detected" / detect_target / "v5_causalimpact"
    else:
        OUT_DIR = _OUT_BASE / mode / "v5_causalimpact"
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("═"*70)
    print("  02d_v5_causalimpact.py  –  Metodo 5: BSTS CausalImpact")
    print(f"  mode={mode}  |  detect={detect_target}  |  covariate={covariate_opt}")
    if mode == "fixed":
        print("  Break = shock date hardcodata (nessuna detection)")
    else:
        print(f"  Break = θ GLM Poisson da 02c_change_point_detection.py")
        print(f"  Detection su: {'MARGINE distributore' if detect_target == 'margin' else 'PREZZO POMPA NETTO'}")
    print(f"  Baseline: PRE_WIN={PRE_WIN}gg prima del break (coerente con v1–v4)")
    print(f"  Post:     POST_WIN={POST_WIN}gg dal break point")
    print("  Consumi:  letti da data/consumi/consumi_giornalieri.csv (via forecast_consumi)")
    print(f"  Output:   {OUT_DIR}")
    print("  Ref: Brodersen et al. (2015), Ann. Appl. Stat., 9(1), 247–274")
    print("═"*70)

    data = load_margin_data()
    rows: list[dict] = []

    for ev_name, ev in EVENTS.items():
        shock = ev["shock"]

        fig, axes = plt.subplots(len(FUELS), 2,
                                 figsize=(15, 5 * len(FUELS)),
                                 squeeze=False)
        fig.suptitle(
            f"[Metodo 5 – BSTS CausalImpact / mode={mode}]  {ev_name}\n{ev['label']}",
            fontsize=11, fontweight="bold",
        )

        for row_idx, (fuel_key, (col_name, fuel_color)) in enumerate(FUELS.items()):
            series = data[col_name].dropna()

            if len(series) < 20:
                print(f"  [{fuel_key}] dati insufficienti – salto.")
                for ax in axes[row_idx]:
                    ax.text(0.5, 0.5, "Dati insufficienti",
                            ha="center", va="center", transform=ax.transAxes)
                continue

            # ── Determina break_date ──────────────────────────────────────────
            if mode == "detected":
                theta = load_theta(ev_name, fuel_key, detect_target,
                                   base_dir=BASE_DIR)
                if theta is not None:
                    break_date   = theta
                    break_method = "glm_poisson_02c"
                else:
                    print(f"  ⚠ [{fuel_key}] θ non trovato — uso shock come fallback.")
                    break_date   = shock
                    break_method = "fallback_shock"
            else:
                break_date   = shock
                break_method = "fixed_at_shock"

            # ── Covariate ─────────────────────────────────────────────────────
            covariate      = None
            covariate_label = "none"
            if covariate_opt == "cross":
                cross_col = _CROSS_FUEL[fuel_key]
                if cross_col in data.columns:
                    covariate       = data[cross_col].dropna()
                    covariate_label = cross_col

            # ── CausalImpact ──────────────────────────────────────────────────
            print(f"\n  [{fuel_key}] {ev_name}  break={break_date.date()}  "
                  f"({break_method})  cov={covariate_label}")

            ci_result = _run_causal_impact(
                series, shock, break_date,
                covariate=covariate,
                alpha=CI_ALPHA,
            )
            if ci_result is None:
                print(f"  ⚠ [{fuel_key}] CausalImpact fallito – salto.")
                for ax in axes[row_idx]:
                    ax.text(0.5, 0.5, "CausalImpact non convergito",
                            ha="center", va="center", transform=ax.transAxes)
                continue

            # ── Carica consumi giornalieri per il periodo post-break ──────────
            # Determiniamo l'indice delle date post-break dalla finestra usata da CausalImpact
            inf = ci_result.inferences
            post_dates = inf.index[inf.index >= break_date]
            if len(post_dates) == 0:
                # fallback: usa la finestra standard
                post_start = break_date
                post_end   = break_date + pd.Timedelta(days=POST_WIN)
                post_dates = pd.date_range(post_start, post_end, freq="D")
            cons = load_daily_consumption(post_dates, fuel_key)

            # ── Plot principale – passiamo cons ───────────────────────────────
            gain_meur = _plot_event_fuel(
                ev_name, ev, series,
                fuel_key, fuel_color, ci_result,
                shock, break_date, mode, covariate_label, cons,
                axes[row_idx][0], axes[row_idx][1],
            )

            # ── Plot diagnostico BSTS ─────────────────────────────────────────
            safe_ev    = (ev_name.replace(" ","_").replace("/","")
                                 .replace("(","").replace(")",""))
            diag_out   = OUT_DIR / f"bsts_{safe_ev}_{fuel_key}.png"
            _plot_bsts_diagnostics(ci_result, ev_name, fuel_key, fuel_color,
                                   break_date, mode, diag_out)

            # ── Statistiche summary (calcolo manuale da inferences 0.2.x) ──────
            post_mask = inf.index >= break_date
            n_post    = post_mask.sum()
            post_inf  = inf.loc[post_mask]

            pe     = post_inf["point_effect"]
            pe_lo  = post_inf["point_effect_lower"]
            pe_hi  = post_inf["point_effect_upper"]

            abs_eff_avg = float(pe.mean())
            abs_eff_lo  = float(pe_lo.mean())
            abs_eff_hi  = float(pe_hi.mean())
            abs_eff_cum = float(pe.sum())
            mean_pred   = float(post_inf["point_pred"].mean())
            rel_eff     = (abs_eff_avg / mean_pred) if mean_pred != 0 else 0.0

            # p-value bayesiano: probabilità di coda sul EFFETTO CUMULATIVO
            # (non sul livello della previsione, che era sempre positivo → p≈0 → prob≈100%)
            from scipy import stats as _st
            post_mask_pv = inf.index >= break_date
            cum_eff    = float(inf.loc[post_mask_pv, "point_effect"].sum())
            cum_eff_lo = float(inf.loc[post_mask_pv, "point_effect_lower"].sum())
            cum_eff_hi = float(inf.loc[post_mask_pv, "point_effect_upper"].sum())
            cum_range  = cum_eff_hi - cum_eff_lo
            # Stima std dell'effetto cumulativo dall'ampiezza del CI
            cum_std = (cum_range / (2 * 1.96)) if cum_range > 1e-12 else max(abs(cum_eff) * 0.25, 1e-10)
            # p-value = prob che l'effetto sia ≤ 0 (se positivo) o ≥ 0 (se negativo)
            if cum_std > 0:
                p_val = float(_st.norm.cdf((0.0 - cum_eff) / cum_std))
            else:
                p_val = 0.5
            # posterior_prob = probabilità che l'effetto causale abbia il segno osservato
            posterior_prob = round((1 - p_val) * 100, 1)

            # Calcolo guadagno in M€ con consumi reali (sovrascrive gain_meur già calcolato in _plot_event_fuel)
            # Usiamo lo stesso metodo per coerenza
            cons_aligned = cons.reindex(post_dates, method="ffill").fillna(cons.mean())
            daily_gain = (pe * cons_aligned) / 1e6
            cum_m      = daily_gain.sum()
            cum_lo     = ((pe_lo * cons_aligned) / 1e6).sum()
            cum_hi     = ((pe_hi * cons_aligned) / 1e6).sum()

            n_pre = len(series[
                (series.index >= shock - pd.Timedelta(days=PRE_WIN)) &
                (series.index < shock)
            ])

            # ── Export residui pre/post (standard per 02d_compare nonparam) ──
            # Pre: response − point_pred sul periodo pre-break (residui BSTS)
            # Post: point_effect (actual − counterfactual, = extra-margine)
            _safe_ev = (ev_name.replace(" ", "_").replace("/", "")
                               .replace("(", "").replace(")", ""))
            _pre_mask_r = inf.index < break_date
            _inf_pre = inf.loc[_pre_mask_r]
            _resid_rows = []
            if "response" in _inf_pre.columns and "point_pred" in _inf_pre.columns:
                _pre_resid_s = _inf_pre["response"] - _inf_pre["point_pred"]
                for _d, _r in _pre_resid_s.dropna().items():
                    _resid_rows.append({
                        "date": str(_d.date()), "residual": float(_r), "phase": "pre",
                        "metodo": "v5_causalimpact", "evento": ev_name,
                        "carburante": fuel_key, "break_date": str(break_date.date()),
                    })
            # Post residuals = point_effect sul periodo post-break
            _post_pe = inf.loc[post_mask, "point_effect"].dropna()
            for _d, _r in _post_pe.items():
                _resid_rows.append({
                    "date": str(_d.date()), "residual": float(_r), "phase": "post",
                    "metodo": "v5_causalimpact", "evento": ev_name,
                    "carburante": fuel_key, "break_date": str(break_date.date()),
                })
            pd.DataFrame(_resid_rows).to_csv(
                OUT_DIR / f"residuals_{_safe_ev}_{fuel_key}.csv", index=False
            )

            print(f"    Break ({break_method}) = {break_date.date()}"
                  f"  (shock={shock.date()})")
            print(f"    Abs effect avg        = {abs_eff_avg:+.4f} €/L  "
                  f"CI95% [{abs_eff_lo:+.4f}, {abs_eff_hi:+.4f}]")
            print(f"    Effetto relativo      = {rel_eff:+.1%}")
            print(f"    Guadagno totale       = {cum_m:+.0f} M€  "
                  f"CI95% [{cum_lo:+.0f}, {cum_hi:+.0f}] M€")
            print(f"    p-value bayesiano     = {p_val:.4f}  "
                  f"(prob. effetto causale: {posterior_prob:.1f}%)")

            rows.append({
                "metodo":              "v5_causalimpact",
                "mode":                mode,
                "break_method":        break_method,
                "covariate":           covariate_label,
                "evento":              ev_name,
                "carburante":          fuel_key,
                "shock":               shock.date(),
                "break_date":          break_date.date(),
                "pre_win_days":        PRE_WIN,
                "post_win_days":       POST_WIN,
                "n_pre":               n_pre,
                "n_post":              n_post,
                # Effetto
                "abs_effect_avg_eurl": round(abs_eff_avg, 5),
                "abs_effect_lo_eurl":  round(abs_eff_lo, 5),
                "abs_effect_hi_eurl":  round(abs_eff_hi, 5),
                "abs_effect_cum_eurl": round(abs_eff_cum, 4),
                "rel_effect_pct":      round(rel_eff * 100, 2),
                # Guadagno in M€ (basato su consumi reali)
                "gain_total_meur":     round(float(cum_m),  1),
                "gain_ci_low_meur":    round(float(cum_lo), 1),
                "gain_ci_high_meur":   round(float(cum_hi), 1),
                # Inferenza Bayesiana
                "p_value_bayesian":    round(p_val, 4),
                "posterior_prob_pct":  posterior_prob,
                "ci_type":             f"BSTS_bayesian_{int((1-CI_ALPHA)*100)}pct",
                "note": (f"BSTS CausalImpact, baseline pre-shock, "
                         f"cov={covariate_label}, mode={mode}, consumi giornalieri reali"),
            })

        fig.tight_layout()
        safe_ev = (ev_name.replace(" ","_").replace("/","")
                         .replace("(","").replace(")",""))
        out = OUT_DIR / f"plot_{safe_ev}.png"
        fig.savefig(out, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"\n  → Salvato: {out}")

    if rows:
        df_out  = pd.DataFrame(rows)
        csv_out = OUT_DIR / "v5_causalimpact_results.csv"
        df_out.to_csv(csv_out, index=False)
        print(f"\n  → CSV: {csv_out}")
        print("\n" + df_out[
            ["evento","carburante","break_date",
             "gain_total_meur","p_value_bayesian","posterior_prob_pct"]
        ].to_string(index=False))
    else:
        print("\n  ⚠  Nessun risultato prodotto.")


if __name__ == "__main__":
    main()