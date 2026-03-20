#!/usr/bin/env python3
"""
02d_v2_intermediate.py  ─  Metodo 2: OLS HAC + GLM Poisson
============================================================
Approccio intermedio: corregge i due principali difetti del metodo naïve
e aggiunge un secondo modello di regressione (GLM Poisson) come robustness check.

────────────────────────────────────────────────────────────────────────
MODELLO 1 — OLS con HAC Newey-West  (come v1, ma con SE robuste)
  ─ Baseline     : trend OLS lineare sui PRE_WIN giorni pre-break
  ─ SE           : Newey-West HAC  (corregge autocorr. + eterosch.)
  ─ CI           : intervallo di previsione basato su HAC
  ─ Chow test    : verifica formale di rottura strutturale in τ

MODELLO 2 — GLM Poisson con link logaritmico
  Motivazione:
    Il margine (€/L) è continuo e può essere negativo → Poisson non si
    applica direttamente. Si usa uno shift additivo pre-stimato:
        y_shift = y − min(y_pre) + ε       (ε = 1e-4 €/L)
    Le predizioni vengono poi back-trasformate sottraendo lo shift.
    Questo approccio è usato in letteratura ITS come robustness check
    quando la serie ha varianza proporzionale al livello (eteroschedasticità
    sistematica), che il GLM Poisson gestisce strutturalmente (Var = μ).

  Nota metodologica:
    Il link log implica che l'effetto del trend sul margine shifted è
    moltiplicativo, non additivo. Se il test di overdispersione
    (Pearson χ²/df >> 1) segnala overdispersione, il GLM Poisson è
    comunque usato per il confronto ma la stima OLS HAC è quella preferita.

  Shift:
    shift = max(0, −min(y_pre)) + ε
    Applicato solo alla serie pre (e propagato sul post) per non
    contaminare la stima della baseline con dati post-break.

  Test di overdispersione: Pearson χ²/df sul periodo pre.
    · χ²/df ≈ 1   → equidispersione (Poisson valido)
    · χ²/df >> 1  → overdispersione (usare quasi-Poisson o preferire OLS)

────────────────────────────────────────────────────────────────────────
Break point detection (--mode detected):
  margin    : Window L2 Discrepancy (paper, Eq. 1–2) sul margine distributore
  price     : Window L2 Discrepancy (paper, Eq. 1–2) sul prezzo pompa netto
  Detection autonoma — non dipende da theta_results.csv / 02c.

Output:
  data/plots/its/{mode}/v2_intermediate/              (se mode=fixed)
  data/plots/its/detected/{detect}/v2_intermediate/   (se mode=detected)
    plot_{evento}.png
    v2_intermediate_results.csv
"""

from __future__ import annotations
from pathlib import Path
import argparse
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
    plot_residual_diagnostics,
)
from theta_loader import load_theta

try:
    import statsmodels.api as sm
    from statsmodels.stats.stattools import durbin_watson
    HAS_SM = True
except ImportError:
    HAS_SM = False
    warnings.warn("statsmodels non installato – HAC e GLM Poisson non disponibili. "
                  "pip install statsmodels")

# ── Configurazione ─────────────────────────────────────────────────────────────
BASE_DIR    = Path(__file__).parent
DAILY_CSV   = BASE_DIR / "data" / "processed" / "daily_fuel_prices_all.csv"
GASOIL_CSV  = BASE_DIR / "data" / "Futures" / "London Gas Oil Futures Historical Data.csv"
EUROBOB_CSV = BASE_DIR / "data" / "Futures" / "Eurobob_B7H1_date.csv"
EURUSD_CSV  = BASE_DIR / "data" / "raw" / "eurusd.csv"
_OUT_BASE   = BASE_DIR / "data" / "plots" / "its"

PRE_WIN   = 90    # giorni pre per stimare baseline
POST_WIN  = 30    # giorni post per calcolare l'extra profitto
SEARCH    = 30    # ricerca break ±SEARCH giorni dallo shock (mode=detected)
HALF_WIN  = 30    # semi-finestra del Chow test intorno al break (giorni per lato)
CI_ALPHA  =  0.05   # α → CI/PI al 90%
POISSON_EPS = 1e-4  # epsilon per lo shift Poisson (€/L)

DAILY_CONSUMPTION_L = {
    "benzina": 12_000_000,
    "gasolio": 25_000_000,
}

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

PRICE_COLS: dict[str, str] = {
    "benzina": "benzina_net",
    "gasolio": "gasolio_net",
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
    daily = (pd.read_csv(DAILY_CSV, parse_dates=["date"])
               .sort_values("date").set_index("date"))
    eurusd  = load_eurusd(csv_path=EURUSD_CSV if EURUSD_CSV.exists() else None,
                          start="2015-01-01", end="2026-12-31")
    gasoil  = _load_gasoil_futures(eurusd)
    eurobob = _load_eurobob_futures(eurusd)
    df = daily[["benzina_net","gasolio_net"]].copy()
    df["margin_gasolio"] = df["gasolio_net"] - gasoil.reindex(df.index, method="ffill")
    df["margin_benzina"] = (df["benzina_net"] - eurobob.reindex(df.index, method="ffill")
                            if eurobob is not None else np.nan)
    return df


# ══════════════════════════════════════════════════════════════════════════════
# Break point detection — Window L2 Discrepancy (Paper BLOCCO 1, Eq. 1–2)
# Questo è il metodo canonico del paper (lo stesso usato da 02c per produrre θ),
# ora implementato autonomamente in v2 senza dipendere da theta_results.csv.
#
# Per ogni candidato v ∈ [u + min_seg, w − min_seg]:
#   d(y_uv, y_vw) = c(y_uw) − c(y_uv) − c(y_vw)
#   dove c(y_I) = Σ_{t∈I} ||y_t − ȳ||²  (costo L2 intra-finestra)
# θ = argmax_v  d(...)
# Pre-processing: moving average a 7 giorni per ridurre il rumore.
# ══════════════════════════════════════════════════════════════════════════════

MA_WIN   = 7    # finestra moving average per pre-processing (Paper)
MIN_SEG  = 14   # segmento minimo per lato (giorni)


def _l2_cost(y: np.ndarray) -> float:
    """c(y_I) = Σ ||y_t − ȳ||²   (Paper, Eq. 2)"""
    if len(y) < 2:
        return 0.0
    return float(np.sum((y - y.mean()) ** 2))


def detect_breakpoint(series: pd.Series, shock: pd.Timestamp) -> dict:
    """
    Window L2 Discrepancy (Paper BLOCCO 1, Eq. 1–2).
    Cerca τ in [shock-SEARCH, shock+SEARCH].

    Selezione: argmax della discrepanza d(v) — il picco coincide con il
    punto in cui la serie è massimamente disomogenea tra i due segmenti.
    Selezione del primo candidato nel top-quartile per catturare l'inizio
    della rottura strutturale.
    """
    mask = (series.index >= shock - pd.Timedelta(days=SEARCH)) & \
           (series.index <= shock + pd.Timedelta(days=SEARCH))
    win = series[mask].dropna()

    fallback = {"tau": shock, "d_max": 0.0, "method": "window_l2_nofound",
                "_df": pd.DataFrame()}

    if len(win) < 2 * MIN_SEG + MA_WIN:
        return fallback

    # Pre-processing: smoothing MA-7
    ma  = win.rolling(MA_WIN, center=True, min_periods=1).mean()
    y   = ma.values
    n   = len(y)
    c_uw = _l2_cost(y)

    rows = []
    for v in range(MIN_SEG, n - MIN_SEG):
        d = c_uw - _l2_cost(y[:v]) - _l2_cost(y[v:])
        rows.append({"tau": ma.index[v], "d": d})

    if not rows:
        return {**fallback}

    df_c = pd.DataFrame(rows)
    threshold = float(np.percentile(df_c["d"], 75))
    top = df_c[df_c["d"] >= threshold]
    if top.empty:
        top = df_c
    best = top.sort_values("tau").iloc[0]

    return {
        "tau":    best["tau"],
        "d_max":  round(float(df_c["d"].max()), 6),
        "method": "window_l2_discrepancy",
        "_df":    df_c,
    }


# ══════════════════════════════════════════════════════════════════════════════
# Diagnostic plot — finestra di selezione del break (mode=detected)
# ══════════════════════════════════════════════════════════════════════════════

def _plot_detection_window(
    detect_series: pd.Series,
    shock: pd.Timestamp,
    df_cands: pd.DataFrame,
    threshold: float,
    selected_tau: pd.Timestamp,
    fuel_key: str,
    fuel_color: str,
    ev_name: str,
    detect_target: str,
    out_path: Path,
) -> None:
    """Plot diagnostico della finestra di detection (Window L2 Discrepancy)."""
    if df_cands.empty:
        return

    # La colonna della metrica è "d" (discrepanza L2)
    metric_col   = "d"
    metric_label = "d(y_uv, y_vw)  [discrepanza L2]"

    win_start  = shock - pd.Timedelta(days=SEARCH + 15)
    win_end    = shock + pd.Timedelta(days=SEARCH + 15)
    series_win = detect_series[(detect_series.index >= win_start) &
                               (detect_series.index <= win_end)]

    fig, (ax_s, ax_m) = plt.subplots(2, 1, figsize=(12, 7),
                                     gridspec_kw={"height_ratios": [1.6, 1]})
    fig.suptitle(
        f"[v2 HAC+Poisson – detection finestra]  {ev_name}  ·  {fuel_key.capitalize()}\n"
        f"Serie: {'prezzo pompa netto' if detect_target == 'price' else 'margine dist.'}"
        f"  |  Shock: {shock.date()}  |  Break scelto: {selected_tau.date()}"
        f"  ({(selected_tau - shock).days:+d}gg)",
        fontsize=9, fontweight="bold"
    )

    ax_s.plot(series_win.index, series_win.values,
              color="grey", lw=0.9, alpha=0.7, label=detect_target)
    is_top = df_cands[metric_col] >= threshold
    for _, row in df_cands[~is_top].iterrows():
        ax_s.axvline(row["tau"], color="#cccccc", lw=0.4, alpha=0.5)
    for _, row in df_cands[is_top].iterrows():
        ax_s.axvline(row["tau"], color=fuel_color, lw=0.6, alpha=0.35)
    ax_s.axvline(shock, color="#e74c3c", lw=1.4, ls="--",
                 label=f"Shock {shock.date()}")
    ax_s.axvline(selected_tau, color=fuel_color, lw=2.2,
                 label=f"Break scelto {selected_tau.date()} ({(selected_tau-shock).days:+d}gg)")

    y_rng = series_win.max() - series_win.min() if not series_win.empty else 1
    y_top = series_win.max() + 0.05 * y_rng if not series_win.empty else 1
    ax_s.annotate(
        f"τ = {selected_tau.date()}\n({(selected_tau-shock).days:+d}gg dallo shock)",
        xy=(selected_tau, y_top),
        xytext=(10, 0), textcoords="offset points",
        fontsize=7.5, color=fuel_color, fontweight="bold",
        arrowprops=dict(arrowstyle="-", color=fuel_color, lw=0.8),
    )
    ax_s.set_ylabel("€/L", fontsize=8)
    ax_s.legend(fontsize=7, loc="upper left")
    ax_s.grid(axis="y", alpha=0.20)
    ax_s.xaxis.set_major_formatter(mdates.DateFormatter("%d %b %y"))
    ax_s.xaxis.set_major_locator(mdates.WeekdayLocator(byweekday=0, interval=2))
    plt.setp(ax_s.xaxis.get_majorticklabels(), rotation=30, ha="right", fontsize=7)

    taus   = pd.to_datetime(df_cands["tau"])
    metric = df_cands[metric_col].values
    ax_m.fill_between(taus, metric, threshold,
                      where=(metric >= threshold),
                      alpha=0.18, color=fuel_color, label="Top quartile (≥ 75°p)")
    ax_m.plot(taus, metric, color="steelblue", lw=1.0, zorder=3)
    ax_m.axhline(threshold, color="darkorange", lw=1.0, ls="--",
                 label=f"Soglia 75° pct = {threshold:.3f}")
    ax_m.axvline(shock, color="#e74c3c", lw=1.2, ls="--", alpha=0.7)
    ax_m.axvline(selected_tau, color=fuel_color, lw=1.8, alpha=0.9)

    match   = df_cands[df_cands["tau"] == selected_tau]
    sel_val = float(match[metric_col].iloc[0]) if not match.empty else threshold
    ax_m.scatter([selected_tau], [sel_val], marker="*", s=180,
                 color=fuel_color, zorder=5, label="Scelto (primo nel top quartile)")
    ax_m.set_ylabel(metric_label, fontsize=8)
    ax_m.set_xlabel("τ candidato", fontsize=8)
    ax_m.legend(fontsize=7, loc="upper left")
    ax_m.grid(axis="y", alpha=0.20)
    ax_m.xaxis.set_major_formatter(mdates.DateFormatter("%d %b %y"))
    ax_m.xaxis.set_major_locator(mdates.WeekdayLocator(byweekday=0, interval=2))
    plt.setp(ax_m.xaxis.get_majorticklabels(), rotation=30, ha="right", fontsize=7)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"    → Detection window plot: {out_path}")


# ══════════════════════════════════════════════════════════════════════════════
# MODELLO 1 — OLS con HAC (Newey-West)
# ══════════════════════════════════════════════════════════════════════════════

def _hac_maxlags(n: int) -> int:
    return max(1, int(np.floor(4 * (n / 100) ** (2/9))))


def fit_ols_hac(series: pd.Series, break_date: pd.Timestamp, shock: pd.Timestamp) -> dict | None:
    """
    OLS lineare sui PRE_WIN giorni pre-break (o pre-shock se break > shock)
    con errori standard Newey-West HAC.
    Restituisce un dizionario con coefficienti, R², DW, residui e design matrix.
    """
    anchor = shock if break_date > shock else break_date
    pre = series[
        (series.index >= anchor - pd.Timedelta(days=PRE_WIN)) &
        (series.index < anchor)
    ].dropna()
    if len(pre) < 10:
        return None

    x = np.array([(d - anchor).days for d in pre.index], dtype=float)

    if not HAS_SM:
        slope, intercept, r, *_ = stats.linregress(x, pre.values)
        return dict(slope=slope, intercept=intercept, r2=float(r**2),
                    anchor=anchor, pre=pre, hac=False,
                    mse=float(np.var(pre.values - (slope*x + intercept), ddof=2)))

    X       = sm.add_constant(x)
    model   = sm.OLS(pre.values, X)
    fit     = model.fit()
    maxlags = _hac_maxlags(len(pre))
    fit_hac = fit.get_robustcov_results(cov_type="HAC", maxlags=maxlags)
    dw      = durbin_watson(fit.resid)

    return dict(
        fit_hac=fit_hac,
        fit_ols=fit,
        slope=float(fit_hac.params[1]),
        intercept=float(fit_hac.params[0]),
        r2=float(fit.rsquared),
        anchor=anchor, pre=pre,
        hac=True, dw=dw, maxlags=maxlags,
        mse=float(fit.mse_resid),
        residuals=fit.resid,
        X_bg=X,
    )


def project_hac(info: dict, post_index: pd.DatetimeIndex
                ) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Proietta la baseline OLS HAC sul periodo post-break con PI al (1-CI_ALPHA)."""
    anchor  = info["anchor"]
    x_post  = np.array([(d - anchor).days for d in post_index], dtype=float)
    baseline = info["slope"] * x_post + info["intercept"]

    if info["hac"] and HAS_SM:
        cov      = info["fit_hac"].cov_params()
        X_post   = np.column_stack([np.ones(len(x_post)), x_post])
        var_param = np.einsum("ni,ij,nj->n", X_post, cov, X_post)
        var_pred  = var_param + info["mse"]
        se_pred   = np.sqrt(np.maximum(var_pred, 0))
        t_crit    = stats.norm.ppf(1 - CI_ALPHA / 2)
    else:
        n   = len(info["pre"])
        x_p = np.array([(d - anchor).days for d in info["pre"].index], dtype=float)
        sxx = np.sum((x_p - x_p.mean())**2)
        se_pred = np.sqrt(info["mse"] * (1 + 1/n + (x_post - x_p.mean())**2 / sxx))
        t_crit  = stats.t.ppf(1 - CI_ALPHA / 2, df=n - 2)

    return (
        pd.Series(baseline,                  index=post_index),
        pd.Series(baseline - t_crit*se_pred, index=post_index),
        pd.Series(baseline + t_crit*se_pred, index=post_index),
    )


# ══════════════════════════════════════════════════════════════════════════════
# MODELLO 2 — GLM Poisson con link log
# ══════════════════════════════════════════════════════════════════════════════

def _compute_shift(pre: pd.Series, eps: float = POISSON_EPS) -> float:
    """
    Shift additivo minimo per portare la serie pre nel dominio (0, +∞).
    shift = max(0, −min(y_pre)) + eps
    Con shift > 0 tutti i valori diventano > 0; il link log è applicabile.
    """
    min_val = float(pre.min())
    return max(0.0, -min_val) + eps


def fit_glm_poisson(
    series: pd.Series,
    break_date: pd.Timestamp,
    shock: pd.Timestamp,
) -> dict | None:
    """
    Stima un GLM Poisson (link log) sul periodo pre-break.

    Procedura:
      1. Estrae la finestra pre: [anchor − PRE_WIN, anchor)
      2. Calcola lo shift = max(0, −min_pre) + ε  →  y_shifted = y + shift > 0
      3. Stima GLM Poisson: log(E[y_shifted]) = β₀ + β₁·t
         dove t = giorni dall'anchor
      4. Test di overdispersione: Pearson χ²/df
         · ≈ 1  →  equidispersione, Poisson coerente
         · >> 1 →  overdispersione (quasi-Poisson preferibile, ma non altera il β)

    Returns
    -------
    dict con shift, params, cov, pearson_dispersion, aic, n_pre, anchor, pre,
    o None se statsmodels non è disponibile o dati insufficienti.
    """
    if not HAS_SM:
        return None

    anchor = shock if break_date > shock else break_date
    pre = series[
        (series.index >= anchor - pd.Timedelta(days=PRE_WIN)) &
        (series.index < anchor)
    ].dropna()

    if len(pre) < 10:
        return None

    shift    = _compute_shift(pre)
    y_shifted = pre.values + shift               # tutti > 0

    x = np.array([(d - anchor).days for d in pre.index], dtype=float)
    X = sm.add_constant(x)

    try:
        glm_model = sm.GLM(
            y_shifted, X,
            family=sm.families.Poisson(link=sm.families.links.Log()),
        )
        glm_fit = glm_model.fit(maxiter=200)
    except Exception as e:
        warnings.warn(f"GLM Poisson fit fallito: {e}")
        return None

    # ── Test di overdispersione: Pearson χ²/df ────────────────────────────────
    # χ²/df = Σ[(y_i − μ̂_i)²/μ̂_i] / (n − p)
    mu_hat   = glm_fit.fittedvalues                # μ̂ sulla scala originale shifted
    pearson_chi2 = float(np.sum((y_shifted - mu_hat)**2 / mu_hat))
    df_resid     = len(pre) - 2                    # p = 2 parametri (intercetta + trend)
    pearson_disp = pearson_chi2 / df_resid if df_resid > 0 else np.nan

    # ── Residui sulla scala originale (non shifted) per diagnostica ───────────
    resid_raw = pre.values - (mu_hat - shift)

    return dict(
        glm_fit          = glm_fit,
        shift            = shift,
        params           = glm_fit.params,        # [β₀, β₁] scala log
        cov              = glm_fit.cov_params(),
        pearson_chi2     = pearson_chi2,
        pearson_disp     = pearson_disp,
        aic              = float(glm_fit.aic),
        deviance         = float(glm_fit.deviance),
        null_deviance    = float(glm_fit.null_deviance),
        anchor           = anchor,
        pre              = pre,
        y_shifted        = y_shifted,
        mu_hat_pre       = mu_hat,
        residuals        = resid_raw,              # residui su scala originale
        overdispersed    = bool(pearson_disp > 2.0),  # soglia indicativa
    )


def project_glm_poisson(
    info_glm: dict,
    post_index: pd.DatetimeIndex,
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """
    Proietta la baseline GLM Poisson sul periodo post-break.

    Predizione sulla scala shifted: μ̂(t) = exp(β₀ + β₁·t)
    Back-trasformazione: baseline(t) = μ̂(t) − shift

    Intervallo di confidenza al (1−CI_ALPHA) via delta method:
      Var[log μ̂] = x'Cov(β̂)x
      SE[log μ̂] = sqrt(Var[log μ̂])
      CI: exp(log μ̂ ± z·SE[log μ̂]) − shift
    """
    anchor  = info_glm["anchor"]
    shift   = info_glm["shift"]
    params  = info_glm["params"]
    cov     = info_glm["cov"]
    z_crit  = stats.norm.ppf(1 - CI_ALPHA / 2)

    x_post  = np.array([(d - anchor).days for d in post_index], dtype=float)
    X_post  = np.column_stack([np.ones(len(x_post)), x_post])

    log_mu   = X_post @ params                             # predizione log-scala
    var_log  = np.einsum("ni,ij,nj->n", X_post, cov, X_post)
    se_log   = np.sqrt(np.maximum(var_log, 0))

    mu_hat   = np.exp(log_mu)                              # scala shifted
    baseline = mu_hat - shift                              # scala originale

    ci_low_shifted  = np.exp(log_mu - z_crit * se_log)
    ci_high_shifted = np.exp(log_mu + z_crit * se_log)
    ci_low  = ci_low_shifted  - shift
    ci_high = ci_high_shifted - shift

    return (
        pd.Series(baseline, index=post_index),
        pd.Series(ci_low,   index=post_index),
        pd.Series(ci_high,  index=post_index),
    )


# ══════════════════════════════════════════════════════════════════════════════
# Chow test
# ══════════════════════════════════════════════════════════════════════════════

def chow_test(series: pd.Series, tau: pd.Timestamp) -> dict:
    pre  = series[(series.index >= tau - pd.Timedelta(days=HALF_WIN)) &
                  (series.index < tau)].dropna()
    post = series[(series.index >= tau) &
                  (series.index < tau + pd.Timedelta(days=HALF_WIN))].dropna()

    if len(pre) < 5 or len(post) < 5:
        return {"F": np.nan, "p": np.nan, "note": "dati insufficienti"}

    def _ols_ssr(y, x_days):
        sl, ic, *_ = stats.linregress(x_days, y)
        resid = y - (sl * x_days + ic)
        return float(np.sum(resid**2))

    x_pre  = np.array([(d - tau).days for d in pre.index],  dtype=float)
    x_post = np.array([(d - tau).days for d in post.index], dtype=float)
    combined = pd.concat([pre, post])
    x_comb   = np.array([(d - tau).days for d in combined.index], dtype=float)

    ssr_r = _ols_ssr(combined.values, x_comb)
    ssr_u = _ols_ssr(pre.values, x_pre) + _ols_ssr(post.values, x_post)

    k, n = 2, len(combined)
    if ssr_u < 1e-14 or n <= 2*k:
        return {"F": np.nan, "p": np.nan, "note": "degenerate"}

    F = ((ssr_r - ssr_u) / k) / (ssr_u / (n - 2*k))
    p = float(stats.f.sf(F, dfn=k, dfd=n - 2*k))
    return {"F": float(F), "p": p, "note": ""}


# ══════════════════════════════════════════════════════════════════════════════
# Plot principale — confronto OLS HAC vs GLM Poisson
# ══════════════════════════════════════════════════════════════════════════════

def _plot_event_fuel(
    ev_name: str, ev: dict,
    series: pd.Series,
    fuel_key: str, fuel_color: str,
    break_date: pd.Timestamp, delta: float, p_bonf: float,
    info_ols: dict, baseline_ols: pd.Series,
    ci_low_ols: pd.Series, ci_high_ols: pd.Series,
    extra_ols: pd.Series, gain_ols: float,
    info_glm: dict | None,
    baseline_glm: pd.Series | None,
    ci_low_glm: pd.Series | None, ci_high_glm: pd.Series | None,
    extra_glm: pd.Series | None, gain_glm: float | None,
    chow: dict, mode: str, break_method: str,
    ax_main: plt.Axes, ax_gain: plt.Axes,
) -> None:
    shock = ev["shock"]
    win = series[
        (series.index >= shock - pd.Timedelta(days=PRE_WIN)) &
        (series.index <= shock + pd.Timedelta(days=POST_WIN))
    ].dropna()

    hac_label = (f"OLS HAC NW (lag={info_ols.get('maxlags','?')})"
                 if info_ols.get("hac") else "OLS i.i.d.")
    sig   = "★" if (not np.isnan(p_bonf) and p_bonf < 0.05) else ""
    chow_str = (f"Chow F={chow['F']:.2f} p={chow['p']:.3f}"
                if not np.isnan(chow.get("F", np.nan)) else "Chow: n/a")
    mode_str  = (f"τ={break_date.date()} ({break_method})" if mode == "detected"
                 else f"Break=shock ({shock.date()})")

    # ── Serie effettiva ───────────────────────────────────────────────────────
    ax_main.plot(win.index, win.values, color=fuel_color, lw=1.0, zorder=3,
                 label=f"{fuel_key.capitalize()} effettivo")

    # ── Baseline OLS HAC: fitted nel pre + proiezione nel post ───────────────
    _anchor_ols = info_ols["anchor"]
    _pre_idx    = info_ols["pre"].index
    _x_pre_ols  = np.array([(d - _anchor_ols).days for d in _pre_idx], dtype=float)
    _pre_fit_ols = info_ols["slope"] * _x_pre_ols + info_ols["intercept"]
    # unisci pre fitted + post projection in una linea continua
    _full_idx_ols = _pre_idx.append(baseline_ols.index)
    _full_val_ols = np.concatenate([_pre_fit_ols, baseline_ols.values])
    ax_main.plot(_full_idx_ols, _full_val_ols,
                 color="dimgrey", lw=1.4, ls="--",
                 label=f"Baseline {hac_label} (R²={info_ols['r2']:.2f})")
    ax_main.fill_between(ci_low_ols.index, ci_low_ols.values, ci_high_ols.values,
                         alpha=0.12, color="grey",
                         label=f"PI {int((1-CI_ALPHA)*100)}% OLS HAC")

    # ── Baseline GLM Poisson: fitted nel pre + proiezione nel post ────────────
    if baseline_glm is not None and info_glm is not None:
        glm_color   = "#2980b9"
        disp_tag    = (f"χ²/df={info_glm['pearson_disp']:.2f}"
                       f"{'  ⚠overdispersione' if info_glm['overdispersed'] else ''}")
        _anchor_glm = info_glm["anchor"]
        _pre_idx_g  = info_glm["pre"].index
        _x_pre_g    = np.array([(d - _anchor_glm).days for d in _pre_idx_g], dtype=float)
        _X_pre_g    = np.column_stack([np.ones(len(_x_pre_g)), _x_pre_g])
        _pre_fit_glm = np.exp(_X_pre_g @ info_glm["params"]) - info_glm["shift"]
        _full_idx_glm = _pre_idx_g.append(baseline_glm.index)
        _full_val_glm = np.concatenate([_pre_fit_glm, baseline_glm.values])
        ax_main.plot(_full_idx_glm, _full_val_glm,
                     color=glm_color, lw=1.4, ls="-.",
                     label=f"Baseline GLM Poisson ({disp_tag})")
        ax_main.fill_between(ci_low_glm.index, ci_low_glm.values, ci_high_glm.values,
                             alpha=0.10, color=glm_color,
                             label=f"CI {int((1-CI_ALPHA)*100)}% GLM Poisson")

    # ── Extra profitto OLS (ombreggiatura) ────────────────────────────────────
    ax_main.fill_between(extra_ols.index,
                         win.reindex(extra_ols.index), baseline_ols.values,
                         where=(extra_ols >= 0), alpha=0.18, color="green",
                         label="Extra OLS (≥0)")
    ax_main.fill_between(extra_ols.index,
                         win.reindex(extra_ols.index), baseline_ols.values,
                         where=(extra_ols < 0), alpha=0.18, color="red",
                         label="Sotto-baseline OLS (<0)")

    # ── Linee verticali ───────────────────────────────────────────────────────
    ax_main.axvline(shock, color=ev["color"], lw=1.6, ls="--",
                    label=f"Shock ({shock.date()})")
    if mode == "detected":
        ax_main.axvline(break_date, color=fuel_color, lw=1.2, ls=":",
                        label=f"τ={break_date.date()} Δ={delta:+.3f}{sig}")

    ax_main.set_title(
        f"[V2-HAC+Poisson / mode={mode}]  {fuel_key.capitalize()} – {ev_name}\n"
        f"{mode_str}  |  {chow_str}",
        fontsize=8, fontweight="bold"
    )
    ax_main.set_ylabel("Margine (€/L)", fontsize=8)
    ax_main.legend(fontsize=6, loc="upper left", ncol=2)
    ax_main.grid(axis="y", alpha=0.20)
    ax_main.xaxis.set_major_formatter(mdates.DateFormatter("%d %b %y"))
    ax_main.xaxis.set_major_locator(mdates.WeekdayLocator(byweekday=0, interval=2))
    plt.setp(ax_main.xaxis.get_majorticklabels(), rotation=35, ha="right", fontsize=7)

    # ── Zoom y-axis: ignora i CI ampi, mostra solo dati + baseline ───────────
    _y_data = [win.values, _full_val_ols]
    if baseline_glm is not None:
        _y_data.append(_full_val_glm)
    _y_all  = np.concatenate([v for v in _y_data if len(v) > 0])
    _y_all  = _y_all[np.isfinite(_y_all)]
    if len(_y_all) > 0:
        _ymin, _ymax = float(np.nanmin(_y_all)), float(np.nanmax(_y_all))
        _pad = max(abs(_ymax - _ymin) * 0.25, 0.005)
        ax_main.set_ylim(_ymin - _pad, _ymax + _pad)

    # ── Pannello guadagno cumulato: OLS vs Poisson ────────────────────────────
    cum_ols = (extra_ols * DAILY_CONSUMPTION_L[fuel_key] / 1e6).cumsum()
    ax_gain.plot(cum_ols.index, cum_ols.values,
                 color="dimgrey", lw=1.2, ls="--",
                 label=f"OLS HAC → {gain_ols:+.0f} M€")

    if extra_glm is not None and gain_glm is not None:
        cum_glm = (extra_glm * DAILY_CONSUMPTION_L[fuel_key] / 1e6).cumsum()
        ax_gain.plot(cum_glm.index, cum_glm.values,
                     color="#2980b9", lw=1.2, ls="-.",
                     label=f"GLM Poisson → {gain_glm:+.0f} M€")

    ax_gain.axhline(0, color="grey", lw=0.7, ls="--")
    ax_gain.fill_between(cum_ols.index, cum_ols.values, 0,
                         where=(cum_ols >= 0), alpha=0.20, color="green")
    ax_gain.fill_between(cum_ols.index, cum_ols.values, 0,
                         where=(cum_ols < 0), alpha=0.20, color="red")
    ax_gain.axvline(break_date, color=fuel_color, lw=1.0, ls=":", alpha=0.7)

    post_days = len(extra_ols)
    ax_gain.set_title(
        f"Guadagno extra cumulato ({post_days}gg post-break)\n"
        f"[{DAILY_CONSUMPTION_L[fuel_key]/1e6:.0f} ML/giorno]",
        fontsize=7
    )
    ax_gain.set_ylabel("M€ cumulati", fontsize=8)
    ax_gain.legend(fontsize=7, loc="upper left")
    ax_gain.grid(axis="y", alpha=0.20)
    ax_gain.xaxis.set_major_formatter(mdates.DateFormatter("%d %b %y"))
    ax_gain.xaxis.set_major_locator(mdates.WeekdayLocator(byweekday=0, interval=2))
    plt.setp(ax_gain.xaxis.get_majorticklabels(), rotation=35, ha="right", fontsize=7)


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    parser = argparse.ArgumentParser(description="V2 OLS HAC + GLM Poisson – ITS pipeline")
    parser.add_argument("--mode", choices=["fixed", "detected"], default="fixed",
                        help="fixed = usa shock date; detected = sliding t-test AR(1)")
    parser.add_argument("--detect", choices=["margin", "price"], default="margin",
                        help="(solo mode=detected) serie di detection: "
                             "margin = sliding t-test AR(1) sul margine [default], "
                             "price  = sliding t-test AR(1) sul prezzo pompa netto")
    args, _ = parser.parse_known_args()
    mode          = args.mode
    detect_target = args.detect

    if mode == "detected":
        OUT_DIR = _OUT_BASE / "detected" / detect_target / "v2_intermediate"
    else:
        OUT_DIR = _OUT_BASE / mode / "v2_intermediate"
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("═"*70)
    print(f"  02d_v2_intermediate.py  –  OLS HAC + GLM Poisson  [mode={mode}]")
    if mode == "fixed":
        print("  Break = shock date hardcodata (nessuna detection)")
    else:
        print(f"  Break = sliding t-test AR(1)  (ricerca ±{SEARCH}gg, finestra {HALF_WIN}gg)")
        print(f"  Detection su: {'MARGINE' if detect_target == 'margin' else detect_target.upper()}")
    print(f"  Finestra: PRE={PRE_WIN}gg / POST={POST_WIN}gg dal break point")
    print(f"  statsmodels: {'OK (OLS HAC + GLM Poisson abilitati)' if HAS_SM else 'MANCANTE – fallback i.i.d., Poisson disabilitato'}")
    print(f"  GLM Poisson shift ε = {POISSON_EPS} €/L")
    print(f"  Output: {OUT_DIR}")
    print("═"*70)

    data = load_margin_data()
    rows: list[dict] = []

    for ev_name, ev in EVENTS.items():
        shock = ev["shock"]

        fig, axes = plt.subplots(len(FUELS), 2,
                                 figsize=(15, 5 * len(FUELS)),
                                 squeeze=False)
        fig.suptitle(
            f"[Metodo 2 – OLS HAC + GLM Poisson / mode={mode}]  {ev_name}\n{ev['label']}",
            fontsize=11, fontweight="bold"
        )

        for row_idx, (fuel_key, (col_name, fuel_color)) in enumerate(FUELS.items()):
            series = data[col_name].dropna()

            # ── Determina break date ─────────────────────────────────────────
            if mode == "detected":
                # Carica θ canonico (GLM Poisson) da 02c_change_point_detection.py
                theta = load_theta(ev_name, fuel_key, detect_target,
                                   base_dir=BASE_DIR)
                if theta is not None:
                    break_date   = theta
                    break_method = "glm_poisson_02c"
                else:
                    print(f"  ⚠ [{fuel_key}] θ non trovato — uso shock come fallback.")
                    break_date   = shock
                    break_method = "fallback_shock"
                d_max  = np.nan
                delta  = 0.0
                p_bonf = np.nan
                t_stat = np.nan
            else:
                break_date   = shock
                d_max        = np.nan
                delta        = 0.0
                p_bonf       = np.nan
                t_stat       = np.nan
                break_method = "fixed_at_shock"

            # ── MODELLO 1: OLS HAC ────────────────────────────────────────────
            info_ols = fit_ols_hac(series, break_date, shock)
            if info_ols is None:
                print(f"  [{fuel_key}] fit OLS fallito – salto.")
                continue

            post = series[
                (series.index >= break_date) &
                (series.index < shock + pd.Timedelta(days=POST_WIN))
            ].dropna()
            if len(post) < 5:
                print(f"  [{fuel_key}] dati post-break insufficienti – salto.")
                continue

            baseline_ols, ci_low_ols, ci_high_ols = project_hac(info_ols, post.index)
            extra_ols    = post - baseline_ols
            gain_ols     = float(extra_ols.sum() * DAILY_CONSUMPTION_L[fuel_key] / 1e6)
            gain_ci_low  = float((post - ci_high_ols).sum() * DAILY_CONSUMPTION_L[fuel_key] / 1e6)
            gain_ci_high = float((post - ci_low_ols).sum()  * DAILY_CONSUMPTION_L[fuel_key] / 1e6)

            chow = chow_test(series, break_date)

            # ── MODELLO 2: GLM Poisson ────────────────────────────────────────
            info_glm = fit_glm_poisson(series, break_date, shock)
            baseline_glm = ci_low_glm = ci_high_glm = extra_glm = None
            gain_glm = glm_disp = glm_aic = np.nan
            glm_overdispersed = False

            if info_glm is not None:
                baseline_glm, ci_low_glm, ci_high_glm = project_glm_poisson(info_glm, post.index)
                extra_glm         = post - baseline_glm
                gain_glm          = float(extra_glm.sum() * DAILY_CONSUMPTION_L[fuel_key] / 1e6)
                glm_disp          = info_glm["pearson_disp"]
                glm_aic           = info_glm["aic"]
                glm_overdispersed = info_glm["overdispersed"]

            # ── Plot ──────────────────────────────────────────────────────────
            _plot_event_fuel(
                ev_name, ev, series, fuel_key, fuel_color,
                break_date, delta, p_bonf,
                info_ols, baseline_ols, ci_low_ols, ci_high_ols,
                extra_ols, gain_ols,
                info_glm, baseline_glm, ci_low_glm, ci_high_glm,
                extra_glm, gain_glm,
                chow, mode, break_method,
                axes[row_idx][0], axes[row_idx][1],
            )

            # ── Diagnostics residui OLS pre-periodo ───────────────────────────
            pre_resid = np.asarray(info_ols.get("residuals", []))
            X_bg      = info_ols.get("X_bg")
            diag = run_diagnostic_tests(pre_resid, x_for_bg=X_bg, n_lags=None)

            safe_ev = ev_name.replace(" ","_").replace("/","").replace("(","").replace(")","")
            diag_plot_out = OUT_DIR / f"diag_{safe_ev}_{fuel_key}.png"
            plot_residual_diagnostics(
                resid=pre_resid,
                dates=info_ols["pre"].index,
                title=(f"[V2-HAC] Diagnostica residui OLS pre-periodo\n"
                       f"{ev_name} · {fuel_key.capitalize()}  "
                       f"(break={break_date.date()})"),
                out_path=diag_plot_out,
                diag_stats=diag,
            )

            # ── Stampa a video ────────────────────────────────────────────────
            print(f"\n  {ev_name}  [{fuel_key.upper()}]")
            print(f"    Break ({break_method}) = {break_date.date()}  (shock={shock.date()})")
            if mode == "detected":
                print(f"    L2 d_max          = {d_max:.6f}  (Window L2 Discrepancy)")
            print(f"    OLS R²            = {info_ols['r2']:.3f}   "
                  f"DW = {info_ols.get('dw', float('nan')):.2f}")
            if not np.isnan(chow.get("F", np.nan)):
                print(f"    Chow test         = F={chow['F']:.2f}  p={chow['p']:.3f}  "
                      f"({'break confermato' if chow['p'] < 0.05 else 'break non confermato'})")
            print(f"    Guadagno OLS HAC  = {gain_ols:+.0f} M€  "
                  f"CI90% [{gain_ci_low:+.0f}, {gain_ci_high:+.0f}] M€")
            if info_glm is not None:
                disp_warn = "  ⚠ overdispersione" if glm_overdispersed else ""
                print(f"    Guadagno Poisson  = {gain_glm:+.0f} M€  "
                      f"(shift={info_glm['shift']:.4f} €/L"
                      f"  χ²/df={glm_disp:.2f}{disp_warn}"
                      f"  AIC={glm_aic:.1f})")
            if not np.isnan(diag.get("sw_p", np.nan)):
                print(f"    SW (normalità)    = W={diag['sw_stat']:.3f}  "
                      f"p={diag['sw_p']:.3f}  "
                      f"{'OK' if diag['sw_p'] > 0.05 else '⚠ non normal.'}")
            if not np.isnan(diag.get("lb_p", np.nan)):
                print(f"    LB({diag['n_lags']}) (autocorr.) = "
                      f"Q={diag['lb_stat']:.2f}  p={diag['lb_p']:.3f}  "
                      f"{'OK' if diag['lb_p'] > 0.05 else '⚠ autocorr.'}")
            if not np.isnan(diag.get("bg_p", np.nan)):
                print(f"    BG({diag['n_lags']}) (autocorr.) = "
                      f"LM={diag['bg_stat']:.2f}  p={diag['bg_p']:.3f}  "
                      f"{'OK' if diag['bg_p'] > 0.05 else '⚠ autocorr.'}")

            rows.append({
                "metodo":            "v2_intermediate",
                "mode":              mode,
                "detect_target":     detect_target if mode == "detected" else "fixed",
                "evento":            ev_name,
                "carburante":        fuel_key,
                "shock":             shock.date(),
                "break_date":        break_date.date(),
                "break_method":      break_method,
                "detection_algo":    "window_l2_discrepancy",
                "l2_d_max":          round(d_max, 6) if not np.isnan(d_max) else np.nan,
                "pre_win_days":      PRE_WIN,
                "post_win_days":     POST_WIN,
                "n_pre":             len(info_ols["pre"]),
                "n_post":            len(post),
                # OLS HAC
                "extra_mean_eurl":   round(float(extra_ols.mean()), 5),
                "extra_sum_eurl":    round(float(extra_ols.sum()), 4),
                "gain_ols_meur":     round(gain_ols, 1),
                "gain_ci_low_meur":  round(gain_ci_low, 1),
                "gain_ci_high_meur": round(gain_ci_high, 1),
                "r2_ols":            round(info_ols["r2"], 4),
                "slope_eurl_day":    round(info_ols["slope"], 6),
                "t_stat_ar1":        np.nan,   # non calcolato da L2
                "p_bonf_ar1":        np.nan,   # non calcolato da L2
                "chow_F":            round(chow.get("F", np.nan), 3),
                "chow_p":            round(chow.get("p", np.nan), 4),
                "dw_stat":           round(info_ols.get("dw", np.nan), 3),
                "hac_maxlags":       info_ols.get("maxlags", np.nan),
                "ci_type_ols":       (f"HAC_NW_{int((1-CI_ALPHA)*100)}pct"
                                      if HAS_SM else f"OLS_iid_{int((1-CI_ALPHA)*100)}pct"),
                # GLM Poisson
                "gain_poisson_meur": round(gain_glm, 1) if not np.isnan(gain_glm) else np.nan,
                "poisson_shift_eurl": round(info_glm["shift"], 5) if info_glm else np.nan,
                "poisson_pearson_disp": round(glm_disp, 3) if not np.isnan(glm_disp) else np.nan,
                "poisson_overdispersed": glm_overdispersed,
                "poisson_aic":       round(glm_aic, 2) if not np.isnan(glm_aic) else np.nan,
                "poisson_available": info_glm is not None,
                # Diagnostics OLS residui
                "sw_stat":           round(diag.get("sw_stat", np.nan), 4),
                "sw_p":              round(diag.get("sw_p", np.nan), 4),
                "lb_stat":           round(diag.get("lb_stat", np.nan), 3),
                "lb_p":              round(diag.get("lb_p", np.nan), 4),
                "bg_stat":           round(diag.get("bg_stat", np.nan), 3),
                "bg_p":              round(diag.get("bg_p", np.nan), 4),
                "diag_n_lags":       diag.get("n_lags", np.nan),
                "note": (f"OLS HAC Newey-West + GLM Poisson (shift), mode={mode}"
                         + (f", detect={detect_target}" if mode == "detected" else "")),
            })

        fig.tight_layout()
        safe = ev_name.replace(" ","_").replace("/","").replace("(","").replace(")","")
        out  = OUT_DIR / f"plot_{safe}.png"
        fig.savefig(out, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"\n  → Salvato: {out}")

    if rows:
        df = pd.DataFrame(rows)
        csv_out = OUT_DIR / "v2_intermediate_results.csv"
        df.to_csv(csv_out, index=False)
        print(f"\n  → CSV: {csv_out}")
        cols_show = ["evento", "carburante", "break_date",
                     "gain_ols_meur", "gain_poisson_meur", "poisson_pearson_disp"]
        print("\n" + df[cols_show].to_string(index=False))
    else:
        print("\n  ⚠ Nessun risultato prodotto.")


if __name__ == "__main__":
    main()