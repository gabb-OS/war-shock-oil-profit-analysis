#!/usr/bin/env python3
"""
02d_v6_glm_gamma.py  ─  Metodo 6: GLM Gamma log-link (ITS)
============================================================
Modello di regressione GLM con distribuzione Gamma e link logaritmico per
la stima dell'extra-profitto speculativo sul margine distributori nei periodi
successivi a eventi geopolitici.

... (commento invariato) ...
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
from forecast_consumi import load_daily_consumption   # <-- nuovo import

try:
    import statsmodels.api as sm
    from statsmodels.stats.stattools import durbin_watson
    HAS_SM = True
except ImportError:
    HAS_SM = False
    warnings.warn(
        "statsmodels non installato — GLM Gamma non disponibile. "
        "pip install statsmodels"
    )

# ── Configurazione ─────────────────────────────────────────────────────────────
BASE_DIR    = Path(__file__).parent
DAILY_CSV   = BASE_DIR / "data" / "processed" / "daily_fuel_prices_all.csv"
GASOIL_CSV  = BASE_DIR / "data" / "Futures" / "London Gas Oil Futures Historical Data.csv"
EUROBOB_CSV = BASE_DIR / "data" / "Futures" / "Eurobob_B7H1_date.csv"
EURUSD_CSV  = BASE_DIR / "data" / "raw" / "eurusd.csv"
_OUT_BASE   = BASE_DIR / "data" / "plots" / "its"

PRE_WIN     = 40       # giorni pre-break per stimare la baseline Gamma
POST_WIN    = 40       # giorni post-break per calcolare l'extra profitto
SEARCH      = 30       # finestra ±SEARCH intorno allo shock (detection L2)
HALF_WIN    = 30       # semi-finestra Chow test
CI_ALPHA    = 0.05     # α → CI/PI al 95%  (z ≈ 1.96)
GAMMA_EPS   = 1e-4     # shift ε per portare y_shifted > 0  (€/L)
DISP_WARN   = 2.0      # soglia: φ > DISP_WARN → segnala overdispersione

MA_WIN      = 7        # smoothing detection L2
MIN_SEG     = 14       # segmento minimo detection L2 (giorni)

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

FUELS: dict[str, tuple[str, str]] = {
    "benzina": ("margin_benzina", "#E63946"),
    "gasolio": ("margin_gasolio", "#1D3557"),
}

PRICE_COLS: dict[str, str] = {
    "benzina": "benzina_net",
    "gasolio": "gasolio_net",
}

GAMMA_COLOR = "#27ae60"   # verde per distinguere Gamma da Poisson (blu)


# ══════════════════════════════════════════════════════════════════════════════
# Caricamento dati (identico)
# ══════════════════════════════════════════════════════════════════════════════

def _load_gasoil_futures(eurusd: pd.Series) -> pd.Series:
    df = pd.read_csv(GASOIL_CSV, encoding="utf-8-sig", dtype=str)
    df["date"]  = pd.to_datetime(df["Date"], format="%m/%d/%Y", errors="coerce")
    df["price"] = (df["Price"].str.replace(",", "", regex=False)
                   .pipe(pd.to_numeric, errors="coerce"))
    return (df.dropna(subset=["date", "price"]).sort_values("date")
              .set_index("date")
              .pipe(lambda d: usd_ton_to_eur_liter(d["price"], eurusd, GAS_OIL)))


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
    df = (df.dropna(subset=["date", "price"])
            .sort_values("date").set_index("date"))
    df = df[~df.index.duplicated(keep="first")]
    return usd_ton_to_eur_liter(df["price"], eurusd, EUROBOB_HC)


def load_margin_data() -> pd.DataFrame:
    daily = (pd.read_csv(DAILY_CSV, parse_dates=["date"])
               .sort_values("date").set_index("date"))
    eurusd  = load_eurusd(
        csv_path=EURUSD_CSV if EURUSD_CSV.exists() else None,
        start="2015-01-01", end="2026-12-31"
    )
    gasoil  = _load_gasoil_futures(eurusd)
    eurobob = _load_eurobob_futures(eurusd)
    df = daily[["benzina_net", "gasolio_net"]].copy()
    df["margin_gasolio"] = (
        df["gasolio_net"] - gasoil.reindex(df.index, method="ffill")
    )
    df["margin_benzina"] = (
        df["benzina_net"] - eurobob.reindex(df.index, method="ffill")
        if eurobob is not None else np.nan
    )
    return df


# ══════════════════════════════════════════════════════════════════════════════
# Break point detection — Window L2 Discrepancy (Paper BLOCCO 1, Eq. 1–2)
# ══════════════════════════════════════════════════════════════════════════════

def _l2_cost(y: np.ndarray) -> float:
    """c(y_I) = Σ ||y_t − ȳ||²   (Paper, Eq. 2)"""
    if len(y) < 2:
        return 0.0
    return float(np.sum((y - y.mean()) ** 2))


def detect_breakpoint(series: pd.Series, shock: pd.Timestamp) -> dict:
    """Window L2 Discrepancy — τ = argmax d(v) nel top-quartile."""
    mask = (
        (series.index >= shock - pd.Timedelta(days=SEARCH)) &
        (series.index <= shock + pd.Timedelta(days=SEARCH))
    )
    win = series[mask].dropna()

    fallback = {"tau": shock, "d_max": 0.0, "method": "window_l2_nofound",
                "_df": pd.DataFrame()}

    if len(win) < 2 * MIN_SEG + MA_WIN:
        return fallback

    ma    = win.rolling(MA_WIN, center=True, min_periods=1).mean()
    y     = ma.values
    n     = len(y)
    c_uw  = _l2_cost(y)

    rows = [
        {"tau": ma.index[v], "d": c_uw - _l2_cost(y[:v]) - _l2_cost(y[v:])}
        for v in range(MIN_SEG, n - MIN_SEG)
    ]
    if not rows:
        return fallback

    df_c      = pd.DataFrame(rows)
    threshold = float(np.percentile(df_c["d"], 75))
    top       = df_c[df_c["d"] >= threshold]
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
# Utilità
# ══════════════════════════════════════════════════════════════════════════════

def _hac_maxlags(n: int) -> int:
    return max(1, int(np.floor(4 * (n / 100) ** (2 / 9))))


def _compute_shift(pre: pd.Series, eps: float = GAMMA_EPS) -> float:
    """
    Shift additivo per portare tutta la serie nel dominio (0, +∞).
    Gamma richiede y > 0; per i margini in generale y_min > 0 ma in periodi
    particolari può scendere sotto zero.
      shift = max(0, −min(y_pre)) + ε
    """
    min_val = float(pre.min())
    return max(0.0, -min_val) + eps


# ══════════════════════════════════════════════════════════════════════════════
# OLS HAC (Newey-West) — modello di riferimento / confronto
# ══════════════════════════════════════════════════════════════════════════════

def fit_ols_hac(
    series: pd.Series,
    break_date: pd.Timestamp,
    shock: pd.Timestamp,
) -> dict | None:
    """OLS lineare sui PRE_WIN gg pre-break con SE Newey-West HAC."""
    anchor = break_date
    pre = series[
        (series.index >= anchor - pd.Timedelta(days=PRE_WIN)) &
        (series.index < anchor)
    ].dropna()

    if len(pre) < 10:
        return None

    x = np.array([(d - anchor).days for d in pre.index], dtype=float)

    if not HAS_SM:
        slope, intercept, r, *_ = stats.linregress(x, pre.values)
        return dict(
            slope=slope, intercept=intercept, r2=float(r ** 2),
            anchor=anchor, pre=pre, hac=False,
            mse=float(np.var(pre.values - (slope * x + intercept), ddof=2)),
        )

    X       = sm.add_constant(x)
    model   = sm.OLS(pre.values, X)
    fit     = model.fit()
    maxlags = _hac_maxlags(len(pre))
    fit_hac = fit.get_robustcov_results(cov_type="HAC", maxlags=maxlags)
    dw      = durbin_watson(fit.resid)

    return dict(
        fit_hac=fit_hac, fit_ols=fit,
        slope=float(fit_hac.params[1]),
        intercept=float(fit_hac.params[0]),
        r2=float(fit.rsquared),
        anchor=anchor, pre=pre,
        hac=True, dw=dw, maxlags=maxlags,
        mse=float(fit.mse_resid),
        residuals=fit.resid,
        X_bg=X,
        aic_ols=float(fit.aic),
        bic_ols=float(fit.bic),
    )


def project_hac(
    info: dict,
    post_index: pd.DatetimeIndex,
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Proietta baseline OLS HAC sul post con PI al (1-CI_ALPHA)."""
    anchor   = info["anchor"]
    x_post   = np.array([(d - anchor).days for d in post_index], dtype=float)
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
        sxx = np.sum((x_p - x_p.mean()) ** 2)
        se_pred = np.sqrt(
            info["mse"] * (1 + 1 / n + (x_post - x_p.mean()) ** 2 / sxx)
        )
        t_crit = stats.t.ppf(1 - CI_ALPHA / 2, df=n - 2)

    return (
        pd.Series(baseline,                  index=post_index),
        pd.Series(baseline - t_crit * se_pred, index=post_index),
        pd.Series(baseline + t_crit * se_pred, index=post_index),
    )


# ══════════════════════════════════════════════════════════════════════════════
# GLM Gamma log-link  ← modello principale
# ══════════════════════════════════════════════════════════════════════════════

def fit_glm_gamma(
    series: pd.Series,
    break_date: pd.Timestamp,
    shock: pd.Timestamp,
) -> dict | None:
    """
    Stima GLM Gamma (link log) sul periodo pre-break.

    Procedura
    ---------
    1. Estrai la finestra pre: [anchor − PRE_WIN, anchor)
    2. Calcola shift = max(0, −min_pre) + ε  →  y_shifted > 0 per costruzione
    3. Stima via IRLS:
         log(E[y_shifted]) = β₀ + β₁·t    (t = giorni dall'anchor)
    4. Estrai:
         · params  β̂, cov_params  (Wald CI)
         · φ = Pearson χ²/df  (dispersione; ≈ 1 = equidisperso)
         · deviance/df         (bontà fit alternativa)
         · AIC, BIC            (confronto modelli)
    5. Diagnostica pre-serie:
         · skewness             (motivazione Gamma)
         · CV = std/mean        (stima empirica di sqrt(φ))
         · Pregibon link test   (correttezza del link log)
         · SW test su log(y)    (proxy log-normalità residui)

    Returns
    -------
    dict o None (se statsmodels mancante o dati insufficienti)
    """
    if not HAS_SM:
        return None

    anchor = break_date
    pre = series[
        (series.index >= anchor - pd.Timedelta(days=PRE_WIN)) &
        (series.index < anchor)
    ].dropna()

    if len(pre) < 10:
        return None

    shift     = _compute_shift(pre)
    y_shifted = pre.values + shift      # tutti > 0
    x         = np.array([(d - anchor).days for d in pre.index], dtype=float)
    X         = sm.add_constant(x)

    try:
        glm_model = sm.GLM(
            y_shifted, X,
            family=sm.families.Gamma(link=sm.families.links.Log()),
        )
        glm_fit = glm_model.fit(maxiter=300)
    except Exception as exc:
        warnings.warn(f"GLM Gamma fit fallito: {exc}")
        return None

    # ── Dispersione φ ─────────────────────────────────────────────────────────
    # Pearson χ²/df = Σ[(y − μ̂)²/μ̂²] / (n − p)   per famiglia Gamma
    mu_hat   = glm_fit.fittedvalues
    n_obs, p = len(pre), 2
    pearson_chi2 = float(np.sum(((y_shifted - mu_hat) / mu_hat) ** 2))
    phi_pearson  = pearson_chi2 / (n_obs - p) if n_obs > p else np.nan

    # Scala stimata ML (= phi via scale ML per Gamma)
    phi_ml = float(glm_fit.scale)

    # Deviance/df
    dev_df = float(glm_fit.deviance / (n_obs - p)) if n_obs > p else np.nan

    # ── Statistiche pre-serie ─────────────────────────────────────────────────
    y_pre    = pre.values
    skew_pre = float(stats.skew(y_pre))
    kurt_pre = float(stats.kurtosis(y_pre))
    mean_pre = float(y_pre.mean())
    std_pre  = float(y_pre.std(ddof=1))
    cv_pre   = std_pre / abs(mean_pre) if abs(mean_pre) > 1e-9 else np.nan

    # ── Pregibon link test ────────────────────────────────────────────────────
    # P-link: aggiunge μ̂² come predittore; t-stat non sign. → link corretto
    link_pval = np.nan
    try:
        eta_hat = glm_fit.predict(X, which="linear")  # η̂ = X β̂
        mu_sq   = (np.exp(eta_hat)) ** 2
        X_aug   = np.column_stack([X, mu_sq])
        glm_aug = sm.GLM(
            y_shifted, X_aug,
            family=sm.families.Gamma(link=sm.families.links.Log()),
        ).fit(maxiter=200)
        link_pval = float(glm_aug.pvalues[-1])
    except Exception:
        pass

    # ── Test SW su log(y_shifted): proxy normalità del predittore lineare ─────
    sw_stat = sw_p = np.nan
    if len(y_shifted) >= 3:
        try:
            sw_stat, sw_p = stats.shapiro(np.log(y_shifted))
            sw_stat, sw_p = float(sw_stat), float(sw_p)
        except Exception:
            pass

    # ── Residui Pearson (scala originale non shifted) ─────────────────────────
    resid_pearson_raw = (y_shifted - mu_hat) / np.sqrt(phi_ml) / mu_hat

    return dict(
        # Core
        glm_fit    = glm_fit,
        params     = glm_fit.params,          # [β₀, β₁]
        cov_params = glm_fit.cov_params(),
        shift      = shift,
        anchor     = anchor,
        pre        = pre,
        # Dispersione
        phi_pearson    = phi_pearson,
        phi_ml         = phi_ml,
        dev_df         = dev_df,
        overdispersed  = (phi_pearson > DISP_WARN) if not np.isnan(phi_pearson) else False,
        # Bontà del fit
        aic        = float(glm_fit.aic),
        bic        = float(glm_fit.bic),
        llf        = float(glm_fit.llf),
        # Statistiche pre-serie
        skew_pre   = skew_pre,
        kurt_pre   = kurt_pre,
        cv_pre     = cv_pre,
        mean_pre   = mean_pre,
        std_pre    = std_pre,
        # Diagnostics
        link_pval     = link_pval,
        sw_stat_log   = sw_stat,
        sw_p_log      = sw_p,
        resid_pearson = resid_pearson_raw,
        n_pre         = n_obs,
    )


def project_glm_gamma(
    info: dict,
    post_index: pd.DatetimeIndex,
) -> tuple[pd.Series, pd.Series, pd.Series, pd.Series, pd.Series]:
    """
    Proietta la baseline GLM Gamma sul periodo post-break.

    Restituisce 5 serie:
      baseline    : μ̂ − shift   (media prevista, scala originale)
      ci_low      : CI medio inf (Wald sul predittore lineare, asimmetrico)
      ci_high     : CI medio sup
      pi_low      : PI previsione inf (include varianza risposta Gamma)
      pi_high     : PI previsione sup
    """
    anchor    = info["anchor"]
    shift     = info["shift"]
    params    = info["params"]
    cov       = info["cov_params"]
    phi_ml    = info["phi_ml"]
    z_crit    = stats.norm.ppf(1 - CI_ALPHA / 2)

    x_post = np.array([(d - anchor).days for d in post_index], dtype=float)
    X_post = np.column_stack([np.ones(len(x_post)), x_post])

    # Predittore lineare e varianza parametrica
    eta_hat   = X_post @ params
    var_param = np.einsum("ni,ij,nj->n", X_post, cov, X_post)
    se_param  = np.sqrt(np.maximum(var_param, 0))

    # Baseline (μ̂ su scala risposta, poi back-shift)
    baseline  = np.exp(eta_hat) - shift

    # CI medio (solo incertezza parametrica)
    ci_low    = np.exp(eta_hat - z_crit * se_param) - shift
    ci_high   = np.exp(eta_hat + z_crit * se_param) - shift

    # PI previsione (parametrica + risposta Gamma)
    se_total  = np.sqrt(np.maximum(var_param + phi_ml, 0))
    pi_low    = np.exp(eta_hat - z_crit * se_total) - shift
    pi_high   = np.exp(eta_hat + z_crit * se_total) - shift

    return (
        pd.Series(baseline, index=post_index),
        pd.Series(ci_low,   index=post_index),
        pd.Series(ci_high,  index=post_index),
        pd.Series(pi_low,   index=post_index),
        pd.Series(pi_high,  index=post_index),
    )


# ══════════════════════════════════════════════════════════════════════════════
# Chow test
# ══════════════════════════════════════════════════════════════════════════════

def chow_test(series: pd.Series, tau: pd.Timestamp) -> dict:
    pre  = series[
        (series.index >= tau - pd.Timedelta(days=HALF_WIN)) &
        (series.index < tau)
    ].dropna()
    post = series[
        (series.index >= tau) &
        (series.index < tau + pd.Timedelta(days=HALF_WIN))
    ].dropna()

    if len(pre) < 5 or len(post) < 5:
        return {"F": np.nan, "p": np.nan}

    def _ols_ssr(y, x_days):
        sl, ic, *_ = stats.linregress(x_days, y)
        return float(np.sum((y - (sl * x_days + ic)) ** 2))

    x_pre    = np.array([(d - tau).days for d in pre.index],  dtype=float)
    x_post   = np.array([(d - tau).days for d in post.index], dtype=float)
    combined = pd.concat([pre, post])
    x_comb   = np.array([(d - tau).days for d in combined.index], dtype=float)

    ssr_r = _ols_ssr(combined.values, x_comb)
    ssr_u = _ols_ssr(pre.values, x_pre) + _ols_ssr(post.values, x_post)

    k, n = 2, len(combined)
    if ssr_u < 1e-14 or n <= 2 * k:
        return {"F": np.nan, "p": np.nan}

    F = ((ssr_r - ssr_u) / k) / (ssr_u / (n - 2 * k))
    p = float(stats.f.sf(F, dfn=k, dfd=n - 2 * k))
    return {"F": float(F), "p": p}


# ══════════════════════════════════════════════════════════════════════════════
# Plot diagnostico distribuzione Gamma (panel specifico)
# ══════════════════════════════════════════════════════════════════════════════

def _plot_gamma_diagnostics(
    info_gamma: dict,
    ev_name: str,
    fuel_key: str,
    out_path: Path,
) -> None:
    # ... (invariato) ...
    pre      = info_gamma["pre"]
    shift    = info_gamma["shift"]
    glm_fit  = info_gamma["glm_fit"]
    phi_ml   = info_gamma["phi_ml"]
    y_shift  = pre.values + shift
    mu_hat   = glm_fit.fittedvalues

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    fig.suptitle(
        f"[V6 GLM Gamma – diagnostica distribuzione]  {ev_name}  ·  {fuel_key.capitalize()}\n"
        f"φ_ML={phi_ml:.3f}  |  skewness={info_gamma['skew_pre']:.3f}  "
        f"|  CV={info_gamma['cv_pre']:.3f}  "
        f"|  AIC={info_gamma['aic']:.1f}",
        fontsize=9, fontweight="bold"
    )

    # ── 1. Histogram + PDF Gamma ──────────────────────────────────────────────
    ax = axes[0]
    ax.hist(y_shift, bins=20, density=True, alpha=0.55, color=GAMMA_COLOR,
            edgecolor="white", label=f"y_shifted (n={len(y_shift)})")
    mu_bar = float(y_shift.mean())
    if phi_ml > 0 and mu_bar > 0:
        k_hat    = 1.0 / phi_ml
        theta_hat = mu_bar * phi_ml
        x_pdf    = np.linspace(y_shift.min() * 0.9, y_shift.max() * 1.1, 200)
        ax.plot(x_pdf, stats.gamma.pdf(x_pdf, a=k_hat, scale=theta_hat),
                color="darkgreen", lw=2.0, label=f"Gamma(k={k_hat:.2f}, θ={theta_hat:.4f})")
    ax.set_xlabel("y_shifted (€/L)", fontsize=8)
    ax.set_ylabel("Densità", fontsize=8)
    ax.set_title("Distribuzione pre-periodo\nvs PDF Gamma stimata", fontsize=8)
    ax.legend(fontsize=7)
    ax.grid(axis="y", alpha=0.20)

    # ── 2. Residui Pearson vs η̂ ──────────────────────────────────────────────
    ax = axes[1]
    eta_hat_pre = glm_fit.predict(linear=True)
    resid_p     = (y_shift - mu_hat) / (phi_ml * mu_hat) if phi_ml > 0 else np.zeros_like(mu_hat)
    ax.scatter(eta_hat_pre, resid_p, s=15, alpha=0.55, color=GAMMA_COLOR, edgecolors="none")
    ax.axhline(0, color="grey", lw=0.9, ls="--")
    ax.axhline( 2, color="orange", lw=0.7, ls=":", alpha=0.7)
    ax.axhline(-2, color="orange", lw=0.7, ls=":", alpha=0.7)
    ax.set_xlabel("Predittore lineare η̂", fontsize=8)
    ax.set_ylabel("Residui Pearson standardizzati", fontsize=8)
    ax.set_title("Residui Pearson vs η̂\n(eteroschedasticità residua)", fontsize=8)
    ax.grid(alpha=0.20)

    # ── 3. Q-Q residui deviance vs N(0,1) ────────────────────────────────────
    ax = axes[2]
    dev_resid = glm_fit.resid_deviance
    n_r       = len(dev_resid)
    q_th      = stats.norm.ppf((np.arange(1, n_r + 1) - 0.375) / (n_r + 0.25))
    q_obs     = np.sort(dev_resid)
    ax.scatter(q_th, q_obs, s=15, alpha=0.55, color=GAMMA_COLOR, edgecolors="none")
    lim = max(abs(q_th).max(), abs(q_obs).max()) * 1.1
    ax.plot([-lim, lim], [-lim, lim], color="grey", lw=0.9, ls="--", label="y=x")
    ax.set_xlabel("Quantili teorici N(0,1)", fontsize=8)
    ax.set_ylabel("Residui deviance (ordinati)", fontsize=8)
    ax.set_title("Q-Q residui deviance\nvs N(0,1)", fontsize=8)
    ax.legend(fontsize=7)
    ax.grid(alpha=0.20)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"    → Gamma diagnostics: {out_path}")


# ══════════════════════════════════════════════════════════════════════════════
# Plot principale — 3 pannelli: serie + CI/PI, guadagno cumulato, CI shape
# ══════════════════════════════════════════════════════════════════════════════

def _plot_event_fuel(
    ev_name: str, ev: dict,
    series: pd.Series,
    fuel_key: str, fuel_color: str,
    break_date: pd.Timestamp,
    info_ols: dict,
    baseline_ols: pd.Series, ci_low_ols: pd.Series, ci_high_ols: pd.Series,
    extra_ols: pd.Series, gain_ols: float,
    info_gamma: dict | None,
    baseline_gamma: pd.Series | None,
    ci_low_gamma: pd.Series | None, ci_high_gamma: pd.Series | None,
    pi_low_gamma: pd.Series | None, pi_high_gamma: pd.Series | None,
    extra_gamma: pd.Series | None, gain_gamma: float | None,
    chow: dict, mode: str, break_method: str,
    cons: pd.Series,                    # <-- consumi giornalieri (L/giorno)
    ax_main: plt.Axes, ax_gain: plt.Axes, ax_ci: plt.Axes,
) -> None:
    shock = ev["shock"]
    win = series[
        (series.index >= shock - pd.Timedelta(days=PRE_WIN)) &
        (series.index <= shock + pd.Timedelta(days=POST_WIN))
    ].dropna()

    hac_label = (f"OLS HAC NW (lag={info_ols.get('maxlags','?')})"
                 if info_ols.get("hac") else "OLS i.i.d.")
    chow_str = (f"Chow F={chow['F']:.2f} p={chow['p']:.3f}"
                if not np.isnan(chow.get("F", np.nan)) else "Chow: n/a")
    mode_str  = (f"τ={break_date.date()} ({break_method})" if mode == "detected"
                 else f"Break=shock ({shock.date()})")

    # ── Serie effettiva ───────────────────────────────────────────────────────
    ax_main.plot(win.index, win.values, color=fuel_color, lw=1.0, zorder=3,
                 label=f"{fuel_key.capitalize()} effettivo")

    # ── Baseline OLS HAC ──────────────────────────────────────────────────────
    _anchor_ols = info_ols["anchor"]
    _pre_idx    = info_ols["pre"].index
    _x_pre_ols  = np.array([(d - _anchor_ols).days for d in _pre_idx], dtype=float)
    _pre_fit_ols = info_ols["slope"] * _x_pre_ols + info_ols["intercept"]
    _full_idx_ols = _pre_idx.append(baseline_ols.index)
    _full_val_ols = np.concatenate([_pre_fit_ols, baseline_ols.values])
    ax_main.plot(_full_idx_ols, _full_val_ols,
                 color="dimgrey", lw=1.4, ls="--",
                 label=f"Baseline {hac_label} (R²={info_ols['r2']:.2f})")
    ax_main.fill_between(ci_low_ols.index,
                         ci_low_ols.values, ci_high_ols.values,
                         alpha=0.10, color="grey",
                         label=f"PI {int((1-CI_ALPHA)*100)}% OLS (simmetrico)")

    # ── Baseline GLM Gamma ────────────────────────────────────────────────────
    if baseline_gamma is not None and info_gamma is not None:
        phi_str  = f"φ={info_gamma['phi_ml']:.3f}"
        disp_tag = f" ⚠" if info_gamma["overdispersed"] else ""
        _anchor_g  = info_gamma["anchor"]
        _pre_idx_g = info_gamma["pre"].index
        _x_pre_g   = np.array([(d - _anchor_g).days for d in _pre_idx_g], dtype=float)
        _X_pre_g   = np.column_stack([np.ones(len(_x_pre_g)), _x_pre_g])
        _pre_fit_g = np.exp(_X_pre_g @ info_gamma["params"]) - info_gamma["shift"]
        _full_idx_g = _pre_idx_g.append(baseline_gamma.index)
        _full_val_g = np.concatenate([_pre_fit_g, baseline_gamma.values])
        ax_main.plot(_full_idx_g, _full_val_g,
                     color=GAMMA_COLOR, lw=1.8, ls="-.",
                     label=f"Baseline GLM Gamma ({phi_str}{disp_tag}  AIC={info_gamma['aic']:.1f})")
        # CI medio (interno, più stretto)
        ax_main.fill_between(
            ci_low_gamma.index,
            ci_low_gamma.values, ci_high_gamma.values,
            alpha=0.14, color=GAMMA_COLOR,
            label=f"CI {int((1-CI_ALPHA)*100)}% Gamma (asimmetrico)"
        )
        # PI previsione (esterno, più largo)
        if pi_low_gamma is not None:
            ax_main.fill_between(
                pi_low_gamma.index,
                pi_low_gamma.values, pi_high_gamma.values,
                alpha=0.06, color=GAMMA_COLOR,
                label=f"PI {int((1-CI_ALPHA)*100)}% Gamma (param+risposta)"
            )

    # ── Extra profitto Gamma (ombreggiatura) ──────────────────────────────────
    if extra_gamma is not None and baseline_gamma is not None:
        ax_main.fill_between(
            extra_gamma.index,
            win.reindex(extra_gamma.index), baseline_gamma.values,
            where=(extra_gamma >= 0), alpha=0.22, color="lime",
            label="Extra Gamma (≥0)"
        )
        ax_main.fill_between(
            extra_gamma.index,
            win.reindex(extra_gamma.index), baseline_gamma.values,
            where=(extra_gamma < 0), alpha=0.22, color="tomato",
            label="Sotto-baseline Gamma (<0)"
        )

    # ── Linee verticali ───────────────────────────────────────────────────────
    ax_main.axvline(shock, color=ev["color"], lw=1.6, ls="--",
                    label=f"Shock ({shock.date()})")
    if mode == "detected":
        ax_main.axvline(break_date, color=fuel_color, lw=1.2, ls=":")

    ax_main.set_title(
        f"[V6-GLM Gamma / mode={mode}]  {fuel_key.capitalize()} – {ev_name}\n"
        f"{mode_str}  |  {chow_str}",
        fontsize=8, fontweight="bold"
    )
    ax_main.set_ylabel("Margine (€/L)", fontsize=8)
    ax_main.legend(fontsize=5.5, loc="upper left", ncol=2)
    ax_main.grid(axis="y", alpha=0.20)
    ax_main.xaxis.set_major_formatter(mdates.DateFormatter("%d %b %y"))
    ax_main.xaxis.set_major_locator(mdates.WeekdayLocator(byweekday=0, interval=2))
    plt.setp(ax_main.xaxis.get_majorticklabels(), rotation=35, ha="right", fontsize=7)

    # zoom y
    _y_data = [win.values, _full_val_ols]
    if baseline_gamma is not None:
        _y_data.append(_full_val_g)
    _y_all = np.concatenate([v for v in _y_data if len(v) > 0])
    _y_all = _y_all[np.isfinite(_y_all)]
    if len(_y_all) > 0:
        _ymin, _ymax = float(np.nanmin(_y_all)), float(np.nanmax(_y_all))
        _pad = max(abs(_ymax - _ymin) * 0.25, 0.005)
        ax_main.set_ylim(_ymin - _pad, _ymax + _pad)

    # ── Pannello guadagno cumulato (ora con consumi reali) ────────────────────
    # Allinea i consumi ai giorni del post
    cons_aligned = cons.reindex(baseline_ols.index, method="ffill").fillna(cons.mean())
    cum_ols = (extra_ols * cons_aligned.values / 1e6).cumsum()
    ax_gain.plot(cum_ols.index, cum_ols.values,
                 color="dimgrey", lw=1.2, ls="--",
                 label=f"OLS HAC → {gain_ols:+.0f} M€")

    if extra_gamma is not None and gain_gamma is not None:
        cum_gamma = (extra_gamma * cons_aligned.values / 1e6).cumsum()
        ax_gain.plot(cum_gamma.index, cum_gamma.values,
                     color=GAMMA_COLOR, lw=1.6,
                     label=f"GLM Gamma → {gain_gamma:+.0f} M€")

    ax_gain.axhline(0, color="grey", lw=0.7, ls="--")
    ax_gain.fill_between(cum_ols.index, cum_ols.values, 0,
                         where=(cum_ols >= 0), alpha=0.18, color="green")
    ax_gain.fill_between(cum_ols.index, cum_ols.values, 0,
                         where=(cum_ols < 0), alpha=0.18, color="red")
    ax_gain.axvline(break_date, color=fuel_color, lw=1.0, ls=":", alpha=0.7)

    # Calcola consumo medio (ML/giorno) per il titolo
    avg_cons_ml = cons.mean() / 1e6
    ax_gain.set_title(
        f"Guadagno extra cumulato ({len(extra_ols)}gg post-break)\n"
        f"[consumo medio {avg_cons_ml:.1f} ML/giorno]",
        fontsize=7
    )
    ax_gain.set_ylabel("M€ cumulati", fontsize=8)
    ax_gain.legend(fontsize=7, loc="upper left")
    ax_gain.grid(axis="y", alpha=0.20)
    ax_gain.xaxis.set_major_formatter(mdates.DateFormatter("%d %b %y"))
    ax_gain.xaxis.set_major_locator(mdates.WeekdayLocator(byweekday=0, interval=2))
    plt.setp(ax_gain.xaxis.get_majorticklabels(), rotation=35, ha="right", fontsize=7)

    # ── Pannello confronto ampiezza CI: OLS vs Gamma ──────────────────────────
    if baseline_gamma is not None and ci_low_gamma is not None:
        idx = baseline_ols.index

        # Larghezza CI OLS (simmetrica)
        width_ols_up  = (ci_high_ols   - baseline_ols).values
        width_ols_dn  = (baseline_ols  - ci_low_ols).values

        # Larghezza CI Gamma (asimmetrica)
        bg = baseline_gamma.reindex(idx)
        cg_lo = ci_low_gamma.reindex(idx)
        cg_hi = ci_high_gamma.reindex(idx)
        width_g_up = (cg_hi - bg).values
        width_g_dn = (bg - cg_lo).values

        x_days_post = np.arange(len(idx))
        ax_ci.fill_between(x_days_post,  width_ols_up, alpha=0.25, color="grey",
                           label="CI+ OLS (simmetrico)")
        ax_ci.fill_between(x_days_post, -width_ols_dn, alpha=0.25, color="grey")
        ax_ci.fill_between(x_days_post,  width_g_up,   alpha=0.30, color=GAMMA_COLOR,
                           label="CI+ Gamma (asimmetrico)")
        ax_ci.fill_between(x_days_post, -width_g_dn,   alpha=0.30, color=GAMMA_COLOR)
        ax_ci.axhline(0, color="black", lw=0.7)
        ax_ci.set_xlabel("Giorni post-break", fontsize=8)
        ax_ci.set_ylabel("Ampiezza semi-CI (€/L)", fontsize=8)
        ax_ci.set_title(
            "Asimmetria CI: OLS (grigio) vs Gamma (verde)\n"
            "Gamma upper > lower  →  distribuzione right-skewed",
            fontsize=7.5
        )
        ax_ci.legend(fontsize=7, loc="upper left")
        ax_ci.grid(axis="y", alpha=0.20)
    else:
        ax_ci.text(0.5, 0.5, "GLM Gamma non disponibile",
                   ha="center", va="center", transform=ax_ci.transAxes, fontsize=9)
        ax_ci.set_axis_off()


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    parser = argparse.ArgumentParser(
        description="V6 GLM Gamma log-link – ITS pipeline"
    )
    parser.add_argument(
        "--mode", choices=["fixed", "detected"], default="fixed",
        help="fixed = usa shock date; detected = θ da 02c_change_point_detection.py"
    )
    parser.add_argument(
        "--detect", choices=["margin", "price"], default="margin",
        help="(solo mode=detected) serie di detection: margin [default] o price"
    )
    args, _ = parser.parse_known_args()
    mode          = args.mode
    detect_target = args.detect

    if mode == "detected":
        OUT_DIR = _OUT_BASE / "detected" / detect_target / "v6_glm_gamma"
    else:
        OUT_DIR = _OUT_BASE / mode / "v6_glm_gamma"
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("═" * 70)
    print("  02d_v6_glm_gamma.py  –  GLM Gamma log-link  "
          f"[mode={mode}]")
    print(f"  Distribuzione  : Gamma(φ·μ²)  link=log  "
          f"(skewness > 0, CV ~ sqrt(φ))")
    if mode == "fixed":
        print("  Break          : shock date hardcodata")
    else:
        print(f"  Break          : θ da 02c  (detect={detect_target})")
    print(f"  Finestra       : PRE={PRE_WIN}gg / POST={POST_WIN}gg dal break")
    print(f"  CI/PI          : Wald asimmetrico sul link log  (α={CI_ALPHA})")
    print(f"  Soglia overdispersione: φ > {DISP_WARN}")
    print(f"  statsmodels    : {'OK' if HAS_SM else 'MANCANTE – solo OLS fallback'}")
    print("  Consumi        : letti da data/consumi/consumi_giornalieri.csv (via forecast_consumi)")
    print(f"  Output         : {OUT_DIR}")
    print("═" * 70)

    data = load_margin_data()
    rows: list[dict] = []

    for ev_name, ev in EVENTS.items():
        shock = ev["shock"]

        # Layout: n_fuels righe × 3 colonne (serie, guadagno, CI shape)
        fig, axes = plt.subplots(
            len(FUELS), 3,
            figsize=(21, 5.5 * len(FUELS)),
            squeeze=False
        )
        fig.suptitle(
            f"[Metodo 6 – GLM Gamma log-link / mode={mode}]  {ev_name}\n{ev['label']}",
            fontsize=11, fontweight="bold"
        )

        for row_idx, (fuel_key, (col_name, fuel_color)) in enumerate(FUELS.items()):
            series = data[col_name].dropna()

            # ── Break date ────────────────────────────────────────────────────
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

            # ── OLS HAC (reference) ───────────────────────────────────────────
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

            # ── Carica consumi giornalieri per il periodo post ────────────────
            cons = load_daily_consumption(post.index, fuel_key)   # Serie L/giorno

            baseline_ols, ci_low_ols, ci_high_ols = project_hac(info_ols, post.index)
            extra_ols = post - baseline_ols
            # Calcolo guadagno OLS con consumi reali
            gain_ols  = float((extra_ols * cons).sum() / 1e6)
            gain_ci_low_ols  = float(((post - ci_high_ols) * cons).sum() / 1e6)
            gain_ci_high_ols = float(((post - ci_low_ols) * cons).sum() / 1e6)

            chow = chow_test(series, break_date)

            # ── GLM Gamma ─────────────────────────────────────────────────────
            info_gamma = fit_glm_gamma(series, break_date, shock)

            baseline_gamma = ci_low_gamma = ci_high_gamma = None
            pi_low_gamma   = pi_high_gamma = extra_gamma = None
            gain_gamma = gain_ci_low_gamma = gain_ci_high_gamma = np.nan

            if info_gamma is not None:
                (baseline_gamma, ci_low_gamma, ci_high_gamma,
                 pi_low_gamma, pi_high_gamma) = project_glm_gamma(info_gamma, post.index)
                extra_gamma = post - baseline_gamma
                # Calcolo guadagno Gamma con consumi reali
                gain_gamma         = float((extra_gamma * cons).sum() / 1e6)
                gain_ci_low_gamma  = float(((post - ci_high_gamma) * cons).sum() / 1e6)
                gain_ci_high_gamma = float(((post - ci_low_gamma) * cons).sum() / 1e6)

                # Diagnostica distribuzione Gamma (plot separato)
                safe_ev    = (ev_name.replace(" ", "_").replace("/", "")
                              .replace("(", "").replace(")", ""))
                gamma_diag_path = OUT_DIR / f"diag_gamma_{safe_ev}_{fuel_key}.png"
                _plot_gamma_diagnostics(info_gamma, ev_name, fuel_key, gamma_diag_path)

            # ── Diagnostica residui OLS (compatibilità pipeline) ──────────────
            pre_resid = np.asarray(info_ols.get("residuals", []))
            X_bg      = info_ols.get("X_bg")
            diag = run_diagnostic_tests(pre_resid, x_for_bg=X_bg, n_lags=None)

            safe_ev   = (ev_name.replace(" ", "_").replace("/", "")
                         .replace("(", "").replace(")", ""))
            diag_path = OUT_DIR / f"diag_ols_{safe_ev}_{fuel_key}.png"
            plot_residual_diagnostics(
                resid=pre_resid,
                dates=info_ols["pre"].index,
                title=(f"[V6-OLS ref] Diagnostica residui pre-periodo\n"
                       f"{ev_name} · {fuel_key.capitalize()}  "
                       f"(break={break_date.date()})"),
                out_path=diag_path,
                diag_stats=diag,
            )

            # ── Plot principale (ora con cons) ────────────────────────────────
            _plot_event_fuel(
                ev_name, ev, series, fuel_key, fuel_color,
                break_date,
                info_ols, baseline_ols, ci_low_ols, ci_high_ols,
                extra_ols, gain_ols,
                info_gamma, baseline_gamma,
                ci_low_gamma, ci_high_gamma,
                pi_low_gamma, pi_high_gamma,
                extra_gamma, gain_gamma,
                chow, mode, break_method,
                cons,                              # <-- passato
                axes[row_idx][0], axes[row_idx][1], axes[row_idx][2],
            )

            # ── Stampa a video (invariata tranne i guadagni già ricalcolati) ─────
            print(f"\n  {ev_name}  [{fuel_key.upper()}]")
            print(f"    Break ({break_method}) = {break_date.date()}  "
                  f"(shock={shock.date()})")
            print(f"    OLS HAC  R²={info_ols['r2']:.3f}   "
                  f"DW={info_ols.get('dw', float('nan')):.2f}   "
                  f"AIC={info_ols.get('aic_ols', float('nan')):.1f}")
            if not np.isnan(chow.get("F", np.nan)):
                print(f"    Chow test  F={chow['F']:.2f}  p={chow['p']:.3f}  "
                      f"({'break confermato' if chow['p'] < 0.05 else 'non confermato'})")
            print(f"    Guadagno OLS HAC = {gain_ols:+.0f} M€   "
                  f"CI95% [{gain_ci_low_ols:+.0f}, {gain_ci_high_ols:+.0f}] M€")

            if info_gamma is not None:
                phi_str    = f"{info_gamma['phi_ml']:.3f}"
                disp_warn  = "  ⚠ overdispersione" if info_gamma["overdispersed"] else ""
                link_str   = (f"{info_gamma['link_pval']:.3f}"
                              if not np.isnan(info_gamma['link_pval']) else "n/a")
                print(f"    GLM Gamma  φ_ML={phi_str}{disp_warn}")
                print(f"               φ_Pearson={info_gamma['phi_pearson']:.3f}   "
                      f"deviance/df={info_gamma['dev_df']:.3f}   "
                      f"AIC={info_gamma['aic']:.1f}")
                print(f"               skewness={info_gamma['skew_pre']:.3f}   "
                      f"CV={info_gamma['cv_pre']:.3f}   "
                      f"Link test p={link_str}")
                print(f"    Guadagno Gamma   = {gain_gamma:+.0f} M€   "
                      f"CI95% [{gain_ci_low_gamma:+.0f}, {gain_ci_high_gamma:+.0f}] M€")
                print(f"    (CI asimmetrico: lower={gain_ci_low_gamma:+.0f}  "
                      f"upper={gain_ci_high_gamma:+.0f}  Δ={gain_ci_high_gamma-gain_ci_low_gamma:.0f} M€)")
            else:
                print("    GLM Gamma: non disponibile (statsmodels mancante o dati insuff.)")

            if not np.isnan(diag.get("sw_p", np.nan)):
                print(f"    SW residui OLS  W={diag['sw_stat']:.3f}  "
                      f"p={diag['sw_p']:.3f}  "
                      f"{'OK' if diag['sw_p'] > 0.05 else '⚠ non norm.'}")

            # ── Record CSV (usa già gain calcolati con cons) ────────────────────
            _gain_total      = gain_gamma         if not np.isnan(gain_gamma)         else gain_ols
            _gain_ci_low     = gain_ci_low_gamma  if not np.isnan(gain_ci_low_gamma)  else gain_ci_low_ols
            _gain_ci_high    = gain_ci_high_gamma if not np.isnan(gain_ci_high_gamma) else gain_ci_high_ols
            _extra_mean      = (float(extra_gamma.mean()) if extra_gamma is not None
                                else float(extra_ols.mean()))

            rows.append({
                "metodo":          "v6_glm_gamma",
                "mode":            mode,
                "detect_target":   detect_target if mode == "detected" else "fixed",
                "evento":          ev_name,
                "carburante":      fuel_key,
                "shock":           shock.date(),
                "break_date":      break_date.date(),
                "break_method":    break_method,
                "pre_win_days":    PRE_WIN,
                "post_win_days":   POST_WIN,
                "n_pre":           len(info_ols["pre"]),
                "n_post":          len(post),
                # ── Colonne standard per compare.py ───────────────────────────
                "gain_total_meur":    round(_gain_total,   1),
                "gain_ci_low_meur":   round(_gain_ci_low,  1),
                "gain_ci_high_meur":  round(_gain_ci_high, 1),
                "extra_mean_eurl":    round(_extra_mean,   5),
                # ── OLS HAC (reference) ───────────────────────────────────────
                "r2_ols":          round(info_ols["r2"], 4),
                "dw_stat":         round(info_ols.get("dw", np.nan), 3),
                "aic_ols":         round(info_ols.get("aic_ols", np.nan), 2),
                "gain_ols_meur":   round(gain_ols, 1),
                "gain_ols_ci_low": round(gain_ci_low_ols, 1),
                "gain_ols_ci_high":round(gain_ci_high_ols, 1),
                "chow_F":          round(chow.get("F", np.nan), 3),
                "chow_p":          round(chow.get("p", np.nan), 4),
                # ── GLM Gamma ─────────────────────────────────────────────────
                "gamma_available": info_gamma is not None,
                "gain_gamma_meur": round(gain_gamma, 1) if not np.isnan(gain_gamma) else np.nan,
                "gain_gamma_ci_low":  round(gain_ci_low_gamma, 1)  if not np.isnan(gain_ci_low_gamma) else np.nan,
                "gain_gamma_ci_high": round(gain_ci_high_gamma, 1) if not np.isnan(gain_ci_high_gamma) else np.nan,
                "gamma_phi_ml":       round(info_gamma["phi_ml"], 4)       if info_gamma else np.nan,
                "gamma_phi_pearson":  round(info_gamma["phi_pearson"], 4)  if info_gamma else np.nan,
                "gamma_dev_df":       round(info_gamma["dev_df"], 4)       if info_gamma else np.nan,
                "gamma_overdispersed":info_gamma["overdispersed"]           if info_gamma else np.nan,
                "gamma_aic":          round(info_gamma["aic"], 2)           if info_gamma else np.nan,
                "gamma_bic":          round(info_gamma["bic"], 2)           if info_gamma else np.nan,
                "gamma_shift_eurl":   round(info_gamma["shift"], 5)         if info_gamma else np.nan,
                "gamma_skew_pre":     round(info_gamma["skew_pre"], 4)      if info_gamma else np.nan,
                "gamma_kurt_pre":     round(info_gamma["kurt_pre"], 4)      if info_gamma else np.nan,
                "gamma_cv_pre":       round(info_gamma["cv_pre"], 4)        if info_gamma else np.nan,
                "gamma_link_pval":    round(info_gamma["link_pval"], 4)     if info_gamma and not np.isnan(info_gamma["link_pval"]) else np.nan,
                "gamma_sw_p_log":     round(info_gamma["sw_p_log"], 4)      if info_gamma and not np.isnan(info_gamma["sw_p_log"]) else np.nan,
                # ── OLS residui diagnostics ───────────────────────────────────
                "sw_stat":         round(diag.get("sw_stat", np.nan), 4),
                "sw_p":            round(diag.get("sw_p", np.nan), 4),
                "lb_stat":         round(diag.get("lb_stat", np.nan), 3),
                "lb_p":            round(diag.get("lb_p", np.nan), 4),
                "bg_stat":         round(diag.get("bg_stat", np.nan), 3),
                "bg_p":            round(diag.get("bg_p", np.nan), 4),
                "note": (
                    f"GLM Gamma log-link, mode={mode}"
                    + (f", detect={detect_target}" if mode == "detected" else "")
                    + (f", phi_ml={info_gamma['phi_ml']:.3f}" if info_gamma else "")
                    + ", consumi giornalieri reali"
                ),
            })

        fig.tight_layout()
        safe = (ev_name.replace(" ", "_").replace("/", "")
                .replace("(", "").replace(")", ""))
        out  = OUT_DIR / f"plot_{safe}.png"
        fig.savefig(out, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"\n  → Salvato: {out}")

    if rows:
        df = pd.DataFrame(rows)
        csv_out = OUT_DIR / "v6_glm_gamma_results.csv"
        df.to_csv(csv_out, index=False)
        print(f"\n  → CSV: {csv_out}")
        cols_show = [
            "evento", "carburante", "break_date",
            "gain_ols_meur", "gain_gamma_meur",
            "gamma_phi_ml", "gamma_phi_pearson",
            "gamma_aic", "gamma_skew_pre", "gamma_cv_pre",
        ]
        print("\n" + df[cols_show].to_string(index=False))
    else:
        print("\n  ⚠ Nessun risultato prodotto.")


if __name__ == "__main__":
    main()