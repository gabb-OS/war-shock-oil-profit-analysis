#!/usr/bin/env python3
"""
02c_change_point_detection.py  (v2 – metodo canonico: GLM Poisson)
====================================================================
Change-point detection sul MARGINE o sul PREZZO NETTO alla pompa
(benzina e gasolio) in corrispondenza di eventi geopolitici rilevanti.

Metodo canonico — θ via GLM Poisson (Likelihood Ratio)
────────────────────────────────────────────────────────
  La serie (margine o prezzo) viene shiftata ad essere strettamente positiva:
      y_shift = y − min(y) + ε      (ε = 1e-4)

  Per ogni candidato τ ∈ [shock−SEARCH, shock+SEARCH], con min_seg ≥ 14 gg:
      λ_pre  = mean(y_shift[u:τ])
      λ_post = mean(y_shift[τ:w])
      LL(τ)  = Σ_{t<τ} [y_t·log λ_pre  − λ_pre]
             + Σ_{t≥τ} [y_t·log λ_post − λ_post]

  LR(τ) = 2·[LL(τ) − LL_null]      dove LL_null = LL(modello unico)
  θ = argmax_τ LR(τ)

  Sotto H₀ (nessun break), LR ∼ χ²(1) asintoticamente.
  theta_confirmed = True se p-value(LR_max) < 0.05.

Output primario
───────────────
  data/plots/change_point/{detect}/theta_results.csv
    → consumato dagli script ITS in modalità detected (colonne: evento,
      shock, carburante, detect_type, theta, lr_stat, theta_confirmed)

  data/plots/change_point/{detect}/cp_{detect}_{evento}.png
    → diagnostica per evento (pannelli GLM Poisson + supporto)

  data/plots/change_point/{detect}/comparison_all_methods.png
    → griglia eventi × carburanti con tutte le date rilevate a confronto

Metodi di confronto (non producono θ canoncio, solo diagnostica)
─────────────────────────────────────────────────────────────────
  L1 · Sliding-window Welch t-test corretto AR(1)  [usato in v1]
  L2 · CUSUM delle deviazioni dalla media pre-evento
  L3 · Binary Segmentation BIC (ruptures, costo RBF)
  L4 · PELT BIC (ruptures, costo RBF)              [usato in v3]
  Lw · Window Discrepancy L2  (vecchio metodo canonico 02c v1)  [usato in v2]

Uso:
  python3 02c_change_point_detection.py                    # margine (default)
  python3 02c_change_point_detection.py --detect margin
  python3 02c_change_point_detection.py --detect price
  python3 02c_change_point_detection.py --detect both      # entrambi

Dipendenze:
  pip install ruptures statsmodels scipy
"""

from __future__ import annotations

import argparse
import warnings
from pathlib import Path
from typing import Optional

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd
from scipy import stats

import sys
sys.path.insert(0, str(Path(__file__).parent / "utils"))
from conversions import GAS_OIL, EUROBOB as EUROBOB_HC, load_eurusd, usd_ton_to_eur_liter

try:
    import ruptures as rpt
    HAS_RUPTURES = True
except ImportError:
    HAS_RUPTURES = False
    warnings.warn("ruptures non installato – L3/L4 disabilitati. pip install ruptures")

# ── Configurazione ──────────────────────────────────────────────────────────
BASE_DIR     = Path(__file__).parent
DAILY_CSV    = BASE_DIR / "data" / "processed" / "daily_fuel_prices_all.csv"
GASOIL_CSV   = BASE_DIR / "data" / "Futures" / "London Gas Oil Futures Historical Data.csv"
EUROBOB_CSV  = BASE_DIR / "data" / "Futures" / "Eurobob_B7H1_date.csv"
EURUSD_CSV   = BASE_DIR / "data" / "raw" / "eurusd.csv"

HALF_WIN   = 40    # giorni per lato nelle finestre di supporto
SEARCH     = 40    # ricerca τ in [shock±SEARCH] giorni
STEP       = 1
MAX_BKPS   = 5
MIN_SIZE   = 14    # segmento minimo (giorni) per GLM e PELT
ZOOM_WIN   = 25    # giorni prima/dopo θ nel pannello zoom

SHIFT_EPS  = 1e-4  # shift per rendere y_shift strettamente positivo

FUELS: dict[str, tuple[str, str]] = {
    "benzina": ("margin_benzina", "#E63946"),
    "gasolio": ("margin_gasolio", "#1D3557"),
}
PRICE_COLS: dict[str, str] = {
    "benzina": "benzina_net",
    "gasolio": "gasolio_net",
}

EVENTS: dict[str, dict] = {
    "Ucraina (Feb 2022)": {
        "shock":     pd.Timestamp("2022-02-24"),
        "pre_start": pd.Timestamp("2021-12-01"),
        "post_end":  pd.Timestamp("2022-04-24"),
        "color":     "#e74c3c",
        "label":     "Russia-Ucraina\n(24 feb 2022)",
    },
    "Iran-Israele (Giu 2025)": {
        "shock":     pd.Timestamp("2025-06-13"),
        "pre_start": pd.Timestamp("2025-04-13"),
        "post_end":  pd.Timestamp("2025-08-13"),
        "color":     "#e67e22",
        "label":     "Iran-Israele\n(13 giu 2025)",
    },
    "Hormuz (Feb 2026)": {
        "shock":     pd.Timestamp("2026-02-28"),
        "pre_start": pd.Timestamp("2025-12-28"),
        "post_end":  pd.Timestamp("2026-04-30"),
        "color":     "#8e44ad",
        "label":     "Stretto di Hormuz\n(28 feb 2026)",
    },
}

# Etichette colori per il confronto tra metodi
METHOD_STYLES: dict[str, dict] = {
    "GLM Poisson θ": dict(color="#2ecc71", lw=2.5, ls="-",  zorder=6),
    "L1 Sliding t":  dict(color="#3498db", lw=1.2, ls=":",  zorder=4),
    "L2 CUSUM":      dict(color="#e67e22", lw=1.2, ls="-.", zorder=4),
    "L3 BinSeg":     dict(color="#9b59b6", lw=1.2, ls=":",  zorder=4),
    "L4 PELT":       dict(color="#27ae60", lw=1.2, ls="--", zorder=4),
    "Lw WinDisc L2": dict(color="#95a5a6", lw=1.2, ls="--", zorder=3),
    "Shock":         dict(color="#e74c3c", lw=1.8, ls="--", zorder=5),
}


# ══════════════════════════════════════════════════════════════════════════════
# Helpers
# ══════════════════════════════════════════════════════════════════════════════

def _phi_ar1(series: pd.Series) -> float:
    if len(series) < 4:
        return 0.0
    s = series.values - series.mean()
    if s.std() < 1e-12:
        return 0.0
    return float(np.clip(np.corrcoef(s[:-1], s[1:])[0, 1], -0.99, 0.99))


def _n_eff(n: int, phi: float) -> float:
    return max(2.0, n * (1.0 - phi) / (1.0 + phi))


def _welch_neff(pre: pd.Series, post: pd.Series) -> tuple[float, float]:
    phi_pre  = _phi_ar1(pre)
    phi_post = _phi_ar1(post)
    n1 = _n_eff(len(pre),  phi_pre)
    n2 = _n_eff(len(post), phi_post)
    m1, m2 = float(pre.mean()), float(post.mean())
    v1 = float(pre.var(ddof=1))  / n1
    v2 = float(post.var(ddof=1)) / n2
    se = np.sqrt(v1 + v2)
    if se < 1e-12:
        return 0.0, 1.0
    t_stat = (m2 - m1) / se
    df = (v1 + v2) ** 2 / (v1**2 / (n1 - 1) + v2**2 / (n2 - 1))
    df = max(1.0, df)
    p  = float(2.0 * stats.t.sf(abs(t_stat), df=df))
    return float(t_stat), p


# ══════════════════════════════════════════════════════════════════════════════
# METODO CANONICO — GLM Poisson (Likelihood Ratio)
# ══════════════════════════════════════════════════════════════════════════════

def glm_poisson_detect(
    series: pd.Series,
    shock:  pd.Timestamp,
    search: int    = SEARCH,
    min_seg: int   = MIN_SIZE,
    eps: float     = SHIFT_EPS,
) -> dict:
    """
    Stima il break canonico θ via massimizzazione del Likelihood Ratio
    sotto un modello Poisson a due segmenti.

    La serie viene shiftata per essere strettamente positiva prima del fit:
        y_shift = y − min(y_win) + eps

    Per ogni τ candidato:
        λ_pre  = mean(y_shift[u:τ])
        λ_post = mean(y_shift[τ:w])
        LL(τ)  = Σ_{pre}(y·log λ_pre − λ_pre) + Σ_{post}(y·log λ_post − λ_post)
        LR(τ)  = 2·[LL(τ) − LL_null]

    θ = argmax LR(τ).  Sotto H₀: LR ~ χ²(1).
    theta_confirmed = True se p(LR_max) < 0.05.

    Returns
    -------
    dict con chiavi:
      theta            – pd.Timestamp
      lr_stat          – float (LR massimo)
      lr_values        – pd.Series (curva LR su tutte le date)
      y_shift          – pd.Series (serie shiftata nella finestra)
      theta_confirmed  – bool
      p_value          – float
    """
    mask = (
        (series.index >= shock - pd.Timedelta(days=search)) &
        (series.index <= shock + pd.Timedelta(days=search))
    )
    win = series[mask].dropna()

    _fallback = {
        "theta": shock, "lr_stat": 0.0,
        "lr_values": pd.Series(dtype=float),
        "y_shift":   pd.Series(dtype=float),
        "theta_confirmed": False, "p_value": 1.0,
    }

    if len(win) < 2 * min_seg + 1:
        return _fallback

    # Shift per positività
    y_shift = win - win.min() + eps
    y = y_shift.values
    n = len(y)

    # LL del modello nullo (un unico λ su tutta la finestra)
    lam_null = y.mean()
    ll_null  = float(np.sum(y * np.log(lam_null) - lam_null))

    lr_vals:    list[float]        = []
    cand_dates: list[pd.Timestamp] = []

    for v in range(min_seg, n - min_seg):
        y_pre  = y[:v]
        y_post = y[v:]
        lam_pre  = y_pre.mean()
        lam_post = y_post.mean()
        if lam_pre <= 0 or lam_post <= 0:
            continue
        ll = (np.sum(y_pre  * np.log(lam_pre)  - lam_pre) +
              np.sum(y_post * np.log(lam_post) - lam_post))
        lr_vals.append(2.0 * (float(ll) - ll_null))
        cand_dates.append(win.index[v])

    if not cand_dates:
        return {**_fallback, "y_shift": y_shift}

    lr_series = pd.Series(lr_vals, index=cand_dates)
    theta     = lr_series.idxmax()
    lr_max    = float(lr_series.max())
    p_val     = float(stats.chi2.sf(lr_max, df=1))
    confirmed = p_val < 0.05

    return {
        "theta":           theta,
        "lr_stat":         lr_max,
        "lr_values":       lr_series,
        "y_shift":         y_shift,
        "theta_confirmed": confirmed,
        "p_value":         p_val,
    }


# ══════════════════════════════════════════════════════════════════════════════
# L1 — Sliding-window Welch t-test (supporto)
# ══════════════════════════════════════════════════════════════════════════════

def sliding_ttest(
    series: pd.Series,
    shock: pd.Timestamp,
    half_win: int = HALF_WIN,
    search:   int = SEARCH,
    step:     int = STEP,
) -> pd.DataFrame:
    idx = series.index
    rows: list = []
    n_tests = 0
    candidates = pd.date_range(
        shock - pd.Timedelta(days=search),
        shock + pd.Timedelta(days=search),
        freq=f"{step}D",
    )
    for tau in candidates:
        pre  = series[(idx >= tau - pd.Timedelta(days=half_win)) & (idx < tau)].dropna()
        post = series[(idx >= tau) & (idx < tau + pd.Timedelta(days=half_win))].dropna()
        if len(pre) < 5 or len(post) < 5:
            continue
        t, p = _welch_neff(pre, post)
        delta = post.mean() - pre.mean()
        rows.append({"tau": tau, "t_stat": t, "p_raw": p, "delta_mean": delta,
                     "n_pre": len(pre), "n_post": len(post)})
        n_tests += 1
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    df["p_bonf"] = (df["p_raw"] * n_tests).clip(upper=1.0)
    df["abs_t"]  = df["t_stat"].abs()
    return df.sort_values("abs_t", ascending=False).reset_index(drop=True)


# ══════════════════════════════════════════════════════════════════════════════
# L2 — CUSUM (supporto)
# ══════════════════════════════════════════════════════════════════════════════

def cusum(
    series: pd.Series,
    shock: pd.Timestamp,
    pre_window: int = HALF_WIN,
) -> tuple[pd.Series, pd.Timestamp]:
    baseline = series[series.index < shock].tail(pre_window).mean()
    dev      = series - baseline
    cs       = dev.cumsum()
    peak_idx = cs.abs().idxmax()
    return cs, peak_idx


# ══════════════════════════════════════════════════════════════════════════════
# L3 — Binary Segmentation BIC (supporto, ruptures)
# ══════════════════════════════════════════════════════════════════════════════

def binseg_detect(
    series: pd.Series,
    max_bkps: int = MAX_BKPS,
    min_size: int = MIN_SIZE,
    model:    str = "rbf",
) -> dict:
    if not HAS_RUPTURES:
        return {}
    signal = series.values.reshape(-1, 1)
    algo   = rpt.Binseg(model=model, min_size=min_size).fit(signal)
    results: dict = {}
    costs: list   = []
    for n in range(1, max_bkps + 1):
        try:
            bkps = algo.predict(n_bkps=n)
        except Exception:
            continue
        dates = [series.index[b - 1] for b in bkps[:-1]]
        results[n] = dates
        cost = algo.cost.sum_of_costs(bkps)
        bic  = cost + n * np.log(len(signal))
        costs.append((n, bic))
    if costs:
        best_n = min(costs, key=lambda x: x[1])[0]
        results["best"]      = results[best_n]
        results["best_n"]    = best_n
        results["bic_curve"] = costs
    return results


# ══════════════════════════════════════════════════════════════════════════════
# L4 — PELT BIC (supporto, ruptures)
# ══════════════════════════════════════════════════════════════════════════════

def pelt_detect(
    series: pd.Series,
    min_size: int = MIN_SIZE,
    model:    str = "rbf",
) -> list[pd.Timestamp]:
    if not HAS_RUPTURES:
        return []
    signal = series.values.reshape(-1, 1)
    pen    = np.log(len(signal))
    try:
        algo = rpt.Pelt(model=model, min_size=min_size).fit(signal)
        bkps = algo.predict(pen=pen)
    except Exception as e:
        warnings.warn(f"PELT fallito: {e}")
        return []
    return [series.index[b - 1] for b in bkps[:-1]]


# ══════════════════════════════════════════════════════════════════════════════
# Lw — Window Discrepancy L2  (vecchio metodo canonico, mantenuto per confronto)
# ══════════════════════════════════════════════════════════════════════════════

def _l2_cost(y: np.ndarray) -> float:
    if len(y) < 2:
        return 0.0
    return float(np.sum((y - y.mean()) ** 2))


def window_l2_detect(
    series: pd.Series,
    shock:  pd.Timestamp,
    search: int    = SEARCH,
    ma_win: int    = 7,
    min_seg: int   = MIN_SIZE,
) -> dict:
    """
    Window Discrepancy L2 — vecchio metodo canonico di 02c v1.
    Mantenuto solo per confronto nel comparison plot.
    d(y_uv, y_vw) = c(y_uw) − c(y_uv) − c(y_vw)
    θ_old = argmax d
    """
    mask = (
        (series.index >= shock - pd.Timedelta(days=search)) &
        (series.index <= shock + pd.Timedelta(days=search))
    )
    win = series[mask].dropna()
    _fb = {"theta_old": shock, "d_max": 0.0, "d_values": pd.Series(dtype=float)}
    if len(win) < 2 * min_seg + ma_win:
        return _fb
    ma   = win.rolling(ma_win, center=True, min_periods=1).mean()
    y    = ma.values
    n    = len(y)
    c_uw = _l2_cost(y)
    d_vals:     list[float]        = []
    cand_dates: list[pd.Timestamp] = []
    for v in range(min_seg, n - min_seg):
        d_vals.append(c_uw - _l2_cost(y[:v]) - _l2_cost(y[v:]))
        cand_dates.append(ma.index[v])
    if not cand_dates:
        return _fb
    d_series  = pd.Series(d_vals, index=cand_dates)
    theta_old = d_series.idxmax()
    return {"theta_old": theta_old, "d_max": float(d_series.max()), "d_values": d_series}


# ══════════════════════════════════════════════════════════════════════════════
# Grafico diagnostico — un evento, un carburante (5 pannelli)
# ══════════════════════════════════════════════════════════════════════════════

def plot_event_fuel(
    ev_name:    str,
    ev:         dict,
    series:     pd.Series,
    fuel_label: str,
    fuel_color: str,
    axes:       list,           # [ax_glm, ax_zoom, ax_cusum, ax_ttest, ax_bic]
    detect_type: str = "margin",
) -> dict | None:
    """
    Riempie 5 axes verticali per (evento, carburante).
    Restituisce il dict dei risultati da includere in theta_results.csv.
    """
    shock = ev["shock"]
    win   = series[(series.index >= ev["pre_start"]) &
                   (series.index <= ev["post_end"])].dropna()

    if len(win) < 2 * MIN_SIZE + 1:
        for ax in axes:
            ax.text(0.5, 0.5, "Dati insufficienti",
                    ha="center", va="center", transform=ax.transAxes, fontsize=8)
        return None

    ax_glm, ax_zoom, ax_cusum, ax_ttest, ax_bic = axes
    color = fuel_color

    # ── Esegui tutti i metodi ────────────────────────────────────────────────
    glm_res    = glm_poisson_detect(series, shock)
    ttest_df   = sliding_ttest(win, shock)
    cs, cusum_peak = cusum(win, shock)
    binseg_res = binseg_detect(win)
    pelt_bkps  = pelt_detect(win)
    lw_res     = window_l2_detect(series, shock)

    theta = glm_res["theta"]
    lr_max = glm_res["lr_stat"]
    p_val  = glm_res["p_value"]
    confirmed = glm_res["theta_confirmed"]

    l1_tau   = ttest_df.iloc[0]["tau"]        if not ttest_df.empty else shock
    l1_p     = ttest_df.iloc[0]["p_bonf"]     if not ttest_df.empty else 1.0
    l1_delta = ttest_df.iloc[0]["delta_mean"] if not ttest_df.empty else 0.0
    l1_t     = ttest_df.iloc[0]["t_stat"]     if not ttest_df.empty else 0.0

    binseg_best = binseg_res.get("best", [])
    theta_old   = lw_res["theta_old"]

    conf_str   = "✓ p<0.05" if confirmed else f"⚠ p={p_val:.3f}"
    conf_color = "#2ecc71"  if confirmed else "#e67e22"

    # ── Pannello 1: curva LR (GLM Poisson) ──────────────────────────────────
    lr_vals = glm_res["lr_values"]
    if not lr_vals.empty:
        ax_glm.plot(lr_vals.index, lr_vals.values, color=color, lw=1.1, label="LR(τ)")
        ax_glm.axhline(stats.chi2.ppf(0.95, df=1), color="grey", lw=0.8, ls="--",
                       label="χ²(1) α=0.05")
        ax_glm.axvline(shock, color=ev["color"],  lw=1.5, ls="--", label="Shock")
        ax_glm.axvline(theta, color=color,        lw=2.2, ls="-",
                       label=f"θ={theta.date()} {conf_str}")
        ax_glm.scatter([theta], [lr_max], color=color, s=90, zorder=6, marker="*")
        for i, b in enumerate(pelt_bkps):
            if lr_vals.index.min() <= b <= lr_vals.index.max():
                ax_glm.axvline(b, color="#27ae60", lw=0.9, ls=":",
                               alpha=0.7, label="L4 PELT" if i == 0 else "")
        ax_glm.set_ylabel("LR statistic", fontsize=8)
        ax_glm.set_title(
            f"GLM Poisson θ={theta.date()}  LR={lr_max:.2f}  {conf_str}",
            fontsize=8, fontweight="bold", color=conf_color, pad=3,
        )
        ax_glm.legend(fontsize=6, loc="upper left", ncol=2)
    else:
        ax_glm.text(0.5, 0.5, "GLM: dati insufficienti",
                    ha="center", va="center", transform=ax_glm.transAxes, fontsize=8)
    ax_glm.grid(alpha=0.25)
    ax_glm.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))

    # ── Pannello 2: zoom ±ZOOM_WIN giorni attorno a θ + tutti i break ───────
    z0 = theta - pd.Timedelta(days=ZOOM_WIN)
    z1 = theta + pd.Timedelta(days=ZOOM_WIN)
    wz = win[(win.index >= z0) & (win.index <= z1)]

    ax_zoom.plot(wz.index, wz.values, color=color, lw=0.9, label=fuel_label)
    ax_zoom.axvline(shock,      color=ev["color"],   lw=1.8, ls="--",  label="Shock")
    ax_zoom.axvline(theta,      color=color,         lw=2.2, ls="-",
                    alpha=0.9, label=f"θ GLM={theta.date()}")
    ax_zoom.axvline(l1_tau,     color="#3498db",     lw=1.1, ls=":",
                    alpha=0.8, label=f"L1 τ={l1_tau.date()}")
    ax_zoom.axvline(cusum_peak, color="#e67e22",     lw=1.1, ls="-.",
                    alpha=0.8, label=f"L2 CUSUM={cusum_peak.date()}")
    ax_zoom.axvline(theta_old,  color="#95a5a6",     lw=1.1, ls="--",
                    alpha=0.8, label=f"Lw WinDisc={theta_old.date()}")
    for i, b in enumerate(pelt_bkps):
        if z0 <= b <= z1:
            ax_zoom.axvline(b, color="#27ae60", lw=0.9, ls=":", alpha=0.7,
                            label="L4 PELT" if i == 0 else "")
    for i, b in enumerate(binseg_best):
        if z0 <= b <= z1:
            ax_zoom.axvline(b, color="#9b59b6", lw=0.9, ls=":", alpha=0.7,
                            label="L3 BinSeg" if i == 0 else "")
    ax_zoom.set_xlim(z0, z1)
    series_label = "Margine (€/L)" if detect_type == "margin" else "Prezzo netto (€/L)"
    ax_zoom.set_ylabel(series_label, fontsize=8)
    ax_zoom.set_title(f"Zoom ±{ZOOM_WIN}gg — tutti i metodi a confronto", fontsize=8, pad=3)
    ax_zoom.legend(fontsize=6, loc="upper left", ncol=2)
    ax_zoom.grid(axis="y", alpha=0.25)
    ax_zoom.xaxis.set_major_formatter(mdates.DateFormatter("%d %b"))
    ax_zoom.xaxis.set_major_locator(mdates.WeekdayLocator(interval=2))
    plt.setp(ax_zoom.xaxis.get_majorticklabels(), rotation=30, ha="right", fontsize=7)

    # ── Pannello 3: CUSUM ────────────────────────────────────────────────────
    ax_cusum.plot(cs.index, cs.values, color=color, lw=0.9)
    ax_cusum.axhline(0, color="grey", lw=0.7, ls="--")
    ax_cusum.axvline(shock,      color=ev["color"],  lw=1.5, ls="--")
    ax_cusum.axvline(theta,      color=color,        lw=1.8, ls="-",
                     alpha=0.8, label=f"θ={theta.date()}")
    ax_cusum.axvline(cusum_peak, color="#e67e22",    lw=1.0, ls="-.",
                     alpha=0.8, label=f"L2={cusum_peak.date()}")
    ax_cusum.fill_between(cs.index, cs.values, 0, where=cs.values > 0,
                          alpha=0.15, color=color)
    ax_cusum.fill_between(cs.index, cs.values, 0, where=cs.values < 0,
                          alpha=0.15, color="green")
    ax_cusum.set_ylabel("CUSUM (€/L)", fontsize=8)
    ax_cusum.set_title("L2 — CUSUM (supporto)", fontsize=8, pad=3)
    ax_cusum.legend(fontsize=6)
    ax_cusum.grid(axis="y", alpha=0.25)
    ax_cusum.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))

    # ── Pannello 4: sliding t-stat ────────────────────────────────────────
    if not ttest_df.empty:
        rising = ttest_df[ttest_df["delta_mean"] > 0]
        ax_ttest.plot(rising["tau"], rising["t_stat"].abs(),
                      color=color, lw=0.9, label="Δ>0")
        ax_ttest.axvline(shock,  color=ev["color"], lw=1.5, ls="--", label="Shock")
        ax_ttest.axvline(theta,  color=color,       lw=1.8, ls="-",
                         alpha=0.8, label=f"θ={theta.date()}")
        ax_ttest.axvline(l1_tau, color="#3498db",   lw=1.0, ls=":",
                         alpha=0.8, label=f"L1 τ={l1_tau.date()}")
        ax_ttest.axhline(stats.t.ppf(0.975, df=HALF_WIN * 2 - 2),
                         color="grey", lw=0.7, ls=":", label="α=0.05")
        ax_ttest.set_ylabel("|t-stat| (Δ>0)", fontsize=8)
        ax_ttest.set_title(
            f"L1 — sliding t-test  |  τ={l1_tau.date()}  "
            f"Δ={l1_delta:+.4f} €/L  p_bonf={l1_p:.3f}",
            fontsize=7, pad=2,
        )
        ax_ttest.legend(fontsize=6)
    ax_ttest.grid(axis="y", alpha=0.25)
    ax_ttest.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))

    # ── Pannello 5: BIC curve BinSeg ─────────────────────────────────────────
    if binseg_res.get("bic_curve"):
        ns, bics = zip(*binseg_res["bic_curve"])
        ax_bic.plot(ns, bics, marker="o", color=color, lw=1)
        best_n = binseg_res.get("best_n", 1)
        ax_bic.axvline(best_n, color="grey", lw=0.8, ls="--", label=f"N={best_n}")
        ax_bic.set_xlabel("N break point", fontsize=8)
        ax_bic.set_ylabel("BIC", fontsize=8)
        ax_bic.set_xticks(list(ns))
        ax_bic.legend(fontsize=6)
        ax_bic.set_title(
            f"L3 BinSeg: {len(binseg_best)} break  |  "
            f"L4 PELT: {len(pelt_bkps)} break  →  "
            + (", ".join(str(d.date()) for d in pelt_bkps) if pelt_bkps else "nessuno"),
            fontsize=7, pad=2,
        )
    else:
        ax_bic.text(0.5, 0.5, "ruptures non disponibile",
                    ha="center", va="center", transform=ax_bic.transAxes, fontsize=8)
    ax_bic.grid(alpha=0.25)

    return {
        "theta":           theta,
        "lr_stat":         lr_max,
        "theta_confirmed": confirmed,
        "p_value":         p_val,
        "L1_tau":          l1_tau,
        "L1_delta_eurl":   round(l1_delta, 5),
        "L1_p_bonf":       round(l1_p,     4),
        "L1_t":            round(l1_t,     4),
        "L2_cusum":        cusum_peak,
        "L3_binseg":       binseg_best,
        "L3_best_n":       binseg_res.get("best_n"),
        "L4_pelt":         pelt_bkps,
        "Lw_theta_old":    theta_old,
        "Lw_d_max":        round(lw_res["d_max"], 6),
    }


# ══════════════════════════════════════════════════════════════════════════════
# Comparison plot — griglia eventi × carburanti con tutti i metodi
# ══════════════════════════════════════════════════════════════════════════════

def comparison_plot(
    all_results:  dict,   # {ev_name: {fuel_key: result_dict}}
    series_data:  dict,   # {fuel_key: pd.Series}
    detect_type:  str,
    out_path:     Path,
) -> None:
    """
    Genera una figura riepilogativa con una griglia:
      righe    = eventi geopolitici
      colonne  = carburanti (benzina, gasolio)
    In ogni pannello mostra la serie temporale con linee verticali
    per ciascun metodo di detection.
    """
    ev_names   = [e for e in EVENTS if e in all_results]
    fuel_keys  = list(FUELS.keys())
    n_rows     = len(ev_names)
    n_cols     = len(fuel_keys)

    if n_rows == 0:
        return

    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(7 * n_cols, 4.5 * n_rows),
        squeeze=False,
    )
    series_label = "Margine distributore (€/L)" if detect_type == "margin" \
                   else "Prezzo netto pompa (€/L)"
    fig.suptitle(
        f"Confronto metodi change-point — {detect_type.upper()}\n"
        f"Metodo canonico: GLM Poisson (θ); Metodi confronto: L1 L2 L3 L4 Lw",
        fontsize=12, fontweight="bold",
    )

    for r_i, ev_name in enumerate(ev_names):
        ev = EVENTS[ev_name]
        for c_i, fuel_key in enumerate(fuel_keys):
            ax  = axes[r_i][c_i]
            res = all_results.get(ev_name, {}).get(fuel_key)
            ser = series_data.get(fuel_key)
            _, fuel_color = FUELS[fuel_key]

            if ser is None:
                ax.text(0.5, 0.5, "N/A", ha="center", va="center",
                        transform=ax.transAxes)
                continue

            # Traccia la serie nella finestra dell'evento
            mask = (ser.index >= ev["pre_start"]) & (ser.index <= ev["post_end"])
            wser = ser[mask].dropna()
            if wser.empty:
                ax.text(0.5, 0.5, "Nessun dato",
                        ha="center", va="center", transform=ax.transAxes)
                continue

            ax.plot(wser.index, wser.values, color=fuel_color,
                    lw=0.8, alpha=0.7, label=fuel_key)
            ax.axvline(ev["shock"], **METHOD_STYLES["Shock"], label="Shock")

            if res:
                theta = res.get("theta")
                if theta:
                    ax.axvline(theta, **METHOD_STYLES["GLM Poisson θ"],
                               label=f"θ GLM = {theta.date()}")
                for key, style_key, label_tpl in [
                    ("L1_tau",       "L1 Sliding t",  "L1 = {d}"),
                    ("L2_cusum",     "L2 CUSUM",      "L2 = {d}"),
                    ("Lw_theta_old", "Lw WinDisc L2", "Lw = {d}"),
                ]:
                    val = res.get(key)
                    if val:
                        ax.axvline(val, **METHOD_STYLES[style_key],
                                   label=label_tpl.format(d=pd.Timestamp(val).date()))
                for i, b in enumerate(res.get("L3_binseg", [])):
                    ax.axvline(b, **METHOD_STYLES["L3 BinSeg"],
                               label="L3 BinSeg" if i == 0 else "")
                for i, b in enumerate(res.get("L4_pelt", [])):
                    ax.axvline(b, **METHOD_STYLES["L4 PELT"],
                               label="L4 PELT" if i == 0 else "")

            ax.set_title(
                f"{ev_name} — {fuel_key.capitalize()}",
                fontsize=8, fontweight="bold", pad=3,
            )
            ax.set_ylabel(series_label, fontsize=7)
            ax.grid(axis="y", alpha=0.2)
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=30, ha="right", fontsize=7)
            ax.legend(fontsize=6, loc="upper left", ncol=1)

    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  → Comparison plot: {out_path}")


# ══════════════════════════════════════════════════════════════════════════════
# Caricamento dati
# ══════════════════════════════════════════════════════════════════════════════

def _load_futures_eurl(path: Path, hc, eurusd: pd.Series) -> pd.Series:
    df = pd.read_csv(path, encoding="utf-8-sig", dtype=str)
    df["date"] = pd.to_datetime(df["Date"], format="%m/%d/%Y", errors="coerce")
    df["price_usd_ton"] = (df["Price"].str.replace(",", "", regex=False)
                           .pipe(pd.to_numeric, errors="coerce"))
    df = df.dropna(subset=["date", "price_usd_ton"]).sort_values("date").set_index("date")
    return usd_ton_to_eur_liter(df["price_usd_ton"], eurusd, hc)


def _load_futures_b7h1(path: Path, hc, eurusd: pd.Series) -> pd.Series:
    df = pd.read_csv(path, encoding="utf-8-sig", dtype=str)
    _IT = {
        "gen": "Jan", "feb": "Feb", "mar": "Mar", "apr": "Apr",
        "mag": "May", "giu": "Jun", "lug": "Jul", "ago": "Aug",
        "set": "Sep", "ott": "Oct", "nov": "Nov", "dic": "Dec",
    }
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
    df["price_usd_ton"] = pd.to_numeric(df["chiusura"], errors="coerce")
    df = (df.dropna(subset=["date", "price_usd_ton"])
            .sort_values("date").set_index("date"))
    df = df[~df.index.duplicated(keep="first")]
    return usd_ton_to_eur_liter(df["price_usd_ton"], eurusd, hc)


def load_all_data() -> pd.DataFrame:
    """Carica daily prices, futures, costruisce margin e price netto."""
    daily = (pd.read_csv(DAILY_CSV, parse_dates=["date"])
               .sort_values("date").set_index("date"))
    print("Carico futures e EUR/USD...")
    eurusd = load_eurusd(
        csv_path=EURUSD_CSV if EURUSD_CSV.exists() else None,
        start="2015-01-01", end="2026-12-31",
    )
    gasoil_eurl  = _load_futures_eurl(GASOIL_CSV, GAS_OIL, eurusd)
    eurobob_eurl = _load_futures_b7h1(EUROBOB_CSV, EUROBOB_HC, eurusd) \
                   if EUROBOB_CSV.exists() else None

    df = daily[["benzina_net", "gasolio_net"]].copy()
    df["margin_gasolio"] = df["gasolio_net"] - gasoil_eurl.reindex(df.index, method="ffill")
    if eurobob_eurl is not None:
        df["margin_benzina"] = df["benzina_net"] - eurobob_eurl.reindex(df.index, method="ffill")
    else:
        df["margin_benzina"] = np.nan
    return df


# ══════════════════════════════════════════════════════════════════════════════
# Run per un singolo detect_type ("margin" o "price")
# ══════════════════════════════════════════════════════════════════════════════

def run_detect(daily: pd.DataFrame, detect_type: str) -> None:
    """Esegue la detection (GLM Poisson + supporto) su margine o prezzo."""
    assert detect_type in ("margin", "price")

    out_dir = BASE_DIR / "data" / "plots" / "change_point" / detect_type
    out_dir.mkdir(parents=True, exist_ok=True)

    # Seleziona le colonne in base a detect_type
    if detect_type == "margin":
        col_map = {fk: col for fk, (col, _) in FUELS.items()}
    else:  # price
        col_map = {fk: PRICE_COLS[fk] for fk in FUELS}

    series_data: dict[str, pd.Series] = {}
    for fk, col in col_map.items():
        if col in daily.columns:
            series_data[fk] = daily[col].dropna()
        else:
            print(f"  ⚠ Colonna '{col}' non trovata — {fk} saltato.")

    all_results: dict = {}
    theta_rows:  list = []

    for ev_name, ev in EVENTS.items():
        shock = ev["shock"]
        ev_results: dict = {}

        # Controlla disponibilità dati
        any_data = any(
            not ser[(ser.index >= ev["pre_start"]) & (ser.index <= ev["post_end"])].empty
            for ser in series_data.values()
        )
        if not any_data:
            print(f"  ⚠ {ev_name}: nessun dato disponibile, salto.")
            continue

        print(f"\n{'═'*70}")
        print(f"  EVENTO: {ev_name}  ({detect_type})  shock={shock.date()}")
        print(f"  Finestra: {ev['pre_start'].date()} → {ev['post_end'].date()}")
        print(f"{'═'*70}")

        # Figura: 5 righe × 2 colonne
        fig = plt.figure(figsize=(18, 20))
        fig.suptitle(
            f"GLM Poisson θ + diagnostica — {ev_name} [{detect_type.upper()}]\n"
            f"Shock: {ev['label'].replace(chr(10), ' ')}",
            fontsize=13, fontweight="bold",
        )
        gs = gridspec.GridSpec(5, 2, figure=fig, hspace=0.60, wspace=0.30)

        for col_idx, (fuel_key, (_, fuel_color)) in enumerate(FUELS.items()):
            ser = series_data.get(fuel_key)
            if ser is None:
                continue

            axes = [
                fig.add_subplot(gs[row, col_idx])
                for row in range(5)
            ]
            # Etichetta colonna in testa
            axes[0].set_title(
                f"{fuel_key.capitalize()} — GLM Poisson θ [{detect_type}]",
                fontsize=9, fontweight="bold", pad=4,
            )
            res = plot_event_fuel(
                ev_name, ev, ser,
                fuel_label=fuel_key.capitalize(),
                fuel_color=fuel_color,
                axes=axes,
                detect_type=detect_type,
            )
            ev_results[fuel_key] = res

            if res:
                conf_icon = "✓" if res["theta_confirmed"] else "⚠"
                print(f"\n  [{fuel_key.upper()}]")
                print(f"    θ  = {res['theta'].date()}  "
                      f"LR={res['lr_stat']:.2f}  p={res['p_value']:.4f}  "
                      f"{conf_icon}")
                print(f"    L1 = {res['L1_tau'].date()}  Δ={res['L1_delta_eurl']:+.4f}  "
                      f"p_bonf={res['L1_p_bonf']:.3f}")
                print(f"    L2 CUSUM = {res['L2_cusum'].date()}")
                print(f"    L3 BinSeg: "
                      + (", ".join(str(d.date()) for d in res["L3_binseg"]) or "—"))
                print(f"    L4 PELT:   "
                      + (", ".join(str(d.date()) for d in res["L4_pelt"]) or "—"))
                print(f"    Lw (vecchio L2 disc.) = {res['Lw_theta_old'].date()}")

                theta_rows.append({
                    "evento":          ev_name,
                    "shock":           shock.date(),
                    "carburante":      fuel_key,
                    "detect_type":     detect_type,
                    "theta":           res["theta"].date(),
                    "lr_stat":         round(res["lr_stat"],    4),
                    "p_value":         round(res["p_value"],    5),
                    "theta_confirmed": res["theta_confirmed"],
                    "L1_tau":          res["L1_tau"].date(),
                    "L1_delta_eurl":   res["L1_delta_eurl"],
                    "L1_p_bonf":       res["L1_p_bonf"],
                    "L2_cusum":        res["L2_cusum"].date(),
                    "L3_binseg":       "; ".join(str(d.date()) for d in res["L3_binseg"]),
                    "L4_pelt":         "; ".join(str(d.date()) for d in res["L4_pelt"]),
                    "Lw_theta_old":    res["Lw_theta_old"].date(),
                    "Lw_d_max":        res["Lw_d_max"],
                })

        all_results[ev_name] = ev_results

        fname = f"cp_{detect_type}_{ev_name.replace(' ', '_').replace('/', '')}.png"
        fig.savefig(out_dir / fname, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  → {out_dir / fname}")

    # ── Export theta_results.csv ─────────────────────────────────────────────
    if theta_rows:
        theta_csv = out_dir / "theta_results.csv"
        pd.DataFrame(theta_rows).to_csv(theta_csv, index=False, encoding="utf-8-sig")
        print(f"\n  → Esportato: {theta_csv}")

    # ── Comparison plot ──────────────────────────────────────────────────────
    cmp_path = out_dir / "comparison_all_methods.png"
    comparison_plot(all_results, series_data, detect_type, cmp_path)

    # ── Riepilogo ────────────────────────────────────────────────────────────
    print(f"\n{'═'*70}")
    print(f"  RIEPILOGO θ (GLM Poisson) — detect_type = {detect_type}")
    print(f"{'═'*70}")
    print(f"  {'Evento':<28} {'Carb.':<10} {'shock':<13} {'θ':<13} "
          f"{'LR':>7} {'p':>7} {'OK'}")
    print("  " + "-" * 75)
    for row in theta_rows:
        conf = "✓" if row["theta_confirmed"] else "⚠"
        print(f"  {row['evento']:<28} {row['carburante']:<10} "
              f"{str(row['shock']):<13} {str(row['theta']):<13} "
              f"{row['lr_stat']:>7.2f} {row['p_value']:>7.4f}  {conf}")


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    ap = argparse.ArgumentParser(
        description="Change-point detection (GLM Poisson) su margine o prezzo."
    )
    ap.add_argument(
        "--detect",
        choices=["margin", "price", "both"],
        default="margin",
        help="Serie su cui eseguire la detection (default: margin)",
    )
    args = ap.parse_args()

    daily = load_all_data()
    print(f"\nDati: {daily.index.min().date()} → {daily.index.max().date()}")
    print(f"Config: SEARCH=±{SEARCH}g  MIN_SIZE={MIN_SIZE}g\n")

    detect_types = ["margin", "price"] if args.detect == "both" else [args.detect]
    for dt in detect_types:
        print(f"\n{'█'*70}")
        print(f"  DETECTION: {dt.upper()}")
        print(f"{'█'*70}")
        run_detect(daily, dt)


if __name__ == "__main__":
    main()