#!/usr/bin/env python3
"""
02_change_point_detection_margin.py
=====================================
Change-point detection sul MARGINE (prezzo netto pompa − futures €/L)
benzina e gasolio in corrispondenza di eventi geopolitici rilevanti.

Metodologia — Rasoio di Occam (dal più semplice al più sofisticato)
────────────────────────────────────────────────────────────────────
  L1 · Sliding-window two-sample t-test
       Finestra di HALF_WIN=40 giorni per lato; il punto di rottura τ
       scorre da –SEARCH giorni a +SEARCH giorni rispetto all'evento
       con step=1 (risoluzione giornaliera). Correzione p-value: Bonferroni.
       → restituisce 1 break point (il più forte).

  L2 · CUSUM (Cumulative SUM of deviations)
       Deviazione cumulata dalla media pre-evento sul livello di prezzo.
       Picco/valle = momento di rottura strutturale. Semplice, visivo.
       → 1 break point.

  L3 · Binary Segmentation (ruptures – BinSeg, costo L2)
       Cerca esattamente N break points; usa BIC per scegliere N ottimale
       tra 1 e MAX_BKPS. → potenzialmente MULTIPLI break point.

  L4 · PELT – Pruned Exact Linear Time (ruptures, penalizzazione BIC)
       Soluzione globale esatta con N incognito; O(n log n).
       → potenzialmente MULTIPLI break point.

Finestra ottimale e CLT
────────────────────────
  Prezzi giornalieri: autocorrelazione AR(1) con φ ≈ 0.2–0.3.
  Dimensione effettiva del campione: n_eff ≈ n·(1-φ)/(1+φ).
  Con φ=0.3 → n_eff = n·0.54.
  Per n_eff ≥ 30 (CLT solido) serve n ≥ 56 → uso HALF_WIN=40 (minimo
  pratico, n_eff≈22) con la consapevolezza che il test t è approssimato.
  Se vuoi CLT garantito imposta HALF_WIN=60.

  Sliding step = 1 giorno (risoluzione massima, 2·SEARCH+1 test totali;
  si applica correzione Bonferroni sui p-value).
  Step = 7 riduce il multiple testing ma perde risoluzione infrasettimanale.

Dipendenze extra:
  pip install ruptures
"""

from __future__ import annotations

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
    warnings.warn("ruptures non installato – L3/L4 disabilitati. "
                  "Installa con: pip install ruptures")

# ══════════════════════════════════════════════════════════════════════════════
# n_eff helpers (AR(1) autocorrelation correction)
# ══════════════════════════════════════════════════════════════════════════════
def _phi_ar1(x: np.ndarray) -> float:
    """Lag-1 autocorrelation (AR(1) coefficient) — clamped to (-0.99, 0.99)."""
    if len(x) < 3:
        return 0.0
    x = x - x.mean()
    r = float(np.corrcoef(x[:-1], x[1:])[0, 1])
    return float(np.clip(r, -0.99, 0.99))


def _n_eff(x: np.ndarray) -> float:
    """Effective sample size corrected for AR(1) autocorrelation."""
    phi = _phi_ar1(x)
    return max(2.0, len(x) * (1 - phi) / (1 + phi))


def _welch_neff(a: np.ndarray, b: np.ndarray):
    """
    Welch t-test with n_eff-corrected degrees of freedom.
    Returns (t_stat, p_value).
    """
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    na, nb = _n_eff(a), _n_eff(b)
    va, vb = a.var(ddof=1) / na, b.var(ddof=1) / nb
    se = np.sqrt(va + vb)
    if se == 0:
        return 0.0, 1.0
    t = (a.mean() - b.mean()) / se
    # Welch–Satterthwaite df
    df = (va + vb) ** 2 / (va ** 2 / (na - 1) + vb ** 2 / (nb - 1))
    p = 2 * stats.t.sf(abs(t), df=df)
    return float(t), float(p)


# ── Configurazione ─────────────────────────────────────────────────────────────
BASE_DIR     = Path(__file__).parent
DAILY_CSV    = BASE_DIR / "data" / "processed" / "daily_fuel_prices_all.csv"
GASOIL_CSV   = BASE_DIR / "data" / "Futures" / "London Gas Oil Futures Historical Data.csv"
EUROBOB_CSV  = BASE_DIR / "data" / "Futures" / "Eurobob_B7H1_date.csv"
EURUSD_CSV   = BASE_DIR / "data" / "raw" / "eurusd.csv"
OUT_DIR      = BASE_DIR / "data" / "plots" / "change_point" / "margin"
OUT_DIR.mkdir(parents=True, exist_ok=True)

HALF_WIN   = 40
SEARCH     = 40
STEP       = 1
MAX_BKPS   = 5
MIN_SIZE   = 14

FUELS = {
    "benzina": ("margin_benzina", "#E63946"),
    "gasolio": ("margin_gasolio", "#1D3557"),
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


# ══════════════════════════════════════════════════════════════════════════════
# L1 – Sliding-window t-test
# ══════════════════════════════════════════════════════════════════════════════
def _bh_correction(p_values: np.ndarray) -> np.ndarray:
    """
    Benjamini-Hochberg FDR correction (Benjamini & Hochberg, 1995).
    Preferibile a Bonferroni per test correlati (finestre sliding si sovrappongono):
      • Bonferroni assume indipendenza e ipercorregge su test correlati.
      • BH controlla il False Discovery Rate; mantiene maggiore potenza.
    Restituisce p-value aggiustati (clip a 1.0).
    """
    n = len(p_values)
    if n == 0:
        return p_values
    order = np.argsort(p_values)
    rank  = np.empty(n, dtype=float)
    rank[order] = np.arange(1, n + 1)
    # p_adj[i] = min(p_raw[i] * n / rank[i], 1)  — calcolo BH step-up
    p_adj = np.minimum(p_values * n / rank, 1.0)
    # Step-up: garantisce monotonia (p_adj deve essere non-decrescente in rank)
    p_adj_sorted = p_adj[order]
    for i in range(n - 2, -1, -1):
        p_adj_sorted[i] = min(p_adj_sorted[i], p_adj_sorted[i + 1])
    p_adj[order] = p_adj_sorted
    return p_adj


def sliding_ttest(
    series: pd.Series,
    shock: pd.Timestamp,
    half_win: int = HALF_WIN,
    search: int   = SEARCH,
    step: int     = STEP,
) -> pd.DataFrame:
    """
    Per ogni candidato τ in [shock–search, shock+search] (step=step giorni),
    esegue un Welch t-test con n_eff tra i HALF_WIN giorni prima e dopo τ.
    Restituisce DataFrame ordinato per |t_stat| decrescente.

    Correzione p-value: Benjamini-Hochberg (FDR).
      La correzione Bonferroni (precedente) assume indipendenza dei test,
      ma le finestre sliding si sovrappongono → test correlati → Bonferroni
      ipercorregge (rigetta meno del dovuto). BH è più appropriato e mantiene
      maggiore potenza in presenza di correlazione positiva tra test.

    Nota sul CLT:
      Con half_win=40 e autocorrelazione AR(1) φ≈0.3,
      n_eff ≈ 40 * 0.54 ≈ 22 < 30  →  CLT approssimativo.
      Imposta half_win=60 per CLT garantito.
    """
    idx  = series.index
    rows = []

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
        t, p = _welch_neff(pre.values, post.values)
        delta = post.mean() - pre.mean()
        rows.append({"tau": tau, "t_stat": t, "p_raw": p, "delta_mean": delta,
                     "n_pre": len(pre), "n_post": len(post)})

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    # Correzione BH (FDR) — sostituisce Bonferroni per test sliding correlati
    df["p_bh"]  = _bh_correction(df["p_raw"].values)
    # Mantieni p_bonf per confronto/retrocompatibilità CSV
    df["p_bonf"] = (df["p_raw"] * len(df)).clip(upper=1.0)
    df["abs_t"]  = df["t_stat"].abs()
    return df.sort_values("abs_t", ascending=False).reset_index(drop=True)


# ══════════════════════════════════════════════════════════════════════════════
# L2 – CUSUM
# ══════════════════════════════════════════════════════════════════════════════
def cusum(
    series: pd.Series,
    shock: pd.Timestamp,
    pre_window: int = HALF_WIN,
) -> tuple[pd.Series, pd.Timestamp]:
    """
    CUSUM delle deviazioni dalla media pre-evento.
    Restituisce (serie CUSUM, data del massimo |CUSUM|).
    """
    baseline = series[series.index < shock].tail(pre_window).mean()
    dev      = series - baseline
    cs       = dev.cumsum()
    peak_idx = cs.abs().idxmax()
    return cs, peak_idx


# ══════════════════════════════════════════════════════════════════════════════
# L3 – Binary Segmentation  (ruptures)
# ══════════════════════════════════════════════════════════════════════════════
def binseg_detect(
    series: pd.Series,
    max_bkps: int  = MAX_BKPS,
    min_size: int  = MIN_SIZE,
    model: str     = "rbf",
) -> dict[int, list[pd.Timestamp]]:
    """
    BinSeg su 1..max_bkps break point; sceglie N ottimale con BIC.
    Restituisce {n_bkps: [date break point]} per ogni N testato,
    più la chiave 'best' con N ottimale.
    """
    if not HAS_RUPTURES:
        return {}

    signal = series.values.reshape(-1, 1)
    algo   = rpt.Binseg(model=model, min_size=min_size).fit(signal)

    results: dict = {}
    costs   = []

    for n in range(1, max_bkps + 1):
        try:
            bkps = algo.predict(n_bkps=n)  # indici (1-based, ultimo = len)
        except Exception:
            continue
        dates = [series.index[b - 1] for b in bkps[:-1]]  # escludi sentinel
        results[n] = dates
        # Costo BIC: costo_residuo + n * log(T)
        cost = algo.cost.sum_of_costs(bkps)
        bic  = cost + n * np.log(len(signal))
        costs.append((n, bic))

    if costs:
        best_n = min(costs, key=lambda x: x[1])[0]
        results["best"] = results[best_n]
        results["best_n"] = best_n
        results["bic_curve"] = costs

    return results


# ══════════════════════════════════════════════════════════════════════════════
# L4 – PELT  (ruptures)
# ══════════════════════════════════════════════════════════════════════════════
def pelt_detect(
    series: pd.Series,
    min_size: int = MIN_SIZE,
    model: str    = "rbf",
) -> list[pd.Timestamp]:
    """
    PELT con penalizzazione BIC (pen = log(T)).
    Restituisce lista di date dei change point rilevati.
    """
    if not HAS_RUPTURES:
        return []

    signal = series.values.reshape(-1, 1)
    pen    = np.log(len(signal))  # BIC
    try:
        algo = rpt.Pelt(model=model, min_size=min_size).fit(signal)
        bkps = algo.predict(pen=pen)
    except Exception as e:
        warnings.warn(f"PELT fallito: {e}")
        return []

    return [series.index[b - 1] for b in bkps[:-1]]


# ══════════════════════════════════════════════════════════════════════════════
# Plotting per evento + carburante
# ══════════════════════════════════════════════════════════════════════════════
def plot_event_fuel(
    event_name: str,
    ev: dict,
    series: pd.Series,
    fuel_label: str,
    fuel_color: str,
    ax_price:  plt.Axes,
    ax_cusum:  plt.Axes,
    ax_tstat:  plt.Axes,
    ax_bic:    plt.Axes,
) -> dict:
    """Disegna i 4 pannelli per un singolo carburante e restituisce risultati."""

    # ── Slice della finestra ──────────────────────────────────────────────────
    win = series[(series.index >= ev["pre_start"]) &
                 (series.index <= ev["post_end"])].dropna()
    if len(win) < 2 * HALF_WIN:
        ax_price.set_title(f"{fuel_label} – dati insufficienti", fontsize=9)
        return {}

    shock = ev["shock"]
    color = fuel_color

    # ── L1: sliding t-test ────────────────────────────────────────────────────
    ttest_df = sliding_ttest(win, shock)
    best_tau = ttest_df.iloc[0]["tau"]   if not ttest_df.empty else shock
    best_t   = ttest_df.iloc[0]["t_stat"] if not ttest_df.empty else 0
    best_p   = ttest_df.iloc[0]["p_bh"]  if not ttest_df.empty else 1  # BH (ex Bonferroni)
    delta    = ttest_df.iloc[0]["delta_mean"] if not ttest_df.empty else 0

    # ── L2: CUSUM ─────────────────────────────────────────────────────────────
    cs, cusum_peak = cusum(win, shock)

    # ── L3: BinSeg ───────────────────────────────────────────────────────────
    binseg_res = binseg_detect(win)
    binseg_best = binseg_res.get("best", [])

    # ── L4: PELT ─────────────────────────────────────────────────────────────
    pelt_breaks = pelt_detect(win)

    # ════ PANNELLO 1: prezzi + break lines  (zoom ±50 giorni dallo shock) ══════
    zoom_start = shock - pd.Timedelta(days=50)
    zoom_end   = shock + pd.Timedelta(days=50)
    win_zoom   = win[(win.index >= zoom_start) & (win.index <= zoom_end)]

    ax_price.plot(win_zoom.index, win_zoom.values, color=color, lw=0.9, label=fuel_label)
    ax_price.axvline(shock,     color=ev["color"], lw=1.5, ls="--", label="Evento")
    ax_price.axvline(best_tau,  color=color, lw=1.2, ls=":", alpha=0.8,
                     label=f"L1 τ={best_tau.date()}")
    ax_price.axvline(cusum_peak, color="darkorange", lw=1.0, ls="-.",
                     label=f"L2 CUSUM={cusum_peak.date()}")
    for i, d in enumerate(pelt_breaks):
        if zoom_start <= d <= zoom_end:
            ax_price.axvline(d, color="green", lw=0.8, ls=":", alpha=0.7,
                             label=f"L4 PELT" if i == 0 else "")
    for i, d in enumerate(binseg_best):
        if zoom_start <= d <= zoom_end:
            ax_price.axvline(d, color="purple", lw=0.8, ls=":", alpha=0.7,
                             label=f"L3 BinSeg" if i == 0 else "")
    ax_price.set_xlim(zoom_start, zoom_end)
    ax_price.set_ylabel("Margine (€/L)", fontsize=8)
    ax_price.legend(fontsize=6, loc="upper left", ncol=2)
    ax_price.grid(axis="y", alpha=0.25)
    ax_price.xaxis.set_major_formatter(mdates.DateFormatter("%d %b %Y"))
    ax_price.xaxis.set_major_locator(mdates.WeekdayLocator(interval=2))
    plt.setp(ax_price.xaxis.get_majorticklabels(), rotation=30, ha="right", fontsize=7)

    # ════ PANNELLO 2: CUSUM ═══════════════════════════════════════════════════
    ax_cusum.plot(cs.index, cs.values, color=color, lw=0.9)
    ax_cusum.axhline(0, color="grey", lw=0.7, ls="--")
    ax_cusum.axvline(shock,      color=ev["color"], lw=1.5, ls="--")
    ax_cusum.axvline(cusum_peak, color="darkorange", lw=1.2, ls="-.")
    ax_cusum.set_ylabel("CUSUM margine (€/L)", fontsize=8)
    ax_cusum.fill_between(cs.index, cs.values, 0,
                          where=cs.values > 0, alpha=0.15, color=color)
    ax_cusum.fill_between(cs.index, cs.values, 0,
                          where=cs.values < 0, alpha=0.15, color="green")
    ax_cusum.grid(axis="y", alpha=0.25)
    ax_cusum.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))

    # ════ PANNELLO 3: t-stat sliding — solo τ dove margine CRESCE (Δ > 0) ══════
    if not ttest_df.empty:
        rising = ttest_df[ttest_df["delta_mean"] > 0]
        ax_tstat.plot(rising["tau"], rising["t_stat"].abs(),
                      color=color, lw=0.9, label="Δ > 0 (margine cresce)")
        ax_tstat.axvline(shock,    color=ev["color"], lw=1.5, ls="--", label="Evento")
        if not rising.empty and best_tau in rising["tau"].values:
            ax_tstat.axvline(best_tau, color=color, lw=1.2, ls=":", label=f"τ={best_tau.date()}")
        ax_tstat.axhline(stats.t.ppf(0.975, df=HALF_WIN * 2 - 2),
                         color="grey", lw=0.7, ls=":", label="α=0.05")
        ax_tstat.set_ylabel("|t-stat|  (solo Δ>0)", fontsize=8)
        ax_tstat.legend(fontsize=6)
        sig = "★" if best_p < 0.05 else ""
        ax_tstat.set_title(
            f"L1 sliding t-test  |  τ={best_tau.date()}  "
            f"Δ={delta:+.4f} €/L  p_bh={best_p:.3f}{sig}  [BH-FDR]",
            fontsize=7, pad=2
        )
    ax_tstat.grid(axis="y", alpha=0.25)
    ax_tstat.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))

    # ════ PANNELLO 4: BIC curve BinSeg ════════════════════════════════════════
    if binseg_res.get("bic_curve"):
        ns, bics = zip(*binseg_res["bic_curve"])
        ax_bic.plot(ns, bics, marker="o", color=color, lw=1)
        best_n = binseg_res.get("best_n", 1)
        ax_bic.axvline(best_n, color="grey", lw=0.8, ls="--",
                       label=f"N ottimale={best_n}")
        ax_bic.set_xlabel("N break point", fontsize=8)
        ax_bic.set_ylabel("BIC", fontsize=8)
        ax_bic.set_xticks(list(ns))
        ax_bic.legend(fontsize=6)
        ax_bic.set_title(
            f"L3 BinSeg: {len(binseg_best)} break  |  "
            f"L4 PELT: {len(pelt_breaks)} break  →  "
            + (", ".join(str(d.date()) for d in pelt_breaks) if pelt_breaks else "nessuno"),
            fontsize=7, pad=2
        )
    else:
        ax_bic.text(0.5, 0.5, "ruptures non disponibile",
                    ha="center", va="center", transform=ax_bic.transAxes, fontsize=8)
    ax_bic.grid(alpha=0.25)

    return {
        "L1_tau":      best_tau,
        "L1_t":        best_t,
        "L1_p_bh":     best_p,    # BH-FDR (ex Bonferroni — più appropriato per sliding window)
        "L1_delta":    delta,
        "L2_cusum":    cusum_peak,
        "L3_binseg":   binseg_best,
        "L3_best_n":   binseg_res.get("best_n"),
        "L4_pelt":     pelt_breaks,
    }


# ══════════════════════════════════════════════════════════════════════════════
# Main
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
        _IT_MONTHS = {
            "gen": "Jan", "feb": "Feb", "mar": "Mar", "apr": "Apr",
            "mag": "May", "giu": "Jun", "lug": "Jul", "ago": "Aug",
            "set": "Sep", "ott": "Oct", "nov": "Nov", "dic": "Dec",
        }
        def _parse_it_date(s: str) -> pd.Timestamp:
            for it, en in _IT_MONTHS.items():
                s = s.replace(it, en)
            return pd.to_datetime(s, dayfirst=True, errors="coerce")
        df["date"] = df["data"].astype(str).apply(_parse_it_date)
    df["price_usd_ton"] = pd.to_numeric(df["chiusura"], errors="coerce")
    df = (df.dropna(subset=["date", "price_usd_ton"])
            .sort_values("date").set_index("date"))
    df = df[~df.index.duplicated(keep="first")]
    print(f"  B7H1: {len(df)} righe  "
          f"({df.index.min().date()} → {df.index.max().date()})")
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
        import numpy as np
        df["margin_benzina"] = np.nan
    return df


# ══════════════════════════════════════════════════════════════════════════════
# Grafico 0 – Serie storica completa con PELT globale
# ══════════════════════════════════════════════════════════════════════════════
def _plot_global_margins(daily: pd.DataFrame, out_dir: Path) -> None:
    """
    Grafico unico con:
      • margine benzina e gasolio sull'intera serie storica
      • break point PELT rilevati sull'intera serie (linee verdi)
      • linee verticali per i tre eventi geopolitici
    Salva: 00_margine_serie_storica_globale.png
    """
    fig, axes = plt.subplots(2, 1, figsize=(16, 8), sharex=True)
    fig.suptitle("Serie storica completa dei margini  –  Benzina e Gasolio\n"
                 "(con change point PELT globali e eventi geopolitici)",
                 fontsize=12, fontweight="bold")

    fuel_cfg = [
        ("margin_benzina", "Benzina", "#E63946"),
        ("margin_gasolio", "Gasolio", "#1D3557"),
    ]

    for ax, (col, label, color) in zip(axes, fuel_cfg):
        series = daily[col].dropna()
        if series.empty:
            ax.set_title(f"{label} – dati non disponibili", fontsize=9)
            continue

        # PELT sull'intera serie
        pelt_global = pelt_detect(series)

        ax.plot(series.index, series.values, color=color, lw=0.85, label=label)
        ax.axhline(series.mean(), color=color, lw=0.7, ls="--", alpha=0.5,
                   label=f"Media = {series.mean():.3f} €/L")

        # Break point PELT
        for i, d in enumerate(pelt_global):
            ax.axvline(d, color="green", lw=1.0, ls=":", alpha=0.75,
                       label="PELT break" if i == 0 else "")

        # Eventi geopolitici
        for ev_name, ev in EVENTS.items():
            ax.axvline(ev["shock"], color=ev["color"], lw=1.4, ls="--", alpha=0.85,
                       label=ev["label"].replace("\n", " "))

        ax.set_ylabel("Margine (€/L)", fontsize=9)
        ax.set_title(f"{label}  –  {len(pelt_global)} break point PELT rilevati",
                     fontsize=9, fontweight="bold")
        ax.legend(fontsize=7, loc="upper left", ncol=3)
        ax.grid(axis="y", alpha=0.2)
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=6))
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=30, ha="right", fontsize=8)

    fig.tight_layout()
    out = out_dir / "00_margine_serie_storica_globale.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  → Salvato: {out}")


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════
def main() -> None:
    # ── Carica dati pompa ─────────────────────────────────────────────────────
    daily = (pd.read_csv(DAILY_CSV, parse_dates=["date"])
               .sort_values("date").set_index("date"))

    # ── Carica futures e costruisce il margine ────────────────────────────────
    print("Carico futures e EUR/USD...")
    eurusd = load_eurusd(
        csv_path=EURUSD_CSV if EURUSD_CSV.exists() else None,
        start="2015-01-01", end="2026-12-31"
    )
    gasoil_eurl  = load_futures_eurl(GASOIL_CSV, GAS_OIL, eurusd)
    eurobob_eurl = load_futures_b7h1(EUROBOB_CSV, EUROBOB_HC, eurusd) \
                   if EUROBOB_CSV.exists() else None

    daily = build_margin(daily, gasoil_eurl, eurobob_eurl)

    print(f"\nDati margine: {daily.index.min().date()} → {daily.index.max().date()}")
    benz_avail = daily["margin_benzina"].dropna()
    if not benz_avail.empty:
        print(f"  margin_benzina: {benz_avail.index.min().date()} → {benz_avail.index.max().date()}")
    print(f"  margin_gasolio: {daily['margin_gasolio'].dropna().index.min().date()} → "
          f"{daily['margin_gasolio'].dropna().index.max().date()}")
    print(f"\nConfigurazione: HALF_WIN={HALF_WIN}g  SEARCH=±{SEARCH}g  STEP={STEP}g")
    print(f"Nota CLT: con HALF_WIN={HALF_WIN} e φ≈0.3 → n_eff≈{int(HALF_WIN*0.54)} "
          f"({'OK' if int(HALF_WIN*0.54)>=30 else 'approssimato, considera HALF_WIN≥60'})\n")

    # ── Grafico 0: serie storica completa + PELT globale ─────────────────────
    print("Grafico 0: margine completo con change point globali (PELT)...")
    _plot_global_margins(daily, OUT_DIR)

    all_results: dict = {}

    for ev_name, ev in EVENTS.items():
        shock = ev["shock"]

        available = daily[(daily.index >= ev["pre_start"]) &
                          (daily.index <= ev["post_end"])]
        if available.empty:
            print(f"⚠  {ev_name}: nessun dato disponibile, salto.")
            continue

        print(f"{'═'*70}")
        print(f"  EVENTO: {ev_name}  (shock={shock.date()})")
        print(f"  Finestra: {ev['pre_start'].date()} → {ev['post_end'].date()}")
        print(f"{'═'*70}")

        fig = plt.figure(figsize=(18, 14))
        fig.suptitle(
            f"Change-point detection sul MARGINE – {ev_name}\n"
            f"Shock: {ev['label'].replace(chr(10), ' ')}",
            fontsize=13, fontweight="bold"
        )

        gs = gridspec.GridSpec(4, 2, figure=fig, hspace=0.55, wspace=0.30)

        ev_results: dict = {}

        for col_idx, (fuel_key, (col_name, fuel_color)) in enumerate(FUELS.items()):
            if col_name not in daily.columns:
                print(f"  Colonna {col_name} non trovata, salto.")
                continue

            series = daily[col_name].dropna()

            # Benzina: salta evento se non ci sono dati nel range
            win_check = series[(series.index >= ev["pre_start"]) &
                               (series.index <= ev["post_end"])]
            if len(win_check) < 2 * HALF_WIN:
                print(f"  [{fuel_key}] dati insufficienti nel range "
                      f"(n={len(win_check)}), salto.")
                continue

            ax_price = fig.add_subplot(gs[0, col_idx])
            ax_cusum = fig.add_subplot(gs[1, col_idx])
            ax_tstat = fig.add_subplot(gs[2, col_idx])
            ax_bic   = fig.add_subplot(gs[3, col_idx])

            ax_price.set_title(
                f"{fuel_key.capitalize()} – {ev['label'].replace(chr(10),' ')}",
                fontsize=9, fontweight="bold", pad=4
            )

            res = plot_event_fuel(
                ev_name, ev, series,
                fuel_label=fuel_key.capitalize(),
                fuel_color=fuel_color,
                ax_price=ax_price,
                ax_cusum=ax_cusum,
                ax_tstat=ax_tstat,
                ax_bic=ax_bic,
            )
            ev_results[fuel_key] = res

            if res:
                print(f"\n  [{fuel_key.upper()}]")
                print(f"    L1 τ={res['L1_tau'].date()}  Δ={res['L1_delta']:+.4f} €/L"
                      f"  p_bh={res['L1_p_bh']:.3f}  [BH-FDR]"
                      f"  {'★ significativo' if res['L1_p_bh']<0.05 else '– non sig.'}")
                print(f"    L2 CUSUM peak = {res['L2_cusum'].date()}")
                print(f"    L3 BinSeg ({res['L3_best_n']} break ottimale): "
                      + (", ".join(str(d.date()) for d in res['L3_binseg']) or "nessuno"))
                print(f"    L4 PELT ({len(res['L4_pelt'])} break): "
                      + (", ".join(str(d.date()) for d in res['L4_pelt']) or "nessuno"))

        all_results[ev_name] = ev_results

        out = OUT_DIR / f"cp_margin_{ev_name.replace(' ','_').replace('/','')}.png"
        fig.savefig(out, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"\n  → Salvato: {out}\n")

    # ── Riepilogo finale ──────────────────────────────────────────────────────
    print(f"\n{'═'*70}")
    print("RIEPILOGO CHANGE POINT MARGINE – tutti gli eventi")
    print(f"{'═'*70}")
    print(f"{'Evento':<28} {'Carb.':<10} {'L1 τ':<13} {'Δ €/L':>8} "
          f"{'p_bh':>8} {'L4 PELT break'}")
    print("-"*80)
    for ev_name, fuels in all_results.items():
        for fuel, res in fuels.items():
            if not res:
                continue
            pelt_str = ", ".join(str(d.date()) for d in res["L4_pelt"]) or "—"
            print(f"{ev_name:<28} {fuel:<10} {str(res['L1_tau'].date()):<13} "
                  f"{res['L1_delta']:>+8.4f} {res['L1_p_bh']:>8.3f}  {pelt_str}")


if __name__ == "__main__":
    main()