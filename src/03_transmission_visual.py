#!/usr/bin/env python3
"""
03_transmission_visual.py
==========================
Catena di trasmissione del prezzo durante gli shock geopolitici:

  Brent crude  →  Futures wholesale (Eurobob / Gasoil)
               →  Prezzi retail (benzina / gasolio)
               →  Margini distributori
               →  Guadagni extra cumulati (M€)

Per ogni evento geopolitico genera una figura a 5 pannelli:
  ① Prezzi indicizzati a 100 allo shock (Brent + futures + retail)
  ② Margine benzina (€/L) con CI 90 % bootstrap
  ③ Margine gasolio (€/L) con CI 90 % bootstrap
  ④ Guadagni extra cumulati benzina + gasolio (M€)
  ⑤ Shift timeline: giorni di ritardo τ − shock per ogni livello della catena

Output: data/plots/transmission/
"""

from __future__ import annotations

import warnings
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
from scipy import stats

try:
    import yfinance as yf
    HAS_YF = True
except ImportError:
    HAS_YF = False
    warnings.warn("yfinance non trovato – Brent non disponibile. pip install yfinance")

import sys
sys.path.insert(0, str(Path(__file__).parent / "utils"))
from conversions import GAS_OIL, EUROBOB as EUROBOB_HC, load_eurusd, usd_ton_to_eur_liter

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR    = Path(__file__).parent
DAILY_CSV   = BASE_DIR / "data" / "processed" / "daily_fuel_prices_all.csv"
GASOIL_CSV  = BASE_DIR / "data" / "Futures" / "London Gas Oil Futures Historical Data.csv"
EUROBOB_CSV = BASE_DIR / "data" / "Futures" / "Eurobob_B7H1_date.csv"
EURUSD_CSV  = BASE_DIR / "data" / "raw" / "eurusd.csv"
OUT_DIR     = BASE_DIR / "data" / "plots" / "transmission"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Config ────────────────────────────────────────────────────────────────────
HALF_WIN  = 40    # finestra sliding t-test (per τ)
SEARCH    = 60    # ricerca τ nell'intervallo ±SEARCH giorni dallo shock
PRE_IDX   = 30    # giorni pre-shock per baseline indicizzazione (→ 100)
PRE_WIN   = 60    # giorni per fit baseline controfattuale
N_BOOT       = 400    # repliche bootstrap CI
CI_LEVEL     = (5, 95)
R2_MIN_SLOPE   = 0.15   # sotto questa soglia → baseline piatta (slope=0)
MAX_TREND_MOVE = 0.30   # trend > 40% della media pre nel pre-window → piatto
DAILY_VOL = {"benzina": 12_000_000, "gasolio": 25_000_000}  # L/giorno

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

# Colori e stili per la chain
CHAIN_STYLE: dict[str, dict] = {
    "brent":          dict(color="#1C2833", lw=2.0,  ls="-",  label="Brent crude"),
    "eurobob":        dict(color="#CA6F1E", lw=1.5,  ls="-",  label="Eurobob futures"),
    "gasoil_fut":     dict(color="#1F618D", lw=1.5,  ls="-",  label="Gasoil futures"),
    "benzina_retail": dict(color="#E74C3C", lw=1.2,  ls="--", label="Benzina retail (netto)"),
    "gasolio_retail": dict(color="#154360", lw=1.2,  ls="--", label="Gasolio retail (netto)"),
}
MARGIN_STYLE = {
    "benzina": dict(color="#E67E22", lw=1.4, label="Margine benzina"),
    "gasolio": dict(color="#2471A3", lw=1.4, label="Margine gasolio"),
}
GAIN_STYLE = {
    "benzina": dict(color="#E67E22", lw=1.4),
    "gasolio": dict(color="#2471A3", lw=1.4),
}

# Etichette leggibili per la shift timeline
SHIFT_LABELS = {
    "brent":          "Brent crude",
    "eurobob":        "Eurobob fut.",
    "gasoil_fut":     "Gasoil fut.",
    "benzina_retail": "Benzina retail",
    "gasolio_retail": "Gasolio retail",
    "benz_margin":    "Margine benzina",
    "gas_margin":     "Margine gasolio",
}


# ══════════════════════════════════════════════════════════════════════════════
# Helper: n_eff e Welch t-test corretto per AR(1)
# ══════════════════════════════════════════════════════════════════════════════

def _phi_ar1(s: pd.Series) -> float:
    if len(s) < 4:
        return 0.0
    v = s.values - s.mean()
    if v.std() < 1e-12:
        return 0.0
    return float(np.clip(np.corrcoef(v[:-1], v[1:])[0, 1], -0.99, 0.99))


def _n_eff(n: int, phi: float) -> float:
    return max(2.0, n * (1.0 - phi) / (1.0 + phi))


def _welch_neff(pre: pd.Series, post: pd.Series) -> tuple[float, float]:
    n1 = _n_eff(len(pre),  _phi_ar1(pre))
    n2 = _n_eff(len(post), _phi_ar1(post))
    m1, m2 = float(pre.mean()), float(post.mean())
    v1 = float(pre.var(ddof=1))  / n1
    v2 = float(post.var(ddof=1)) / n2
    se = np.sqrt(v1 + v2)
    if se < 1e-12:
        return 0.0, 1.0
    t = (m2 - m1) / se
    df = max(1.0, (v1 + v2)**2 / (v1**2 / (n1-1) + v2**2 / (n2-1)))
    return float(t), float(2.0 * stats.t.sf(abs(t), df=df))


def find_tau(series: pd.Series, shock: pd.Timestamp,
             half_win: int = HALF_WIN, search: int = SEARCH) -> pd.Timestamp | None:
    """Trova τ con sliding t-test corretto per n_eff. Restituisce None se non significativo."""
    idx = series.index
    best_t, best_tau = 0.0, None
    for tau in pd.date_range(shock - pd.Timedelta(days=search),
                             shock + pd.Timedelta(days=search), freq="1D"):
        pre  = series[(idx >= tau - pd.Timedelta(days=half_win)) & (idx < tau)].dropna()
        post = series[(idx >= tau) & (idx < tau + pd.Timedelta(days=half_win))].dropna()
        if len(pre) < 5 or len(post) < 5:
            continue
        t, _ = _welch_neff(pre, post)
        if abs(t) > abs(best_t):
            best_t, best_tau = t, tau
    return best_tau


# ══════════════════════════════════════════════════════════════════════════════
# Helper: bootstrap CI sulla baseline controfattuale
# ══════════════════════════════════════════════════════════════════════════════

def fit_baseline(series: pd.Series, tau: pd.Timestamp, shock: pd.Timestamp,
                 pre_win: int = PRE_WIN) -> tuple[float, float, float, pd.Timestamp]:
    """Anchor corretto + R² cap (identica a 02d)."""
    pre_end = shock if tau > shock else tau
    pre = series[
        (series.index >= pre_end - pd.Timedelta(days=pre_win)) &
        (series.index < pre_end)
    ].dropna()
    if len(pre) < 10:
        return 0.0, float(pre.mean()) if len(pre) else 0.0, 0.0, pre_end
    x = np.array([(d - pre_end).days for d in pre.index], dtype=float)
    slope, intercept, r, *_ = stats.linregress(x, pre.values)
    r2 = float(r ** 2)
    pre_mean = float(pre.mean())
    pre_trend_move = abs(slope) * (len(pre) - 1)
    if r2 < R2_MIN_SLOPE or (pre_mean != 0 and pre_trend_move > MAX_TREND_MOVE * abs(pre_mean)):
        slope, intercept, r2 = 0.0, pre_mean, 0.0
    return float(slope), float(intercept), r2, pre_end


def project_baseline(tau: pd.Timestamp, pre_end: pd.Timestamp,
                     post_index: pd.DatetimeIndex,
                     slope: float, intercept: float) -> pd.Series:
    """valore(d) = intercept + slope * (d − pre_end).days"""
    x = np.array([(d - pre_end).days for d in post_index], dtype=float)
    return pd.Series(slope * x + intercept, index=post_index)


def bootstrap_ci(series: pd.Series, tau: pd.Timestamp, shock: pd.Timestamp,
                 post_index: pd.DatetimeIndex,
                 pre_win: int = PRE_WIN, n_boot: int = N_BOOT,
                 ci: tuple[int, int] = CI_LEVEL) -> tuple[pd.Series, pd.Series]:
    """Circular block bootstrap CI con stesso anchor di fit_baseline."""
    pre_end = shock if tau > shock else tau
    pre = series[
        (series.index >= pre_end - pd.Timedelta(days=pre_win)) &
        (series.index < pre_end)
    ].dropna()
    if len(pre) < 10:
        flat = pd.Series(np.full(len(post_index), float(pre.mean()) if len(pre) else np.nan),
                         index=post_index)
        return flat, flat
    vals, n = pre.values, len(pre.values)
    n_post  = len(post_index)
    x_fit   = np.array([(d - pre_end).days for d in pre.index], dtype=float)
    x_post  = np.array([(d - pre_end).days for d in post_index], dtype=float)
    phi = abs(_phi_ar1(pre))
    bs  = max(5, int(round(-1.0 / np.log(phi)))) if phi > 0.01 else 5
    bs  = min(bs, n // 3)
    rng = np.random.default_rng(42)
    boot = np.empty((n_boot, n_post))
    for b in range(n_boot):
        bv: list[float] = []
        while len(bv) < n:
            s = rng.integers(0, n)
            bv.extend([vals[(s + k) % n] for k in range(bs)])
        sample = np.array(bv[:n])
        sl, ic, r, *_ = stats.linregress(x_fit, sample)
        trend_move = abs(sl) * (n - 1)
        smean = float(sample.mean())
        if r**2 < R2_MIN_SLOPE or (smean != 0 and trend_move > MAX_TREND_MOVE * abs(smean)):
            sl, ic = 0.0, smean
        boot[b] = sl * x_post + ic
    return (pd.Series(np.percentile(boot, ci[0], axis=0), index=post_index),
            pd.Series(np.percentile(boot, ci[1], axis=0), index=post_index))


def index_100(series: pd.Series, shock: pd.Timestamp, pre_days: int = PRE_IDX) -> pd.Series:
    """Indicizza la serie a 100 = media dei `pre_days` giorni pre-shock."""
    baseline = series[(series.index >= shock - pd.Timedelta(days=pre_days)) &
                      (series.index < shock)].mean()
    if pd.isna(baseline) or baseline == 0:
        return series * np.nan
    return series / baseline * 100.0


# ══════════════════════════════════════════════════════════════════════════════
# Caricamento dati
# ══════════════════════════════════════════════════════════════════════════════

def load_brent_eurl(eurusd: pd.Series) -> pd.Series | None:
    """Scarica Brent da yfinance e converte in EUR/L."""
    if not HAS_YF:
        return None
    try:
        raw = yf.download("BZ=F", start="2015-01-01", end="2026-12-31",
                          progress=False, auto_adjust=True)
        # yfinance può restituire MultiIndex
        if isinstance(raw.columns, pd.MultiIndex):
            raw.columns = raw.columns.get_level_values(0)
        price_usd_bbl = raw["Close"].dropna()
        price_usd_bbl.index = pd.to_datetime(price_usd_bbl.index).tz_localize(None)
        # 1 barile = 158.987 L
        eurusd_aligned = eurusd.reindex(price_usd_bbl.index, method="ffill")
        brent_eur_l = price_usd_bbl / 158.987 / eurusd_aligned
        brent_eur_l.name = "brent_eurl"
        print(f"  Brent scaricato: {brent_eur_l.index.min().date()} → {brent_eur_l.index.max().date()}")
        return brent_eur_l.dropna()
    except Exception as e:
        warnings.warn(f"Brent non disponibile: {e}")
        return None


def load_futures_eurl(path: Path, hc, eurusd: pd.Series) -> pd.Series:
    df = pd.read_csv(path, encoding="utf-8-sig", dtype=str)
    df["date"] = pd.to_datetime(df["Date"], format="%m/%d/%Y", errors="coerce")
    df["price"] = (df["Price"].str.replace(",", "", regex=False)
                   .pipe(pd.to_numeric, errors="coerce"))
    df = df.dropna(subset=["date", "price"]).sort_values("date").set_index("date")
    return usd_ton_to_eur_liter(df["price"], eurusd, hc)


def load_futures_b7h1(path: Path, hc, eurusd: pd.Series) -> pd.Series:
    df = pd.read_csv(path, encoding="utf-8-sig", dtype=str)
    _IT = {"gen":"Jan","feb":"Feb","mar":"Mar","apr":"Apr","mag":"May","giu":"Jun",
           "lug":"Jul","ago":"Aug","set":"Sep","ott":"Oct","nov":"Nov","dic":"Dec"}
    if "timestamp" in df.columns:
        ts = pd.to_numeric(df["timestamp"], errors="coerce")
        df["date"] = pd.to_datetime(ts, unit="s", utc=True).dt.tz_localize(None).dt.normalize()
    else:
        def _p(s: str) -> pd.Timestamp:
            for it, en in _IT.items():
                s = s.replace(it, en)
            return pd.to_datetime(s, dayfirst=True, errors="coerce")
        df["date"] = df["data"].astype(str).apply(_p)
    df["price"] = pd.to_numeric(df["chiusura"], errors="coerce")
    df = df.dropna(subset=["date", "price"]).sort_values("date").set_index("date")
    df = df[~df.index.duplicated(keep="first")]
    return usd_ton_to_eur_liter(df["price"], eurusd, hc)


def load_all(eurusd: pd.Series) -> dict[str, pd.Series]:
    daily = (pd.read_csv(DAILY_CSV, parse_dates=["date"])
               .sort_values("date").set_index("date"))

    gasoil_eurl  = load_futures_eurl(GASOIL_CSV, GAS_OIL, eurusd)
    eurobob_eurl = load_futures_b7h1(EUROBOB_CSV, EUROBOB_HC, eurusd) \
                   if EUROBOB_CSV.exists() else None

    brent = load_brent_eurl(eurusd)

    m_gas  = (daily["gasolio_net"] - gasoil_eurl.reindex(daily.index, method="ffill")).dropna()
    m_benz = (daily["benzina_net"] - eurobob_eurl.reindex(daily.index, method="ffill")).dropna() \
             if eurobob_eurl is not None else pd.Series(dtype=float)

    return {
        "brent":          brent,
        "eurobob":        eurobob_eurl,
        "gasoil_fut":     gasoil_eurl,
        "benzina_retail": daily["benzina_net"].dropna(),
        "gasolio_retail": daily["gasolio_net"].dropna(),
        "benz_margin":    m_benz,
        "gas_margin":     m_gas,
    }


# ══════════════════════════════════════════════════════════════════════════════
# Plot per evento
# ══════════════════════════════════════════════════════════════════════════════

def _setup_date_axis(ax: plt.Axes, interval_months: int = 1) -> None:
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=interval_months))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=30, ha="right", fontsize=7)


def plot_event(ev_name: str, ev: dict, data: dict[str, pd.Series],
               out_dir: Path) -> None:
    shock     = ev["shock"]
    pre_start   = ev["pre_start"]
    post_end  = ev["post_end"]
    ev_color  = ev["color"]

    # ── Trova τ per ogni livello della catena ────────────────────────────────
    tau: dict[str, pd.Timestamp | None] = {}
    print(f"\n  Calcolo τ per {ev_name}:")
    for key, series in data.items():
        if series is None or series.empty:
            tau[key] = None
            continue
        win = series[(series.index >= pre_start) & (series.index <= post_end)].dropna()
        if len(win) < 2 * HALF_WIN:
            tau[key] = None
            continue
        t = find_tau(win, shock)
        tau[key] = t
        lag = int((t - shock).days) if t is not None else None
        print(f"    {key:<18}  τ = {t.date() if t else 'N/A':12}  lag = {lag:+d}d" if lag is not None
              else f"    {key:<18}  τ = N/A")

    # ── Finestra temporale del plot ──────────────────────────────────────────
    t_start = pre_start
    t_end   = post_end

    # ── Layout figura ─────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(20, 16), facecolor="white")
    fig.suptitle(
        f"Catena di trasmissione  ·  {ev['label']}",
        fontsize=14, fontweight="bold", y=0.98
    )

    gs = gridspec.GridSpec(
        4, 2,
        figure=fig,
        height_ratios=[2.2, 1.4, 1.4, 1.0],
        hspace=0.52, wspace=0.28,
    )

    ax_chain = fig.add_subplot(gs[0, :])          # ① prezzi indicizzati
    ax_bm    = fig.add_subplot(gs[1, 0])           # ② margine benzina
    ax_gm    = fig.add_subplot(gs[1, 1])           # ③ margine gasolio
    ax_gain  = fig.add_subplot(gs[2, :])           # ④ guadagni cumulati
    ax_shift = fig.add_subplot(gs[3, :])           # ⑤ shift timeline

    # ════════════════════════════════════════════════════════════════════════
    # ① PREZZI INDICIZZATI A 100 (Brent + futures + retail)
    # ════════════════════════════════════════════════════════════════════════
    ax_chain.set_title("① Prezzi indicizzati a 100 allo shock  (Brent · Futures · Retail netto)",
                       fontsize=9, fontweight="bold", loc="left")
    ax_chain.axhline(100, color="grey", lw=0.6, ls=":", alpha=0.5)
    ax_chain.axvline(shock, color=ev_color, lw=2.0, ls="--", alpha=0.85, label=f"Shock  {shock.date()}")

    for key in ["brent", "eurobob", "gasoil_fut", "benzina_retail", "gasolio_retail"]:
        s = data.get(key)
        if s is None or s.empty:
            continue
        s_win = s[(s.index >= t_start) & (s.index <= t_end)]
        s_idx = index_100(s_win, shock)
        sty   = CHAIN_STYLE[key]
        ax_chain.plot(s_idx.index, s_idx.values,
                      color=sty["color"], lw=sty["lw"], ls=sty["ls"],
                      label=sty["label"], alpha=0.9)
        # segna τ
        if tau.get(key) is not None:
            ax_chain.axvline(tau[key], color=sty["color"], lw=1.0, ls=":", alpha=0.6)

    ax_chain.set_ylabel("Indice (100 = media 30gg pre-shock)", fontsize=8)
    ax_chain.set_xlim(t_start, t_end)
    ax_chain.legend(fontsize=7, loc="upper left", ncol=3, framealpha=0.85)
    ax_chain.grid(axis="y", alpha=0.20)
    _setup_date_axis(ax_chain, interval_months=2)

    # ════════════════════════════════════════════════════════════════════════
    # ② MARGINE BENZINA  ③ MARGINE GASOLIO
    # ════════════════════════════════════════════════════════════════════════
    for ax, key, label in [(ax_bm, "benz_margin", "② Margine benzina (€/L)"),
                           (ax_gm, "gas_margin",   "③ Margine gasolio (€/L)")]:
        fuel = "benzina" if "benz" in key else "gasolio"
        s = data.get(key)
        ax.set_title(label, fontsize=9, fontweight="bold", loc="left")

        if s is None or s.empty:
            ax.text(0.5, 0.5, "Dati non disponibili",
                    ha="center", va="center", transform=ax.transAxes, fontsize=9)
            continue

        s_win = s[(s.index >= t_start) & (s.index <= t_end)].dropna()
        sty   = MARGIN_STYLE[fuel]
        ax.plot(s_win.index, s_win.values, **sty, alpha=0.9)

        # Baseline + CI bootstrap se τ disponibile
        if tau.get(key) is not None:
            τ = tau[key]
            sl, ic, r2, pre_end = fit_baseline(s, τ, shock)
            post_idx_range = s_win[s_win.index >= τ].index
            if len(post_idx_range) > 1:
                baseline = project_baseline(τ, pre_end, post_idx_range, sl, ic)
                ci_lo, ci_hi = bootstrap_ci(s, τ, shock, post_idx_range)
                ax.plot(baseline.index, baseline.values,
                        color="grey", lw=1.1, ls="--", alpha=0.7,
                        label=f"Baseline (R²={r2:.2f})")
                ax.fill_between(ci_lo.index, ci_lo.values, ci_hi.values,
                                color="grey", alpha=0.15,
                                label=f"CI 90% bootstrap (N={N_BOOT})")
                # fill guadagno/perdita
                actual = s_win[s_win.index >= τ]
                extra  = actual - baseline
                ax.fill_between(actual.index, actual.values, baseline.values,
                                where=(extra >= 0), alpha=0.22, color="green")
                ax.fill_between(actual.index, actual.values, baseline.values,
                                where=(extra < 0),  alpha=0.22, color="red")

            ax.axvline(τ, color=sty["color"], lw=1.2, ls=":",
                       label=f"τ={τ.date()}  lag={int((τ-shock).days):+d}d")

        ax.axvline(shock, color=ev_color, lw=1.5, ls="--", alpha=0.7)
        ax.axhline(0, color="grey", lw=0.5, ls=":")
        ax.set_ylabel("€/L", fontsize=8)
        ax.set_xlim(t_start, t_end)
        ax.legend(fontsize=6, loc="upper left", ncol=2, framealpha=0.85)
        ax.grid(axis="y", alpha=0.18)
        _setup_date_axis(ax, interval_months=2)

    # ════════════════════════════════════════════════════════════════════════
    # ④ GUADAGNI EXTRA CUMULATI (M€)
    # ════════════════════════════════════════════════════════════════════════
    ax_gain.set_title("④ Guadagni extra cumulati  (stima controfattuale,  M€)",
                      fontsize=9, fontweight="bold", loc="left")
    ax_gain.axhline(0, color="grey", lw=0.7, ls="--")
    ax_gain.axvline(shock, color=ev_color, lw=1.5, ls="--", alpha=0.7)

    gain_legend: list = [Line2D([0], [0], color=ev_color, ls="--", lw=1.5, label=f"Shock {shock.date()}")]

    for key, fuel in [("benz_margin", "benzina"), ("gas_margin", "gasolio")]:
        s = data.get(key)
        if s is None or s.empty or tau.get(key) is None:
            continue
        τ = tau[key]
        sl, ic, r2, pre_end = fit_baseline(s, τ, shock)
        post = s[(s.index >= τ) & (s.index <= t_end)].dropna()
        if len(post) < 2:
            continue
        baseline = project_baseline(τ, pre_end, post.index, sl, ic)
        extra   = post - baseline
        cum_meur = extra.cumsum() * DAILY_VOL[fuel] / 1e6

        sty = GAIN_STYLE[fuel]
        final = float(cum_meur.iloc[-1])
        ax_gain.plot(cum_meur.index, cum_meur.values, **sty,
                     label=f"{fuel.capitalize()}  →  {final:+.0f} M€")
        ax_gain.fill_between(cum_meur.index, cum_meur.values, 0,
                             where=(cum_meur >= 0), alpha=0.18, color=sty["color"])
        ax_gain.fill_between(cum_meur.index, cum_meur.values, 0,
                             where=(cum_meur < 0),  alpha=0.18, color="red")
        ax_gain.axvline(τ, color=sty["color"], lw=0.9, ls=":", alpha=0.7)

    ax_gain.set_ylabel("M€  (cumulato)", fontsize=8)
    ax_gain.set_xlim(t_start, t_end)
    ax_gain.legend(fontsize=7, loc="upper left", ncol=3, framealpha=0.85)
    ax_gain.grid(axis="y", alpha=0.18)
    _setup_date_axis(ax_gain, interval_months=2)

    # ════════════════════════════════════════════════════════════════════════
    # ⑤ SHIFT TIMELINE: giorni di ritardo τ − shock per livello della catena
    # ════════════════════════════════════════════════════════════════════════
    ax_shift.set_title(
        "⑤ Shift temporale  ·  giorni tra lo shock e il cambio strutturale τ  "
        "(←  anticipa  |  ritarda  →)",
        fontsize=9, fontweight="bold", loc="left"
    )

    labels_order = ["brent", "eurobob", "gasoil_fut",
                    "benzina_retail", "gasolio_retail",
                    "benz_margin", "gas_margin"]
    y_pos = {k: i for i, k in enumerate(labels_order)}
    n_levels = len(labels_order)

    ax_shift.set_xlim(-SEARCH - 5, SEARCH + 5)
    ax_shift.set_ylim(-0.6, n_levels - 0.4)
    ax_shift.axvline(0, color=ev_color, lw=2.0, ls="--", alpha=0.9, label=f"Shock")
    ax_shift.set_xlabel("Giorni rispetto allo shock (τ − shock)", fontsize=8)

    for key in labels_order:
        y = y_pos[key]
        t_ = tau.get(key)
        sty = {**CHAIN_STYLE.get(key, {}), **MARGIN_STYLE.get(
            "benzina" if "benz" in key else "gasolio", {})}
        dot_color = sty.get("color", "#555555")

        if t_ is None:
            ax_shift.text(0, y, "  N/D", va="center", ha="left", fontsize=7,
                          color="#aaaaaa", fontstyle="italic")
        else:
            lag = int((t_ - shock).days)
            line_color = "#27AE60" if lag < 0 else "#E74C3C"
            ax_shift.hlines(y, 0, lag, colors=line_color, lw=2.5, alpha=0.7)
            ax_shift.scatter(lag, y, s=90, color=dot_color, zorder=5,
                             edgecolors="white", linewidths=0.8)
            ax_shift.text(lag + (1.5 if lag >= 0 else -1.5), y,
                          f"{lag:+d}d", va="center",
                          ha="left" if lag >= 0 else "right",
                          fontsize=7.5, fontweight="bold", color=dot_color)

    ax_shift.set_yticks(list(y_pos.values()))
    ax_shift.set_yticklabels([SHIFT_LABELS[k] for k in labels_order], fontsize=8)
    ax_shift.grid(axis="x", alpha=0.20)
    ax_shift.spines[["top", "right"]].set_visible(False)

    # ── Salva ────────────────────────────────────────────────────────────────
    safe = ev_name.replace(" ", "_").replace("/", "").replace("(", "").replace(")", "")
    out  = out_dir / f"transmission_{safe}.png"
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  → Salvato: {out}")


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    print("Carico EUR/USD e dati...")
    eurusd = load_eurusd(
        csv_path=EURUSD_CSV if EURUSD_CSV.exists() else None,
        start="2015-01-01", end="2026-12-31",
    )
    data = load_all(eurusd)

    for k, s in data.items():
        if s is not None and not s.empty:
            print(f"  {k:<18}: {s.index.min().date()} → {s.index.max().date()}")

    for ev_name, ev in EVENTS.items():
        print(f"\n{'═'*65}")
        print(f"  EVENTO: {ev_name}")
        print(f"{'═'*65}")
        plot_event(ev_name, ev, data, OUT_DIR)

    print(f"\nDone. Figure salvate in: {OUT_DIR}")


if __name__ == "__main__":
    main()
