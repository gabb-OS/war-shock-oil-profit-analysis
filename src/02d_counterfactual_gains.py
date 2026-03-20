#!/usr/bin/env python3
"""
02d_counterfactual_gains.py
===========================
Analisi controfattuale dei margini benzina / gasolio post-shock.

Per ogni evento geopolitico e carburante:

  1. Identifica il break point (L1 τ dal sliding t-test di 02a).
  2. Stima una baseline "pre-break" con regressione lineare sui
     PRE_WIN giorni precedenti τ, poi la proietta in avanti.
  3. Calcola il guadagno extra giornaliero:
         gain(t) = margine_effettivo(t) − baseline_proiettata(t)
  4. Somma gain su tutta la finestra post-break disponibile →
         guadagno_extra_totale  (€/L × giorni = "€/L-giorno")
  5. Stima volumetrica (opzionale, se disponibili vendite nazionali):
         guadagno_extra_€ = guadagno_extra_totale × volume_giornaliero_medio

Output:
  data/plots/change_point/margin/
    04_counterfactual_{evento}_{carburante}.png
    04_counterfactual_summary.csv
"""

from __future__ import annotations

import warnings
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
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

# ── Configurazione ─────────────────────────────────────────────────────────────
BASE_DIR     = Path(__file__).parent
DAILY_CSV    = BASE_DIR / "data" / "processed" / "daily_fuel_prices_all.csv"
GASOIL_CSV   = BASE_DIR / "data" / "Futures" / "London Gas Oil Futures Historical Data.csv"
EUROBOB_CSV  = BASE_DIR / "data" / "Futures" / "Eurobob_B7H1_date.csv"
EURUSD_CSV   = BASE_DIR / "data" / "raw" / "eurusd.csv"
OUT_DIR      = BASE_DIR / "data" / "plots" / "change_point" / "margin"
OUT_DIR.mkdir(parents=True, exist_ok=True)

HALF_WIN   = 40    # finestra sliding t-test
SEARCH     = 40    # ricerca τ ±SEARCH giorni dallo shock
STEP       = 1
PRE_WIN    = 60    # giorni pre-τ usati per fit baseline

# Bootstrap CI sulla baseline controfattuale
N_BOOT       = 500    # repliche bootstrap
CI_LEVEL     = (5, 95)   # percentili (→ CI al 90%)
R2_MIN_SLOPE      = 0.15   # sotto questa soglia la slope è inaffidabile → baseline piatta
MAX_TREND_MOVE    = 0.30   # se il trend sposta la baseline > 40% della media pre nel solo
                            # pre-window → il trend è volatilità strutturale, non estrapolabile

# Consumi italiani stimati (litri/giorno, fonte MISE/MIMIT media 2022-2025)
# Usiamo stime conservative; non critiche per il ratio relativo.
DAILY_CONSUMPTION_L = {
    "benzina": 12_000_000,   # ~12 ML/giorno
    "gasolio": 25_000_000,   # ~25 ML/giorno (autotrazione)
}

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
# Utilità
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
        def _parse(s):
            for it, en in _IT_MONTHS.items():
                s = s.replace(it, en)
            return pd.to_datetime(s, dayfirst=True, errors="coerce")
        df["date"] = df["data"].astype(str).apply(_parse)
    df["price_usd_ton"] = pd.to_numeric(df["chiusura"], errors="coerce")
    df = df.dropna(subset=["date", "price_usd_ton"]).sort_values("date").set_index("date")
    df = df[~df.index.duplicated(keep="first")]
    return usd_ton_to_eur_liter(df["price_usd_ton"], eurusd, hc)


def build_margin(daily, gasoil_eurl, eurobob_eurl):
    df = daily[["benzina_net", "gasolio_net"]].copy()
    ws_gas = gasoil_eurl.reindex(df.index, method="ffill")
    df["margin_gasolio"] = df["gasolio_net"] - ws_gas
    if eurobob_eurl is not None:
        ws_benz = eurobob_eurl.reindex(df.index, method="ffill")
        df["margin_benzina"] = df["benzina_net"] - ws_benz
    else:
        df["margin_benzina"] = np.nan
    return df


def sliding_ttest(series, shock, half_win=HALF_WIN, search=SEARCH, step=STEP):
    idx = series.index
    rows = []
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
        t, p = _welch_neff(pre, post)   # corretto per autocorrelazione AR(1)
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
# Helpers: n_eff e Welch t-test corretto per autocorrelazione AR(1)
# ══════════════════════════════════════════════════════════════════════════════

def _phi_ar1(series: pd.Series) -> float:
    """Stima il coefficiente AR(1) via correlazione lag-1."""
    if len(series) < 4:
        return 0.0
    s = series.values - series.mean()
    if s.std() < 1e-12:
        return 0.0
    return float(np.clip(np.corrcoef(s[:-1], s[1:])[0, 1], -0.99, 0.99))


def _n_eff(n: int, phi: float) -> float:
    """Dimensione campionaria effettiva per AR(1): n*(1-φ)/(1+φ)."""
    return max(2.0, n * (1.0 - phi) / (1.0 + phi))


def _welch_neff(pre: pd.Series, post: pd.Series) -> tuple[float, float]:
    """
    Welch t-test corretto per autocorrelazione AR(1).
    Stima φ su pre e post separatamente, corregge n nei gradi di libertà
    di Welch-Satterthwaite → p-value meno ottimistici con serie autocorrelate.
    """
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
# Baseline controfattuale
# ══════════════════════════════════════════════════════════════════════════════

def fit_baseline(series: pd.Series, tau: pd.Timestamp, shock: pd.Timestamp,
                 pre_win: int = PRE_WIN) -> tuple[float, float, float, pd.Timestamp]:
    """
    Stima baseline OLS con anchor corretto.

    Anchor della finestra pre:
      τ ≤ shock  →  [τ − pre_win,   τ)      finestra pre-break pura
      τ > shock  →  [shock − pre_win, shock) evita il periodo già contaminato
                                              tra shock e τ (e.g. crisi già in atto)

    R² cap: se R² < R2_MIN_SLOPE la slope è rumore → baseline piatta = media.
    Questo evita proiezioni lineari che scappano a 0 o negativo su finestre lunghe.

    Restituisce (slope, intercept, r2, pre_end) dove pre_end è l'ancora usata.
    """
    pre_end = shock if tau > shock else tau
    pre = series[
        (series.index >= pre_end - pd.Timedelta(days=pre_win)) &
        (series.index < pre_end)
    ].dropna()
    if len(pre) < 10:
        return 0.0, float(pre.mean()) if len(pre) else 0.0, 0.0, pre_end
    # x in giorni rispetto a pre_end (valori negativi → "giorni prima dello shock/τ")
    x = np.array([(d - pre_end).days for d in pre.index], dtype=float)
    slope, intercept, r, *_ = stats.linregress(x, pre.values)
    r2 = float(r ** 2)
    pre_mean = float(pre.mean())
    pre_trend_move = abs(slope) * (len(pre) - 1)   # spostamento totale nel pre-window
    if r2 < R2_MIN_SLOPE or (pre_mean != 0 and pre_trend_move > MAX_TREND_MOVE * abs(pre_mean)):
        # slope inaffidabile o trend troppo aggressivo per essere extrapolato:
        # usa media piatta → baseline difendibile anche su finestre post lunghe
        slope, intercept, r2 = 0.0, pre_mean, 0.0
    return float(slope), float(intercept), r2, pre_end


def project_baseline(tau: pd.Timestamp, pre_end: pd.Timestamp,
                     post_index: pd.DatetimeIndex,
                     slope: float, intercept: float) -> pd.Series:
    """
    Proietta la baseline in avanti su post_index.

    Coordinate unificate: valore(d) = intercept + slope * (d − pre_end).days
    Funziona sia quando pre_end = τ (τ ≤ shock) sia quando pre_end = shock (τ > shock):
    nel secondo caso la retta è già estrapolata dal gap [shock, τ] automaticamente.
    """
    x = np.array([(d - pre_end).days for d in post_index], dtype=float)
    return pd.Series(slope * x + intercept, index=post_index)


def block_bootstrap_baseline(
    series: pd.Series,
    tau: pd.Timestamp,
    shock: pd.Timestamp,
    post_index: pd.DatetimeIndex,
    pre_win: int = PRE_WIN,
    n_boot: int  = N_BOOT,
    ci: tuple[int, int] = CI_LEVEL,
) -> tuple[pd.Series, pd.Series]:
    """
    Circular block bootstrap CI sulla baseline (stessa logica di fit_baseline).
    Usa pre_end = shock se τ > shock, altrimenti τ.
    Applica R2_MIN_SLOPE: se r² < soglia → slope=0 per ogni replica.
    """
    pre_end = shock if tau > shock else tau
    pre = series[
        (series.index >= pre_end - pd.Timedelta(days=pre_win)) &
        (series.index < pre_end)
    ].dropna()

    if len(pre) < 10:
        flat = pd.Series(np.full(len(post_index), float(pre.mean()) if len(pre) else np.nan),
                         index=post_index)
        return flat, flat

    vals   = pre.values
    n      = len(vals)
    n_post = len(post_index)
    x_fit  = np.array([(d - pre_end).days for d in pre.index], dtype=float)
    x_post = np.array([(d - pre_end).days for d in post_index], dtype=float)

    phi = abs(_phi_ar1(pre))
    block_size = max(5, int(round(-1.0 / np.log(phi)))) if phi > 0.01 else 5
    block_size = min(block_size, n // 3)

    rng = np.random.default_rng(42)
    boot_projections = np.empty((n_boot, n_post))

    for b in range(n_boot):
        boot_vals: list[float] = []
        while len(boot_vals) < n:
            start = rng.integers(0, n)
            boot_vals.extend([vals[(start + k) % n] for k in range(block_size)])
        sample = np.array(boot_vals[:n])

        sl, ic, r, *_ = stats.linregress(x_fit, sample)
        trend_move = abs(sl) * (n - 1)
        sample_mean = float(sample.mean())
        if r**2 < R2_MIN_SLOPE or (sample_mean != 0 and trend_move > MAX_TREND_MOVE * abs(sample_mean)):
            sl, ic = 0.0, sample_mean
        boot_projections[b] = sl * x_post + ic

    ci_low  = pd.Series(np.percentile(boot_projections, ci[0], axis=0), index=post_index)
    ci_high = pd.Series(np.percentile(boot_projections, ci[1], axis=0), index=post_index)
    return ci_low, ci_high


# ══════════════════════════════════════════════════════════════════════════════
# Plot controfattuale singolo evento+carburante
# ══════════════════════════════════════════════════════════════════════════════

def plot_counterfactual(
    ev_name: str, ev: dict,
    series: pd.Series,
    fuel_label: str, fuel_color: str,
    tau: pd.Timestamp, delta: float, p_bonf: float,
    ax_main: plt.Axes,
    ax_gain: plt.Axes,
    daily_consumption_l: float,
):
    """
    ax_main: margine effettivo vs controfattuale (baseline proiettata)
    ax_gain: guadagno extra cumulato (€/L-giorno e €M totali)
    """
    shock = ev["shock"]
    color = fuel_color

    # ── fit baseline (anchor corretto + R² cap) ──────────────────────────────
    slope, intercept, r2, pre_end = fit_baseline(series, tau, shock)

    # ── finestra post-τ disponibile ──────────────────────────────────────────
    post = series[series.index >= tau].dropna()
    post = post[post.index <= ev["post_end"]]

    baseline = project_baseline(tau, pre_end, post.index, slope, intercept)

    # ── Bootstrap CI al 90% sulla baseline ──────────────────────────────────
    ci_low, ci_high = block_bootstrap_baseline(series, tau, shock, post.index)

    # extra margin per giorno (rispetto alla baseline puntuale)
    extra = post - baseline

    # ── ax_main: serie + baseline + CI + evento ──────────────────────────────
    # mostra finestra completa (pre_start → post_end)
    win_full = series[(series.index >= ev["pre_start"]) &
                      (series.index <= ev["post_end"])].dropna()

    ax_main.plot(win_full.index, win_full.values, color=color, lw=1.0, label=f"{fuel_label} effettivo")
    ax_main.plot(baseline.index, baseline.values, color="grey", lw=1.2, ls="--",
                 label=f"Baseline OLS (R²={r2:.2f})")

    # CI bootstrap: fascia grigia attorno alla baseline
    ax_main.fill_between(
        ci_low.index, ci_low.values, ci_high.values,
        alpha=0.18, color="grey", label=f"CI 90% bootstrap (N={N_BOOT})"
    )

    # fill: guadagno extra (positivo=verde, negativo=rosso)
    ax_main.fill_between(
        post.index, post.values, baseline.values,
        where=(extra >= 0), alpha=0.20, color="green", label="Guadagno extra (≥0)"
    )
    ax_main.fill_between(
        post.index, post.values, baseline.values,
        where=(extra < 0), alpha=0.20, color="red", label="Perdita rispetto baseline (<0)"
    )

    ax_main.axvline(shock, color=ev["color"], lw=1.5, ls="--",
                    label=f"Evento ({shock.date()})")
    ax_main.axvline(tau,   color=color, lw=1.2, ls=":",
                    label=f"L1 τ ({tau.date()})  Δ={delta:+.3f} €/L")

    ax_main.set_ylabel("Margine (€/L)", fontsize=8)
    sig = "★" if p_bonf < 0.05 else ""
    ax_main.set_title(
        f"{fuel_label} – {ev_name}   |   τ={tau.date()}  p_bonf={p_bonf:.3f}{sig}",
        fontsize=8, fontweight="bold", pad=3
    )
    ax_main.legend(fontsize=6, loc="upper left", ncol=2)
    ax_main.grid(axis="y", alpha=0.20)
    ax_main.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    ax_main.xaxis.set_major_locator(mdates.MonthLocator())
    plt.setp(ax_main.xaxis.get_majorticklabels(), rotation=30, ha="right", fontsize=7)

    # ── ax_gain: guadagno cumulato ───────────────────────────────────────────
    cum_extra_eurl_day = extra.cumsum()   # €/L × giorni cumulati
    cum_extra_eur      = cum_extra_eurl_day * daily_consumption_l  # €

    cum_meur = cum_extra_eur / 1e6

    ax_gain.plot(cum_meur.index, cum_meur.values, color=color, lw=1.2)
    ax_gain.axhline(0, color="grey", lw=0.7, ls="--")
    ax_gain.fill_between(cum_meur.index, cum_meur.values, 0,
                         where=(cum_meur >= 0), alpha=0.25, color="green")
    ax_gain.fill_between(cum_meur.index, cum_meur.values, 0,
                         where=(cum_meur < 0), alpha=0.25, color="red")
    ax_gain.axvline(tau, color=color, lw=1.0, ls=":", alpha=0.8)

    final_meur = cum_meur.iloc[-1] if len(cum_meur) else 0
    n_days     = len(post)
    ax_gain.set_title(
        f"Guadagno extra cumulato  →  {final_meur:+.0f} M€  ({n_days}gg post-τ)\n"
        f"[consumo stimato {daily_consumption_l/1e6:.0f} ML/giorno]",
        fontsize=7, pad=3
    )
    ax_gain.set_ylabel("Guadagno cumulato (M€)", fontsize=8)
    ax_gain.grid(axis="y", alpha=0.20)
    ax_gain.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    ax_gain.xaxis.set_major_locator(mdates.MonthLocator())
    plt.setp(ax_gain.xaxis.get_majorticklabels(), rotation=30, ha="right", fontsize=7)

    # CI bootstrap sul guadagno totale (usando la traiettoria worst/best case)
    extra_low  = (post - ci_high).fillna(0)   # baseline alta → guadagno minimo
    extra_high = (post - ci_low).fillna(0)    # baseline bassa → guadagno massimo
    gain_ci_low  = float(extra_low.cumsum().iloc[-1]  * daily_consumption_l / 1e6) if len(extra_low)  else np.nan
    gain_ci_high = float(extra_high.cumsum().iloc[-1] * daily_consumption_l / 1e6) if len(extra_high) else np.nan

    return {
        "extra_mean_eurl":   float(extra.mean()),
        "extra_cumsum_eurl": float(cum_extra_eurl_day.iloc[-1]) if len(extra) else np.nan,
        "gain_total_meur":   float(final_meur),
        "gain_ci_low_meur":  gain_ci_low,
        "gain_ci_high_meur": gain_ci_high,
        "n_days_post_tau":   n_days,
        "r2_baseline":       r2,
        "slope_pre":         slope,
    }


# ══════════════════════════════════════════════════════════════════════════════
# Grafico 5: margini benzina e gasolio su assi separati, serie completa
# ══════════════════════════════════════════════════════════════════════════════

def plot_margin_overview(daily: pd.DataFrame, out_dir: Path) -> None:
    """
    Grafico pulito con solo i margini benzina e gasolio (serie completa).
    Pannelli sovrapposti con asse Y indipendente + eventi in verticale.
    """
    fig, axes = plt.subplots(2, 1, figsize=(16, 7), sharex=True)
    fig.suptitle("Margine distributori – Benzina e Gasolio (serie storica completa)",
                 fontsize=12, fontweight="bold")

    cfg = [
        ("margin_benzina", "Benzina", "#E63946"),
        ("margin_gasolio", "Gasolio", "#1D3557"),
    ]

    for ax, (col, label, color) in zip(axes, cfg):
        s = daily[col].dropna()
        if s.empty:
            ax.set_title(f"{label} – dati non disponibili", fontsize=9)
            continue

        ax.plot(s.index, s.values, color=color, lw=0.9, label=label)
        ax.axhline(s.mean(), color=color, lw=0.8, ls="--", alpha=0.5,
                   label=f"Media = {s.mean():.3f} €/L")

        for ev_name, ev in EVENTS.items():
            ax.axvline(ev["shock"], color=ev["color"], lw=1.4, ls="--", alpha=0.8,
                       label=ev["label"].replace("\n", " "))

        ax.set_ylabel("Margine (€/L)", fontsize=9)
        ax.set_title(label, fontsize=9, fontweight="bold")
        ax.legend(fontsize=7, loc="upper left", ncol=4)
        ax.grid(axis="y", alpha=0.20)
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=6))
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=30, ha="right", fontsize=8)

    fig.tight_layout()
    out = out_dir / "05_margini_benzina_gasolio.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  → Salvato: {out}")


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    # ── Carica dati ──────────────────────────────────────────────────────────
    daily = (pd.read_csv(DAILY_CSV, parse_dates=["date"])
               .sort_values("date").set_index("date"))

    print("Carico futures e EUR/USD...")
    eurusd = load_eurusd(
        csv_path=EURUSD_CSV if EURUSD_CSV.exists() else None,
        start="2015-01-01", end="2026-12-31"
    )
    gasoil_eurl  = load_futures_eurl(GASOIL_CSV, GAS_OIL, eurusd)
    eurobob_eurl = load_futures_b7h1(EUROBOB_CSV, EUROBOB_HC, eurusd) \
                   if EUROBOB_CSV.exists() else None

    daily = build_margin(daily, gasoil_eurl, eurobob_eurl)

    # ── Grafico 5: margini puri benzina/gasolio ───────────────────────────────
    print("\nGrafico 5: margini benzina e gasolio (serie completa)...")
    plot_margin_overview(daily, OUT_DIR)

    # ── Analisi controfattuale per evento ─────────────────────────────────────
    print("\nAnalisi controfattuale guadagni extra:")
    summary_rows = []

    for ev_name, ev in EVENTS.items():
        print(f"\n  {'─'*60}")
        print(f"  EVENTO: {ev_name}  (shock={ev['shock'].date()})")

        # una figura per evento (2 righe × 2 colonne: benzina|gasolio × main|gain)
        fig, axes = plt.subplots(2, 2, figsize=(16, 9))
        fig.suptitle(
            f"Analisi Controfattuale – {ev_name}\n"
            f"Shock: {ev['label'].replace(chr(10), ' ')}",
            fontsize=12, fontweight="bold"
        )

        for col_idx, (fuel_key, (col_name, fuel_color)) in enumerate(FUELS.items()):
            series = daily[col_name].dropna()

            win = series[(series.index >= ev["pre_start"]) &
                         (series.index <= ev["post_end"])]
            if len(win) < 2 * HALF_WIN:
                print(f"    [{fuel_key}] dati insufficienti, salto.")
                continue

            # Trova τ con sliding t-test
            ttest_df = sliding_ttest(win, ev["shock"])
            if ttest_df.empty:
                print(f"    [{fuel_key}] t-test vuoto, salto.")
                continue

            tau     = ttest_df.iloc[0]["tau"]
            delta   = ttest_df.iloc[0]["delta_mean"]
            p_bonf  = ttest_df.iloc[0]["p_bonf"]

            ax_main = axes[col_idx][0]
            ax_gain = axes[col_idx][1]

            res = plot_counterfactual(
                ev_name, ev, series,
                fuel_label=fuel_key.capitalize(),
                fuel_color=fuel_color,
                tau=tau, delta=delta, p_bonf=p_bonf,
                ax_main=ax_main, ax_gain=ax_gain,
                daily_consumption_l=DAILY_CONSUMPTION_L[fuel_key],
            )

            sig = "★" if p_bonf < 0.05 else "–"
            print(f"    [{fuel_key.upper()}]  τ={tau.date()}  "
                  f"Δ={delta:+.4f} €/L  p={p_bonf:.3f} {sig}")
            print(f"      Extra medio:   {res['extra_mean_eurl']:+.4f} €/L/gg")
            print(f"      Guadagno tot:  {res['gain_total_meur']:+.0f} M€  "
                  f"({res['n_days_post_tau']} gg post-τ)")

            summary_rows.append({
                "evento":             ev_name,
                "carburante":         fuel_key,
                "shock":              ev["shock"].date(),
                "tau_L1":             tau.date(),
                "delta_eurl":         round(delta, 4),
                "p_bonf":             round(p_bonf, 4),
                "extra_mean_eurl":    round(res["extra_mean_eurl"], 4),
                "gain_total_meur":    round(res["gain_total_meur"], 1),
                "gain_ci_low_meur":   round(res["gain_ci_low_meur"],  1) if res["gain_ci_low_meur"]  is not None else np.nan,
                "gain_ci_high_meur":  round(res["gain_ci_high_meur"], 1) if res["gain_ci_high_meur"] is not None else np.nan,
                "n_days_post_tau":    res["n_days_post_tau"],
                "r2_baseline":        round(res["r2_baseline"], 3),
                "slope_pre_eurl_day": round(res["slope_pre"], 6),
                "consumo_ML_die":     DAILY_CONSUMPTION_L[fuel_key] / 1e6,
            })

        fig.tight_layout()
        safe_name = ev_name.replace(" ", "_").replace("/", "").replace("(", "").replace(")", "")
        out = OUT_DIR / f"04_counterfactual_{safe_name}.png"
        fig.savefig(out, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  → Salvato: {out}")

    # ── CSV riepilogativo ─────────────────────────────────────────────────────
    if summary_rows:
        df_sum = pd.DataFrame(summary_rows)
        csv_out = OUT_DIR / "04_counterfactual_summary.csv"
        df_sum.to_csv(csv_out, index=False)
        print(f"\n  → CSV summary: {csv_out}")

        print(f"\n{'═'*70}")
        print("RIEPILOGO GUADAGNI EXTRA (stima controfattuale)")
        print(f"{'═'*70}")
        print(df_sum.to_string(index=False))

    # ── Grafico riepilogativo: guadagni per evento+carburante ─────────────────
    if summary_rows:
        df_sum = pd.DataFrame(summary_rows)
        fig, ax = plt.subplots(figsize=(10, 5))
        fig.suptitle("Guadagno extra stimato per evento e carburante\n"
                     "(baseline = trend lineare pre-breakpoint, volume nazionale medio)",
                     fontsize=10, fontweight="bold")

        ev_names   = df_sum["evento"].unique()
        x          = np.arange(len(ev_names))
        width      = 0.35
        colors_map = {"benzina": "#E63946", "gasolio": "#1D3557"}

        for i, fuel in enumerate(["benzina", "gasolio"]):
            sub = df_sum[df_sum["carburante"] == fuel].set_index("evento")
            vals = [sub.loc[ev, "gain_total_meur"] if ev in sub.index else 0 for ev in ev_names]
            bars = ax.bar(x + (i - 0.5) * width, vals, width, label=fuel.capitalize(),
                          color=colors_map[fuel], alpha=0.85, edgecolor="white")
            for bar, v in zip(bars, vals):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + (0.5 if v >= 0 else -3),
                        f"{v:+.0f} M€", ha="center", va="bottom", fontsize=7)

        ax.axhline(0, color="black", lw=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels(ev_names, fontsize=9)
        ax.set_ylabel("Guadagno extra cumulato (M€)", fontsize=9)
        ax.legend(fontsize=9)
        ax.grid(axis="y", alpha=0.20)
        fig.tight_layout()
        out_bar = OUT_DIR / "04_counterfactual_barplot.png"
        fig.savefig(out_bar, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  → Barplot: {out_bar}")


if __name__ == "__main__":
    main()