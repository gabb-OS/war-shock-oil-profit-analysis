#!/usr/bin/env python3
"""
02d_v7_theilsen.py  ─  Metodo 7: Theil-Sen + Block Bootstrap CI
=================================================================
Stima completamente non-parametrica dell'extra-profitto speculativo sul
margine distributori (prezzo pompa netto − futures €/L) in corrispondenza
di eventi geopolitici.

Perché Theil-Sen invece di OLS / GLM Gamma:
  ─ OLS      : assume residui i.i.d. normali → distorto con code pesanti
  ─ GLM Gamma: richiede y > 0 → shift artificiale sui margini negativi
  ─ Theil-Sen: slope = mediana di tutti i rapporti (y_j−y_i)/(t_j−t_i)
               zero assunzioni distributive, breakdown point ≈ 29%,
               funziona con margini negativi/estremi/asimmetrici

Stima baseline:
  slope    = theilslopes(y_pre, t_pre).slope     (mediana dei pendii)
  intercept= theilslopes(y_pre, t_pre).intercept (mediana compatibile)
  baseline = slope * t_post + intercept

Intervalli di confidenza:
  Block bootstrap circolare (Künsch 1989):
    1. Calcola residui pre-periodo:  r_t = y_pre_t − baseline_pre_t
    2. Per B iterazioni:
         · Ricampiona blocchi di lunghezza L (circolare) dai residui
         · Ricostruisce y_boot = baseline_pre + r_boot
         · Re-stima Theil-Sen su y_boot → slope_b, intercept_b
         · Proietta baseline_b sul post-periodo
         · Calcola gain_b = Σ[(y_post_t − baseline_b_t) · cons_t] / 1e6
    3. CI [α/2, 1−α/2] = percentili bootstrap di {gain_b}
  Lunghezza blocco: L = max(3, ceil(√n_pre))   — regola del pollice
                    per autocorrelazione moderata (DW < 2 → L più lungo)

Output CSV compatibile con 02d_compare.py (stesse colonne standard).

Modalità (--mode):
  fixed     : break = data dello shock hardcodata               [default]
  detected  : break θ letto da theta_results.csv (02c_change_point_detection)

Parametro --detect (solo mode=detected):
  margin  : usa θ rilevato sul margine distributore            [default]
  price   : usa θ rilevato sul prezzo alla pompa netto (€/L)

Output:
  data/plots/its/fixed/v7_theilsen/                    (mode=fixed)
  data/plots/its/detected/{margin|price}/v7_theilsen/  (mode=detected)
    plot_{evento}.png
    diag_{evento}_{carburante}.png
    v7_theilsen_results.csv
"""

from __future__ import annotations
from pathlib import Path
import argparse
import math
import sys
import warnings

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).parent / "utils"))
from conversions import GAS_OIL, EUROBOB as EUROBOB_HC, load_eurusd, usd_ton_to_eur_liter
from diagnostics import run_diagnostic_tests, plot_residual_diagnostics
from theta_loader import load_theta
from forecast_consumi import load_daily_consumption

# ── Configurazione ─────────────────────────────────────────────────────────────
BASE_DIR    = Path(__file__).parent
DAILY_CSV   = BASE_DIR / "data" / "processed" / "daily_fuel_prices_all.csv"
GASOIL_CSV  = BASE_DIR / "data" / "Futures" / "London Gas Oil Futures Historical Data.csv"
EUROBOB_CSV = BASE_DIR / "data" / "Futures" / "Eurobob_B7H1_date.csv"
EURUSD_CSV  = BASE_DIR / "data" / "raw" / "eurusd.csv"
_OUT_BASE   = BASE_DIR / "data" / "plots" / "its"

PRE_WIN   = 40      # giorni pre-break per stimare la baseline
POST_WIN  = 40      # giorni post-break per calcolare l'extra profitto
CI_ALPHA  = 0.05    # α → CI al 95%
N_BOOT    = 2000    # iterazioni bootstrap
SEED      = 42      # riproducibilità

METHOD_COLOR = "#6c3483"   # viola per distinguere da altri metodi

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
# Caricamento dati  (identico agli altri metodi)
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
    eurusd  = load_eurusd(
        csv_path=EURUSD_CSV if EURUSD_CSV.exists() else None,
        start="2015-01-01", end="2026-12-31",
    )
    gasoil  = _load_gasoil_futures(eurusd)
    eurobob = _load_eurobob_futures(eurusd)
    df = daily[["benzina_net", "gasolio_net"]].copy()
    df["margin_gasolio"] = df["gasolio_net"] - gasoil.reindex(df.index, method="ffill")
    df["margin_benzina"] = (
        df["benzina_net"] - eurobob.reindex(df.index, method="ffill")
        if eurobob is not None else np.nan
    )
    return df


# ══════════════════════════════════════════════════════════════════════════════
# Theil-Sen fit
# ══════════════════════════════════════════════════════════════════════════════

def fit_theilsen(series: pd.Series, break_date: pd.Timestamp) -> dict | None:
    """
    Stima Theil-Sen sulla finestra pre-break.

    Theil-Sen:
      slope     = mediana di { (y_j − y_i) / (t_j − t_i) : j > i }
      intercept = mediana(y) − slope · mediana(t)

    Resistente a outlier e code pesanti; nessuna assunzione sulla
    distribuzione dei residui; funziona con valori negativi.

    Ritorna dict con:
      slope, intercept, break_date, pre (Series),
      residuals (array), x (array giorni), x_mean,
      n, ts_result (oggetto scipy per CI interni slope)
    """
    pre = series[
        (series.index >= break_date - pd.Timedelta(days=PRE_WIN)) &
        (series.index <  break_date)
    ].dropna()

    if len(pre) < 10:
        return None

    x = np.array([(d - break_date).days for d in pre.index], dtype=float)
    y = pre.values

    # scipy.stats.theilslopes restituisce (slope, intercept, low_slope, high_slope)
    ts = stats.theilslopes(y, x, alpha=1 - CI_ALPHA)

    slope     = float(ts.slope)
    intercept = float(ts.intercept)
    fitted    = slope * x + intercept
    residuals = y - fitted

    # Pseudo-R² (varianza spiegata dalla mediana lineare)
    ss_res = float(np.sum(residuals ** 2))
    ss_tot = float(np.sum((y - np.median(y)) ** 2))
    pseudo_r2 = 1.0 - ss_res / ss_tot if ss_tot > 1e-12 else 0.0

    # Design matrix (per compatibilità diagnostics BG test)
    X_bg = np.column_stack([np.ones(len(x)), x])

    return dict(
        slope      = slope,
        intercept  = intercept,
        pseudo_r2  = pseudo_r2,
        break_date = break_date,
        pre        = pre,
        x          = x,
        x_mean     = float(x.mean()),
        n          = len(pre),
        residuals  = residuals,
        X_bg       = X_bg,
        ts_result  = ts,          # conserva per diagnostica slope
    )


def project_theilsen(
    fit: dict,
    post_index: pd.DatetimeIndex,
) -> pd.Series:
    """
    Proietta la retta Theil-Sen sul periodo post-break.
    Ritorna solo la baseline puntuale — i CI vengono dal block bootstrap.
    """
    x_post   = np.array([(d - fit["break_date"]).days for d in post_index], dtype=float)
    baseline = fit["slope"] * x_post + fit["intercept"]
    return pd.Series(baseline, index=post_index)


# ══════════════════════════════════════════════════════════════════════════════
# Block Bootstrap per CI sul guadagno
# ══════════════════════════════════════════════════════════════════════════════

def _block_len(n: int, dw: float | None = None) -> int:
    """
    Lunghezza blocco per block bootstrap circolare.

    Regola base: L = ceil(√n)
    Se DW < 1.5 (autocorrelazione positiva forte) → L × 1.5
    """
    base = math.ceil(math.sqrt(n))
    if dw is not None and dw < 1.5:
        base = math.ceil(base * 1.5)
    return max(3, base)


def _circular_block_resample(
    residuals: np.ndarray,
    block_len: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Ricampionamento a blocchi circolare (wrap-around) dei residui.
    Produce un array della stessa lunghezza di residuals.
    """
    n       = len(residuals)
    n_blocks = math.ceil(n / block_len)
    # Indice di partenza di ogni blocco: uniforme su [0, n)
    starts  = rng.integers(0, n, size=n_blocks)
    indices = np.concatenate([
        np.arange(s, s + block_len) % n for s in starts
    ])
    return residuals[indices[:n]]


def bootstrap_gain_ci(
    fit:        dict,
    post_data:  pd.Series,
    cons:       pd.Series,
    n_boot:     int = N_BOOT,
    seed:       int = SEED,
) -> tuple[float, float, float, np.ndarray]:
    """
    Block bootstrap circolare per CI sul guadagno cumulato (M€).

    Per ogni iterazione b:
      1. Ricampiona residui pre in blocchi → r_boot
      2. y_boot = fitted_pre + r_boot
      3. Re-stima Theil-Sen su y_boot
      4. Proietta baseline_b sul post
      5. gain_b = Σ[(y_post_t − baseline_b_t) · cons_t] / 1e6

    Ritorna:
      gain_point   : guadagno puntuale (stima originale)
      ci_low       : percentile α/2
      ci_high      : percentile 1−α/2
      boot_gains   : array di tutti i gain bootstrap (per diagnostica)
    """
    break_date = fit["break_date"]
    slope      = fit["slope"]
    intercept  = fit["intercept"]
    x_pre      = fit["x"]
    residuals  = fit["residuals"]
    n_pre      = fit["n"]

    # Guadagno puntuale
    baseline_post = project_theilsen(fit, post_data.index)
    extra_point   = post_data - baseline_post
    gain_point    = float((extra_point * cons).sum() / 1e6)

    # DW approssimato sui residui pre (per scegliere lunghezza blocco)
    if n_pre > 4:
        dw_approx = float(np.sum(np.diff(residuals) ** 2) / (np.sum(residuals ** 2) + 1e-12))
    else:
        dw_approx = 2.0
    L   = _block_len(n_pre, dw_approx)
    rng = np.random.default_rng(seed)

    # Fitted pre-baseline (sul quale reinnestare i residui ricampionati)
    fitted_pre = slope * x_pre + intercept

    x_post = np.array(
        [(d - break_date).days for d in post_data.index], dtype=float
    )

    boot_gains = np.empty(n_boot, dtype=float)
    for b in range(n_boot):
        r_boot   = _circular_block_resample(residuals, L, rng)
        y_boot   = fitted_pre + r_boot

        ts_b     = stats.theilslopes(y_boot, x_pre)
        base_b   = ts_b.slope * x_post + ts_b.intercept
        gain_b   = float(((post_data.values - base_b) * cons.values).sum() / 1e6)
        boot_gains[b] = gain_b

    ci_low  = float(np.percentile(boot_gains, 100 * CI_ALPHA / 2))
    ci_high = float(np.percentile(boot_gains, 100 * (1 - CI_ALPHA / 2)))

    return gain_point, ci_low, ci_high, boot_gains


# ══════════════════════════════════════════════════════════════════════════════
# Plot per singolo evento + carburante
# ══════════════════════════════════════════════════════════════════════════════

def _plot_event_fuel(
    ev_name:    str,
    ev:         dict,
    series:     pd.Series,
    fuel_key:   str,
    fuel_color: str,
    fit:        dict,
    baseline:   pd.Series,
    ci_low_arr: pd.Series,    # CI bootstrap sul baseline (per la banda)
    ci_high_arr: pd.Series,
    extra:      pd.Series,
    gain_meur:  float,
    ci_low_gain: float,
    ci_high_gain: float,
    cons:       pd.Series,
    boot_gains: np.ndarray,
    break_date: pd.Timestamp,
    mode:       str,
    ax_main:    plt.Axes,
    ax_gain:    plt.Axes,
    ax_boot:    plt.Axes,
) -> None:
    shock = ev["shock"]

    win = series[
        (series.index >= shock - pd.Timedelta(days=PRE_WIN)) &
        (series.index <= shock + pd.Timedelta(days=POST_WIN))
    ].dropna()

    # ── Pannello 1: margine effettivo vs baseline ─────────────────────────────
    ax_main.plot(win.index, win.values, color=fuel_color, lw=1.0,
                 label=f"{fuel_key.capitalize()} effettivo")
    ax_main.plot(baseline.index, baseline.values, color=METHOD_COLOR, lw=1.4,
                 ls="--", label=f"Baseline Theil-Sen (R²={fit['pseudo_r2']:.2f})")
    ax_main.fill_between(
        baseline.index, ci_low_arr.values, ci_high_arr.values,
        alpha=0.15, color=METHOD_COLOR,
        label=f"CI {int((1-CI_ALPHA)*100)}% bootstrap",
    )
    ax_main.fill_between(extra.index,
                         win.reindex(extra.index), baseline.values,
                         where=(extra >= 0), alpha=0.22, color="green",
                         label="Extra profitto (≥0)")
    ax_main.fill_between(extra.index,
                         win.reindex(extra.index), baseline.values,
                         where=(extra < 0), alpha=0.22, color="red",
                         label="Sotto-baseline (<0)")
    ax_main.axvline(shock, color=ev["color"], lw=1.6, ls="--",
                    label=f"Shock ({shock.date()})")
    if mode == "detected" and break_date != shock:
        ax_main.axvline(break_date, color="black", lw=1.2, ls=":",
                        label=f"τ rilevato ({break_date.date()})")

    mode_str = (f"Break=θ {break_date.date()} (GLM Poisson 02c)"
                if mode == "detected" else f"Break=shock ({shock.date()})")
    ax_main.set_title(
        f"[V7-TheilSen / mode={mode}]  {fuel_key.capitalize()} – {ev_name}\n"
        f"{mode_str}  |  slope={fit['slope']:+.5f} €/L/g",
        fontsize=8, fontweight="bold",
    )
    ax_main.set_ylabel("Margine (€/L)", fontsize=8)
    ax_main.legend(fontsize=6, loc="upper left", ncol=2)
    ax_main.grid(axis="y", alpha=0.20)
    ax_main.xaxis.set_major_formatter(mdates.DateFormatter("%d %b %y"))
    ax_main.xaxis.set_major_locator(mdates.WeekdayLocator(byweekday=0, interval=2))
    plt.setp(ax_main.xaxis.get_majorticklabels(), rotation=35, ha="right", fontsize=7)

    # ── Pannello 2: guadagno cumulato ─────────────────────────────────────────
    cum = (extra * cons.values / 1e6).cumsum()
    ax_gain.plot(cum.index, cum.values, color=fuel_color, lw=1.2)
    ax_gain.axhline(0, color="grey", lw=0.7, ls="--")
    ax_gain.fill_between(cum.index, cum.values, 0,
                         where=(cum >= 0), alpha=0.25, color="green")
    ax_gain.fill_between(cum.index, cum.values, 0,
                         where=(cum < 0), alpha=0.25, color="red")
    ax_gain.axhline(ci_low_gain,  color=METHOD_COLOR, lw=0.9, ls=":",
                    label=f"CI bootstrap [{ci_low_gain:+.0f}, {ci_high_gain:+.0f}] M€")
    ax_gain.axhline(ci_high_gain, color=METHOD_COLOR, lw=0.9, ls=":")
    avg_cons_ml = cons.mean() / 1e6
    ax_gain.set_title(
        f"Guadagno extra cumulato → {gain_meur:+.0f} M€\n"
        f"CI95% bootstrap [{ci_low_gain:+.0f}, {ci_high_gain:+.0f}] M€  "
        f"[cons. medio {avg_cons_ml:.1f} ML/g]",
        fontsize=7,
    )
    ax_gain.set_ylabel("M€ cumulati", fontsize=8)
    ax_gain.legend(fontsize=6)
    ax_gain.grid(axis="y", alpha=0.20)
    ax_gain.xaxis.set_major_formatter(mdates.DateFormatter("%d %b %y"))
    ax_gain.xaxis.set_major_locator(mdates.WeekdayLocator(byweekday=0, interval=2))
    plt.setp(ax_gain.xaxis.get_majorticklabels(), rotation=35, ha="right", fontsize=7)

    # ── Pannello 3: distribuzione bootstrap dei gain ──────────────────────────
    ax_boot.hist(boot_gains, bins=60, color=METHOD_COLOR, alpha=0.65,
                 edgecolor="white", linewidth=0.4)
    ax_boot.axvline(gain_meur,   color="black",       lw=1.6, ls="-",
                    label=f"Stima puntuale {gain_meur:+.0f} M€")
    ax_boot.axvline(ci_low_gain, color=METHOD_COLOR,  lw=1.2, ls="--",
                    label=f"CI95% [{ci_low_gain:+.0f}, {ci_high_gain:+.0f}]")
    ax_boot.axvline(ci_high_gain, color=METHOD_COLOR, lw=1.2, ls="--")
    ax_boot.axvline(0, color="grey", lw=0.8, ls=":")
    ax_boot.set_xlabel("Guadagno bootstrap (M€)", fontsize=7)
    ax_boot.set_ylabel("Frequenza", fontsize=7)
    ax_boot.set_title(
        f"Distribuzione bootstrap ({N_BOOT} iter.)  |  "
        f"SD={float(boot_gains.std()):.1f} M€",
        fontsize=7,
    )
    ax_boot.legend(fontsize=6)
    ax_boot.grid(axis="y", alpha=0.15)


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    parser = argparse.ArgumentParser(
        description="V7 Theil-Sen + Block Bootstrap – ITS pipeline"
    )
    parser.add_argument(
        "--mode", choices=["fixed", "detected"], default="fixed",
        help="fixed = usa shock date hardcodata; detected = usa θ da 02c",
    )
    parser.add_argument(
        "--detect", choices=["margin", "price"], default="margin",
        help="(solo mode=detected) serie su cui è stata fatta detection",
    )
    args, _       = parser.parse_known_args()
    mode          = args.mode
    detect_target = args.detect

    if mode == "detected":
        OUT_DIR = _OUT_BASE / "detected" / detect_target / "v7_theilsen"
    else:
        OUT_DIR = _OUT_BASE / mode / "v7_theilsen"
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("═" * 70)
    print(f"  02d_v7_theilsen.py  –  Metodo 7: Theil-Sen + Block Bootstrap  [mode={mode}]")
    if mode == "fixed":
        print("  Break = shock date hardcodata (nessuna detection)")
    else:
        print("  Break = θ GLM Poisson da 02c_change_point_detection.py")
        print(f"  Detection su: {'MARGINE distributore' if detect_target == 'margin' else 'PREZZO POMPA NETTO'}")
    print(f"  Finestra: PRE={PRE_WIN}gg / POST={POST_WIN}gg dal break point")
    print(f"  Bootstrap: B={N_BOOT}  seed={SEED}  CI={int((1-CI_ALPHA)*100)}%")
    print(f"  Output: {OUT_DIR}")
    print("═" * 70)

    data = load_margin_data()
    rows: list[dict] = []

    for ev_name, ev in EVENTS.items():
        shock = ev["shock"]

        fig, axes = plt.subplots(
            len(FUELS), 3,
            figsize=(18, 5 * len(FUELS)),
            squeeze=False,
        )
        fig.suptitle(
            f"[Metodo 7 – Theil-Sen + Block Bootstrap / mode={mode}]  {ev_name}\n"
            f"{ev['label']}",
            fontsize=11, fontweight="bold",
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

            # ── Finestre dati ─────────────────────────────────────────────────
            pre_data  = series[
                (series.index >= break_date - pd.Timedelta(days=PRE_WIN)) &
                (series.index <  break_date)
            ]
            post_data = series[
                (series.index >= break_date) &
                (series.index <  shock + pd.Timedelta(days=POST_WIN))
            ]

            if len(pre_data) < 10 or len(post_data) < 5:
                print(f"  [{fuel_key}] dati insufficienti – salto.")
                for ax in axes[row_idx]:
                    ax.text(0.5, 0.5, "Dati insufficienti",
                            ha="center", va="center", transform=ax.transAxes)
                continue

            # ── Theil-Sen fit ─────────────────────────────────────────────────
            fit = fit_theilsen(series, break_date)
            if fit is None:
                print(f"  [{fuel_key}] fit fallito – salto.")
                continue

            baseline  = project_theilsen(fit, post_data.index)
            extra     = post_data - baseline

            # ── Consumi giornalieri ───────────────────────────────────────────
            cons = load_daily_consumption(post_data.index, fuel_key)

            # ── Block bootstrap CI sul guadagno ───────────────────────────────
            print(f"  [{fuel_key}] block bootstrap {N_BOOT} iter. ...", end="", flush=True)
            gain_meur, ci_low_gain, ci_high_gain, boot_gains = bootstrap_gain_ci(
                fit, post_data, cons,
            )
            print(f" done  ({gain_meur:+.0f} M€  CI95%=[{ci_low_gain:+.0f}, {ci_high_gain:+.0f}])")

            # ── CI sul baseline per la banda nel plot ─────────────────────────
            # Ricaviamo la banda con percentili bootstrap sulle proiezioni giorno-per-giorno
            rng    = np.random.default_rng(SEED + 1)
            x_pre  = fit["x"]
            residuals_pre = fit["residuals"]
            fitted_pre    = fit["slope"] * x_pre + fit["intercept"]
            x_post = np.array(
                [(d - break_date).days for d in post_data.index], dtype=float
            )
            L = _block_len(fit["n"])

            all_baselines = np.empty((N_BOOT, len(post_data)), dtype=float)
            for b in range(N_BOOT):
                r_b   = _circular_block_resample(residuals_pre, L, rng)
                y_b   = fitted_pre + r_b
                ts_b  = stats.theilslopes(y_b, x_pre)
                all_baselines[b] = ts_b.slope * x_post + ts_b.intercept

            ci_low_base  = pd.Series(
                np.percentile(all_baselines, 100 * CI_ALPHA / 2, axis=0),
                index=post_data.index,
            )
            ci_high_base = pd.Series(
                np.percentile(all_baselines, 100 * (1 - CI_ALPHA / 2), axis=0),
                index=post_data.index,
            )

            # ── Diagnostica residui pre (compatibilità pipeline) ──────────────
            pre_resid = fit["residuals"]
            diag = run_diagnostic_tests(
                pre_resid,
                x_for_bg=fit["X_bg"],
                n_lags=None,
            )

            safe_ev = (ev_name.replace(" ", "_").replace("/", "")
                       .replace("(", "").replace(")", ""))
            diag_path = OUT_DIR / f"diag_{safe_ev}_{fuel_key}.png"
            plot_residual_diagnostics(
                resid=pre_resid,
                dates=fit["pre"].index,
                title=(f"[V7-TheilSen] Diagnostica residui pre-periodo\n"
                       f"{ev_name} · {fuel_key.capitalize()}  "
                       f"(break={break_date.date()})"),
                out_path=diag_path,
                diag_stats=diag,
            )

            # ── Plot ──────────────────────────────────────────────────────────
            _plot_event_fuel(
                ev_name, ev, series, fuel_key, fuel_color,
                fit, baseline,
                ci_low_base, ci_high_base,
                extra, gain_meur, ci_low_gain, ci_high_gain,
                cons, boot_gains, break_date, mode,
                axes[row_idx][0], axes[row_idx][1], axes[row_idx][2],
            )

            # ── Stampa a video ────────────────────────────────────────────────
            ts_res  = fit["ts_result"]
            boot_sd = float(boot_gains.std())
            print(f"\n  {ev_name}  [{fuel_key.upper()}]")
            print(f"    Break ({break_method}) = {break_date.date()}  (shock={shock.date()})")
            print(f"    Theil-Sen  slope = {fit['slope']:+.5f} €/L/g  "
                  f"[CI slope: {ts_res.low_slope:+.5f}, {ts_res.high_slope:+.5f}]")
            print(f"    Pseudo-R²  = {fit['pseudo_r2']:.3f}")
            print(f"    Extra medio = {extra.mean():+.4f} €/L/g")
            print(f"    Guadagno    = {gain_meur:+.0f} M€   "
                  f"CI{int((1-CI_ALPHA)*100)}% bootstrap [{ci_low_gain:+.0f}, {ci_high_gain:+.0f}] M€")
            print(f"    SD bootstrap = {boot_sd:.1f} M€  |  "
                  f"blocco L={_block_len(fit['n'])} gg")
            if not np.isnan(diag.get("sw_p", np.nan)):
                print(f"    SW residui  W={diag['sw_stat']:.3f}  p={diag['sw_p']:.3f}  "
                      f"{'OK' if diag['sw_p'] > 0.05 else '⚠ non norm.'}")
            if not np.isnan(diag.get("lb_p", np.nan)):
                print(f"    LB({diag['n_lags']}) autocorr.  "
                      f"Q={diag['lb_stat']:.2f}  p={diag['lb_p']:.3f}  "
                      f"{'OK' if diag['lb_p'] > 0.05 else '⚠ autocorr.'}")

            # ── Export residui pre/post (standard per 02d_compare nonparam) ──
            _safe_ev = (ev_name.replace(" ", "_").replace("/", "")
                               .replace("(", "").replace(")", ""))
            _resid_rows = []
            for _d, _r in zip(fit["pre"].index, fit["residuals"]):
                _resid_rows.append({
                    "date": str(_d.date()), "residual": float(_r), "phase": "pre",
                    "metodo": "v7_theilsen", "evento": ev_name,
                    "carburante": fuel_key, "break_date": str(break_date.date()),
                })
            for _d, _r in zip(post_data.index, extra.values):
                _resid_rows.append({
                    "date": str(_d.date()), "residual": float(_r), "phase": "post",
                    "metodo": "v7_theilsen", "evento": ev_name,
                    "carburante": fuel_key, "break_date": str(break_date.date()),
                })
            pd.DataFrame(_resid_rows).to_csv(
                OUT_DIR / f"residuals_{_safe_ev}_{fuel_key}.csv", index=False
            )

            # ── Record CSV ────────────────────────────────────────────────────
            rows.append({
                # ── Colonne standard per compare.py ───────────────────────────
                "metodo":             "v7_theilsen",
                "mode":               mode,
                "detect_target":      detect_target if mode == "detected" else "fixed",
                "evento":             ev_name,
                "carburante":         fuel_key,
                "shock":              shock.date(),
                "break_date":         break_date.date(),
                "break_method":       break_method,
                "pre_win_days":       PRE_WIN,
                "post_win_days":      POST_WIN,
                "n_pre":              len(pre_data),
                "n_post":             len(post_data),
                "pre_std_eurl":       round(float(pre_data.std(ddof=1)), 6),
                "extra_mean_eurl":    round(float(extra.mean()), 5),
                "gain_total_meur":    round(gain_meur,    1),
                "gain_ci_low_meur":   round(ci_low_gain,  1),
                "gain_ci_high_meur":  round(ci_high_gain, 1),
                # ── Diagnostica Theil-Sen ─────────────────────────────────────
                "ts_slope":           round(fit["slope"],        6),
                "ts_intercept":       round(fit["intercept"],    5),
                "ts_slope_ci_low":    round(float(ts_res.low_slope),  6),
                "ts_slope_ci_high":   round(float(ts_res.high_slope), 6),
                "pseudo_r2":          round(fit["pseudo_r2"],    4),
                # ── Bootstrap ────────────────────────────────────────────────
                "boot_n":             N_BOOT,
                "boot_sd_meur":       round(boot_sd,             2),
                "boot_block_len":     _block_len(fit["n"]),
                "ci_type":            f"block_bootstrap_{int((1-CI_ALPHA)*100)}pct",
                # ── Diagnostica residui pre ───────────────────────────────────
                "sw_stat":            round(diag.get("sw_stat", np.nan), 4),
                "sw_p":               round(diag.get("sw_p",    np.nan), 4),
                "lb_stat":            round(diag.get("lb_stat", np.nan), 3),
                "lb_p":               round(diag.get("lb_p",    np.nan), 4),
                "bg_stat":            round(diag.get("bg_stat", np.nan), 3),
                "bg_p":               round(diag.get("bg_p",    np.nan), 4),
                "diag_n_lags":        diag.get("n_lags", np.nan),
                "note": (
                    f"Theil-Sen non-parametrico + block bootstrap circolare, "
                    f"mode={mode}"
                    + (f", detect={detect_target}" if mode == "detected" else "")
                    + f", B={N_BOOT}, L={_block_len(fit['n'])}"
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
        df      = pd.DataFrame(rows)
        csv_out = OUT_DIR / "v7_theilsen_results.csv"
        df.to_csv(csv_out, index=False)
        print(f"\n  → CSV: {csv_out}")
        print("\n" + df[["evento", "carburante", "break_date",
                          "gain_total_meur", "gain_ci_low_meur", "gain_ci_high_meur",
                          "pseudo_r2", "boot_sd_meur"]].to_string(index=False))
    else:
        print("\n  ⚠ Nessun risultato prodotto.")


if __name__ == "__main__":
    main()