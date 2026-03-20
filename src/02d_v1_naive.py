#!/usr/bin/env python3
"""
02d_v1_naive.py  ─  Metodo 1: OLS Naïve
=========================================
Il metodo più semplice possibile per stimare l'extra-profitto speculativo
sul margine distributori (prezzo pompa netto − futures €/L) in corrispondenza
di eventi geopolitici.

Assunzioni (e limitazioni):
  ─ Baseline     : trend OLS lineare sui PRE_WIN giorni precedenti il break
  ─ Proiezione   : la retta OLS viene estrapolata in avanti (controfattuale)
  ─ Extra profitto : margine_effettivo(t) − baseline_proiettata(t)
  ─ CI           : intervallo di previsione OLS standard (i.i.d.)

Modalità (--mode):
  fixed     : break = data dello shock hardcodata [default]
  detected  : break θ letto da theta_results.csv prodotto da
              02c_change_point_detection.py (GLM Poisson canonico).
              Eseguire prima: python3 02c_change_point_detection.py --detect {margin|price}

Parametro --detect (solo mode=detected):
  margin  : usa θ rilevato sul margine distributore  [default]
  price   : usa θ rilevato sul prezzo alla pompa netto (€/L)

Output:
  data/plots/its/fixed/v1_naive/                    (mode=fixed)
  data/plots/its/detected/{margin|price}/v1_naive/  (mode=detected)
    plot_{evento}.png
    diag_{evento}_{carburante}.png
    v1_naive_results.csv
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
from forecast_consumi import load_daily_consumption

try:
    import statsmodels.api as _sm
    HAS_SM = True
except ImportError:
    HAS_SM = False

# ── Configurazione ─────────────────────────────────────────────────────────────
BASE_DIR    = Path(__file__).parent
DAILY_CSV   = BASE_DIR / "data" / "processed" / "daily_fuel_prices_all.csv"
GASOIL_CSV  = BASE_DIR / "data" / "Futures" / "London Gas Oil Futures Historical Data.csv"
EUROBOB_CSV = BASE_DIR / "data" / "Futures" / "Eurobob_B7H1_date.csv"
EURUSD_CSV  = BASE_DIR / "data" / "raw" / "eurusd.csv"
_OUT_BASE   = BASE_DIR / "data" / "plots" / "its"

PRE_WIN   = 40    # giorni pre-break per stimare la baseline
POST_WIN  = 40    # giorni post-break per calcolare l'extra profitto
CI_ALPHA  = 0.05   # livello α → intervallo di previsione al 90%

# SEARCH e MIN_SEG rimossi: la detection è centralizzata in 02c (GLM Poisson)


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
    df["date"] = pd.to_datetime(df["Date"], format="%m/%d/%Y", errors="coerce")
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
    eurusd = load_eurusd(
        csv_path=EURUSD_CSV if EURUSD_CSV.exists() else None,
        start="2015-01-01", end="2026-12-31"
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
# Metodo 1 – OLS Naïve
# ══════════════════════════════════════════════════════════════════════════════


def _fit_ols(series: pd.Series, break_date: pd.Timestamp) -> dict | None:
    """
    OLS lineare su PRE_WIN giorni prima di break_date.
    Nessuna correzione per autocorrelazione o eteroschedasticità.
    break_date può essere la shock date (mode=fixed) o il τ rilevato (mode=detected).
    """
    pre = series[
        (series.index >= break_date - pd.Timedelta(days=PRE_WIN)) &
        (series.index < break_date)
    ].dropna()

    if len(pre) < 10:
        return None

    x = np.array([(d - break_date).days for d in pre.index], dtype=float)
    y = pre.values

    slope, intercept, r, _, _ = stats.linregress(x, y)
    y_hat     = slope * x + intercept
    residuals = y - y_hat
    n         = len(pre)
    mse       = np.sum(residuals**2) / (n - 2)
    sxx       = np.sum((x - x.mean())**2)

    # Design matrix per BG test (costante + t): shape (n, 2)
    X_bg = np.column_stack([np.ones(n), x])

    return dict(slope=slope, intercept=intercept, r2=float(r**2),
                mse=mse, sxx=sxx, x_mean=float(x.mean()), n=n,
                break_date=break_date, pre=pre,
                residuals=residuals, X_bg=X_bg)


def _project_ols(fit: dict, post_index: pd.DatetimeIndex
                 ) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Proietta la retta OLS + PI (1-α)% in avanti dal break_date."""
    x = np.array([(d - fit["break_date"]).days for d in post_index], dtype=float)

    baseline = fit["slope"] * x + fit["intercept"]

    se_pred = np.sqrt(
        fit["mse"] * (1 + 1/fit["n"] + (x - fit["x_mean"])**2 / fit["sxx"])
    )
    t_crit  = stats.t.ppf(1 - CI_ALPHA / 2, df=fit["n"] - 2)

    return (
        pd.Series(baseline,                  index=post_index),
        pd.Series(baseline - t_crit*se_pred, index=post_index),
        pd.Series(baseline + t_crit*se_pred, index=post_index),
    )


# ══════════════════════════════════════════════════════════════════════════════
# Plot per singolo evento + carburante
# ══════════════════════════════════════════════════════════════════════════════

def _plot_event_fuel(
    ev_name: str, ev: dict,
    series: pd.Series,
    fuel_key: str, fuel_color: str,
    fit: dict, baseline: pd.Series,
    ci_low: pd.Series, ci_high: pd.Series,
    extra: pd.Series, gain_meur: float,
    cons: pd.Series,                     # consumo giornaliero (litri)
    ax_main: plt.Axes, ax_gain: plt.Axes,
    break_date: pd.Timestamp, mode: str,
    break_score: float = np.nan,
) -> None:
    shock = ev["shock"]

    win = series[
        (series.index >= shock - pd.Timedelta(days=PRE_WIN)) &
        (series.index <= shock + pd.Timedelta(days=POST_WIN))
    ].dropna()

    ax_main.plot(win.index, win.values, color=fuel_color, lw=1.0,
                 label=f"{fuel_key.capitalize()} effettivo")
    ax_main.plot(baseline.index, baseline.values, color="dimgrey", lw=1.3,
                 ls="--", label=f"Baseline OLS (R²={fit['r2']:.2f})")
    ax_main.fill_between(ci_low.index, ci_low.values, ci_high.values,
                         alpha=0.15, color="grey",
                         label=f"PI {int((1-CI_ALPHA)*100)}% (i.i.d.)")
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
                        label=f"τ rilevato ({break_date.date()}, score={break_score:.2f})")

    mode_str = f"Break=θ {break_date.date()} (GLM Poisson 02c)" \
               if mode == "detected" else f"Break=shock ({shock.date()})"
    ax_main.set_title(
        f"[V1-Naïve / mode={mode}]  {fuel_key.capitalize()} – {ev_name}\n"
        f"{mode_str}  |  Baseline OLS  |  CI i.i.d.",
        fontsize=8, fontweight="bold"
    )
    ax_main.set_ylabel("Margine (€/L)", fontsize=8)
    ax_main.legend(fontsize=6, loc="upper left", ncol=2)
    ax_main.grid(axis="y", alpha=0.20)
    ax_main.xaxis.set_major_formatter(mdates.DateFormatter("%d %b %y"))
    ax_main.xaxis.set_major_locator(mdates.WeekdayLocator(byweekday=0, interval=2))
    plt.setp(ax_main.xaxis.get_majorticklabels(), rotation=35, ha="right", fontsize=7)

    # Calcola cumulato usando i consumi giornalieri
    cum = (extra * cons.values / 1e6).cumsum()
    ax_gain.plot(cum.index, cum.values, color=fuel_color, lw=1.2)
    ax_gain.axhline(0, color="grey", lw=0.7, ls="--")
    ax_gain.fill_between(cum.index, cum.values, 0,
                         where=(cum >= 0), alpha=0.25, color="green")
    ax_gain.fill_between(cum.index, cum.values, 0,
                         where=(cum < 0), alpha=0.25, color="red")
    # Calcola consumo medio per l'annotazione
    avg_cons_ml = cons.mean() / 1e6
    ax_gain.set_title(
        f"Guadagno extra cumulato → {gain_meur:+.0f} M€  "
        f"({len(extra)}gg post-break)\n"
        f"[consumo medio {avg_cons_ml:.1f} ML/giorno]",
        fontsize=7
    )
    ax_gain.set_ylabel("M€ cumulati", fontsize=8)
    ax_gain.grid(axis="y", alpha=0.20)
    ax_gain.xaxis.set_major_formatter(mdates.DateFormatter("%d %b %y"))
    ax_gain.xaxis.set_major_locator(mdates.WeekdayLocator(byweekday=0, interval=2))
    plt.setp(ax_gain.xaxis.get_majorticklabels(), rotation=35, ha="right", fontsize=7)


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    parser = argparse.ArgumentParser(description="V1 Naïve OLS – ITS pipeline")
    parser.add_argument("--mode", choices=["fixed", "detected"], default="fixed",
                        help="fixed = usa shock date hardcodata; "
                             "detected = sliding window naïve autonomo")
    parser.add_argument("--detect", choices=["margin", "price"],
                        default="margin",
                        help="(solo mode=detected) serie su cui fare detection: "
                             "margin = sliding window sul margine [default], "
                             "price  = sliding window sul prezzo pompa netto")
    args, _ = parser.parse_known_args()
    mode          = args.mode
    detect_target = args.detect

    if mode == "detected":
        OUT_DIR = _OUT_BASE / "detected" / detect_target / "v1_naive"
    else:
        OUT_DIR = _OUT_BASE / mode / "v1_naive"
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("═"*70)
    print(f"  02d_v1_naive.py  –  Metodo 1: OLS Naïve  [mode={mode}]")
    if mode == "fixed":
        print("  Break = shock date hardcodata (nessuna detection)")
    else:
        print(f"  Break = θ GLM Poisson da 02c_change_point_detection.py")
        print(f"  Detection su: {'MARGINE distributore' if detect_target == 'margin' else 'PREZZO POMPA NETTO'}")
    print(f"  Finestra: PRE={PRE_WIN}gg / POST={POST_WIN}gg dal break point")
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
            f"[Metodo 1 – Naïve OLS / mode={mode}]  {ev_name}\n{ev['label']}",
            fontsize=11, fontweight="bold"
        )

        for row_idx, (fuel_key, (col_name, fuel_color)) in enumerate(FUELS.items()):
            series = data[col_name].dropna()

            # ── Determina break date ──────────────────────────────────────────
            if mode == "detected":
                # Carica θ canonico (GLM Poisson) prodotto da 02c
                theta = load_theta(ev_name, fuel_key, detect_target,
                                   base_dir=BASE_DIR)
                if theta is not None:
                    break_date   = theta
                    break_method = "glm_poisson_02c"
                    break_score  = np.nan
                else:
                    print(f"  ⚠ [{fuel_key}] θ non trovato in theta_results.csv "
                          f"— uso shock come fallback.")
                    break_date   = shock
                    break_method = "fallback_shock"
                    break_score  = np.nan
            else:
                break_date   = shock
                break_method = "fixed_at_shock"
                break_score  = np.nan

            pre_data = series[
                (series.index >= break_date - pd.Timedelta(days=PRE_WIN)) &
                (series.index < break_date)
            ]
            post_data = series[
                (series.index >= break_date) &
                (series.index < shock + pd.Timedelta(days=POST_WIN))
            ]

            if len(pre_data) < 10 or len(post_data) < 5:
                print(f"  [{fuel_key}] dati insufficienti – salto.")
                for ax in axes[row_idx]:
                    ax.text(0.5, 0.5, "Dati insufficienti",
                            ha="center", va="center", transform=ax.transAxes)
                continue

            fit = _fit_ols(series, break_date)
            if fit is None:
                continue

            baseline, ci_low, ci_high = _project_ols(fit, post_data.index)
            extra     = post_data - baseline
            cons      = load_daily_consumption(post_data.index, fuel_key)
            gain_meur = float((extra * cons).sum() / 1e6)
            gain_ci_low  = float(((post_data - ci_high) * cons).sum() / 1e6)
            gain_ci_high = float(((post_data - ci_low) * cons).sum() / 1e6)

            _plot_event_fuel(
                ev_name, ev, series, fuel_key, fuel_color,
                fit, baseline, ci_low, ci_high, extra, gain_meur, cons,
                axes[row_idx][0], axes[row_idx][1],
                break_date=break_date, mode=mode, break_score=break_score,
            )

            # ── Diagnostics residui pre-periodo ──────────────────────────────
            pre_resid = fit.get("residuals", np.array([]))
            diag = run_diagnostic_tests(
                pre_resid,
                x_for_bg=fit.get("X_bg"),
                n_lags=None,
            )

            safe_ev = ev_name.replace(" ","_").replace("/","").replace("(","").replace(")","")
            diag_plot_out = OUT_DIR / f"diag_{safe_ev}_{fuel_key}.png"
            plot_residual_diagnostics(
                resid=pre_resid,
                dates=fit["pre"].index,
                title=(f"[V1-Naïve] Diagnostica residui OLS pre-periodo\n"
                       f"{ev_name} · {fuel_key.capitalize()}  "
                       f"(break={break_date.date()})"),
                out_path=diag_plot_out,
                diag_stats=diag,
            )

            print(f"\n  {ev_name}  [{fuel_key.upper()}]")
            print(f"    Break ({break_method}) = {break_date.date()}  (shock={shock.date()})")
            if not np.isnan(break_score):
                print(f"    Score naive           = {break_score:.4f}  "
                      f"(Δ={break_date.date()-shock.date()} dal shock)")
            print(f"    OLS R²                = {fit['r2']:.3f}  "
                  f"slope = {fit['slope']:+.5f} €/L/giorno")
            print(f"    Extra medio           = {extra.mean():+.4f} €/L/giorno")
            print(f"    Guadagno totale       = {gain_meur:+.0f} M€  "
                  f"CI90% [{gain_ci_low:+.0f}, {gain_ci_high:+.0f}] M€")
            sw_ok = not np.isnan(diag.get("sw_p", np.nan))
            lb_ok = not np.isnan(diag.get("lb_p", np.nan))
            if sw_ok:
                print(f"    SW (normalità)        = W={diag['sw_stat']:.3f}  "
                      f"p={diag['sw_p']:.3f}  "
                      f"{'OK' if diag['sw_p'] > 0.05 else '⚠ non normal.'}")
            if lb_ok:
                print(f"    LB({diag['n_lags']}) (autocorr.)   = "
                      f"Q={diag['lb_stat']:.2f}  p={diag['lb_p']:.3f}  "
                      f"{'OK' if diag['lb_p'] > 0.05 else '⚠ autocorr.'}")
            bg_ok = not np.isnan(diag.get("bg_p", np.nan))
            if bg_ok:
                print(f"    BG({diag['n_lags']}) (autocorr.)   = "
                      f"LM={diag['bg_stat']:.2f}  p={diag['bg_p']:.3f}  "
                      f"{'OK' if diag['bg_p'] > 0.05 else '⚠ autocorr.'}")

            # ── Export residui pre/post (standard per 02d_compare nonparam) ──
            _safe_ev = (ev_name.replace(" ", "_").replace("/", "")
                               .replace("(", "").replace(")", ""))
            _resid_rows = []
            for _d, _r in zip(fit["pre"].index, fit["residuals"]):
                _resid_rows.append({
                    "date": str(_d.date()), "residual": float(_r), "phase": "pre",
                    "metodo": "v1_naive", "evento": ev_name,
                    "carburante": fuel_key, "break_date": str(break_date.date()),
                })
            for _d, _r in zip(post_data.index, extra.values):
                _resid_rows.append({
                    "date": str(_d.date()), "residual": float(_r), "phase": "post",
                    "metodo": "v1_naive", "evento": ev_name,
                    "carburante": fuel_key, "break_date": str(break_date.date()),
                })
            pd.DataFrame(_resid_rows).to_csv(
                OUT_DIR / f"residuals_{_safe_ev}_{fuel_key}.csv", index=False
            )

            rows.append({
                "metodo":            "v1_naive",
                "mode":              mode,
                "break_method":      break_method,
                "evento":            ev_name,
                "carburante":        fuel_key,
                "shock":             shock.date(),
                "break_date":        break_date.date(),
                "break_score_naive": round(break_score, 4) if not np.isnan(break_score) else np.nan,
                "pre_win_days":      PRE_WIN,
                "post_win_days":     POST_WIN,
                "n_pre":             len(pre_data),
                "n_post":            len(post_data),
                "pre_std_eurl":      round(float(pre_data.std(ddof=1)), 6),
                "extra_mean_eurl":   round(float(extra.mean()), 5),
                "extra_sum_eurl":    round(float(extra.sum()), 4),
                "gain_total_meur":   round(gain_meur, 1),
                "gain_ci_low_meur":  round(gain_ci_low, 1),
                "gain_ci_high_meur": round(gain_ci_high, 1),
                "r2_ols":            round(fit["r2"], 4),
                "slope_eurl_day":    round(fit["slope"], 6),
                "ci_type":           f"OLS_iid_{int((1-CI_ALPHA)*100)}pct",
                # ── Diagnostics ──────────────────────────────────────────────
                "sw_stat":           round(diag.get("sw_stat", np.nan), 4),
                "sw_p":              round(diag.get("sw_p", np.nan), 4),
                "lb_stat":           round(diag.get("lb_stat", np.nan), 3),
                "lb_p":              round(diag.get("lb_p", np.nan), 4),
                "bg_stat":           round(diag.get("bg_stat", np.nan), 3),
                "bg_p":              round(diag.get("bg_p", np.nan), 4),
                "diag_n_lags":       diag.get("n_lags", np.nan),
                "note":              f"OLS naïve, residui i.i.d., mode={mode}",
            })

        fig.tight_layout()
        safe = ev_name.replace(" ", "_").replace("/", "").replace("(", "").replace(")", "")
        out  = OUT_DIR / f"plot_{safe}.png"
        fig.savefig(out, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"\n  → Salvato: {out}")

    if rows:
        df = pd.DataFrame(rows)
        csv_out = OUT_DIR / "v1_naive_results.csv"
        df.to_csv(csv_out, index=False)
        print(f"\n  → CSV: {csv_out}")
        print("\n" + df[["evento","carburante","break_date","gain_total_meur"]].to_string(index=False))
    else:
        print("\n  ⚠ Nessun risultato prodotto.")


if __name__ == "__main__":
    main()