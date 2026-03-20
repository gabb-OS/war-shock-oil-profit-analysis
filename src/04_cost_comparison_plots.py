#!/usr/bin/env python3
"""
04_cost_comparison_plots.py
═══════════════════════════
Genera due famiglie di grafici per ogni combinazione
(modello × evento × carburante):

  FIGURA A — "Per modello" (style originale esteso)
    Panel 1 : Margine €/L  —  effettivo vs baseline controfattuale
    Panel 2 : Prezzo pompa netto €/L  —  effettivo vs controfattuale
    Panel 3 : Guadagno extra cumulato M€ con CI

  FIGURA B — "Costo reale vs controfattuale"  (NUOVO)
    Panel 1 : Prezzo pompa netto €/L  —  effettivo vs controfattuale (area)
    Panel 2 : Consumo giornaliero ML/giorno (barre)
    Panel 3 : Costo giornaliero M€  —  effettivo vs controfattuale (area)
    Panel 4 : Risparmio/sovra-costo cumulato M€

Uso:
    python3 04_cost_comparison_plots.py                   # tutti i modelli
    python3 04_cost_comparison_plots.py --models v1 v7    # solo v1 e v7
    python3 04_cost_comparison_plots.py --mode fixed      # solo modalità fixed
    python3 04_cost_comparison_plots.py --fig b           # solo Figura B
    python3 04_cost_comparison_plots.py --events ucraina  # solo Ucraina

Output:
    data/plots/cost_comparison/
        {fig_a|fig_b}_{mode}/{modello}/{evento}_{carburante}.png
"""

from __future__ import annotations
import argparse
import sys
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR      = Path(__file__).parent
DAILY_CSV     = BASE_DIR / "data" / "processed" / "daily_fuel_prices_all.csv"
ITS_BASE      = BASE_DIR / "data" / "plots" / "its"
OUT_BASE      = BASE_DIR / "data" / "plots" / "cost_comparison"

# ── Costanti consumo giornaliero [litri] — fallback se forecast_consumi
# non è disponibile (media annua italiana approssimata da MISE/ENEA)
CONS_FALLBACK = {
    "benzina": 26_850_000,   # ~26.85 ML/giorno
    "gasolio": 78_000_000,   # ~78 ML/giorno (diesel trasporti + auto)
}

EVENTS = {
    "Ucraina (Feb 2022)":    {"shock": pd.Timestamp("2022-02-24"), "color": "#c0392b", "short": "ucraina"},
    "Iran-Israele (Giu 2025)": {"shock": pd.Timestamp("2025-06-13"), "color": "#e67e22", "short": "iran"},
    "Hormuz (Feb 2026)":     {"shock": pd.Timestamp("2026-02-28"), "color": "#8e44ad", "short": "hormuz"},
}

MODELS = {
    "v1_naive":       {"label": "OLS Naïve",         "color": "#2980b9"},
    "v3_arima":       {"label": "ARIMA/ETS",          "color": "#27ae60"},
    "v5_causalimpact":{"label": "BSTS CausalImpact",  "color": "#8e44ad"},
    "v7_theilsen":    {"label": "Theil-Sen",          "color": "#d35400"},
    "v8_pymc":        {"label": "PyMC Bayesiano",     "color": "#16a085"},
}

FUELS = {
    "benzina": {"col": "benzina_net", "label": "Benzina",
                "color": "#e74c3c", "cons_color": "#f39c12"},
    "gasolio": {"col": "gasolio_net", "label": "Gasolio",
                "color": "#2c3e50", "cons_color": "#7f8c8d"},
}

PRE_WIN  = 40
POST_WIN = 40

# ── Stile globale ──────────────────────────────────────────────────────────────
plt.rcParams.update({
    "figure.facecolor":  "white",
    "axes.facecolor":    "white",
    "axes.grid":         True,
    "grid.alpha":        0.18,
    "grid.linestyle":    "--",
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "font.size":         8,
    "axes.titlesize":    8,
    "axes.labelsize":    8,
    "xtick.labelsize":   7,
    "ytick.labelsize":   7,
    "legend.fontsize":   6.5,
})

DATE_FMT   = mdates.DateFormatter("%d %b '%y")
DATE_LOC   = mdates.WeekdayLocator(byweekday=0, interval=2)


# ══════════════════════════════════════════════════════════════════════════════
# Caricamento dati
# ══════════════════════════════════════════════════════════════════════════════

def load_price_series() -> pd.DataFrame | None:
    """Carica prezzi pompa netti da daily_fuel_prices_all.csv."""
    if not DAILY_CSV.exists():
        print(f"  ⚠  {DAILY_CSV} non trovato — skip pannello prezzi.")
        return None
    df = (pd.read_csv(DAILY_CSV, parse_dates=["date"])
            .sort_values("date").set_index("date"))
    return df


def load_consumption(index: pd.DatetimeIndex, fuel_key: str) -> pd.Series:
    """
    Prova a usare load_daily_consumption dal modulo utils.
    Se non disponibile usa il fallback costante.
    """
    try:
        sys.path.insert(0, str(BASE_DIR / "utils"))
        from forecast_consumi import load_daily_consumption  # type: ignore
        return load_daily_consumption(index, fuel_key)
    except Exception:
        val = CONS_FALLBACK.get(fuel_key, 30_000_000)
        return pd.Series(val, index=index, name="consumption")


def load_residuals(model: str, mode: str, ev_name: str,
                   fuel_key: str, detect_target: str = "fixed") -> pd.DataFrame | None:
    """
    Cerca il CSV dei residuals nel path standard del modello.
    Prova prima fixed, poi detected/margin, poi detected/price.
    """
    safe_ev = (ev_name.replace(" ", "_").replace("/", "")
                      .replace("(", "").replace(")", ""))

    candidates = [
        ITS_BASE / mode / model / f"residuals_{safe_ev}_{fuel_key}.csv",
    ]
    if mode == "detected":
        candidates += [
            ITS_BASE / "detected" / detect_target / model / f"residuals_{safe_ev}_{fuel_key}.csv",
        ]

    for path in candidates:
        if path.exists():
            df = pd.read_csv(path, parse_dates=["date"])
            return df

    return None


def get_window(ev_name: str, price_df: pd.DataFrame | None,
               fuel_key: str) -> tuple[pd.Series | None, pd.Timestamp, pd.Timestamp]:
    """Estrae la finestra pre+post del prezzo pompa."""
    shock = EVENTS[ev_name]["shock"]
    if price_df is None:
        return None, shock, shock

    col = FUELS[fuel_key]["col"]
    if col not in price_df.columns:
        return None, shock, shock

    series = price_df[col].dropna()
    # date di trading intorno allo shock
    pre_start  = series.loc[:shock].index[-PRE_WIN]  if len(series.loc[:shock]) >= PRE_WIN  else series.index[0]
    post_end_candidates = series.loc[shock:].index
    post_end   = post_end_candidates[POST_WIN - 1] if len(post_end_candidates) >= POST_WIN else post_end_candidates[-1]

    window = series.loc[pre_start:post_end]
    return window, pre_start, post_end


# ══════════════════════════════════════════════════════════════════════════════
# FIGURA A — Margine + Prezzo + Cumulato (per modello)
# ══════════════════════════════════════════════════════════════════════════════

def plot_fig_a(model: str, mode: str, ev_name: str, fuel_key: str,
               price_df: pd.DataFrame | None, out_dir: Path) -> None:

    resid_df = load_residuals(model, mode, ev_name, fuel_key)
    if resid_df is None:
        print(f"    ⚠  residuals non trovati: {model}/{mode}/{ev_name}/{fuel_key}")
        return

    shock   = EVENTS[ev_name]["shock"]
    ev_col  = EVENTS[ev_name]["color"]
    m_cfg   = MODELS[model]
    f_cfg   = FUELS[fuel_key]

    pre_r  = resid_df[resid_df["phase"] == "pre"].set_index("date").sort_index()
    post_r = resid_df[resid_df["phase"] == "post"].set_index("date").sort_index()

    pre_r.index  = pd.to_datetime(pre_r.index)
    post_r.index = pd.to_datetime(post_r.index)

    cons = load_consumption(post_r.index, fuel_key)

    # prezzo effettivo vs controfattuale
    price_window, _, _ = get_window(ev_name, price_df, fuel_key)

    fig, axes = plt.subplots(3, 1, figsize=(11, 9),
                             gridspec_kw={"height_ratios": [2, 2, 1.5]})
    fig.suptitle(
        f"[{m_cfg['label']}  /  mode={mode}]   "
        f"{f_cfg['label'].upper()} — {ev_name}",
        fontsize=9, fontweight="bold", y=0.98,
    )

    # ── Panel 1: Margine ──────────────────────────────────────────────────────
    ax1 = axes[0]
    all_r  = pd.concat([pre_r["residual"], post_r["residual"]])
    # baseline (residui pre = scarto pre, non useful here; plot 0 line)
    ax1.axhline(0, color="grey", lw=0.8, ls="--", zorder=1)
    ax1.plot(pre_r.index,  pre_r["residual"],  color="grey",      lw=1.0,
             label="Residui pre-break")
    ax1.fill_between(post_r.index, post_r["residual"], 0,
                     where=(post_r["residual"] >= 0),
                     color=m_cfg["color"], alpha=0.35, label="Extra (>0)")
    ax1.fill_between(post_r.index, post_r["residual"], 0,
                     where=(post_r["residual"] < 0),
                     color="#e74c3c", alpha=0.30, label="Sotto-baseline (<0)")
    ax1.plot(post_r.index, post_r["residual"], color=m_cfg["color"], lw=1.2)
    ax1.axvline(shock, color=ev_col, lw=1.4, ls="--",
                label=f"Shock ({shock.date()})")
    ax1.set_ylabel("Extra margine (€/L)")
    ax1.set_title("Margine distributore: effettivo − baseline controfattuale")
    ax1.legend(ncol=4, loc="upper left")
    ax1.xaxis.set_major_formatter(DATE_FMT)
    ax1.xaxis.set_major_locator(DATE_LOC)
    plt.setp(ax1.xaxis.get_majorticklabels(), rotation=30, ha="right")

    # ── Panel 2: Prezzo pompa effettivo vs controfattuale ─────────────────────
    ax2 = axes[1]
    if price_window is not None:
        p_actual = price_window
        # prezzo controfattuale = prezzo effettivo − residuo (solo post)
        p_cf_post = p_actual.reindex(post_r.index) - post_r["residual"]

        ax2.plot(p_actual.index, p_actual.values, color=f_cfg["color"],
                 lw=1.4, label="Prezzo effettivo (€/L)")
        ax2.plot(p_cf_post.index, p_cf_post.values, color="grey",
                 lw=1.2, ls="--", label="Baseline controfattuale (€/L)")
        ax2.fill_between(p_cf_post.index,
                         p_actual.reindex(p_cf_post.index).values,
                         p_cf_post.values,
                         color=m_cfg["color"], alpha=0.20,
                         label="Differenziale prezzo")
        ax2.axvline(shock, color=ev_col, lw=1.4, ls="--")
    else:
        ax2.text(0.5, 0.5, "daily_fuel_prices_all.csv non disponibile",
                 ha="center", va="center", transform=ax2.transAxes,
                 color="grey", fontsize=8)
    ax2.set_ylabel("Prezzo pompa netto (€/L)")
    ax2.set_title("Prezzo effettivo vs baseline controfattuale")
    ax2.legend(ncol=3, loc="upper left")
    ax2.xaxis.set_major_formatter(DATE_FMT)
    ax2.xaxis.set_major_locator(DATE_LOC)
    plt.setp(ax2.xaxis.get_majorticklabels(), rotation=30, ha="right")

    # ── Panel 3: Cumulato M€ ──────────────────────────────────────────────────
    ax3 = axes[2]
    extra  = post_r["residual"].values
    cum    = (extra * cons.values / 1e6).cumsum()
    cum_s  = pd.Series(cum, index=post_r.index)
    ax3.plot(cum_s.index, cum_s.values, color=m_cfg["color"], lw=1.4)
    ax3.fill_between(cum_s.index, cum_s.values, 0,
                     where=(cum_s >= 0), color=m_cfg["color"], alpha=0.25)
    ax3.fill_between(cum_s.index, cum_s.values, 0,
                     where=(cum_s < 0), color="#e74c3c", alpha=0.25)
    ax3.axhline(0, color="grey", lw=0.8, ls="--")
    ax3.set_ylabel("M€ cumulati")
    ax3.set_title(f"Guadagno extra cumulato → {cum[-1]:+.1f} M€")
    ax3.xaxis.set_major_formatter(DATE_FMT)
    ax3.xaxis.set_major_locator(DATE_LOC)
    plt.setp(ax3.xaxis.get_majorticklabels(), rotation=30, ha="right")

    fig.tight_layout(rect=[0, 0, 1, 0.97])
    safe_ev = ev_name.replace(" ", "_").replace("/", "").replace("(", "").replace(")", "")
    fname = out_dir / f"{safe_ev}_{fuel_key}.png"
    fig.savefig(fname, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"    ✓  {fname.relative_to(BASE_DIR)}")


# ══════════════════════════════════════════════════════════════════════════════
# FIGURA B — Prezzo + Consumo + Costo reale vs controfattuale  [NUOVO]
# ══════════════════════════════════════════════════════════════════════════════

def plot_fig_b(model: str, mode: str, ev_name: str, fuel_key: str,
               price_df: pd.DataFrame | None, out_dir: Path) -> None:

    resid_df = load_residuals(model, mode, ev_name, fuel_key)
    if resid_df is None:
        print(f"    ⚠  residuals non trovati: {model}/{mode}/{ev_name}/{fuel_key}")
        return

    shock   = EVENTS[ev_name]["shock"]
    ev_col  = EVENTS[ev_name]["color"]
    m_cfg   = MODELS[model]
    f_cfg   = FUELS[fuel_key]

    post_r = (resid_df[resid_df["phase"] == "post"]
              .set_index("date").sort_index())
    post_r.index = pd.to_datetime(post_r.index)

    cons       = load_consumption(post_r.index, fuel_key)
    cons_ml    = cons / 1e6           # in ML/giorno per il grafico

    price_window, _, _ = get_window(ev_name, price_df, fuel_key)

    fig, axes = plt.subplots(4, 1, figsize=(11, 12),
                             gridspec_kw={"height_ratios": [2, 1.5, 2, 2]})
    fig.suptitle(
        f"[{m_cfg['label']}  /  mode={mode}]   "
        f"{f_cfg['label'].upper()} — {ev_name}\n"
        f"Costo reale vs controfattuale · finestra {PRE_WIN}+{POST_WIN} giorni",
        fontsize=9, fontweight="bold", y=0.99,
    )

    # ── Panel 1: Prezzo pompa €/L ─────────────────────────────────────────────
    ax1 = axes[0]
    if price_window is not None:
        p_actual  = price_window
        p_cf_post = p_actual.reindex(post_r.index) - post_r["residual"].values

        # sfondo pre-break
        ax1.axvspan(price_window.index[0], shock,
                    color="lightgrey", alpha=0.12, zorder=0)
        ax1.plot(p_actual.index, p_actual.values,
                 color=f_cfg["color"], lw=1.6, label="Prezzo effettivo")
        ax1.plot(p_cf_post.index, p_cf_post.values,
                 color="grey", lw=1.3, ls="--", label="Baseline controfattuale")
        ax1.fill_between(p_cf_post.index,
                         p_actual.reindex(p_cf_post.index).values,
                         p_cf_post.values,
                         color=m_cfg["color"], alpha=0.18,
                         label="Differenziale")
        ax1.axvline(shock, color=ev_col, lw=1.4, ls="--",
                    label=f"Shock ({shock.date()})")
    else:
        ax1.text(0.5, 0.5, "Prezzi non disponibili (daily_fuel_prices_all.csv mancante)",
                 ha="center", va="center", transform=ax1.transAxes,
                 color="grey", fontsize=8)

    ax1.set_ylabel("€/L")
    ax1.set_title("Prezzo pompa netto (€/L) — effettivo vs controfattuale")
    ax1.legend(ncol=4, loc="upper left", framealpha=0.8)
    ax1.xaxis.set_major_formatter(DATE_FMT)
    ax1.xaxis.set_major_locator(DATE_LOC)
    plt.setp(ax1.xaxis.get_majorticklabels(), rotation=30, ha="right")

    # ── Panel 2: Consumo giornaliero ML/giorno ────────────────────────────────
    ax2 = axes[1]
    bar_colors = [f_cfg["cons_color"]] * len(post_r)
    ax2.bar(post_r.index, cons_ml.values, width=0.8,
            color=bar_colors, alpha=0.75, label="Consumo giorn. (ML/g)")
    ax2.set_ylabel("ML / giorno")
    ax2.set_title("Consumo giornaliero stimato")
    mean_cons = cons_ml.mean()
    ax2.axhline(mean_cons, color="grey", lw=0.9, ls="--",
                label=f"Media {mean_cons:.1f} ML/g")
    ax2.legend(ncol=2, loc="upper right")
    ax2.xaxis.set_major_formatter(DATE_FMT)
    ax2.xaxis.set_major_locator(DATE_LOC)
    plt.setp(ax2.xaxis.get_majorticklabels(), rotation=30, ha="right")

    # ── Panel 3: Costo giornaliero M€ — effettivo vs controfattuale ───────────
    ax3 = axes[2]
    if price_window is not None:
        p_actual_post = price_window.reindex(post_r.index)
        p_cf_post     = p_actual_post - post_r["residual"].values

        cost_actual = p_actual_post * cons / 1e6   # M€/giorno
        cost_cf     = p_cf_post     * cons / 1e6   # M€/giorno

        ax3.plot(post_r.index, cost_actual.values,
                 color=f_cfg["color"], lw=1.5, label="Costo effettivo (M€/g)")
        ax3.plot(post_r.index, cost_cf.values,
                 color="grey", lw=1.3, ls="--",
                 label="Costo controfattuale (M€/g)")
        ax3.fill_between(post_r.index,
                         cost_actual.values, cost_cf.values,
                         where=(cost_actual.values >= cost_cf.values),
                         color=m_cfg["color"], alpha=0.25,
                         label="Sovra-costo")
        ax3.fill_between(post_r.index,
                         cost_actual.values, cost_cf.values,
                         where=(cost_actual.values < cost_cf.values),
                         color="#27ae60", alpha=0.25,
                         label="Risparmio")
    else:
        # senza prezzi assoluti uso solo il differenziale dal residuo
        daily_extra = post_r["residual"].values * cons.values / 1e6
        pos = daily_extra.copy(); pos[pos < 0] = 0
        neg = daily_extra.copy(); neg[neg > 0] = 0
        ax3.bar(post_r.index, pos, width=0.8,
                color=m_cfg["color"], alpha=0.70, label="Sovra-costo (M€/g)")
        ax3.bar(post_r.index, neg, width=0.8,
                color="#27ae60", alpha=0.70, label="Risparmio (M€/g)")
        ax3.axhline(0, color="grey", lw=0.8)
        ax3.set_title("Differenziale costo giornaliero (senza prezzi assoluti)")

    ax3.set_ylabel("M€ / giorno")
    ax3.set_title("Costo giornaliero effettivo vs controfattuale")
    ax3.legend(ncol=4, loc="upper left", framealpha=0.8)
    ax3.xaxis.set_major_formatter(DATE_FMT)
    ax3.xaxis.set_major_locator(DATE_LOC)
    plt.setp(ax3.xaxis.get_majorticklabels(), rotation=30, ha="right")

    # ── Panel 4: Sovra-costo/risparmio cumulato M€ ────────────────────────────
    ax4 = axes[3]
    extra_daily = post_r["residual"].values * cons.values / 1e6
    cum = np.cumsum(extra_daily)
    cum_s = pd.Series(cum, index=post_r.index)

    ax4.plot(cum_s.index, cum_s.values, color=m_cfg["color"],
             lw=1.8, label=f"Cumulato → {cum[-1]:+.1f} M€")
    ax4.fill_between(cum_s.index, cum_s.values, 0,
                     where=(cum_s >= 0), color=m_cfg["color"], alpha=0.22,
                     label="Consumatori pagano di più")
    ax4.fill_between(cum_s.index, cum_s.values, 0,
                     where=(cum_s < 0), color="#27ae60", alpha=0.22,
                     label="Consumatori risparmiano")
    ax4.axhline(0, color="grey", lw=0.8, ls="--")

    # annotazione finale
    ax4.annotate(f"{cum[-1]:+.1f} M€",
                 xy=(cum_s.index[-1], cum[-1]),
                 xytext=(-40, 8 if cum[-1] >= 0 else -16),
                 textcoords="offset points",
                 fontsize=8, fontweight="bold",
                 color=m_cfg["color"],
                 arrowprops=dict(arrowstyle="->", color=m_cfg["color"],
                                 lw=0.8))

    ax4.set_ylabel("M€ cumulati")
    ax4.set_title("Sovra-costo (o risparmio) cumulato per i consumatori")
    ax4.legend(ncol=3, loc="upper left", framealpha=0.8)
    ax4.xaxis.set_major_formatter(DATE_FMT)
    ax4.xaxis.set_major_locator(DATE_LOC)
    plt.setp(ax4.xaxis.get_majorticklabels(), rotation=30, ha="right")

    fig.tight_layout(rect=[0, 0, 1, 0.97])
    safe_ev = ev_name.replace(" ", "_").replace("/", "").replace("(", "").replace(")", "")
    fname = out_dir / f"{safe_ev}_{fuel_key}.png"
    fig.savefig(fname, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"    ✓  {fname.relative_to(BASE_DIR)}")


# ══════════════════════════════════════════════════════════════════════════════
# FIGURA C — Confronto tutti i modelli sullo stesso grafico (BONUS)
# ══════════════════════════════════════════════════════════════════════════════

def plot_fig_c_compare(mode: str, ev_name: str, fuel_key: str,
                       price_df: pd.DataFrame | None, out_dir: Path) -> None:
    """Un unico pannello con i cumulati di tutti i modelli sovrapposti."""
    shock  = EVENTS[ev_name]["shock"]
    ev_col = EVENTS[ev_name]["color"]
    f_cfg  = FUELS[fuel_key]

    fig, axes = plt.subplots(2, 1, figsize=(11, 7),
                             gridspec_kw={"height_ratios": [1.8, 2]})
    fig.suptitle(
        f"Confronto modelli — {f_cfg['label'].upper()} · {ev_name}\n"
        f"Prezzo effettivo vs controfattuale  &  Sovra-costo cumulato",
        fontsize=9, fontweight="bold", y=0.99,
    )

    ax1, ax2 = axes
    price_plotted = False

    for model, m_cfg in MODELS.items():
        resid_df = load_residuals(model, mode, ev_name, fuel_key)
        if resid_df is None:
            continue

        post_r = (resid_df[resid_df["phase"] == "post"]
                  .set_index("date").sort_index())
        post_r.index = pd.to_datetime(post_r.index)
        cons = load_consumption(post_r.index, fuel_key)

        # Panel 1: differenziale prezzo (solo se non già tracciato il reale)
        if price_df is not None and not price_plotted:
            p_actual, _, _ = get_window(ev_name, price_df, fuel_key)
            if p_actual is not None:
                ax1.plot(p_actual.index, p_actual.values,
                         color=f_cfg["color"], lw=1.8, zorder=5,
                         label="Prezzo effettivo")
                price_plotted = True

        if price_df is not None:
            p_actual, _, _ = get_window(ev_name, price_df, fuel_key)
            if p_actual is not None:
                p_cf = p_actual.reindex(post_r.index) - post_r["residual"].values
                ax1.plot(p_cf.index, p_cf.values,
                         color=m_cfg["color"], lw=1.0, ls="--", alpha=0.80,
                         label=f"CF {m_cfg['label']}")

        # Panel 2: cumulato
        extra_daily = post_r["residual"].values * cons.values / 1e6
        cum = np.cumsum(extra_daily)
        cum_s = pd.Series(cum, index=post_r.index)
        ax2.plot(cum_s.index, cum_s.values,
                 color=m_cfg["color"], lw=1.5,
                 label=f"{m_cfg['label']}  ({cum[-1]:+.0f} M€)")

    ax1.axvline(shock, color=ev_col, lw=1.3, ls="--",
                label=f"Shock ({shock.date()})")
    ax1.set_ylabel("€/L")
    ax1.set_title("Prezzo pompa netto — effettivo e baseline controfattuale per modello")
    ax1.legend(ncol=3, loc="upper left", fontsize=6)
    ax1.xaxis.set_major_formatter(DATE_FMT)
    ax1.xaxis.set_major_locator(DATE_LOC)
    plt.setp(ax1.xaxis.get_majorticklabels(), rotation=30, ha="right")

    ax2.axhline(0, color="grey", lw=0.8, ls="--")
    ax2.set_ylabel("M€ cumulati")
    ax2.set_title("Sovra-costo cumulato per i consumatori — confronto tra modelli")
    ax2.legend(ncol=3, loc="upper left", fontsize=6.5)
    ax2.xaxis.set_major_formatter(DATE_FMT)
    ax2.xaxis.set_major_locator(DATE_LOC)
    plt.setp(ax2.xaxis.get_majorticklabels(), rotation=30, ha="right")

    fig.tight_layout(rect=[0, 0, 1, 0.96])
    safe_ev = ev_name.replace(" ", "_").replace("/", "").replace("(", "").replace(")", "")
    fname = out_dir / f"{safe_ev}_{fuel_key}.png"
    fig.savefig(fname, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"    ✓  {fname.relative_to(BASE_DIR)}")


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Cost comparison plots")
    p.add_argument("--models",  nargs="+", default=list(MODELS.keys()),
                   choices=list(MODELS.keys()),
                   help="Modelli da includere (default: tutti)")
    p.add_argument("--mode",    default="fixed",
                   choices=["fixed", "detected"],
                   help="Modalità break (default: fixed)")
    p.add_argument("--fig",     nargs="+", default=["a", "b", "c"],
                   choices=["a", "b", "c"],
                   help="Figure da generare: a=margine+prezzo+cum, b=costo_reale, c=compare (default: tutte)")
    p.add_argument("--events",  nargs="+", default=None,
                   help="Filtro eventi per short-name: ucraina, iran, hormuz")
    p.add_argument("--fuels",   nargs="+", default=list(FUELS.keys()),
                   choices=list(FUELS.keys()),
                   help="Carburanti (default: benzina gasolio)")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    price_df = load_price_series()

    # Filtra eventi
    events_to_run = {
        k: v for k, v in EVENTS.items()
        if args.events is None or v["short"] in args.events
    }

    total = 0
    for ev_name in events_to_run:
        for fuel_key in args.fuels:
            # ── Fig A: per ogni modello ──────────────────────────────────────
            if "a" in args.fig:
                for model in args.models:
                    out_dir = OUT_BASE / f"fig_a_{args.mode}" / model
                    out_dir.mkdir(parents=True, exist_ok=True)
                    print(f"  [Fig A] {model} | {ev_name} | {fuel_key}")
                    plot_fig_a(model, args.mode, ev_name, fuel_key,
                               price_df, out_dir)
                    total += 1

            # ── Fig B: per ogni modello ──────────────────────────────────────
            if "b" in args.fig:
                for model in args.models:
                    out_dir = OUT_BASE / f"fig_b_{args.mode}" / model
                    out_dir.mkdir(parents=True, exist_ok=True)
                    print(f"  [Fig B] {model} | {ev_name} | {fuel_key}")
                    plot_fig_b(model, args.mode, ev_name, fuel_key,
                               price_df, out_dir)
                    total += 1

            # ── Fig C: confronto tutti i modelli ─────────────────────────────
            if "c" in args.fig:
                out_dir = OUT_BASE / f"fig_c_{args.mode}"
                out_dir.mkdir(parents=True, exist_ok=True)
                print(f"  [Fig C] all models | {ev_name} | {fuel_key}")
                plot_fig_c_compare(args.mode, ev_name, fuel_key,
                                   price_df, out_dir)
                total += 1

    print(f"\n  ✓  Generati {total} grafici in {OUT_BASE}")


if __name__ == "__main__":
    main()