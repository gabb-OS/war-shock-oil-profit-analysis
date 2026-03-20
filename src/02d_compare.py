#!/usr/bin/env python3
"""
02d_compare.py  ─  Confronto dei 4 Metodi ITS
===============================================
Legge i CSV di output prodotti da v1, v3, v5, v7 e crea:

  1. Tabella comparativa guadagni extra (M€) per evento × carburante × metodo
  2. Barplot gruppi: un gruppo per evento, barre per metodo (benzina + gasolio)
  3. Scatter plot: coppie v1/v3/v5/v7 → misura di accordo
  4. Heatmap: accordo tra metodi per ogni combinazione evento+carburante

Metodi attivi: v1 (Naïve OLS) · v3 (ARIMA) · v5 (BSTS CausalImpact) · v7 (Theil-Sen Bootstrap)
Palette pastello coerente: blu=#7EC8E3 · verde=#90D4A0 · lavanda=#C9A8E0 · arancio=#FFBC80

Modalità (--mode):
  fixed     : legge da data/plots/its/fixed/{metodo}/         [default]
  detected  : legge da data/plots/its/detected/{detect}/{metodo}/

Parametro --detect (solo quando --mode detected):
  margin  : legge la variante detection-su-margine  [default]
  price   : legge la variante detection-su-prezzo

Output:
  data/plots/its/{mode}/compare/              (se mode=fixed)
  data/plots/its/detected/{detect}/compare/   (se mode=detected)
    compare_table.csv
    compare_barplot.png
    compare_scatter.png
    compare_heatmap.png
"""

from __future__ import annotations
from pathlib import Path
import argparse
import sys

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent / "utils"))
try:
    from forecast_consumi import load_daily_consumption as _load_cons
    _HAS_FORECAST_CONSUMI = True
except ImportError:
    _HAS_FORECAST_CONSUMI = False

try:
    from nonparametric_tests import nonparam_h0_battery, print_battery_results
    _HAS_NONPARAM = True
except ImportError:
    _HAS_NONPARAM = False

# ── Configurazione ─────────────────────────────────────────────────────────────
BASE_DIR  = Path(__file__).parent
_OUT_BASE = BASE_DIR / "data" / "plots" / "its"

# ── Palette pastello a 4 colori (coerente in tutti i grafici) ─────────────────
# v1 = blu pastello  |  v3 = verde pastello
# v5 = lavanda       |  v7 = arancio pastello
COLORS = {
    "v1_naive":        "#96b0fd",   # blu pastello
    "v3_arima":        "#FDFD96",   # giallo pastello
    "v7_theilsen":     "#D32E2E",   
    "v8_pymc": "#fd9696",   # 
}

LABELS = {
    "v1_naive":        "V1 – Naïve OLS",
    "v3_arima":        "V3 – ARIMA/ITS",
    "v7_theilsen":     "V7 – Theil-Sen Bootstrap",
    "v8_pymc":         "V4 – PYMC",

}

FUEL_PATTERNS = {"benzina": "/", "gasolio": ""}

# ── Tolleranza CI per il test H0 ──────────────────────────────────────────────
# Utilizziamo esattamente 2σ come soglia.  Un guadagno è "anomalo" (H0 rigettata)
# se e solo se il limite INFERIORE del CI ±2σ è > 0, cioè l'extra-profitto è
# statisticamente distinguibile da zero anche nella stima conservativa.
CI_SIGMA_H0 = 2.0   # deve coincidere con CI_SIGMA nei metodi ITS

# ── Consumo giornaliero di fallback (usato solo se forecast_consumi non è disponibile)
DAILY_CONSUMPTION_L_FALLBACK: dict[str, int] = {
    "benzina": 12_000_000,
    "gasolio": 25_000_000,
}
POST_WIN_DEFAULT = 40   # giorni di default se n_post manca dal CSV

# Shock dates per fallback (corrispondono a quelli nei metodi ITS)
EVENTS_SHOCKS: dict[str, pd.Timestamp] = {
    "Ucraina (Feb 2022)":      pd.Timestamp("2022-02-24"),
    "Iran-Israele (Giu 2025)": pd.Timestamp("2025-06-13"),
    "Hormuz (Feb 2026)":       pd.Timestamp("2026-02-28"),
}

# Colori banda volatilità (uno per carburante)
VOL_COLORS: dict[str, str] = {
    "benzina": "#E67E22",   # arancio
    "gasolio": "#2471A3",   # blu
}


# ══════════════════════════════════════════════════════════════════════════════
# Caricamento e normalizzazione risultati
# ══════════════════════════════════════════════════════════════════════════════

def load_results(mode: str, detect_target: str = "margin") -> pd.DataFrame:
    if mode == "detected":
        its_dir = _OUT_BASE / "detected" / detect_target
    else:
        its_dir = _OUT_BASE / mode
    csv_paths = {
        "v1_naive":        its_dir / "v1_naive"        / "v1_naive_results.csv",
        "v3_arima":        its_dir / "v3_arima"        / "v3_arima_results.csv",
        "v7_theilsen":     its_dir / "v7_theilsen"     / "v7_theilsen_results.csv",
        "v8_pymc": its_dir / "v8_pymc" / "v8_pymc_results.csv",
    }

    frames = []
    for method, path in csv_paths.items():
        if not path.exists():
            print(f"  ⚠ {method}: file non trovato ({path}) – salto.")
            continue

        df = pd.read_csv(path)

        if method in ("v3_arima",) and "is_best" in df.columns:
            df = df[df["is_best"].astype(bool)].copy()

        df["metodo"] = method

        # Normalizza colonne method-specific → nomi comuni:
        # v2 usa gain_ols_meur invece di gain_total_meur
        if "gain_total_meur" not in df.columns and "gain_ols_meur" in df.columns:
            df = df.rename(columns={"gain_ols_meur": "gain_total_meur"})
        # v5 usa abs_effect_avg_eurl invece di extra_mean_eurl
        if "extra_mean_eurl" not in df.columns and "abs_effect_avg_eurl" in df.columns:
            df = df.rename(columns={"abs_effect_avg_eurl": "extra_mean_eurl"})

        cols_keep = ["metodo", "evento", "carburante",
                     "gain_total_meur", "gain_ci_low_meur", "gain_ci_high_meur",
                     "extra_mean_eurl",
                     "pre_std_eurl", "n_post"]
        missing = [c for c in cols_keep if c not in df.columns]
        for c in missing:
            df[c] = np.nan
        frames.append(df[cols_keep])

    if not frames:
        print(f"  ✗ Nessun CSV trovato in {its_dir}. Eseguire prima v1, v3, v5, v7 con --mode {mode}"
              + (f" --detect {detect_target}" if mode == "detected" else "") + ".")
        return pd.DataFrame()

    combined = pd.concat(frames, ignore_index=True)
    print(f"  Caricati {len(frames)} metodi: {[f['metodo'].iloc[0] for f in frames]}")
    return combined


# ══════════════════════════════════════════════════════════════════════════════
# Caricamento residui (per test non-parametrici)
# ══════════════════════════════════════════════════════════════════════════════

def load_residuals(mode: str, detect_target: str = "margin") -> pd.DataFrame:
    """
    Carica tutti i CSV residuals_{evento}_{carburante}.csv prodotti dai
    modelli v1, v3, v5, v7 e li concatena in un unico DataFrame.

    Schema atteso per ogni CSV:
      date, residual, phase (pre|post), metodo, evento, carburante, break_date
    """
    if mode == "detected":
        its_dir = _OUT_BASE / "detected" / detect_target
    else:
        its_dir = _OUT_BASE / mode

    METHOD_DIRS = {
        "v1_naive":        its_dir / "v1_naive",
        "v3_arima":        its_dir / "v3_arima",
        "v7_theilsen":     its_dir / "v7_theilsen",
        "v8_pymc":         its_dir / "v8_pymc"
    }

    frames = []
    for method, mdir in METHOD_DIRS.items():
        if not mdir.exists():
            continue
        csv_files = sorted(mdir.glob("residuals_*.csv"))
        if not csv_files:
            print(f"  ⚠ {method}: nessun residuals_*.csv in {mdir}")
            continue
        for csv_path in csv_files:
            try:
                df = pd.read_csv(csv_path, parse_dates=["date"])
                # Normalizza: se metodo non è nel file usa nome directory
                if "metodo" not in df.columns:
                    df["metodo"] = method
                frames.append(df)
            except Exception as e:
                print(f"  ⚠ {csv_path.name}: errore lettura ({e})")

    if not frames:
        print(f"  ⚠ Nessun file residuals_*.csv trovato sotto {its_dir}")
        return pd.DataFrame()

    combined = pd.concat(frames, ignore_index=True)
    print(f"  Residui caricati: {len(combined)} righe "
          f"da {combined['metodo'].nunique()} modelli, "
          f"{combined['evento'].nunique()} eventi, "
          f"{combined['carburante'].nunique()} carburanti")
    return combined


# ══════════════════════════════════════════════════════════════════════════════
# 1. Tabella comparativa
# ══════════════════════════════════════════════════════════════════════════════

def make_comparison_table(df: pd.DataFrame) -> pd.DataFrame:
    pivot = df.pivot_table(
        index=["evento", "carburante"],
        columns="metodo",
        values="gain_total_meur",
        aggfunc="first",
    )
    available = [m for m in ["v1_naive","v3_arima","v5_causalimpact","v7_theilsen"] if m in pivot.columns]
    if len(available) > 1:
        pivot["range_meur"] = pivot[available].max(axis=1) - pivot[available].min(axis=1)
        pivot["mean_meur"]  = pivot[available].mean(axis=1)
        pivot["sign_agree"] = (pivot[available].gt(0).all(axis=1) |
                               pivot[available].lt(0).all(axis=1)).map({True:"✓", False:"✗"})
    pivot = pivot.rename(columns=LABELS)
    return pivot.reset_index()


# ══════════════════════════════════════════════════════════════════════════════
# Soglia di volatilità pre-shock
# ══════════════════════════════════════════════════════════════════════════════

def compute_volatility_thresholds(df: pd.DataFrame) -> dict:
    """
    Calcola la soglia ±2σ di volatilità naturale per ogni (evento, carburante).

    Formula (somma cumulata di variazioni giornaliere i.i.d.):
        threshold_meur = 2 × σ_pre  ×  √n_post  ×  consumo_giornaliero_L  / 1e6

    dove σ_pre è la deviazione standard del margine giornaliero (€/L) nei
    PRE_WIN giorni prima del break point, stimata dal CSV di ogni metodo.

    Il consumo giornaliero viene letto da consumi_giornalieri.csv (via
    forecast_consumi) se disponibile; in alternativa si usa il valore
    hardcodato di fallback.

    Ritorna:
        dict[(evento, carburante)] → {
            "threshold_meur": float,  # soglia ±2σ cumulata (M€)
            "sigma_eurl":     float,  # σ_pre media tra metodi (€/L)
            "n_post":         int,    # giorni post usati
            "cons_source":    str,    # "csv" | "fallback"
        }
    """
    thresholds: dict = {}

    if "pre_std_eurl" not in df.columns or df["pre_std_eurl"].isna().all():
        print("  ⚠  pre_std_eurl assente nei CSV — banda volatilità non disponibile.")
        print("     Assicurati che 02d_v1_naive (o altri metodi) esporti pre_std_eurl.")
        return thresholds

    for (ev, fuel), grp in df.groupby(["evento", "carburante"]):
        valid = grp.dropna(subset=["pre_std_eurl"])
        if valid.empty:
            continue

        sigma_eurl = float(valid["pre_std_eurl"].mean())

        if "n_post" in valid.columns and not valid["n_post"].isna().all():
            n_post = int(valid["n_post"].dropna().median())
        else:
            n_post = POST_WIN_DEFAULT

        # ── Consumo giornaliero da CSV o fallback ──────────────────────────
        cons_source = "fallback"
        if _HAS_FORECAST_CONSUMI:
            try:
                shock_row = EVENTS_SHOCKS.get(ev)
                if shock_row is not None:
                    post_dates = pd.date_range(shock_row, periods=n_post, freq="D")
                    cons_series = _load_cons(post_dates, fuel)
                    consumption  = float(cons_series.mean())
                    cons_source  = "csv"
                else:
                    consumption = DAILY_CONSUMPTION_L_FALLBACK.get(fuel, 10_000_000)
            except Exception:
                consumption = DAILY_CONSUMPTION_L_FALLBACK.get(fuel, 10_000_000)
        else:
            consumption = DAILY_CONSUMPTION_L_FALLBACK.get(fuel, 10_000_000)

        # ±2σ cumulata
        threshold_meur = CI_SIGMA_H0 * sigma_eurl * np.sqrt(n_post) * consumption / 1e6

        thresholds[(ev, fuel)] = {
            "threshold_meur": threshold_meur,
            "sigma_eurl":     sigma_eurl,
            "n_post":         n_post,
            "cons_source":    cons_source,
            "consumption_L":  consumption,
        }
        print(f"    σ_pre [{ev} / {fuel}]:  {sigma_eurl:.5f} €/L  "
              f"→  soglia ±{CI_SIGMA_H0:.0f}σ = ±{threshold_meur:.0f} M€  "
              f"(√{n_post}≈{np.sqrt(n_post):.1f}d  cons={consumption/1e6:.1f}ML/g  [{cons_source}])")

    return thresholds


# ══════════════════════════════════════════════════════════════════════════════
# 2. Barplot comparativo
# ══════════════════════════════════════════════════════════════════════════════

def plot_barplot(df: pd.DataFrame, out_dir: Path, mode: str) -> None:
    """Un PNG per ogni evento."""
    events  = df["evento"].unique()
    methods = df["metodo"].unique()
    fuels   = df["carburante"].unique()
    n_bars  = len(methods) * len(fuels)
    width   = 0.8 / n_bars

    for ev in events:
        df_ev   = df[df["evento"] == ev]
        fig, ax = plt.subplots(figsize=(max(6, n_bars * 1.3), 6))
        fig.suptitle(
            f"Confronto Guadagni Extra Speculativi  [mode={mode}]\n"
            f"{ev}  –  basati su stime controfattuali, ±CI 90%",
            fontsize=11, fontweight="bold",
        )

        x = np.arange(1)
        bar_idx = 0

        for method in methods:
            for fuel in fuels:
                sub = df_ev[(df_ev["metodo"] == method) & (df_ev["carburante"] == fuel)]
                offset = (bar_idx - n_bars / 2 + 0.5) * width
                if not sub.empty:
                    g   = sub["gain_total_meur"].values[0]
                    clo = sub["gain_ci_low_meur"].values[0]
                    chi = sub["gain_ci_high_meur"].values[0]
                    ci_lo = max(0, g - clo) if not np.isnan(clo) else 0
                    ci_hi = max(0, chi - g) if not np.isnan(chi) else 0

                    bar = ax.bar(
                        x + offset, [g], width,
                        label=f"{LABELS.get(method, method)} – {fuel}",
                        color=COLORS.get(method, "grey"),
                        hatch=FUEL_PATTERNS.get(fuel, ""),
                        alpha=0.85,
                        edgecolor="white",
                        linewidth=0.5,
                    )
                    ax.errorbar(x + offset, [g], yerr=[[ci_lo], [ci_hi]],
                                fmt="none", color="black", capsize=3, lw=0.8)
                    if not np.isnan(g):
                        ax.text(
                            bar[0].get_x() + bar[0].get_width() / 2,
                            bar[0].get_height() + (1 if g >= 0 else -4),
                            f"{g:+.0f}",
                            ha="center", va="bottom", fontsize=7,
                        )
                bar_idx += 1

        ax.axhline(0, color="black", lw=0.9)
        ax.set_xticks([])
        ax.set_ylabel("Guadagno extra cumulato (M€)", fontsize=9)

        handles, labels_l = ax.get_legend_handles_labels()
        ax.legend(handles, labels_l,
                  fontsize=7, loc="upper right", ncol=2,
                  title="Metodo – Carburante", title_fontsize=7)

        ax.grid(axis="y", alpha=0.20)
        fig.tight_layout()
        slug = ev.replace(" ", "_").replace("/", "-").replace("(", "").replace(")", "")
        out  = out_dir / f"compare_barplot_{slug}.png"
        fig.savefig(out, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  → Barplot [{ev}]: {out}")


# ══════════════════════════════════════════════════════════════════════════════
# 3. Scatter
# ══════════════════════════════════════════════════════════════════════════════

def plot_scatter(pivot_df: pd.DataFrame, out_dir: Path, mode: str) -> None:
    """Un PNG per ogni evento (+ uno aggregato se ci sono più eventi)."""
    events = pivot_df["evento"].unique() if "evento" in pivot_df.columns else [None]
    for ev in events:
        df_ev = pivot_df[pivot_df["evento"] == ev] if ev is not None else pivot_df
        _plot_scatter_single(df_ev, out_dir, mode, ev)


def _plot_scatter_single(pivot_df: pd.DataFrame, out_dir: Path, mode: str, ev_label) -> None:
    available = [m for m in ["v1_naive","v3_arima","v7_theilsen","v8_pymc"]
                 if m in pivot_df.columns]
    pairs = [(a, b) for i, a in enumerate(available) for b in available[i+1:]]

    if not pairs:
        print("  ⚠ Scatter: meno di 2 metodi, salto.")
        return

    n = len(pairs)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 5))
    if n == 1:
        axes = [axes]
    fig.suptitle(f"Accordo tra Metodi – Guadagni Extra (M€)  [mode={mode}]", fontsize=11)

    markers = {"benzina": "o", "gasolio": "s"}
    fcolors = {"benzina": "#E07B7B", "gasolio": "#5B8DB8"}   # rosso/blu per carburante
    fuels   = pivot_df["carburante"].unique() if "carburante" in pivot_df.columns else []

    for ax, (m1, m2) in zip(axes, pairs):
        for fuel in fuels:
            sub  = pivot_df[pivot_df["carburante"] == fuel]
            xs   = sub[m1].values
            ys   = sub[m2].values
            mask = ~(np.isnan(xs) | np.isnan(ys))
            if not mask.any():
                continue
            ax.scatter(xs[mask], ys[mask],
                       marker=markers.get(fuel, "o"),
                       color=fcolors.get(fuel, "grey"),
                       s=60, alpha=0.8, label=fuel.capitalize(), zorder=5)
            if "evento" in pivot_df.columns:
                for x_, y_, ev in zip(xs[mask], ys[mask], sub["evento"].values[mask]):
                    ax.annotate(ev.split("(")[0].strip()[:10], (x_, y_),
                                fontsize=5, xytext=(3, 3), textcoords="offset points")

        all_v = np.concatenate([pivot_df[m1].dropna().values, pivot_df[m2].dropna().values])
        if len(all_v):
            lo, hi = all_v.min() - 5, all_v.max() + 5
            ax.plot([lo, hi], [lo, hi], "k--", lw=0.8, alpha=0.5, label="y=x")
            ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)

        ax.set_xlabel(LABELS.get(m1, m1) + " (M€)", fontsize=8)
        ax.set_ylabel(LABELS.get(m2, m2) + " (M€)", fontsize=8)
        ax.set_title(f"{LABELS.get(m1,m1)}\nvs\n{LABELS.get(m2,m2)}", fontsize=8)
        ax.legend(fontsize=6)
        ax.grid(alpha=0.20)
        ax.axhline(0, color="grey", lw=0.6, ls=":")
        ax.axvline(0, color="grey", lw=0.6, ls=":")

    fig.tight_layout()
    slug = (ev_label.replace(" ", "_").replace("/", "-").replace("(", "").replace(")", "")
            if ev_label else "all")
    out = out_dir / f"compare_scatter_{slug}.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  → Scatter [{ev_label or 'all'}]: {out}")


# ══════════════════════════════════════════════════════════════════════════════
# 4. Heatmap
# ══════════════════════════════════════════════════════════════════════════════

def plot_heatmap(df: pd.DataFrame, out_dir: Path, mode: str) -> None:
    """Un PNG per ogni evento."""
    for ev in df["evento"].unique():
        _plot_heatmap_single(df[df["evento"] == ev], out_dir, mode, ev)


def _plot_heatmap_single(df: pd.DataFrame, out_dir: Path, mode: str, ev_label: str) -> None:
    available = [m for m in ["v1_naive","v3_arima","v7_theilsen","v8_pymc"]
                 if m in df["metodo"].unique()]

    if len(available) < 2:
        print("  ⚠ Heatmap: meno di 2 metodi, salto.")
        return

    pivot = df[df["metodo"].isin(available)].pivot_table(
        index=["evento", "carburante"], columns="metodo",
        values="gain_total_meur", aggfunc="first",
        dropna=False,
    )

    if pivot.empty or pivot.values.size == 0:
        print("  ⚠ Heatmap: pivot vuoto, salto.")
        return

    data_np = pivot.values.astype(float)
    if np.all(np.isnan(data_np)):
        print("  ⚠ Heatmap: tutti i valori NaN, salto.")
        return

    fig, ax = plt.subplots(figsize=(max(6, len(available)*2), max(4, len(pivot)*0.8)))
    fig.suptitle(f"Guadagni Extra (M€) – Confronto Metodi  [mode={mode}]\n{ev_label}",
                 fontsize=11, fontweight="bold")

    vmax    = np.nanmax(np.abs(data_np)) + 1e-6
    norm    = mcolors.TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)
    im      = ax.imshow(data_np, aspect="auto", cmap=plt.cm.RdYlGn, norm=norm)
    plt.colorbar(im, ax=ax, label="Guadagno (M€)", shrink=0.8)

    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels([LABELS.get(c, c) for c in pivot.columns], fontsize=8)
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels([f"{ev}\n{fuel}" for ev, fuel in pivot.index], fontsize=7)

    for i in range(len(pivot.index)):
        for j in range(len(pivot.columns)):
            v = data_np[i, j]
            if not np.isnan(v):
                ax.text(j, i, f"{v:+.0f}", ha="center", va="center",
                        fontsize=8, fontweight="bold",
                        color="white" if abs(v) > vmax*0.5 else "black")

    fig.tight_layout()
    slug = ev_label.replace(" ", "_").replace("/", "-").replace("(", "").replace(")", "")
    out  = out_dir / f"compare_heatmap_{slug}.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  → Heatmap [{ev_label}]: {out}")


# ══════════════════════════════════════════════════════════════════════════════
# 5. Test H₀ / H₁ — Profitto Anomalo
# ══════════════════════════════════════════════════════════════════════════════

ACTIVE_METHODS = ["v1_naive", "v3_arima", "v7_theilsen","v8_pymc"]

# Ipotesi:
#   H₀ : i distributori NON generano profitti anomali in prossimità di shock
#         geopolitici → gain_ci_low_meur ≤ 0  (il CI a ±2σ include lo zero)
#   H₁ : i distributori GENERANO profitti anomali
#         → gain_ci_low_meur > 0  (il CI a ±2σ esclude lo zero dal basso)
#
# Criterio per metodo : h0_rejected = True  ↔  gain_ci_low_meur > 0
# Verdetto combinato  : maggioranza semplice (≥ ceil(n_metodi/2)) rifiuta H₀


def make_h0_test_table(df: pd.DataFrame,
                       vol_thresholds: dict | None = None) -> pd.DataFrame:
    """
    Produce una tabella con il verdetto H₀/H₁ per ogni
    (evento, carburante, metodo) e un verdetto combinato.

    Criteri (per metodo), in ordine di priorità:
      1. Se la colonna 'h0_rejected' è presente nel CSV del metodo → la usa direttamente
      2. Se vol_thresholds è fornito: h0_rejected = gain_total_meur > vol_threshold  (±2σ pre-shock)
      3. Fallback: h0_rejected = gain_ci_low_meur > 0  (CI modello, molto conservativo)

    Colonne output:
        evento | carburante | v1_naive | v3_arima | v5_causalimpact | v7_theilsen
        | n_methods_available | n_h0_rejected | verdict | gain_mean_meur | gain_range_meur
    """
    rows_out = []
    available_methods = [m for m in ACTIVE_METHODS if m in df["metodo"].unique()]

    for (ev, fuel), grp in df.groupby(["evento", "carburante"]):
        row: dict = {"evento": ev, "carburante": fuel}

        n_available = 0
        n_rejected  = 0

        # Soglia ±2σ pre-shock per questa coppia (evento, carburante)
        vol_thr: float | None = None
        if vol_thresholds:
            info = vol_thresholds.get((ev, fuel))
            if info is not None:
                vol_thr = float(info["threshold_meur"])

        for method in available_methods:
            sub = grp[grp["metodo"] == method]
            if sub.empty:
                row[method]              = np.nan
                row[f"{method}_gain"]    = np.nan
                row[f"{method}_ci_low"]  = np.nan
                row[f"{method}_ci_high"] = np.nan
                continue

            # Prendi la riga con is_best=True (v3) o la prima disponibile
            if "is_best" in sub.columns:
                best = sub[sub["is_best"].astype(bool)]
                s = best.iloc[0] if not best.empty else sub.iloc[0]
            else:
                s = sub.iloc[0]

            gain    = float(s.get("gain_total_meur", np.nan))

            # Criterio H0:
            #   1. h0_rejected dal CSV del metodo (se presente e valido)
            #   2. gain supera soglia ±2σ pre-shock nella sua direzione  ← criterio principale
            #   3. fallback direzionale: gain>0→ci_low>0 | gain<0→ci_high<0
            if "h0_rejected" in s and not pd.isna(s["h0_rejected"]):
                h0_rej = bool(s["h0_rejected"])
            elif vol_thr is not None and not np.isnan(gain):
                # Solo extra-profitti: H0 rigettata ↔ gain > +soglia
                # Gain negativi → sempre non rigettata (non valutiamo perdite anomale)
                h0_rej = bool(gain > vol_thr)
            else:
                ci_lo_s = s.get("gain_ci_low_meur", np.nan)
                # Solo lato positivo: CI inferiore > 0
                h0_rej = bool(ci_lo_s > 0) if not pd.isna(ci_lo_s) else False

            ci_lo_v  = float(s.get("gain_ci_low_meur",  np.nan))
            ci_hi_v  = float(s.get("gain_ci_high_meur", np.nan))

            # Soglia direzionale usata per la decisione (mostrata come "lim" nel print):
            #   Solo lato positivo — sempre +vol_thr o CI inferiore
            if vol_thr is not None and not np.isnan(gain):
                lim_used = vol_thr
            else:
                lim_used = ci_lo_v

            row[method]              = "✔ RIGETTATA" if h0_rej else "✘ NON RIGETTATA"
            row[f"{method}_gain"]    = round(gain,    1) if not np.isnan(gain)    else np.nan
            row[f"{method}_lim"]     = round(lim_used, 1) if not np.isnan(lim_used) else np.nan
            row[f"{method}_ci_low"]  = round(ci_lo_v, 1) if not np.isnan(ci_lo_v) else np.nan
            row[f"{method}_ci_high"] = round(ci_hi_v, 1) if not np.isnan(ci_hi_v) else np.nan

            n_available += 1
            if h0_rej:
                n_rejected += 1

        row["n_methods_available"] = n_available
        row["n_h0_rejected"]       = n_rejected

        import math
        majority = math.ceil(n_available / 2) if n_available > 0 else 999
        if n_available == 0:
            verdict = "INDETERMINATO"
        elif n_rejected >= majority:
            verdict = "H₀ RIGETTATA  (profitto anomalo)"
        else:
            verdict = "H₀ NON RIGETTATA"
        row["verdict"] = verdict

        # Guadagno medio e range tra metodi disponibili
        gains = [row.get(f"{m}_gain", np.nan) for m in available_methods]
        gains_valid = [g for g in gains if not (isinstance(g, float) and np.isnan(g))]
        row["gain_mean_meur"]  = round(float(np.mean(gains_valid)),  1) if gains_valid else np.nan
        row["gain_range_meur"] = round(float(np.max(gains_valid) - np.min(gains_valid)), 1) if len(gains_valid) > 1 else np.nan

        rows_out.append(row)

    return pd.DataFrame(rows_out)


def print_h0_summary(h0_df: pd.DataFrame) -> None:
    """Stampa la tabella H₀/H₁ in formato leggibile a terminale."""
    available_methods = [m for m in ACTIVE_METHODS if m in h0_df.columns]
    SEP = "═" * 100

    print(f"\n{SEP}")
    print("  TEST IPOTESI H₀/H₁ — PROFITTO ANOMALO DEI DISTRIBUTORI  (±2σ CI)")
    print(f"  H₀: aumenti di prezzo coerenti con i costi  |  H₁: profitti anomali")
    print(f"  Criterio: H₀ rigettata per un metodo  ↔  gain > +{CI_SIGMA_H0:.0f}σ  (solo extra-profitti positivi)")
    print(f"  Gain negativi → sempre H₀ non rigettata (non valutiamo perdite anomale)")
    print(f"  (fallback se soglia non disponibile: CI inferiore del modello > 0)")
    print(f"  Formato celle: gain [lim: +soglia]  —  ✔ se gain > lim, ✘ altrimenti")
    print(f"  Verdetto combinato: maggioranza semplice (≥ ceil(n_metodi/2))")
    print(f"{SEP}")

    # Header
    hdr = f"  {'Evento':<28} {'Carb.':<10}"
    for m in available_methods:
        hdr += f"  {LABELS.get(m, m)[:16]:>16}"
    hdr += f"  {'Rifiuti':>7}  {'Verdetto'}"
    print(hdr)
    print("  " + "─" * (len(hdr) - 2))

    prev_ev = None
    for _, r in h0_df.iterrows():
        ev   = str(r.get("evento",     "?"))
        fuel = str(r.get("carburante", "?"))
        if ev != prev_ev:
            if prev_ev is not None:
                print()
            prev_ev = ev

        line = f"  {ev[:27]:<28} {fuel[:9]:<10}"
        n_rej = int(r.get("n_h0_rejected", 0))
        n_tot = int(r.get("n_methods_available", 0))
        for m in available_methods:
            cell   = str(r.get(m, "N/D"))
            flag    = "✔" if "RIGETTATA" in cell and "NON" not in cell else "✘"
            gain    = r.get(f"{m}_gain", np.nan)
            gain_f  = float(gain) if not (isinstance(gain, float) and np.isnan(gain)) else np.nan
            # Limite direzionale salvato in make_h0_test_table: ±vol_thr oppure CI bound
            lim_val = r.get(f"{m}_lim", np.nan)
            gain_str = f"{gain_f:+.0f}" if not np.isnan(gain_f) else "n/d"
            lim_str  = (f"{float(lim_val):+.0f}"
                        if not (isinstance(lim_val, float) and np.isnan(lim_val))
                        else "n/d")
            line += f"  {flag} {gain_str:>6}[lim:{lim_str:>6}]M€  "
        line += f"  {n_rej}/{n_tot}  "
        v = str(r.get("verdict", "?"))
        icon = "🔴" if "NON RIGETTATA" in v else ("🟢" if "RIGETTATA" in v else "⚪")
        line += f"{icon}  {v}"
        print(line)

    print(f"{SEP}\n")


def plot_h0_heatmap(h0_df: pd.DataFrame, out_dir: Path, mode: str) -> None:
    """
    Heatmap semaforo: verde = H₀ rigettata (profitto anomalo),
                      rosso  = H₀ non rigettata.
    Righe = evento × carburante, colonne = metodi + verdetto.
    """
    available_methods = [m for m in ACTIVE_METHODS if m in h0_df.columns]
    if not available_methods:
        return

    all_cols = available_methods + ["VERDETTO"]
    nrows    = len(h0_df)
    ncols    = len(all_cols)

    # Matrice numerica: 1 = rigettata, 0 = non rigettata, -1 = N/D
    mat = np.full((nrows, ncols), -1.0)

    for i, (_, r) in enumerate(h0_df.iterrows()):
        for j, col in enumerate(all_cols):
            if col == "VERDETTO":
                v = str(r.get("verdict", ""))
                mat[i, j] = 1.0 if ("RIGETTATA" in v and "NON" not in v) else 0.0
            else:
                cell = str(r.get(col, ""))
                if "RIGETTATA" in cell and "NON" not in cell:
                    mat[i, j] = 1.0
                elif "NON RIGETTATA" in cell:
                    mat[i, j] = 0.0

    # Colormap: rosso=0, bianco=non definito, verde=1
    cmap = mcolors.LinearSegmentedColormap.from_list(
        "h0_traffic",
        [(0.0, "#e74c3c"), (0.5, "#ecf0f1"), (1.0, "#27ae60")],
    )
    norm = mcolors.Normalize(vmin=0, vmax=1)

    fig, ax = plt.subplots(figsize=(max(8, ncols * 2.2), max(4, nrows * 0.9 + 1.5)))
    fig.suptitle(
        f"Test H₀/H₁ — Profitti Anomali Distributori  [mode={mode}]\n"
        f"Verde = H₀ rigettata (profitto anomalo ±{CI_SIGMA_H0:.0f}σ)  |  "
        f"Rosso = H₀ non rigettata",
        fontsize=11, fontweight="bold",
    )

    # Disegna celle colorate (solo dove mat ≥ 0)
    for i in range(nrows):
        for j in range(ncols):
            v = mat[i, j]
            color = cmap(norm(v)) if v >= 0 else "#bdc3c7"
            rect  = plt.Rectangle([j, nrows - i - 1], 1, 1, facecolor=color,
                                   edgecolor="white", linewidth=1.5)
            ax.add_patch(rect)

            # Testo
            r = h0_df.iloc[i]
            col = all_cols[j]
            if col == "VERDETTO":
                text = "✔" if v == 1 else "✘"
                fontweight = "bold"
            else:
                g = r.get(f"{col}_gain", np.nan)
                g_str = f"{float(g):+.0f}" if not (isinstance(g, float) and np.isnan(g)) else "n/d"
                text = ("✔\n" if v == 1 else "✘\n" if v == 0 else "?\n") + g_str + " M€"
                fontweight = "bold" if v == 1 else "normal"

            text_color = "white" if v in (0.0, 1.0) else "black"
            ax.text(j + 0.5, nrows - i - 0.5, text,
                    ha="center", va="center", fontsize=8,
                    fontweight=fontweight, color=text_color)

    # Assi
    ax.set_xlim(0, ncols)
    ax.set_ylim(0, nrows)
    ax.set_xticks(np.arange(ncols) + 0.5)
    ax.set_xticklabels(
        [LABELS.get(c, c) if c != "VERDETTO" else "VERDETTO\nCombinato" for c in all_cols],
        fontsize=8, rotation=20, ha="right",
    )
    ax.set_yticks(np.arange(nrows) + 0.5)
    ax.set_yticklabels(
        [f"{r.get('carburante','?').upper()}\n{r.get('evento','?')[:22]}"
         for _, r in h0_df[::-1].iterrows()],
        fontsize=7,
    )
    ax.tick_params(length=0)

    fig.tight_layout()
    out = out_dir / "h0_test_heatmap.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  → H₀ heatmap: {out}")


# ══════════════════════════════════════════════════════════════════════════════
# Accordo tra segni
# ══════════════════════════════════════════════════════════════════════════════

def print_sign_agreement(pivot_df: pd.DataFrame) -> None:
    available = [m for m in ["v1_naive","v3_arima","v7_theilsen","v8_pymc"]
                 if m in pivot_df.columns]
    if len(available) < 2:
        return

    print("\n  ACCORDO TRA METODI (segno del guadagno):")
    print(f"  {'Evento':<28} {'Carb.':<10}", end="")
    for m in available:
        print(f"  {LABELS.get(m,m)[:14]:>14}", end="")
    print("  Accordo segno")
    print("  " + "─"*90)

    for _, row in pivot_df.iterrows():
        ev   = str(row.get("evento",""))[:27]
        fuel = str(row.get("carburante",""))[:9]
        vals = [row.get(m, np.nan) for m in available]
        signs = [np.sign(v) for v in vals if not (isinstance(v, float) and np.isnan(v))]
        agree = "✓" if len(set(signs)) == 1 else "✗"
        print(f"  {ev:<28} {fuel:<10}", end="")
        for v in vals:
            s = f"{v:+.0f}" if not (isinstance(v, float) and np.isnan(v)) else "n/a"
            print(f"  {s:>14}", end="")
        print(f"  {agree}")


# ══════════════════════════════════════════════════════════════════════════════
# 6. Test H₀ non-parametrici su residui ITS
# ══════════════════════════════════════════════════════════════════════════════

def run_nonparam_h0_tests(
    resid_df: pd.DataFrame,
    alpha: float = 0.05,
    n_perm: int = 4999,
) -> pd.DataFrame:
    """
    Per ogni (evento, carburante, metodo) applica la batteria non-parametrica
    sulle serie di residui pre/post esportate dai modelli ITS.

    Testa H₀: i residui post-break hanno mediana ≤ 0 (nessun extra-profitto).
    H₁: i residui post-break hanno mediana > 0 (extra-profitto anomalo).

    Ritorna DataFrame con una riga per ogni (evento, carburante, metodo).
    """
    if not _HAS_NONPARAM:
        print("  ⚠ utils/nonparametric_tests.py non trovato – sezione 6 saltata.")
        return pd.DataFrame()

    if resid_df.empty:
        return pd.DataFrame()

    required_cols = {"date", "residual", "phase", "metodo", "evento", "carburante"}
    if not required_cols.issubset(resid_df.columns):
        missing = required_cols - set(resid_df.columns)
        print(f"  ⚠ Colonne mancanti nei residui: {missing}")
        return pd.DataFrame()

    rows = []
    rng  = np.random.default_rng(42)

    groups = resid_df.groupby(["evento", "carburante", "metodo"])
    for (ev_name, fuel_key, metodo), grp in groups:
        pre_resid  = grp.loc[grp["phase"] == "pre",  "residual"].dropna().values
        post_resid = grp.loc[grp["phase"] == "post", "residual"].dropna().values

        if len(post_resid) < 4:
            print(f"  [{metodo} | {fuel_key} | {ev_name}] "
                  f"post_resid insufficienti ({len(post_resid)}) — salto.")
            continue

        label = f"{metodo}  ·  {fuel_key.upper()}  ·  {ev_name}"
        print(f"\n  ── {label}")
        result = nonparam_h0_battery(
            post_resid=post_resid,
            pre_resid=pre_resid,
            alpha=alpha,
            n_perm=n_perm,
            rng=rng,
        )
        print_battery_results(result, label=label, alpha=alpha)

        row = {
            "evento":     ev_name,
            "carburante": fuel_key,
            "metodo":     metodo,
        }
        row.update(result)
        rows.append(row)

    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows)


def plot_nonparam_heatmap(
    np_df: pd.DataFrame,
    out_dir: Path,
    mode_label: str,
    alpha: float = 0.05,
) -> None:
    """
    Heatmap 2D: righe = (evento × carburante), colonne = metodo.
    Colore: verde = H₀ rigettata (profitto anomalo), rosso = H₀ non rigettata.
    Intensità proporzionale al numero di test che rifiutano H₀.
    """
    if np_df.empty:
        return

    methods  = [m for m in ["v1_naive", "v3_arima", "v7_theilsen","v8_pymc"]
                if m in np_df["metodo"].values]
    ev_fuel_combos = (np_df[["evento", "carburante"]]
                      .drop_duplicates()
                      .sort_values(["carburante", "evento"])
                      .reset_index(drop=True))

    nrows = len(ev_fuel_combos)
    ncols = len(methods) + 1  # +1 per colonna VERDETTO aggregato

    # Matrice valore: [0,1] proporzionale a n_reject/n_valid; -1 = N/D
    mat = np.full((nrows, ncols), -1.0)

    for i, (_, combo) in enumerate(ev_fuel_combos.iterrows()):
        ev   = combo["evento"]
        fuel = combo["carburante"]
        sub  = np_df[(np_df["evento"] == ev) & (np_df["carburante"] == fuel)]

        method_verdicts = []
        for j, met in enumerate(methods):
            row_m = sub[sub["metodo"] == met]
            if row_m.empty:
                continue
            r = row_m.iloc[0]
            n_valid  = int(r.get("n_tests_valid", 0))
            n_reject = int(r.get("n_tests_reject", 0))
            val = (n_reject / n_valid) if n_valid > 0 else -1.0
            mat[i, j] = val
            method_verdicts.append(bool(r.get("h0_rejected", False)))

        # Colonna aggregata: majority vote tra i metodi disponibili
        if method_verdicts:
            agg_reject = sum(method_verdicts)
            agg_val    = agg_reject / len(method_verdicts)
            mat[i, ncols - 1] = agg_val

    cmap = mcolors.LinearSegmentedColormap.from_list(
        "h0_nonparam",
        [(0.0, "#e74c3c"), (0.5, "#f9e4b7"), (1.0, "#27ae60")],
    )
    norm = mcolors.Normalize(vmin=0.0, vmax=1.0)

    col_labels = [LABELS.get(m, m) for m in methods] + ["VERDETTO\nAggregato"]

    fig, ax = plt.subplots(figsize=(max(10, ncols * 2.4), max(4, nrows * 0.85 + 2.0)))
    fig.suptitle(
        f"Test H₀ Non-Parametrici — Profitti Anomali Distributori  [mode={mode_label}]\n"
        f"Verde = H₀ rigettata (extra-profitto)  |  Rosso = H₀ non rigettata  |  "
        f"α={alpha}",
        fontsize=11, fontweight="bold",
    )

    for i in range(nrows):
        for j in range(ncols):
            v = mat[i, j]
            color = cmap(norm(v)) if v >= 0 else "#bdc3c7"
            rect = plt.Rectangle([j, nrows - i - 1], 1, 1, facecolor=color,
                                  edgecolor="white", linewidth=1.8)
            ax.add_patch(rect)

            # Testo: percentuale test che rifiutano H₀
            if v < 0:
                text = "N/D"
            else:
                sub2 = np_df[
                    (np_df["evento"] == ev_fuel_combos.iloc[i]["evento"]) &
                    (np_df["carburante"] == ev_fuel_combos.iloc[i]["carburante"])
                ]
                if j < len(methods):
                    row_m2 = sub2[sub2["metodo"] == methods[j]]
                    if not row_m2.empty:
                        r2 = row_m2.iloc[0]
                        n_valid2  = int(r2.get("n_tests_valid", 0))
                        n_reject2 = int(r2.get("n_tests_reject", 0))
                        hl = r2.get("hodges_lehmann_eurl", float("nan"))
                        hl_s = f"\nHL={float(hl):+.4f}" if isinstance(hl, (int, float)) and np.isfinite(float(hl)) else ""
                        text = f"{n_reject2}/{n_valid2}{hl_s}"
                    else:
                        text = "N/D"
                else:
                    # Aggregato
                    verdicts = []
                    for met2 in methods:
                        rm2 = sub2[sub2["metodo"] == met2]
                        if not rm2.empty:
                            verdicts.append(bool(rm2.iloc[0].get("h0_rejected", False)))
                    agree  = sum(verdicts)
                    total  = len(verdicts)
                    symbol = "✔" if agree > total / 2 else "✘"
                    text   = f"{symbol}\n{agree}/{total}"

            text_color = "white" if v in (0.0, 1.0) or (v > 0.85) or (v < 0.10) else "black"
            ax.text(j + 0.5, nrows - i - 0.5, text,
                    ha="center", va="center", fontsize=7.5,
                    fontweight="bold" if j == ncols - 1 else "normal",
                    color=text_color)

    ax.set_xlim(0, ncols)
    ax.set_ylim(0, nrows)
    ax.set_xticks(np.arange(ncols) + 0.5)
    ax.set_xticklabels(col_labels, fontsize=8, rotation=20, ha="right")
    ax.set_yticks(np.arange(nrows) + 0.5)
    ax.set_yticklabels(
        [f"{ev_fuel_combos.iloc[nrows - i - 1]['carburante'].upper()}\n"
         f"{ev_fuel_combos.iloc[nrows - i - 1]['evento'][:22]}"
         for i in range(nrows)],
        fontsize=7,
    )
    ax.tick_params(length=0)

    # Colorbar
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, fraction=0.025, pad=0.02)
    cbar.set_label("Frazione test che rifiutano H₀", fontsize=8)

    fig.tight_layout()
    out = out_dir / "nonparam_h0_heatmap.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  → Heatmap non-parametrica: {out}")


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    parser = argparse.ArgumentParser(description="Confronto 3 metodi ITS")
    parser.add_argument("--mode", choices=["fixed", "detected"], default="fixed",
                        help="fixed o detected: deve corrispondere ai file già prodotti")
    parser.add_argument("--detect", choices=["margin", "price"], default="margin",
                        help="(solo mode=detected) variante da confrontare: "
                             "margin [default] o price")
    args, _ = parser.parse_known_args()
    mode          = args.mode
    detect_target = args.detect

    if mode == "detected":
        OUT_DIR = _OUT_BASE / "detected" / detect_target / "compare"
    else:
        OUT_DIR = _OUT_BASE / mode / "compare"
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("═"*70)
    print(f"  02d_compare.py  –  Confronto 4 Metodi ITS  [mode={mode}]")
    if mode == "detected":
        print(f"  Variante detection: {detect_target}")
        print(f"  Legge da: {_OUT_BASE / 'detected' / detect_target}")
    else:
        print(f"  Legge da: {_OUT_BASE / mode}")
    print(f"  Output:   {OUT_DIR}")
    print("═"*70)

    df = load_results(mode, detect_target)
    if df.empty:
        return

    n_methods_with_data = df.groupby("metodo")["gain_total_meur"].apply(
        lambda s: s.notna().any()
    ).sum()
    if n_methods_with_data == 0:
        print("  ✗ gain_total_meur assente in tutti i metodi caricati. "
              "Verificare i nomi colonne nei CSV.")
        return

    mode_label = f"detected/{detect_target}" if mode == "detected" else mode

    # ── Soglie di volatilità pre-shock ────────────────────────────────────────
    print(f"\n  VOLATILITÀ PRE-SHOCK (soglia ±{CI_SIGMA_H0:.0f}σ cumulata)")
    print("  " + "─"*60)
    vol_thresholds = compute_volatility_thresholds(df)
    if not vol_thresholds:
        print("  (nessuna soglia calcolata – barplot senza banda)")

    # ── 1. Tabella ────────────────────────────────────────────────────────────
    table = make_comparison_table(df)
    csv_out = OUT_DIR / "compare_table.csv"
    table.to_csv(csv_out, index=False)
    print(f"\n  → Tabella: {csv_out}")
    print("\n" + table.to_string(index=False))

    # ── Accordo segni ─────────────────────────────────────────────────────────
    pivot_raw = df.pivot_table(
        index=["evento","carburante"], columns="metodo",
        values="gain_total_meur", aggfunc="first",
    ).reset_index()
    print_sign_agreement(pivot_raw)

    # ── 2–4. Plot ─────────────────────────────────────────────────────────────
    plot_barplot(df, OUT_DIR, mode_label)
    plot_scatter(pivot_raw, OUT_DIR, mode_label)
    plot_heatmap(df, OUT_DIR, mode_label)

    # ── 5. Test H₀/H₁ ────────────────────────────────────────────────────────
    print(f"\n{'═'*70}")
    print("  5. TEST FORMALE H₀/H₁  (±{:.0f}σ, CI inferiore)".format(CI_SIGMA_H0))
    print(f"{'═'*70}")
    h0_df = make_h0_test_table(df, vol_thresholds=vol_thresholds)
    if not h0_df.empty:
        print_h0_summary(h0_df)
        plot_h0_heatmap(h0_df, OUT_DIR, mode_label)
        h0_csv = OUT_DIR / "h0_test_summary.csv"
        h0_df.to_csv(h0_csv, index=False)
        print(f"  → CSV H₀ test: {h0_csv}")

    # ── 6. Test H₀ NON-PARAMETRICI su residui ITS ────────────────────────────
    print(f"\n{'═'*70}")
    print("  6. TEST NON-PARAMETRICI H₀/H₁  (batteria su residui ITS)")
    print(f"     H₀: residui post-break hanno mediana ≤ 0  (nessun extra-profitto)")
    print(f"     H₁: residui post-break hanno mediana  > 0  (profitto anomalo)")
    print(f"{'═'*70}")

    resid_df = load_residuals(mode, detect_target)
    if not resid_df.empty and _HAS_NONPARAM:
        np_df = run_nonparam_h0_tests(resid_df, alpha=0.05, n_perm=4999)
        if not np_df.empty:
            np_csv = OUT_DIR / "nonparam_h0_summary.csv"
            np_df.to_csv(np_csv, index=False)
            print(f"\n  → CSV non-parametrico: {np_csv}")

            # Stampa riepilogo compatto
            print(f"\n{'─'*70}")
            print("  RIEPILOGO VERDETTI NON-PARAMETRICI")
            print(f"  {'Evento':<28} {'Carb.':<10} {'Metodo':<22} "
                  f"{'n_post':>6} {'HL (€/L)':>10} {'Rigetti':>7}  Verdetto")
            print(f"  {'─'*100}")
            for _, r in np_df.sort_values(
                    ["carburante", "evento", "metodo"]).iterrows():
                hl  = r.get("hodges_lehmann_eurl", float("nan"))
                hl_s = f"{float(hl):+.5f}" if isinstance(hl, (int, float)) and np.isfinite(float(hl)) else "  N/D   "
                icon = "🔴" if r.get("verdict") == "H0_RIGETTATA" else (
                       "🟡" if r.get("verdict") == "INDETERMINATO" else "🟢")
                print(f"  {str(r['evento'])[:27]:<28} "
                      f"{str(r['carburante'])[:9]:<10} "
                      f"{str(r['metodo'])[:21]:<22} "
                      f"{int(r.get('n_post', 0)):>6} "
                      f"{hl_s:>10} "
                      f"{int(r.get('n_tests_reject', 0)):>2}/{int(r.get('n_tests_valid', 0)):<4} "
                      f" {icon} {r.get('verdict', '?')}")

            plot_nonparam_heatmap(np_df, OUT_DIR, mode_label, alpha=0.05)
    elif not _HAS_NONPARAM:
        print("  ⚠ utils/nonparametric_tests.py non trovato — sezione saltata.")
        print("    Assicurarsi che il file esista nella cartella utils/.")
    else:
        print("  ⚠ Nessun file residuals_*.csv trovato — eseguire prima i modelli ITS.")

    # ── Statistiche ───────────────────────────────────────────────────────────
    available = [m for m in ["v1_naive","v3_arima","v7_theilsen","v8_pymc"]
                 if m in pivot_raw.columns]
    if len(available) >= 2:
        gains  = pivot_raw[available].values.astype(float)
        ranges = np.nanmax(gains, axis=1) - np.nanmin(gains, axis=1)
        means  = np.nanmean(gains, axis=1)
        cv     = np.abs(ranges / np.where(means != 0, means, np.nan))
        print(f"\n  STATISTICHE ACCORDO INTER-METODO:")
        print(f"    Range medio (M€):  {np.nanmean(ranges):+.1f}")
        print(f"    Range max  (M€):   {np.nanmax(ranges):+.1f}")
        print(f"    CV medio:          {np.nanmean(cv)*100:.1f}%")


if __name__ == "__main__":
    main()