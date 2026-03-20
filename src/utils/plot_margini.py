#!/usr/bin/env python3
"""
plot_margini.py
===============
Grafico storico dei margini distributori (crack spread) 2015–2026.

  Pannello superiore : margine benzina  = benzina_net  − Eurobob  (€/L)
  Pannello inferiore : margine gasolio  = gasolio_net  − Gas Oil  (€/L)

Linea di baseline 2019 (media ± 2σ) e linee verticali dei tre shock.

Uso:
  python3 utils/plot_margini.py
  python3 utils/plot_margini.py --out path.png
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.patches as mpatches
import pandas as pd
import numpy as np

# Importa conversioni dal modulo utils
import sys
sys.path.insert(0, str(Path(__file__).parent))
from conversions import load_eurusd, usd_ton_to_eur_liter, GAS_OIL, EUROBOB

# ── Percorsi ──────────────────────────────────────────────────────────────────
BASE_DIR     = Path(__file__).parent.parent
PRICES_CSV   = BASE_DIR / "data" / "processed" / "daily_fuel_prices_stradale.csv"
EUROBOB_CSV  = BASE_DIR / "data" / "Futures" / "Eurobob_B7H1_date.csv"
GASOIL_CSV   = BASE_DIR / "data" / "Futures" / "London Gas Oil Futures Historical Data.csv"
EURUSD_CSV   = BASE_DIR / "data" / "raw" / "eurusd.csv"
OUT_DIR      = BASE_DIR / "data" / "plots" / "utils"
DEFAULT_OUT  = OUT_DIR / "margini_2015_2026.png"

# ── Costanti ──────────────────────────────────────────────────────────────────#
#BASELINE_START = "2019-01-01"
#BASELINE_END   = "2019-12-31"

EVENTS = [
    ("2022-02-24", "Ucraina\n(feb 2022)",       "#e74c3c"),
    ("2025-06-13", "Iran–Israele\n(giu 2025)",  "#e67e22"),
    ("2026-02-28", "Hormuz\n(feb 2026)",          "#8e44ad"),
]

C_BENZ   = "#E63946"
C_GAS    = "#1D3557"
C_BASE   = "#607d8b"
C_2SIGMA = "#b0bec5"


# ── Carica futures ─────────────────────────────────────────────────────────────
def load_eurobob() -> pd.Series:
    df = pd.read_csv(EUROBOB_CSV, dtype=str)
    df.columns = [c.strip().lower() for c in df.columns]
    df["date"] = pd.to_datetime(df["data"], format="%m/%d/%Y", errors="coerce")
    # Il CSV Eurobob usa già il punto come separatore decimale → parse diretto
    df["price"] = pd.to_numeric(df["chiusura"], errors="coerce")
    df = df.dropna(subset=["date", "price"]).sort_values("date")
    return df.set_index("date")["price"].rename("eurobob_usd_ton")


def load_gasoil() -> pd.Series:
    df = pd.read_csv(GASOIL_CSV, dtype=str, encoding="utf-8-sig")
    df.columns = [c.strip().strip('"') for c in df.columns]
    df["date"] = pd.to_datetime(df["Date"].str.strip().str.strip('"'),
                                format="%m/%d/%Y", errors="coerce")
    df["price"] = (df["Price"].str.strip().str.strip('"')
                   .str.replace(",", "", regex=False)
                   .pipe(pd.to_numeric, errors="coerce"))
    df = df.dropna(subset=["date", "price"]).sort_values("date")
    return df.set_index("date")["price"].rename("gasoil_usd_ton")


# ── Calcola margini ────────────────────────────────────────────────────────────
def build_margins() -> pd.DataFrame:
    print("Carico prezzi netti...")
    prices = pd.read_csv(PRICES_CSV, parse_dates=["date"]).set_index("date")
    prices = prices[["benzina_net", "gasolio_net"]].dropna()

    print("Carico EUR/USD...")
    eurusd = load_eurusd(
        csv_path=EURUSD_CSV if EURUSD_CSV.exists() else None,
        start="2015-01-01", end="2026-12-31",
    )

    print("Carico futures Eurobob...")
    eurobob_usd = load_eurobob()
    print("Carico futures Gas Oil...")
    gasoil_usd  = load_gasoil()

    print("Converto in €/L...")
    eurobob_eur = usd_ton_to_eur_liter(eurobob_usd, eurusd, EUROBOB)
    gasoil_eur  = usd_ton_to_eur_liter(gasoil_usd,  eurusd, GAS_OIL)

    # Allinea tutto su base giornaliera con forward-fill (weekends futures)
    idx = prices.index
    eurobob_eur = eurobob_eur.reindex(idx, method="ffill")
    gasoil_eur  = gasoil_eur.reindex(idx, method="ffill")

    df = prices.copy()
    df["eurobob_eur_l"] = eurobob_eur
    df["gasoil_eur_l"]  = gasoil_eur
    df["margine_benz"]  = df["benzina_net"] - df["eurobob_eur_l"]
    df["margine_gas"]   = df["gasolio_net"] - df["gasoil_eur_l"]
    df = df.dropna(subset=["margine_benz", "margine_gas"])

    return df


# ── Baseline 2019 ──────────────────────────────────────────────────────────────
def baseline_stats(series: pd.Series):
    bl = series.loc[BASELINE_START:BASELINE_END].dropna()
    return bl.mean(), bl.std()


# ── Pannello singolo ───────────────────────────────────────────────────────────
def draw_panel(ax, series: pd.Series, color: str, title: str, show_xlabels: bool):
    #mean, std = baseline_stats(series)

    # Area ±2σ baseline
    #ax.axhspan(mean - 2 * std, mean + 2 * std,color=C_2SIGMA, alpha=0.35, zorder=1, label="Baseline 2019 ±2σ")
    # Linea media baseline
    #ax.axhline(mean, color=C_BASE, lw=1.0, ls="--", zorder=2,label=f"Media 2019: {mean:.3f} €/L")

    # Serie margine
    ax.plot(series.index, series, color=color, lw=0.8, zorder=3, label="Margine (€/L)")

    # Media mobile 30gg per leggere il trend
    #roll = series.rolling(30, center=True, min_periods=5).mean()
    #ax.plot(roll.index, roll, color=color, lw=2.0, alpha=0.75, zorder=4,label="Media mobile 30gg")

    # Shock geopolitici
    ymin, ymax = ax.get_ylim()
    for date_str, label, ec in EVENTS:
        d = pd.Timestamp(date_str)
        ax.axvline(d, color=ec, lw=1.3, ls="--", alpha=0.85, zorder=5)
        ax.text(d, ymax - (ymax - ymin) * 0.03, label,
                rotation=90, va="top", ha="right",
                fontsize=6.5, color=ec, alpha=0.9)

    ax.set_ylabel("€/L", fontsize=9)
    ax.set_title(title, fontsize=10, fontweight="bold", pad=4)
    ax.legend(fontsize=7.5, loc="upper left", framealpha=0.92)
    ax.grid(axis="y", alpha=0.18, linestyle="--")
    ax.spines[["top", "right"]].set_visible(False)

    if show_xlabels:
        loc = mdates.AutoDateLocator()
        ax.xaxis.set_major_locator(loc)
        ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(loc))
        ax.tick_params(axis="x", labelsize=8)
    else:
        ax.tick_params(axis="x", labelbottom=False)


# ── Main ───────────────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--out", type=Path, default=DEFAULT_OUT)
    return p.parse_args()


def main():
    args = parse_args()
    df = build_margins()

    fig, (ax_b, ax_g) = plt.subplots(
        2, 1, figsize=(15, 9), sharex=True,
        facecolor="white",
        gridspec_kw={"hspace": 0.08},
    )
    fig.subplots_adjust(top=0.93, bottom=0.07, left=0.07, right=0.97)

    draw_panel(ax_b, df["margine_benz"], C_BENZ,
               "Margine benzina — prezzo netto − Eurobob ARA (€/L)",
               show_xlabels=False)
    draw_panel(ax_g, df["margine_gas"],  C_GAS,
               "Margine gasolio — prezzo netto − Gas Oil ICE (€/L)",
               show_xlabels=True)

    # Legenda shock condivisa in basso
    event_patches = [mpatches.Patch(color=c, label=lbl.replace("\n", " "))
                     for _, lbl, c in EVENTS]
    fig.legend(handles=event_patches, loc="lower center", ncol=3,
               fontsize=8, title="Shock geopolitici", title_fontsize=8,
               frameon=True, bbox_to_anchor=(0.5, 0.01))

    fig.suptitle("Margini distributori carburanti in Italia — 2015/2026",
                 fontsize=13, fontweight="bold")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=150, bbox_inches="tight", pad_inches=0.15)
    plt.close(fig)
    print(f"\n✓  Salvato: {args.out}")


if __name__ == "__main__":
    main()
