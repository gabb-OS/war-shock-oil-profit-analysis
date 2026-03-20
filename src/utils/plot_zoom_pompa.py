#!/usr/bin/env python3
"""
plot_zoom_pompa.py
==================
Tre pannelli in verticale, uno per evento geopolitico.
Asse X condiviso: giorni relativi rispetto alla data shock.
Mostra benzina e gasolio al prezzo alla pompa (IVA inclusa).

Linea verticale tratteggiata al giorno 0 (shock).
Area grigia tra le due curve.
Annotazione Δ medio pre→post per ciascun carburante.

Output:
  data/plots/utils/zoom_pompa.png

Uso:
  python3 utils/plot_zoom_pompa.py
  python3 utils/plot_zoom_pompa.py --pre 40 --post 60
  python3 utils/plot_zoom_pompa.py --out path.png
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import pandas as pd

BASE_DIR    = Path(__file__).parent.parent
PRICES_CSV  = BASE_DIR / "data" / "processed" / "daily_fuel_prices_stradale.csv"
OUT_DIR     = BASE_DIR / "data" / "plots" / "utils"
DEFAULT_OUT = OUT_DIR / "zoom_pompa.png"

EVENTS = [
    ("2022-02-24", "Invasione russa dell'Ucraina",  "#e74c3c"),
    ("2025-06-13", "Guerra Iran–Israele",            "#e67e22"),
    ("2026-02-28", "Chiusura Stretto di Hormuz",     "#8e44ad"),
]

C_BENZ = "#E63946"
C_GAS  = "#1D3557"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--pre",  type=int, default=40,
                   help="Giorni prima dello shock (default: 40)")
    p.add_argument("--post", type=int, default=40,
                   help="Giorni dopo lo shock (default: 40)")
    p.add_argument("--out",  type=Path, default=DEFAULT_OUT)
    return p.parse_args()


def window_df(df: pd.DataFrame, shock: pd.Timestamp, pre: int, post: int) -> pd.DataFrame:
    mask = (df["date"] >= shock - pd.Timedelta(days=pre)) & \
           (df["date"] <= shock + pd.Timedelta(days=post))
    sub = df[mask].copy()
    sub["day"] = (sub["date"] - shock).dt.days
    return sub


def draw_panel(ax, sub: pd.DataFrame, shock_str: str, label: str,
               color: str, pre: int, post: int):
    ax.plot(sub["day"], sub["benzina_pump"], color=C_BENZ, lw=1.4, label="Benzina")
    ax.plot(sub["day"], sub["gasolio_pump"], color=C_GAS,  lw=1.4, label="Gasolio")
    ax.fill_between(sub["day"], sub["benzina_pump"], sub["gasolio_pump"],
                    alpha=0.07, color="#888888")

    ax.axvline(0, color=color, lw=1.8, ls="--", zorder=4,
               label=f"Shock ({shock_str})")
    ax.axvspan(0, post, alpha=0.04, color=color)

    ax.text(0.01, 0.97, "◀ pre",  transform=ax.transAxes,
            ha="left",  va="top", fontsize=7, color=color, style="italic", alpha=0.8)
    ax.text(0.99, 0.97, "post ▶", transform=ax.transAxes,
            ha="right", va="top", fontsize=7, color=color, style="italic", alpha=0.8)

    pre_data  = sub[sub["day"] < 0]
    post_data = sub[sub["day"] > 0]
    if not pre_data.empty and not post_data.empty:
        for col, c, y_frac in [("benzina_pump", C_BENZ, 0.18),
                                ("gasolio_pump", C_GAS,  0.08)]:
            delta = post_data[col].mean() - pre_data[col].mean()
            sign  = "+" if delta >= 0 else ""
            ax.text(0.98, y_frac, f"{sign}{delta:.3f} €/L",
                    transform=ax.transAxes, ha="right", va="bottom",
                    fontsize=8, color=c, fontweight="bold",
                    bbox=dict(boxstyle="round,pad=0.2", facecolor="white",
                              edgecolor=c, linewidth=0.7, alpha=0.85))

    ax.set_ylabel("€/L", fontsize=9)
    ax.set_title(f"{label}  —  {shock_str}", fontsize=10,
                 fontweight="bold", color=color, pad=5)
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("€%.3f"))
    ax.grid(axis="y", alpha=0.18, linestyle="--")
    ax.grid(axis="x", alpha=0.10, linestyle=":")
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(fontsize=8, loc="upper left", framealpha=0.9)


def main():
    args = parse_args()

    df = pd.read_csv(PRICES_CSV, parse_dates=["date"]).sort_values("date")
    df = df.dropna(subset=["benzina_pump", "gasolio_pump"])

    fig, axes = plt.subplots(
        len(EVENTS), 1,
        figsize=(13, 4.2 * len(EVENTS)),
        sharex=False,
        facecolor="white",
        gridspec_kw={"hspace": 0.35},
    )
    fig.subplots_adjust(top=0.93, bottom=0.06, left=0.08, right=0.97)

    for ax, (shock_str, label, color) in zip(axes, EVENTS):
        shock = pd.Timestamp(shock_str)
        sub   = window_df(df, shock, args.pre, args.post)

        if sub.empty:
            ax.text(0.5, 0.5, "Dati non disponibili",
                    transform=ax.transAxes, ha="center", va="center",
                    fontsize=10, color="grey")
            ax.set_title(label, fontsize=10, fontweight="bold")
            continue

        draw_panel(ax, sub, shock_str, label, color, args.pre, args.post)

        ax.set_xlabel("Giorni rispetto alla data shock", fontsize=9)
        ax.xaxis.set_major_locator(mticker.MultipleLocator(10))
        ax.xaxis.set_minor_locator(mticker.MultipleLocator(5))
        ax.tick_params(axis="x", labelsize=8)

    fig.suptitle(
        f"Zoom ±{args.pre}/{args.post} giorni attorno agli shock geopolitici\n"
        "(Prezzo alla pompa IVA inclusa, €/L — impianti stradali)",
        fontsize=12, fontweight="bold",
    )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=150, bbox_inches="tight", pad_inches=0.15)
    plt.close(fig)
    print(f"✓  Salvato: {args.out}")


if __name__ == "__main__":
    main()
