#!/usr/bin/env python3
"""
plot_andamento.py
=================
Produce tre PNG separati sull'andamento storico benzina e gasolio (2015-oggi):

  01_prezzi_pompa.png        — prezzi alla pompa (€/L)
  02_prezzi_netti.png        — prezzi netti ex-tasse (€/L)
  03_cuneo_fiscale.png       — cuneo fiscale accise+IVA (€/L)

Ogni grafico ha le tre linee verticali degli shock geopolitici.

Uso:
  python3 utils/plot_andamento.py
  python3 utils/plot_andamento.py --out-dir path/to/dir
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.patches as mpatches
import pandas as pd

# ── Percorsi ──────────────────────────────────────────────────────────────────
BASE_DIR    = Path(__file__).parent.parent
PRICES_CSV  = BASE_DIR / "data" / "processed" / "daily_fuel_prices_stradale.csv"
DEFAULT_OUT = BASE_DIR / "data" / "plots" / "utils"

# ── Eventi geopolitici ────────────────────────────────────────────────────────
EVENTS = [
    ("2022-02-24", "Ucraina (feb 2022)",       "#e74c3c"),
    ("2025-06-13", "Iran–Israele (giu 2025)",  "#e67e22"),
    ("2026-02-28", "Hormuz (feb 2026)",         "#8e44ad"),
]

# ── Colori ────────────────────────────────────────────────────────────────────
C_BENZ = "#E63946"
C_GAS  = "#1D3557"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    return p.parse_args()


def _base_fig():
    fig, ax = plt.subplots(figsize=(15, 5), facecolor="white")
    return fig, ax


def _add_events(ax):
    ymin, ymax = ax.get_ylim()
    for date_str, label, color in EVENTS:
        d = pd.Timestamp(date_str)
        ax.axvline(d, color=color, lw=1.2, ls="--", alpha=0.8, zorder=3)
        ax.text(d, ymax - (ymax - ymin) * 0.03, label,
                rotation=90, va="top", ha="right",
                fontsize=7, color=color, alpha=0.9)


def _event_legend():
    return [mpatches.Patch(color=c, label=lbl) for _, lbl, c in EVENTS]


def _finish(ax, fig, title, out_path):
    loc = mdates.AutoDateLocator()
    ax.xaxis.set_major_locator(loc)
    ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(loc))
    ax.tick_params(axis="x", labelsize=8)
    ax.set_ylabel("€/L", fontsize=9)
    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.grid(axis="y", alpha=0.2, linestyle="--")
    ax.spines[["top", "right"]].set_visible(False)

    data_leg  = ax.get_legend_handles_labels()
    event_leg = _event_legend()
    ax.legend(
        data_leg[0] + event_leg,
        data_leg[1] + [p.get_label() for p in event_leg],
        fontsize=8, loc="upper left", framealpha=0.9,
    )

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"✓  {out_path.name}")


def plot_pompa(df, out_dir):
    fig, ax = _base_fig()
    ax.plot(df["date"], df["benzina_pump"], color=C_BENZ, lw=0.9, label="Benzina")
    ax.plot(df["date"], df["gasolio_pump"], color=C_GAS,  lw=0.9, label="Gasolio")
    _add_events(ax)
    _finish(ax, fig, "Prezzi alla pompa — benzina vs gasolio (€/L)",
            out_dir / "01_prezzi_pompa.png")


def plot_netti(df, out_dir):
    fig, ax = _base_fig()
    ax.plot(df["date"], df["benzina_net"], color=C_BENZ, lw=0.9, label="Benzina netto")
    ax.plot(df["date"], df["gasolio_net"], color=C_GAS,  lw=0.9, label="Gasolio netto")
    ax.fill_between(df["date"], df["benzina_net"], df["gasolio_net"],
                    alpha=0.08, color="#999999")
    _add_events(ax)
    _finish(ax, fig, "Prezzi netti ex-tasse — benzina vs gasolio (€/L)",
            out_dir / "02_prezzi_netti.png")


def plot_cuneo(df, out_dir):
    fig, ax = _base_fig()
    ax.plot(df["date"], df["tax_wedge_benz"], color=C_BENZ, lw=0.9, label="Tax wedge benzina")
    ax.plot(df["date"], df["tax_wedge_gas"],  color=C_GAS,  lw=0.9, label="Tax wedge gasolio")
    ax.fill_between(df["date"], df["tax_wedge_benz"], 0, alpha=0.10, color=C_BENZ)
    ax.fill_between(df["date"], df["tax_wedge_gas"],  0, alpha=0.10, color=C_GAS)
    _add_events(ax)
    _finish(ax, fig, "Cuneo fiscale (accise + IVA) — benzina vs gasolio (€/L)",
            out_dir / "03_cuneo_fiscale.png")


def plot_combined(df, out_dir):
    """Tutti e tre i pannelli in verticale, asse X condiviso."""
    fig, axes = plt.subplots(
        3, 1, figsize=(15, 13), sharex=True, facecolor="white",
        gridspec_kw={"hspace": 0.08},
    )
    fig.subplots_adjust(top=0.93, bottom=0.07, left=0.06, right=0.98)

    configs = [
        (axes[0], "benzina_pump", "gasolio_pump",
         "Prezzi alla pompa (€/L)"),
        (axes[1], "benzina_net",  "gasolio_net",
         "Prezzi netti ex-tasse (€/L)"),
        (axes[2], "tax_wedge_benz", "tax_wedge_gas",
         "Cuneo fiscale — accise + IVA (€/L)"),
    ]

    for ax, col_b, col_g, title in configs:
        ax.plot(df["date"], df[col_b], color=C_BENZ, lw=0.9, label="Benzina")
        ax.plot(df["date"], df[col_g], color=C_GAS,  lw=0.9, label="Gasolio")

        # area riempimento leggero per il terzo pannello
        if "wedge" in col_b:
            ax.fill_between(df["date"], df[col_b], 0, alpha=0.10, color=C_BENZ)
            ax.fill_between(df["date"], df[col_g], 0, alpha=0.10, color=C_GAS)
        else:
            ax.fill_between(df["date"], df[col_b], df[col_g],
                            alpha=0.07, color="#999999")

        ymin, ymax = ax.get_ylim()
        for date_str, label, color in EVENTS:
            d = pd.Timestamp(date_str)
            ax.axvline(d, color=color, lw=1.2, ls="--", alpha=0.8, zorder=3)
            ax.text(d, ymax - (ymax - ymin) * 0.03, label,
                    rotation=90, va="top", ha="right",
                    fontsize=6.5, color=color, alpha=0.9)

        ax.set_ylabel("€/L", fontsize=9)
        ax.set_title(title, fontsize=10, fontweight="bold", pad=4)
        ax.grid(axis="y", alpha=0.18, linestyle="--")
        ax.spines[["top", "right"]].set_visible(False)
        ax.legend(fontsize=8, loc="upper left", framealpha=0.9)

    # data labels solo sull'asse X dell'ultimo pannello
    loc = mdates.AutoDateLocator()
    axes[-1].xaxis.set_major_locator(loc)
    axes[-1].xaxis.set_major_formatter(mdates.ConciseDateFormatter(loc))
    axes[-1].tick_params(axis="x", labelsize=8)

    # legenda shock in basso
    event_patches = [mpatches.Patch(color=c, label=lbl) for _, lbl, c in EVENTS]
    fig.legend(
        handles=event_patches,
        loc="lower center", ncol=3, fontsize=8,
        title="Shock geopolitici", title_fontsize=8,
        frameon=True, bbox_to_anchor=(0.5, 0.01),
    )

    fig.suptitle("Andamento prezzi carburanti in Italia (2015–oggi)",
                 fontsize=13, fontweight="bold")

    out_path = out_dir / "00_andamento_combinato.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight",
                pad_inches=0.15)
    plt.close(fig)
    print(f"✓  {out_path.name}")


def main():
    args = parse_args()

    df = pd.read_csv(PRICES_CSV, parse_dates=["date"]).sort_values("date")
    df = df.dropna(subset=["benzina_pump", "gasolio_pump",
                            "benzina_net",  "gasolio_net"])

    plot_combined(df, args.out_dir)   # verticale combinato
    plot_pompa(df, args.out_dir)
    plot_netti(df, args.out_dir)
    plot_cuneo(df, args.out_dir)

    print(f"\nSalvati in: {args.out_dir}")


if __name__ == "__main__":
    main()
