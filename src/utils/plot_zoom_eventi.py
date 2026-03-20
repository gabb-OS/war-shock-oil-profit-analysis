#!/usr/bin/env python3
"""
plot_zoom_eventi.py
===================
Per ogni evento: un PNG con benzina (sinistra) e gasolio (destra).
Ogni pannello mostra il prezzo alla pompa reale + le controfattuali di tutti i
metodi ITS disponibili (colori diversi, tratto tratteggiato).

Il residual ITS è calcolato sul prezzo netto; il tax wedge è fisso, quindi
pump_cf = pump_actual − residual è corretto senza nessuna correzione aggiuntiva.

Finestra: PRE giorni prima dello shock → POST giorni dopo.
Linea verticale tratteggiata al giorno 0.

Output:
  data/plots/utils/zoom_eventi/{evento_slug}.png

Uso:
  python3 utils/plot_zoom_eventi.py
  python3 utils/plot_zoom_eventi.py --pre 3 --post 40
  python3 utils/plot_zoom_eventi.py --out-dir path/dir
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import pandas as pd

BASE_DIR   = Path(__file__).parent.parent
PRICES_CSV = BASE_DIR / "data" / "processed" / "daily_fuel_prices_stradale.csv"
ITS_DIR    = BASE_DIR / "data" / "plots" / "its" / "fixed"
OUT_DIR    = BASE_DIR / "data" / "plots" / "utils" / "zoom_eventi"

EVENTS = [
    ("Ucraina_Feb_2022",      "2022-02-24", "Invasione Ucraina — feb 2022"),
    ("Iran-Israele_Giu_2025", "2025-06-13", "Guerra Iran–Israele — giu 2025"),
    ("Hormuz_Feb_2026",       "2026-02-28", "Chiusura Stretto di Hormuz — feb 2026"),
]
FUELS = [
    ("benzina", "benzina_pump", "Benzina"),
    ("gasolio", "gasolio_pump", "Gasolio"),
]
PUMP_COL = {"benzina": "benzina_pump", "gasolio": "gasolio_pump"}

METHOD_COLORS = {
    "v1_naive":    "#e74c3c",
    "v3_arima":    "#2980b9",
    "v7_theilsen": "#8e44ad",
    "v8_pymc":     "#e67e22",
}
FALLBACK_COLORS = ["#1abc9c", "#f39c12", "#c0392b", "#16a085", "#d35400"]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--pre",     type=int, default=3,
                   help="Giorni prima dello shock (default: 3)")
    p.add_argument("--post",    type=int, default=40,
                   help="Giorni dopo lo shock (default: 40)")
    p.add_argument("--out-dir", type=Path, default=OUT_DIR)
    return p.parse_args()


def discover_methods() -> list[str]:
    methods = []
    for d in sorted(ITS_DIR.iterdir()):
        if d.is_dir() and d.name != "compare":
            if any(d.glob("residuals_*.csv")):
                methods.append(d.name)
    return methods


def load_residuals(metodo: str, evento_slug: str, fuel: str) -> pd.DataFrame | None:
    path = ITS_DIR / metodo / f"residuals_{evento_slug}_{fuel}.csv"
    if not path.exists():
        return None
    df = pd.read_csv(path, parse_dates=["date"])
    return df.sort_values("date").reset_index(drop=True)


def build_window(
    residuals: pd.DataFrame,
    prices: pd.DataFrame,
    shock: pd.Timestamp,
    fuel: str,
    pre: int,
    post: int,
) -> pd.DataFrame | None:
    col      = PUMP_COL[fuel]
    start    = shock - pd.Timedelta(days=pre)
    post_res = residuals[residuals["phase"] == "post"][["date", "residual"]]
    post_end = post_res["date"].max() if not post_res.empty else shock
    end      = min(shock + pd.Timedelta(days=post), post_end)

    win = prices[(prices["date"] >= start) & (prices["date"] <= end)].copy()
    if win.empty:
        return None

    win["day"] = (win["date"] - shock).dt.days
    win = win.merge(post_res, on="date", how="left")
    win["residual"]    = win["residual"].fillna(0.0)
    win["pump_actual"] = win[col]
    win["pump_cf"]     = win[col] - win["residual"]

    return win[["date", "day", "pump_actual", "pump_cf"]]


def draw_panel(
    ax,
    prices: pd.DataFrame,
    shock: pd.Timestamp,
    slug: str,
    fuel: str,
    methods: list[str],
    pre: int,
    post: int,
    fuel_label: str,
    method_colors: dict,
):
    win_actual = None
    for metodo in methods:
        res = load_residuals(metodo, slug, fuel)
        if res is not None:
            win_actual = build_window(res, prices, shock, fuel, pre, post)
            if win_actual is not None:
                break

    if win_actual is None:
        ax.text(0.5, 0.5, "Dati non disponibili",
                transform=ax.transAxes, ha="center", va="center",
                fontsize=9, color="grey")
        ax.set_title(fuel_label, fontsize=9, fontweight="bold")
        return

    ax.plot(win_actual["day"], win_actual["pump_actual"],
            color="black", lw=1.8, ls="-", label="Prezzo pompa reale", zorder=5)

    for metodo in methods:
        res = load_residuals(metodo, slug, fuel)
        if res is None:
            continue
        win = build_window(res, prices, shock, fuel, pre, post)
        if win is None or win.empty:
            continue
        color = method_colors.get(metodo, "#999999")
        ax.plot(win["day"], win["pump_cf"],
                color=color, lw=1.2, ls="--", label=metodo, zorder=4)

    ax.axvline(0, color="black", lw=0.8, ls=":", alpha=0.5)
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("€%.3f"))
    ax.tick_params(axis="both", labelsize=7)
    ax.xaxis.set_major_locator(mticker.MultipleLocator(10))
    ax.xaxis.set_minor_locator(mticker.MultipleLocator(5))
    ax.set_title(fuel_label, fontsize=9, fontweight="bold", pad=3)
    ax.set_xlabel("Giorni dallo shock", fontsize=8)
    ax.set_ylabel("€/L", fontsize=8)
    ax.grid(axis="y", alpha=0.15, linestyle="--")
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(fontsize=7, loc="upper left", framealpha=0.9)


def main():
    args = parse_args()
    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    prices  = pd.read_csv(PRICES_CSV, parse_dates=["date"])
    methods = [m for m in discover_methods() if m in METHOD_COLORS]
    print(f"Metodi trovati: {methods}")

    method_colors = {}
    fallback_idx  = 0
    for m in methods:
        if m in METHOD_COLORS:
            method_colors[m] = METHOD_COLORS[m]
        else:
            method_colors[m] = FALLBACK_COLORS[fallback_idx % len(FALLBACK_COLORS)]
            fallback_idx += 1

    for slug, shock_str, evento_label in EVENTS:
        shock = pd.Timestamp(shock_str)

        fig, axes = plt.subplots(
            1, len(FUELS),
            figsize=(7 * len(FUELS), 4.5),
            facecolor="white",
            gridspec_kw={"wspace": 0.30},
        )
        fig.subplots_adjust(top=0.84, bottom=0.12, left=0.08, right=0.97)

        for ax, (fuel, pump_col, fuel_label) in zip(axes, FUELS):
            draw_panel(
                ax, prices, shock, slug, fuel, methods,
                args.pre, args.post, fuel_label, method_colors,
            )

        fig.suptitle(
            f"{evento_label}\n"
            f"Prezzo alla pompa reale vs controfattuali — finestra −{args.pre}/+{args.post}gg",
            fontsize=11, fontweight="bold",
        )

        out_path = out_dir / f"{slug}.png"
        fig.savefig(out_path, dpi=150, bbox_inches="tight", pad_inches=0.15)
        plt.close(fig)
        print(f"✓  {out_path}")


if __name__ == "__main__":
    main()
