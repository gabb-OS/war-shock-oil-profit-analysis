#!/usr/bin/env python3
"""
plot_fit_pre.py
===============
Per ogni evento: un PNG che mostra come ogni modello ITS ha imparato
la serie prezzi nel periodo PRE-shock (training fit).

  prezzo_reale[t]   = actual
  prezzo_fitted[t]  = actual − residual   (in-sample fit del modello)

Pannello sinistro: benzina — pannello destro: gasolio.
Linea verticale punteggiata al giorno 0 (inizio post).
I giorni sull'asse X sono relativi alla data shock.

Output:
  data/plots/utils/fit_pre/{evento_slug}.png

Uso:
  python3 utils/plot_fit_pre.py
  python3 utils/plot_fit_pre.py --post-days 10   # mostra anche N giorni post
  python3 utils/plot_fit_pre.py --out-dir path/dir
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import pandas as pd

BASE_DIR   = Path(__file__).parent.parent
PRICES_CSV = BASE_DIR / "data" / "processed" / "daily_fuel_prices_stradale.csv"
ITS_DIR    = BASE_DIR / "data" / "plots" / "its" / "fixed"
OUT_DIR    = BASE_DIR / "data" / "plots" / "utils" / "fit_pre"

EVENTS = [
    ("Ucraina_Feb_2022",      "2022-02-24", "Invasione Ucraina — feb 2022"),
    ("Iran-Israele_Giu_2025", "2025-06-13", "Guerra Iran–Israele — giu 2025"),
    ("Hormuz_Feb_2026",       "2026-02-28", "Chiusura Stretto di Hormuz — feb 2026"),
]
FUELS = [
    ("benzina", "benzina_net", "Benzina"),
    ("gasolio", "gasolio_net", "Gasolio"),
]
PRICE_COL = {"benzina": "benzina_net", "gasolio": "gasolio_net"}

METHOD_COLORS = {
    "v1_naive":    "#e74c3c",
    "v3_arima":    "#2980b9",
    "v7_theilsen": "#8e44ad",
    "v8_pymc":     "#e67e22",
}

METHOD_LABELS = {
    "v1_naive":    "OLS Naïve",
    "v3_arima":    "ARIMA",
    "v7_theilsen": "Theil-Sen",
    "v8_pymc":     "PyMC Bayesiano",
}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--post-days", type=int, default=5,
                   help="Giorni post-shock da mostrare per vedere la divergenza (default: 5)")
    p.add_argument("--out-dir", type=Path, default=OUT_DIR)
    return p.parse_args()


def discover_methods() -> list[str]:
    methods = []
    for d in sorted(ITS_DIR.iterdir()):
        if d.is_dir() and d.name in METHOD_COLORS:
            if any(d.glob("residuals_*.csv")):
                methods.append(d.name)
    return methods


def load_residuals(metodo: str, slug: str, fuel: str) -> pd.DataFrame | None:
    path = ITS_DIR / metodo / f"residuals_{slug}_{fuel}.csv"
    if not path.exists():
        return None
    df = pd.read_csv(path, parse_dates=["date"])
    return df.sort_values("date").reset_index(drop=True)


def draw_panel(
    ax,
    prices: pd.DataFrame,
    shock: pd.Timestamp,
    slug: str,
    fuel: str,
    methods: list[str],
    post_days: int,
    fuel_label: str,
):
    price_col = PRICE_COL[fuel]

    # Raccogli l'unione di tutte le date pre da tutti i metodi
    all_pre_dates = set()
    method_data: dict[str, pd.DataFrame] = {}

    for metodo in methods:
        res = load_residuals(metodo, slug, fuel)
        if res is None:
            continue
        pre = res[res["phase"] == "pre"].copy()
        # Aggiungi anche qualche giorno post per vedere la divergenza
        post = res[res["phase"] == "post"].copy()
        post = post[post["date"] <= shock + pd.Timedelta(days=post_days)]
        combined = pd.concat([pre, post]).sort_values("date")
        if combined.empty:
            continue
        method_data[metodo] = combined
        all_pre_dates.update(pre["date"].tolist())

    if not method_data:
        ax.text(0.5, 0.5, "Dati non disponibili",
                transform=ax.transAxes, ha="center", va="center",
                fontsize=9, color="grey")
        ax.set_title(fuel_label, fontsize=9, fontweight="bold")
        return

    # Prezzo reale dal CSV prezzi (non dai residuals)
    start = min(all_pre_dates) - pd.Timedelta(days=1)
    end   = shock + pd.Timedelta(days=post_days)
    win_prices = prices[
        (prices["date"] >= start) & (prices["date"] <= end)
    ].copy()
    win_prices["day"] = (win_prices["date"] - shock).dt.days

    ax.plot(
        win_prices["day"], win_prices[price_col],
        color="black", lw=2.0, ls="-", label="Prezzo netto reale", zorder=6,
    )

    # Linea fitted per ogni metodo: fitted = actual − residual
    for metodo, df in method_data.items():
        df = df.merge(
            prices[["date", price_col]].rename(columns={price_col: "actual"}),
            on="date", how="left",
        )
        df["day"]    = (df["date"] - shock).dt.days
        df["fitted"] = df["actual"] - df["residual"]

        color = METHOD_COLORS[metodo]
        label = METHOD_LABELS.get(metodo, metodo)

        # Pre: linea continua (fit in-sample)
        pre_df = df[df["phase"] == "pre"]
        ax.plot(pre_df["day"], pre_df["fitted"],
                color=color, lw=1.3, ls="-", label=f"{label} (fit)", zorder=4)

        # Post (pochi giorni): tratteggiato per mostrare dove diverge il CF
        post_df = df[df["phase"] == "post"]
        if not post_df.empty:
            # Aggiungi punto di giunzione (ultimo pre) per continuità visiva
            last_pre = pre_df.iloc[[-1]]
            bridge   = pd.concat([last_pre, post_df])
            ax.plot(bridge["day"], bridge["fitted"],
                    color=color, lw=1.0, ls="--", alpha=0.7, zorder=3)

    # Linea verticale shock
    ax.axvline(0, color="black", lw=1.0, ls=":", alpha=0.6)
    ax.text(0.01, 0.97, "◀ training (pre)",
            transform=ax.transAxes, ha="left", va="top",
            fontsize=7, color="#555555", style="italic")
    if post_days > 0:
        ax.text(0.99, 0.97, "post ▶",
                transform=ax.transAxes, ha="right", va="top",
                fontsize=7, color="#555555", style="italic")

    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("€%.3f"))
    ax.tick_params(axis="both", labelsize=7)
    ax.xaxis.set_major_locator(mticker.MultipleLocator(10))
    ax.xaxis.set_minor_locator(mticker.MultipleLocator(5))
    ax.set_title(fuel_label, fontsize=9, fontweight="bold", pad=3)
    ax.set_xlabel("Giorni rispetto allo shock", fontsize=8)
    ax.set_ylabel("€/L", fontsize=8)
    ax.grid(axis="y", alpha=0.15, linestyle="--")
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(fontsize=7, loc="upper left", framealpha=0.92, ncol=1)


def main():
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    prices  = pd.read_csv(PRICES_CSV, parse_dates=["date"])
    methods = discover_methods()
    print(f"Metodi: {methods}")

    for slug, shock_str, evento_label in EVENTS:
        shock = pd.Timestamp(shock_str)

        fig, axes = plt.subplots(
            1, len(FUELS),
            figsize=(7 * len(FUELS), 4.8),
            facecolor="white",
            gridspec_kw={"wspace": 0.28},
        )
        fig.subplots_adjust(top=0.83, bottom=0.12, left=0.08, right=0.97)

        for ax, (fuel, price_col, fuel_label) in zip(axes, FUELS):
            draw_panel(ax, prices, shock, slug, fuel, methods,
                       args.post_days, fuel_label)

        fig.suptitle(
            f"{evento_label}\n"
            f"Fit in-sample dei modelli nel periodo pre-shock"
            f"  (+{args.post_days}gg post per confronto)",
            fontsize=11, fontweight="bold",
        )

        out_path = args.out_dir / f"{slug}.png"
        fig.savefig(out_path, dpi=150, bbox_inches="tight", pad_inches=0.15)
        plt.close(fig)
        print(f"✓  {out_path}")


if __name__ == "__main__":
    main()
