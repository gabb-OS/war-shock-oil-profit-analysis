#!/usr/bin/env python3
"""
plot_cf_margine_post.py
=======================
Per ogni evento: un PNG con benzina (sinistra) e gasolio (destra).
Mostra il margine reale (prezzo netto − futures) vs le controfattuali di
tutti i modelli ITS nel periodo POST-shock (0 → +post gg).

  margin_actual[t] = prezzo_netto[t] − futures[t]
  margin_cf[t]     = margin_actual[t] − residual[t]

Output:
  data/plots/utils/cf_margine_post/{evento_slug}.png

Uso:
  python3 utils/plot_cf_margine_post.py
  python3 utils/plot_cf_margine_post.py --post 60
  python3 utils/plot_cf_margine_post.py --out-dir path/dir
"""

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from conversions import load_eurusd, usd_ton_to_eur_liter, GAS_OIL, EUROBOB
from plot_margini import load_eurobob as _load_eurobob, load_gasoil as _load_gasoil

BASE_DIR   = Path(__file__).parent.parent
PRICES_CSV = BASE_DIR / "data" / "processed" / "daily_fuel_prices_stradale.csv"
EURUSD_CSV = BASE_DIR / "data" / "raw" / "eurusd.csv"
ITS_DIR    = BASE_DIR / "data" / "plots" / "its" / "fixed"
OUT_DIR    = BASE_DIR / "data" / "plots" / "utils" / "cf_margine_post"

EVENTS = [
    ("Ucraina_Feb_2022",      "2022-02-24", "Invasione Ucraina — feb 2022"),
    ("Iran-Israele_Giu_2025", "2025-06-13", "Guerra Iran–Israele — giu 2025"),
    ("Hormuz_Feb_2026",       "2026-02-28", "Chiusura Stretto di Hormuz — feb 2026"),
]
FUELS = [
    ("benzina", "benzina_net", "Benzina"),
    ("gasolio", "gasolio_net", "Gasolio"),
]
PRICE_COL   = {"benzina": "benzina_net",   "gasolio": "gasolio_net"}
FUTURES_COL = {"benzina": "eurobob_eur_l", "gasolio": "gasoil_eur_l"}

METHOD_COLORS = {
    "v1_naive":    "#e74c3c",
    "v3_arima":    "#2980b9",
    "v7_theilsen": "#8e44ad",
    "v8_pymc":     "#e67e22",
}
METHOD_LABELS = {
    "v1_naive":    "OLS Naïve (CF)",
    "v3_arima":    "ARIMA (CF)",
    "v7_theilsen": "Theil-Sen (CF)",
    "v8_pymc":     "PyMC Bayesiano (CF)",
}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--post",    type=int, default=40,
                   help="Giorni post-shock da mostrare (default: 40)")
    p.add_argument("--out-dir", type=Path, default=OUT_DIR)
    return p.parse_args()


def load_futures() -> pd.DataFrame:
    eurusd = load_eurusd(
        csv_path=EURUSD_CSV if EURUSD_CSV.exists() else None,
        start="2015-01-01", end="2026-12-31",
    )
    eurobob_eur = usd_ton_to_eur_liter(_load_eurobob(), eurusd, EUROBOB)
    gasoil_eur  = usd_ton_to_eur_liter(_load_gasoil(),  eurusd, GAS_OIL)
    # ffill dopo il join: i due exchange hanno calendari diversi,
    # l'union degli indici introduce NaN che il reindex non ripara.
    return pd.DataFrame({
        "eurobob_eur_l": eurobob_eur,
        "gasoil_eur_l":  gasoil_eur,
    }).ffill()


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


def build_post_window(
    residuals: pd.DataFrame,
    prices: pd.DataFrame,
    futures: pd.DataFrame,
    shock: pd.Timestamp,
    fuel: str,
    post: int,
) -> pd.DataFrame | None:
    price_col   = PRICE_COL[fuel]
    futures_col = FUTURES_COL[fuel]

    # Solo fase post — l'end viene dai residuals stessi, non dal parametro `post`.
    # Così evitiamo che i giorni fuori dal CSV residuals ricevano residual=0
    # e facciano uno spike artificiale dove la CF torna sull'actual.
    post_res = residuals[residuals["phase"] == "post"].copy()
    post_res = post_res[post_res["date"] <= shock + pd.Timedelta(days=post)]
    if post_res.empty:
        return None
    end = post_res["date"].max()

    # Prezzi nella finestra post (fino all'ultimo giorno coperto dai residuals)
    win = prices[(prices["date"] >= shock) & (prices["date"] <= end)].copy()
    if win.empty:
        return None
    win["day"] = (win["date"] - shock).dt.days

    # Futures con ffill sulla finestra
    fut_win = (
        futures[[futures_col]]
        .reindex(pd.date_range(shock, end, freq="D"), method="ffill")
        .rename_axis("date")
        .reset_index()
    )
    win = win.merge(fut_win, on="date", how="left")
    win = win.merge(post_res[["date", "residual"]], on="date", how="left")
    # Qui non dovrebbero esserci NaN: se ci sono è per date senza residual nel CSV
    win = win.dropna(subset=["residual"])

    win["margin_actual"] = win[price_col] - win[futures_col]
    win["margin_cf"]     = win["margin_actual"] - win["residual"]

    return win[["date", "day", "margin_actual", "margin_cf"]]


def draw_panel(
    ax,
    prices: pd.DataFrame,
    futures: pd.DataFrame,
    shock: pd.Timestamp,
    slug: str,
    fuel: str,
    methods: list[str],
    post: int,
    fuel_label: str,
):
    # Costruisci il margine reale dal primo metodo disponibile
    margin_actual = None
    for metodo in methods:
        res = load_residuals(metodo, slug, fuel)
        if res is None:
            continue
        win = build_post_window(res, prices, futures, shock, fuel, post)
        if win is not None:
            margin_actual = win[["day", "margin_actual"]].drop_duplicates("day")
            break

    if margin_actual is None:
        ax.text(0.5, 0.5, "Dati non disponibili",
                transform=ax.transAxes, ha="center", va="center",
                fontsize=9, color="grey")
        ax.set_title(fuel_label, fontsize=9, fontweight="bold")
        return

    # Linea reale
    ax.plot(margin_actual["day"], margin_actual["margin_actual"],
            color="black", lw=2.2, ls="-", label="Margine reale", zorder=6)

    # Controfattuali per modello
    for metodo in methods:
        res = load_residuals(metodo, slug, fuel)
        if res is None:
            continue
        win = build_post_window(res, prices, futures, shock, fuel, post)
        if win is None or win.empty:
            continue
        color = METHOD_COLORS[metodo]
        label = METHOD_LABELS.get(metodo, metodo)
        ax.plot(win["day"], win["margin_cf"],
                color=color, lw=1.4, ls="-", label=label, zorder=4)

    ax.axhline(0, color="#aaaaaa", lw=0.8, ls="--", zorder=2)
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("€%.3f"))
    ax.tick_params(axis="both", labelsize=7)
    ax.xaxis.set_major_locator(mticker.MultipleLocator(10))
    ax.xaxis.set_minor_locator(mticker.MultipleLocator(5))
    ax.set_title(fuel_label, fontsize=10, fontweight="bold", pad=4)
    ax.set_xlabel("Giorni dallo shock", fontsize=8)
    ax.set_ylabel("Margine (€/L)", fontsize=8)
    ax.grid(axis="y", alpha=0.15, linestyle="--")
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(fontsize=7.5, loc="upper left", framealpha=0.92, ncol=1)


def main():
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    prices  = pd.read_csv(PRICES_CSV, parse_dates=["date"])
    print("Carico futures...")
    futures = load_futures()
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

        for ax, (fuel, _, fuel_label) in zip(axes, FUELS):
            draw_panel(ax, prices, futures, shock, slug, fuel,
                       methods, args.post, fuel_label)

        fig.suptitle(
            f"{evento_label}\n"
            f"Margine reale vs controfattuali — post-shock  (+{args.post}gg)",
            fontsize=11, fontweight="bold",
        )

        out_path = args.out_dir / f"{slug}.png"
        fig.savefig(out_path, dpi=150, bbox_inches="tight", pad_inches=0.15)
        plt.close(fig)
        print(f"✓  {out_path}")


if __name__ == "__main__":
    main()
