#!/usr/bin/env python3
"""
05_barplot_costo.py
════════════════════════════════════════════════════════════════════════
Genera un PNG per ogni combinazione evento × metodologia trovata in
  data/plots/its/fixed/{metodo}/residuals_{evento}_{carburante}.csv

Ogni figura: 2 colonne (benzina | gasolio) × 2 righe (barplot + cumulato).

Logica prezzi:
  price_actual[t] = prezzo netto reale da daily_fuel_prices_stradale.csv
  price_cf[t]     = price_actual[t] - residual[t]   (controfatuale)
  overpaid[t]     = (price_actual[t] - price_cf[t]) × litri[t]

Uso:
  python3 05_barplot_costo.py               # genera tutti i grafici
  python3 05_barplot_costo.py --mode fixed  # solo modalità 'fixed' (default)
  python3 05_barplot_costo.py --out-dir path/to/dir
"""

import argparse
import re
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE_DIR     = Path(__file__).parent
PRICES_CSV   = BASE_DIR / "data" / "processed" / "daily_fuel_prices_stradale.csv"
CONSUMI_CSV  = BASE_DIR / "data" / "consumi" / "consumi_giornalieri.csv"
ITS_DIR      = BASE_DIR / "data" / "plots" / "its"
DEFAULT_OUT  = BASE_DIR / "data" / "plots" / "cost_comparison"

FUELS = ["benzina", "gasolio"]

# Colonne prezzi netti da usare (al netto tasse)
PRICE_COL = {"benzina": "benzina_net",  "gasolio": "gasolio_net"}
PUMP_COL  = {"benzina": "benzina_pump", "gasolio": "gasolio_pump"}
VOL_COL   = {"benzina": "benzina_L",    "gasolio": "gasolio_L"}

METHOD_LABEL = {
    "v1_naive":        "OLS Naïve",
    "v3_arima":        "ARIMA",
    "v5_causalimpact": "BSTS CausalImpact",
    "v6_glm_gamma":    "GLM Gamma",
    "v7_theilsen":     "Theil-Sen",
    "v8_pymc":         "PyMC (Bayesiano)",
}

BAR_RED   = "#c0392b"
BAR_GREEN = "#27ae60"
BAR_GREY  = "#b0bec5"
CUM_RED   = "#7f0000"


# ── CLI ────────────────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--mode",    default="fixed",
                   help="Sotto-cartella ITS: fixed | detected/margin | detected/price (default: fixed)")
    p.add_argument("--out-dir", type=Path, default=DEFAULT_OUT,
                   help="Directory di output per i PNG")
    return p.parse_args()


# ── Carica dati base ───────────────────────────────────────────────────────────
def load_prices() -> pd.DataFrame:
    df = pd.read_csv(PRICES_CSV, parse_dates=["date"])
    return df.set_index("date")


def load_consumi() -> pd.DataFrame:
    df = pd.read_csv(CONSUMI_CSV, parse_dates=["data"])
    return df.rename(columns={"data": "date"}).set_index("date")


# ── Scopri combinazioni evento × metodo ───────────────────────────────────────
def discover_combos(mode: str) -> dict[tuple[str, str], dict[str, Path]]:
    """
    Ritorna { (evento_slug, metodo): {carburante: Path} }
    Usa i file in data/plots/its/{mode}/{metodo}/residuals_{evento}_{carburante}.csv
    """
    mode_dir = ITS_DIR / mode.replace("/", "/")   # es. ITS_DIR / "fixed"
    combos: dict[tuple[str, str], dict[str, Path]] = {}

    if not mode_dir.exists():
        raise FileNotFoundError(f"Cartella mode non trovata: {mode_dir}")

    pattern = re.compile(r"^residuals_(.+)_(benzina|gasolio)\.csv$")

    for method_dir in sorted(mode_dir.iterdir()):
        if not method_dir.is_dir():
            continue
        metodo = method_dir.name
        for csv_path in sorted(method_dir.glob("residuals_*.csv")):
            m = pattern.match(csv_path.name)
            if not m:
                continue
            evento_slug, carburante = m.group(1), m.group(2)
            key = (evento_slug, metodo)
            combos.setdefault(key, {})[carburante] = csv_path

    return combos


# ── Prepara dati per una combo + carburante ───────────────────────────────────
def prepare_fuel_data(
    csv_path: Path,
    prices: pd.DataFrame,
    consumi: pd.DataFrame,
    fuel: str,
) -> pd.DataFrame | None:
    df = pd.read_csv(csv_path, parse_dates=["date"])
    post = df[df["phase"] == "post"].sort_values("date").reset_index(drop=True)

    if post.empty:
        return None

    # Prezzi reali giornalieri (netto + pompa)
    post = post.join(prices[[PRICE_COL[fuel], PUMP_COL[fuel]]], on="date")
    post = post.join(consumi[[VOL_COL[fuel]]], on="date")
    post = post.rename(columns={
        PRICE_COL[fuel]: "price_actual",
        PUMP_COL[fuel]:  "price_pump",
        VOL_COL[fuel]:   "volume",
    })

    # Rimuovi giorni senza prezzo o volume
    post = post.dropna(subset=["price_actual", "volume"]).reset_index(drop=True)
    if post.empty:
        return None

    post["price_cf"]      = post["price_actual"] - post["residual"]
    post["cost_actual"]   = post["price_actual"] * post["volume"]
    post["cost_cf"]       = post["price_cf"]     * post["volume"]
    post["cost_pump"]     = post["price_pump"]   * post["volume"]
    post["overpaid_day"]  = post["cost_actual"]  - post["cost_cf"]

    return post


# ── Pannello line + area ──────────────────────────────────────────────────────
def plot_line_panel(ax, post: pd.DataFrame, fuel: str, metodo: str) -> dict:
    dates = post["date"]
    p_act = post["price_actual"]   # €/L reale
    p_cf  = post["price_cf"]       # €/L controfattuale
    ov_m  = post["overpaid_day"] / 1e6   # M€/giorno

    # ── Asse sinistro: prezzi €/L ────────────────────────────────────────────
    ax.plot(dates, p_act, color="#c0392b", lw=1.6, zorder=4,
            label="Prezzo netto reale (€/L)")
    ax.plot(dates, p_cf,  color="#2c3e50", lw=1.6, ls="--", zorder=4,
            label="Prezzo controfattuale (€/L)")

    # Area tra le due linee: rosso quando pagato di più, verde quando meno
    ax.fill_between(dates, p_act, p_cf,
                    where=(p_act >= p_cf),
                    color=BAR_RED,   alpha=0.25, zorder=2,
                    label="Extra-costo (shock)")
    ax.fill_between(dates, p_act, p_cf,
                    where=(p_act < p_cf),
                    color=BAR_GREEN, alpha=0.25, zorder=2,
                    label="Risparmio (shock)")

    ax.set_ylabel("Prezzo netto €/L", fontsize=8, color="#2c3e50")
    ax.tick_params(axis="y", labelcolor="#2c3e50", labelsize=7)
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("€%.3f"))

    # ── Asse destro: M€/giorno ────────────────────────────────────────────────
    ax2 = ax.twinx()
    ax2.bar(dates, ov_m, width=pd.Timedelta(hours=16),
            color=np.where(ov_m >= 0, BAR_RED, BAR_GREEN),
            alpha=0.35, zorder=1)
    ax2.axhline(0, color="grey", lw=0.6, ls=":")
    ax2.set_ylabel("M€ / giorno (extra-costo)", fontsize=8, color="#7f0000")
    ax2.tick_params(axis="y", labelcolor="#7f0000", labelsize=7)
    ax2.yaxis.set_major_formatter(mticker.FormatStrFormatter("%+.1fM"))
    ax2.spines[["top"]].set_visible(False)

    # ── Totali ────────────────────────────────────────────────────────────────
    total_actual = post["cost_actual"].sum()
    total_cf     = post["cost_cf"].sum()
    overpaid     = total_actual - total_cf

    textbox = (
        f"Tot. CF:       {total_cf/1e6:>8.1f} M€\n"
        f"Tot. effettivo:{total_actual/1e6:>8.1f} M€\n"
        f"Tot. extra:   {overpaid/1e6:>+8.1f} M€"
    )
    ax.text(0.015, 0.97, textbox, transform=ax.transAxes,
            va="top", ha="left", fontsize=7, fontfamily="monospace",
            bbox=dict(boxstyle="round,pad=0.4", facecolor="white",
                      edgecolor="#90a4ae", linewidth=0.7))

    ax.set_title(
        f"{fuel.capitalize()} — {METHOD_LABEL.get(metodo, metodo)}\n"
        "Linea rossa = prezzo reale  ·  Linea grigia = prezzo CF  ·  "
        "Area = guadagno/perdita in M€",
        fontsize=8, fontweight="bold",
    )
    lines1, labels1 = ax.get_legend_handles_labels()
    ax.legend(lines1, labels1, fontsize=6.5, loc="upper right")
    ax.grid(axis="y", alpha=0.13, linestyle="--")
    ax.tick_params(axis="x", rotation=90, labelsize=6)
    ax.spines[["top", "right"]].set_visible(False)

    return {"total_actual": total_actual, "total_cf": total_cf, "overpaid": overpaid}


# ── Pannello confronto spesa: esentasse vs controfattuale ────────────────────
def plot_cum_panel(ax, post: pd.DataFrame, fuel: str) -> None:
    dates       = post["date"]
    cost_actual = post["cost_actual"] / 1e6   # spesa al prezzo netto reale (esentasse)
    cost_cf     = post["cost_cf"]     / 1e6   # spesa al prezzo controfattuale

    ax.plot(dates, cost_actual, color="#c0392b", lw=1.6, zorder=4,
            label="Spesa esentasse reale (M€)")
    ax.plot(dates, cost_cf,     color="#2c3e50", lw=1.4, ls="--", zorder=4,
            label="Spesa al prezzo CF (M€)")

    ax.fill_between(dates, cost_actual, cost_cf,
                    where=(cost_actual >= cost_cf),
                    color=BAR_RED,   alpha=0.22, zorder=2, label="Extra-costo shock")
    ax.fill_between(dates, cost_actual, cost_cf,
                    where=(cost_actual < cost_cf),
                    color=BAR_GREEN, alpha=0.22, zorder=2, label="Risparmio shock")

    mean_act = cost_actual.mean()
    mean_cf  = cost_cf.mean()
    ax.annotate(f"media {mean_act:.1f} M€/g",
                xy=(dates.iloc[-1], cost_actual.iloc[-1]),
                xytext=(-90, 6), textcoords="offset points",
                fontsize=7, fontweight="bold", color="#c0392b",
                arrowprops=dict(arrowstyle="->", color="#c0392b", lw=0.7))
    ax.annotate(f"media {mean_cf:.1f} M€/g",
                xy=(dates.iloc[-1], cost_cf.iloc[-1]),
                xytext=(-90, -14), textcoords="offset points",
                fontsize=7, fontweight="bold", color="#2c3e50",
                arrowprops=dict(arrowstyle="->", color="#2c3e50", lw=0.7))

    ax.set_ylabel("M€ / giorno", fontsize=8)
    ax.set_title("Spesa giornaliera: prezzo netto esentasse vs prezzo controfattuale", fontsize=7.5)
    ax.legend(fontsize=6.5, loc="upper left")
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.1f"))
    ax.grid(axis="y", alpha=0.15, linestyle="--")
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(axis="x", rotation=90, labelsize=6)


# ── Genera figura per una combo evento × metodo ───────────────────────────────
def generate_figure(
    evento_slug: str,
    metodo: str,
    fuel_paths: dict[str, Path],
    prices: pd.DataFrame,
    consumi: pd.DataFrame,
    out_dir: Path,
) -> None:
    fuel_data: dict[str, pd.DataFrame | None] = {}
    for fuel in FUELS:
        if fuel not in fuel_paths:
            fuel_data[fuel] = None
            continue
        fuel_data[fuel] = prepare_fuel_data(fuel_paths[fuel], prices, consumi, fuel)

    available = [f for f in FUELS if fuel_data[f] is not None]
    if not available:
        print(f"  ⚠  nessun dato post per {evento_slug} / {metodo} — salto")
        return

    n_cols = len(available)
    fig, axes = plt.subplots(
        2, n_cols,
        figsize=(9 * n_cols, 10),
        gridspec_kw={"height_ratios": [3, 1.2]},
        facecolor="white",
        squeeze=False,
    )

    evento_label = evento_slug.replace("_", " ")

    for col, fuel in enumerate(available):
        post = fuel_data[fuel]
        plot_line_panel(axes[0, col], post, fuel, metodo)
        plot_cum_panel(axes[1, col], post, fuel)

    metodo_label = METHOD_LABEL.get(metodo, metodo)
    fig.suptitle(
        f"Evento: {evento_label}  ·  Metodo: {metodo_label}\n"
        "Prezzi reali giornalieri netti (al netto imposte)",
        fontsize=11, fontweight="bold", y=1.01,
    )
    fig.tight_layout(h_pad=2.5, w_pad=3)

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"barplot_{evento_slug}_{metodo}.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓  {out_path.name}")


# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    args = parse_args()

    prices  = load_prices()
    consumi = load_consumi()

    combos = discover_combos(args.mode)
    if not combos:
        print(f"Nessun file residuals trovato in data/plots/its/{args.mode}/")
        return

    print(f"Trovate {len(combos)} combinazioni evento×metodo in modalità '{args.mode}'")
    print(f"Output in: {args.out_dir}\n")

    for (evento_slug, metodo), fuel_paths in sorted(combos.items()):
        generate_figure(evento_slug, metodo, fuel_paths, prices, consumi, args.out_dir)

    print(f"\n✓  Completato — {len(combos)} grafici generati.")


if __name__ == "__main__":
    main()
