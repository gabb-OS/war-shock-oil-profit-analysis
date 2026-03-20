#!/usr/bin/env python3
"""
utils/plot_wholesale_comparison.py
===================================
Quattro grafici separati:

  Grafico 1a – Gasolio pompa (€/L) vs London Gas Oil Futures (€/L, convertiti)
  Grafico 1b – Benzina pompa (€/L) vs Eurobob Futures (€/L, convertiti)
               Stesso asse Y in €/L, dal 2017 in poi.

  Grafico 2 – Decomposizione prezzo finale gasolio (area stackata)
               Componenti: wholesale · crack+distribuzione · accisa · IVA · margine retailer

  Grafico 3 – Stessa decomposizione per benzina

Conversione USD/ton → €/L
  Usa conversions.py: approccio molare + densità ICE + EUR/USD storico.
  Se data/raw/eurusd.csv non esiste → fallback medie annuali BCE.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd

# Importa il modulo di conversione fisico-chimica
import sys
sys.path.insert(0, str(Path(__file__).parent))
from conversions import (
    GAS_OIL, EUROBOB as EUROBOB_HC,
    load_eurusd, usd_ton_to_eur_liter,
    print_conversion_summary,
)

# ── Configurazione ─────────────────────────────────────────────────────────────
BASE_DIR     = Path(__file__).parent.parent
DAILY_CSV    = BASE_DIR / "data" / "processed" / "daily_fuel_prices_all.csv"
SISEN_CSV    = BASE_DIR / "data" / "raw"       / "sisen_prezzi_settimanali.csv"
GASOIL_CSV   = BASE_DIR / "data" / "Futures"   / "London Gas Oil Futures Historical Data.csv"
EUROBOB_CSV  = BASE_DIR / "data" / "Futures"   / "Eurobob_B7H1_date.csv"
EURUSD_CSV   = BASE_DIR / "data" / "raw"       / "eurusd.csv"   # scarica da investing.com
OUT_DIR      = BASE_DIR / "data" / "plots" / "wholesale"
OUT_DIR.mkdir(parents=True, exist_ok=True)

START_DATE   = "2017-01-01"

# Colori
C = {
    "wholesale":    "#2C3E50",   # blu scuro     – materia prima (Gas Oil/Eurobob)
    "crack":        "#27AE60",   # verde         – crack spread + distribuzione
    "accisa":       "#E74C3C",   # rosso         – accisa (imposta fissa)
    "iva":          "#E67E22",   # arancio       – IVA (imposta percentuale)
    "margin":       "#9B59B6",   # viola         – margine retailer giornaliero
    "pump":         "#1A1A2E",   # quasi nero    – prezzo pompa (linea verifica)
    "futures":      "#3498DB",   # celeste       – future line destra
}


# ══════════════════════════════════════════════════════════════════════════════
# Caricamento dati
# ══════════════════════════════════════════════════════════════════════════════

def _parse_it(s: pd.Series) -> pd.Series:
    return (s.astype(str).str.strip()
             .str.replace(r"\.(?=\d{3})", "", regex=True)
             .str.replace(",", ".", regex=False)
             .pipe(pd.to_numeric, errors="coerce"))


def load_daily() -> pd.DataFrame:
    df = pd.read_csv(DAILY_CSV, parse_dates=["date"]).sort_values("date")
    df = df[df["date"] >= START_DATE]
    return df.set_index("date")


def load_sisen() -> pd.DataFrame:
    """Ritorna DataFrame con date settimanali e colonne accisa_b, iva_b, netto_b, prezzo_b (benzina)
    e accisa_g, iva_g, netto_g, prezzo_g (gasolio auto), in €/L."""
    raw = pd.read_csv(SISEN_CSV, sep=",", encoding="utf-8-sig",
                      on_bad_lines="skip", engine="python", dtype=str)
    raw.columns = [c.strip().lower().replace(" ", "_") for c in raw.columns]
    raw = raw.rename(columns={"data_rilevazione": "date", "codice_prodotto": "codice",
                               "nome_prodotto": "fuel"})
    raw["date"] = pd.to_datetime(raw["date"], errors="coerce")
    for col in ("prezzo", "accisa", "iva", "netto"):
        if col in raw.columns:
            raw[col] = _parse_it(raw[col])

    def pivot_fuel(code: str, suffix: str) -> pd.DataFrame:
        sub = raw[raw["codice"].str.strip() == code][["date","prezzo","accisa","iva","netto"]].copy()
        sub.columns = ["date"] + [f"{c}_{suffix}" for c in ("prezzo","accisa","iva","netto")]
        sub = sub.drop_duplicates("date").sort_values("date")
        # Converti €/1000L → €/L se necessario
        for col in sub.columns[1:]:
            if sub[col].dropna().max() > 10:
                sub[col] = sub[col] / 1000
        return sub

    benz = pivot_fuel("1", "b")
    gas  = pivot_fuel("2", "g")
    merged = pd.merge(benz, gas, on="date", how="outer").sort_values("date")
    merged = merged[merged["date"] >= START_DATE]
    return merged.set_index("date")


def load_futures(path: Path, eurusd: pd.Series, hc) -> pd.DataFrame:
    """Carica futures e converte in €/L via conversions.py.

    Supporta tre layout CSV:
      • Inglese  (Gas Oil):  colonne 'Date' (MM/DD/YYYY) e 'Price' (USD/ton, virgola migliaia)
      • Italiano (Eurobob):  colonne 'data' e 'chiusura'.
          - data: investing.com esporta MM/DD/YYYY oppure DD/MM/YYYY → auto-detect
          - chiusura: formato italiano "1.234,56" OPPURE già numerico "1025.129"
    """
    df = pd.read_csv(path, encoding="utf-8-sig", dtype=str)
    cols_lower = {c.strip().lower(): c for c in df.columns}

    # ── Colonna data ──────────────────────────────────────────────────────────
    if "date" in cols_lower:
        # formato inglese: MM/DD/YYYY
        df["date"] = pd.to_datetime(df[cols_lower["date"]], format="%m/%d/%Y", errors="coerce")
    elif "data" in cols_lower:
        raw_dates = df[cols_lower["data"]].str.strip()
        # Auto-detect: prova MM/DD/YYYY (investing.com recente), poi DD/MM/YYYY
        parsed = pd.to_datetime(raw_dates, format="%m/%d/%Y", errors="coerce")
        if parsed.isna().mean() > 0.3:   # troppe date non parsate → prova l'altro formato
            parsed = pd.to_datetime(raw_dates, format="%d/%m/%Y", errors="coerce")
        df["date"] = parsed
    else:
        raise KeyError(f"Colonna data non trovata in {path.name}. Colonne: {list(df.columns)}")

    # ── Colonna prezzo ────────────────────────────────────────────────────────
    if "price" in cols_lower:
        # Gas Oil: separatore migliaia = virgola, decimale = punto  →  "1,234.56"
        raw_price = df[cols_lower["price"]].str.replace(",", "", regex=False)
    elif "chiusura" in cols_lower:
        raw_col = df[cols_lower["chiusura"]].str.strip()
        # Distingui formato:
        #   italiano classico "1.234,56": contiene virgola → strip punto-migliaia, virgola→punto
        #   numerico inglese  "1025.129": nessuna virgola  → già parsabile, non toccare i punti
        has_comma = raw_col.str.contains(",", na=False).any()
        if has_comma:
            # italiano: il punto prima di esattamente 3 cifre seguite da virgola è migliaia
            raw_price = (raw_col
                         .str.replace(r"\.(?=\d{3},)", "", regex=True)  # sep. migliaia
                         .str.replace(",", ".", regex=False))            # virgola → punto
        else:
            # già in formato numerico con punto decimale (investing.com export recente)
            raw_price = raw_col
    else:
        raise KeyError(f"Colonna prezzo non trovata in {path.name}. Colonne: {list(df.columns)}")

    df["price_usd_ton"] = pd.to_numeric(raw_price, errors="coerce")
    df = df.dropna(subset=["date", "price_usd_ton"]).sort_values("date")
    df = df[df["date"] >= START_DATE].set_index("date")

    prices = df["price_usd_ton"]
    df["price_eurl"] = usd_ton_to_eur_liter(prices, eurusd, hc)
    return df[["price_usd_ton", "price_eurl"]]


def merge_daily_sisen(daily: pd.DataFrame, sisen: pd.DataFrame) -> pd.DataFrame:
    """Merge_asof giornaliero ← settimanale SISEN (backward)."""
    d = daily.reset_index().sort_values("date")
    s = sisen.reset_index().sort_values("date")
    merged = pd.merge_asof(d, s, on="date", direction="backward")
    return merged.set_index("date")


# ══════════════════════════════════════════════════════════════════════════════
# Grafico 1 (generico) – Pompa vs Futures
# ══════════════════════════════════════════════════════════════════════════════

def plot_pump_vs_futures(
    daily: pd.DataFrame,
    futures: pd.DataFrame,
    fuel: str,           # "gasolio" | "benzina"
    pump_col: str,       # colonna in daily, es. "gasolio_pump" | "benzina_pump"
    hc,                  # oggetto hydrocarbon da conversions.py (GAS_OIL | EUROBOB_HC)
    futures_label: str,  # es. "Gas Oil" | "Eurobob"
    fig_id: str,         # es. "1a" | "1b"
) -> None:
    fuel_label = fuel.capitalize()

    fig, axes = plt.subplots(2, 1, figsize=(14, 9), sharex=True)
    fig.suptitle(
        f"Prezzo pompa {fuel_label} (€/L) vs {futures_label} Futures convertiti (€/L)\n"
        f"dal {START_DATE}  —  conversione: {hc.l_per_ton_eff:.1f} L/ton "
        f"(ρ_ICE={hc.rho_eff} kg/L) · EUR/USD storico",
        fontsize=12, fontweight="bold"
    )

    # Allinea per date comuni — usiamo price_eurl (€/L)
    df = daily[[pump_col]].join(futures[["price_eurl"]], how="left")
    df["price_eurl"] = df["price_eurl"].interpolate("time")

    # ── Pannello superiore: entrambe le curve in €/L sullo stesso asse ─────────
    ax1 = axes[0]
    ax1.plot(df.index, df[pump_col],       color=C["pump"],    lw=1.0,
             label=f"{fuel_label} pompa (€/L)  [tasse incluse]")
    ax1.plot(df.index, df["price_eurl"],   color=C["futures"], lw=0.9, ls="--", alpha=0.85,
             label=f"{futures_label} Futures (€/L)  [{hc.l_per_ton_eff:.0f} L/ton · EUR/USD storico]")
    ax1.set_ylabel("€/L", fontsize=9)
    ax1.set_title("Andamento assoluto — stesso asse €/L", fontsize=9)
    ax1.grid(axis="y", alpha=0.2)
    ax1.legend(fontsize=8, loc="upper left")

    # ── Pannello inferiore: spread (pompa €/L – wholesale €/L) ───────────────
    ax2 = axes[1]
    spread = df[pump_col] - df["price_eurl"]
    ax2.fill_between(df.index, spread, 0,
                     where=spread >= 0, color=C["crack"],  alpha=0.30)
    ax2.fill_between(df.index, spread, 0,
                     where=spread <  0, color="red",       alpha=0.30)
    ax2.plot(df.index, spread, color=C["crack"], lw=0.8)
    ax2.axhline(spread.mean(), color="grey", lw=0.8, ls=":",
                label=f"media {spread.mean():+.3f} €/L")
    ax2.axhline(0, color="black", lw=0.5)
    ax2.set_ylabel("Δ €/L  (pompa − wholesale)", fontsize=9)
    ax2.set_title(
        f"Spread pompa − wholesale (tasse incluse vs futures €/L)  "
        "— nota: spread include accisa + IVA + margine",
        fontsize=8
    )
    ax2.legend(fontsize=8)
    ax2.grid(axis="y", alpha=0.2)
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

    fig.tight_layout()
    out = OUT_DIR / f"01{fig_id}_{fuel}_pompa_vs_futures.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Salvato: {out}")


# ══════════════════════════════════════════════════════════════════════════════
# Grafico 2/3 – Decomposizione stackata prezzo finale
# ══════════════════════════════════════════════════════════════════════════════

def plot_decomposition(
    merged: pd.DataFrame,
    futures: pd.DataFrame | None,
    fuel: str,           # "gasolio" | "benzina"
    suffix: str,         # "g" | "b"
    fig_num: int,
) -> None:
    """
    Decomposizione stackata:
      ┌──────────────────────────────────────────────────────────────┐
      │ Δ daily−SISEN  (rumore timing tra media MIMIT e SISEN)       │  ← pump - prezzo_sisen
      │ IVA                                                          │
      │ Accisa                                                       │
      │ Margine ind.+logistica+distribuzione  (crack spread)        │  ← NETTO_sisen - wholesale
      │ Wholesale (futures €/L)                                      │  ← Gas Oil / Eurobob
      └──────────────────────────────────────────────────────────────┘

    Note:
    - "Margine ind.+logistica+distribuzione" include tutto ciò che è tra
      il prezzo all'ingrosso ICE e il prezzo al netto delle tasse SISEN:
      margine raffineria, costi trasporto, distribuzione, retailer.
      Non è possibile separare queste componenti senza dati aggiuntivi.
    - "Δ daily−SISEN" è molto piccolo (~0-5 €/L cent) e rappresenta solo
      il disallineamento tra la rilevazione settimanale SISEN e la media
      giornaliera MIMIT (campioni e orari di rilevazione diversi).
    - Se wholesale > netto_sisen (shock estremi), il crack è negativo:
      viene mostrato in rosso come componente "sotto zero".
    """
    pump_col   = f"{'gasolio' if suffix=='g' else 'benzina'}_pump"
    prezzo_col = f"prezzo_{suffix}"
    accisa_col = f"accisa_{suffix}"
    iva_col    = f"iva_{suffix}"
    netto_col  = f"netto_{suffix}"

    df = merged[[pump_col, prezzo_col, accisa_col, iva_col, netto_col]].dropna()

    # Merge con futures se disponibile
    has_futures = futures is not None and not futures.empty
    if has_futures:
        fut = futures[["price_eurl"]].rename(columns={"price_eurl": "wholesale"})
        # forward-fill è più robusto di nearest+tolerance per buchi weekend/festivi
        fut = fut.reindex(df.index, method="ffill")
        df = df.join(fut)
        df["wholesale"] = df["wholesale"].interpolate("time").clip(lower=0)
        # crack può essere negativo (futures > netto): lo mostriamo senza clip
        df["crack"] = df[netto_col] - df["wholesale"]
    else:
        df["wholesale"] = 0.0
        df["crack"]     = df[netto_col]

    # Δ daily−SISEN: piccolo rumore tra campionamenti diversi, NON margine retailer
    df["delta_timing"] = df[pump_col] - df[prezzo_col]

    # ── Figure ─────────────────────────────────────────────────────────────────
    fuel_label   = fuel.capitalize()
    fut_label    = "Gas Oil" if suffix == "g" else "Eurobob"
    fig, (ax_stack, ax_line) = plt.subplots(
        2, 1, figsize=(14, 10), sharex=True,
        gridspec_kw={"height_ratios": [2, 1]}
    )
    fig.suptitle(
        f"Decomposizione prezzo {fuel_label} alla pompa (€/L)\n"
        + (f"Wholesale: {fut_label} futures convertiti (€/L · EUR/USD storico)"
           if has_futures else "Wholesale non disponibile – mostrato solo prezzo netto SISEN"),
        fontsize=11, fontweight="bold"
    )

    # Resample settimanale per leggibilità (troppi giorni per stacked)
    dw = df.resample("W").mean()

    # ── Pannello superiore: stack ──────────────────────────────────────────────
    # Stack order (bottom→top): wholesale → crack_pos → accisa → iva → delta_pos
    bottoms = np.zeros(len(dw))

    if has_futures:
        # Wholesale
        w_vals = dw["wholesale"].clip(lower=0).fillna(0).values
        ax_stack.bar(dw.index, w_vals, bottom=bottoms, width=7,
                     color=C["wholesale"], label="Wholesale (futures €/L)", alpha=0.85, linewidth=0)
        bottoms += w_vals

        # Crack positivo
        crack_pos = dw["crack"].clip(lower=0).fillna(0).values
        ax_stack.bar(dw.index, crack_pos, bottom=bottoms, width=7,
                     color=C["crack"], label="Margine ind.+logistica+distribuzione", alpha=0.85, linewidth=0)
        bottoms += crack_pos

        # Crack negativo (futures > netto): barra rossa che scende
        crack_neg = dw["crack"].clip(upper=0).fillna(0).values
        if crack_neg.any():
            ax_stack.bar(dw.index, crack_neg, bottom=bottoms + crack_neg,
                         width=7, color="red", label="Crack negativo (futures > netto)", alpha=0.5, linewidth=0)
    else:
        netto_vals = dw[netto_col].clip(lower=0).fillna(0).values
        ax_stack.bar(dw.index, netto_vals, bottom=bottoms, width=7,
                     color=C["crack"], label="Prezzo netto ex-tasse (SISEN)", alpha=0.85, linewidth=0)
        bottoms += netto_vals

    # Accisa
    acc_vals = dw[accisa_col].clip(lower=0).fillna(0).values
    ax_stack.bar(dw.index, acc_vals, bottom=bottoms, width=7,
                 color=C["accisa"], label="Accisa (imposta fissa)", alpha=0.85, linewidth=0)
    bottoms += acc_vals

    # IVA
    iva_vals = dw[iva_col].clip(lower=0).fillna(0).values
    ax_stack.bar(dw.index, iva_vals, bottom=bottoms, width=7,
                 color=C["iva"], label="IVA (22%)", alpha=0.85, linewidth=0)
    bottoms += iva_vals

    # Δ daily−SISEN (positivo e negativo separati)
    dt_pos = dw["delta_timing"].clip(lower=0).fillna(0).values
    dt_neg = dw["delta_timing"].clip(upper=0).fillna(0).values
    ax_stack.bar(dw.index, dt_pos, bottom=bottoms, width=7,
                 color=C["margin"], label="Δ daily−SISEN (+)", alpha=0.70, linewidth=0)
    if dt_neg.any():
        ax_stack.bar(dw.index, dt_neg, bottom=bottoms,
                     width=7, color="purple", label="Δ daily−SISEN (−)", alpha=0.40, linewidth=0)

    # Linea prezzo pompa come verifica chiusura stack
    ax_stack.plot(df.index, df[pump_col], color=C["pump"], lw=0.7,
                  label=f"Prezzo pompa {fuel_label} (verifica)", zorder=5)

    ax_stack.set_ylabel("€/L", fontsize=9)
    ax_stack.legend(fontsize=7.5, loc="upper left", ncol=2)
    ax_stack.grid(axis="y", alpha=0.2)
    ax_stack.set_title(f"Composizione prezzo {fuel_label} (media settimanale)", fontsize=9)

    # ── Pannello inferiore: % di ogni componente sul totale ───────────────────
    total = dw[pump_col].clip(lower=0.01)
    if has_futures:
        ax_line.plot(dw.index, dw["wholesale"] / total * 100,
                     color=C["wholesale"], lw=0.9, label="% Wholesale")
        ax_line.plot(dw.index, dw["crack"].clip(lower=0) / total * 100,
                     color=C["crack"], lw=0.9, ls="--", label="% Margine+Dist.")
    else:
        ax_line.plot(dw.index, dw[netto_col] / total * 100,
                     color=C["crack"], lw=0.9, label="% Netto SISEN")

    ax_line.plot(dw.index, dw[accisa_col] / total * 100,
                 color=C["accisa"], lw=0.9, label="% Accisa")
    ax_line.plot(dw.index, dw[iva_col] / total * 100,
                 color=C["iva"], lw=0.9, label="% IVA")
    ax_line.plot(dw.index, dw["delta_timing"] / total * 100,
                 color=C["margin"], lw=0.7, ls=":", alpha=0.7, label="% Δ daily−SISEN")
    ax_line.axhline(0, color="black", lw=0.4)
    ax_line.set_ylabel("% del prezzo finale", fontsize=9)
    ax_line.set_title("Incidenza % di ogni componente sul prezzo pompa", fontsize=9)
    ax_line.legend(fontsize=7.5, ncol=3)
    ax_line.grid(alpha=0.2)
    ax_line.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.0f%%"))
    ax_line.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

    fig.tight_layout()
    out = OUT_DIR / f"0{fig_num}_decomposizione_{fuel}.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Salvato: {out}")


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    print_conversion_summary()

    print("\nCarico dati...")
    daily  = load_daily()
    sisen  = load_sisen()
    eurusd = load_eurusd(
        csv_path=EURUSD_CSV if EURUSD_CSV.exists() else None,
        start=START_DATE, end="2026-12-31"
    )

    gasoil  = load_futures(GASOIL_CSV,  eurusd, GAS_OIL)
    eurobob = load_futures(EUROBOB_CSV, eurusd, EUROBOB_HC) if EUROBOB_CSV.exists() else None

    print(f"  Daily:   {daily.index.min().date()} → {daily.index.max().date()}")
    print(f"  SISEN:   {sisen.index.min().date()} → {sisen.index.max().date()}")
    print(f"  Gas Oil: {gasoil.index.min().date()} → {gasoil.index.max().date()}")
    if eurobob is not None and not eurobob.empty:
        print(f"  Eurobob: {eurobob.index.min().date()} → {eurobob.index.max().date()}")

    # Merge giornaliero + SISEN
    merged = merge_daily_sisen(daily, sisen)

    # ── Grafico 1a: gasolio pompa vs Gas Oil Futures ──────────────────────────
    print("\nGrafico 1a: gasolio pompa vs Gas Oil Futures...")
    plot_pump_vs_futures(
        daily, gasoil,
        fuel="gasolio", pump_col="gasolio_pump",
        hc=GAS_OIL, futures_label="Gas Oil",
        fig_id="a",
    )

    # ── Grafico 1b: benzina pompa vs Eurobob Futures ──────────────────────────
    if eurobob is not None and not eurobob.empty:
        print("Grafico 1b: benzina pompa vs Eurobob Futures...")
        plot_pump_vs_futures(
            daily, eurobob,
            fuel="benzina", pump_col="benzina_pump",
            hc=EUROBOB_HC, futures_label="Eurobob",
            fig_id="b",
        )
    else:
        print("⚠  Eurobob non disponibile — grafico 1b saltato.")

    # ── Grafico 2: decomposizione gasolio ─────────────────────────────────────
    print("Grafico 2: decomposizione gasolio...")
    plot_decomposition(merged, gasoil, "gasolio", "g", fig_num=2)

    # ── Grafico 3: decomposizione benzina ─────────────────────────────────────
    print("Grafico 3: decomposizione benzina...")
    # Eurobob molto corto (2021-2022 only) → uso None se periodo troppo breve
    eb = eurobob if (eurobob is not None and len(eurobob) > 100) else None
    plot_decomposition(merged, eb, "benzina", "b", fig_num=3)

    print("\nDone. Output in:", OUT_DIR)


if __name__ == "__main__":
    main()