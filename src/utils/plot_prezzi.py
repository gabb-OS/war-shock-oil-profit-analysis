#!/usr/bin/env python3
"""
plot_prezzi.py
==============
Quattro grafici:
  1. Prezzi pompa giornalieri (daily_fuel_prices_all.csv)
     vs prezzi settimanali SISEN — benzina e gasolio
  2. Differenza giornaliero − settimanale per entrambi i carburanti
  3. Benzina e gasolio: curva ALL vs curva STRADALE (senza autostrade)
  4. Differenza ALL − STRADALE (premio autostradale)
"""

import re
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd

# ── Percorsi ──────────────────────────────────────────────────────────────────
BASE_DIR      = Path(__file__).parent.parent
print(BASE_DIR)
DAILY_CSV     = BASE_DIR / "data" / "processed" / "daily_fuel_prices_all.csv"
STRADALE_CSV  = BASE_DIR / "data" / "processed" / "daily_fuel_prices_stradale.csv"
SISEN_CSV     = BASE_DIR / "data" / "raw"       / "sisen_prezzi_settimanali.csv"
OUT_DIR       = BASE_DIR / "data" / "plots" / "utils"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Carica giornalieri ────────────────────────────────────────────────────────
daily    = pd.read_csv(DAILY_CSV,    parse_dates=["date"]).sort_values("date")
stradale = pd.read_csv(STRADALE_CSV, parse_dates=["date"]).sort_values("date")
daily    = daily.dropna(subset=["benzina_pump", "gasolio_pump"])
stradale = stradale.dropna(subset=["benzina_pump", "gasolio_pump"])

# ── Carica SISEN settimanale ──────────────────────────────────────────────────
def _parse_it(s: pd.Series) -> pd.Series:
    return (s.astype(str).str.strip()
             .str.replace(r"\.(?=\d{3})", "", regex=True)
             .str.replace(",", ".", regex=False)
             .pipe(pd.to_numeric, errors="coerce"))

raw = pd.read_csv(SISEN_CSV, sep=",", encoding="utf-8-sig",
                  on_bad_lines="skip", engine="python", dtype=str)
raw.columns = [c.strip().lower().replace(" ", "_") for c in raw.columns]
raw = raw.rename(columns={"data_rilevazione": "date",
                           "nome_prodotto":    "fuel",
                           "codice_prodotto":  "codice",
                           "netto":            "netto"})
raw["date"]  = pd.to_datetime(raw["date"], errors="coerce")
raw["prezzo"] = _parse_it(raw["prezzo"])

is_benz = raw["codice"].str.strip() == "1"
is_gas  = raw["codice"].str.strip() == "2"

benz_w = (raw[is_benz][["date", "prezzo"]].copy()
          .rename(columns={"prezzo": "prezzo_benz_w"}))
gas_w  = (raw[is_gas][["date", "prezzo"]].copy()
          .rename(columns={"prezzo": "prezzo_gas_w"}))

sisen = (pd.merge(benz_w, gas_w, on="date", how="outer")
         .sort_values("date")
         .drop_duplicates("date")
         .dropna(subset=["prezzo_benz_w", "prezzo_gas_w"]))

# Converti €/1000L → €/L se necessario
for col in ("prezzo_benz_w", "prezzo_gas_w"):
    if sisen[col].dropna().max() > 10:
        sisen[col] = sisen[col] / 1000

# ── Merge per calcolare differenza ───────────────────────────────────────────
daily["_dt"]  = daily["date"]
sisen["_dt"]  = sisen["date"]
merged = pd.merge_asof(
    daily.sort_values("_dt"),
    sisen[["_dt", "prezzo_benz_w", "prezzo_gas_w"]],
    on="_dt", direction="backward"
).drop(columns=["_dt"])

merged["diff_benz"] = merged["benzina_pump"] - merged["prezzo_benz_w"]
merged["diff_gas"]  = merged["gasolio_pump"] - merged["prezzo_gas_w"]

# ── Stile ─────────────────────────────────────────────────────────────────────
COLORS = {
    "benz_day":      "#E63946",   # rosso acceso  – benzina all
    "benz_week":     "#FF9F9F",   # rosso chiaro  – benzina SISEN
    "benz_str":      "#FF6B35",   # arancione     – benzina stradale
    "gas_day":       "#1D3557",   # blu scuro     – gasolio all
    "gas_week":      "#457B9D",   # blu chiaro    – gasolio SISEN
    "gas_str":       "#2A9D8F",   # verde-acqua   – gasolio stradale
    "diff_benz":     "#E63946",
    "diff_gas":      "#1D3557",
    "diff_aut_benz": "#C77DFF",   # viola         – delta autostradale benzina
    "diff_aut_gas":  "#FF9500",   # arancio       – delta autostradale gasolio
    "zero":          "#AAAAAA",
}

loc  = mdates.AutoDateLocator()
fmt  = mdates.ConciseDateFormatter(loc)

# ════════════════════════════════════════════════════════════════════════════
# GRAFICO 1 – Prezzi pompa giornalieri vs SISEN settimanale
# ════════════════════════════════════════════════════════════════════════════
fig1, (ax_b, ax_g) = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
fig1.suptitle("Prezzi pompa giornalieri vs SISEN settimanale (€/L)", fontsize=14, fontweight="bold")

# — Benzina —
ax_b.plot(daily["date"], daily["benzina_pump"],
          color=COLORS["benz_day"], lw=0.8, label="Benzina – giornaliero (pompa)")
ax_b.step(sisen["date"], sisen["prezzo_benz_w"],
          color=COLORS["benz_week"], lw=1.5, where="post", label="Benzina – settimanale (SISEN)")
ax_b.set_ylabel("€/L")
ax_b.legend(loc="upper right", fontsize=8)
ax_b.grid(axis="y", alpha=0.3)
ax_b.set_title("Benzina", fontsize=10)

# — Gasolio —
ax_g.plot(daily["date"], daily["gasolio_pump"],
          color=COLORS["gas_day"], lw=0.8, label="Gasolio – giornaliero (pompa)")
ax_g.step(sisen["date"], sisen["prezzo_gas_w"],
          color=COLORS["gas_week"], lw=1.5, where="post", label="Gasolio – settimanale (SISEN)")
ax_g.set_ylabel("€/L")
ax_g.legend(loc="upper right", fontsize=8)
ax_g.grid(axis="y", alpha=0.3)
ax_g.set_title("Gasolio", fontsize=10)

for ax in (ax_b, ax_g):
    ax.xaxis.set_major_locator(loc)
    ax.xaxis.set_major_formatter(fmt)

fig1.tight_layout()
out1 = OUT_DIR / "01_prezzi_giornalieri_vs_sisen.png"
fig1.savefig(out1, dpi=150, bbox_inches="tight")
print(f"Salvato: {out1}")

# ════════════════════════════════════════════════════════════════════════════
# GRAFICO 2 – Differenza giornaliero − settimanale
# ════════════════════════════════════════════════════════════════════════════
fig2, (ax_db, ax_dg) = plt.subplots(2, 1, figsize=(14, 7), sharex=True)
fig2.suptitle("Differenza prezzi pompa − SISEN settimanale (€/L)", fontsize=14, fontweight="bold")

for ax, col, color, label in [
    (ax_db, "diff_benz", COLORS["diff_benz"], "Benzina"),
    (ax_dg, "diff_gas",  COLORS["diff_gas"],  "Gasolio"),
]:
    ax.axhline(0, color=COLORS["zero"], lw=1, ls="--")
    ax.fill_between(merged["date"], merged[col], 0,
                    where=merged[col] >= 0, alpha=0.25, color=color, label="+")
    ax.fill_between(merged["date"], merged[col], 0,
                    where=merged[col] < 0,  alpha=0.25, color="green", label="−")
    ax.plot(merged["date"], merged[col], color=color, lw=0.7)

    # Annotazioni statistiche
    mu  = merged[col].mean()
    std = merged[col].std()
    ax.axhline(mu, color=color, lw=1, ls=":", alpha=0.8)
    ax.set_title(f"{label}  |  media={mu:+.4f} €/L   σ={std:.4f} €/L", fontsize=10)
    ax.set_ylabel("Δ €/L")
    ax.grid(axis="y", alpha=0.3)
    ax.xaxis.set_major_locator(loc)
    ax.xaxis.set_major_formatter(fmt)

fig2.tight_layout()
out2 = OUT_DIR / "02_differenza_giornaliero_sisen.png"
fig2.savefig(out2, dpi=150, bbox_inches="tight")
print(f"Salvato: {out2}")

# ════════════════════════════════════════════════════════════════════════════
# GRAFICO 3 – ALL vs STRADALE (senza autostrade)
# ════════════════════════════════════════════════════════════════════════════
fig3, (ax3_b, ax3_g) = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
fig3.suptitle("Prezzi pompa: tutti gli impianti vs senza autostrade (€/L)",
              fontsize=14, fontweight="bold")

for ax, fuel_col, c_all, c_str, label in [
    (ax3_b, "benzina_pump", COLORS["benz_day"], COLORS["benz_str"], "Benzina"),
    (ax3_g, "gasolio_pump", COLORS["gas_day"],  COLORS["gas_str"],  "Gasolio"),
]:
    ax.plot(daily["date"],    daily[fuel_col],    color=c_all, lw=0.8, label="Tutti gli impianti (ALL)")
    ax.plot(stradale["date"], stradale[fuel_col], color=c_str, lw=0.8, ls="--", label="Senza autostrade (STRADALE)")
    ax.set_ylabel("€/L")
    ax.set_title(label, fontsize=10)
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(axis="y", alpha=0.3)
    ax.xaxis.set_major_locator(mdates.AutoDateLocator())
    ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(mdates.AutoDateLocator()))

fig3.tight_layout()
out3 = OUT_DIR / "03_all_vs_stradale.png"
fig3.savefig(out3, dpi=150, bbox_inches="tight")
print(f"Salvato: {out3}")

# ════════════════════════════════════════════════════════════════════════════
# GRAFICO 4 – Differenza ALL − STRADALE (premio autostradale)
# ════════════════════════════════════════════════════════════════════════════

# Merge sui giorni comuni
comp = pd.merge(
    daily[["date", "benzina_pump", "gasolio_pump"]].rename(
        columns={"benzina_pump": "benz_all", "gasolio_pump": "gas_all"}),
    stradale[["date", "benzina_pump", "gasolio_pump"]].rename(
        columns={"benzina_pump": "benz_str", "gasolio_pump": "gas_str"}),
    on="date", how="inner"
)
comp["diff_benz"] = comp["benz_all"] - comp["benz_str"]
comp["diff_gas"]  = comp["gas_all"]  - comp["gas_str"]

fig4, (ax4_b, ax4_g) = plt.subplots(2, 1, figsize=(14, 7), sharex=True)
fig4.suptitle("Differenza prezzi ALL − STRADALE (premio autostradale, €/L)",
              fontsize=14, fontweight="bold")

for ax, col, color, label in [
    (ax4_b, "diff_benz", COLORS["diff_aut_benz"], "Benzina"),
    (ax4_g, "diff_gas",  COLORS["diff_aut_gas"],  "Gasolio"),
]:
    mu  = comp[col].mean()
    std = comp[col].std()
    ax.axhline(0,  color=COLORS["zero"], lw=1, ls="--")
    ax.axhline(mu, color=color,          lw=1, ls=":", alpha=0.9)
    ax.fill_between(comp["date"], comp[col], 0,
                    where=comp[col] >= 0, alpha=0.30, color=color)
    ax.fill_between(comp["date"], comp[col], 0,
                    where=comp[col] <  0, alpha=0.30, color="green")
    ax.plot(comp["date"], comp[col], color=color, lw=0.7)
    ax.set_title(f"{label}  |  media={mu:+.4f} €/L   σ={std:.4f} €/L", fontsize=10)
    ax.set_ylabel("Δ €/L")
    ax.grid(axis="y", alpha=0.3)
    ax.xaxis.set_major_locator(mdates.AutoDateLocator())
    ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(mdates.AutoDateLocator()))

fig4.tight_layout()
out4 = OUT_DIR / "04_differenza_all_stradale.png"
fig4.savefig(out4, dpi=150, bbox_inches="tight")
print(f"Salvato: {out4}")

plt.show()
print("\nDone.")
