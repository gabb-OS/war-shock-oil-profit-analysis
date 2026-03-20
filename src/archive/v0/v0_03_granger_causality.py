"""
03_granger_causality.py — paper quality (versione migliorata)
==============================================================
Miglioramenti rispetto all'originale:
  1. Figura a DUE PANNELLI affiancati (Benzina + Diesel) in un'unica immagine
  2. Barre con gradiente di significatività (rosso scuro → più significativo)
  3. Annotazioni p-value più pulite (sopra la barra, con asterischi)
  4. Linea α e soglia fisica con label integrate nel plot (no legend ridondante)
  5. rcParams paper-quality coerenti con 02_changepoint
  6. Tabelle riassuntive dei lag significativi stampate a console
  7. Figure singole per carburante mantenute per compatibilità
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap
from statsmodels.tsa.stattools import grangercausalitytests, adfuller
import warnings
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
# rcParams paper-quality
# ─────────────────────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family":        "serif",
    "font.serif":         ["Times New Roman", "DejaVu Serif", "serif"],
    "axes.spines.top":    False,
    "axes.spines.right":  False,
    "axes.linewidth":     0.9,
    "axes.grid":          True,
    "grid.color":         "#e0e0e0",
    "grid.linewidth":     0.5,
    "grid.linestyle":     "-",
    "xtick.direction":    "out",
    "ytick.direction":    "out",
    "legend.framealpha":  0.9,
    "legend.edgecolor":   "#cccccc",
    "figure.dpi":         180,
})

MAX_LAG = 8
ALPHA   = 0.05
DPI     = 180

# ─────────────────────────────────────────────────────────────────────────────
# Dati e test ADF
# ─────────────────────────────────────────────────────────────────────────────
merged = pd.read_csv("data/dataset_merged.csv", index_col=0, parse_dates=True)

print("TEST ADF (stazionarietà)\n" + "─" * 40)
for col in ["log_brent", "log_benzina", "log_diesel"]:
    if col in merged.columns:
        p = adfuller(merged[col].dropna(), autolag="AIC")[1]
        print(f"  ADF {col}: p = {p:.4f}  {'[stazionario]' if p < 0.05 else '[→ uso diff]'}")

merged["d_log_brent"]   = merged["log_brent"].diff()
merged["d_log_benzina"] = merged["log_benzina"].diff()
merged["d_log_diesel"]  = merged["log_diesel"].diff()
merged.dropna(inplace=True)

# ─────────────────────────────────────────────────────────────────────────────
# Granger causality test
# ─────────────────────────────────────────────────────────────────────────────
print("\nGRANGER: Brent → Pompa\n" + "=" * 50)

granger_results = {}
for fuel_col, fuel_name in [("d_log_benzina", "Benzina"), ("d_log_diesel", "Diesel")]:
    data2 = merged[[fuel_col, "d_log_brent"]].dropna()
    try:
        gc = grangercausalitytests(data2, maxlag=MAX_LAG, verbose=False)
    except Exception as e:
        print(f"  Errore {fuel_name}: {e}")
        continue

    rows = []
    for lag in range(1, MAX_LAG + 1):
        f_stat, p_val = gc[lag][0]["ssr_ftest"][:2]
        rows.append({
            "lag_weeks": lag,
            "lag_days":  lag * 7,
            "F_stat":    round(f_stat, 4),
            "p_value":   round(p_val, 4),
            "significant": p_val < ALPHA,
        })
        stars = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else ""
        flag  = "  ← H₀ rifiutata" if (p_val < ALPHA and lag * 7 < 30) else ""
        print(f"  {fuel_name} lag={lag}sett ({lag*7}gg): F={f_stat:.3f}  "
              f"p={p_val:.4f} {stars}{flag}")

    granger_results[fuel_name] = pd.DataFrame(rows)
    granger_results[fuel_name].to_csv(f"data/granger_{fuel_name.lower()}.csv", index=False)


# ─────────────────────────────────────────────────────────────────────────────
# Helper: colore barra per significatività e direzione (rosso = più sign.)
# ─────────────────────────────────────────────────────────────────────────────
def _bar_color(p):
    """
    Mappa il p-value su un colore:
      p < 0.001  → rosso scuro   (#8b1a1a)
      p < 0.01   → rosso medio   (#c0392b)
      p < 0.05   → arancio-rosso (#e74c3c)
      p ≥ 0.05   → grigio blu    (#95a5a6)
    """
    if p < 0.001:
        return "#8b1a1a"
    elif p < 0.01:
        return "#c0392b"
    elif p < 0.05:
        return "#e74c3c"
    else:
        return "#95a5a6"


def _stars(p):
    if p < 0.001: return "***"
    if p < 0.01:  return "**"
    if p < 0.05:  return "*"
    return ""


# ─────────────────────────────────────────────────────────────────────────────
# Helper: disegna un singolo pannello Granger
# ─────────────────────────────────────────────────────────────────────────────
def _granger_panel(ax, df_gc, fuel_name, fuel_color):
    """
    Pannello paper-quality per Granger causality:
      - barre con colore per significatività
      - p-value + asterischi sopra ogni barra
      - linea α tratteggiata con label integrata
      - soglia fisica 30gg con label e shading
    """
    lags_d = df_gc["lag_days"].values
    pvals  = df_gc["p_value"].values
    bar_w  = 5.5

    bar_colors = [_bar_color(p) for p in pvals]
    ax.yaxis.grid(True, color="#e0e0e0", linewidth=0.5, linestyle="-")
    ax.set_axisbelow(True)
    bars = ax.bar(lags_d, pvals,
                  color=bar_colors, edgecolor="black",
                  linewidth=0.6, alpha=0.88, width=bar_w, zorder=3)

    # ── Linea α = 0.05
    ax.axhline(ALPHA, color="#2c3e50", lw=1.4, linestyle="--", zorder=4)
    ax.text(lags_d[-1] + 1, ALPHA + 0.005,
            f"α = {ALPHA}", ha="right", va="bottom",
            fontsize=8, color="#2c3e50", style="italic")

    # ── Soglia fisica 30 giorni
    ax.axvline(30, color="#e67e22", lw=1.6, linestyle="--", zorder=4)
    ax.axvspan(0, 30, alpha=0.06, color="#e74c3c", zorder=1)
    ax.text(15, ax.get_ylim()[1] * 0.97 if ax.get_ylim()[1] > 0.05 else 0.14,
            "< 30 gg\n(speculazione)", ha="center", va="top",
            fontsize=7, color="#c0392b", style="italic")

    # ── Annotazioni sopra le barre: p-value + asterischi
    y_max_data = max(pvals) * 1.30
    for bar, p in zip(bars, pvals):
        bx   = bar.get_x() + bar.get_width() / 2
        by   = bar.get_height()
        star = _stars(p)
        # p-value numerico
        ax.text(bx, by + y_max_data * 0.02,
                f"{p:.3f}", ha="center", va="bottom",
                fontsize=6.8,
                color=_bar_color(p) if p < ALPHA else "#777777",
                fontweight="bold" if p < ALPHA else "normal")
        # asterischi (sopra il p-value)
        if star:
            ax.text(bx, by + y_max_data * 0.09,
                    star, ha="center", va="bottom",
                    fontsize=8, color=_bar_color(p), fontweight="bold")

    # ── Decorazioni
    ax.set_xlabel("Lag (settimane → giorni)", fontsize=10)
    ax.set_ylabel("p-value  (F-test)", fontsize=10)
    ax.set_xticks(lags_d)
    ax.set_xticklabels(
        [f"{int(d)}d\n(w{int(d)//7})" for d in lags_d],
        fontsize=8,
    )
    ax.tick_params(axis="y", labelsize=8)
    ax.set_ylim(0, max(y_max_data, 0.18))

    # ── Titolo
    n_sig_30 = sum(p < ALPHA and d <= 30 for p, d in zip(pvals, lags_d))
    ax.set_title(
        f"Granger Causality: Brent → {fuel_name}\n"
        f"Lag significativi < 30 gg: {n_sig_30}  "
        f"(H₀ rifiutata p < {ALPHA})",
        fontsize=11, fontweight="bold",
        color="#b03030" if n_sig_30 > 0 else "black",
        pad=6,
    )

    # ── Legenda significatività
    legend_patches = [
        mpatches.Patch(color="#8b1a1a", label="p < 0.001  (***)"),
        mpatches.Patch(color="#c0392b", label="p < 0.01   (**)"),
        mpatches.Patch(color="#e74c3c", label="p < 0.05   (*)"),
        mpatches.Patch(color="#95a5a6", label="p ≥ 0.05   (n.s.)"),
    ]
    ax.legend(handles=legend_patches, fontsize=7.5,
              loc="upper right", title="Significatività",
              title_fontsize=7.5)


# ─────────────────────────────────────────────────────────────────────────────
# FIGURA COMBINATA: Benzina + Diesel affiancati (paper-quality)
# ─────────────────────────────────────────────────────────────────────────────
if len(granger_results) == 2:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=False)
    fig.subplots_adjust(wspace=0.35)

    fuel_colors = {"Benzina": "#d6604d", "Diesel": "#31a354"}
    for ax, fuel_name in zip(axes, ["Benzina", "Diesel"]):
        if fuel_name in granger_results:
            _granger_panel(ax, granger_results[fuel_name],
                           fuel_name, fuel_colors[fuel_name])

    fig.suptitle(
        "Granger Causality: Brent Crude Oil → Prezzi Carburanti Italia\n"
        "Barre rosse = lag significativi  |  Soglia fisica 30 gg = tempo minimo di raffinazione/distribuzione",
        fontsize=11, fontweight="bold", y=1.03,
    )
    fig.tight_layout(pad=1.5)
    fig.savefig("plots/03_granger_combined.png", dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print("\n  Salvato: plots/03_granger_combined.png  (figura combinata Benzina+Diesel)")


# ─────────────────────────────────────────────────────────────────────────────
# FIGURE SINGOLE per carburante (compatibilità con pipeline)
# ─────────────────────────────────────────────────────────────────────────────
fuel_colors = {"Benzina": "#d6604d", "Diesel": "#31a354"}
for fuel_name, df_gc in granger_results.items():
    fig, ax = plt.subplots(figsize=(8, 5))
    _granger_panel(ax, df_gc, fuel_name, fuel_colors.get(fuel_name, "#3498db"))
    fig.tight_layout(pad=1.2)
    fig.savefig(f"plots/03_granger_{fuel_name.lower()}.png",
                dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  Salvato: plots/03_granger_{fuel_name.lower()}.png")


# ─────────────────────────────────────────────────────────────────────────────
# Tabella riassuntiva: lag significativi < 30 giorni
# ─────────────────────────────────────────────────────────────────────────────
print("\nRIASSUNTO LAG SIGNIFICATIVI (p < 0.05 E lag < 30 gg):\n" + "─" * 50)
for fuel_name, df_gc in granger_results.items():
    sig = df_gc[(df_gc["p_value"] < ALPHA) & (df_gc["lag_days"] <= 30)]
    if not sig.empty:
        for _, row in sig.iterrows():
            print(f"  {fuel_name}: lag={int(row['lag_weeks'])}sett "
                  f"({int(row['lag_days'])}gg)  p={row['p_value']:.4f}  "
                  f"F={row['F_stat']:.3f}  {_stars(row['p_value'])}")
    else:
        print(f"  {fuel_name}: nessun lag significativo < 30 gg")

# Ripristina rcParams
plt.rcParams.update(plt.rcParamsDefault)

print("\nScript 03 completato.")