"""
02c_zoom_plots.py — Grafici zoom ±4 settimane intorno allo shock
================================================================
Risolve i problemi della figura originale:
  - Scala LOG → scala prezzi reali (EUR/litro o EUR/barile)
  - Finestra enorme → ±4 settimane intorno allo shock
  - Punti minuscoli → scatter s=90 con bordo bianco
  - Doppio asse Y confuso → asse singolo chiaro
  - Se il changepoint τ è fuori dalla finestra → annotazione testuale
  - Δ% medio pre/post shock in sovrimpressione

Input:
  - data/dataset_merged.csv      → prezzi settimanali
  - data/table1_changepoints.csv → risultati MCMC (τ, CI, lag)

Output:
  - plots/02c_zoom_{evento}_{serie}.png      → singolo (9 totali)
  - plots/02c_zoom_combined_{serie}.png      → 3 eventi affiancati (3 totali)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import warnings
warnings.filterwarnings("ignore")

# ─── Configurazione ──────────────────────────────────────────────────────────
ZOOM_WEEKS = 4          # settimane prima e dopo lo shock
DPI        = 180

EVENTS_CFG = {
    "Ucraina (Feb 2022)":      ("2022-02-24", "#e74c3c"),
    "Iran-Israele (Giu 2025)": ("2025-06-13", "#e67e22"),
    "Hormuz (Feb 2026)":       ("2026-02-28", "#8e44ad"),
}

SERIES_CFG = {
    "Brent":   ("brent_7d_eur", "EUR / barile", "#2166ac"),
    "Benzina": ("benzina_4w",   "EUR / litro",  "#d6604d"),
    "Diesel":  ("diesel_4w",    "EUR / litro",  "#31a354"),
}

plt.rcParams.update({
    "font.family":        "serif",
    "font.serif":         ["Times New Roman", "DejaVu Serif", "serif"],
    "axes.spines.top":    False,
    "axes.spines.right":  False,
    "axes.linewidth":     0.9,
    "axes.grid":          True,
    "grid.color":         "#e8e8e8",
    "grid.linewidth":     0.6,
    "grid.linestyle":     "-",
    "xtick.direction":    "out",
    "ytick.direction":    "out",
    "figure.dpi":         DPI,
})

# ─── Carica dati ─────────────────────────────────────────────────────────────
merged = pd.read_csv("data/dataset_merged.csv", index_col=0, parse_dates=True)
cps    = pd.read_csv("data/table1_changepoints.csv")

print(f"Dataset: {len(merged)} settimane  "
      f"({merged.index[0].date()} – {merged.index[-1].date()})")
print(f"Changepoints disponibili: {len(cps)} righe\n")


# ─────────────────────────────────────────────────────────────────────────────
# Core: disegna un pannello zoom su un singolo asse
# ─────────────────────────────────────────────────────────────────────────────
def plot_zoom_panel(ax, event_name, series_name,
                    show_ylabel=True, show_title=True, zoom_weeks=ZOOM_WEEKS):
    """
    Disegna il grafico zoom ±zoom_weeks settimane intorno allo shock.

    Elementi:
      • Shading pre (leggero) / post (più scuro) dello shock
      • Linea + scatter grandi per ogni punto settimanale
      • Linea verticale continua per lo shock
      • Linea verticale tratteggiata per τ̂ (se nella finestra)
        oppure annotazione testuale (se fuori finestra)
      • Banda CI 95% per τ̂ (se in finestra)
      • Box Δ% medio pre/post
    """
    shock_date_str, shock_color = EVENTS_CFG[event_name]
    col, unit, fuel_color       = SERIES_CFG[series_name]
    shock_dt = pd.Timestamp(shock_date_str)

    # ── Filtra finestra ±zoom_weeks settimane
    win_start = shock_dt - pd.Timedelta(weeks=zoom_weeks)
    win_end   = shock_dt + pd.Timedelta(weeks=zoom_weeks)
    df_zoom   = merged.loc[win_start:win_end, col].dropna()

    if df_zoom.empty:
        ax.text(0.5, 0.5, "Dati non\ndisponibili",
                ha="center", va="center", transform=ax.transAxes,
                fontsize=12, color="#888888")
        ax.set_title(f"{event_name}\n{series_name}", fontsize=10,
                     fontweight="bold")
        return

    # ── Shading pre/post shock
    ax.axvspan(df_zoom.index[0], shock_dt,
               color=shock_color, alpha=0.05, zorder=1, label="_nolegend_")
    ax.axvspan(shock_dt, df_zoom.index[-1],
               color=shock_color, alpha=0.11, zorder=1, label="_nolegend_")

    # ── Linea di collegamento + punti grandi
    ax.plot(df_zoom.index, df_zoom.values,
            color=fuel_color, lw=2.0, alpha=0.55, zorder=3)
    ax.scatter(df_zoom.index, df_zoom.values,
               color=fuel_color, s=95, zorder=5,
               edgecolors="white", linewidths=1.4,
               label=f"{series_name} (settimanale)")

    # ── Linea shock (verticale continua)
    ax.axvline(shock_dt, color=shock_color, lw=2.8, linestyle="-",
               zorder=6, label=f"Shock  {shock_date_str}")

    # ── Changepoint τ̂: in finestra → linea + CI; fuori → box testo
    row = cps[(cps["Evento"] == event_name) & (cps["Serie"] == series_name)]
    if not row.empty:
        cp_date = pd.Timestamp(str(row["tau"].values[0]))
        lag_gg  = int(row["Lag (gg)"].values[0])
        ci_lo   = pd.Timestamp(str(row["CI_95_lo"].values[0]))
        ci_hi   = pd.Timestamp(str(row["CI_95_hi"].values[0]))

        if win_start <= cp_date <= win_end:
            # τ̂ visibile nella finestra
            ax.axvline(cp_date, color="#2980b9", lw=2.2, linestyle="--",
                       zorder=4, label=f"τ̂ = {cp_date.strftime('%-d %b %Y')}")
            # CI band (solo la parte nel window)
            ax.axvspan(max(ci_lo, win_start), min(ci_hi, win_end),
                       color="#2980b9", alpha=0.13, zorder=2,
                       label=f"CI 95% τ̂")
        else:
            # τ̂ fuori finestra → annotazione
            lag_sign = "prima" if lag_gg < 0 else "dopo"
            ax.text(0.985, 0.97,
                    f"τ̂ = {cp_date.strftime('%-d %b %Y')}\n"
                    f"({abs(lag_gg)} gg {lag_sign} shock)",
                    ha="right", va="top", transform=ax.transAxes,
                    fontsize=8.5, color="#2980b9",
                    bbox=dict(boxstyle="round,pad=0.35", fc="white",
                              ec="#2980b9", alpha=0.90, lw=1.2))

    # ── Δ% pre/post shock
    pre_vals  = df_zoom[df_zoom.index <  shock_dt]
    post_vals = df_zoom[df_zoom.index >= shock_dt]
    if len(pre_vals) > 0 and len(post_vals) > 0:
        delta_pct = (post_vals.mean() - pre_vals.mean()) / pre_vals.mean() * 100
        sign      = "+" if delta_pct >= 0 else ""
        color_d   = "#c0392b" if delta_pct > 0.5 else (
                    "#27ae60" if delta_pct < -0.5 else "#555555")
        ax.text(0.015, 0.97,
                f"Δ media: {sign}{delta_pct:.1f}%",
                ha="left", va="top", transform=ax.transAxes,
                fontsize=10, fontweight="bold", color=color_d,
                bbox=dict(boxstyle="round,pad=0.35", fc="white",
                          ec="#cccccc", alpha=0.90, lw=0.8))

    # ── Assi e tick
    # X: una tacca per settimana, formato "3 Feb\n2026"
    ax.xaxis.set_major_locator(mdates.WeekdayLocator(byweekday=0, interval=1))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%-d %b\n%Y"))
    ax.tick_params(axis="x", labelsize=9, rotation=0)
    ax.tick_params(axis="y", labelsize=10)

    if show_ylabel:
        ax.set_ylabel(unit, fontsize=11)
    ax.set_xlabel("")

    # Y range con margine generoso
    y_vals   = df_zoom.values
    y_rng    = y_vals.max() - y_vals.min()
    y_margin = y_rng * 0.30 if y_rng > 0 else y_vals.mean() * 0.03
    ax.set_ylim(y_vals.min() - y_margin, y_vals.max() + y_margin)

    # ── Titolo
    if show_title:
        ax.set_title(
            f"{event_name}  —  {series_name}",
            fontsize=11, fontweight="bold", pad=8,
        )

    # ── Legenda compatta
    ax.legend(fontsize=8.5, loc="lower right",
              framealpha=0.92, edgecolor="#cccccc",
              handlelength=1.5, labelspacing=0.4)


# ─────────────────────────────────────────────────────────────────────────────
# 1. PLOT SINGOLI: uno per ogni combinazione evento × serie (9 totali)
# ─────────────────────────────────────────────────────────────────────────────
print("── Plot singoli ─────────────────────────────────────────────────")
for event_name in EVENTS_CFG:
    for series_name in SERIES_CFG:
        fig, ax = plt.subplots(figsize=(8, 4.8))
        plot_zoom_panel(ax, event_name, series_name)
        fig.tight_layout(pad=1.5)

        safe_e = (event_name
                  .replace(" ", "_")
                  .replace("(", "").replace(")", "")
                  .replace("/", ""))
        safe_s = series_name.lower()
        fname  = f"plots/02c_zoom_{safe_e}_{safe_s}.png"
        fig.savefig(fname, dpi=DPI, bbox_inches="tight")
        plt.close(fig)
        print(f"  Salvato: {fname}")


# ─────────────────────────────────────────────────────────────────────────────
# 2. PLOT COMBINATI: 3 eventi affiancati per ogni serie (3 figure totali)
#    Formato ideale per paper: confronto diretto dei tre shock per un carburante
# ─────────────────────────────────────────────────────────────────────────────
print("\n── Plot combinati (3 eventi × 1 serie) ──────────────────────────")
event_list  = list(EVENTS_CFG.keys())
shock_colors = [EVENTS_CFG[e][1] for e in event_list]

for series_name, (col, unit, fuel_color) in SERIES_CFG.items():
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.2), sharey=False)
    fig.subplots_adjust(wspace=0.30)

    for idx, (ax, event_name) in enumerate(zip(axes, event_list)):
        plot_zoom_panel(
            ax, event_name, series_name,
            show_ylabel=(idx == 0),
            show_title=True,
        )

    fig.suptitle(
        f"Prezzi {series_name}  ·  ±{ZOOM_WEEKS} settimane intorno allo shock  "
        f"({unit})\n"
        f"Punti = osservazione settimanale  |  "
        f"Linea continua = data shock  |  "
        f"Tratteggio = changepoint τ̂ (se nella finestra)",
        fontsize=11, fontweight="bold", y=1.035,
    )
    fig.tight_layout(pad=1.8)

    safe_s = series_name.lower()
    fname  = f"plots/02c_zoom_combined_{safe_s}.png"
    fig.savefig(fname, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  Salvato: {fname}")


# ─────────────────────────────────────────────────────────────────────────────
# 3. MEGA-GRIGLIA: tutti e 3 i carburanti × tutti e 3 gli eventi
#    Un colpo d'occhio su tutta la matrice evento/serie
# ─────────────────────────────────────────────────────────────────────────────
print("\n── Mega-griglia 3×3 ─────────────────────────────────────────────")
series_list = list(SERIES_CFG.keys())

fig, axes = plt.subplots(3, 3, figsize=(18, 13))
fig.subplots_adjust(wspace=0.28, hspace=0.55)

for row_idx, series_name in enumerate(series_list):
    for col_idx, event_name in enumerate(event_list):
        ax = axes[row_idx, col_idx]
        plot_zoom_panel(
            ax, event_name, series_name,
            show_ylabel=(col_idx == 0),
            show_title=True,
        )

fig.suptitle(
    f"Effetto degli shock sui prezzi energetici — zoom ±{ZOOM_WEEKS} settimane\n"
    f"Scala: prezzi reali (non logaritmici)  |  "
    f"Δ% = variazione media pre/post shock",
    fontsize=13, fontweight="bold", y=1.01,
)
fig.tight_layout(pad=2.0)
fig.savefig("plots/02c_zoom_grid_3x3.png", dpi=DPI, bbox_inches="tight")
plt.close(fig)
print("  Salvato: plots/02c_zoom_grid_3x3.png  (griglia 3×3 completa)")

plt.rcParams.update(plt.rcParamsDefault)

print("\nScript 02c completato.")
print("  Singoli:   plots/02c_zoom_{evento}_{serie}.png  (9 file)")
print("  Combinati: plots/02c_zoom_combined_{brent|benzina|diesel}.png  (3 file)")
print("  Griglia:   plots/02c_zoom_grid_3x3.png  (1 file — overview completo)")