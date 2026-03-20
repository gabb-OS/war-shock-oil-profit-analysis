"""
07b_compare_methods.py — CONFRONTO TRE METODI
==============================================
Confronta sistematicamente:
  1. Yield fisso (Brent × 0.45 / 0.52)
  2. Crack spread Eurobob (benzina)
  3. Crack spread Gas Oil (diesel)

Produce:
  - data/table2d_method_comparison.csv  → matrice completa confronto
  - plots/07b_comparison_methods.png    → barre affiancate per evento
  - plots/07b_correlation_matrix.png    → heatmap correlazioni Δmargini
  - plots/07b_scatter_yield_vs_crack.png → scatter yield vs crack

Usage:
  python 07b_compare_methods.py
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

DPI = 180

print("="*80)
print("CONFRONTO METODI: Yield vs Crack Spread")
print("="*80)

# ─────────────────────────────────────────────────────────────────────────────
# 1. CARICA CSV DA SCRIPT 07
# ─────────────────────────────────────────────────────────────────────────────
csv_all = "data/table2_margini_anomaly_all.csv"

try:
    df = pd.read_csv(csv_all)
    print(f"\nCaricato: {csv_all} ({len(df)} righe)")
except FileNotFoundError:
    print(f"\nERRORE: {csv_all} non trovato")
    print("Esegui prima: python 07_margine_speculation_test.py --method=all")
    exit(1)

# ─────────────────────────────────────────────────────────────────────────────
# 2. RISTRUTTURA DATI PER CONFRONTO
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "─"*80)
print("RISTRUTTURAZIONE DATI")
print("─"*80)

# Pivot: evento × serie × metodo → Δmargine
pivot_rows = []

for evento in df["Evento"].unique():
    for serie in df["Serie"].unique():
        row = {"Evento": evento, "Serie": serie}
        
        # Yield
        y = df[(df["Evento"]==evento) & (df["Serie"]==serie) & 
               (df["Metodo"].str.contains("Yield", case=False, na=False))]
        if len(y) > 0:
            row["Yield_delta"] = y["delta_margine_eur"].values[0]
            row["Yield_clas"]  = y["classificazione"].values[0]
            row["Yield_ks_p"]  = y["ks_p"].values[0]
        else:
            row["Yield_delta"] = np.nan
            row["Yield_clas"]  = ""
            row["Yield_ks_p"]  = np.nan
        
        # Crack
        c = df[(df["Evento"]==evento) & (df["Serie"]==serie) & 
               (df["Metodo"].str.contains("Crack", case=False, na=False))]
        if len(c) > 0:
            row["Crack_delta"] = c["delta_margine_eur"].values[0]
            row["Crack_clas"]  = c["classificazione"].values[0]
            row["Crack_ks_p"]  = c["ks_p"].values[0]
        else:
            row["Crack_delta"] = np.nan
            row["Crack_clas"]  = ""
            row["Crack_ks_p"]  = np.nan
        
        # Aggiungi solo se almeno un metodo disponibile
        if not (np.isnan(row.get("Yield_delta", np.nan)) and 
                np.isnan(row.get("Crack_delta", np.nan))):
            pivot_rows.append(row)

df_pivot = pd.DataFrame(pivot_rows)
print(f"  Comparazioni possibili: {len(df_pivot)}")

# ─────────────────────────────────────────────────────────────────────────────
# 3. ANALISI CONSISTENZA CLASSIFICAZIONI
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "="*80)
print("CONSISTENZA CLASSIFICAZIONI")
print("="*80)

n_totali = 0
n_consistenti = 0
dettagli = []

for _, row in df_pivot.iterrows():
    # Skip se uno dei due metodi mancante
    if pd.isna(row.get("Yield_delta")) or pd.isna(row.get("Crack_delta")):
        continue
    
    n_totali += 1
    
    # Estrai categoria principale (rimuovi dettagli dopo "—")
    y_cat = str(row["Yield_clas"]).split("—")[0].strip().split("(")[0].strip()
    c_cat = str(row["Crack_clas"]).split("—")[0].strip().split("(")[0].strip()
    
    consistente = (y_cat == c_cat)
    if consistente:
        n_consistenti += 1
        simbolo = "ACCORDO"
    else:
        simbolo = "DIVERGENZA"
    
    ev_short = row["Evento"].split("(")[0].strip()
    print(f"\n  {ev_short:15} | {row['Serie']:7}")
    print(f"    Yield: Δ={row['Yield_delta']:+.5f}  →  {y_cat}")
    print(f"    Crack: Δ={row['Crack_delta']:+.5f}  →  {c_cat}")
    print(f"    → {simbolo}")
    
    dettagli.append({
        "Evento": ev_short,
        "Serie": row["Serie"],
        "Yield_delta": row["Yield_delta"],
        "Yield_clas": y_cat,
        "Crack_delta": row["Crack_delta"],
        "Crack_clas": c_cat,
        "Consistente": consistente,
    })

# Sommario
print("\n" + "─"*80)
if n_totali > 0:
    pct = 100 * n_consistenti / n_totali
    print(f"  Consistenza: {n_consistenti}/{n_totali} ({pct:.0f}%)")
    if pct >= 80:
        print("  ALTA consistenza — metodi convergono, risultati robusti")
    elif pct >= 60:
        print("  MODERATA consistenza — alcune divergenze, cautela")
    else:
        print("  BASSA consistenza — metodi divergenti, possibile basis risk")
else:
    print("  Nessun confronto disponibile")

pd.DataFrame(dettagli).to_csv("data/table2d_method_comparison.csv", index=False)
print(f"\n  Salvato: data/table2d_method_comparison.csv")

# ─────────────────────────────────────────────────────────────────────────────
# 4. CORRELAZIONE Δmargini
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "="*80)
print("CORRELAZIONE Δmargini")
print("="*80)

# Filtra coppie valide
valid_pairs = [(row["Yield_delta"], row["Crack_delta"]) 
               for _, row in df_pivot.iterrows()
               if not (pd.isna(row.get("Yield_delta")) or pd.isna(row.get("Crack_delta")))]

if len(valid_pairs) > 0:
    y_vals, c_vals = zip(*valid_pairs)
    corr = np.corrcoef(y_vals, c_vals)[0, 1]
    print(f"  Correlazione Pearson: r = {corr:.3f}")
    if abs(corr) > 0.8:
        print("    FORTE correlazione — metodi coerenti")
    elif abs(corr) > 0.5:
        print("    MODERATA correlazione")
    else:
        print("    BASSA correlazione — metodi divergenti")
else:
    print("  Nessuna coppia valida per correlazione")
    corr = np.nan

# ─────────────────────────────────────────────────────────────────────────────
# 5. PLOT 1: BARRE AFFIANCATE PER EVENTO
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "="*80)
print("Generazione plot...")
print("="*80)

fig, ax = plt.subplots(figsize=(16, 10))

# Prepara dati
labels = []
deltas_yield = []
deltas_crack = []
clas_yield = []
clas_crack = []

for _, row in df_pivot.iterrows():
    ev_short = row["Evento"].split("(")[0].strip()
    labels.append(f"{ev_short}\n{row['Serie']}")
    deltas_yield.append(row.get("Yield_delta", np.nan))
    deltas_crack.append(row.get("Crack_delta", np.nan))
    clas_yield.append(row.get("Yield_clas", ""))
    clas_crack.append(row.get("Crack_clas", ""))

n_groups = len(labels)
y_pos = np.arange(n_groups)
width = 0.38

# Colori
colors_y = {"Yield": "#2166ac"}
colors_c = {"Crack": "#e67e22"}

# Barre
offset_y = -width/2
offset_c = +width/2

bars_y = ax.barh(y_pos + offset_y, deltas_yield, height=width,
                 label="Yield (Brent)", color="#2166ac",
                 alpha=0.85, edgecolor="black", lw=0.7)

bars_c = ax.barh(y_pos + offset_c, deltas_crack, height=width,
                 label="Crack spread", color="#e67e22",
                 alpha=0.85, edgecolor="black", lw=0.7)

# Annotazioni classificazioni
for i, (y_d, c_d, y_c, c_c) in enumerate(zip(deltas_yield, deltas_crack, clas_yield, clas_crack)):
    if not np.isnan(y_d):
        if "SPECULAZIONE" in str(y_c):
            y_col = "#8b1a1a"
        elif "NEUTRO" in str(y_c) or "ANTICIPAZIONE" in str(y_c):
            y_col = "#27ae60"
        elif "COMPRESSIONE" in str(y_c):
            y_col = "#2980b9"
        else:
            y_col = "#555555"
        
        x_txt = y_d + (0.004 if y_d >= 0 else -0.004)
        ha_txt = "left" if y_d >= 0 else "right"
        y_short = str(y_c).split("—")[0].strip()[:20]
        ax.text(x_txt, y_pos[i] + offset_y, y_short,
                fontsize=7, va="center", ha=ha_txt, color=y_col,
                fontweight="bold" if "SPECULAZIONE" in str(y_c) else "normal")
    
    if not np.isnan(c_d):
        if "SPECULAZIONE" in str(c_c):
            c_col = "#8b1a1a"
        elif "NEUTRO" in str(c_c) or "ANTICIPAZIONE" in str(c_c):
            c_col = "#27ae60"
        elif "COMPRESSIONE" in str(c_c):
            c_col = "#2980b9"
        else:
            c_col = "#555555"
        
        x_txt = c_d + (0.004 if c_d >= 0 else -0.004)
        ha_txt = "left" if c_d >= 0 else "right"
        c_short = str(c_c).split("—")[0].strip()[:20]
        ax.text(x_txt, y_pos[i] + offset_c, c_short,
                fontsize=7, va="center", ha=ha_txt, color=c_col,
                fontweight="bold" if "SPECULAZIONE" in str(c_c) else "normal")

# Decorazioni
ax.axvline(0, color="black", lw=1.2, alpha=0.5)
ax.set_yticks(y_pos)
ax.set_yticklabels(labels, fontsize=9)
ax.set_xlabel("Δ margine post-shock (EUR/litro)", fontsize=12, fontweight="bold")
ax.set_title(
    f"Confronto metodologie: Yield (Brent) vs Crack spread (Eurobob/Gas Oil)\n"
    f"Consistenza classificazioni: {n_consistenti}/{n_totali} ({pct:.0f}%)  |  "
    f"Correlazione Δmargini: r={corr:.3f}\n"
    "Rosso=SPECULAZIONE | Verde=NEUTRO | Blu=COMPRESSIONE",
    fontsize=12, fontweight="bold", pad=12
)
ax.legend(fontsize=11, loc="lower right")
ax.grid(axis="x", alpha=0.3)

plt.tight_layout()
plt.savefig("plots/07b_comparison_methods.png", dpi=DPI, bbox_inches="tight")
plt.close()
print("  Salvato: plots/07b_comparison_methods.png")

# ─────────────────────────────────────────────────────────────────────────────
# 6. PLOT 2: SCATTER YIELD vs CRACK
# ─────────────────────────────────────────────────────────────────────────────
if len(valid_pairs) > 0:
    fig, ax = plt.subplots(figsize=(9, 9))
    
    y_vals_arr = np.array(y_vals)
    c_vals_arr = np.array(c_vals)
    
    # Scatter
    ax.scatter(y_vals_arr, c_vals_arr, s=180, alpha=0.7, 
               edgecolors="black", linewidths=1.5, c="#2c3e50")
    
    # Linea y=x (perfetta consistenza)
    lim_min = min(y_vals_arr.min(), c_vals_arr.min())
    lim_max = max(y_vals_arr.max(), c_vals_arr.max())
    padding = (lim_max - lim_min) * 0.1
    ax.plot([lim_min - padding, lim_max + padding], 
            [lim_min - padding, lim_max + padding], 
            'k--', lw=2, alpha=0.5, label="Perfetta consistenza (y=x)")
    
    # Regressione lineare
    from scipy.stats import linregress
    slope, intercept, r_val, p_val, std_err = linregress(y_vals_arr, c_vals_arr)
    x_line = np.array([lim_min - padding, lim_max + padding])
    y_line = slope * x_line + intercept
    ax.plot(x_line, y_line, 'r-', lw=2, alpha=0.7,
            label=f"Regressione (y={slope:.2f}x{intercept:+.3f}, r²={r_val**2:.3f})")
    
    # Annotazioni
    ax.text(0.05, 0.95, 
            f"Correlazione: r = {corr:.3f}\n"
            f"N = {len(valid_pairs)} coppie\n"
            f"p-value = {p_val:.4f}",
            transform=ax.transAxes, fontsize=11, va="top",
            bbox=dict(boxstyle="round,pad=0.5", fc="white", ec="black", alpha=0.9))
    
    ax.set_xlabel("Δ margine Yield (Brent) [EUR/L]", fontsize=12, fontweight="bold")
    ax.set_ylabel("Δ margine Crack spread [EUR/L]", fontsize=12, fontweight="bold")
    ax.set_title("Correlazione Δmargini: Yield vs Crack spread\n"
                 "Punti vicini alla diagonale = alta consistenza",
                 fontsize=13, fontweight="bold", pad=12)
    ax.legend(fontsize=10, loc="lower right")
    ax.grid(alpha=0.3)
    ax.axhline(0, color="black", lw=0.8, alpha=0.5)
    ax.axvline(0, color="black", lw=0.8, alpha=0.5)
    
    plt.tight_layout()
    plt.savefig("plots/07b_scatter_yield_vs_crack.png", dpi=DPI, bbox_inches="tight")
    plt.close()
    print("  Salvato: plots/07b_scatter_yield_vs_crack.png")

# ─────────────────────────────────────────────────────────────────────────────
# 7. PLOT 3: HEATMAP CORRELAZIONI (se ci sono abbastanza dati)
# ─────────────────────────────────────────────────────────────────────────────
# Costruisci matrice pivot: righe=evento×serie, colonne=metodo
matrix_data = []
matrix_labels = []

for _, row in df_pivot.iterrows():
    label = f"{row['Evento'].split('(')[0].strip()}\n{row['Serie']}"
    matrix_labels.append(label)
    matrix_data.append([
        row.get("Yield_delta", np.nan),
        row.get("Crack_delta", np.nan)
    ])

if len(matrix_data) > 2:
    fig, ax = plt.subplots(figsize=(10, len(matrix_labels) * 0.6 + 2))
    
    df_matrix = pd.DataFrame(matrix_data, 
                              columns=["Yield (Brent)", "Crack spread"],
                              index=matrix_labels)
    
    # Heatmap
    sns.heatmap(df_matrix, annot=True, fmt=".4f", cmap="RdYlGn", 
                center=0, cbar_kws={"label": "Δ margine (EUR/L)"},
                linewidths=0.5, linecolor="gray", ax=ax)
    
    ax.set_title("Heatmap Δmargini: confronto metodi per evento\n"
                 "Verde = margine compresso | Rosso = margine espanso",
                 fontsize=12, fontweight="bold", pad=12)
    
    plt.tight_layout()
    plt.savefig("plots/07b_heatmap_methods.png", dpi=DPI, bbox_inches="tight")
    plt.close()
    print("  Salvato: plots/07b_heatmap_methods.png")

# ─────────────────────────────────────────────────────────────────────────────
# 8. SOMMARIO FINALE
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "="*80)
print("SOMMARIO FINALE")
print("="*80)
print(f"  Confronti totali:       {n_totali}")
print(f"  Classificazioni uguali: {n_consistenti} ({pct:.0f}%)")
print(f"  Divergenze:             {n_totali - n_consistenti}")

if len(valid_pairs) > 0:
    print(f"\n  Correlazione Δmargini:  r = {corr:.3f}")
    print(f"  Regressione:            y = {slope:.2f}x {intercept:+.3f}")
    print(f"  R² = {r_val**2:.3f}, p = {p_val:.4f}")

print("\n  Interpretazione:")
print("    • Alta consistenza (>80%) + alta correlazione (r>0.8):")
print("      → Risultati ROBUSTI, ipotesi speculazione affidabile")
print("    • Bassa consistenza o bassa correlazione:")
print("      → Possibile basis risk (prezzi USA vs EU)")
print("      → Yield teorici potrebbero non riflettere raffinerie EU")
print("      → Serve cautela nell'interpretazione")

print("\n  Output prodotti:")
print("    data/table2d_method_comparison.csv")
print("    plots/07b_comparison_methods.png")
print("    plots/07b_scatter_yield_vs_crack.png")
if len(matrix_data) > 2:
    print("    plots/07b_heatmap_methods.png")

print("\n" + "="*80)
print("Script 07b completato.")
print("="*80)