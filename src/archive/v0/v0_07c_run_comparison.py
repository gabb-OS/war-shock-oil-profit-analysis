#!/usr/bin/env python3
"""
07c_run_comparison.py — Wrapper completo per analisi margini
=============================================================
Esegue automaticamente:
  1. Script 07 con --method=all → calcola margini con tutti i metodi
  2. Script 07b → confronto visivo yield vs crack
  3. Script 07a (opzionale) → plot margini nel tempo

Output:
  - data/dataset_merged_with_futures_all.csv  → dataset completo con tutti i futures
  - data/table2_margini_anomaly_all.csv       → risultati test speculazione (tutti i metodi)
  - data/table2d_method_comparison.csv        → matrice confronto metodi
  - plots/07_summary_all.png                  → pannello riassuntivo
  - plots/07b_comparison_methods.png          → barre affiancate metodi
  - plots/07b_scatter_yield_vs_crack.png      → correlazione metodi
  - plots/07b_heatmap_methods.png             → heatmap Δmargini
=============================================================
"""

import subprocess
import sys
import os

os.chdir(os.path.dirname(os.path.abspath(__file__)))

print("="*80)
print(" ANALISI COMPLETA: Margini lordi carburanti Italia")
print(" Confronto metodologie: Yield (Brent) vs Crack spread (Eurobob/Gas Oil)")
print("="*80)

# ─────────────────────────────────────────────────────────────────────────────
# 1. VERIFICA PRESENZA CSV FUTURES
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "─"*80)
print("[0/3] Verifica dati futures...")
print("─"*80)

required_files = [
    "data/Brent Oil Futures Historical Data.csv",
    "data/Eurobob Futures Historical Data.csv",
    "data/London Gas Oil Futures Historical Data.csv",
]

missing = [f for f in required_files if not os.path.exists(f)]
if missing:
    print("\nERRORE: File futures mancanti:")
    for f in missing:
        print(f"  {f}")
    print("\nScarica i dati da Investing.com e salvali in data/")
    sys.exit(1)

print("  OK: tutti i file futures presenti")
for f in required_files:
    size_kb = os.path.getsize(f) / 1024
    print(f"    {f:<55} ({size_kb:>7.1f} KB)")

# ─────────────────────────────────────────────────────────────────────────────
# 2. SCRIPT 07: Calcolo margini con tutti i metodi
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "─"*80)
print("[1/3] Esecuzione: 07_margine_speculation_test.py --method=all")
print("─"*80)

result1 = subprocess.run([sys.executable, "07_margine_speculation_test.py", "--method=all"],
                         capture_output=False, text=True)
if result1.returncode != 0:
    print("\nERRORE nell'esecuzione script 07")
    sys.exit(1)

# ─────────────────────────────────────────────────────────────────────────────
# 3. SCRIPT 07b: Confronto metodi
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "─"*80)
print("[2/3] Esecuzione: 07b_compare_methods.py")
print("─"*80)

result2 = subprocess.run([sys.executable, "07b_compare_methods.py"],
                         capture_output=False, text=True)
if result2.returncode != 0:
    print("\nERRORE nel confronto metodi")
    sys.exit(1)

# ─────────────────────────────────────────────────────────────────────────────
# 4. SCRIPT 07a (opzionale): Plot margini nel tempo
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "─"*80)
print("[3/3] Esecuzione: 07a_margin_plots.py (opzionale)")
print("─"*80)

if os.path.exists("07a_margin_plots.py"):
    result3 = subprocess.run([sys.executable, "07a_margin_plots.py"],
                             capture_output=False, text=True)
    if result3.returncode != 0:
        print("\nWARNING: errore in 07a (non critico)")
else:
    print("  Script 07a non trovato (opzionale, skip)")

# ─────────────────────────────────────────────────────────────────────────────
# 5. SOMMARIO FINALE
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "="*80)
print(" ANALISI COMPLETATA")
print("="*80)

print("\n File prodotti:")
output_files = [
    "data/dataset_merged_with_futures_all.csv",
    "data/table2_margini_anomaly_all.csv",
    "data/table2d_method_comparison.csv",
    "plots/07_summary_all.png",
    "plots/07b_comparison_methods.png",
    "plots/07b_scatter_yield_vs_crack.png",
    "plots/07b_heatmap_methods.png",
    "plots/01d_margine_benzina.png",
    "plots/01d_margine_diesel.png",
]

for fname in output_files:
    if os.path.exists(fname):
        size_kb = os.path.getsize(fname) / 1024
        print(f"   {fname:<50} ({size_kb:>7.1f} KB)")
    else:
        print(f"   {fname:<50} (non trovato)")

print("\n Interpretazione:")
print("   1. Apri plots/07_summary_all.png")
print("      → Pannello riassuntivo con tutti i metodi")
print("   2. Apri plots/07b_comparison_methods.png")
print("      → Confronto yield vs crack per ogni evento")
print("   3. Apri plots/07b_scatter_yield_vs_crack.png")
print("      → Correlazione tra i metodi (r > 0.8 = robusto)")
print("   4. Leggi data/table2d_method_comparison.csv")
print("      → Dettaglio numerico del confronto")

print("\n Criteri di robustezza:")
print("   • Consistenza classificazioni > 80%  → risultati AFFIDABILI")
print("   • Correlazione Δmargini r > 0.8      → metodi COERENTI")
print("   • Se entrambe soddisfatte            → ipotesi speculazione ROBUSTA")
print("   • Se una o entrambe falliscono       → cautela, possibile basis risk")

print("\n Prossimi passi:")
print("   • Se consistenza ALTA → usa crack spread (più preciso)")
print("   • Se consistenza BASSA → integra con dati ARERA per validazione")
print("   • Crack spread preferito per paper (wholesale europeo vs yield teorico)")

print("\n" + "="*80)
print("Pipeline 07 completata.")
print("="*80)