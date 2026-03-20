"""
run_all.py
===========
Esegue la pipeline completa in sequenza.

Ordine logico:
  1. 01_data_pipeline.py     → scarica dati (da 2019-01-01), produce dataset_merged.csv
  2. 02_core_analysis.py     → MCMC changepoint prezzi + test anomalia margine
                               (test primario H0 + BH correction FDR 5% locale)
  3. 03_statistical_tests.py → Granger + R&F + KS + ANOVA + Chow + Bootstrap
                               (test ausiliari) + DiD con parallel trends test
  4. 04_global_corrections.py→ BH correction GLOBALE su tutti i p-value del paper
                               (confirmatory: Welch t + DiD; exploratory: tutto resto)

Output principali:
  data/table1_changepoints.csv  → Table 1 del paper
  data/table2_margin_anomaly.csv→ Table 2 del paper (con BH local + global)
  data/global_bh_corrections.csv→ tutti i p-value con BH globale
  plots/                        → tutte le figure

NOTA METODOLOGICA — Baseline 2019:
  La baseline per la soglia 2σ è il full year 2019 (pre-COVID, pre-crisi).
  2020 escluso (COVID, WTI negativo). 2021 H1/Full usati come sensitivity check.
  Sensitivity analysis: data/baseline_sensitivity.csv
"""

import subprocess
import sys
import os
import time

os.chdir(os.path.dirname(os.path.abspath(__file__)))

SCRIPTS = [
    ("01_data_pipeline.py",      "Raccolta dati (yfinance 2019+ + EU Oil Bulletin)"),
    ("02_core_analysis.py",      "Analisi principale: changepoint + test margine (H0)"),
    ("03_statistical_tests.py",  "Test ausiliari: Granger + R&F + KS + ANOVA + Chow + DiD (parallel trends)"),
    ("04_global_corrections.py", "BH correction globale su tutti i p-value del paper"),
    ("05_nonparametric_validation.py", "Test non parametrici per validazione"),
]

print("=" * 70)
print(" PIPELINE: Speculazione carburanti Italia — tre crisi energetiche")
print("=" * 70)
print(" H0: il margine lordo (crack spread wholesale) non aumenta anomalmente (> 2σ baseline 2019)")
print(" H1: aumento anomalo post-shock → margine anomalo positivo (causa da verificare)")
print(" Baseline primaria: 2019 full year (pre-COVID, pre-crisi)")
print("=" * 70)

total_start = time.time()

for script, description in SCRIPTS:
    print(f"\n{'─'*70}")
    print(f"▶  {script}")
    print(f"   {description}")
    print(f"{'─'*70}")
    t0 = time.time()
    result = subprocess.run([sys.executable, script], capture_output=False, text=True)
    elapsed = time.time() - t0
    if result.returncode != 0:
        print(f"\n  ERRORE in {script} (exit code {result.returncode})")
        sys.exit(1)
    print(f"  ✓ Completato in {elapsed:.0f}s")

total_elapsed = time.time() - total_start
print(f"\n{'='*70}")
print(f" PIPELINE COMPLETATA in {total_elapsed/60:.1f} minuti")
print(f"{'='*70}")

# ── Sommario file prodotti ────────────────────────────────────────────────────
print("\n File chiave prodotti:")
key_files = [
    ("data/table1_changepoints.csv",       "Table 1 — changepoints MCMC (τ, CI 95%, lag D)"),
    ("data/table2_margin_anomaly.csv",     "Table 2 — test anomalia margine + BH local FDR 5%"),
    ("data/global_bh_corrections.csv",     "BH globale — tutti i p-value con adjusted p"),
    ("data/baseline_sensitivity.csv",      "Sensitivity baseline (2019 primary, 2021 H1/Full check)"),
    ("data/rockets_feathers_results.csv",  "R&F — asimmetria rialzo/ribasso"),
    ("data/did_results.csv",               "DiD — δ IT vs DE + parallel trends test (PTA)"),
    ("plots/02_*.png",                     "Changepoint plots (prezzi) — 9 figure"),
    ("plots/07_delta_summary.png",         "Δmargine per evento × metodo × carburante"),
    ("plots/07_margins_*crack*.png",       "Margini lordi crack spread nel tempo"),
    ("plots/03_granger_combined.png",      "Granger causality Brent → pompa"),
    ("plots/04_rf_combined.png",           "Rockets & Feathers scatter"),
    ("plots/06_statistical_tests.png",     "KS/ANOVA/Chow/CCF/Bootstrap riepilogo"),
]

for fname_pattern, description in key_files:
    if "*" in fname_pattern:
        print(f"   {fname_pattern:<45} {description}")
    elif os.path.exists(fname_pattern):
        size_kb = os.path.getsize(fname_pattern) / 1024
        print(f"   {fname_pattern:<45} ({size_kb:>7.1f} KB)  {description}")
    else:
        print(f"   {fname_pattern:<45} (non trovato)")

print("\n Per leggere i risultati:")
print("   import pandas as pd")
print("   t1 = pd.read_csv('data/table1_changepoints.csv')")
print("   t2 = pd.read_csv('data/table2_margin_anomaly.csv')")
print("   # Filtra test BH-locali significativi (Table 2):")
print("   t2_sig = t2[t2['BH_reject_FDR5%'] == True]")
print("   # Filtra test BH-globali significativi:")
print("   t2_sig_global = t2[t2['BH_global_reject'] == True]")
print("   # Tutti i p-value corretti globalmente:")
print("   g = pd.read_csv('data/global_bh_corrections.csv')")
print("   g_conf = g[g['tipo'] == 'confirmatory']")

print(f"\n{'='*70}\n")