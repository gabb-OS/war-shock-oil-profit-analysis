"""
run_all_v2.py
==============
Esegue la pipeline v2 in sequenza.

ARCHITETTURA EPISTEMICA v2
───────────────────────────
  LIVELLO 1 — CONFIRMATORY   (famiglia BH-A, 16 test)
    H₀: μ_post = μ_2019  (livello assoluto, one-sided upper)
    Split: shock_hard + τ_price (esogeni al margine)
    Test: Welch 1-sample + Mann-Whitney
    Script: 03_margin_hypothesis_v2.py

  LIVELLO 2 — EXPLORATORY    (no BH, riportati separatamente)
    τ_margin: solo timing relativo a τ_price (Δ giorni, no p-value)
    Block permutation, HAC Andrews, n_eff, ρ̂
    Script: 03_margin_hypothesis_v2.py

  LIVELLO 3 — AUXILIARY      (famiglia BH-B separata, 8 test DiD)
    H₀: δ_DiD = 0  (specificità italiana)
    Script: 04_auxiliary_evidence_v2.py

  CORREZIONE BH GLOBALE      (24 test: 16 BH-A + 8 BH-B)
    Script: 05_global_corrections_v2.py

  DIAGNOSTICA DISTRIBUZIONE  (n_eff, Andrews BW, inflazione potenza)
    Script: 06_distribution_check_v2.py

  PRE-SHOCK ANOMALY          (esplorativo, no BH)
    Script: 07_preshock_anomaly.py

ORDINE ESECUZIONE
─────────────────
  1. 01_data_pipeline.py         → dataset_merged.csv
  2. 02_core_analysis.py         → MCMC changepoint + τ_price → table1_changepoints.csv
  3. 03_margin_hypothesis_v2.py  → test confirmativi + esplorativi margine
  4. 04_auxiliary_evidence_v2.py → DiD, Granger, R&F, windfall v2, Hormuz
  5. 05_global_corrections_v2.py → BH globale famiglia A (16) + B (8)
  6. 06_distribution_check_v2.py → n_eff, Andrews BW, diagnostiche
  7. 07_preshock_anomaly.py      → anomalia pre-shock Iran-Israele

DIFETTI CORRETTI RISPETTO A v1
───────────────────────────────
  [CRITICO] Circolarità τ_margin: rimosso dalla famiglia BH → solo descrittivo
  [CRITICO] H₀ miste: Welch+MW (livello vs 2019) separati da block perm+HAC (salto locale)
  [ALTO]    Bug script 05: match chiave (evento,carburante) invece di posizionale
  [MEDIO]   HAC maxlags=4 → Andrews automatic bandwidth
  [MEDIO]   n_eff non riportato → ora in data/neff_report_v2.csv e testo paper
  [MEDIO]   Anomalia pre-shock Iran-Israele → script 07 dedicato
  [BASSA]   DiD negativo minimizzato → interpretazione corretta in did_results_v2.csv
  [BASSA]   Windfall volumi fissi 2022 → correzione trend lineare -1.5%/anno

Output chiave v2:
  data/confirmatory_pvalues_v2.csv  → 16 test confirmativi (Welch+MW, shock_hard+τ_price)
  data/exploratory_results_v2.csv   → esplorativi (block perm, HAC, n_eff)
  data/tau_margin_descriptive.csv   → τ_margin solo descrittivo
  data/did_results_v2.csv           → 8 test DiD con BH famiglia B
  data/global_bh_v2.csv             → BH globale 24 test
  data/neff_report_v2.csv           → n_eff, Andrews BW, inflazione potenza
  data/preshock_anomaly.csv         → analisi pre-shock Iran-Israele
"""

import subprocess
import sys
import os
import time

os.chdir(os.path.dirname(os.path.abspath(__file__)))

SCRIPTS_V2 = [
    ("01_data_pipeline.py",
     "Raccolta dati (yfinance 2019+ + EU Oil Bulletin)",
     True),   # può essere skip se i dati esistono già
    ("02_changepoint.py",
     "Analisi principale: MCMC changepoint prezzi → τ_price per ogni evento/serie",
     False),
    ("03_margin_hypothesis.py",
     "Test confirmativi margine (BH-A: 16 test) + esplorativi + τ_margin descrittivo",
     False),
    ("04_auxiliary_evidence_v2.py",
     "DiD IT vs DE/SE (BH-B: 8 test) + Granger + R&F + windfall v2 + Hormuz",
     False),
    ("05_global_corrections.py",
     "BH globale: famiglia A (16) + famiglia B (8) = 24 test totali",
     False),
    ("06_distribution_check_v2.py",
     "Diagnostica: n_eff, Andrews BW, inflazione potenza, BP/DW/SW",
     False),
    ("07_preshock_anomaly.py",
     "Analisi pre-shock Iran-Israele: Bai-Perron + confronto annuale [esplorativo]",
     False),
]

print("=" * 72)
print(" PIPELINE v2: Speculazione carburanti Italia — tre crisi energetiche")
print("=" * 72)
print(" ARCHITETTURA v2:")
print("   BH-A (16 test) — confirmativi: H₀: μ_post = μ_2019, split esogeni")
print("   BH-B  (8 test) — ausiliari DiD: H₀: δ_DiD = 0 (specificità IT)")
print("   Esplorativi     — τ_margin, block perm, HAC, n_eff [no BH]")
print("   Pre-shock       — anomalia strutturale 2025-H1 [no BH]")
print("=" * 72)

total_start = time.time()

for script, description, can_skip in SCRIPTS_V2:
    if not os.path.exists(script):
        print(f"\n  SKIP {script} — file non trovato (eseguire manualmente se necessario)")
        continue

    print(f"\n{'─'*72}")
    print(f"▶  {script}")
    print(f"   {description}")
    print(f"{'─'*72}")
    t0     = time.time()
    result = subprocess.run([sys.executable, script], capture_output=False, text=True)
    elapsed = time.time() - t0

    if result.returncode != 0:
        if can_skip:
            print(f"  ⚠ {script} terminato con errori (exit {result.returncode}) — pipeline continua")
        else:
            print(f"\n  ERRORE CRITICO in {script} (exit code {result.returncode})")
            print("  Pipeline interrotta. Correggere l'errore e rieseguire.")
            sys.exit(1)
    else:
        print(f"  ✓ Completato in {elapsed:.0f}s")

total_elapsed = time.time() - total_start
print(f"\n{'='*72}")
print(f" PIPELINE v2 COMPLETATA in {total_elapsed/60:.1f} minuti")
print(f"{'='*72}")

# ── Sommario file prodotti ─────────────────────────────────────────────────────
print("\n File chiave v2:")
key_files = [
    # Livello 1 — confirmatory
    ("data/confirmatory_pvalues_v2.csv",
     "BH-A — 16 test confirmativi (Welch+MW, shock_hard+τ_price)"),
    ("data/exploratory_results_v2.csv",
     "Esplorativi — block perm, HAC, n_eff [no BH]"),
    ("data/tau_margin_descriptive.csv",
     "τ_margin — solo timing descrittivo, NESSUN p-value"),
    # Livello 3 — auxiliary DiD
    ("data/did_results_v2.csv",
     "BH-B — 8 test DiD IT vs DE/SE con BH famiglia separata"),
    # Global BH
    ("data/global_bh_v2.csv",
     "BH globale — 24 test (16+8) con adjusted p"),
    # Diagnostica
    ("data/neff_report_v2.csv",
     "n_eff, Andrews BW, fattore inflazione potenza"),
    ("data/distribution_diagnostics_v2.csv",
     "Diagnostica BP/DW/SW per ogni serie/evento"),
    # Pre-shock anomaly
    ("data/preshock_anomaly.csv",
     "Bai-Perron pre-shock Iran-Israele [esplorativo]"),
    ("data/preshock_annual_stats.csv",
     "Statistiche annuali 2019/2023/2024/2025-H1"),
    ("data/windfall_v2.csv",
     "Windfall con correzione trend consumi v2"),
    # Legacy (compatibilità backward)
    ("data/table1_changepoints.csv",
     "Table 1 — MCMC τ_price (da 02_core_analysis.py)"),
]

any_missing = False
for fname, desc in key_files:
    if os.path.exists(fname):
        size_kb = os.path.getsize(fname) / 1024
        print(f"   ✓ {fname:<42} ({size_kb:>6.1f} KB)  {desc}")
    else:
        print(f"   ✗ {fname:<42} (non trovato)  {desc}")
        any_missing = True

if any_missing:
    print("\n  ⚠ Alcuni file non trovati. Verificare log degli script.")

print("\n Lettura rapida risultati:")
print("   import pandas as pd")
print()
print("   # Confirmativi (BH-A)")
print("   c = pd.read_csv('data/confirmatory_pvalues_v2.csv')")
print("   print(c[c['BH_A_reject']==True])")
print()
print("   # DiD (BH-B)")
print("   d = pd.read_csv('data/did_results_v2.csv')")
print("   print(d[['Evento','Paese_controllo','Carburante','delta_DiD_EUR_L','p_value','BH_DiD_reject','interpretation_note']])")
print()
print("   # n_eff — inflazione potenza")
print("   n = pd.read_csv('data/neff_report_v2.csv')")
print("   print(n[['Evento','Carburante','n_post','rho_hat_AR1','n_eff','inflation_factor','inflazione_nota']])")
print()
print("   # Pre-shock anomaly")
print("   p = pd.read_csv('data/preshock_anomaly.csv')")
print("   print(p)")
print(f"\n{'='*72}\n")
