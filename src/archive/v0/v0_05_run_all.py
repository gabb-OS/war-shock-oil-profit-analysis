"""
05_run_all.py
==============
Esegue tutta la pipeline in sequenza e stampa il sommario finale.
Equivale a runnare tutti i notebook in ordine.
"""

import subprocess
import sys
import os

os.chdir(os.path.dirname(os.path.abspath(__file__)))

SCRIPTS = [
    "01_data_pipeline.py",
    "02_changepoint_detection.py",
    "02c_zoom_plots.py",
    "03_granger_causality.py",
    "04_rocket_feather.py",
    "06_statistical_tests.py", 
    "07_margine_speculation_test.py"
]

print("=" * 65)
print(" PIPELINE COMPLETA — Speculazione sui carburanti")
print("=" * 65)

for script in SCRIPTS:
    print(f"\n{'─'*65}")
    print(f"▶  Esecuzione: {script}")
    print(f"{'─'*65}")
    result = subprocess.run([sys.executable, script], capture_output=False, text=True)
    if result.returncode != 0:
        print(f" Errore in {script}")
        sys.exit(1)

print("\n" + "=" * 65)
print(" PIPELINE COMPLETATA")
print("=" * 65)

# Stampa sommario dei file prodotti
import os
print("\n File prodotti:")
for folder in ["data", "plots"]:
    if os.path.exists(folder):
        files = os.listdir(folder)
        for f in sorted(files):
            size = os.path.getsize(f"{folder}/{f}")
            print(f"   {folder}/{f}  ({size/1024:.1f} KB)")

print("\n Interpretazione dei risultati:")
print("   - plots/01_overview.png     → andamento prezzi con eventi guerra")
print("   - plots/02_changepoints.png → regressione piecewise (StudentT + AR(1) MCMC)")
print("   - plots/03_granger.png      → Granger causality p-values per lag")
print("   - plots/04_rockets_feathers.png → asimmetria razzo/piuma (GLSAR AR(1) + HAC)")
print("   - data/lag_results.csv      → tabella lag D per ogni evento")
print("   - data/table1_changepoints.csv  → Table 1 analoga al paper (con nu + rho posteriori)")
print("   - data/granger_benzina.csv  → risultati Granger per benzina")
print("   - data/rockets_feathers_results.csv → test asimmetria (SE HAC + rho AR(1))")
print("   - plots/06_statistical_tests.png → KS, CCF, rolling corr, bootstrap CI")
print("   - plots/08_regression_selection.png → selezione tipo regressione (BP/LB/AIC/SE)")
print("   - data/ks_results.csv            → Kolmogorov-Smirnov test")
print("   - data/anova_results.csv         → ANOVA 3 periodi")
print("   - data/chow_results.csv          → Chow structural break test")
print("   - data/bootstrap_ci.csv          → Bootstrap 95% CI sul lag D")
print("   - data/fit_quality.csv           → RMSE/MAE regressione piecewise")
print("   - data/regression_selection.csv  → raccomandazione tipo regressione per serie")