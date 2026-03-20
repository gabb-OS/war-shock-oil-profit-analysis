"""
3_run.py  — Orchestrator pipeline v3 (minimale)
================================================
Esegue in sequenza i 4 script della pipeline v3,
corrispondenti alle tre sub-ipotesi di H₀.

  3_01_data.py   → data/3_dataset.csv          (dati + HICP deflazione)
  3_02_tests.py  → data/3_AB.csv               (Famiglia A + B)
  3_03_did.py    → data/3_C.csv + plots/3_did  (Famiglia C)
  3_04_bh.py     → data/3_bh.csv + plots/3_summary (BH + sommario)

Uso:
  cd /path/to/src
  python 3_run.py
"""

import subprocess
import sys
import os
import time

SCRIPTS = [
    # ── Passo 1: dati grezzi ──────────────────────────────────────────────
    ("3_01_data.py",        "Preparazione dati + HICP deflazione"),

    # ── Passo 2: rileva QUANDO il mercato si rompe (τ MCMC) ──────────────
    # DEVE girare prima di 3_02 e 3_03: produce data/3_cp.csv con i tau
    # MAP dei changepoint, che i test successivi usano come cutoff pre/post
    # invece della data geopolitica hardcoded.
    ("3_05_changepoint.py", "MCMC Change Point Detection → tau pre/post"),

    # ── Passo 3-4: test H₀(i)(ii)(iii) con τ come data di taglio ─────────
    # 3_02 e 3_03 leggono data/3_cp.csv (prodotto al passo 2) per
    # sostituire la data dell'evento con il τ empirico rilevato dai dati.
    ("3_02_tests.py",       "Famiglie A + B: HAC_t e Mann-Whitney (con τ MCMC)"),
    ("3_03_did.py",         "Famiglia C: DiD IT vs EU (con τ MCMC)"),

    # ── Passo 5: correzione multipla e sommario finale ────────────────────
    ("3_04_bh.py",          "Correzione BH + sommario H₀ macro"),
]


def run_script(script_name: str, description: str) -> bool:
    """Esegue uno script Python e restituisce True se completato senza errori."""
    print(f"\n{'='*65}")
    print(f"▶  {script_name}  —  {description}")
    print(f"{'='*65}")
    t0 = time.time()
    result = subprocess.run(
        [sys.executable, script_name],
        capture_output=False,   # mostra output in tempo reale
    )
    elapsed = time.time() - t0
    if result.returncode != 0:
        print(f"\n  ✗ ERRORE in {script_name} (returncode={result.returncode})")
        return False
    print(f"\n  ✓ {script_name} completato in {elapsed:.1f}s")
    return True


if __name__ == "__main__":
    # Cambia directory nella cartella degli script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)
    print(f"Workdir: {script_dir}")

    t_start = time.time()
    successes = []

    for script, desc in SCRIPTS:
        ok = run_script(script, desc)
        successes.append((script, ok))
        if not ok:
            print(f"\n  Pipeline interrotta a {script}.")
            break

    total = time.time() - t_start
    print(f"\n{'='*65}")
    print(f"PIPELINE v3 — SOMMARIO  ({total:.1f}s totali)")
    print(f"{'='*65}")
    for script, ok in successes:
        status = "✓" if ok else "✗"
        print(f"  {status}  {script}")

    all_ok = all(ok for _, ok in successes) and len(successes) == len(SCRIPTS)
    if all_ok:
        print("\n  Tutti i passi completati.")
        print("\n  ORDINE LOGICO:")
        print("    3_01 → dati grezzi")
        print("    3_05 → τ MCMC (changepoint): QUANDO si rompe il mercato")
        print("    3_02 → test H₀(i)(ii) con τ come cutoff pre/post")
        print("    3_03 → DiD H₀(iii) con τ come cutoff pre/post")
        print("    3_04 → correzione multipla BH + sommario finale")
        print("\n  DATI:")
        print("    data/3_dataset.csv        — crack spread IT nominali + reali (HICP)")
        print("    data/3_hicp.csv           — HICP Italy mensile")
        print("    data/3_AB.csv             — 16 test famiglie A+B")
        print("    data/3_C.csv              — 8 test famiglia C (DiD)")
        print("    data/3_bh.csv             — tutti i test con BH reject per famiglia")
        print("    data/3_table_results.csv  — tabella riassuntiva per paper")
        print("    data/3_neff_report.csv    — diagnostica n_eff e ρ̂")
        print("    data/3_annual_margins.csv — margini medi annuali")
        print("    data/3_windfall.csv       — windfall con sensitività ±30%")
        print("\n  GRAFICI (3_01 — dati grezzi):")
        print("    plots/3_01a_brent.png     — Brent EUR/barile")
        print("    plots/3_01b_pompa_it.png  — prezzi pompa IT")
        print("    plots/3_01c_crack.png     — crack spread nominale vs reale")
        print("    plots/3_01d_confronto.png — IT vs DE vs SE")
        print("\n  GRAFICI (3_02 — test H₀(i): famiglie A+B):")
        print("    plots/3_02a_margins.png   — crack spread con bande 2019 + finestre evento")
        print("    plots/3_02b_delta.png     — confronto medie 2019/pre/post")
        print("    plots/3_02c_annual.png    — margini medi annuali")
        print("    plots/3_02d_neff.png      — diagnostica n_eff e autocorrelazione")
        print("\n  GRAFICI (3_03 — test H₀(iii)):")
        print("    plots/3_03a_did.png       — forest plot DiD IT vs DE/SE")
        print("    plots/3_03b_windfall.png  — windfall stimati con sensitività")
        print("\n  GRAFICI (3_04 — sommario):")
        print("    plots/3_04_summary.png    — heatmap BH tutti i test")
        print("\n  GRAFICI (3_05 — MCMC change point):")
        print("    plots/3_05a_cp_benzina.png — CP posteriori benzina + regimi")
        print("    plots/3_05b_cp_diesel.png  — CP posteriori diesel + regimi")
        print("    plots/3_05_summary.png     — CP MAP+CI95 vs eventi geopolitici")
        print("\n  DATI (3_05):")
        print("    data/3_cp.csv             — MAP e CI 95% dei change point")
    else:
        sys.exit(1)