"""
run_all.py
===========
Esegue la pipeline completa in sequenza.

NARRATIVA DELLA PIPELINE
--------------------------
Il punto di partenza e' una domanda empirica: il margine lordo dei
distributori italiani e' aumentato in modo anomalo durante le tre
crisi energetiche del periodo 2022-2026?

La risposta segue una catena decisionale:

  PASSO 1 — Raccolta dati
    Scarichiamo le serie settimanali necessarie: Brent crude in EUR/barile,
    prezzi alla pompa italiani al netto delle tasse (EU Weekly Oil Bulletin),
    futures wholesale europei (Eurobob per benzina, Gas Oil ICE per diesel).
    Il baseline pre-crisi e' il 2019: mercato maturo, Brent stabile 60-70 $/bbl.

  PASSO 2 — Quando si sono mossi i prezzi?
    Prima di testare i margini, stabiliamo QUANDO la dinamica dei prezzi
    si e' rotta rispetto allo shock geopolitico. Usiamo un modello
    piecewise-lineare bayesiano sui log-prezzi (likelihood StudentT per
    robustezza alle code pesanti). Il lag D = tau - shock_date misura
    l'anticipo (D < 0) o ritardo (D > 0) del changepoint rispetto all'evento.
    Come effetto collaterale, i diagnostici OLS prodotti qui (DW, SW, BP)
    motivano la scelta dei test nel passo successivo.

  PASSO 3 — Il margine e' anomalo?
    Calcoliamo il crack spread (prezzo pompa - costo wholesale) come proxy
    del margine lordo. Testiamo H0 con una batteria di test scelti in base
    ai diagnostici del passo 2:
      - DW = 0.15-0.42 (rho ~0.90): le SE del t-test sono gonfiate -> aggiungiamo
        HAC Newey-West e block permutation che non assumono indipendenza.
      - SW p < 0.05 per alcuni scenari: aggiungiamo Mann-Whitney che non
        assume normalita'.
      - Il Welch t rimane test primario per confrontabilita' con la letteratura
        e per la BH correction locale.

  PASSO 4 — L'anomalia e' specifica all'Italia?
    Tre domande ausiliarie:
      a. Con quale velocita' il Brent predice i prezzi pompa? (Granger)
         Un lag breve e' ambiguo: coerente sia con efficienza che con
         pricing opportunistico.
      b. La trasmissione e' strutturalmente asimmetrica? (Rockets & Feathers)
         beta_up > beta_down indica che i prezzi salgono piu' veloce di
         quanto scendono, indipendentemente dagli shock.
      c. L'eventuale aumento di margine e' specifico all'Italia o comune
         a tutti i mercati EU? (Difference-in-Differences, IT vs DE e SE)
         Solo il DiD e' confirmatory e contribuisce alla BH globale.

  PASSO 5 — Quanti risultati reggono alla correzione per test multipli?
    Raccogliamo tutti i p-value confirmatory (script 03 + DiD script 04)
    e applichiamo la Benjamini-Hochberg correction globale (FDR <= 5%).
    I test esplorativi (Granger, R&F, KS, ANOVA, Chow) sono riportati
    come evidenza di contesto ma non entrano nella famiglia BH.

  PASSO 6 — L'assunzione distributiva StudentT e' corretta?
    Per ogni scenario (evento x serie) fittiamo quattro distribuzioni
    sui residui OLS piecewise (log-prezzi) e sui crack spread post-shock:
    Normale, StudentT (scelta attuale), Skew-Normal, Skewed-T (Fernandez-Steel).
    Il confronto AIC identifica se e' necessaria una distribuzione asimmetrica.
    La guida operativa mostra come modificare il modello PyMC in script 02.

Output principali:
  data/table1_changepoints.csv      -> Table 1: tau, CI 95%, lag D
  data/table2_margin_anomaly.csv    -> Table 2: test anomalia + BH locale + globale
  data/global_bh_corrections.csv   -> tutti i p-value confirmatory con BH
  data/baseline_sensitivity.csv    -> robustezza alla scelta di baseline
  data/did_results.csv             -> DiD delta con PTA test
  plots/                           -> tutte le figure
"""

import subprocess
import sys
import os
import time

os.chdir(os.path.dirname(os.path.abspath(__file__)))

PIPELINE = [
    (
        "01_data_pipeline.py",
        "Raccolta dati",
        "Brent EUR + prezzi pompa senza tasse (EU Bulletin 2019-oggi) + futures",
    ),
    (
        "02_changepoint.py",
        "Changepoint bayesiano sui log-prezzi",
        "Quando si e' rotta la dinamica dei prezzi? Lag D = tau - shock_date",
    ),
    (
        "03_margin_hypothesis.py",
        "Test anomalia margine lordo",
        "Welch t + Mann-Whitney + block perm + HAC, motivati dai diagnostici OLS",
    ),
    (
        "04_auxiliary_evidence.py",
        "Evidenza ausiliaria",
        "Granger (velocita') + R&F (asimmetria) + DiD (specificita' italiana)",
    ),
    (
        "05_global_corrections.py",
        "BH correction globale",
        "FDR <= 5% su tutti i test confirmatory degli script 03 e 04",
    ),
    (
        "06_distribution_check.py",
        "Verifica assunzione distributiva",
        "StudentT vs Skew-Normal vs Skewed-T sui residui OLS e crack spread",
    ),
]

print("=" * 70)
print(" PIPELINE: Margini carburanti Italia — tre crisi energetiche")
print("=" * 70)
print()
print(" H0: il margine lordo (crack spread wholesale) non aumenta anomalmente")
print("     rispetto al baseline 2019 dopo lo shock geopolitico.")
print()
print(" Baseline: 2019 full year (pre-COVID, mercato maturo, Brent 60-70 $/bbl)")
print("=" * 70)

t_start = time.time()
timings = {}

for script, titolo, descrizione in PIPELINE:
    sep = "─" * 70
    print(f"\n{sep}")
    print(f"  {titolo}")
    print(f"  {descrizione}")
    print(f"  Script: {script}")
    print(sep)

    t0     = time.time()
    result = subprocess.run([sys.executable, script], capture_output=False, text=True)
    elapsed = time.time() - t0
    timings[script] = elapsed

    if result.returncode != 0:
        print(f"\n  ERRORE in {script} (exit code {result.returncode})")
        print("  Pipeline interrotta.")
        sys.exit(1)

    print(f"  Completato in {elapsed:.0f}s")

t_total = time.time() - t_start
print(f"\n{'=' * 70}")
print(f" PIPELINE COMPLETATA in {t_total / 60:.1f} minuti")
print(f"{'=' * 70}")

# ── Riepilogo file chiave ─────────────────────────────────────────────────────
print("\n File chiave prodotti:")

KEY_FILES = [
    ("data/table1_changepoints.csv",    "Table 1 — changepoints (tau, CI 95%, lag D, nu, Rhat)"),
    ("data/table2_margin_anomaly.csv",  "Table 2 — test margine + BH locale + BH globale"),
    ("data/global_bh_corrections.csv", "BH globale — tutti i p-value confirmatory"),
    ("data/baseline_sensitivity.csv",  "Sensitivity — soglia 2sigma con baseline 2019 vs 2021"),
    ("data/did_results.csv",           "DiD — delta IT vs DE/SE + parallel trends test"),
    ("data/rockets_feathers_results.csv", "R&F — asimmetria beta_up vs beta_down"),
    ("plots/02_*.png",                 "Changepoint plots (9 figure + diagnostiche)"),
    ("plots/03_delta_summary.png",     "Delta margine post-shock per evento x carburante"),
    ("plots/03_margins.png",           "Crack spread nel tempo con eventi e banda baseline"),
    ("plots/04_granger.png",           "Granger causality Brent -> pompa"),
    ("plots/04_rf.png",                "Rockets & Feathers scatter"),
    ("plots/04_did.png",               "DiD delta con CI 95% HC3"),
    ("data/distribution_check.csv",    "Distrib. check — AIC per 4 distribuzioni + raccomandazione"),
    ("plots/06_distrib_summary.png",   "ΔAIC heatmap: StudentT vs Skew-Normal vs Skewed-T"),
]

for fname, descrizione in KEY_FILES:
    if "*" in fname:
        print(f"   {fname:<45} {descrizione}")
        continue
    if os.path.exists(fname):
        kb = os.path.getsize(fname) / 1024
        print(f"   {fname:<45} ({kb:>7.1f} KB)  {descrizione}")
    else:
        print(f"   {fname:<45} (non prodotto)  {descrizione}")

# ── Tempi per script ──────────────────────────────────────────────────────────
print(f"\n Tempi di esecuzione:")
for script, t in timings.items():
    print(f"   {script:<35} {t:>5.0f}s")

# ── Come leggere i risultati ──────────────────────────────────────────────────
print("""
 Come leggere i risultati:

   import pandas as pd

   # Table 1: lag D tra changepoint e shock
   t1 = pd.read_csv("data/table1_changepoints.csv")
   print(t1[["Evento","Serie","tau","Lag (gg)","H0_rif","DW"]])

   # Table 2: classificazione finale con BH globale
   t2 = pd.read_csv("data/table2_margin_anomaly.csv")
   print(t2[["Evento","Carburante","delta_mean_eur","t_p",
             "BH_reject_local","BH_global_reject","classificazione_BH"]])

   # Tutti i test confirmatory con p aggiustato
   bh = pd.read_csv("data/global_bh_corrections.csv")
   print(bh[bh["BH_global_reject"] == True])

   # Verifica assunzione distributiva (script 06)
   dc = pd.read_csv("data/distribution_check.csv")
   print(dc[["label","best_distribution","raccomandazione","ΔAIC_skewt_vs_t"]])
""")

print("=" * 70)