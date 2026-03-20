#!/usr/bin/env python3
"""
run_all.py
──────────
Esegue l'intera pipeline in ordine.

Pipeline base (senza argomenti):
  02a     →  diagnostics prezzi
  02b     →  diagnostics margini
  02c_m   →  change point detection sul margine
  02c_p   →  change point detection sul prezzo
  02d     →  analisi controfattuale (legacy)

Pipeline ITS (argomento positivo: v1/v3/v5/v6/cmp o nessuno = tutti):
  Per OGNI metodo ITS selezionato la pipeline esegue AUTOMATICAMENTE:
    1. 02c --detect margin       → produce theta_results.csv (margine)
    2. 02c --detect price        → produce theta_results.csv (prezzo)
    3. metodo --mode fixed       → break = data shock hardcodata
    4. metodo --mode detected --detect margin  → break = θ rilevato sul margine
    5. metodo --mode detected --detect price   → break = θ rilevato sul prezzo
    6. 02d_compare.py            → confronto metodi (solo se tutti i metodi girano)

  Metodi attivi (triangolazione):
    v1  — OLS Naïve          (frequentista, baseline)
    v3  — ARIMA             (serie temporali, autocorrelazione)
    v5  — BSTS CausalImpact  (bayesiano, state-space)
    v6  — GLM Gamma          (distribuzione continua positiva, log-link)
    v7  — Theil-Sen          (non-parametrico, block bootstrap CI)

  I passi 1-2 vengono eseguiti una sola volta all'inizio (non ripetuti per ogni
  metodo), a meno che non venga usato --skip-02c per saltarli (utile se
  theta_results.csv è già aggiornato).

Uso:
  python3 run_all.py                   # solo pipeline base (02a→02d legacy)
  python3 run_all.py its               # tutti i metodi ITS (v1+v3+v5+v6+cmp)
  python3 run_all.py v1                # solo v1 in tutte e 3 le modalità
  python3 run_all.py v1 v3             # v1 e v3 in tutte e 3 le modalità
  python3 run_all.py v5 v6 cmp         # v5, v6 e compare
  python3 run_all.py its --skip-02c    # salta 02c (usa theta_results.csv esistente)
  python3 run_all.py 02a 02b           # solo step base indicati
"""

import subprocess
import sys
import time
from pathlib import Path

BASE_DIR = Path(__file__).parent

# ── Step base (non richiedono mode) ──────────────────────────────────────────
BASE_STEPS = [
    ("02a",   "02a_diagnostics_price.py",      "Diagnostics — Prezzi",               []),
    ("02b",   "02b_diagnostics_margin.py",      "Diagnostics — Margini",              []),
    ("02c_m", "02c_change_point_detection.py",  "Change Point Detection — Margine",   ["--detect", "margin"]),
    ("02c_p", "02c_change_point_detection.py",  "Change Point Detection — Prezzo",    ["--detect", "price"]),
    ("02d",   "02d_counterfactual_gains.py",    "Analisi Controfattuale — Legacy",    []),
]

# ── Step ITS ─────────────────────────────────────────────────────────────────
ITS_STEPS = [
    ("v1",  "02d_v1_naive.py",        "ITS Metodo 1 — OLS Naïve"),
    ("v3",  "02d_v3_arima.py",       "ITS Metodo 3 — ARIMA"),
    ("v5",  "02d_v5_causalimpact.py", "ITS Metodo 5 — BSTS CausalImpact"),
    #("v6",  "02d_v6_glm_gamma.py",    "ITS Metodo 6 — GLM Gamma log-link"),
    ("v7",  "02d_v7_theilsen.py",     "ITS Metodo 7 — Theil-Sen + Block Bootstrap"),
    ("v8",  "02d_v8_pymc.py",     "Pymc Metodo 8"),
    ("cmp", "02d_compare.py",         "ITS Confronto — Metodi"),
    ("02e", "02e_statistical_tests.py", "Test Statistici — Batteria Completa"),
]

# Per ogni metodo ITS, queste sono le 3 varianti che vengono eseguite in ordine:
#   (tag_label, extra_args)
ITS_VARIANTS = [
    ("fixed",            ["--mode", "fixed"]),
    ("detected/margin",  ["--mode", "detected", "--detect", "margin"]),
    ("detected/price",   ["--mode", "detected", "--detect", "price"]),
]

ITS_KEYS  = {k for k, _, _ in ITS_STEPS}
BASE_KEYS = {k for k, *_ in BASE_STEPS}

SEP = "═" * 70


def run_step(filename: str, extra_args: list[str]) -> tuple[str, float]:
    script = BASE_DIR / filename
    if not script.exists():
        print(f"\n  ⚠  {filename} non trovato — salto.")
        return "SKIP", 0.0

    cmd  = [sys.executable, str(script)] + extra_args
    t0   = time.time()
    proc = subprocess.run(cmd, cwd=str(BASE_DIR))
    elapsed = time.time() - t0

    if proc.returncode == 0:
        print(f"\n  ✓  completato in {elapsed:.1f}s")
        return "OK", elapsed
    else:
        print(f"\n  ✗  errore (exit {proc.returncode}) dopo {elapsed:.1f}s")
        return "FAIL", elapsed


def print_banner(tag: str, label: str) -> None:
    print(f"\n{SEP}")
    print(f"  ▶  {tag}  |  {label}")
    print(f"{SEP}\n")


def main() -> None:
    raw_args = sys.argv[1:]

    # ── Parsing argomenti ─────────────────────────────────────────────────────
    skip_02c    = False
    step_targets: list[str] = []
    run_its     = False

    # Alias backward-compat: "fixed" e "detected" erano le vecchie modalità ITS
    ITS_ALIASES = {"fixed", "detected", "its"}

    for a in raw_args:
        if a == "--skip-02c":
            skip_02c = True
        elif a in ITS_ALIASES:
            # "fixed", "detected", "its" → avvia tutti i metodi ITS
            # (ogni metodo gira sempre in tutte e 3 le varianti: fixed, detected/margin, detected/price)
            run_its = True
        elif a in ITS_KEYS or a in BASE_KEYS or a.startswith("02"):
            step_targets.append(a)
        else:
            print(f"  ⚠  Argomento non riconosciuto: '{a}' — ignorato.")

    # "its" senza altri step_targets = tutti i metodi ITS
    if run_its and not any(t in ITS_KEYS for t in step_targets):
        step_targets += [k for k, _, _ in ITS_STEPS]

    # Deduplica mantenendo ordine
    seen: set[str] = set()
    step_targets_dedup: list[str] = []
    for t in step_targets:
        if t not in seen:
            step_targets_dedup.append(t)
            seen.add(t)
    step_targets = step_targets_dedup

    # Determina cosa eseguire
    its_targets  = [t for t in step_targets if t in ITS_KEYS]
    base_targets = [t for t in step_targets if t in BASE_KEYS or
                    (t.startswith("02") and t not in ITS_KEYS)]

    no_targets = not step_targets and not run_its

    total_start = time.time()
    results: list[tuple] = []

    # ══════════════════════════════════════════════════════════════════════════
    # A) Pipeline base
    # ══════════════════════════════════════════════════════════════════════════
    if no_targets:
        # Nessun argomento → esegui solo pipeline base
        for key, filename, label, extra_args in BASE_STEPS:
            print_banner(key, label)
            status, elapsed = run_step(filename, extra_args)
            results.append((key, label, "–", status, elapsed))

    elif base_targets and not its_targets:
        # Solo step base esplicitamente richiesti
        for key, filename, label, extra_args in BASE_STEPS:
            if any(t in key for t in base_targets):
                print_banner(key, label)
                status, elapsed = run_step(filename, extra_args)
                results.append((key, label, "–", status, elapsed))

    # ══════════════════════════════════════════════════════════════════════════
    # B) Pipeline ITS
    # ══════════════════════════════════════════════════════════════════════════
    if its_targets:

        # ── B1. Prerequisito: esegui 02c per margin e price ──────────────────
        if not skip_02c:
            print(f"\n{SEP}")
            print("  PRE-REQUISITO ITS: Change Point Detection (02c)")
            print(f"  (usa --skip-02c per saltare se theta_results.csv è già aggiornato)")
            print(f"{SEP}")

            for detect_variant in ["margin", "price"]:
                tag   = f"02c_{detect_variant[0]}"
                label = f"Change Point Detection — {'Margine' if detect_variant == 'margin' else 'Prezzo'}"
                print_banner(tag, label)
                status, elapsed = run_step(
                    "02c_change_point_detection.py",
                    ["--detect", detect_variant]
                )
                results.append((tag, label, "prerequisito", status, elapsed))
                if status == "FAIL":
                    print(f"\n  ✗  02c_{detect_variant[0]} fallito — "
                          f"i metodi detected/{detect_variant} potrebbero usare il fallback (shock date).")
        else:
            print(f"\n  ⚙  --skip-02c: salto 02c, uso theta_results.csv esistente.")

        # ── B2. Per ogni metodo ITS: gira tutte e 3 le varianti ──────────────
        for key, filename, label in ITS_STEPS:
            if key not in its_targets:
                continue

            for variant_tag, variant_args in ITS_VARIANTS:
                tag = f"{key}[{variant_tag}]"
                print_banner(tag, f"{label}  [{variant_tag}]")
                status, elapsed = run_step(filename, variant_args)
                results.append((tag, label, variant_tag, status, elapsed))

    # ══════════════════════════════════════════════════════════════════════════
    # Riepilogo
    # ══════════════════════════════════════════════════════════════════════════
    total_elapsed = time.time() - total_start
    print(f"\n{SEP}")
    print("  RIEPILOGO PIPELINE")
    print(f"{SEP}")
    for key, label, variant, status, elapsed in results:
        icon    = "✓" if status == "OK" else ("⚠" if status == "SKIP" else "✗")
        var_str = f"[{variant}]" if variant != "–" else ""
        print(f"  {icon}  {key:<35}  {var_str:<22}  {status:<5}  {elapsed:.1f}s")
    print(f"{SEP}")
    print(f"  Totale: {total_elapsed:.1f}s")
    print(f"{SEP}\n")

    if any(s == "FAIL" for *_, s, _ in results):
        sys.exit(1)


if __name__ == "__main__":
    main()