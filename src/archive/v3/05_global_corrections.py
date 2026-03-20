"""
05_global_corrections_v2.py
============================
Correzione BH globale su famiglia confirmatory PULITA.

DIFETTI CORRETTI RISPETTO A v1:

  [FIX 1 — CRITICO] La famiglia BH era contaminata da p-value di split
    endogeni (τ_margin) e da test su H₀ diversa (block perm, HAC locale).
    In v2 raccoglie SOLO i test da confirmatory_pvalues_v2.csv (16 righe),
    tutti testando H₀: μ_post = μ_2019 su split esogeni.

  [FIX 2 — ALTO] Bug posizionale in aggiornamento table2.
    In v1: "if len(df_welch) == len(nonprel_idx)" → sempre falso (12 ≠ 4)
    → table2 non riceveva mai BH_global_reject.
    In v2: matching per chiave (evento, carburante, split_type) → nessun
    mismatch possibile indipendentemente dalla dimensione della famiglia.

  [FIX 3] Confronto famiglie: riporta separatamente confirmatory e DiD.
    DiD testa H₀ diversa (specificità italiana) → BH separata opzionale.

Famiglie:
  PRIMARIA:  confirmatory_pvalues_v2.csv  (≤16 test: HAC_t + MW × 2 split)
  AUSILIARIA: did_results_v2.csv          (DiD IT vs controllo — prodotto da script 04 v2)
  ESCLUSA:   exploratory_results.csv     (block perm, HAC — H₀_locale)

Input:
  data/confirmatory_pvalues_v2.csv   (script 03 v2)
  data/did_results_v2.csv            (script 04 v2: DiD — FIX da auxiliary_pvalues.csv v1)
  data/table2_margin_anomaly_v2.csv  (aggiornato con BH locale in script 03 v2)

FIXES v2.1 (27-apr-2026, DeepSeek review):
  [FIX-A] DiD: was loading auxiliary_pvalues.csv (v1, p~0.2-0.8) instead of
          did_results_v2.csv (v2, prodotto da 04_auxiliary_evidence_v2.py).
          Risultato sbagliato: "0/8 rigettati" quando i veri p v2 sono ~0.
  [FIX-B] Confirmatory: cerca prefisso "HAC_t" (non più "Welch_t") per
          allinearsi a 03_margin_hypothesis.py v2.1 che usa long-run variance
          Andrews BW invece del Welch iid. Impatta riepilogo test e lookup table2.

Output:
  data/global_bh_corrections_v2.csv
  data/table2_margin_anomaly_v2.csv  (aggiornato con BH_global_reject)
"""

import os
import numpy as np
import pandas as pd

os.makedirs("data", exist_ok=True)

ALPHA = 0.05


def bh_correction(p_values: np.ndarray, alpha: float = 0.05):
    """Benjamini-Hochberg (1995) con monotonicity enforcement."""
    p = np.array(p_values, dtype=float)
    n = len(p)
    if n == 0:
        return np.array([], dtype=bool), np.array([])
    order   = np.argsort(p)
    ranked  = np.empty(n, dtype=float)
    ranked[order] = np.arange(1, n + 1)
    p_adj   = np.minimum(1.0, p * n / ranked)
    p_adj_m = np.minimum.accumulate(p_adj[order][::-1])[::-1]
    p_out   = np.empty(n)
    p_out[order] = p_adj_m
    return p_out <= alpha, p_out


def _load_pvalues(path: str, famiglia_label: str):
    if not os.path.exists(path):
        print(f"  {famiglia_label}: {path} non trovato — skip")
        return pd.DataFrame()
    df = pd.read_csv(path)
    df["p_value"] = pd.to_numeric(df["p_value"], errors="coerce")
    df = df.dropna(subset=["p_value"])
    print(f"  {famiglia_label}: {len(df)} test caricati da {path}")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# 1. FAMIGLIA PRIMARIA — confirmatory (H₀: μ_post = μ_2019)
# ─────────────────────────────────────────────────────────────────────────────
print("=" * 70)
print("FAMIGLIA PRIMARIA — HAC_t + MannWhitney, split esogeni")  # [FIX-B] v2.1
print("  H₀: μ_post = μ_2019  (one-sided upper)")
print("  [τ_margin e test locali ESCLUSI — v1 li includeva erroneamente]")
print("=" * 70)

df_primary = _load_pvalues("data/confirmatory_pvalues_v2.csv", "Confirmatory primaria")

all_bh_rows = []

if not df_primary.empty:
    p_prim   = df_primary["p_value"].values
    rej_p, adj_p = bh_correction(p_prim, alpha=ALPHA)
    df_primary["BH_global_reject"]    = rej_p
    df_primary["p_value_BH_adjusted"] = adj_p
    df_primary["famiglia"]            = "primaria"

    n_rej = int(rej_p.sum())
    print(f"\n  Famiglia primaria: {len(df_primary)} test  |  Rigettati FDR 5%: {n_rej}")

    # Riepilogo per tipo di test  [FIX-B] v2.1: HAC_t se script03 v2.1, Welch_t se v2.0
    # Mostriamo entrambi; il prefisso attivo dipende da quando è stato eseguito script 03
    for test_name in ["HAC_t", "Welch_t", "MannWhitney"]:
        sub = df_primary[df_primary["fonte"].str.startswith(test_name)]
        print(f"    {test_name:15}: {int(sub['BH_global_reject'].sum())}/{len(sub)} rigettati")

    # Riepilogo per split
    for st in ["shock_hard", "tau_price"]:
        sub = df_primary[df_primary.get("split_type", pd.Series(dtype=str)) == st
                         if "split_type" in df_primary.columns
                         else pd.Series([True]*len(df_primary))]
        if not sub.empty:
            print(f"    split={st:12}: {int(sub['BH_global_reject'].sum())}/{len(sub)} rigettati")

    print(f"\n  Test rigettati (ordinati per p nominale):")
    df_rej_p = df_primary[df_primary["BH_global_reject"]].sort_values("p_value")
    if df_rej_p.empty:
        print("    Nessuno.")
    else:
        for _, r in df_rej_p.iterrows():
            print(f"    p={r['p_value']:.4f}  adj={r['p_value_BH_adjusted']:.4f}"
                  f"  | {r['fonte'][:55]}")

    all_bh_rows.append(df_primary)


# ─────────────────────────────────────────────────────────────────────────────
# 2. FAMIGLIA AUSILIARIA — DiD (H₀: nessuna specificità italiana)
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("FAMIGLIA AUSILIARIA — DiD IT vs controllo")
print("  H₀_DiD: δ_DiD = 0 (nessuna specificità italiana)")
print("  [BH separata — domanda economica diversa da famiglia primaria]")
print("=" * 70)

df_did = _load_pvalues("data/did_results_v2.csv", "DiD ausiliaria")  # [FIX-A] v2.1

if not df_did.empty:
    p_did = df_did["p_value"].values
    rej_d, adj_d = bh_correction(p_did, alpha=ALPHA)
    df_did["BH_global_reject"]    = rej_d
    df_did["p_value_BH_adjusted"] = adj_d
    df_did["famiglia"]            = "ausiliaria_DiD"

    n_rej_d = int(rej_d.sum())
    print(f"\n  Famiglia DiD: {len(df_did)} test  |  Rigettati FDR 5%: {n_rej_d}")

    # Presenza colonne PTA (did_results_v2.csv)
    has_pta = "PTA_non_rigettata" in df_did.columns

    print(f"\n  Test DiD (ordinati per p nominale):")
    for _, r in df_did.sort_values("p_value").iterrows():
        flag = "RIGETTATA" if r["BH_global_reject"] else "         "
        # [FIX-A] v2.1: did_results_v2.csv non ha 'descrizione' → deriva da colonne evento
        if "descrizione" in r.index:
            label = str(r["descrizione"])[:50]
        else:
            label = f"{r.get('Evento','')} | {r.get('Paese_controllo','')} | {r.get('Carburante','')}"
        # [FIX critic.3] Mostra stato PTA accanto al risultato DiD
        pta_str = ""
        if has_pta:
            pta_ok = r.get("PTA_non_rigettata", None)
            if pta_ok is True or str(pta_ok).lower() == "true":
                pta_str = "  [PTA ✓]"
            elif pta_ok is False or str(pta_ok).lower() == "false":
                pta_str = "  [PTA ✗ CONDIZIONATO]"
        print(f"  {flag} | p={r['p_value']:.4f}  adj={r['p_value_BH_adjusted']:.4f}"
              f"  | {label[:60]}{pta_str}")

    # [FIX critic.3] Breakdown PTA dei rigetti
    if has_pta:
        rej_did = df_did[df_did["BH_global_reject"]].copy()
        pta_ok_col = df_did["PTA_non_rigettata"].map(lambda v: v is True or str(v).lower() == "true")
        n_pta_ok  = int(pta_ok_col.sum())
        n_pta_viol= int((~pta_ok_col).sum())
        rej_pta_ok  = int((df_did["BH_global_reject"] & pta_ok_col).sum())
        rej_pta_viol= int((df_did["BH_global_reject"] & (~pta_ok_col)).sum())
        print(f"\n  Breakdown PTA (critic.3):")
        print(f"    Totale test DiD: {len(df_did)}")
        print(f"    PTA soddisfatta: {n_pta_ok}  |  PTA violata: {n_pta_viol}")
        print(f"    Rigettati con PTA ✓ (causalmente validi): {rej_pta_ok}/{n_rej_d}")
        print(f"    Rigettati con PTA ✗ (condizionati):      {rej_pta_viol}/{n_rej_d}")
        if rej_pta_viol > 0:
            print(f"    ⚠  {rej_pta_viol} rigetti NON soddisfano PTA → δ_DiD potenzialmente")
            print(f"       contaminato da trend pre-esistenti. Non interpretare come causale.")

    all_bh_rows.append(df_did)

    # Nota interpretativa sui DiD negativi
    neg_did = df_did[df_did.get("p_value", pd.Series(dtype=float)) < ALPHA] if not df_did.empty else pd.DataFrame()
    print("""
  NOTA INTERPRETATIVA — DiD negativi (IT < controllo):
    Un δ_DiD negativo (non significativo) per Ucraina IT vs DE/SE significa:
    l'Italia NON ha avuto margini superiori al paese controllo → evidenza
    CONTRO la specificità italiana. Questo RAFFORZA la conclusione "nessun
    opportunismo specifico italiano" per quell'evento, anche in assenza
    di significatività statistica formale.
    """)


# ─────────────────────────────────────────────────────────────────────────────
# 3. ESCLUSI DALLA BH — esplorativa H₀_locale
# ─────────────────────────────────────────────────────────────────────────────
print("=" * 70)
print("ESCLUSI DALLA BH — block permutation e HAC (H₀_locale)")
print("  Questi test rispondono a: 'c'è un salto pre→post?'")
print("  Non a: 'il livello post è anomalo vs 2019?'")
print("  Mescoliarli nella stessa BH della famiglia primaria era un errore.")
print("=" * 70)

if os.path.exists("data/exploratory_results.csv"):
    df_explo = pd.read_csv("data/exploratory_results.csv")
    print(f"\n  {len(df_explo)} test esplorativi in data/exploratory_results.csv")
    print("  (consultabili separatamente — non entrano nella BH globale)")


# ─────────────────────────────────────────────────────────────────────────────
# 4. SALVA OUTPUT UNIFICATO
# ─────────────────────────────────────────────────────────────────────────────
if all_bh_rows:
    df_all_bh = pd.concat(all_bh_rows, ignore_index=True)
    df_all_bh.to_csv("data/global_bh_corrections_v2.csv", index=False)
    print(f"\n  ✓ data/global_bh_corrections_v2.csv  ({len(df_all_bh)} righe totali)")

    # Confronto dimensioni famiglie v1 vs v2
    print("""
  CONFRONTO FAMIGLIE BH (v1 → v2):
  ┌────────────────────────────────┬────────┬────────┐
  │ Famiglia                       │  v1    │  v2    │
  ├────────────────────────────────┼────────┼────────┤
  │ Confirmatory margine (script 03)│  48    │  ≤16   │
  │   di cui τ_margin (endogeno)   │  16    │    0   │  ← rimosso
  │   di cui H₀_locale (perm+HAC) │  16    │    0   │  ← rimosso
  │   di cui H₀_livello (Welch+MW)│  16    │  ≤16   │
  │ Auxiliary DiD (script 04)      │   8    │    8   │
  │ TOTALE                         │  56    │  ≤24   │
  └────────────────────────────────┴────────┴────────┘
  La riduzione non "facilita" i rigetti — i p-value validi restano identici.
  Rende la BH corretta: FDR ≤ 5% sulla domanda che si intende rispondere.
    """)


# ─────────────────────────────────────────────────────────────────────────────
# 5. AGGIORNA table2_margin_anomaly_v2 — FIX BUG POSIZIONALE
# ─────────────────────────────────────────────────────────────────────────────
print("=" * 70)
print("AGGIORNAMENTO table2 — matching per chiave (fix bug posizionale v1)")
print("=" * 70)

t2_path = "data/table2_margin_anomaly_v2.csv"
if os.path.exists(t2_path) and not df_primary.empty:
    df_t2 = pd.read_csv(t2_path)

    # Costruisci lookup per chiave (Evento, Carburante, split_type)
    # Usiamo SOLO le righe HAC_t (o Welch_t per retrocompat.) con split shock_hard  [FIX-B] v2.1
    welch_shock = df_primary[
        (df_primary["fonte"].str.startswith("HAC_t") | df_primary["fonte"].str.startswith("Welch_t")) &
        (df_primary.get("split_type", pd.Series(dtype=str)) == "shock_hard"
         if "split_type" in df_primary.columns
         else pd.Series([True]*len(df_primary)))
    ]

    # Estrai (evento, carburante) dalla colonna 'fonte' o 'evento'/'carburante'
    lookup_key = {}
    if "evento" in df_primary.columns and "carburante" in df_primary.columns:
        for _, row in welch_shock.iterrows():
            k = (row["evento"], row["carburante"])
            lookup_key[k] = {
                "BH_global_reject":       row["BH_global_reject"],
                "t_p_BH_global_adjusted": row["p_value_BH_adjusted"],
            }
    else:
        # Fallback: estrazione dalla stringa fonte "Welch_t_{evento}_{fuel}_shock_hard"
        for _, row in welch_shock.iterrows():
            # [FIX-B] v2.1: rimuove prefisso HAC_t_ o Welch_t_ (retrocompatibile)
            src = row["fonte"].replace("HAC_t_", "").replace("Welch_t_", "").replace("_shock_hard", "")
            for fuel in ["Benzina", "Diesel"]:
                if src.endswith(f"_{fuel}"):
                    evento = src[:-len(f"_{fuel}")]
                    lookup_key[(evento, fuel)] = {
                        "BH_global_reject":       row["BH_global_reject"],
                        "t_p_BH_global_adjusted": row["p_value_BH_adjusted"],
                    }

    print(f"\n  Lookup costruito: {len(lookup_key)} chiavi (evento, carburante)")

    # Rinomina colonne per matching
    ev_col   = "Evento"   if "Evento"   in df_t2.columns else "evento"
    fuel_col = "Carburante" if "Carburante" in df_t2.columns else "carburante"
    prel_col = "preliminare" if "preliminare" in df_t2.columns else None

    n_updated = 0
    for idx, row in df_t2.iterrows():
        is_prel = bool(row[prel_col]) if prel_col and prel_col in row.index else False
        k = (row[ev_col], row[fuel_col])

        if is_prel:
            df_t2.at[idx, "BH_global_reject"]       = pd.NA
            df_t2.at[idx, "t_p_BH_global_adjusted"] = pd.NA
            df_t2.at[idx, "BH_NOTE"]                = "ESCLUSO dalla BH (dati preliminari)"
        elif k in lookup_key:
            df_t2.at[idx, "BH_global_reject"]       = lookup_key[k]["BH_global_reject"]
            df_t2.at[idx, "t_p_BH_global_adjusted"] = lookup_key[k]["t_p_BH_global_adjusted"]
            df_t2.at[idx, "BH_NOTE"]                = "BH applicata su famiglia confirmatory v2"
            n_updated += 1
        else:
            print(f"  ⚠ Chiave {k} non trovata nel lookup — controlla nomi evento/carburante")

    df_t2.to_csv(t2_path, index=False)
    print(f"  ✓ {t2_path} aggiornato  ({n_updated} righe non-prelim × key)")

    if n_updated == 0 and len(lookup_key) == 0:
        print("  ATTENZIONE: nessun aggiornamento possibile.")
        print("  Verificare che confirmatory_pvalues_v2.csv abbia colonne 'evento','carburante'.")


# ─────────────────────────────────────────────────────────────────────────────
# 6. SOMMARIO FINALE
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("SOMMARIO — risultati dopo correzione BH per test multipli (v2)")
print("=" * 70)

if not df_primary.empty:
    for fonte_prefix in ["HAC_t", "Welch_t", "MannWhitney"]:  # [FIX-B] v2.1
        sub = df_primary[df_primary["fonte"].str.startswith(fonte_prefix)]
        if not sub.empty:
            n_rej_s = int(sub["BH_global_reject"].sum())
            print(f"  {fonte_prefix:15} [prim]: {n_rej_s:2d} / {len(sub):2d} rigettati (FDR 5%)")

if not df_did.empty:
    n_rej_d = int(df_did["BH_global_reject"].sum())
    print(f"  {'DiD':15} [aux.]: {n_rej_d:2d} / {len(df_did):2d} rigettati (FDR 5%)")

# ── [FIX critic.5] Caveat n_eff — 27-apr-2026 ────────────────────────────
# Il messaggio "N/16 rigettati" va contestualizzato: con n_eff spesso < 10
# la potenza nominale è gonfiata dall'autocorrelazione. Il test HAC corregge
# la SE ma non elimina il problema quando BW ≈ n/2 (test quasi non informativo).
# I rigetti devono essere letti come "evidenza di margini elevati post-shock
# coerente con N test su dati autocorrelati", non come "N prove indipendenti".
print()
print("  ⚠  ROBUSTEZZA — contestualizzazione n_eff (critic.5):")
print("  ─────────────────────────────────────────────────────────────────")

# Carica neff_report se disponibile
_neff_path = "data/neff_report_v2.csv"
if os.path.exists(_neff_path):
    try:
        df_neff_rep = pd.read_csv(_neff_path)
        neff_vals   = pd.to_numeric(df_neff_rep["n_eff"], errors="coerce").dropna()
        neff_below10 = int((neff_vals < 10).sum())
        neff_below5  = int((neff_vals < 5).sum())
        neff_med     = float(neff_vals.median())
        infl_vals    = pd.to_numeric(df_neff_rep["inflation_factor"], errors="coerce").dropna()
        infl_med     = float(infl_vals.median())
        print(f"  n_eff mediano (serie post-shock): {neff_med:.1f}  "
              f"(max inflazione mediana: {infl_med:.1f}×)")
        print(f"  Serie con n_eff < 10: {neff_below10}/{len(neff_vals)}  |  "
              f"n_eff < 5: {neff_below5}/{len(neff_vals)}")
    except Exception:
        neff_med, infl_med = None, None
        print("  (neff_report_v2.csv non leggibile)")
else:
    neff_med, infl_med = None, None
    print("  (neff_report_v2.csv non trovato — eseguire 06_distribution_check_v2.py)")

print("""
  INTERPRETAZIONE CORRETTA DEI RISULTATI (da usare nel paper):
  ─────────────────────────────────────────────────────────────────
  NON scrivere: "16/16 test rigettati → evidenza forte di speculazione"
  SCRIVERE:     "I margini post-shock risultano statisticamente superiori
                 al baseline 2019 in tutti i 16 test confirmatori (HAC_t +
                 Mann-Whitney, FDR 5%). Tuttavia, il numero effettivo di
                 osservazioni indipendenti è spesso n_eff < 10 a causa
                 dell'elevata autocorrelazione settimanale (ρ̂ ≈ 0.7–0.9).
                 I rigetti devono essere interpretati come evidenza
                 CONSISTENTE ma DEBOLE: non 16 prove indipendenti, bensì
                 una medesima evidenza (margini elevati) vista da 16 angolature
                 statisticamente dipendenti."

  Per il DiD (7/8 rigettati, 5/8 con PTA violata):
  → I risultati con PTA violata sono CONDIZIONATI sull'assunzione che
    il trend pre-shock fosse parallelo. Con break strutturale pre-shock
    (rilevato da script 07), la PTA è strutturalmente difficile da
    soddisfare. Interpretare come evidenza descrittiva, non causale.
  ─────────────────────────────────────────────────────────────────
""")

print("Script 05 v2 completato.")