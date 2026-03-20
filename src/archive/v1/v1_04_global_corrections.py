"""
04_global_corrections.py
=========================
Raccoglie TUTTI i p-value prodotti dagli script 02 e 03 e applica una
Benjamini-Hochberg correction GLOBALE a livello di paper.

MOTIVAZIONE METODOLOGICA
──────────────────────────
Lo script 02 applica BH solo ai test di Table 2 (margine, ~6-8 test).
Lo script 03 produce decine di test aggiuntivi (Granger, R&F, KS, ANOVA,
Chow, DiD) che non partecipano alla correction.
Controllare il FDR solo su un subset gonfia artificialmente il Type I error
globale: con 50 test a α=0.05 ci aspettiamo ~2.5 falsi positivi per caso.

APPROCCIO
──────────
• Tutti i p-value sono marcati come "confirmatory" o "exploratory".
• La BH globale è applicata solo ai confirmatory (test primari H0).
• I test explorativi sono riportati ma non entrano nella correction globale.
  Rif: Benjamini & Hochberg (1995); Tukey (1991) distinzione conf./expl.

CONFIRMATORY (test primari su H0 margine):
  - Welch t-test su Δmargine (table2_margin_anomaly.csv → colonna t_p)
  - DiD δ su margine (did_results.csv → colonna p_value)

EXPLORATORY (evidenza ausiliaria):
  - Granger (granger_*.csv)
  - Rockets & Feathers (rockets_feathers_results.csv)
  - KS test su prezzi (ks_results.csv)
  - ANOVA (anova_results.csv)
  - Chow test (chow_results.csv)

Output:
  data/global_bh_corrections.csv   → tutti i p-value con BH globale
  data/table2_margin_anomaly.csv   → aggiornato con colonna BH_global_reject
"""

import os
import pandas as pd
import numpy as np


def bh_correction(p_values, alpha=0.05):
    """Benjamini-Hochberg FDR correction. Restituisce (reject_array, adjusted_p)."""
    p = np.array(p_values, dtype=float)
    n = len(p)
    if n == 0:
        return np.array([], dtype=bool), np.array([])
    order   = np.argsort(p)
    ranked  = np.empty(n, dtype=float)
    ranked[order] = np.arange(1, n + 1)
    p_adj   = np.minimum(1.0, p * n / ranked)
    # Monotonicity: adjusted p-value = min(p_adj[k:]) per ciascun rango k
    p_adj_mono = np.minimum.accumulate(p_adj[order][::-1])[::-1]
    p_adj_out  = np.empty(n)
    p_adj_out[order] = p_adj_mono
    reject = p_adj_out <= alpha
    return reject, p_adj_out


ALPHA = 0.05

os.makedirs("data", exist_ok=True)

all_rows = []

# ── 1. CONFIRMATORY: Welch t-test su Δmargine (Table 2) ──────────────────────
t2_path = "data/table2_margin_anomaly.csv"
if os.path.exists(t2_path):
    df_t2 = pd.read_csv(t2_path)
    for _, row in df_t2.iterrows():
        all_rows.append({
            "fonte":        "table2_Welch_t",
            "tipo":         "confirmatory",
            "descrizione":  f"Δmargine {row.get('Evento','?')} | {row.get('Serie','?')} | {row.get('Metodo','?')}",
            "p_value":      float(row["t_p"]),
        })
    print(f"  table2 → {len(df_t2)} test Welch t caricati")
else:
    print(f"  SKIP {t2_path} (non trovato — eseguire 02_core_analysis.py)")

# ── 2. CONFIRMATORY: DiD δ ───────────────────────────────────────────────────
did_path = "data/did_results.csv"
if os.path.exists(did_path):
    df_did = pd.read_csv(did_path)
    for _, row in df_did.iterrows():
        paese = row.get("Paese_controllo", row.get("Carburante", "?"))
        all_rows.append({
            "fonte":       "DiD",
            "tipo":        "confirmatory",
            "descrizione": f"DiD {row.get('Evento','?')} | IT vs {paese} | {row.get('Carburante','?')}",
            "p_value":     float(row["p_value"]),
        })
    print(f"  DiD → {len(df_did)} test caricati")
else:
    print(f"  SKIP {did_path} (non trovato)")

# ── 3. EXPLORATORY: Granger ──────────────────────────────────────────────────
for fuel in ["benzina", "diesel"]:
    path = f"data/granger_{fuel}.csv"
    if os.path.exists(path):
        df_g = pd.read_csv(path)
        for _, row in df_g.iterrows():
            all_rows.append({
                "fonte":       f"Granger_{fuel}",
                "tipo":        "exploratory",
                "descrizione": f"Granger {fuel} lag={row.get('lag_weeks','?')}w",
                "p_value":     float(row["p_value"]),
            })
        print(f"  Granger {fuel} → {len(df_g)} test caricati")

# ── 4. EXPLORATORY: Rockets & Feathers ───────────────────────────────────────
rf_path = "data/rockets_feathers_results.csv"
if os.path.exists(rf_path):
    df_rf = pd.read_csv(rf_path)
    for _, row in df_rf.iterrows():
        for pcol in [c for c in df_rf.columns if "p_value" in c.lower() or c.lower() == "p"]:
            try:
                all_rows.append({
                    "fonte":       "R&F",
                    "tipo":        "exploratory",
                    "descrizione": f"R&F {row.get('Carburante', row.get('fuel','?'))} {pcol}",
                    "p_value":     float(row[pcol]),
                })
            except (ValueError, TypeError):
                pass
    print(f"  R&F → {len(df_rf)} righe caricate")

# ── 5. EXPLORATORY: KS test ──────────────────────────────────────────────────
ks_path = "data/ks_results.csv"
if os.path.exists(ks_path):
    df_ks = pd.read_csv(ks_path)
    for _, row in df_ks.iterrows():
        for pcol in [c for c in df_ks.columns if "p" in c.lower()]:
            try:
                all_rows.append({
                    "fonte":       "KS",
                    "tipo":        "exploratory",
                    "descrizione": f"KS {row.get('Evento','?')} | {row.get('Carburante','?')}",
                    "p_value":     float(row[pcol]),
                })
                break
            except (ValueError, TypeError):
                pass
    print(f"  KS → {len(df_ks)} test caricati")

# ── 6. EXPLORATORY: ANOVA ────────────────────────────────────────────────────
anova_path = "data/anova_results.csv"
if os.path.exists(anova_path):
    df_an = pd.read_csv(anova_path)
    for _, row in df_an.iterrows():
        for pcol in [c for c in df_an.columns if "p" in c.lower()]:
            try:
                all_rows.append({
                    "fonte":       "ANOVA",
                    "tipo":        "exploratory",
                    "descrizione": f"ANOVA {row.get('Evento','?')} | {row.get('Carburante','?')}",
                    "p_value":     float(row[pcol]),
                })
                break
            except (ValueError, TypeError):
                pass
    print(f"  ANOVA → {len(df_an)} test caricati")

# ── 7. EXPLORATORY: Chow test ────────────────────────────────────────────────
chow_path = "data/chow_results.csv"
if os.path.exists(chow_path):
    df_ch = pd.read_csv(chow_path)
    for _, row in df_ch.iterrows():
        for pcol in [c for c in df_ch.columns if "p" in c.lower()]:
            try:
                val = row[pcol]
                if val == "N/A":
                    continue
                all_rows.append({
                    "fonte":       "Chow",
                    "tipo":        "exploratory",
                    "descrizione": f"Chow {row.get('Evento','?')} | {row.get('Carburante','?')}",
                    "p_value":     float(val),
                })
                break
            except (ValueError, TypeError):
                pass
    print(f"  Chow → {len(df_ch)} test caricati")


# ── Applica BH GLOBALE solo ai confirmatory ───────────────────────────────────
if not all_rows:
    print("\n  Nessun p-value caricato — eseguire prima 02 e 03.")
else:
    df_all = pd.DataFrame(all_rows)
    df_all["p_value"] = pd.to_numeric(df_all["p_value"], errors="coerce")
    df_all = df_all.dropna(subset=["p_value"])

    # BH globale su confirmatory
    conf_mask = df_all["tipo"] == "confirmatory"
    if conf_mask.sum() > 0:
        p_conf = df_all.loc[conf_mask, "p_value"].values
        reject_conf, p_adj_conf = bh_correction(p_conf, alpha=ALPHA)
        df_all.loc[conf_mask, "BH_global_reject"]   = reject_conf
        df_all.loc[conf_mask, "p_value_BH_adjusted"] = p_adj_conf
    else:
        print("  Nessun test confirmatory trovato.")

    # Exploratory: riporta p-value nominale con flag "exploratory — no correction"
    df_all.loc[~conf_mask, "BH_global_reject"]    = np.nan
    df_all.loc[~conf_mask, "p_value_BH_adjusted"] = np.nan

    df_all.to_csv("data/global_bh_corrections.csv", index=False)
    print(f"\n  Salvato: data/global_bh_corrections.csv ({len(df_all)} p-value totali)")

    n_conf   = conf_mask.sum()
    n_expl   = (~conf_mask).sum()
    n_reject = int(df_all.loc[conf_mask, "BH_global_reject"].sum()) if conf_mask.sum() > 0 else 0
    print(f"  Confirmatory: {n_conf} test  |  BH global rigettati: {n_reject} / {n_conf}")
    print(f"  Exploratory:  {n_expl} test  |  (nessuna correction applicata — evidenza ausiliaria)")

    # ── Aggiorna table2 con colonna BH_global_reject ──────────────────────────
    if os.path.exists(t2_path):
        df_t2    = pd.read_csv(t2_path)
        t2_conf  = df_all[df_all["fonte"] == "table2_Welch_t"].reset_index(drop=True)
        if len(t2_conf) == len(df_t2):
            df_t2["BH_global_reject"] = t2_conf["BH_global_reject"].values
            df_t2["t_p_BH_adjusted"]  = t2_conf["p_value_BH_adjusted"].values
            df_t2.to_csv(t2_path, index=False)
            print(f"  Aggiornato: {t2_path} (colonne BH_global_reject, t_p_BH_adjusted aggiunte)")
        else:
            print(f"  ATTENZIONE: mismatch righe table2 ({len(df_t2)}) vs confirmatory ({len(t2_conf)})")

    # ── Sommario leggibile ────────────────────────────────────────────────────
    print(f"\n{'─'*65}")
    print("  SOMMARIO BH GLOBALE — test confirmatory")
    print(f"{'─'*65}")
    for _, row in df_all[conf_mask].sort_values("p_value").iterrows():
        rej = "✓ RIGETTATA" if row["BH_global_reject"] else "  non rigett."
        print(f"  {rej} | p={row['p_value']:.4f} → p_adj={row['p_value_BH_adjusted']:.4f} "
              f"| {row['descrizione'][:55]}")

print("\nScript 04 completato.")
print("  Per leggere i risultati:")
print("  import pandas as pd")
print("  g = pd.read_csv('data/global_bh_corrections.csv')")
print("  g[g['tipo']=='confirmatory']")