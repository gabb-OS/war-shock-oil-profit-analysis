"""
3_04_bh.py  — Correzione BH + sommario finale (v3, completo)
=============================================================
Applica Benjamini-Hochberg (FDR 5%) per famiglia di test e produce
il sommario delle tre sub-ipotesi con tabella risultati formattata.

STRUTTURA FAMIGLIE
──────────────────
  Famiglia A (H₀_i)   — livello reale vs 2019:         8 test (HAC_t + MW)
  Famiglia B (H₀_ii)  — salto pre→post shock:          8 test (HAC_t + MW)
  Famiglia C (H₀_iii) — specificità italiana (DiD):   N×4 test (OLS HC3)
                         N = paesi controllo (DE, NL, BE, DK, FI, AT)
                         = 7 paesi × 2 eventi × 2 carburanti = 28 test

BH separata per famiglia perché ogni famiglia risponde a una domanda
economica distinta. H₀ macro rifiutata se ≥1 famiglia rigetta.

NOTA n_eff:
  I rigetti vanno interpretati come "evidenza consistente" non come
  "N prove indipendenti". Con autocorrelazione alta (ρ̂ ≈ 0.7–0.9)
  si hanno spesso n_eff < 10: il test HAC corregge la SE ma la
  potenza nominale rimane limitata.

Input:
  data/3_AB.csv           (da 3_02_tests.py)
  data/3_C.csv            (da 3_03_did.py)
  data/3_neff_report.csv  (da 3_02_tests.py)

Output:
  data/3_bh.csv             — tutti i test con BH reject per famiglia
  data/3_table_results.csv  — tabella riassuntiva formattata per paper
  plots/3_04_summary.png    — heatmap risultati BH per famiglia
"""

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

warnings.filterwarnings("ignore")
os.makedirs("data",  exist_ok=True)
os.makedirs("plots", exist_ok=True)

ALPHA = 0.05


# ════════════════════════════════════════════════════════════════════════════
# FUNZIONE BH
# ════════════════════════════════════════════════════════════════════════════

def bh_correction(p_values: np.ndarray, alpha: float = 0.05):
    """Benjamini-Hochberg (1995) — step-up con monotonicity enforcement."""
    p = np.asarray(p_values, dtype=float)
    n = len(p)
    if n == 0:
        return np.array([], dtype=bool), np.array([])
    order   = np.argsort(p)
    ranked  = np.empty(n, dtype=float)
    ranked[order] = np.arange(1, n + 1)
    p_adj   = np.minimum(1.0, p * n / ranked)
    p_mono  = np.minimum.accumulate(p_adj[order][::-1])[::-1]
    p_out   = np.empty(n)
    p_out[order] = p_mono
    return p_out <= alpha, p_out


# ════════════════════════════════════════════════════════════════════════════
# CARICA FAMIGLIE
# ════════════════════════════════════════════════════════════════════════════
print("Carico risultati famiglie A, B, C...")

frames = []
for path, label in [("data/3_AB.csv", "A+B"), ("data/3_C.csv", "C")]:
    if os.path.exists(path):
        df_ = pd.read_csv(path)
        frames.append(df_)
        print(f"   {path}: {len(df_)} test")
    else:
        print(f"   MANCANTE: {path} — eseguire gli script precedenti")

if not frames:
    raise SystemExit("Nessun dato disponibile. Eseguire prima 3_02 e 3_03.")

df_all = pd.concat(frames, ignore_index=True)
df_all["p_value"] = pd.to_numeric(df_all["p_value"], errors="coerce")
df_all["BH_reject"] = False
df_all["p_BH_adj"]  = np.nan


# ════════════════════════════════════════════════════════════════════════════
# BH PER FAMIGLIA
# ════════════════════════════════════════════════════════════════════════════
print("\n" + "="*62)
print("CORREZIONE BH (FDR 5%) per famiglia di test")
print("="*62)

FAMILY_LABELS = {
    "A": "Analisi strutturale — livello reale vs 2019 (complementare, non H₀ dichiarata)",
    "B": "H₀(i)  — salto pre→post shock (test causale dichiarato)",
    "C": "H₀(iii) — specificità italiana vs EU (DiD)",
}

family_summary = {}

for fam in ["A", "B", "C"]:
    mask  = df_all["famiglia"] == fam
    sub   = df_all[mask].copy()
    label = FAMILY_LABELS.get(fam, fam)

    if sub.empty:
        print(f"\n   Famiglia {fam}: nessun test disponibile")
        family_summary[fam] = {"label": label, "n": 0, "n_rej": 0, "rejected": False}
        continue

    p_vals = sub["p_value"].values
    valid  = ~np.isnan(p_vals)

    rej_all = np.zeros(len(p_vals), dtype=bool)
    adj_all = np.full(len(p_vals), np.nan)
    if valid.sum() > 0:
        rej_v, adj_v = bh_correction(p_vals[valid], alpha=ALPHA)
        rej_all[valid] = rej_v
        adj_all[valid] = adj_v

    df_all.loc[mask, "BH_reject"] = rej_all
    df_all.loc[mask, "p_BH_adj"]  = adj_all

    n_rej    = int(rej_all.sum())
    # Per Family C: H₀(iii) è rifiutata solo se esistono δ>0 significativi.
    # δ<0 significativi vanno nella direzione opposta a H₁ e non la supportano.
    if fam == "C" and "delta_DiD_EUR_L" in df_all.columns:
        n_rej_h1 = int(df_all[mask & df_all["BH_reject"] &
                               (df_all["delta_DiD_EUR_L"] > 0)].shape[0])
        rejected = n_rej_h1 > 0
    else:
        rejected = n_rej > 0

    print(f"\n   Famiglia {fam} — {label}")
    print(f"   {n_rej}/{len(sub)} test rigettati a FDR 5%")

    sub_up = df_all[mask]

    if fam in ("A", "B"):
        for test_t in sorted(sub_up["test"].dropna().unique()):
            ts = sub_up[sub_up["test"] == test_t]
            print(f"     {test_t:<15}: {int(ts['BH_reject'].sum())}/{len(ts)}")
        for ev in sub_up["evento"].dropna().unique():
            es = sub_up[sub_up["evento"] == ev]
            rej_e = int(es["BH_reject"].sum())
            flag  = "✓" if rej_e > 0 else " "
            print(f"     {flag} {ev[:40]}: {rej_e}/{len(es)} rigettati")
    else:  # C
        n_rej_pos = 0  # δ>0 significativi (evidenza contro H₀(iii))
        n_rej_neg = 0  # δ<0 significativi (a favore di IT)
        for ev in sub_up["evento"].dropna().unique():
            for paese in sub_up["paese_controllo"].dropna().unique():
                ep = sub_up[(sub_up["evento"] == ev) & (sub_up["paese_controllo"] == paese)]
                if ep.empty:
                    continue
                rej_ep  = int(ep["BH_reject"].sum())
                n_pta_ok = ep["PTA_non_rigettata"].astype(str).str.lower().eq("true").sum()
                # Conta direzione del δ̂ per i test rigettati
                if "delta_DiD_EUR_L" in ep.columns:
                    rej_pos_ep = int(ep[ep["BH_reject"] & (ep["delta_DiD_EUR_L"] > 0)].shape[0])
                    rej_neg_ep = int(ep[ep["BH_reject"] & (ep["delta_DiD_EUR_L"] < 0)].shape[0])
                    n_rej_pos += rej_pos_ep
                    n_rej_neg += rej_neg_ep
                    flag = "✓" if rej_pos_ep > 0 else ("←" if rej_neg_ep > 0 else " ")
                else:
                    flag = "✓" if rej_ep > 0 else " "
                    rej_pos_ep = rej_neg_ep = "n/a"
                print(f"     {flag} {ev[:22]} vs {paese:<10}: "
                      f"{rej_ep}/{len(ep)} rigettati  "
                      f"(δ>0: {rej_pos_ep}  δ<0: {rej_neg_ep})  "
                      f"PTA✓={n_pta_ok}/{len(ep)}")
        if n_rej_neg > 0:
            print(f"     ⚠ {n_rej_neg} rigetti con δ<0 → IT ≤ controllo (contro H₁(ii), "
                  "NON contati come evidenza di extraprofitti)")

    family_summary[fam] = {
        "label": label, "n": len(sub), "n_rej": n_rej, "rejected": rejected
    }


# ════════════════════════════════════════════════════════════════════════════
# SOMMARIO H₀ MACRO
# ════════════════════════════════════════════════════════════════════════════
print("\n" + "="*62)
print("SOMMARIO FINALE — H₀ macro")
print("="*62)
print()

print()
print("  MAPPA FAMIGLIE → H₀ DICHIARATE:")
print("  Famiglia A: analisi descrittiva livello strutturale (NON è H₀ dichiarata)")
print("  Famiglia B: H₀(i)  — salto pre→post shock (test causale)")
print("  Famiglia C: H₀(iii) — specificità italiana vs EU (DiD)")
print()
for fam, res in family_summary.items():
    if res["n"] == 0:
        continue
    status = "RIFIUTATA ✓" if res["rejected"] else "non rifiutata"
    if fam == "A":
        role_note = "  ← complementare (non testa H₀ causale)"
    else:
        role_note = ""
    print(f"  [{fam}] {res['label']}: {status}  ({res['n_rej']}/{res['n']}){role_note}")

any_rej = any(r["rejected"] for f, r in family_summary.items()
              if r["n"] > 0 and f in ("B", "C"))  # A è analisi descrittiva, non testa H₀ dichiarata
print()
if any_rej:
    rej_fams = [f for f, r in family_summary.items()
                if r.get("rejected") and f in ("B", "C")]
    print(f"  → H₀ MACRO RIFIUTATA (famiglie: {', '.join(rej_fams)})")
    print("    Esiste evidenza di almeno una componente di margine anomalo.")
    if "A" in family_summary and family_summary["A"].get("rejected"):
        print("    Famiglia A (livello strutturale) avvalora il risultato in modo")
        print("    descrittivo ma non costituisce test causale della H₀ dichiarata.")
else:
    print("  → H₀ MACRO NON RIFIUTATA")
    print("    Nessuna evidenza che gli shock abbiano causato extraprofitti anomali.")
    if "A" in family_summary and family_summary["A"].get("rejected"):
        print("    Nota: Famiglia A mostra margini strutturalmente alti vs 2019,")
        print("    ma questo non implica che gli shock siano la causa (Family A ≠ H₀(i)).")

# Nota n_eff
neff_path = "data/3_neff_report.csv"
if os.path.exists(neff_path):
    df_neff = pd.read_csv(neff_path)
    neff_vals = pd.to_numeric(df_neff["n_eff_post"], errors="coerce").dropna()
    if len(neff_vals):
        n_caut = int((neff_vals < 5).sum())
        n_att  = int(((neff_vals >= 5) & (neff_vals < 10)).sum())
        print(f"""
  ⚠  ROBUSTEZZA n_eff:
     n_eff mediano (serie post-shock): {neff_vals.median():.1f}
     Test con n_eff < 5  (⚠ CAUTELA):      {n_caut}/{len(neff_vals)}
     Test con 5 ≤ n_eff < 10 (attenzione): {n_att}/{len(neff_vals)}

     I rigetti vanno letti come "evidenza consistente ma basata su
     pochi punti indipendenti effettivi", non come N prove separate.
""")

# Nota test con n_post=0 o n_post<4 (τ_MCMC oltre post_end)
if os.path.exists("data/3_AB.csv"):
    df_ab = pd.read_csv("data/3_AB.csv")
    if "n_post" in df_ab.columns:
        zero_post = df_ab[(df_ab["famiglia"] == "B") &
                          (pd.to_numeric(df_ab["n_post"], errors="coerce") == 0)]
        tiny_post = df_ab[(df_ab["famiglia"] == "B") &
                          (pd.to_numeric(df_ab["n_post"], errors="coerce").between(1, 3,
                                                                                    inclusive="both"))]
        if not zero_post.empty:
            print("  🔴 AVVISO CRITICO — τ_MCMC oltre la finestra post_end:")
            for _, r in zero_post.iterrows():
                print(f"     {r.get('evento','?')} | {r.get('carburante','?')}: "
                      f"n_post=0 → test INDEFINITO (δ=nan).")
            print("     Il τ MCMC di questi casi cade oltre post_end: nessun dato")
            print("     post-shock disponibile. Escludere dall'analisi Family B o")
            print("     estendere post_end nelle EVENTS di 3_02_tests.py.")
        if not tiny_post.empty:
            print("  🟡 AVVISO — n_post molto ridotto (1–3 settimane):")
            for _, r in tiny_post.iterrows():
                print(f"     {r.get('evento','?')} | {r.get('carburante','?')}: "
                      f"n_post={int(pd.to_numeric(r.get('n_post',0), errors='coerce'))} → "
                      f"potenza del test quasi nulla.")

# Nota Family B: rigetto singolo = evidenza fragile
if "B" in family_summary:
    fs_b = family_summary["B"]
    if fs_b.get("rejected") and fs_b.get("n_rej", 0) == 1:
        print("\n  🟡 AVVISO ROBUSTEZZA Family B:")
        print("     H₀(i) è rifiutata su 1 solo test (su 8).")
        print("     Con n_eff≈2-3, anche un singolo rigetto MW va interpretato")
        print("     come evidenza 'consistente ma fragile', non come prova forte.")

# Nota PTA (Famiglia C)
if "3_C.csv" in [os.path.basename(p) for p in ["data/3_C.csv"]]:
    if os.path.exists("data/3_C.csv"):
        df_c = pd.read_csv("data/3_C.csv")
        if "PTA_non_rigettata" in df_c.columns:
            n_pta_viol = df_c["PTA_non_rigettata"].astype(str).str.lower().eq("false").sum()
            n_c_total  = len(df_c)
            n_c_rej    = family_summary.get("C", {}).get("n_rej", 0)
            if n_pta_viol > 0:
                print(f"  ⚠  NOTA PTA (Famiglia C):")
                print(f"     {n_pta_viol}/{n_c_total} test DiD hanno PTA violata → δ̂ potenzialmente")
                print(f"     contaminato da trend pre-esistenti. Interpretare come")
                print(f"     evidenza DESCRITTIVA, non causale.")


# ════════════════════════════════════════════════════════════════════════════
# SALVA CSV
# ════════════════════════════════════════════════════════════════════════════
df_all.to_csv("data/3_bh.csv", index=False)
print(f"\n✓ data/3_bh.csv  ({len(df_all)} test totali)")

# Tabella riassuntiva per paper
events   = ["Ucraina (Feb 2022)", "Iran-Israele (Giu 2025)"]
fuels    = ["Benzina", "Diesel"]
table_rows = []

for ev in events:
    for fuel in fuels:
        row = {"Evento": ev, "Carburante": fuel}   # ← inizializzazione obbligatoria
        # Famiglia A — analisi descrittiva (NON testa H₀(i) causale)
        sub_a = df_all[(df_all["famiglia"] == "A") & (df_all["evento"] == ev) &
                       (df_all["carburante"] == fuel) & (df_all["test"] == "HAC_t")]
        if not sub_a.empty:
            r = sub_a.iloc[0]
            row["A_delta_real_vs2019"]  = round(float(r.get("delta", np.nan)), 4)
            row["A_p_HAC"]              = round(float(r["p_value"]), 4)
            row["A_n_eff"]              = r.get("n_eff")
            row["A_BH_reject"]          = bool(r["BH_reject"])
            row["A_nota"]               = "descrittivo: livello vs 2019"
        # Famiglia B — H₀(i): salto pre→post shock
        sub_b = df_all[(df_all["famiglia"] == "B") & (df_all["evento"] == ev) &
                       (df_all["carburante"] == fuel) & (df_all["test"] == "HAC_t")]
        if not sub_b.empty:
            r = sub_b.iloc[0]
            row["B_delta_nom_preppost"] = round(float(r.get("delta", np.nan)), 4)
            row["B_p_HAC"]              = round(float(r["p_value"]), 4)
            row["B_BH_reject"]          = bool(r["BH_reject"])
            row["B_H0i_esito"]          = (
                "H₀(i) rifiutata" if bool(r["BH_reject"]) and float(r.get("delta", 0)) > 0
                else "H₀(i) non rifiutata (δ<0)" if float(r.get("delta", 0)) < 0
                else "H₀(i) non rifiutata"
            )
        # Famiglia C (Germania)
        sub_c = df_all[(df_all["famiglia"] == "C") & (df_all["evento"] == ev) &
                       (df_all["carburante"] == fuel) &
                       (df_all.get("paese_controllo", pd.Series(dtype=str)) == "Germania")]
        if not sub_c.empty:
            r = sub_c.iloc[0]
            row["C_delta_DiD"]   = round(float(r.get("delta_DiD_EUR_L", np.nan)), 4)
            row["C_p_DiD"]       = round(float(r["p_value"]), 4)
            row["C_PTA_ok"]      = r.get("PTA_non_rigettata")
            row["C_BH_reject"]   = bool(r["BH_reject"])
        table_rows.append(row)

df_table = pd.DataFrame(table_rows)
df_table.to_csv("data/3_table_results.csv", index=False)
print("✓ data/3_table_results.csv")
print()
print(df_table.to_string(index=False))


# ════════════════════════════════════════════════════════════════════════════
# GRAFICO: heatmap riassuntiva BH
# ════════════════════════════════════════════════════════════════════════════
def _plot_summary(df_all, family_summary, fpath):
    events = ["Ucraina (Feb 2022)", "Iran-Israele (Giu 2025)"]
    fuels  = ["Benzina", "Diesel"]

    # Colonne famiglie A e B (fisse)
    col_defs = [
        ("A", "HAC_t",       None, "A: HAC_t\nliv.strutturale"),
        ("A", "MannWhitney", None, "A: MW\nliv.strutturale"),
        ("B", "HAC_t",       None, "B: HAC_t\nH₀(i) salto"),
        ("B", "MannWhitney", None, "B: MW\nH₀(i) salto"),
    ]

    # Colonne famiglia C — generate dinamicamente dai paesi in 3_C.csv
    # Etichette brevi: usa i primi 2 caratteri del nome paese (es. "Germania" → "DE")
    PAESE_TO_CODE = {
        "Germania": "DE", "Svezia": "SE", "Paesi Bassi": "NL",
        "Belgio": "BE", "Danimarca": "DK", "Finlandia": "FI", "Austria": "AT",
    }
    if "paese_controllo" in df_all.columns:
        c_countries = (df_all[df_all["famiglia"] == "C"]["paese_controllo"]
                       .dropna().unique().tolist())
        # Ordine stabile: prima Germania poi gli altri in ordine alfabetico
        c_countries = sorted(c_countries,
                             key=lambda x: (0 if x == "Germania" else 1, x))
    else:
        c_countries = []

    for paese in c_countries:
        code = PAESE_TO_CODE.get(paese, paese[:2].upper())
        col_defs.append(("C", None, paese, f"C: DiD\nvs {code}"))

    nrows = len(events) * len(fuels)
    ncols = len(col_defs)
    Z     = np.full((nrows, ncols), np.nan)
    pta_viol = np.zeros((nrows, ncols), dtype=bool)
    row_labels = []

    for ri, (ev, fuel) in enumerate([(e, f) for e in events for f in fuels]):
        row_labels.append(f"{ev.split('(')[0].strip()}\n{fuel}")
        for ci, (fam, test_t, paese, _) in enumerate(col_defs):
            flt = (df_all["famiglia"] == fam) & \
                  (df_all["evento"] == ev) & \
                  (df_all["carburante"] == fuel)
            if test_t:
                flt &= (df_all["test"] == test_t)
            if paese and "paese_controllo" in df_all.columns:
                flt &= (df_all["paese_controllo"] == paese)
            sub = df_all[flt]
            if sub.empty:
                continue
            r = sub.iloc[0]
            Z[ri, ci] = 1.0 if r["BH_reject"] else 0.0
            if fam == "C" and "PTA_non_rigettata" in r.index:
                pta_viol[ri, ci] = (str(r["PTA_non_rigettata"]).lower() == "false")

    fig, ax = plt.subplots(figsize=(11, max(4.5, nrows + 2)))
    cmap = mcolors.ListedColormap(["#a5d6a7", "#ef9a9a"])  # verde=non rigettata, rosso=rigettata
    ax.imshow(np.where(np.isnan(Z), -1, Z), cmap=cmap, vmin=-0.5, vmax=1.5, aspect="auto")

    col_labels = [c[3] for c in col_defs]
    ax.set_xticks(range(ncols))
    ax.set_xticklabels(col_labels, fontsize=9, ha="center")
    ax.set_yticks(range(nrows))
    ax.set_yticklabels(row_labels, fontsize=9)

    for ri in range(nrows):
        for ci in range(ncols):
            v = Z[ri, ci]
            if np.isnan(v):
                ax.text(ci, ri, "n/a", ha="center", va="center", fontsize=8, color="#9e9e9e")
            else:
                txt   = "✗ H₀" if v == 1 else "✓ H₀"
                color = "#b71c1c" if v == 1 else "#1b5e20"
                ax.text(ci, ri, txt, ha="center", va="center",
                        fontsize=9, fontweight="bold", color=color)
                # Marcatore PTA violata
                if pta_viol[ri, ci]:
                    ax.text(ci + 0.35, ri - 0.35, "PTA✗",
                            ha="right", va="top", fontsize=6, color="#e65100")

    # Separatori tra famiglie (A|B a 1.5, B|C a 3.5)
    for xsep in [1.5, 3.5]:
        ax.axvline(xsep, color="#37474f", lw=1.8)

    # Intestazioni famiglie — posizione C calcolata dinamicamente
    n_c_cols = len(c_countries)
    c_mid    = 4 + (n_c_cols - 1) / 2 if n_c_cols > 0 else 4
    for xi, lbl in [(0.5,   "Famiglia A\n(livello strutturale)"),
                    (2.5,   "Famiglia B\nH₀(i) salto"),
                    (c_mid, "Famiglia C\nH₀(iii) DiD")]:
        ax.text(xi, -0.75, lbl, ha="center", va="bottom", fontsize=9,
                fontweight="bold", color="#37474f",
                transform=ax.get_xaxis_transform())

    ax.set_title("Risultati correzione BH (FDR 5%) per famiglia di test\n"
                 "verde = H₀ non rifiutata  |  rosso = H₀ rifiutata\n"
                 "Famiglia A = analisi descrittiva (complementare) | B = H₀(i) | C = H₀(iii)",
                 fontsize=10, fontweight="bold", pad=28)
    plt.tight_layout()
    fig.savefig(fpath, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"✓ {fpath}")


try:
    _plot_summary(df_all, family_summary, "plots/3_04_summary.png")
except Exception as exc:
    print(f"   Plot summary: {exc}")

print("Script 3_04 completato.")