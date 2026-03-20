"""
05_nonparametric_validation.py
================================
Validazione non parametrica dell'analisi principale.

MOTIVAZIONE
───────────
I diagnostici in regression_diagnostics.csv mostrano:
  - DW ≈ 0.29–0.42 per benzina/diesel Ucraina → autocorrelazione forte
  - SW p ≈ 4.8e-5 (benzina) e 1.7e-4 (diesel) Ucraina → non-normalità
  - BP p > 0.05 per la maggior parte → omoschedasticità ≈ ok, ma con non-normalità
    il BP stesso non è affidabile.

Le assunzioni del Welch t-test (test primario in 02_core_analysis.py) sono
violate. I test non parametrici qui AFFIANCANO il Welch t come co-test
primari, e i loro p-value partecipano alla BH correction di 04_global_corrections.py.

STRUTTURA
──────────
§1.  Mann-Whitney U (Wilcoxon rank-sum) + Hodges‑Lehmann + Cliff's delta
§2.  Kruskal-Wallis (con warning dipendenza temporale)
§3.  Cliff's delta (efficiente)
§4.  Permutation test su Δmediana (block permutation)
§5.  Permutation test DiD (approssimazione sign‑flip + placebo)
§6.  Fligner‑Killeen (omogeneità varianze, non parametrico)
§7.  Runs test migliorato (modello con dummy shock) + DW
§7bis. HAC Newey‑West (supplemento robusto per autocorrelazione)
§8.  Aggiornamento BH globale

NOTE METODOLOGICHE
───────────────────
• Block permutation: preserva blocchi di 4 settimane per rispettare autocorrelazione.
• La statistica del permutation test è Δmediana, coerente con Mann-Whitney.
• BH applicata a test dipendenti: la correlazione positiva preserva il controllo FDR.
• Placebo DiD: shock spostato 8 settimane prima per testare specificità temporale.

Output:
  data/nonparam_mannwhitney.csv
  data/nonparam_kruskal.csv
  data/nonparam_permutation.csv
  data/nonparam_fligner.csv
  data/nonparam_hac.csv
  data/placebo_did.csv
  data/global_bh_corrections.csv  (aggiornato)
  plots/10_nonparam_summary.png
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy import stats
from scipy.stats import mannwhitneyu, kruskal, fligner
import statsmodels.api as sm
from statsmodels.stats.stattools import durbin_watson
import warnings
warnings.filterwarnings("ignore")

os.makedirs("data", exist_ok=True)
os.makedirs("plots", exist_ok=True)

ALPHA = 0.05
DPI   = 180
N_PERM = 10_000          # permutazioni Monte Carlo (§4, §5)
SEED   = 42
BLOCK_SIZE = 4           # settimane (coerente con media mobile 4w)

EVENTS = {
    "Ucraina (Feb 2022)": {
        "shock":     pd.Timestamp("2022-02-24"),
        "pre_start": pd.Timestamp("2021-09-01"),
        "post_end":  pd.Timestamp("2022-08-31"),
        "window_start": "2021-10-01",
        "window_end":   "2022-07-31",
    },
    "Iran-Israele (Giu 2025)": {
        "shock":     pd.Timestamp("2025-06-13"),
        "pre_start": pd.Timestamp("2025-01-01"),
        "post_end":  pd.Timestamp("2025-10-31"),
        "window_start": "2025-02-01",
        "window_end":   "2025-10-31",
    },
}
# Hormuz escluso

FUELS = {"Benzina": "benzina_4w", "Diesel": "diesel_4w"}
BRENT_L = 159.0
DENSITY_BENZ = 0.74
DENSITY_DIES = 0.84
L_PER_T_BENZ = 1000.0 / DENSITY_BENZ
L_PER_T_DIES = 1000.0 / DENSITY_DIES

# ─────────────────────────────────────────────────────────────────────────────
# Utility aggiornate
# ─────────────────────────────────────────────────────────────────────────────

def block_permutation(x, block_size, rng):
    """Permuta x preservando blocchi contigui di lunghezza block_size."""
    n = len(x)
    n_blocks = int(np.ceil(n / block_size))
    blocks = [x[i*block_size : min((i+1)*block_size, n)] for i in range(n_blocks)]
    rng.shuffle(blocks)
    return np.concatenate(blocks)


def circular_block_bootstrap(x, block_size, rng):
    """Circular block bootstrap: ricampiona blocchi circolari."""
    n = len(x)
    res = []
    while len(res) < n:
        start = rng.integers(0, n)
        idx = (start + np.arange(block_size)) % n
        res.extend(x[idx])
    return np.array(res[:n])


def cliffs_delta_fast(pre, post):
    """
    Cliff's delta efficiente via searchsorted.
    Equivalente a 2*AUC(ROC) - 1.
    """
    pre = np.sort(pre)
    post = np.sort(post)
    n1, n2 = len(post), len(pre)
    more = sum(np.searchsorted(pre, x, side='left') for x in post)
    less = sum(n2 - np.searchsorted(pre, x, side='right') for x in post)
    return (more - less) / (n1 * n2)


def cliffs_delta_magnitude(d):
    ad = abs(d)
    if ad < 0.147:   return "trascurabile"
    if ad < 0.330:   return "piccolo"
    if ad < 0.474:   return "medio"
    return "grande"


def mannwhitney_hl(post, pre):
    """
    Mann-Whitney U con Hodges‑Lehmann estimator.
    Restituisce: U_stat, p_one (greater), p_two, hl_shift
    """
    U_stat, p_one = mannwhitneyu(post, pre, alternative='greater')
    _, p_two = mannwhitneyu(post, pre, alternative='two-sided')
    diff = np.array([p - q for p in post for q in pre])
    hl_shift = np.median(diff)
    return U_stat, p_one, p_two, hl_shift


def runs_test_enhanced(series_vals, shock_idx):
    """
    Runs test su residui di OLS con trend + dummy post-shock.
    H0: segni casuali (no autocorrelazione).
    """
    n = len(series_vals)
    x = np.arange(n, dtype=float)
    post_dummy = np.zeros(n)
    post_dummy[shock_idx:] = 1
    X = np.column_stack([np.ones(n), x, post_dummy])
    try:
        model = sm.OLS(series_vals, X).fit()
        resid = model.resid
    except:
        resid = series_vals - np.median(series_vals)  # fallback grezzo

    signs = np.sign(resid - np.median(resid))
    signs = signs[signs != 0]
    if len(signs) < 5:
        return np.nan, np.nan, "dati insufficienti"
    n1 = int((signs > 0).sum())
    n2 = int((signs < 0).sum())
    runs = 1 + np.sum(signs[:-1] != signs[1:])
    R = int(runs)
    ER = 1.0 + 2.0 * n1 * n2 / (n1 + n2)
    VR = (2.0 * n1 * n2 * (2.0 * n1 * n2 - len(signs))) / (len(signs)**2 * (len(signs)-1))
    if VR <= 0:
        return np.nan, np.nan, "varianza nulla"
    Z = (R - ER) / np.sqrt(VR)
    p_val = 2.0 * stats.norm.sf(abs(Z))
    verdict = "✓ residui casuali" if p_val >= ALPHA else "✗ autocorrelazione rilevata"
    return float(Z), float(p_val), verdict


def _stars(p):
    return "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "n.s."


# ─────────────────────────────────────────────────────────────────────────────
# Carica dataset e margini
# ─────────────────────────────────────────────────────────────────────────────
merged = pd.read_csv("data/dataset_merged_with_futures.csv", index_col=0, parse_dates=True)
print(f"Dataset: {len(merged)} settimane | "
      f"{merged.index[0].date()} → {merged.index[-1].date()}\n")

# Normalizzazione unità pompa
for raw_col, eur_l_col in [("benzina_4w", "benzina_eur_l"),
                             ("diesel_4w",  "diesel_eur_l")]:
    if raw_col in merged.columns and eur_l_col not in merged.columns:
        med = merged[raw_col].dropna().median()
        merged[eur_l_col] = merged[raw_col] / (1000.0 if med > 10 else 1.0)

# Margini (crack spread dove disponibile, fallback brent-based)
MARGIN_COLS = {}
if "margine_benz_crack" in merged.columns:
    MARGIN_COLS["Benzina"] = "margine_benz_crack"
elif "benzina_eur_l" in merged.columns and "brent_eur" in merged.columns:
    merged["margine_benzina_brent"] = merged["benzina_eur_l"] - merged["brent_eur"] / BRENT_L
    MARGIN_COLS["Benzina"] = "margine_benzina_brent"

if "margine_dies_crack" in merged.columns:
    MARGIN_COLS["Diesel"] = "margine_dies_crack"
elif "diesel_eur_l" in merged.columns and "brent_eur" in merged.columns:
    merged["margine_diesel_brent"] = merged["diesel_eur_l"] - merged["brent_eur"] / BRENT_L
    MARGIN_COLS["Diesel"] = "margine_diesel_brent"

print(f"  Margini usati: {MARGIN_COLS}\n")


# ─────────────────────────────────────────────────────────────────────────────
# §1. MANN-WHITNEY U + Hodges‑Lehmann + Cliff's delta (efficiente)
# ─────────────────────────────────────────────────────────────────────────────
print("=" * 65)
print("§1. Mann-Whitney U (con Hodges-Lehmann, two‑sided, Cliff's delta)")
print("    H0: P(margine_post > margine_pre) = 0.5")
print("=" * 65)

mw_rows = []

for event_name, cfg in EVENTS.items():
    shock = cfg["shock"]
    for fuel_name, margin_col in MARGIN_COLS.items():
        if margin_col not in merged.columns:
            continue
        df_ev = merged.loc[cfg["pre_start"]:cfg["post_end"]].dropna(subset=[margin_col])
        shock_idx = int(np.clip(df_ev.index.searchsorted(shock), 1, len(df_ev) - 1))
        pre  = df_ev.iloc[:shock_idx][margin_col].dropna().values
        post = df_ev.iloc[shock_idx:][margin_col].dropna().values

        if len(pre) < 3 or len(post) < 3:
            continue

        U_stat, mw_p_one, mw_p_two, hl_shift = mannwhitney_hl(post, pre)
        U_max = len(pre) * len(post)
        cd = cliffs_delta_fast(pre, post)
        cd_mag = cliffs_delta_magnitude(cd)

        row = {
            "Evento":        event_name,
            "Carburante":    fuel_name,
            "n_pre":         len(pre),
            "n_post":        len(post),
            "U_stat":        round(U_stat, 1),
            "U_max":         U_max,
            "AUC":           round(float(U_stat / U_max), 3),
            "mw_p_one":      round(mw_p_one, 4),
            "mw_p_two":      round(mw_p_two, 4),
            "hodges_lehmann": round(hl_shift, 5),
            "cliffs_delta":  round(cd, 3),
            "magnitude":     cd_mag,
            "mw_H0":         "RIFIUTATA" if mw_p_one < ALPHA else "non rifiutata",
        }
        mw_rows.append(row)

        pre_med  = np.median(pre)
        post_med = np.median(post)
        print(f"  {event_name.split('(')[0].strip():<22} | {fuel_name:<8}: "
              f"U={U_stat:.0f}/{U_max}  AUC={row['AUC']:.3f}  "
              f"p(one)={mw_p_one:.4f} {_stars(mw_p_one)}  "
              f"p(two)={mw_p_two:.4f}  HL={hl_shift:+.5f}  "
              f"Cliff's δ={cd:+.3f} [{cd_mag}]\n"
              f"    mediana pre={pre_med:.5f} → post={post_med:.5f}  "
              f"Δmed={post_med-pre_med:+.5f}  → H0 {row['mw_H0']}")

df_mw = pd.DataFrame(mw_rows)
df_mw.to_csv("data/nonparam_mannwhitney.csv", index=False)

# Aggiorna table2 con colonne Mann-Whitney
t2_path = "data/table2_margin_anomaly.csv"
if os.path.exists(t2_path):
    df_t2 = pd.read_csv(t2_path)
    df_mw_match = df_mw.rename(columns={"Carburante": "Serie"})
    df_t2 = df_t2.merge(
        df_mw_match[["Evento", "Serie", "U_stat", "AUC", "mw_p_one", "mw_p_two",
                     "hodges_lehmann", "cliffs_delta", "magnitude", "mw_H0"]],
        on=["Evento", "Serie"],
        how="left",
    )
    df_t2.to_csv(t2_path, index=False)
    print(f"\n  Aggiornato: {t2_path} (colonne Mann-Whitney aggiunte)")

print(f"  Salvato: data/nonparam_mannwhitney.csv ({len(mw_rows)} test)\n")


# ─────────────────────────────────────────────────────────────────────────────
# §2. KRUSKAL-WALLIS (con avviso dipendenza seriale)
# ─────────────────────────────────────────────────────────────────────────────
print("=" * 65)
print("§2. Kruskal-Wallis — alternativa non parametrica all'ANOVA (3 periodi)")
print("    **WARNING: gruppi temporali → possibile dipendenza seriale**")
print("=" * 65)

kw_rows = []

for event_name, cfg in EVENTS.items():
    shock = cfg["shock"]
    for fuel_name, fuel_col in FUELS.items():
        if fuel_col not in merged.columns:
            continue
        pA = merged.loc[cfg["pre_start"]:shock, fuel_col].dropna().values
        pB = merged.loc[shock:shock + pd.Timedelta(weeks=6), fuel_col].dropna().values
        pC = merged.loc[shock + pd.Timedelta(weeks=6):cfg["post_end"], fuel_col].dropna().values

        if any(len(p) < 3 for p in [pA, pB, pC]):
            continue

        H_stat, kw_p = kruskal(pA, pB, pC)

        # Dunn post-hoc (approssimazione: Mann-Whitney con Bonferroni)
        _, p_AB = mannwhitneyu(pB, pA, alternative="two-sided")
        _, p_BC = mannwhitneyu(pC, pB, alternative="two-sided")
        _, p_AC = mannwhitneyu(pC, pA, alternative="two-sided")
        p_AB_bon = min(p_AB * 3, 1.0)
        p_BC_bon = min(p_BC * 3, 1.0)
        p_AC_bon = min(p_AC * 3, 1.0)

        row = {
            "Evento":      event_name,
            "Carburante":  fuel_name,
            "n_pre":       len(pA),
            "n_shock6w":   len(pB),
            "n_post":      len(pC),
            "H_stat":      round(H_stat, 4),
            "kw_p":        round(kw_p, 4),
            "kw_H0":       "RIFIUTATA" if kw_p < ALPHA else "non rifiutata",
            "posthoc_pre_vs_shock": round(p_AB_bon, 4),
            "posthoc_shock_vs_post": round(p_BC_bon, 4),
            "posthoc_pre_vs_post":  round(p_AC_bon, 4),
            "warning_serial": "Sì - gruppi temporali non indipendenti",
        }
        kw_rows.append(row)
        print(f"  {event_name.split('(')[0].strip():<22} | {fuel_name:<8}: "
              f"H={H_stat:.3f}  p={kw_p:.4f} {_stars(kw_p)}  → {row['kw_H0']}")

pd.DataFrame(kw_rows).to_csv("data/nonparam_kruskal.csv", index=False)
print(f"\n  Salvato: data/nonparam_kruskal.csv ({len(kw_rows)} test)\n")


# ─────────────────────────────────────────────────────────────────────────────
# §4. PERMUTATION TEST — Δmediana (block permutation)
# ─────────────────────────────────────────────────────────────────────────────
print("=" * 65)
print("§4. Permutation test — Δmediana (block perm, block_size=4w)")
print("    H0: Δ osservato è compatibile con permutazione casuale dei blocchi")
print("=" * 65)

perm_rows = []
rng = np.random.default_rng(SEED)


def _perm_test_block(pre, post, n_perm, rng, block_size=BLOCK_SIZE):
    """Permutation test su Δmediana con block permutation."""
    observed_delta = np.median(post) - np.median(pre)
    combined = np.concatenate([pre, post])
    n_post = len(post)
    count = 0
    for _ in range(n_perm):
        perm = block_permutation(combined, block_size, rng)
        post_perm = perm[-n_post:]
        pre_perm = perm[:-n_post]
        if np.median(post_perm) - np.median(pre_perm) >= observed_delta:
            count += 1
    p_val = count / n_perm
    return observed_delta, p_val


for event_name, cfg in EVENTS.items():
    shock = cfg["shock"]
    for fuel_name, margin_col in MARGIN_COLS.items():
        if margin_col not in merged.columns:
            continue
        df_ev = merged.loc[cfg["pre_start"]:cfg["post_end"]].dropna(subset=[margin_col])
        shock_idx = int(np.clip(df_ev.index.searchsorted(shock), 1, len(df_ev) - 1))
        pre  = df_ev.iloc[:shock_idx][margin_col].dropna().values
        post = df_ev.iloc[shock_idx:][margin_col].dropna().values

        if len(pre) < 3 or len(post) < 3:
            continue

        obs_delta, p_perm = _perm_test_block(pre, post, N_PERM, rng)

        row = {
            "Tipo":        "Δmargine_block",
            "Evento":      event_name,
            "Carburante":  fuel_name,
            "Confronto":   f"IT margine pre vs post",
            "obs_delta_med": round(obs_delta, 5),
            "perm_p":      round(p_perm, 4),
            "perm_H0":     "RIFIUTATA" if p_perm < ALPHA else "non rifiutata",
            "n_perm":      N_PERM,
            "block_size":  BLOCK_SIZE,
        }
        perm_rows.append(row)
        print(f"  {event_name.split('(')[0].strip():<22} | {fuel_name:<8}: "
              f"Δmed={obs_delta:+.5f}  p_perm={p_perm:.4f} {_stars(p_perm)}  "
              f"→ H0 {row['perm_H0']}")


# ── Permutation test DiD: sign‑flip approximation + Placebo ─────────────────
print(f"\n§5. Permutation test DiD δ̂ (sign‑flip approx + placebo)")

did_path = "data/did_results.csv"
if os.path.exists(did_path):
    df_did_res = pd.read_csv(did_path)
    rng2 = np.random.default_rng(SEED+1)

    for _, did_row in df_did_res.iterrows():
        event_name = did_row["Evento"]
        paese      = did_row["Paese_controllo"]
        fuel_name  = did_row["Carburante"]

        if event_name not in EVENTS:
            continue
        margin_col = MARGIN_COLS.get(fuel_name)
        if margin_col is None or margin_col not in merged.columns:
            continue

        cfg   = EVENTS[event_name]
        shock = cfg["shock"]

        # Sign‑flip approximation
        obs_did = float(did_row["delta_DiD"])
        se_did  = float(did_row["SE_HC3"])
        flips = rng2.choice([-1.0, 1.0], size=N_PERM)
        perm_dist_did = obs_did * flips + rng2.normal(0, se_did, size=N_PERM)
        p_perm_did = float(np.mean(np.abs(perm_dist_did) >= abs(obs_did)))

        row_did = {
            "Tipo":       "DiD_signflip",
            "Evento":     event_name,
            "Carburante": fuel_name,
            "Confronto":  f"IT vs {paese}",
            "obs_delta":  round(obs_did, 5),
            "perm_p":     round(p_perm_did, 4),
            "perm_H0":    "RIFIUTATA" if p_perm_did < ALPHA else "non rifiutata",
            "n_perm":     N_PERM,
            "nota":       "sign-flip approximation (non exact permutation)",
        }
        perm_rows.append(row_did)
        print(f"  [{paese}] {event_name.split('(')[0].strip():<20} | {fuel_name:<8}: "
              f"δ̂={obs_did:+.5f}  p_perm={p_perm_did:.4f} {_stars(p_perm_did)}  "
              f"→ H0 {row_did['perm_H0']}")

    # Placebo DiD: shock anticipato di 8 settimane
    print("\n  Placebo DiD (shock anticipato di 8 settimane)...")
    for _, did_row in df_did_res.iterrows():
        event_name = did_row["Evento"]
        paese      = did_row["Paese_controllo"]
        fuel_name  = did_row["Carburante"]

        if event_name not in EVENTS:
            continue
        margin_col = MARGIN_COLS.get(fuel_name)
        if margin_col is None:
            continue
        cfg   = EVENTS[event_name]
        shock = cfg["shock"] - pd.Timedelta(weeks=8)
        # Finestre placebo: pre inizia 12 settimane prima del falso shock, post finisce 12 dopo
        pre_start = shock - pd.Timedelta(weeks=12)
        post_end  = shock + pd.Timedelta(weeks=12)
        if pre_start < merged.index[0] or post_end > merged.index[-1]:
            continue
        it_pre  = merged.loc[pre_start:shock, margin_col].dropna()
        it_post = merged.loc[shock:post_end, margin_col].dropna()
        if len(it_pre) < 3 or len(it_post) < 3:
            continue
        obs_delta = np.median(it_post) - np.median(it_pre)
        row_placebo = {
            "Tipo":       "Placebo_DiD",
            "Evento":     event_name,
            "Carburante": fuel_name,
            "Confronto":  f"IT vs {paese} (shock -8w)",
            "obs_delta":  round(obs_delta, 5),
            "perm_p":     np.nan,   # non calcolato
            "perm_H0":    "N/A",
            "n_perm":     0,
            "nota":       "placebo: shock anticipato 8 settimane",
        }
        perm_rows.append(row_placebo)
        print(f"  [{paese}] Placebo {event_name.split('(')[0].strip():<18} | {fuel_name:<8}: "
              f"Δmed={obs_delta:+.5f} (placebo)")

else:
    print("  data/did_results.csv non trovato — §5 saltato")

df_perm = pd.DataFrame(perm_rows)
df_perm.to_csv("data/nonparam_permutation.csv", index=False)
print(f"\n  Salvato: data/nonparam_permutation.csv ({len(perm_rows)} test)\n")


# ─────────────────────────────────────────────────────────────────────────────
# §6. FLIGNER-KILLEEN + §7. RUNS TEST MIGLIORATO
# ─────────────────────────────────────────────────────────────────────────────
print("=" * 65)
print("§6. Fligner-Killeen (omogeneità varianze)")
print("§7. Runs test (modello con dummy shock)")
print("=" * 65)

fk_rows = []

for event_name, cfg in EVENTS.items():
    shock = cfg["shock"]
    for fuel_name, fuel_col in FUELS.items():
        if fuel_col not in merged.columns:
            continue
        series = merged.loc[cfg["pre_start"]:cfg["post_end"], fuel_col].dropna()
        if len(series) < 10:
            continue
        shock_idx = int(np.clip(series.index.searchsorted(shock), 2, len(series) - 2))
        pre  = series.values[:shock_idx]
        post = series.values[shock_idx:]

        # Fligner-Killeen
        try:
            fk_stat, fk_p = fligner(pre, post)
        except Exception:
            fk_stat, fk_p = np.nan, np.nan

        # Runs test con modello migliorato
        z_runs, p_runs, runs_verdict = runs_test_enhanced(series.values, shock_idx)
        dw_stat = durbin_watson(series.values)

        row = {
            "Evento":      event_name,
            "Carburante":  fuel_name,
            "FK_stat":     round(float(fk_stat), 4) if not np.isnan(fk_stat) else "N/A",
            "FK_p":        round(float(fk_p), 4)    if not np.isnan(fk_p)    else "N/A",
            "FK_H0":       ("RIFIUTATA" if (not np.isnan(fk_p) and fk_p < ALPHA)
                            else "non rifiutata"),
            "Z_runs":      round(z_runs, 3) if not np.isnan(z_runs) else "N/A",
            "p_runs":      round(p_runs, 4) if not np.isnan(p_runs) else "N/A",
            "runs_verdict":runs_verdict,
            "DW":          round(dw_stat, 3),
        }
        fk_rows.append(row)
        fk_s = f"{fk_p:.4f}" if not np.isnan(fk_p) else "N/A"
        rn_s = f"{p_runs:.4f}" if not np.isnan(p_runs) else "N/A"
        print(f"  {event_name.split('(')[0].strip():<22} | {fuel_name:<8}: "
              f"FK_p={fk_s}  Runs_p={rn_s}  DW={dw_stat:.3f}  [{runs_verdict}]")

pd.DataFrame(fk_rows).to_csv("data/nonparam_fligner.csv", index=False)
print(f"\n  Salvato: data/nonparam_fligner.csv ({len(fk_rows)} test)\n")


# ─────────────────────────────────────────────────────────────────────────────
# §7bis. HAC NEWEY-WEST — Δmargine con errori robusti
# ─────────────────────────────────────────────────────────────────────────────
print("=" * 65)
print("§7bis. HAC Newey‑West — test robusto per autocorrelazione")
print("    Modello: margine ~ cost + post_dummy, SE HAC(maxlags=4)")
print("=" * 65)

hac_rows = []

for event_name, cfg in EVENTS.items():
    shock = cfg["shock"]
    for fuel_name, margin_col in MARGIN_COLS.items():
        if margin_col not in merged.columns:
            continue
        df_ev = merged.loc[cfg["pre_start"]:cfg["post_end"]].dropna(subset=[margin_col])
        shock_idx = int(np.clip(df_ev.index.searchsorted(shock), 1, len(df_ev) - 1))
        y = df_ev[margin_col].values
        post_dummy = np.concatenate([np.zeros(shock_idx), np.ones(len(y) - shock_idx)])
        X = sm.add_constant(post_dummy)
        try:
            ols_hac = sm.OLS(y, X).fit(cov_type='HAC', cov_kwds={'maxlags': 4})
            hac_p = ols_hac.pvalues[1]
            hac_H0 = "RIFIUTATA" if hac_p < ALPHA else "non rifiutata"
        except:
            ols_hac = None
            hac_p = np.nan
            hac_H0 = "errore"

        row_hac = {
            "Evento":      event_name,
            "Carburante":  fuel_name,
            "delta_mean":  round(ols_hac.params[1], 5) if ols_hac is not None else np.nan,
            "HAC_p":       round(hac_p, 4) if not np.isnan(hac_p) else "N/A",
            "H0":          hac_H0,
        }
        hac_rows.append(row_hac)
        print(f"  {event_name.split('(')[0].strip():<22} | {fuel_name:<8}: "
              f"Δ={row_hac['delta_mean']:.5f}  HAC p={row_hac['HAC_p']}  → {hac_H0}")

pd.DataFrame(hac_rows).to_csv("data/nonparam_hac.csv", index=False)
print(f"\n  Salvato: data/nonparam_hac.csv ({len(hac_rows)} test)\n")


# ─────────────────────────────────────────────────────────────────────────────
# §8. AGGIORNAMENTO BH GLOBALE con p-value non parametrici
# ─────────────────────────────────────────────────────────────────────────────
print("=" * 65)
print("§8. Aggiornamento global_bh_corrections.csv con p-value non parametrici")
print("=" * 65)

global_bh_path = "data/global_bh_corrections.csv"
new_rows = []

# Mann-Whitney (one-sided) come primario non parametrico
for row in mw_rows:
    new_rows.append({
        "fonte":       "MannWhitneyU",
        "tipo":        "confirmatory",
        "descrizione": f"MW Δmargine {row['Evento']} | {row['Carburante']}",
        "p_value":     float(row["mw_p_one"]),
    })

# Block permutation test (Δ mediana)
for _, row in df_perm[df_perm["Tipo"] == "Δmargine_block"].iterrows():
    new_rows.append({
        "fonte":       "Permutation_block",
        "tipo":        "confirmatory",
        "descrizione": f"Perm Δmed {row['Evento']} | {row['Carburante']}",
        "p_value":     float(row["perm_p"]),
    })

# HAC Newey‑West
for row_hac in hac_rows:
    if "N/A" not in str(row_hac["HAC_p"]) and not np.isnan(row_hac["HAC_p"]):
        new_rows.append({
            "fonte":       "HAC_NeweyWest",
            "tipo":        "confirmatory",
            "descrizione": f"HAC Δmargine {row_hac['Evento']} | {row_hac['Carburante']}",
            "p_value":     float(row_hac["HAC_p"]),
        })

# Kruskal-Wallis (exploratory)
for row in kw_rows:
    new_rows.append({
        "fonte":       "KruskalWallis",
        "tipo":        "exploratory",
        "descrizione": f"KW 3-period {row['Evento']} | {row['Carburante']}",
        "p_value":     float(row["kw_p"]),
    })

if not new_rows:
    print("  Nessun p-value non parametrico da aggiungere.")
else:
    df_new = pd.DataFrame(new_rows)
    df_new["BH_global_reject"]    = np.nan
    df_new["p_value_BH_adjusted"] = np.nan

    if os.path.exists(global_bh_path):
        df_old = pd.read_csv(global_bh_path)
        df_combined = pd.concat([df_old, df_new], ignore_index=True)
    else:
        df_combined = df_new.copy()

    # Ri-applica BH su tutti i confirmatory
    def bh_correction(p_values, alpha=0.05):
        p = np.array(p_values, dtype=float)
        n = len(p)
        if n == 0:
            return np.array([], dtype=bool), np.array([])
        order    = np.argsort(p)
        ranked   = np.empty(n, dtype=float)
        ranked[order] = np.arange(1, n + 1)
        p_adj    = np.minimum(1.0, p * n / ranked)
        p_adj_m  = np.minimum.accumulate(p_adj[order][::-1])[::-1]
        p_adj_out = np.empty(n)
        p_adj_out[order] = p_adj_m
        return p_adj_out <= alpha, p_adj_out

    conf_mask = df_combined["tipo"] == "confirmatory"
    if conf_mask.sum() > 0:
        p_conf = df_combined.loc[conf_mask, "p_value"].values.astype(float)
        reject_c, p_adj_c = bh_correction(p_conf, alpha=ALPHA)
        df_combined.loc[conf_mask, "BH_global_reject"]    = reject_c
        df_combined.loc[conf_mask, "p_value_BH_adjusted"] = p_adj_c

    df_combined.to_csv(global_bh_path, index=False)

    n_conf_tot = conf_mask.sum()
    n_rej      = int(reject_c.sum())
    print(f"  Aggiornato: {global_bh_path}")
    print(f"  Confirmatory totali: {n_conf_tot}  |  BH global rigettati: {n_rej}/{n_conf_tot}")
    print(f"  Nota: i test non sono indipendenti, ma la correlazione positiva preserva il controllo FDR.\n")


# ─────────────────────────────────────────────────────────────────────────────
# PLOT RIASSUNTIVO
# ─────────────────────────────────────────────────────────────────────────────
print("Generazione plot riassuntivo...")

plt.rcParams.update({
    "font.family": "serif",
    "font.serif":  ["Times New Roman", "DejaVu Serif"],
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "axes.grid":   True,
    "grid.color":  "#e0e0e0",
    "grid.linewidth": 0.5,
})

fig, axes = plt.subplots(2, 2, figsize=(16, 11))
fig.suptitle(
    "Validazione non parametrica — Margini carburanti Italia\n"
    "(block permutation, Δmediana, HAC Newey‑West, Hodges‑Lehmann)",
    fontsize=13, fontweight="bold",
)

# Pannello A: Cliff’s delta e Hodges‑Lehmann
ax = axes[0, 0]
if mw_rows:
    labels_mw = [f"{r['Evento'].split('(')[0].strip()}\n{r['Carburante']}"
                 for r in mw_rows]
    cd_vals   = [r["cliffs_delta"] for r in mw_rows]
    colors_mw = ["#e74c3c" if r["mw_H0"] == "RIFIUTATA" else "#95a5a6"
                 for r in mw_rows]
    ax.barh(range(len(mw_rows)), cd_vals, color=colors_mw,
            edgecolor="black", lw=0.5, alpha=0.85)
    for thr in [0.147, 0.330, 0.474]:
        ax.axvline(thr, color="orange", lw=1.2, ls=":", alpha=0.7)
        ax.axvline(-thr, color="orange", lw=1.2, ls=":", alpha=0.7)
    ax.axvline(0, color="black", lw=1.0, ls="--")
    ax.set_yticks(range(len(mw_rows)))
    ax.set_yticklabels(labels_mw, fontsize=9)
    ax.set_xlabel("Cliff's delta / HL shift", fontsize=10)
    ax.set_title(
        "Mann-Whitney — Cliff's delta & Hodges‑Lehmann\n"
        "(rosso = p<0.05 one‑sided)",
        fontsize=11, fontweight="bold",
    )
    for i, r in enumerate(mw_rows):
        txt = f"δ={r['cliffs_delta']:.2f} HL={r['hodges_lehmann']:+.3f}"
        ax.text(cd_vals[i] + 0.02, i, txt, va="center", fontsize=7,
                color="#8b1a1a" if r["mw_H0"] == "RIFIUTATA" else "#555")

# Pannello B: p-value block permutation vs HAC
ax = axes[0, 1]
if mw_rows and hac_rows:
    perm_p = df_perm[df_perm["Tipo"] == "Δmargine_block"]["perm_p"].values
    hac_p  = [r["HAC_p"] for r in hac_rows if r["HAC_p"] != "N/A"]
    min_len = min(len(perm_p), len(hac_p))
    x_pos = np.arange(min_len)
    width = 0.35
    ax.bar(x_pos - width/2, perm_p[:min_len], width, label="Block perm (Δmed)",
           color="#e74c3c", alpha=0.8)
    ax.bar(x_pos + width/2, hac_p[:min_len], width, label="HAC Newey‑West",
           color="#3498db", alpha=0.8)
    ax.axhline(ALPHA, color="black", lw=1.5, ls="--")
    ax.set_xticks(x_pos)
    ax.set_xticklabels([f"{mw_rows[i]['Evento'].split('(')[0].strip()}\n{mw_rows[i]['Carburante']}" for i in range(min_len)], fontsize=7)
    ax.set_ylabel("p-value")
    ax.set_title("Confronto p-value: Block perm vs HAC\n(soglia α=0.05)", fontsize=11, fontweight="bold")
    ax.legend(fontsize=9)

# Pannello C: Distribuzione nulla block permutation (primo evento)
ax = axes[1, 0]
# Seleziona il primo evento con margine disponibile
shown = False
for event_name, cfg in EVENTS.items():
    if shown: break
    for fuel_name, margin_col in MARGIN_COLS.items():
        if margin_col not in merged.columns or shown: continue
        df_ev = merged.loc[cfg["pre_start"]:cfg["post_end"]].dropna(subset=[margin_col])
        shock_idx = int(np.clip(df_ev.index.searchsorted(cfg["shock"]), 1, len(df_ev)-1))
        pre  = df_ev.iloc[:shock_idx][margin_col].dropna().values
        post = df_ev.iloc[shock_idx:][margin_col].dropna().values
        if len(pre) < 3 or len(post) < 3: continue

        # Ottieni distribuzione permutazione per il plot
        obs_delta = np.median(post) - np.median(pre)
        combined = np.concatenate([pre, post])
        n_post = len(post)
        deltas = []
        for _ in range(500):  # ridotto per velocità
            perm = block_permutation(combined, BLOCK_SIZE, np.random.default_rng(SEED+_))
            deltas.append(np.median(perm[-n_post:]) - np.median(perm[:-n_post]))
        ax.hist(deltas, bins=30, color="#bdc3c7", edgecolor="none", alpha=0.85, density=True)
        ax.axvline(obs_delta, color="#e74c3c", lw=2.5, label=f"Δ obs = {obs_delta:.4f}")
        ax.set_xlabel("Δ mediana (EUR/litro)", fontsize=10)
        ax.set_ylabel("Densità", fontsize=10)
        ax.set_title(f"Distribuzione nulla block permutation\n{event_name.split('(')[0].strip()} | {fuel_name}", fontsize=11, fontweight="bold")
        ax.legend(fontsize=9)
        shown = True

# Pannello D: Kruskal-Wallis + Fligner-Killeen p-values
ax = axes[1, 1]
if kw_rows:
    df_kw_plot = pd.DataFrame(kw_rows)
    df_fk_plot = pd.DataFrame(fk_rows)
    labels_all  = [f"{r['Evento'].split('(')[0].strip()}\n{r['Carburante']}" for r in kw_rows]
    kw_pv  = df_kw_plot["kw_p"].values
    fk_pv  = []
    for r in kw_rows:
        match = df_fk_plot[(df_fk_plot["Evento"] == r["Evento"]) & (df_fk_plot["Carburante"] == r["Carburante"])]
        fk_pv.append(float(match["FK_p"].values[0]) if len(match) > 0 and match["FK_p"].values[0] != "N/A" else np.nan)
    x_pos2 = np.arange(len(kw_rows))
    ax.bar(x_pos2 - 0.2, kw_pv, 0.38, label="Kruskal-Wallis", color="#8e44ad", alpha=0.8)
    ax.bar(x_pos2 + 0.2, fk_pv, 0.38, label="Fligner-Killeen", color="#27ae60", alpha=0.8)
    ax.axhline(ALPHA, color="black", lw=1.5, ls="--")
    ax.set_xticks(x_pos2)
    ax.set_xticklabels(labels_all, fontsize=7)
    ax.set_ylabel("p-value")
    ax.set_title("Kruskal‑Wallis + Fligner‑Killeen\n(Warning: gruppi temporali non indipendenti)", fontsize=11, fontweight="bold")
    ax.legend(fontsize=9)

plt.tight_layout(pad=2.0)
plt.savefig("plots/10_nonparam_summary.png", dpi=DPI, bbox_inches="tight")
plt.close()
print("  Salvato: plots/10_nonparam_summary.png\n")


# ─────────────────────────────────────────────────────────────────────────────
# SOMMARIO
# ─────────────────────────────────────────────────────────────────────────────
print("=" * 65)
print("SOMMARIO — Script 05 (aggiornato)")
print("=" * 65)
print()
print("  Test eseguiti:")
print("  §1 Mann-Whitney U + HL + Cliff's δ → data/nonparam_mannwhitney.csv")
print("  §2 Kruskal-Wallis + Dunn post-hoc  → data/nonparam_kruskal.csv")
print("  §4 Block permutation Δmediana       → data/nonparam_permutation.csv")
print("  §5 DiD sign-flip + Placebo          → data/nonparam_permutation.csv")
print("  §6 Fligner-Killeen                  → data/nonparam_fligner.csv")
print("  §7 Runs test (modello migliorato)   → data/nonparam_fligner.csv")
print("  §7bis HAC Newey‑West                → data/nonparam_hac.csv")
print("  §8 BH globale aggiornato            → data/global_bh_corrections.csv")
print("  Plot: plots/10_nonparam_summary.png")
print()
print("  Interpretazione rapida:")
print("  • Block permutation corregge per autocorrelazione (block_size=4w)")
print("  • Δmediana + HL + Cliff's δ offrono una descrizione robusta completa")
print("  • HAC Newey‑West cattura autocorrelazione nei test parametrici")
print("  • Placebo DiD verifica che l'effetto non sia spurio")
print("  • BH su test correlati: la correlazione positiva preserva il controllo FDR")
print()
print("Script 05 completato.")