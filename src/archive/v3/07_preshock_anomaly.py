"""
07_preshock_anomaly.py
=======================
Analisi dell'anomalia PRE-SHOCK del margine Iran-Israele (Giu 2025).

MOTIVAZIONE
───────────
Tutti i 4 casi Iran-Israele mostrano pre_anomalo_2sigma=True nei dati v1
(δ_pre ≈ +0.086 EUR/L benzina, +0.081 EUR/L diesel). Il margine era già
strutturalmente elevato nei mesi Jan–Jun 2025, prima del conflitto.
La pipeline v1 lo rilevava ma non investigava la domanda naturale: da quando?

DOMANDA SCIENTIFICA
────────────────────
Il margine era elevato per tutto il 2025, o c'è stato un cambio strutturale
identificabile? Se τ_pre ≈ inizio 2025 → l'anomalia preesiste al conflitto.
Se τ_pre >> inizio 2025 → possibile anticipazione dai mercati futures (hedging).

NOTA EPISTEMICA
───────────────
Questa analisi è DESCRITTIVO-ESPLORATIVA. I test in questo script
NON entrano nella famiglia BH globale (05_global_corrections_v2.py).
Scopo: contestualizzare l'anomalia pre-shock; non testare H₀ primaria.

SEZIONI
───────
  §1. Carica dati e prepara finestra 2023-01-01 / 2025-06-12
  §2. Bai-Perron brute-force — struttura breaks nella serie pre-shock
  §3. Confronto annuale: 2023, 2024, 2025-H1 vs baseline 2019
  §4. Stagionalità: Q1-Q2 confronto inter-annuale
  §5. Mann-Whitney: 2025-H1 pre-shock vs baseline 2019 e vs 2023/2024
  §6. Plot: serie temporale 2023–2025 con τ_pre evidenziati
  §7. Plot: confronto annuale boxplot

FINESTRA ANALISI
────────────────
  start:  2023-01-01
  end:    2025-06-12  (giorno prima del conflitto Iran-Israele: 2025-06-13)
  shock:  2025-06-13

Questa finestra esclude deliberatamente 2022 (contaminata da Ucraina)
e include 2023-2024 come riferimento "normale post-Ucraina".

Input:
  data/dataset_merged_with_futures.csv  (o dataset_merged.csv)

Output:
  data/preshock_anomaly.csv           → struttura breaks + test per carburante
  data/preshock_annual_stats.csv      → statistiche annuali 2019/2023/2024/2025-H1
  data/preshock_seasonal.csv          → confronto stagionale Q1-Q2
  plots/07_preshock_timeseries.png    → serie 2023-2025 con breaks evidenziati
  plots/07_preshock_annual.png        → boxplot annuali

Rif: Bai & Perron (1998) Econometrica 66(1) 47-78;
     Bai & Perron (2003) J. Applied Econometrics 18(1) 1-22;
     Casini & Perron (2021) Annual Review of Economics.
"""

from __future__ import annotations

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.dates as mdates
from scipy import stats
from scipy.stats import mannwhitneyu

warnings.filterwarnings("ignore")

# ─── Configurazione ───────────────────────────────────────────────────────────
ALPHA  = 0.05
DPI    = 160

ANALYSIS_WINDOW = {
    "start": pd.Timestamp("2023-01-01"),
    "end":   pd.Timestamp("2025-06-12"),    # giorno prima dello shock
    "shock": pd.Timestamp("2025-06-13"),
}

BASELINE_YEARS = [2019]                      # baseline primaria (pre-COVID)
COMPARISON_YEARS = [2023, 2024, "2025-H1"]  # anni di confronto

os.makedirs("data",  exist_ok=True)
os.makedirs("plots", exist_ok=True)


# ─────────────────────────────────────────────────────────────────────────────
# §1. CARICA DATI
# ─────────────────────────────────────────────────────────────────────────────

print("=" * 70)
print("§1. CARICA DATI — finestra 2023-01 / 2025-06-12")
print("=" * 70)

def _load_merged() -> pd.DataFrame:
    for fname in ["data/dataset_merged_with_futures.csv", "data/dataset_merged.csv"]:
        if os.path.exists(fname):
            return pd.read_csv(fname, index_col=0, parse_dates=True)
    raise FileNotFoundError("dataset_merged.csv non trovato — eseguire 01_data_pipeline.py")

try:
    merged = _load_merged()
    print(f"  Dataset caricato: {len(merged)} righe  ({merged.index.min().date()} → {merged.index.max().date()})")
except FileNotFoundError as e:
    print(f"  ERRORE: {e}")
    merged = None

# ─── Individua colonne margine ────────────────────────────────────────────────
MARGIN_COLS = {
    "Benzina": None,
    "Diesel":  None,
}
if merged is not None:
    cands_b = ["margine_benzina", "margine_benz_crack", "margine_benz"]
    cands_d = ["margine_diesel",  "margine_dies_crack", "margine_dies"]
    for c in cands_b:
        if c in merged.columns:
            MARGIN_COLS["Benzina"] = c
            break
    for c in cands_d:
        if c in merged.columns:
            MARGIN_COLS["Diesel"] = c
            break
    print(f"  Colonne margine: Benzina={MARGIN_COLS['Benzina']}, Diesel={MARGIN_COLS['Diesel']}")

FUELS = [f for f, c in MARGIN_COLS.items() if c is not None]
if not FUELS:
    print("  ATTENZIONE: nessuna colonna margine trovata. Uscita.")
    exit(0)


# ─────────────────────────────────────────────────────────────────────────────
# §2. BAI-PERRON BRUTE-FORCE sulla serie 2023-2025-H1
# ─────────────────────────────────────────────────────────────────────────────

print("\n" + "=" * 70)
print("§2. BAI-PERRON BRUTE-FORCE — struttura breaks pre-shock")
print("    (Finestra: 2023-01-01 → 2025-06-12; esclude post-shock Ucraina)")
print("=" * 70)

def bai_perron_brute(
    series: np.ndarray,
    dates: pd.DatetimeIndex,
    min_seg_frac: float = 0.15,
    max_breaks: int = 2,
) -> list[dict]:
    """
    Bai-Perron brute force su modello a media costante a tratti.
    Cerca il numero ottimale di break (0..max_breaks) tramite BIC.

    Restituisce lista di break:
        [{"date": ..., "t_stat": ..., "delta_before": ..., "delta_after": ...}]

    NOTA: questo implementa il caso più semplice (solo media, no regressori).
    Per un'implementazione completa usare il pacchetto `ruptures`.
    """
    n          = len(series)
    min_seg    = max(int(n * min_seg_frac), 3)
    # RSS per sotto-segmento [i:j]
    rss_cache: dict[tuple[int, int], float] = {}

    def seg_rss(i: int, j: int) -> float:
        key = (i, j)
        if key not in rss_cache:
            seg = series[i:j]
            rss_cache[key] = float(np.sum((seg - seg.mean()) ** 2))
        return rss_cache[key]

    # BIC per modello con k break
    def bic_k(break_idx: list[int]) -> float:
        """Break indices = indici *dopo* il break (segmento finale a destra)."""
        segs   = [0] + break_idx + [n]
        rss    = sum(seg_rss(segs[i], segs[i+1]) for i in range(len(segs)-1))
        k_par  = len(break_idx) + (len(segs) - 1)   # break + medie
        sigma2 = rss / n if n > k_par else 1e-9
        bic    = n * np.log(sigma2 + 1e-12) + k_par * np.log(n)
        return bic

    # ── BIC baseline (0 breaks) ────────────────────────────────────────────
    best_bic   = bic_k([])
    best_breaks: list[int] = []

    # ── Cerca 1 break ─────────────────────────────────────────────────────
    for k in range(min_seg, n - min_seg):
        b = bic_k([k])
        if b < best_bic:
            best_bic    = b
            best_breaks = [k]

    # ── Cerca 2 breaks ────────────────────────────────────────────────────
    if max_breaks >= 2:
        for k1 in range(min_seg, n - 2 * min_seg):
            for k2 in range(k1 + min_seg, n - min_seg):
                b = bic_k([k1, k2])
                if b < best_bic:
                    best_bic    = b
                    best_breaks = [k1, k2]

    # ── Converti break indices in statistiche ─────────────────────────────
    results: list[dict] = []
    if not best_breaks:
        return results

    segs = [0] + best_breaks + [n]
    means = [series[segs[i]:segs[i+1]].mean() for i in range(len(segs)-1)]

    for b_idx, brk in enumerate(best_breaks):
        seg_len_before = segs[b_idx + 1] - segs[b_idx]
        seg_len_after  = segs[b_idx + 2] - segs[b_idx + 1]

        before = series[segs[b_idx]:brk]
        after  = series[brk:segs[b_idx + 2]]

        delta = float(means[b_idx + 1] - means[b_idx])

        # t-stat Welch tra i due segmenti adiacenti
        t_stat, p_welch = np.nan, np.nan
        if len(before) >= 3 and len(after) >= 3:
            try:
                t_stat, p_welch = stats.ttest_ind(after, before, equal_var=False)
            except Exception:
                pass

        results.append({
            "break_index": int(brk),
            "date":        dates[brk] if brk < len(dates) else pd.NaT,
            "delta":       round(delta, 5),
            "mean_before": round(float(before.mean()), 5),
            "mean_after":  round(float(after.mean()), 5),
            "n_before":    int(len(before)),
            "n_after":     int(len(after)),
            "t_stat_welch": round(float(t_stat), 3) if not np.isnan(t_stat) else "N/A",
            "p_welch_descrittivo": round(float(p_welch), 5) if not np.isnan(p_welch) else "N/A",
            "nota_epistemica": (
                "DESCRITTIVO — break cercato con BIC, t-stat NON corretto per data-snooping. "
                "Non usare p_welch per inferenza confermativa."
            ),
        })

    return results


preshock_rows: list[dict] = []

BP_results: dict[str, list[dict]] = {}   # per i plot

for fuel in FUELS:
    mcol = MARGIN_COLS[fuel]
    series_full = merged[mcol].dropna()

    # Finestra analisi: 2023-01 → 2025-06-12
    window_series = series_full.loc[
        ANALYSIS_WINDOW["start"]:ANALYSIS_WINDOW["end"]
    ]
    baseline_2019 = series_full.loc["2019-01-01":"2019-12-31"]

    if len(window_series) < 15:
        print(f"  {fuel}: troppo pochi dati nella finestra ({len(window_series)}) — skip")
        continue

    print(f"\n  {fuel} — n={len(window_series)}  "
          f"({window_series.index[0].date()} → {window_series.index[-1].date()})")

    # ── Bai-Perron ─────────────────────────────────────────────────────────
    bp_breaks = bai_perron_brute(
        window_series.values,
        window_series.index,
        min_seg_frac=0.12,
        max_breaks=2
    )
    BP_results[fuel] = bp_breaks

    if bp_breaks:
        print(f"    Break strutturale rilevato (BIC-ottimale):")
        for br in bp_breaks:
            print(
                f"      τ_pre={br['date'].date() if pd.notna(br['date']) else 'N/A'}  "
                f"δ={br['delta']:+.4f}  "
                f"mean_before={br['mean_before']:.4f}  mean_after={br['mean_after']:.4f}  "
                f"[n_before={br['n_before']}, n_after={br['n_after']}]"
            )
            preshock_rows.append({
                "Carburante":         fuel,
                "finestra":           f"{ANALYSIS_WINDOW['start'].date()} / {ANALYSIS_WINDOW['end'].date()}",
                "tau_pre_BIC":        str(br["date"].date()) if pd.notna(br["date"]) else "N/A",
                "delta_break":        br["delta"],
                "mean_before_break":  br["mean_before"],
                "mean_after_break":   br["mean_after"],
                "n_before":           br["n_before"],
                "n_after":            br["n_after"],
                "t_stat_descrittivo": br["t_stat_welch"],
                "p_welch_NO_INFERENZA": br["p_welch_descrittivo"],
                "nota": br["nota_epistemica"],
            })
    else:
        print(f"    Nessun break strutturale (BIC preferisce modello senza break)")
        preshock_rows.append({
            "Carburante":         fuel,
            "finestra":           f"{ANALYSIS_WINDOW['start'].date()} / {ANALYSIS_WINDOW['end'].date()}",
            "tau_pre_BIC":        "NESSUN BREAK",
            "delta_break":        np.nan,
            "mean_before_break":  np.nan,
            "mean_after_break":   np.nan,
            "n_before":           np.nan,
            "n_after":            np.nan,
            "t_stat_descrittivo": np.nan,
            "p_welch_NO_INFERENZA": np.nan,
            "nota": "Nessun break strutturale BIC-ottimale nella finestra 2023-2025-H1",
        })


# ─────────────────────────────────────────────────────────────────────────────
# §3. CONFRONTO ANNUALE — statistiche descrittive
# ─────────────────────────────────────────────────────────────────────────────

print("\n" + "=" * 70)
print("§3. CONFRONTO ANNUALE — 2019 / 2023 / 2024 / 2025-H1")
print("=" * 70)

annual_rows: list[dict] = []

for fuel in FUELS:
    mcol = MARGIN_COLS[fuel]
    series_full = merged[mcol].dropna()

    print(f"\n  {fuel}:")
    for year_label in [2019, 2023, 2024, "2025-H1"]:
        if year_label == "2025-H1":
            yr_data = series_full.loc[
                "2025-01-01":ANALYSIS_WINDOW["end"]
            ]
            yr_str  = "2025-H1 (pre-shock)"
        else:
            yr_data = series_full.loc[f"{year_label}-01-01":f"{year_label}-12-31"]
            yr_str  = str(year_label)

        if len(yr_data) < 3:
            continue

        mean_v  = float(yr_data.mean())
        std_v   = float(yr_data.std())
        median_v= float(yr_data.median())
        q25_v   = float(yr_data.quantile(0.25))
        q75_v   = float(yr_data.quantile(0.75))
        max_v   = float(yr_data.max())

        # MW vs baseline 2019
        base_2019 = series_full.loc["2019-01-01":"2019-12-31"]
        mw_p_vs19 = np.nan
        if len(base_2019) >= 3 and year_label != 2019:
            _, mw_p_vs19 = mannwhitneyu(yr_data.values, base_2019.values, alternative="two-sided")

        print(
            f"    {yr_str:<20}: mean={mean_v:.4f}  std={std_v:.4f}  "
            f"med={median_v:.4f}  n={len(yr_data)}"
            + (f"  MW_vs2019 p={mw_p_vs19:.4f}" if not np.isnan(mw_p_vs19) else "")
        )

        annual_rows.append({
            "Carburante":       fuel,
            "anno":             yr_str,
            "n":                len(yr_data),
            "mean":             round(mean_v, 5),
            "std":              round(std_v, 5),
            "median":           round(median_v, 5),
            "q25":              round(q25_v, 5),
            "q75":              round(q75_v, 5),
            "max":              round(max_v, 5),
            "MW_p_vs_2019":     round(float(mw_p_vs19), 5) if not np.isnan(mw_p_vs19) else "N/A",
            "tipo":             "esplorativo",
        })

if annual_rows:
    pd.DataFrame(annual_rows).to_csv("data/preshock_annual_stats.csv", index=False)
    print(f"\n  Salvato: data/preshock_annual_stats.csv ({len(annual_rows)} righe)")


# ─────────────────────────────────────────────────────────────────────────────
# §4. STAGIONALITÀ — Q1-Q2 confronto inter-annuale
# ─────────────────────────────────────────────────────────────────────────────

print("\n" + "=" * 70)
print("§4. STAGIONALITÀ — Q1-Q2 (gen-giu) confronto 2023/2024/2025")
print("=" * 70)

seasonal_rows: list[dict] = []

for fuel in FUELS:
    mcol = MARGIN_COLS[fuel]
    series_full = merged[mcol].dropna()

    print(f"\n  {fuel} — Q1-Q2:")
    q1q2_data: dict[str | int, pd.Series] = {}

    for year in [2019, 2023, 2024, 2025]:
        q1q2 = series_full.loc[
            f"{year}-01-01":f"{year}-06-12"
            if year == 2025 else f"{year}-06-30"
        ]
        if len(q1q2) >= 3:
            q1q2_data[year] = q1q2
            print(
                f"    {year} Q1-Q2: mean={q1q2.mean():.4f}  "
                f"std={q1q2.std():.4f}  n={len(q1q2)}"
            )

    # MW tests Q1-Q2 2025 vs anni precedenti
    ref_2025 = q1q2_data.get(2025)
    base_2019_q1q2 = q1q2_data.get(2019)
    if ref_2025 is not None:
        for ref_year, ref_data in [(y, d) for y, d in q1q2_data.items() if y != 2025]:
            _, mw_p = mannwhitneyu(ref_2025.values, ref_data.values, alternative="two-sided")
            mean_diff = ref_2025.mean() - ref_data.mean()
            print(
                f"    MW Q1-Q2-2025 vs Q1-Q2-{ref_year}: "
                f"δ={mean_diff:+.4f}  p={mw_p:.4f}"
            )
            seasonal_rows.append({
                "Carburante":    fuel,
                "periodo_A":     "2025-Q1Q2",
                "periodo_B":     f"{ref_year}-Q1Q2",
                "n_A":           len(ref_2025),
                "n_B":           len(ref_data),
                "mean_A":        round(float(ref_2025.mean()), 5),
                "mean_B":        round(float(ref_data.mean()), 5),
                "mean_diff_A_B": round(float(mean_diff), 5),
                "MW_p_twosided": round(float(mw_p), 5),
                "tipo":          "esplorativo",
                "nota":          "Stagionalità Q1-Q2 pre-shock — non in BH",
            })

if seasonal_rows:
    pd.DataFrame(seasonal_rows).to_csv("data/preshock_seasonal.csv", index=False)
    print(f"\n  Salvato: data/preshock_seasonal.csv ({len(seasonal_rows)} test)")


# ─────────────────────────────────────────────────────────────────────────────
# §5. MANN-WHITNEY: 2025-H1 pre-shock vs baseline e vs 2023/2024
# ─────────────────────────────────────────────────────────────────────────────

print("\n" + "=" * 70)
print("§5. MANN-WHITNEY — 2025-H1 pre-shock vs baseline 2019 e vs 2023/2024")
print("=" * 70)

mw_rows: list[dict] = []

if merged is not None:
    for fuel in FUELS:
        mcol = MARGIN_COLS[fuel]
        series_full = merged[mcol].dropna()

        h1_2025 = series_full.loc["2025-01-01":ANALYSIS_WINDOW["end"]]
        b_2019  = series_full.loc["2019-01-01":"2019-12-31"]
        b_2023  = series_full.loc["2023-01-01":"2023-12-31"]
        b_2024  = series_full.loc["2024-01-01":"2024-12-31"]

        if len(h1_2025) < 3:
            print(f"  {fuel}: 2025-H1 dati insufficienti ({len(h1_2025)})")
            continue

        comparisons = [
            ("2019-full",   b_2019,  "greater"),
            ("2023-full",   b_2023,  "two-sided"),
            ("2024-full",   b_2024,  "two-sided"),
        ]
        print(f"\n  {fuel} — 2025-H1 (n={len(h1_2025)}, mean={h1_2025.mean():.4f}):")
        for ref_label, ref_data, alt in comparisons:
            if len(ref_data) < 3:
                continue
            mw_stat, mw_p = mannwhitneyu(h1_2025.values, ref_data.values, alternative=alt)
            delta_mu = h1_2025.mean() - ref_data.mean()
            print(
                f"    vs {ref_label:<15} (n={len(ref_data)}): "
                f"δ={delta_mu:+.4f}  MW_stat={mw_stat:.1f}  p={mw_p:.5f}  "
                f"[{alt}]"
            )
            mw_rows.append({
                "Carburante":      fuel,
                "serie_A":         f"2025-H1 pre-shock (n={len(h1_2025)})",
                "serie_B":         f"{ref_label} (n={len(ref_data)})",
                "mean_A":          round(float(h1_2025.mean()), 5),
                "mean_B":          round(float(ref_data.mean()), 5),
                "delta_A_minus_B": round(float(delta_mu), 5),
                "MW_stat":         round(float(mw_stat), 1),
                "p_value":         round(float(mw_p), 6),
                "alternativa":     alt,
                "tipo":            "esplorativo",
                "nota":            "Non in BH — contestualizzazione anomalia pre-shock",
            })

    if mw_rows:
        df_mw = pd.DataFrame(mw_rows)
        # Aggiungi a preshock_anomaly.csv come sezione aggiuntiva
        df_mw.to_csv("data/preshock_mw_tests.csv", index=False)
        print(f"\n  Salvato: data/preshock_mw_tests.csv ({len(mw_rows)} test)")


# ─────────────────────────────────────────────────────────────────────────────
# §6. PLOT — Serie temporale 2023-2025 con τ_pre evidenziati
# ─────────────────────────────────────────────────────────────────────────────

print("\n§6. PLOT — Serie temporale pre-shock + breaks strutturali...")

if merged is not None and FUELS:
    nfuels = len(FUELS)
    fig, axes = plt.subplots(nfuels, 1, figsize=(14, 4 * nfuels), squeeze=False)

    colors_fuel = {"Benzina": "#d6604d", "Diesel": "#4393c3"}

    for i, fuel in enumerate(FUELS):
        ax    = axes[i][0]
        mcol  = MARGIN_COLS[fuel]
        col   = colors_fuel.get(fuel, "steelblue")

        # Serie completa 2021-2025 (per contesto)
        context_start = "2021-01-01"
        full_ctx = merged[mcol].loc[context_start:].dropna()
        # Evidenzia finestra analisi
        window_s = full_ctx.loc[ANALYSIS_WINDOW["start"]:ANALYSIS_WINDOW["end"]]
        # Baseline 2019 media + banda 2σ
        base_2019 = merged[mcol].loc["2019-01-01":"2019-12-31"].dropna()
        mu_19     = base_2019.mean()
        sd_19     = base_2019.std()

        ax.plot(full_ctx.index, full_ctx.values, color="grey", lw=0.8, alpha=0.4, label="_nolegend_")
        ax.plot(window_s.index, window_s.values, color=col, lw=1.4,
                label=f"{fuel} — finestra analisi 2023–2025-H1")

        # Banda 2σ baseline 2019
        ax.axhline(mu_19, color="black", lw=1.0, ls="--", alpha=0.7, label=f"μ_2019={mu_19:.3f}")
        ax.axhspan(mu_19 - 2*sd_19, mu_19 + 2*sd_19, alpha=0.08, color="green",
                   label="±2σ baseline 2019")
        ax.axhline(mu_19 + 2*sd_19, color="green", lw=0.6, ls=":", alpha=0.5)

        # τ shock
        ax.axvline(ANALYSIS_WINDOW["shock"], color="crimson", lw=1.8, ls="-",
                   label=f"Shock Iran-Israele {ANALYSIS_WINDOW['shock'].date()}")

        # τ_pre da Bai-Perron
        breaks = BP_results.get(fuel, [])
        bp_colors = ["#ff7f00", "#984ea3"]
        for j, br in enumerate(breaks):
            tau_date = br.get("date")
            if tau_date is not None and pd.notna(tau_date):
                ax.axvline(tau_date, color=bp_colors[j % len(bp_colors)],
                           lw=1.4, ls="--",
                           label=f"τ_pre (BIC) = {tau_date.date()}")
                ax.annotate(
                    f"τ_pre\n{tau_date.strftime('%b %Y')}",
                    xy=(tau_date, ax.get_ylim()[1] if ax.get_ylim()[1] != 0 else 0.4),
                    xytext=(10, -15), textcoords="offset points",
                    fontsize=7, color=bp_colors[j % len(bp_colors)],
                    arrowprops=dict(arrowstyle="->", color=bp_colors[j], lw=0.8)
                )

        # Ombra area pre-shock anomala (2025-H1)
        h1_data = merged[mcol].loc["2025-01-01":ANALYSIS_WINDOW["end"]].dropna()
        if len(h1_data) > 0:
            ax.fill_between(h1_data.index, mu_19 + 2*sd_19, h1_data.values,
                            where=(h1_data.values > mu_19 + 2*sd_19),
                            alpha=0.25, color="orange",
                            label="Anomalia > 2σ_2019 (pre-shock)")

        ax.set_ylabel("Margine (EUR/L)")
        ax.set_title(
            f"Margine crack spread — {fuel}\n"
            f"Analisi struttura pre-shock Iran-Israele",
            fontsize=10
        )
        ax.legend(fontsize=7, loc="upper left", ncol=2)
        ax.xaxis.set_major_locator(mdates.MonthLocator(bymonth=[1, 4, 7, 10]))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%b\n%Y"))
        ax.grid(True, alpha=0.2)

    fig.suptitle(
        "Anomalia pre-shock margine carburanti — Iran-Israele (Giu 2025)\n"
        "Il margine era strutturalmente elevato prima del conflitto. Da quando?",
        fontsize=11, y=1.01
    )
    fig.tight_layout()
    fig.savefig("plots/07_preshock_timeseries.png", dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print("  Plot: plots/07_preshock_timeseries.png")


# ─────────────────────────────────────────────────────────────────────────────
# §7. PLOT — Boxplot annuali + Q1-Q2
# ─────────────────────────────────────────────────────────────────────────────

print("\n§7. PLOT — Boxplot annuali...")

if merged is not None and annual_rows:
    nfuels = len(FUELS)
    fig, axes = plt.subplots(1, nfuels, figsize=(6 * nfuels, 6), squeeze=False)

    year_groups = [2019, 2022, 2023, 2024, "2025-H1"]
    palette     = ["#b2df8a", "#e31a1c", "#1f78b4", "#33a02c", "#ff7f00"]

    for i, fuel in enumerate(FUELS):
        ax   = axes[0][i]
        mcol = MARGIN_COLS[fuel]
        series_full = merged[mcol].dropna()

        data_to_plot = []
        labels_plot  = []
        colors_plot  = []

        for yr, pal in zip(year_groups, palette):
            if yr == "2025-H1":
                seg = series_full.loc["2025-01-01":ANALYSIS_WINDOW["end"]]
                lbl = "2025\nH1"
            elif yr == 2022:
                seg = series_full.loc["2022-01-01":"2022-12-31"]
                lbl = "2022\n(Ucraina)"
            else:
                seg = series_full.loc[f"{yr}-01-01":f"{yr}-12-31"]
                lbl = str(yr)
            if len(seg) >= 3:
                data_to_plot.append(seg.values)
                labels_plot.append(lbl)
                colors_plot.append(pal)

        if data_to_plot:
            bp = ax.boxplot(data_to_plot, labels=labels_plot, patch_artist=True,
                            medianprops=dict(color="black", lw=1.5))
            for patch, col in zip(bp["boxes"], colors_plot):
                patch.set_facecolor(col)
                patch.set_alpha(0.75)

        # Banda 2σ 2019
        base_2019 = series_full.loc["2019-01-01":"2019-12-31"].dropna()
        mu_19, sd_19 = base_2019.mean(), base_2019.std()
        ax.axhspan(mu_19 - 2*sd_19, mu_19 + 2*sd_19, alpha=0.08, color="green")
        ax.axhline(mu_19 + 2*sd_19, color="green", lw=0.8, ls="--",
                   label=f"μ_2019 + 2σ = {mu_19+2*sd_19:.3f}")

        ax.set_ylabel("Margine (EUR/L)")
        ax.set_title(
            f"Distribuzione annuale margine — {fuel}\n"
            f"Confronto 2019 / 2022-2024 / 2025-H1 pre-shock",
            fontsize=9
        )
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.2, axis="y")

    fig.suptitle(
        "Anomalia strutturale 2025-H1 nel contesto storico\n"
        "(bande verdi = ±2σ baseline 2019)",
        fontsize=11, y=1.01
    )
    fig.tight_layout()
    fig.savefig("plots/07_preshock_annual.png", dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print("  Plot: plots/07_preshock_annual.png")


# ─────────────────────────────────────────────────────────────────────────────
# SALVA output principale e SOMMARIO
# ─────────────────────────────────────────────────────────────────────────────

if preshock_rows:
    pd.DataFrame(preshock_rows).to_csv("data/preshock_anomaly.csv", index=False)
    print(f"\n  Salvato: data/preshock_anomaly.csv ({len(preshock_rows)} righe)")

print("\n" + "=" * 70)
print("  SOMMARIO 07_preshock_anomaly.py")
print("=" * 70)
print("  Output:")
print("    data/preshock_anomaly.csv       — break strutturali BIC (Bai-Perron)")
print("    data/preshock_annual_stats.csv  — statistiche 2019/2023/2024/2025-H1")
print("    data/preshock_seasonal.csv      — confronto Q1-Q2 inter-annuale")
print("    data/preshock_mw_tests.csv      — MW: 2025-H1 vs baseline e anni prec.")
print("    plots/07_preshock_timeseries.png")
print("    plots/07_preshock_annual.png")
print()
print("  INTERPRETAZIONE:")
print("  • Se τ_pre ≈ gen-feb 2025 → anomalia strutturale indipendente dal conflitto")
print("  • Se τ_pre ≈ mag-giu 2025 → possibile anticipazione dai mercati futures")
print("  • Se nessun break → margine uniformemente elevato per tutto 2025-H1")
print()
print("  ⚠  NOTA: tutti i test in questo script sono ESPLORATIVI.")
print("     I p-value MW NON entrano nella BH globale (05_global_corrections_v2.py).")
print("=" * 70)
print("\nScript 07 completato.")
