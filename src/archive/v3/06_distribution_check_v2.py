"""
06_distribution_check_v2.py
============================
Diagnostica delle distribuzioni e quantificazione dell'impatto dell'autocorrelazione.

OBIETTIVO PRINCIPALE (v2)
──────────────────────────
Rispondere a: "quanto è affidabile il p-value del Welch t-test dato che DW ≈ 0.3?"

Il problema:
  Con ρ̂ ≈ 0.85, il numero effettivo di osservazioni indipendenti è:
    n_eff = n × (1 − ρ̂) / (1 + ρ̂) ≈ 27 × 0.15/1.85 ≈ 2
  Il Welch t tratta 27 obs come 27 indipendenti → gonfia la potenza di ~3.5×.
  Questo script lo documenta esplicitamente per ogni serie.

SEZIONI
───────
  §1. Diagnostica distribuzione (BP, DW, SW) per ogni serie/evento
  §2. Calcolo n_eff e Andrews bandwidth ottimale per ogni serie post-shock
  §3. Tabella "inflazione potenza": confronto df_nominale vs df_effettivo
  §4. Sommario raccomandazioni per il paper

FORMULE CHIAVE
──────────────
  n_eff   = n × (1 − ρ̂) / (1 + ρ̂)              [Kish, 1965]
  df_eff  = n_eff − 1                              [gradi di libertà effettivi]
  BW_nw   = 1.1447 × (4ρ²/(1−ρ)⁴ × n)^(1/3)     [Andrews 1991, AR(1)]
  inflation_factor = n / n_eff                      [≈ 3.5 per ρ=0.85, n=27]

  Interpretazione inflation_factor:
    ~1.0 : dati quasi indipendenti, test valido
    1–2  : leggera autocorrelazione, cautela moderata
    2–5  : autocorrelazione significativa, p-value nominali conservativi
    >5   : dati fortemente autocorrelati, p-value poco affidabili

Input:
  data/dataset_merged_with_futures.csv  (o dataset_merged.csv)
  data/confirmatory_pvalues_v2.csv      (per recuperare p-value confirmativi)
  data/exploratory_results_v2.csv       (per rho_hat se disponibile)

Output:
  data/distribution_diagnostics_v2.csv  → BP, DW, SW, rho_hat per ogni serie
  data/neff_report_v2.csv               → n_eff, Andrews BW, inflazione potenza
  plots/06_autocorr_v2.png              → ACF plots serie post-shock
  plots/06_neff_v2.png                  → grafico inflazione potenza

Rif: Kish (1965) Survey Sampling; Andrews (1991) Econometrica 59(3) 817-858;
     Newey & West (1987) Econometrica 55(3) 703-708;
     Durbin & Watson (1950) Biometrika 37(3-4) 409-428.
"""

from __future__ import annotations

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats
from statsmodels.stats.stattools import durbin_watson
from statsmodels.stats.diagnostic import het_breuschpagan
from statsmodels.graphics.tsaplots import plot_acf
import statsmodels.api as sm

warnings.filterwarnings("ignore")

# ─── Configurazione ───────────────────────────────────────────────────────────
ALPHA = 0.05
DPI   = 150

os.makedirs("data",  exist_ok=True)
os.makedirs("plots", exist_ok=True)

# ─── Carica dataset principale ────────────────────────────────────────────────
def _load_merged() -> pd.DataFrame:
    for fname in ["data/dataset_merged_with_futures.csv", "data/dataset_merged.csv"]:
        if os.path.exists(fname):
            return pd.read_csv(fname, index_col=0, parse_dates=True)
    raise FileNotFoundError("dataset_merged.csv non trovato")

try:
    merged = _load_merged()
    print(f"  Dataset caricato: {len(merged)} righe")
except FileNotFoundError as e:
    print(f"  ERRORE: {e}")
    merged = None

# ─── Configurazione eventi ────────────────────────────────────────────────────
EVENTS = {
    "Ucraina (Feb 2022)": {
        "shock":      pd.Timestamp("2022-02-24"),
        "pre_start":  pd.Timestamp("2021-01-11"),
        "post_end":   pd.Timestamp("2022-12-31"),
    },
    "Iran-Israele (Giu 2025)": {
        "shock":      pd.Timestamp("2025-06-13"),
        "pre_start":  pd.Timestamp("2024-01-01"),
        "post_end":   pd.Timestamp("2025-12-31"),
    },
}

FUELS    = ["Benzina", "Diesel"]
SERIES   = ["Brent", "Benzina", "Diesel"]   # per diagnostiche prezzo

# ─── Helper: colonne ──────────────────────────────────────────────────────────
PRICE_COLS = {
    "Brent":   "log_brent",
    "Benzina": "log_benzina",
    "Diesel":  "log_diesel",
}
MARGIN_COLS_CANDIDATES = {
    "Benzina": ["margine_benzina", "margine_benz_crack", "margine_benz"],
    "Diesel":  ["margine_diesel",  "margine_dies_crack", "margine_dies"],
}

def _col(df: pd.DataFrame, candidates: list[str]) -> str | None:
    for c in candidates:
        if c in df.columns:
            return c
    return None


# ─────────────────────────────────────────────────────────────────────────────
# §1. DIAGNOSTICA DISTRIBUZIONE
# ─────────────────────────────────────────────────────────────────────────────

print("\n" + "=" * 70)
print("§1. DIAGNOSTICA DISTRIBUZIONE (BP, DW, SW)")
print("=" * 70)

diag_rows: list[dict] = []

if merged is not None:
    for event_name, ecfg in EVENTS.items():
        shock     = ecfg["shock"]
        pre_start = ecfg["pre_start"]
        post_end  = ecfg["post_end"]

        for serie in SERIES:
            # Usa colonna log-prezzo per diagnostica
            col = PRICE_COLS.get(serie)
            if col is None or col not in merged.columns:
                continue

            window = merged.loc[pre_start:post_end, col].dropna()
            if len(window) < 10:
                continue

            # Residui da regressione lineare su tempo (detrend minimo)
            t = np.arange(len(window), dtype=float)
            X = sm.add_constant(t)
            try:
                ols = sm.OLS(window.values, X).fit()
                resid = ols.resid
            except Exception:
                resid = window.diff().dropna().values

            # ── Breusch-Pagan ──────────────────────────────────────────────
            bp_lm, bp_p = np.nan, np.nan
            try:
                bp_lm, bp_p, _, _ = het_breuschpagan(resid, X[:len(resid)])
            except Exception:
                pass

            # ── Durbin-Watson ──────────────────────────────────────────────
            dw_stat = np.nan
            rho_hat = np.nan
            try:
                dw_stat = float(durbin_watson(resid))
                rho_hat = 1.0 - dw_stat / 2.0   # approssimazione DW → rho
            except Exception:
                pass

            # ── Shapiro-Wilk ───────────────────────────────────────────────
            sw_w, sw_p = np.nan, np.nan
            try:
                if len(resid) >= 3:
                    sw_w, sw_p = stats.shapiro(resid)
            except Exception:
                pass

            # ── Ljung-Box (lag 1, lag 4) ───────────────────────────────────
            lb_p1, lb_p4 = np.nan, np.nan
            try:
                from statsmodels.stats.diagnostic import acorr_ljungbox
                lb = acorr_ljungbox(resid, lags=[1, 4], return_df=True)
                lb_p1 = float(lb["lb_pvalue"].iloc[0])
                lb_p4 = float(lb["lb_pvalue"].iloc[1]) if len(lb) > 1 else np.nan
            except Exception:
                pass

            dw_verdict = (
                "autocorrelazione positiva" if dw_stat < 1.5 else
                "ok" if dw_stat <= 2.5 else
                "autocorrelazione negativa"
            )

            row = {
                "Evento":       event_name,
                "Serie":        serie,
                "n_obs":        len(window),
                "BP_LM":        round(bp_lm, 4) if not np.isnan(bp_lm) else "N/A",
                "BP_p":         round(bp_p, 5)  if not np.isnan(bp_p)  else "N/A",
                "BP_H0":        (
                    "eteroschedasticità" if (not np.isnan(bp_p) and bp_p < ALPHA)
                    else "omoschedasticità"
                ),
                "DW":           round(dw_stat, 4) if not np.isnan(dw_stat) else "N/A",
                "DW_H0":        dw_verdict,
                "rho_hat":      round(rho_hat, 4) if not np.isnan(rho_hat) else "N/A",
                "LjungBox_p1":  round(lb_p1, 5) if not np.isnan(lb_p1) else "N/A",
                "LjungBox_p4":  round(lb_p4, 5) if not np.isnan(lb_p4) else "N/A",
                "SW_W":         round(sw_w, 4) if not np.isnan(sw_w) else "N/A",
                "SW_p":         round(sw_p, 5) if not np.isnan(sw_p) else "N/A",
                "SW_H0":        (
                    "NON normale" if (not np.isnan(sw_p) and sw_p < ALPHA)
                    else "normalità non rigettata"
                ),
            }
            diag_rows.append(row)

            print(
                f"  {event_name.split('(')[0].strip():<22} | {serie:<8}: "
                f"DW={row['DW']}  ρ̂={row['rho_hat']}  "
                f"BP_p={row['BP_p']}  SW_p={row['SW_p']}"
            )

if diag_rows:
    df_diag = pd.DataFrame(diag_rows)
    df_diag.to_csv("data/distribution_diagnostics_v2.csv", index=False)
    print(f"\n  Salvato: data/distribution_diagnostics_v2.csv ({len(df_diag)} righe)")
else:
    df_diag = pd.DataFrame()
    print("  Nessun risultato diagnostico prodotto.")


# ─────────────────────────────────────────────────────────────────────────────
# §2. n_eff E ANDREWS BANDWIDTH
# ─────────────────────────────────────────────────────────────────────────────

print("\n" + "=" * 70)
print("§2. n_eff, ANDREWS BW, INFLAZIONE POTENZA — serie post-shock")
print("=" * 70)

def n_eff_from_rho(n: int, rho: float) -> float:
    """Kish (1965): n_eff = n × (1−ρ)/(1+ρ)."""
    rho = np.clip(rho, -0.9999, 0.9999)
    return n * (1 - rho) / (1 + rho)

def andrews_bw_ar1(rho: float, n: int) -> int:
    """
    Andrews (1991) plug-in bandwidth per Newey-West con AR(1).
    α̂_NW = 4ρ²/(1−ρ)⁴
    BW = 1.1447 × (α̂_NW × n)^(1/3)
    Ritorna min(ceil(BW), n//2).
    """
    rho = np.clip(rho, -0.9999, 0.9999)
    alpha_hat = 4 * rho**2 / (1 - rho)**4
    bw_raw    = 1.1447 * (alpha_hat * n) ** (1 / 3)
    return int(np.clip(np.ceil(bw_raw), 4, n // 2))

def inflation_factor(n: int, rho: float) -> float:
    """Fattore inflazione potenza = n / n_eff."""
    neff = n_eff_from_rho(n, rho)
    return n / neff if neff > 0 else np.inf

def interpret_inflation(factor: float) -> str:
    if factor < 1.5:
        return "ok"
    if factor < 2.5:
        return "CAUTELA: inflazione lieve"
    if factor < 5.0:
        return "ATTENZIONE: inflazione significativa"
    return "CRITICO: n_eff molto piccolo — p-value non affidabili"

neff_rows: list[dict] = []

if merged is not None:
    for event_name, ecfg in EVENTS.items():
        shock    = ecfg["shock"]
        post_end = ecfg["post_end"]

        for fuel in FUELS:
            cands = MARGIN_COLS_CANDIDATES[fuel]
            mcol  = _col(merged, cands)
            if mcol is None:
                print(f"  {event_name} {fuel}: colonna margine non trovata")
                continue

            post_series = merged.loc[shock:post_end, mcol].dropna()
            n_post      = len(post_series)
            if n_post < 5:
                print(f"  {event_name} {fuel}: troppo pochi dati post-shock ({n_post})")
                continue

            # ρ̂ stimato da AR(1) su residui (più robusto di DW)
            try:
                t_arr = np.arange(n_post, dtype=float)
                X_t   = sm.add_constant(t_arr)
                res   = sm.OLS(post_series.values, X_t).fit()
                rho_hat_ar1 = float(
                    sm.OLS(res.resid[1:], res.resid[:-1]).fit().params[0]
                )
            except Exception:
                dw_v    = float(durbin_watson(post_series.diff().dropna().values))
                rho_hat_ar1 = 1.0 - dw_v / 2.0

            rho_hat_ar1 = np.clip(rho_hat_ar1, -0.999, 0.999)

            n_eff_val    = n_eff_from_rho(n_post, rho_hat_ar1)
            df_nominal   = n_post - 1
            df_effective = max(n_eff_val - 1, 0.5)
            bw_andrews   = andrews_bw_ar1(rho_hat_ar1, n_post)
            inf_factor   = inflation_factor(n_post, rho_hat_ar1)
            inf_note     = interpret_inflation(inf_factor)

            # Correlazione al lag bw_fixed=4 (vecchio default): quanto rimane?
            rho_at_lag4   = rho_hat_ar1 ** 4
            rho_at_lag_bw = rho_hat_ar1 ** bw_andrews if bw_andrews > 0 else np.nan

            # t critico a df_eff vs t critico nominale
            from scipy.stats import t as t_dist
            t_crit_nominal  = float(t_dist.ppf(0.95, df=df_nominal))
            t_crit_eff      = float(t_dist.ppf(0.95, df=max(df_effective, 1)))

            # Carica p-value confirmatorio se disponibile
            conf_p = "N/A"
            if os.path.exists("data/confirmatory_pvalues_v2.csv"):
                try:
                    df_conf = pd.read_csv("data/confirmatory_pvalues_v2.csv")
                    mask = (
                        (df_conf["evento"].str.contains(event_name.split("(")[0].strip(), na=False)) &
                        (df_conf["carburante"].str.lower() == fuel.lower()) &
                        (df_conf.get("test", pd.Series(["Welch_t"] * len(df_conf))) == "Welch_t") &
                        (df_conf.get("split_type", pd.Series(["shock_hard"] * len(df_conf))) == "shock_hard")
                    )
                    if mask.any():
                        conf_p = float(df_conf.loc[mask, "p_value"].iloc[0])
                except Exception:
                    pass

            print(
                f"  {event_name.split('(')[0].strip():<22} | {fuel:<8}: "
                f"n={n_post}  ρ̂={rho_hat_ar1:.3f}  "
                f"n_eff={n_eff_val:.1f}  "
                f"Andrews_BW={bw_andrews}  "
                f"inflation={inf_factor:.1f}×  "
                f"→ {inf_note}"
            )

            neff_rows.append({
                "Evento":              event_name,
                "Carburante":          fuel,
                "serie":               f"margine_{fuel.lower()}",
                "n_post":              n_post,
                "rho_hat_AR1":         round(rho_hat_ar1, 4),
                "n_eff":               round(n_eff_val, 1),
                "df_nominale":         df_nominal,
                "df_effettivo":        round(df_effective, 1),
                "inflation_factor":    round(inf_factor, 2),
                "inflazione_nota":     inf_note,
                "t_critico_nominale":  round(t_crit_nominal, 3),
                "t_critico_effettivo": round(t_crit_eff, 3),
                "andrews_BW":          bw_andrews,
                "bw_fixato_v1":        4,
                "rho_al_lag4":         round(rho_at_lag4, 3),
                "rho_al_lag_andrews":  round(rho_at_lag_bw, 6) if not np.isnan(rho_at_lag_bw) else "N/A",
                "p_confirmatorio_v2":  conf_p,
                "raccomandazione": (
                    f"Riportare n_eff≈{n_eff_val:.0f} nel paper. "
                    f"Usare Andrews BW={bw_andrews} per HAC."
                    + (" TEST POCO INFORMATIVO." if bw_andrews >= n_post // 2 else "")
                ),
            })

if neff_rows:
    df_neff = pd.DataFrame(neff_rows)
    df_neff.to_csv("data/neff_report_v2.csv", index=False)
    print(f"\n  Salvato: data/neff_report_v2.csv ({len(df_neff)} righe)")
else:
    df_neff = pd.DataFrame()
    print("  Nessun risultato n_eff prodotto.")


# ─────────────────────────────────────────────────────────────────────────────
# §3. PLOT — ACF e inflazione potenza
# ─────────────────────────────────────────────────────────────────────────────

print("\n§3. PLOT diagnostici...")

if merged is not None and not df_neff.empty:

    # ── Plot 1: ACF delle serie post-shock ──────────────────────────────────
    n_series  = len(neff_rows)
    ncols     = min(n_series, 4)
    nrows     = (n_series + ncols - 1) // ncols

    if n_series > 0:
        fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 3.5 * nrows),
                                 squeeze=False)
        idx = 0
        for event_name, ecfg in EVENTS.items():
            shock    = ecfg["shock"]
            post_end = ecfg["post_end"]
            for fuel in FUELS:
                if idx >= n_series:
                    break
                cands = MARGIN_COLS_CANDIDATES[fuel]
                mcol  = _col(merged, cands)
                if mcol is None:
                    idx += 1
                    continue

                post_s = merged.loc[shock:post_end, mcol].dropna()
                r, c   = divmod(idx, ncols)
                ax     = axes[r][c]
                try:
                    plot_acf(post_s.values, lags=min(20, len(post_s) // 2),
                             ax=ax, alpha=0.05, zero=False)
                except Exception:
                    ax.text(0.5, 0.5, "ACF non disponibile",
                            ha="center", va="center", transform=ax.transAxes)

                row_r = df_neff[
                    (df_neff["Evento"] == event_name) &
                    (df_neff["Carburante"] == fuel)
                ]
                inf_str = ""
                if not row_r.empty:
                    inf_str = f"ρ̂={row_r['rho_hat_AR1'].iloc[0]:.3f}  n_eff={row_r['n_eff'].iloc[0]:.1f}"

                ax.set_title(
                    f"{event_name.split('(')[0].strip()} — {fuel}\n{inf_str}",
                    fontsize=8
                )
                ax.set_xlabel("Lag (settimane)")
                idx += 1

        # Nascondi assi inutilizzati
        for jj in range(idx, nrows * ncols):
            r, c = divmod(jj, ncols)
            axes[r][c].set_visible(False)

        fig.suptitle(
            "ACF serie margini post-shock\n"
            "Autocorrelazione alta → n_eff << n → inflazione potenza",
            fontsize=10, y=1.01
        )
        fig.tight_layout()
        fig.savefig("plots/06_autocorr_v2.png", dpi=DPI, bbox_inches="tight")
        plt.close(fig)
        print("  Plot: plots/06_autocorr_v2.png")

    # ── Plot 2: n_eff vs n_nominale + inflazione ─────────────────────────────
    if not df_neff.empty:
        n_series_plot = len(df_neff)
        fig2, (ax_top, ax_bot) = plt.subplots(2, 1, figsize=(max(8, n_series_plot * 1.5), 7))

        x_pos   = np.arange(n_series_plot)
        labels  = [
            f"{row['Evento'].split('(')[0].strip()}\n{row['Carburante']}"
            for _, row in df_neff.iterrows()
        ]
        colors_bar = ["#d6604d" if r["Carburante"] == "Benzina" else "#4393c3"
                      for _, r in df_neff.iterrows()]

        # Top: n vs n_eff
        ax_top.bar(x_pos - 0.2, df_neff["n_post"].values, 0.35,
                   label="n nominale", color="steelblue", alpha=0.7)
        ax_top.bar(x_pos + 0.2, df_neff["n_eff"].values,  0.35,
                   label="n_eff (Kish)",  color="darkorange", alpha=0.9)
        ax_top.axhline(5, color="red", lw=1, ls="--", label="n_eff = 5 (soglia cautela)")
        ax_top.set_xticks(x_pos)
        ax_top.set_xticklabels(labels, fontsize=8)
        ax_top.set_ylabel("Osservazioni")
        ax_top.set_title("n nominale vs n_eff (Kish 1965)")
        ax_top.legend(fontsize=8)

        # Bot: inflation factor + colori critici
        bar_cols = [
            "green" if f < 1.5 else "orange" if f < 5 else "red"
            for f in df_neff["inflation_factor"].values
        ]
        ax_bot.bar(x_pos, df_neff["inflation_factor"].values,
                   color=bar_cols, alpha=0.85)
        ax_bot.axhline(1.0, color="black", lw=0.8, ls="--")
        ax_bot.axhline(2.5, color="orange", lw=0.8, ls="--", label="soglia 2.5×")
        ax_bot.axhline(5.0, color="red",    lw=0.8, ls="--", label="soglia 5×")
        for j, (val, row) in enumerate(zip(df_neff["inflation_factor"].values,
                                           df_neff.itertuples())):
            ax_bot.text(j, val + 0.05, f"{val:.1f}×", ha="center", va="bottom", fontsize=8)
        ax_bot.set_xticks(x_pos)
        ax_bot.set_xticklabels(labels, fontsize=8)
        ax_bot.set_ylabel("Inflazione potenza (n/n_eff)")
        ax_bot.set_title("Fattore inflazione potenza — Welch t tratta n obs come n_eff indipendenti")
        ax_bot.legend(fontsize=8)

        fig2.tight_layout()
        fig2.savefig("plots/06_neff_v2.png", dpi=DPI, bbox_inches="tight")
        plt.close(fig2)
        print("  Plot: plots/06_neff_v2.png")


# ─────────────────────────────────────────────────────────────────────────────
# §4. SOMMARIO RACCOMANDAZIONI PER IL PAPER
# ─────────────────────────────────────────────────────────────────────────────

print("\n" + "=" * 70)
print("§4. SOMMARIO RACCOMANDAZIONI PER IL PAPER")
print("=" * 70)

if not df_neff.empty:
    print("\n  RACCOMANDAZIONI n_eff (da includere nel paper come nota metodologica):")
    print(f"  {'Evento + Carburante':<35} {'n':>5} {'ρ̂':>6} {'n_eff':>7} "
          f"{'infl.':>7} {'Andrews_BW':>11} {'valutazione'}")
    print("  " + "─" * 100)
    for _, row in df_neff.iterrows():
        label = f"{row['Evento'].split('(')[0].strip()} — {row['Carburante']}"
        print(
            f"  {label:<35} {row['n_post']:>5} {row['rho_hat_AR1']:>6.3f} "
            f"{row['n_eff']:>7.1f} {row['inflation_factor']:>7.1f}× "
            f"{row['andrews_BW']:>11}  {row['inflazione_nota']}"
        )

    # Avvisi critici
    critical = df_neff[df_neff["n_eff"] < 5]
    if not critical.empty:
        print(f"\n  ⚠  {len(critical)} serie con n_eff < 5:")
        for _, row in critical.iterrows():
            print(
                f"     {row['Evento'].split('(')[0].strip()} — {row['Carburante']}: "
                f"n_eff={row['n_eff']:.1f}  "
                f"(p-value confirmativi da interpretare con estrema cautela)"
            )

    # Andrews BW critico (≥ n//2)
    bw_crit = df_neff[df_neff["andrews_BW"] >= df_neff["n_post"] // 2]
    if not bw_crit.empty:
        print(f"\n  ⚠  {len(bw_crit)} serie dove Andrews BW ≥ n/2:")
        for _, row in bw_crit.iterrows():
            print(
                f"     {row['Evento'].split('(')[0].strip()} — {row['Carburante']}: "
                f"BW={row['andrews_BW']} ≥ {row['n_post']//2} → HAC test quasi non informativo"
            )

print("\n  TEMPLATE testo paper (nota metodologica):")
print("  " + "-" * 65)
if not df_neff.empty:
    for _, row in df_neff.iterrows():
        ev_short = row["Evento"].split("(")[0].strip()
        print(
            f"  [{ev_short} — {row['Carburante']}] "
            f"La serie post-shock presenta ρ̂ AR(1) = {row['rho_hat_AR1']:.3f} "
            f"(DW ≈ {2*(1-row['rho_hat_AR1']):.2f}), riducendo il numero effettivo "
            f"di osservazioni a n_eff ≈ {row['n_eff']:.0f} su n = {row['n_post']} nominali "
            f"(fattore inflazione {row['inflation_factor']:.1f}×). "
            f"Il bandwidth Newey-West ottimale Andrews (1991) è {row['andrews_BW']} settimane."
        )
print()

print("=" * 70)
print("\nScript 06 completato.")
print("  data/distribution_diagnostics_v2.csv — diagnostiche BP/DW/SW")
print("  data/neff_report_v2.csv              — n_eff, Andrews BW, inflazione")
print("  plots/06_autocorr_v2.png             — ACF post-shock")
print("  plots/06_neff_v2.png                 — inflazione potenza")
