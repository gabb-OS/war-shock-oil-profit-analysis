"""
07_margine_speculation_test.py (v4 — TRE METODI comparati)
===========================================================
TEST DI SPECULAZIONE vs ANTICIPAZIONE RAZIONALE
Confronto sistematico di 3 metodologie per il calcolo del margine lordo:

METODO 1: Yield fisso da Brent futures
  margine = prezzo_pompa - (brent_fut_eur/159) × yield
  yield_benzina = 0.45, yield_diesel = 0.52

METODO 2: Crack spread Eurobob (benzina Europa)
  margine_benzina = prezzo_pompa - eurobob_eur_l

METODO 3: Crack spread London Gas Oil (diesel Europa)
  margine_diesel = prezzo_pompa - gasoil_eur_l

Classificazione:
  SPECULAZIONE        → Δmargine > 2σ, p < 0.05, CI esclude 0
  COMPRESSIONE MARGINE→ Δmargine < −2σ, p < 0.05
  ANTICIPAZIONE RAZ.  → Δmargine ≤ 2σ o p ≥ 0.05
  INCONCLUSIVO        → segnali contrastanti

Usage:
  python 07_margine_speculation_test.py --method=yield
  python 07_margine_speculation_test.py --method=crack
  python 07_margine_speculation_test.py --method=all     (default)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.patches as mpatches
import pymc as pm
import pytensor.tensor as pt
from pytensor.scan import scan
from scipy import stats
import warnings
import argparse
import sys
warnings.filterwarnings("ignore")

# ─── Parsing argomenti ───────────────────────────────────────────────────────
parser = argparse.ArgumentParser(description='Test speculazione margini carburanti')
parser.add_argument('--method', type=str, default='all',
                    choices=['yield', 'crack', 'all'],
                    help='Metodo calcolo margine: yield (Brent), crack (Eurobob/GasOil), all (entrambi)')
args = parser.parse_args()

METHOD = args.method

# ─── Configurazione ──────────────────────────────────────────────────────────
DPI          = 180
MCMC_DRAWS   = 2000
MCMC_TUNE    = 1000
MCMC_CHAINS  = 2
ALPHA        = 0.05

BARREL_LITRES  = 159.0
YIELD_GASOLINE = 0.45
YIELD_DIESEL   = 0.52

# Densità (kg/L) per conversione tonnellata → litri
# Fonte: Engineering ToolBox, petroleum industry standards
DENSITY_GASOLINE_KG_L = 0.74   # benzina ≈ 0.72–0.76 kg/L
DENSITY_DIESEL_KG_L   = 0.84   # diesel  ≈ 0.82–0.85 kg/L

# Tonnellata = 1000 kg → litri = 1000 / densità
LITERS_PER_TONNE_GASOLINE = 1000.0 / DENSITY_GASOLINE_KG_L  # ≈ 1351 L
LITERS_PER_TONNE_DIESEL   = 1000.0 / DENSITY_DIESEL_KG_L    # ≈ 1190 L

BASELINE_START = "2021-01-01"
BASELINE_END   = "2021-12-31"
EDGE_FRACTION  = 0.15

EVENTS = {
    "Ucraina (Feb 2022)": {
        "shock_date":   "2022-02-24",
        "window_start": "2021-10-01",
        "window_end":   "2022-07-31",
        "color":        "#e74c3c",
    },
    "Iran-Israele (Giu 2025)": {
        "shock_date":   "2025-06-13",
        "window_start": "2025-02-01",
        "window_end":   "2025-10-31",
        "color":        "#e67e22",
    },
    "Hormuz (Feb 2026)": {
        "shock_date":   "2026-02-28",
        "window_start": "2025-10-01",
        "window_end":   "2026-03-17",
        "color":        "#8e44ad",
    },
}

CLAS_COLOR = {
    "SPECULAZIONE":                       "#c0392b",
    "COMPRESSIONE MARGINE":               "#2980b9",
    "ANTICIPAZIONE RAZIONALE / NEUTRO":   "#27ae60",
    "VARIAZIONE STATISTICA (non anomala)":"#e67e22",
    "INCONCLUSIVO":                       "#95a5a6",
}

print("\n" + "="*80)
print(f"SCRIPT 07 — Metodo: {METHOD.upper()}")
print("="*80)

# ─────────────────────────────────────────────────────────────────────────────
# 1. CARICA DATASET PREZZI POMPA
# ─────────────────────────────────────────────────────────────────────────────
print("\n[1/6] Caricamento prezzi pompa...")
merged = pd.read_csv("data/dataset_merged.csv", index_col=0, parse_dates=True)

# Rilevamento automatico unità (EU Bulletin può pubblicare in EUR/1000L)
pump_sample_benz = merged["benzina_4w"].dropna().mean()
pump_sample_dies = merged["diesel_4w"].dropna().mean()

unit_factor_benz = 1000.0 if pump_sample_benz > 10 else 1.0
unit_factor_dies = 1000.0 if pump_sample_dies > 10 else 1.0

merged["benzina_eur_l"] = merged["benzina_4w"] / unit_factor_benz
merged["diesel_eur_l"]  = merged["diesel_4w"]  / unit_factor_dies

print(f"  Benzina: media {pump_sample_benz:.2f} → "
      f"unità {'EUR/1000L (conv. a EUR/L)' if unit_factor_benz > 1 else 'EUR/L (già corretto)'}")
print(f"  Diesel:  media {pump_sample_dies:.2f} → "
      f"unità {'EUR/1000L (conv. a EUR/L)' if unit_factor_dies > 1 else 'EUR/L (già corretto)'}")
print(f"  Post-conversione: Benzina {merged['benzina_eur_l'].mean():.4f} EUR/L, "
      f"Diesel {merged['diesel_eur_l'].mean():.4f} EUR/L")

# ─────────────────────────────────────────────────────────────────────────────
# 2. CARICA EUR/USD
# ─────────────────────────────────────────────────────────────────────────────
print("\n[2/6] Caricamento EUR/USD...")
if "eurusd" in merged.columns and merged["eurusd"].dropna().mean() > 0.5:
    print("  EUR/USD già presente in dataset_merged")
else:
    print("  EUR/USD non trovato → uso fallback 1.08")
    merged["eurusd"] = 1.08

merged["eurusd"] = merged["eurusd"].ffill().bfill()
print(f"  EUR/USD medio: {merged['eurusd'].mean():.4f}")

# ─────────────────────────────────────────────────────────────────────────────
# 3. CARICA FUTURES DA CSV
# ─────────────────────────────────────────────────────────────────────────────
print("\n[3/6] Caricamento futures da CSV...")

def load_investing_csv(filepath, price_col="Price"):
    """
    Carica CSV da Investing.com con formato:
      Date,Price,Open,High,Low,Vol.,Change %
    Gestisce:
      - BOM UTF-8 (﻿)
      - Virgole nelle migliaia ("1,234.56")
      - Date formato MM/DD/YYYY
    """
    df = pd.read_csv(filepath, thousands=',')
    # Rimuovi BOM se presente nella prima colonna
    df.columns = [col.lstrip('\ufeff') for col in df.columns]
    df['Date'] = pd.to_datetime(df['Date'], format='%m/%d/%Y', errors='coerce')
    df = df.dropna(subset=['Date'])
    df.set_index('Date', inplace=True)
    df.sort_index(inplace=True)
    # Converti Price in numerico (gestisce "1,234.56" → 1234.56)
    if price_col in df.columns:
        df[price_col] = pd.to_numeric(df[price_col].astype(str).str.replace(',', ''), 
                                       errors='coerce')
    return df

# ── Brent Crude Oil (USD/barile)
try:
    brent_csv = load_investing_csv("data/Brent Oil Futures Historical Data.csv")
    brent_csv.rename(columns={"Price": "brent_fut_usd_bbl"}, inplace=True)
    print(f"  Brent: {len(brent_csv)} righe, "
          f"{brent_csv.index[0].date()} → {brent_csv.index[-1].date()}, "
          f"media ${brent_csv['brent_fut_usd_bbl'].mean():.2f}/bbl")
except Exception as e:
    print(f"  Errore caricamento Brent CSV: {e}")
    brent_csv = None

# ── Eurobob Gasoline (USD/tonnellata)
try:
    eurobob_csv = load_investing_csv("data/Eurobob Futures Historical Data.csv")
    eurobob_csv.rename(columns={"Price": "eurobob_usd_tonne"}, inplace=True)
    print(f"  Eurobob: {len(eurobob_csv)} righe, "
          f"{eurobob_csv.index[0].date()} → {eurobob_csv.index[-1].date()}, "
          f"media ${eurobob_csv['eurobob_usd_tonne'].mean():.2f}/tonne")
except Exception as e:
    print(f"  Errore caricamento Eurobob CSV: {e}")
    eurobob_csv = None

# ── London Gas Oil (USD/tonnellata)
try:
    gasoil_csv = load_investing_csv("data/London Gas Oil Futures Historical Data.csv")
    gasoil_csv.rename(columns={"Price": "gasoil_usd_tonne"}, inplace=True)
    print(f"  Gas Oil: {len(gasoil_csv)} righe, "
          f"{gasoil_csv.index[0].date()} → {gasoil_csv.index[-1].date()}, "
          f"media ${gasoil_csv['gasoil_usd_tonne'].mean():.2f}/tonne")
except Exception as e:
    print(f"  Errore caricamento Gas Oil CSV: {e}")
    gasoil_csv = None

# ─────────────────────────────────────────────────────────────────────────────
# 4. RESAMPLE A SETTIMANALE E MERGE
# ─────────────────────────────────────────────────────────────────────────────
print("\n[4/6] Resample settimanale e merge...")

if brent_csv is not None:
    brent_w = brent_csv[["brent_fut_usd_bbl"]].resample("W-MON").mean()
    merged = merged.join(brent_w, how="left")
    merged["brent_fut_usd_bbl"] = merged["brent_fut_usd_bbl"].ffill()
    # USD/barile → EUR/litro
    merged["brent_fut_eur_l"] = (merged["brent_fut_usd_bbl"] / merged["eurusd"]) / BARREL_LITRES
    print(f"  Brent EUR/L: media {merged['brent_fut_eur_l'].dropna().mean():.4f} EUR/L "
          f"(atteso ~0.4–0.8)")
else:
    merged["brent_fut_eur_l"] = np.nan
    print("  Brent futures non disponibile")

if eurobob_csv is not None:
    eurobob_w = eurobob_csv[["eurobob_usd_tonne"]].resample("W-MON").mean()
    merged = merged.join(eurobob_w, how="left")
    merged["eurobob_usd_tonne"] = merged["eurobob_usd_tonne"].ffill()
    # USD/tonnellata → EUR/litro
    merged["eurobob_eur_l"] = (merged["eurobob_usd_tonne"] / merged["eurusd"]) / LITERS_PER_TONNE_GASOLINE
    print(f"  Eurobob EUR/L: media {merged['eurobob_eur_l'].dropna().mean():.4f} EUR/L "
          f"(atteso ~0.4–0.9)")
else:
    merged["eurobob_eur_l"] = np.nan
    print("  Eurobob futures non disponibile")

if gasoil_csv is not None:
    gasoil_w = gasoil_csv[["gasoil_usd_tonne"]].resample("W-MON").mean()
    merged = merged.join(gasoil_w, how="left")
    merged["gasoil_usd_tonne"] = merged["gasoil_usd_tonne"].ffill()
    # USD/tonnellata → EUR/litro
    merged["gasoil_eur_l"] = (merged["gasoil_usd_tonne"] / merged["eurusd"]) / LITERS_PER_TONNE_DIESEL
    print(f"  Gas Oil EUR/L: media {merged['gasoil_eur_l'].dropna().mean():.4f} EUR/L "
          f"(atteso ~0.4–0.9)")
else:
    merged["gasoil_eur_l"] = np.nan
    print("  Gas Oil futures non disponibile")

# ─────────────────────────────────────────────────────────────────────────────
# 5. CALCOLA MARGINI (3 METODI)
# ─────────────────────────────────────────────────────────────────────────────
print("\n[5/6] Calcolo margini...")

# ── METODO 1: Yield fisso (Brent)
if not merged["brent_fut_eur_l"].isna().all():
    merged["margine_benzina_yield"] = (merged["benzina_eur_l"] 
                                        - merged["brent_fut_eur_l"] * YIELD_GASOLINE)
    merged["margine_diesel_yield"]  = (merged["diesel_eur_l"]
                                        - merged["brent_fut_eur_l"] * YIELD_DIESEL)
    print(f"  YIELD: Benzina {merged['margine_benzina_yield'].dropna().mean():.4f} EUR/L, "
          f"Diesel {merged['margine_diesel_yield'].dropna().mean():.4f} EUR/L")
else:
    merged["margine_benzina_yield"] = np.nan
    merged["margine_diesel_yield"]  = np.nan
    print("  YIELD: non disponibile (Brent mancante)")

# ── METODO 2: Crack spread Eurobob (benzina)
if not merged["eurobob_eur_l"].isna().all():
    merged["margine_benzina_crack"] = merged["benzina_eur_l"] - merged["eurobob_eur_l"]
    print(f"  CRACK EUROBOB: Benzina {merged['margine_benzina_crack'].dropna().mean():.4f} EUR/L")
else:
    merged["margine_benzina_crack"] = np.nan
    print("  CRACK EUROBOB: non disponibile")

# ── METODO 3: Crack spread Gas Oil (diesel)
if not merged["gasoil_eur_l"].isna().all():
    merged["margine_diesel_crack"] = merged["diesel_eur_l"] - merged["gasoil_eur_l"]
    print(f"  CRACK GAS OIL: Diesel {merged['margine_diesel_crack'].dropna().mean():.4f} EUR/L")
else:
    merged["margine_diesel_crack"] = np.nan
    print("  CRACK GAS OIL: non disponibile")

# Salva dataset con tutti i margini
merged.to_csv("data/dataset_merged_with_futures_all.csv")
print(f"\n  Salvato: data/dataset_merged_with_futures_all.csv")

# ─────────────────────────────────────────────────────────────────────────────
# 6. SELEZIONA METODO DA USARE
# ─────────────────────────────────────────────────────────────────────────────
print(f"\n[6/6] Selezione metodo: {METHOD}")

if METHOD == "yield":
    SERIES_TO_TEST = [
        ("Benzina", "margine_benzina_yield", "Yield fisso (Brent × 0.45)"),
        ("Diesel",  "margine_diesel_yield",  "Yield fisso (Brent × 0.52)"),
    ]
    output_suffix = "yield"
elif METHOD == "crack":
    SERIES_TO_TEST = [
        ("Benzina", "margine_benzina_crack", "Crack spread (Eurobob)"),
        ("Diesel",  "margine_diesel_crack",  "Crack spread (Gas Oil)"),
    ]
    output_suffix = "crack"
else:  # all
    SERIES_TO_TEST = [
        ("Benzina", "margine_benzina_yield", "Yield fisso (Brent × 0.45)"),
        ("Benzina", "margine_benzina_crack", "Crack spread (Eurobob)"),
        ("Diesel",  "margine_diesel_yield",  "Yield fisso (Brent × 0.52)"),
        ("Diesel",  "margine_diesel_crack",  "Crack spread (Gas Oil)"),
    ]
    output_suffix = "all"

# Filtra solo serie disponibili
SERIES_TO_TEST = [(s, col, desc) for s, col, desc in SERIES_TO_TEST 
                  if col in merged.columns and not merged[col].isna().all()]

if not SERIES_TO_TEST:
    print("  ERRORE: nessuna serie disponibile per il metodo selezionato")
    sys.exit(1)

print(f"  Serie da testare: {len(SERIES_TO_TEST)}")
for s, col, desc in SERIES_TO_TEST:
    n_valid = merged[col].dropna().shape[0]
    print(f"    {s:8} | {desc:35} | {n_valid:4} obs")

# ─────────────────────────────────────────────────────────────────────────────
# 7. SOGLIE BASELINE (2σ su 2021)
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "─"*80)
print("SOGLIE BASELINE (2σ su 2021)")
print("─"*80)

baseline_thresholds = {}
baseline_means      = {}

for series_name, margine_col, desc in SERIES_TO_TEST:
    key = f"{series_name}_{margine_col}"
    b   = merged.loc[BASELINE_START:BASELINE_END, margine_col].dropna()
    if len(b) < 8:
        baseline_thresholds[key] = 0.030
        baseline_means[key]      = float(b.mean()) if len(b) > 0 else np.nan
        print(f"  {key:40}: baseline insufficiente → fallback 0.030 EUR/L")
    else:
        thr = 2.0 * float(b.std())
        baseline_thresholds[key] = thr
        baseline_means[key]      = float(b.mean())
        print(f"  {key:40}: μ={b.mean():.4f}, σ={b.std():.4f} → 2σ={thr:.4f} EUR/L")

# ─────────────────────────────────────────────────────────────────────────────
# 8. BAYESIAN CHANGEPOINT (StudentT + AR(1))
# ─────────────────────────────────────────────────────────────────────────────
def bayesian_changepoint(x_vals, y_vals, alpha=0.05):
    n, sd_y, rng_x = len(x_vals), float(np.std(y_vals)), float(x_vals[-1]-x_vals[0])
    with pm.Model():
        tau   = pm.Uniform("tau",   lower=x_vals[0], upper=x_vals[-1])
        sigma = pm.HalfNormal("sigma", sigma=sd_y)
        nu    = pm.Exponential("nu", lam=1/30)
        rho   = pm.Uniform("rho",   lower=-1, upper=1)
        b1    = pm.StudentT("b1", mu=0, sigma=3*sd_y, nu=3)
        b2    = pm.StudentT("b2", mu=0, sigma=3*sd_y, nu=3)
        a1    = pm.StudentT("a1", mu=0, sigma=sd_y/max(rng_x,1), nu=3)
        a2    = pm.Deterministic("a2", a1 + tau*(b1-b2))
        x_pt  = pt.as_tensor_variable(x_vals.astype(float))
        step  = pm.math.sigmoid((x_pt - tau) * 50)
        mu    = (a1 + b1*x_pt)*(1-step) + (a2 + b2*x_pt)*step
        eps_init = pm.Normal("eps_init", mu=0, sigma=sigma)
        eta      = pm.Normal("eta", mu=0, sigma=sigma, shape=n-1)
        eps_rest, _ = scan(
            fn=lambda eta_t, eps_prev, rho_v: rho_v*eps_prev + eta_t,
            sequences=[eta], outputs_info=[eps_init], non_sequences=[rho],
        )
        eps = pt.concatenate([[eps_init], eps_rest])
        pm.StudentT("obs", nu=nu, mu=mu+eps, sigma=sigma, observed=y_vals)
        trace = pm.sample(
            draws=MCMC_DRAWS, tune=MCMC_TUNE, chains=MCMC_CHAINS,
            progressbar=False, random_seed=42, target_accept=0.9,
            return_inferencedata=True,
        )
    tau_post = trace.posterior["tau"].values.flatten()
    lo, hi   = (alpha/2)*100, (1-alpha/2)*100
    return {
        "tau_mean": float(np.mean(tau_post)),
        "tau_lo":   float(np.percentile(tau_post, lo)),
        "tau_hi":   float(np.percentile(tau_post, hi)),
        "tau_idx":  int(np.clip(round(float(np.median(tau_post))), 1, n-2)),
        "tau_post": tau_post,
        "nu_mean":  float(np.mean(trace.posterior["nu"].values.flatten())),
        "rho_mean": float(np.mean(trace.posterior["rho"].values.flatten())),
    }

def is_edge(idx, n, frac=EDGE_FRACTION):
    return (idx < int(n * frac)) or (idx > int(n * (1 - frac)))

def bootstrap_delta(pre_vals, post_vals, n_boot=1000, seed=42):
    rng = np.random.default_rng(seed)
    deltas = [rng.choice(post_vals, len(post_vals), replace=True).mean() -
              rng.choice(pre_vals,  len(pre_vals),  replace=True).mean()
              for _ in range(n_boot)]
    arr = np.array(deltas)
    return float(np.mean(arr)), float(np.percentile(arr, 2.5)), float(np.percentile(arr, 97.5))

# ─────────────────────────────────────────────────────────────────────────────
# 9. RUN TEST PER OGNI SERIE
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "="*80)
print("TEST SPECULAZIONE vs ANTICIPAZIONE RAZIONALE")
print("="*80)

results = []

for event_name, cfg in EVENTS.items():
    shock = pd.Timestamp(cfg["shock_date"])
    
    for series_name, margine_col, desc_method in SERIES_TO_TEST:
        key = f"{series_name}_{margine_col}"
        
        df = merged.loc[cfg["window_start"]:cfg["window_end"]].copy()
        df = df.dropna(subset=[margine_col])
        if len(df) < 10:
            print(f"  SKIP {event_name}|{series_name}|{desc_method}: <10 obs")
            continue
        
        # Split pre/post sulla data shock
        shock_idx = int(np.clip(df.index.searchsorted(shock), 2, len(df)-2))
        pre_m  = df.iloc[:shock_idx][margine_col].dropna()
        post_m = df.iloc[shock_idx:][margine_col].dropna()
        
        if len(pre_m) < 2 or len(post_m) < 2:
            print(f"  SKIP {event_name}|{series_name}|{desc_method}: gruppo pre/post <2")
            continue
        
        # Test statistici
        t_stat, t_p   = stats.ttest_ind(post_m.values, pre_m.values, equal_var=False)
        ks_stat, ks_p = stats.ks_2samp(pre_m.values, post_m.values)
        delta_mean    = float(post_m.mean() - pre_m.mean())
        boot_mean, boot_lo, boot_hi = bootstrap_delta(pre_m.values, post_m.values)
        
        # MCMC changepoint sul margine
        print(f"\n  MCMC → {event_name} | {series_name} | {desc_method}")
        x_vals    = np.arange(len(df), dtype=float)
        y_margine = df[margine_col].values.astype(float)
        ci_m      = bayesian_changepoint(x_vals, y_margine)
        
        cp_idx   = ci_m["tau_idx"]
        cp_date  = df.index[cp_idx]
        no_break = is_edge(cp_idx, len(df))
        lag_vs_shock = (cp_date - shock).days
        
        # Classificazione
        soglia = baseline_thresholds.get(key, 0.030)
        anomalo = abs(delta_mean) > soglia
        
        t_sig  = (not np.isnan(t_p)) and (t_p < ALPHA)
        ks_sig = ks_p < ALPHA
        ci_non_zero = (boot_lo > 0) or (boot_hi < 0)
        stat_sig = (t_sig or ks_sig) and ci_non_zero
        
        if stat_sig and anomalo and delta_mean > 0:
            clas = "SPECULAZIONE"
        elif stat_sig and anomalo and delta_mean < 0:
            clas = "COMPRESSIONE MARGINE"
        elif not anomalo:
            clas = "ANTICIPAZIONE RAZIONALE / NEUTRO"
        elif stat_sig and not anomalo:
            clas = "VARIAZIONE STATISTICA (non anomala)"
        else:
            clas = "INCONCLUSIVO"
        
        print(f"    Δ={delta_mean:+.5f} EUR/L [CI: {boot_lo:+.5f},{boot_hi:+.5f}]")
        print(f"    Soglia 2σ={soglia:.5f} | KS p={ks_p:.4f} | τ={cp_date.date()} "
              f"(lag {lag_vs_shock:+d}gg) | break={not no_break}")
        print(f"    → {clas}")
        
        results.append({
            "Evento":             event_name,
            "Serie":              series_name,
            "Metodo":             desc_method,
            "n_pre":              len(pre_m),
            "n_post":             len(post_m),
            "delta_margine_eur":  round(delta_mean, 5),
            "boot_CI_lo":         round(boot_lo, 5),
            "boot_CI_hi":         round(boot_hi, 5),
            "soglia_2sigma":      round(soglia, 5),
            "delta_anomalo":      anomalo,
            "t_p":                round(float(t_p), 4) if not np.isnan(t_p) else "nan",
            "ks_p":               round(ks_p, 4),
            "tau_margine":        cp_date.date(),
            "lag_tau_vs_shock":   lag_vs_shock,
            "break_strutturale":  not no_break,
            "nu_StudentT":        round(ci_m["nu_mean"], 2),
            "rho_AR1":            round(ci_m["rho_mean"], 3),
            "classificazione":    clas,
        })

# ─────────────────────────────────────────────────────────────────────────────
# 10. SALVA RISULTATI
# ─────────────────────────────────────────────────────────────────────────────
csv_out = f"data/table2_margini_anomaly_{output_suffix}.csv"
pd.DataFrame(results).to_csv(csv_out, index=False)
print(f"\n  Salvato: {csv_out} ({len(results)} righe)")

# ─────────────────────────────────────────────────────────────────────────────
# 11. PLOT RIASSUNTIVO
# ─────────────────────────────────────────────────────────────────────────────
if results:
    df_r = pd.DataFrame(results)
    n_rows = len(df_r)
    
    fig_s, ax_s = plt.subplots(figsize=(14, max(5, n_rows * 0.8)))
    fig_s.suptitle(
        f"Variazione margine lordo post-shock — Metodo: {METHOD.upper()}\n"
        f"Barre = Δ EUR/L  |  Whisker = Bootstrap CI 95%",
        fontsize=12, fontweight="bold",
    )
    
    labels = [f"{r['Evento'].split('(')[0].strip()}\n{r['Serie']}\n{r['Metodo'][:20]}"
              for _, r in df_r.iterrows()]
    deltas = df_r["delta_margine_eur"].values
    ci_lo  = df_r["boot_CI_lo"].values
    ci_hi  = df_r["boot_CI_hi"].values
    colors = [CLAS_COLOR.get(c, "#555555") for c in df_r["classificazione"]]
    
    bars = ax_s.barh(range(n_rows), deltas, color=colors,
                     alpha=0.75, edgecolor="black", lw=0.7)
    for i in range(n_rows):
        ax_s.errorbar(deltas[i], i,
                      xerr=[[deltas[i]-ci_lo[i]], [ci_hi[i]-deltas[i]]],
                      fmt="none", color="black", capsize=6, lw=1.8)
        ax_s.text(max(ci_hi[i], deltas[i]) + 0.003, i,
                  f"{df_r.iloc[i]['classificazione'][:25]}",
                  va="center", fontsize=8)
    
    ax_s.axvline(0, color="black", lw=0.8)
    ax_s.set_yticks(range(n_rows))
    ax_s.set_yticklabels(labels, fontsize=8)
    ax_s.set_xlabel("Δ margine lordo post-shock (EUR/litro)", fontsize=11)
    ax_s.grid(alpha=0.3, axis="x")
    ax_s.legend(handles=[
        mpatches.Patch(color=c, label=k) for k, c in CLAS_COLOR.items()
    ], fontsize=8, loc="lower right")
    
    plt.tight_layout(pad=1.5)
    plt.savefig(f"plots/07_summary_{output_suffix}.png", dpi=DPI, bbox_inches="tight")
    plt.close()
    print(f"  Salvato: plots/07_summary_{output_suffix}.png")

# ─────────────────────────────────────────────────────────────────────────────
# 12. SOMMARIO FINALE
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "="*80)
print(f"SOMMARIO — Metodo: {METHOD.upper()}")
print("="*80)

for res in results:
    print(f"\n  {res['Evento']} | {res['Serie']} | {res['Metodo']}")
    print(f"    Δ={res['delta_margine_eur']:+.5f} EUR/L "
          f"[CI: {res['boot_CI_lo']:+.5f},{res['boot_CI_hi']:+.5f}]")
    print(f"    Soglia 2σ={res['soglia_2sigma']:.5f} | KS p={res['ks_p']} | "
          f"τ={res['tau_margine']} ({res['lag_tau_vs_shock']:+d}gg)")
    print(f"    → {res['classificazione']}")

print("\n  Output:")
print(f"    {csv_out}")
print(f"    plots/07_summary_{output_suffix}.png")

print("\n" + "="*80)
print("Script 07 completato.")
print("="*80)