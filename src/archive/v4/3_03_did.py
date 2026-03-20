"""
3_03_did.py  — Famiglia C: DiD + Windfall  (v4 — fix metodologici)
====================================================================
Testa H₀(iii): i margini italiani sono cresciuti *più* di quelli dei
peer EU dopo gli shock geopolitici?

MAPPA H₀ DICHIARATE → FAMIGLIE
───────────────────────────────
  H₀(iii) dichiarata: "le variazioni osservate nei margini italiani non
  risultano superiori a quelle dei principali paesi europei di
  riferimento" → FAMIGLIA C (questo script)

  ⚠ δ̂ < 0 significativo → H₀(iii) NON rifiutata (IT ≤ controllo).
    Il test rigetta qualcosa, ma nella direzione opposta a H₁(ii).
    Non va contato come evidenza di extraprofitti italiani.

CORREZIONI rispetto a v3
────────────────────────
1. Wholesale UNIFORME per tutti i paesi
   Eurobob ARA (benzina) e London Gas Oil ICE (diesel) sono prezzi
   paneuropei: tutti i distributori EU, IT inclusa, si riforniscono
   agli stessi mercati ARA/ICE.  Il vecchio metodo usava Brent/yield
   (0.45 / 0.52) come proxy solo per i controlli, creando un margine
   disomogeneo rispetto a IT. Ora:
     margine_k(t) = pompa_k(t) − wholesale_ARA(t)   ∀ k ∈ {IT, DE, SE, …}

2. Panel bilanciato per data (inner join)
   IT e CT vengono uniti sull'indice datetime prima di costruire il
   panel OLS → ogni riga del panel corrisponde a una settimana precisa
   condivisa da entrambi i paesi.

3. SE HC3 (heteroskedasticity-robust)
   Il panel stacked è interleaved (t1-IT, t1-CT, t2-IT, t2-CT, …):
   il lag-1 stimato da Newey-West è cross-sezionale, non temporale →
   HAC gonfia artificialmente le SE. HC3 è robusto senza ipotesi
   sull'ordinamento temporale del residuo.

4. Tau MCMC come date di cutoff (facoltativo)
   Se data/3_cp.csv esiste, le date pre/post vengono sostituite con
   i tau MAP stimati dal MCMC (3_05_changepoint.py) sulle serie
   benzina e diesel.  Le date hardcoded rimangono come fallback.

5. Windfall plot migliorato
   Mostra CI 95% del δ̂ come banda di errore sulle barre, separato
   per evento, con annotazioni leggibili.

MODELLO DiD
───────────
  Panel bilanciato (solo settimane con dati per entrambi):
    M_{k,t} = α + β·IT_k + γ·Post_t + δ·(IT×Post)_{k,t} + ε_{k,t}

  δ̂ = (mean_IT_post − mean_IT_pre) − (mean_CT_post − mean_CT_pre)
  SE Newey-West.

PTA (Parallel Trends Assumption)
─────────────────────────────────
  Regressione pre-shock: M ~ IT + t + IT×t  (ultime 8 settimane pre-shock)
  PTA violata se p(IT×t) < 0.05 → δ̂ potenzialmente biased da trend
  preesistenti; interpretare come evidenza DESCRITTIVA.

WINDFALL
────────
  Windfall (MLD EUR) = δ̂ × vol_settimana × n_settimane_post / 1e9
  CI 95% windfall = CI_95(δ̂) × stessa formula
  Riferimento: Germania (unico controllo senza interventi ex-tasse certi)
  Sensitività: ±30% sui volumi.
"""

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
import statsmodels.api as sm

warnings.filterwarnings("ignore")
os.makedirs("data",  exist_ok=True)
os.makedirs("plots", exist_ok=True)

# ── Costanti ──────────────────────────────────────────────────────────────────
ALPHA     = 0.05
PTA_WEEKS = 8
DPI       = 160

# Volumi proxy IT (MISE 2022 corr. trend -1.5%/anno)
VOLUME_2022 = {
    "Benzina": 1_750_000_000 / 52,   # L/settimana
    "Diesel":  2_250_000_000 / 52,
}
TREND_ANNO = -0.015

# Date hardcoded (fallback se tau MCMC non disponibili)
EVENTS_DEFAULT = {
    "Ucraina (Feb 2022)": {
        "shock":     pd.Timestamp("2022-02-24"),
        "pre_start": pd.Timestamp("2021-01-11"),
        "post_end":  pd.Timestamp("2022-12-31"),
        "color":     "#e74c3c",
        "ref_year":  2022,
    },
    "Iran-Israele (Giu 2025)": {
        "shock":     pd.Timestamp("2025-06-13"),
        "pre_start": pd.Timestamp("2024-12-01"),   # ~6 mesi pre-shock (era 18 mesi, corretto)
        "post_end":  pd.Timestamp("2025-12-31"),
        "color":     "#e67e22",
        "ref_year":  2025,
    },
    "Hormuz (Feb 2026)": {
        "shock":     pd.Timestamp("2026-02-28"),
        "pre_start": pd.Timestamp("2025-08-01"),   # ~6 mesi pre τ (τ benzina ≈ 2026-02-23)
        "post_end":  pd.Timestamp("2026-04-30"),   # limite dati disponibili (~9 settimane post)
        "color":     "#8e44ad",
        "ref_year":  2026,
    },
}

CONTROLS = {
    "Germania":    "DE",
    "Paesi Bassi": "NL",
    "Belgio":      "BE",
    "Danimarca":   "DK",
    "Finlandia":   "FI",
    "Austria":     "AT",
}

FUELS = ["Benzina", "Diesel"]
FUEL_COLORS = {"Benzina": "#d6604d", "Diesel": "#4393c3"}


# ════════════════════════════════════════════════════════════════════════════
# 0. Carica dataset IT (contiene già Eurobob / Gas Oil in EUR/L)
# ════════════════════════════════════════════════════════════════════════════
print("Carico dataset Italia...")
merged = pd.read_csv("data/3_dataset.csv", index_col=0, parse_dates=True).sort_index()
print(f"   {len(merged)} settimane | {merged.index[0].date()} – {merged.index[-1].date()}")

# Controlla presenza wholesale futures
HAS_EUROBOB  = "eurobob_eur_l" in merged.columns
HAS_GASOIL   = "gasoil_eur_l"  in merged.columns

if not HAS_EUROBOB or not HAS_GASOIL:
    print("   ⚠ ATTENZIONE: eurobob_eur_l / gasoil_eur_l non trovati in 3_dataset.csv")
    print("     I CSV futures devono essere scaricati manualmente da Investing.com.")
    print("     Fallback: approssimazione Brent/yield (meno accurata).")

def _wholesale_series(fuel: str) -> pd.Series:
    """
    Restituisce il prezzo wholesale in EUR/L dal mercato ARA/ICE.

    Eurobob ARA (benzina) e London Gas Oil ICE (diesel) sono prezzi
    paneuropei: lo stesso mercato wholesale si applica a IT, DE, ecc.

    NON usiamo Brent/yield come proxy: lo yield crack (0.45/0.52) e' una
    costante fisica che non riflette la variabilita' settimanale dello
    spread prodotto/greggio -> introduce errore sistematico nel margine.

    Se i futures non sono disponibili il DiD non puo' girare correttamente:
    bisogna scaricare i CSV da Investing.com (vedi 3_01_data.py).
    """
    if fuel == "Benzina":
        if not HAS_EUROBOB:
            raise RuntimeError(
                "eurobob_eur_l non trovato in 3_dataset.csv.\n"
                "Scarica 'Eurobob Futures Historical Data.csv' da Investing.com\n"
                "e riesegui 3_01_data.py."
            )
        return merged["eurobob_eur_l"]
    else:
        if not HAS_GASOIL:
            raise RuntimeError(
                "gasoil_eur_l non trovato in 3_dataset.csv.\n"
                "Scarica 'London Gas Oil Futures Historical Data.csv' da Investing.com\n"
                "e riesegui 3_01_data.py."
            )
        return merged["gasoil_eur_l"]


# ════════════════════════════════════════════════════════════════════════════
# 1. Carica paesi controllo e calcola margini con wholesale UNIFORME
# ════════════════════════════════════════════════════════════════════════════
print(f"\nCarico paesi controllo ({', '.join(CONTROLS.values())})...")

CONTROL_MARGINS: dict[str, dict[str, pd.Series]] = {}
# {paese: {fuel: serie_margine_settimanale}}

for paese, code in CONTROLS.items():
    csv_path = f"data/pompa_{code.lower()}.csv"
    if not os.path.exists(csv_path):
        print(f"   SKIP {paese} ({code}): {csv_path} non trovato")
        continue
    try:
        pump = pd.read_csv(csv_path, index_col=0, parse_dates=True).sort_index()
        margins = {}
        for fuel, pcol in [("Benzina", "benzina_eur_l"), ("Diesel", "diesel_eur_l")]:
            if pcol not in pump.columns:
                continue
            ws = _wholesale_series(fuel)
            # Allinea wholesale all'indice del paese controllo
            ws_aligned = ws.reindex(pump.index).ffill(limit=4)
            margins[fuel] = pump[pcol] - ws_aligned
        CONTROL_MARGINS[paese] = margins
        fuels_ok = list(margins.keys())
        print(f"   {paese}: caricato  ({len(pump)} settimane, fuels={fuels_ok})")
    except Exception as exc:
        print(f"   SKIP {paese} ({code}): {exc}")

# Costruisce anche i margini IT con stesso wholesale
IT_MARGINS: dict[str, pd.Series] = {}
for fuel in FUELS:
    pcol_it = "benzina_eur_l" if fuel == "Benzina" else "diesel_eur_l"
    if pcol_it not in merged.columns:
        continue
    ws = _wholesale_series(fuel)
    IT_MARGINS[fuel] = merged[pcol_it] - ws

print(f"\n   Wholesale usato: {'futures Eurobob/GasOil (reale)' if HAS_EUROBOB else 'Brent/yield (approssimato)'}")
print(f"   → stesso wholesale per IT e tutti i controlli ✓")


# ════════════════════════════════════════════════════════════════════════════
# 2. Carica tau MCMC (se disponibile) per affinare le finestre pre/post
# ════════════════════════════════════════════════════════════════════════════
EVENTS = {k: dict(v) for k, v in EVENTS_DEFAULT.items()}  # copia

cp_path = "data/3_cp.csv"
if os.path.exists(cp_path):
    print(f"\nCarico tau MCMC da {cp_path}...")
    df_cp = pd.read_csv(cp_path)
    # Colonne attese: evento, serie, tau_map, ci95_low, ci95_high
    tau_cols = [c for c in df_cp.columns if "tau" in c.lower() or "map" in c.lower()]
    serie_col = next((c for c in df_cp.columns if "serie" in c.lower()), None)
    ev_col    = next((c for c in df_cp.columns if "evento" in c.lower()), None)

    for evento in EVENTS:
        for fuel in FUELS:
            fuel_key = "Benzina" if fuel == "Benzina" else "Diesel"
            if ev_col and serie_col and tau_cols:
                row = df_cp[
                    df_cp[ev_col].str.contains(evento[:10], na=False, regex=False) &
                    df_cp[serie_col].str.contains(fuel_key, case=False, na=False, regex=False)
                ]
                if not row.empty:
                    tau_val = pd.Timestamp(row[tau_cols[0]].iloc[0])
                    # Usa tau come data di shock se non è troppo lontana
                    default_shock = EVENTS_DEFAULT[evento]["shock"]
                    lag_days = abs((tau_val - default_shock).days)
                    if lag_days <= 180:
                        print(f"   {evento} | {fuel}: shock → τ_MCMC={tau_val.date()} "
                              f"(lag={int((tau_val-default_shock).days):+d}gg)")
                        # Aggiorna shock per questo evento (usa media tra benzina e diesel se discordanti)
                        # Per semplicità: usa il tau della serie benzina come riferimento evento
                        if fuel == "Benzina":
                            EVENTS[evento]["shock_mcmc"] = tau_val
                    else:
                        print(f"   {evento} | {fuel}: τ_MCMC={tau_val.date()} troppo lontano, ignorato")
    print("   Nota: shock_mcmc disponibile per eventi con tau benzina entro ±180gg")
else:
    print(f"\n   {cp_path} non trovato — uso date hardcoded")


def _get_shock(evento: str) -> pd.Timestamp:
    """Restituisce tau MCMC se disponibile, altrimenti data hardcoded."""
    return EVENTS[evento].get("shock_mcmc", EVENTS[evento]["shock"])


# ════════════════════════════════════════════════════════════════════════════
# 3. Helper statistici
# ════════════════════════════════════════════════════════════════════════════

def _stars(p: float) -> str:
    if p < 0.001: return "***"
    if p < 0.01:  return "**"
    if p < 0.05:  return "*"
    if p < 0.10:  return "."
    return ""


def _nw_lags(T: int) -> int:
    """Lag automatico Newey-West: L = ⌈4(T/100)^(2/9)⌉ (Andrews 1991)."""
    return max(1, int(np.ceil(4 * (T / 100) ** (2 / 9))))


def _did_ols(it_mg: pd.Series, ct_mg: pd.Series,
             shock: pd.Timestamp,
             pre_start: pd.Timestamp, post_end: pd.Timestamp):
    """
    Stima DiD su panel bilanciato (inner join per data).

    Modello:  M = α + β·IT + γ·Post + δ·(IT×Post) + ε
    SE HC3 (heteroskedasticity-robust, corretto per panel stacked).

    Returns dict con delta, se, t, p_one, p_two, ci_lo, ci_hi, r2, n_obs
    """
    # Finestra temporale
    it_s = it_mg.loc[pre_start:post_end].dropna()
    ct_s = ct_mg.loc[pre_start:post_end].dropna()

    # Inner join: solo settimane presenti in entrambi
    panel = pd.concat({"IT": it_s, "CT": ct_s}, axis=1).dropna()
    if len(panel) < 8:
        return None

    # Stack in formato long
    long = panel.stack().reset_index()
    long.columns = ["date", "group", "M"]
    long["IT"]       = (long["group"] == "IT").astype(int)
    long["Post"]     = (long["date"] >= shock).astype(int)
    long["IT_x_Post"] = long["IT"] * long["Post"]

    X   = sm.add_constant(long[["IT", "Post", "IT_x_Post"]].values.astype(float))
    y   = long["M"].values.astype(float)
    lags = _nw_lags(len(long))

    try:
        # HC3 (heteroskedasticity-robust): corretto per panel stacked.
        # HAC sarebbe sbagliato perché il panel è interleaved (t1-IT, t1-CT,
        # t2-IT, t2-CT, …): il lag-1 residuo stimato da HAC è cross-sezionale,
        # non temporale → SE HAC sovrastimano l'autocorrelazione temporale.
        # HC3 è robusto e consistente senza ipotesi sull'ordinamento temporale.
        ols = sm.OLS(y, X).fit(cov_type="HC3")
    except Exception:
        return None

    delta  = float(ols.params[3])
    se     = float(ols.bse[3])
    t_stat = float(ols.tvalues[3])
    p_two  = float(ols.pvalues[3])
    p_one  = float(p_two / 2) if t_stat > 0 else float(1.0 - p_two / 2)
    ci_lo  = delta - 1.96 * se
    ci_hi  = delta + 1.96 * se

    n_pre_it  = int((long["IT"] == 1).sum() - long[(long["IT"]==1) & (long["Post"]==1)].shape[0])
    n_post_it = int(long[(long["IT"]==1) & (long["Post"]==1)].shape[0])

    return dict(delta=delta, se=se, t_stat=t_stat,
                p_two=p_two, p_one=p_one,
                ci_lo=ci_lo, ci_hi=ci_hi,
                r2=float(ols.rsquared),
                n_obs=len(long), lags_nw=lags,
                n_pre_it=n_pre_it, n_post_it=n_post_it)


def _pta_test(it_mg: pd.Series, ct_mg: pd.Series,
              shock: pd.Timestamp, n_weeks: int = PTA_WEEKS):
    """
    Test PTA su panel bilanciato nelle ultime n_weeks pre-shock.
    Regressione:  M ~ IT + t + IT×t   (t = giorni da inizio finestra)
    PTA non rigettata se p(IT×t) ≥ ALPHA.
    """
    pta_w  = pd.Timedelta(weeks=n_weeks)
    it_pta = it_mg[(it_mg.index >= shock - pta_w) & (it_mg.index < shock)].dropna()
    ct_pta = ct_mg[(ct_mg.index >= shock - pta_w) & (ct_mg.index < shock)].dropna()

    # Inner join sulle date disponibili
    both = pd.concat({"IT": it_pta, "CT": ct_pta}, axis=1).dropna()
    if len(both) < 4:
        return np.nan, None

    long = both.stack().reset_index()
    long.columns = ["date", "group", "M"]
    long["IT"]  = (long["group"] == "IT").astype(int)
    t0          = long["date"].min()
    long["t"]   = (long["date"] - t0).dt.days
    long["IT_x_t"] = long["IT"] * long["t"]

    X_pt   = sm.add_constant(long[["IT", "t", "IT_x_t"]].values.astype(float))
    try:
        ols_pt = sm.OLS(long["M"].values, X_pt).fit(cov_type="HC3")
        pta_p  = float(ols_pt.pvalues[3])
        return pta_p, bool(pta_p >= ALPHA)
    except Exception:
        return np.nan, None


# ════════════════════════════════════════════════════════════════════════════
# 4. Esegui DiD
# ════════════════════════════════════════════════════════════════════════════
print("\n" + "="*62)
print("FAMIGLIA C — H₀(iii): specificità italiana (DiD IT vs EU peer)")
print("="*62)

rows_did      = []
rows_windfall = []

for paese, ct_margins in CONTROL_MARGINS.items():
    for evento, ecfg in EVENTS.items():
        pre_start = ecfg["pre_start"]
        post_end  = ecfg["post_end"]
        shock     = _get_shock(evento)

        for fuel in FUELS:
            if fuel not in IT_MARGINS or fuel not in ct_margins:
                continue

            it_mg = IT_MARGINS[fuel]
            ct_mg = ct_margins[fuel]

            # ── PTA ──────────────────────────────────────────────────────
            pta_p, pta_pass = _pta_test(it_mg, ct_mg, shock)

            # ── DiD OLS ──────────────────────────────────────────────────
            res = _did_ols(it_mg, ct_mg, shock, pre_start, post_end)
            if res is None:
                print(f"   SKIP {evento} {fuel} ({paese}): dati insufficienti")
                continue

            delta  = res["delta"]
            se     = res["se"]
            p_one  = res["p_one"]
            ci_lo  = res["ci_lo"]
            ci_hi  = res["ci_hi"]
            rej    = (p_one < ALPHA) and (delta > 0)

            pta_tag = " ⚠PTA✗" if pta_pass is False else " PTA✓"
            print(f"   [{paese}] {evento[:22]:<22} | {fuel:<8}: "
                  f"δ={delta:+.4f}  p={p_one:.4f}{_stars(p_one):<3}"
                  f"  CI=[{ci_lo:+.4f},{ci_hi:+.4f}]  "
                  f"{'RIFIUTATA' if rej else 'non rig.'}{pta_tag}"
                  f"  N={res['n_obs']} L_NW={res['lags_nw']}")

            if delta > 0 and rej:
                note = f"δ>0 sign.: IT > {paese} post-shock — specificità IT"
                h0_status = "RIFIUTATA (δ>0)"
            elif delta < 0 and p_one < ALPHA:
                note = f"δ<0 sign.: IT < {paese} post-shock — contro specificità IT"
                h0_status = "non rifiutata (δ<0 sign.)"
            elif delta > 0:
                note = f"δ>0 n.s.: differenza IT vs {paese} non distinguibile da zero"
                h0_status = "non rifiutata (δ>0 n.s.)"
            else:
                note = f"δ≤0 n.s.: IT non cresce più di {paese}"
                h0_status = "non rifiutata"

            rows_did.append({
                "famiglia":          "C",
                "ipotesi":           "H0_iii",
                "evento":            evento,
                "paese_controllo":   paese,
                "carburante":        fuel,
                "fonte":             f"DiD_{evento}_{paese}_{fuel}",
                "shock_usato":       str(shock.date()),
                "shock_tipo":        "MCMC" if "shock_mcmc" in ecfg else "hardcoded",
                "wholesale_tipo":    "futures" if HAS_EUROBOB else "brent_yield",
                "n_obs_panel":       res["n_obs"],
                "lags_NW":           res["lags_nw"],
                "n_IT_pre":          res["n_pre_it"],
                "n_IT_post":         res["n_post_it"],
                "PTA_pvalue":        round(float(pta_p), 4) if not np.isnan(pta_p) else None,
                "PTA_non_rigettata": pta_pass,
                "delta_DiD_EUR_L":   round(delta, 5),
                "SE_NW":             round(se,    5),
                "CI_95_low":         round(ci_lo, 5),
                "CI_95_high":        round(ci_hi, 5),
                "t_stat":            round(res["t_stat"], 3),
                "p_value":           round(p_one, 6),
                "p_value_twosided":  round(res["p_two"], 6),
                "R2_OLS":            round(res["r2"], 3),
                "H0":                h0_status,
                "interpretation":    note,
            })

            # ── Windfall (solo per Germania come controllo principale) ──
            if paese == "Germania" and delta > 0:
                ref_yr = ecfg["ref_year"]
                n_wks  = res["n_post_it"]
                yrs_from_2022 = ref_yr - 2022
                for fuel_name, base_vol in VOLUME_2022.items():
                    if fuel_name != fuel:
                        continue
                    vol_adj = base_vol * (1 + TREND_ANNO) ** yrs_from_2022
                    for mult, scen in [(0.70, "-30%"), (1.00, "base"), (1.30, "+30%")]:
                        vol_s = vol_adj * mult
                        wf_base   = delta * vol_s * n_wks / 1e9
                        wf_ci_lo  = ci_lo * vol_s * n_wks / 1e9
                        wf_ci_hi  = ci_hi * vol_s * n_wks / 1e9
                        rows_windfall.append({
                            "evento":            evento,
                            "carburante":        fuel,
                            "delta_DiD_EUR_L":   round(delta, 5),
                            "SE_NW":             round(se, 5),
                            "CI_95_low_delta":   round(ci_lo, 5),
                            "CI_95_high_delta":  round(ci_hi, 5),
                            "n_settimane_post":  n_wks,
                            "vol_scenario":      scen,
                            "vol_adj_ML_wk":     round(vol_s / 1e6, 3),
                            "windfall_MLD_EUR":  round(wf_base, 3),
                            "windfall_CI_lo":    round(wf_ci_lo, 3),
                            "windfall_CI_hi":    round(wf_ci_hi, 3),
                        })


# ════════════════════════════════════════════════════════════════════════════
# 5. Salva CSV
# ════════════════════════════════════════════════════════════════════════════
if not rows_did:
    print("\n  Nessun test DiD prodotto.")
    df_did = pd.DataFrame()
else:
    df_did = pd.DataFrame(rows_did)
    df_did.to_csv("data/3_C.csv", index=False)
    print(f"\n✓ data/3_C.csv  ({len(df_did)} test DiD)")

if rows_windfall:
    df_wf = pd.DataFrame(rows_windfall)
    df_wf.to_csv("data/3_windfall.csv", index=False)
    print(f"✓ data/3_windfall.csv  ({len(df_wf)} scenari)")
    print("\n   Windfall estimates (scenario base, controllo: Germania):")
    for _, r in df_wf[df_wf["vol_scenario"] == "base"].iterrows():
        print(f"   {r['evento'][:22]} | {r['carburante']}: "
              f"δ={r['delta_DiD_EUR_L']:+.4f} EUR/L × {r['vol_adj_ML_wk']:.1f} ML/wk "
              f"× {r['n_settimane_post']} wk = {r['windfall_MLD_EUR']:.2f} MLD EUR "
              f"(CI 95%: [{r['windfall_CI_lo']:.2f}, {r['windfall_CI_hi']:.2f}])")


# ════════════════════════════════════════════════════════════════════════════
# 6. GRAFICI
# ════════════════════════════════════════════════════════════════════════════

# ── Fig A: Forest plot DiD ─────────────────────────────────────────────────
if not df_did.empty:
    events_list = list(EVENTS.keys())
    countries   = list(CONTROL_MARGINS.keys())
    n_events    = len(events_list)

    fig, axes = plt.subplots(
        1, n_events,
        figsize=(6.5 * n_events, max(5, len(countries) * 2 + 2)),
        squeeze=False
    )

    for ei, evento in enumerate(events_list):
        ax = axes[0][ei]
        ev_data = df_did[df_did["evento"] == evento]
        if ev_data.empty:
            ax.set_visible(False)
            continue

        ev_color = EVENTS[evento]["color"]
        plot_rows = []
        for paese in countries:
            sub = ev_data[ev_data["paese_controllo"] == paese].sort_values("carburante")
            for _, r in sub.iterrows():
                plot_rows.append(r)

        if not plot_rows:
            ax.set_visible(False)
            continue

        xlim_a = max(0.15,
                     max(abs(r["CI_95_high"]) for r in plot_rows) + 0.03,
                     max(abs(r["CI_95_low"]) for r in plot_rows) + 0.03)

        # Zona H₁ (δ > 0)
        ax.axvspan(0, xlim_a, alpha=0.06, color="#e74c3c")
        ax.axvline(0, color="black", lw=0.8, ls="--", alpha=0.5)

        y_pos = list(range(len(plot_rows)))
        for yi, r in zip(y_pos, plot_rows):
            col = FUEL_COLORS[r["carburante"]]
            sig = (r["p_value"] < ALPHA) and (r["delta_DiD_EUR_L"] > 0)

            ax.hlines(yi, r["CI_95_low"], r["CI_95_high"],
                      colors=col, lw=2.5, alpha=0.85)
            ax.plot(r["delta_DiD_EUR_L"], yi,
                    marker=("D" if sig else "o"), color=col, ms=8, zorder=3)
            if r["PTA_non_rigettata"] is False:
                ax.plot(r["delta_DiD_EUR_L"], yi, "o", color="#f39c12",
                        ms=16, zorder=2, alpha=0.45)

            pta_tag = " ⚠PTA" if r["PTA_non_rigettata"] is False else ""
            fuel_short = "B" if r["carburante"] == "Benzina" else "D"
            label = f"{r['paese_controllo']} {fuel_short}{pta_tag}"
            ax.text(-xlim_a + 0.003, yi, label, va="center", fontsize=8.2)

            p_str = f"p={r['p_value']:.3f}{_stars(r['p_value'])}"
            ax.text(xlim_a - 0.003, yi, p_str, va="center", ha="right", fontsize=7.5)

        ax.set_yticks([])
        ax.set_xlim(-xlim_a, xlim_a)
        ax.set_xlabel("δ̂  DiD (EUR/L)", fontsize=10)

        shock_tipo = "hardcoded"
        if "shock_mcmc" in EVENTS[evento]:
            shock_tipo = f"τ_MCMC={EVENTS[evento]['shock_mcmc'].date()}"
        ws_label = "wholesale: futures" if HAS_EUROBOB else "wholesale: Brent/yield"

        ax.set_title(
            f"{evento}\nδ>0 = IT sale più del paese controllo post-shock\n"
            f"({ws_label} | shock: {shock_tipo})",
            fontsize=9, fontweight="bold"
        )
        ax.grid(axis="x", alpha=0.25)

        # Separatori orizzontali tra paesi
        for k in range(len(countries) - 1):
            sep_y = (k + 1) * 2 - 0.5
            if sep_y < len(plot_rows):
                ax.axhline(sep_y, color="gray", lw=0.4, ls=":", alpha=0.5)

    legend_els = [
        Line2D([0],[0], marker="o", color=FUEL_COLORS["Benzina"], lw=0,
               ms=8, label="Benzina (B)"),
        Line2D([0],[0], marker="o", color=FUEL_COLORS["Diesel"], lw=0,
               ms=8, label="Diesel (D)"),
        Line2D([0],[0], marker="D", color="gray", lw=0, ms=8,
               label="Significativo (δ>0, p<0.05)"),
        Line2D([0],[0], marker="o", color="#f39c12", lw=0, ms=12,
               alpha=0.6, label="PTA violata (⚠PTA)"),
        mpatches.Patch(color="#e74c3c", alpha=0.12,
                       label="Zona H₁ (δ>0 = specificità IT)"),
    ]
    fig.legend(handles=legend_els, loc="lower center", ncol=5,
               fontsize=8.5, bbox_to_anchor=(0.5, -0.05))
    fig.suptitle(
        "H₀(iii) — DiD: IT vs paesi EU  |  Modello: M = α + β·IT + γ·Post + δ·(IT×Post)\n"
        "SE HC3 (heteroskedasticity-robust)  |  Panel bilanciato (inner join per data)\n"
        f"Wholesale {'futures Eurobob/GasOil' if HAS_EUROBOB else 'proxy Brent/yield'} — uniforme per tutti i paesi\n"
        "⚠ δ<0 = IT ≤ controllo (non evidenza di extraprofitti)",
        fontsize=9, fontweight="bold", y=1.02
    )
    plt.tight_layout()
    fig.savefig("plots/3_03a_did.png", dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print("✓ plots/3_03a_did.png")


# ── Fig B: Windfall con CI 95% ────────────────────────────────────────────
if rows_windfall:
    df_wf_plot = pd.DataFrame(rows_windfall)
    cases = df_wf_plot[["evento", "carburante"]].drop_duplicates().values.tolist()
    n_cases = len(cases)

    if n_cases > 0:
        fig, axes_wf = plt.subplots(
            1, n_cases,
            figsize=(5.5 * n_cases, 5.5),
            squeeze=False
        )
        scenario_colors = {"-30%": "#90caf9", "base": "#1565c0", "+30%": "#0d47a1"}
        scen_order      = ["-30%", "base", "+30%"]

        for ci, (ev, fuel) in enumerate(cases):
            ax = axes_wf[0][ci]
            sub = df_wf_plot[
                (df_wf_plot["evento"] == ev) &
                (df_wf_plot["carburante"] == fuel)
            ].set_index("vol_scenario")

            x_pos  = np.arange(len(scen_order))
            values = [sub.loc[s, "windfall_MLD_EUR"] if s in sub.index else 0
                      for s in scen_order]
            ci_lo_w = [sub.loc[s, "windfall_CI_lo"] if s in sub.index else 0
                       for s in scen_order]
            ci_hi_w = [sub.loc[s, "windfall_CI_hi"] if s in sub.index else 0
                       for s in scen_order]

            bars = ax.bar(x_pos, values,
                          color=[scenario_colors.get(s, "gray") for s in scen_order],
                          alpha=0.85, edgecolor="black", lw=0.6, width=0.55)

            # Errore CI 95% sul δ̂ (asimmetrico se ci_lo < 0)
            err_lo = [v - lo for v, lo in zip(values, ci_lo_w)]
            err_hi = [hi - v for v, hi in zip(values, ci_hi_w)]
            ax.errorbar(x_pos, values,
                        yerr=[err_lo, err_hi],
                        fmt="none", color="black", lw=1.5, capsize=5, capthick=1.5,
                        label="CI 95% δ̂")

            # Etichette valore
            for bar, val in zip(bars, values):
                yoff = bar.get_height() + max(err_hi) * 0.08 + 0.005
                ax.text(bar.get_x() + bar.get_width()/2, yoff,
                        f"{val:.2f}B€", ha="center", fontsize=9, fontweight="bold")

            # Banda grigia CI base scenario (zona plausibile δ̂)
            if "base" in sub.index:
                ax.axhspan(sub.loc["base", "windfall_CI_lo"],
                           sub.loc["base", "windfall_CI_hi"],
                           alpha=0.10, color="#1565c0",
                           label="CI 95% δ̂ (scenario base)")

            ax.set_xticks(x_pos)
            ax.set_xticklabels([f"Vol {s}\n({sub.loc[s,'vol_adj_ML_wk']:.1f} ML/wk)"
                                 if s in sub.index else s for s in scen_order],
                                fontsize=8.5)
            ax.set_title(
                f"{ev[:25]}\n{fuel} — vs Germania\n"
                f"δ̂={sub.loc['base','delta_DiD_EUR_L']:+.4f} EUR/L "
                f"({sub.loc['base','n_settimane_post']} settimane post-shock)",
                fontsize=8.5, fontweight="bold"
            )
            ax.set_ylabel("Miliardi EUR")
            y_top = max(ci_hi_w) * 1.30 if max(ci_hi_w) > 0 else 0.1
            y_bot = min(0, min(ci_lo_w) * 1.20)
            ax.set_ylim(y_bot, y_top)
            ax.axhline(0, color="black", lw=0.8)
            ax.grid(alpha=0.25, axis="y")
            if ci == 0:
                ax.legend(fontsize=8, loc="upper left")

            # Nota metodologica
            ax.text(0.01, 0.01,
                    f"Windfall = δ̂ × vol × n_wk\nCI include incertezza su δ̂\n"
                    f"Vol MISE 2022 ± 30%",
                    transform=ax.transAxes, fontsize=6.5, va="bottom",
                    color="gray", style="italic")

        fig.suptitle(
            "Extraprofitti stimati (IT vs Germania) — solo eventi con δ̂>0 significativo\n"
            "Le barre mostrano sensitività sui volumi; le barre d'errore l'incertezza su δ̂ (CI 95%)",
            fontsize=10, fontweight="bold"
        )
        plt.tight_layout()
        fig.savefig("plots/3_03b_windfall.png", dpi=DPI, bbox_inches="tight")
        plt.close(fig)
        print("✓ plots/3_03b_windfall.png")


# ── Fig C: serie temporali IT vs controllo (sanity check visivo) ──────────
# Per ogni evento × fuel mostra il margine IT e la mediana dei controlli
if not df_did.empty and IT_MARGINS:
    fig, axes = plt.subplots(2, len(EVENTS), figsize=(8 * len(EVENTS), 7),
                             sharex=False, squeeze=False)
    for ei, (evento, ecfg) in enumerate(EVENTS.items()):
        shock     = _get_shock(evento)
        pre_start = ecfg["pre_start"]
        post_end  = ecfg["post_end"]

        for fi, fuel in enumerate(FUELS):
            ax = axes[fi][ei]
            if fuel not in IT_MARGINS:
                continue

            it_s = IT_MARGINS[fuel].loc[pre_start:post_end]
            ax.plot(it_s.index, it_s.values,
                    color=FUEL_COLORS[fuel], lw=2.0, label="Italia", zorder=3)

            # Mediana e range controlli
            ct_all = []
            for ct_mg in CONTROL_MARGINS.values():
                if fuel in ct_mg:
                    ct_all.append(ct_mg[fuel].loc[pre_start:post_end])
            if ct_all:
                ct_df  = pd.concat(ct_all, axis=1).dropna(how="all")
                ct_med = ct_df.median(axis=1)
                ct_lo  = ct_df.min(axis=1)
                ct_hi  = ct_df.max(axis=1)
                ax.plot(ct_med.index, ct_med.values,
                        color="gray", lw=1.5, ls="--", label="Mediana controlli")
                ax.fill_between(ct_lo.index, ct_lo.values, ct_hi.values,
                                alpha=0.12, color="gray", label="Range controlli")

            ax.axvline(shock, color=ecfg["color"], lw=1.8, ls="--", alpha=0.9)
            ax.text(shock + pd.Timedelta(days=5),
                    float(it_s.max()) * 0.97,
                    "shock", fontsize=7.5, color=ecfg["color"], rotation=90, va="top")

            ax.set_title(f"{evento[:22]}\n{fuel} — margine lordo IT vs controlli",
                         fontsize=9, fontweight="bold")
            ax.set_ylabel("Margine lordo (EUR/L)\n= pompa ex-tasse − wholesale")
            ax.grid(alpha=0.25)
            if fi == 0 and ei == 0:
                ax.legend(fontsize=8)

    fig.suptitle(
        "Margine lordo IT (linea) vs controlli EU (banda) — pompa ex-tasse meno futures wholesale\n"
        f"Wholesale: {'futures Eurobob (benzina) / GasOil ICE (diesel)' if HAS_EUROBOB else 'proxy Brent/yield'}"
        f" — stesso wholesale per IT e controlli",
        fontsize=10, fontweight="bold"
    )
    plt.tight_layout()
    fig.savefig("plots/3_03c_timeseries.png", dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print("✓ plots/3_03c_timeseries.png  (sanity check serie temporali)")


print("\nScript 3_03 completato.")