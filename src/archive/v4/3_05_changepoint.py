"""
3_05_changepoint.py  — Bayesian Change Point Detection (MCMC/NUTS)
====================================================================
Quando si è rotta la dinamica dei prezzi carburante in Italia?

Modella le serie di prezzo (log) con un modello piecewise-lineare bayesiano
raccordato al changepoint τ, con likelihood Student-T per gestire le code
pesanti. Usa NUTS tramite PyMC per campionamento efficiente.

Il lag D = τ - shock_date misura l'anticipo (D < 0) o ritardo (D > 0).
Un lag fortemente negativo è consistente con mercati forward-looking, non
con speculazione post-shock.

Note sul modello MCMC:
  - τ parametrizzazione default: τ_raw ~ Beta(2,2) su [0,1].
  - τ parametrizzazione logit (use_logit_tau=True): τ_logit ~ Normal(0,1.5),
    τ_raw = sigmoid(τ_logit). NUTS campiona in ℝ → geometria migliore per
    posteriori multimodali (usato per Ucraina|Brent, forte AR(1)).
  - τ prior informativo (tau_prior_center + tau_prior_conc): Beta centrata
    sul τ di una serie correlata già nota (usato per Hormuz|Crack_Benz).
  - Likelihood default: Student-T, ν ~ Gamma(2, 0.1).
  - Likelihood SkewNormal (use_skewnorm=True): quando ΔAIC OLS indica
    asimmetria forte (usato per Hormuz|Crack_Benz, ΔAIC=-4.7).
  - AR(1) non modellato esplicitamente: l'autocorrelazione è riportata
    come diagnostico (DW) ma il CI posteriore di τ già assorbe l'incertezza.

Output sezione 8: stampa cronologia esplicita per ogni evento:
  Shock ufficiale → τ_Brent → τ_Benzina → τ_Diesel → τ_Crack
  con lag D e CI 95%.

Diagnostici OLS (motivano la scelta dei test in 3_02/3_03):
  - Breusch-Pagan (eteroschedasticità)
  - Shapiro-Wilk (non-normalità)
  - Durbin-Watson (autocorrelazione)
  - AIC confronto Student-T vs Skew-Normal sui residui OLS piecewise

Input:
  data/3_dataset.csv

Output:
  data/3_cp.csv                        — MAP, CI 95%, lag, diagnostici
  data/3_cp_diagnostics.csv            — OLS diagnostics per serie×evento
  plots/3_05_{evento}_{serie}.png      — serie + piecewise fit + KDE posteriore
  plots/3_05_{evento}_{serie}_diag.png — QQ, residui, ACF
  plots/3_05_summary.png               — lag D per tutti gli eventi/serie
"""

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from scipy import stats
from scipy.stats import gaussian_kde
from statsmodels.stats.stattools import durbin_watson
from statsmodels.stats.diagnostic import het_breuschpagan
from statsmodels.graphics.tsaplots import plot_acf
import statsmodels.api as sm

warnings.filterwarnings("ignore")
os.makedirs("data",  exist_ok=True)
os.makedirs("plots", exist_ok=True)

# ── Controlla PyMC ─────────────────────────────────────────────────────────────
try:
    import pymc as pm
    import pytensor.tensor as pt
    import arviz as az
    HAS_PYMC = True
    print("✓ PyMC disponibile — uso NUTS")
except ImportError:
    HAS_PYMC = False
    print("✗ PyMC non trovato — installa con: pip install pymc")
    print("  Fallback: Metropolis-Hastings puro NumPy")

# ── Costanti ──────────────────────────────────────────────────────────────────
DPI         = 180
ALPHA       = 0.05
LAG_THRESH  = 30        # giorni: |D| < 30gg = trasmissione rapida

# Configurazione MCMC (PyMC/NUTS)
_MCMC_DEFAULT = dict(draws=2000, tune=2000, chains=4,
                     target_accept=0.95, init="adapt_diag", max_treedepth=15)
MCMC_OVERRIDES = {
    # ── Ucraina|Brent: Rhat=1.191 nella config precedente ─────────────────
    # Causa: posteriore multimodale su 43 settimane con AR(1) forte.
    # Fix: reparametrizzazione logit di τ → NUTS si muove in spazio non
    # vincolato ℝ invece che [0,1], geometria molto più regolare.
    # tune=8000 + target_accept=0.995 per step size molto piccolo.
    "Ucraina (Feb 2022)__Brent": dict(
        tune=8000,
        target_accept=0.995,
        init="adapt_full",
        max_treedepth=20,
        use_logit_tau=True,     # ← reparametrizzazione chiave
    ),
    # ── Hormuz|Crack_Benz: Rhat=1.447, 15 divergenze ──────────────────────
    # Causa: likelihood Student-T inadatta (ΔAIC=-4.7 vs SkewNormal,
    # α_skew=-4.8 → coda sinistra pesante). Geometria degenere → divergenze.
    # Fix 1: SkewNormal come likelihood (più vicino alla vera distribuzione).
    # Fix 2: prior informativo su τ centrato sul τ benzina wholesale
    #        (≈ 2026-02-23, settimana ~22/30 nella finestra → pos ≈ 0.72).
    # Fix 3: target_accept=0.999 per step size aggressivamente piccolo.
    "Hormuz (Feb 2026)__Crack_Benz": dict(
        tune=8000,
        target_accept=0.999,
        init="jitter+adapt_diag_grad",
        max_treedepth=20,
        use_skewnorm=True,          # ← SkewNormal invece di Student-T
        tau_prior_center=0.72,      # centrato su τ benzina wholesale noto
        tau_prior_conc=6.0,         # concentrazione moderata (non troppo rigido)
    ),
}

# Configurazione MCMC fallback (MH puro NumPy)
_MH_N_ITER   = 30_000
_MH_WARMUP   = 8_000
_MH_TAU_STEP = 8

EVENTS = {
    "Ucraina (Feb 2022)": {
        "shock":     "2022-02-24",
        "win_start": "2021-10-01",
        "win_end":   "2022-07-31",
        "color":     "#e74c3c",
    },
    "Iran-Israele (Giu 2025)": {
        "shock":     "2025-06-13",
        "win_start": "2025-02-01",
        "win_end":   "2025-12-31",
        "color":     "#e67e22",
    },
    "Hormuz (Feb 2026)": {
        "shock":     "2026-02-28",
        "win_start": "2025-10-01",
        "win_end":   "2026-04-30",
        "color":     "#8e44ad",
    },
}

# Le serie su cui girare il change point
# is_log=True → serie in log (prezzi), is_log=False → livelli (crack spread)
SERIES_DEF = {
    "Brent":        ("log_brent",          "#2166ac", True),
    "Benzina":      ("log_benzina",        "#d6604d", True),
    "Diesel":       ("log_diesel",         "#31a354", True),
    "Crack_Benz":   ("margine_benz_real",  "#d6604d", False),
    "Crack_Dies":   ("margine_dies_real",  "#4393c3", False),
}


# ════════════════════════════════════════════════════════════════════════════
# 0. Carico dataset e calcolo colonne log
# ════════════════════════════════════════════════════════════════════════════
print("Carico data/3_dataset.csv...")
df = pd.read_csv("data/3_dataset.csv", index_col=0, parse_dates=True)
df = df.loc["2019-01-01":].copy()
print(f"   {len(df)} settimane | {df.index[0].date()} – {df.index[-1].date()}")

# Log dei prezzi (per series con is_log=True)
if "brent_eur"    in df.columns: df["log_brent"]   = np.log(df["brent_eur"].clip(lower=0.01))
if "benzina_eur_l" in df.columns: df["log_benzina"] = np.log(df["benzina_eur_l"].clip(lower=0.001))
if "diesel_eur_l"  in df.columns: df["log_diesel"]  = np.log(df["diesel_eur_l"].clip(lower=0.001))


# ════════════════════════════════════════════════════════════════════════════
# 1. Modelli MCMC
# ════════════════════════════════════════════════════════════════════════════

def _mcmc_cfg(key: str) -> dict:
    cfg = dict(_MCMC_DEFAULT)
    cfg.update(MCMC_OVERRIDES.get(key, {}))
    return cfg


# ── PyMC / NUTS ───────────────────────────────────────────────────────────────
def _bayesian_nuts(x: np.ndarray, y: np.ndarray, cfg: dict) -> dict:
    """
    Modello piecewise-lineare Bayesiano campionato con NUTS via PyMC.

    Parametrizzazione di τ (selezionabile via cfg):
      - Default: τ_raw ~ Beta(2,2) su [0,1]  →  τ ∈ [x_lo, x_hi]
      - use_logit_tau=True: τ_logit ~ Normal(0,1.5) → τ_raw = sigmoid(τ_logit)
        NUTS si muove in ℝ invece di [0,1]: molto meglio per posteriori
        multimodali o con geometria irregolare (es. Ucraina|Brent, AR forte).
      - tau_prior_center + tau_prior_conc: Beta informativa centrata su un
        valore noto (es. τ da wholesale già stimato). Usato per Hormuz|Crack_Benz.

    Likelihood (selezionabile via cfg):
      - Default: Student-T (code pesanti, robusto)
      - use_skewnorm=True: SkewNormal (quando ΔAIC OLS indica asimmetria forte)
        → α_skew ~ Normal(0,5), stima la direzione e intensità dell'asimmetria.
    """
    import scipy.special as _scipy_special

    n      = len(x)
    sd_y   = float(np.std(y))
    mean_y = float(np.mean(y))
    x_lo, x_hi = float(x[0]), float(x[-1])
    x_rng  = max(x_hi - x_lo, 1.0)

    # Estrae flag specifici (non vengono passati a pm.sample)
    use_logit_tau    = cfg.pop("use_logit_tau",    False)
    use_skewnorm     = cfg.pop("use_skewnorm",     False)
    tau_prior_center = cfg.pop("tau_prior_center", None)   # float in (0,1)
    tau_prior_conc   = cfg.pop("tau_prior_conc",  None)   # float > 0

    with pm.Model():

        # ── Parametrizzazione τ ──────────────────────────────────────────
        if use_logit_tau:
            # Reparametrizzazione non-centrata: NUTS campiona in ℝ
            # → risolve multimodalità e autocorrelazione alta nel posteriore
            if tau_prior_center is not None:
                logit_center = float(_scipy_special.logit(
                    float(np.clip(tau_prior_center, 0.01, 0.99))
                ))
                tau_logit = pm.Normal("tau_logit", mu=logit_center, sigma=1.5)
            else:
                tau_logit = pm.Normal("tau_logit", mu=0.0, sigma=1.5)
            tau_raw = pm.Deterministic("tau_raw", pm.math.sigmoid(tau_logit))
        else:
            if tau_prior_center is not None and tau_prior_conc is not None:
                # Beta informativa: modo ≈ tau_prior_center
                c = float(np.clip(tau_prior_center, 0.05, 0.95))
                k = float(tau_prior_conc)
                alpha_b = max(c * k, 0.5)
                beta_b  = max((1.0 - c) * k, 0.5)
                tau_raw = pm.Beta("tau_raw", alpha=alpha_b, beta=beta_b)
            else:
                tau_raw = pm.Beta("tau_raw", alpha=2, beta=2)

        tau = pm.Deterministic("tau", x_lo + tau_raw * x_rng)

        sigma = pm.HalfNormal("sigma", sigma=sd_y)

        b1 = pm.Normal("b1", mu=0.0, sigma=3.0 * sd_y)
        b2 = pm.Normal("b2", mu=0.0, sigma=3.0 * sd_y)
        a1 = pm.Normal("a1", mu=mean_y, sigma=sd_y)
        a2 = pm.Deterministic("a2", a1 + tau * (b1 - b2))

        x_t  = pt.as_tensor_variable(x.astype(float))
        step = pm.math.sigmoid((x_t - tau) * 50)
        mu   = (a1 + b1 * x_t) * (1.0 - step) + (a2 + b2 * x_t) * step

        # ── Likelihood ───────────────────────────────────────────────────
        if use_skewnorm:
            # SkewNormal: gestisce asimmetria nei residui (ΔAIC < -2 vs StudentT)
            # α_skew > 0 → coda destra; α_skew < 0 → coda sinistra
            alpha_skew = pm.Normal("alpha_skew", mu=0.0, sigma=5.0)
            pm.SkewNormal("obs", mu=mu, sigma=sigma, alpha=alpha_skew, observed=y)
        else:
            nu = pm.Gamma("nu", alpha=2, beta=0.1)
            pm.StudentT("obs", nu=nu, mu=mu, sigma=sigma, observed=y)

        trace = pm.sample(
            draws=cfg["draws"], tune=cfg["tune"], chains=cfg["chains"],
            progressbar=True,
            random_seed=list(range(42, 42 + cfg["chains"])),
            target_accept=cfg["target_accept"],
            nuts_sampler_kwargs={"max_treedepth": cfg["max_treedepth"]},
            init=cfg["init"],
            return_inferencedata=True,
        )

    try:
        rhat_max = float(np.nanmax(az.rhat(trace).to_array().values.flatten()))
        ess_vals = az.ess(trace, method="bulk").to_array().values.flatten()
        ess_min  = float(np.min(ess_vals[np.isfinite(ess_vals) & (ess_vals > 0)]))
    except Exception:
        rhat_max, ess_min = np.nan, np.nan

    if np.isnan(rhat_max):
        print("  Diagnostica convergenza non disponibile")
    elif rhat_max > 1.05:
        print(f"  ⚠ CONVERGENZA DUBBIA: Rhat={rhat_max:.3f} > 1.05")
    elif rhat_max > 1.01:
        print(f"  Rhat={rhat_max:.3f} (lieve)")
    else:
        print(f"  ✓ Convergenza ok: Rhat={rhat_max:.3f}  ESS={ess_min:.0f}")

    tau_post = trace.posterior["tau"].values.flatten()
    lo, hi   = ALPHA / 2 * 100, (1 - ALPHA / 2) * 100

    return {
        "tau_mean":  float(np.mean(tau_post)),
        "tau_lo":    float(np.percentile(tau_post, lo)),
        "tau_hi":    float(np.percentile(tau_post, hi)),
        "tau_idx":   int(np.clip(round(float(np.median(tau_post))), 1, n - 2)),
        "tau_post":  tau_post,
        "b1_mean":   float(np.mean(trace.posterior["b1"].values.flatten())),
        "b1_lo":     float(np.percentile(trace.posterior["b1"].values.flatten(), lo)),
        "b1_hi":     float(np.percentile(trace.posterior["b1"].values.flatten(), hi)),
        "b2_mean":   float(np.mean(trace.posterior["b2"].values.flatten())),
        "b2_lo":     float(np.percentile(trace.posterior["b2"].values.flatten(), lo)),
        "b2_hi":     float(np.percentile(trace.posterior["b2"].values.flatten(), hi)),
        "a1_mean":   float(np.mean(trace.posterior["a1"].values.flatten())),
        # nu non esiste nel trace quando use_skewnorm=True → np.nan
        "nu_mean":   (float(np.mean(trace.posterior["nu"].values.flatten()))
                      if "nu" in trace.posterior else np.nan),
        "rhat_max":  rhat_max,
        "ess_min":   ess_min,
        "sampler":   "NUTS",
    }


# ── Fallback: MH puro NumPy ───────────────────────────────────────────────────
def _bayesian_mh(x: np.ndarray, y: np.ndarray) -> dict:
    """
    Metropolis-Hastings + Gibbs (puro NumPy) per ambienti senza PyMC.
    Modello piecewise-costante a 2 regime (media + std condiviso) — meno
    sofisticato del NUTS ma funzionale.
    """
    from scipy import stats as _stats

    rng  = np.random.default_rng(42)
    n    = len(y)
    MU0, SIG0 = 0.0, float(np.std(y)) * 2
    SIG_SCALE  = float(np.std(y)) * 1.5

    t1  = n // 3;  t2 = 2 * n // 3
    mu1 = float(y[:t1].mean());  mu2 = float(y[t1:t2].mean());  mu3 = float(y[t2:].mean())
    sig = float(np.std(y)) * 1.1

    def _ll(y_, t1_, t2_, m1, m2, m3, s):
        n_ = len(y_)
        segs = [(y_[:t1_], m1), (y_[t1_:t2_], m2), (y_[t2_:], m3)]
        ll = 0.0
        for seg, m in segs:
            if len(seg) == 0:
                return -np.inf
            ll += _stats.norm.logpdf(seg, loc=m, scale=s).sum()
        return ll

    def _gibbs_mu(y_, t1_, t2_, s_):
        n_ = len(y_)
        out = np.empty(3)
        prec_p = 1.0 / SIG0**2
        for k, (a, b) in enumerate([(0, t1_), (t1_, t2_), (t2_, n_)]):
            seg = y_[a:b];  nk = len(seg)
            if nk == 0:
                out[k] = rng.normal(MU0, SIG0); continue
            prec_l = nk / s_**2
            prec_q = prec_p + prec_l
            mu_q   = (MU0 * prec_p + seg.sum() / s_**2) / prec_q
            out[k] = rng.normal(mu_q, 1.0 / np.sqrt(prec_q))
        return out

    c_t1 = np.empty(_MH_N_ITER, dtype=int)
    c_t2 = np.empty(_MH_N_ITER, dtype=int)
    c_ms = np.empty((_MH_N_ITER, 3))
    c_sg = np.empty(_MH_N_ITER)
    ll_c = _ll(y, t1, t2, mu1, mu2, mu3, sig)

    for i in range(_MH_N_ITER):
        t1p = int(t1 + rng.integers(-_MH_TAU_STEP, _MH_TAU_STEP + 1))
        if 3 <= t1p < t2 - 3:
            llp = _ll(y, t1p, t2, mu1, mu2, mu3, sig)
            if np.log(rng.uniform() + 1e-300) < llp - ll_c:
                t1, ll_c = t1p, llp
        t2p = int(t2 + rng.integers(-_MH_TAU_STEP, _MH_TAU_STEP + 1))
        if t1 + 3 <= t2p < n - 3:
            llp = _ll(y, t1, t2p, mu1, mu2, mu3, sig)
            if np.log(rng.uniform() + 1e-300) < llp - ll_c:
                t2, ll_c = t2p, llp
        mu1, mu2, mu3 = _gibbs_mu(y, t1, t2, sig)
        ll_c = _ll(y, t1, t2, mu1, mu2, mu3, sig)
        ls_c = np.log(sig)
        ls_p = ls_c + rng.normal(0, 0.04)
        sp   = np.exp(ls_p)
        lp_c = _stats.halfnorm.logpdf(sig, scale=SIG_SCALE) + ls_c
        lp_p = _stats.halfnorm.logpdf(sp,  scale=SIG_SCALE) + ls_p
        llp  = _ll(y, t1, t2, mu1, mu2, mu3, sp)
        if np.log(rng.uniform() + 1e-300) < (llp + lp_p) - (ll_c + lp_c):
            sig, ll_c = sp, llp
        c_t1[i] = t1;  c_t2[i] = t2
        c_ms[i] = [mu1, mu2, mu3];  c_sg[i] = sig

    sl = slice(_MH_WARMUP, None)
    t1_post = c_t1[sl];  t2_post = c_t2[sl]
    # Restituisce il change point più vicino al centro della serie (τ₁ per fenomeni a inizio finestra)
    lo, hi  = ALPHA / 2 * 100, (1 - ALPHA / 2) * 100
    return {
        "tau_mean":  float(t1_post.mean()),
        "tau_lo":    float(np.percentile(t1_post, lo)),
        "tau_hi":    float(np.percentile(t1_post, hi)),
        "tau_idx":   int(np.median(t1_post)),
        "tau_post":  t1_post.astype(float),
        "b1_mean":   0.0, "b1_lo": 0.0, "b1_hi": 0.0,
        "b2_mean":   0.0, "b2_lo": 0.0, "b2_hi": 0.0,
        "a1_mean":   float(c_ms[sl, 0].mean()),
        "nu_mean":   np.nan,
        "rhat_max":  np.nan,
        "ess_min":   float((_MH_N_ITER - _MH_WARMUP)),
        "sampler":   "MH-fallback",
    }


def bayesian_changepoint(x: np.ndarray, y: np.ndarray, cfg: dict) -> dict:
    if HAS_PYMC:
        return _bayesian_nuts(x, y, cfg)
    else:
        return _bayesian_mh(x, y)


# ════════════════════════════════════════════════════════════════════════════
# 2. Confronto AIC: Student-T vs Skew-Normal sui residui OLS
# ════════════════════════════════════════════════════════════════════════════
def distribution_aic_comparison(resid: np.ndarray) -> dict:
    """
    Confronta via AIC Student-T e Skew-Normal MLE sui residui OLS piecewise.
    ΔAIC = AIC(SkewNorm) − AIC(StudentT) < −2 → SkewNormal preferita.
    """
    r = np.asarray(resid, dtype=float)
    r = r[np.isfinite(r)];  n = len(r)

    aic_t = np.nan
    try:
        params_t = stats.t.fit(r)
        ll_t     = float(np.sum(stats.t.logpdf(r, *params_t)))
        aic_t    = -2.0 * ll_t + 2.0 * 3
    except Exception:
        pass

    aic_sn = np.nan;  alpha_sn = np.nan
    try:
        params_sn = stats.skewnorm.fit(r)
        ll_sn     = float(np.sum(stats.skewnorm.logpdf(r, *params_sn)))
        aic_sn    = -2.0 * ll_sn + 2.0 * 3
        alpha_sn  = float(params_sn[0])
    except Exception:
        pass

    delta_aic = float(aic_sn - aic_t) if (np.isfinite(aic_sn) and np.isfinite(aic_t)) else np.nan

    skew_emp, skew_p = np.nan, np.nan
    try:
        skew_emp = float(stats.skew(r))
        if n >= 8:
            skew_p = float(stats.skewtest(r).pvalue)
    except Exception:
        pass

    use_skewed = False
    if np.isfinite(delta_aic):
        if delta_aic < -2.0:
            rec = (f"⚠  ΔAIC={delta_aic:.1f}: SkewNormal migliore. "
                   f"α_skew={alpha_sn:.3f}. Considerare Skewed-T nel MCMC.")
            use_skewed = True
        elif delta_aic > 2.0:
            rec = f"✓  ΔAIC={delta_aic:.1f}: Student-T adeguato."
        else:
            rec = f"~  ΔAIC={delta_aic:.1f}: modelli equivalenti (|ΔAIC|≤2)."
    else:
        rec = "ΔAIC non disponibile"

    return {
        "AIC_StudentT":      round(float(aic_t),  2) if np.isfinite(aic_t)  else None,
        "AIC_SkewNormal":    round(float(aic_sn), 2) if np.isfinite(aic_sn) else None,
        "delta_AIC_Skew_T":  round(delta_aic, 2)     if np.isfinite(delta_aic) else None,
        "alpha_SkewNorm":    round(alpha_sn, 3)       if np.isfinite(alpha_sn) else None,
        "skewness_empirica": round(skew_emp, 3)       if np.isfinite(skew_emp) else None,
        "skewtest_p":        round(skew_p, 4)         if np.isfinite(skew_p)   else None,
        "use_skewed_t":      use_skewed,
        "raccomandazione_dist": rec,
    }


# ════════════════════════════════════════════════════════════════════════════
# 3. Diagnostici OLS piecewise
# ════════════════════════════════════════════════════════════════════════════
def ols_diagnostics(x: np.ndarray, y: np.ndarray, cp_idx: int) -> dict:
    """
    Breusch-Pagan, Shapiro-Wilk, Durbin-Watson e AIC distribuzionale
    sui residui del fit OLS piecewise al change point stimato.
    """
    cp = cp_idx
    s1, i1, *_ = stats.linregress(x[:cp], y[:cp])
    s2, *_     = stats.linregress(x[cp:], y[cp:])
    i2         = i1 + cp * (s1 - s2)
    y_hat      = np.concatenate([i1 + s1 * x[:cp], i2 + s2 * x[cp:]])
    resid      = y - y_hat

    try:
        _, bp_p, _, _ = het_breuschpagan(resid, sm.add_constant(y_hat))
    except Exception:
        bp_p = np.nan
    try:
        _, sw_p = stats.shapiro(resid[:5000])
    except Exception:
        sw_p = np.nan
    dw = durbin_watson(resid)

    aic_info = distribution_aic_comparison(resid)
    return {"BP_p": bp_p, "SW_p": sw_p, "DW": dw,
            "resid": resid, "y_hat": y_hat, **aic_info}


# ════════════════════════════════════════════════════════════════════════════
# 4. Setup matplotlib
# ════════════════════════════════════════════════════════════════════════════
plt.rcParams.update({
    "font.family":        "serif",
    "font.serif":         ["Times New Roman", "DejaVu Serif"],
    "axes.spines.top":    False,
    "axes.spines.right":  False,
    "axes.grid":          True,
    "grid.color":         "#e8e8e8",
    "grid.linewidth":     0.6,
})


# ════════════════════════════════════════════════════════════════════════════
# 5. Loop principale: eventi × serie
# ════════════════════════════════════════════════════════════════════════════
table_rows = []
diag_rows  = []

for ev_name, ev_cfg in EVENTS.items():
    shock    = pd.Timestamp(ev_cfg["shock"])
    ev_color = ev_cfg["color"]
    safe_ev  = (ev_name.replace(" ", "_").replace("(", "").replace(")", "")
                       .replace("/", "").replace(",", ""))

    for ser_name, (col, ser_color, is_log) in SERIES_DEF.items():
        if col not in df.columns:
            continue

        df_ev = df.loc[ev_cfg["win_start"]:ev_cfg["win_end"], col].dropna()
        if len(df_ev) < 8:
            print(f"  SKIP {ev_name}|{ser_name}: {len(df_ev)} obs < 8")
            continue

        print(f"\n{'='*62}")
        print(f"MCMC → {ev_name} | {ser_name}  ({len(df_ev)} settimane)")
        print(f"{'='*62}")

        x = np.arange(len(df_ev), dtype=float)
        y = df_ev.values

        ci = bayesian_changepoint(x, y, _mcmc_cfg(f"{ev_name}__{ser_name}"))

        cp      = ci["tau_idx"]
        cp_date = df_ev.index[0] + pd.Timedelta(weeks=int(ci["tau_idx"]))
        cp_lo   = df_ev.index[0] + pd.Timedelta(weeks=max(0, round(ci["tau_lo"])))
        cp_hi   = df_ev.index[0] + pd.Timedelta(
                      weeks=min(len(df_ev) - 1, round(ci["tau_hi"])))
        lag_days = (cp_date - shock).days

        # OLS piecewise per slope leggibili e R²
        def _lr(xv, yv):
            if len(xv) < 2: return 0., 0., 0.
            s, i, r, *_ = stats.linregress(xv, yv)
            return s, i, r ** 2

        b1_ols, a1_ols, r2_pre  = _lr(x[:cp], y[:cp])
        b2_ols, _, r2_post      = _lr(x[cp:], y[cp:])

        dt1 = np.log(2) / (b1_ols / 7) if b1_ols > 0 else np.inf
        dt2 = np.log(2) / (b2_ols / 7) if b2_ols > 0 else np.inf
        a2_ols = a1_ols + cp * (b1_ols - b2_ols)

        # OLS diagnostics
        diag = ols_diagnostics(x, y, cp)
        diag_rows.append({
            "Evento": ev_name, "Serie": ser_name,
            "BP_p":              round(float(diag["BP_p"]), 4) if not np.isnan(diag["BP_p"]) else None,
            "SW_p":              round(float(diag["SW_p"]), 4) if not np.isnan(diag["SW_p"]) else None,
            "DW":                round(float(diag["DW"]),   3),
            "AIC_StudentT":      diag.get("AIC_StudentT"),
            "AIC_SkewNormal":    diag.get("AIC_SkewNormal"),
            "delta_AIC_Skew_T":  diag.get("delta_AIC_Skew_T"),
            "alpha_SkewNorm":    diag.get("alpha_SkewNorm"),
            "skewness_empirica": diag.get("skewness_empirica"),
            "skewtest_p":        diag.get("skewtest_p"),
            "use_skewed_t":      diag.get("use_skewed_t", False),
            "dist_raccomandazione": diag.get("raccomandazione_dist", ""),
        })
        print(f"  Distr. AIC: {diag.get('raccomandazione_dist', 'N/A')}")

        rhat_flag = ("CONVERGENZA DUBBIA" if ci["rhat_max"] > 1.05
                     else "ok" if not np.isnan(ci["rhat_max"]) else "N/A")

        table_rows.append({
            "Evento":      ev_name,
            "Serie":       ser_name,
            "tau":         cp_date.date(),
            "CI_95_lo":    cp_lo.date(),
            "CI_95_hi":    cp_hi.date(),
            "Lag_gg":      lag_days,
            "trasmissione_rapida": "SI" if abs(lag_days) < LAG_THRESH else "NO",
            "b1_OLS":      round(b1_ols, 5),
            "b2_OLS":      round(b2_ols, 5),
            "DT1_gg":      round(dt1, 1) if dt1 != np.inf else "inf",
            "DT2_gg":      round(dt2, 1) if dt2 != np.inf else "inf",
            "R2_pre":      round(r2_pre, 4),
            "R2_post":     round(r2_post, 4),
            "nu_StudentT": round(ci["nu_mean"], 2) if not np.isnan(ci["nu_mean"]) else None,
            "rhat_max":    round(ci["rhat_max"], 3) if not np.isnan(ci["rhat_max"]) else None,
            "ess_min":     round(ci["ess_min"], 0)  if not np.isnan(ci["ess_min"])  else None,
            "rhat_flag":   rhat_flag,
            "sampler":     ci["sampler"],
            "BP_p":        diag_rows[-1]["BP_p"],
            "SW_p":        diag_rows[-1]["SW_p"],
            "DW":          diag_rows[-1]["DW"],
        })

        print(f"  τ={cp_date.date()}  lag={lag_days:+d}gg  "
              f"DW={diag['DW']:.2f}  BP_p={diag['BP_p']:.4f}  SW_p={diag['SW_p']:.4f}")

        # ── Plot principale con KDE posteriore ────────────────────────────
        fig = plt.figure(figsize=(12, 5.8))
        gs  = fig.add_gridspec(2, 1, height_ratios=[4, 1], hspace=0.0)
        ax  = fig.add_subplot(gs[0])
        axk = fig.add_subplot(gs[1], sharex=ax)

        ax.scatter(df_ev.index, y, s=18, color="black", alpha=0.55, zorder=3,
                   label="dato osservato")
        # Rette pre/post
        ax.plot(df_ev.index[:cp],
                a1_ols + ci["b1_mean"] * x[:cp],
                color="#27ae60", lw=2.5,
                label=f"Pre-shock β={ci['b1_mean']:.4f}")
        ax.fill_between(df_ev.index[:cp],
                        a1_ols + ci["b1_lo"] * x[:cp],
                        a1_ols + ci["b1_hi"] * x[:cp],
                        color="#27ae60", alpha=0.10)
        ax.plot(df_ev.index[cp:],
                a2_ols + ci["b2_mean"] * x[cp:],
                color="#e74c3c", lw=2.5,
                label=f"Post-shock β={ci['b2_mean']:.4f}")
        ax.fill_between(df_ev.index[cp:],
                        a2_ols + ci["b2_lo"] * x[cp:],
                        a2_ols + ci["b2_hi"] * x[cp:],
                        color="#e74c3c", alpha=0.10)
        # Change point MAP
        ax.axvline(cp_date, color="#2980b9", lw=2.0, ls="--",
                   label=f"τ MAP = {cp_date.date()}")
        ax.axvspan(cp_lo, cp_hi, alpha=0.13, color="#2980b9",
                   label=f"CI 95%: {cp_lo.strftime('%d %b %y')} – {cp_hi.strftime('%d %b %y')}")
        # Shock geopolitico
        ax.axvline(shock, color=ev_color, lw=1.8, ls=":",
                   label=f"Shock {ev_cfg['shock']}")

        ylabel = "log(prezzo)" if is_log else "EUR/L (reale)"
        nu_str = f"  ν={ci['nu_mean']:.1f}" if not np.isnan(ci["nu_mean"]) else ""
        ax.set_title(
            f"{ev_name} — {ser_name}   |   D={lag_days:+d}gg{nu_str}   "
            f"[{ci['sampler']}]",
            fontsize=11, fontweight="bold",
            color="#c0392b" if abs(lag_days) < LAG_THRESH else "black",
        )
        ax.set_ylabel(ylabel, fontsize=10)
        ax.legend(fontsize=8, loc="upper left")
        ax.tick_params(axis="x", labelbottom=False)

        # KDE posteriore di τ
        base_num = mdates.date2num(df_ev.index[0])
        tau_num  = base_num + ci["tau_post"] * 7
        kde_fn   = gaussian_kde(tau_num, bw_method=0.25)
        t_grid   = np.linspace(tau_num.min(), tau_num.max(), 400)
        density  = kde_fn(t_grid) / kde_fn(t_grid).max()
        axk.fill_between(mdates.num2date(t_grid), 0, density,
                         alpha=0.5, color="#2980b9")
        axk.axvline(cp_date, color="#2980b9", lw=2.0, ls="--")
        axk.axvspan(cp_lo, cp_hi, alpha=0.13, color="#2980b9")
        axk.axvline(shock, color=ev_color, lw=1.8, ls=":")
        axk.set_ylim(0, 1.6);  axk.set_yticks([])
        axk.set_ylabel("p(τ)", fontsize=9)
        axk.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
        axk.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
        axk.tick_params(axis="x", rotation=45, labelsize=9)

        safe_s = ser_name.lower()
        fpath  = f"plots/3_05_{safe_ev}_{safe_s}.png"
        fig.tight_layout(pad=1.5)
        fig.savefig(fpath, dpi=DPI, bbox_inches="tight")
        plt.close(fig)
        print(f"  ✓ {fpath}")

        # ── Plot diagnostica ──────────────────────────────────────────────
        fig_d, axes_d = plt.subplots(1, 3, figsize=(14, 4))
        fig_d.suptitle(
            f"Diagnostica OLS piecewise — {ev_name} | {ser_name}\n"
            "(motivano la scelta dei test in 3_02_tests.py)",
            fontsize=10, fontweight="bold")

        axes_d[0].scatter(diag["y_hat"], diag["resid"],
                          s=14, alpha=0.5, color="#3498db")
        axes_d[0].axhline(0, color="red", lw=1.5, ls="--")
        bp_s = (f"BP p={diag['BP_p']:.4f} "
                f"({'eterosch.' if not np.isnan(diag['BP_p']) and diag['BP_p'] < ALPHA else 'ok'})")
        axes_d[0].set_title(f"Residui vs Fitted\n{bp_s}", fontsize=9)

        stats.probplot(diag["resid"], dist="norm", plot=axes_d[1])
        sw_s = (f"SW p={diag['SW_p']:.4f} "
                f"({'non norm.' if not np.isnan(diag['SW_p']) and diag['SW_p'] < ALPHA else 'ok'})")
        axes_d[1].set_title(f"QQ Plot\n{sw_s}", fontsize=9)

        plot_acf(diag["resid"],
                 lags=min(20, len(diag["resid"]) // 2 - 1),
                 ax=axes_d[2], zero=False)
        dw_s = f"DW={diag['DW']:.3f} ({'autocorr.' if diag['DW'] < 1.5 else 'ok'})"
        axes_d[2].set_title(f"ACF Residui\n{dw_s}", fontsize=9)

        issues = []
        if not np.isnan(diag["BP_p"]) and diag["BP_p"] < ALPHA:
            issues.append("eteroschedasticità")
        if not np.isnan(diag["SW_p"]) and diag["SW_p"] < ALPHA:
            issues.append("non-normalità")
        if diag["DW"] < 1.5:
            issues.append(f"autocorrelazione (DW={diag['DW']:.2f})")
        footer = (f"Violazioni: {', '.join(issues)} → test non parametrici + HAC in 3_02"
                  if issues else "Ipotesi OLS soddisfatte")
        fig_d.text(0.5, -0.02, footer, ha="center", fontsize=9,
                   color="#c0392b" if issues else "#1a7a1a",
                   bbox=dict(boxstyle="round,pad=0.3", fc="#f8f8f8",
                             ec="#c0392b" if issues else "#1a7a1a", lw=1.2))
        fig_d.tight_layout(pad=1.5)
        diag_f = f"plots/3_05_{safe_ev}_{safe_s}_diag.png"
        fig_d.savefig(diag_f, dpi=DPI, bbox_inches="tight")
        plt.close(fig_d)
        print(f"  ✓ {diag_f}")


# ════════════════════════════════════════════════════════════════════════════
# 6. Grafico sommario: lag D per tutti gli eventi/serie
# ════════════════════════════════════════════════════════════════════════════
if table_rows:
    t_df = pd.DataFrame(table_rows)

    fig, ax = plt.subplots(figsize=(12, max(5, len(t_df) * 0.45 + 2)))
    colors_ev = {ev: cfg["color"] for ev, cfg in EVENTS.items()}

    y_labels = []
    for i, row in t_df.iterrows():
        lag  = row["Lag_gg"]
        col  = colors_ev.get(row["Evento"], "#555")
        fast = abs(lag) < LAG_THRESH
        ax.barh(i, lag, color=col, alpha=0.75,
                edgecolor="#333", linewidth=0.6, height=0.65)
        ax.text(lag + (3 if lag >= 0 else -3), i,
                f"{lag:+d}gg", va="center", ha="left" if lag >= 0 else "right",
                fontsize=8.5, color="#222",
                fontweight="bold" if fast else "normal")
        y_labels.append(f"{row['Evento'][:22]} | {row['Serie']}")

    ax.axvline(0, color="#333", lw=1.2)
    ax.axvspan(-LAG_THRESH, LAG_THRESH, alpha=0.07, color="#2ecc71",
               label=f"Trasmissione rapida (|D|<{LAG_THRESH}gg)")
    ax.set_yticks(range(len(t_df)));  ax.set_yticklabels(y_labels, fontsize=9)
    ax.set_xlabel("Lag D = τ_change_point − shock_date (giorni)\n"
                  "D < 0 → cambio prima dello shock (anticipazione futures)\n"
                  "D > 0 → cambio dopo lo shock (trasmissione ritardata)", fontsize=9)
    ax.set_title("Lag tra change point MCMC e shock geopolitico\n"
                 "per serie e evento", fontweight="bold", fontsize=12)
    ax.legend(fontsize=9);  ax.grid(axis="x", alpha=0.3)
    plt.tight_layout()
    fig.savefig("plots/3_05_summary.png", dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print("\n✓ plots/3_05_summary.png")


# ════════════════════════════════════════════════════════════════════════════
# 7. Salva output CSV
# ════════════════════════════════════════════════════════════════════════════
pd.DataFrame(table_rows).to_csv("data/3_cp.csv", index=False)
pd.DataFrame(diag_rows).to_csv("data/3_cp_diagnostics.csv", index=False)
print(f"\n✓ data/3_cp.csv  ({len(table_rows)} righe)")
print(f"✓ data/3_cp_diagnostics.csv  ({len(diag_rows)} righe)")

print("\nRiepilogo lag D (change point vs shock):")
for r in table_rows:
    print(f"  {r['Evento'][:30]} | {r['Serie']:12}: "
          f"D={r['Lag_gg']:+4d}gg  "
          f"τ={r['tau']}  "
          f"[{r['sampler']}]")


# ════════════════════════════════════════════════════════════════════════════
# 8. Stampa esplicita: Shock ufficiale → τ_Brent → τ_Benzina → τ_Diesel
#    Per ciascun evento mostra QUANDO si è mosso ogni livello della catena
#    Brent grezzo → prezzo alla pompa → crack spread (margine)
# ════════════════════════════════════════════════════════════════════════════
if table_rows:
    t_df = pd.DataFrame(table_rows)
    print(f"\n{'═'*72}")
    print("CRONOLOGIA CHANGEPOINT PER EVENTO")
    print("  (negativo = mercato anticipa; positivo = trasmissione ritardata)")
    print(f"{'═'*72}")

    SERIES_ORDER = ["Brent", "Benzina", "Diesel", "Crack_Benz", "Crack_Dies"]
    SERIES_LABEL = {
        "Brent":      "Brent (grezzo EUR/bbl)",
        "Benzina":    "Benzina (pompa IT)",
        "Diesel":     "Diesel  (pompa IT)",
        "Crack_Benz": "Crack Benzina (margine reale)",
        "Crack_Dies": "Crack Diesel  (margine reale)",
    }

    for ev_name, ev_cfg in EVENTS.items():
        shock_date = pd.Timestamp(ev_cfg["shock"])
        ev_rows = t_df[t_df["Evento"] == ev_name]

        print(f"\n  ┌─ {ev_name}")
        print(f"  │  Shock ufficiale : {shock_date.date()}")
        print(f"  │")

        for serie in SERIES_ORDER:
            row = ev_rows[ev_rows["Serie"] == serie]
            if row.empty:
                continue
            r      = row.iloc[0]
            lag    = int(r["Lag_gg"])
            tau    = r["tau"]
            ci_lo  = r["CI_95_lo"]
            ci_hi  = r["CI_95_hi"]
            rhat   = r.get("rhat_max", float("nan"))
            conv   = ""
            if not (isinstance(rhat, float) and np.isnan(rhat)):
                if float(rhat) > 1.05:
                    conv = f"  ⚠ Rhat={float(rhat):.3f}"
                elif float(rhat) > 1.01:
                    conv = f"  ~ Rhat={float(rhat):.3f}"
            sign   = "⬆ ANTICIPA" if lag < -LAG_THRESH else ("≈ SINCRONO" if abs(lag) <= LAG_THRESH else "⬇ RITARDA")
            label  = SERIES_LABEL.get(serie, serie)
            print(f"  │  τ {label:<32}: {str(tau):<12}  "
                  f"D={lag:+4d}gg  CI95=[{ci_lo}–{ci_hi}]  {sign}{conv}")

        print(f"  └{'─'*65}")

    print(f"\n  Legenda lag D = τ − shock_ufficiale:")
    print(f"    D << 0 → futures prezzavano l'evento prima (mercato forward-looking)")
    print(f"    D ≈ 0  → trasmissione rapida (entro {LAG_THRESH}gg)")
    print(f"    D >> 0 → margine si è allargato DOPO lo shock (analizzare perché)\n")


# Reset stile matplotlib
plt.rcParams.update(plt.rcParamsDefault)
print("\nScript 3_05 completato.")