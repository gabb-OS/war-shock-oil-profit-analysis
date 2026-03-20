"""
02d_v8_pymc.py  ─  Metodo 8: Bayesian ITS con PyMC (AR(1) Student-T)
======================================================================
Stima completamente bayesiana dell'extra-profitto speculativo sul margine
distributori (prezzo pompa netto − futures €/L) in corrispondenza di eventi
geopolitici.

Perché PyMC invece di BSTS/CausalImpact (V5):
  ─ V5 (BSTS) : richiede serie di controllo; senza di esse il modello state-
                space deriva liberamente nel post-periodo → stime instabili
  ─ V8 (PyMC) : modello ITS esplicito con priors interpretabili
                Baseline = proiezione lineare stimata SOLO sul pre-periodo
                Il posterior del guadagno è calcolato direttamente
                Non richiede serie di controllo

Modello generativo (pre-periodo):
  alpha  ~ Normal(mu_pre, 3 * std_pre)          [intercetta]
  beta   ~ Normal(0, 0.005)                      [trend giornaliero €/L]
  sigma  ~ HalfNormal(std_pre)                   [scala residui]
  rho    ~ Uniform(-0.95, 0.95)                  [autocorrelazione AR(1)]
  nu     ~ Exponential(1/30) + 2                 [gradi libertà Student-T, > 2]

  mu_t   = alpha + beta * t_pre_t                [media lineare]
  eps_0  ~ StudentT(nu, 0, sigma/√(1−ρ²))       [primo residuo, stazionario]
  eps_t  ~ StudentT(nu, ρ·eps_{t-1}, sigma)      [AR(1) Student-T]

Il processo AR(1) è implementato come Potential in PyMC — nessuna assunzione
sulla normalità dei residui, robusto agli outlier tipici dei margini carburante.

Posterior del guadagno:
  Per ogni campione MCMC (α_s, β_s):
    cf_s(t) = α_s + β_s · t_post             [baseline controfattuale campionata]
    gain_s  = Σ_t [(y_post_t − cf_s(t)) · cons_t] / 1e6  [M€]
  → distribuzione completa del guadagno, HDI come intervallo di credibilità

Output CSV compatibile con 02d_compare.py (stesse colonne standard).

Modalità (--mode):
  fixed     : break = data dello shock hardcodata               [default]
  detected  : break θ letto da theta_results.csv (02c)

Parametro --detect (solo mode=detected):
  margin  : usa θ rilevato sul margine distributore             [default]
  price   : usa θ rilevato sul prezzo alla pompa netto (€/L)

Argomenti aggiuntivi:
  --draws  N   : numero di campioni MCMC post-tuning [default=1000]
  --tune   N   : numero di step di tuning NUTS       [default=600]
  --chains N   : numero di catene                    [default=2]
  --no-progress: sopprime la progress bar PyMC

Output:
  data/plots/its/fixed/v8_pymc/                    (mode=fixed)
  data/plots/its/detected/{margin|price}/v8_pymc/  (mode=detected)
    plot_{evento}.png
    diag_{evento}_{carburante}.png
    v8_pymc_results.csv
"""

from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).parent / "utils"))
from conversions import GAS_OIL, EUROBOB as EUROBOB_HC, load_eurusd, usd_ton_to_eur_liter
from diagnostics import run_diagnostic_tests, plot_residual_diagnostics
from theta_loader import load_theta
from forecast_consumi import load_daily_consumption

# ── PyMC (obbligatorio per V8) ────────────────────────────────────────────────
try:
    import pymc as pm
    import pytensor.tensor as pt
    import arviz as az
    HAS_PYMC = True
except ImportError:
    HAS_PYMC = False
    warnings.warn(
        "PyMC non trovato — V8 non può girare.\n"
        "  Installa con: pip install pymc\n"
        "  (installa automaticamente anche pytensor e arviz)"
    )

# ── Configurazione ─────────────────────────────────────────────────────────────
BASE_DIR    = Path(__file__).parent
DAILY_CSV   = BASE_DIR / "data" / "processed" / "daily_fuel_prices_all.csv"
GASOIL_CSV  = BASE_DIR / "data" / "Futures" / "London Gas Oil Futures Historical Data.csv"
EUROBOB_CSV = BASE_DIR / "data" / "Futures" / "Eurobob_B7H1_date.csv"
EURUSD_CSV  = BASE_DIR / "data" / "raw" / "eurusd.csv"
_OUT_BASE   = BASE_DIR / "data" / "plots" / "its"

PRE_WIN   = 40      # giorni pre-break per stimare la baseline
POST_WIN  = 40      # giorni post-break per calcolare l'extra profitto
CI_ALPHA  = 0.05    # α → HDI al 95%
SEED      = 42

# Default MCMC (sovrascrivibili da CLI)
DEFAULT_DRAWS  = 1000
DEFAULT_TUNE   = 600
DEFAULT_CHAINS = 2

METHOD_COLOR = "#1a6f8a"   # blu-teal per distinguere da altri metodi

EVENTS: dict[str, dict] = {
    "Ucraina (Feb 2022)": {
        "shock": pd.Timestamp("2022-02-24"),
        "color": "#e74c3c",
        "label": "Russia-Ucraina (24 feb 2022)",
    },
    "Iran-Israele (Giu 2025)": {
        "shock": pd.Timestamp("2025-06-13"),
        "color": "#e67e22",
        "label": "Iran-Israele (13 giu 2025)",
    },
    "Hormuz (Feb 2026)": {
        "shock": pd.Timestamp("2026-02-28"),
        "color": "#8e44ad",
        "label": "Stretto di Hormuz (28 feb 2026)",
    },
}

FUELS: dict[str, tuple[str, str]] = {
    "benzina": ("margin_benzina", "#E63946"),
    "gasolio": ("margin_gasolio", "#1D3557"),
}


# ══════════════════════════════════════════════════════════════════════════════
# Caricamento dati  (identico agli altri metodi)
# ══════════════════════════════════════════════════════════════════════════════

def _load_gasoil_futures(eurusd: pd.Series) -> pd.Series:
    df = pd.read_csv(GASOIL_CSV, encoding="utf-8-sig", dtype=str)
    df["date"]  = pd.to_datetime(df["Date"], format="%m/%d/%Y", errors="coerce")
    df["price"] = (df["Price"].str.replace(",", "", regex=False)
                   .pipe(pd.to_numeric, errors="coerce"))
    df = df.dropna(subset=["date", "price"]).sort_values("date").set_index("date")
    return usd_ton_to_eur_liter(df["price"], eurusd, GAS_OIL)


def _load_eurobob_futures(eurusd: pd.Series) -> pd.Series | None:
    if not EUROBOB_CSV.exists():
        return None
    df = pd.read_csv(EUROBOB_CSV, encoding="utf-8-sig", dtype=str)
    _IT = {"gen":"Jan","feb":"Feb","mar":"Mar","apr":"Apr","mag":"May","giu":"Jun",
           "lug":"Jul","ago":"Aug","set":"Sep","ott":"Oct","nov":"Nov","dic":"Dec"}
    if "timestamp" in df.columns:
        ts = pd.to_numeric(df["timestamp"], errors="coerce")
        df["date"] = (pd.to_datetime(ts, unit="s", utc=True)
                      .dt.tz_localize(None).dt.normalize())
    else:
        def _parse(s):
            for it, en in _IT.items():
                s = s.replace(it, en)
            return pd.to_datetime(s, dayfirst=True, errors="coerce")
        df["date"] = df["data"].astype(str).apply(_parse)
    df["price"] = pd.to_numeric(df["chiusura"], errors="coerce")
    df = df.dropna(subset=["date", "price"]).sort_values("date").set_index("date")
    df = df[~df.index.duplicated(keep="first")]
    return usd_ton_to_eur_liter(df["price"], eurusd, EUROBOB_HC)


def load_margin_data() -> pd.DataFrame:
    daily = (pd.read_csv(DAILY_CSV, parse_dates=["date"])
               .sort_values("date").set_index("date"))
    eurusd  = load_eurusd(
        csv_path=EURUSD_CSV if EURUSD_CSV.exists() else None,
        start="2015-01-01", end="2026-12-31",
    )
    gasoil  = _load_gasoil_futures(eurusd)
    eurobob = _load_eurobob_futures(eurusd)
    df = daily[["benzina_net", "gasolio_net"]].copy()
    df["margin_gasolio"] = df["gasolio_net"] - gasoil.reindex(df.index, method="ffill")
    df["margin_benzina"] = (
        df["benzina_net"] - eurobob.reindex(df.index, method="ffill")
        if eurobob is not None else np.nan
    )
    return df


# ══════════════════════════════════════════════════════════════════════════════
# Modello PyMC — AR(1) Student-T sul pre-periodo
# ══════════════════════════════════════════════════════════════════════════════

def fit_pymc_its(
    pre_data:  pd.Series,
    break_date: pd.Timestamp,
    draws:     int,
    tune:      int,
    chains:    int,
    show_progress: bool,
) -> dict | None:
    """
    Stima bayesiana del trend lineare con errori AR(1) Student-T sul pre-periodo.

    Modello:
      alpha ~ Normal(media_pre, 3·std_pre)
      beta  ~ Normal(0, 0.005)
      sigma ~ HalfNormal(std_pre)
      rho   ~ Uniform(-0.95, 0.95)
      nu    ~ Exponential(1/30) + 2      [ν > 2: varianza finita garantita]

      mu_t  = alpha + beta * t_pre_t     [t in giorni dal break_date]
      eps_t = y_t - mu_t                 [residui]
      eps_0 ~ StudentT(ν, 0, σ/√(1-ρ²))
      eps_t ~ StudentT(ν, ρ·eps_{t-1}, σ)  t > 0

    Restituisce dict con trace ArviZ, parametri summary, e arrays pre-periodo.
    Restituisce None se PyMC non è disponibile o se fit fallisce.
    """
    if not HAS_PYMC:
        return None

    n = len(pre_data)
    if n < 10:
        return None

    y_arr = pre_data.values.astype(float)
    t_arr = np.array([(d - break_date).days for d in pre_data.index], dtype=float)

    mu_prior  = float(y_arr.mean())
    sig_prior = max(float(y_arr.std(ddof=1)), 1e-6)

    with pm.Model() as model:  # noqa: F841

        # ── Priors ────────────────────────────────────────────────────────────
        alpha = pm.Normal("alpha", mu=mu_prior,  sigma=sig_prior * 3)
        beta  = pm.Normal("beta",  mu=0.0,        sigma=0.005)
        sigma = pm.HalfNormal("sigma",             sigma=sig_prior)
        rho   = pm.Uniform("rho",  lower=-0.95,   upper=0.95)

        # ν > 2 garantisce varianza finita; Exp(1/30) → mediana ≈ 21 (robusto ma
        # non degenera verso normalità)
        nu_raw   = pm.Exponential("nu_raw", lam=1.0 / 30.0)
        nu_param = pm.Deterministic("nu", nu_raw + 2.0)

        # ── AR(1) Student-T likelihood via Potential ───────────────────────────
        mu_t  = alpha + beta * t_arr        # tensore (n,)
        resid = y_arr - mu_t               # tensore (n,) — y_arr è costante

        # Varianza stazionaria del processo AR(1)
        sigma_0 = sigma / pt.sqrt(1.0 - rho ** 2 + 1e-8)

        # Log-verosimiglianza primo punto (distribuzione stazionaria)
        ll_init = pm.logp(
            pm.StudentT.dist(nu=nu_param, mu=0.0, sigma=sigma_0),
            resid[0],
        )
        # Log-verosimiglianza punti successivi (condizionale AR(1))
        ll_cond = pm.logp(
            pm.StudentT.dist(nu=nu_param, mu=rho * resid[:-1], sigma=sigma),
            resid[1:],
        )
        pm.Potential("ar1_loglik", ll_init + pt.sum(ll_cond))

        # ── Campionamento NUTS ─────────────────────────────────────────────────
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                trace = pm.sample(
                    draws=draws,
                    tune=tune,
                    chains=chains,
                    target_accept=0.92,
                    progressbar=show_progress,
                    random_seed=SEED,
                    return_inferencedata=True,
                    idata_kwargs={"log_likelihood": False},
                )
            except Exception as exc:
                print(f"\n  ✗ PyMC campionamento fallito: {exc}")
                return None

    # ── Convergenza ───────────────────────────────────────────────────────────
    try:
        summary = az.summary(
            trace,
            var_names=["alpha", "beta", "sigma", "rho", "nu"],
            round_to=4,
        )
        rhat_max = float(summary["r_hat"].max())
        ess_min  = float(summary["ess_bulk"].min())
        converged = rhat_max <= 1.05 and ess_min >= 200
    except Exception:
        rhat_max  = np.nan
        ess_min   = np.nan
        converged = False

    if not converged:
        print(f"\n  ⚠ Convergenza PyMC dubbia: R̂_max={rhat_max:.3f}  ESS_min={ess_min:.0f}")

    # ── Estrai campioni posteriori ────────────────────────────────────────────
    post = trace.posterior
    alpha_s  = post["alpha"].values.reshape(-1)
    beta_s   = post["beta"].values.reshape(-1)
    sigma_s  = post["sigma"].values.reshape(-1)
    rho_s    = post["rho"].values.reshape(-1)
    nu_s     = post["nu"].values.reshape(-1)

    # Residui OLS (per diagnostica compatibile con pipeline)
    alpha_med  = float(np.median(alpha_s))
    beta_med   = float(np.median(beta_s))
    fitted_pre = alpha_med + beta_med * t_arr
    residuals  = y_arr - fitted_pre
    X_bg       = np.column_stack([np.ones(n), t_arr])

    # Pseudo-R² (varianza spiegata dal trend mediano)
    ss_res    = float(np.sum(residuals ** 2))
    ss_tot    = float(np.sum((y_arr - y_arr.mean()) ** 2))
    pseudo_r2 = 1.0 - ss_res / ss_tot if ss_tot > 1e-12 else 0.0

    return dict(
        trace       = trace,
        alpha_s     = alpha_s,
        beta_s      = beta_s,
        sigma_s     = sigma_s,
        rho_s       = rho_s,
        nu_s        = nu_s,
        alpha_med   = alpha_med,
        beta_med    = beta_med,
        pseudo_r2   = pseudo_r2,
        rhat_max    = rhat_max,
        ess_min     = ess_min,
        converged   = converged,
        break_date  = break_date,
        pre         = pre_data,
        x           = t_arr,
        x_mean      = float(t_arr.mean()),
        n           = n,
        residuals   = residuals,
        X_bg        = X_bg,
        n_samples   = len(alpha_s),
    )


def compute_posterior_gain(
    fit:       dict,
    post_data: pd.Series,
    cons:      pd.Series,
) -> tuple[float, float, float, np.ndarray, pd.Series, pd.Series, pd.Series]:
    """
    Calcola il posterior del guadagno cumulato (M€) e il counterfactual.

    Per ogni campione MCMC (α_s, β_s):
      cf_s(t) = α_s + β_s · t_post_t        [baseline controfattuale campionata]
      gain_s  = Σ_t [(y_post_t − cf_s(t)) · cons_t] / 1e6

    Restituisce:
      gain_median   : mediana posteriore del gain (M€)
      ci_low, ci_high : HDI 95%
      gain_samples  : distribuzione completa del posterior
      baseline      : controfatuale puntuale (mediana parametri)
      cf_ci_low, cf_ci_high : banda credibile sul baseline giornaliero
    """
    break_date = fit["break_date"]
    alpha_s    = fit["alpha_s"]    # (n_samples,)
    beta_s     = fit["beta_s"]

    t_post = np.array(
        [(d - break_date).days for d in post_data.index], dtype=float
    )
    y_post  = post_data.values.astype(float)
    cons_arr = cons.values.astype(float)
    n_s      = fit["n_samples"]

    # Matrice baseline: (n_samples, n_post_days)
    cf_matrix = alpha_s[:, None] + beta_s[:, None] * t_post[None, :]

    # Gain campionario — incertezza epistemica (solo parametri, no sigma)
    gain_samples = ((y_post[None, :] - cf_matrix) * cons_arr[None, :]).sum(axis=1) / 1e6

    # Stima puntuale: mediana posteriore
    gain_median = float(np.median(gain_samples))

    # HDI 95%
    try:
        hdi = az.hdi(gain_samples, hdi_prob=1.0 - CI_ALPHA)
        ci_low  = float(hdi[0])
        ci_high = float(hdi[1])
    except Exception:
        ci_low  = float(np.percentile(gain_samples, 100 * CI_ALPHA / 2))
        ci_high = float(np.percentile(gain_samples, 100 * (1 - CI_ALPHA / 2)))

    # Baseline puntuale (mediana parametri)
    baseline = pd.Series(
        fit["alpha_med"] + fit["beta_med"] * t_post,
        index=post_data.index,
    )

    # Banda credibile sul baseline (percentili del posterior predittivo della media)
    cf_ci_low  = pd.Series(
        np.percentile(cf_matrix, 100 * CI_ALPHA / 2, axis=0),
        index=post_data.index,
    )
    cf_ci_high = pd.Series(
        np.percentile(cf_matrix, 100 * (1 - CI_ALPHA / 2), axis=0),
        index=post_data.index,
    )

    return gain_median, ci_low, ci_high, gain_samples, baseline, cf_ci_low, cf_ci_high


# ══════════════════════════════════════════════════════════════════════════════
# Plot per singolo evento + carburante
# ══════════════════════════════════════════════════════════════════════════════

def _plot_event_fuel(
    ev_name:     str,
    ev:          dict,
    series:      pd.Series,
    fuel_key:    str,
    fuel_color:  str,
    fit:         dict,
    baseline:    pd.Series,
    cf_ci_low:   pd.Series,
    cf_ci_high:  pd.Series,
    extra:       pd.Series,
    gain_median: float,
    ci_low_gain: float,
    ci_high_gain: float,
    cons:        pd.Series,
    gain_samples: np.ndarray,
    break_date:  pd.Timestamp,
    mode:        str,
    ax_main:     plt.Axes,
    ax_gain:     plt.Axes,
    ax_post:     plt.Axes,
) -> None:
    shock = ev["shock"]

    win = series[
        (series.index >= shock - pd.Timedelta(days=PRE_WIN)) &
        (series.index <= shock + pd.Timedelta(days=POST_WIN))
    ].dropna()

    # ── Pannello 1: margine effettivo vs baseline bayesiana ───────────────────
    ax_main.plot(win.index, win.values, color=fuel_color, lw=1.0,
                 label=f"{fuel_key.capitalize()} effettivo")
    ax_main.plot(baseline.index, baseline.values, color=METHOD_COLOR, lw=1.5,
                 ls="--",
                 label=f"Baseline Bayes (R²={fit['pseudo_r2']:.2f})")
    ax_main.fill_between(
        baseline.index, cf_ci_low.values, cf_ci_high.values,
        alpha=0.18, color=METHOD_COLOR,
        label=f"Banda credibile {int((1-CI_ALPHA)*100)}% (parametri)",
    )
    ax_main.fill_between(
        extra.index,
        win.reindex(extra.index, fill_value=np.nan), baseline.values,
        where=(extra >= 0), alpha=0.22, color="green",
        label="Extra profitto (≥0)",
    )
    ax_main.fill_between(
        extra.index,
        win.reindex(extra.index, fill_value=np.nan), baseline.values,
        where=(extra < 0), alpha=0.22, color="red",
        label="Sotto-baseline (<0)",
    )
    ax_main.axvline(shock, color=ev["color"], lw=1.6, ls="--",
                    label=f"Shock ({shock.date()})")
    if mode == "detected" and break_date != shock:
        ax_main.axvline(break_date, color="black", lw=1.2, ls=":",
                        label=f"τ rilevato ({break_date.date()})")

    conv_str = (f"R̂={fit['rhat_max']:.3f}  ESS={fit['ess_min']:.0f}"
                if not np.isnan(fit.get("rhat_max", np.nan)) else "conv.?")
    mode_str = (f"Break=θ {break_date.date()} (GLM Poisson 02c)"
                if mode == "detected" else f"Break=shock ({shock.date()})")
    ax_main.set_title(
        f"[V8-PyMC / mode={mode}]  {fuel_key.capitalize()} – {ev_name}\n"
        f"{mode_str}  |  β={fit['beta_med']:+.5f} €/L/g  |  {conv_str}",
        fontsize=8, fontweight="bold",
    )
    ax_main.set_ylabel("Margine (€/L)", fontsize=8)
    ax_main.legend(fontsize=6, loc="upper left", ncol=2)
    ax_main.grid(axis="y", alpha=0.20)
    ax_main.xaxis.set_major_formatter(mdates.DateFormatter("%d %b %y"))
    ax_main.xaxis.set_major_locator(mdates.WeekdayLocator(byweekday=0, interval=2))
    plt.setp(ax_main.xaxis.get_majorticklabels(), rotation=35, ha="right", fontsize=7)

    # ── Pannello 2: guadagno cumulato ─────────────────────────────────────────
    cum = (extra * cons.values / 1e6).cumsum()
    ax_gain.plot(cum.index, cum.values, color=fuel_color, lw=1.2)
    ax_gain.axhline(0, color="grey", lw=0.7, ls="--")
    ax_gain.fill_between(cum.index, cum.values, 0,
                         where=(cum >= 0), alpha=0.25, color="green")
    ax_gain.fill_between(cum.index, cum.values, 0,
                         where=(cum < 0), alpha=0.25, color="red")
    ax_gain.axhline(ci_low_gain,  color=METHOD_COLOR, lw=0.9, ls=":",
                    label=f"HDI95% [{ci_low_gain:+.0f}, {ci_high_gain:+.0f}] M€")
    ax_gain.axhline(ci_high_gain, color=METHOD_COLOR, lw=0.9, ls=":")
    avg_cons_ml = cons.mean() / 1e6
    ax_gain.set_title(
        f"Guadagno extra cumulato → {gain_median:+.0f} M€  (mediana posteriore)\n"
        f"HDI95% [{ci_low_gain:+.0f}, {ci_high_gain:+.0f}] M€  "
        f"[cons. medio {avg_cons_ml:.1f} ML/g]",
        fontsize=7,
    )
    ax_gain.set_ylabel("M€ cumulati", fontsize=8)
    ax_gain.legend(fontsize=6)
    ax_gain.grid(axis="y", alpha=0.20)
    ax_gain.xaxis.set_major_formatter(mdates.DateFormatter("%d %b %y"))
    ax_gain.xaxis.set_major_locator(mdates.WeekdayLocator(byweekday=0, interval=2))
    plt.setp(ax_gain.xaxis.get_majorticklabels(), rotation=35, ha="right", fontsize=7)

    # ── Pannello 3: distribuzione posteriore del guadagno ─────────────────────
    ax_post.hist(gain_samples, bins=60, color=METHOD_COLOR, alpha=0.60,
                 edgecolor="white", linewidth=0.4, density=True)

    # KDE sovrapposta
    try:
        from scipy.stats import gaussian_kde
        kde = gaussian_kde(gain_samples)
        x_kde = np.linspace(gain_samples.min(), gain_samples.max(), 300)
        ax_post.plot(x_kde, kde(x_kde), color=METHOD_COLOR, lw=1.5)
    except Exception:
        pass

    ax_post.axvline(gain_median,  color="black",      lw=1.8, ls="-",
                    label=f"Mediana {gain_median:+.0f} M€")
    ax_post.axvline(ci_low_gain,  color=METHOD_COLOR, lw=1.2, ls="--",
                    label=f"HDI95% [{ci_low_gain:+.0f}, {ci_high_gain:+.0f}]")
    ax_post.axvline(ci_high_gain, color=METHOD_COLOR, lw=1.2, ls="--")
    ax_post.axvline(0, color="grey", lw=0.8, ls=":", alpha=0.7)

    # Probabilità posteriore che il guadagno > 0
    prob_pos = float((gain_samples > 0).mean())
    ax_post.set_xlabel("Guadagno (M€)", fontsize=7)
    ax_post.set_ylabel("Densità posteriore", fontsize=7)
    ax_post.set_title(
        f"Posterior gain  |  SD={float(gain_samples.std()):.1f} M€\n"
        f"P(gain > 0) = {prob_pos:.2%}",
        fontsize=7,
    )
    ax_post.legend(fontsize=6)
    ax_post.grid(axis="y", alpha=0.15)


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    if not HAS_PYMC:
        print("\n  ✗ PyMC non installato — impossibile eseguire V8.")
        print("    pip install pymc")
        sys.exit(1)

    parser = argparse.ArgumentParser(
        description="V8 Bayesian ITS (PyMC AR(1) Student-T) – ITS pipeline"
    )
    parser.add_argument(
        "--mode", choices=["fixed", "detected"], default="fixed",
        help="fixed = usa shock date hardcodata; detected = usa θ da 02c",
    )
    parser.add_argument(
        "--detect", choices=["margin", "price"], default="margin",
        help="(solo mode=detected) serie su cui è stata fatta detection",
    )
    parser.add_argument(
        "--draws", type=int, default=DEFAULT_DRAWS,
        help=f"Campioni MCMC post-tuning [default={DEFAULT_DRAWS}]",
    )
    parser.add_argument(
        "--tune", type=int, default=DEFAULT_TUNE,
        help=f"Step di tuning NUTS [default={DEFAULT_TUNE}]",
    )
    parser.add_argument(
        "--chains", type=int, default=DEFAULT_CHAINS,
        help=f"Numero di catene [default={DEFAULT_CHAINS}]",
    )
    parser.add_argument(
        "--no-progress", action="store_true",
        help="Sopprime la progress bar di PyMC",
    )
    args, _       = parser.parse_known_args()
    mode          = args.mode
    detect_target = args.detect
    draws         = args.draws
    tune          = args.tune
    chains        = args.chains
    show_progress = not args.no_progress

    if mode == "detected":
        OUT_DIR = _OUT_BASE / "detected" / detect_target / "v8_pymc"
    else:
        OUT_DIR = _OUT_BASE / mode / "v8_pymc"
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("═" * 70)
    print(f"  02d_v8_pymc.py  –  Metodo 8: Bayesian ITS  [mode={mode}]")
    if mode == "fixed":
        print("  Break = shock date hardcodata (nessuna detection)")
    else:
        print("  Break = θ GLM Poisson da 02c_change_point_detection.py")
        print(f"  Detection su: {'MARGINE distributore' if detect_target == 'margin' else 'PREZZO POMPA NETTO'}")
    print(f"  Finestra: PRE={PRE_WIN}gg / POST={POST_WIN}gg dal break point")
    print(f"  MCMC: draws={draws}  tune={tune}  chains={chains}  seed={SEED}")
    print(f"  Modello: AR(1) Student-T  |  HDI={int((1-CI_ALPHA)*100)}%")
    print(f"  Output: {OUT_DIR}")
    print("═" * 70)

    # Sopprime output numerici PyMC/pytensor al log di root
    import logging
    logging.getLogger("pymc").setLevel(logging.ERROR)
    logging.getLogger("pytensor").setLevel(logging.ERROR)

    data = load_margin_data()
    rows: list[dict] = []

    for ev_name, ev in EVENTS.items():
        shock = ev["shock"]

        fig, axes = plt.subplots(
            len(FUELS), 3,
            figsize=(18, 5 * len(FUELS)),
            squeeze=False,
        )
        fig.suptitle(
            f"[Metodo 8 – Bayesian ITS PyMC AR(1) / mode={mode}]  {ev_name}\n"
            f"{ev['label']}",
            fontsize=11, fontweight="bold",
        )

        for row_idx, (fuel_key, (col_name, fuel_color)) in enumerate(FUELS.items()):
            series = data[col_name].dropna()

            # ── Break date ────────────────────────────────────────────────────
            if mode == "detected":
                theta = load_theta(ev_name, fuel_key, detect_target,
                                   base_dir=BASE_DIR)
                if theta is not None:
                    break_date   = theta
                    break_method = "glm_poisson_02c"
                else:
                    print(f"  ⚠ [{fuel_key}] θ non trovato — uso shock come fallback.")
                    break_date   = shock
                    break_method = "fallback_shock"
            else:
                break_date   = shock
                break_method = "fixed_at_shock"

            # ── Finestre dati ─────────────────────────────────────────────────
            pre_data = series[
                (series.index >= break_date - pd.Timedelta(days=PRE_WIN)) &
                (series.index <  break_date)
            ]
            post_data = series[
                (series.index >= break_date) &
                (series.index <  shock + pd.Timedelta(days=POST_WIN))
            ]

            if len(pre_data) < 10 or len(post_data) < 5:
                print(f"  [{fuel_key}] dati insufficienti – salto.")
                for ax in axes[row_idx]:
                    ax.text(0.5, 0.5, "Dati insufficienti",
                            ha="center", va="center", transform=ax.transAxes)
                continue

            # ── Fit PyMC ──────────────────────────────────────────────────────
            print(f"\n  [{ev_name}]  [{fuel_key.upper()}]  campionamento MCMC ...",
                  flush=True)
            fit = fit_pymc_its(
                pre_data=pre_data,
                break_date=break_date,
                draws=draws,
                tune=tune,
                chains=chains,
                show_progress=show_progress,
            )
            if fit is None:
                print(f"  ✗ [{fuel_key}] fit PyMC fallito — salto.")
                for ax in axes[row_idx]:
                    ax.text(0.5, 0.5, "PyMC fit fallito",
                            ha="center", va="center", transform=ax.transAxes)
                continue

            # ── Consumi ───────────────────────────────────────────────────────
            cons = load_daily_consumption(post_data.index, fuel_key)

            # ── Posterior del guadagno ────────────────────────────────────────
            (gain_median, ci_low_gain, ci_high_gain,
             gain_samples, baseline, cf_ci_low, cf_ci_high) = compute_posterior_gain(
                fit, post_data, cons
            )

            extra = post_data - baseline

            # ── Diagnostica residui pre (compatibilità pipeline) ──────────────
            pre_resid = fit["residuals"]
            diag = run_diagnostic_tests(
                pre_resid,
                x_for_bg=fit["X_bg"],
                n_lags=None,
            )

            safe_ev   = (ev_name.replace(" ", "_").replace("/", "")
                                .replace("(", "").replace(")", ""))
            diag_path = OUT_DIR / f"diag_{safe_ev}_{fuel_key}.png"
            plot_residual_diagnostics(
                resid      = pre_resid,
                dates      = fit["pre"].index,
                title      = (f"[V8-PyMC] Diagnostica residui pre-periodo\n"
                              f"{ev_name} · {fuel_key.capitalize()}  "
                              f"(break={break_date.date()})"),
                out_path   = diag_path,
                diag_stats = diag,
            )

            # ── Plot ──────────────────────────────────────────────────────────
            _plot_event_fuel(
                ev_name, ev, series, fuel_key, fuel_color,
                fit, baseline, cf_ci_low, cf_ci_high,
                extra, gain_median, ci_low_gain, ci_high_gain,
                cons, gain_samples, break_date, mode,
                axes[row_idx][0], axes[row_idx][1], axes[row_idx][2],
            )

            # ── Stampa a video ────────────────────────────────────────────────
            prob_pos = float((gain_samples > 0).mean())
            print(f"    Break ({break_method}) = {break_date.date()}  "
                  f"(shock={shock.date()})")
            print(f"    α (intercetta)   = {fit['alpha_med']:+.5f} €/L  "
                  f"σ={float(np.std(fit['alpha_s'])):.5f}")
            print(f"    β (trend)        = {fit['beta_med']:+.6f} €/L/g  "
                  f"σ={float(np.std(fit['beta_s'])):.6f}")
            print(f"    ρ (AR1 mediana)  = {float(np.median(fit['rho_s'])):+.3f}")
            print(f"    ν (StudentT med) = {float(np.median(fit['nu_s'])):.1f}  "
                  f"[>> 2: {'robusto' if float(np.median(fit['nu_s'])) > 10 else '⚠ code pesanti'}]")
            print(f"    Pseudo-R²        = {fit['pseudo_r2']:.3f}")
            print(f"    R̂_max            = {fit['rhat_max']:.3f}  "
                  f"ESS_min = {fit['ess_min']:.0f}  "
                  f"[{'✓ converge' if fit['converged'] else '⚠ non converge'}]")
            print(f"    Extra medio      = {extra.mean():+.4f} €/L/g")
            print(f"    Guadagno         = {gain_median:+.0f} M€  "
                  f"HDI95% [{ci_low_gain:+.0f}, {ci_high_gain:+.0f}] M€  "
                  f"SD={float(gain_samples.std()):.1f} M€")
            print(f"    P(guadagno > 0)  = {prob_pos:.2%}")

            if not np.isnan(diag.get("sw_p", np.nan)):
                print(f"    SW residui  W={diag['sw_stat']:.3f}  p={diag['sw_p']:.3f}  "
                      f"{'OK' if diag['sw_p'] > 0.05 else '⚠ non norm.'}")
            if not np.isnan(diag.get("lb_p", np.nan)):
                print(f"    LB({diag['n_lags']}) autocorr.  "
                      f"Q={diag['lb_stat']:.2f}  p={diag['lb_p']:.3f}  "
                      f"{'OK' if diag['lb_p'] > 0.05 else '⚠ autocorr.'}")

            # ── Export residui pre/post (standard per 02d_compare nonparam) ──
            _resid_rows = []
            for _d, _r in zip(fit["pre"].index, fit["residuals"]):
                _resid_rows.append({
                    "date": str(_d.date()), "residual": float(_r), "phase": "pre",
                    "metodo": "v8_pymc", "evento": ev_name,
                    "carburante": fuel_key, "break_date": str(break_date.date()),
                })
            for _d, _r in zip(post_data.index, extra.values):
                _resid_rows.append({
                    "date": str(_d.date()), "residual": float(_r), "phase": "post",
                    "metodo": "v8_pymc", "evento": ev_name,
                    "carburante": fuel_key, "break_date": str(break_date.date()),
                })
            pd.DataFrame(_resid_rows).to_csv(
                OUT_DIR / f"residuals_{safe_ev}_{fuel_key}.csv", index=False
            )

            # ── Record CSV (colonne standard per compare.py + V8-specifiche) ──
            rows.append({
                # ── Colonne standard compare.py ────────────────────────────────
                "metodo":             "v8_pymc",
                "mode":               mode,
                "detect_target":      detect_target if mode == "detected" else "fixed",
                "evento":             ev_name,
                "carburante":         fuel_key,
                "shock":              shock.date(),
                "break_date":         break_date.date(),
                "break_method":       break_method,
                "pre_win_days":       PRE_WIN,
                "post_win_days":      POST_WIN,
                "n_pre":              len(pre_data),
                "n_post":             len(post_data),
                "pre_std_eurl":       round(float(pre_data.std(ddof=1)), 6),
                "extra_mean_eurl":    round(float(extra.mean()), 5),
                "gain_total_meur":    round(gain_median,    1),
                "gain_ci_low_meur":   round(ci_low_gain,   1),
                "gain_ci_high_meur":  round(ci_high_gain,  1),
                "ci_type":            f"HDI_bayesian_{int((1-CI_ALPHA)*100)}pct",
                # ── Parametri posteriori ───────────────────────────────────────
                "alpha_med":          round(fit["alpha_med"],                     5),
                "alpha_sd":           round(float(np.std(fit["alpha_s"])),        5),
                "beta_med":           round(fit["beta_med"],                      6),
                "beta_sd":            round(float(np.std(fit["beta_s"])),         6),
                "sigma_med":          round(float(np.median(fit["sigma_s"])),     5),
                "rho_med":            round(float(np.median(fit["rho_s"])),       4),
                "nu_med":             round(float(np.median(fit["nu_s"])),        2),
                "pseudo_r2":          round(fit["pseudo_r2"],                     4),
                # ── Posterior del guadagno ─────────────────────────────────────
                "gain_sd_meur":       round(float(gain_samples.std()),            2),
                "prob_gain_pos":      round(float((gain_samples > 0).mean()),     4),
                "n_mcmc_samples":     fit["n_samples"],
                # ── Convergenza MCMC ───────────────────────────────────────────
                "rhat_max":           round(fit["rhat_max"],                      4),
                "ess_min":            round(fit["ess_min"],                       0),
                "converged":          fit["converged"],
                # ── Diagnostica residui pre ────────────────────────────────────
                "sw_stat":            round(diag.get("sw_stat", np.nan),          4),
                "sw_p":               round(diag.get("sw_p",    np.nan),          4),
                "lb_stat":            round(diag.get("lb_stat", np.nan),          3),
                "lb_p":               round(diag.get("lb_p",    np.nan),          4),
                "bg_stat":            round(diag.get("bg_stat", np.nan),          3),
                "bg_p":               round(diag.get("bg_p",    np.nan),          4),
                "diag_n_lags":        diag.get("n_lags", np.nan),
                "note": (
                    f"Bayesian ITS PyMC AR(1) Student-T, mode={mode}"
                    + (f", detect={detect_target}" if mode == "detected" else "")
                    + f", draws={draws}, tune={tune}, chains={chains}"
                    + (f", R_hat={fit['rhat_max']:.3f}" if not np.isnan(fit["rhat_max"]) else "")
                ),
            })

        fig.tight_layout()
        safe = (ev_name.replace(" ", "_").replace("/", "")
                .replace("(", "").replace(")", ""))
        out  = OUT_DIR / f"plot_{safe}.png"
        fig.savefig(out, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"\n  → Salvato: {out}")

    if rows:
        df      = pd.DataFrame(rows)
        csv_out = OUT_DIR / "v8_pymc_results.csv"
        df.to_csv(csv_out, index=False)
        print(f"\n  → CSV: {csv_out}")
        print(
            "\n"
            + df[[
                "evento", "carburante", "break_date",
                "gain_total_meur", "gain_ci_low_meur", "gain_ci_high_meur",
                "prob_gain_pos", "rho_med", "nu_med",
                "rhat_max", "converged",
            ]].to_string(index=False)
        )
    else:
        print("\n  ⚠ Nessun risultato prodotto.")


if __name__ == "__main__":
    main()