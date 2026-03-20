"""
utils/diagnostics.py
────────────────────
Utility condivisa per la diagnostica dei residui nei modelli ITS.

Funzioni esportate
──────────────────
run_diagnostic_tests(resid, x_for_bg=None, n_lags=None)
    Calcola Shapiro-Wilk, Ljung-Box (LB), Breusch-Godfrey (BG opzionale).
    → dict con sw_stat, sw_p, lb_stat, lb_p, bg_stat, bg_p, n, n_lags

fit_sarima_benchmark(pre_series, n_steps, s=12)
    Fitta SARIMA(0,1,1)(0,1,0)_s su pre_series, proietta n_steps in avanti.
    → dict con fit, resid, forecast, ci_lo, ci_hi, aic, bic, gain_fn

plot_residual_diagnostics(resid, dates, title, out_path,
                          diag_stats=None, n_lags=None)
    Figura 4-panel (modello primario):
      (a) Serie dei residui nel tempo
      (b) ACF dei residui
      (c) PACF dei residui
      (d) Istogramma residui standardizzati + curva N(0,1)

plot_sarima_diagnostics(resid, dates, title, out_path, diag_stats=None)
    Figura 3-panel (benchmark SARIMA, come da Masena & Shongwe 2024):
      (a) Serie dei residui
      (b) ACF
      (c) Istogramma residui standardizzati + N(0,1)
"""

from __future__ import annotations
import warnings
import numpy as np
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# ── Dipendenze opzionali ───────────────────────────────────────────────────────
try:
    from scipy import stats as _sp
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

try:
    from statsmodels.stats.diagnostic import acorr_ljungbox, acorr_breusch_godfrey
    from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
    from statsmodels.tsa.statespace.sarimax import SARIMAX
    import statsmodels.api as sm
    HAS_SM = True
except ImportError:
    HAS_SM = False


# ══════════════════════════════════════════════════════════════════════════════
# Diagnostic tests
# ══════════════════════════════════════════════════════════════════════════════

def run_diagnostic_tests(
    resid,
    x_for_bg: np.ndarray | None = None,
    n_lags: int | None = None,
) -> dict:
    """
    Calcola un pannello di test sui residui:

    · Shapiro-Wilk  (SW) — normalità
    · Ljung-Box     (LB) — autocorrelazione
    · Breusch-Godfrey (BG) — autocorrelazione nei residui OLS
                             (richiede x_for_bg = matrice X del modello)

    Parametri
    ---------
    resid     : 1D array-like dei residui del modello
    x_for_bg  : matrice di design per BG (n_obs × k), include costante.
                Se None, il BG viene saltato (es. per residui ARIMA).
    n_lags    : numero di lag per LB e BG; default = min(n//4, 20, 1+)

    Restituisce
    -----------
    dict con chiavi:
      n, n_lags,
      sw_stat, sw_p,
      lb_stat, lb_p,
      bg_stat, bg_p    (NaN se x_for_bg=None)
    """
    resid = np.asarray(resid, dtype=float)
    resid = resid[~np.isnan(resid)]
    n = len(resid)

    if n_lags is None:
        n_lags = max(1, min(n // 4, 20))

    out = {
        "n": n,
        "n_lags": n_lags,
        "sw_stat": np.nan, "sw_p": np.nan,
        "lb_stat": np.nan, "lb_p": np.nan,
        "bg_stat": np.nan, "bg_p": np.nan,
    }

    # ── Shapiro-Wilk ──────────────────────────────────────────────────────────
    if HAS_SCIPY and n >= 3:
        try:
            sw_s, sw_p = _sp.shapiro(resid[:5000])
            out["sw_stat"] = float(sw_s)
            out["sw_p"]    = float(sw_p)
        except Exception:
            pass

    # ── Ljung-Box ─────────────────────────────────────────────────────────────
    if HAS_SM and n > n_lags:
        try:
            lb = acorr_ljungbox(resid, lags=[n_lags], return_df=True)
            out["lb_stat"] = float(lb["lb_stat"].iloc[-1])
            out["lb_p"]    = float(lb["lb_pvalue"].iloc[-1])
        except Exception:
            pass

    # ── Breusch-Godfrey (solo OLS) ───────────────────────────────────────────
    if HAS_SM and x_for_bg is not None and n > n_lags:
        try:
            # Ricostruisce il modello OLS per BG
            ols_fit = sm.OLS(resid, x_for_bg).fit()
            # BG usa il modello OLS ausiliario internamente
            bg = acorr_breusch_godfrey(ols_fit, nlags=n_lags)
            out["bg_stat"] = float(bg[0])
            out["bg_p"]    = float(bg[1])
        except Exception:
            pass

    return out


# ══════════════════════════════════════════════════════════════════════════════
# SARIMA(0,1,1)(0,1,0)_s benchmark
# ══════════════════════════════════════════════════════════════════════════════

def fit_sarima_benchmark(
    pre_series,
    n_steps: int,
    s: int = 12,
    alpha: float = 0.10,
    max_iter: int = 300,
) -> dict | None:
    """
    Fitta SARIMA(0,1,1)(0,1,0)_s sulla serie pre-intervento e proietta
    n_steps periodi in avanti (controfattuale per il post-intervento).

    Modello "airline-type" stagionale. Per serie giornaliere con s=12,
    cattura un ciclo bimensile. Usato come benchmark rispetto ad auto-ARIMA.

    Restituisce None se statsmodels non è disponibile o il fit fallisce.
    """
    if not HAS_SM:
        return None

    pre = np.asarray(pre_series, dtype=float)
    n   = len(pre)

    # Serve almeno 2*s + 2 punti per seasonal diff + MA
    if n < max(2 * s + 2, 15):
        return None

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            m   = SARIMAX(
                pre,
                order=(0, 1, 1),
                seasonal_order=(0, 1, 0, s),
                trend="n",
                enforce_stationarity=False,
                enforce_invertibility=False,
            )
            fit = m.fit(disp=False, maxiter=max_iter, method="lbfgs")

        fc   = fit.get_forecast(steps=n_steps)
        mean = fc.predicted_mean
        ci   = fc.conf_int(alpha=alpha)

        resid_arr = np.asarray(fit.resid, dtype=float)

        return {
            "fit":      fit,
            "resid":    resid_arr,
            "forecast": np.asarray(mean),
            "ci_lo":    ci.iloc[:, 0].values if hasattr(ci, "iloc") else ci[:, 0],
            "ci_hi":    ci.iloc[:, 1].values if hasattr(ci, "iloc") else ci[:, 1],
            "aic":      float(fit.aic),
            "bic":      float(fit.bic),
            "order":    f"SARIMA(0,1,1)(0,1,0)_{s}",
        }
    except Exception as e:
        warnings.warn(f"SARIMA benchmark fit fallito: {e}")
        return None


# ══════════════════════════════════════════════════════════════════════════════
# Plot helper: format date axis
# ══════════════════════════════════════════════════════════════════════════════

def _fmt_date_ax(ax, dates=None):
    """Applica formatter date se disponibile."""
    if dates is None:
        return
    try:
        import matplotlib.dates as mdates
        import pandas as pd
        span_days = (max(dates) - min(dates)).days if len(dates) > 1 else 0
        if span_days <= 90:
            ax.xaxis.set_major_locator(mdates.WeekdayLocator(byweekday=0, interval=2))
        else:
            ax.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%d %b %y"))
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=30, ha="right", fontsize=6.5)
    except Exception:
        pass


def _stat_annotation(ax, diag_stats: dict, include_bg: bool = True):
    """Aggiunge box con statistiche dei test nell'angolo in alto a sinistra."""
    if diag_stats is None:
        return

    def _fmt(stat, p):
        if np.isnan(stat):
            return "N/A"
        sig = "✓" if p > 0.05 else "✗"
        return f"{stat:.3f}  (p={p:.3f}) {sig}"

    sw_ok = not np.isnan(diag_stats.get("sw_stat", np.nan))
    lb_ok = not np.isnan(diag_stats.get("lb_stat", np.nan))
    bg_ok = include_bg and not np.isnan(diag_stats.get("bg_stat", np.nan))

    lines = []
    if sw_ok:
        lines.append(f"SW  W={_fmt(diag_stats['sw_stat'], diag_stats['sw_p'])}")
    if lb_ok:
        lines.append(f"LB({diag_stats.get('n_lags','?')})  Q={_fmt(diag_stats['lb_stat'], diag_stats['lb_p'])}")
    if bg_ok:
        lines.append(f"BG({diag_stats.get('n_lags','?')})  LM={_fmt(diag_stats['bg_stat'], diag_stats['bg_p'])}")
    if not lines:
        return

    lines.append("─ ✓=no rej. H₀  ✗=reject")
    ax.text(
        0.02, 0.98, "\n".join(lines),
        transform=ax.transAxes, fontsize=6.2, va="top", ha="left",
        family="monospace",
        bbox=dict(boxstyle="round,pad=0.35", fc="lightyellow", alpha=0.85, lw=0.5),
    )


# ══════════════════════════════════════════════════════════════════════════════
# QQ plot helper
# ══════════════════════════════════════════════════════════════════════════════

def _qq_plot(ax, std_rv: np.ndarray, color: str = "steelblue") -> None:
    """
    Normal QQ plot dei residui standardizzati.
    Usa scipy.stats.probplot se disponibile, altrimenti calcolo manuale.
    """
    if HAS_SCIPY:
        (osm, osr), (slope, intercept, _r) = _sp.probplot(std_rv, dist="norm")
        ax.scatter(osm, osr, color=color, s=12, alpha=0.7, zorder=3,
                   label="Quantili campione")
        # Linea di riferimento N(0,1)
        lo, hi = osm[0], osm[-1]
        ax.plot([lo, hi], [slope * lo + intercept, slope * hi + intercept],
                color="red", lw=1.3, ls="--", zorder=4, label="Linea N(0,1)")
    else:
        # Fallback manuale
        n  = len(std_rv)
        ps = (np.arange(1, n + 1) - 0.5) / n
        th = np.sort(std_rv)
        q_th = np.interp(ps, np.linspace(0, 1, n), np.sort(std_rv))
        ax.scatter(ps, q_th, color=color, s=12, alpha=0.7)
        ax.plot([0, 1], [0, 1], color="red", lw=1.3, ls="--")

    ax.set_xlabel("Quantili teorici N(0,1)", fontsize=7)
    ax.set_ylabel("Quantili campione", fontsize=7)
    ax.legend(fontsize=6.5, loc="upper left")
    ax.grid(alpha=0.20)


# ══════════════════════════════════════════════════════════════════════════════
# 5-panel diagnostics (primary model)
# ══════════════════════════════════════════════════════════════════════════════

def plot_residual_diagnostics(
    resid,
    dates,
    title: str,
    out_path,
    diag_stats: dict | None = None,
    n_lags: int | None = None,
) -> None:
    """
    Figura diagnostica a 5 pannelli per il modello primario (layout 2×3):
      (a) row0-col0 — Serie dei residui nel tempo
      (b) row0-col1 — ACF dei residui
      (c) row0-col2 — PACF dei residui
      (d) row1-col0 — Istogramma residui standardizzati + N(0,1)
      (e) row1-col1 — QQ plot residui standardizzati vs N(0,1)
      [row1-col2 vuoto — spazio per eventuale espansione]

    Parametri
    ---------
    resid      : 1D array residui
    dates      : indice temporale (DatetimeIndex o None)
    title      : titolo del grafico
    out_path   : percorso output (Path o str)
    diag_stats : dict da run_diagnostic_tests() (opzionale)
    n_lags     : lag per ACF/PACF; default min(n//4, 20)
    """
    resid = np.asarray(resid, dtype=float)
    mask  = ~np.isnan(resid)
    rv    = resid[mask]
    n     = len(rv)
    if n < 4:
        return

    if n_lags is None:
        n_lags = max(1, min(n // 4, 20))

    std_rv = (rv - rv.mean()) / (rv.std() + 1e-12)
    try:
        import pandas as pd
        dv = pd.DatetimeIndex(dates)[mask] if dates is not None else None
    except Exception:
        dv = None

    fig = plt.figure(figsize=(18, 8))
    gs  = gridspec.GridSpec(2, 3, figure=fig, hspace=0.50, wspace=0.33)
    ax_ts   = fig.add_subplot(gs[0, 0])
    ax_acf  = fig.add_subplot(gs[0, 1])
    ax_pacf = fig.add_subplot(gs[0, 2])
    ax_hist = fig.add_subplot(gs[1, 0])
    ax_qq   = fig.add_subplot(gs[1, 1])

    fig.suptitle(title, fontsize=9, fontweight="bold", y=1.00)

    # (a) Residual series ─────────────────────────────────────────────────────
    if dv is not None:
        ax_ts.plot(dv, rv, color="steelblue", lw=0.85, alpha=0.85)
    else:
        ax_ts.plot(rv, color="steelblue", lw=0.85, alpha=0.85)
    ax_ts.axhline(0, color="black", lw=0.7, ls="--")
    ax_ts.set_title("(a) Residui nel tempo", fontsize=8)
    ax_ts.set_ylabel("Residuo", fontsize=7)
    ax_ts.grid(axis="y", alpha=0.20)
    _fmt_date_ax(ax_ts, dv)
    _stat_annotation(ax_ts, diag_stats)

    # (b) ACF ─────────────────────────────────────────────────────────────────
    if HAS_SM and n > n_lags + 1:
        try:
            plot_acf(rv, lags=n_lags, ax=ax_acf, alpha=0.05,
                     color="steelblue", vlines_kwargs={"colors": "steelblue"})
        except Exception:
            ax_acf.text(0.5, 0.5, "ACF N/A", ha="center", va="center",
                        transform=ax_acf.transAxes, fontsize=9)
    else:
        ax_acf.text(0.5, 0.5, "ACF N/A", ha="center", va="center",
                    transform=ax_acf.transAxes, fontsize=9)
    ax_acf.set_title(f"(b) ACF dei residui (lag={n_lags})", fontsize=8)
    ax_acf.set_xlabel("Lag", fontsize=7)
    ax_acf.grid(axis="y", alpha=0.20)

    # (c) PACF ────────────────────────────────────────────────────────────────
    if HAS_SM and n > n_lags + 1:
        try:
            plot_pacf(rv, lags=n_lags, ax=ax_pacf, alpha=0.05,
                      method="ywm",
                      color="darkorange", vlines_kwargs={"colors": "darkorange"})
        except Exception:
            ax_pacf.text(0.5, 0.5, "PACF N/A", ha="center", va="center",
                         transform=ax_pacf.transAxes, fontsize=9)
    else:
        ax_pacf.text(0.5, 0.5, "PACF N/A", ha="center", va="center",
                     transform=ax_pacf.transAxes, fontsize=9)
    ax_pacf.set_title(f"(c) PACF dei residui (lag={n_lags})", fontsize=8)
    ax_pacf.set_xlabel("Lag", fontsize=7)
    ax_pacf.grid(axis="y", alpha=0.20)

    # (d) Histogram of standardised residuals ─────────────────────────────────
    bins = max(10, n // 5)
    ax_hist.hist(std_rv, bins=bins, density=True,
                 color="steelblue", alpha=0.6, edgecolor="white",
                 label="Residui std.")
    xx = np.linspace(std_rv.min() - 0.5, std_rv.max() + 0.5, 300)
    if HAS_SCIPY:
        ax_hist.plot(xx, _sp.norm.pdf(xx), color="red", lw=1.5, label="N(0,1)")
    ax_hist.set_title("(d) Istogramma residui std.", fontsize=8)
    ax_hist.set_xlabel("Residuo standardizzato", fontsize=7)
    ax_hist.legend(fontsize=7)
    ax_hist.grid(axis="y", alpha=0.20)

    # (e) QQ plot ─────────────────────────────────────────────────────────────
    ax_qq.set_title("(e) QQ plot vs N(0,1)", fontsize=8)
    _qq_plot(ax_qq, std_rv, color="steelblue")

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"    → Diag plot: {out_path.name}")


# ══════════════════════════════════════════════════════════════════════════════
# 4-panel SARIMA benchmark diagnostics (come Masena & Shongwe 2024 + QQ)
# ══════════════════════════════════════════════════════════════════════════════

def plot_sarima_diagnostics(
    resid,
    dates,
    title: str,
    out_path,
    diag_stats: dict | None = None,
) -> None:
    """
    Figura diagnostica a 4 pannelli per il modello SARIMA benchmark (2×2):
      (a) top-left  — Serie dei residui
      (b) top-right — ACF dei residui
      (c) bot-left  — Istogramma residui standardizzati + N(0,1)
      (d) bot-right — QQ plot residui standardizzati vs N(0,1)
    """
    resid = np.asarray(resid, dtype=float)
    mask  = ~np.isnan(resid)
    rv    = resid[mask]
    n     = len(rv)
    if n < 4:
        return

    n_lags = max(1, min(n // 4, 20))
    std_rv = (rv - rv.mean()) / (rv.std() + 1e-12)

    try:
        import pandas as pd
        dv = pd.DatetimeIndex(dates)[mask] if dates is not None else None
    except Exception:
        dv = None

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle(title, fontsize=9, fontweight="bold", y=1.00)
    ax_ts   = axes[0, 0]
    ax_acf  = axes[0, 1]
    ax_hist = axes[1, 0]
    ax_qq   = axes[1, 1]

    # (a) Residual series ─────────────────────────────────────────────────────
    if dv is not None:
        ax_ts.plot(dv, rv, color="steelblue", lw=0.85, alpha=0.85)
    else:
        ax_ts.plot(rv, color="steelblue", lw=0.85)
    ax_ts.axhline(0, color="black", lw=0.7, ls="--")
    ax_ts.set_title("(a) Residual series", fontsize=8)
    ax_ts.set_ylabel("Residuo", fontsize=7)
    ax_ts.grid(axis="y", alpha=0.20)
    _fmt_date_ax(ax_ts, dv)
    _stat_annotation(ax_ts, diag_stats, include_bg=False)

    # (b) ACF ─────────────────────────────────────────────────────────────────
    if HAS_SM and n > n_lags + 1:
        try:
            plot_acf(rv, lags=n_lags, ax=ax_acf, alpha=0.05,
                     color="steelblue", vlines_kwargs={"colors": "steelblue"})
        except Exception:
            ax_acf.text(0.5, 0.5, "ACF N/A", ha="center", va="center",
                        transform=ax_acf.transAxes, fontsize=9)
    else:
        ax_acf.text(0.5, 0.5, "ACF N/A", ha="center", va="center",
                    transform=ax_acf.transAxes, fontsize=9)
    ax_acf.set_title(f"(b) ACF of residuals (lag={n_lags})", fontsize=8)
    ax_acf.set_xlabel("Lag", fontsize=7)
    ax_acf.grid(axis="y", alpha=0.20)

    # (c) Histogram ───────────────────────────────────────────────────────────
    bins = max(10, n // 5)
    ax_hist.hist(std_rv, bins=bins, density=True,
                 color="steelblue", alpha=0.6, edgecolor="white",
                 label="Std. residuals")
    xx = np.linspace(std_rv.min() - 0.5, std_rv.max() + 0.5, 300)
    if HAS_SCIPY:
        ax_hist.plot(xx, _sp.norm.pdf(xx), color="red", lw=1.5, label="N(0,1)")
    ax_hist.set_title("(c) Histogram of std. residuals", fontsize=8)
    ax_hist.set_xlabel("Standardised residual", fontsize=7)
    ax_hist.legend(fontsize=7)
    ax_hist.grid(axis="y", alpha=0.20)

    # (d) QQ plot ─────────────────────────────────────────────────────────────
    ax_qq.set_title("(d) QQ plot vs N(0,1)", fontsize=8)
    _qq_plot(ax_qq, std_rv, color="steelblue")

    fig.tight_layout()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"    → SARIMA diag plot: {out_path.name}")
