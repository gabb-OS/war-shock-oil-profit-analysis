#!/usr/bin/env python3
"""
02f_triangulation_tests.py  —  Triangolazione statistica formale tra modelli ITS
==================================================================================
Legge i file residuals_*.csv prodotti da tutti i modelli ITS (v1–v8) e applica
una batteria di test statistici sugli extra-margini post-intervento, poi combina
i p-value tra modelli con metodi formali per produrre un verdetto unico per
ogni (evento × carburante).

Perché serve questo file
─────────────────────────
Il criterio "3/4 modelli rifiutano H0" è una regola euristica. Non ha un p-value
associato e non controlla il tasso di falsi positivi globale. Questo modulo
sostituisce quella regola con:

  1. Test per singolo modello  (sul vettore extra_post = actual − CF)
     ├─ Sign test            binomtest(n_pos, n_total, p=0.5)   ← non-param, robusto
     ├─ Wilcoxon signed-rank wilcoxon(extra, alternative="greater") ← non-param
     └─ t-test n_eff-corretto  corregge per autocorrelazione AR(1) ← parametrico

  2. Combinazione tra modelli  (Fisher + Stouffer)
     ├─ Fisher χ²  = −2 Σ ln(pᵢ)   ~ χ²(2k)      ← classico, sempre usabile
     ├─ Stouffer Z = Σ zᵢ / √k      ~ N(0,1)      ← più potente, assume indip.
     └─ Vote count  n_reject / n_models             ← leggibile, non sostituto

  3. Verdetto finale per (evento × carburante)
     ├─ Basato su Fisher e Stouffer combinati (p_combined)
     └─ Con effect size mediano (mediana di Hodges-Lehmann tra modelli)

Come funziona la combinazione di p-value
─────────────────────────────────────────
Fisher (1932): se i modelli sono indipendenti e H0 vera,
  X² = −2 Σᵢ ln(pᵢ)  ~  χ²(2k)
Rifiuta H0 globale se X² > χ²_{1-α}(2k).

Stouffer (1949): converte ogni pᵢ in zᵢ = Φ⁻¹(1 − pᵢ) (one-sided),
  Z = Σzᵢ / √k  ~  N(0,1)
Più potente di Fisher quando i segnali hanno uguale dimensione.

I due test sono complementari: Fisher è sensibile a p piccoli isolati,
Stouffer bilancia segnali uniformi. Entrambi vengono riportati.

Nota sui modelli non del tutto indipendenti
────────────────────────────────────────────
I modelli condividono gli stessi dati di input, quindi i loro p-value non
sono strettamente indipendenti. La combinazione resta valida in senso
conservativo (tende a sovrastimare p_combined), per cui un rifiuto rimane
probante; una non-rejection è meno conclusiva.

Output
──────
  data/plots/triangulation/
    triangulation_by_model.csv     — un test per ogni modello × evento × carburante
    triangulation_combined.csv     — Fisher + Stouffer per ogni evento × carburante
    plot_pvalues_{evento}.png      — heatmap p-value per modello
    plot_combined_verdict.png      — grafico verdetti finali con effect size

Uso
───
  python3 02f_triangulation_tests.py
  python3 02f_triangulation_tests.py --alpha 0.05
  python3 02f_triangulation_tests.py --mode fixed
  python3 02f_triangulation_tests.py --mode detected --detect margin
  python3 02f_triangulation_tests.py --all-modes          # aggrega tutti i mode
"""

from __future__ import annotations

import argparse
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import binomtest, wilcoxon, chi2

# ── Configurazione ────────────────────────────────────────────────────────────
BASE_DIR  = Path(__file__).parent
ITS_BASE  = BASE_DIR / "data" / "plots" / "its"
OUT_DIR   = BASE_DIR / "data" / "plots" / "triangulation"
OUT_DIR.mkdir(parents=True, exist_ok=True)

ALPHA_DEFAULT = 0.05
SEP = "═" * 72

# Pattern dei file residuals prodotti dai vari modelli
# Ogni modello salva: residuals_{evento_safe}_{carburante}.csv
# con colonne: date, residual, phase, metodo, evento, carburante, break_date
RESIDUAL_GLOB = "residuals_*.csv"

# Nomi eventi (per matching con i file safe-name)
EVENTS_SAFE = {
    "Ucraina_Feb_2022":       "Ucraina (Feb 2022)",
    "Iran-Israele_Giu_2025":  "Iran-Israele (Giu 2025)",
    "Hormuz_Feb_2026":        "Hormuz (Feb 2026)",
}

# Colori per i plot
EVENT_COLORS = {
    "Ucraina (Feb 2022)":       "#e74c3c",
    "Iran-Israele (Giu 2025)":  "#e67e22",
    "Hormuz (Feb 2026)":        "#8e44ad",
}

FUEL_COLORS = {
    "benzina": "#E63946",
    "gasolio": "#1D3557",
}


# ══════════════════════════════════════════════════════════════════════════════
# 1. Caricamento dei residui
# ══════════════════════════════════════════════════════════════════════════════

def _scan_residuals(modes: list[str]) -> pd.DataFrame:
    """
    Scansiona le directory ITS per trovare tutti i file residuals_*.csv.
    Restituisce un DataFrame con tutte le righe post-intervento.
    """
    all_dfs = []
    searched = []

    for mode_path_str in modes:
        mode_path = ITS_BASE / mode_path_str
        if not mode_path.exists():
            continue
        # Cerca ricorsivamente in tutte le sottodirectory (v1/, v3/, ecc.)
        for csv_path in sorted(mode_path.rglob(RESIDUAL_GLOB)):
            searched.append(str(csv_path))
            try:
                df = pd.read_csv(csv_path)
                # Aggiungi info sul percorso (mode, variante) se non già presenti
                if "mode_path" not in df.columns:
                    df["mode_path"] = mode_path_str
                all_dfs.append(df)
            except Exception as e:
                warnings.warn(f"Impossibile leggere {csv_path}: {e}")

    if not all_dfs:
        print(f"  ⚠ Nessun file residuals trovato nelle path: {modes}")
        print(f"    Cercato in: {ITS_BASE}")
        return pd.DataFrame()

    full = pd.concat(all_dfs, ignore_index=True)

    # Tieni solo le righe post-intervento (extra = actual − CF)
    post = full[full["phase"] == "post"].copy()

    print(f"  Trovati {len(searched)} file residuals")
    print(f"  Righe post-intervento totali: {len(post)}")
    print(f"  Modelli distinti: {sorted(post['metodo'].unique())}")
    print(f"  Carburanti: {sorted(post['carburante'].unique())}")

    return post


# ══════════════════════════════════════════════════════════════════════════════
# 2. Test statistici per singolo modello
# ══════════════════════════════════════════════════════════════════════════════

def _phi_ar1(x: np.ndarray) -> float:
    """Autocorrelazione lag-1, clampata a (−0.99, 0.99)."""
    x = np.asarray(x, float)
    if len(x) < 3:
        return 0.0
    xc = x - x.mean()
    r = float(np.corrcoef(xc[:-1], xc[1:])[0, 1])
    return float(np.clip(r, -0.99, 0.99))


def _n_eff(x: np.ndarray) -> float:
    """N effettivo corretto per autocorrelazione AR(1)."""
    phi = _phi_ar1(x)
    return max(2.0, len(x) * (1 - phi) / (1 + phi))


def _hodges_lehmann(x: np.ndarray) -> float:
    """
    Stimatore di Hodges-Lehmann: mediana di tutti i (xᵢ + xⱼ)/2.
    È lo stimatore di posizione associato al test di Wilcoxon.
    Equivale alla mediana dei margini extra in senso robusto.
    """
    x = np.asarray(x, float)
    n = len(x)
    if n == 0:
        return np.nan
    # Walsh averages
    i_idx, j_idx = np.triu_indices(n, k=0)
    averages = (x[i_idx] + x[j_idx]) / 2.0
    return float(np.median(averages))


def _block_bootstrap_mean(x: np.ndarray, n_boot: int = 999,
                           block_size: int | None = None,
                           rng: np.random.Generator | None = None) -> float:
    """
    Bootstrap a blocchi (circular block bootstrap) per stimare il p-value
    di H0: media(x) = 0, alternativa 'greater'.
    Robusto alla correlazione seriale — non assume indipendenza delle osservazioni.
    """
    if rng is None:
        rng = np.random.default_rng(42)
    x = np.asarray(x, float)
    n = len(x)
    if n < 4:
        return np.nan

    # Block size default: n^(1/3), almeno 2
    if block_size is None:
        block_size = max(2, int(round(n ** (1 / 3))))

    obs_mean = float(x.mean())
    x_centered = x - obs_mean  # centra sotto H0

    # Genera blocchi circolari
    n_blocks = int(np.ceil(n / block_size))
    boot_means = np.empty(n_boot, dtype=float)
    for b in range(n_boot):
        starts = rng.integers(0, n, size=n_blocks)
        boot_sample = np.concatenate([
            x_centered[np.arange(s, s + block_size) % n] for s in starts
        ])[:n]
        boot_means[b] = float(boot_sample.mean())

    # p-value one-sided: P(T* >= obs_mean | H0)
    p = float(np.mean(boot_means >= obs_mean))
    return max(p, 1.0 / (n_boot + 1))  # evita p=0 esatto


def test_single_model(extra: np.ndarray, label: str) -> dict:
    """
    Testa H0: mediana(extra_post) = 0  contro H1: mediana > 0.

    Restituisce un dict con p-value di ogni test e le statistiche principali.
    Tutti i test sono one-sided (H1: margine post > atteso dal CF).
    """
    extra = np.asarray(extra, float)
    extra = extra[~np.isnan(extra)]
    n = len(extra)

    row: dict = {
        "label": label,
        "n":     n,
        "mean_extra":   round(float(extra.mean()), 6) if n > 0 else np.nan,
        "median_extra": round(float(np.median(extra)), 6) if n > 0 else np.nan,
        "std_extra":    round(float(extra.std(ddof=1)), 6) if n > 1 else np.nan,
        "phi_ar1":      round(_phi_ar1(extra), 4),
        "n_eff":        round(_n_eff(extra), 1),
        "hodges_lehmann": round(_hodges_lehmann(extra), 6) if n > 0 else np.nan,
    }

    if n < 5:
        for key in ["sign_p", "wilcox_p", "ttest_neff_p", "bootstrap_p"]:
            row[key] = np.nan
        row["skip_reason"] = "n_insufficiente"
        return row

    # ── Test del segno ────────────────────────────────────────────────────────
    # H0: P(extra > 0) = 0.5  (la mediana è 0)
    # H1: P(extra > 0) > 0.5  (la mediana è positiva)
    n_pos   = int(np.sum(extra > 0))
    n_nonzero = int(np.sum(extra != 0))
    if n_nonzero > 0:
        bt = binomtest(n_pos, n_nonzero, p=0.5, alternative="greater")
        row["sign_p"]     = round(float(bt.pvalue), 6)
        row["sign_n_pos"] = n_pos
        row["sign_n"]     = n_nonzero
    else:
        row["sign_p"] = np.nan

    # ── Wilcoxon signed-rank ──────────────────────────────────────────────────
    # H0: distribuzione simmetrica attorno a 0
    # H1: distribuzione spostata verso destra (guadagni > 0)
    # Più potente del test del segno se la distribuzione è simmetrica.
    try:
        wx_stat, wx_p = wilcoxon(extra, alternative="greater",
                                 zero_method="wilcox")
        row["wilcox_stat"] = round(float(wx_stat), 3)
        row["wilcox_p"]    = round(float(wx_p), 6)
    except Exception:
        row["wilcox_stat"] = np.nan
        row["wilcox_p"]    = np.nan

    # ── t-test one-sample corretto per autocorrelazione (n_eff) ──────────────
    # Usa n_eff invece di n per il calcolo di SE e gradi di libertà.
    # Robusto alla correlazione seriale tipica dei margini giornalieri.
    ne  = _n_eff(extra)
    se  = float(extra.std(ddof=1) / np.sqrt(ne))
    if se > 0:
        t_stat = float(extra.mean() / se)
        t_p    = float(stats.t.sf(t_stat, df=ne - 1))  # one-sided
        row["ttest_neff_t"]  = round(t_stat, 4)
        row["ttest_neff_p"]  = round(t_p, 6)
        row["ttest_neff_df"] = round(ne - 1, 1)
    else:
        row["ttest_neff_t"] = row["ttest_neff_p"] = row["ttest_neff_df"] = np.nan

    # ── Bootstrap a blocchi ───────────────────────────────────────────────────
    # Non assume né normalità né indipendenza. Più lento ma il più robusto.
    # Usa block_size = n^(1/3) per bilanciare bias/varianza.
    row["bootstrap_p"] = round(
        _block_bootstrap_mean(extra, n_boot=1999, rng=np.random.default_rng(42)),
        6
    )

    return row


# ══════════════════════════════════════════════════════════════════════════════
# 3. Combinazione dei p-value tra modelli (Fisher + Stouffer)
# ══════════════════════════════════════════════════════════════════════════════

def _fisher_combined(p_values: list[float]) -> tuple[float, float]:
    """
    Fisher's method: X² = −2 Σ ln(pᵢ) ~ χ²(2k).
    Restituisce (statistica, p_value).
    P-value piccolo = evidenza combinata che almeno un p_i è piccolo.
    """
    ps = np.asarray([p for p in p_values if not np.isnan(p) and 0 < p <= 1])
    if len(ps) == 0:
        return np.nan, np.nan
    chi2_stat = -2.0 * np.sum(np.log(ps))
    df = 2 * len(ps)
    p_combined = float(chi2.sf(chi2_stat, df))
    return float(chi2_stat), p_combined


def _stouffer_combined(p_values: list[float]) -> tuple[float, float]:
    """
    Stouffer's Z-method: Z = Σ Φ⁻¹(1 − pᵢ) / √k ~ N(0,1).
    Restituisce (Z, p_value).
    Più potente di Fisher quando i segnali sono uniformi tra modelli.
    """
    ps = np.asarray([p for p in p_values if not np.isnan(p) and 0 < p < 1])
    if len(ps) == 0:
        return np.nan, np.nan
    z_scores = stats.norm.ppf(1.0 - ps)  # converte p one-sided in z one-sided
    Z = float(np.sum(z_scores) / np.sqrt(len(ps)))
    p_combined = float(stats.norm.sf(Z))
    return Z, p_combined


def combine_models(model_rows: list[dict], test_key: str = "wilcox_p") -> dict:
    """
    Combina i p-value di `test_key` tra tutti i modelli con Fisher e Stouffer.
    `test_key` può essere: 'sign_p', 'wilcox_p', 'ttest_neff_p', 'bootstrap_p'.

    Restituisce un dict con le statistiche combinate.
    """
    p_values = [r.get(test_key, np.nan) for r in model_rows]
    valid_ps  = [p for p in p_values if not np.isnan(p)]
    n_models  = len(model_rows)
    n_valid   = len(valid_ps)

    row: dict = {
        "n_models":         n_models,
        "n_valid":          n_valid,
        "test_used":        test_key,
        "p_values":         [round(p, 6) for p in p_values],
        "labels":           [r.get("label", "?") for r in model_rows],
        "median_hl":        round(float(np.nanmedian(
                                [r.get("hodges_lehmann", np.nan) for r in model_rows]
                            )), 6),
        "median_mean_extra": round(float(np.nanmedian(
                                [r.get("mean_extra", np.nan) for r in model_rows]
                            )), 6),
    }

    if n_valid == 0:
        row["fisher_stat"] = row["fisher_p"] = np.nan
        row["stouffer_z"]  = row["stouffer_p"] = np.nan
        row["n_reject_05"] = row["n_reject_01"] = np.nan
        return row

    # Fisher
    fisher_stat, fisher_p = _fisher_combined(valid_ps)
    row["fisher_stat"] = round(fisher_stat, 4) if not np.isnan(fisher_stat) else np.nan
    row["fisher_p"]    = round(fisher_p, 6)    if not np.isnan(fisher_p)    else np.nan

    # Stouffer
    stouffer_z, stouffer_p = _stouffer_combined(valid_ps)
    row["stouffer_z"] = round(stouffer_z, 4) if not np.isnan(stouffer_z) else np.nan
    row["stouffer_p"] = round(stouffer_p, 6) if not np.isnan(stouffer_p) else np.nan

    # Voto semplice (per confronto con la vecchia euristica)
    row["n_reject_05"] = int(sum(p < 0.05 for p in valid_ps))
    row["n_reject_01"] = int(sum(p < 0.01 for p in valid_ps))
    row["vote_frac"]   = round(row["n_reject_05"] / n_valid, 3)

    # Verdetto combinato (usa il più conservativo tra Fisher e Stouffer)
    combined_p = max(
        fisher_p  if not np.isnan(fisher_p)  else 1.0,
        stouffer_p if not np.isnan(stouffer_p) else 1.0,
    )
    row["combined_p_conservative"] = round(combined_p, 6)

    return row


# ══════════════════════════════════════════════════════════════════════════════
# 4. Plot
# ══════════════════════════════════════════════════════════════════════════════

def _pstar(p: float) -> str:
    if np.isnan(p): return ""
    if p < 0.001: return "***"
    if p < 0.01:  return "**"
    if p < 0.05:  return "*"
    return "ns"


def plot_pvalue_heatmap(
    by_model_df: pd.DataFrame,
    evento: str,
    carburante: str,
    test_key: str,
    alpha: float,
    out_path: Path,
) -> None:
    """
    Heatmap p-value per tutti i modelli, per un dato evento × carburante.
    Celle verdi = rifiuto H0, rosse = non rifiuto.
    """
    sub = by_model_df[
        (by_model_df["evento"] == evento) &
        (by_model_df["carburante"] == carburante)
    ].copy()

    if sub.empty:
        return

    sub = sub.sort_values("metodo")
    models  = sub["metodo"].tolist()
    ps      = sub[test_key].tolist()
    hl_vals = sub["hodges_lehmann"].tolist()
    n_vals  = sub["n"].tolist()

    fig, ax = plt.subplots(figsize=(max(8, len(models) * 1.2), 3.5))

    for i, (m, p, hl, n) in enumerate(zip(models, ps, hl_vals, n_vals)):
        rejected = (not np.isnan(p)) and (p < alpha)
        color    = "#27ae60" if rejected else "#e74c3c" if not np.isnan(p) else "#bdc3c7"
        ax.bar(i, 1, color=color, alpha=0.75, edgecolor="white", linewidth=1.5)
        p_str = f"{p:.4f}" if not np.isnan(p) else "N/D"
        star  = _pstar(p) if not np.isnan(p) else ""
        hl_str = f"{hl:+.4f}" if not np.isnan(hl) else "N/D"
        ax.text(i, 0.75, f"{m}", ha="center", va="center", fontsize=7.5,
                fontweight="bold", color="white")
        ax.text(i, 0.50, f"p={p_str}{star}", ha="center", va="center", fontsize=8,
                color="white")
        ax.text(i, 0.25, f"HL={hl_str}", ha="center", va="center", fontsize=7,
                color="white")
        ax.text(i, 0.05, f"n={n}", ha="center", va="center", fontsize=6.5,
                color="white", alpha=0.85)

    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlim(-0.5, len(models) - 0.5)
    ax.set_ylim(0, 1)
    ax.set_title(
        f"P-value per modello  |  {evento}  ×  {carburante}  |  Test: {test_key}\n"
        f"Verde = rifiuto H0 (p<{alpha})  |  HL = Hodges-Lehmann stimatore di posizione",
        fontsize=9, fontweight="bold"
    )

    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor="#27ae60", alpha=0.75, label=f"Rifiuto H₀ (p < {alpha})"),
        Patch(facecolor="#e74c3c", alpha=0.75, label=f"Non rifiuto H₀"),
    ]
    ax.legend(handles=legend_elements, loc="lower right", fontsize=7)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_combined_verdict(combined_df: pd.DataFrame, alpha: float, out_path: Path) -> None:
    """
    Grafico riassuntivo: per ogni evento × carburante mostra Fisher p,
    Stouffer p, voto semplice e effect size mediano (Hodges-Lehmann).
    """
    if combined_df.empty:
        return

    rows   = combined_df.copy()
    n_rows = len(rows)

    fig = plt.figure(figsize=(14, 3.5 * n_rows + 2), constrained_layout=True)
    fig.suptitle(
        "Triangolazione formale  —  Fisher + Stouffer combined test\n"
        f"H₀: mediana(extra_post) = 0  |  H₁: margine post > atteso dal controfattuale  |  α={alpha}",
        fontsize=11, fontweight="bold"
    )
    gs = gridspec.GridSpec(n_rows, 3, figure=fig)

    for row_i, (_, row) in enumerate(rows.iterrows()):
        ev  = row["evento"]
        fuel = row["carburante"]
        ev_color = EVENT_COLORS.get(ev, "#333333")
        fuel_color = FUEL_COLORS.get(fuel, "#555555")

        # ── Pannello 1: p-value individuali ──────────────────────────────────
        ax1 = fig.add_subplot(gs[row_i, 0])
        p_vals = eval(str(row["p_values"])) if isinstance(row["p_values"], str) else row["p_values"]
        labels = eval(str(row["labels"]))   if isinstance(row["labels"], str)   else row["labels"]
        p_arr  = np.asarray(p_vals, dtype=float)
        colors_bar = ["#27ae60" if (not np.isnan(p) and p < alpha) else "#e74c3c"
                      for p in p_arr]
        y_pos = np.arange(len(labels))
        bars  = ax1.barh(y_pos, np.where(np.isnan(p_arr), 0, p_arr),
                         color=colors_bar, alpha=0.75, edgecolor="white", height=0.6)
        ax1.axvline(alpha, color="black", lw=1.5, ls="--",
                    label=f"α={alpha}", zorder=5)
        for i, (p, lbl) in enumerate(zip(p_arr, labels)):
            if not np.isnan(p):
                ax1.text(min(p + 0.005, 0.98), i, f"{p:.3f}{_pstar(p)}",
                         va="center", fontsize=6.5)
        ax1.set_yticks(y_pos)
        ax1.set_yticklabels([str(l).replace("v", "v").split("_")[0]
                              for l in labels], fontsize=7)
        ax1.set_xlim(0, 1.05)
        ax1.set_title(f"{ev}\n{fuel.capitalize()} — p per modello",
                      fontsize=8, fontweight="bold", color=ev_color)
        ax1.set_xlabel("p-value", fontsize=7)
        ax1.legend(fontsize=6, loc="lower right")

        # ── Pannello 2: risultati combinati ──────────────────────────────────
        ax2 = fig.add_subplot(gs[row_i, 1])
        ax2.axis("off")

        fisher_p   = row.get("fisher_p", np.nan)
        stouffer_p = row.get("stouffer_p", np.nan)
        cons_p     = row.get("combined_p_conservative", np.nan)
        vote_frac  = row.get("vote_frac", np.nan)
        n_rej05    = int(row.get("n_reject_05", 0))
        n_valid    = int(row.get("n_valid", 0))
        hl_med     = row.get("median_hl", np.nan)
        mean_med   = row.get("median_mean_extra", np.nan)

        def _verdict_color(p):
            if np.isnan(p): return "#bdc3c7"
            return "#27ae60" if p < alpha else "#e74c3c"

        def _fmt_p(p, label):
            if np.isnan(p): return f"{label}: N/D"
            return f"{label}: {p:.4f} {_pstar(p)}"

        lines = [
            ("", "#ffffff", 0.95),
            (_fmt_p(fisher_p,   "Fisher p"),    _verdict_color(fisher_p),   0.82),
            (f"  χ²={row.get('fisher_stat', np.nan):.2f}" if not np.isnan(row.get('fisher_stat', np.nan)) else "", "#ffffff", 0.72),
            (_fmt_p(stouffer_p, "Stouffer p"),  _verdict_color(stouffer_p), 0.60),
            (f"  Z={row.get('stouffer_z', np.nan):.2f}"   if not np.isnan(row.get('stouffer_z', np.nan))   else "", "#ffffff", 0.50),
            (f"Voto: {n_rej05}/{n_valid} modelli (p<{alpha})", "#2c3e50", 0.37),
            (f"HL mediano: {hl_med:+.4f} €/L" if not np.isnan(hl_med) else "HL mediano: N/D", "#2c3e50", 0.25),
            (f"Extra mediano: {mean_med:+.4f} €/L" if not np.isnan(mean_med) else "", "#2c3e50", 0.13),
        ]
        for text, color, y in lines:
            if text:
                ax2.text(0.05, y, text, transform=ax2.transAxes,
                         fontsize=9, va="center", color=color,
                         fontweight="bold" if "p" in text or "Voto" in text else "normal")
        ax2.set_title("Combinazione Fisher + Stouffer", fontsize=8, fontweight="bold")

        # ── Pannello 3: verdetto finale ───────────────────────────────────────
        ax3 = fig.add_subplot(gs[row_i, 2])
        ax3.axis("off")

        fisher_reject   = (not np.isnan(fisher_p))   and (fisher_p < alpha)
        stouffer_reject = (not np.isnan(stouffer_p)) and (stouffer_p < alpha)

        if fisher_reject and stouffer_reject:
            verdict     = "RIFIUTO H₀"
            subtext     = "Entrambi Fisher e Stouffer\nrifiutano H₀"
            bg_color    = "#27ae60"
            text_color  = "white"
            confidence  = "ALTA"
        elif fisher_reject or stouffer_reject:
            verdict     = "RIFIUTO PARZIALE"
            subtext     = "Solo uno dei due test\ncombinati rifiuta H₀"
            bg_color    = "#f39c12"
            text_color  = "white"
            confidence  = "MEDIA"
        else:
            verdict     = "NON RIFIUTO H₀"
            subtext     = "Nessun test combinato\nrifiuta H₀"
            bg_color    = "#e74c3c"
            text_color  = "white"
            confidence  = "BASSA"

        bbox = dict(boxstyle="round,pad=1", facecolor=bg_color, alpha=0.85, edgecolor="white")
        ax3.text(0.5, 0.65, verdict, transform=ax3.transAxes,
                 fontsize=13, fontweight="bold", ha="center", va="center",
                 color=text_color, bbox=bbox)
        ax3.text(0.5, 0.35, subtext, transform=ax3.transAxes,
                 fontsize=8, ha="center", va="center", color="#2c3e50")
        ax3.text(0.5, 0.15,
                 f"p conservativo = {cons_p:.4f}" if not np.isnan(cons_p) else "",
                 transform=ax3.transAxes,
                 fontsize=8, ha="center", va="center", color="#7f8c8d")
        ax3.set_title("Verdetto triangolazione", fontsize=8, fontweight="bold")

    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  → Plot verdetti: {out_path}")


# ══════════════════════════════════════════════════════════════════════════════
# 5. Main
# ══════════════════════════════════════════════════════════════════════════════

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Triangolazione statistica formale tra modelli ITS",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--alpha",     type=float, default=ALPHA_DEFAULT,
                   help=f"Livello α [default={ALPHA_DEFAULT}]")
    p.add_argument("--mode",      choices=["fixed", "detected"], default="fixed",
                   help="Modalità ITS da analizzare [default=fixed]")
    p.add_argument("--detect",    choices=["margin", "price"], default="margin",
                   help="(solo mode=detected) variante di detection [default=margin]")
    p.add_argument("--all-modes", action="store_true",
                   help="Aggrega residui da TUTTE le modalità (fixed + detected/*)")
    p.add_argument("--test",      choices=["sign_p", "wilcox_p", "ttest_neff_p", "bootstrap_p"],
                   default="wilcox_p",
                   help="Test primario per la combinazione Fisher/Stouffer [default=wilcox_p]")
    p.add_argument("--no-plots",  action="store_true",
                   help="Salta la generazione dei plot")
    return p.parse_args()


def main() -> None:
    args    = parse_args()
    alpha   = args.alpha
    test_key = args.test

    # Determina quali directory scansionare
    if args.all_modes:
        modes_to_scan = ["fixed", "detected/margin", "detected/price"]
    elif args.mode == "detected":
        modes_to_scan = [f"detected/{args.detect}"]
    else:
        modes_to_scan = ["fixed"]

    print(f"\n{SEP}")
    print("  02f_triangulation_tests.py  —  Triangolazione statistica formale")
    print(f"  Mode: {modes_to_scan}  |  Test primario: {test_key}  |  α={alpha}")
    print(f"  Output: {OUT_DIR}")
    print(f"{SEP}\n")

    # ── Caricamento ───────────────────────────────────────────────────────────
    print("Scansione file residuals...")
    post_df = _scan_residuals(modes_to_scan)

    if post_df.empty:
        print("\n  ✗ Nessun dato trovato. Esegui prima almeno un modello ITS (v1, v3, ecc.).")
        return

    # ── Test per singolo modello ──────────────────────────────────────────────
    print(f"\n{SEP}")
    print("  TEST PER SINGOLO MODELLO")
    print(f"{SEP}")

    model_rows: list[dict] = []

    # Raggruppa per (evento, carburante, metodo)
    grp_cols = ["evento", "carburante", "metodo"]
    for (evento, carburante, metodo), grp in post_df.groupby(grp_cols):
        extra = grp["residual"].values
        row = test_single_model(extra, label=metodo)
        row["evento"]     = evento
        row["carburante"] = carburante
        row["metodo"]     = metodo
        model_rows.append(row)

        # Stampa riga
        sign_p  = row.get("sign_p", np.nan)
        wx_p    = row.get("wilcox_p", np.nan)
        t_p     = row.get("ttest_neff_p", np.nan)
        boot_p  = row.get("bootstrap_p", np.nan)
        hl      = row.get("hodges_lehmann", np.nan)
        n       = row.get("n", 0)
        print(
            f"  {metodo:<12} {carburante:<10} {evento:<30} "
            f"n={n:>3}  sign={sign_p:.4f}{_pstar(sign_p) if not np.isnan(sign_p) else '   '}"
            f"  WX={wx_p:.4f}{_pstar(wx_p) if not np.isnan(wx_p) else '   '}"
            f"  t*={t_p:.4f}{_pstar(t_p) if not np.isnan(t_p) else '   '}"
            f"  boot={boot_p:.4f}{_pstar(boot_p) if not np.isnan(boot_p) else '   '}"
            f"  HL={hl:+.4f}" if not np.isnan(hl) else ""
        )

    df_model = pd.DataFrame(model_rows)
    out_model = OUT_DIR / "triangulation_by_model.csv"
    df_model.to_csv(out_model, index=False)
    print(f"\n  → CSV modelli: {out_model}")

    # ── Combinazione Fisher + Stouffer ────────────────────────────────────────
    print(f"\n{SEP}")
    print(f"  COMBINAZIONE FISHER + STOUFFER  (test primario: {test_key})")
    print(f"{SEP}\n")
    print(f"  {'Evento':<30} {'Carburante':<10} {'Fisher-p':>10} {'Stouffer-p':>12}"
          f"  {'Voto':>8}  {'HL-med':>10}  Verdetto")
    print("  " + "─" * 100)

    combined_rows: list[dict] = []

    for (evento, carburante), grp in df_model.groupby(["evento", "carburante"]):
        model_subset = grp.to_dict(orient="records")
        comb = combine_models(model_subset, test_key=test_key)
        comb["evento"]     = evento
        comb["carburante"] = carburante
        combined_rows.append(comb)

        fp  = comb.get("fisher_p", np.nan)
        sp  = comb.get("stouffer_p", np.nan)
        vf  = comb.get("vote_frac", np.nan)
        nrj = int(comb.get("n_reject_05", 0))
        nv  = int(comb.get("n_valid", 0))
        hl  = comb.get("median_hl", np.nan)

        fisher_r   = (not np.isnan(fp)) and (fp < alpha)
        stouffer_r = (not np.isnan(sp)) and (sp < alpha)
        if fisher_r and stouffer_r:
            verdict = "✅ RIFIUTO H₀ (alta)"
        elif fisher_r or stouffer_r:
            verdict = "⚠  RIFIUTO PARZIALE"
        else:
            verdict = "❌ NON RIFIUTO"

        print(
            f"  {evento:<30} {carburante:<10}"
            f" {fp:>10.4f}{_pstar(fp):<3}"
            f" {sp:>10.4f}{_pstar(sp):<3}"
            f"  {nrj}/{nv:>2}"
            f"  {hl:>+10.4f}"
            f"  {verdict}"
        )

    print(f"\n  Legenda: *** p<0.001  ** p<0.01  * p<0.05  ns p≥{alpha}")
    print(f"  Fisher: robusto a segnali isolati  |  Stouffer: più potente se segnale uniforme")
    print(f"  Verdetto finale basato sul più conservativo tra i due")

    df_combined = pd.DataFrame(combined_rows)
    out_combined = OUT_DIR / "triangulation_combined.csv"
    df_combined.to_csv(out_combined, index=False)
    print(f"\n  → CSV combinato: {out_combined}")

    # ── Plot ──────────────────────────────────────────────────────────────────
    if not args.no_plots:
        print(f"\n{SEP}")
        print("  GENERAZIONE PLOT")
        print(f"{SEP}")

        # Heatmap p-value per ogni evento
        for evento in df_model["evento"].unique():
            for carburante in df_model["carburante"].unique():
                sub = df_model[
                    (df_model["evento"] == evento) &
                    (df_model["carburante"] == carburante)
                ]
                if sub.empty:
                    continue
                safe_ev = (evento.replace(" ", "_").replace("/", "")
                                  .replace("(", "").replace(")", ""))
                out_hmap = OUT_DIR / f"plot_heatmap_{safe_ev}_{carburante}.png"
                plot_pvalue_heatmap(df_model, evento, carburante,
                                    test_key, alpha, out_hmap)
                print(f"  → Heatmap: {out_hmap}")

        # Verdetto combinato
        out_verdict = OUT_DIR / "plot_combined_verdict.png"
        plot_combined_verdict(df_combined, alpha, out_verdict)

    # ── Riepilogo finale ──────────────────────────────────────────────────────
    print(f"\n{SEP}")
    print("  RIEPILOGO FINALE TRIANGOLAZIONE")
    print(f"{SEP}")
    print(f"  α={alpha}  |  Test primario: {test_key}")
    print(f"  N modelli analizzati: {df_model['metodo'].nunique()}")
    print()

    for _, row in df_combined.iterrows():
        ev   = row["evento"]
        fuel = row["carburante"]
        fp   = row.get("fisher_p", np.nan)
        sp   = row.get("stouffer_p", np.nan)
        cp   = row.get("combined_p_conservative", np.nan)
        hl   = row.get("median_hl", np.nan)
        nrj  = int(row.get("n_reject_05", 0))
        nv   = int(row.get("n_valid", 0))

        fisher_r   = (not np.isnan(fp)) and (fp < alpha)
        stouffer_r = (not np.isnan(sp)) and (sp < alpha)

        if fisher_r and stouffer_r:
            icon = "🔴"
            interp = "PROFITTO ANOMALO STATISTICAMENTE SIGNIFICATIVO"
        elif fisher_r or stouffer_r:
            icon = "🟡"
            interp = "EVIDENZA PARZIALE DI PROFITTO ANOMALO"
        else:
            icon = "🟢"
            interp = "NESSUNA EVIDENZA SIGNIFICATIVA"

        hl_str = f"HL={hl:+.4f} €/L" if not np.isnan(hl) else "HL=N/D"
        print(f"  {icon}  {ev:<30}  {fuel:<10}  {interp}")
        print(f"       Fisher p={fp:.4f}{_pstar(fp)}  Stouffer p={sp:.4f}{_pstar(sp)}"
              f"  p_cons={cp:.4f}  {hl_str}"
              f"  Voto: {nrj}/{nv}")
        print()

    print(f"  Output → {OUT_DIR}")
    print(f"    triangulation_by_model.csv   — test per ogni modello")
    print(f"    triangulation_combined.csv   — Fisher + Stouffer combinati")
    print(f"    plot_heatmap_*.png           — heatmap p-value per evento")
    print(f"    plot_combined_verdict.png    — grafico verdetti finali")
    print(f"{SEP}\n")


if __name__ == "__main__":
    main()