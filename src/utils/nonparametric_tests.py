#!/usr/bin/env python3
"""
utils/nonparametric_tests.py  —  Batteria non-parametrica H₀/H₁ sui residui ITS
==================================================================================
Testa se i residui post-break (= extra-margine rispetto alla baseline modellata)
sono significativamente superiori a zero — cioè se i distributori hanno generato
profitti anomali al netto del trend storico.

H₀: la distribuzione dei residui post-break ha mediana ≤ 0
    (nessun extra-profitto anomalo)
H₁: la distribuzione dei residui post-break ha mediana > 0
    (extra-profitto anomalo, distributori beneficiano dello shock)

Test implementati
─────────────────
  ONE-SAMPLE (post residuals vs 0):
    T1. Sign test (binomiale esatto, one-sided)
    T2. Wilcoxon signed-rank (one-sided, alternative="greater")
    T3. Permutation test sulla media (one-sided, B=9999 permutazioni)

  TWO-SAMPLE (post residuals vs pre residuals):
    T4. Mann-Whitney U (one-sided, alternative="greater")
    T5. Kolmogorov-Smirnov 2-campioni (bilateral, poi interpretato)
    T6. Mood's median test (k=2)

  EFFECT SIZE:
    ES1. Hodges-Lehmann (one-sample): mediana di {(y_i + y_j)/2  ∀ i≤j}
    ES2. P(post_resid > 0)  — probabilità di superiorità su zero
    ES3. Cliff's δ (post vs pre)
    ES4. Cohen d (post vs pre, pooled std)

  VERDETTO COMBINATO:
    majority_vote: H₀ rigettata se ≥ ceil(n_tests_validi / 2) test rifiutano
    n_reject_count: numero di test che rifiutano H₀

Uso
────
  from utils.nonparametric_tests import nonparam_h0_battery

  results = nonparam_h0_battery(
      post_resid  = np.array([...]),   # residui periodo post-break (€/L)
      pre_resid   = np.array([...]),   # residui periodo pre-break  (€/L)
      alpha       = 0.05,
      n_perm      = 9999,
  )
  # results è un dict con p-value, statistiche, effect size, verdetto
"""

from __future__ import annotations

import math
import warnings
from typing import Sequence

import numpy as np
from scipy import stats
from scipy.stats import (
    mannwhitneyu,
    ks_2samp,
    wilcoxon,
    median_test,
    binom_test,
)


# ══════════════════════════════════════════════════════════════════════════════
# Helpers interni
# ══════════════════════════════════════════════════════════════════════════════

def _safe_float(x) -> float:
    try:
        v = float(x)
        return v if math.isfinite(v) else float("nan")
    except (TypeError, ValueError):
        return float("nan")


def _cliffs_delta(a: np.ndarray, b: np.ndarray) -> float:
    """Cliff δ ∈ [-1,1]: proporzione (b>a) − proporzione (b<a)."""
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    if len(a) == 0 or len(b) == 0:
        return float("nan")
    dominance = (np.sum(b[:, None] > a[None, :]) -
                 np.sum(b[:, None] < a[None, :]))
    return float(dominance / (len(a) * len(b)))


def _cohens_d(a: np.ndarray, b: np.ndarray) -> float:
    """Cohen d (pooled std, campioni indipendenti)."""
    na, nb = len(a), len(b)
    if na < 2 or nb < 2:
        return float("nan")
    sp = math.sqrt(
        ((na - 1) * float(np.var(b, ddof=1)) +
         (nb - 1) * float(np.var(a, ddof=1))) /
        (na + nb - 2)
    )
    if sp < 1e-14:
        return float("nan")
    return float((np.mean(b) - np.mean(a)) / sp)


def _interpret_cliffs(delta: float) -> str:
    if not math.isfinite(delta):
        return "N/A"
    ad = abs(delta)
    if ad < 0.147:
        return "negligible"
    if ad < 0.330:
        return "small"
    if ad < 0.474:
        return "medium"
    return "large"


def _interpret_cohens_d(d: float) -> str:
    if not math.isfinite(d):
        return "N/A"
    ad = abs(d)
    if ad < 0.20:
        return "negligible"
    if ad < 0.50:
        return "small"
    if ad < 0.80:
        return "medium"
    return "large"


def _hodges_lehmann_1s(x: np.ndarray) -> float:
    """
    Stimatore di Hodges-Lehmann one-sample: mediana di {(x_i + x_j)/2, i ≤ j}.
    Robusto al 29 %; compatibile con il Wilcoxon signed-rank.
    """
    x = np.asarray(x, float)
    x = x[~np.isnan(x)]
    if len(x) == 0:
        return float("nan")
    # Walsh averages (i ≤ j) include diagonal (i=j) → (x_i + x_i)/2 = x_i
    i_idx, j_idx = np.triu_indices(len(x), k=0)
    walsh = (x[i_idx] + x[j_idx]) / 2.0
    return float(np.median(walsh))


def _permutation_mean_test(x: np.ndarray, n_perm: int = 9999,
                            rng: np.random.Generator | None = None) -> float:
    """
    Permutation test one-sample per H₀: media(x) = 0 (H₁: media > 0).
    Riflette il segno di ogni osservazione (±x_i) per generare la distribuzione
    nulla — equivalente al test della media su dati simmetrizzati rispetto a 0.

    Ritorna il p-value one-sided (proporzione di permutazioni con media ≥ t_obs).
    """
    x = np.asarray(x, float)
    x = x[~np.isnan(x)]
    n = len(x)
    if n == 0:
        return float("nan")
    t_obs = float(np.mean(x))
    if rng is None:
        rng = np.random.default_rng(42)
    # Genera segni casuali: ogni permutazione riflette un sottoinsieme di elementi
    count = 0
    for _ in range(n_perm):
        signs = rng.choice([-1.0, 1.0], size=n, replace=True)
        t_perm = float(np.mean(np.abs(x) * signs))
        if t_perm >= t_obs:
            count += 1
    return float((count + 1) / (n_perm + 1))   # +1 per inclusione del caso osservato


# ══════════════════════════════════════════════════════════════════════════════
# Batteria principale
# ══════════════════════════════════════════════════════════════════════════════

def nonparam_h0_battery(
    post_resid: Sequence | np.ndarray,
    pre_resid:  Sequence | np.ndarray,
    alpha:   float = 0.05,
    n_perm:  int   = 9999,
    rng:     np.random.Generator | None = None,
) -> dict:
    """
    Batteria completa di test non-parametrici per H₀/H₁ sui residui ITS.

    Parametri
    ----------
    post_resid : array-like
        Residui del periodo post-break (extra-margine effettivo vs baseline).
        Valori positivi = profitto anomalo; negativi = sotto-baseline.
    pre_resid : array-like
        Residui del periodo pre-break (usati come campione di riferimento).
    alpha : float
        Livello di significatività (default 0.05).
    n_perm : int
        Numero di permutazioni per il permutation test (default 9999).
    rng : np.random.Generator | None
        Generatore casuale (per riproducibilità).

    Ritorna
    -------
    dict con tutti i risultati, p-value, effect size e verdetto.
    """
    b = np.asarray(post_resid, float)
    a = np.asarray(pre_resid,  float)
    b = b[~np.isnan(b)]
    a = a[~np.isnan(a)]

    nb, na = len(b), len(a)
    if rng is None:
        rng = np.random.default_rng(42)

    row: dict = {
        "n_post": nb,
        "n_pre":  na,
        "post_mean":   _safe_float(np.mean(b)) if nb > 0 else float("nan"),
        "post_median": _safe_float(np.median(b)) if nb > 0 else float("nan"),
        "post_std":    _safe_float(np.std(b, ddof=1)) if nb > 1 else float("nan"),
        "pre_mean":    _safe_float(np.mean(a)) if na > 0 else float("nan"),
        "pre_median":  _safe_float(np.median(a)) if na > 0 else float("nan"),
        "pre_std":     _safe_float(np.std(a, ddof=1)) if na > 1 else float("nan"),
    }

    MIN_ONE = 4    # minimo per test one-sample
    MIN_TWO = 4    # minimo per test two-sample

    # ── T1. Sign test (binomiale esatto, one-sided) ───────────────────────────
    if nb >= MIN_ONE:
        n_pos = int(np.sum(b > 0))
        n_neg = int(np.sum(b < 0))
        n_nonzero = n_pos + n_neg
        if n_nonzero > 0:
            try:
                # binom_test: k=n_pos, n=n_nonzero, p=0.5, alternative="greater"
                sign_p = float(binom_test(n_pos, n_nonzero, p=0.5, alternative="greater"))
            except Exception:
                sign_p = float("nan")
        else:
            sign_p = float("nan")
        row["sign_n_pos"]    = n_pos
        row["sign_n_neg"]    = n_neg
        row["sign_n_nonzero"] = n_nonzero
        row["sign_p_1s"]     = round(sign_p, 5) if math.isfinite(sign_p) else float("nan")
        row["sign_reject"]   = bool(sign_p < alpha) if math.isfinite(sign_p) else None
        row["prob_pos"]      = round(n_pos / n_nonzero, 4) if n_nonzero > 0 else float("nan")
    else:
        row.update({
            "sign_n_pos": float("nan"), "sign_n_neg": float("nan"),
            "sign_n_nonzero": float("nan"), "sign_p_1s": float("nan"),
            "sign_reject": None, "prob_pos": float("nan"),
        })

    # ── T2. Wilcoxon signed-rank (one-sample, alternative="greater") ──────────
    if nb >= MIN_ONE:
        try:
            wx_stat, wx_p = wilcoxon(b, alternative="greater", zero_method="wilcox",
                                     correction=False)
            row["wilcoxon_1s_stat"]   = round(float(wx_stat), 3)
            row["wilcoxon_1s_p"]      = round(float(wx_p), 5)
            row["wilcoxon_1s_reject"] = bool(wx_p < alpha)
        except Exception as e:
            row["wilcoxon_1s_stat"]   = float("nan")
            row["wilcoxon_1s_p"]      = float("nan")
            row["wilcoxon_1s_reject"] = None
    else:
        row["wilcoxon_1s_stat"]   = float("nan")
        row["wilcoxon_1s_p"]      = float("nan")
        row["wilcoxon_1s_reject"] = None

    # ── T3. Permutation test sulla media (one-sided) ──────────────────────────
    if nb >= MIN_ONE:
        perm_p = _permutation_mean_test(b, n_perm=n_perm, rng=rng)
        row["perm_p_1s"]   = round(perm_p, 5) if math.isfinite(perm_p) else float("nan")
        row["perm_reject"] = bool(perm_p < alpha) if math.isfinite(perm_p) else None
        row["perm_n"]      = n_perm
    else:
        row["perm_p_1s"]   = float("nan")
        row["perm_reject"] = None
        row["perm_n"]      = n_perm

    # ── T4. Mann-Whitney U (two-sided, H1: post > pre) ────────────────────────
    if nb >= MIN_TWO and na >= MIN_TWO:
        try:
            mw_stat, mw_p = mannwhitneyu(b, a, alternative="greater")
            row["mw_stat"]   = round(float(mw_stat), 2)
            row["mw_p_1s"]   = round(float(mw_p), 5)
            row["mw_reject"] = bool(mw_p < alpha)
        except Exception:
            row["mw_stat"]   = float("nan")
            row["mw_p_1s"]   = float("nan")
            row["mw_reject"] = None
    else:
        row["mw_stat"]   = float("nan")
        row["mw_p_1s"]   = float("nan")
        row["mw_reject"] = None

    # ── T5. Kolmogorov-Smirnov 2-campioni ────────────────────────────────────
    if nb >= MIN_TWO and na >= MIN_TWO:
        try:
            ks_stat, ks_p = ks_2samp(b, a, alternative="greater")
            row["ks_stat"]   = round(float(ks_stat), 4)
            row["ks_p"]      = round(float(ks_p), 5)
            row["ks_reject"] = bool(ks_p < alpha)
        except Exception:
            row["ks_stat"]   = float("nan")
            row["ks_p"]      = float("nan")
            row["ks_reject"] = None
    else:
        row["ks_stat"]   = float("nan")
        row["ks_p"]      = float("nan")
        row["ks_reject"] = None

    # ── T6. Mood's median test ────────────────────────────────────────────────
    if nb >= MIN_TWO and na >= MIN_TWO:
        try:
            stat_mood, p_mood, _, _ = median_test(b, a)
            row["mood_stat"]   = round(float(stat_mood), 4)
            row["mood_p"]      = round(float(p_mood), 5)
            row["mood_reject"] = bool(p_mood < alpha)
        except Exception:
            row["mood_stat"]   = float("nan")
            row["mood_p"]      = float("nan")
            row["mood_reject"] = None
    else:
        row["mood_stat"]   = float("nan")
        row["mood_p"]      = float("nan")
        row["mood_reject"] = None

    # ── Effect sizes ─────────────────────────────────────────────────────────
    # ES1. Hodges-Lehmann one-sample
    hl = _hodges_lehmann_1s(b)
    row["hodges_lehmann_eurl"] = round(hl, 6) if math.isfinite(hl) else float("nan")

    # ES2. P(post_resid > 0)
    if nb > 0:
        row["prob_resid_pos"] = round(float(np.mean(b > 0)), 4)
    else:
        row["prob_resid_pos"] = float("nan")

    # ES3. Cliff's δ (post vs pre)
    cd = _cliffs_delta(a, b)
    row["cliffs_delta"]  = round(cd, 4) if math.isfinite(cd) else float("nan")
    row["cliffs_interp"] = _interpret_cliffs(cd)

    # ES4. Cohen d (post vs pre)
    cohe = _cohens_d(a, b)
    row["cohens_d"]       = round(cohe, 4) if math.isfinite(cohe) else float("nan")
    row["cohens_d_interp"] = _interpret_cohens_d(cohe)

    # ── Verdetto combinato (majority vote) ────────────────────────────────────
    test_rejects: list[bool | None] = [
        row["sign_reject"],
        row["wilcoxon_1s_reject"],
        row["perm_reject"],
        row["mw_reject"],
        row["ks_reject"],
        row["mood_reject"],
    ]
    valid_results = [r for r in test_rejects if r is not None]
    n_valid    = len(valid_results)
    n_reject   = sum(valid_results)
    threshold  = math.ceil(n_valid / 2) if n_valid > 0 else 0

    row["n_tests_valid"]  = n_valid
    row["n_tests_reject"] = n_reject
    row["alpha"]          = alpha

    if n_valid == 0:
        row["verdict"] = "INDETERMINATO"
        row["h0_rejected"] = None
    elif n_reject >= threshold and n_reject > 0:
        row["verdict"]    = "H0_RIGETTATA"
        row["h0_rejected"] = True
    else:
        row["verdict"]    = "H0_NON_RIGETTATA"
        row["h0_rejected"] = False

    return row


# ══════════════════════════════════════════════════════════════════════════════
# Utilità: stampa tabella risultati
# ══════════════════════════════════════════════════════════════════════════════

def print_battery_results(
    row: dict,
    label: str = "",
    alpha: float = 0.05,
) -> None:
    """Stampa a video il riepilogo della batteria per un singolo (evento, carburante)."""
    ALPHA = alpha
    sep = "─" * 70

    def _pstr(p) -> str:
        if p is None or not math.isfinite(float(p)):
            return "N/D "
        pf = float(p)
        flag = " *" if pf < ALPHA else "  "
        return f"{pf:.4f}{flag}"

    def _rej(r) -> str:
        if r is None:
            return "N/D"
        return "RIFIUTA" if r else "non rif."

    print(f"\n  {label}")
    print(f"  {sep}")
    print(f"  n_post={row['n_post']}  n_pre={row['n_pre']}"
          f"  |  post_media={row['post_mean']:+.5f} €/L"
          f"  post_mediana={row['post_median']:+.5f} €/L")
    print(f"  {'Test':<40} {'p-value':>10}  Decisione")
    print(f"  {sep}")
    print(f"  {'T1. Sign test (binom, one-sided)':<40} {_pstr(row['sign_p_1s']):>10}  {_rej(row['sign_reject'])}"
          f"   P(>0)={row.get('prob_pos', float('nan')):.3f}")
    print(f"  {'T2. Wilcoxon signed-rank (1-camp.)':<40} {_pstr(row['wilcoxon_1s_p']):>10}  {_rej(row['wilcoxon_1s_reject'])}")
    print(f"  {'T3. Permutation test media':<40} {_pstr(row['perm_p_1s']):>10}  {_rej(row['perm_reject'])}"
          f"   (B={row['perm_n']})")
    print(f"  {'T4. Mann-Whitney U (post > pre)':<40} {_pstr(row['mw_p_1s']):>10}  {_rej(row['mw_reject'])}")
    print(f"  {'T5. KS 2-campioni (post > pre)':<40} {_pstr(row['ks_p']):>10}  {_rej(row['ks_reject'])}")
    print(f"  {'T6. Mood median test':<40} {_pstr(row['mood_p']):>10}  {_rej(row['mood_reject'])}")
    print(f"  {sep}")
    print(f"  Effect sizes:")
    print(f"    Hodges-Lehmann (1-campione)  = {row['hodges_lehmann_eurl']:+.5f} €/L")
    print(f"    P(resid_post > 0)            = {row['prob_resid_pos']:.3f}")
    print(f"    Cliff δ (post vs pre)        = {row['cliffs_delta']:+.3f}  ({row['cliffs_interp']})")
    print(f"    Cohen d (post vs pre)        = {row['cohens_d']:+.3f}  ({row['cohens_d_interp']})")
    print(f"  {sep}")
    icon = "🔴" if row["verdict"] == "H0_RIGETTATA" else ("🟡" if row["verdict"] == "INDETERMINATO" else "🟢")
    print(f"  {icon} VERDETTO: {row['verdict']}"
          f"  ({row['n_tests_reject']}/{row['n_tests_valid']} test rifiutano H₀ a α={alpha})")