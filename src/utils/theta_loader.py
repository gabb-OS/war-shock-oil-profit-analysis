"""
utils/theta_loader.py
─────────────────────
Funzione condivisa per caricare il break canonico θ dal file
theta_results.csv prodotto da 02c_change_point_detection.py.

Usata dai metodi ITS (v1–v4) in modalità --mode detected
al posto dei loro algoritmi di detection autonomi.

Schema theta_results.csv:
  evento, shock, carburante, detect_type, theta, lr_stat, p_value,
  theta_confirmed, L1_tau, L1_delta_eurl, L1_p_bonf, L2_cusum,
  L3_binseg, L4_pelt, Lw_theta_old, Lw_d_max
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd


def load_theta(
    event_name:  str,
    fuel_key:    str,
    detect_type: str,
    base_dir:    Path | None = None,
    strict:      bool        = False,
) -> pd.Timestamp | None:
    """
    Restituisce il break canonico θ (GLM Poisson) per la combinazione
    (event_name, fuel_key, detect_type), oppure None se non trovato.

    Parameters
    ----------
    event_name  : nome dell'evento, es. "Ucraina (Feb 2022)"
    fuel_key    : "benzina" o "gasolio"
    detect_type : "margin" o "price"
    base_dir    : directory radice del progetto (default: parent del file chiamante)
    strict      : se True, solleva ValueError se il csv non esiste o la riga
                  non viene trovata; altrimenti restituisce None silenziosamente

    Returns
    -------
    pd.Timestamp | None
    """
    if base_dir is None:
        # Risale di due livelli: utils/ → cartella principale
        base_dir = Path(__file__).parent.parent

    csv_path = (base_dir / "data" / "plots" / "change_point"
                / detect_type / "theta_results.csv")

    if not csv_path.exists():
        msg = (
            f"theta_results.csv non trovato in {csv_path}.\n"
            f"Esegui prima: python3 02c_change_point_detection.py --detect {detect_type}"
        )
        if strict:
            raise FileNotFoundError(msg)
        print(f"  ⚠  {msg}")
        return None

    df = pd.read_csv(csv_path, dtype=str)

    mask = (
        (df["evento"].str.strip()      == event_name.strip()) &
        (df["carburante"].str.strip()  == fuel_key.strip()) &
        (df["detect_type"].str.strip() == detect_type.strip())
    )
    rows = df[mask]

    if rows.empty:
        msg = (
            f"Nessuna riga in theta_results.csv per:\n"
            f"  evento='{event_name}'  carburante='{fuel_key}'  detect='{detect_type}'\n"
            f"  Esegui: python3 02c_change_point_detection.py --detect {detect_type}"
        )
        if strict:
            raise ValueError(msg)
        print(f"  ⚠  {msg}")
        return None

    theta_str = rows.iloc[0]["theta"]
    return pd.Timestamp(theta_str)


def load_theta_results(
    detect_type: str,
    base_dir:    Path | None = None,
) -> pd.DataFrame:
    """
    Carica l'intero theta_results.csv per un dato detect_type.
    Restituisce DataFrame vuoto se il file non esiste.
    """
    if base_dir is None:
        base_dir = Path(__file__).parent.parent

    csv_path = (base_dir / "data" / "plots" / "change_point"
                / detect_type / "theta_results.csv")

    if not csv_path.exists():
        print(f"  ⚠  theta_results.csv non trovato: {csv_path}")
        return pd.DataFrame()

    df = pd.read_csv(csv_path, dtype=str)
    # Converti theta in Timestamp
    if "theta" in df.columns:
        df["theta"] = pd.to_datetime(df["theta"], errors="coerce")
    return df