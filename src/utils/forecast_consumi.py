#!/usr/bin/env python3
"""
utils/forecast_consumi.py
─────────────────────────
Legge i consumi mensili da ../data/consumi/bilancio.xlsx e produce
  data/consumi/consumi_giornalieri.csv

che tutti i metodi ITS leggono per calcolare i profitti in M€.

METODO (universale):
  Per ogni giorno D nel range generato:
    - Cerca qta del mese di D nel dataset storico
    - daily_rate = qta_mese / giorni_reali_mese  (28/29/30/31)
    - Se il mese manca → carry-forward del mese noto più recente

CONVERSIONE mila t → litri:
  Benzina : × 1.000 × DENSITY_BENZINA_L_T   (default 1337 L/t, ~748 kg/m³)
  Gasolio : × 1.000 × DENSITY_GASOLIO_L_T   (default 1198 L/t, ~835 kg/m³)

CSV prodotto (data/consumi/consumi_giornalieri.csv):
  data | benzina_mila_t | gasolio_mila_t | benzina_L | gasolio_L | fonte_benzina | fonte_gasolio

Uso standalone (rigenera il CSV):
  python3 utils/forecast_consumi.py

Uso da altri script (import):
  from utils.forecast_consumi import load_daily_consumption
  s = load_daily_consumption(post_data.index, "benzina")  # → pd.Series L/giorno
"""

import calendar
import sys
from pathlib import Path

import pandas as pd

# ── Percorsi ──────────────────────────────────────────────────────────────────
UTILS_DIR   = Path(__file__).resolve().parent          # src/utils/
SRC_DIR     = UTILS_DIR.parent                         # src/
EXCEL_PATH  = SRC_DIR / "data" / "consumi" / "bilancio.xlsx"
OUT_CSV     = SRC_DIR / "data" / "consumi" / "consumi_giornalieri.csv"

# Intervallo date generato nel CSV (copre tutti gli eventi della pipeline)
CSV_START = pd.Timestamp("2021-01-01")
CSV_END   = pd.Timestamp("2027-12-31")

# ── Densità standard (litri per tonnellata) ────────────────────────────────────
# Benzina senza piombo 95: ~748 kg/m³  → 1 t = 1000/0.748 ≈ 1337 L
# Gasolio EN590:           ~835 kg/m³  → 1 t = 1000/0.835 ≈ 1198 L
DENSITY_L_T = {
    "benzina": 1_337,
    "gasolio": 1_198,
}

TIPOS = ["Benzina", "Gasolio"]   # nomi canonici nel file Excel

# ── Parsing periodo ────────────────────────────────────────────────────────────

def _parse_periodo(val) -> pd.Timestamp | None:
    if pd.isna(val):
        return None
    if isinstance(val, pd.Timestamp):
        return val.replace(day=1)
    if hasattr(val, "year"):
        return pd.Timestamp(val).replace(day=1)
    s = str(val).strip()
    mesi_it = {
        "gen": 1, "feb": 2, "mar": 3, "apr": 4,
        "mag": 5, "giu": 6, "lug": 7, "ago": 8,
        "set": 9, "ott": 10, "nov": 11, "dic": 12,
    }
    if "-" in s and len(s) == 6:
        parts = s.lower().split("-")
        if parts[0] in mesi_it:
            return pd.Timestamp(2000 + int(parts[1]), mesi_it[parts[0]], 1)
    try:
        return pd.to_datetime(s).replace(day=1)
    except Exception:
        return None

# ── Normalizzazione tipo ───────────────────────────────────────────────────────

_TIPO_ALIASES = {
    "benzina": "Benzina", "benizna": "Benzina", "benz": "Benzina",
    "gasolio": "Gasolio", "gasoil":  "Gasolio", "diesel": "Gasolio",
}

def _normalize_tipo(val: str) -> str:
    v = str(val).strip().lower()
    if v in _TIPO_ALIASES:
        return _TIPO_ALIASES[v]
    for alias, canonical in _TIPO_ALIASES.items():
        if alias in v:
            return canonical
    return str(val).strip().capitalize()

# ── Lettura Excel ──────────────────────────────────────────────────────────────

def _find_col(wanted: str, cols: list) -> str:
    w = wanted.lower()
    for c in cols:
        if str(c).lower() == w:
            return c
    for c in cols:
        if w in str(c).lower():
            return c
    raise KeyError(
        f"Colonna '{wanted}' non trovata. Colonne: {cols}\n"
        f"Usa --col-periodo / --col-tipo / --col-qta per specificarle."
    )

def load_excel(path: Path = EXCEL_PATH,
               sheet: str | None = None,
               col_periodo: str = "periodo",
               col_tipo:    str = "tipo",
               col_qta:     str = "quantità") -> pd.DataFrame:
    """
    Carica bilancio.xlsx e restituisce DataFrame normalizzato:
      data (Timestamp 1°mese) | tipo | qta_mila_t
    """
    if not path.exists():
        print(f"  ✗  File Excel non trovato: {path}", file=sys.stderr)
        sys.exit(1)

    xf     = pd.ExcelFile(path)
    sheets = xf.sheet_names
    tgt    = sheet if (sheet and sheet in sheets) else sheets[0]
    df     = xf.parse(tgt)
    cols   = list(df.columns)

    try:
        c_per  = _find_col(col_periodo, cols)
        c_tipo = _find_col(col_tipo,    cols)
        c_qta  = _find_col(col_qta,     cols)
    except KeyError as e:
        print(f"  ✗  {e}", file=sys.stderr)
        sys.exit(1)

    out = pd.DataFrame()
    out["data"]  = df[c_per].apply(_parse_periodo)
    out["tipo"]  = df[c_tipo].astype(str).apply(_normalize_tipo)
    raw_qta      = pd.to_numeric(df[c_qta], errors="coerce")

    # Auto-detect unità: se mediana > 10.000 sono tonnellate → converti in mila t
    if raw_qta.median() > 10_000:
        out["qta_mila_t"] = raw_qta / 1_000
    else:
        out["qta_mila_t"] = raw_qta

    out = out.dropna(subset=["data", "qta_mila_t"])
    out = out[out["tipo"].isin(["Benzina", "Gasolio"])]
    return out.sort_values(["tipo", "data"]).reset_index(drop=True)


# ── Lookup mensile ─────────────────────────────────────────────────────────────

def _build_lookup(hist: pd.DataFrame, tipo: str) -> dict:
    """{ (year, month): daily_rate_mila_t }"""
    sub    = hist[hist["tipo"] == tipo]
    lookup = {}
    for _, row in sub.iterrows():
        y = row["data"].year
        m = row["data"].month
        lookup[(y, m)] = row["qta_mila_t"] / calendar.monthrange(y, m)[1]
    return lookup


# ── Costruzione serie giornaliera ─────────────────────────────────────────────

def build_daily_series(hist: pd.DataFrame,
                       start: pd.Timestamp = CSV_START,
                       end:   pd.Timestamp = CSV_END) -> pd.DataFrame:
    """
    Serie giornaliera completa [start, end] per Benzina e Gasolio.
    Carry-forward per i mesi non presenti nel dataset.
    """
    date_range = pd.date_range(start=start, end=end, freq="D")
    rows = []

    for tipo in TIPOS:
        fuel_key = tipo.lower()
        lookup   = _build_lookup(hist, tipo)

        # Primo carry-forward disponibile
        last_rate  = None
        last_label = "n/d"
        for (y, m) in sorted(lookup.keys()):
            last_rate  = lookup[(y, m)]
            last_label = f"{y}-{m:02d}"
            break  # prende il primo disponibile per date antecedenti

        for d in date_range:
            key = (d.year, d.month)
            if key in lookup:
                rate       = lookup[key]
                fonte      = "storico"
                last_rate  = rate
                last_label = d.strftime("%Y-%m")
            else:
                rate  = last_rate if last_rate is not None else 0.0
                fonte = f"carry ({last_label})"

            rows.append({
                "data":         d,
                "tipo":         tipo,
                "mila_t_giorno": round(rate, 5) if rate else 0.0,
                "L_giorno":     round(rate * 1_000 * DENSITY_L_T[fuel_key], 0)
                                if rate else 0.0,
                "fonte":        fonte,
            })

    df = pd.DataFrame(rows)
    # Pivot: una riga per data, colonne per tipo
    piv = df.pivot(index="data", columns="tipo",
                   values=["mila_t_giorno", "L_giorno", "fonte"])
    piv.columns = [f"{t.lower()}_{v}" for v, t in piv.columns]
    piv = piv.reset_index().rename(columns={
        "benzina_mila_t_giorno": "benzina_mila_t",
        "gasolio_mila_t_giorno": "gasolio_mila_t",
        "benzina_L_giorno":      "benzina_L",
        "gasolio_L_giorno":      "gasolio_L",
        "benzina_fonte":         "fonte_benzina",
        "gasolio_fonte":         "fonte_gasolio",
    })
    return piv


# ── API pubblica per gli altri script ─────────────────────────────────────────

_cache: pd.DataFrame | None = None   # cache in-process per evitare reletture

def _get_master(csv_path: Path = OUT_CSV) -> pd.DataFrame:
    global _cache
    if _cache is None:
        if not csv_path.exists():
            print(f"  ⚠  {csv_path} non trovato — rigenero da Excel...",
                  file=sys.stderr)
            main()   # rigenera
        _cache = pd.read_csv(csv_path, parse_dates=["data"])
    return _cache


def load_daily_consumption(dates: pd.DatetimeIndex,
                           fuel_key: str,
                           csv_path: Path = OUT_CSV) -> pd.Series:
    """
    Restituisce una pd.Series con i litri/giorno per ogni data in `dates`.

    Parametri:
      dates    : DatetimeIndex delle date post-breakpoint del metodo ITS
      fuel_key : "benzina" oppure "gasolio"
      csv_path : percorso del CSV master (default: data/consumi/consumi_giornalieri.csv)

    Uso tipico nei metodi ITS:
      from utils.forecast_consumi import load_daily_consumption
      cons = load_daily_consumption(post_data.index, fuel_key)
      gain_meur = float((extra * cons).sum() / 1e6)
    """
    col  = f"{fuel_key.lower()}_L"
    df   = _get_master(csv_path).set_index("data")
    vals = df[col].reindex(dates, method="ffill")   # ffill per weekend/festivi
    vals.index = dates
    return vals


# ── Main (rigenera il CSV) ────────────────────────────────────────────────────

def main(excel: Path = EXCEL_PATH, out: Path = OUT_CSV) -> None:
    print("=" * 60)
    print("  forecast_consumi.py  —  generazione CSV master")
    print(f"  Excel : {excel}")
    print(f"  Output: {out}")
    print("=" * 60)

    hist  = load_excel(excel)
    print(f"  Righe caricate: {len(hist)}")

    daily = build_daily_series(hist)
    out.parent.mkdir(parents=True, exist_ok=True)
    daily.to_csv(out, index=False)

    n_storico = (daily["fonte_benzina"] == "storico").sum()
    n_carry   = (daily["fonte_benzina"].str.startswith("carry")).sum()
    print(f"  Giorni totali : {len(daily)}")
    print(f"  Da dato reale : {n_storico}")
    print(f"  Carry-forward : {n_carry}")
    print(f"  Salvato       : {out}")

    # Campione di verifica
    sample = daily[daily["data"].dt.is_month_start].head(8)[
        ["data", "benzina_mila_t", "gasolio_mila_t",
         "benzina_L", "gasolio_L", "fonte_benzina"]]
    print("\n  Campione (primo giorno di ogni mese):")
    print(sample.to_string(index=False))
    print("=" * 60)


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--excel", default=str(EXCEL_PATH))
    p.add_argument("--out",   default=str(OUT_CSV))
    a = p.parse_args()
    main(Path(a.excel), Path(a.out))