#!/usr/bin/env python3
"""
01_data_ingestion.py
====================
Step 1 della pipeline crack-spread Eurobob / prezzo-pompa esentasse.

Fonti:
  - MIMIT/MISE : tar trimestrali prezzi giornalieri per impianto
  - MIMIT/MISE : tar trimestrali anagrafica (per filtro autostradale)
  - SISEN/MASE : CSV settimanale prezzi + tasse

Output:
  data/processed/daily_fuel_prices_all.csv
  data/processed/daily_fuel_prices_stradale.csv

Uso:
  pip install -r requirements.txt
  python 01_data_ingestion.py                         # run completo
  python 01_data_ingestion.py --keep-cache            # mantieni i tar
  python 01_data_ingestion.py --quarter 2015 1        # solo Q1 2015
  python 01_data_ingestion.py --inspect-quarter 2015 1  # diagnosi struttura tar
  python 01_data_ingestion.py --inspect-sisen         # diagnosi CSV SISEN
"""

import argparse
import contextlib
import io
import json
import logging
import re
import tarfile
import tempfile
import textwrap
from datetime import date
from pathlib import Path
from typing import Iterator, Optional

import numpy as np
import pandas as pd
import requests
from tqdm import tqdm

# ── Logging ────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ── Configurazione ─────────────────────────────────────────────────────────────
START_DATE = date(2015, 1, 1)
END_DATE   = date(2026, 3, 31)

BASE_DIR  = Path(__file__).parent
DATA_DIR  = BASE_DIR / "data"
RAW_DIR   = DATA_DIR / "raw"
TAR_CACHE = RAW_DIR / "tars"
PROC_DIR  = DATA_DIR / "processed"

OUTPUT_ALL      = PROC_DIR / "daily_fuel_prices_all.csv"
OUTPUT_STRADALE = PROC_DIR / "daily_fuel_prices_stradale.csv"

_MISE_BASE         = "https://opendatacarburanti.mise.gov.it/categorized"
PREZZO_URL_TPL     = _MISE_BASE + "/prezzo_alle_8/{y}/{y}_{q}_tr.tar.gz"
ANAGRAFICA_URL_TPL = _MISE_BASE + "/anagrafica_impianti_attivi/{y}/{y}_{q}_tr.tar.gz"
SISEN_URL          = (
    "https://sisen.mase.gov.it/dgsaie/api/v1/"
    "weekly-prices/report/export?type=ALL&format=CSV&lang=it"
)

FUEL_TYPES = {"Benzina", "Gasolio"}
PRICE_MIN, PRICE_MAX = 0.50, 3.50
MIN_OBS = 5

ACCISE_JSON = DATA_DIR / "accise_variazioni.json"

SESSION = requests.Session()
SESSION.headers.update({"User-Agent": "Mozilla/5.0 (crack-spread-research/1.0)"})


# ── Helpers trimestri ──────────────────────────────────────────────────────────
def quarters_in_range(start: date, end: date) -> list[tuple[int, int]]:
    result = []
    y, q = start.year, (start.month - 1) // 3 + 1
    ey, eq = end.year, (end.month - 1) // 3 + 1
    while (y, q) <= (ey, eq):
        result.append((y, q))
        q += 1
        if q > 4:
            q, y = 1, y + 1
    return result


# ── Download ───────────────────────────────────────────────────────────────────
def download_file(url: str, dest: Path, desc: str = "") -> bool:
    try:
        dest.parent.mkdir(parents=True, exist_ok=True)
        r = SESSION.get(url, stream=True, timeout=180)
        r.raise_for_status()
        total = int(r.headers.get("content-length", 0))
        with open(dest, "wb") as fh, tqdm(
            total=total, unit="B", unit_scale=True, desc=desc or dest.name, leave=False
        ) as bar:
            for chunk in r.iter_content(chunk_size=1 << 17):
                fh.write(chunk)
                bar.update(len(chunk))
        return True
    except requests.HTTPError as e:
        log.warning("HTTP %s per %s", e.response.status_code, url)
    except Exception as e:
        log.warning("Errore scaricando %s: %s", url, e)
    return False


@contextlib.contextmanager
def quarter_workspace(year: int, quarter: int, keep_cache: bool):
    if keep_cache:
        d = TAR_CACHE
        d.mkdir(parents=True, exist_ok=True)
        yield d
    else:
        with tempfile.TemporaryDirectory(prefix=f"crack_{year}q{quarter}_") as tmpdir:
            yield Path(tmpdir)


# ── Separatore ────────────────────────────────────────────────────────────────
def detect_sep(first_bytes: bytes) -> str:
    line = first_bytes.decode("utf-8", errors="replace").split("\n")[0]
    counts = {"|": line.count("|"), ";": line.count(";"), ",": line.count(",")}
    return max(counts, key=lambda k: counts[k])


# ── Lettura CSV robusta (gestisce header "Estrazione del YYYY-MM-DD") ──────────
_ESTRAZIONE_RE = re.compile(r"^Estrazione\s+del\s+\d{4}-\d{2}-\d{2}", re.I)

def _smart_read_csv(content: bytes, **kwargs) -> pd.DataFrame:
    """
    Alcuni tar MIMIT (es. 2015 Q2 in poi) hanno come prima riga
    'Estrazione del YYYY-MM-DD' prima degli header reali.
    Rileva automaticamente il separatore dalla riga corretta
    (non da "Estrazione del..." che non contiene separatori dati).
    """
    lines = content.split(b"\n")
    first_line = lines[0].decode("utf-8", errors="replace").strip()
    if _ESTRAZIONE_RE.match(first_line):
        log.debug("   Rilevata riga \'Estrazione del …\' — saltata come header spuria.")
        skiprows = 1
        sep_line = lines[1].decode("utf-8", errors="replace") if len(lines) > 1 else first_line
    else:
        skiprows = None
        sep_line = first_line
    sep = detect_sep(sep_line.encode())
    return pd.read_csv(
        io.BytesIO(content), sep=sep, header=0,
        skiprows=skiprows, dtype=str,
        on_bad_lines="skip", engine="python", **kwargs
    )


# ── Data da nome file ─────────────────────────────────────────────────────────
def extract_date(name: str) -> Optional[date]:
    """
    Prova vari formati di data nel nome file:
      20150101           (8 cifre consecutive)
      2015-01-01         (ISO con trattino)
      2015_01_01         (ISO con underscore)
      01-01-2015         (DD-MM-YYYY con trattino)
      01012015           (DDMMYYYY 8 cifre — meno comune)
    """
    # YYYYMMDD
    m = re.search(r"(?<!\d)(\d{4})(\d{2})(\d{2})(?!\d)", name)
    if m:
        try:
            return date(int(m.group(1)), int(m.group(2)), int(m.group(3)))
        except ValueError:
            pass
    # YYYY-MM-DD o YYYY_MM_DD
    m = re.search(r"(\d{4})[-_](\d{2})[-_](\d{2})", name)
    if m:
        try:
            return date(int(m.group(1)), int(m.group(2)), int(m.group(3)))
        except ValueError:
            pass
    # DD-MM-YYYY o DD_MM_YYYY
    m = re.search(r"(\d{2})[-_](\d{2})[-_](\d{4})", name)
    if m:
        try:
            return date(int(m.group(3)), int(m.group(2)), int(m.group(1)))
        except ValueError:
            pass
    return None


# ── Anagrafica ────────────────────────────────────────────────────────────────
def load_stradale_ids(tar_path: Path) -> Optional[set[int]]:
    try:
        with tarfile.open(tar_path, "r:*") as tf:
            csv_members = [m for m in tf.getmembers() if m.name.endswith(".csv")]
            if not csv_members:
                log.warning("Nessun CSV trovato in %s", tar_path.name)
                return None
            member  = max(csv_members, key=lambda m: m.size)
            content = tf.extractfile(member).read()

        df  = _smart_read_csv(content)
        df.columns = [c.strip() for c in df.columns]

        id_col   = next((c for c in df.columns if re.match(r"id.?impianto", c, re.I)), None)
        tipo_col = next((c for c in df.columns if re.match(r"tipo.?impianto", c, re.I)), None)

        if id_col is None or tipo_col is None:
            raise RuntimeError(
                f"Anagrafica {tar_path.name}: colonne 'idImpianto' / 'TipoImpianto' non trovate.\n"
                f"Colonne presenti: {df.columns.tolist()}\n"
                f"Prime 3 righe:\n{df.head(3).to_string()}\n\n"
                "→ Correggi il parsing (es. separatore sbagliato o nuova struttura) e ri-runnare."
            )

        # ── DIAGNOSTICA: mostra i valori unici reali ──────────────────────────
        unique_tipo = df[tipo_col].str.strip().value_counts()
        log.info("   Valori unici '%s': %s", tipo_col, unique_tipo.to_dict())

        # Escludi solo gli impianti autostradali — teniamo "Strada Statale", "Statale", "Altro" ecc.
        # Il valore cambia nel tempo ("Strada Statale" → "Statale"), match su sottostringa
        tipo_norm = df[tipo_col].str.strip().str.lower()
        mask = ~tipo_norm.str.contains("autostradale", na=False)

        if mask.sum() == 0:
            log.warning(
                "   Nessun impianto non-autostradale trovato — "
                "controlla i valori sopra e aggiorna il filtro se necessario."
            )

        ids = pd.to_numeric(df.loc[mask, id_col], errors="coerce").dropna().astype(int)
        result_set = set(ids)
        total = df[id_col].notna().sum()
        log.info("   → %d totale | %d stradali | %d altro", total, len(result_set),
                 total - len(result_set))
        return result_set if result_set else None   # None = non filtrare

    except Exception as e:
        log.warning("Errore leggendo anagrafica %s: %s", tar_path.name, e)
        return None


# ── Parsing CSV giornaliero ───────────────────────────────────────────────────
def _normalise_price_cols(df: pd.DataFrame) -> pd.DataFrame:
    rename: dict[str, str] = {}
    for col in df.columns:
        c = col.strip().lower().replace(" ", "").replace("_", "")
        if c in ("desccarburante", "tipocarburante", "carburante", "nome"):
            rename[col] = "descCarburante"
        elif c in ("self", "selfservice", "self_service", "isself", "is_self"):
            rename[col] = "self"
        elif c == "prezzo":
            rename[col] = "prezzo"
        elif c in ("idimpianto", "id", "codicepv"):
            rename[col] = "idImpianto"
    return df.rename(columns=rename)


def iter_daily_csvs(tar_path: Path) -> Iterator[tuple[date, pd.DataFrame]]:
    found_any = False
    with tarfile.open(tar_path, "r:*") as tf:
        members = sorted(tf.getmembers(), key=lambda m: m.name)

        # ── DIAGNOSTICA: logga i primi file trovati ───────────────────────────
        names = [m.name for m in members if not m.isdir()]
        if names:
            log.info("   Tar contiene %d file. Esempi: %s", len(names), names[:5])
        else:
            log.warning("   Tar vuoto o senza file regolari.")
            return

        for member in members:
            if member.isdir() or not member.name.endswith(".csv"):
                continue

            d = extract_date(member.name)
            if d is None:
                log.debug("   Data non trovata in: %s", member.name)
                continue
            if not (START_DATE <= d <= END_DATE):
                continue

            raw_file = tf.extractfile(member)
            if raw_file is None:
                continue
            content = raw_file.read()
            if not content:
                continue

            try:
                df = _smart_read_csv(content)
            except Exception as e:
                log.debug("Impossibile leggere %s: %s", member.name, e)
                continue

            df = _normalise_price_cols(df)

            # ── DIAGNOSTICA: mostra colonne del primo file processato ──────────
            if not found_any:
                log.info("   Colonne CSV prezzo: %s", df.columns.tolist())
                required = {"prezzo", "descCarburante", "self"}
                missing  = required - set(df.columns)
                if missing:
                    raise RuntimeError(
                        f"CSV prezzi {member.name}: colonne mancanti dopo normalizzazione: {missing}\n"
                        f"Colonne trovate: {df.columns.tolist()}\n"
                        f"Prime 3 righe:\n{df.head(3).to_string()}\n\n"
                        "→ Aggiorna _normalise_price_cols() o la logica di parsing e ri-runnare."
                    )
                found_any = True

            for col, dtype in [("prezzo", "float32"), ("self", "Int8"), ("idImpianto", "Int32")]:
                if col in df.columns:
                    df[col] = (df[col].astype(str)
                                      .str.replace(",", ".", regex=False)
                                      .pipe(pd.to_numeric, errors="coerce")
                                      .astype(dtype))

            yield d, df

    if not found_any:
        raise RuntimeError(
            f"Nessun CSV giornaliero valido trovato nel tar {tar_path.name}.\n"
            "Controlla i nomi file e le colonne con --inspect-quarter ANNO Q, poi ri-runnare."
        )


def _aggregate_subset(df: pd.DataFrame) -> dict[str, float]:
    out: dict[str, float] = {}
    for fuel in FUEL_TYPES:
        mask   = df["descCarburante"].str.strip() == fuel
        prices = pd.to_numeric(df.loc[mask, "prezzo"], errors="coerce").dropna()
        prices = prices[(prices >= PRICE_MIN) & (prices <= PRICE_MAX)]
        if len(prices) >= MIN_OBS:
            out[fuel] = float(prices.mean())   # media aritmetica (non mediana)
    return out


def aggregate_daily_both(
    df: pd.DataFrame, stradale_ids: Optional[set[int]]
) -> tuple[dict[str, float], dict[str, float]]:
    if not {"self", "prezzo", "descCarburante"}.issubset(df.columns):
        return {}, {}
    self_df = df[df["self"] == 1]
    agg_all = _aggregate_subset(self_df)
    if stradale_ids is not None and "idImpianto" in self_df.columns:
        agg_stradale = _aggregate_subset(self_df[self_df["idImpianto"].isin(stradale_ids)])
    else:
        agg_stradale = {}
    return agg_all, agg_stradale


# ── Pipeline per trimestre ────────────────────────────────────────────────────
def process_quarter(year: int, quarter: int, keep_cache: bool) -> tuple[list[dict], list[dict]]:
    log.info("─── %d Q%d ───────────────────────────────────────", year, quarter)

    with quarter_workspace(year, quarter, keep_cache) as workdir:

        # anagrafica
        ana_name = f"anagrafica_{year}_{quarter}.tar.gz"
        ana_path = workdir / ana_name
        stradale_ids: Optional[set[int]] = None
        if not ana_path.exists():
            log.info("⬇  Anagrafica %d Q%d …", year, quarter)
            ok = download_file(ANAGRAFICA_URL_TPL.format(y=year, q=quarter), ana_path, desc=ana_name)
        else:
            log.info("✓  Cache anagrafica %d Q%d", year, quarter); ok = True
        if ok:
            stradale_ids = load_stradale_ids(ana_path)
            if not keep_cache:
                ana_path.unlink(missing_ok=True); log.info("   Anagrafica cancellata.")
        else:
            log.warning("   Anagrafica non disponibile.")

        # prezzi
        prezzo_name = f"prezzo_{year}_{quarter}.tar.gz"
        prezzo_path = workdir / prezzo_name
        if not prezzo_path.exists():
            log.info("⬇  Prezzi    %d Q%d …", year, quarter)
            ok = download_file(PREZZO_URL_TPL.format(y=year, q=quarter), prezzo_path, desc=prezzo_name)
        else:
            log.info("✓  Cache prezzi    %d Q%d", year, quarter); ok = True
        if not ok:
            log.warning("   Prezzi non disponibili — trimestre saltato.")
            return [], []

        rows_all: list[dict] = []
        rows_str: list[dict] = []
        for d, df in iter_daily_csvs(prezzo_path):
            a_all, a_str = aggregate_daily_both(df, stradale_ids)
            if a_all:
                rows_all.append({"date": d, "benzina_pump": a_all.get("Benzina"),
                                              "gasolio_pump": a_all.get("Gasolio")})
            if a_str:
                rows_str.append({"date": d, "benzina_pump": a_str.get("Benzina"),
                                              "gasolio_pump": a_str.get("Gasolio")})

        if not keep_cache:
            prezzo_path.unlink(missing_ok=True); log.info("   Prezzi cancellati.")

        log.info("   → %d giorni (all) | %d giorni (stradale)", len(rows_all), len(rows_str))
    return rows_all, rows_str


# ── SISEN ─────────────────────────────────────────────────────────────────────
# Formato reale: space-separated, una riga per carburante per settimana
# Header: DATA_RILEVAZIONE CODICE_PRODOTTO NOME_PRODOTTO PREZZO ACCISA IVA NETTO VARIAZIONE
# Es:     2005-01-03       1               Benzina       1.115,75 558,64 185,96 371,15 -1,57
# Numeri in formato italiano: il punto è separatore migliaia, la virgola è decimale
# ACCISA e PREZZO in €/1000L → convertiamo in €/L dividendo per 1000

_SISEN_COL_MAP: dict[str, str] = {
    "data_rilevazione": "date",
    "data":             "date",
    "settimana":        "date",
    "data_settimana":   "date",
    "nome_prodotto":    "fuel",
    "nomeprodotto":     "fuel",
    "codice_prodotto":  "codice",
    "codiceprodotto":   "codice",
    "accisa":           "accisa",
    "iva":              "iva_rate",
    "aliquota_iva":     "iva_rate",
}


def _parse_it_number(s: pd.Series) -> pd.Series:
    """Converte formato italiano (1.115,75) in float (1115.75)."""
    return (s.astype(str)
             .str.strip()
             .str.replace(r"\.(?=\d{3})", "", regex=True)   # rimuove sep migliaia
             .str.replace(",", ".", regex=False)              # virgola decimale → punto
             .pipe(pd.to_numeric, errors="coerce"))


def _detect_sep(first_line: str) -> str:
    for sep in ("\t", ";", ","):
        if sep in first_line:
            return sep
    return r"\s+"   # space-separated (fallback)


def load_sisen() -> pd.DataFrame:
    dest = RAW_DIR / "sisen_prezzi_settimanali.csv"
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    if not dest.exists():
        log.info("⬇  Download SISEN settimanale …")
        ok = download_file(SISEN_URL, dest, desc="sisen_prezzi_settimanali.csv")
        if not ok:
            raise RuntimeError(f"Impossibile scaricare il CSV SISEN.\nURL: {SISEN_URL}")

    raw_text   = dest.read_text(encoding="utf-8-sig", errors="replace")
    first_line = raw_text.strip().split("\n")[0]
    sep        = _detect_sep(first_line)

    df = pd.read_csv(dest, sep=sep, encoding="utf-8-sig", on_bad_lines="skip",
                     engine="python", dtype=str)
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]
    log.info("SISEN raw: %d righe, colonne: %s", len(df), df.columns.tolist())

    rename = {c: _SISEN_COL_MAP[c] for c in df.columns if c in _SISEN_COL_MAP}
    df = df.rename(columns=rename)

    if "date" not in df.columns:
        raise ValueError(
            f"Colonna data non trovata nel CSV SISEN.\n"
            f"Colonne: {df.columns.tolist()}\n"
            "Esegui --inspect-sisen per vedere la struttura raw del file."
        )

    df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.date
    df = df.dropna(subset=["date"])

    # Determina tipo carburante da nome o codice
    # Convenzione SISEN: codice 1 = Benzina, 2 = Gasolio AUTO.
    # Usiamo il codice numerico come discriminante primario per evitare di
    # catturare "Gasolio riscaldamento" (cod. 3) con un semplice str.contains("gasolio").
    if "codice" in df.columns:
        is_benz = df["codice"].str.strip() == "1"
        is_gas  = df["codice"].str.strip() == "2"
    elif "fuel" in df.columns:
        fl = df["fuel"].str.strip().str.lower()
        is_benz = fl.str.contains("benzina", na=False)
        # Usa "gasolio auto" per escludere "gasolio riscaldamento"
        is_gas  = fl.str.contains("gasolio auto", na=False)
    else:
        raise ValueError("Impossibile identificare il tipo carburante nel CSV SISEN. "
                         "Colonne: " + str(df.columns.tolist()))

    if "accisa" not in df.columns:
        raise ValueError("Colonna 'accisa' non trovata nel CSV SISEN. "
                         "Colonne: " + str(df.columns.tolist()))

    # Colonne numeriche: prezzo pompa settimanale e netto (già presenti nel CSV)
    # Il cuneo fiscale = prezzo - netto  (accise + IVA insieme, senza algebra IVA)
    for col in ("accisa", "prezzo", "netto"):
        if col in df.columns:
            df[col] = _parse_it_number(df[col])

    # Pivot: una riga per data con benzina e gasolio separati
    benz_cols = ["date", "prezzo", "netto"]
    gas_cols  = ["date", "prezzo", "netto"]
    benz = (df[is_benz][benz_cols].copy()
            .rename(columns={"prezzo": "prezzo_benz_w", "netto": "netto_benz_w"}))
    gas  = (df[is_gas][gas_cols].copy()
            .rename(columns={"prezzo": "prezzo_gas_w",  "netto": "netto_gas_w"}))

    merged = pd.merge(benz, gas, on="date", how="outer").sort_values("date")
    merged = merged.drop_duplicates("date").reset_index(drop=True)

    # Converti €/1000L → €/L (i valori SISEN sono > 10 quando in €/1000L)
    for col in ("prezzo_benz_w", "netto_benz_w", "prezzo_gas_w", "netto_gas_w"):
        if col in merged.columns and merged[col].dropna().max() > 10:
            merged[col] = merged[col] / 1000

    # Cuneo fiscale settimanale = prezzo_pompa - netto  (accise + IVA reali, senza formula)
    merged["tax_wedge_benz"] = merged["prezzo_benz_w"] - merged["netto_benz_w"]
    merged["tax_wedge_gas"]  = merged["prezzo_gas_w"]  - merged["netto_gas_w"]

    log.info("SISEN: %d settimane (%s → %s)  cuneo medio benz=%.4f gas=%.4f €/L",
             len(merged), merged["date"].min(), merged["date"].max(),
             merged["tax_wedge_benz"].mean(), merged["tax_wedge_gas"].mean())
    return merged[["date", "tax_wedge_benz", "tax_wedge_gas"]]


# ── Build output ──────────────────────────────────────────────────────────────
# ── Caricamento variazioni accise ─────────────────────────────────────────────
def load_accise_changes(json_path: Path) -> list[pd.Timestamp]:
    """
    Legge accise_variazioni.json e restituisce la lista di data_effettiva
    per ogni evento non-pending.

    build_output calcola autonomamente la fine della finestra di correzione
    come il primo lunedì successivo a data_effettiva: da quel lunedì in poi
    il backward-merge standard prende già la riga SISEN corretta.
    """
    if not json_path.exists():
        log.warning(
            "accise_variazioni.json non trovato (%s) — nessuna correzione SISEN applicata.",
            json_path,
        )
        return []

    with open(json_path, encoding="utf-8") as fh:
        data = json.load(fh)

    changes: list[pd.Timestamp] = []
    for v in data.get("variazioni", []):
        if v.get("sisen_corretta") is None:
            log.info(
                "Variazione accise '%s' (%s) senza sisen_corretta — rinviata.",
                v["id"], v["data_effettiva"],
            )
            continue
        try:
            changes.append(pd.Timestamp(v["data_effettiva"]))
        except Exception as e:
            log.warning("Variazione '%s': data_effettiva non parsabile (%s)", v.get("id"), e)

    log.info("Variazioni accise caricate: %d eventi.", len(changes))
    return changes


def _next_monday(dt: pd.Timestamp) -> pd.Timestamp:
    """Primo lunedì strettamente successivo a dt."""
    days = (7 - dt.weekday()) % 7 or 7   # se dt è lunedì → 7, altrimenti 1-6
    return dt + pd.Timedelta(days=days)


# ── Build output ──────────────────────────────────────────────────────────────
def build_output(
    rows: list[dict],
    sisen: pd.DataFrame,
    accise_change_dates: Optional[list[pd.Timestamp]] = None,
) -> pd.DataFrame:
    """
    Unisce i prezzi giornalieri MIMIT con il cuneo fiscale settimanale SISEN.

    Problema: quando le accise cambiano a metà settimana SISEN, i giorni
    successivi al decreto (ma prima del lunedì seguente) vengono abbinati
    dal backward-merge alla riga SISEN contaminata (vecchia aliquota) →
    netto artificialmente basso → margine apparentemente compresso.

    Soluzione: per ogni data_effettiva, i giorni in
    (data_effettiva, next_monday(data_effettiva)) ricevono il tax_wedge
    del primo lunedì SISEN disponibile da quel lunedì in poi.
    Dal lunedì successivo il backward-merge standard è già corretto.
    """
    if not rows:
        return pd.DataFrame()

    daily = pd.DataFrame(rows).sort_values("date").reset_index(drop=True)
    sisen = sisen.sort_values("date").copy()

    daily["_dt"] = pd.to_datetime(daily["date"])
    sisen["_dt"] = pd.to_datetime(sisen["date"])

    # ── 1. Merge backward standard ────────────────────────────────────────────
    merged = pd.merge_asof(
        daily,
        sisen[["_dt", "tax_wedge_benz", "tax_wedge_gas"]],
        on="_dt",
        direction="backward",
    )

    # ── 2. Correzione settimane con cambio di accisa ──────────────────────────
    if accise_change_dates:
        sisen_sorted = sisen.sort_values("_dt").reset_index(drop=True)

        for change_dt in accise_change_dates:
            # Fine finestra: primo lunedì dopo data_effettiva.
            # Da quel lunedì il backward-merge standard prende già la riga
            # SISEN aggiornata — non serve correggere oltre.
            next_mon = _next_monday(change_dt)

            fwd_rows = sisen_sorted[sisen_sorted["_dt"] >= next_mon]
            if fwd_rows.empty:
                log.debug("Variazione %s: nessuna riga SISEN da %s — skip.",
                          change_dt.date(), next_mon.date())
                continue
            fwd_row = fwd_rows.iloc[0]

            # Giorni da correggere: (data_effettiva, next_monday) esclusivi
            # Il giorno del decreto mantiene il vecchio tax_wedge (< next_mon
            # garantisce che il lunedì successivo venga lasciato al merge standard)
            mask = (merged["_dt"] > change_dt) & (merged["_dt"] < next_mon)
            n = mask.sum()
            if n == 0:
                continue

            merged.loc[mask, "tax_wedge_benz"] = fwd_row["tax_wedge_benz"]
            merged.loc[mask, "tax_wedge_gas"]  = fwd_row["tax_wedge_gas"]
            log.info(
                "Accise %s → corretto (%s, %s): %d giorni  "
                "tax_wedge_benz=%.4f  tax_wedge_gas=%.4f",
                change_dt.date(), change_dt.date(), next_mon.date(), n,
                fwd_row["tax_wedge_benz"],
                fwd_row["tax_wedge_gas"],
            )

    # ── 3. Calcolo netto ──────────────────────────────────────────────────────
    merged = merged.drop(columns=["_dt"])
    merged["benzina_net"] = merged["benzina_pump"] - merged["tax_wedge_benz"]
    merged["gasolio_net"] = merged["gasolio_pump"] - merged["tax_wedge_gas"]
    return merged[["date", "benzina_pump", "gasolio_pump",
                   "tax_wedge_benz", "tax_wedge_gas",
                   "benzina_net",   "gasolio_net"]]


# ── Inspect modes ─────────────────────────────────────────────────────────────
def inspect_quarter(year: int, quarter: int) -> None:
    """
    Scarica i due tar del trimestre (li mantiene in TAR_CACHE) e
    stampa la struttura interna: nomi file, separatori, colonne, valori unici.
    Utile per diagnosticare errori di parsing senza modificare il codice.
    """
    TAR_CACHE.mkdir(parents=True, exist_ok=True)
    SEP = "=" * 64

    for kind, url_tpl in [("ANAGRAFICA", ANAGRAFICA_URL_TPL), ("PREZZI", PREZZO_URL_TPL)]:
        tar_name = f"{kind.lower()}_{year}_{quarter}_inspect.tar.gz"
        tar_path = TAR_CACHE / tar_name
        url      = url_tpl.format(y=year, q=quarter)

        if not tar_path.exists():
            print(f"\n⬇  Download {kind} {year} Q{quarter} …")
            ok = download_file(url, tar_path, desc=tar_name)
            if not ok:
                print(f"   ⚠  Impossibile scaricare {url}"); continue

        print(f"\n{SEP}\n  {kind}  {year} Q{quarter}  —  {tar_path.name}\n{SEP}")
        with tarfile.open(tar_path, "r:*") as tf:
            members = tf.getmembers()
            files   = [m for m in members if not m.isdir()]
            print(f"File nel tar: {len(files)}")
            for m in files[:15]:
                d = extract_date(m.name)
                print(f"  {m.name:<50}  {m.size:>10,} B  →  data={d}")
            if len(files) > 15:
                print(f"  ... e altri {len(files)-15} file")

            csv_files = [m for m in files if m.name.endswith(".csv")]
            if not csv_files:
                print("  ⚠  Nessun .csv trovato!")
                continue

            # Ispeziona il primo CSV
            sample = csv_files[0]
            print(f"\nIspezione: {sample.name}")
            content = tf.extractfile(sample).read()
            sep     = detect_sep(content[:512])
            print(f"Separatore rilevato: '{sep}'")

            df = pd.read_csv(io.BytesIO(content), sep=sep, dtype=str,
                             on_bad_lines="skip", engine="python")
            print(f"Colonne ({len(df.columns)}): {df.columns.tolist()}")
            print(f"\nPrime 5 righe:")
            print(df.head(5).to_string(index=False))

            # Valori unici delle colonne chiave
            for pat in [r"tipo.?impianto", r"desc.*carb", r"tipo.*carb", r"carb"]:
                col = next((c for c in df.columns if re.search(pat, c, re.I)), None)
                if col:
                    df_full = pd.read_csv(io.BytesIO(content), sep=sep, dtype=str,
                                          on_bad_lines="skip", engine="python")
                    vc = df_full[col].str.strip().value_counts()
                    print(f"\nValori unici '{col}':\n{vc.to_string()}")
                    break

    print(f"\n{SEP}")
    print("Usa queste info per aggiornare extract_date() o i mapping se necessario.")
    print(f"{SEP}\n")


def inspect_sisen() -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    dest = RAW_DIR / "sisen_prezzi_settimanali.csv"
    if not dest.exists():
        download_file(SISEN_URL, dest, desc="sisen_prezzi_settimanali.csv")
    raw  = dest.read_text(encoding="utf-8-sig", errors="replace")
    fl   = raw.split("\n")[0]
    sep  = ";" if fl.count(";") >= fl.count(",") else ","
    df   = pd.read_csv(dest, sep=sep, encoding="utf-8-sig", nrows=10, on_bad_lines="skip")
    print(f"\nSeparatore: '{sep}'")
    print(f"Colonne ({len(df.columns)}): {df.columns.tolist()}")
    print("\nPrime 10 righe:")
    print(df.to_string(index=False))


# ── Salvataggio incrementale ──────────────────────────────────────────────────
def save_incremental(df_new: pd.DataFrame, path: Path) -> int:
    """
    Crea il CSV se non esiste, altrimenti aggiunge solo le righe
    con date non ancora presenti (deduplicazione per colonna 'date').
    Restituisce il numero di righe effettivamente scritte.
    """
    if df_new.empty:
        return 0

    df_new = df_new.copy()
    df_new["date"] = pd.to_datetime(df_new["date"]).dt.date

    if path.exists():
        try:
            df_old = pd.read_csv(path)
            df_old["date"] = pd.to_datetime(df_old["date"]).dt.date
            existing_dates = set(df_old["date"])
        except Exception as e:
            log.warning("Impossibile leggere %s per deduplicazione: %s — si sovrascrive.", path.name, e)
            existing_dates = set()

        df_append = df_new[~df_new["date"].isin(existing_dates)].sort_values("date")
        if df_append.empty:
            log.info("   ✓ Nessuna data nuova da aggiungere a %s", path.name)
            return 0
        df_append.to_csv(path, mode="a", header=False, index=False, float_format="%.5f")
        log.info("   +%d righe nuove → %s  (date: %s … %s)",
                 len(df_append), path.name, df_append["date"].min(), df_append["date"].max())
        return len(df_append)
    else:
        df_new = df_new.sort_values("date")
        df_new.to_csv(path, index=False, float_format="%.5f")
        log.info("   Creato %s  (%d righe, %s … %s)",
                 path.name, len(df_new), df_new["date"].min(), df_new["date"].max())
        return len(df_new)


# ── Pipeline principale ───────────────────────────────────────────────────────
def main(args: argparse.Namespace) -> None:
    PROC_DIR.mkdir(parents=True, exist_ok=True)

    # ── FASE 0: variazioni accise (per correzione SISEN nelle settimane di cambio)
    log.info("═══ FASE 0 — Variazioni accise (%s) ═══", args.accise_json)
    accise_changes = load_accise_changes(Path(args.accise_json))

    # ── FASE 1: tasse settimanali (servono subito per ogni trimestre) ──────────
    log.info("═══ FASE 1 — Tasse settimanali SISEN ═══")
    sisen_df = load_sisen()

    # ── FASE 2: prezzi giornalieri MIMIT, trimestre per trimestre ─────────────
    log.info("═══ FASE 2 — Prezzi giornalieri MIMIT (%s → %s) ═══", START_DATE, END_DATE)
    quarters = ([tuple(args.quarter)] if args.quarter
                else quarters_in_range(START_DATE, END_DATE))

    total_all = total_str = 0
    for year, q in quarters:
        r_all, r_str = process_quarter(year, q, keep_cache=args.keep_cache)

        if r_all:
            df_q = build_output(r_all, sisen_df, accise_change_dates=accise_changes)
            total_all += save_incremental(df_q, OUTPUT_ALL)

        if r_str:
            df_q = build_output(r_str, sisen_df, accise_change_dates=accise_changes)
            total_str += save_incremental(df_q, OUTPUT_STRADALE)

    log.info("═══ Fine pipeline: +%d righe ALL | +%d righe STRADALE ═══",
             total_all, total_str)

    # ── Riepilogo finale ───────────────────────────────────────────────────────
    for path, label in [(OUTPUT_ALL, "ALL"), (OUTPUT_STRADALE, "STRADALE")]:
        if not path.exists():
            continue
        df = pd.read_csv(path)
        if df.empty:
            continue
        print(f"\n{'─'*60}  {label}")
        print(df.tail(5).to_string(index=False))
        cols = [c for c in ("benzina_pump", "gasolio_pump", "benzina_net", "gasolio_net")
                if c in df.columns]
        if cols:
            print(df[cols].describe().round(4).to_string())


# ── CLI ───────────────────────────────────────────────────────────────────────
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Ingestion prezzi carburante IT (MIMIT + SISEN)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""
        Esempi:
          python 01_data_ingestion.py
          python 01_data_ingestion.py --keep-cache
          python 01_data_ingestion.py --quarter 2015 1
          python 01_data_ingestion.py --inspect-quarter 2015 1
          python 01_data_ingestion.py --inspect-sisen
        """),
    )
    p.add_argument("--keep-cache", action="store_true",
                   help="Mantieni i tar in data/raw/tars/")
    p.add_argument("--quarter", nargs=2, type=int, metavar=("ANNO", "Q"))
    p.add_argument("--inspect-quarter", nargs=2, type=int, metavar=("ANNO", "Q"),
                   help="Diagnosi struttura tar per un trimestre")
    p.add_argument("--inspect-sisen", action="store_true")
    p.add_argument(
        "--accise-json",
        default=str(ACCISE_JSON),
        help=f"Path al JSON delle variazioni di accisa (default: {ACCISE_JSON})",
    )
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.inspect_quarter:
        inspect_quarter(*args.inspect_quarter)
    elif args.inspect_sisen:
        inspect_sisen()
    else:
        main(args)