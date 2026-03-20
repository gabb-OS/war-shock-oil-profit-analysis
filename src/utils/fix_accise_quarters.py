#!/usr/bin/env python3
"""
utils/fix_accise_quarters.py
============================
Ricalcola solo i trimestri del CSV che contengono eventi di cambio
accisa con sisen_corretta nota, garantendo la correzione forward-match.

Problema che risolve:
  save_incremental non sovrascrive righe già presenti → se il CSV è stato
  generato prima della correzione SISEN, i giorni in [data_effettiva, sisen_corretta)
  hanno ancora il tax_wedge contaminato → netto artificialmente basso.

Soluzione:
  1. Legge accise_variazioni.json → trova i (year, quarter) degli eventi
  2. Cancella quelle righe dal CSV esistente
  3. Riesegue process_quarter solo per quei trimestri
  4. Salva con build_output corretto (forward-match SISEN)

Uso:
  python3 utils/fix_accise_quarters.py
  python3 utils/fix_accise_quarters.py --keep-cache
  python3 utils/fix_accise_quarters.py --dry-run   # mostra i trimestri senza modificare
"""

import argparse
import importlib.util
import json
import logging
from pathlib import Path

import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

SRC_DIR     = Path(__file__).parent.parent
ACCISE_JSON = SRC_DIR / "data" / "accise_variazioni.json"

# ── Import dinamico di 01_data_ingestion (nome inizia con cifra) ──────────────
_spec = importlib.util.spec_from_file_location(
    "data_ingestion", SRC_DIR / "01_data_ingestion.py"
)
ing = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ing)


# ── Helpers ───────────────────────────────────────────────────────────────────
def _next_monday(dt: pd.Timestamp) -> pd.Timestamp:
    """Primo lunedì strettamente successivo a dt."""
    days = (7 - dt.weekday()) % 7 or 7
    return dt + pd.Timedelta(days=days)


def _date_to_quarter(d: pd.Timestamp) -> tuple[int, int]:
    return d.year, (d.month - 1) // 3 + 1


def _quarter_bounds(year: int, q: int) -> tuple[pd.Timestamp, pd.Timestamp]:
    """Prima e ultima data del trimestre (incluse)."""
    first_month = (q - 1) * 3 + 1
    last_month  = first_month + 2
    start = pd.Timestamp(year, first_month, 1)
    # fine trimestre: primo giorno del mese successivo meno 1
    if last_month == 12:
        end = pd.Timestamp(year, 12, 31)
    else:
        end = pd.Timestamp(year, last_month + 1, 1) - pd.Timedelta(days=1)
    return start, end


def affected_quarters(json_path: Path) -> list[tuple[int, int]]:
    """
    Legge il JSON e restituisce i (year, quarter) da ricalcolare.
    Per ogni evento con sisen_corretta non-null include sia il trimestre
    di data_effettiva che quello di sisen_corretta (possono differire
    per eventi a cavallo di fine anno o fine trimestre).
    """
    with open(json_path, encoding="utf-8") as fh:
        data = json.load(fh)

    quarters: set[tuple[int, int]] = set()
    for v in data.get("variazioni", []):
        if v.get("sisen_corretta") is None:
            continue
        eff      = pd.Timestamp(v["data_effettiva"])
        eff_next = eff + pd.Timedelta(days=1)          # correzione parte dal giorno dopo
        next_mon = _next_monday(eff)                   # fine finestra (esclusa)
        quarters.add(_date_to_quarter(eff_next))
        quarters.add(_date_to_quarter(next_mon))       # include trimestre del lunedì se cambia
        log.info(
            "  Evento '%s': decreto %s → finestra (%s, %s)  (Q%s … Q%s)",
            v["id"], eff.date(), eff_next.date(), next_mon.date(),
            "%dQ%d" % _date_to_quarter(eff_next),
            "%dQ%d" % _date_to_quarter(next_mon),
        )

    return sorted(quarters)


def drop_quarter_rows(csv_path: Path, year: int, q: int) -> int:
    """
    Rimuove dal CSV le righe del trimestre (year, q).
    Sovrascrive il file sul posto. Ritorna il numero di righe rimosse.
    """
    if not csv_path.exists():
        return 0

    df = pd.read_csv(csv_path, parse_dates=["date"])
    start, end = _quarter_bounds(year, q)
    mask = (df["date"] >= start) & (df["date"] <= end)
    n = int(mask.sum())

    if n > 0:
        df[~mask].to_csv(csv_path, index=False, float_format="%.5f")
        log.info(
            "  Rimosso %d righe (%d Q%d: %s … %s) da %s",
            n, year, q, start.date(), end.date(), csv_path.name,
        )
    else:
        log.info("  Nessuna riga per %d Q%d in %s", year, q, csv_path.name)

    return n


# ── Main ──────────────────────────────────────────────────────────────────────
def main() -> None:
    p = argparse.ArgumentParser(
        description="Ricalcola i trimestri con cambio accisa nel CSV prezzi."
    )
    p.add_argument("--keep-cache", action="store_true",
                   help="Mantieni i tar scaricati in data/raw/tars/")
    p.add_argument("--dry-run", action="store_true",
                   help="Mostra i trimestri interessati senza modificare nulla")
    args = p.parse_args()

    if not ACCISE_JSON.exists():
        log.error("File non trovato: %s", ACCISE_JSON)
        return

    quarters = affected_quarters(ACCISE_JSON)
    if not quarters:
        log.info("Nessun evento con sisen_corretta trovato — niente da fare.")
        return

    log.info("Trimestri da ricalcolare (%d): %s", len(quarters),
             ", ".join(f"{y}Q{q}" for y, q in quarters))

    if args.dry_run:
        print("\nDry-run — nessuna modifica al CSV.")
        return

    # Carica SISEN e variazioni accise una volta sola
    log.info("═══ Caricamento SISEN ═══")
    sisen_df      = ing.load_sisen()
    accise_changes = ing.load_accise_changes(ACCISE_JSON)

    updated = 0
    for year, q in quarters:
        log.info("═══ %d Q%d ═══", year, q)

        # Rimuovi le righe già presenti per questo trimestre
        drop_quarter_rows(ing.OUTPUT_ALL,      year, q)
        drop_quarter_rows(ing.OUTPUT_STRADALE, year, q)

        # Ricalcola i prezzi per il trimestre
        r_all, r_str = ing.process_quarter(year, q, keep_cache=args.keep_cache)

        if r_all:
            df_q = ing.build_output(r_all, sisen_df, accise_change_dates=accise_changes)
            ing.save_incremental(df_q, ing.OUTPUT_ALL)

        if r_str:
            df_q = ing.build_output(r_str, sisen_df, accise_change_dates=accise_changes)
            ing.save_incremental(df_q, ing.OUTPUT_STRADALE)

        updated += 1

    # ── Riordina entrambi i CSV per data (save_incremental appende in coda) ────
    for csv_path in (ing.OUTPUT_ALL, ing.OUTPUT_STRADALE):
        if not csv_path.exists():
            continue
        df = pd.read_csv(csv_path, parse_dates=["date"])
        df = df.sort_values("date").drop_duplicates("date").reset_index(drop=True)
        df.to_csv(csv_path, index=False, float_format="%.5f")
        log.info("Riordinato %s (%d righe)", csv_path.name, len(df))

    log.info("═══ Fine: %d trimestri aggiornati ═══", updated)


if __name__ == "__main__":
    main()
