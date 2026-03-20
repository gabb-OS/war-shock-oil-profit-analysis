"""
Converte la colonna 'data' del CSV da formato italiano TradingView
  "mer 29 Apr '26"  →  "04/29/2026"

Uso:
    python3 convert_dates.py                        # legge b7h1_data.csv, scrive b7h1_data_us.csv
    python3 convert_dates.py input.csv output.csv   # file personalizzati
"""

import csv
import sys
import re
from pathlib import Path

# ── Mappa mesi italiani ───────────────────────────────────────────────────────

MESI = {
    "Gen": "01", "Feb": "02", "Mar": "03", "Apr": "04",
    "Mag": "05", "Giu": "06", "Lug": "07", "Ago": "08",
    "Set": "09", "Ott": "10", "Nov": "11", "Dic": "12",
}

# ── Conversione singola data ──────────────────────────────────────────────────

def converti(data_it: str) -> str:
    """
    Input:  "mer 29 Apr '26"
    Output: "04/29/2026"
    """
    # estrai giorno, mese abbreviato, anno abbreviato
    m = re.search(r"(\d{1,2})\s+([A-Za-z]+)\s+'(\d{2})", data_it)
    if not m:
        return data_it  # lascia invariato se non riconosce il formato

    giorno = m.group(1).zfill(2)
    mese   = MESI.get(m.group(2), "??")
    anno   = "20" + m.group(3)

    return f"{mese}/{giorno}/{anno}"

# ── Lettura / scrittura CSV ───────────────────────────────────────────────────

def converti_csv(input_csv: str, output_csv: str):
    src = Path(input_csv)
    if not src.exists():
        sys.exit(f"File non trovato: {input_csv}")

    with open(src, newline="", encoding="utf-8") as fin:
        reader = csv.DictReader(fin)
        if "data" not in (reader.fieldnames or []):
            sys.exit("Colonna 'data' non trovata nel CSV.")
        fieldnames = reader.fieldnames
        rows = list(reader)

    for row in rows:
        row["data"] = converti(row["data"])

    with open(output_csv, "w", newline="", encoding="utf-8") as fout:
        writer = csv.DictWriter(fout, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"✅  Convertite {len(rows)} righe → {output_csv}")
    print(f"   Esempio: {rows[0]['data']}")

# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    inp = sys.argv[1] if len(sys.argv) > 1 else "b7h1_data.csv"
    out = sys.argv[2] if len(sys.argv) > 2 else inp.replace(".csv", "_us.csv")
    converti_csv(inp, out)