"""
utils/conversions.py
====================
Conversioni fisico-chimiche e valutarie per prezzi dei carburanti.

TON → LITRI  (approccio molare)
────────────────────────────────
I futures ICE sono quotati in USD/tonnellata metrica (USD/ton).
Per convertire in USD/litro usiamo la densità liquida del prodotto,
derivata dalla composizione molecolare media (approccio molare).

  Densità liquida  ρ = MW / Vm_liq
  dove Vm_liq = volume molare liquido (L/mol), ricavato dalla
  correlazione di Rackett per idrocarburi puri o dalla specifica
  contrattuale ICE per le miscele.

  1 ton → L  =  1000 kg / ρ [kg/L]

Prodotti:
  Gas Oil (ICE Low Sulphur Gasoil)
    Specifica ICE: densità 0.820–0.845 kg/L a 15 °C
    Molecola rappresentativa: miscela C10–C22, media ≈ C13H28
      MW  = 13×12.011 + 28×1.008  = 156.14 + 28.224 = 184.36 g/mol
      ρ_pura (Rackett C13 n-tridecano, 15°C) ≈ 0.756 kg/L
      ρ_effettiva ICE ≈ 0.840 kg/L (aromati + cicloalcani aumentano ρ)
    → si usa ρ_ICE = 0.840 kg/L  ⟹  1 ton = 1000/0.840 = 1190.5 L

  Eurobob (unleaded gasoline)
    Specifica ICE: densità 0.720–0.775 kg/L a 15 °C
    Molecola rappresentativa: miscela C5–C12, media ≈ C8H18 (isoottano)
      MW  = 8×12.011 + 18×1.008  = 96.088 + 18.144 = 114.23 g/mol
      ρ_pura isoottano (15°C) ≈ 0.695 kg/L
      ρ_effettiva ICE ≈ 0.745 kg/L (blend commerciale con aromati)
    → si usa ρ_ICE = 0.745 kg/L  ⟹  1 ton = 1000/0.745 = 1342.3 L

EUR/USD
────────
Il modulo usa tre sorgenti in ordine di priorità:

  1. CSV locale (investing.com o simili) — se csv_path fornito
       Formato: "Date","Price","Open","High","Low","Change %"
       "Apr 28, 2026","1.1346","1.1358","1.1398","1.1265","-0.11%"
       Scaricabile da: https://www.investing.com/currencies/eur-usd-historical-data
       → "Download Data" → salva come  data/raw/eurusd.csv

  2. yfinance (EURUSD=X) — dati daily dal 2003 ad oggi
       pip install yfinance
       Sorgente consigliata: precisa, aggiornata automaticamente.
       Attenzione: richiede connessione internet al momento dell'esecuzione.

  3. Medie annuali BCE (fallback hard-coded)
       Precisione: ±5–8% in anni con forte volatilità EUR/USD (es. 2022).
       Sufficiente per grafici di trend a lungo periodo, ma non per
       calcolare crack spread precisi intorno agli shock geopolitici.

NOTA: per analisi su dati 2022 (Ucraina), EUR/USD è passato da 1.13 a 0.96
(-15% in 9 mesi). Usare la media annuale 1.053 introduce errori sistematici
sui prezzi all'ingrosso che distorcono il crack spread stimato.
"""

from __future__ import annotations

from pathlib import Path
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

# ── Costanti molecolari ────────────────────────────────────────────────────────

@dataclass(frozen=True)
class Hydrocarbon:
    """Rappresenta un idrocarburo o una miscela con formula CnHm."""
    name:        str
    formula:     str
    n_carbon:    float   # numero atomi C (può essere frazionario per miscele)
    n_hydrogen:  float   # numero atomi H
    rho_pure:    float   # densità idrocarburo puro a 15°C [kg/L]
    rho_eff:     float   # densità effettiva prodotto commerciale ICE [kg/L]

    # Pesi atomici IUPAC 2021
    _MW_C: float = 12.011
    _MW_H: float = 1.008

    @property
    def mw(self) -> float:
        """Peso molecolare [g/mol]."""
        return self.n_carbon * self._MW_C + self.n_hydrogen * self._MW_H

    @property
    def vm_pure(self) -> float:
        """Volume molare del liquido puro [L/mol] = MW / (ρ_pure * 1000)."""
        return self.mw / (self.rho_pure * 1000)

    @property
    def vm_eff(self) -> float:
        """Volume molare effettivo della miscela ICE [L/mol]."""
        return self.mw / (self.rho_eff * 1000)

    @property
    def l_per_ton_pure(self) -> float:
        """L/ton usando densità pura: 1 ton = 1_000_000 g / MW mol → × Vm."""
        moles_per_ton = 1_000_000 / self.mw   # mol/ton
        return moles_per_ton * self.vm_pure     # L/ton

    @property
    def l_per_ton_eff(self) -> float:
        """L/ton usando densità effettiva ICE (valore da usare in pratica)."""
        return 1000 / self.rho_eff              # = 1_000_000 g/ton / (ρ_eff * 1000 g/L)


# Prodotti ICE
GAS_OIL = Hydrocarbon(
    name       = "ICE Low Sulphur Gasoil (diesel)",
    formula    = "C13H28",          # n-tridecano come proxy della miscela C10–C22
    n_carbon   = 13.0,
    n_hydrogen = 28.0,
    rho_pure   = 0.756,             # n-tridecano puro a 15°C [kg/L]
    rho_eff    = 0.840,             # specifica contrattuale ICE [kg/L]
)

EUROBOB = Hydrocarbon(
    name       = "Eurobob Oxy Gasoline (benzina)",
    formula    = "C8H18",           # isoottano come proxy del blend C5–C12
    n_carbon   = 8.0,
    n_hydrogen = 18.0,
    rho_pure   = 0.695,             # isoottano puro a 15°C [kg/L]
    rho_eff    = 0.745,             # specifica ICE per benzina commerciale [kg/L]
)


def print_conversion_summary() -> None:
    """Stampa tabella riassuntiva delle conversioni molare/ICE."""
    print("╔══════════════════════════════════════════════════════════════════╗")
    print("║  CONVERSIONI FISICO-CHIMICHE  USD/ton → USD/L                  ║")
    print("╠══════════════════════════════════════════════════════════════════╣")
    for hc in (GAS_OIL, EUROBOB):
        print(f"║  {hc.name}")
        print(f"║    Formula proxy   : {hc.formula}")
        print(f"║    MW              : {hc.mw:.2f} g/mol")
        print(f"║    Vm puro (15°C)  : {hc.vm_pure*1000:.2f} mL/mol  "
              f"→ {hc.l_per_ton_pure:.1f} L/ton  (ρ_pura={hc.rho_pure} kg/L)")
        print(f"║    Vm ICE (15°C)   : {hc.vm_eff*1000:.2f} mL/mol  "
              f"→ {hc.l_per_ton_eff:.1f} L/ton  (ρ_ICE={hc.rho_eff} kg/L)  ← usato")
        print(f"║    Delta ρ         : +{(hc.rho_eff-hc.rho_pure)/hc.rho_pure*100:.1f}%"
              f"  (aromati/cicloalcani nella miscela reale)")
        print("║")
    print("╚══════════════════════════════════════════════════════════════════╝")


# ── EUR/USD ────────────────────────────────────────────────────────────────────

# Fallback: tassi annui medi storici EUR/USD (fonte: BCE, medie annuali)
_EURUSD_ANNUAL_FALLBACK: dict[int, float] = {
    2005: 1.2441, 2006: 1.2556, 2007: 1.3705, 2008: 1.4726, 2009: 1.3948,
    2010: 1.3257, 2011: 1.3920, 2012: 1.2848, 2013: 1.3281, 2014: 1.3285,
    2015: 1.0859, 2016: 1.1069, 2017: 1.1297, 2018: 1.1810, 2019: 1.1195,
    2020: 1.1422, 2021: 1.1827, 2022: 1.0530, 2023: 1.0813, 2024: 1.0820,
    2025: 1.0750, 2026: 1.0900,   # stima
}


def _build_fallback_series(start: str, end: str) -> pd.Series:
    """Costruisce una serie giornaliera EUR/USD dai valori annui medi (fallback)."""
    idx = pd.date_range(start, end, freq="D")
    vals = [_EURUSD_ANNUAL_FALLBACK.get(d.year, 1.08) for d in idx]
    s = pd.Series(vals, index=idx, name="eurusd")
    return s


def _load_eurusd_from_csv(csv_path: Path, start: str, end: str) -> Optional[pd.Series]:
    """
    Carica EUR/USD da file CSV (formato investing.com o simili).
    Restituisce None se il file non esiste o è illeggibile.
    """
    if not Path(csv_path).exists():
        return None
    try:
        df = pd.read_csv(csv_path, encoding="utf-8-sig", dtype=str)
        df.columns = [c.strip().strip('"') for c in df.columns]

        # Prova formato investing.com: "Apr 28, 2026"
        date_col  = next((c for c in df.columns if "date" in c.lower()), df.columns[0])
        price_col = next((c for c in df.columns if "price" in c.lower()), df.columns[1])

        df[date_col]  = df[date_col].str.strip().str.strip('"')
        df[price_col] = (df[price_col].str.strip().str.strip('"')
                         .str.replace(",", "", regex=False)
                         .pipe(pd.to_numeric, errors="coerce"))

        # Prova più formati di data
        for fmt in ("%b %d, %Y", "%m/%d/%Y", "%Y-%m-%d", "%d/%m/%Y"):
            try:
                df["date"] = pd.to_datetime(df[date_col], format=fmt)
                break
            except Exception:
                continue
        else:
            df["date"] = pd.to_datetime(df[date_col], infer_datetime_format=True,
                                        errors="coerce")

        df = df.dropna(subset=["date", price_col]).sort_values("date")
        s  = df.set_index("date")[price_col].rename("eurusd")

        # Interpola weekends/festivi e clip range ragionevole
        s = (s.reindex(pd.date_range(s.index.min(), s.index.max(), freq="D"))
              .interpolate("time")
              .clip(0.80, 1.60))

        print(f"  EUR/USD caricato da file: {Path(csv_path).name}  "
              f"({len(s)} osservazioni, "
              f"{s.index.min().date()} → {s.index.max().date()})")
        return s

    except Exception as e:
        print(f"  ⚠ Errore lettura EUR/USD da file ({e})")
        return None


def _load_eurusd_from_yfinance(start: str, end: str) -> Optional[pd.Series]:
    """
    Scarica EUR/USD da yfinance (ticker EURUSD=X).
    Restituisce None se yfinance non è installato o la rete non è disponibile.
    """
    try:
        import yfinance as yf  # type: ignore
        raw = yf.download("EURUSD=X", start=start, end=end,
                          auto_adjust=True, progress=False)
        if raw.empty:
            return None
        # yfinance restituisce MultiIndex o singolo livello a seconda della versione
        if isinstance(raw.columns, pd.MultiIndex):
            s = raw["Close"]["EURUSD=X"].rename("eurusd")
        else:
            s = raw["Close"].rename("eurusd")
        s = s.dropna().astype(float)
        # Interpola fine settimana/festivi
        s = (s.reindex(pd.date_range(s.index.min(), s.index.max(), freq="D"))
              .interpolate("time")
              .clip(0.80, 1.60))
        print(f"  EUR/USD scaricato da yfinance (EURUSD=X): "
              f"{s.index.min().date()} → {s.index.max().date()}")
        return s
    except Exception as e:
        print(f"  ⚠ yfinance non disponibile o errore di rete ({e})")
        return None


def load_eurusd(
    csv_path: Path | None = None,
    start: str = "2004-01-01",
    end:   str = "2026-12-31",
    use_yfinance: bool = True,
) -> pd.Series:
    """
    Carica serie storica giornaliera EUR/USD.

    Priorità:
      1. csv_path (investing.com o ECB) se esiste e leggibile
      2. yfinance (EURUSD=X) — dati daily precisi — se use_yfinance=True
      3. Fallback su medie annuali BCE hard-coded

    Il CSV di investing.com ha formato:
      "Date","Price","Open","High","Low","Change %"
      "Apr 28, 2026","1.1346",...

    Restituisce pd.Series con DatetimeIndex e valori EUR/USD
    (quanti USD vale 1 EUR; per convertire USD→EUR: dividi per EUR/USD).

    Note
    ----
    Usare yfinance per analisi precise su periodi con forte volatilità
    EUR/USD (es. 2022: range 0.96–1.13). Le medie annuali introducono
    errori del ±5–8% sui prezzi all'ingrosso in tali periodi.
    """
    # Costruiamo il fallback annuale come base su cui sovrascrivere
    fb  = _build_fallback_series(start, end)
    out = fb.copy()

    # Priorità 1: CSV locale
    if csv_path is not None:
        s = _load_eurusd_from_csv(csv_path, start, end)
        if s is not None:
            out.update(s)
            return out.loc[start:end]

    # Priorità 2: yfinance (daily, preciso)
    if use_yfinance:
        s = _load_eurusd_from_yfinance(start, end)
        if s is not None:
            out.update(s)
            return out.loc[start:end]

    # Priorità 3: fallback medie annuali BCE
    print("  EUR/USD: uso fallback medie annuali BCE (precisione ±5-8% in anni volatili).")
    print("    Per maggiore precisione:")
    print("    • installa yfinance:  pip install yfinance")
    print("    • oppure scarica CSV: https://www.investing.com/currencies/eur-usd-historical-data")
    print("      → 'Download Data' → salva in  data/raw/eurusd.csv")
    return out.loc[start:end]


def usd_ton_to_eur_liter(
    prices_usd_ton: pd.Series,
    eurusd:         pd.Series,
    hc:             Hydrocarbon,
) -> pd.Series:
    """
    Converte prezzi futures [USD/ton] in [€/L].

    Passaggi:
      1. USD/ton  →  USD/L   :  dividi per l_per_ton_eff  (molare + ρ_ICE)
      2. USD/L    →  EUR/L   :  dividi per EUR/USD (quanti USD per 1 EUR)

    Parameters
    ----------
    prices_usd_ton : Series con DatetimeIndex, valori in USD/ton
    eurusd         : Series con DatetimeIndex, tasso EUR/USD giornaliero
    hc             : Hydrocarbon (GAS_OIL o EUROBOB)

    Returns
    -------
    Series EUR/L allineata all'indice di prices_usd_ton
    """
    # Allinea EUR/USD all'indice dei prezzi (forward-fill weekends)
    rate = eurusd.reindex(prices_usd_ton.index, method="ffill")

    usd_per_liter = prices_usd_ton / hc.l_per_ton_eff   # USD/L
    eur_per_liter = usd_per_liter  / rate                # EUR/L

    return eur_per_liter.rename(f"{hc.formula}_eur_l")
