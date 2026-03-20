"""
TradingView Selenium Scraper - B7H1!
Virtual scrolling: raccoglie le righe visibili ad ogni step di scroll.

Dipendenze:
    pip install selenium webdriver-manager beautifulsoup4 lxml
"""

import csv
import datetime
import re
import time
import sys
from bs4 import BeautifulSoup

try:
    from selenium import webdriver
    from selenium.webdriver.chrome.options import Options
    from selenium.webdriver.chrome.service import Service
    from selenium.webdriver.common.by import By
    from selenium.webdriver.support.ui import WebDriverWait
    from selenium.webdriver.support import expected_conditions as EC
    from webdriver_manager.chrome import ChromeDriverManager
except ImportError:
    sys.exit("Installa: pip install selenium webdriver-manager beautifulsoup4 lxml")


URL        = "https://it.tradingview.com/chart/?symbol=NYMEX%3AB7H1%21"
OUTPUT_CSV = "b7h1_data.csv"

# Timestamp Unix della data limite (inclusa): 2015-01-01 00:00:00 UTC
STOP_DATE     = datetime.datetime(2015, 1, 1, tzinfo=datetime.timezone.utc)
STOP_TS       = int(STOP_DATE.timestamp())   # 1420070400


# ── Chrome ────────────────────────────────────────────────────────────────────

def build_driver():
    opts = Options()
    opts.add_argument("--window-size=1920,1080")
    opts.add_argument("--disable-blink-features=AutomationControlled")
    opts.add_experimental_option("excludeSwitches", ["enable-automation"])
    opts.add_experimental_option("useAutomationExtension", False)
    opts.add_argument(
        "user-agent=Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
    )
    opts.add_argument("--no-sandbox")
    opts.add_argument("--disable-dev-shm-usage")
    opts.add_argument("--disable-gpu")
    driver = webdriver.Chrome(
        service=Service(ChromeDriverManager().install()), options=opts
    )
    driver.execute_cdp_cmd(
        "Page.addScriptToEvaluateOnNewDocument",
        {"source": "Object.defineProperty(navigator, 'webdriver', {get: () => undefined})"},
    )
    return driver


# ── Parsing di un singolo <tr> ────────────────────────────────────────────────

def parse_value(raw):
    if not raw:
        return ""
    cleaned = raw.replace(".", "").replace(",", ".")
    if re.search(r"[a-zA-Z%+\-\u2212]", cleaned):
        return raw
    return cleaned

def parse_row(tr):
    cells = tr.find_all(["td", "th"])
    if len(cells) < 7:
        return None
    def cv(i):
        return cells[i].get("data-copy-value") or cells[i].get_text(strip=True)
    return {
        "data":       cells[0].get_text(strip=True),
        "timestamp":  tr.get("data-row-time", ""),
        "apertura":   parse_value(cv(1)),
        "massimo":    parse_value(cv(2)),
        "minimo":     parse_value(cv(3)),
        "chiusura":   parse_value(cv(4)),
        "variazione": cv(5),
        "volume":     cv(6),
    }


# ── Raccolta con virtual scroll ───────────────────────────────────────────────

def collect_all_rows(driver, stop_ts=STOP_TS, pause=1.2, max_scrolls=2000):
    """
    Ad ogni step di scroll legge le righe visibili e le accumula
    in un dizionario {timestamp: row}. Continua finché il timestamp
    della riga più vecchia visibile è > stop_ts (2015-01-01).
    Si ferma solo quando ha effettivamente superato quella soglia
    oppure quando non arrivano più dati nuovi.
    """
    stop_date_str = datetime.datetime.utcfromtimestamp(stop_ts).strftime("%Y-%m-%d")
    print(f"   Data limite: {stop_date_str} (ts={stop_ts})")

    # Trova il contenitore scrollabile
    container = None
    for sel in [".wrapper-Tv7LSjUz", ".overlayScrollWrap-Tv7LSjUz", "tbody"]:
        try:
            el  = driver.find_element(By.CSS_SELECTOR, sel)
            sh  = driver.execute_script("return arguments[0].scrollHeight", el)
            ch  = driver.execute_script("return arguments[0].clientHeight", el)
            if sh > ch:
                container = el
                print(f"   Contenitore scrollabile: {sel}")
                break
        except Exception:
            continue

    collected   = {}   # timestamp → row  (evita duplicati)
    stable      = 0
    prev_oldest = None

    for i in range(max_scrolls):
        # Leggi le righe attualmente visibili nel DOM
        soup    = BeautifulSoup(driver.page_source, "lxml")
        visible = soup.find_all("tr", attrs={"data-row-time": True})

        new        = 0
        oldest_ts  = None
        oldest_lbl = "?"

        for tr in visible:
            ts = tr.get("data-row-time", "")
            if ts and ts not in collected:
                row = parse_row(tr)
                if row:
                    collected[ts] = row
                    new += 1
            if ts:
                oldest_ts  = ts
                span = tr.find("span")
                oldest_lbl = span.get_text(strip=True) if span else ts

        total = len(collected)
        print(
            f"   Scroll {i+1:>4} | visibili: {len(visible):>3} | "
            f"nuove: {new:>3} | totale: {total:>5} | "
            f"fino a: {oldest_lbl}   ",
            end="\r"
        )

        # ── Controllo data di stop ──────────────────────────────────────────
        # Ci fermiamo SOLO quando il timestamp più vecchio visibile
        # è <= stop_ts, cioè abbiamo già caricato dati precedenti a 2015-01-01.
        if oldest_ts:
            try:
                ots = int(oldest_ts)
                if ots > 0 and ots <= stop_ts:
                    oldest_date = datetime.datetime.utcfromtimestamp(ots).strftime("%Y-%m-%d")
                    print(f"\n   ✅ Raggiunto {oldest_date} (≤ {stop_date_str}), stop.")
                    break
            except ValueError:
                pass

        # ── Stop se non arrivano più righe nuove ───────────────────────────
        if oldest_ts == prev_oldest:
            stable += 1
            if stable >= 10:          # 10 cicli senza novità → fine dati
                print(f"\n   ⚠️  Nessun nuovo dato da {stable} cicli, stop.")
                break
        else:
            stable = 0
        prev_oldest = oldest_ts

        # Scrolla verso il basso
        if container:
            driver.execute_script("arguments[0].scrollTop += 2000", container)
        else:
            driver.execute_script("window.scrollBy(0, 2000)")
        time.sleep(pause)

    return collected


# ── Main ──────────────────────────────────────────────────────────────────────

def scrape_live(url=URL, output_csv=OUTPUT_CSV):
    driver = build_driver()
    wait   = WebDriverWait(driver, 40)

    try:
        print(f"Apro {url} ...")
        driver.get(url)
        wait.until(EC.presence_of_element_located((By.CSS_SELECTOR, ".chart-markup-table")))
        time.sleep(2)

        print("\n" + "="*55)
        print("➡️   APRI LA VISTA TABELLA NEL BROWSER")
        print("    right-click sulla legenda → Vista tabella")
        print("    Quando la tabella è visibile, torna qui.")
        print("="*55)
        input("\nPremi INVIO per iniziare lo scraping: ")

        wait.until(EC.presence_of_element_located((By.CSS_SELECTOR, "tr[data-row-time]")))
        time.sleep(2)

        stop_date_str = STOP_DATE.strftime("%d/%m/%Y")
        print(f"\nRaccolta dati fino al {stop_date_str} ...")
        collected = collect_all_rows(driver, stop_ts=STOP_TS)

        if not collected:
            driver.save_screenshot("debug_screenshot.png")
            print("Nessuna riga trovata. Controlla debug_screenshot.png")
            return []

        # Ordina per timestamp decrescente (più recente prima)
        rows = sorted(collected.values(), key=lambda r: int(r["timestamp"]), reverse=True)

        # Filtra eventuali righe antecedenti alla data limite
        # (teniamo tutto perché potremmo voler vedere anche cosa c'è prima)
        rows_in_range = [r for r in rows if int(r["timestamp"]) >= STOP_TS]
        discarded     = len(rows) - len(rows_in_range)
        if discarded:
            print(f"   (scartate {discarded} righe precedenti al {stop_date_str})")

        output_rows = rows_in_range if rows_in_range else rows

        fieldnames = ["data", "timestamp", "apertura", "massimo",
                      "minimo", "chiusura", "variazione", "volume"]
        with open(output_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(output_rows)

        print(f"\n✅  CSV salvato: {output_csv}  ({len(output_rows)} righe)")
        print(f"   Prima riga : {output_rows[0]['data']}")
        print(f"   Ultima riga: {output_rows[-1]['data']}")
        return output_rows

    finally:
        driver.quit()


if __name__ == "__main__":
    scrape_live()