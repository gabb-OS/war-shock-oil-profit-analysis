# Report Metodologico: Catena Decisionale dei Test Statistici
## Analisi dei Margini sui Carburanti Italiani — Tre Crisi Energetiche
### Versione aggiornata post-run 27 aprile 2026

---

## 1. Obiettivo e Ipotesi Nulla

L'analisi testa se il margine lordo dei distributori italiani — calcolato come differenza
fra prezzo alla pompa al netto delle tasse e costo wholesale europeo (Eurobob per la benzina,
Gas Oil ICE per il diesel, entrambi convertiti in EUR/litro) — aumenti in modo
statisticamente anomalo rispetto al baseline pre-crisi (2019) durante tre eventi energetici:
invasione dell'Ucraina (feb. 2022), guerra Iran-Israele (giu. 2025), chiusura dello
Stretto di Hormuz (feb. 2026).

**H₀:** μ_post = μ_2019 (il margine lordo medio nel periodo post-shock è uguale alla
media del baseline 2019). Il test è **unilaterale superiore** (one-sided upper).

**Baseline:** 2019 full year (52 settimane). Dataset: 381 settimane, 2019-01-07 – 2026-04-20.

| Carburante | μ₂₀₁₉ (EUR/L) | σ₂₀₁₉ | Soglia 2σ |
|---|---|---|---|
| Benzina | 0.168 | 0.019 | 0.038 |
| Diesel | 0.149 | 0.018 | 0.037 |

---

## 2. Problemi Metodologici — Sezione Critica (aggiornata v2)

Questa sezione documenta le limitazioni della pipeline nell'ordine di gravità.
I problemi risolti in v2 sono marcati **[RISOLTO]**; quelli nuovi sono marcati **[NUOVO]**.

### 2.1 ~~CIRCOLARITÀ: τ_margin nella BH confirmatory~~ [RISOLTO in v2]

**Problema originale:** τ_margin era incluso nella famiglia BH confirmatory come split,
ma è stimato come argmax del segnale sulla serie del margine (data-snooping). P-value
non validi sotto H₀.

**Fix v2:** τ_margin rimosso completamente dalla BH. Viene ora riportato solo come
timing descrittivo (nessun p-value), coerentemente con la sua natura endogena.
**La famiglia BH-A è ora pulita: 16 test, solo split esogeni (shock_hard + τ_price).**

### 2.2 ~~H₀ Miste nella Stessa Famiglia BH~~ [RISOLTO in v2]

**Problema originale:** Welch/MW (H₀_A: livello vs 2019) e block perm/HAC (H₀_B:
salto locale pre→post) erano nella stessa famiglia BH.

**Fix v2:** block perm e HAC sono ora **esplorativi [no BH]**. BH-A contiene solo
Welch t + Mann-Whitney (H₀_A). BH-B contiene solo DiD (domanda diversa, famiglia separata).

### 2.3 ~~Bug table2_margin_anomaly aggiornamento BH~~ [RISOLTO in v2]

**Fix v2:** script 05 ora usa join su chiave composita (evento, carburante, split_type)
invece di posizione. Le colonne `BH_global_reject` e `t_p_BH_global_adjusted` in
`table2_margin_anomaly_v2.csv` sono ora popolate correttamente.

### 2.4 HAC maxlags → Andrews BW [PARZIALMENTE RISOLTO in v2]

**Fix v2:** script 03 ora usa il bandwidth ottimale Andrews (1991) invece di maxlags=4 fisso.
Script 06 riporta Andrews BW per ogni serie.

**Problema residuo:** per Ucraina benzina (BW=22 ≥ n/2=22) e Iran-Is. benzina (BW=14 ≥ n/2=14)
il test HAC è quasi non informativo — la finestra ottimale è pari a metà della serie.
Questo è segnalato con flag nei report ma i p-value HAC vanno trattati con cautela estrema.

### 2.5 ~~n_eff Non Riportato~~ [RISOLTO in v2]

**Fix v2:** n_eff calcolato e riportato per ogni combinazione evento×carburante×split.
Flag automatico "CAUTELA" se n_eff < 5 (test non informativo). File: `neff_report_v2.csv`.

### 2.6 BUG CRITICO: Script 05 carica DiD v1 invece di v2 [NUOVO — CRITICO]

**Problema:** script 05 legge `data/auxiliary_pvalues.csv` (file v1, p-value 0.2–0.8,
tutti n.s.) invece di `data/did_results_v2.csv` (v2, 7/8 con p≈0). La conclusione
"BH-DiD: 0/8 rigettati" nel sommario del run è **basata su dati obsoleti**.

**Conseguenza:** il `global_bh_corrections_v2.csv` contiene p-value DiD sbagliati
nelle righe ausiliaria_DiD. I rigetti BH-A (16/16) non sono impattati.

**Fix necessario:** in script 05, riga di caricamento DiD deve puntare a
`did_results_v2.csv` usando la colonna `p_value` di quel file.

### 2.7 DiD v2: PTA violata in 5/8 casi [NUOVO — METODOLOGICO]

**Problema:** il test di parallel trends assumption (PTA) è rigettato in 5/8 casi v2:
- Ucraina DE benzina: PTA_p=0.000 → **VIOLATA**
- Iran-Is. DE benzina: PTA_p=0.015 → **VIOLATA**
- Ucraina SE benzina: PTA_p=0.000 → **VIOLATA**
- Ucraina SE diesel: PTA_p=0.001 → **VIOLATA**
- Iran-Is. SE benzina: PTA_p=0.024 → **VIOLATA**

Casi validi: Ucraina DE diesel (PTA ok), Iran-Is. DE diesel (PTA ok), Iran-Is. SE diesel (PTA ok).

**Conseguenza:** per i 5 casi con PTA violata, il DiD non fornisce una stima causale
valida della specificità italiana. Solo i 3 casi PTA-validi sono interpretabili.

**Fix:** riportare separatamente i 3 casi validi e i 5 invalidi. Considerare
l'uso di un DiD con tendenze lineari parallele (event study) per gestire la violazione PTA.

### 2.8 DiD v2: δ molto diversi da v1 [NUOVO — DA INVESTIGARE]

**Problema:** il DiD Ucraina IT vs DE benzina è passato da −0.024 (v1, n.s.) a +0.184 EUR/L
(v2, p≈0). Un cambiamento di 7× del parametro suggerisce una modifica di finestra,
baseline o definizione del margine tra v1 e v2 che non è documentata.

**Ipotesi:** v2 potrebbe usare una finestra post-shock più lunga (che include anni
con dinamiche diverse tra IT e DE) o confrontare prezzi pompa invece di margini.
Da verificare nel codice di script 04.

### 2.9 MCMC Brent Ucraina: Rhat=1.162 [PERSISTENTE]

**Non risolto.** Il τ_Brent-Ucraina rimane inaffidabile (Rhat=1.162 >> 1.01).
Il problema non impatta i test sul margine (che usano τ_Benzina e τ_Diesel), ma
τ_Brent-Ucraina non dovrebbe essere citato come punto puntuale.

**Fix:** re-run MCMC con prior più informativo, chains più lunghe (tune=10000),
o prior su tau_raw ristretto a ±30% attorno all'evento.

---

## 3. Diagnostici OLS (Script 02)

Applicati ai residui della regressione piecewise-lineare sui log-prezzi.
DW < 1.5 in 9/9 serie: autocorrelazione sistematica. Questo è strutturale nelle serie
temporali settimanali energetiche, non un artefatto del campione.

| Evento | Serie | DW | Esito DW | BP_p | SW_p | ν_StudentT | Rhat |
|---|---|---|---|---|---|---|---|
| Ucraina | Brent | 0.307 | **AUTOCORR** | 0.198 ok | 0.119 ok | 16.92 | 1.162 ⚠ |
| Ucraina | Benzina | 0.366 | **AUTOCORR** | 0.349 ok | 0.000 **NON-NORM** | 1.47 | 1.001 ✓ |
| Ucraina | Diesel | 0.294 | **AUTOCORR** | 0.368 ok | 0.000 **NON-NORM** | 5.23 | 1.004 ✓ |
| Iran-Israele | Brent | 1.202 | **AUTOCORR** | 0.453 ok | 0.067 ok | 16.71 | 1.002 ✓ |
| Iran-Israele | Benzina | 0.422 | **AUTOCORR** | 0.123 ok | 0.869 ok | 21.77 | 1.002 ✓ |
| Iran-Israele | Diesel | 0.329 | **AUTOCORR** | 0.021 **ETEROSC** | 0.489 ok | 22.96 | 1.001 ✓ |
| Hormuz | Brent | 1.091 | **AUTOCORR** | 0.169 ok | 0.150 ok | 22.20 | 1.002 ✓ |
| Hormuz | Benzina | 1.316 | **AUTOCORR** | 0.010 **ETEROSC** | 0.000 **NON-NORM** | 13.95 | 1.002 ✓ |
| Hormuz | Diesel | 0.929 | **AUTOCORR** | 0.136 ok | 0.682 ok | 21.69 | 1.001 ✓ |

**Nota Brent Ucraina:** Rhat=1.162 > 1.05 → convergenza MCMC dubbia. Il τ_price
per i test sul margine proviene dalle run Benzina/Diesel (convergenti), non da Brent.
Il problema non si propaga ai test del margine, ma τ_Brent-Ucraina non è affidabile.

---

## 4. Batteria di Test (Script 03 v2)

### 4.1 Architettura Multi-Split v2

In v2 i split nella BH sono solo quelli esogeni:

| Split | Definizione | Ruolo | In BH-A? |
|---|---|---|---|
| shock_hard | Data geopolitica fissa | Baseline di letteratura | ✓ |
| τ_price | Changepoint MCMC su log-prezzo | Esogeno al margine | ✓ |
| τ_margin | Changepoint Bai-Perron sul margine | **Endogeno** | ✗ (solo timing descrittivo) |

### 4.2 Test 1 — Welch t (Primario, BH-A)

H₀: μ_post = μ₂₀₁₉, one-sided upper. Famiglia BH-A: 8 Welch (2 eventi × 2 carb. × 2 split).

**Risultati v2 (split shock_hard e τ_price):**

| Evento | Carb. | Split | δ_vs_2019 | p | n_eff | Flag | BH-A |
|---|---|---|---|---|---|---|---|
| Ucraina | Benzina | shock_hard | +0.089 | 0.0001 | 4.0 | CAUTELA | ✓ (inconcl.) |
| Ucraina | Benzina | τ_price | +0.074 | 0.0000 | 5.4 | ATTENZIONE | ✓ |
| Ucraina | Diesel | shock_hard | +0.073 | 0.0008 | 8.0 | ATTENZIONE | ✓ |
| Ucraina | Diesel | τ_price | +0.056 | 0.0009 | 10.4 | ok | ✓ |
| Iran-Is. | Benzina | shock_hard | +0.079 | 0.0000 | 6.3 | ATTENZIONE | ✓ |
| Iran-Is. | Benzina | τ_price | +0.076 | 0.0000 | 7.5 | ATTENZIONE | ✓ |
| Iran-Is. | Diesel | shock_hard | +0.060 | 0.0000 | 17.5 | ok | ✓ |
| Iran-Is. | Diesel | τ_price | +0.061 | 0.0000 | 14.2 | ok | ✓ |

**Nota n_eff:** Ucraina benzina ρ̂=0.695 → n_eff=7.9, fattore inflazione 5.6×, Andrews BW=22.
Il t_stat = 4.49 (shock_hard) è gonfiato; con df_eff=6.9 il t_critico è 1.899 invece
di 1.681 nominale. Il rigetto regge ma con molto meno margine.

**Iran-Israele pre_anomalo = TRUE:** script 07 ha identificato il break strutturale
a luglio 2024 (benzina) e ottobre 2023 (diesel). Il pre-shock di Iran-Is. era già
strutturalmente diverso dal 2019. Il Welch conferma il livello elevato, ma l'anomalia
non è stata causata dallo shock.

### 4.3 Test 2 — Mann-Whitney (BH-A)

Confronto ordinale post-shock vs 52 settimane del 2019. 8 test in BH-A (stessi split).

| Evento | Carb. | p (shock_hard) | p (τ_price) | BH-A |
|---|---|---|---|---|
| Ucraina | Benzina | 0.0001 | 0.0000 | ✓ |
| Ucraina | Diesel | 0.0000 | 0.0003 | ✓ |
| Iran-Is. | Benzina | 0.0000 | 0.0000 | ✓ |
| Iran-Is. | Diesel | 0.0000 | 0.0000 | ✓ |

**16/16 test BH-A rigettati.** MW non dipende dallo split (confronto vs 2019 fisso),
quindi τ_price e shock_hard danno risultati quasi identici.

### 4.4 Test 3 — Block Permutation (H₀_locale, esplorativo)

**[non in BH-A da v2]** Testa salto locale pre→post. Con τ_price come split:

| Evento | Carb. | Δ mediano | p_perm | Note |
|---|---|---|---|---|
| Ucraina | Benzina | +0.058 | 0.014 * | gap τ_price→τ_margin = 70gg |
| Ucraina | Diesel | +0.047 | 0.035 * | |
| Iran-Is. | Benzina | −0.006 | 0.724 n.s. | δ_local negativo (margine sceso) |
| Iran-Is. | Diesel | −0.021 | 0.922 n.s. | δ_local negativo |

*(Valori da split shock_hard, coerenti con output script 03)*

**HAC con Andrews BW:** Ucraina benzina HAC_p=0.020 (BW=22≥n/2 → non informativo).
Iran-Is. benzina HAC_p=0.312 n.s., diesel HAC_p=0.054 ~borderline.

### 4.5 τ_margin — Timing Descrittivo (non in BH)

| Evento | Carb. | τ_price | τ_margin | τ_lag (gg) | Tipo |
|---|---|---|---|---|---|
| Ucraina | Benzina | 2022-01-03 | 2022-03-14 | **+70** | REATTIVO |
| Ucraina | Diesel | 2022-01-03 | 2022-03-14 | **+70** | REATTIVO |
| Iran-Is. | Benzina | 2025-04-28 | 2025-04-21 | −7 | SINCRONO |
| Iran-Is. | Diesel | 2025-05-05 | 2025-06-02 | +28 | REATTIVO |
| Hormuz | Benzina | 2026-03-02 | 2025-11-24 | **−98** | ANTICIPATORIO ⚠ |
| Hormuz | Diesel | 2026-02-23 | 2026-01-05 | **−49** | ANTICIPATORIO ⚠ |

Il segnale anticipatorio di Hormuz (margine si espande 3 mesi prima del prezzo
wholesale) è potenzialmente rilevante ma basato su soli 12 settimane post-shock
con n_eff molto basso (3.1 benzina, 1.3 diesel). Da verificare a run aggiornata.

---

## 5. Correzione BH Globale (Script 05 v2)

### 5.1 Composizione della Famiglia v2 (pulita)

| Famiglia | Fonte | Tipo | N test | H₀ | Note |
|---|---|---|---|---|---|
| **BH-A primaria** | Welch t (shock_hard + τ_price) | confirmatory | 8 | μ_post = μ₂₀₁₉ | ✓ |
| **BH-A primaria** | Mann-Whitney (shock_hard + τ_price) | confirmatory | 8 | domina vs 2019 | ✓ |
| **BH-B ausiliaria** | DiD IT vs DE/SE | confirmatory | 8 | δ_DiD = 0 | H₀ diversa |
| Esplorativi [no BH] | Block perm + HAC | esplorativo | 16 | μ_post = μ_pre | H₀_B ≠ H₀_A |
| Esplorativi [no BH] | τ_margin timing | descrittivo | — | — | nessun p-value |
| **TOTALE BH** | | | **24** | | (vs 56 in v1) |

### 5.2 Risultati BH-A (famiglia primaria, 16 test)

**16/16 rigettati a FDR 5%.** Tutti i p-value BH-adjusted: ≤ 0.0009.

| Evento | Carb. | Split | p_welch | p_adj | p_MW | p_adj | Esito |
|---|---|---|---|---|---|---|---|
| Ucraina | Benzina | shock_hard | 0.0001 | 0.0001 | 0.0001 | 0.0001 | ✓ |
| Ucraina | Benzina | τ_price | 0.0000 | 0.0000 | 0.0000 | 0.0000 | ✓ |
| Ucraina | Diesel | shock_hard | 0.0008 | 0.0009 | 0.0000 | 0.0000 | ✓ |
| Ucraina | Diesel | τ_price | 0.0009 | 0.0009 | 0.0003 | 0.0003 | ✓ |
| Iran-Is. | Benzina | shock_hard | 0.0000 | 0.0000 | 0.0000 | 0.0000 | ✓ |
| Iran-Is. | Benzina | τ_price | 0.0000 | 0.0000 | 0.0000 | 0.0000 | ✓ |
| Iran-Is. | Diesel | shock_hard | 0.0000 | 0.0000 | 0.0000 | 0.0000 | ✓ |
| Iran-Is. | Diesel | τ_price | 0.0000 | 0.0000 | 0.0000 | 0.0000 | ✓ |

### 5.3 Risultati BH-B (famiglia ausiliaria DiD, 8 test)

> ⚠ **BUG:** script 05 usa `auxiliary_pvalues.csv` (v1) → report "0/8 rigettati" ERRATO.
> I risultati corretti sono in `did_results_v2.csv`: **7/8 significativi** (script 04).
> Ma molti δ sono negativi (IT < controllo): rigetto ≠ specificità italiana.
> Vedere §7 per interpretazione completa.

### 5.4 Confronto v1 vs v2

| Metrica | v1 | v2 |
|---|---|---|
| N test BH totali | 56 | 24 |
| Test τ_margin (endogeni) | 16 | 0 |
| Test H₀_locale (perm+HAC) in BH | 16 | 0 |
| Rigetti BH-A (primaria) | 24/24 | 16/16 |
| Rigetti BH-B (DiD) | 0/8 | 0/8* (*basati su v1) |
| Bug aggiornamento table2 | presente | risolto |

---

## 6. Classificazione Finale v2 (Script 03, post-BH-A)

La classificazione v2 usa BH-A (16 test, split esogeni) e integra script 07 per Iran-Is.
τ_margin non è in BH ma rimane come timing descrittivo.

| Evento | Carb. | δ_vs_2019 (shock_hard) | δ_local | pre_anom | n_eff | BH-A | Classificazione v2 |
|---|---|---|---|---|---|---|---|
| Ucraina | Benzina | **+0.089** | +0.058 | no | 4.0 | ✓* | **Confermato (cautela n_eff=4)** |
| Ucraina | Diesel | **+0.073** | +0.047 | no | 8.0 | ✓ | **Confermato** |
| Iran-Is. | Benzina | **+0.079** | −0.006 | **SÌ** | 6.3 | ✓ | **Anomalia strutturale pre-shock** |
| Iran-Is. | Diesel | **+0.060** | −0.021 | **SÌ** | 17.5 | ✓ | **Anomalia strutturale pre-shock** |
| Hormuz | Benzina | **+0.108** | +0.008 | **SÌ** | 3.1 | ✗ prel. | **Inconclusivo** ⚠ |
| Hormuz | Diesel | −0.010 | −0.083 | **SÌ** | 1.3 | ✗ prel. | **Neutro** ⚠ |

*Ucraina benzina shock_hard: BH-A ✓ ma classificazione "Inconclusivo (n_eff<5)" — il test è
tecnicamente rigettato ma non informativo per gli scopi confirmativi.

**Cambio terminologico v2 per Iran-Israele:** da "Compressione margine" (descriveva il
δ_local negativo, potenzialmente confondente) a "Anomalia strutturale pre-shock" (descrive
la causa — il break del luglio 2024 — identificata da script 07). Il livello post-shock
è anomalo rispetto al 2019 (BH-A ✓) ma il pre-shock era già al picco storico.

**Hormuz (aggiornamento da 7 a 12 settimane post-shock):**
- Benzina: δ_vs_2019=+0.104, z=5.5, ma n_eff=3.1 (test non informativo)
- Diesel: δ_vs_2019=+0.015, z=0.8, n_eff=1.3 (nessun segnale)
- τ_margin benzina=2026-03-23, diesel=2026-04-06 (da hormuz_preliminary.csv)
- Entrambe ESCLUSE dalla BH, da aggiornare a n_post≥20

---

## 7. Evidenza Ausiliaria (Script 04 v2)

### 7.1 Granger v2

| Carburante | Lag 1w | Lag 2w | Lag 3w | Lag 4w |
|---|---|---|---|---|
| Benzina | F=58.2 p<0.001 | F=37.3 p<0.001 | F=30.3 p<0.001 | F=22.9 p<0.001 |
| Diesel | F=42.2 p<0.001 | F=35.3 p<0.001 | F=26.3 p<0.001 | F=20.9 p<0.001 |

Trasmissione Brent → pompa confermata a tutti i lag. **Esplorativo — no BH.**

### 7.2 Rockets & Feathers v2

| Carburante | β_up | β_down | R&F index | p asimmetria |
|---|---|---|---|---|
| Benzina | +0.0039 | +0.0022 | 1.765 | 0.239 n.s. |
| Diesel | +0.0056 | +0.0017 | **3.324** | **0.091 ~** |

**Aggiornamento rispetto a v1:** diesel R&F index passa da 1.3 a 3.3 e p da 0.757 a 0.091.
Con più dati (inclusa crisi Iran-Is. 2025) emerge un segnale quasi-significativo di
asimmetria per il diesel: le salite sono ~3× più veloci dei ribassi. Da monitorare.

### 7.3 Difference-in-Differences v2

> ⚠ **ATTENZIONE:** i risultati v2 sono molto diversi da v1. Bug §2.6 impatta la BH
> (script 05 usa file sbagliato). PTA violata in 5/8 casi (§2.7). Interpretare con cautela.

**Risultati da `did_results_v2.csv` (fonte corretta, non usata da script 05):**

| Evento | Paese | Carb. | δ_DiD (EUR/L) | CI 95% | p | PTA | Interpretazione |
|---|---|---|---|---|---|---|---|
| Ucraina | Germania | Benzina | **+0.184** | [+0.150, +0.219] | 0.000*** | **VIOLATA** | *non valido* |
| Ucraina | Germania | Diesel | **−0.064** | [−0.105, −0.024] | 0.002** | valida | IT < DE ← contro specificità IT |
| Ucraina | Svezia | Benzina | **+0.202** | [+0.165, +0.239] | 0.000*** | **VIOLATA** | *non valido* |
| Ucraina | Svezia | Diesel | +0.002 | [−0.040, +0.043] | 0.928 n.s. | **VIOLATA** | *non valido* |
| Iran-Is. | Germania | Benzina | **−0.118** | [−0.140, −0.097] | 0.000*** | **VIOLATA** | *non valido* |
| Iran-Is. | Germania | Diesel | **−0.119** | [−0.144, −0.094] | 0.000*** | valida | IT < DE ← contro specificità IT |
| Iran-Is. | Svezia | Benzina | **−0.118** | [−0.138, −0.097] | 0.000*** | **VIOLATA** | *non valido* |
| Iran-Is. | Svezia | Diesel | **−0.119** | [−0.144, −0.094] | 0.000*** | valida | IT < SE ← contro specificità IT |

**Lettura dei 3 casi PTA-validi:**
- Ucraina DE diesel: IT ha margini inferiori alla Germania (δ=−0.064) → contro specificità IT
- Iran-Is. DE diesel: IT inferiore a Germania (δ=−0.119) → contro specificità IT
- Iran-Is. SE diesel: IT inferiore a Svezia (δ=−0.119) → contro specificità IT

**Conclusione provvisoria (solo casi PTA-validi):** i 3 DiD interpretabili mostrano tutti
δ negativo (IT ≤ controllo). L'evidenza è **contro** l'ipotesi di opportunismo specificamente
italiano per entrambi gli eventi.

**Nota sui δ molto grandi:** i δ di +0.18 e +0.20 EUR/L nei casi PTA-violati di Ucraina
sembrano eccessivamente grandi. Potrebbe riflettere una divergenza strutturale pre-esistente
non rimossa dalla PTA — da investigare (§2.8).

### 7.4 Windfall v2 (con correzione trend consumi −1.5%/anno)

| Evento | Carburante | δ_margine | n_sett. | Windfall | CI boot 95% |
|---|---|---|---|---|---|
| Ucraina | Benzina | +0.096 EUR/L | 44 | **+0.14 Mld EUR** | [+0.10, +0.18] |
| Ucraina | Diesel | +0.087 EUR/L | 44 | **+0.17 Mld EUR** | [+0.10, +0.23] |
| Iran-Is. | Benzina | +0.087 EUR/L | 29 | **+0.08 Mld EUR** | [+0.07, +0.09] |
| Iran-Is. | Diesel | +0.072 EUR/L | 29 | **+0.09 Mld EUR** | [+0.07, +0.10] |

*Descrittivo — nessun p-value. Volumi corretti per trend −1.5%/anno (EV + efficienza).*

---

## 8. Changepoint Bayesiano — Table 1

| Evento | Serie | τ̂ | CI 95% | Lag D | DW | SW_p | H₀ 30gg | ν | Rhat |
|---|---|---|---|---|---|---|---|---|---|
| Ucraina | Brent | 2021-12-13 | [Nov 08 – Jul 04] | −73 gg | 0.31 | 0.119 | NO | 16.9 | **1.162 ⚠** |
| Ucraina | Benzina | 2022-01-03 | [Dec 27 – Jan 17] | −52 gg | 0.37 | 0.000 | NO | 1.47 | 1.001 ✓ |
| Ucraina | Diesel | 2022-01-03 | [Dec 06 – Jan 17] | −52 gg | 0.29 | 0.000 | NO | 5.23 | 1.004 ✓ |
| Iran-Is. | Brent | 2025-04-28 | [Apr 07 – May 12] | −46 gg | 1.20 | 0.067 | NO | 16.7 | 1.002 ✓ |
| Iran-Is. | Benzina | 2025-04-28 | [Apr 14 – May 12] | −46 gg | 0.42 | 0.869 | NO | 21.8 | 1.002 ✓ |
| Iran-Is. | Diesel | 2025-05-05 | [Apr 21 – May 19] | −39 gg | 0.33 | 0.489 | NO | 23.0 | 1.001 ✓ |
| Hormuz | Brent | 2026-02-16 | [Feb 09 – Feb 23] | −12 gg | 1.09 | 0.150 | **SÌ** | 22.2 | 1.002 ✓ |
| Hormuz | Benzina | 2026-03-02 | [Feb 23 – Mar 02] | +2 gg | 1.32 | 0.000 | **SÌ** | 13.9 | 1.002 ✓ |
| Hormuz | Diesel | 2026-02-23 | [Feb 23 – Mar 02] | −5 gg | 0.93 | 0.682 | **SÌ** | 21.7 | 1.001 ✓ |

**ν molto basso per Ucraina Benzina (ν=1.47 → distribuzione quasi-Cauchy):** questo
è consistente con i residui fortemente non-normali (SW_p≈0) e spiega perché la
Skewed-T è raccomandata dallo script 06 per questo scenario.

---

## 9. Verifica Assunzione Distributiva (Script 06)

| Scenario | Tipo | Raccomandazione | ΔAIC(skewt vs t) |
|---|---|---|---|
| Ucraina Benzina (log-prezzo) | Residui OLS | **Skewed-T** | < −2 |
| Ucraina Diesel (log-prezzo) | Residui OLS | **Skewed-T** | < −2 |
| Ucraina Brent (log-prezzo) | Residui OLS | StudentT (ok) | ≈0 |
| Iran-Is. Brent (log-prezzo) | Residui OLS | StudentT (ok) | ≈0 |
| Iran-Is. Benzina (log-prezzo) | Residui OLS | StudentT (ok) | ≈0 |
| Iran-Is. Diesel (log-prezzo) | Residui OLS | Normale | > 0 |
| Hormuz Brent (log-prezzo) | Residui OLS | **Skewed-T** | < −2 |
| Hormuz Benzina (log-prezzo) | Residui OLS | **Skewed-T** | < −2 |
| Hormuz Diesel (log-prezzo) | Residui OLS | Normale | > 0 |
| Ucraina Benzina (crack spread) | Post-shock | **Skewed-T** | < −2 |
| Ucraina Diesel (crack spread) | Post-shock | **Skewed-T** | < −2 |
| Iran-Is. Benzina (crack spread) | Post-shock | Normale | > 0 |
| Iran-Is. Diesel (crack spread) | Post-shock | **Skewed-T** | < −2 |
| Hormuz Benzina/Diesel | Post-shock | N/A (n<6) | — |

La Skewed-T è raccomandata per 7/13 scenari con dati sufficienti. L'impatto è sulla
stima del changepoint τ, non sui test di script 03 (non parametrici o HAC).

---

## 10. Analisi Annuale — Sintesi

**Benzina** (margine medio annuo vs μ₂₀₁₉=0.168 EUR/L, soglia 2σ=0.038 EUR/L):

| Anno | μ (EUR/L) | δ vs 2019 | Anomalo 2σ | MW p vs 2019 | Windfall M€ |
|---|---|---|---|---|---|
| 2019 | 0.168 | 0.000 | no | 0.501 | 0 |
| 2020 | 0.204 | +0.036 | no* | 0.000 | +340 |
| 2021 | 0.182 | +0.013 | no | 0.008 | +126 |
| 2022 | 0.254 | **+0.085** | **SÌ** | 0.000 | **+807** |
| 2023 | 0.235 | **+0.067** | **SÌ** | 0.000 | +633 |
| 2024 | 0.233 | **+0.064** | **SÌ** | 0.000 | +622 |
| 2025 | 0.255 | **+0.086** | **SÌ** | 0.000 | **+818** |
| 2026 (parz.) | 0.274 | **+0.106** | **SÌ** | 0.000 | +288 |

*2020: δ=+0.036 sotto la soglia 2σ=0.038 di pochissimo, ma volumes COVID molto ridotti:
il windfall per liter è quasi anomalo ma il windfall totale è sovrastimato (volumi fissi).

I margini sono rimasti **strutturalmente sopra il 2019 per tutti gli anni dal 2022 in
poi**. Il 2025 supera il 2022 (benzina +0.086 vs +0.085). Il fenomeno non è transitorio.

**Diesel** — stesso pattern con windfall maggiori: 2022 +2.297 M€, 2025 +2.362 M€.

---

## 11. Pre-shock Anomaly (Script 07 v2)

### 11.1 Break Strutturali Bai-Perron (finestra 2023-01 / 2025-06-12)

| Carburante | τ_pre | δ_break | Livello prima | Livello dopo | n |
|---|---|---|---|---|---|
| Benzina | **2024-07-15** | +0.019 | 0.230 EUR/L | 0.250 EUR/L | 80/48 |
| Diesel | **2023-05-15** | −0.067 | 0.245 → 0.178 | (discesa) | 19/20 |
| Diesel | **2023-10-02** | +0.040 | 0.178 → 0.218 | (rialzo) | 20/89 |

### 11.2 Statistiche Annuali e 2025-H1 pre-shock

| Anno | Benzina μ (EUR/L) | δ vs 2019 | MW p vs 2019 | Diesel μ | δ vs 2019 |
|---|---|---|---|---|---|
| 2019 | 0.168 | 0 | — | 0.149 | 0 |
| 2023 | 0.235 | +0.067 | 0.000 | 0.215 | +0.066 |
| 2024 | 0.233 | +0.065 | 0.000 | 0.210 | +0.061 |
| **2025-H1 pre-shock** | **0.254** | **+0.086** | **0.000** | **0.230** | **+0.081** |

**MW 2025-H1 vs 2024:** benzina p=0.001, diesel p=0.008 → il 2025-H1 è significativamente
più alto anche del 2024, non solo del 2019.

### 11.3 Implicazioni per l'interpretazione Iran-Israele

Il fenomeno Iran-Israele è meglio descritto come "**anomalia strutturale con tre break
successivi (2022, 2023, 2024) che si stabilizza al picco storico nel 2025-H1**".
Lo shock del giugno 2025 non ha causato il rialzo — ha semmai generato una compressione
del margine rispetto al pre-shock. La classificazione v2 è coerente con questo.

---

## 12. Roadmap Correttiva — Priorità Aggiornate v2

| Priorità | Issue | Stato | Fix | Script |
|---|---|---|---|---|
| 1 | Script 05 carica DiD v1 (§2.6) | **APERTO — CRITICO** | Cambiare path da `auxiliary_pvalues.csv` a `did_results_v2.csv` | 05 |
| 2 | PTA violata in 5/8 DiD (§2.7) | **APERTO** | DiD con trend lineari (event study) | 04 |
| 3 | δ_DiD anomali v2 vs v1 (§2.8) | **DA INVESTIGARE** | Verificare finestra/definizione in script 04 | 04 |
| 4 | MCMC Brent Ucraina Rhat=1.162 (§2.9) | **APERTO — PERSISTENTE** | Re-run con tune=10000, prior ristretto | 02 |
| 5 | HAC BW≥n/2 per Ucraina/Iran-Is. benzina (§2.4) | **PARZIALMENTE GESTITO** | Segnalato in output; considerare subsample bootstrap | 03 |
| 6 | Naming mismatch file (run_all_v2 checker) | **APERTO — MINORE** | Aggiornare 3 path nel checker: `exploratory_results_v2`, `tau_margin_descriptive`, `global_bh_v2` | run_all_v2 |
| 7 | Convergenza Hormuz (n_post=12, n_eff<5) | **STRUTTURALE** | Attendere n_post≥20 prima di re-run confermativo | 03, 04 |
| ~~1~~ | ~~τ_margin nella BH confirmatory~~ | ✓ **RISOLTO** | — | — |
| ~~2~~ | ~~H₀ miste nella famiglia BH~~ | ✓ **RISOLTO** | — | — |
| ~~3~~ | ~~Bug table2 aggiornamento BH~~ | ✓ **RISOLTO** | — | — |
| ~~5~~ | ~~n_eff non riportato~~ | ✓ **RISOLTO** | — | — |
| ~~7~~ | ~~Windfall volumi fissi 2022~~ | ✓ **RISOLTO** (trend -1.5%) | — | — |

---

## 12. Limitazioni Strutturali

**Proxy del margine:** il crack spread ARA non cattura il differenziale di base
ARA-Mediterraneo, che può variare sistematicamente durante le crisi (rerouting navi,
costi assicurazione). Una parte del "margine anomalo" potrebbe riflettere questo.

**Causalità vs correlazione:** le classificazioni descrivono pattern statistici.
"Margine anomalo positivo" è consistente con opportunismo ma anche con: effetti
FIFO/LIFO su inventario, risk premium razionale, cost-push non catturato dalla
proxy ARA/ICE, riduzione temporanea della concorrenza.

**n_eff ridotto:** con ρ≈0.85 e finestre di 20–27 settimane, l'evidenza effettiva
è di circa 2–3 osservazioni indipendenti per evento. I test parametrici sovrastimano
sistematicamente la certezza statistica.

**Hormuz è preliminare:** 7 settimane post-shock. I risultati sono direzionali.