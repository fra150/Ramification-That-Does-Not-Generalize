# Findings v0.1 — chiusura del primo sweep

> **Avvertenza storica.** I numeri di questo documento sono stati prodotti con
> il bug F1 del training BPE (heap lazy che perde coppie valide). Non vanno usati
> nel paper. Restano qui per tracciare l'errore e la sua correzione. I numeri
> corretti e il nuovo esperimento sull'assegnazione morfologica sono in
> [`docs/findings-v0.2.md`](findings-v0.2.md).

Questo documento chiude la v0.1: i sei run sperimentali del primo sweep sono
arrivati, il baseline flat sul sintetico è incluso, e il bug del codec è stato
corretto. I numeri sono presi dai CSV in `results/agent*/results.csv`. La
matrice v0.2 a 3 seed è in corso e avrà un documento separato
(`docs/findings-v0.2.md`).

## 1. Risultati

### Corpus sintetico (ground truth morfologico)

| metodo | vocab | tok/char | bits/char | boundary F1 | precision | recall | decode word |
|---|---:|---:|---:|---:|---:|---:|---:|
| flat | 256 | 0.4027 | 2.8221 | 0.7934 | 0.7212 | 0.9310 | 1.0 |
| flat | 512 | 0.3152 | 2.3834 | 0.8741 | 0.8882 | 0.8793 | 1.0 |
| flat | 1024 | 0.2849 | 2.4057 | 0.8063 | 0.8952 | 0.7665 | 1.0 |
| hier2 | 256 | 0.3986 | 2.8076 | 0.7941 | 0.7202 | 0.9308 | 1.0 |
| hier2 | 512 | 0.3112 | 2.3420 | **0.8917** | 0.9124 | 0.8867 | 1.0 |
| hier2 | 1024 | 0.2813 | 2.3660 | **0.8246** | 0.9188 | 0.7778 | 1.0 |
| hier3 | 256 | 0.3975 | 2.8150 | 0.7849 | 0.7122 | 0.9188 | 1.0 |
| hier3 | 512 | 0.3097 | **2.3286** | **0.8934** | 0.9152 | 0.8867 | 1.0 |
| hier3 | 1024 | 0.2816 | 2.3737 | 0.8235 | 0.9187 | 0.7765 | 1.0 |

### Corpus codice (stdlib Python, 400k caratteri)

| metodo | vocab | tok/char | bits/char | decode word |
|---|---:|---:|---:|---:|
| flat | 256 | 0.5166 | 3.7430 | 1.0 |
| flat | 512 | 0.4052 | 3.0977 | 1.0 |
| flat | 1024 | 0.3526 | 2.7565 | 1.0 |
| hier2 | 256 | 0.5166 | 3.7430 | 1.0 |
| hier2 | 512 | 0.4052 | 3.0982 | 1.0 |
| hier2 | 1024 | 0.3512 | 2.7460 | 1.0 |
| hier3 | 256 | 0.5167 | 3.7400 | 1.0 |
| hier3 | 512 | 0.4047 | 3.0948 | 1.0 |
| hier3 | 1024 | 0.3512 | 2.7467 | 1.0 |

## 2. Cosa dicono i numeri

1. **Sul codice la gerarchia è neutra.** Flat, hier2 e hier3 differiscono per
   meno dello 0,5% su tok/char e bpc. La struttura ramificata non penalizza la
   compressione: condizione necessaria superata. Non è un guadagno, è un'assenza
   di danno.
2. **Sul sintetico il vantaggio c'è, ma è piccolo e non monotono.** A 512 la
   gerarchia guadagna ~1,8 punti di F1 sul piatto (0,892/0,893 contro 0,874); a
   1024 ~1,8 punti (0,825/0,824 contro 0,806); a 256 il vantaggio è zero
   (0,794/0,785 contro 0,793). Il piatto non è un baseline debole: è già vicino
   al soffitto a 256 e resta competitivo a 1024.
3. **La caduta a 1024 è un effetto di over-merging, non di "più vocabolario".**
   La precisione sale (0,72 → 0,92) mentre la recall crolla (0,93 → 0,78): il
   livello 1 riceve 501 merge e divora i confini morfologici.
4. **hier2 e hier3 sono quasi identici.** Il terzo livello non aggiunge nulla su
   questo corpus. La ramificazione utile, se esiste, è quella che separa morfemi
   da parole, non quella che aggiunge un livello di frase.
5. **Il bpc non è monotono.** A 1024 è peggiore che a 512. L'add-1 smoothing
   penalizza i vocabolari grandi per costruzione; la distribuzione del test
   (uniforme sulle parole held-out) è diversa da quella del training (Zipfiana).
   La v0.2 usa add-alpha 0.1 e riporta i numeri su un denominatore corretto.

## 3. Il codec: la correzione che cambia la lettura

Il codec v0.1 era esatto a livello di parola ma non di documento:
`"a . b"` decodificava `"a .b"`. Il codec a chunk (`ramified/codec.py`) 
normalizza gli spazi, codifica ogni chunk con `EOW` e decodifica `EOW` come
spazio. Il round-trip è esatto sul testo normalizzato, senza token aggiuntivi.

Il primo test v0.2 sul codice sembrava un peggioramento grave: bpc 3.885 contro
3.098 della v0.1 a vocab 512. Il confronto mescolava due cambiamenti, codec e
denominatore. Il testo del codice si riduce del 20,2% normalizzando gli spazi;
la v0.1 non codificava gli spazi ma li contava nel denominatore.

Confronto a parità di corpus e split (`experiments/codec_compare.py`):

| config | token test | tok/char grezzo | tok/char normalizzato | bpc grezzo | bpc normalizzato | round-trip doc |
|---|---:|---:|---:|---:|---:|---:|
| words/words (v0.1) | 16.568 | 0.414 | 0.519 | 3.138 | 3.934 | no |
| words/chunks | 18.604 | 0.465 | 0.583 | 3.501 | 4.389 | sì |
| chunks/chunks (v0.2) | **14.950** | **0.374** | **0.469** | **3.099** | **3.885** | **sì** |

A parità di denominatore il chunk codec è **meglio** della v0.1: 3,099 contro
3,138 sul grezzo, con 1.618 token in meno. Il 3,885 non è una regressione del
codec: è il costo onesto di contare anche gli spazi normalizzati. La v0.1 era
ottimista, non la v0.2 pessimista.

Sul sintetico i tre codec sono identici (6.105 token, 0.315 tok/char, 2.384 bpc,
F1 0.874): il cambio di codec non tocca il segnale morfologico.

**Decisione.** Adottare il chunk codec; riportare il bpc sul testo normalizzato;
dichiarare nel paper che il dominio del codec è il testo normalizzato, non il
testo grezzo. Per il codice questo significa perdere indentazione e newline: se
servirà il round-trip grezzo, servirà un codec dei whitespace separato.

## 4. L'ablazione del budget di livello 1

| vocab | metodo | budget L1 | F1 | precision | recall |
|---:|---|---:|---:|---:|---:|
| 512 | flat | — | 0.874 | 0.888 | 0.879 |
| 512 | hier2 | 32 | 0.871 | 0.886 | 0.876 |
| 512 | hier2 | 64 | 0.839 | 0.855 | 0.841 |
| 512 | hier2 | 128 | 0.874 | 0.892 | 0.873 |
| 512 | hier2 | **256** | **0.892** | **0.912** | **0.887** |
| 1024 | flat | — | 0.806 | 0.895 | 0.766 |
| 1024 | hier2 | 32 | 0.803 | 0.892 | 0.764 |
| 1024 | hier2 | 64 | 0.781 | 0.868 | 0.741 |
| 1024 | hier2 | 128 | 0.812 | 0.906 | 0.767 |
| 1024 | hier2 | **256** | **0.834** | **0.932** | **0.786** |
| 1024 | hier2 | 512 | 0.823 | 0.920 | 0.775 |

Il budget fisso di 256 merge al livello 1 è il migliore nei due vocabolari
testati. A vocab 256 il budget totale è 234 (sintetico) o 160 (codice), quindi
`hier2_b256` collassa a flat: è un controllo, non un errore. Il runner v0.2
riporta `n_levels` effettivi e i merge per livello.

## 5. Bug e questioni aperte

- **Codec:** risolto con il chunk codec; il dominio è il testo normalizzato.
- **Smoothing:** risolto con add-alpha 0.1; i confronti v0.1/v0.2 vanno fatti a
  parità di estimatore.
- **Budget del livello 1:** l'ablazione indica 256; la matrice v0.2 dirà se
  regge a 2048/4096. Il test di scala a 4096 dà F1 0,673, quindi il budget fisso
  non scala automaticamente.
- **Timing:** i tempi del primo sweep sono rumorosi perché 10 agenti giravano in
  parallelo. Non sono un risultato di scaling.
- **Corpus codice:** 400k caratteri sono un toy. Per il paper serve un corpus
  più grande e un secondo corpus reale (es. testo italiano pubblico dominio).
- **Review 7-8 e design Fase 2:** ancora in corso. Vanno letti prima di
  congelare i numeri.
- **Letteratura:** il meccanismo è noto (SLP / Re-Pair / BPE gerarchico);
  l'embedding sull'albero di merge è anticipato da Bass (2026). Il contributo
  difendibile è il confronto multi-livello a parità di compressione, non
  l'invenzione del meccanismo.

## 6. Prossimo giro

1. Matrice v0.2: 4 metodi × 5 vocabolari × 2 corpus × 3 seed = 120 righe.
2. Aggregato con media e deviazione standard; confronto flat vs hier2_b256.
3. Lettura delle review 7-8 e del design Fase 2.
4. Decisione: l'ablazione morfema-first vs frequenza-first è il test più forte
   della ramificazione; va aggiunta prima del paper.
5. Solo se la Fase 1 dà un segnale stabile: Fase 2 embedding, poi Fase 3 canale,
   poi preprint.
