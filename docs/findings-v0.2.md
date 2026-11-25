# Findings v0.2 — il bug, il null, e la finestra morfologica

> **Nota (2026-11-24).** Questo documento è storico. La Fase 1f lo ha superato:
> con un generatore a lunghezze variabili e uno split pair-disjoint l'effetto
> non sopravvive al criterio pre-registrato. La sintesi corrente è
> [`findings-v0.3.md`](findings-v0.3.md); la decisione è il paper A
> (negativo/metodologico).

Questo documento sostituisce la lettura dei risultati v0.1. La v0.1 era stata
prodotta con un BPE non standard; la v0.2 ha corretto il bug, ha mostrato che
la gerarchia a partizione di budget è equivalente al BPE piatto, e ha testato
l'ablazione che conta davvero: l'assegnazione dei merge ai livelli per
provenienza morfologica.

I numeri vengono da:

- `results/fixed_smoke/results.csv` — smoke corretto, 4 metodi × 2 corpus × v512, seed 42
- `results/level_assignment.csv` — 3 seed × 3 vocabolari × 5 modalità
- `results/level_assignment_v512_10seeds.csv` — 10 seed × v512 × 4 modalità
- `results/level_assignment_stats_v512.csv` — statistiche appaiate

## 1. Il bug F1 e perché la matrice a 20 agenti è invalida

La review avversariale di `ramified/bpe.py` ha trovato un bug critico nel
training. L'heap lazy perde coppie valide: quando un merge distrugge
un'occorrenza di una coppia in una parola ma la coppia sopravvive in un'altra,
il conteggio viene decrementato ma non reinserito nell'heap. L'heap conserva
solo la vecchia entrata, più grande; quando viene estratta, la validazione la
scarta, e la coppia diventa invisibile. Il training non è quindi BPE greedy:
può scegliere una coppia con conteggio inferiore, o fermarsi con coppie
positive ancora presenti.

Repro minimale:

```text
sequences=[[1,0,2],[1,0],[0,2]], freqs=[1,9,4], 5 merge
buggy:  (1,0) count 10, poi (3,2) count 1, heap vuoto con (0,2): 4
fixed:  (1,0) count 10, poi (0,2) count 4, poi (10,2) count 1
```

Sul sintetico v512, con il codice buggy, al confine di livello 294 il flat
sceglie `u+re` con conteggio 352, mentre la coppia più frequente nello stato è
`p+i` con conteggio 547. Il livello 2 della gerarchia ricostruisce l'heap e
trova `p+i`: la gerarchia stava riparando accidentalmente il bug del flat a
ogni nuovo livello. Il primo disallineamento tra flat e hier2 è esattamente al
confine 294, e le regole comuni sono 459/490.

Dopo il fix (push di ogni conteggio cambiato, inclusi i decrementi):

- `check_levels` sul sintetico v512: flat e hier2 hanno **490/490 regole
  identiche** e tokenizzazione identica.
- Smoke corretto a v512, seed 42:

| metodo | tok/ch | bpc | F1 |
|---|---:|---:|---:|
| flat | 0.309 | 2.354 | 0.867 |
| hier2 | 0.309 | 2.354 | 0.867 |
| hier2_b256 | 0.309 | 2.354 | 0.867 |
| hier3 | 0.309 | 2.354 | 0.867 |

Sul codice, tutti i metodi danno 0.469 tok/ch e 3.886 bpc. **La gerarchia a
partizione di budget è un no-op.** I 120 run della matrice a 20 agenti sono un
artefatto del bug: restano in `results/matrix/` come documentazione, ma non
vanno usati nel paper.

## 2. L'ablazione che testa la ramificazione

### Disegno

Si tiene fisso il BPE flat (stesse regole, stesso vocabolario) e si cambia solo
l'assegnazione delle regole ai due livelli. Per ogni regola si contano le
occorrenze intra-morfema e cross-morfema nell'albero di derivazione delle parole
di training. Le regole intra vanno al livello 1, le cross al livello 2, con
chiusura delle dipendenze: una regola di livello 1 non può dipendere da una
regola di livello 2, altrimenti non potrebbe mai applicarsi.

Controlli:

- `frequency`: i primi n1 merge per rango — la vecchia gerarchia a budget, che
  dopo il fix è identica al flat;
- `random_closed`: un prefisso casuale di un ordine topologico, della stessa
  dimensione, chiuso per dipendenze;
- `reverse`: cross-first, per verificare se basta un riordino qualsiasi.

La metrica principale è `boundary_f1` sulle parole held-out; la secondaria è
`bits_per_char` con add-alpha 0.1 sul testo normalizzato. Il confronto è
appaiato per seed: lo stesso seed usa lo stesso corpus sintetico.

### v512, 10 seed (42-51)

| confronto | Δ F1 | t | W/L/T | Δ bpc | t |
|---|---:|---:|---:|---:|---:|
| morph − frequency | **+0.0092** | 6.52 | 10/0/0 | **−0.0258** | −5.38 |
| random_closed − frequency | −0.0064 | −1.73 | 3/7/0 | +0.0361 | 3.04 |
| morph − random_closed | **+0.0156** | 3.81 | 9/1/0 | **−0.0619** | −4.73 |

Intervalli al 95%:

- morph − frequency, F1: **[+0.0060, +0.0125]**
- morph − frequency, bpc: **[−0.0366, −0.0150]**
- morph − random_closed, F1: **[+0.0064, +0.0249]**
- morph − random_closed, bpc: **[−0.0916, −0.0323]**

Morph migliora sia l'allineamento sia la compressione su **tutti e 10 i seed**.
Random_closed peggiora entrambi in media. Non è un riordino generico: è la
provenienza morfologica che conta. La F1 in tabella è macro-averaged; il
controllo micro-F1 parziale (5 seed) è nella sezione 4b.

### v1024 e v2048, 3 seed (42-44)

A v1024 e v2048, morph = frequency **esattamente**: Δ F1 0.000, Δ bpc 0.000,
3 pareggi su 3 seed. L'effetto è confinato a v512.

## 3. Interpretazione

Il risultato non è "più livelli è meglio". È:

1. La gerarchia a partizione di budget è equivalente al BPE flat. I livelli sono
   etichette, non vincoli. Questo è il caveat F9 della review.
2. Un'assegnazione a due livelli per provenienza morfologica cambia la
   tokenizzazione e migliora boundary F1 e bpc a v512, contro i controlli
   frequency e random_closed.
3. L'effetto svanisce a v1024/2048. La lettura più plausibile è una finestra di
   budget: il livello 1 funziona quando il suo budget è vicino alla scala dei
   morfemi. A vocabolari più grandi il prefisso per frequenza contiene già i
   merge intra-morfema, e l'assegnazione non cambia più nulla.

È un risultato piccolo (+0.9 punti F1, −0.026 bpc) e condizionato, ma è
controllato: stesse regole, stesso vocabolario, stesso corpus, stesso numero di
livelli, cambia solo l'assegnazione.

## 4. Cosa resta aperto

1. **Corpus reale morfologico.** Il sintetico è controllato ma artificiale. Serve
   un corpus con confini verificabili e licenza chiara; poi si rifà il confronto
   morph vs frequency vs random_closed.
2. **Fase 2 — embedding composizionali.** Il guadagno di F1 non implica un
   guadagno di rappresentazione. Arnett et al. (2025) trovano che
   l'allineamento spiega poco della performance downstream. Il test tree-
   recursive vs flat mean vs GRU/Transformer vs oracle è il prossimo passo.
3. **Baseline morfologiche.** MorphBPE e TreeTokenizer sono i confronti giusti;
   flat BPE da solo non basta più.
4. **Statistica.** 10 seed sul sintetico sono un buon inizio, ma l'effetto è
   piccolo e va replicato su corpus e seed indipendenti. Il codice della
   tabella è deterministico: i tre seed del corpus codice sono repliche, non
   campioni indipendenti.
5. **Bug minori della review.** F2 (punctuation words-mode), F3 (EOW nel decode),
   F4 (unk e offset dei confini), F6/F7 (budget), F10/F11 (API edge). La v0.2
   usa il chunk codec, quindi F2 non tocca i numeri correnti, ma va sistemato
   prima del rilascio.
6. **Canale steganografico.** La non-iniettività dell'espansione resta un
   cassetto separato: modello di attaccante, capacità, rilevabilità. Non è
   encryption e non va presentato come tale.

## 4b. Caveat del benchmark (review dei dati, agente 8)

La review dei dati conferma C0 in modo indipendente e aggiunge limiti che vanno
scritti nel paper:

- **H1 — interpolazione, non composizione.** Ogni terna di test ha tutte e tre le
  coppie di componenti attestate in training (prefix-root, root-suffix,
  prefix-suffix: 2304/2304 ciascuna). Il filtro di attestazione sposta zero
  parole al default. Il risultato morph a v512 è quindi un risultato su **terne
  nuove con coppie note**, non su composizione di componenti mai viste. Serve
  uno split pair-disjoint (o zero-shot) prima di qualsiasi claim composizionale.
- **H3 — corpus codice.** Il corpus è composto dai primi 16 file in ordine
  alfabetico; `_pydecimal.py` è il 46% e il test è la coda dello stesso file.
  Non è un corpus di codice held-out. I numeri sul codice non vanno usati come
  evidenza generale.
- **C1/C2 — codec.** Il bpc è calcolato sul testo normalizzato, che è ciò che il
  codec codifica; il codec perde newline e indentazione (circa 7.020 caratteri
  sul test code). `decode_doc` è esatto solo sul testo normalizzato, non sul
  documento grezzo.
- **H4 — bpc.** La non monotonicità 512→1024 non è solo smoothing; add-alpha 0.1
  non la rimuove. Il bpc unigramma è una misura secondaria, da riportare con OOV
  e una seconda modellazione.
- **M1/M2 — metriche.** Macro e micro possono differire più dell'effetto; il
  codice ora riporta entrambi e i TP/FP/FN pooled. Bpc e boundary F1 usano
  campioni diversi (1324 tipi vs 2304) e i root sono sbilanciati: per il paper
  serve un unico test set condiviso.

Il controllo micro-F1 parziale (5 seed, run interrotto) dà macro ≈ micro e
morph ancora positivo su 4 seed su 5; il run completo a 10 seed va rifatto.

## 4c. Red team — cosa ha retto e cosa no

La prima ondata di report (`docs/redteam/`) conferma che l'implementazione è
sana: il fix F1 ha 0 divergenze da un riferimento indipendente su 10.325
decisioni, la provenienza ha 489/489 regole e 151.299 span corretti, la
chiusura ha zero demotion, il tie-break è positivo in 35/35 celle, macro ≈ micro.

Ma il red team restringe il claim:

- **Posizionale, non morfologico.** Confini costanti `{2, L-2}` riproducono morph
  esattamente su 10/10 seed (stesso L1, F1, bpc); cambiano solo 8/490 statistiche
  di regola.
- **Dipende dalla skew Zipfiana.** Con campionamento uniforme morph = frequency
  su 5/5 seed; togliere i periodi non basta a spegnere l'effetto.
- **Dipende dal numero di root.** A 60 root: +0.000087 (2/10 vittorie, 8
  pareggi). A 120 root: +0.0092.
- **Dipende dalla frazione del livello 1.** Finestra ~0.55-0.65; a 0.5 il segno
  della F1 si inverte. 0.55 batte 0.6, quindi 0.6 non è un massimo post-hoc, ma
  la frazione resta un grado di libertà dominante.
- **Supervisionato.** Usa gli stessi confini della metrica; il “10/0/0” è 9
  vittorie + 1 pareggio di una parola.
- **Non è compressione.** Un bigramma rimuove l'86-92% del guadagno bpc; sul
  testo in-distribution il segno si inverte.
- **Interpolazione.** Tutte le coppie di componenti sono attestate; il 96.8-98.0%
  delle parole di test è a distanza di Levenshtein 1 da una parola di training.
- **Vettori mai usati.** La Fase 1 non legge `Word.vector`; un modello bilineare
  per coppia arriva a cosine 0.999998 sullo split attuale.

**Stato:** C0 è morta (gerarchia a budget = flat). C1 è un effetto piccolo e
fragile, non un vantaggio morfologico generale. C2 non è testabile senza P5 e
senza split pair-disjoint. Il bpc è secondario. Il corpus codice non è generale.

**Decisione:** o un paper negativo/metodologico (A), o un ciclo time-boxed con
P5, generatore a lunghezze variabili, split pair-disjoint e corpus reale (B). La
raccomandazione è B time-boxed; se l'effetto non sopravvive, si scrive A.

## 5. La frase del paper

Non: "la tokenizzazione ramificata generalizza meglio".

Sì: "la gerarchia a partizione di budget è equivalente al BPE piatto;
un'assegnazione a due livelli per provenienza morfologica migliora boundary F1
di ~0.5-0.9 punti a vocabolario 512, su 10-20 seed, contro i controlli frequency
e random_closed, **su un holdout di terne nuove con coppie di componenti
attestate (interpolazione), in una finestra di frazione 0.55-0.65 e con skew
Zipfiana**; l'effetto svanisce a vocabolari più grandi, con campionamento
uniforme, a 60 root e sotto controlli posizionali. Non è un vantaggio
morfologico generale. La generalizzazione composizionale non è ancora testata:
richiede uno split pair-disjoint. Il meccanismo è noto (SLP / BPE gerarchico);
il confronto controllato e la caratterizzazione dei limiti sono il contributo."

## 6. Comandi

```powershell
# smoke corretto
python -m experiments.run_matrix_v2 --corpora synthetic,code `
  --methods flat,hier2,hier2_b256,hier3 --vocab-sizes 512 --seeds 42 `
  --out-dir results/fixed_smoke

# ablazione livelli, 3 seed, 3 vocabolari
python -m experiments.level_assignment --seeds 42,43,44 `
  --vocabs 512,1024,2048 --modes frequency,morph,random_closed,reverse `
  --out results/level_assignment.csv

# 10 seed a v512
python -m experiments.level_assignment --seeds 42,43,44,45,46,47,48,49,50,51 `
  --vocabs 512 --modes frequency,morph,random_closed `
  --out results/level_assignment_v512_10seeds.csv

# statistiche appaiate
python -m experiments.analyze_levels `
  --csv results/level_assignment_v512_10seeds.csv `
  --out results/level_assignment_stats_v512.csv
```
