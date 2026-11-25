# Ramification-That-Does-Not-Generalize
Ramification That Does Not Generalize Hierarchical Tokenization, Positional Slots, and a Controlled Negative Result
# unibull — Ramified Tokenization

Ipotesi di lavoro: **il finito presenta l'infinito ramificato**. Un tokenizzatore
gerarchico con embedding composizionali dovrebbe generalizzare meglio sui
composti mai visti rispetto a BPE piatto, a parità di vocabolario e di
compressione. **La Fase 1 non sostiene l'ipotesi; la v0.3 dice perché.**

Questo repository è il banco di prova che ha deciso che il paper positivo non
c'è. La v0.1 ha misurato la struttura; la v0.2 ha corretto un bug critico del
training e ha mostrato che la gerarchia a partizione di budget è equivalente al
BPE piatto; la v0.3 ha mostrato che l'effetto morfologico apparente è uno slot
posizionale e che il residuo non sopravvive a uno split pair-disjoint. Il
diario di laboratorio è in [`diario.md`](diario.md); la sintesi corrente è
[`docs/findings-v0.3.md`](docs/findings-v0.3.md).

## Stato

- [x] Spina dorsale formale: `docs/ramified-v0.md`
- [x] Tokenizzatore BPE piatto + gerarchico: `ramified/bpe.py`
- [x] Corpus sintetico con ground truth morfologico: `ramified/synthetic.py`
- [x] Metriche: `ramified/metrics.py`
- [x] Primo sweep (10 agenti)
- [x] Codec a chunk con round-trip documentale: `ramified/codec.py`
- [x] Ablazione del budget di livello 1
- [x] Correzione dello smoothing (add-alpha 0.1) e del denominatore bpc
- [x] Review avversariale del BPE: trovato il bug F1 e corretto
- [x] Verifica del null: con il BPE corretto, flat = hier2 = hier3
- [x] Assegnazione morfologica ai livelli: +0.0092 F1 a v512 su 10/10 seed
- [x] Red team, prima ondata: l'effetto è posizionale, dipende da skew, root e frazione L1
- [x] Fase 1f: generatore a lunghezze variabili, split pair-disjoint, controlli const2/random2/near2 — **negativo**
- [ ] Corpus morfologico reale (validità esterna, non sulla linea critica)
- [ ] Embedding composizionali (Fase 2, non sulla linea critica)
- [ ] Canale steganografico (Fase 3)
- [x] Paper A: Markdown, LaTeX e PDF in `paper/` (`paper-it.*`, `paper-en.*`)

## Struttura

```
diario.md                 quaderno di laboratorio, una voce per fase
docs/
  ramified-v0.md          definizioni e ipotesi falsificabile
  literature.md           scan della letteratura (agente 9)
  findings-v0.1.md        numeri storici, prodotti con il bug F1 (non usare)
  findings-v0.2.md        bug, null, assegnazione morfologica e statistica
  findings-v0.3.md        Fase 1f: morfologia variabile, split pair-disjoint, negativo
  reviews/bpe-review.md   review avversariale del tokenizzatore
  embedding-experiment.md design Fase 2 (agente 10)
ramified/
  bpe.py                  BPE piatto e gerarchico, alberi di derivazione
  levels.py               assegnazione dei merge ai livelli per provenienza
  codec.py                codec a chunk, round-trip sul testo normalizzato
  synthetic.py            corpus morfologico sintetico con vettori ground truth
  metrics.py              bits/char, boundary F1, decode
  corpora.py              corpus di codice reale (stdlib Python)
experiments/
  run_matrix.py           matrice v0.1
  run_matrix_v2.py        matrice v0.2 (chunk codec, add-alpha, hier2_b256)
  ablate_level1.py        ablazione del budget di livello 1
  codec_compare.py        confronto codec a parità di denominatore
  verify.py               verifica indipendente di CSV, codec e leakage
  check_levels.py         verifica se la gerarchia è più di una partizione
  level_assignment.py     morph vs frequency vs random_closed vs reverse
  split_check.py          integrità del corpus e dello split pair-disjoint
  analyze_levels.py       statistiche appaiate per seed
  analyze_f1f.py          delta appaiati del 2x2 morfologia x split
  make_f1f_figure.py      Figura 1 del paper
  aggregate.py            unisce i CSV, media e deviazione standard
paper/
  paper-it.md             paper A, versione italiana (sorgente)
  paper-en.md             paper A, English version (source)
  paper-it.tex            versione LaTeX
  paper-en.tex            LaTeX version
  paper-it.pdf            PDF italiano, 18 pagine
  paper-en.pdf            PDF inglese, 17 pagine
  md2tex.py               convertitore Markdown -> LaTeX
  README.md               frontespizio, citazione, impronte dei dati
  figures/f1f_2x2.png     Figura 1: F1 per cella e differenze appaiate
results/
  agentXX_*/results.csv   output del primo sweep
  matrix/                 matrice v0.2 a 20 agenti: INVALIDA per il bug F1
  fixed_smoke/            smoke corretto, flat = hier2 = hier3
  level_assignment*.csv   ablazione morfologica e statistiche
  f1f_*.csv               2x2 Fase 1f: 4 celle, 10 seed, 7 modalità
  summary.csv             aggregato
```

## Come si esegue

Smoke corretto (tutti i metodi, v512):

```powershell
python -m experiments.run_matrix_v2 --corpora synthetic,code `
  --methods flat,hier2,hier2_b256,hier3 --vocab-sizes 512 --seeds 42 `
  --out-dir results/fixed_smoke
```

Ablazione morfologica (3 seed, 3 vocabolari):

```powershell
python -m experiments.level_assignment --seeds 42,43,44 `
  --vocabs 512,1024,2048 --modes frequency,morph,random_closed,reverse `
  --out results/level_assignment.csv
```

Confronto a 10 seed sul punto in cui il segnale esiste (v512):

```powershell
python -m experiments.level_assignment --seeds 42,43,44,45,46,47,48,49,50,51 `
  --vocabs 512 --modes frequency,morph,random_closed `
  --out results/level_assignment_v512_10seeds.csv
```

Statistiche appaiate:

```powershell
python -m experiments.analyze_levels `
  --csv results/level_assignment_v512_10seeds.csv `
  --out results/level_assignment_stats_v512.csv
```

Fase 1f (2x2 morfologia x split, una cella per comando):

```powershell
python -m experiments.level_assignment --seeds 42,43,44,45,46,47,48,49,50,51 `
  --vocabs 512 --morphology variable --holdout-mode pair `
  --modes frequency,morph,random_closed `
  --boundary-controls const2,random2,near2 `
  --out results/f1f_variable_pair.csv

python -m experiments.split_check --seeds 42,43,44,45,46,47,48,49,50,51 `
  --morphologies fixed,variable --holdout-modes triple,pair

python -m experiments.analyze_f1f
```

Confronto codec a parità di denominatore:

```powershell
python -m experiments.codec_compare --corpus code --vocab 512
```

## Metriche

| Metrica | Cosa misura | Direzione |
|---|---|---|
| `tokens_per_char` | token per carattere normalizzato | più basso è meglio |
| `bits_per_char` | cross-entropy unigramma sui token, add-alpha 0.1 | più basso è meglio |
| `boundary_f1` | allineamento token / confini morfologici veri | più alto è meglio |
| `decode_word` | `dec(enc(w)) == w` per parola | deve essere 1.0 |
| `decode_doc` | round-trip dell'intero documento normalizzato | deve essere 1.0 |

Il corpus sintetico è l'unico con ground truth morfologico: ogni parola è
`prefisso + radice + suffisso` e il vettore semantico è `P @ S @ r`. Due
generatori: `fixed` (slot 2-3/2, quello dei run v0.1-v0.2) e `variable`
(prefissi 1-4, suffissi 1-3, inventari prefix-free/suffix-free). Due split:
`triple` (20% delle terne, interpolazione) e `pair` (10% delle coppie
`(prefix, root)` e `(root, suffix)`; ogni parola di test ha almeno una coppia
non vista, componenti attestate).

`boundary_f1` misura l'allineamento su terne nuove con coppie note
(`triple`), non la generalizzazione composizionale; per quella serve `pair`.
**Esito Fase 1f:** sul generatore fisso `{2, L-2}` riproduce morph
esattamente, quindi il guadagno è posizionale. Con il generatore variabile
resta +0.9 F1 su `triple`, ma su `pair` il delta è +0.7 (t 1.28, non
significativo) e non si distingue dal controllo near-true. Dettagli in
[`docs/findings-v0.3.md`](docs/findings-v0.3.md).

**Nota sul denominatore e sul codec.** Il codec normalizza gli spazi
(`\s+` → un spazio) e lavora su quel testo. Il `bits_per_char` è quindi
calcolato sulla lunghezza normalizzata, cioè su ciò che il codec codifica
davvero. I numeri v0.1 sono ottimisti e non confrontabili direttamente; il
confronto va fatto a parità di denominatore. `decode_doc` è esatto solo sul
testo normalizzato: newline e indentazione non sono preservati.

## Protocollo

1. **Fase 1 — struttura.** BPE piatto vs gerarchico a parità di vocabolario.
   Criterio di morte: il gerarchico non deve essere peggiore su compressione e
   decodifica, e deve essere migliore su `boundary_f1`.
   **Esito:** la gerarchia a partizione di budget è equivalente al flat.
   L'assegnazione morfologica ai livelli dà +0.0092 F1 e −0.0258 bpc sul
   generatore a slot fissi, ma `{2, L-2}` riproduce la stessa assegnazione.
   Con morfologia a lunghezze variabili il residuo è +0.009 F1 su `triple`,
   non significativo su `pair` (+0.007, t 1.28) e non distinguibile da un
   controllo near-true; il bpc non replica. **Claim originale non supportato.**
2. **Fase 2 — significato.** Embedding composizionali sul derivation forest.
   Criterio: cosine similarity e retrieval sui composti held-out. **Non sulla
   linea critica:** il residuo del tokenizer non è abbastanza solido da
   giustificare il test. Resta un'ipotesi aperta, non il prossimo passo.
3. **Fase 3 — canale.** Non-iniettività dell'espansione come canale
   steganografico. Modello di attaccante, capacità, rilevabilità.
4. **Fase 4 — scrittura.** Paper A: negativo/metodologico. Il risultato è la
   caratterizzazione dei gradi di libertà che simulano l'effetto, non un
   vantaggio della ramificazione.

## Scala dello sciame

| Agenti | Fase | Cosa fanno |
|---|---|---|
| 10 | primo sweep | 6 run + 2 review + literature + design Fase 2 |
| 20 | matrice completa | 4 metodi × 5 vocabolari × 2 corpus × 3 seed = 120 righe; **invalidata dal bug F1** |
| 20 | red team, prima ondata | report su assunzioni, casi limite e attacchi; nessuna seconda ondata |

La regola resta: **prima i test, poi il paper**. Se un risultato è negativo, si
riporta e si riformula l'ipotesi; non si difende. La lezione della v0.2 è che
un risultato positivo può essere un bug che si nasconde: la review avversariale
è servita più di 20 run. La lezione della v0.3 è che può essere anche uno slot
fisso nel generatore.
