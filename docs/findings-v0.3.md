# Findings v0.3 — il test decisivo, e il negativo

Questo documento chiude la Fase 1. Sostituisce `findings-v0.2.md` come lettura
dei risultati: la v0.2 aveva ancora il claim condizionato “morph migliora a
v512”; la v0.3 mostra che quel guadagno è riproducibile da uno slot
posizionale, e che il residuo non sopravvive a uno split pair-disjoint.

I numeri vengono da:

- `results/f1f_variable_triple.csv`, `results/f1f_variable_pair.csv`,
  `results/f1f_fixed_triple.csv`, `results/f1f_fixed_pair.csv` — 4 celle × 10
  seed × 7 modalità = 280 run;
- `results/f1f_analysis.txt`, `results/f1f_summary.csv` — delta appaiati;
- `results/f1f_split_check.txt` — integrità dello split su 10 seed.

Il criterio è stato scritto in `diario.md` (2026-11-24) **prima** di leggere i
CSV.

## 1. Il disegno

2 × 2: morfologia `{fixed, variable}` × split `{triple, pair}`, 10 seed
(42-51), v512, frazione L1 0,6.

- `fixed`: prefissi 2-3 caratteri, suffissi tutti di 2. È il generatore dei run
  precedenti.
- `variable`: prefissi di 1-4, suffissi di 1-3, inventari prefix-free e
  suffix-free. La posizione del confine dipende dal morfema.
- `triple`: holdout del 20% delle terne, con tutte le coppie di componenti
  attestate in training (interpolazione).
- `pair`: holdout del 10% delle coppie `(prefix, root)` e `(root, suffix)`.
  Ogni parola di test ha almeno una coppia non vista; le componenti restano
  attestate.

Controlli di confine, applicati alla **provenienza** dei merge e non alla
metrica:

- `const2`: taglio fisso `{2, L-2}`;
- `random2`: due tagli uniformi;
- `near2`: ogni confine vero spostato di −1/0/+1, con il caso (0,0) rifiutato.

## 2. I numeri

F1 macro, media su 10 seed:

| cella | freq | morph | const2 | near2 | random2 | random_closed |
|---|---:|---:|---:|---:|---:|---:|
| variable × triple | 0.8829 | **0.8922** | 0.8671 | 0.8835 | 0.8827 | 0.8696 |
| variable × pair | 0.8857 | **0.8928** | 0.8672 | 0.8885 | 0.8840 | 0.8690 |
| fixed × triple | 0.8742 | **0.8835** | 0.8835 | 0.8794 | 0.8826 | 0.8678 |
| fixed × pair | 0.8725 | **0.8763** | 0.8763 | 0.8748 | 0.8741 | 0.8594 |

Delta appaiati morph − controllo, con t a due code (10 seed, soglia 2,262):

| cella | vs frequency | vs const2 | vs near2 | vs random2 | bpc vs freq |
|---|---:|---:|---:|---:|---:|
| variable × triple | **+0.0094** (3.80) | **+0.0252** (5.51) | **+0.0087** (3.91) | **+0.0095** (3.61) | −0.0073 (−0.56) |
| variable × pair | +0.0071 (1.28) | **+0.0257** (5.02) | +0.0043 (0.81) | +0.0088 (1.26) | −0.0004 (−0.03) |
| fixed × triple | **+0.0092** (6.52) | 0.0000 (10 pareggi) | **+0.0040** (3.55) | +0.0009 (0.54) | **−0.0258** (−5.38) |
| fixed × pair | **+0.0038** (4.57) | 0.0000 (10 pareggi) | +0.0015 (2.23) | +0.0022 (2.21) | **−0.0099** (−2.46) |

Nella cella decisiva (variable × pair): morph−frequency 8/2/0,
morph−const2 10/0/0, morph−near2 7/3/0, morph−random2 7/3/0.

Lo split è pulito su 10 seed: `all_pairs_seen=0`, `missing_component=0`,
`dropped=0`, `dup=0`; test 4.372-4.399 parole, training 18.641-18.675. Sul
generatore variabile `const2` coincide con i confini veri nel 7,3-9,4% delle
parole; sul fisso nel 48,4-54,5%.

## 3. Interpretazione

1. **Il risultato storico si riproduce, ma è posizionale.** Sul generatore
   fisso, `const2` produce esattamente la stessa assegnazione di `morph` (10
   pareggi su 10 in entrambi gli split). Il +0.0092 F1 e il −0.026 bpc non
   richiedono l'identità dei morfemi: bastano due tagli fissi.
2. **La morfologia variabile toglie lo slot.** `const2` non è più una copia e
   peggiora frequency di 1,6-1,9 punti. Il generatore nuovo non è una variante
   cosmetica: cambia la natura del test.
3. **Sullo split triple resta un effetto di confine.** morph batte frequency,
   near2, random2 e const2, con 9/1 su frequency. Non è puramente posizionale:
   la posizione esatta dei confini porta informazione. L'effetto è piccolo
   (+0.009 F1) e non tocca il bpc.
4. **Sullo split pair il criterio fallisce.** Il punto stima è +0.0071, ma
   t = 1.28, CI95 [−0.0054, +0.0195]; e morph−near2 è +0.0043, t = 0.81. Non
   si distingue da zero né da una perturbazione near-true. Anche `near2` batte
   frequency (+0.0028, t 2.33): conta la regione del confine, non il punto
   esatto.
5. **Il bpc era un effetto dello slot.** Il guadagno −0.0258 non si replica con
   la morfologia variabile (−0.0073 sul triple, −0.0004 sul pair). Non c'è
   evidenza di compressione indipendente dall'allineamento.
6. **Lo split da solo non basta a spiegare il crollo.** fixed × pair conserva
   +0.0038 (t 4.57), ma `const2` lo riproduce esattamente: è ancora
   posizionale. È il generatore variabile a togliere l'artefatto; lo split
   toglie la significatività al residuo.

## 4. Il claim del paper

Il claim originale — “la tokenizzazione ramificata generalizza meglio sui
composti mai visti” — non è supportato. Il claim pubblicabile è metodologico:

> Una gerarchia BPE a partizione di budget è equivalente al BPE piatto. Su un
> corpus sintetico a slot fissi, l'assegnazione dei merge ai livelli per
> provenienza morfologica migliora boundary F1 e bpc, ma un taglio posizionale
> costante riproduce la stessa assegnazione: il guadagno non è evidenza
> morfologica. Con un generatore a lunghezze variabili resta un piccolo
> effetto di allineamento di confine sullo split di interpolazione (+0.9 F1),
> che non produce guadagno di compressione e non sopravvive a uno split
> pair-disjoint (non significativo a 10 seed; non distinguibile da un
> controllo near-true). La metrica è supervisionata dagli stessi confini che
> assegnano i livelli, quindi anche il residuo è allineamento supervisionato,
> non generalizzazione composizionale.

In una frase: **la gerarchia a budget è flat; l'effetto morfologico apparente è
uno slot fisso; il residuo è piccolo, non compressivo e non composizionale.**

## 5. Cosa è morto

- C0 — la gerarchia a partizione di budget è un no-op. Resta morta.
- C1 — l'assegnazione morfologica come vantaggio generale. Ridotta a effetto
  di allineamento in una finestra sintetica.
- Il bpc come evidenza di compressione. Secondario, e non replica.
- Il corpus codice come evidenza generale. Non è un corpus held-out.
- P5 come prossimo passo. Non c'è un effetto abbastanza solido da giustificare
  un tokenizer nuovo.
- Il red team come prossima ondata. Non c'è un claim positivo da attaccare.

Restano vivi: il bug F1 e la sua correzione; il null della gerarchia a budget;
la sanità dell'implementazione; il protocollo di split pair-disjoint; la lista
dei gradi di libertà che simulano l'effetto.

## 6. Limiti

- Lo split pair-disjoint tiene fuori coppie, non componenti: prefix, root e
  suffix restano attestati. È “coppie mai viste”, non zero-shot.
- Il corpus è sintetico e l'inventario è ricostruibile dalla stringa. Un corpus
  reale con confini verificati indipendentemente resta il test di validità
  esterna, non fatto.
- Con 10 seed il CI del delta su pair è largo ±1,2 punti F1. Il test è
  sottopotenza per un effetto di 0,7; il controllo near2 è ciò che rende
  impraticabile la strada B, non la sola significatività.
- Il bpc unigramma non è una prova di compressione.
- La metrica di boundary F1 usa gli stessi confini della provenienza. È il
  limite strutturale del disegno, non un bug.

## 7. Riproduzione

```powershell
# 2 x 2 completo, 10 seed, v512
python -m experiments.level_assignment --seeds 42,43,44,45,46,47,48,49,50,51 `
  --vocabs 512 --morphology variable --holdout-mode triple `
  --modes frequency,morph,random_closed `
  --boundary-controls const2,random2,near2 `
  --out results/f1f_variable_triple.csv

python -m experiments.level_assignment --seeds 42,43,44,45,46,47,48,49,50,51 `
  --vocabs 512 --morphology variable --holdout-mode pair `
  --modes frequency,morph,random_closed `
  --boundary-controls const2,random2,near2 `
  --out results/f1f_variable_pair.csv

# idem con --morphology fixed per le altre due celle

# integrità dello split
python -m experiments.split_check --seeds 42,43,44,45,46,47,48,49,50,51 `
  --morphologies fixed,variable --holdout-modes triple,pair

# delta appaiati
python -m experiments.analyze_f1f
```

## 8. Decisione

**Paper A, negativo/metodologico.** Non P5, non una corsa a 40 seed, non un
corpus reale usato come salvataggio. Se in futuro si vuole riaprire la
domanda, serve un test nuovo e non circolare: corpus reale con confini
verificati da un segmentatore indipendente, e l'effetto deve battere tagli
near-true. È un progetto separato, non la Fase 2 di questo.
