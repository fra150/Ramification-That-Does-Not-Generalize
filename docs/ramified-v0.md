# Ramified Tokenization — v0.1 (spina dorsale)

> Documento di lavoro. Non è un paper. È la definizione minima che rende
> falsificabile l'ipotesi: il finito presenta l'infinito ramificato, e la
> struttura di presentazione migliora la generalizzazione composizionale.

## 1. Presentazione finita

Sia \(A\) un alfabeto finito (caratteri, byte, code point Unicode usati come
terminali). Sia \(R\) un insieme finito di regole di raffinamento:

\[
R \subseteq A \times A^{*}, \qquad R \text{ finito.}
\]

Una **grammatica ramificata** è la coppia

\[
\mathcal{G} = (A, R).
\]

Un **simbolo** è un termine finito generato da \(A\) e \(R\). L'insieme dei
simboli finiti è l'algebra iniziale del funtore polinomiale associato a
\(\mathcal{G}\):

\[
S(\mathcal{G}) \;=\; \mu X.\; A + R(X).
\]

Lo **spazio ramificato completo** è la coalgebra finale:

\[
T(\mathcal{G}) \;=\; \nu X.\; A + R(X).
\]

Esiste un'unica mappa \(S(\mathcal{G}) \to T(\mathcal{G})\): il finito si
immerge nell'infinito e non lo esaurisce. Questo è il senso preciso di
«lavorare sul finito per generare l'infinito ramificato». Appena \(R\) contiene
una regola binaria, \(T(\mathcal{G})\) contiene uno spazio di Cantor e non è
numerabile; non lo enumeriamo, lo usiamo come spazio di riferimento.

## 2. Tokenizzatore

Un tokenizzatore è una coppia di mappe

\[
\mathrm{enc}: A^{*} \to S(\mathcal{G})^{*}, \qquad
\mathrm{dec}: S(\mathcal{G})^{*} \to A^{*}
\]

tali che \(\mathrm{dec}(\mathrm{enc}(x)) = x\) per ogni stringa \(x\) nel
dominio. Nel prototipo, \(\mathrm{enc}\) è un BPE gerarchico: i merge sono
organizzati in livelli, e ogni livello opera sui simboli prodotti dal livello
precedente. Il vocabolario è l'unione dei simboli di tutti i livelli.

## 3. Embedding composizionale

Un embedding è un omomorfismo dall'algebra dei simboli a uno spazio vettoriale:

\[
e: S(\mathcal{G}) \to \mathbb{R}^{d}, \qquad
e(r(a_1,\dots,a_k)) = g\big(e(a_1),\dots,e(a_k)\big).
\]

Per un simbolo mai visto, \(e\) è definito dalla sua struttura, non dalla sua
occorrenza. È questa la proprietà che un vocabolario piatto non ha.

## 4. Compressione

Il costo del modello è la dimensione della presentazione finita,
\(|\mathcal{G}|\), non il numero di token. La compressione si misura in
bit per carattere con un modello unigramma sui token, a parità di vocabolario.

## 5. Canale

Se l'espansione \(S(\mathcal{G}) \to A^{*}\) non è iniettiva, più percorsi
collassano sulla stessa stringa. La scelta del percorso trasporta informazione:
è un canale steganografico, non un cifrario. Capacità e rilevabilità sono
misurabili. Questo resta un cassetto separato.

## 6. Ipotesi falsificabile (v0.1)

> A parità di vocabolario e di compressione, un tokenizzatore ramificato
> produce una segmentazione più allineata ai morfemi e un embedding
> composizionale migliore sui composti mai visti, rispetto a BPE piatto.

Misure:

1. **Allineamento morfologico**: precision, recall, F1 dei confini di token
   rispetto ai confini morfologici veri, su parole tenute fuori dal training.
2. **Compressione**: bit per carattere su testo di test, a parità di
   vocabolario.
3. **Decodifica esatta**: \(\mathrm{dec}(\mathrm{enc}(x)) = x\).
4. **Costo**: tokens per carattere, tempo di training.

Criterio di morte della v0.1: se il tokenizzatore ramificato non è almeno
comparabile a BPE piatto su compressione e decodifica, e non è migliore
sull'allineamento morfologico, l'ipotesi va riformulata.

## 7. Nota sui cinque pezzi

Unicode è \(A\). La matematica ramificata è \(R\) e la chiusura \(S\). I token
sono il parsing in \(S\). Gli embedding sono l'omomorfismo \(e\). La
crittografia/steganografia è la non-iniettività di \(S \to A^{*}\). Cinque
proiezioni di un solo oggetto; gli esperimenti restano sequenziali.
