# Literature scan — ramified tokenization (v0.1)

**Scan date:** 22 November 2026  
**Target document:** [`docs/ramified-v0.md`](ramified-v0.md)  
**Method:** targeted web, arXiv and ACL Anthology searches; not a systematic review.  
**Scope:** 25 primary works across the five areas named in the brief, plus a small number of adjacent works mentioned in prose. Recent 2025–2026 items are identified as preprints where their venue is not confirmed.

**Labels used below**

- **Supports** — gives evidence or a conceptual argument for a component of the hypothesis.
- **Overlaps** — already does part of what the proposal claims, usually under another name; reduces novelty.
- **Undermines** — gives evidence against a premise, or makes the claimed advantage less plausible.
- **Background** — standard result or tool; relevant to the framing, neither for nor against.

---

## 0. Sintesi (per il progetto)

La scansione dice tre cose. Primo, il tokenizzatore proposto è formalmente una
grammatica *straight-line* (SLP) costruita da BPE multi-livello: la letteratura
su Re-Pair, Sequitur e BPE come compressione grammaticale è direttamente
pertinente e toglie novità all'idea «tokenizzazione = grammatica». Secondo, la
coppia algebra iniziale / coalgebra finale è il teorema standard dei termini
finiti contro alberi finiti e infiniti (Goguen; Rutten; Jacobs;
Adámek–Milius–Moss): è matematica corretta ma da manuale, e nella v0.1 la
coalgebra finale non ha ruolo empirico. Terzo, l'embedding come omomorfismo è
la semantica composizionale standard (Montague/DisCoCat, Tree-LSTM, LEAR);
l'uso della merge tree del tokenizzatore come schema di ricorsione è già oggetto
di un preprint del 2026 (Bass). La parte potenzialmente nuova resta l'estensione
multi-livello allineata alla morfologia, con un confronto a parità di
vocabolario e compressione. La tesi morfologica è già in parte dimostrata da
MorphBPE e TreeTokenizer; il salto da allineamento morfologico a
generalizzazione composizionale è la parte più fragile, anche perché Arnett et
al. (2025) trovano che l'allineamento spiega poco delle prestazioni downstream.

**Hypothesis restated (for the labels below).** H1: at equal vocabulary and
equal compression, a ramified (multi-level grammar) tokenizer produces a
segmentation more aligned with morphological boundaries than flat BPE. H2: at
equal vocabulary and compression, the grammar-recursive embedding gives better
compositional representations of unseen compounds than a flat BPE embedding.
The v0.1 death criterion is failure on both.

---

## 1. Bottom line

1. **The tokenizer is not new as a mechanism.** A finite acyclic grammar with
   rules `X -> ...` is a straight-line program / SLP; BPE is a greedy
   pair-replacement grammar compressor (Gage 1994), with Re-Pair as its
   better-analyzed offline sibling; multi-level BPE already exists as
   "hierarchical BPE" and "SuperBPE". A 2026 preprint (Bass) already claims the
   coupling of the merge tree to the embedding, at least for rare-token
   embeddings. The remaining space is the multi-level, morphology-aligned
   version plus a matched-vocabulary/matched-compression falsifiable test.
2. **The algebra/coalgebra pair is textbook.** For a finitary signature,
   the initial algebra is the set of finite terms and the terminal coalgebra is
   the set of finite and infinite terms. The unique map between them is the
   inclusion. The final coalgebra is not used by any experiment in v0.1.
3. **The embedding homomorphism is standard compositional semantics.** It is
   the Montague/DisCoCat idea, the Tree-LSTM/recursive-network idea, and
   explicitly the LEAR "homomorphism between a syntactic algebra and a semantic
   algebra" idea. The choice of the tokenizer's merge grammar as the syntactic
   algebra is also claimed by Bass (2026) for rare-token embeddings; what is not
   yet shown is that the *multi-level* grammar improves morphological alignment
   and unseen-compound composition.
4. **The empirical claim is plausible but partly anticipated.** Morphology-aware
   tokenizers already improve morphological alignment (MorphBPE, TreeTokenizer,
   Morfessor). The stronger claim — that this yields better compositional
   embeddings on unseen compounds at equal vocabulary and compression — is not
   established, and the main available evidence on alignment as a predictor of
   downstream performance is negative or weak (Arnett et al. 2025).
5. **The weakest methodological link is the benchmark.** SCAN and CFQ measure
   syntactic/rule composition, not morphological composition of unseen
   compounds. If the hypothesis is about compounds, it needs a compound-specific
   split (e.g. CompoundPiece-style hard compounds), not SCAN.

---

## 2. The 25 works

### 2.1 Hierarchical / multi-level BPE and tokenization

**1. Sennrich, R., Haddow, B., & Birch, A. (2016).** *Neural Machine Translation of Rare Words with Subword Units.* ACL 2016.
- **Contribution:** Imports byte-pair encoding into NLP as a subword tokenizer for open-vocabulary translation; the flat-BPE baseline against which the hypothesis is stated.
- **Relation:** **Supports** the need for subword composition; neutral on hierarchy. This is the baseline, not prior art for ramification.

**2. Zouhar, V., Meister, C., Gastaldi, J. L., Du, L., Vieira, T., Sachan, M., & Cotterell, R. (2023).** *A Formal Perspective on Byte-Pair Encoding.* Findings of ACL 2023, 598–614. arXiv:2306.16837.
- **Contribution:** Formalizes BPE as a merge-sequence compression problem, gives an approximation guarantee for a submodular compression objective, and explicitly situates BPE in grammar-based compression.
- **Relation:** **Overlaps strongly / undermines novelty.** The identification "BPE is a restricted grammar compressor" is already in the literature. The paper does not propose a multi-level morphology-aware tokenizer, but it removes any claim that the compression framing is new.

**3. Whittington, P., Bachmann, G., & Pimentel, T. (2025).** *Tokenisation is NP-Complete.* ACL 2025; arXiv:2412.15210 (2024).
- **Contribution:** Proves that finding an optimal vocabulary, or an optimal sequence of merge operations, under a symbol budget is NP-complete.
- **Relation:** **Undermines** any claim of optimal compression. It supports the use of greedy/heuristic grammars, but it also means "the grammar is the model" cannot be sold as an optimality result. Kozma et al. (2024/2026) sharpen this to APX-completeness for pair encoding and give BPE a constant-factor guarantee.

**4. Dolga, R., Maystre, L., Berariu, T., & Barber, D. (2025).** *From Characters to Tokens: Dynamic Grouping with Hierarchical BPE.* Findings of EMNLP 2025, 11154–11162. arXiv:2510.15517.
- **Contribution:** A two-level BPE: first-level BPE tokens become character patches with end-of-patch markers; a second BPE stage compresses patches; matches or beats entropy- and whitespace-based dynamic patching.
- **Relation:** **Overlaps strongly.** This is hierarchical BPE in the same design space as the project's `hier2`/`hier3`. It does not tie levels to morphology or define embeddings recursively, but the tokenizer mechanism is a sibling, not a new species.

**5. Liu, A., Hayase, J., Hofmann, V., Oh, S., Smith, N. A., & Choi, Y. (2025).** *SuperBPE: Space Travel for Language Models.* COLM 2025; arXiv:2503.13423.
- **Contribution:** A two-stage BPE curriculum that first learns subwords and then superwords crossing whitespace; at fixed vocabulary and compute, 33% fewer tokens and +4.0 average downstream across 30 tasks.
- **Relation:** **Overlaps.** It demonstrates that multi-stage BPE can improve efficiency and downstream performance, but the stages are not morphology-aligned and there is no compositional embedding argument. It is the strongest recent evidence that "more levels" can help; it is not evidence for the ramified hypothesis specifically.

### 2.2 Morphology and morphological alignment

**6. Creutz, M., & Lagus, K. (2002).** *Unsupervised Discovery of Morphemes.* ACL-02 Workshop on Morphological and Phonological Learning, 21–30.
- **Contribution:** Morfessor, an MDL-based unsupervised morpheme segmentation model; the canonical morphology-aware alternative to BPE.
- **Relation:** **Overlaps.** It establishes both the objective (compression/MDL) and the baseline (unsupervised morphology) that the ramified tokenizer is implicitly reinventing. Any claim that compression can induce morpheme-like units must engage with Morfessor.

**7. Zhu, Q., Hu, X., Ji, P., Wu, W., & Tu, K. (2025).** *Unsupervised Morphological Tree Tokenizer.* Findings of ACL 2025; arXiv:2406.15245.
- **Contribution:** Induces character-level morphological trees with a self-supervised model and tokenizes words top-down by vocabulary matching; outperforms BPE and WordPiece on morphological segmentation and language-modeling tasks.
- **Relation:** **Supports H1 / undermines novelty.** Tree-structured tokenization aligned with morphology is already implemented and evaluated. The project's contribution cannot be "tree tokenization improves morphological alignment" *tout court*; it must be the specific grammar-recursive embedding and the matched comparison.

**8. Asgari, E., El Kheir, Y., & Sadraei Javaheri, M. A. (2025).** *MorphBPE: A Morpho-Aware Tokenizer Bridging Linguistic Complexity for Efficient LLM Training Across Morphologies.* arXiv:2502.00894 (preprint).
- **Contribution:** Constrains BPE merges to respect morpheme boundaries and proposes morphological consistency F1 and morphological edit distance; improves alignment and convergence on English, Russian, Hungarian and Arabic at 300M and 1B parameters.
- **Relation:** **Supports H1 / undermines novelty.** The hypothesis's first empirical claim — better morphological alignment than flat BPE — is already demonstrated by a morphology-constrained BPE. The correct baseline for the project is MorphBPE or TreeTokenizer, not only flat BPE.

**9. Arnett, C., Hudspeth, M., & O'Connor, B. (2025).** *Evaluating Morphological Alignment of Tokenizers in 70 Languages.* ICML 2025 TokShop; arXiv:2507.06378.
- **Contribution:** Extends MorphScore to 70 languages and correlates alignment with downstream performance for five pretrained models on seven tasks; finds that morphological alignment explains little variance in performance.
- **Relation:** **Undermines the H1 → H2 bridge.** Even if the ramified tokenizer improves boundary F1, that improvement may not translate into better compositional embeddings or downstream behavior. This is the most important negative result for the project's causal story.

### 2.3 Byte-level and dynamic tokenization

**10. Xue, L., Barua, A., Constant, N., Al-Rfou, R., Narang, S., Kale, M., Roberts, A., & Raffel, C. (2022).** *ByT5: Towards a Token-Free Future with Pre-trained Byte-to-Byte Models.* TACL 10:291–306.
- **Contribution:** A byte-level T5 that removes subword tokenization; competitive with tokenized models on many tasks, with advantages on character-level, orthographic and multilingual tasks.
- **Relation:** **Overlaps / undermines the necessity of morphology-aware tokenization.** If a byte-level model already handles morphology and spelling without an explicit morpheme tokenizer, the marginal value of a ramified tokenizer must be shown, not assumed.

**11. Pagnoni, A., Pasunuru, R., Rodriguez, P., Nguyen, J., Muller, B., Li, M., Zhou, C., Yu, L., Weston, J., Zettlemoyer, L., Ghosh, G., Lewis, M., Holtzman, A., & Iyer, S. (2025).** *Byte Latent Transformer: Patches Scale Better Than Tokens.* ACL 2025; arXiv:2412.09871.
- **Contribution:** Dynamically segments bytes into entropy-based patches and matches tokenized LLM performance at scale with better inference efficiency and robustness.
- **Relation:** **Overlaps / undermines.** Dynamic, non-morphological patching is a strong alternative to a fixed hierarchical vocabulary. BLT shows that the segmentation criterion can be information-theoretic rather than morphological, which weakens the claim that morphology is the right level of ramification.

**12. Feher, D., Vulić, I., & Minixhofer, B. (2025).** *Retrofitting Large Language Models with Dynamic Tokenization.* ACL 2025, 29866–29883.
- **Contribution:** Dynamically merges frequent subword sequences per batch and predicts token embeddings on the fly with a hypernetwork; reduces sequence length by >20% across 14 languages with <2% degradation.
- **Relation:** **Overlaps.** Dynamic tokenization is already practical and deployed as a retrofit. The project's static multi-level vocabulary is one point in this design space; the burden is to show that grammar levels add something beyond dynamic merge frequency.

### 2.4 Grammar-based compression

**13. Nevill-Manning, C. G., & Witten, I. H. (1997).** *Identifying Hierarchical Structure in Sequences: A Linear-Time Algorithm.* Journal of Artificial Intelligence Research 7:67–82.
- **Contribution:** Sequitur, an online grammar compressor that maintains digram uniqueness and rule utility and builds a hierarchical grammar from a sequence.
- **Relation:** **Overlaps strongly.** Hierarchical grammar induction from a sequence is exactly the kind of object the proposal calls a ramified grammar. Sequitur is the canonical algorithm and already provides a "hierarchy of rules" view.

**14. Larsson, N. J., & Moffat, A. (2000).** *Off-line Dictionary-Based Compression.* Proceedings of the IEEE 88(11):1722–1732.
- **Contribution:** Re-Pair, greedy replacement of the most frequent pair, with linear-time implementations and a principled stopping rule; a more sophisticated offline sibling of BPE.
- **Relation:** **Overlaps strongly / undermines novelty.** BPE (Gage 1994) and Re-Pair are the same greedy pair-replacement idea; Re-Pair adds a stopping rule and a sharper analysis, while BPE fixes the merge budget. The project's grammar is in this family; the "ramified" tokenizer is a multi-level Re-Pair/BPE, not a new compression principle.

**15. Charikar, M., Lehman, E., Liu, D., Panigrahy, R., Prabhakaran, M., Sahai, A., & Shelat, A. (2005).** *The Smallest Grammar Problem.* IEEE Transactions on Information Theory 51(7):2554–2576.
- **Contribution:** Proves the smallest grammar problem is NP-hard, gives approximation lower bounds, and analyzes Sequitur, Re-Pair, LZ78 and other grammar compressors.
- **Relation:** **Undermines optimality claims.** It supports the use of greedy heuristics, but "the cost of the model is the size of the grammar" is not a computable optimum in general. Bannai et al. (2021, IEEE TIT 67(1):317–328) sharpen several of these bounds.

**16. De Marcken, C. (1996).** *Unsupervised Language Acquisition.* PhD thesis, MIT.
- **Contribution:** MDL-based unsupervised grammar and morphology induction; treats a grammar as a compact model of the corpus and the lexicon as part of the code.
- **Relation:** **Overlaps.** "Grammar as model, compression as objective" is the founding idea of the MDL grammar-induction tradition. The project's compression claim is a modern tokenizer instance of it, not a new theoretical move.

### 2.5 Compositional generalization and compositional embeddings

**17. Lake, B. M., & Baroni, M. (2018).** *Generalization without Systematicity: On the Compositional Skills of Sequence-to-Sequence Recurrent Networks.* ICML 2018.
- **Contribution:** Introduces SCAN and shows that standard seq2seq models fail on systematic splits (e.g. add-jump), exposing poor compositional generalization.
- **Relation:** **Supports the motivation**, but does not test morphology or tokenization. SCAN's primitives are words/commands, not morphemes; success on SCAN would not establish the ramified hypothesis.

**18. Bass, L. (2026).** *Merge-Tree Compositional Embeddings: The Tokenizer Already Knows What Rare Tokens Are Made Of.* Preprint, ResearchGate, October 2026. DOI:10.13140/RG.2.2.16236.53127.
- **Contribution:** Observes that every BPE token is the concatenation of two existing tokens, yet language models give each token an independent embedding; proposes computing rare-token embeddings from the tokenizer's own merge tree, and quantifies the consequences in a low-resource multilingual setting. (Full text was not independently accessible in this scan; this characterization is from the public abstract and metadata.)
- **Relation:** **Overlaps very strongly / undermines core novelty.** The public abstract describes essentially the proposal's central embedding move — use the tokenizer's merge tree as the recursion scheme — applied to rare-token embeddings. It is a non-peer-reviewed preprint and it does not contain the multi-level morphology/compression hypothesis, but it means "the tokenizer already knows what tokens are made of" is not a new idea. It is dated October 2026, close to the project's v0.1; check the project's own timestamp for priority. The technical overlap is direct. The project must cite it and differentiate on the multi-level, morphology-aligned, falsifiable-comparison axis.

**19. Tai, K. S., Socher, R., & Manning, C. D. (2015).** *Improved Semantic Representations From Tree-Structured Long Short-Term Memory Networks.* ACL 2015.
- **Contribution:** Tree-LSTM, a recursive neural network that composes child representations along a syntactic tree.
- **Relation:** **Overlaps.** The project's embedding homomorphism `e(r(a1..ak)) = g(e(a1)..e(ak))` is the same recursive-composition principle. The only difference is the source of the tree: the tokenizer grammar rather than a linguistic parse.

**20. Liu, C., An, S., Lin, Z., Liu, Q., Chen, B., Lou, J.-G., Wen, L., Zheng, N., & Zhang, D. (2021).** *Learning Algebraic Recombination for Compositional Generalization.* Findings of ACL-IJCNLP 2021, 1129–1144.
- **Contribution:** LEAR models semantic parsing as a homomorphism between a latent syntactic algebra and a semantic algebra, with a Tree-LSTM composer and an interpreter for semantic operations.
- **Relation:** **Overlaps very strongly.** The project's `e: S(G) -> R^d` with a recursive composition function is a special case of the LEAR/DisCoCat algebraic-homomorphism view. The novelty must be the grammar's role as tokenizer, not the homomorphism itself.

**21. Reif, Y., Kaplan, G., & Schwartz, R. (2026).** *More Than Words: Compositional Tokenization for Efficient Language Models.* COLM 2026; arXiv:2610.05597.
- **Contribution:** CoBPE represents phrases as a lexical base token plus reusable surface modifiers, composes them in embedding space at input, and predicts them jointly at output; 30% shorter sequences and +1.2 downstream at 780M/1.3B under matched compute.
- **Relation:** **Overlaps strongly.** Compositional tokenization plus embedding-space composition is already a working recipe. It does not use a grammar hierarchy or morphology, but it occupies the conceptual slot "tokenization can be compositional and improve efficiency/performance". The project needs to differentiate on the grammar-recursive, morphology-aligned axis.

### 2.6 Initial algebra, final coalgebra, and tree algebras

**22. Goguen, J. A., Thatcher, J. W., Wagner, E. G., & Wright, J. B. (1977).** *Initial Algebra Semantics and Continuous Algebras.* Journal of the ACM 24(1):68–95.
- **Contribution:** Initial algebra semantics for abstract data types and programming languages; the canonical source for "syntax is an initial algebra".
- **Relation:** **Background / supports the formalism.** The project's `S(G) = mu X. A + R(X)` is an instance of this tradition. The paper does not discuss tokenization; it establishes that the move is standard.

**23. Rutten, J. J. M. M. (2000).** *Universal Coalgebra: A Theory of Systems.* Theoretical Computer Science 249(1):3–80.
- **Contribution:** Universal coalgebra; final coalgebras as spaces of behaviours/observations, with coinduction and bisimulation as the dual of induction and congruence.
- **Relation:** **Background / supports the formalism.** The project's `T(G) = nu X. A + R(X)` is a standard final coalgebra. Rutten supplies the general theory, not a tokenization-specific result.

**24. Jacobs, B. (2016).** *Introduction to Coalgebra: Towards Mathematics of States and Observation.* Cambridge University Press.
- **Contribution:** Textbook treatment of coalgebra, the duality with algebra, final coalgebras and coinductive definitions.
- **Relation:** **Background.** Confirms that the algebra/coalgebra duality used in v0.1 is textbook material. There is no novelty in stating that finite terms embed into a final coalgebra of finite and infinite terms.

**25. Adámek, J., Milius, S., & Moss, L. S. (2018).** *Fixed Points of Functors.* Journal of Logical and Algebraic Methods in Programming 95:41–81.
- **Contribution:** For a finitary signature `Sigma`, proves that `mu H_Sigma` is the algebra of finite `Sigma`-trees and `nu H_Sigma` is the coalgebra of all finite and infinite `Sigma`-trees.
- **Relation:** **Directly supports the math / directly undermines mathematical novelty.** This is exactly the theorem the v0.1 document invokes. The unique map `S(G) -> T(G)` is the inclusion of finite trees into finite+infinite trees. It is correct, but it is a known instance of a general result, not a new contribution.

---

## 3. What the literature actually establishes

### 3.1 The tokenizer is an SLP, and BPE is already a grammar compressor

A finite set of rules `X -> w` with acyclic dependencies is a straight-line
program (SLP), also called a straight-line grammar or a macro scheme. BPE
(Gage 1994) produces exactly such a grammar: each merge introduces a new
nonterminal whose right-hand side is a pair, and the merge order gives the
dependency order. Re-Pair is the same greedy idea with a stopping rule instead
of a fixed vocabulary budget; Sequitur is the online version with digram
uniqueness. The smallest grammar problem is NP-hard, so every practical method
is a heuristic. This is the established view in Zouhar et al. (2023), Charikar
et al. (2005), Larsson & Moffat (2000) and Nevill-Manning & Witten (1997). For
the SLP side specifically, Rytter (2003, *Theoretical Computer Science*
302:211–222) relates SLPs to LZ77 factorization and gives approximation
guarantees; Bannai et al. (2021) give the current bounds for Re-Pair and LZ78.

Consequence for the project: the phrase "ramified grammar" describes a known
object. The multi-level BPE implementation is a configuration of a known design
space. What is not known is whether the *specific* level structure — levels
chosen to track morphological composition — buys anything over a flat BPE with
the same vocabulary and merge budget. That is an empirical question, and it is
the only defensible reading of the hypothesis.

Two caveats on the compression axis. Bostrom & Durrett (2020, Findings of
EMNLP) show that BPE is suboptimal for language-model pretraining relative to
unigram tokenization, and Schmidt et al. (2024, EMNLP) show that compression
alone does not predict downstream quality. So "equal compression versus flat
BPE" is a necessary control, not a sufficient measure of tokenizer quality. A
ramified tokenizer that merely matches BPE on bpc has not shown that it is a
better tokenizer; it has shown that it is not worse on one axis.

A very recent preprint goes further in the same direction: Kemal & Lieck (2026,
arXiv:2610.26695) train a hierarchical token-merging autoencoder that
recursively combines adjacent tokens and marginalizes over binary trees, and
show that compression alone can induce parse-like structure on a synthetic
ambiguous grammar (93% parse validity, 93% exact reconstruction). It is grammar
induction rather than language-model tokenization, but it is the closest
existing proof of concept for "hierarchical token merging learns structure from
compression", and the project should cite and differentiate it.

### 3.2 The algebra/coalgebra pair is the standard signature theorem

For a finitary signature functor `H_Sigma(X) = coprod_n Sigma_n x X^n`, the
initial algebra is the set of finite `Sigma`-trees and the terminal coalgebra is
the set of finite and infinite `Sigma`-trees. Adámek, Milius & Moss (2018)
state this explicitly. The unique map from the initial algebra to the terminal
coalgebra is the inclusion. The Cantor-space remark is true only when some node
has at least two available constructors; a single binary rule over a singleton
alphabet gives a one-point terminal coalgebra, not a Cantor space. For the
tree-language side, the standard reference is Comon et al. (2007), *Tree
Automata Techniques and Applications*; the project does not use recognizable
tree languages, but that is the mature "tree algebra" tradition adjacent to
this construction.

Consequence for the project: the final coalgebra is a correct reference space,
but it does no empirical work in v0.1. No experiment uses `T(G)`, and the
hypothesis is entirely about `S(G)`. The coalgebra should be presented as
framing, not as a result. If the project wants the coalgebra to matter, it needs
an operational use — e.g. a prior over infinite continuations, a coinductive
definition of a language model, or a bisimulation-based equivalence of
tokenizations — and that is a separate paper.

### 3.3 The embedding homomorphism is standard compositional semantics

`e(r(a1,...,ak)) = g(e(a1),...,e(ak))` is the definition of a compositional
representation. It is the Montague/DisCoCat tradition in categorical semantics
(Coecke, Sadrzadeh & Clark 2010), the recursive-neural-network tradition
(Socher et al. 2013; Tai et al. 2015), and the explicit "syntactic algebra to
semantic algebra" homomorphism of LEAR (Liu et al. 2021). Subword-compositional
word embeddings for OOV words are also well established: fastText (Bojanowski
et al. 2017), Mimick (Pinter et al. 2017) and Charagram (Wieting et al. 2016)
all compute a word vector from its subword units.

Consequence for the project: "unseen symbol has an embedding determined by its
structure" is not new. What could be new is that the structure is the
*tokenizer's own merge grammar*, so the embedding and the segmentation are
derived from one object rather than from a separate parser or a separate
character n-gram model. That is a real design difference, but it is a
difference in the choice of the algebra, not in the principle of compositional
embedding.

### 3.4 Morphological alignment is not a sufficient endpoint

MorphBPE and TreeTokenizer already show that morphology-aware tokenization
improves morphological alignment and can improve language-modeling metrics.
Morfessor shows that MDL/compression can induce morpheme-like units without
supervision. So the first half of the hypothesis — "a ramified tokenizer is
more morphologically aligned than flat BPE" — is plausible and partly
anticipated.

The second half is the problem. Arnett et al. (2025) find that morphological
alignment explains little variance in downstream performance across five
models and seven tasks. That does not prove alignment is useless, but it means
alignment cannot be used as a proxy for "better compositional embedding"
without an independent downstream test. The project's own findings file
(`docs/findings-v0.1.md`) already shows a non-monotonic boundary F1
(0.79 → 0.89 → 0.82 across vocabulary sizes), which is a warning that the metric
is sensitive to the level-1 budget. A stronger endpoint would be a held-out
compound-composition task, not boundary F1 alone.

### 3.5 Compositional generalization benchmarks do not measure compound morphology

SCAN (Lake & Baroni 2018) and CFQ (Keysers et al. 2020) test the recombination
of known primitives into novel syntactic structures. They do not test unseen
morphological compounds, and they do not vary the tokenizer. COGS (Kim & Linzen
2020) is similar: it tests syntactic rule composition, not morpheme
composition. Hupkes et al. (2020) decompose compositionality into five tests,
but none of them is a morphological-compounding test.

If the hypothesis is about "composti mai visti", the relevant evaluation is a
compound split with constituent-held-out items, in the spirit of CompoundPiece
(Minixhofer, Pfeiffer & Vulić 2023), which defines "hard compounds" relative to
a tokenizer and evaluates decompounding. A SCAN result would be interesting
for the embedding recursion, but it would not test the morphological claim.

### 3.6 Byte-level and dynamic tokenization raise the bar

ByT5, BLT and dynamic-tokenization retrofits show that competitive language
models can be built without a fixed morphological vocabulary. BLT's patches are
entropy-based; SpaceByte's patches are whitespace-aligned; Feher et al. merge
frequent subword sequences dynamically. None of these uses a grammar hierarchy
tied to morphology, and they already achieve strong efficiency and robustness.
This does not refute the ramified hypothesis, but it means the burden of proof
is higher: the ramified tokenizer must beat not only flat BPE but also a
dynamic, non-morphological segmentation at matched compute. The open-vocabulary
modeling tradition is surveyed by Mielke et al. (2021, arXiv:2112.10508); the
project's "open vocabulary" framing should cite it rather than present
open-vocabulary tokenization as new.

---

## 4. Formal caveats in v0.1 (not literature)

These are not objections from the literature; they are places where the v0.1
document's formalism should be tightened before it is used in a paper.

1. **The functor needs arities.** `R` is written as a subset of `A x A*`, but
   `A + R(X)` is only a polynomial functor if `R` is a signature with a fixed
   arity for each rule. The correct form is
   `A + coprod_{r in R} X^{|rhs(r)|}`. If rules may share a left-hand side,
   the object is a CFG/regular-tree grammar, not an SLP. To match the BPE
   implementation, use `R` subset of `N x (A union N)*` with acyclic
   dependencies, or state explicitly that `A` is the set of all symbols.
2. **The Cantor claim is too strong as stated.** `nu X. A + R(X)` is a Cantor
   space only if some node has at least two available constructors (e.g.
   `|A| >= 2`, or two rules of the same arity). A single binary rule over a
   singleton alphabet gives a singleton terminal coalgebra.
3. **The unique map is the inclusion.** `S(G) -> T(G)` is the unique algebra
   morphism from the initial algebra to the terminal coalgebra; for a signature
   functor it is the inclusion of finite terms into finite+infinite terms. It is
   not a special property of the tokenizer, and it is not used experimentally.
4. **Injectivity is about `S(G)* -> A*`, not `S(G) -> A*`.** The doc says "the
   expansion `S(G) -> A*`" but the channel is the monoid homomorphism
   `S(G)* -> A*`. Non-injectivity means distinct token sequences decode to the
   same string (segmentation ambiguity), not that distinct symbols expand to
   the same string. Standard BPE's symbol map is injective if the vocabulary has
   no duplicate strings; the ambiguity is in the segmentation, i.e. in `enc`.
5. **Compression accounting is incomplete.** The model cost is not only `|G|`:
   the embedding function `e` has parameters. If `e` is a lookup table, the
   table is `|S(G)| x d`; the homomorphism is supposed to avoid that table, but
   then the cost of `g` and the leaf embeddings must be counted. The unigram
   token bpc is a weak proxy; a neural LM or a hierarchical likelihood is a
   stronger test, and it is the standard in the BPE-compression literature.
6. **The "equal vocabulary and equal compression" constraint is two-dimensional.**
   In practice there is a Pareto frontier: a larger vocabulary reduces tokens
   per character but increases the embedding table and changes bpc. The
   hypothesis should specify the matching protocol (same vocabulary size? same
   bpc? same total parameter count? same training FLOPs?) before the experiment
   is run, otherwise the comparison is post hoc.

---

## 5. What would make the hypothesis genuinely novel

The literature suggests four ways to turn the current proposal into something
that is not already covered:

1. **One grammar, four roles.** Bass (2026) already couples the tokenizer and
   the embedding for rare tokens. The remaining coupling is to add the
   compression model and the ambiguity channel as measurable roles, and to show
   that the *multi-level, morphology-aligned* grammar beats a flat merge tree on
   a held-out compound task. The individual roles are known; the four-role,
   multi-level version is not yet demonstrated.
2. **A morphology-held-out compound benchmark.** Build or adopt a compound
   split with constituent-held-out items and compare flat BPE, MorphBPE,
   TreeTokenizer, CoBPE and the ramified tokenizer at matched vocabulary,
   matched bpc and matched parameter count. This is the experiment the current
   hypothesis actually needs.
3. **Ablate the grammar, not just the number of levels.** The current `hier2`
   vs `hier3` comparison is a comparison of level counts. A stronger ablation
   holds the vocabulary and merge budget fixed and varies only the *level
   assignment* of merges (morpheme-first vs frequency-first). If the level
   assignment does not matter, the "ramified" part is decorative.
4. **Give the final coalgebra an operational role.** For example, define a
   coinductive prior over infinite token continuations, or use bisimulation to
   quotient tokenizations that are behaviorally equivalent for a downstream
   model. Without such a role, `T(G)` is a mathematically correct but
   empirically idle reference space.

---

## 6. Novelty assessment

The proposal's central move — use the tokenizer's own merge tree as the
recursion scheme for token embeddings — is anticipated by a 2026 preprint. The
public abstract of Bass (2026) describes exactly that: BPE tokens receive
independent embeddings despite being concatenations of existing tokens, and
rare-token embeddings can be computed from the merge tree. If the preprint's
claims hold, this is the v0.1 embedding move in a low-resource multilingual
setting. The rest of the proposal is a package around that move: a multi-level
grammar, a morphology-alignment objective, a compression model, an ambiguity
channel, and a falsifiable comparison against flat BPE. The package is a
legitimate, narrow contribution, but the document should not present the
merge-tree embedding idea as its own.

Under their own names, the pieces are: the grammar is a **straight-line
grammar / SLP / macro scheme**, and BPE and Re-Pair are greedy pair-replacement
grammar compressors, with Re-Pair the better-analyzed offline sibling (Zouhar
et al. 2023; Larsson & Moffat 2000; Nevill-Manning & Witten 1997; Charikar et
al. 2005). The multi-level tokenizer is **hierarchical BPE / two-stage BPE /
dynamic patching** (Dolga et al. 2025; Liu et al. 2025; Pagnoni et al. 2025;
Slagle 2024). The embedding homomorphism is **Montague / DisCoCat-style
compositional semantics**, the **Tree-LSTM / recursive neural network**
recursion, and explicitly **LEAR's syntactic-to-semantic algebra homomorphism**
(Coecke et al. 2010; Tai et al. 2015; Liu et al. 2021), with
**subword-compositional OOV embeddings** as the lexical precedent (fastText,
Mimick, Charagram). The algebra/coalgebra pair is the **standard initial-algebra
/ terminal-coalgebra theory of signatures**: for a finitary signature, finite
terms versus finite+infinite terms (Goguen et al. 1977; Rutten 2000; Jacobs
2016; Adámek, Milius & Moss 2018). The ambiguity channel is **segmentation
ambiguity / non-unique tokenization**, already studied in tokenizer robustness
and linguistic steganography (e.g. Nozaki et al. 2022, "Addressing Segmentation
Ambiguity in Neural Linguistic Steganography", AACL 2022). The compression
objective is the **smallest grammar problem / MDL grammar induction**, which is
NP-hard in general (Charikar et al. 2005; De Marcken 1996).

What survives as potentially new is therefore: (a) the extension of
merge-tree compositional embeddings from rare tokens to a *multi-level* grammar
whose levels are chosen for morphological alignment, and the demonstration that
this improves unseen-compound composition at matched compression; (b) the
matched-vocabulary, matched-compression comparison protocol, if it is specified
carefully; and (c) the use of the grammar's ambiguity as a measurable channel,
if that is developed beyond the current "separate drawer" remark. Everything
else — hierarchical BPE, grammar compression, merge-tree compositional
embeddings, recursive neural embeddings, algebraic semantics, morphological
alignment, segmentation ambiguity — is already known under another name.

The honest verdict on the current v0.1 is this. As a **mathematical framing**,
it is coherent but textbook: the initial/terminal coalgebra distinction is
correct, and the final coalgebra is not used. As a **tokenizer proposal**, it is
a multi-level BPE / SLP, and the literature already contains morphology-aware
and hierarchical variants that would be the right baselines. As an **embedding
proposal**, the central merge-tree recursion is claimed in a 2026 preprint
(Bass); the project can claim the multi-level, morphology-aligned extension, not
the idea itself. As an **empirical hypothesis**, the first half (better
morphological alignment at equal compression) is plausible and partly
anticipated by MorphBPE and TreeTokenizer; the second half (better compositional
embeddings on unseen compounds) is the only genuinely open claim in the v0.1
package, and it is not tested by SCAN/CFQ or by boundary F1. The project should
reframe v0.1 around that
second half, cite Bass (2026) explicitly, compare against CoBPE-style
compositional tokenization and subword-compositional embeddings rather than flat
BPE alone, and treat the coalgebra as a conceptual appendix rather than a
result.
