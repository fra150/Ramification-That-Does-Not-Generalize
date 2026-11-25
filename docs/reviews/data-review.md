# Adversarial review — data and evaluation pipeline (v0.1)

Scope: `docs/ramified-v0.md`, `ramified/synthetic.py`, `ramified/metrics.py`,
`ramified/corpora.py`, `experiments/run_matrix.py`. Supporting code inspected
where the reviewed files call it (`ramified/bpe.py`).

No source files were modified. All probes ran from the repository root on this
machine:

- Python 3.13.13 (Anaconda), NumPy 2.2.6, Windows
- stdlib path `C:\Users\acese\miniconda3\Lib`
- probe scripts written under `C:\Users\acese\AppData\Local\Temp\opencode`
  (not part of the repository)

The concurrent v0.2 artifacts (`ramified/codec.py`,
`experiments/run_matrix_v2.py`) are noted where relevant, but the review target
is the v0.1 path actually used to produce `docs/findings-v0.1.md`.

**Revision note.** `ramified/bpe.py` changed while this review was running: the
pair-count update was corrected to push decrements, and `train_tokenizer`
gained a `split` argument. All numeric evidence below was regenerated against
the current tree. The numbers in `docs/findings-v0.1.md:13-23` predate that
correction and are not reproducible with the current BPE; C0 and M3 call this
out explicitly. This is itself a symptom of the missing revision/hash metadata
in the pipeline. Snapshot SHA-256 prefixes for the files under review:
`bpe.py 73093a3e62fc`, `synthetic.py e7d4e5fbb01e`,
`metrics.py 6a2243521250`, `corpora.py 3dc1fd9db67b`,
`run_matrix.py e47093a7ca60`.

Severity scale: **Critical** = a headline number is invalid or the stated
property is not measured; **High** = can change a scientific conclusion or
leaves the central claim untested; **Medium** = materially biases values,
statistical power, or reproducibility; **Low** = robustness/hygiene.

## Summary, ordered by severity

| ID | Severity | Finding | Location |
|---|---|---|---|
| C0 | Critical | `hier2` and `hier3` produce exactly the same tokenizer as `flat` at the same total vocabulary: levels only partition one global greedy merge sequence, so there is no hierarchy treatment to compare | `ramified/bpe.py:68-137,213-259`, `experiments/run_matrix.py:25-29,112-117` |
| C1 | Critical | `bits_per_char` is not bits per character for code: the tokenizer discards all whitespace while the denominator counts it | `ramified/metrics.py:11-24`, `ramified/bpe.py:262-278`, `experiments/run_matrix.py:47-63` |
| C2 | Critical | `decode_exact` is word-level and tautological; it cannot fail for in-alphabet words and misses document/whitespace corruption entirely | `ramified/metrics.py:53-59`, `experiments/run_matrix.py:43,54` |
| H1 | High | The synthetic holdout tests interpolation, not compositional generalisation: every test triple has all three component pairs attested in training, often many times | `ramified/synthetic.py:101-116`, `README.md:100-103`, `docs/ramified-v0.md:81-90` |
| H2 | High | The compositional vectors are generated but never evaluated or exported with their component ground truth; the central hypothesis is not tested | `ramified/synthetic.py:26-32,35-43,84-99,138-145`, `experiments/run_matrix.py:32-44` |
| H3 | High | The "code" corpus is 16 alphabetically first files, 46% `_pydecimal.py`; its test set is the last 40k chars of the same file as training, not a held-out document, and is install-dependent | `ramified/corpora.py:10-25`, `experiments/run_matrix.py:105-109` |
| H4 | High | Unigram bits/char is non-monotonic in vocabulary; the 1024 regression is not explained by add-1 smoothing alone, so it is not a robust compression comparison | `ramified/metrics.py:19-23`, `experiments/run_matrix.py:35,50` |
| M1 | Medium | Boundary F1 mixes macro and micro conventions; reported P/R/F1 are not mutually consistent, macro precision is inflated by empty predictions, and the averaging choice materially changes reported precision | `ramified/metrics.py:27-36,39-50` |
| M2 | Medium | BPC and boundary F1 are computed on different test samples (BPC covers 1324/2304 held-out types); test roots are imbalanced (10–31 words per root) | `ramified/synthetic.py:101-116,134-136`, `experiments/run_matrix.py:35-36` |
| M3 | Medium | Seed variation is large and the v0.1 per-method deltas are not reproducible with the current BPE; code-corpus seed rows are identical pseudo-replicates | `experiments/run_matrix.py:81,93-95,131`, `docs/findings-v0.1.md:13-23` |
| M4 | Medium | CSV schema lacks corpus/tokenizer metadata, is overwritten each run, loses everything on a mid-run exception, and crashes on an empty/unknown corpus | `experiments/run_matrix.py:125-142` |
| M5 | Medium | Corpus loader silently ignores decode errors, bypasses PEP 263 encoding declarations, truncates mid-line, and has no corpus identity | `ramified/corpora.py:13,18,24-25` |
| L1 | Low | Zipf sampling is implemented through a shuffled rank vector (distributionally Zipf but confusing); 18.9% of training types never appear; punctuation appears only in training | `ramified/synthetic.py:120-132` |
| L2 | Low | `holdout=0`/`1` crashes; the attestation filter is order-dependent at extreme holdout and does not use the realised training sample | `ramified/synthetic.py:102-116,135` |
| L3 | Low | `vocab_target` can be below the minimum alphabet size, in which case `vocab_actual > vocab_target` and BPC uses the larger actual vocabulary with no parity guard | `ramified/bpe.py:244`, `experiments/run_matrix.py:129-130` |
| L4 | Low | OOV characters expand to `<unk>` of length 5, corrupting `word_boundaries` character offsets | `ramified/bpe.py:185-196`, `ramified/metrics.py:42` |
| L5 | Low | CLI does not validate corpus/method names or empty lists; `--corpora ""` raises `IndexError`, an unknown corpus raises `TypeError` | `experiments/run_matrix.py:85-88,100-109,140` |

---

## C0 — `hier2` and `hier3` are the same tokenizer as `flat`

**Where:** `ramified/bpe.py:68-137` (`train_bpe`), `ramified/bpe.py:213-259`
(`train_tokenizer`), `experiments/run_matrix.py:25-29` (`METHODS`) and
`experiments/run_matrix.py:112-117` (training call).

**Why it matters.** `train_bpe` greedily emits the globally best pair merges
from the current sequences. `train_tokenizer` then splits one total merge
budget into level budgets and calls `train_bpe` again on the sequences left by
the previous level. That is exactly the continuation of the same global BPE
merge sequence, not a different constraint: the levels do not restrict
candidate pairs, do not use different alphabets or boundaries, and do not
change the objective. `Tokenizer._apply_level` merely applies the first chunk
of the merge list, then the second, etc. With the same total vocabulary, every
partition of the merge budget produces the same final merge set and therefore
the same tokenization. The `n_levels`/`level_weights` parameters relabel work,
they do not change the model.

This is not a small equivalence: for the evaluated surface (token IDs,
boundaries, BPC, decode) the matrix has **no hierarchy treatment**. Any
observed `flat` vs `hier2`/`hier3` difference is either random sampling noise
in a different corpus seed or an implementation bug. The v0.1
numbers in `docs/findings-v0.1.md:13-23` show such differences; the current
`bpe.py` docstring at lines 40-65 records that the old pair-count update
(stale heap entries, no decrement pushes) was a bug. The v0.1 differences are
consistent with that bug and cannot be reproduced with the current code; they
are not evidence for hierarchy. I did not modify the source; the current file
already contains the correction.

**Evidence.**

```powershell
python C:\Users\acese\AppData\Local\Temp\opencode\audit_hierarchy_equivalence.py
```

```text
synthetic v=128  flat levels=[106] ids=9662
synthetic v=128  hier2 levels=[64, 42] ids=9662
synthetic v=128  hier3 levels=[53, 32, 21] ids=9662
synthetic v=128: flat==hier2 True, flat==hier3 True
synthetic v=256  flat levels=[234] ids=7700
synthetic v=256  hier2 levels=[140, 94] ids=7700
synthetic v=256  hier3 levels=[117, 70, 47] ids=7700
synthetic v=256: flat==hier2 True, flat==hier3 True
synthetic v=512  flat levels=[490] ids=5982
synthetic v=512  hier2 levels=[294, 196] ids=5982
synthetic v=512  hier3 levels=[245, 147, 98] ids=5982
synthetic v=512: flat==hier2 True, flat==hier3 True
synthetic43 v=512 flat levels=[490] ids=6012
synthetic43 v=512 hier2 levels=[294, 196] ids=6012
synthetic43 v=512 hier3 levels=[245, 147, 98] ids=6012
synthetic43 v=512: flat==hier2 True, flat==hier3 True
code v=256  flat levels=[160] ids=20662
code v=256  hier2 levels=[96, 64] ids=20662
code v=256  hier3 levels=[80, 48, 32] ids=20662
code v=256: flat==hier2 True, flat==hier3 True
code v=512  flat levels=[416] ids=16180
code v=512  hier2 levels=[250, 166] ids=16180
code v=512  hier3 levels=[208, 125, 83] ids=16180
code v=512: flat==hier2 True, flat==hier3 True
code v=1024 flat levels=[928] ids=14005
code v=1024 hier2 levels=[557, 371] ids=14005
code v=1024 hier3 levels=[464, 278, 186] ids=14005
code v=1024: flat==hier2 True, flat==hier3 True
```

Extreme weight splits do not change it either:

```powershell
python -c "import sys; sys.path.insert(0,r'C:\Users\acese\Desktop\unibull'); from ramified.synthetic import generate_synthetic; from ramified.bpe import train_tokenizer,tokenize_normalized; s=generate_synthetic(seed=42); ids={}; [ids.setdefault(str(w),tokenize_normalized(s.test_text,train_tokenizer([s.train_text],512,n_levels=2,level_weights=w))) for w in ([0.1,0.9],[0.9,0.1],[0.5,0.5])]; print({k:len(v) for k,v in ids.items()}); vals=list(ids.values()); print('all equal',vals[0]==vals[1],vals[0]==vals[2])"
```

```text
{'[0.1, 0.9]': 5982, '[0.9, 0.1]': 5982, '[0.5, 0.5]': 5982}
all equal True True
```

**Fix.**

1. Give each level a real constraint. Examples: level 1 may merge only inside
   morpheme spans or only pairs whose expansion stays within a word; level 2
   operates on cross-boundary symbols with a different candidate set. A level
   that only re-runs the same greedy merge on the same sequences is flat.
2. If the intent is simply to allocate a merge budget in chunks, delete the
   hierarchy methods and report one baseline; do not call the result
   ramified/hierarchical.
3. Add a treatment-fidelity test: on at least one corpus and vocabulary, assert
   that the token IDs of flat and each hierarchical method differ; otherwise
   fail the run before producing a matrix.
4. Rebuild `docs/findings-v0.1.md` from the current fixed BPE; the old
   per-method deltas are not reproducible and must not be reported as a
   hierarchy effect.

---

## C1 — `bits_per_char` is not bits per character for code

**Where:** `ramified/metrics.py:11-24` (denominator is `test_char_len`,
supplied by `experiments/run_matrix.py:50` as `len(test_text)`, raw);
`ramified/bpe.py:262-273` (`tokenize_normalized` normalises and then matches
only `\w+|[^\w\s]`); `ramified/bpe.py:276-278` (`detokenize` inserts spaces at
EOW). For synthetic, `experiments/run_matrix.py:35` uses
`len(corpus.test_text)`.

**Why it matters.** `tokenize_normalized` deletes every whitespace character
before tokenisation. For code, that is 28.5% of the test text (newlines,
indentation, runs of spaces). The cross-entropy is then divided by the raw
character count, so the model is never charged for the characters it did not
encode. The result is not a compression rate for the corpus; it is a rate for
a lossy whitespace-normalised projection of the corpus. The same error affects
`tokens_per_char` (`run_matrix.py:57`), so the reported token cost is also
understated. A tokenizer that dropped more whitespace would look better for
free. This directly contradicts `docs/ramified-v0.md:68-72` ("bits per
character") and the decode requirement at line 93.

**Evidence.**

```powershell
python C:\Users\acese\AppData\Local\Temp\opencode\audit_corpora.py
```

```text
loaded chars=400000 files_read=16
split=360014 (newline found at 360014); test chars=39986
test whitespace chars=11409/39986 (28.5%)
len(test)=39986 len(normalized)=31740 reduction=20.6%
actual vocab=512 train ids=135104 test ids=16180
bpc (raw char denominator)=3.0948; bpc (normalized denominator)=3.8988 ratio=1.260
tokens/raw-char=0.4046; tokens/normalized-char=0.5098
whitespace-only different inputs tokenize identically: True
detokenized test len=32966 vs raw=39986; equal=False
raw   prefix='\n    __hash__ = None\n\n    def Etiny(self):\n        """Returns Etiny (= Emin - pr'
detok prefix='__hash__ =None def Etiny (self ):"""Returns Etiny (=Emin -prec +1 )"""return int'
```

A concrete whitespace case:

```powershell
python C:\Users\acese\AppData\Local\Temp\opencode\audit_decode.py
```

```text
in='a . b' -> out='a .b' equal=False ids=[149, 13, 266]
in='if x :\n    y = 1\n' -> out='if x :y =1' equal=False ids=[124, 199, 25, 131, 28, 143]
full test:
raw len 39986 rec len 32966 equal False
nonspace equal True
space count raw 10259 rec 4389
newline count raw 1150 rec 0
chars lost 7020
```

So the code BPC in `docs/findings-v0.1.md:27-37` is understated by ~26% on
this corpus using the normalised length; even using the codec's own recovered
length (32,966) gives 3.75, still 21% above 3.09. No denominator can charge for
whitespace the tokenizer never encodes. The code codec cannot reconstruct the
input at all.

**Fix.**

1. Decide what is being compressed. If it is source code, whitespace must be
   part of the codec: encode spaces/newlines/indentation (a space terminal, or
   a separate whitespace stream), and make `dec(enc(x)) == x` on the original
   text.
2. If the corpus is deliberately whitespace-normalised, normalise the corpus
   before measuring, pass the normalised length as the denominator, and label
   the metric "bits per normalised character"; never call it code compression
   or compare it with synthetic BPC as if the units were the same.
3. Add a document-level round-trip assertion to the code path, not only a
   word-list check.

Note: the concurrent `ramified/codec.py:22-39` normalises before comparing
(`roundtrip_ok(text, tok) == normalize(text)`), so it fixes the space-after-
punctuation bug but still does not restore original newlines/indentation.
`run_matrix_v2.py:73-92` also uses `len(normalize(test_text))`; that is a
consistent normalised-character rate, but not a rate for the raw code corpus.

---

## C2 — `decode_exact` cannot detect the failure it is supposed to detect

**Where:** `ramified/metrics.py:53-59`; called at
`experiments/run_matrix.py:43` (synthetic) and duplicated at
`experiments/run_matrix.py:51-55` (code).

**Why it matters.** `decode_exact` does:

```python
decoded = tokenizer.decode(tokenizer.encode(w.word, eow=True)).replace(EOW, "")
if decoded == w.word: ...
```

`encode` emits symbols; `decode` concatenates those symbols' expansions. For
every character present in `char_to_id`, this is the identity by construction.
The only way it can return <1.0 is an out-of-alphabet character mapped to
`<unk>`, which makes the metric an OOV-coverage statistic, not an invertibility
test. It never tokenises whitespace or punctuation at the document level, so it
passes while the actual codec is losing 7,020 characters on the code test set.
The doc's requirement (`docs/ramified-v0.md:93`) is
`dec(enc(x)) = x`; the implementation tests only `x = word` over synthetic
words and a 1,000-word sample of code.

**Evidence.** Every synthetic cell in the table below, and the code path,
reported `decode_exact = 1.0000`:

```text
v=256  flat  actual=256  dec=1.0000
v=256  hier2 actual=256  dec=1.0000
v=256  hier3 actual=256  dec=1.0000
v=512  flat  actual=512  dec=1.0000
v=512  hier2 actual=512  dec=1.0000
v=512  hier3 actual=512  dec=1.0000
v=1024 flat  actual=1024 dec=1.0000
v=1024 hier2 actual=1024 dec=1.0000
v=1024 hier3 actual=1024 dec=1.0000
```

and the code path reports `code decode_exact=1.0000 failures=0/1000`
(from the same `audit_corpora.py` run) while
`detokenize(tokenize_normalized(test_text)) != test_text` (C1).

The only real failure mode is a coverage failure, not a codec bug:

```text
encode('café') = [3, 1, 20, 20, 21]
decode = 'ca<unk><unk>\x00</w>'
```

**Fix.**

1. Replace the word-list check with a document-level test on the actual test
   text: `detokenize(tokenize(text)) == text`, including punctuation and
   whitespace, for both corpora.
2. Keep a separate OOV rate/metric; do not call it exact decode.
3. Add a byte/character fallback for characters outside the training alphabet
   if the domain is arbitrary Unicode; otherwise document that the input
   alphabet is closed and filter the test set accordingly.
4. Report `decode_exact` only for in-alphabet inputs, and report `decode_doc`
   separately (the v0.2 code handles the latter, but only after normalisation).

---

## H1 — The holdout tests interpolation, not compositional generalisation

**Where:** `ramified/synthetic.py:101-116` (random word-level holdout and
component-attestation filter); claim at `README.md:100-103` and
`docs/ramified-v0.md:81-90`.

**Why it matters.** The split is a random 20% of the 11,520 word triples. Every
held-out word has all three single components in training, which is exactly
what the filter enforces. More importantly, all three **pairwise** combinations
of a held-out triple are also in training:

```text
prefix-root: test words with pair seen in train = 2304/2304; min/median/max train occurrences for seen pairs = 4/9/11
root-suffix: test words with pair seen in train = 2304/2304; min/median/max train occurrences for seen pairs = 1/6/7
prefix-suffix: test words with pair seen in train = 2304/2304; min/median/max train occurrences for seen pairs = 85/95/108
total test words whose (prefix,root) and (root,suffix) and (prefix,suffix) are all seen: 2304
```

Thus only the exact triple is novel. The same holds in the realised
120,000-token training text, not just in the type list: every test triple still
has all three pairs attested there (minimum counts 3 / 1 / 63 for
prefix-root / root-suffix / prefix-suffix). A model that memorises pairwise
morpheme interactions, or even a lookup table over `(prefix, root)` and
`(root, suffix)`, can score well. This is interpolation over triples, not the
systematic/compositional extrapolation the hypothesis needs. It also means the
"every component attested" check (`synthetic.py:106-116`) is vacuous at the
default: `n_test=2304`, and **zero** words are moved back to training. Even at
`holdout=0.95` (576 training types) no word is moved, because each root has 96
words and a 5% type sample still covers every root. The filter only starts to
bind around `holdout=0.97` (826 train types after moves, 10,694 test words) and
at `holdout=1.0` it crashes (L2).

**Evidence.**

```powershell
python C:\Users\acese\AppData\Local\Temp\opencode\audit_holdout_pairs.py
```

```text
prefix-root: test words with pair seen in train = 2304/2304; min/median/max train occurrences for seen pairs = 4/9/11
root-suffix: test words with pair seen in train = 2304/2304; min/median/max train occurrences for seen pairs = 1/6/7
prefix-suffix: test words with pair seen in train = 2304/2304; min/median/max train occurrences for seen pairs = 85/95/108
total test words whose (prefix,root) and (root,suffix) and (prefix,suffix) are all seen: 2304
prefix-root pairs in train=960, avg suffixes per pair=9.6
root-suffix pairs in train=1440, avg prefixes per pair=6.4
```

The default filter behaviour:

```text
n_roots=120 n_triples=11520
n_train_types=9216 n_test_words=2304
initial holdout slots=2304; kept=2304; moved-back-to-train=0
unique word strings=11520; string collisions=0
train/test word-string overlap=0 []
test words whose type-level components are not all in train types: 0
roots in train=120/120 prefixes=8/8 suffixes=12/12
```

**Fix.**

1. Define the generalisation regime explicitly and build the split accordingly:
   - **Interpolation:** current random triple holdout; fine, but rename it.
   - **Pair-disjoint:** hold out a set of `(prefix, root)` pairs (and
     separately `(root, suffix)` pairs), so test triples contain an unseen
     pair; report pair coverage, not just component coverage.
   - **Zero-shot morphemes:** hold out entire roots/prefixes/suffixes; this is
     the only setting that tests unseen-component composition.
2. Replace the single-component attestation filter with a pair-coverage report
   and a split that cannot leak a held-out pair.
3. Stratify the holdout by component so every root has the same number of test
   words; the current root counts range from 10 to 31 (M2).

---

## H2 — Compositional vectors are generated but never evaluated, and the component ground truth is not exported

**Where:** `ramified/synthetic.py:26-32` (`Word.vector`),
`synthetic.py:84-99` (root vectors and prefix/suffix matrices; `P @ S @ r` at
line 89), `synthetic.py:35-43` and `synthetic.py:138-145` (the corpus now
returns `train_word_objects` as well as `test_words`, but no component
matrices/root vectors). `experiments/run_matrix.py:32-44` evaluates only BPC,
boundaries, tokens/char and decode; no vector metric exists.

**Why it matters.** The specification's second claim
(`docs/ramified-v0.md:81-90`, README phase 2) is that the ramified tokenizer
produces a better compositional embedding on unseen compounds. The pipeline
cannot test that claim at all: nothing in `run_matrix.py` reads `Word.vector`,
and `SyntheticCorpus` does not expose `root_vec`, `prefix_mat` or `suffix_mat`.
The recent addition of `train_word_objects` makes train word vectors available,
but even a future embedding experiment cannot recover the component ground
truth from the returned corpus; it only has word vectors, and the component
ground truth must be re-derived by replaying the RNG. The formula itself is
correct: the code computes
`prefix_mat[p] @ (suffix_mat[s] @ root_vec[r])`, matching the docstring and
the doc's `P @ S @ r`; all 2,304 test vectors are unit norm and distinct.
This is a coverage gap, not an algebra bug.

**Evidence.**

```powershell
python -c "import sys,numpy as np; sys.path.insert(0,r'C:\Users\acese\Desktop\unibull'); from ramified.synthetic import generate_synthetic; s=generate_synthetic(seed=42); n=np.array([np.linalg.norm(w.vector) for w in s.test_words]); V=np.stack([w.vector for w in s.test_words]); print('norms',n.min(),n.max()); print('distinct',len({tuple(np.round(v,9)) for v in V}))"
```

```text
norms 0.9999999999999994 1.0000000000000007
distinct 2304
```

No metric in `run_matrix.py` consumes `Word.vector`.

**Fix.**

1. Add a Phase 2 evaluation or explicitly scope v0.1 out: report that the
   embedding claim is not tested.
2. Export the component ground truth: prefix matrices, suffix matrices, root
   vectors, plus a `compose(prefix, root, suffix)` function, so an embedding
   model does not have to re-derive internals from the RNG. Train/test word
   vectors are now available and should be used by the Phase 2 evaluation.
3. Evaluate on the pair-disjoint/zero-shot split from H1, not the current
   interpolation split. Add noise or a nonlinear composition if the intent is
   to make the task nontrivial; the current map is a linear group action and is
   exactly representable.

---

## H3 — The code corpus is one file's tail, not a held-out real-text corpus

**Where:** `ramified/corpora.py:10-25` (sorted top-level glob, `max_chars`);
`experiments/run_matrix.py:105-109` (90% character split).

**Why it matters.** The loader takes the first files in alphabetical order
until 400,000 characters. On this install that is 16 of 152 top-level files,
and `_pydecimal.py` alone supplies 46.0% of the corpus. The 90% split falls
inside `_pydecimal.py`; the test set is its last 39,986 characters, while
143,996 characters of the same file are in training. Fifteen of the sixteen
files contribute nothing to the test set. This is not a held-out document or
package, and the result depends on the local Python installation and its
version (`C:\Users\acese\miniconda3\Lib`, Python 3.13.13 here). The code BPC
table in `docs/findings-v0.1.md` is therefore a within-file measurement on a
single large, highly repetitive pure-Python module, not evidence about "code"
generally.

**Evidence.**

```powershell
python C:\Users\acese\AppData\Local\Temp\opencode\audit_corpus_composition.py
```

```text
files available=152 files read=16
file                       source  in_corpus   share    train    test
__future__.py                5218       5218    1.3%     5218       0
...
_pydatetime.py              91983      91983   23.0%    91983       0
_pydecimal.py              227526     183982   46.0%   143996   39986
total corpus=400000 train=360014 test=39986
```

```powershell
python C:\Users\acese\AppData\Local\Temp\opencode\audit_corpora.py
```

```text
split=360014 (newline found at 360014); test chars=39986
file containing split: [('_pydecimal.py', 216018, 443544)]
test starts with='\n    __hash__ = None\n\n    def Etiny(self):\n        """Return'
split is line start: False
files in train=16 test=1 both=1 ['_pydecimal.py']
test lines=1151 nonblank duplicate-in-train lines=96 chars=2210/38832 (5.7%)
most repeated duplicated nonblank lines:
      48x '        """'
```

The `False` is an artifact of my check comparing `corpus[split-1]` to a
newline; the split index points at the newline itself, so the test text begins
at a line start (with a leading newline).

Only one file is tested, and 5.7% of nonblank test characters are duplicate
lines already in training (mostly `"""`). The duplicate lines are not the main
problem; the same-file split and the corpus composition are.

**Fix.**

1. Vendor or pin a corpus artifact with a SHA-256 and a manifest; record the
   Python version and source path in the CSV. Do not let "the local stdlib"
   be the dataset.
2. Cap the characters contributed by any one file and sample files/packages
   rather than taking the alphabetically first sixteen.
3. Split by file/document: all text in the test set must come from files that
   do not appear in training. At minimum, choose the split at a file boundary
   and test several held-out files.
4. Report corpus statistics (files, chars, per-file shares, duplicate-line
   rate) next to the metrics.

---

## H4 — Unigram bits/char is non-monotonic and the 1024 regression is not a smoothing artifact

**Where:** `ramified/metrics.py:19-23` (`denom = total + vocab_size`,
add-one); `experiments/run_matrix.py:35,50` pass the actual vocabulary.
`docs/findings-v0.1.md:56-59` attributes the 1024 regression to this
denominator.

**Why it matters.** Add-one is an arbitrary smoothing prior whose total mass
grows with `vocab_size`, and `vocab_size` includes symbols that are consumed by
later merges and never emitted at the top level. More importantly, the data do
not support the claim that smoothing is the cause of the 512→1024 regression. Keeping the same
tokenisation and varying only the smoothing vocabulary moves BPC by ~0.017 bits
over a 32× range of V; changing the vocabulary/tokenisation from 512 to 1024
moves it by ~0.024, and add-alpha 0.1 does not remove the regression:

```powershell
python C:\Users\acese\AppData\Local\Temp\opencode\audit_bpc_oov.py
```

```text
flat synthetic seed 42: OOV test tokens and smoothing sensitivity
v=256   actual=256   tokens=7700   oov_types=0  oov_occ=0   (0.00%) bpc_add1=2.8159 bpc_alpha0.1=2.8159
v=512   actual=512   tokens=5982   oov_types=0  oov_occ=0   (0.00%) bpc_add1=2.3536 bpc_alpha0.1=2.3539
v=1024  actual=1024  tokens=5449   oov_types=1  oov_occ=1   (0.02%) bpc_add1=2.3778 bpc_alpha0.1=2.3789
v=2048  actual=2048  tokens=4544   oov_types=2  oov_occ=2   (0.04%) bpc_add1=2.3214 bpc_alpha0.1=2.3266
v=4096  actual=4096  tokens=4088   oov_types=12 oov_occ=28  (0.68%) bpc_add1=2.3709 bpc_alpha0.1=2.3897
```

With the same tokenisation, the smoothing-only sensitivity is small:

```powershell
python C:\Users\acese\AppData\Local\Temp\opencode\audit_metrics.py
```

```text
same tokenization / same counts, different smoothing vocabulary V:
  V=512   bpc=2.3536
  V=256   bpc=2.3531
  V=1024  bpc=2.3547
  V=4096  bpc=2.3612
  V=8192  bpc=2.3698
```

So the 512→1024 degradation is a property of the unigram model and the
tokenisation (larger vocabularies produce shorter but flatter/rarer token
distributions), not only the add-one denominator. `docs/findings-v0.1.md:56-59`
should not say the regression is explained by smoothing; smoothing correction
alone does not reverse it.

There is also a parity failure at small targets: `vocab_actual` can exceed
`vocab_target`. At target 16 the tokenizer has 22 symbols (chars + `<unk>` +
EOW), and BPC uses 22 with no warning (L3). Actual vocabularies matched targets
at 256–4096 in the tested configurations, so the current headline table is not
broken by this, but the pipeline does not enforce parity.

**Fix.**

1. Do not treat a single unigram BPC point as "compression". Report at least a
   second model (bigram/Kneser-Ney or an adaptive smoothing model) and a
   held-out smoothing/temperature selection.
2. Report alpha sensitivity, OOV rate, and both nominal and emitted vocabulary
   sizes. Use a fixed reference vocabulary or normalise the model description
   length if the goal is a fair comparison at equal V.
3. Enforce `vocab_actual == vocab_target` or fail/warn; if actual differs, the
   rows are not a parity comparison.

---

## M1 — Boundary F1 mixes macro and micro conventions and has inconsistent P/R/F1

**Where:** `ramified/metrics.py:27-36` (`boundary_scores`),
`metrics.py:39-50` (`mean_boundary_f1`, macro average); used at
`experiments/run_matrix.py:36`.

**Why it matters.** The CSV reports one precision, one recall and one F1, but
they are not mutually consistent. `mean_boundary_f1` averages per-word F1 while
also averaging per-word precision and recall; the mean of per-word F1 is not
the harmonic mean of the mean P and mean R. Example from the current seed-42
runs (all three methods are identical under C0):

```text
v=256  macro P=0.7122 R=0.9191 F1=0.7850; harmonic of mean P/R = 0.8025; micro P=0.6539 F1=0.7641
v=512  macro P=0.8889 R=0.8594 F1=0.8665; harmonic of mean P/R = 0.8739; micro P=0.8746 F1=0.8669
v=1024 macro P=0.9150 R=0.7726 F1=0.8199; harmonic of mean P/R = 0.8378; micro P=0.9052 F1=0.8336
```

A reader who computes F1 from the reported P/R gets a different number; the
gap is 0.018 at both 256 and 1024. Macro and micro precision also differ by
0.058 at v=256, 0.014 at v=512 and 0.010 at v=1024, while macro F1 is 0.021
above micro at 256 and 0.014 below it at 1024. These shifts are the same order
as the per-method deltas the experiment is trying to interpret. With the
current code all methods are identical (C0), so this cannot flip a method
ranking here; it will as soon as a real treatment exists.

`boundary_scores([], [2,3])` returns `(1.0, 0.0, 0.0)`: a word emitted as one
token gets **perfect precision** for predicting no boundaries. The current
synthetic runs happen to have zero such words at the tested vocabularies, but
the API is wrong for degenerate/OOV cases and will inflate macro precision when
it triggers.

**Evidence (key rows; the full table has all three methods identical).**

```powershell
python C:\Users\acese\AppData\Local\Temp\opencode\audit_metrics.py
```

```text
v=256  flat  bpc=2.8159 macroP=0.7122 macroR=0.9191 macroF1=0.7850 microP=0.6539 microR=0.9191 microF1=0.7641 pred_bounds=6477 true_bounds=4608 pred_empty_words=0
v=512  flat  bpc=2.3536 macroP=0.8889 macroR=0.8594 macroF1=0.8665 microP=0.8746 microR=0.8594 microF1=0.8669 pred_bounds=4528 true_bounds=4608 pred_empty_words=0
v=1024 flat  bpc=2.3778 macroP=0.9150 macroR=0.7726 macroF1=0.8199 microP=0.9052 microR=0.7726 microF1=0.8336 pred_bounds=3933 true_bounds=4608 pred_empty_words=0
...
all three methods (flat/hier2/hier3) produced identical ids and identical rows at each vocabulary (C0)
boundary_scores edge cases:
  pred=[] true=[] -> (1.0, 1.0, 1.0)
  pred=[] true=[2, 3] -> (1.0, 0.0, 0.0)
  pred=[2] true=[] -> (0.0, 1.0, 0.0)
  pred=[2] true=[2, 3] -> (1.0, 0.5, 0.6666666666666666)
```

**Fix.**

1. Report **micro** pooled TP/FP/FN for segmentation (standard CoNLL-style),
   and optionally macro as a secondary number. If macro is reported, compute
   P, R and F1 separately and state that F1 is the mean of per-word F1, not the
   harmonic mean of the displayed P/R.
2. Use `zero_division=0`: precision 0 when `pred` is empty and `true` is not
   empty; recall 0 when `true` is empty and `pred` is not. Do not return 1.0
   for an undefined precision.
3. Report `n_pred_boundaries`, `n_true_boundaries`, and `n_empty_pred` in the
   CSV so the reader can see when the convention matters.
4. Add a bootstrap CI over words; the small method differences in this dataset
   are not interpretable without one.

---

## M2 — BPC and boundary F1 are measured on different test sets; class balance is uneven

**Where:** `ramified/synthetic.py:101-116` (random split), `synthetic.py:134-136`
(test text sampled with replacement); `experiments/run_matrix.py:35-36`
(BPC uses `test_text`, boundaries use `test_words`); `ramified/metrics.py:39-50`
(macro average).

**Why it matters.** The held-out word list has 2,304 types, but BPC is computed
on 2,000 draws **with replacement**, which cover only 1,324 distinct types
(57.5%). 980 held-out types are never seen by the compression metric, while
the boundary metric evaluates all 2,304. The two headline numbers therefore
describe different samples and can move in opposite directions without any
contradiction in the data. The test set is also unbalanced by root: 10 to 31
words per root (mean 19.2), while each root contributes 96 words to the full
corpus; macro F1 weights a root with 31 test words more than one with 10.

The test text has no `.` tokens, while training inserts `.` with probability
0.07 after each sampled word; BPC therefore evaluates a distribution missing a
symbol that the model spent probability mass on. This is a smaller version of
the same train/test distribution mismatch.

**Evidence.**

```powershell
python -c "import sys,collections; sys.path.insert(0,r'C:\Users\acese\Desktop\unibull'); from ramified.synthetic import generate_synthetic; s=generate_synthetic(seed=42); c=collections.Counter(s.test_text.split()); print('test draws',sum(c.values()),'distinct',len(c),'of',len(s.test_words)); print('max repeat',max(c.values()),'singletons',sum(1 for v in c.values() if v==1)); print('types never drawn for bpc',len(s.test_words)-len(c))"
```

```text
test draws 2000 distinct 1324 of 2304
max repeat 6 singletons 825
types never drawn for bpc 980
```

```text
test prefix: distinct=8 min=261 max=312 mean=288.00
test suffix: distinct=12 min=173 max=220 mean=192.00
test root: distinct=120 min=10 max=31 mean=19.20
```

**Fix.**

1. Build one held-out evaluation set and use it for every metric: for example,
   take a random permutation of all 2,304 test words, use it once for BPC
   (optionally repeat the whole permutation k times), and use the same word
   list for boundary and decode metrics.
2. If subsampling is necessary, sample uniformly over types with a fixed seed
   and report coverage (`n_test_words`, `n_test_types_evaluated`).
3. Stratify the holdout by prefix/root/suffix so each component has the same
   number of test words; report per-component counts.
4. Either include `.` in the test text or remove punctuation from training;
   record the decision.

---

## M3 — Seed variation is large, the reported per-method deltas are not reproducible, and code seeds are pseudo-replicates

**Where:** `experiments/run_matrix.py:81` (`--seeds` default `"42"`),
`run_matrix.py:93-95` (synthetic corpus regenerated per seed; code reloaded,
not resampled), `run_matrix.py:131` (seed in row);
`ramified/synthetic.py:120-136` (Zipf sample and test sample consume
independent RNG streams).

**Why it matters.** On synthetic, seed variation is large even though all
methods are currently identical (C0). A five-seed sweep at vocab 512 gives:

```powershell
python C:\Users\acese\AppData\Local\Temp\opencode\audit_seeds.py
```

```text
seed method  bpc     macroF1
42   flat    2.3536  0.8665
42   hier2   2.3536  0.8665
43   flat    2.3916  0.8011
43   hier2   2.3916  0.8011
44   flat    2.2401  0.9142
44   hier2   2.2401  0.9142
45   flat    2.2552  0.8935
45   hier2   2.2552  0.8935
46   flat    2.3668  0.8029
46   hier2   2.3668  0.8029
flat:  bpc mean=2.3214 sd=0.0690 range=0.1516; F1 mean=0.8556 sd=0.0518 range=0.1131
hier2: bpc mean=2.3214 sd=0.0690 range=0.1516; F1 mean=0.8556 sd=0.0518 range=0.1131
paired method differences per seed:
  d_bpc(flat-hier2)=+0.0000 d_F1=+0.0000
  ...
```

With the current BPE every `flat`/`hier2`/`hier3` row is identical in the runs
tested. The v0.1 table in `docs/findings-v0.1.md:13-23` reports non-zero
differences (for example flat/hier2 at 512: 2.3834/2.3420); those numbers are
not reproducible with the current code (C0). A reader cannot tell from the CSV
which code revision produced the table, so the difference cannot be attributed
to hierarchy rather than to the stale-heap bug that was fixed.

The seed spread is still large: BPC sd 0.069 and macro F1 sd 0.052 across five
corpora. A one-seed run is not a stable estimate of the absolute metric, and a
three-seed matrix is unlikely to resolve the small deltas claimed in v0.1 even
if a real treatment existed.

On code, `seed` changes nothing: the corpus is loaded identically and the BPE
trainer is deterministic. The smoke run writes:

```text
code,flat,1,128,128,42,0.55,0.6890661731606063,4.19856427196522,,,,1.0
code,flat,1,128,128,43,0.51,0.6890661731606063,4.19856427196522,,,,1.0
```

Only `train_seconds` differs. Treating those as independent seeds inflates the
apparent sample size and makes `aggregate.py`'s standard deviation meaningless
for the scored metrics (they are exactly identical); only timing varies.

**Fix.**

1. Fix C0 first; no number of seeds can estimate a hierarchy effect when the
   hierarchical tokenizer is the flat tokenizer.
2. Pin the code revision (hash) with every result and regenerate
   `docs/findings-v0.1.md` from the current BPE. The current table is not
   reproducible from the current source.
3. Default to at least 5 seeds for synthetic; report per-seed absolute values,
   paired differences (if a real treatment exists) and a bootstrap/t confidence
   interval, not just means.
4. For code, define real replicates (different file subsets/corpora/splits) or
   run a single seed and label it as such. Do not duplicate code rows across
   seeds.
5. Increase the synthetic test size (or evaluate all 2,304 types) to reduce
   sampling noise; the current 2,000-token draw contributes to the BPC spread.
6. Keep timing measurements separate from statistical claims; they are
   wall-clock and noisy under parallel agent load (the findings doc already
   notes this).

---

## M4 — CSV schema lacks metadata, overwrites runs, and has no empty-run guard

**Where:** `experiments/run_matrix.py:125-142`; row fields at
`run_matrix.py:125-134`; header construction at line 140.

**Why it matters.** The CSV is the only output. It contains:

```text
corpus,method,n_levels,vocab_target,vocab_actual,seed,train_seconds,tokens_per_char,bits_per_char,boundary_precision,boundary_recall,boundary_f1,decode_exact
```

It does not contain the corpus hash/version, Python/NumPy version, synthetic
`holdout`/`n_roots`/`dim`/`n_train_tokens`/`n_test_tokens`, train/test character
lengths, number of boundary words, number of decode words, level weights/budget,
or the command line. None of the synthetic configuration can be reconstructed
from the CSV. The code test is one file (H3) but the CSV has no file identity.

The file is opened with `"w"` at line 139 and all rows are written only at the
end. A crash in the last run loses every earlier row; two runs with different
arguments overwrite each other when they share an out-dir (the default is
`results/results.csv`). With no runs, `rows[0]` raises:

```powershell
python -m experiments.run_matrix --corpora "" --methods flat --vocab-sizes 128 --seeds 42 --out-dir C:\Users\acese\AppData\Local\Temp\opencode\results_empty
```

```text
Traceback (most recent call last):
  File "C:\Users\acese\Desktop\unibull\experiments\run_matrix.py", line 147, in <module>
    main()
  File "C:\Users\acese\Desktop\unibull\experiments\run_matrix.py", line 140, in main
    writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
IndexError: list index out of range
```

An unknown corpus name is silently routed to the code branch and fails later:

```powershell
python -m experiments.run_matrix --corpora foo --methods flat --vocab-sizes 128 --seeds 42 --out-dir C:\Users\acese\AppData\Local\Temp\opencode\results_foo
```

```text
File "C:\Users\acese\Desktop\unibull\experiments\run_matrix.py", line 105, in main
    split = int(len(code) * 0.9)
TypeError: object of type 'NoneType' has no len()
```

**Fix.**

1. Write an atomic output with a run ID and a sidecar `config.json` containing
   every argument and every dataset/tokenizer parameter. Include SHA-256 of
   the corpus and the Python/NumPy versions.
2. Append/resume rather than overwrite; write rows incrementally with
   `flush`, or write a temp file and rename at the end. Never lose completed
   runs to a later exception.
3. Validate `corpora`, `methods`, `seeds`, `vocab_sizes` up front; fail with a
   clear message if no runs would be produced or a name is unknown.
4. Add `n_train_chars`, `n_test_chars`, `n_test_words`, `n_boundary_words`,
   `n_decode_words`, `decode_doc` and `vocab_parity_ok` columns.
5. Since this working tree is not a git repository, archive the code snapshot
   or record a content hash alongside each result.

---

## M5 — Corpus loader silently changes text and has no corpus identity

**Where:** `ramified/corpora.py:13` (sorted top-level glob),
`corpora.py:18` (`encoding="utf-8", errors="ignore"`),
`corpora.py:24-25` (`break` then hard `[:max_chars]` truncation);
`experiments/run_matrix.py:105-109` (newline-adjusted split).

**Why it matters.** `errors="ignore"` silently drops undecodable bytes, and
hardcoding UTF-8 ignores PEP 263 coding declarations. A file with a different
coding cookie would decode as mojibake or lose bytes without an error; the
corpus length and all metrics would silently change. `load_code_corpus` also
truncates at exactly `max_chars` rather than at a line/file boundary, so the
last line can be incomplete. On this install the risk is latent: all 152
top-level files decode strictly as UTF-8, no coding cookies were found, and
`errors="ignore"` changed no file. But the loader has no way to report that,
and another Python installation/version can differ.

The truncation is observable: the corpus ends inside a line:

```text
corpus ends with='2)\n        Decimal(\'1\')\n        """\n    '
last source file ends with='H_MODULUS - 2, _PyHASH_MODULUS)\ndel sys\n'
```

The test text therefore ends with an incomplete fragment, which is not a clean
document. The sorted glob also determines the corpus composition (H3), so the
choice of `max_chars` and file order is a data-selection policy, not a neutral
cap.

**Evidence.**

```powershell
python C:\Users\acese\AppData\Local\Temp\opencode\audit_corpora.py
```

```text
files=152 strict-utf8 failures=0
non-utf8 coding cookies=0
files changed by errors='ignore'=0 []
loaded chars=400000 files_read=16
last file read=_pydecimal.py source_len=227526 chars kept=183982
corpus ends with='2)\n        Decimal(\'1\')\n        """\n    '
```

**Fix.**

1. Read source files with `tokenize.open` or `importlib.util.decode_source` so
   coding declarations are honoured; use `errors="strict"` and log/skip a file
   explicitly on `UnicodeDecodeError`.
2. Truncate at a newline or file boundary (`text[:max_chars].rsplit("\n", 1)[0]`
   plus a newline), so the corpus never ends mid-line.
3. Record the manifest, per-file character counts and SHA-256 in the run
   metadata; reject an empty corpus or an empty test split.
4. Make file selection explicit (sampled packages/files, per-file caps) rather
   than "first N sorted paths" (H3).

---

## L1 — Zipf sampling is confusing and leaves 19% of training types unsampled

**Where:** `ramified/synthetic.py:120-132`.

**Why it matters.** The code:

```python
order = list(range(len(train_words)))
rng.shuffle(order)
probs = np.array([1.0 / (rank + 1) ** 1.1 for rank in order])
```

assigns `probs[k]` the weight of `order[k]`, not of rank position `k`. Because
`order` is a uniform permutation, the resulting weights are still a random
Zipf assignment, so this is not a distributional bug, but the variable name is
misleading and the exact assignment is not the obvious "rank 1 is the first
shuffled word" implementation. The realised sample covers 7,478/9,216 training
types (81.1%); 1,738 types never appear. All components still appear in the
realised text for seed 42, but `train_words` includes the unsampled types and
the attestation filter uses that type list, not the realised text (H1/L2).
The 0.07 probability of appending `.` gives training punctuation that the test
text never contains.

**Evidence.**

```powershell
python C:\Users\acese\AppData\Local\Temp\opencode\audit_synthetic.py
```

```text
realized distinct word types=7478/9216 (81.1%)
top 1% of realized types hold 62.2% of tokens
top 10 counts=[18222, 8434, 5415, 3920, 3218, 2561, 2258, 1813, 1667, 1422]
types sampled once=2194
```

**Fix.**

1. Make the rank assignment explicit:
   `probs[order[i]] = 1.0 / (i + 1) ** 1.1` (if the first shuffled word should
   be rank 1), or document that a random Zipf weight is intentional.
2. Report type coverage, and if the experiment claims every component is
   attested in training, compute that on the realised token sample.
3. Either include `.` in test sampling or remove it from training; record the
   choice.

---

## L2 — Holdout validation and extreme values are unguarded

**Where:** `ramified/synthetic.py:102-116` (split/filter),
`synthetic.py:135` (test sampling).

**Why it matters.** `holdout=0` produces an empty test list and crashes at
`np_rng.integers(0, 0, ...)`; `holdout=1.0` produces an empty training set and
the same crash before a useful error. At high holdout, rejected test words are
appended to `train_words_objs`, but `train_roots`/`train_prefixes`/
`train_suffixes` are not updated. The filter is therefore order-dependent: a
rejected word can make its component attested for later words, but the code
never notices, so it can move more words than necessary. At the default it
moves zero words (H1), so this is low priority, but it makes the holdout
fraction unreportable.

```powershell
python C:\Users\acese\AppData\Local\Temp\opencode\audit_holdout_extreme.py
```

```text
holdout=0.0: ValueError: high <= 0
holdout=0.9: train_types=1152 test_words=10368
holdout=0.95: train_types=576 test_words=10944
holdout=0.97: train_types=826 test_words=10694
holdout=0.98: train_types=2151 test_words=9369
holdout=0.99: train_types=4820 test_words=6700
holdout=1.0: ValueError: high <= 0
```

At 0.97 the initial train set is 346 words; after moving rejected test words it
is 826, while the test set shrinks from 11,174 to 10,694. The actual holdout
fraction is not returned or recorded.

**Fix.**

1. Validate `0 < holdout < 1`, `n_test > 0`, `n_train > 0`, and non-empty
   test set; raise a descriptive `ValueError`.
2. Sample training text first, compute the realised component/pair sets, then
   filter test words against those sets; update sets as words move.
3. Return the realised holdout counts and the seed/params in `SyntheticCorpus`
   so the CSV can record them.

---

## L3 — `vocab_target` can be exceeded with no parity check

**Where:** `ramified/bpe.py:244` (`total_merges = max(0, vocab_size - len(chars) - 2)`);
`experiments/run_matrix.py:129-130` records both target and actual, but
`ramified/metrics.py:19` uses `vocab_size` (actual).

**Why it matters.** A target below `len(chars) + 2` silently produces a larger
vocabulary. At target 16 the synthetic tokenizer has 22 symbols:

```powershell
python C:\Users\acese\AppData\Local\Temp\opencode\audit_boundary_empty.py
```

```text
v=16 actual=22 macroF1=0.4182
```

BPC then smooths with 22, not 16, and a row labelled `vocab_target=16` is not
comparable to another method at 16. This does not affect the 256–4096 headline
rows (actual matched target in the tested configurations), but the pipeline
should not rely on that.

**Fix.** Validate `vocab_size >= len(chars) + 2` at the CLI, or fail the run
with `vocab_actual != vocab_target`; if actual differs, report it and do not
compare it at the target value. `experiments/verify.py` already flags this in
CSV, but the matrix should prevent it.

---

## L4 — OOV characters corrupt boundary offsets

**Where:** `ramified/bpe.py:185-196` (`word_boundaries` measures
`len(expansions[n.symbol])`); `ramified/metrics.py:42` passes test words
without checking for unknown characters.

**Why it matters.** `expansions[unk_id]` is the five-character string
`"<unk>"`, so an OOV character contributes five character positions to the
cumulative boundary count. The resulting offsets no longer correspond to the
original word. Synthetic test words currently use only the training alphabet,
and the code path does not evaluate boundaries, so this is latent; it will
break as soon as a held-out word contains an unseen character.

```powershell
python -c "import sys; sys.path.insert(0,r'C:\Users\acese\Desktop\unibull'); from ramified.bpe import train_tokenizer; from ramified.synthetic import generate_synthetic; s=generate_synthetic(seed=42); t=train_tokenizer([s.train_text],vocab_size=512); w='caf\u00e9x'; print('word len',len(w)); print('boundaries',t.word_boundaries(w)); print('encode',t.encode(w)); print('decode',repr(t.decode(t.encode(w))))"
```

```text
word len 5
boundaries [1, 2]
encode [3, 1, 20, 20, 49]
decode 'ca<unk><unk>x\x00</w>'
```

For `caféx`, the returned `[1, 2]` offsets are not valid character positions
for a two-boundary segmentation of a five-character word; the unknown token was
counted as five characters.

**Fix.** Represent unknown characters as one-character spans (or byte fallback)
for boundary accounting, skip words containing `<unk>` in boundary metrics and
report the skipped count, or move to a byte-level alphabet. Add a regression
test with an OOV character.

---

## L5 — CLI argument validation

**Where:** `experiments/run_matrix.py:85-88` (split names),
`run_matrix.py:100-109` (unknown corpus falls through to code branch),
`run_matrix.py:140` (empty rows).

**Why it matters.** `--corpora ""` and `--seeds ""` crash with `IndexError` at
the CSV writer; `--corpora foo` crashes with `TypeError` at `len(None)`;
`--methods foo` raises `KeyError` at `METHODS[method]`. None of these produce a
useful diagnostic, and an empty run leaves no output. This is separate from
M4's schema issue but has the same fix surface.

```powershell
python -m experiments.run_matrix --corpora "" --methods flat --vocab-sizes 128 --seeds 42 --out-dir C:\Users\acese\AppData\Local\Temp\opencode\results_empty
```

```text
File "C:\Users\acese\Desktop\unibull\experiments\run_matrix.py", line 140, in main
    writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
IndexError: list index out of range
```

```powershell
python -m experiments.run_matrix --corpora foo --methods flat --vocab-sizes 128 --seeds 42 --out-dir C:\Users\acese\AppData\Local\Temp\opencode\results_foo
```

```text
File "C:\Users\acese\Desktop\unibull\experiments\run_matrix.py", line 105, in main
    split = int(len(code) * 0.9)
TypeError: object of type 'NoneType' has no len()
```

**Fix.** Validate against known names, require at least one corpus/method/seed,
and raise `argparse.ArgumentTypeError`/`SystemExit` with a clear message before
training starts.

---

## Leakage check

- **Synthetic:** exact train/test word-string overlap is 0, test text contains
  only test words, and the tokenizer is trained only on `train_text`. BPC
  counts come from `train_text`; boundary and decode metrics use `test_words`.
  I found no label/target leakage in this path. The H1 problem is novelty, not
  leakage: the test triples are genuinely unseen, but all their component pairs
  are seen.
- **Code:** the same file (`_pydecimal.py`) is on both sides of the split; 5.7%
  of nonblank test characters are duplicate lines already in training (H3).
  That is a file-level leak, not an exact-document leak; it makes the test
  in-domain and single-file rather than a held-out code document.
- **Metric construction:** `bits_per_char` uses train counts and test tokens,
  so the unigram model is not fitted on test. `decode_exact` and
  `mean_boundary_f1` take test words. The failures in C1/C2 are measurement
  validity, not train/test contamination.

## What checked out

These checks are important context: the synthetic data has several good
properties, and the synthetic leakage claim is not the problem.

1. **No exact synthetic leakage.** 11,520 unique word strings, no string
   collisions, and zero train/test word-string overlap:
   ```text
   unique word strings=11520; string collisions=0
   train/test word-string overlap=0 []
   test words in train text: 0
   train text tokens outside train_words + dot: 0
   test text tokens outside test_words: 0
   ```
2. **Boundary tuples are valid.** All 2,304 test words have
   `boundaries == (len(prefix), len(prefix)+len(root))` with
   `0 < b0 < b1 < len(word)`; `invalid boundary tuples: 0`.
3. **Component attestation is true, if trivially so.** At seed 42 all 120
   roots, 8 prefixes and 12 suffixes are present in training types and in the
   realised training text; zero test words have an absent component. This is
   why H1 is about pair/triple novelty, not leakage.
4. **Actual vocabulary equals target at normal sizes.** At 256, 512, 1024,
   2048 and 4096, the configurations tested reported
   `vocab_actual == vocab_target` (all three methods at 256–1024; flat at
   2048/4096, on both corpora). The exception is the deliberately too-small
   target in L3.
5. **Same-seed reproducibility works in this environment.** Two separate CLI
   runs with the same seeds produced identical CSV rows except `train_seconds`:
   ```text
   identical_excluding_timing = True
   n_rows 4
   ```
6. **Strict UTF-8 succeeds on this stdlib.** All 152 top-level files decode
   strictly, no non-UTF-8 coding cookies were found, and `errors="ignore"`
   changed no file here. The loader is still fragile (M5), but this particular
   corpus does not contain a decode error.
7. **The vector formula matches the docstring.** `P @ (S @ r)` computes the
   same expression as `P @ S @ r`; all test vectors have unit norm and 2,304
   distinct values. The issue is that no experiment consumes them (H2).

---

## Bottom line

The synthetic **word-level separation** is clean: no exact word or token
leakage, valid boundaries, all components attested. The boundary numbers are
not contaminated by train/test overlap.

The evaluation does not yet support the claims in `docs/findings-v0.1.md`:

- **There is no hierarchy treatment (C0).** With the current fixed BPE,
  `flat`, `hier2` and `hier3` produce identical token IDs at every tested
  vocabulary on both corpora. The v0.1 flat/hier differences are consistent with
  the stale pair-count bug and are not reproducible with the current source.
  This is the first thing to fix; no amount of rerunning or seed averaging can
  turn an identity into a hierarchy effect.
- **Code compression and decode are not measured correctly (C1, C2).** The
  code BPC is a whitespace-normalised rate divided by raw characters, and the
  decode metric is guaranteed to be 1.0 while the codec loses all newlines and
  ~7,000 characters on the test set. The code baseline is also a single
  `_pydecimal.py` tail (H3), not a held-out code corpus.
- **Synthetic "compositional generalisation" is interpolation (H1).** Every
  test triple has all component pairs in training; the guard moves zero words
  at the default. A pair-disjoint or zero-shot split is needed before the
  boundary result is evidence for composition.
- **The central embedding claim is not tested (H2).** Vectors exist but are
  neither used nor accompanied by component ground truth in the returned
  corpus.
- **The statistics are too weak for the reported deltas (M1–M3).** Macro/micro
  conventions are mixed, BPC and F1 use different samples, roots are
  imbalanced, and seed variation is larger than the small method deltas in the
  old table. At minimum, use paired multi-seed comparisons and micro
  segmentation scores on one shared test set.
- **The output is not reproducible or self-describing (M4, M5).** No corpus
  hash, code revision, versions, holdout parameters or tokenizer budgets;
  overwritten CSVs; install-dependent corpus.

Recommended order: first fix the hierarchy treatment and re-establish that
methods actually differ (C0); fix the codec/denominator and add document-level
decode (C1/C2); redesign the synthetic holdout (H1) and add the embedding
evaluation or scope it out (H2); then rerun with a pinned corpus/revision,
micro boundary scores, a shared test sample and 5–10 paired seeds (M1–M5).
Until then, the v0.1 numbers should not be used as evidence for the ramified
hypothesis: the code metrics are invalid, the synthetic differences are not
reproducible, and the hierarchy comparison is an identity.
