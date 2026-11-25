# W1-10 — BPE edge cases vs the v0.2 v512 result

**Agent:** W1-10 (red team, read-only)
**Date:** 2026-11-22
**Python:** 3.13.13, Windows (PowerShell)
**File tested:** `ramified/bpe.py`, 278 lines, SHA-256
`73093A3E62FC66A59594F29CDB8549F4523D358DF04EBAB46E412C7C47C5B0C2`,
last write 22:02:02.
**Codec tested with:** `ramified/codec.py`, SHA-256
`3771AA4D36C092E06EA2E4A7A8A8CA86BE436857DF4B58F02D711C272A402B3B`.
**Review document:** `docs/reviews/bpe-review.md` describes a **different, older
revision**: 255 lines, SHA-256 `F4E6F2C8...`. The current file already contains
the F1 fix (the docstrings in `_update_pair_counts` and the merge loop cite
"review F1"). Line numbers in the review do not map to this revision.

**Constraint:** no project source was modified. The only project write is this
report; the test scripts ran from `%TEMP%\opencode` and wrote only there.

---

## Verdict

**The v512 result survives.** All ten edge cases are either benign, confined to
the public API, or latent; none is reachable from the v0.2 v512 configuration.
Section 4 shows the concrete input-domain checks and the row-level reproduction.

**No critical or high-severity edge case was found in this assignment.** The
review's critical F1 (heap frontier) and F5 (stale `pair_words`) findings are
already fixed in the tested revision; the minimal repro now performs all three
legal merges instead of stopping with a positive pair.

**Strongest edge finding (latent, Medium):** unknown characters still corrupt
exact decode and every subsequent boundary offset, because `<unk>` expands to
five characters. `min_freq > 1` can *manufacture* such characters by building
the alphabet after filtering. This does not touch v0.2 because all v0.2 corpora
have full alphabet coverage, but it is the defect most likely to bite a real
corpus.

**Confidence:** high (≈0.9) that no edge case corrupts the v512 numbers;
medium-high (≈0.85) that the v512 result survives entirely, limited only by the
absence of a full reference-BPE replay (Section 4.5).

---

## 1. Edge-case matrix

`v0.2 impact` answers "can this corrupt the v0.2 v512 numbers as run?" — not
"is the API defect acceptable".

| # | Case | Expected (contract / API) | Actual | Severity | v0.2 impact |
|---|---|---|---|---|---|
| E1 | Empty string | `dec(enc("")) = ""`; empty document costs no tokens | `encode("") -> [EOW]`; `decode -> "\x00</w>"`; `tokenize_chunks("") -> [EOW]`; round-trip still `True`; training on `[""]` raises `ValueError` | Low | **No**: 0 empty documents/chunks in all corpora |
| E2 | One-character words | Works; no internal boundaries | Works; `"a"` may merge with EOW into one token; `word_boundaries("a") -> []`; chunk codec round-trips | None | **No**: one-char chunk `"."` occurs in v512 training and works |
| E3 | Unknown characters | `dec(enc("aZc")) = "aZc"` (after EOW strip); boundaries `[1, 2]` | `"a<unk>c"`; boundaries `[1]`; unknown expansion length 5 shifts offsets; chunk round-trip `False`; `decode_exact = 0` | **Medium** | **No**: 0 missing chars in 10 synthetic seeds and code split; all rows `decode_word = decode_doc = 1.0` |
| E4 | EOW sentinel in input | Reject, or preserve exactly | Accepted; `detokenize_chunks` replaces the literal sentinel with a space; `roundtrip_ok = False` | Low | **No**: literal EOW/NUL absent from all corpora |
| E5 | Zero merges | Works; vocabulary equals `alphabet + 2` | Works: target 6 -> actual 6, one empty level, encode/decode fine | None | **No**: v512 has 490 merges |
| E5b | Zero budget at a level, later budget positive (F7) | Later level still runs | Level loop breaks; target 20 -> actual 17; 3-level zero-middle also stops at one empty level | **Medium** (latent) | **No**: all v0.2 budgets are positive |
| E6 | Vocab smaller than alphabet | Raise, or document a lower target | Silent overrun: target 2 (and target 0) -> actual 6 | Low | **No**: target 512, actual 512 |
| E7 | `n_levels=0` | Raise `ValueError` | Default weights: `ZeroDivisionError`; `level_weights=[1.0]`: silently returns no levels and ignores the budget; `level_weights=[]` + positive budget: `ZeroDivisionError` | Low | **No**: v0.2 uses 1/2/3 levels |
| E8 | Weight-length mismatch | Raise `ValueError` | Shorter list -> `IndexError`; empty list -> `ZeroDivisionError`; extra entries silently ignored and budget under-spent | Low | **No**: v0.2 passes the exact 2/3-element vectors |
| E9 | `min_freq > 1` | Alphabet from all text before filtering | Alphabet built after filtering; `['aa aa bb'], min_freq=2 -> {'a': 0}`; `"bb" -> [unk, unk]`; decode `"<unk><unk>"`; vocab 5 vs target 30; round-trip `False` | **Medium** (latent) | **No**: every v0.2 run uses the default `min_freq=1` |
| E10 | Invalid `split` | Raise `ValueError` | Any non-`"chunks"` value silently behaves as `"words"`; on `"a,a b.b"` the rules/vocab differ from chunk training (10 vs 12), and `",+EOW"` is trained but inference emits bare `","` | Low (Medium if a future result script forgets `split="chunks"`) | **No**: both v0.2 drivers pass `split="chunks"` explicitly |

Severity distribution: **0 Critical, 0 High, 3 Medium (E3, E5b, E9), 6 Low,
2 None**.

---

## 2. Case-by-case evidence

All commands are run from the project root. The full battery is
`python %TEMP%\opencode\w1_10_edges.py`; the outputs below are excerpts of that
run (the complete log is `%TEMP%\opencode\w1_10_edges.log`). The script only
reads project modules.

### E1 — Empty string

```powershell
python -c "
from ramified.bpe import train_tokenizer
from ramified.codec import roundtrip_ok, tokenize_chunks
t = train_tokenizer(['abc abc abd'], 20)
print('encode:', t.encode(''))
print('decode:', repr(t.decode(t.encode(''))))
print('encode eow=False:', t.encode('', eow=False))
print('boundaries:', t.word_boundaries(''))
print('chunk ids:', tokenize_chunks('', t))
print('roundtrip:', roundtrip_ok('', t))
try: train_tokenizer([''], 20)
except Exception as e: print('train [\"\"]:', type(e).__name__, e)
"
```

```text
encode: [5]
decode: '\x00</w>'
encode eow=False: []
boundaries: []
chunk ids: [5]
roundtrip: True
train [""]: ValueError empty corpus
```

Expected: the raw API should satisfy `dec(enc("")) = ""`. Actual: the sentinel is
emitted, so only the caller-side strip (`metrics.decode_exact`,
`detokenize_chunks`) satisfies the contract. An empty document costs one token
but round-trips. Training on only the empty string is rejected cleanly; an empty
text among non-empty texts is silently ignored.

**v0.2:** synthetic train/test chunks and the code split contain **0** empty
chunks (Section 4.2). No row is affected.

### E2 — One-character words

```powershell
python -c "
from ramified.bpe import train_tokenizer
from ramified.codec import roundtrip_ok, tokenize_chunks
t = train_tokenizer(['a a b c ab ab'], 15)
print('rules:', [[(m.left, m.right, m.new) for m in l] for l in t.levels])
print('encode a:', t.encode('a'))
print('decode a:', repr(t.decode(t.encode('a'))))
print('plain decode:', repr(t.decode(t.encode('a', eow=False))))
print('boundaries:', t.word_boundaries('a'))
print('chunk roundtrip:', tokenize_chunks('a b', t), roundtrip_ok('a b', t))
"
```

```text
rules: [[(1, 4, 5), (0, 4, 6), (0, 5, 7), (2, 4, 8)]]
encode a: [6]
decode a: 'a\x00</w>'
plain decode: 'a'
boundaries: []
chunk roundtrip: [6, 5] True
```

Expected: a one-character word is legal, has no internal boundary, and may be
one token when the `(char, EOW)` merge exists. Actual: exactly that. No defect.

**v0.2:** the synthetic corpus contains one-character chunks (`"."`), and the
v512 tokenizer learns `. + EOW`; `decode_doc` is 1.0 on all 40 rows.

### E3 — Unknown characters

```powershell
python -c "
from ramified.bpe import train_tokenizer
from ramified.codec import detokenize_chunks, roundtrip_ok, tokenize_chunks
from ramified.metrics import decode_exact, mean_boundary_f1
t = train_tokenizer(['abc abc abc'], 20)
print('alphabet:', t.char_to_id)
print('ids:', t.encode('aZc'))
print('decode:', repr(t.decode(t.encode('aZc'))))
print('boundaries:', t.word_boundaries('aZc'))
print('lengths:', [(t.expansions[n.symbol], len(t.expansions[n.symbol])) for n in t.encode_nodes('aZc')])
W = type('W', (), {'word': 'aZc', 'boundaries': (1, 2)})
print('decode_exact:', decode_exact(t, [W]))
print('boundary P/R/F1:', mean_boundary_f1(t, [W]))
print('chunk roundtrip:', repr(detokenize_chunks(tokenize_chunks('aZc', t), t)), roundtrip_ok('aZc', t))
"
```

```text
alphabet: {'a': 0, 'b': 1, 'c': 2}
ids: [0, 3, 6]
decode: 'a<unk>c\x00</w>'
boundaries: [1]
lengths: [('a', 1), ('<unk>', 5), ('c\x00</w>', 6)]
decode_exact: 0.0
boundary P/R/F1: (1.0, 0.5, 0.6666666666666666)
chunk roundtrip: 'a<unk>c' False
```

Expected: `[1, 2]` boundaries and exact decode for a word containing an
out-of-alphabet character. Actual: `<unk>` is a **five-character** expansion,
so the token after it starts at offset 6; the boundary at offset 2 is never
produced. This is review F4, still present. It is worse than a boundary metric:
the chunk codec's document round-trip also fails, and `decode_exact` is 0.

`min_freq` (E9) is a direct route to this state: filter out every word that
contains a character, and that character becomes unknown even though it was in
the corpus.

**v0.2:** no held-out word or document character is missing from the training
alphabet for any of the 10 synthetic seeds or the code split (Section 4.2). The
CSV itself is corroborating evidence: all 40 rows have `decode_word = 1.0` and
`decode_doc = 1.0`. An unknown character anywhere in a test word would make
`decode_word < 1`. The defect is latent, not active.

### E4 — Literal EOW in the input

```powershell
python -c "
from ramified.bpe import EOW, train_tokenizer
from ramified.codec import detokenize_chunks, normalize, roundtrip_ok, tokenize_chunks
text = 'x ' + EOW + ' x x'
t = train_tokenizer([text], 30, split='chunks')
ids = tokenize_chunks(text, t)
print('alphabet has NUL:', chr(0) in t.char_to_id)
print('ids:', ids)
print('decode:', repr(t.decode(ids)))
print('detokenize:', repr(detokenize_chunks(ids, t)))
print('normalize:', repr(normalize(text)))
print('roundtrip:', roundtrip_ok(text, t))
"
```

```text
alphabet has NUL: True
ids: [8, 13, 8, 8]
decode: 'x\x00</w>\x00</w>\x00</w>x\x00</w>x\x00</w>'
detokenize: 'x   x x'
normalize: 'x \x00</w> x x'
roundtrip: False
```

Expected: either reject the sentinel or preserve it. Actual: the trainer accepts
it as ordinary characters, and the detokeniser replaces every literal
`"\x00</w>"` with a space. The comment `EOW = "\u0000</w>"  # never appears in
real text` is not enforced. This is review F11.

**v0.2:** literal EOW and NUL are absent from all 10 synthetic corpora and from
the code corpus (`EOW in code == False`, `NUL in code == False`). All rows have
`decode_doc = 1.0`.

### E5 — Zero merges

```powershell
python -c "
from ramified.bpe import train_tokenizer
t = train_tokenizer(['abcd abcd'], 6)
print('vocab:', t.vocab_size, 'levels:', [len(l) for l in t.levels])
print('ids:', t.encode('abcd'))
print('decode:', repr(t.decode(t.encode('abcd'))))
t2 = train_tokenizer(['abcd abcd'], 6, n_levels=2)
print('2 levels, zero budget:', t2.vocab_size, [len(l) for l in t2.levels])
"
```

```text
vocab: 6 levels: [0]
ids: [0, 1, 2, 3, 5]
decode: 'abcd\x00</w>'
2 levels, zero budget: 6 [0]
```

Expected: zero merges is a valid tokenizer (alphabet + unk + EOW), and it is.
The only wrinkle is that `n_levels=2` reports one empty level, not two, because
the loop stops on the first empty merge list. With all budgets zero that loses
nothing. **No v0.2 impact:** v512 has 490 merges.

### E5b — Zero budget at one level, positive budget later (review F7)

This is the case the "zero merges" edge does *not* cover, and it is still broken.

```powershell
python -c "
from ramified.bpe import train_tokenizer
text = 'ab ac ad ae af ag ah ai aj ak al am an ao'
t = train_tokenizer([text], 20, n_levels=2, level_weights=[0.0, 1.0])
print('2 levels, level0=0:', t.vocab_size, [len(l) for l in t.levels])
t3 = train_tokenizer([text], 20, n_levels=3, level_weights=[0.1, 0.0, 0.9])
print('3 levels, middle=0:', t3.vocab_size, [len(l) for l in t3.levels])
"
```

```text
2 levels, level0=0: 17 [0]
3 levels, middle=0: 17 [0]
```

Expected: level 1 (or level 2) spends its positive budget. Actual: `train_bpe`
returns `[]` for the zero-budget level, `train_tokenizer` treats that as "corpus
exhausted", and breaks; the later positive budget is silently dropped. Target
20 -> actual 17. This is review F7, still present.

**v0.2:** the budgets are `[294, 196]`, `[256, 234]`, `[245, 147, 98]`, and
`[256, 160]` for the code smoke — all positive. `split_budget` allocates zero
only for a zero weight; no v0.2 vector contains one. No impact.

### E6 — Vocabulary smaller than the alphabet

```powershell
python -c "
from ramified.bpe import train_tokenizer
for target in (2, 0):
    t = train_tokenizer(['abcd abcd'], target)
    print('target', target, '-> actual', t.vocab_size, 'levels', [len(l) for l in t.levels])
"
```

```text
target 2 -> actual 6 levels [0]
target 0 -> actual 6 levels [0]
```

Expected: raise a clear error (the minimum vocabulary is `alphabet + 2`), or
document that the size is a lower bound. Actual: `total_merges` clamps to 0 and
the tokenizer is silently 4 tokens larger than requested. `verify.py` would flag
`vocab_actual != vocab_target`, but training itself succeeds. This is review F10.

**v0.2:** the `fixed_smoke` rows have `vocab_actual = 512` and every
`level_assignment` row carries `vocab = 512`; target 512 is far above
`alphabet + 2` (22 for synthetic, 96 for code). No impact.

### E7 — `n_levels=0`

```powershell
python -c "
from ramified.bpe import train_tokenizer
def show(label, fn):
    try: print(label, '->', fn())
    except Exception as e: print(label, '->', type(e).__name__, e)
show('default', lambda: train_tokenizer(['abcd abcd'], 20, n_levels=0).vocab_size)
show('weights [1.0]', lambda: (train_tokenizer(['abcd abcd'], 20, n_levels=0, level_weights=[1.0]).vocab_size,
                               [len(l) for l in train_tokenizer(['abcd abcd'], 20, n_levels=0, level_weights=[1.0]).levels]))
show('weights [] positive', lambda: train_tokenizer(['abcd abcd'], 20, n_levels=0, level_weights=[]).vocab_size)
show('weights [] zero', lambda: train_tokenizer(['abcd abcd'], 6, n_levels=0, level_weights=[]).encode('abcd'))
"
```

```text
default -> ZeroDivisionError float division by zero
weights [1.0] -> (6, [])
weights [] positive -> ZeroDivisionError integer modulo by zero
weights [] zero -> [0, 1, 2, 3, 5]
```

Expected: reject `n_levels=0`. Actual: three different behaviours. The
`level_weights=[1.0]` branch is the dangerous one: it returns a tokenizer with
no levels and silently ignores the entire merge budget (vocab 6 instead of 20).
This is review F10.

**v0.2:** `run_matrix_v2` uses 1/2/3 levels; `level_assignment` trains 1 level.
No impact.

### E8 — Weight-length mismatch

```powershell
python -c "
from ramified.bpe import train_tokenizer
text = 'ab ac ad ae af ag ah ai aj ak al am an ao'
def show(label, fn):
    try: print(label, '->', fn())
    except Exception as e: print(label, '->', type(e).__name__, e)
show('2 levels, weights [1.0]', lambda: train_tokenizer([text], 20, n_levels=2, level_weights=[1.0]).vocab_size)
show('2 levels, weights []', lambda: train_tokenizer([text], 20, n_levels=2, level_weights=[]).vocab_size)
t = train_tokenizer([text], 40, n_levels=1, level_weights=[0.5, 0.5])
print('1 level, extra weights -> vocab', t.vocab_size, 'target 40, levels', [len(l) for l in t.levels])
"
```

```text
2 levels, weights [1.0] -> IndexError list index out of range
2 levels, weights [] -> ZeroDivisionError integer modulo by zero
1 level, extra weights -> vocab 29 target 40, levels [12]
```

Expected: validate `len(level_weights) == n_levels`. Actual: a shorter list
raises an unhelpful `IndexError` at the point where the missing budget is read;
an empty list crashes in `split_budget`; extra entries are ignored, so the
remaining budget is never spent (target 40 -> actual 29; the second 12-merge
share is discarded). This is review F10 (and the empty-list branch of F6).

**v0.2:** `run_matrix_v2` passes exactly 2 or 3 weights; `level_assignment`
passes none (default equal split for 1 level). No impact.

### E9 — `min_freq > 1`

```powershell
python -c "
from ramified.bpe import train_tokenizer
from ramified.codec import detokenize_chunks, roundtrip_ok, tokenize_chunks
t = train_tokenizer(['aa aa bb'], 30, min_freq=2)
print('alphabet:', t.char_to_id, 'vocab:', t.vocab_size, 'target: 30')
print('ids bb:', t.encode('bb'))
print('decode:', repr(t.decode(t.encode('bb'))))
print('chunk roundtrip:', repr(detokenize_chunks(tokenize_chunks('bb', t), t)), roundtrip_ok('bb', t))
try: train_tokenizer(['aa aa bb'], 30, min_freq=3)
except Exception as e: print('min_freq=3:', type(e).__name__, e)
print('control min_freq=1:', train_tokenizer(['aa aa bb bb cc'], 30, min_freq=1).char_to_id)
"
```

```text
alphabet: {'a': 0} vocab: 5 target: 30
ids bb: [1, 1, 2]
decode: '<unk><unk>\x00</w>'
chunk roundtrip: '<unk><unk>' False
min_freq=3: ValueError empty corpus
control min_freq=1: {'a': 0, 'b': 1, 'c': 2}
```

Expected: the alphabet is the set of characters in the corpus, independent of
word-frequency filtering; or `min_freq` explicitly defines the domain and the
docs say so. Actual: the alphabet is built *after* the filter, so `b` disappears
entirely. Held-out words containing `b` become `<unk><unk>`, with the same
five-character boundary corruption as E3, and the vocabulary under-fills
(target 30 -> 5). Filtering everything raises `ValueError("empty corpus")`,
which is reasonable. This is review F10.

**v0.2:** `min_freq` is never passed by any experiment; the default is 1. No
impact. This is the most plausible way the E3 defect would become active in a
future real-corpus run.

### E10 — Invalid `split`

```powershell
python -c "
from ramified.bpe import train_tokenizer
def rules(t): return [[(m.left, m.right, m.new) for m in l] for l in t.levels]
b = train_tokenizer(['a,a b.b'], 30, split='bogus')
w = train_tokenizer(['a,a b.b'], 30, split='words')
c = train_tokenizer(['a,a b.b'], 30, split='chunks')
print('bogus == words rules:', rules(b) == rules(w), 'chars:', b.char_to_id == w.char_to_id)
print('bogus == chunks rules:', rules(b) == rules(c))
print('bogus rules:', rules(b))
print('chunks rules:', rules(c))
print('vocab:', b.vocab_size, c.vocab_size)
t = train_tokenizer(['a,a'], 30, split='bogus')
print('comma train expansion:', [t.expansions[i] for i in t.encode(',', eow=True)])
print('comma inference:', t.char_to_id[','], repr(t.expansions[t.char_to_id[',']]))
"
```

```text
bogus == words rules: True chars: True
bogus == chunks rules: False
bogus rules: [[(2, 5, 6), (3, 5, 7), (0, 5, 8), (1, 5, 9)]]
chunks rules: [[(0, 2, 6), (1, 3, 7), (2, 6, 8), (3, 7, 9), (8, 5, 10), (9, 5, 11)]]
vocab: 10 12
comma train expansion: [',\x00</w>']
comma inference: 1 ','
```

Expected: raise `ValueError` for a typo. Actual: anything other than exactly
`"chunks"` falls back to `"words"`. This silently trains punctuation with an
EOW suffix while `tokenize_normalized` emits punctuation bare (review F2/F10);
on the sample the merge set and vocabulary differ from chunk training.

**v0.2:** `experiments/run_matrix_v2.py:153` and
`experiments/level_assignment.py:94` pass `split="chunks"` explicitly, so no
v0.2 CSV row is produced by the fallback. The default-words call sites are:
`experiments/run_matrix.py` (v0.1 matrix), `experiments/ablate_level1.py`
(v0.1-era ablation), and `experiments/verify.py` (a diagnostic that then reports
the chunk codec). Those are not v0.2 result producers, but `verify.py`'s
"current codec" report trains the wrong tokenizer and should not be cited as a
v0.2 check.

---

## 3. Review status against the tested revision

The review is precise about the older revision but stale about the current one:

- **F1, critical — fixed.** The minimal repro now produces
  `[(1,0,3,10), (0,2,4,4), (3,2,5,1)]` and final words `[[5], [3], [4]]`; no
  positive pair is left behind. The old buggy result was two merges and
  `pair_counts == {(0,2): 4}` with an empty heap.
- **F5, medium (stale `pair_words`) — addressed.** `_update_pair_counts` now
  discards the word index on a decrement and removes zero-count keys. The
  performance claim is not re-benchmarked here.
- **F2, F3, F4, F6, F7, F9, F10, F11 — still present** in the tested revision.
  This assignment exercised F3 (E1), F4 (E3), F7 (E5b), F10 (E6–E9), and F11
  (E4). F2 is visible through E10. F6 is visible through E8's empty-weights
  crash; unnormalised/negative weights were not re-tested here.
- **F8, F9 — not re-tested** (performance and strict-level semantics are out of
  scope for W1-10).

The hash/mtime mismatch matters for citation: quoting `bpe-review.md`'s
"critical F1" against the current results would be wrong; the current revision
contains the fix and the v512 artifacts post-date it.

---

## 4. Does the v512 result survive?

### 4.1 Provenance

| Artifact | Timestamp | Relationship to `bpe.py` |
|---|---|---|
| `ramified/bpe.py` (SHA `73093A…`) | 22:02:02 | tested revision |
| `results/fixed_smoke/results.csv` | 22:04:07 | after the last `bpe.py` write |
| `results/level_assignment_v512_10seeds.csv` | 22:18:51 | after the last `bpe.py` write |
| `results/level_assignment_stats_v512.csv` | 22:19:12 | derived from the previous file |
| `ramified/metrics.py`, `experiments/level_assignment.py` | 22:22:42, 22:23:04 | **edited after the results** (other agents; not by W1-10) |

The `bpe.py` hash was stable throughout this review. The v512 CSVs were
therefore produced with the revision tested here. The later edits to
`metrics.py`/`level_assignment.py` are a separate provenance caveat: the current
evaluator reports macro/micro boundary fields and treats an empty prediction as
precision 0, whereas the CSV has the older `boundary_precision`/`recall`
columns. `boundary_f1` is still the macro F1 alias, and macro F1 is unchanged by
that precision change for these words (all have two true boundaries, so an empty
prediction scores F1 0 under both conventions). A literal rerun will not
reproduce the `boundary_precision` column byte-for-byte. The F1 columns
reproduce on all 40 rows (Section 4.4); bpc is verified exactly for seed 42 and
its published paired statistics recompute from the CSV (Section 4.3).

### 4.2 Input-domain coverage (the edge cases are unreachable)

Commands: `python %TEMP%\opencode\w1_10_coverage.py` and
`python %TEMP%\opencode\w1_10_edges.py` (last section). Results:

```text
SYNTHETIC (10 seeds, 42-51):
  train_chars=20 for every seed
  missing held-out word chars: [] for every seed
  missing document chars:      [] for every seed
  empty chunks:                0 / 0 for every seed
  literal EOW:                 False for every seed
  literal NUL:                 False for every seed

CODE (90/10 split of load_code_corpus(), chunk training):
  train chars=94, test chars=74, missing=[]
  test chunks with an unknown char: 0 / 3164 (0.0000%)
  EOW in code: False; NUL in code: False
  empty chunks train/test: 0 / 0
  one-char test chunks: 16 types
```

Mapping to the matrix:

| Case | v512 synthetic | v512 code smoke |
|---|---|---|
| E1 empty | no empty chunks/documents | no empty chunks |
| E2 one-char | `"."` occurs and works | 16 one-char chunk types |
| E3 unknown | 0 missing chars | 0/3164 unknown chunks |
| E4 EOW literal | absent | absent |
| E5 zero merges | 490 merges | 416 merges |
| E5b zero budget | budgets `[294,196]` | `[256,160]`, `[250,166]`, `[208,125,83]` |
| E6 vocab < alphabet | 512 > 22 | 512 > 96 |
| E7 `n_levels=0` | 1/2/3 | 1/2/3 |
| E8 weight mismatch | exact 2/3 vectors | exact 2/3 vectors |
| E9 `min_freq` | default 1 | default 1 |
| E10 invalid split | explicit chunks | explicit chunks |

### 4.3 CSV self-evidence

`python -c "...csv integrity..."` over the 40 v512 rows:

```text
rows 40, seeds [42..51], modes [flat, frequency, morph, random_closed]
all decode_word == 1.0: True
all decode_doc  == 1.0: True
any non-finite metric: []
flat vs frequency metric mismatches: []   # same rules, same tokenization
level sizes: flat (490,0); frequency/morph (294,196);
             random_closed (294,196)/(293,197)/(292,198) after closure
```

The `decode_word = 1.0` / `decode_doc = 1.0` equalities are direct evidence that
no E3/E4 defect fired: a single unknown character or literal sentinel in any
held-out word/document would drop one of them below 1.0. The absence of
`NaNs` rules out the empty-document bpc path.

Pair statistics also reproduce exactly from the CSV with
`python -m experiments.analyze_levels --csv ... --out %TEMP%\w1_10_stats_recomputed.csv`:
6/6 comparisons match `results/level_assignment_stats_v512.csv` numerically
(0 mismatches; mean/t/CI/W-L-T all identical). So the published `+0.0092 F1`,
`−0.0258 bpc` numbers are a correct summary of the CSV.

### 4.4 Current-code reproduction

`python %TEMP%\opencode\w1_10_repro_fast.py`, read-only, in memory:

```text
seed 42: flat v=512, l1=490
exact provenance: 489 keys
weighted provenance: 489 keys, identical=True
seed 42 frequency:     all 7 checked fields OK (n_l1, n_l2, tok/ch, bpc, F1, dW, dD)
seed 42 morph:         all 7 checked fields OK
seed 42 random_closed: all 7 checked fields OK
F1-only, seeds 42-51:
  seed 42  flat=0.866507523  frequency=0.866507523  morph=0.870630787  random_closed=0.875723380
  seed 43  flat=0.801085069  frequency=0.801085069  morph=0.810807292  random_closed=0.801707176
  seed 44  flat=0.914192708  frequency=0.914192708  morph=0.923828125  random_closed=0.888353588
  seed 45  flat=0.893504051  frequency=0.893504051  morph=0.903486690  random_closed=0.891608796
  seed 46  flat=0.802893519  frequency=0.802893519  morph=0.803038194  random_closed=0.794884673
  seed 47  flat=0.869068287  frequency=0.869068287  morph=0.878211806  random_closed=0.842129630
  seed 48  flat=0.894343171  frequency=0.894343171  morph=0.909939236  random_closed=0.885383598
  seed 49  flat=0.925549769  frequency=0.925549769  morph=0.940089699  random_closed=0.923367229
  seed 50  flat=0.890248843  frequency=0.890248843  morph=0.901041667  random_closed=0.889221644
  seed 51  flat=0.884765625  frequency=0.884765625  morph=0.893518519  random_closed=0.885923032

F1 mismatches: 0 (40/40 rows)
```

`weighted provenance` is an independent re-implementation that walks each
unique training word once and multiplies by its frequency; it is asserted
identical to the real `collect_provenance` on seed 42 (489/489 keys, exact
counts). **F1 reproduces on all 10 seeds and all 4 modes: 40/40 rows.** The
seed-42 full-row reproduction matches `tokens_per_char`, `bits_per_char`,
`boundary_f1`, `decode_word`, `decode_doc`, `n_l1`, `n_l2` exactly for all three
assigned modes, including morph `bpc = 2.3484206602229314` and random_closed
`bpc = 2.349558098164956`; the flat row equals the frequency row by the CSV
identity checked in Section 4.3. The other nine seeds were F1-only in the fast
run; their `bits_per_char` was not recomputed row-by-row, but the published
paired bpc statistics recompute exactly from the CSV (Section 4.3).

**F1 status on the tested revision.** The minimal repro from the review now
gives three merges and a fully merged word list; the flat v512 tokenizer learns
the full 490-merge budget; and flat vs frequency produce identical token
counts, bpc and F1 on all 10 seeds. A wrong merge order could in principle still
exist beyond the minimal case, but the specific early-stop/invisible-pair
failure is gone, and the v512 artifacts were generated after that fix.

### 4.5 What would change this verdict

The edge cases cannot reach v512, so the only remaining risks are (a) a
training-order defect not covered by the minimal repro or by F1 equality
between flat and frequency, and (b) artifact provenance. The single check that
would most change the verdict is an **independent reference-BPE replay of all
490 merges on each of the 10 v512 seeds**, comparing the merge sequence, not
just F1. If any merge-order divergence remains, the v512 claim is invalid even
though the edge cases are clean. A weaker but cheaper check is a fresh
`train_tokenizer` + full `evaluate` for all 10 seeds against the CSV, which is
what Section 4.4 does for seed 42 and partially for the rest.

---

## 5. Severity summary

| Severity | Findings |
|---|---|
| **Critical** | none |
| **High** | none |
| **Medium** | E3 unknown characters (`<unk>` is five chars; decode/boundary corruption); E5b zero-budget level drops later positive budgets (F7); E9 `min_freq` removes characters from the alphabet and manufactures E3 |
| **Low** | E1 empty-string sentinel; E4 literal EOW collision; E6 vocab below alphabet silently overruns; E7 `n_levels=0` inconsistent failure; E8 weight-length mismatch unvalidated; E10 invalid `split` silently falls back to words |
| **None** | E2 one-character words; E5 zero merges |

No finding in this set corrupts the v0.2 v512 results as run. The two worth
fixing before any real-corpus run are E3/E9 and E5b (review F4/F10 and F7).

---

## 6. Commands run (read-only)

```powershell
# Full edge battery (writes only to %TEMP%)
python C:\Users\acese\AppData\Local\Temp\opencode\w1_10_edges.py

# Corpus coverage for synthetic seeds 42-51 and the code split
python C:\Users\acese\AppData\Local\Temp\opencode\w1_10_coverage.py

# Fast reproduction: seed 42 full rows, F1 for 42-51
python C:\Users\acese\AppData\Local\Temp\opencode\w1_10_repro_fast.py

# Paired statistics recomputation
python -m experiments.analyze_levels `
  --csv results/level_assignment_v512_10seeds.csv `
  --out C:\Users\acese\AppData\Local\Temp\opencode\w1_10_stats_recomputed.csv

# Minimal F1 repro
python -c "
from ramified.bpe import train_bpe
exp = {0: 'a', 1: 'y', 2: 'b'}
merges, words = train_bpe([[1,0,2],[1,0],[0,2]], [1,9,4], 5, 3, 0, exp)
print([(m.left, m.right, m.new, m.count) for m in merges])
print(words)
"
```

Expected minimal F1 output (and actual on this revision):

```text
[(1, 0, 3, 10), (0, 2, 4, 4), (3, 2, 5, 1)]
[[5], [3], [4]]
```

No project file other than this report was created or modified by W1-10.
