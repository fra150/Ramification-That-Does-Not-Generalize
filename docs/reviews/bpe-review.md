# Adversarial review — `ramified/bpe.py`

**Reviewed revision:** `ramified/bpe.py`, 255 lines, SHA-256
`F4E6F2C8EA9C4C63F0AEE77CFA6AC6AA781B20953FF323F7FF2F3652F4CD2A17`
(2026-11-22). This revision includes the recently added
`split="words"|"chunks"` parameter in `train_tokenizer`. All line numbers below
refer to this revision. This review did not modify the file; the hash above
identifies the snapshot reviewed.

**Scope:** `ramified/bpe.py`; integration points are cited only where they
change the impact. `docs/ramified-v0.md` §2 and §6 are the contract being
checked (`dec(enc(x)) = x`, hierarchical levels, compression at equal
vocabulary).

**Bottom line:** one critical training bug. The pair *counts* are correct, but
the lazy heap does not contain a valid entry after a pair's count decreases
while it survives in another word. Training therefore is not greedy BPE: it can
skip the true best pair (observed: chose count 17 while a count-21 pair was
invisible) and can stop with positive-count pairs left (minimal repro: heap
empty, `pair_counts == {(0,2): 4}`). There is no double-counting bug, and no
structural hierarchy or word-boundary bug. The remaining issues are the
words-mode punctuation mismatch, the EOW/decode contract, unknown-character
boundary offsets, stale `pair_words`, `split_budget`'s non-normalised contract,
a zero-budget level dropping later levels, and quadratic encoding.

## Severity summary

| ID | Severity | Area | One-line |
|----|----------|------|----------|
| F1 | **Critical** | training | Heap frontier loses positive-count pairs; wrong merges and early stop |
| F2 | **High** | train/infer | Default `words` mode trains punctuation with EOW but emits it bare |
| F3 | Medium | EOW/decode | `decode` includes the EOW sentinel; `dec(enc(x)) = x` needs a caller-side strip |
| F4 | Medium | EOW/boundaries | Unknown characters use a 5-char `<unk>` expansion, shifting all later boundaries |
| F5 | Medium | training/perf | `pair_words` is never pruned; stale entries and scans grow without bound |
| F6 | Medium | budgets | `split_budget` only sums to `total` for normalised non-negative weights |
| F7 | Medium | budgets/levels | A zero budget at one level breaks the loop and drops later levels' budgets |
| F8 | Medium (perf) | encoding | `_apply_level` is O(n²) per word; chunk mode makes long inputs likely |
| F9 | Info / validity | hierarchy | Levels are budget-partitioned BPE stages, not strict symbol-provenance strata |
| F10 | Low | API edges | `n_levels=0`, weight-length mismatch, tiny vocab, `min_freq`, invalid `split` |
| F11 | Low | EOW | `"\u0000</w>"` is not impossible in real text; `detokenize` corrupts it |

---

## Findings

### F1 — Critical: the heap frontier drops valid pairs, so training is not greedy BPE and can stop with merges left

**Where:** `ramified/bpe.py:40-47` (`_update_pair_counts`), `:73-79`
(pop + validate), `:95` (decrement), `:108-112` (only pairs of `new_seq` are
pushed).

**What happens.** `pair_counts` is updated correctly in both directions, but
`_update_pair_counts` has no access to the heap and only *adds* index entries.
The only pushes inside the merge loop are lines 110-112, for the pairs of the
newly merged word. If a merge of pair `q` destroys an occurrence of pair `p` in
word `w`, but `p` survives in a different word not touched by `q`, then:

1. `pair_counts[p]` decreases by `f` (line 95);
2. `p` is not in `new_seq`, so line 112 never pushes its new count;
3. the heap entries for `p` still carry the old, larger count;
4. when those entries reach the top, line 75 rejects them, and `p` is now
   invisible although `pair_counts[p] > 0`.

Training then selects the next *visible* pair, which may have a lower count, or
stops completely if the heap empties. This is not the greedy BPE the design doc
describes.

**Minimal repro.** `sequences=[[1,0,2],[1,0],[0,2]]`, `freqs=[1,9,4]`, 5
merges. `(1,0)` has count 10 and merges first. That destroys one `(0,2)` in
`[1,0,2]`, leaving `(0,2)` with count 4 in `[0,2]`; no `-4` entry is pushed.
The next chosen pair is `(3,2)` with count 1, then the heap is empty while
`pair_counts` still holds `{(0,2): 4}`. Only 2 of 5 merges are produced and
`[0,2]` is left unmerged (see T1/T2).

**Evidence on less contrived data.** On a 120-word random corpus, the first
divergence from a recomputed reference is at merge 10: `train_bpe` chose
`(4,2)` count 17, while `(2,1)` had count 21 and was present in the reference
state (T3). On the synthetic corpus (`generate_synthetic(seed=42,
n_train_tokens=4000)`, 234 merges) the first divergence is at merge 171 of 234.
The result CSVs present at review time all have `vocab_actual ==
vocab_target`, so the hard early-stop variant did not fire in those particular
runs. That equality does not certify the merge order: the wrong-order variant
was observed on the same synthetic corpus at merge 171 of 234, so it is not
ruled out for those runs. A direct replay would quantify it; the minimal case
proves the early stop is reachable.

**Why it matters.** Merge ranks, the learned vocabulary, compression and
boundary metrics are all produced by a different algorithm than the one
claimed. A vocabulary may silently under-fill, and a level's unused budget may
be misattributed. It also compounds F7: if a positive-budget level drains its
heap early, `train_tokenizer` breaks and drops later levels.

**Concrete fix.** Push a fresh `(-new_count, pair)` for *every* pair whose
count changes, including decrements that leave a positive count. One shape:

```python
# pass heap into _update_pair_counts, or return touched pairs and push in caller
def _update_pair_counts(seq, delta, pair_counts, pair_words, wi, heap):
    for i in range(len(seq) - 1):
        p = (seq[i], seq[i + 1])
        new_count = pair_counts.get(p, 0) + delta
        if new_count <= 0:
            pair_counts.pop(p, None)
            pair_words.pop(p, None)      # zero-count pairs cannot reappear
        else:
            pair_counts[p] = new_count
            if delta > 0:
                pair_words[p].add(wi)
            else:
                pair_words[p].discard(wi)   # +f pass re-adds it if it survives
            heapq.heappush(heap, (-new_count, p))
```

`_update_pair_counts(new_seq, +f, ...)` may then be the only place new pairs
are pushed; remove the separate push loop at 110-112 (or keep it, duplicates
are harmless with count validation). A version counter per pair is a cleaner
long-term variant. Add the minimal regression test and a property test against
a reference BPE. Rebuilding the heap when it empties is a useful safety net but
is *not sufficient*: it would fix the early stop, not the wrong selection at
merge 10.

---

### F2 — High: default `words` mode trains punctuation with EOW but `tokenize_normalized` emits it bare

**Where:** `bpe.py:196-201` (words from the regex), `:218` (EOW appended to
every word), `:243-249` (punctuation emitted as a bare character),
`:253-255` (`detokenize` spacing). Integration:
`experiments/run_matrix.py`, `experiments/ablate_level1.py`.

**What happens.** In the default `split="words"` mode, a punctuation match such
as `","` becomes the training sequence `[comma_id, eow_id]`, so merges like
`(comma, EOW)` are learned. `tokenize_normalized` appends EOW only to `\w+`
matches; punctuation is appended with `tok.char_to_id.get(c, tok.unk_id)` and
*no* EOW. The learned punctuation+EOW merge can therefore never fire at
inference, and the same text has a different tokenisation in training and
evaluation:

```text
training tokens per word ("a", ",", "a"): [[4], [5], [4]]
training expansions: [['a\x00</w>'], [',\x00</w>'], ['a\x00</w>']]
inference ids: [4, 0, 4]
inference expansions: ['a\x00</w>', ',', 'a\x00</w>']
```

The comma+EOW token 5 is learned but never produced at inference; the comma
is emitted as bare id 0. `detokenize(tokenize_normalized("a,a b.b"))` returns `"a ,a b .b"`, not
`"a,a b.b"` (T7). The v2 path (`split="chunks"` + `codec.tokenize_chunks`) is
self-consistent and round-trips exactly (T11); the issue is the default
`words` path used by v1 and the level-1 ablation, and any code that trains one
mode and evaluates with the other (`experiments/verify.py` trains the default
words mode but reports the chunk codec).

**Why it matters.** Vocab slots and merges are spent on rules that inference
never uses. Train/test token distributions differ, which directly biases
bits-per-character and tokens-per-character. Document round-trip fails at
punctuation. On a code corpus, punctuation is frequent, so this is not a corner
case.

**Concrete fix.** Pick one policy and enforce it:

- **Recommended:** use `split="chunks"` + `codec.tokenize_chunks` for documents
  everywhere (already works), and make the pairing explicit (assert it, or make
  chunks the default for document experiments). Keep the words path for
  word-level morphology only.
- Or make `tokenize_normalized` encode every regex match with EOW
  (`ids.extend(tok.encode(s, eow=True))`), matching words training; then fix
  `detokenize` spacing, which is harder because EOW alone cannot distinguish
  “space before punctuation” from “space between words”.
- Or train punctuation without EOW in words mode (append EOW only to `\w+`
  matches), matching inference.

Add a consistency test: the tokenisation used for training must equal the one
used for evaluation on the same text, and document-level `roundtrip_ok` should
be checked in CI. Note that chunk mode changes the EOW distribution for
word-final morphology; if boundary F1 is the target, train a words-mode
tokenizer for that metric or state the trade-off.

---

### F3 — Medium: `decode` includes the EOW sentinel; the round-trip contract only holds with a caller-side strip

**Where:** `bpe.py:21` (`EOW`), `:130-131` (append EOW), `:159-160`
(`decode`), `:215` (EOW expansion).

**What happens.** `decode(encode("abc"))` returns `'abc\x00</w>'`, and
`decode(encode(""))` returns `'\x00</w>'`. `docs/ramified-v0.md` §2 states
`dec(enc(x)) = x`; the implementation only satisfies this after
`.replace(EOW, "")`, which `metrics.decode_exact` does manually
(`ramified/metrics.py:56`) and `detokenize` replaces with a space. A caller
using `decode` directly gets sentinel-laden text. `decode(encode(x,
eow=False)) == x` for in-vocabulary `x` (verified for `a`, `ab`, `abc`), but
the default is `eow=True`, and the segmentation differs from training when
EOW merges are skipped.

**Why it matters.** The headline correctness property is not true of the public
API. Empty input emits a token, and the sentinel can leak into outputs or
serialisation. Tests that compare `decode(encode(x))` to `x` fail for a reason
unrelated to the tokenizer's segmentation.

**Concrete fix.** Add `decode(ids, strip_eow=True)` (default strips) and a
`decode_raw`/`decode_tokens` used by `detokenize`, or give EOW an empty
expansion and track boundaries structurally. Separately, decide whether
`encode("", eow=True)` should return `[EOW]` or `[]`; for word-level encoding,
`[]` is more natural and also avoids the spurious empty chunk in
`codec.tokenize_chunks("")` (see the edge-case table). Document the chosen
contract and test empty, one-character and default words.

---

### F4 — Medium: unknown characters break exact decode and every subsequent boundary offset

**Where:** `bpe.py:129` (`unk_id` mapping), `:165` (length from expansion),
`:214` (`expansions[unk_id] = "<unk>"`).

**What happens.** An out-of-alphabet character becomes `unk_id`, whose
expansion is the five-character string `"<unk>"`. `word_boundaries` computes
source offsets from `len(self.expansions[symbol])`, so an unknown character
contributes 5 instead of 1 and shifts (or filters out) all later boundaries.
On a tokenizer trained on `"abc"`:

```text
word_boundaries("aZc") = [1]          # correct would be [1, 2]
decode(encode("aZc")) = "a<unk>c\x00</w>"
tokens of aZc: [('a', 1), ('<unk>', 5), ('c\x00</w>', 6)]
```

**Why it matters.** Boundary precision/recall is wrong for any test word with
an unseen character. `decode_exact` returns 0 for those words. This can happen
on a real code corpus split, and `min_freq` can cause it for characters that
are in the corpus but only in filtered words (F10).

**Concrete fix.** Give `Node` a source-length field: terminal = 1, EOW = 0,
merge = sum of children; use that in `word_boundaries` instead of expansion
length. As a cheap partial fix, make the unk expansion a single code point
(e.g., `"\uFFFD"`) so arithmetic stays correct, but that still does not recover
the character. For exact out-of-alphabet decode, add a byte/char fallback or
document that exact decode is only guaranteed over the training alphabet. Build
the alphabet from all text before `min_freq` filtering (F10). Add a test with
an unseen character.

---

### F5 — Medium: `pair_words` is never pruned; stale entries and scans grow without bound

**Where:** `bpe.py:40-47` (only `.add(wi)`; no discard on a negative update),
`:88-92` (`list(pair_words.get(pair, ()))` copy plus an O(len(seq)) `any`
scan), `:46-47` (`pair_counts` is pruned, `pair_words` is not).

**What happens.** When a merge destroys a pair in a word, `pair_counts` is
decremented (and popped at zero), but the word index remains in
`pair_words[p]`. It is only removed if `p` is later selected and the stale
`wi` is visited at line 91. Zero-count keys are never deleted, so both keys and
word-index entries grow monotonically. Instrumented after 100 merges on 300
words (T10):

```text
pair_words keys: 674   actual live pair keys: 415
pair_words (pair,word) entries: 1588
stale keys (not in final adjacency set): 259
```

**Why it matters.** This is not a correctness bug: the `any` check filters
stale words, and the counts themselves were consistent in 100/100 random runs.
It is a performance and memory trap: the loop copies potentially large sets
and scans words that no longer contain the pair, which is exactly the hot path
of training. On large corpora the index can dominate memory.

**Concrete fix.** Maintain `pair_words` exactly. On the negative pass, discard
`wi` from every old adjacent pair and re-add it on the positive pass for every
new adjacent pair (the current decrement/increment structure already visits
both sets). Remove zero-count keys from both `pair_counts` and `pair_words`;
zero-count pairs cannot reappear because merges are monotone (symbols are never
split). Then the `any` check at line 90 becomes a debug assertion rather than
load-bearing filtering. This also makes `list(pair_words[pair])` small and
removes the stale-scan overhead.

---

### F6 — Medium: `split_budget` violates its “sums exactly to total” contract for non-normalised, zero, negative or empty weights

**Where:** `bpe.py:176-187`.

**What happens.** The docstring promises an exact sum. For non-negative weights
that sum to 1 this is true: 2000/2000 random normalised cases passed, including
small budgets (total 0..50), with non-negative outputs. Outside that input
domain it fails:

| Input | Output | Problem |
|---|---|---|
| `split_budget(10, [2.0, 2.0])` | `[20, 20]`, sum 40 | `remainder` is negative; `range(remainder)` does nothing |
| `split_budget(5, [])` | `ZeroDivisionError` | `order[i % 0]` at line 186 |
| `split_budget(5, [0.0, 0.0])` | `[3, 2]` | Budget assigned to zero-weight levels |
| `split_budget(10, [1.5, -0.5])` | `[15, -5]` | Negative budget for level 1; level 0 over-allocated |
| `split_budget(10, [0.1, 0.1])` | `[5, 5]` | Weights sum to 0.2; cycling extras is not proportional |

The `for i in range(remainder)` loop also becomes an O(total) loop when
weights sum well below 1, so a bad weight vector can be a denial-of-service
input.

**Why it matters.** `level_weights` is caller-supplied. A malformed vector can
crash training, silently overrun the vocabulary, or hand a negative budget to a
level (which `train_bpe` treats as “no merges”). The exact-sum failure for
weights that sum to more than 1 is a direct contradiction of the docstring.

**Concrete fix.**

```python
import math

def split_budget(total, weights):
    if not isinstance(total, int) or total < 0:
        raise ValueError("total must be a non-negative int")
    if not weights:
        if total:
            raise ValueError("empty weights with positive total")
        return []
    if any(not isinstance(w, (int, float)) or not math.isfinite(w) or w < 0
           for w in weights):
        raise ValueError("weights must be finite and non-negative")
    s = sum(weights)
    if s == 0:
        if total:
            raise ValueError("all weights are zero")
        return [0] * len(weights)
    raw = [total * w / s for w in weights]      # normalise
    out = [math.floor(x) for x in raw]
    rem = total - sum(out)                      # 0 <= rem < len(weights)
    order = sorted(range(len(raw)), key=lambda i: raw[i] - out[i], reverse=True)
    for i in order[:rem]:
        out[i] += 1
    return out
```

Also validate `len(level_weights) == n_levels` in `train_tokenizer` (F10).
Add unit tests for unnormalised, empty, all-zero and negative weights.

---

### F7 — Medium: a zero budget at one level makes `train_tokenizer` break and drop all later levels' budgets

**Where:** `bpe.py:226-234`, especially `:233-234`.

**What happens.** `train_bpe(..., n_merges=0, ...)` returns `[]` without
touching the sequences. `train_tokenizer` interprets *any* empty merge list as
“nothing left” and breaks, so a later level that still has a positive budget is
never run. This happens whenever a level is allocated zero merges, which is
easy with small vocabularies or skewed weights:

```text
# target 7, alphabet 4, total merges 1, weights [0.4, 0.6]
split_budget(1, [0.4, 0.6]) == [0, 1]     # level 0 gets 0
levels: [0]                                # level 1 never runs
vocab actual: 6                            # target was 7; one merge lost

# n_levels=3, weights [0.1, 0.0, 0.9], target 30, alphabet 8, total 20
split_budget(20, [0.1, 0.0, 0.9]) == [2, 0, 18]
levels: [2, 0]                             # level 2 never runs
vocab actual: 12                           # target was 30
# manual continuation: level 2 learns 6 of its 18-merge budget

# n_levels=2, weights [0.0, 1.0], total 22
levels: [0]                                # everything lost
vocab actual: 8                            # target was 30
```

**Why it matters.** Hierarchical tokenizers silently under-fill the vocabulary;
`experiments/verify.py` flags `vocab_actual != vocab_target`; comparisons “at
equal vocabulary” become invalid. It also compounds F1: a positive-budget level
that drains its heap early triggers the same break and drops later budgets.

**Concrete fix.** Only stop when there is genuinely nothing left to merge:

```python
if not merges and budgets[level] > 0:
    break
```

If `budgets[level] == 0`, append an empty level and continue; the next level
gets the same sequences and can spend its budget. Alternatively skip
zero-budget levels explicitly. Add a regression test with total merges 1 and
weights `[0.4, 0.6]`, asserting `levels == [[], [1]]` (or at least
`vocab_actual == 7`).

---

### F8 — Medium (performance): `_apply_level` is O(n²) per word; chunk mode makes long inputs likely

**Where:** `bpe.py:136-154` (full rescan per merge, lines 142-147), called once
per word per level from `:132-133`; `tokenize_normalized` at `:245-246` and the
chunk codec call it per word/chunk.

**What happens.** Every merge rescans the whole node list to find the minimum
rank; a length-`n` word with `n/2` merges does roughly `n²/2` pair lookups.
Measured with a two-rule tokenizer on `"a" * n` (T9):

```text
n= 500  tokens= 125  seconds= 0.0871
n= 1000 tokens= 250  seconds= 0.3241
n= 2000 tokens= 500  seconds= 1.033
n= 4000 tokens= 1000 seconds= 4.3704
```

The time grows about 4× per doubling, i.e. quadratic. Long `\w+` runs in code
(long identifiers, URLs, base64) and long chunks in `split="chunks"` mode
(minified lines without spaces) are pathological. The `Node` allocation per
character and per merge adds constant overhead, and `encode_nodes` is called
once per occurrence, with no cache for repeated words.

**Why it matters.** Tokenisation is part of every experiment; a single long
chunk can dominate runtime. Chunk mode, now used by v2, makes long chunks more
likely than word mode.

**Concrete fix.** Use a linked list of nodes plus a heap of adjacent candidate
pairs keyed by `(rank, position, version)`: pop the minimum valid rank, merge,
and push only the two newly-created neighbour pairs. This is O(n log n) per
word per level. Or apply each level's rules in rank order with a single
left-to-right scan per rule and a position index; benchmark both. Add a
long-input performance test. Optional but cheap: cache `tokenize_normalized`
results per word/chunk (`functools.lru_cache`) because the corpus is Zipfian.

Two related memory traps:

- `expansions[new_id] = expansions[a] + expansions[b]` (`:86`) stores a full
  string per merge. Repeatedly merging a long chunk stores lengths
  `2 + 3 + ... + n`, i.e. O(n²) bytes for that chunk. Store `(left, right)` on
  `MergeRule` and expand lazily in `decode`, or cap chunk length.
- The heap at `:110-112` grows with every pair of every updated word; stale
  entries are only discarded when they reach the top. Versioned entries and a
  periodic compact (or the F1/F5 rewrite) keep this bounded.

---

### F9 — Info / validity: levels are budget-partitioned BPE stages, not strict symbol-provenance strata

**Where:** `bpe.py:136-154` (`_apply_level`); training chaining at `:97-108`;
doc claim in `docs/ramified-v0.md` §2 (“each level operates on symbols produced
by the previous level”).

**Answer to the specific question: there is no structural boundary bug.**
`_apply_level` applies only the rules of the requested level. At entry, the
nodes are the previous level’s tokens (plus symbols created earlier in the
current level), so a previous-level token is a single atomic node: no later
level can merge through its internal boundary, and no level can merge across a
word boundary because EOW is always the last node. Encoding also reproduces
training segmentation: 0 mismatches in 200 random corpora × {1, 2, 3} levels
(600 configurations), and the EOW/boundary invariants held in 50 random
corpora × 3 level counts.

**What is not enforced.** `_apply_level` has no notion of provenance. Within a
level, `train_bpe` creates a new symbol, appends it to `words` (line 101/107),
and later merges in the *same* level can use it as an operand. This is standard
BPE within a level, but it violates a strict reading of the doc. Demonstrated
on a 2-level tokenizer (T4): the first id created in level 1 is 12, and level 1
contains rules `(9, 12, 15)` and `(13, 7, 17)`, i.e. rules whose operands were
created by level 1 itself. A level can therefore build arbitrarily deep trees;
the level budget does not bound depth or guarantee a granularity.

**Why it matters.** This is a design/validity caveat, not a decode or tree
bug. `hier2`/`hier3` differ from flat BPE only in where the merge budget is
spent and in the input segmentation, not in a hard structural constraint. The
level-1 budget ablation (“over-merges morphemes”) may be confounded by
within-level chaining. If the hypothesis is “hierarchy imposes structure”, the
implementation does not enforce it.

**Concrete fix.** Choose and document one semantics:

- If a level is meant to be a BPE pass over the previous level’s tokens and
  chaining within a level is intended, say so explicitly (rename “stages”,
  adjust the doc’s “each level operates on symbols produced by the previous
  level”), and interpret level budgets accordingly.
- If strict strata are intended, record the allowed symbol set at level entry
  and only merge pairs whose operands are in that set, in both training and
  `_apply_level`. Accept that some levels may then have unusable budget because
  there are fewer adjacent previous-level pairs.

Add a test that asserts the chosen semantics (either no same-level operand, or
explicitly allow it).

---

### F10 — Low: input validation and edge cases

Each item has its own `file:line`, why it matters, and fix.

| Item | Where | What / why | Fix |
|---|---|---|---|
| `n_levels=0` | `bpe.py:222-223` | `1.0 / n_levels` raises `ZeroDivisionError`; if `level_weights` is supplied, it silently returns a tokenizer with no levels and ignores the budget. | Require `n_levels >= 1`. |
| `level_weights` length mismatch | `bpe.py:224`, `:229` | `budgets[level]` raises `IndexError` when the list is shorter; extra entries are ignored. | Validate `len(level_weights) == n_levels`. |
| `vocab_size < len(chars) + 2` | `bpe.py:221` | `total_merges` clamps to 0, so actual vocab is `len(chars)+2`, silently larger than the target (target 2 → actual 4). `verify.py` flags this as a mismatch. | Raise `ValueError` with the minimum, or explicitly document that `vocab_size` is a lower target and return the actual size. |
| `min_freq` drops characters from the alphabet | `bpe.py:204-205`, `:209` | The alphabet is built after filtering. `train_tokenizer(['aa aa bb'], 30, min_freq=2)` gives `char_to_id == {'a': 0}`; `encode('bb')` → `[unk, unk]`; decode → `"<unk><unk>"`. | Build the alphabet from all words/chunks before `min_freq` filtering, or document that `min_freq` defines the domain. |
| Invalid `split` | `bpe.py:198-201` | Anything other than `"chunks"` silently falls back to words. | Validate `split in {"words", "chunks"}`. |
| `texts` as a bare string | `bpe.py:196-197` | Iterates characters, training on single characters. | Require a sequence of strings or convert a single string to `[texts]`. |
| `train_bpe` input assumptions | `bpe.py:56-65` | `len(freqs) != len(sequences)` gives `IndexError` or wrong counts; non-positive freqs create zero/negative counts; `next_id` collisions overwrite expansions. | Assert equal lengths, positive freqs, and `next_id > max id in sequences`. |
| `decode` unknown id | `bpe.py:159-160` | `KeyError` with no context. | Raise a clear `ValueError`, or support a fallback. |
| Empty string | `bpe.py:128-131` | `encode("")` returns `[EOW]`; `word_boundaries("")` returns `[]`; training on `[""]` raises `ValueError("empty corpus")` (reasonable). | Decide `[]` vs `[EOW]` for empty input; see F3. |
| One-character word | `bpe.py:128-134` | Works: `encode("a")` is `[a, EOW]` (possibly one merged token); `word_boundaries("a")` is `[]`. | No fix needed; add a regression test. |
| Zero merges | `bpe.py:221-234` | `vocab_size = len(chars)+2` works; `train_bpe` returns `[]`; `Tokenizer` has an empty level; `encode` works. | No fix needed. |
| `split_budget` total type | `bpe.py:176-187` | A float total reaches `range(remainder)` and raises `TypeError`. | Require an int (see F6). |

---

### F11 — Low: the EOW sentinel is not impossible in real text; the comment is false

**Where:** `bpe.py:21` (`EOW = "\u0000</w>"`, comment “never appears in real
text”), `:253-255` (`detokenize` replaces every occurrence).

**What happens.** `"\u0000</w>"` is a five-character string. NUL can occur in
binary or embedded data, and `"</w>"` can occur literally in HTML/XML or in a
word-list corpus. If the training text contains the sentinel, `detokenize`
turns it into a space and `decode` becomes ambiguous; the claim in the comment
is not enforced. The five-character length is also why `word_boundaries` has to
skip the last token; a one-character-out-of-band boundary would be cleaner.

**Why it matters.** Exact document round-trip is not guaranteed for arbitrary
input, which matters for a codec advertised as exact. It is unlikely for Python
source/synthetic text, hence low severity.

**Concrete fix.** Use a dedicated boundary token id whose expansion is empty
(or a single private-use code point) and track boundaries structurally, or
validate the corpus and raise if `EOW in text`. Add a test containing NUL and
`</w>`.

---

## Category-by-category verdicts

### 1. Pair-count bookkeeping (`train_bpe`)

- **Heap lazy invalidation: BUG (F1, critical).** Positive-count pairs can
  become invisible after a decrement; training can choose a lower-count pair
  or stop early. See T1–T3.
- **Stale `pair_words`: BUG (F5, medium, performance/memory).** The index is
  never pruned; 259/674 keys were stale in the instrumented run. It does not
  corrupt counts because line 90 filters, but it is a hot-path scan.
- **Double counting: NO BUG FOUND.** In 100 random corpora, the internal
  `pair_counts` matched a recount from the final words exactly (0/100
  mismatches), and the `pair_words` index had no missing (pair, word) entries
  (0/100). There were no duplicate merge pairs in 200 runs, and every merged
  pair was absent from the final words in 100/100 runs. The counting passes are
  sound; the frontier is what is broken.
- **Tie-breaking:** deterministic by `(-count, pair)` for valid entries, but
  missing entries make it non-greedy (F1).

### 2. Hierarchical level application (`_apply_level`)

- **Boundary restriction: NO BUG FOUND.** It applies only the current level’s
  rules, treats previous-level tokens as atomic, and cannot cross a word
  boundary because EOW is last. Encoding reproduced training segmentation in
  600/600 configurations.
- **Strict level semantics: CAVEAT (F9).** Training allows same-level chaining
  (demonstrated), so levels are budget-partitioned BPE stages, not strict
  strata. This is a spec/experiment-validity issue, not a tree/decode bug.
- **Budget propagation: BUG (F7).** A zero-budget level breaks the level loop
  and drops later budgets.

### 3. EOW handling (`encode_nodes`, `decode`, `word_boundaries`)

- **Last-node invariant: NO BUG FOUND.** In 50 random corpora × 3 level
  counts, the last node always contained EOW and no non-last node ever did.
  This follows structurally: EOW starts last, and every merge replaces two
  adjacent nodes with one at the left position, so an EOW-containing node can
  only remain last.
- **`word_boundaries` for in-vocabulary words: NO BUG FOUND.** It matched
  cumulative expansion lengths in all tested cases, including words where the
  last token contains EOW.
- **Decode contract: BUG (F3).** `decode` returns the sentinel; `dec(enc(x))`
  needs an external strip.
- **Unknown characters: BUG (F4).** `<unk>` is five characters, so offsets
  after it are wrong.
- **Punctuation/train-inference: BUG (F2).** Default words mode is
  inconsistent; chunk mode is consistent.
- **Sentinel collision: LOW (F11).** The comment overstates the guarantee.

### 4. `split_budget`

- **Normalised non-negative weights: NO BUG FOUND.** 2000/2000 random cases
  summed exactly to `total`, stayed non-negative, and preserved length,
  including totals 0..50 and tiny budgets. Small-budget distribution can give
  the extra unit to an earlier level on equal remainders (stable sort), which
  is arbitrary but not a contract violation.
- **Arbitrary weights: BUG (F6).** Unnormalised weights can exceed the total;
  empty weights crash; zero weights receive budget; negative weights produce
  negative budgets; weights summing below 1 cycle extras and can be O(total).

### 5. Edge cases

| Case | Result |
|---|---|
| Empty string | `encode("")` → `[EOW]`; `decode` → sentinel; `word_boundaries("")` → `[]`; `tokenize_normalized("")` → `[]`; training on `[""]` raises `ValueError("empty corpus")`. The only defect is the decode/EOW contract (F3) and a spurious EOW for an empty chunk in `codec.tokenize_chunks` (T11 note). |
| One-character word | Works; `word_boundaries` is `[]`; no crash. |
| Unknown characters | BUG (F4): decode is not identity and boundary offsets shift. |
| `vocab_size` smaller than alphabet | BUG (F10): silent overrun (target 2 → actual 4). |
| Zero merges | NO BUG: `vocab_size = len(chars)+2`; empty levels; `encode` works. The separate zero-budget-level case is F7. |
| `n_levels=0` | BUG (F10): `ZeroDivisionError`. |
| `level_weights` wrong length | BUG (F10): `IndexError`. |
| `min_freq` filtering | BUG (F10): characters from filtered words become unknown. |

### 6. Complexity and performance

- **`_apply_level`: O(n²) per word per level (F8).** Measured ~4× slowdown per
  doubling; long chunks make it visible.
- **`train_bpe` index/scans: (F5).** `list(pair_words[pair])` copies sets;
  `any` and three full-sequence passes per affected word; stale entries add
  work.
- **Heap growth:** every merge pushes every pair of every updated word; stale
  entries only leave when popped. Versioned entries/compaction recommended.
- **Expansion strings:** `expansions[new_id] = expansions[a] + expansions[b]`
  can be O(n²) bytes for repeatedly merged long chunks. Store the tree and
  expand lazily.
- **No caching:** `encode_nodes` is called per occurrence; a per-word cache
  would help on Zipfian corpora.

---

## Reproducing tests

All commands were run from the project root with Python 3.13.13 on Windows
(PowerShell). Timings are machine-dependent; the other outputs are
deterministic for the given seeds.

### T1 — Heap bug: training stops with a positive pair count

```powershell
python -c "
from ramified.bpe import train_bpe
seqs = [[1, 0, 2], [1, 0], [0, 2]]
freqs = [1, 9, 4]
exp = {0: 'a', 1: 'y', 2: 'b'}
merges, words = train_bpe(seqs, freqs, 5, 3, 0, exp)
print('merges:', [(m.left, m.right, m.new, m.count) for m in merges])
print('words:', words)
print('num merges:', len(merges), 'requested: 5')
print('expansions:', exp)
"
```

Output:

```text
merges: [(1, 0, 3, 10), (3, 2, 4, 1)]
words: [[4], [3], [0, 2]]
num merges: 2 requested: 5
expansions: {0: 'a', 1: 'y', 2: 'b', 3: 'ya', 4: 'yab'}
```

The pair `(0, 2)` has count 4 in the final word `[0, 2]` and was never merged.

### T2 — Instrumented proof: heap empty while `pair_counts` is non-empty

```powershell
python -c "
import ramified.bpe as bpe
from collections import Counter as RealCounter
counters = []
class CapturingCounter(RealCounter):
    def __init__(self, *a, **k):
        super().__init__(*a, **k); counters.append(self)
bpe.Counter = CapturingCounter
heaps = []
orig_pop = bpe.heapq.heappop
def capturing_pop(h):
    if not any(h is x for x in heaps): heaps.append(h)
    return orig_pop(h)
bpe.heapq.heappop = capturing_pop
seqs = [[1, 0, 2], [1, 0], [0, 2]]
freqs = [1, 9, 4]
exp = {0: 'a', 1: 'y', 2: 'b'}
merges, words = bpe.train_bpe(seqs, freqs, 5, 3, 0, exp)
print('merges:', [(m.left, m.right, m.new, m.count) for m in merges])
print('pair_counts after training:', dict(counters[-1]))
print('heap length after training:', len(heaps[-1]) if heaps else 'no heap captured')
print('final words:', words)
bpe.heapq.heappop = orig_pop
"
```

Output:

```text
merges: [(1, 0, 3, 10), (3, 2, 4, 1)]
pair_counts after training: {(0, 2): 4}
heap length after training: 0
final words: [[4], [3], [0, 2]]
```

This is the smoking gun: a positive count with an empty frontier.

### T3 — Reference BPE divergence: count 17 chosen while count 21 is invisible

```powershell
python -c "
import random
from collections import Counter
from ramified.bpe import train_bpe

def ref_steps(seqs, freqs, n, next_id):
    words = [list(s) for s in seqs]
    steps = []
    for _ in range(n):
        c = Counter()
        for w, f in zip(words, freqs):
            for i in range(len(w) - 1):
                c[(w[i], w[i + 1])] += f
        c = {p: v for p, v in c.items() if v > 0}
        if not c: break
        (a, b), cnt = min(c.items(), key=lambda kv: (-kv[1], kv[0]))
        steps.append(((a, b, cnt), dict(c)))
        new = next_id; next_id += 1
        for wi, w in enumerate(words):
            out = []; i = 0
            while i < len(w):
                if i < len(w) - 1 and w[i] == a and w[i + 1] == b:
                    out.append(new); i += 2
                else:
                    out.append(w[i]); i += 1
            words[wi] = out
    return steps

random.seed(0)
seqs = []; freqs = []
for _ in range(120):
    seqs.append([random.choice([0, 1, 2, 3, 4]) for _ in range(random.randint(1, 8))])
    freqs.append(random.randint(1, 5))
exp = {i: chr(97 + i) for i in range(5)}
got, _ = train_bpe(seqs, freqs, 60, 5, 0, exp)
steps = ref_steps(seqs, freqs, 60, 5)
for i, (m, (rpair, counts)) in enumerate(zip(got, steps)):
    if (m.left, m.right) != rpair[:2]:
        print('first divergence step', i)
        print(' train_bpe chose', (m.left, m.right), 'count', m.count)
        print(' reference chose  ', rpair[:2], 'count', rpair[2])
        print(' reference top-5:', sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))[:5])
        print(' count of train_bpe pair in reference state:', counts.get((m.left, m.right)))
        break
"
```

Output:

```text
first divergence step 10
 train_bpe chose (4, 2) count 17
 reference chose   (2, 1) count 21
 reference top-5: [((2, 1), 21), ((3, 1), 20), ((4, 0), 20), ((3, 4), 18), ((4, 2), 17)]
 count of train_bpe pair in reference state: 17
```

The first ten merges agree, so the reference word state is exactly the state
`train_bpe` is in at step 10; the count-21 pair is genuinely present and
invisible.

### T4 — Same-level chaining in the “hierarchical” tokenizer

```powershell
python -c "
from ramified.bpe import train_tokenizer
text = 'abcd abef cdab efcd ' * 4
t = train_tokenizer([text], 30, n_levels=2, level_weights=[0.2, 0.8])
print('chars:', t.char_to_id, 'eow:', t.eow_id)
for li, rules in enumerate(t.levels):
    print('level', li, 'n=', len(rules), [(m.left, m.right, m.new, m.rank, m.count) for m in rules])
first1 = t.eow_id + 1 + len(t.levels[0])
print('first id created in level 1:', first1)
print('level1 rules using same-level symbols:',
      [(m.left, m.right, m.new, m.rank, m.count) for m in t.levels[1] if m.left >= first1 or m.right >= first1])
"
```

Output:

```text
chars: {'a': 0, 'b': 1, 'c': 2, 'd': 3, 'e': 4, 'f': 5} eow: 7
level 0 n= 4 [(0, 1, 8, 0, 12), (2, 3, 9, 1, 12), (4, 5, 10, 2, 8), (9, 7, 11, 3, 8)]
level 1 n= 6 [(8, 7, 12, 0, 4), (8, 10, 13, 1, 4), (8, 11, 14, 2, 4), (9, 12, 15, 3, 4), (10, 11, 16, 4, 4), (13, 7, 17, 5, 4)]
first id created in level 1: 12
level1 rules using same-level symbols: [(9, 12, 15, 3, 4), (13, 7, 17, 5, 4)]
```

Level 1 uses ids 12 and 13, which were created by level 1 itself.

### T5 — EOW in decode, empty string, unknown-character boundaries

```powershell
python -c "
from ramified.bpe import train_tokenizer
t = train_tokenizer(['abc abc'], 20)
for s in ['', 'a', 'abc', 'aZc']:
    ids = t.encode(s)
    print(repr(s), 'ids', ids, 'decode', repr(t.decode(ids)))
print('abc eow=False:', repr(t.decode(t.encode('abc', eow=False))))
print('word_boundaries(aZc) =', t.word_boundaries('aZc'))
print('tokens of aZc:', [(t.expansions[n.symbol], len(t.expansions[n.symbol])) for n in t.encode_nodes('aZc')])
"
```

Output:

```text
'' ids [4] decode '\x00</w>'
'a' ids [0, 4] decode 'a\x00</w>'
'abc' ids [7] decode 'abc\x00</w>'
'aZc' ids [0, 3, 6] decode 'a<unk>c\x00</w>'
abc eow=False: 'abc'
word_boundaries(aZc) = [1]
tokens of aZc: [('a', 1), ('<unk>', 5), ('c\x00</w>', 6)]
```

### T6 — `split_budget` contract failures

```powershell
python -c "
from ramified.bpe import split_budget
print('normalized [0.6,0.4], total 10:', split_budget(10, [0.6, 0.4]), sum(split_budget(10, [0.6, 0.4])))
print('unnormalized [2.0,2.0], total 10:', split_budget(10, [2.0, 2.0]), sum(split_budget(10, [2.0, 2.0])))
print('zero weights [0,0], total 5:', split_budget(5, [0.0, 0.0]), sum(split_budget(5, [0.0, 0.0])))
print('negative [1.5,-0.5], total 10:', split_budget(10, [1.5, -0.5]))
print('tiny budget, 3 levels:', split_budget(2, [1/3, 1/3, 1/3]))
try:
    print('empty weights, total 5:', split_budget(5, []))
except Exception as e:
    print('empty weights, total 5 ->', type(e).__name__, str(e))
"
```

Output:

```text
normalized [0.6,0.4], total 10: [6, 4] 10
unnormalized [2.0,2.0], total 10: [20, 20] 40
zero weights [0,0], total 5: [3, 2] 5
negative [1.5,-0.5], total 10: [15, -5]
tiny budget, 3 levels: [1, 1, 0]
empty weights, total 5 -> ZeroDivisionError integer modulo by zero
```

### T7 — Words-mode punctuation mismatch and document detokenisation

```powershell
python -c "
from ramified.bpe import train_tokenizer, tokenize_normalized, detokenize
t = train_tokenizer(['a,a a,a a,a a,a'], 30)
print('char_to_id:', t.char_to_id)
print('levels:', [[(m.left, m.right, m.new, m.rank, m.count) for m in rules] for rules in t.levels])
words = ['a', ',', 'a']
print('training-style per word:', [t.encode(w, eow=True) for w in words])
print('training-style expansions:', [[t.expansions[i] for i in t.encode(w, eow=True)] for w in words])
print('inference ids:', tokenize_normalized('a,a', t))
print('inference expansions:', [t.expansions[i] for i in tokenize_normalized('a,a', t)])
t2 = train_tokenizer(['a,a b.b'], 30)
s = 'a,a b.b'
ids = tokenize_normalized(s, t2)
print('input:', repr(s), 'detokenize:', repr(detokenize(ids, t2)))
"
```

Output:

```text
char_to_id: {',': 0, 'a': 1}
levels: [[(1, 3, 4, 0, 8), (0, 3, 5, 1, 4)]]
training-style per word: [[4], [5], [4]]
training-style expansions: [['a\x00</w>'], [',\x00</w>'], ['a\x00</w>']]
inference ids: [4, 0, 4]
inference expansions: ['a\x00</w>', ',', 'a\x00</w>']
input: 'a,a b.b' detokenize: 'a ,a b .b'
```

### T8 — Zero-budget level drops later budgets

```powershell
python -c "
import re
from collections import Counter
from ramified.bpe import train_bpe, split_budget, train_tokenizer

t = train_tokenizer(['abcd abcd abcd'], 7, n_levels=2, level_weights=[0.4, 0.6])
print('case A target 7, actual vocab:', t.vocab_size, 'levels:', [len(r) for r in t.levels])

text = 'abcdefgh abcdefgh abcdefgh abcdefgh'
words = re.findall(r'\w+|[^\w\s]', text)
freq = Counter(words)
chars = sorted(set(''.join(freq)))
c2i = {c:i for i,c in enumerate(chars)}
unk = len(chars); eow = unk+1
exp = {i:c for c,i in c2i.items()}; exp[unk]='<unk>'; exp[eow]='E'
seqs = [[c2i[c] for c in w]+[eow] for w in freq]
fs = list(freq.values())
total = 30 - len(chars) - 2
budgets = split_budget(total, [0.1, 0.0, 0.9])
print('case B alphabet', len(chars), 'total', total, 'budgets', budgets)
t2 = train_tokenizer([text], 30, n_levels=3, level_weights=[0.1, 0.0, 0.9])
print('case B train_tokenizer levels:', [len(r) for r in t2.levels], 'vocab', t2.vocab_size, 'target 30')
next_id = eow+1
for level in range(3):
    merges, seqs = train_bpe(seqs, fs, budgets[level], next_id, level, exp)
    next_id += len(merges)
    print('case B manual level', level, 'learned', len(merges), 'budget', budgets[level])
"
```

Output:

```text
case A target 7, actual vocab: 6 levels: [0]
case B alphabet 8 total 20 budgets [2, 0, 18]
case B train_tokenizer levels: [2, 0] vocab 12 target 30
case B manual level 0 learned 2 budget 2
case B manual level 1 learned 0 budget 0
case B manual level 2 learned 6 budget 18
```

`split_budget(1, [0.4, 0.6])` is `[0, 1]`; level 0 gets zero and the loop
breaks before level 1 can spend the merge. In case B, continuing manually lets
level 2 learn 6 of its 18-merge budget, so the break loses usable merges (the
vocabulary would still be 18, not 30, because the corpus exhausts).

### T9 — `_apply_level` quadratic scaling

```powershell
python -c "
import time
from ramified.bpe import Tokenizer, MergeRule
exp = {0:'a', 1:'X', 2:'Y'}
# eow/unk ids are irrelevant here: eow=False and the input is all 'a'
rules = [MergeRule(0,0,1,0,0,1), MergeRule(1,1,2,1,0,1)]
tok = Tokenizer(exp, {'a':0}, 1, [rules], 2)
for n in [500, 1000, 2000, 4000]:
    s = 'a' * n
    t = time.perf_counter(); ids = tok.encode(s, eow=False); dt = time.perf_counter() - t
    print('n=', n, 'tokens=', len(ids), 'seconds=', round(dt, 4))
"
```

Output (this machine; absolute times vary):

```text
n= 500 tokens= 125 seconds= 0.0871
n= 1000 tokens= 250 seconds= 0.3241
n= 2000 tokens= 500 seconds= 1.033
n= 4000 tokens= 1000 seconds= 4.3704
```

### T10 — Stale `pair_words` instrumentation

```powershell
python -c "
import random
import ramified.bpe as bpe
from collections import defaultdict as real_dd
created = []
def dd(factory):
    d = real_dd(factory); created.append(d); return d
bpe.defaultdict = dd
random.seed(7)
seqs = []; freqs = []
for _ in range(300):
    seqs.append([random.choice(range(8)) for _ in range(random.randint(1, 8))])
    freqs.append(random.randint(1, 5))
exp = {i: chr(97 + i) for i in range(8)}
merges, words = bpe.train_bpe(seqs, freqs, 100, 8, 0, exp)
pw = created[-1]
actual = {(w[i], w[i+1]) for w in words for i in range(len(w)-1)}
entries = sum(len(s) for s in pw.values())
print('merges learned:', len(merges))
print('pair_words keys:', len(pw), 'actual live pair keys:', len(actual))
print('pair_words (pair,word) entries:', entries)
print('keys whose set is empty:', sum(1 for s in pw.values() if not s))
print('stale keys (not in final adjacency set):', len(set(pw) - actual))
"
```

Output:

```text
merges learned: 100
pair_words keys: 674 actual live pair keys: 415
pair_words (pair,word) entries: 1588
keys whose set is empty: 0
stale keys (not in final adjacency set): 259
```

### T11 — Chunk mode is self-consistent (the F2 fix path works)

```powershell
python -c "
from ramified.bpe import train_tokenizer
from ramified.codec import tokenize_chunks, detokenize_chunks, roundtrip_ok
text = 'a,a b.b c!'
t = train_tokenizer([text], 30, n_levels=1, split='chunks')
ids = tokenize_chunks(text, t)
print('chunk ids:', ids)
print('chunk decode:', repr(detokenize_chunks(ids, t)))
print('roundtrip exactly normalized:', roundtrip_ok(text, t))
t2 = train_tokenizer([text], 30, n_levels=1, split='bogus')
print('invalid split silently defaults, vocab:', t2.vocab_size, 'levels:', len(t2.levels))
"
```

Output:

```text
chunk ids: [14, 15, 13]
chunk decode: 'a,a b.b c!'
roundtrip exactly normalized: True
invalid split silently defaults, vocab: 14 levels: 1
```

Note: `codec.tokenize_chunks("")` currently returns `[EOW]` because
`"".split(" ") == [""]` and `encode("", eow=True)` emits EOW; the round-trip is
still `""`, but the empty document costs one token. This is the empty-string
edge of F3/F10.

---

## Suggested fix order

1. **F1 + F5 together.** Rework `_update_pair_counts` to push every count
   change and keep `pair_words` exact. Add the minimal regression and a
   reference-BPE property test. Re-run all CSVs; expect merge sets and
   compression numbers to move.
2. **F7 + F6.** Fix the zero-budget break and validate/normalise
   `split_budget`; add small-vocab hierarchy tests. This prevents silent
   vocabulary shortfalls.
3. **F2.** Choose the document tokenisation policy (chunks vs punctuation+EOW),
   make training/evaluation use the same one, and add a document round-trip
   test to CI.
4. **F3 + F4.** Make `decode` strip EOW by default, add `decode_raw`, give
   `Node` a source length (or single-code-point unk), and test empty/unknown
   inputs.
5. **F8.** Replace the O(n²) `_apply_level` scan with a rank heap plus local
   neighbour updates; add a long-chunk benchmark.
6. **F9–F11.** Document or enforce the level semantics, validate the remaining
   edge inputs, and harden/replace the EOW sentinel.
