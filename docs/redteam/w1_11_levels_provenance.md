# W1-11 — audit of `collect_provenance` / `_walk` (levels.py)

**Agent:** W1-11 (red team, read-only)
**Date:** 2026-11-22
**Python:** 3.13.13, Windows (PowerShell), numpy 2.2.6
**Files tested**

| file | SHA-256 | last write |
|---|---|---|
| `ramified/levels.py` | `14420E923EE6C4034EDEA192DE161479772DD274406AD80DF95BC184E5CF7DED` | 22:07:03 |
| `ramified/bpe.py` | `73093A3E62FC66A59594F29CDB8549F4523D358DF04EBAB46E412C7C47C5B0C2` | 22:02:02 |
| `ramified/synthetic.py` | `E7D4E5FBB01E86C41541785F3FA277A79AB49B545741C8453D296F8C8A0B6D22` | 22:04:54 |
| `results/level_assignment_v512_10seeds.csv` | — | 22:18:51 |

The result CSV post-dates `levels.py`; the hash was stable throughout this review.

**Constraint:** no project source was modified. The only intentional project
write is this report. All probe scripts ran from `%TEMP%\opencode` and wrote
only there.

---

## Verdict

**The v512 morph result survives this audit.** On the corpus used for the
published result, `collect_provenance` is not merely plausible, it is exact:
for **489/489 word rules** the sum of intra and cross occurrences equals the
rule's training-time weighted count (`max |diff| = 0` over those 489 rules), and
**all 151,299 derivation nodes** checked in 9,216 training words span the correct
character slice, with `_walk` ending exactly at `len(word)` for every word. EOW,
node-edge boundaries, multi-morpheme nodes, duplicate occurrences, and
text-frequency weighting all pass hand-constructed parses. The published F1/bpc
values reproduce for the seeds re-run.

Two provenance defects are real but **inactive in v512-as-run**:

1. **Medium (latent): unknown characters misclassify merges in both
   directions.** `<unk>` expands to 5 characters while ground-truth boundaries
   are character offsets, so a learned rule on known characters that sits after
   an OOV character can be counted cross when it is intra and vice versa.
   Synthetic v512 has **0** missing characters on all 10 seeds, so the published
   numbers are untouched; this blocks reuse on real/OOV corpora.
2. **Low (active, inert): non-word chunks are invisible to provenance.**
   `collect_provenance` silently drops text tokens absent from `word_objects`.
   In v512 that is exactly the 8,408 `"."` tokens, so the `(".", EOW)` rule
   (rank 30, count 8,408) has no stats, receives the `_score = 0.5` default, and
   occupies a level-1 slot. It is labelled `unseen` by `level_summary`. Moving
   it to level 2 with or without a replacement leaves F1 = 0.870631 and
   bpc = 2.348421 **identical**, so the slot is inert here. It is a real
   coverage gap for punctuation-rich corpora.

A separate **Low / interpretation** finding: the raw `intra/(intra+cross)`
score is an untuned choice. Every non-punctuation rule has at least 127 weighted
occurrences, so this is not small-sample noise; nevertheless Laplace or Wilson
smoothing changes which rules fill the 43 cross-majority slots, and on seed 42
both **improve** the result (F1 0.884 / 0.872 vs 0.871 raw, frequency 0.8665).
The sign of morph − frequency is robust to these variants; the reported
magnitude is not intrinsic to "provenance" and should not be read as a maximum.

**Confidence:** high (≈0.9) that no provenance defect corrupts the v512 numbers
as run; medium (≈0.6) that the method transfers to a real corpus without the
OOV/chunk-coverage fixes.

---

## 1. What is being audited

`collect_provenance(tok, word_objects, text=None)` walks the flat derivation
forest of each word with `_walk`, and counts each binary merge node as **intra**
if no ground-truth morpheme boundary falls strictly inside the node's span and
**cross** otherwise. `stats` is keyed by `(left.symbol, right.symbol)`. With
`text`, word occurrences are weighted by their frequency in the text; without
it, each unique word counts once. `_score` returns `intra/(intra+cross)`, or
`0.5` when the pair has no stats.

The audit target is the span arithmetic:

```python
# levels.py:30-51 (abridged)
if not node.children:
    expansion = tok.expansions[node.symbol]
    if expansion == EOW:
        return start
    return start + len(expansion)

end = start
for child in node.children:
    end = _walk(child, end, tok, boundaries, stats)

if len(node.children) == 2:
    key = (left.symbol, right.symbol)
    crosses = any(start < b < end for b in boundaries)
```

Checks requested: EOW, unknown characters, multi-morpheme nodes, node-edge
boundaries, duplicate rule occurrences, text vs unique-word weighting.

---

## 2. Hand-constructed parses with known derivations

The toy tokenizers were built directly (`Tokenizer(expansions, char_to_id,
eow_id, [rules], unk_id)`) so the expected classification is known by hand.
Full script: `%TEMP%\opencode\w1_11_toy.py`; command:

```powershell
$env:PYTHONPATH=(Get-Location).Path
python -u C:\Users\acese\AppData\Local\Temp\opencode\w1_11_toy.py
```

### 2.1 EOW and spans

Word `"ab"`, rule `(a,b)`, then `(ab, EOW)`; ground truth `(1,)` (i.e. `a|b`).
`_walk` must report end = 2, not `len("ab") + len(EOW) = 7`:

```text
[B-eow-merged] stats={(0, 1): (0, 1), (4, 3): (0, 1)}
[B-eow-merged] walk end=2 (len(word)=2, len(EOW)=5)
```

Both merges cross the internal boundary: correct. Control `(b, EOW)` alone in
`"ab"` with `(1,)` gives `(1, 0)` (intra): the boundary equals the span start and
must not count. **EOW contributes length 0; span handling is correct.**

### 2.2 Boundaries exactly at node edges

`"abcd"`, boundaries `(2,)`, rules `(a,b)`, `(c,d)`, `(ab,cd)`:

```text
[A-edges] stats={(0,1):(1,0), (2,3):(1,0), (6,7):(0,1)} -> OK
[E-outer] boundaries (0,6)                -> every rule (1,0)   -> OK
```

`start < b < end` is strict, so a boundary at a node's left or right edge is
intra. That is the correct semantics for a merge that does not extend past a
morpheme edge. Passing the whole-word edges `(0, len)` changes nothing.

### 2.3 A node covering multiple morphemes

`"abcdef"`, boundaries `(2,4)`, rules `(a,b)`, `(c,d)`, `(e,f)`, `(ab,cd)`,
`(abcd,ef)`:

```text
[D-multimorph] stats={(0,1):(1,0),(2,3):(1,0),(4,5):(1,0),(8,9):(0,1),(11,10):(0,1)} -> OK
```

The three pairwise merges are intra, the two-morpheme parent and the
three-morpheme root are each counted **once** as cross. No multiplication by the
number of boundaries crossed. This matches "occurrences of the rule"; see §4.4
for the interpretation caveat.

### 2.4 Duplicate occurrences of one rule

`"abab"`, boundaries `(1,)`, rule `(a,b)` fires at positions 0-1 (crosses) and
2-3 (intra):

```text
[C-duplicate] stats={(0, 1): (1, 1)}
[C-duplicate] total occurrences counted=2 (expected 2)
```

Per-occurrence counting, not per-rule-type. Correct.

### 2.5 Frequency weighting, text vs unique words

One word `"ab"`, one rule `(a,b)`:

```text
[G-weight] text=None      stats={(0,1):(0,1)}
[G-weight] text='ab ab ab' stats={(0,1):(0,3)}
[G-weight] text='ab . ab'  stats={(0,1):(0,2)}
[G-weight] text='ab zz ab' stats={(0,1):(0,2)}
```

Occurrences are weighted literally; tokens not in `word_objects` are silently
skipped (`"."`, `"zz"`). Duplicate `Word` objects with different boundaries are
collapsed by `{w.word: w for w in word_objects}`, **last one wins**:

```text
[H-dup-objects] [W("ab",(1,)), W("ab",())] -> {(0,1):(1,0)}
[H-dup-objects] [W("ab",()), W("ab",(1,))] -> {(0,1):(0,1)}
```

This is dormant in synthetic: across seeds 42-51 there are **0 duplicate
surface forms and 0 boundary conflicts** among the 11,520 words per seed
(§3.1).

### 2.6 Unknown characters — the one span-arithmetic bug

A learned rule can never contain `unk_id`: `train_tokenizer` builds the alphabet
from the corpus, so `unk_id` never appears in a training sequence and no pair
touching it ever has a positive count. (Confirmed: `rules touching unk: []` on
seed 42.) Therefore the original toy with a `(unk,unk)` rule is unreachable;
the reachable case is a **known-character rule after an OOV character**. Its walk
span is shifted right by 4 per preceding `<unk>`, while `word.boundaries` remains
in true character coordinates.

Script `%TEMP%\opencode\w1_11_unk.py`; alphabet `a..g`, `X` unknown, one rule
`(b,c)`:

```text
[U1-false-cross]  "aXbcdefg", boundaries (1,7)
                  true span of bc = [2,4), boundary 7 is outside
                  stats = {(1,2): (0,1)}   expected {(1,2): (1,0)}
[U2-missed-cross] "aXbc",     boundaries (1,3)
                  true span of bc = [2,4), boundary 3 is inside
                  stats = {(1,2): (1,0)}   expected {(1,2): (0,1)}
[control-1] all chars known -> {(1,2): (1,0)}  OK
[control-2] all chars known -> {(1,2): (0,1)}  OK
```

Both directions occur: a true-intra node becomes cross when a later boundary's
true coordinate falls inside the shifted span, and a true-cross node becomes
intra when its interior boundary is less than the shift away from the span
start. This is stronger than review F4's "later boundaries shift": the
provenance labels themselves flip. It is **Medium, latent**: 0 missing
characters in every synthetic seed, and the v512 code smoke also has 0 unknown
chunks per W1-10, so no v0.2 row is affected. It becomes reachable through
`min_freq` (F10) or any real corpus with OOV.

---

## 3. At-scale checks (synthetic, v512, seed 42)

Command:

```powershell
python -u C:\Users\acese\AppData\Local\Temp\opencode\w1_11_scale.py 42
```

### 3.1 Input domain and coverage

```text
seed=42 v=512: rules=490 stats_text_keys=489 stats_unique_keys=489
  train word objects=9216 unique strings=9216 text tokens=128408
  chars=19 missing_from_alphabet=[]
  duplicate surface forms=0 boundary_conflicts=0
  rules with total stats < 50 (text-weighted): 1   # the punctuation rule, total 0
  weighted totals: min 0, p10 162, median 430, max 46336
```

The 19 characters are exactly the word alphabet; the 20th alphabet symbol is
`"."`, which is why the flat tokenizer has 20 chars + unk + EOW + 490 = 512.
The 0 missing characters is direct evidence for seed 42; W1-10's independent
coverage check reports 0 missing held-out/document characters for all 10 seeds.
The 8,408 skipped text tokens are exactly the `"."` chunks (`120,000` word
tokens + `8,408` dots = `128,408`). Every non-punctuation rule has at least 127
weighted occurrences (`p10 = 162`), so no rule ratio is small-sample noise.

### 3.2 Count identity against training

`MergeRule.count` is the weighted occurrence count at merge time. Every
occurrence survives as a node in the final forest, so
`sum(intra+cross) == rule.count` must hold if counting, weighting, EOW, and
span arithmetic are all right. It does:

```text
[text] stats_total vs rule.count: mismatches=1 max|diff|=8408
  rank,left,right,count,stats,exp = (30, 0, 21, 8408, 0, "'.\\x00</w>'")
rules unseen in text stats: [(30, 0, 21, 8408, "'.\\x00</w>'")]
rules unseen in unique stats: [30]
```

The single mismatch is the punctuation rule. All **489/489 word rules match
exactly** (`max |diff| = 0`). The unique-word path returns the same 489 keys
with the same single punctuation key absent; its totals are unweighted by
construction, so they are not compared to the weighted `rule.count`. This is the
strongest single piece of evidence that `collect_provenance` is a faithful
decomposition of the training counts for the corpus that produced the result.

### 3.3 Span identity against the words

For every training word, every node's expansion (with EOW stripped) was compared
to the corresponding character slice, and the top-level `_walk` end to
`len(word)`:

```text
span check: nodes=151299 bad_spans=0 bad_end=0
```

Independently, `word_boundaries` was compared to boundaries derived from
top-level node spans for all 11,520 train+test words:

```text
words checked: 11520 mismatches: 0
sample: [('subtopar', (3,6), [3,5,6]), ('adbibdois', (2,7), [2,7]), ...]
```

`("subtopar", [3,5,6])` is a genuine over-segmentation, not a span bug — the
evaluator is allowed to predict an extra boundary. The important point is that
`_walk` and `word_boundaries` agree on EOW-aware offsets for every word.

### 3.4 Text-frequency weighting vs unique words

`collect_provenance(..., text=syn.train_text)` (the published call) vs
`collect_provenance(...)` on the same 9,216 unique words, seeds 42-44:

```text
seed=42 morph[text]   L1=294 L2=196 L1=250/43/1 F1=0.870631
seed=42 morph[unique] L1=294 L2=196 L1=251/42/1 F1=0.870631
seed=42 assignment xor=0   intra-majority flips=5/490  mean|frac diff|=0.0045
seed=43 morph[text]   L1=294 L2=196 L1=255/38/1 F1=0.810807
seed=43 morph[unique] L1=294 L2=196 L1=256/37/1 F1=0.810807
seed=43 assignment xor=0
seed=44 morph[text]   L1=294 L2=196 L1=278/15/1 F1=0.923828
seed=44 morph[unique] L1=294 L2=196 L1=278/15/1 F1=0.923828
seed=44 assignment xor=0
```

The two weightings change a handful of rule-level majorities (5/490 on seed 42)
but **not the level-1 set** on any tested seed. The `L1=250/43/1` vs
`251/42/1` difference in that table is `level_summary` labelling the same 294
selected rules under the two stats, not a different set. The published choice
(text) is also the one consistent with BPE training, which weights by text
frequency; the unique-word alternative is a robustness check, not a correction.

### 3.5 The punctuation rule and the "unseen" slot

`collect_provenance` cannot see `"."` because there is no `Word` object for it,
yet the flat BPE learned `(".", EOW)` with count 8,408. `_score` returns 0.5 for
it, placing it between intra-majority and cross-majority rules, inside level 1.
Three variants on seed 42:

```text
flat           L1=490 L2=0   L1 intra/cross/unseen=250/239/1 F1=0.866508 bpc=2.353869
morph-current  L1=294 L2=196 L1 intra/cross/unseen=250/43/1  F1=0.870631 bpc=2.348421
V1-promote     L1=294 L2=196 L1 intra/cross/unseen=250/44/0  F1=0.870631 bpc=2.348421
V2-no-promote  L1=293 L2=197 L1 intra/cross/unseen=250/43/0  F1=0.870631 bpc=2.348421
```

V1 moves `(".", EOW)` to level 2 and promotes the first level-2 rule by rank
(rank 136, `ulbeen</w>`, pure cross, stats `(0,869)`); V2 moves it to level 2
without replacement. **F1 and bpc are bit-identical in all three.** The slot
theft is real but immaterial at v512.

### 3.6 Scoring-rule sensitivity (post hoc)

The score is a raw occurrence ratio. Every non-punctuation rule has at least
127 weighted occurrences, so the ratio is not noisy in the small-sample sense; however, the
43 cross-majority rules that fill the remaining level-1 slots are sensitive to
smoothing. A **best-first, dependency-closed, exactly size-294** selection using
the same stats reproduces the published raw assignment exactly, and smoothing
on the same protocol gives:

```text
published raw      L1=294 L2=196 F1=0.870631 bpc=2.348421
best-first raw     L1=294 L2=196 xor_vs_published=0   F1=0.870631 bpc=2.348421
best-first laplace L1=294 L2=196 xor_vs_published=80  F1=0.884115 bpc=2.315632
best-first wilson  L1=294 L2=196 xor_vs_published=34  F1=0.872367 bpc=2.347220
frequency control                           F1=0.866508 bpc=2.353869
```

All three provenance variants beat frequency; the raw-ratio score is the
*weakest* of the three on this seed. This does not invalidate the published
paired comparison (the protocol is held fixed across modes), but it means the
reported +0.0092 F1 / −0.0258 bpc is one point on a protocol-dependent curve,
not an intrinsic size of the morphological effect. The check is single-seed and
post hoc; it is a robustness caveat, not a bug.

### 3.7 Reproduction of the published numbers

```text
seed=42 flat F1=0.866508 bpc=2.353869; morph F1=0.870631 bpc=2.348421
seed=43 flat F1=0.801085; morph F1=0.810807
seed=44 flat F1=0.914193; morph F1=0.923828
```

These match `results/level_assignment_v512_10seeds.csv` at all printed digits
(seed 42 morph bpc `2.3484206602229314`; frequency and flat are identical, as
the CSV says). The published 10-seed paired statistics were not recomputed here
(W1-10/W1-18 did that); the provenance inputs to them are the subject of this
report.

---

## 4. Findings by severity

| # | Severity | Finding | v0.2 impact |
|---|---|---|---|
| 1 | **Medium (latent)** | Unknown characters shift `_walk` spans by 4 per `<unk>`; provenance flips in both directions (false cross and missed cross) for rules after an OOV character. | **None**: 0 missing chars in 10 synthetic seeds; code smoke also 0 unknown chunks. Blocks real-corpus reuse. |
| 2 | **Low (active, inert)** | Non-word chunks are invisible: `text` tokens missing from `word_objects` are dropped. The `(".", EOW)` rule (8,408 occurrences) gets no stats, score 0.5, and a level-1 slot; `level_summary` calls it "unseen". | **None measured**: moving it to L2 changes nothing (F1/bpc identical). Risk grows with punctuation-rich corpora. |
| 3 | **Low (design / DoF)** | `_score` uses an unsmoothed occurrence ratio; a 0-total rule gets an arbitrary 0.5. Laplace/Wilson on a size-matched closed selection change 80/34 level-1 rules and give F1 0.884/0.872 vs 0.871 raw on seed 42. | None for the sign; affects the claimed magnitude and the interpretation of "provenance" as a single principled score. |
| 4 | **Low (latent)** | `{w.word: w for w in word_objects}` silently keeps the last of duplicate surface forms with different parses; all occurrences then use that parse. Also `_walk` is recursive with depth proportional to derivation depth (not exercised here). | **None**: 0 duplicate surfaces in 10 seeds; synthetic words are ≤7 chars. Homographs in a real corpus would need a parse-aware key. |
| 5 | **None (verified)** | EOW spans; node-edge boundaries; multi-morpheme nodes; duplicate occurrences; text weighting; unique `(left,right)` rule keys; `word_boundaries`/`_walk` agreement. | — |

### 4.1 Detail — OOV (finding 1)

Root cause: `len(tok.expansions[node.symbol])` is a *presentation* length, not a
source length. `<unk>` is 5 characters, EOW is 5 (`"\u0000</w>"`). EOW is
special-cased; unk is not. The same root cause is already F4 for
`word_boundaries`; this audit adds that provenance labels, not just offsets, are
affected, and that both error directions occur.

Why the reachable toy needs a known-character rule: no rule can contain `unk_id`
because unk never occurs in training sequences. Thus a tree can never merge
across an unk; the affected nodes are those entirely after an unk. The two
reachable failure modes are shown in §2.6. The U1 false cross requires a later
boundary less than `4k` characters after the true span end, where `k` is the
number of OOV characters before the node — plausible in real words, and common
in punctuation/identifier-heavy text with several OOV characters.

Fix direction: give `Node` a source length (terminal = 1, EOW = 0, merge = sum
of children) and use it for both `_walk` and `word_boundaries`; add an assertion
that the final `_walk` end equals `len(word)` (it holds on 100% of v512 words).
This is the F4 fix already recommended in the BPE review.

### 4.2 Detail — punctuation/chunk coverage (finding 2)

`collect_provenance` documents `text` weighting but not its word-matching
policy. It builds the lookup from `word_objects` and drops every other token:
punctuation, numerals, filtered words, or any chunking mismatch between the
training text and the boundary-annotated word list. At v512 this is exactly one
rule, and the slot experiment shows it is inert. At a real corpus it can be a
large fraction of the merge budget, and `_score`'s 0.5 default is then doing
real work with no provenance. A safer contract is to require that every
encoded chunk in `text` is represented (raise on misses), or to give non-word
chunks an explicit rule (e.g. force them to level 2 or exclude them from the
budget). The duplicate-surface last-wins behavior in the same lookup (finding 4)
should be made explicit rather than silent.

### 4.3 Detail — scoring (finding 3)

Because both the numerator and denominator are occurrence counts from the same
corpus, the ratio is well estimated; the sensitivity is not variance but the
ordering of the *cross-majority* fillers. The published assignment has 250
intra-majority rules, 1 unseen punctuation rule, and 43 cross-majority rules in
level 1 (294 = 60% of 490). All 250 intra-majority rules are selected under
every scoring variant; the differences are which cross-majority rules fill the
last 43 slots. This is why the F1 effect is robust in sign but variable in
magnitude. It also explains why `reverse` is not a size-matched control after
closure (L1 = 124): the closure demotes cross rules that depend on intra rules
in level 2. That is a design property already visible in the reported level
sizes, not a provenance bug.

### 4.4 Detail — interpretation of a multi-morpheme node (Info)

A merge spanning three morphemes contributes **one** cross occurrence, the same
as a merge spanning two. The score therefore measures "fraction of occurrences
that cross at least one boundary", not "number of boundaries crossed". That is
a defensible definition of rule provenance, but a merge that joins long spans
is not penalized more than a local boundary violation. If the intent is to
measure morphological impurity per merge, a boundary-count weight is a
different statistic. At v512 the vast majority of merges are local, so this
does not change the result.

---

## 5. Does the v512 morph result survive?

**As a numerical result, yes.** The evidence is stronger than a re-run:

- The level-1/level-2 assignment is built from provenance counts that reproduce
  the training merge counts **exactly for 489/489 word rules** (seed 42).
- Every node span used for the intra/cross decision matches the true character
  slice (**151,299 nodes, 0 errors**), and the EOW/edge/multi-morpheme/duplicate
  hand cases are all correct.
- The only active coverage gap (punctuation) has **no measurable effect** on F1
  or bpc, and the only weighting alternative (unique words) gives the **same
  level-1 set** on seeds 42-44.
- The published F1/bpc reproduce for seeds 42-44; W1-10/W1-18 independently
  reproduced all 10 seeds and the paired statistics.

**As a reusable method, not yet.** The OOV bug is latent only because the
synthetic alphabet is closed. Any real corpus with unseen characters — the
exact next step in findings-v0.2 §4.1 — will produce biased provenance, and
punctuation/numeral chunks will be invisible unless the coverage contract is
tightened. The scoring-rule sensitivity also means the paper should report the
effect as "this provenance assignment under this ratio score", not as an
intrinsic magnitude; a smoothing robustness line would be honest and, on seed
42, favourable (the raw result is conservative).

**On the specific claim in findings-v0.2 §3.2** ("an assignment to two levels by
morphological provenance improves boundary F1 and bpc at v512"): yes, for the
synthetic v512 setup, with the caveats above. Nothing in `collect_provenance` or
`_walk` invalidates the measured comparison.

---

## 6. What would change my mind

**The single check that would most change the verdict:** rerun the full 10-seed
v512 comparison with a **source-length provenance implementation** (`Node`
carries terminal = 1, EOW = 0, merge = sum of children) **and every encoded
chunk included in the stats**, keeping everything else identical. If
morph − frequency ΔF1 or Δbpc collapses or reverses under that corrected
implementation, the published result was resting on the expansion-length
coordinate artifact or on the punctuation omission. Given that OOV chars are
absent and the punctuation slot is inert on seed 42, I expect it to survive, but
that is the direct test.

A second, external-validity check would change the *scope* of the verdict even
if v512 survives: run the same paired design on a real morphologically annotated
corpus with natural OOV and punctuation, assert the per-rule
`stats_total == rule.count` identity there, and report OOV counts. The identity
will fail loudly if `<unk>` is present, which is exactly the signal this audit
predicts.

---

## 7. Commands and outputs

All commands from the project root. Probe scripts are in `%TEMP%\opencode`; they
import the project read-only.

```powershell
$env:PYTHONPATH=(Get-Location).Path

# hand parses: EOW, edges, multi-morpheme, duplicates, weighting
python -u C:\Users\acese\AppData\Local\Temp\opencode\w1_11_toy.py

# reachable unknown-character cases (known-char rule after an OOV char)
python -u C:\Users\acese\AppData\Local\Temp\opencode\w1_11_unk.py

# at-scale: corpus domain, count identity, span identity, published reproduction
python -u C:\Users\acese\AppData\Local\Temp\opencode\w1_11_scale.py 42

# text vs unique weighting on seeds 43, 44
python -u C:\Users\acese\AppData\Local\Temp\opencode\w1_11_lean.py 43 44

# punctuation-slot variants (seed 42)
python -u C:\Users\acese\AppData\Local\Temp\opencode\w1_11_slot.py

# size-matched dependency-closed scoring variants (seed 42)
python -u C:\Users\acese\AppData\Local\Temp\opencode\w1_11_bestfirst.py
```

Key outputs are quoted in §2-§3. The most important lines, collected:

```text
[A-edges] ... -> OK
[B-eow-merged] walk end=2 (len(word)=2, len(EOW)=5)
[C-duplicate] total occurrences counted=2 (expected 2)
[D-multimorph] ... -> OK
[U1-false-cross] stats={(1,2):(0,1)} expected {(1,2):(1,0)}
[U2-missed-cross] stats={(1,2):(1,0)} expected {(1,2):(0,1)}
[G-weight] text='ab ab ab' stats={(0,1):(0,3)}

seed=42 v=512: rules=490 stats_text_keys=489 stats_unique_keys=489
  chars=19 missing_from_alphabet=[]
  duplicate surface forms=0 boundary_conflicts=0
  [text] stats_total vs rule.count: mismatches=1 max|diff|=8408
    rank,left,right,count,stats,exp = (30, 0, 21, 8408, 0, "'.\\x00</w>'")
  span check: nodes=151299 bad_spans=0 bad_end=0
  flat: F1=0.866508 bpc=2.353869
  morph[text]: L1=294 L2=196 L1=250/43/1 F1=0.870631 bpc=2.348421
  morph[unique]: L1=294 L2=196 L1=251/42/1 F1=0.870631 bpc=2.348421
  assignment diff text vs unique: |L1_t xor L1_u|=0
  intra-majority flips=5/490 mean|frac diff|=0.0045

seed=43 flat F1=0.801085 frequency F1=0.801085 morph F1=0.810807
seed=44 flat F1=0.914193 frequency F1=0.914193 morph F1=0.923828

morph-current  F1=0.870631 bpc=2.348421
V1-promote     F1=0.870631 bpc=2.348421
V2-no-promote  F1=0.870631 bpc=2.348421

published raw      F1=0.870631 bpc=2.348421
best-first raw     xor_vs_published=0   F1=0.870631 bpc=2.348421
best-first laplace xor_vs_published=80  F1=0.884115 bpc=2.315632
best-first wilson  xor_vs_published=34  F1=0.872367 bpc=2.347220
```

No project source file was modified; the only intentional project write is this
report. (Python may have refreshed `__pycache__` on import.)
