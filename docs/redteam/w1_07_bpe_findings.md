# W1-07 — BPE review findings F2–F11: impact on v0.2 and the level-assignment experiment

**Role:** red-team subagent W1-07. **Scope:** F2–F11 of
`docs/reviews/bpe-review.md`, checked against the current tree and the four CSVs
cited by `docs/findings-v0.2.md`. No source file was modified. Probes were run
from the project root and written only under
`C:\Users\acese\AppData\Local\Temp\opencode`.

## 0. Verdict

**The v512 morph result survives F2–F11. None of F2–F11 changes a number in
the cited v0.2 CSVs.** F9 is the only finding that constrains the *claim*: the
two levels are sequential merge stages, not strict provenance strata, so the
result must be described as **morphology-aware merge scheduling / two-stage
rule assignment**, not as structural hierarchy. The v0.2 text already says
“I livelli sono etichette, non vincoli”; that sentence must be kept.

Two companion audits constrain how strongly the surviving result can be
stated. W1-20 finds that the **primary** `morph − frequency` contrast survives
full-family multiplicity correction, but the **secondary** `morph −
random_closed` F1 arm does not (full-family corrected p ≈ 0.09–0.10; the bpc
arm does survive). W1-16 finds that a 60-root generator removes the F1 effect,
and W1-20 finds that a level-1 fraction of 0.5 reverses it. The v512 result is
therefore conditional on the 120-root generator and the 60/40 budget; F2–F11
do not affect it, but it is not a general “morphology always helps” claim.

Two facts change the priority of the review:

- **F5 is already fixed in the current source.** `pair_words` is pruned on
  negative updates, and zero-count entries are removed. The review’s stale-index
  repro does not reproduce (409 keys = 409 live pairs, 0 stale entries).
- **A new control bug exists outside the reviewed file:** `random_closed` in
  `ramified/levels.py` can enqueue the same rule twice, so its level-1 size is
  292–294 instead of 294. A corrected control leaves the v512 result intact and
  slightly *strengthens* it (details in §4, N1).

Prerequisite F1 check: the current `train_bpe` matches an independent reference
BPE on the minimal repro, the review’s T3 case, 30/30 random corpora, and
`check_levels` reports **490/490 identical rules and identical tokenization for
flat vs hier2** on synthetic v512 seed 42. The v0.1 matrix remains invalid as
`docs/findings-v0.2.md` says.

## 1. Snapshots and what was actually run

| Item | Value |
|---|---|
| Review snapshot `bpe.py` SHA-256 | `F4E6F2C8…F4CD2A17` (255 lines) |
| Current `bpe.py` SHA-256 | `73093A3E62FC66A59594F29CDB8549F4523D358DF04EBAB46E412C7C47C5B0C2` (278 lines) |
| Current `codec.py` SHA-256 | `3771AA4D36C092E06EA2E4A7A8A8CA86BE436857DF4B58F02D711C272A402B3B` |
| Current `levels.py` SHA-256 | `14420E923EE6C4034EDEA192DE161479772DD274406AD80DF95BC184E5CF7DED` |
| Current `metrics.py` SHA-256 | `14DBF6B1794ABFD9802C503BBF5C9E3D6B2922FE88E93F12246866A952FA4D7D` |
| Current `experiments/level_assignment.py` SHA-256 | `167C55CC81EE373672A88D018786996716013E76B5982F40072F7731DEEB4517` |

`metrics.py` and `experiments/level_assignment.py` changed during the wider
red-team session (zero-division boundary convention and micro-boundary fields).
The published v512 CSV predates those changes and contains only the macro
`boundary_f1`; W1-17 reports no empty predicted/true boundary sets in the v512
data, so the F1 column is invariant. The micro-F1 check in §5 uses an
independent implementation, not the changed helper.

The current `bpe.py` is **not** the reviewed revision: it contains the F1
decrement-push fix (documented in `_update_pair_counts`) and the F5 pruning.
Line numbers in the review do not apply to the current file. F2, F3, F4, F6,
F7, F8, F9, F10 and F11 still reproduce on the current source; F5 does not.

The v0.2 numbers come from:

- `results/fixed_smoke/results.csv` — methods × {synthetic, code} × v512 × seed 42;
- `results/level_assignment_v512_10seeds.csv` — seeds 42–51 × v512 ×
  {flat, frequency, morph, random_closed};
- `results/level_assignment_stats_v512.csv` — paired statistics;
- `results/level_assignment.csv` — 3 seeds × {512, 1024, 2048} × 5 modes.

Every cited document-level run trains with `split="chunks"` and evaluates with
`ramified.codec.tokenize_chunks`. The level-assignment experiment trains one
**flat** tokenizer per (seed, vocab) and then only reassigns its rules to two
levels; it does not retrain a hierarchy.

## 2. Impact table

“None” means no effect on the cited CSV values. “Interpretive” means the
numbers stand but the claim must be worded differently. “API/perf” means no
numerical effect, only robustness or runtime.

| ID | Finding (short) | v512 morph | Code numbers | Synthetic numbers | API / perf | Review severity → impact on v0.2 | Fix | v512 survives? |
|---|---|---|---|---|---|---|---|---|
| F2 | Default `words` mode trains punctuation with EOW but emits punctuation bare | none — cited runs use chunks | none — chunks | none — chunks | default API is broken; `verify.py` mixes modes | High → **none** | Make chunks the document default, or align words-mode inference with training; add train/eval consistency test | **Yes** |
| F3 | `decode` includes EOW; empty input emits EOW | none — metrics strip/replace EOW | none | none | public `dec(enc(x))` contract is false without caller strip | Medium → **none** | `decode(ids, strip_eow=True)`, add `decode_raw`; decide `encode("")` | **Yes** |
| F4 | `<unk>` expansion is 5 chars; later boundary offsets shift | none — 0 unknown words in 10 seeds | none — code encoded text has no unknown chars | none | wrong boundaries/decode for unseen chars; medium for a general codec | Medium → **none** | `Node` source-length field; one-code-point unk; build alphabet before `min_freq`; fallback | **Yes** |
| F5 | `pair_words` never pruned; stale entries/scans grow | none | none | none | current source already prunes; residual heap growth is perf | Medium → **none (already fixed)** | Keep pruning; versioned heap / periodic compact for the residual | **Yes** |
| F6 | `split_budget` exact only for normalized non-negative weights | none — all cited weights normalized | none | none | malformed weights can crash, overrun, or go negative | Medium → **none** | Validate/normalize weights; reject empty/zero/negative; unit tests | **Yes** |
| F7 | Zero budget at one level breaks the loop and drops later budgets | none — all v512 budgets positive | none | none | still real for other vocab sizes/skewed weights | Medium → **none for cited; config risk** | Break only when `budget > 0`; append empty level or skip zero-budget levels | **Yes** |
| F8 | `_apply_level` is O(n²) per word | none | none | none | runtime only; roughly 4× per doubling confirmed | Medium (perf) → **none** | Rank heap + linked list, or one scan per rule; cache `encode` | **Yes** |
| F9 | Levels are budget/scheduling stages, not strict symbol-provenance strata | **interpretive only** — the measured treatment is assignment/scheduling | none — explains flat=hier identity | none — same | `docs/ramified-v0.md` §2 overclaims; v0.2 §3.1 already corrects it | Info/validity → **Medium interpretive** | Document “stages/scheduling”, or enforce strata and test the chosen semantics | **Yes, with wording** |
| F10 | Input-validation and edge cases (`n_levels=0`, weight length, tiny vocab, `min_freq`, invalid split, bare string, etc.) | none — cited configs valid | none | none | API only | Low → **none** | Add validations; tests | **Yes** |
| F11 | EOW `"\u0000</w>"` is not impossible in real text | none — sentinel absent | none | none | exact round-trip not guaranteed for arbitrary input | Low → **none** | Dedicated boundary id or validate/reject sentinel; test NUL/`</w>` | **Yes** |

**Reading the “survives” column.** “Yes” means the finding does not change the
v512 numbers. It does not mean the v512 claim is unconditional: the primary
`morph − frequency` result is multiplicity-robust, while the secondary
`morph − random_closed` F1 result is provisional under full-family correction
(W1-20), and the effect is absent at 60 roots (W1-16) and reverses at a 0.5
level-1 fraction (W1-20). Those are external-validity/statistics caveats, not
F2–F11 defects. They are collected in §6.

## 3. Finding-by-finding evidence

### F2 — words-mode punctuation mismatch (no impact on current numbers)

Reproduced on the current source:

```text
training tokens per word ("a", ",", "a"): [[a\x00</w>], [,\x00</w>], [a\x00</w>]]
inference ids: [4, 0, 4]
inference expansions: ['a\x00</w>', ',', 'a\x00</w>']
detokenize("a,a b.b"): 'a ,a b .b'
chunk roundtrip exact: True
```

The mismatch is real for the default `split="words"` path: the learned
`(comma, EOW)` rule can never fire at inference because
`tokenize_normalized` emits punctuation as a bare character and appends EOW
only to `\w+` matches. It spends vocabulary/merges on rules inference never
uses and breaks document detokenisation at punctuation.

**Impact on v0.2:** none. `run_matrix_v2` (fixed smoke) and
`level_assignment` both pass `split="chunks"` and evaluate with
`tokenize_chunks`; chunk mode is self-consistent and round-trips exactly on
normalised text. `verify.py` trains the default words mode but is not the
source of any cited number. The review’s own recommendation (chunks for
documents) is the path v0.2 already uses.

**Severity:** release-blocking for the default API, not for the v0.2 results.

### F3 — EOW in `decode` (API only)

Current output:

```text
encode('')    = [4]      decode = '\x00</w>'
encode('a')   = [0, 4]   decode = 'a\x00</w>'
encode('abc') = [7]      decode = 'abc\x00</w>'
```

The headline `dec(enc(x)) = x` is false for the public `decode` unless the
caller strips EOW. The metrics already compensate: `decode_exact` does
`.replace(EOW, "")`, `detokenize_chunks` replaces EOW with a space, and
`word_boundaries` skips the last (EOW-containing) token. No CSV value depends
on the raw `decode` contract.

**Impact on v0.2:** none. **Fix:** strip by default, add `decode_raw`, decide
the empty-input contract.

### F4 — unknown-character offsets (no impact on current corpora)

Review repro still holds:

```text
word_boundaries('aZc') = [1]          # correct would be [1, 2]
decode(encode('aZc'))  = 'a<unk>c\x00</w>'
tokens of 'aZc': [('a', 1), ('<unk>', 5), ('c\x00</w>', 6)]
```

But the cited v0.2 runs never feed an unknown character to the tokenizer:

- Synthetic, all 10 seeds: alphabet = 20 chars, test-word chars = 19, **0
  unknown test words**; every test word uses training characters.
- Code: training-chunk alphabet = 94; the only characters in the raw test text
  outside that alphabet are `\n` and space, both removed by
  `codec.normalize` before encoding. `decode_doc = 1.0` and the code
  `decode_word = 1.0` confirm no encoded unknown.
- The level-assignment test set is synthetic words only.

**Impact on v0.2:** none. For a general corpus it remains a medium codec bug;
the fix (source-length on `Node`, single-code-point unk, alphabet before
`min_freq`) is still required before release.

### F5 — stale `pair_words` is already fixed on the current source

Instrumented current `train_bpe` (300 words, 100 merges, same setup as the
review’s T10):

```text
merges=100 pair_words_keys=409 live_pairs=409 entries=433
empty_sets=0 stale_entries=0 keys_not_live=0
```

Compare the review snapshot: 674 keys, 1588 entries, 259 stale keys. The
current negative update discards the word index and pops zero-count pairs, so
the index is exact. The `any` scan and the ever-growing heap remain, but that
is the F8 performance path, not a correctness bug.

**Impact on v0.2:** none. **Do not “fix F5” again**; add the instrumentation as
a regression test. If performance matters, the remaining work is a versioned
heap / periodic compaction, not `pair_words` pruning.

### F6 — `split_budget` contract (not exercised by the cited runs)

Current failures:

```text
split_budget(10, [2.0, 2.0])  = [20, 20]   # sum 40
split_budget(5,  [0.0, 0.0])  = [3, 2]     # budget to zero-weight levels
split_budget(10, [1.5, -0.5]) = [15, -5]   # negative budget
split_budget(5,  [])          -> ZeroDivisionError
```

All cited runs use normalized weights: flat `None` (one level), hier2
`[0.6, 0.4]`, hier3 `[0.5, 0.3, 0.2]`, and `build_weights` always returns
`w1 + (1 - w1)`. Budgets match `merges_l1/l2/l3` in the CSVs exactly.

**Impact on v0.2:** none. **Fix:** validate and normalize as in the review’s
proposed implementation; add unnormalized/empty/zero/negative tests.

### F7 — zero-budget level break (not triggered at v512)

Current minimal repro still gives the review’s result:

```text
train_tokenizer(['abcd abcd abcd'], 7, n_levels=2, level_weights=[0.4, 0.6])
levels=[0]  vocab=6  target=7
```

But every cited v512 hierarchy has positive budgets:

```text
synthetic v512: hier2 [294, 196]  hier3 [245, 147, 98]  hier2_b256 [256, 234]
code v512:      hier2 [250, 166]  hier3 [208, 125, 83]  hier2_b256 [256, 160]
```

No level receives zero, so no later budget is dropped. At v256,
`hier2_b256` clamps level 1 to the whole budget and level 2 gets zero, but
level 1 already reaches the target; that configuration is not part of the
cited v0.2 numbers. The break still matters for small vocabularies and skewed
weights.

**Impact on v0.2:** none for the cited rows; keep the fix on the release list.
**Fix:** break only when `budget > 0`; append an empty level or skip
zero-budget levels; add the review’s target-7 regression.

### F8 — quadratic encoding (runtime only)

Current timings on the two-rule toy tokenizer:

```text
n= 500  tokens= 125  seconds= 0.485
n=1000  tokens= 250  seconds= 2.163
n=2000  tokens= 500  seconds= 9.742
n=4000  tokens=1000  seconds=32.372
```

About 4× per doubling; absolute times are machine-dependent and slower than
the review’s, but the quadratic growth is confirmed. A long chunk can dominate
runtime; it cannot change the token IDs. `fixed_smoke` code training completed
in ~10 s and the level-assignment flat tokenizer in ~5 s, so no cited run was
truncated or timed out.

**Impact on v0.2:** none. **Fix:** rank heap/linked list or per-rule scans;
cache repeated chunks; store merge trees lazily instead of full expansion
strings.

### F9 — levels are stages, not strata (the one interpretive caveat)

Same-level dependency counts in the v512 morph assignment (seed 42, operand
slots, so a rule with two same-level operands counts twice):

```text
morph level 1: 294 rules, 328 operand slots created by level 1 itself
morph level 2: 196 rules,  84 operand slots created by level 2 itself
```

So `_apply_level` does not enforce provenance: within a level, later rules can
consume symbols created earlier in the same level. `check_levels` on synthetic
v512 seed 42 confirms the consequence for the budget hierarchy:

```text
flat levels=[490]  hier levels=[294, 196]
common rule triples: 490 / 490
rule sequences identical
sample tokens flat/hier: 1540/1540 identical=True
```

The v0.2 reading is therefore correct: the budget hierarchy is a no-op, and
the level-assignment experiment changes **when** rules are allowed to fire
(level 1 rules all before level 2 rules), not a structural stratum constraint.
The measured v512 effect is a property of that scheduling intervention. The
v0.2 text says this (“I livelli sono etichette, non vincoli”), but
`docs/ramified-v0.md` §2 still says each level operates on symbols produced by
the previous level; that sentence is false for same-level chaining and should
be qualified or the implementation changed.

**Impact on v0.2:** numbers stand; the word “hierarchy” and the v0.1 contract
sentence need the caveat. **Severity:** medium interpretive, not invalidating.
**Fix:** document “two-stage rule scheduling” (or enforce strict strata and
accept unusable budget); add a test for the chosen semantics.

### F10 — API edge cases (not exercised)

Current outputs:

```text
n_levels=0                 -> ZeroDivisionError
level_weights length bad   -> IndexError
vocab_size below alphabet  -> actual 4 for target 2
min_freq drops chars       -> char_to_id {'a': 0}; 'bb' becomes unk
invalid split              -> silently falls back to words
empty weights              -> ZeroDivisionError
```

All cited configurations use `n_levels` 1/2/3, correct weight lengths,
`vocab_size` 512/1024/2048 (well above the alphabet), `min_freq=1`, a valid
`split`, and a one-element text list. None of these paths is on the v0.2
result path.

**Impact on v0.2:** none. **Fix:** the validations listed in the review; low
priority for the paper, worth doing before release.

### F11 — EOW sentinel collision (not present)

```text
synthetic train: NUL=0 EOW=0    synthetic test: NUL=0 EOW=0
code train:      NUL=0 EOW=0    code test:      NUL=0 EOW=0
```

The five-character sentinel is a real general-codec risk (NUL/binary,
`</w>` in HTML/XML), but it does not occur in the cited corpora. The comment
“never appears in real text” is still false as a guarantee.

**Impact on v0.2:** none. **Fix:** dedicated boundary id with empty expansion,
or validate/reject the sentinel; add a NUL/`</w>` test.

## 4. New observations outside F2–F11

### N1 — `random_closed` can enqueue a rule twice (control bug, result survives)

`ramified/levels.py` builds `dependents[dependency.new].append(rule.new)` once
per operand. When both operands are created by the **same** rule (e.g. a rule
merging `(S, S)` such as `rere`, `baba`, `otot`, `inin`), the child is appended
twice. The duplicate occupies a slot in `order[:n1]`, so the level-1 set has
fewer than `n1` unique rules.

Evidence on the 10-seed v512 CSV:

```text
seed 42: n_l1=294  duplicate slots=0
seed 44: n_l1=293  duplicate slots=1   (rule (23,23) -> 'rere')
seed 46: n_l1=293  duplicate slots=1   (rule (115,115) -> 'baba')
seed 50: n_l1=292  duplicate slots=2   (rules (92,92)->'otot', (41,41)->'inin')
```

I reimplemented `random_closed` with a deduplicated queue, holding everything
else fixed. (An alternative fix is to take unique order entries until `n1`;
that gives a different random draw, so the control rows must be regenerated
either way.) The v512 paired comparison becomes:

| metric | comparison | current CSV | corrected control |
|---|---|---:|---:|
| boundary F1 | morph − random_closed | +0.015629, t=3.81, 9/1 | **+0.016163, t=4.18, 9/1** |
| bits/char | morph − random_closed | −0.061932, t=−4.73, 0/10 | **−0.065770, t=−4.93, 0/10** |
| boundary F1 | random_closed − frequency | −0.006386, t=−1.73, 3/7 | −0.006919, t=−2.07, 3/7 |
| bits/char | random_closed − frequency | +0.036144, t=3.04, 8/2 | +0.039983, t=3.56, 8/2 |

So the control bug does **not** inflate the morph result; correcting it makes
the morph advantage slightly larger. It is a real control-integrity bug and
should be fixed, but it does not change the v512 verdict.

This agrees with two companion audits. W1-12 checks the dependency closure and
reports PASS (P1–P4; zero demotions on all ten published seeds; morph L1 is
exactly 294 in every seed). W1-14 repeats the random control with **10
independent topological orders per seed**: morph beats the 10-draw mean by
+0.0223 F1 (t = 7.23, 10/10) and −0.0887 bpc (t = −8.13, 10/10); the single
canonical draw in the CSV is conservative relative to that ensemble (only
6/100 random draws beat morph on F1, 2/100 on bpc). The control should still be
fixed for integrity, but its defect is not the source of the v512 effect.

Caveat: the correction changes the random draw sequence on seeds 44/45/46/50,
so a full rerun of the `random_closed` rows is required before publishing those
specific numbers. The `morph` and `frequency` rows are unaffected.

### N2 — `reverse` is not a size-matched control

In `results/level_assignment.csv` (3 seeds), closure shrinks the reverse
level-1 set drastically:

```text
seed 42: morph n_l1=294, reverse n_l1=124
seed 43: morph n_l1=294, reverse n_l1=141
seed 44: morph n_l1=294, reverse n_l1=154
```

The reverse rows are therefore confounded by level-1 size. The 10-seed v512
table does not include `reverse`, and the paper sentence cites only `frequency`
and `random_closed`, so this does not affect the headline. If `reverse` is
reported, match the level-1 size or state the confound.

Related revision-hygiene note: `results/level_assignment_smoke.csv` has
`random_closed` n_l1=145 and F1=0.772 at seed 42, from an older 145-rule
implementation; the current 10-seed CSV has 292–294. Do not mix those files.

### N3 — punctuation rules are absent from provenance statistics

`collect_provenance` iterates `text.split()` but maps only morphological
`Word` objects; the `.` token is skipped. The learned rule
`'.' + EOW -> '.EOW'` (rank 30, count 8408) is therefore “unseen” (score 0.5)
and always lands in level 1. It has no effect on synthetic word F1 because the
held-out word list contains no punctuation, and the rule is applied in both
levels eventually, so document tokenization is unchanged. For a real corpus
with punctuation, punctuation rules would receive an arbitrary level; include
punctuation occurrences in provenance or explicitly exclude punctuation rules.

### N4 — the morph mode is supervised (scope, not a bug)

`collect_provenance` uses the ground-truth `Word.boundaries`; `frequency` and
`random_closed` do not. The v0.2 design discloses this (“Per ogni regola si
contano le occorrenze intra-morfema e cross-morfema”), and the comparison is
fair for the question “given morphological boundaries, does provenance-aware
rule assignment help?”. It is **not** an unsupervised claim: the paper should
say “morphology-supervised assignment”, not “BPE discovers morphology”.

### N5 — rank is not the hidden explanation

For seed 42, level-1 mean/median rank is:

```text
frequency:     mean 146.5  median 146.5
morph:         mean 170.5  median 151.5
random_closed: mean 210.2  median 199.5
```

`random_closed` has *more* high-rank rules than morph yet loses to morph on
9 of 10 seeds (seed 42 is the exception: random is slightly better there), so
the effect is not simply “put high-rank rules first”. Still, `random_closed` is
not rank-matched; a rank-stratified random control (shuffle provenance labels
within rank bands) would isolate the morphological component more cleanly.

## 5. Checks that passed (why the v512 result survives)

| Check | Result |
|---|---|
| `train_bpe` vs independent reference, minimal F1 repro | identical (3 merges) |
| `train_bpe` vs reference, review T3 (120 words, 60 merges) | identical, 60/60 |
| `train_bpe` vs reference, 30 random corpora (3–6 chars, 5–40 merges) | 30/30 identical |
| `check_levels` synthetic v512 seed 42 | 490/490 identical rule triples; identical tokenization |
| `fixed_smoke` synthetic flat vs hier2/hier3 | identical `0.308828 tok/ch`, `2.353869 bpc`, `0.866508 F1` |
| `fixed_smoke` code flat vs hier2/hier3 | identical `0.468579 tok/ch`, `3.885538 bpc` |
| v512 CSV reproduction, random_closed seed 42 | F1 matches to 9 decimals (`0.875723380`) |
| micro vs macro boundary F1, 10 seeds | morph−frequency **+0.009484**, t=6.35, 10/0 (macro +0.009243, t=6.52, 10/0); morph−random **+0.020122**, t=3.92, 9/1 (macro +0.015629, t=3.81) |
| unknown characters, synthetic 10 seeds | 0 unknown words; alphabet 20, test chars 19 |
| unknown characters, code | only `\n`/space in raw test; normalised away; no encoded unknown |
| v512 budgets | all positive (see F7) |
| EOW sentinel in all four corpora | 0 occurrences |
| F5 index | 409 keys = 409 live pairs; 0 stale entries |

The micro-F1 check matters because the data review (M1) warns that the reported
macro P/R/F1 are not mutually consistent. The v512 morph effect is **larger**
under micro F1 than under the macro number in the CSV, so it is not a metric
convention artifact.

### 5.1 Companion audits (not re-run here; cited for completeness)

- **W1-11 (provenance):** for 489/489 word rules the provenance counts equal
  the training-time weighted counts; 151,299 derivation nodes span the correct
  character slices; 0 missing characters on all 10 seeds. The `.`+EOW rule is
  the only unseen rule, and moving it between levels leaves F1 and bpc
  identical (the N3 inertness).
- **W1-12 (closure):** PASS on P1–P4; no level-1 rule depends on level 2, the
  partition preserves every rule, and there are zero demotions on the ten
  published seeds. This confirms the morph L1 size of 294.
- **W1-14 (random-control ensemble):** 10 topological orders per seed; morph −
  10-draw mean is **+0.0223 F1** (t = 7.23, 10/10) and **−0.0887 bpc**
  (t = −8.13, 10/10). The single published draw is conservative relative to
  the ensemble. The control is still not neutral: it is worse than flat on
  10/10 seeds, so the flat-relative effect remains the smaller +0.0092/−0.0258.
- **W1-16 (paired statistics):** the t-test is implemented correctly; the
  primary contrast survives exact/Wilcoxon checks (p = 0.00195, 10/10 signs).
  But a **60-root** generator gives ΔF1 = +0.000087 (2/10 wins, 8 exact ties,
  p = 0.168), so the effect is not stable across generator size.
- **W1-17 (boundary F1):** the macro convention is exact for the v512 data
  (no empty predicted/true boundary sets); the F1 column is invariant to the
  zero-division change. This agrees with the direct micro/macro check above.
- **W1-18 (bpc):** the morph−frequency bpc delta is not an add-alpha,
  denominator, or token-count artifact; it is stable from α = 1.0 to 1e-6 and
  under Witten-Bell. It is a unigram-model effect, not a general compression
  effect.
- **W1-20 (multiplicity/specification curve):** the primary `morph − frequency`
  F1 and bpc contrasts survive Bonferroni/Holm over the full 24-test family
  (F1 p ≈ 0.0026; bpc p ≈ 0.0106). The secondary `morph − random_closed` F1
  arm does **not** (Bonferroni/Holm ≈ 0.0996/0.0872; adjusted CI includes 0);
  its bpc arm does (≈ 0.0260/0.0238). At `level1_fraction` 0.5 the F1 effect
  reverses (−0.01558, 4/6, p = 0.081); at 0.4 it is −0.02495 (0/10,
  p = 0.0017); 0.6 is the reported +0.00924. The “confined to v512” claim is a
  3-seed description, not a tested window. The 10-seed extension is a second
  look; a two-look Bonferroni does not change the primary qualitative result,
  but the published p-values are unadjusted for the interim look.

These companion results do not contradict the F2–F11 analysis; they narrow the
scientific claim. The BPE findings do not change the numbers, but the numbers
themselves are conditional on the synthetic generator and the 60% budget.

## 6. Adjacent data-review caveats (not F2–F11, but they bound the claim)

These do not affect the internal comparison, and none changes a cited number,
but they limit what the v512 result can be said to prove:

- **Generator size and budget fraction (companion audits).** W1-16 reports a
  60-root null (ΔF1 = +0.000087, 8/10 exact ties); W1-20 reports that a
  level-1 fraction of 0.5 reverses the F1 effect (−0.01558) and 0.4 makes it
  clearly negative (−0.02495, 0/10). The 0.6 default sits just above the
  pure-intra threshold, so this is a **construction window**, not a tested
  general mechanism.
- **Multiplicity (companion audit).** W1-20 finds the primary
  `morph − frequency` F1/bpc contrasts survive full-family correction, but the
  secondary `morph − random_closed` F1 contrast does not. Report the latter as
  supportive/provisional, not as a second confirmed effect; the bpc arm of that
  comparison does survive.
- **H1 — interpolation holdout.** The synthetic held-out words have all three
  component pairs attested in training; only the exact triple is novel. The
  boundary-F1 gain is on novel triples, not unseen morphemes.
- **H3 — code corpus.** `load_code_corpus` is the alphabetically first stdlib
  files, with `_pydecimal.py` dominating; the code numbers are a within-file
  measurement and installation-dependent, not a general code-compression
  result.
- **C1 — code bpc denominator.** v0.2 uses `len(normalize(test_text))`, so the
  code bpc is bits per **normalised** character, not raw code. Label it that
  way.
- **M2 — different test samples.** BPC uses 2,000 test-text draws (about 1,324
  distinct types in seed 42); boundary F1 uses all 2,304 held-out types. They
  can move independently; the v512 result is consistent on both, but they are
  not the same sample.
- **H4 — unigram bpc.** The bpc is a unigram model, not a general compression
  benchmark. The morph−frequency bpc delta is consistent across 10 seeds, but
  it should remain secondary to boundary F1.

## 7. What would change the verdict

The internal v0.2 result is well protected against F2–F11. The check most
likely to change the scientific verdict is **replication on a real
morphological corpus with a pair-disjoint or zero-shot-morpheme split**, using
the same paired 10-seed design, a fixed `random_closed` control, a rank-matched
random control, and a level-1 fraction sweep. If the morph advantage vanishes
there, the v512 result is a synthetic-corpus effect, not a general property of
morphology-aware merge scheduling.

Within the synthetic generator, two cheaper checks already narrow the claim:
the 60-root probe removes the F1 effect (W1-16), and the 0.5/0.4 fraction
sweep reverses it (W1-20). A further **rank-stratified random control** (shuffle
provenance labels within rank bands) would test whether the remaining effect is
specifically morphological rather than a by-product of which rank bands stay in
level 1.

## 8. Reproduction

All commands were run from the project root on Python 3.13.13 / Windows. The
probe scripts are under `C:\Users\acese\AppData\Local\Temp\opencode`
(`w1_07_probe*.py`), not in the repository. No source file was modified.

```powershell
# 1. flat vs hier2 rule identity at v512
python -m experiments.check_levels --corpus synthetic --vocab 512 --seed 42

# 2. current-source F2/F3/F4/F5/F6/F7/F8/F11 probes
python C:\Users\acese\AppData\Local\Temp\opencode\w1_07_probe.py
python C:\Users\acese\AppData\Local\Temp\opencode\w1_07_probe3.py

# 3. reference-BPE property test (minimal repro + T3 + 30 random corpora)
python C:\Users\acese\AppData\Local\Temp\opencode\w1_07_probe6.py

# 4. corrected random_closed control (10 seeds, F1 + bpc)
python C:\Users\acese\AppData\Local\Temp\opencode\w1_07_probe4.py
python C:\Users\acese\AppData\Local\Temp\opencode\w1_07_probe4b.py

# 5. micro vs macro boundary F1 (10 seeds)
python C:\Users\acese\AppData\Local\Temp\opencode\w1_07_probe7.py
```

The probe scripts contain the exact review T5/T6/T7/T8 one-liners and the
10-seed scans used above; they are temporary artifacts, not repository files.
The `check_levels` command is a repository entry point and is sufficient to
reproduce the 490/490 hierarchy identity.

Key raw numbers are in §3–§5. The current `bpe.py` hash is
`73093A3E62FC…C47C5B0C2`; the review snapshot hash
`F4E6F2C8…F4CD2A17` is older and should not be used to reason about F5.

## 9. Bottom line for the paper

- **v512 morph result:** survives F2–F11. The primary `morph − frequency`
  contrast (+0.0092 F1 / −0.0258 bpc, 10/10) is multiplicity-robust. The
  secondary `morph − random_closed` F1 contrast is supportive but provisional
  under full-family correction; its bpc arm is robust. Keep the paired 10-seed
  design, the `frequency` control, and the `random_closed` control after fixing
  N1.
- **Wording:** “on this 120-root synthetic generator, at v512 and a 60/40
  level-1 budget, morphology-supervised two-stage merge scheduling improves
  boundary F1 by ~0.9 points and bpc by ~0.026 over flat/frequency; the effect
  is absent at 60 roots, reverses at a 0.5 level-1 fraction (p = 0.081), and is
  clearly negative at 0.4; it is a no-op at v1024/2048.” Do not claim a general
  morphological advantage or that the levels are strict provenance strata.
- **Before release:** fix F2 (default words path), F3/F4 (decode/unk API), F6/F7
  (budget validation), N1 (`random_closed` duplicates), and update the v0.1
  §2 sentence about levels. F5 is already fixed and needs only a regression
  test. F8/F10/F11 are hygiene/performance.
