# W1-17 — boundary_f1 audit (v512 level-assignment result)

Read-only audit. No project source file was modified. All scripts were run from
the repository root and live in the session temp directory
(`$env:TEMP\opencode\w1_17\`).

## 0. What was audited

Source files read:

- `docs/findings-v0.2.md`
- `ramified/metrics.py`
- `ramified/bpe.py`
- `ramified/synthetic.py`
- plus `ramified/levels.py`, `ramified/codec.py`, `experiments/level_assignment.py`,
  `experiments/analyze_levels.py`, and `results/level_assignment_v512_10seeds.csv`

Source hashes at audit time (SHA-256, first 16 hex chars):

| file | hash |
|---|---|
| `ramified/metrics.py` | `14DBF6B1794ABFD9` |
| `ramified/bpe.py` | `73093A3E62FC66A5` |
| `ramified/levels.py` | `14420E923EE6C403` |
| `experiments/level_assignment.py` | `167C55CC81EE3736` |

Note: `metrics.py` changed on disk during the audit (mtime 22:22:42). The change
replaced the empty-prediction precision/recall `1.0` convention with
zero-division `0.0` and added the docstring “Macro-averaged boundary P/R/F1”.
For the v512 data this is F1-invariant: the audit found **no** test word with an
empty predicted or true boundary set. The F1 column is identical under both
versions. Only one-sided-empty P/R values would differ, and there are none here.

## 1. Reproduction

Command:

```powershell
python %TEMP%\opencode\w1_17\audit_f1.py 42
```

Key output (seed 42, v512):

```text
test_words=2304 dup_test_strings=0 train_overlap=0 train_types=9216
boundary tuples all length 2: True
gold boundaries interior: True
flat rules=490 chars=20 unk=20 eow=21 vocab=512
[flat         ] macroF1=0.866508 microF1=0.866900 macroP=0.8889 macroR=0.8594
[frequency    ] macroF1=0.866508 microF1=0.866900 macroP=0.8889 macroR=0.8594
[morph        ] macroF1=0.870631 microF1=0.870992 macroP=0.8926 macroR=0.8637
[random_closed] macroF1=0.875723 microF1=0.874809 macroP=0.8951 macroR=0.8720
```

This matches `results/level_assignment_v512_10seeds.csv` to the printed precision
(e.g. flat 0.8665075231481482, morph 0.8706307870370371,
random_closed 0.8757233796296297), and confirms `flat == frequency` exactly.
The v1024/v2048 window claim also reproduces exactly:

```text
42 1024 dF1 0.0 dbpc 0.0 dtok 0.0 n1 601 601
43 1024 dF1 0.0 dbpc 0.0 dtok 0.0 n1 601 601
44 1024 dF1 0.0 dbpc 0.0 dtok 0.0 n1 601 601
42 2048 dF1 0.0 dbpc 0.0 dtok 0.0 n1 1215 1215
...
```

## 2. Metric definition and averaging

`boundary_scores(pred, true)` de-duplicates both sides with `set`, computes
per-word precision, recall and F1, and `mean_boundary_f1` takes the **macro**
average over words. With every synthetic test word having exactly two gold
internal boundaries, this is a per-word seam-localisation score.

The micro alternative is not a threat on this corpus. Full 10-seed paired
comparison (same words, same tokenizers):

| comparison | macro Δ | t | W/L/T | micro Δ | t | W/L/T |
|---|---:|---:|---:|---:|---:|---:|
| morph − frequency | +0.009243 | 6.52 | 10/0/0 | +0.009484 | 6.35 | 10/0/0 |
| random_closed − frequency | −0.006386 | −1.73 | 3/7/0 | −0.010637 | −2.27 | 3/7/0 |
| morph − random_closed | +0.015629 | 3.81 | 9/1/0 | +0.020122 | 3.92 | 9/1/0 |

The sign and significance of every morph comparison are unchanged. The only
averaging-sensitive result is `random_closed − frequency`: macro CI includes 0
(`[−0.014735, +0.001963]`), micro CI barely excludes it
(`[−0.021221, −0.000054]`). The findings table uses macro, which is the
conservative choice for that control.

Token-weighted macro F1 (weighting each held-out word by its frequency in
`test_text`) gives morph − frequency `+0.009133`, `9/0/1`: seed 46 is an exact
tie at the test-token level.

**Verdict on averaging:** macro is a legitimate choice and the v512 conclusion is
identical under micro. The findings document should say “macro over held-out
word types”, because the current text leaves this ambiguous.

## 3. Empty predicted/true sets

Direct edge cases of the current `boundary_scores`:

```text
pred=[] true=[]       -> (1.0, 1.0, 1.0)
pred=[1] true=[]      -> (0.0, 0.0, 0.0)
pred=[] true=[1]      -> (0.0, 0.0, 0.0)
pred=[1,2] true=[1,2] -> (1.0, 1.0, 1.0)
pred=[1,2,3] true=[1,2] -> (0.667, 1.0, 0.800)
pred=[1] true=[1,2]    -> (1.0, 0.500, 0.667)
```

F1 is well behaved: only both-empty returns 1.0; any one-sided empty returns 0.0.
This is a reasonable convention.

On the v512 synthetic test set the conventions are **not exercised**:

- every gold word has exactly 2 interior boundaries (`boundary tuples all length
  2: True` for all 10 seeds), so `empty_true=0` by construction;
- `empty_pred=0` in all 92,160 (tokenizer, word) checks;
- all 2,304 test words per seed have at least one predicted boundary.

For real corpora with monomorphemic words, macro averaging over both-empty words
would give those words F1=1 and can inflate the corpus mean relative to micro.
That is a design choice to document, not a v512 bug.

## 4. Token boundaries, EOW, and unk

### 4.1 EOW and expansion offsets are correct here

`word_boundaries` walks `encode_nodes(word, eow=True)`, takes
`len(expansions[symbol])` for every node except the last, and keeps offsets
`0 < cum < len(word)`. Because EOW is structurally always the final node, the
last-node skip removes EOW and no word-final boundary is emitted.

Independent check: re-derive the boundaries from the same derivation tree with
one character per terminal and zero for EOW, ignoring expansion strings. Over
**92,160** (tokenizer, word) pairs (10 seeds × 4 modes × 2,304 words):

```text
indep_mismatch=0   eow_bad=0   unk_words=0
```

for every seed and mode. Every EOW-containing node is the last node; no EOW
appears in a non-final node. Expansion-length offsets therefore agree with
character offsets on this corpus.

### 4.2 unk offsets are still broken (known F4), and it matters for real corpora

Minimal repro:

```text
alphabet {'a': 0, 'b': 1, 'c': 2} unk_exp '<unk>' len 5
aZc ids [0, 3, 6] decode 'a<unk>c\x00</w>'
boundaries [1]              # expected internal offsets: [1, 2]
```

The 5-character `<unk>` expansion shifts every later offset by +4. On a real
seed-42 word:

```text
subtopar gold=(3, 6)   pred=[3, 5, 6]
suZtopar gold=(2, 5)   pred=[1, 2, 7]   # Z is unknown
```

The altered word gets a spurious boundary at 1 and the two gold seams are lost
(one offset is shifted past the word end and dropped). This is exactly review
F4. It does **not** affect the v512 synthetic result: the synthetic alphabet is
closed over all seeds (`test_chars_missing=[]` for seeds 42–51) and no test word
contains unk. But boundary F1 on any real corpus with OOV characters is invalid
until unk gets a source length of 1 (or a one-code-point expansion), or the
tokenizer is trained with an alphabet guaranteed to cover the evaluation text.

## 5. What boundary F1 rewards on this corpus

### 5.1 The test set is fixed-shape

Every held-out word is `prefix + root + suffix`, with exactly two gold seams.
Gold first-seam positions are only 2 or 3; second-seam positions are 5–8. There
are no monomorphemic words, no two-morpheme words, and no unseen roots/affixes:
all components are attested in training by construction. The test tests
generalisation to **unseen combinations**, not to unseen morphology or variable
morpheme counts.

Morphology-blind baselines (10-seed mean macro F1):

| strategy | F1 |
|---|---:|
| fixed seams {2,5} | 0.313 |
| fixed seams {3,7} | 0.437 |
| every 2 characters | 0.368 |
| every position | 0.414 |
| midpoint, one boundary | 0.000 |
| oracle | 1.000 |
| flat BPE v512 | 0.874 |

So the metric is not trivially gamed by a fixed split, and 0.874 is substantial.
But the fixed three-morpheme shape means the metric mostly measures seam
localisation, not morpheme count or semantic compositionality.

### 5.2 The gain is position, not over-segmentation

Aggregate predicted-boundary counts over 23,040 test words:

| mode | 1 | 2 | 3 | 4 | 5 |
|---|---:|---:|---:|---:|---:|
| flat / frequency | 1,820 | 20,342 | 811 | 67 | 0 |
| morph | 1,761 | 20,448 | 764 | 67 | 0 |
| random_closed | 1,716 | 19,854 | 1,112 | 326 | 32 |

Morph moves mass from 1-boundary and 3-boundary words to exactly 2, but only
slightly: 88.29% → 88.75% of words have exactly two predicted boundaries.
Mean tokens/char are essentially unchanged (0.303531 → 0.303568, +0.000037).
Random_closed over-segments (0.308950 tokens/char, +102 predicted boundaries per
2,304 words). The F1 gain is therefore not “more boundaries”; it is better
placement, with no token-count inflation.

The mechanism is visible in the level contents (10-seed means):

| mode | L1 intra | L1 cross | L2 intra | L2 cross |
|---|---:|---:|---:|---:|
| frequency | 197.1 | 95.9 | 65.6 | 130.4 |
| morph | 262.7 | 30.3 | 0.0 | 196.0 |
| random_closed | 211.6 | 81.0 | 51.1 | 145.3 |

Morph moves almost every majority-cross rule out of level 1: the level-1 stage
is dominated by intra-morpheme merges, and the second stage's entire 196-rule
inventory is majority-cross. The metric rewards exactly that delay. This is a real
tokenisation effect, but it is also close to the definition of the treatment.
The rule inventory is identical, but the effective grammar is not: on seed 42,
100/196 morph level-2 rules fire on the test words versus 127/196 for frequency,
so “same rules” means same set, not same usage.

Per-word comparison morph vs frequency over all 23,040 words:

```text
changed=443  improved=443  worsened=0
mean delta on changed=+0.480737  sum delta=+212.97
changed/improved by word length: 7:(62,62) 8:(79,79) 9:(200,200) 10:(102,102)
exact-seam words: morph=17439  frequency=17020  (+419)
```

All 443 changed words improve; none regress. This is a clean Pareto improvement
on the held-out combinations. It is also exactly what the treatment is built to
do.

### 5.3 The treatment and the metric use the same gold boundaries

`collect_provenance(flat, syn.train_word_objects, ...)` assigns rules to levels
using the gold morpheme boundaries of the **training** words. The metric then
scores agreement with the gold boundaries of the **held-out** words. The word
sets are disjoint (verified: `train_overlap=0`, `dup_test_strings=0` for all 10
seeds), so this is not direct leakage. But it is a supervised criterion: the
level assignment is selected on the same notion of “correct seam” that F1
measures. The F1 gain is evidence that the supervised routing generalises across
unseen word combinations; it is not evidence that the tokenizer discovers
morphology unsupervised.

The less circular outcome is bpc, and it is small: mean 2.300324 → 2.274536
bits/char, −0.025788 (−1.1%). Across the 30 non-flat rows, F1 and bpc correlate
at r = −0.844, so the two are not independent evidence; both largely track the
same tokenisation change. The findings document presents them as two outcomes,
which is acceptable, but the paper sentence should not imply two independent
confirmations. Note also that F1 uses all 2,304 held-out types, while bpc uses
the 2,000-token test sample drawn with replacement; the two are not evaluated on
identical multisets.

**Verdict on “rewards the right thing”:** it rewards correct seam placement, and
the treatment improves exactly that. The result is a valid supervised
segmentation-alignment result, not a general morphological-generalisation result.

## 6. Statistics and the “10/0/0” claim

Per-seed macro F1 deltas (morph − frequency), from the CSV:

| seed | Δ F1 | Δ bpc | note |
|---:|---:|---:|---|
| 42 | +0.004123 | −0.005449 | random_closed beats both |
| 43 | +0.009722 | −0.024135 | |
| 44 | +0.009635 | −0.029892 | |
| 45 | +0.009983 | −0.023677 | |
| 46 | **+0.000145** | **−0.000997** | one word-equivalent (≈1/3 F1 unit); effectively a tie |
| 47 | +0.009144 | −0.024805 | |
| 48 | +0.015596 | −0.049437 | |
| 49 | +0.014540 | −0.034473 | |
| 50 | +0.010793 | −0.044482 | |
| 51 | +0.008753 | −0.020530 | |

Paired statistics:

```text
F1  mean=+0.009243 sd=0.004484 t=+6.52 CI95=[+0.006036,+0.012451]
    Wilcoxon two-sided p=0.00195, W+=0
    leave-one-out means: +0.008537 .. +0.010254
bpc mean=-0.025788 sd=0.015148 t=-5.38, 10/10 negative
```

The effect is robust to leave-one-out and to a rank test. The correct prose is
“9 clear wins plus 1 negligible (seed 46)”, not “all 10 seeds improve” without
qualification. Under token-weighted F1 seed 46 is an exact tie (`9/0/1`). The
reported `10/0/0` is technically true for the macro float comparison but
overstates the substance.

## 7. Random-closed control: minor size defect, not the source of the gain

The CSV shows the random-closed control is not always size-matched:

```text
seed 44: random_closed L1/L2 = 293/197   (morph/frequency 294/196)
seed 46: random_closed L1/L2 = 293/197
seed 50: random_closed L1/L2 = 292/198
```

Cause: `assign_levels(..., mode="random_closed")` appends a child to `available`
once per occurrence of a dependency. A rule with the same non-terminal on both
sides, e.g. `(X, X) -> Y`, is appended twice when `X` is emitted; the second
occurrence consumes a prefix slot without adding a unique rule. Toy repro:

```text
order [0, 1, 1, 3, 4, 2]  n1=3  unique L1 size=2  closed=True
```

The prefix remains downward-closed (dependencies appear earlier), so the defect
is a size mismatch of 1–2 rules, not an unclosed level. It affects 3/10 seeds.

Stress test with a corrected exact-size Kahn scheduler (deduplicated `queued`
set, 5 draws per seed):

| seed | morph | source random | corrected random mean (range) |
|---:|---:|---:|---:|
| 46 | 0.803038 | 0.794885 | 0.790212 (0.780562–0.800998) |
| 50 | 0.901042 | 0.889222 | 0.884821 (0.882263–0.886863) |

Morph is above all five corrected draws on both seeds, and the source random
control is not weaker than the corrected one (it is at the high end for seed 46
and above the corrected maximum for seed 50). The size defect is real but does
not appear to inflate morph's advantage on these stress seeds; if anything the
corrected control is slightly harder to beat. (Seed 44, the third under-sized
seed, has a morph-random gap of +0.035, far larger than a 1–2-rule correction.)
With only five draws the corrected estimate is noisy (seed 46 range 0.020), so
this is a distributional check, not a precise causal estimate. Full correction
over all seeds is still the right thing to do for the paper, but it is not a
threat to the v512 conclusion.

One design caveat remains: a prefix of a random linear extension is a random
*ideal* of the dependency poset, not a uniform random closed set of size n1. It
is biased toward low-rank, low-dependency rules. This is a control-design
limitation to state, not a bug in `boundary_f1`.

## 8. Verdict on the v512 result

**Survives, numerically.** The +0.0092 macro F1 and −0.0258 bpc reproduce to the
CSV precision. Macro vs micro, empty-set conventions, EOW handling, and
expansion-derived offsets do not change the sign or significance. There is no
unk in the synthetic test, and the random-closed control's 1–2-rule size defect
does not appear to favour morph.

**Narrow the interpretation.** The result is a supervised level-routing effect on
held-out *combinations* of attested morphemes in a fixed three-morpheme synthetic
corpus. It is not evidence of unsupervised morphological generalisation, and it
does not test unseen roots/affixes or variable morpheme counts. Before any claim
about real morphology:

1. fix F4 (unk source length), or guarantee alphabet coverage;
2. rerun morph vs frequency vs a corrected, size-matched random control on a
   real morphological corpus with held-out roots/affixes;
3. document macro averaging and the seed-46 near-tie;
4. regenerate any boundary P/R columns produced with the old empty-prediction
   convention if the dataset contains empty predictions;
5. add an invariant test that EOW is always the final node;
6. report and sensitivity-test the fixed `level1_fraction=0.6` (the v512 window
   may depend on this researcher choice; 0.5/0.7 were not tested here).

**Suggested wording:** “At v512, a two-level assignment selected by gold morpheme
provenance improves boundary F1 by +0.009 macro (9 clear seeds, 1 negligible;
Wilcoxon p=0.002) and bpc by −0.026 over the frequency assignment, with no effect
at v1024/2048. This is a supervised segmentation-alignment result on synthetic
held-out combinations, not evidence of unsupervised morphological
generalisation.”

## 9. Single check that would most change the verdict

Re-run the same three-way comparison on a **real morphological corpus with OOV
handling fixed and genuinely held-out roots/affixes**, scoring variable morpheme
counts including monomorphemic words. That is the check that tests generalisation
rather than re-testing the implementation; if the gain vanishes there, the v512
effect is synthetic-specific. The second-most informative check is a treatment
whose provenance comes from a disjoint corpus, which would separate “learned
morphological routing” from “the treatment was fitted to the evaluation target”.

## Appendix: commands used

```powershell
# reproduction and seed-42 detail
python "$env:TEMP\opencode\w1_17\audit_f1.py" 42

# full 10-seed macro/micro/weighted audit, per-word deltas, unk demo
python "$env:TEMP\opencode\w1_17\audit_fast.py"

# edge cases and morphology-blind baselines
python "$env:TEMP\opencode\w1_17\baselines.py"

# per-seed deltas, Wilcoxon, leave-one-out from the existing CSV
python "$env:TEMP\opencode\w1_17\csv_stats.py"

# corrected random-closed control on the two under-sized seeds
python "$env:TEMP\opencode\w1_17\corrected_random.py" 46,50

# v1024/2048 exact-equality check
python -c "import csv; ..."   # see section 1

# unk toy
python -c "from ramified.bpe import train_tokenizer; ..."   # see section 4.2

# duplicate-append toy
python -c "..."   # see section 7
```

Raw logs: `%TEMP%\opencode\w1_17\audit_fast.log`,
`%TEMP%\opencode\w1_17\corrected_random.log`; the seed-42 reproduction and
baseline outputs are shown inline above and in the session transcript.
