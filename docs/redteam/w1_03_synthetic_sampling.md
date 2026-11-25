# W1-03 — Does the v512 morph effect depend on Zipfian sampling and period insertion?

**Agent:** W1-03 (red team, 3 of 100)
**Date:** 2026-11-22
**Project:** `C:\Users\acese\Desktop\unibull`
**Inputs:** `ramified/synthetic.py`, `experiments/level_assignment.py`,
`ramified/bpe.py`, `ramified/levels.py`, `ramified/metrics.py`,
`ramified/codec.py`, `docs/findings-v0.2.md`, `diario.md`.
**Mode:** read-only on source. The only writes are under `results/redteam/`
and this report.

## Bottom line

The v512 morph−frequency effect **survives removing the 7% period insertion**
and **does not survive replacing Zipfian with uniform word sampling**.

- Zipf(1.1), seeds 42–46: morph−frequency ΔF1 **+0.0067** (95% CI
  [+0.0012, +0.0122], t = 3.40, 5/5 wins), Δbpc **−0.0168** (CI
  [−0.0327, −0.0010], 5/5 wins).
- No periods: ΔF1 **+0.0067** [+0.0013, +0.0121], Δbpc **−0.0163**
  [−0.0316, −0.0009]. The period effect-shift is +0.0000 F1
  [−0.0002, +0.0003] (p = 0.74) and −0.0006 bpc (p = 0.33): removing periods
  changes nothing.
- Uniform sampling: morph and frequency produce **identical held-out
  tokenization on all five seeds** (0 of 2,304 test words changed; ΔF1 =
  0.0000, Δbpc = 0.0000, 5/5 exact ties). Removing periods as well gives the
  same zero.
- Effect shift, baseline − uniform: **+0.0067 F1** [+0.0012, +0.0122]
  (p = 0.027), **−0.0168 bpc** [−0.0327, −0.0010] (p = 0.042). The shift is
  the entire baseline effect.

The mechanism is a **frequency-prefix repair**. Under Zipf, the frequency-ranked
294-rule level 1 leaves 63.2 intra-morpheme rules in level 2 (and puts 100
cross-morpheme rules in level 1); morph promotes the 63 intra rules, changes
1.4% of held-out words, and improves F1/bpc. Under uniform, frequency level 1
leaves only 0.8 intra rules in level 2 (range 0–3); morph promotes those few,
but no held-out word changes, so every metric ties exactly. Periods do not
touch this structure: no-period level 2 has 63.0 intra rules and the effect is
unchanged.

The published effect is therefore conditional on a head-heavy Zipfian training
distribution, not a general advantage of morphological level assignment.
Uniform sampling also raises type coverage from 81.0% to 100%, but coverage is
not the driver: a 1M-token Zipf corpus with 99.99–100% coverage reproduces the
effect almost exactly (+0.0074 F1, shift vs 120k +0.0007 [−0.0020, +0.0034],
p = 0.53). The frequency prefix remains impure at full coverage (62.8 intra
rules left in level 2, versus 63.2 at 120k). The operative variable is the
head/tail frequency skew, not the missing tail.

## 1. Question and design

The published v512 result (`docs/findings-v0.2.md` §2) uses the synthetic
generator in `ramified/synthetic.py`, which has two corpus-level choices:

1. training words are drawn with a Zipf(1.1) rank-frequency law over a randomly
   permuted 9,216-word training inventory;
2. a period is inserted after 7% of sampled word tokens.

The question is whether either choice is necessary for the morph effect.

The ablation is a 2×2 over `{Zipf, uniform} × {7% periods, no periods}` at
v512. For a given seed, all four conditions share:

- the same 11,520-word inventory and the same 20% held-out split;
- the same 2,304 held-out words and the same 2,000-token held-out text;
- 120,000 training word tokens;
- the same flat BPE budget (490 merges with periods, 491 without, because `.`
  enters the terminal alphabet only in the period conditions);
- the same level-assignment fraction (0.6) and the same random seed for the
  `random_closed` draw.

Only the training token distribution changes:

| condition | word sampling | periods |
|---|---|---|
| `baseline` | Zipf(1.1) | 7% |
| `uniform` | i.i.d. uniform over the 9,216 training types | 7% |
| `noperiod` | Zipf(1.1) | 0 |
| `uniform_noperiod` | i.i.d. uniform | 0 |

The `noperiod` corpus is the Zipf word sequence of the baseline with the `.`
tokens removed, so the sampled words are unchanged. The uniform corpora draw
120,000 indices with replacement from `syn.train_words`; realized entropy is
13.11 of a possible 13.17 bits, i.e. effectively uniform. A fifth condition,
`zipf1m`, keeps the Zipf law and the same test words but draws 1,000,000
training word tokens; it covers 99.99–100% of the inventory and is used in §7
to separate skew from coverage.

For each condition the harness trains one flat BPE and evaluates five
assignments: `flat` (reference), `frequency`, `morph`, `random_closed`, and
`morph_type`. `morph_type` is a diagnostic: it computes provenance without
token-frequency weighting (one count per word type). The three required modes
are `frequency`, `morph`, and `random_closed`.

Primary metric: mean boundary F1 on the 2,304 held-out words. Secondary:
bits per character with add-alpha 0.1 on the normalised 2,000-token held-out
text. Comparisons are paired by seed.

## 2. Commands and artifacts

```powershell
# main 2x2 ablation, v512, five seeds
python results/redteam/w1_03_synthetic_sampling.py `
  --seeds 42,43,44,45,46 --vocab 512 `
  --modes frequency,morph,random_closed,morph_type `
  --trainer fast --out-dir results/redteam

# rendered tables used in this report
python results/redteam/w1_03_analyze.py --dir results/redteam `
  > results/redteam/w1_03_tables.md

# coverage-matched follow-up: Zipf with 1M tokens
python results/redteam/w1_03_zipf_budget.py `
  > results/redteam/w1_03_zipf1m_log.txt
```

Tail of the main run:

```text
wrote results/redteam\w1_03_runs.csv (100 rows)
wrote results/redteam\w1_03_stats.csv (50 rows)
wrote results/redteam\w1_03_corpus_stats.csv (20 rows)
wrote results/redteam\w1_03_baseline_check.csv (60 rows)
baseline reproduction: worst abs diff vs published = 6.883e-14
total wall time 1138.4s
```

Artifacts:

| file | contents |
|---|---|
| `results/redteam/w1_03_synthetic_sampling.py` | 2×2 harness |
| `results/redteam/w1_03_runs.csv` | 100 rows: 4 conditions × 5 seeds × 5 modes |
| `results/redteam/w1_03_stats.csv` | paired deltas and effect shifts |
| `results/redteam/w1_03_corpus_stats.csv` | realised corpus statistics |
| `results/redteam/w1_03_baseline_check.csv` | reproduction vs published CSV |
| `results/redteam/w1_03_trainer_check.csv` | source-vs-fast trainer equivalence |
| `results/redteam/w1_03_tables.md` | all rendered tables |
| `results/redteam/w1_03_log.txt` | full stdout |
| `results/redteam/w1_03_zipf1m_*.csv` | 1M-token Zipf follow-up |
| `results/redteam/w1_03_rule_overlap.csv` | level-1 set overlap diagnostic |

## 3. Verification

Three independent checks were run before interpreting the ablation.

1. **Baseline reproduction.** The `baseline` condition is the unmodified
   `generate_synthetic` corpus. Against `results/level_assignment_v512_10seeds.csv`
   for seeds 42–46 and the four published modes, 60 metric comparisons have a
   maximum absolute difference of **6.883e-14** (F1, bpc, tokens/char). The
   harness therefore reproduces the published baseline exactly.
2. **Trainer equivalence.** The harness uses a faster inner BPE loop because
   the uniform corpus makes the source lazy heap quadratic (duplicate pushes
   for the same pair inside one word update). `w1_03_trainer_check.csv` compares
   the fast and source trainers field by field on five condition/vocab pairs:
   same rule tuples `(left, right, new, rank, level, count)`, same expansions,
   same test-word encodings. All five are `True`. The source trainer remains
   the reference.
3. **Cache equivalence.** `collect_provenance_cached` walks one derivation tree
   per word type and multiplies by the text frequency. On baseline and uniform
   it matches `ramified.levels.collect_provenance` exactly, both unweighted
   (full 9k types) and weighted (20k-token sample). The fast integer encoder is
   checked against `Tokenizer.encode` for every unique training chunk in every
   condition; the harness aborts on mismatch, and all 20 conditions completed.

No source file was modified. The fast trainer and cached evaluator are in the
red-team script, not in `ramified/`.

## 4. Corpus manipulation

Mean over seeds 42–46:

| condition | word tokens | periods | types | coverage | top-1 share | top-10 share | entropy (bits/word) |
|---|---:|---:|---:|---:|---:|---:|---:|
| baseline | 120,000 | 8,446 | 7,468 | 0.810 | 0.1520 | 0.4083 | 8.25 |
| uniform | 120,000 | 8,355 | 9,216 | 1.000 | 0.0002 | 0.0022 | 13.11 |
| noperiod | 120,000 | 0 | 7,468 | 0.810 | 0.1520 | 0.4083 | 8.25 |
| uniform_noperiod | 120,000 | 0 | 9,216 | 1.000 | 0.0002 | 0.0022 | 13.11 |

The manipulation is large and in the intended direction: Zipf observes 81% of
the 9,216 types, has a 15.2% most-frequent word and 8.25 bits/word; uniform
observes all 9,216 types, has a 0.02% most-frequent word and 13.11 bits/word.
The held-out words and test text are identical across conditions.

## 5. Results

### 5.1 Mean metrics

Mean (SD) over five seeds:

| condition | flat F1 | frequency F1 | morph F1 | random_closed F1 | flat bpc | frequency bpc | morph bpc | random_closed bpc |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| baseline | 0.8556 (0.0518) | 0.8556 (0.0518) | 0.8624 (0.0541) | 0.8505 (0.0480) | 2.3217 | 2.3217 | 2.3049 | 2.3540 |
| uniform | 0.8954 (0.0735) | 0.8954 (0.0735) | 0.8954 (0.0735) | 0.8751 (0.0643) | 2.0002 | 2.0002 | 2.0002 | 2.0919 |
| noperiod | 0.8560 (0.0521) | 0.8560 (0.0521) | 0.8627 (0.0543) | 0.8464 (0.0480) | 2.3006 | 2.3006 | 2.2843 | 2.3355 |
| uniform_noperiod | 0.8952 (0.0735) | 0.8952 (0.0735) | 0.8952 (0.0735) | 0.8770 (0.0637) | 1.9895 | 1.9895 | 1.9895 | 2.0697 |

`flat` and `frequency` are identical in every cell, as expected: with the fixed
BPE the budget partition is a no-op. `morph_type` is also identical to `morph`
in every cell (5/5 ties on F1 and bpc in all four conditions), so the result
does not depend on weighting provenance by token frequency.

Absolute bpc is **not comparable across sampling conditions**: uniform training
matches the uniform held-out distribution, so its unigram model assigns more
probability to test tokens independent of tokenization quality. Only the
within-condition morph−frequency contrast is used below.

### 5.2 Morph − frequency

Per-seed ΔF1:

| seed | baseline | uniform | noperiod | uniform_noperiod |
|---:|---:|---:|---:|---:|
| 42 | +0.0041 | +0.0000 | +0.0043 | +0.0000 |
| 43 | +0.0097 | +0.0000 | +0.0097 | +0.0000 |
| 44 | +0.0096 | +0.0000 | +0.0096 | +0.0000 |
| 45 | +0.0100 | +0.0000 | +0.0097 | +0.0000 |
| 46 | +0.0001 | +0.0000 | +0.0001 | +0.0000 |
| **mean** | **+0.0067** | **+0.0000** | **+0.0067** | **+0.0000** |

Paired statistics (5 seeds, 95% t interval):

| condition | ΔF1 mean [CI] | t | W/L/T | Δbpc mean [CI] | t | W/L/T |
|---|---:|---:|---:|---:|---:|---:|
| baseline | +0.0067 [+0.0012, +0.0122] | 3.40 | 5/0/0 | −0.0168 [−0.0327, −0.0010] | −2.95 | 5/0/0 |
| uniform | +0.0000 [+0.0000, +0.0000] | — | 0/0/5 | +0.0000 [+0.0000, +0.0000] | — | 0/0/5 |
| noperiod | +0.0067 [+0.0013, +0.0121] | 3.46 | 5/0/0 | −0.0163 [−0.0316, −0.0009] | −2.95 | 5/0/0 |
| uniform_noperiod | +0.0000 [+0.0000, +0.0000] | — | 0/0/5 | +0.0000 [+0.0000, +0.0000] | — | 0/0/5 |

The uniform zero is not a noisy null: it is exact equality of tokenization.
`frac_test_words_diff_vs_frequency` is 0.0 for all 2,304 words on all five
uniform seeds (and all five uniform_noperiod seeds). The baseline changes
1.4% of held-out words on average (5–50 words per seed; seed 46 is the
near-tie with 5), consistent with W1-01's per-word sparsity finding.

### 5.3 Periods vs sampling

The effect shift is paired by seed: `(morph − frequency)` in condition A minus
the same contrast in condition B.

| shift | ΔF1 mean [CI] | t | p | Δbpc mean [CI] | t | p |
|---|---:|---:|---:|---:|---:|---:|
| baseline − uniform | +0.0067 [+0.0012, +0.0122] | 3.40 | 0.0273 | −0.0168 [−0.0327, −0.0010] | −2.95 | 0.0420 |
| baseline − noperiod | +0.0000 [−0.0002, +0.0003] | 0.36 | 0.7396 | −0.0006 [−0.0019, +0.0008] | −1.10 | 0.3338 |
| baseline − uniform_noperiod | +0.0067 [+0.0012, +0.0122] | 3.40 | 0.0273 | −0.0168 [−0.0327, −0.0010] | −2.95 | 0.0420 |

Removing the periods moves the effect by **0.0000 F1** and **−0.0006 bpc**,
with CIs that exclude any effect of practical size. Replacing Zipf with uniform
moves it by the full baseline effect. There is no detectable interaction:
`uniform − uniform_noperiod` is exactly 0.0000 on both metrics, and
`noperiod − uniform_noperiod` reproduces the baseline−uniform shift.

### 5.4 Random_closed control

Random level assignment is not inert. Under uniform, it is clearly harmful:
random_closed − frequency ΔF1 **−0.0204** [−0.0337, −0.0070], 0/5 wins;
Δbpc **+0.0917** [+0.0240, +0.1594], 0/5 wins. Without periods it is also
harmful: ΔF1 −0.0096 [−0.0153, −0.0039], 0/5; Δbpc +0.0349
[+0.0098, +0.0600], 0/5. In the baseline (with periods) the random control is
noisy on five seeds: ΔF1 −0.0052 [−0.0214, +0.0111], 2/5; Δbpc +0.0323
[−0.0170, +0.0816], 1/5. The published 10-seed estimate is −0.0064 F1 /
+0.0361 bpc.

So the level-assignment machinery still changes tokenization under uniform
(7.2% of test words for random_closed vs 0.0% for morph). The zero in §5.2 is
specific to the morph assignment, which has almost nothing left to promote.

### 5.5 It is not a ceiling artifact

Uniform sampling does not simply make the task easier. Flat F1 by seed:

| seed | baseline | uniform | noperiod | uniform_noperiod |
|---:|---:|---:|---:|---:|
| 42 | 0.8665 | 0.9494 | 0.8682 | 0.9491 |
| 43 | 0.8011 | 0.7882 | 0.8007 | 0.7878 |
| 44 | 0.9142 | 0.8486 | 0.9143 | 0.8486 |
| 45 | 0.8935 | 0.9464 | 0.8938 | 0.9461 |
| 46 | 0.8029 | 0.9446 | 0.8030 | 0.9444 |

Uniform is **lower** than baseline on seeds 43 and 44, where there is ample
headroom, yet morph still changes no word. The zero is not caused by F1 being
near 1.0; it is caused by the composition of the frequency prefix.

## 6. Mechanism

Mean level composition over five seeds. `intra`/`cross` classify each rule by
its majority provenance in the training text; `L2 intra` is the number of
intra-morpheme rules that the frequency split leaves in level 2 — the rules
morph can promote.

| condition | freq L1 intra/cross | freq L2 intra | morph L1 intra/cross | morph L2 intra | test words changed by morph | mean ΔF1 |
|---|---:|---:|---:|---:|---:|---:|
| baseline | 193 / 100 | 63.2 | 257 / 36 | 0.0 | 1.4% | +0.0067 |
| uniform | 264 / 29 | 0.8 | 265 / 28 | 0.0 | 0.0% | +0.0000 |
| noperiod | 194 / 100 | 63.0 | 257 / 37 | 0.0 | 1.4% | +0.0067 |
| uniform_noperiod | 264 / 30 | 0.6 | 265 / 29 | 0.0 | 0.0% | +0.0000 |

Under Zipf, the 294-rule frequency prefix is morphologically impure: it spends
about 100 slots on cross-boundary rules and leaves about 63 intra rules below
the cutoff. Morph swaps the two groups. Under uniform, the prefix already
contains 264 of the 265 intra rules (89.7% of level 1 is intra); at most three
intra rules remain in level 2 on any seed, and promoting them changes no
held-out word. The `random_closed` control, which leaves 48–61 intra rules
below the cutoff, moves 3–7% of test words and is harmful. The sign of the
intervention follows whether the reassignment is morphological or random; the
magnitude follows how many intra rules the frequency prefix displaced.

A direct set comparison on seed 42 (`w1_03_rule_overlap.csv`) confirms the
picture. Under Zipf, frequency and morph share 241 of 347 level-1 rules
(Jaccard 0.695): exactly 53 cross rules leave and exactly 53 intra rules enter.
Under uniform, the two level-1 sets are **identical** (294/294, Jaccard 1.000),
so morph is literally the same tokenizer, not merely an equivalent one. On the
uniform seeds with a residual displaced rule (43 and 46) the sets differ by at
most three intra rules, still with zero changed test words.

This also unifies the published v1024/2048 null with the uniform null. At a
larger vocabulary the frequency prefix is long enough to include the intra
rules even under Zipf; at v512, uniform weighting makes the prefix
intra-complete by removing the head/tail contrast, not by raising coverage
(§7 shows that 100% coverage under Zipf leaves 62.8 intra rules in level 2).
The published "budget window" is therefore better described as a
**frequency-prefix purity window**: the effect appears when level 1 is both
impure (cross rules present) and incomplete (intra rules left in level 2).

Periods are inert because they do not enter this structure. A `.` chunk
contributes a character and one `.EOW` merge, and consumes one vocabulary slot
(490 vs 491 merges), but it does not change the intra/cross counts of the
word-internal rules: no-period frequency L1 is 194/100 with 63.0 intra in L2,
versus 193/100 and 63.2 with periods. The small bpc level shift (2.322 → 2.301)
comes from removing an out-of-vocabulary test symbol and adding one merge, not
from a change in the morph contrast.

## 7. Coverage-matched Zipf: 1M tokens

The 120k Zipf corpus observes 81% of the inventory; the uniform corpus observes
100%. To separate skew from coverage, the same Zipf(1.1) generator was run with
1,000,000 training word tokens (12.5× the budget) on seeds 42–46, keeping the
120k baseline's test text and held-out words so F1 and bpc are paired with the
main baseline rows.

Coverage rises to 9,215–9,216 of 9,216 types (0.9999–1.0000) while the head is
unchanged: top-1 share 15.1–15.3%, top-10 40.7–40.9%, entropy 8.29–8.31
bits/word. This is the coverage of the uniform corpus with the skew of the
Zipf corpus.

| metric | morph − frequency at 1M | shift vs Zipf 120k |
|---|---:|---:|
| ΔF1 | **+0.0074** [+0.0025, +0.0123], t = 4.21, 5/5 | +0.0007 [−0.0020, +0.0034], p = 0.53 |
| Δbpc | **−0.0179** [−0.0301, −0.0056], t = −4.05, 5/5 | −0.0010 [−0.0113, +0.0092], p = 0.79 |

The level composition is essentially unchanged from 120k:

| corpus | freq L1 intra/cross | freq L2 intra | morph L1 intra/cross | morph L2 intra |
|---|---:|---:|---:|---:|
| Zipf 120k (81% coverage) | 193 / 100 | 63.2 | 257 / 36 | 0 |
| Zipf 1M (100% coverage) | 192 / 101 | 62.8 | 255 / 38 | 0 |
| uniform 120k (100% coverage) | 264 / 29 | 0.8 | 265 / 28 | 0 |

Closing the coverage gap does not repair the frequency prefix. At full
coverage the prefix still spends about 101 slots on cross-boundary rules and
leaves about 63 intra rules below the cutoff. The effect is statistically
indistinguishable from the 120k baseline on both metrics. The main confound of
the 2×2 is therefore resolved in favour of **frequency skew**: the uniform null
is caused by removing the head/tail frequency contrast, not by the uniform
corpus seeing more types.

## 8. Does the effect survive?

| perturbation | effect survives? | evidence |
|---|---|---|
| remove 7% periods | **yes** | ΔF1 +0.0067 vs +0.0067; shift +0.0000 [−0.0002, +0.0003], p = 0.74 |
| increase Zipf budget to 1M (100% coverage) | **yes** | ΔF1 +0.0074; shift vs 120k +0.0007 [−0.0020, +0.0034], p = 0.53 |
| uniform word sampling | **no** | ΔF1 0.0000, Δbpc 0.0000, 5/5 exact ties; 0/2,304 test words changed |
| both | **no** | same as uniform |

The correct conditional statement is: *on this generator, at v512 with the
0.6 level-1 fraction, the morph advantage requires a head-heavy (Zipfian)
training distribution whose frequency-ranked prefix is morphologically impure.
It does not require period tokens, and it is not an under-coverage artifact:
full type coverage with the same skew reproduces it.*

## 9. What would change my mind

1. **A uniform corpus that still shows the effect.** If a uniform corpus at
   v512 produced morph > frequency with a CI excluding zero, the sampling
   dependence would be falsified.
2. **A controlled head/tail tilt on the uniform corpus.** Keep 100% type
   coverage but promote a random 1–5% of word types to a large share of the
   tokens. If morph then helps, the causal variable is the head/tail skew
   itself; if it does not, the specific Zipfian rank shape (many mid-frequency
   words, not just a head) is required. This is the cleanest next intervention
   because it varies skew while holding coverage fixed.
3. **A real morphological corpus with natural Zipf.** If morph still helps
   there, the practical claim survives even if the synthetic mechanism is
   distributional.
4. **A level-1 fraction sweep under uniform.** At a smaller fraction (e.g.
   0.2–0.4) the frequency prefix becomes incomplete even under uniform, so
   morph may regain something to promote. That would locate the effect at the
   interaction of sampling and budget, not at sampling alone.
5. **A Zipf exponent sweep at 100% coverage.** Exponents 0.3/0.6/0.9/1.1 with
   the token budget scaled to keep coverage high would test whether the effect
   rises monotonically with skew. A threshold would locate the mechanism more
   precisely than the two-point uniform-vs-Zipf contrast.

## 10. Limitations

- Five seeds (42–46). The baseline subset includes the near-tie seed 46, so its
  mean effect (+0.0067) is smaller than the published 10-seed estimate
  (+0.0092). This makes the baseline side conservative, not the uniform null:
  the uniform null is exact equality, not a power failure.
- The 2×2 confounds Zipfian skew with type coverage, but the 1M-token check
  holds coverage at 100% and reproduces the effect, so coverage is ruled out as
  the driver. Which feature of the skew matters (head mass, exponent, or the
  broad mid-frequency mass) is still open; the head/tilt intervention in §9 is
  the next test.
- Boundary F1 is type-balanced over the held-out words; bpc is token-balanced
  on a uniform held-out text. Real evaluation may weight words differently.
- `random_closed` has a known sampler bug: duplicate topological emissions can
  produce 292–294 unique level-1 rules (W1-14). This affects only the secondary
  control; `frequency` and `morph` always have 294. The direction of the
  random-control result is unchanged by the size correction in W1-14.
- The no-period corpus has 491 merges rather than 490 because `.` leaves the
  terminal alphabet. The extra merge is one slot; the level compositions and
  the morph contrast are unchanged.
- The synthetic boundaries are oracle labels. Nothing here tests a real
  tokenizer or a real corpus.

## 11. Relation to other red-team checks

- **W1-01** shows that the v512 gain is sparse (20–50 changed word types per
  seed) and that the holdout is unseen combinations of seen morphemes. The
  present result explains why the sparse changes vanish under uniform: the
  rules that produce them are no longer displaced from level 1.
- **W1-14** shows the published random_closed draw is one point in a wide
  random distribution, and documents the 292–294 size bug. The present
  random_closed numbers are one draw per seed, as in the published design.
- **W1-16** finds a dose-response between level-1 cross rules and the morph
  gain, and reports that a 60-root corpus removes the effect. The present
  result adds a corpus-level manipulation with the same signature: the gain
  tracks the number of intra rules left in level 2 (63 vs 1), and vanishes
  when that number is near zero. It also cautions that a low level-1 cross
  count alone is not sufficient; level 1 must also be intra-complete.

## 12. Bottom line for the paper

The v512 sentence must carry the corpus condition: *with Zipfian training, a
uniform held-out set, vocabulary 512 and a 60/40 level split, morphological
level assignment improves boundary F1 by about 0.7 points and bpc by about
0.017 over the frequency assignment in this five-seed subset (the published
ten-seed estimate is 0.9 points and 0.026); the gain disappears under uniform
word sampling, is unchanged when the 7% period tokens are removed, and is
unchanged when the Zipf budget is raised to 1M tokens with full type
coverage.* The effect is a repair of a morphologically impure frequency prefix,
not a general property of ramified tokenization. Because natural text is
Zipfian, the result may still be practically relevant, but the mechanism is
distributional and must be reported as such.
