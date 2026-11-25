# W1-20 — Multiple comparisons, specification curve, and held-out seeds for the v512 level-assignment claim

**Agent:** W1-20 (red team, read-only)
**Date:** 2026-11-22
**Inputs:** `results/level_assignment.csv`, `results/level_assignment_v512_10seeds.csv`,
`results/level_assignment_stats.csv`, `results/level_assignment_stats_v512.csv`,
`experiments/analyze_levels.py`, `experiments/level_assignment.py`,
`ramified/{bpe,levels,synthetic,metrics}.py`, `docs/findings-v0.2.md`, `diario.md`,
`docs/reviews/data-review.md`.
**Constraint:** no project source was modified. The only project write is this report.
Analysis scripts were run from `%TEMP%\opencode\w1_20_*.py` and only read data /
recomputed from the generator.
**Companion reports:** W1-14 (random-control ensemble), W1-18 (bpc audit),
W1-19 (bootstrap/permutation). This note does not repeat their full tables. It adds
(i) the fully enumerated multiplicity family across vocabularies and mode pairs,
(ii) the post-hoc/held-out-seed audit, (iii) the `level1_fraction` specification
curve, (iv) the `t=inf` reporting bug, and (v) the per-word distribution.

---

## Verdict

- **The primary v512 claim survives.** `morph − frequency` is +0.00924 boundary F1
  (t = 6.52, 10/10 seeds) and −0.02579 bpc (t = −5.38, 10/10). Both survive
  Bonferroni/Holm over the full 24-test family (F1: 0.0026/0.0026; bpc:
  0.0106/0.0102), and the exact sign test is at its 10-pair floor (p = 0.00195;
  ×24 = 0.0469). The fresh seeds 45–51 replicate 7/7 on both metrics; the
  cleanest block 47–51 replicates 5/5 (t p = 0.0011 / 0.0033).
- **The secondary v512 claim does not fully survive.** `morph − random_closed`
  F1 is +0.01563 (t = 3.81, 9/1), but full-family Bonferroni/Holm give 0.0996 /
  0.0872, the exact sign test gives p = 0.0215 (×4 = 0.086), and the adjusted
  95% CI at m = 18/24 includes 0. It survives only if the family is restricted
  to ≤12 tests (Bonferroni 12 = 0.0498, borderline; 6 = 0.0249). The bpc arm of
  the same comparison (−0.06193, 10/10) survives every correction
  (Bonferroni 24 = 0.0260, Holm 24 = 0.0238; exact sign ×24 = 0.0469).
- **“Effect confined to v512” is not statistically established.** The only
  larger-vocabulary data are 3 seeds. The paired v512−v1024 difference is
  p = 0.0517 (F1) and 0.1151 (bpc); the 10-seed run did not include
  v1024/v2048. The exact zeros at v1024/v2048 are structural no-ops at the
  fixed 60% fraction, not a tested absence of a morphological effect.
- **The dominant researcher degree of freedom is the 60% level-1 budget.** At
  `level1_fraction = 0.5` the F1 effect reverses (−0.01558, 4/6 seeds,
  p = 0.081); at 0.4 it is −0.02495 (0/10, p = 0.0017); at 0.6 it is +0.00924
  (10/0, p = 0.00011); at 0.7 +0.00139 (8/0/2); at 0.8 +0.00004 (2/0/8).
  The 0.6 default is the smallest tested grid point above the pure-intra rule
  fraction (~252/490 = 0.514). The v512 result is a **narrow, tuned-window
  effect**, not a robust one.
- **The findings’ summary sentence is overbroad.** “Morph migliora sia
  l’allineamento sia la compressione su tutti e 10 i seed” is true for
  morph−frequency (both metrics) and for bpc-vs-random, but not for
  F1-vs-random: that row is 9/1. The bpc result is a unigram cross-entropy
  result, not a compression result (W1-18); the F1 improvement is carried by
  ~1.9% of held-out words, with no word worsening.

---

## 1. Data, reproduction, and scope

The two design files are not the same experiment:

| file | design | rows | modes |
|---|---|---:|---|
| `results/level_assignment.csv` | 3 seeds (42–44) × 3 vocabs (512/1024/2048) × 5 modes | 45 | flat, frequency, morph, random_closed, reverse |
| `results/level_assignment_v512_10seeds.csv` | 10 seeds (42–51) × v512 × 4 modes | 40 | flat, frequency, morph, random_closed |

Checks:

- `frequency` is bit-identical to `flat` in every metric cell of the 3-seed file
  (0 mismatches over the 9 frequency rows × 5 metrics = 45 cells). The design
  therefore has one flat baseline, not two independent controls.
- The 3-seed v512 rows are reproduced exactly in the 10-seed file for seeds
  42–44 and all four modes (0 mismatches over 3 × 4 × 5 cells). The 10-seed file
  is a superset for v512, except that it drops `reverse`.
- Recomputing the pipeline from the generator reproduces the published v512 CSV
  exactly: 0/10 mismatches for `morph` F1 and 0/10 for the canonical
  `random_closed` F1 (to 1e-9). The published stats files also reproduce
  exactly from the current analyzer (0 mismatched cells for both the 6-row and
  24-row files).
- `n_l1 + n_l2` is 490 at v512, 1002 at v1024, 2026 at v2048. The rules and
  vocabulary are held fixed within a seed; only the assignment changes.
- The 24-row `results/level_assignment_stats.csv` is the **3-seed** design. The
  6-row `results/level_assignment_stats_v512.csv` is the **10-seed** confirmatory
  file. A reader must not quote the 3-seed v512 p = 0.0517 as the headline
  result.

The synthetic generator changes the root inventory, the train/test split, and
the sampled text per seed; it does **not** change the affix inventory, the
prefix–root–suffix grammar, or the boundary definition. “10 seeds” means 10
draws from one synthetic grammar, not 10 corpora or languages. Seeds are nearly
disjoint in roots (W1-19: 0–5 overlap out of 120 per pair), but they share the
same morphology. The treatment (`morph`) and the primary metric (`boundary_f1`)
also share the same boundary ontology by construction: the assignment is built
from training-word boundaries, and the metric scores the same kind of
boundaries on held-out combinations. There is no test leakage (provenance uses
training words only), but the result is a controlled mechanism check, not
external validity.

---

## 2. Was v512 chosen post hoc?

**Yes for the 10-seed confirmation; no for the general project focus.** The
chronology in `diario.md` and the file timestamps is:

1. The 3-seed design was run at v512, v1024, v2048 with four assignment modes.
2. v512 was the only non-null cell. At v1024/v2048, `morph` and `frequency`
   were exactly 0.000 on F1 and bpc for all 3 seeds; at v2048 the level-1 sets
   are literally identical (W1-18: Jaccard 1.0; v1024: 12 rules differ but the
   tokenization is bit-identical).
3. Only after seeing that pattern was the 10-seed run launched, and it was
   launched **only at v512**.

The choice is not arbitrary: v512 was the best F1/bpc cell in the first sweep
and the first corrected smoke, so the project had a legitimate prior interest in
it. But the level-assignment experiment itself treated 512/1024/2048
symmetrically (`--vocabs 512,1024,2048`), and the 10-seed extension was run only
at the one cell that showed a signal. That is post hoc confirmation, not a
pre-registered confirmatory design.

That is a follow-the-signal extension, and the 3-seed v512 result was not
itself significant at 0.05:

| comparison (3 seeds, v512) | mean | t | p (two-sided) | 95% CI |
|---|---:|---:|---:|---|
| morph − frequency, F1 | +0.007827 | +4.226 | 0.0517 | [−0.00014, +0.01580] |
| morph − frequency, bpc | −0.019825 | −2.687 | 0.1151 | [−0.05157, +0.01192] |
| morph − random_closed, F1 | +0.013161 | +1.107 | 0.3835 | [−0.03798, +0.06430] |
| morph − random_closed, bpc | −0.048465 | −1.397 | 0.2974 | [−0.19778, +0.10085] |

The 10-seed extension has the structure of an optional-stopping /
sample-augmentation move. A two-look Bonferroni (×2) does not change the
qualitative picture for the primary claim (F1 p = 0.00011 → 0.00022; bpc
0.00044 → 0.00088), but the published p-values are not adjusted for the interim
look.

**Mitigation: the fresh seeds replicate, and the selection seeds are not
inflated.** The 10 seeds are not all equally clean:

| seed block | status before the 10-seed level-assignment run | F1 mean (morph−freq) | F1 t / p | bpc mean | bpc t / p |
|---|---|---:|---:|---:|---:|
| 42–44 | used in the 3-vocab/4-mode exploration that selected v512 | +0.007827 | 4.226 / 0.0517 | −0.019825 | −2.687 / 0.1151 |
| 45–51 | fresh for the level-assignment effect; 45–46 appear in the data-review's 5-seed **flat/hier2** baseline sweep (not the morph comparison) | +0.009850 | 5.181 / 0.00205 | −0.028343 | −4.612 / 0.00365 |
| 47–51 | cleanest: never used before the 10-seed level-assignment run | +0.011765 | 8.396 / 0.00110 | −0.034745 | −6.283 / 0.00328 |

The fresh and clean blocks have **larger** point estimates than the selection
block. The selection seeds are not driving the effect; if anything, they are
conservative. The 45–46 caveat matters for the phrase “held out”: the data
review had already looked at the baseline F1 of those seeds at v512, although
it had not run the morph-vs-frequency comparison. The cleanest held-out block
is 47–51, n = 5.

**The held-out claim in sign and magnitude:** on 45–51, all four headline
comparisons are 7/7 in the favourable direction; on 47–51, all four are 5/5.
The t-tests are significant uncorrected. Under a 4-test Bonferroni:

| block | morph−freq F1 | morph−freq bpc | morph−random F1 | morph−random bpc |
|---|---:|---:|---:|---:|
| 45–51 (n=7) | 0.0082 | 0.0146 | 0.0210 | 0.0090 |
| 47–51 (n=5) | 0.0044 | 0.0131 | 0.0740 | 0.0534 |

The clean 5-seed block certifies only the primary morph−frequency claim under a
4-test correction. The secondary morph−random F1 claim is not certified by the
cleanest block: its 4-test Bonferroni p is 0.074. The exact sign test on 5/5 is
p = 0.0625, which cannot reach 0.05; on 7/7 it is 0.0156, but ×4 = 0.0625. The
t-test is doing the work; the distribution-free test is underpowered at these
block sizes.

**The “confined to v512” part is weaker than the v512 part.** The paired
v512−v1024 difference on the 3 common seeds is exactly the 3-seed v512 effect,
because every v1024 delta is 0: p = 0.0517 (F1), p = 0.1151 (bpc). That is not
significant at 0.05. The 10-seed run has no v1024/v2048 arm. The exact zeros
are real in the data, but they are 3-seed no-ops at the 60% fraction; they do
not establish a general “window.” The mechanism section below explains why the
no-op is a fraction artifact.

---

## 3. Multiplicity correction

### 3.1 The family

The experiment varied 3 vocabularies × 4 assignment modes × 2 metrics, plus the
flat/frequency identity. The full design family is 24 tests: 3 vocabularies
× 4 pairs (`morph−frequency`, `random_closed−frequency`,
`morph−random_closed`, `reverse−frequency`) × 2 metrics. At v512, the 10-seed
stats are used for the three non-reverse pairs; the v512 reverse row and all
v1024/v2048 rows are the 3-seed stats. This is a conservative sensitivity
family, not a pre-registered family: the 10-seed and 3-seed rows have different
sample sizes. Exact-zero rows are assigned p = 1, not p = 0 (see §3.4).

| family | m | what it represents |
|---|---:|---|
| all reported | 24 | full 3-vocab × 4-pair × 2-metric design |
| no reverse | 18 | 3 vocabs × 3 pairs × 2 metrics |
| v512 + v1024 | 12 | 2 vocabs × 3 pairs × 2 metrics |
| v512 only | 6 | 3 pairs × 2 metrics |
| headline | 4 | morph−frequency and morph−random, × 2 metrics |
| primary | 2 | morph−frequency, × 2 metrics |

### 3.2 Headline rows under correction

| claim | raw t | raw p | Bonf m=6 | Holm m=6 | Bonf m=12 | Holm m=12 | Bonf m=18 | Holm m=18 | Bonf m=24 | Holm m=24 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| morph−freq F1 | +6.518 | 0.00011 | 0.00065 | 0.00065 | 0.00131 | 0.00131 | 0.00196 | 0.00196 | 0.00262 | 0.00262 |
| morph−freq bpc | −5.384 | 0.00044 | 0.00265 | 0.00221 | 0.00531 | 0.00487 | 0.00796 | 0.00752 | 0.01062 | 0.01018 |
| morph−rand bpc | −4.725 | 0.00108 | 0.00649 | 0.00432 | 0.01297 | 0.01081 | 0.01946 | 0.01730 | 0.02595 | 0.02378 |
| morph−rand F1 | +3.810 | 0.00415 | **0.02490** | **0.01245** | **0.04980** | **0.03735** | 0.07470 | 0.06225 | 0.09960 | 0.08715 |

Bold = survives at 0.05. The primary and the bpc-vs-random claim survive every
family. The F1-vs-random claim survives m ≤ 12 (at m = 12 the Bonferroni value
is 0.0498, i.e. a hair below 0.05) and fails the full 18/24-test design.

The multiplicity-adjusted 95% CIs tell the same story (df = 9, per-test
α = 0.05/m):

| claim | mean | 95% CI | m=12 CI | m=18 CI | m=24 CI |
|---|---:|---|---|---|---|
| morph−freq F1 | +0.009243 | [+0.00604, +0.01245] | [+0.00384, +0.01464] | [+0.00346, +0.01502] | [+0.00319, +0.01530] |
| morph−freq bpc | −0.025788 | [−0.03662, −0.01495] | [−0.04403, −0.00755] | [−0.04531, −0.00627] | [−0.04624, −0.00534] |
| morph−rand bpc | −0.061932 | [−0.09158, −0.03228] | [−0.11184, −0.01202] | [−0.11534, −0.00852] | [−0.11788, −0.00598] |
| morph−rand F1 | +0.015629 | [+0.00635, +0.02491] | [+0.00001, +0.03125] | **[−0.00109, +0.03234]** | **[−0.00188, +0.03314]** |

Bold = includes zero. The 95% CI in the findings is not a multiplicity-adjusted
interval; the adjusted interval for the F1-vs-random claim includes zero at the
full-design family sizes.

### 3.3 Exact tests

Paired sign-flip permutation over all \(2^{10}\) sign assignments; the sign test
is the binomial test on wins/losses; Wilcoxon uses the magnitudes.

| claim | W/L/T | sign p | sign ×4 | sign ×24 | sign-flip p | Wilcoxon p | Wilcoxon ×24 |
|---|---:|---:|---:|---:|---:|---:|---:|
| morph−freq F1 | 10/0/0 | 0.00195 | 0.0078 | 0.0469 | 0.00195 | 0.00195 | 0.0469 |
| morph−freq bpc | 0/10/0 | 0.00195 | 0.0078 | 0.0469 | 0.00195 | 0.00195 | 0.0469 |
| morph−rand bpc | 0/10/0 | 0.00195 | 0.0078 | 0.0469 | 0.00195 | 0.00195 | 0.0469 |
| morph−rand F1 | 9/1/0 | 0.02148 | **0.0859** | 0.5156 | 0.00391 | 0.00391 | 0.0938 |

The primary and bpc-vs-random claims survive even the conservative sign test
with a 24-test Bonferroni, but only just (0.0469). The F1-vs-random claim fails
the sign-test correction even at m = 4. The sign-flip/Wilcoxon tests are kinder
(p = 0.00391) because the single loss (seed 42) has a small magnitude; they
still fail at m = 24 (0.0938). This is the statistical core of the
“provisional” verdict on the F1-vs-random arm.

### 3.4 Reporting bug: `t = inf` for exact zeros

`experiments/analyze_levels.py:51` sets `t = mean/se if se > 0 else inf`. Eight
rows in `results/level_assignment_stats.csv` are exact-zero effects (v1024 and
v2048, morph−frequency and reverse−frequency, both metrics): mean = 0, std = 0,
se = 0, CI = [0, 0], **t = inf**. The correct p-value for a zero effect is 1.0,
not 0. A downstream meta-analysis that parses `t` and computes a p-value from
`inf` would rank the 1024/2048 no-ops as the most significant results in the
file. The findings do not quote those rows, and the current text says “3 ties,”
which is correct; but the stats CSV is a trap. Fix: set `t = NaN` (or print
`n/a`) and p = 1.0 when `se == 0 and mean == 0`.

Minor related issue: `T95.get(n, 1.96)` uses the normal 1.96 for any n > 10. If
the analyzer is reused with 11–20 seeds, the CI is too narrow (correct t for
n = 11, df = 10 is 2.228; for n = 20, df = 19 is 2.093). The current data have
n ≤ 10, so this does not affect the published numbers, but it is a reproducibility
trap for the planned larger replication.

---

## 4. Researcher degrees of freedom

### 4.1 The level-1 fraction is the dominant DOF

`experiments/level_assignment.py` defaults to `--fraction 0.6`, inherited from
the `hier2` 0.6/0.4 budget split. There is no morph-specific pre-registration
of 0.6 and no published fraction sweep for `morph`. I swept
`level1_fraction ∈ {0.4, 0.5, 0.6, 0.7, 0.8}` on the same 10 seeds, with exact
provenance and the same evaluation. The 0.6 row reproduces the published morph
F1 exactly (0/10 mismatches), so the sweep is on the published pipeline.

| fraction | mean actual n1 | Δ F1 (morph − flat) | t | p | W/L/T |
|---:|---:|---:|---:|---:|---:|
| 0.4 | 176.5 | **−0.024949** | −4.415 | 0.00168 | 0/10/0 |
| 0.5 | 221.6 | **−0.015580** | −1.962 | 0.08138 | 4/6/0 |
| 0.6 | 294.0 | **+0.009243** | +6.518 | 0.00011 | 10/0/0 |
| 0.7 | 343.0 | +0.001387 | +4.000 | 0.00311 | 8/0/2 |
| 0.8 | 392.0 | +0.000043 | +1.500 | 0.16785 | 2/0/8 |

The sign reverses between 0.5 and 0.6. At 0.4 the effect is significantly
negative (morph is worse than flat); at 0.5 it is negative but not significant;
at 0.6 it is strongly positive; at 0.7 it is small but positive; at 0.8 it is
zero. A pre-registered 0.5 (the natural “half” prior) would have produced a
negative result. The 0.6 default is the only grid point with a large positive
effect and 10/10 wins.

The mechanism is a threshold, not a smooth optimum. The exact provenance count
at v512 is 225–263 pure-intra rules (mean 252.1) out of 490. At 0.5, dependency
closure reduces L1 to 221.6 rules, **below** the pure-intra count; some pure-intra
rules remain in L2, where cross rules can outrank them. At 0.6, L1 has 294 rules,
**above** the pure-intra count; all pure-intra rules fit in L1 and the L2 rules
are cross/mixed. At 0.7/0.8, L1 starts absorbing cross rules and the benefit
shrinks. The exact pure-intra threshold is ~252/490 = 0.514; the CSV's
majority-intra variant gives 261/490 = 0.533. Either way the threshold lies
between the 0.5 and 0.6 grid points, and 0.6 is the first tested point above it.
This is a sharp, corpus-specific threshold, not a broad “window.”

The paper must either (a) report this sweep and the threshold, or (b) choose the
fraction by a rule (e.g. set n1 to the number of pure-intra rules, or a fixed
absolute budget near 250) and pre-register that rule. Presenting 0.6 as an
arbitrary default is not defensible.

### 4.2 The random control is a single draw, and it has a size bug

The published `random_closed` control is one random topological prefix per seed,
with RNG seed `seed * 1000 + vocab`. Its level-1 size is 294 in 7/10 seeds, 293
in seeds 44 and 46, and 292 in seed 50, versus 294 for morph/frequency. W1-14
documents the cause: a rule can be appended twice when it depends twice on the
same creator (a self-pair `(X, X)`), so the topological prefix contains a
duplicate and fewer than n1 unique rules. I replayed the sampler and confirmed
it: seed 44's order has 491 entries for 490 rules (rule 438 appears twice;
L1 = 293), seed 50's order has 492 entries (rules 239 and 381 each twice;
L1 = 292). A deduplicated sampler gives exactly 294 rules.

The size bug is not the main driver. W1-14's 10-draw ensemble gives morph
+0.0223 F1 / −0.0887 bpc versus the random mean, 10/10 seeds; restricting to
size-294 draws gives +0.0215 / −0.0829. My independent corrected sampler (10
draws per seed, exact 294-rule L1) gives +0.02046 F1, t = 8.20, p = 2e-5, 10/10;
morph beats all 10 draws in 9/10 seeds. The published single-draw effect
(+0.01563, 9/1) is **smaller** than the ensemble mean and is conditional on the
particular draws. The published single-draw p = 0.00415 is not the correct
uncertainty for “random assignment” as a factor; a mixed model with multiple
draws per seed would be the right analysis.

But the ensemble is a post-hoc sensitivity check, not the pre-specified test.
The multiplicity correction in §3 applies to the published single-draw test and
fails at the full family. The honest summary: the F1-vs-random effect is
probably real (ensemble 10/10, fresh seeds 7/7), but the published test is
underpowered and not robust to full-family correction; report it as provisional
and pre-register the ensemble for the replication.

### 4.3 `reverse` is not a size-matched control

At v512, `reverse` has n1 = 124, 141, 154 (3 seeds) versus 294 for
morph/frequency, because cross-first closure demotes many rules. Its F1 is
−0.0109 versus frequency (0/3 wins), bpc +0.0192 (3/3 worse), but the level-size
confound is severe. It was dropped from the 10-seed run, so the “any reordering”
control is only 3 seeds and is not comparable. Do not use `reverse` as a clean
control; the random ensemble is the better “generic reordering” comparison.

### 4.4 bpc smoothing and token count

`alpha = 0.1` versus `alpha = 1.0` gives −0.025788 vs −0.025780 bpc, t = −5.384
vs −5.378, 10/10 wins, correlation 1.000. The bpc sign is not a smoothing
artifact. W1-18 has the fuller audit: the gain is a unigram / held-out
recombination effect, it reverses on natural-frequency in-distribution text
(+0.0096, 8/2), and an interpolated bigram code removes 86–92% of it. The
numerical bpc sign survives; the word “compression” does not. Report it as
“held-out unigram bpc.”

Token count is not the confound: morph − frequency Δ tok/char = +0.000037
(t = 0.25, 5/4/1), and morph − random has morph with **fewer** tokens
(−0.00538, t = −4.33) but higher F1. Precision and recall both improve for
morph vs frequency (+0.00849 precision, +0.00953 recall). The F1 gain is not a
token-count or precision/recall tradeoff.

### 4.5 Per-word distribution: the mean is 1.9% of words

On the 10 seeds × 2,304 held-out words = 23,040 word evaluations:

```text
improved = 443 (1.92%)   worsened = 0 (0.00%)   tied = 22,597 (98.08%)
mean ΔF1 = +0.009243     median ΔF1 = 0.000000
p05 = 0.000000           p95 = 0.000000
```

The mean F1 gain is entirely a +0.48 average gain on the 1.9% of words whose
boundaries change; no word gets worse. This is a clean but extremely sparse
effect. It is not a broad segmentation improvement, and it is consistent with
the narrow fraction threshold: the treatment fixes a small set of words and
leaves 98% unchanged. The paper should report this distribution, not just the
mean.

---

## 5. Mechanism: why v1024/v2048 are no-ops, and why the findings’ sentence is wrong

The findings say: “A vocabolari più grandi il prefisso per frequenza contiene
già i merge intra-morfema, e l’assegnazione non cambia più nulla.” The CSV and
the exact provenance do not support that reading. The frequency prefix contains
**more** cross-majority rules at larger vocabularies, and the morph level 1
still contains hundreds of them:

| vocab | seeds | n1 | intra-majority L1 rules (mean) | frequency L1 cross | morph L1 cross | cross rules moved | Δ F1 (morph−freq) |
|---|---:|---:|---:|---:|---:|---:|---:|
| 512 | 3 | 294 | 261.0 (exact pure-intra 252.1; 10-seed range 225–263) | 97.7 | 32.0 | 65.7 | +0.00783 |
| 1024 | 3 | 601 | 270.0 | 334.7 | 330.0 | 4.7 | 0.00000 |
| 2048 | 3 | 1215 | 272.0 | 943.0 | 942.0 | 1.0 | 0.00000 |

(The cross columns are means over the 3 seeds. At v1024/v2048 the CSV’s
`l1_intra`/`l1_cross` are majority-based; at v512 the exact pure-intra count is
252.1 and the CSV majority count is 261.0. Both counts saturate as the
vocabulary grows.)

The correct story is **dilution**, not saturation of the frequency prefix. The
number of intra-morpheme rules saturates at ~252–272 as the vocabulary grows,
while the fixed 60% budget grows to 294 → 601 → 1215. At v1024 the morph level 1
is 55% cross-majority; at v2048 it is 78% cross-majority. The morph ordering has
converged to rank order (W1-18: v2048 Jaccard 1.0; v1024 12 rules differ but the
tokenization is identical), so the assignment is a no-op. This is a
**budget-scaling artifact**, not evidence that morphology stops mattering at
larger vocabularies. A fair larger-vocabulary test would set n1 near the
intra-rule count (or a fixed absolute budget of ~250–270), not 60%.

At v512, the same mechanism explains the fraction sweep: the correlation between
“cross rules moved out of L1” and Δ F1 across the 10 seeds is +0.648. Seed 46,
with the fewest pure-intra rules (225) and the fewest cross rules moved (51),
has Δ F1 = +0.000145 (essentially a tie). Seed 48, with 257 pure-intra rules and
68 moved, has +0.015596. The effect is concentrated where the assignment can
actually separate intra from cross rules.

This also means the “budget window” is not a property of vocabulary size per se.
It is a property of the ratio \(n_1 / \text{intra-rule count}\): ~1.13–1.17 at
v512 (majority vs exact pure-intra), ~2.23 at v1024, ~4.47 at v2048. The
findings’ v512-only claim is consistent with this ratio, but the paper should
state the ratio, not the vocabulary size, and should test the larger
vocabularies at a matched ratio before claiming a window.

---

## 6. What survives, claim by claim

| claim | verdict | basis |
|---|---|---|
| morph > frequency, F1, v512 | **Survives** | 10/10; t p = 0.00011; Bonf/Holm 24 = 0.0026; exact sign ×24 = 0.0469; fresh 45–51 7/7; clean 47–51 5/5 |
| morph > frequency, bpc, v512 | **Survives numerically** | 10/10; t p = 0.00044; Bonf/Holm 24 = 0.0106/0.0102; alpha 0.1/1.0 identical; W1-18: not a compression effect |
| morph > random, bpc, v512 | **Survives** | 10/10; t p = 0.00108; Bonf/Holm 24 = 0.0260/0.0238; exact sign ×24 = 0.0469; ensemble 10/10 |
| morph > random, F1, v512 | **Provisional / fails full correction** | 9/1; t p = 0.00415; Bonf/Holm 24 = 0.0996/0.0872; adjusted CI includes 0; sign ×4 = 0.086; survives m ≤ 12 only; ensemble 10/10 but post hoc; clean 5-seed 4-test Bonferroni = 0.074 |
| “effect confined to v512” | **Not established** | 3-seed interaction p = 0.0517 (F1), 0.115 (bpc); 10-seed run only at v512; exact zeros are 60%-fraction no-ops |
| findings’ “morph improves alignment and compression on all 10 seeds” | **Overbroad** | true for morph−frequency (both metrics) and bpc-vs-random; false for F1-vs-random (9/1); correct the prose |
| “the effect is a window near the morpheme scale” | **Directionally plausible, mechanism misstated** | 0.6 is a tuned threshold at \(n_1\)/intra ≈ 1.1–1.2; 0.5 reverses; larger-vocab no-op is dilution, not frequency-prefix saturation |
| “10 independent seeds” | **Qualified** | 42–44 selected v512; 45–46 seen in the data-review baseline; 47–51 cleanest; all share one synthetic grammar |
| F1 gain is a broad improvement | **No** | 1.9% of words improved, 0% worsened, 98.1% unchanged |

---

## 7. Recommendations

1. **Report the `level1_fraction` sweep** (0.4–0.8) and the pure-intra threshold.
   Either choose the fraction by a rule (n1 = number of pure-intra rules; or a
   fixed absolute budget ~250) or pre-register 0.6 with the sweep as a
   sensitivity analysis. Do not present 0.6 as an unexamined default. Add a
   fine sweep over 0.50–0.56 to locate the transition.
2. **Pre-register a fresh 10+ seed replication at v512** (seeds 52–61) with the
   four-test family, the exact sign test, and multiple random-control draws per
   seed. The clean 5-seed block is too small for a distribution-free
   multiplicity-adjusted test; the fresh 7-seed block is suggestive but its
   exact sign ×4 is 0.0625. A 10+ seed block with 10/10 wins would give exact
   sign p = 0.00195 and survive.
3. **Re-run v1024/v2048 with n1 matched to the pure-intra count** (or a fixed
   absolute budget) before claiming a vocabulary window. This is the single
   check that would most change the scientific verdict: if the effect appears
   at v1024/2048 at matched n1, the “confined to v512” story is a fraction
   artifact; if it disappears, the window is real.
4. **Fix `analyze_levels.py`:** t = NaN / p = 1 for zero-variance zero-effect
   rows; add p-values and Holm/Bonferroni to the output; fix the `T95` fallback
   for n > 10; do not let `t = inf` enter a meta-analysis.
5. **Fix `random_closed`:** deduplicate the available list, assert exactly n1
   unique rules, report the actual L1 size, and use multiple draws per seed (or
   a mixed model) as the primary random-control analysis. W1-14’s ensemble is
   the right design; the single canonical draw should be retained only for
   reproduction.
6. **Correct the paper sentence:** “10/10” applies to morph−frequency and to
   bpc-vs-random; morph−random F1 is 9/1. Report the per-word distribution
   (1.9% improved, 0% worsened) and call bpc “held-out unigram bpc,” not
   “compression” (W1-18).
7. **Update the Phase 2 power calculation.** `docs/embedding-experiment.md`
   (KC5 / power section) uses the pre-fix +0.017–0.019 boundary-F1 advantage at
   512/1024. The corrected level-assignment data give +0.009 at 512 and 0 at
   1024. Recompute the power and the KC5 threshold; otherwise Phase 2 is
   designed around an effect size that the corrected BPE no longer supports.

---

## Appendix — commands and key outputs

All commands were run from `C:\Users\acese\Desktop\unibull`.

### A.1 Reproduce the published stats

```powershell
python -m experiments.analyze_levels `
  --csv results/level_assignment_v512_10seeds.csv `
  --out C:\Users\acese\AppData\Local\Temp\opencode\stats10_repro.csv

python -m experiments.analyze_levels `
  --csv results/level_assignment.csv `
  --out C:\Users\acese\AppData\Local\Temp\opencode\stats3_repro.csv
```

Output: `wrote ...stats10_repro.csv (6 comparisons)` and
`wrote ...stats3_repro.csv (24 comparisons)`. A cell-by-cell comparison against
`results/level_assignment_stats_v512.csv` and
`results/level_assignment_stats.csv` gives **0 mismatched cells** in both
files.

### A.2 Integrity, splits, and per-seed deltas

```powershell
python C:\Users\acese\AppData\Local\Temp\opencode\w1_20_csv_audit.py
```

Key output:

```text
3-seed rows=45  seeds=[42, 43, 44]  vocabs=[512, 1024, 2048]
10-seed rows=40 seeds=[42..51]     vocabs=[512]
frequency != flat metric cells: 0
3-seed v512 rows not reproduced by 10-seed file: 0
10-seed file omits mode(s): ['reverse']
v512 n_l1: frequency 294 x10; morph 294 x10;
          random_closed [294,294,293,294,293,294,294,294,292,294]
```

Held-out splits (F1 / bpc):

```text
morph - frequency
  all 42-51      F1 +0.009243 t=+6.518 p=0.00011 10/0
                 bpc -0.025788 t=-5.384 p=0.00044 0/10
  selection 42-44 F1 +0.007827 t=+4.226 p=0.05169 3/0
                 bpc -0.019825 t=-2.687 p=0.11507 0/3
  fresh 45-51    F1 +0.009850 t=+5.181 p=0.00205 7/0
                 bpc -0.028343 t=-4.612 p=0.00365 0/7
  clean 47-51    F1 +0.011765 t=+8.396 p=0.00110 5/0
                 bpc -0.034745 t=-6.283 p=0.00328 0/5
morph - random_closed
  all 42-51      F1 +0.015629 t=+3.810 p=0.00415 9/1
                 bpc -0.061932 t=-4.725 p=0.00108 0/10
  selection 42-44 F1 +0.013161 t=+1.107 p=0.38351 2/1
                 bpc -0.048465 t=-1.397 p=0.29735 0/3
  fresh 45-51    F1 +0.016687 t=+4.273 p=0.00524 7/0
                 bpc -0.067703 t=-5.081 p=0.00226 0/7
  clean 47-51    F1 +0.019355 t=+3.837 p=0.01850 5/0
                 bpc -0.074757 t=-4.232 p=0.01335 0/5
```

### A.3 Full family correction

```powershell
python C:\Users\acese\AppData\Local\Temp\opencode\w1_20_family.py
```

Key output (combined 24-test family; v512 10-seed rows for the three non-reverse
pairs, v512 reverse + v1024/v2048 3-seed rows otherwise):

```text
    v metric        pair                       src  n       mean        t         p    Bonf24    Holm24
  512 boundary_f1   morph-frequency            10seeds 10  +0.009243   +6.518   0.00011   0.00262   0.00262
  512 boundary_f1   morph-random_closed        10seeds 10  +0.015629   +3.810   0.00415   0.09960   0.08715
  512 bits_per_char morph-frequency            10seeds 10  -0.025788   -5.384   0.00044   0.01062   0.01018
  512 bits_per_char morph-random_closed        10seeds 10  -0.061932   -4.725   0.00108   0.02595   0.02378
 1024 boundary_f1   morph-frequency             3seeds  3  +0.000000     +inf   1.00000   1.00000   1.00000
 2048 boundary_f1   morph-frequency             3seeds  3  +0.000000     +inf   1.00000   1.00000   1.00000
```

The `t=+inf` rows are exact zero effects; their p is set to 1.0, not 0. The
`analyze_levels.py` file itself prints `t=+inf` for these rows (see §3.4).

### A.4 Level-1 fraction sweep, random ensemble, per-word, bpc alpha, mechanism

```powershell
python C:\Users\acese\AppData\Local\Temp\opencode\w1_20_parallel.py
```

Key output (10 seeds, exact `collect_provenance`, 10 corrected random draws per
seed):

```text
A. level1_fraction sensitivity (morph vs flat, F1)
 frac  mean_n1        dF1        t         p   W/L
  0.4    176.5  -0.024949   -4.415   0.00168  0/10
  0.5    221.6  -0.015580   -1.962   0.08138  4/6
  0.6    294.0  +0.009243   +6.518   0.00011  10/0
  0.7    343.0  +0.001387   +4.000   0.00311  8/0 (2 ties)
  0.8    392.0  +0.000043   +1.500   0.16785  2/0 (8 ties)

B. random control ensemble
  morph - original random draw (F1)     mean=+0.015629 t=+3.810 p=0.00415 9/1
  morph - mean of 10 corrected draws    mean=+0.020464 t=+8.199 p=0.00002 10/0
  morph beats all 10 draws in 9/10 seeds
  corrected sampler L1 size = 294 in all draws (original: 292-294)

C. per-word F1 (morph 0.6 vs flat)
  improved=443  worsened=0  tied=22597  (1.92% improved, 0.00% worsened)
  pooled mean=+0.009243 median=0.000000 p05=0 p95=0

D. mechanism at v512
  pure-intra rules 225-263 (mean 252.1); n1=294; frequency L1 cross 88-111;
  morph L1 cross 15-60; moved 51-75; corr(moved, dF1)=+0.648
  seed 46: 225 pure-intra, 51 moved, dF1=+0.000145 (near tie)
  seed 48: 257 pure-intra, 68 moved, dF1=+0.015596

E. bpc alpha
  alpha=0.1: mean=-0.025788 t=-5.384 p=0.00044 0/10
  alpha=1.0: mean=-0.025780 t=-5.378 p=0.00045 0/10
```

### A.5 Random-control size-bug diagnostic

```powershell
python C:\Users\acese\AppData\Local\Temp\opencode\w1_20_random_bug.py
```

```text
{'seed': 44, 'n_rules': 490, 'n1': 294, 'order_len': 491, 'unique': 490,
 'duplicates': [(438, 2)], 'first_dup_indices': [53, 78], 'n_level1': 293}
{'seed': 50, 'n_rules': 490, 'n1': 294, 'order_len': 492, 'unique': 490,
 'duplicates': [(239, 2), (381, 2)], 'first_dup_indices': [4, 45, 63, 108],
 'n_level1': 292}
```

### A.6 Artifacts and hashes

- Input CSV SHA-256:
  `E813CFC019BA0D2D8B46233CCF29573AD7116F8EAEB5D2CCCD10B5A7FFF2F747`
  (`results/level_assignment_v512_10seeds.csv`; same hash as W1-19).
- `w1_20_parallel.py` SHA-256:
  `6FA3F04D81F4FE2958F9D1CC8A1F5F2EB74BD03D0C269356E1616E739E493F21`.
- Full stdout is in the temp logs at the paths printed by the commands; the
  tables above are the durable summary. The scripts only read the project and
  print; they do not write to `results/`.

---

## Bottom line

The v512 **morph − frequency** result is real, reproducible, survives the full
multiplicity correction, and replicates on fresh seeds. The v512
**morph − random_closed F1** result is provisional: it is 9/1, fails the full
family correction and the exact sign correction, and is rescued only by a
post-hoc random-draw ensemble and by fresh-seed t-tests that are too small for a
distribution-free multiplicity-adjusted test. The v512 claim as a whole is
**conditional on a narrow 60% level-1 budget**: at 50% the effect reverses, and
the 60% default sits just above the pure-intra threshold. The “confined to
v512” sentence is a 3-seed descriptive observation, not a tested window, and its
stated mechanism is wrong: the larger-vocabulary no-op is dilution of the morph
level 1, not saturation of the frequency prefix.
