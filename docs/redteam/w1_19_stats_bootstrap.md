# W1-19 — Bootstrap/permutation audit of the v512 level-assignment result

**Agent:** W1-19 (red team, read-only)
**Date:** 2026-11-22
**Python:** 3.13.13, Windows (PowerShell)
**Data:** `results/level_assignment_v512_10seeds.csv`, 40 rows, SHA-256
`E813CFC019BA0D2D8B46233CCF29573AD7116F8EAEB5D2CCCD10B5A7FFF2F747`
**Analysis script:** `%TEMP%\opencode\w1_19_bootstrap.py`, SHA-256
`79EDAF7B83029529741DD499946CDB49C5609F401526DA96437B62E043AAE0E7`.
It only reads the CSV and prints; it writes nothing.
**Constraint:** no project source was modified. The only intentional project
write is this report.

---

## Verdict

**The v512 morph − frequency result survives.** The +0.0092 F1 and −0.0258 bpc
effects are not driven by a single seed. Both paired bootstrap CIs exclude zero,
the exact sign-flip/Wilcoxon/sign tests are at the smallest p attainable with
10 pairs (2/1024 = 0.00195), and dropping the best seed, the worst seed, or both
extremes leaves the sign and significance intact.

**The morph − random_closed result survives for bpc but is only provisional for
F1.** The F1 arm is 9/1 (sign test p = 0.0215; after dropping the best seed,
8/1, p = 0.0391), and it is conditional on a single random control draw per
seed. It should be described as supporting evidence, not as a second decisive
result.

**Strongest finding:** the headline effect is seed-robust but small relative to
between-seed variation (0.21 between-seed SD for F1, −0.44 for bpc), and the
permutation test hits its resolution floor. The published t p-values (1.1e-4,
4.4e-4) cannot be resolved by a 10-pair permutation test; the permutation test
can only say p < 0.002.

**The v512 "window" is post hoc.** At v1024/v2048, morph − frequency is exactly
0.000 F1 and 0.000 bpc on all 3 seeds, so the effect is confined to v512 in this
dataset. That is a real pattern, but it is 3 seeds and was selected after seeing
the larger-vocabulary nulls. It needs independent v512 replication.

---

## 1. Data and design

The CSV contains 10 seeds (42–51), vocabulary 512, and four modes: `flat`,
`frequency`, `morph`, `random_closed`. The three experimental modes share the
same flat BPE rules and vocabulary within a seed; only the assignment of rules
to the two levels changes. `frequency` is the flat prefix-by-rank control.

Each seed generates a fresh synthetic corpus (`generate_synthetic(seed=seed)`):
120 roots, 9,216 training words, 2,304 held-out words, and a 2,000-token test
text. Roots are nearly disjoint across seeds: pairwise overlap is 0–5 roots out
of 120 (total distinct roots 1,133). The held-out sets are therefore new draws,
not the same words relabeled.

Sanity checks on the CSV:

- `flat` and `frequency` agree exactly on every seed and both metrics
  (max absolute difference 0.0 over 20 comparisons). This is the expected
  post-fix identity and a strong internal-consistency check.
- `decode_word = 1.0` and `decode_doc = 1.0` for all 40 rows.
- `morph` has `l1_intra = 233–278`; `frequency` has 182–205; `random_closed` has
  191–221. `morph` has `l2_intra = 0` in all 10 seeds. The treatment does what
  it says: level 1 is the intra-morpheme level.
- `random_closed` has `n_l1 = 294` in 7 seeds, 293 in seeds 44 and 46, and 292
  in seed 50, versus 294 for `morph`/`frequency`. The random control is therefore
  only approximately size-matched (0–2 rules short in 3/10 seeds). This is a
  minor implementation asymmetry, not enough to explain the effect, but it
  should be fixed for exactness.

Group means:

| mode | boundary F1 | bits/char |
|---|---:|---:|
| frequency | 0.874216 | 2.300324 |
| morph | 0.883459 | 2.274536 |
| random_closed | 0.867830 | 2.336468 |

Thus morph − frequency is +0.009243 F1 (+1.06% relative) and −0.025788 bpc
(−1.12% relative). Morph − random_closed is +0.015629 F1 (+1.79%) and −0.061932
bpc (−2.69%).

---

## 2. Methods

All tests are two-sided and paired by seed.

1. **Published t interval.** For paired deltas \(d_i\), mean \(\bar d\),
   sample SD \(s\), SE \(s/\sqrt{10}\), \(t=\bar d/\mathrm{SE}\), df = 9,
   \(t_{0.975,9}=2.262\). This reproduces the intervals in
   `docs/findings-v0.2.md`.
2. **Paired bootstrap.** Resample the 10 seed-pairs with replacement
   (B = 200,000, NumPy PCG64, fixed seed 20261122), recompute the mean delta.
   Report the percentile 95% interval and the BCa 95% interval. BCa uses the
   standard bias correction \(z_0\) and jackknife acceleration; it is the more
   reliable small-n interval when the deltas are skewed.
3. **Exact sign-flip permutation.** Permuting the two mode labels within each
   seed is equivalent to flipping the sign of that seed's paired delta. With
   n = 10 there are \(2^{10}=1024\) assignments; all were enumerated. Report
   the two-sided p for the mean and for the t statistic. The smallest attainable
   two-sided p is \(2/1024 = 0.001953\); comparisons with all 10 deltas in one
   direction therefore hit the floor.
4. **Distribution-free paired tests.** Exact Wilcoxon signed-rank (enumeration
   with tied ranks) and the exact sign test. These do not assume normality.
5. **Influence checks.** Leave-one-seed-out means and t intervals; explicit
   drop-best and drop-worst (defined by improvement direction: largest positive
   delta for F1, most negative delta for bpc); drop both extremes; dfbeta
   \(=\bar d - \bar d_{(-i)}\).
6. **Multiplicity.** Holm and Bonferroni over the 6 v512 comparisons
   (3 pairs × 2 metrics), with a sensitivity check over the broader
   18-comparison family (3 vocabularies × 3 pairs × 2 metrics).

The bootstrap resamples seeds, not words. That is the right unit for the claim
"on a new corpus from this generator," because each seed has a new corpus and a
new held-out set. It does not capture uncertainty from the choice of generator,
from the synthetic morphology, or from the single random control draw. Those are
external-validity limits, discussed in §7.

---

## 3. Results

### 3.1 morph − frequency

Per-seed F1 deltas (all positive):

```text
seed 42 +0.004123   seed 47 +0.009144
seed 43 +0.009722   seed 48 +0.015596
seed 44 +0.009635   seed 49 +0.014540
seed 45 +0.009983   seed 50 +0.010793
seed 46 +0.000145   seed 51 +0.008753
```

Per-seed bpc deltas (all negative):

```text
seed 42 -0.005449   seed 47 -0.024805
seed 43 -0.024135   seed 48 -0.049437
seed 44 -0.029892   seed 49 -0.034473
seed 45 -0.023677   seed 50 -0.044482
seed 46 -0.000997   seed 51 -0.020530
```

| check | F1 | bpc |
|---|---:|---:|
| mean | +0.009243 | −0.025788 |
| SD | 0.004484 | 0.015148 |
| SE | 0.001418 | 0.004790 |
| t(9) | +6.518 | −5.384 |
| t p (two-sided) | 0.000109 | 0.000442 |
| Cohen's dz | +2.06 | −1.70 |
| t 95% CI | [+0.006036, +0.012451] | [−0.036623, −0.014952] |
| bootstrap percentile 95% | [+0.006506, +0.011769] | [−0.034602, −0.016843] |
| BCa 95% | [+0.006188, +0.011525] | [−0.034489, −0.016678] |
| exact sign-flip, mean | p = 0.001953 | p = 0.001953 |
| exact sign-flip, t | p = 0.001953 | p = 0.001953 |
| exact Wilcoxon | p = 0.001953 | p = 0.001953 |
| exact sign test | 10/0, p = 0.001953 | 0/10, p = 0.001953 |

The bootstrap CIs are slightly narrower than the published t intervals and
remain comfortably on the same side of zero. The bootstrap distribution had
0/200,000 resampled means on the wrong side of zero for both metrics.

Drop-one and drop-both:

| subset | mean F1 | BCa F1 | mean bpc | BCa bpc |
|---|---:|---:|---:|---:|
| full (n=10) | +0.009243 | [+0.006188, +0.011525] | −0.025788 | [−0.034489, −0.016678] |
| drop best (seed 48) | +0.008537 | [+0.005369, +0.010640] | −0.023160 | [−0.030849, −0.014276] |
| drop worst (seed 46) | +0.010254 | [+0.008230, +0.012275] | −0.028542 | [−0.036728, −0.020484] |
| drop both (n=8) | +0.009587 | [+0.007545, +0.011344] | −0.025930 | [−0.032879, −0.018181] |

All leave-one-out means stay in [+0.008537, +0.010254] for F1 and
[−0.028542, −0.023160] for bpc; every leave-one-out t interval excludes zero and
every leave-one-out exact sign-flip p is ≤ 0.003906. The largest influence is
seed 46 for F1 (dfbeta −0.001011, 11% of the mean) and seed 46 for bpc
(dfbeta +0.002755, 11%). No single seed drives the result.

### 3.2 morph − random_closed

Per-seed F1 deltas (9 positive, 1 negative):

```text
seed 42 -0.005093   seed 47 +0.036082
seed 43 +0.009100   seed 48 +0.024556
seed 44 +0.035475   seed 49 +0.016722
seed 45 +0.011878   seed 50 +0.011820
seed 46 +0.008154   seed 51 +0.007595
```

Per-seed bpc deltas (all negative):

```text
seed 42 -0.001137   seed 47 -0.118290
seed 43 -0.028164   seed 48 -0.081753
seed 44 -0.116095   seed 49 -0.100673
seed 45 -0.036950   seed 50 -0.055306
seed 46 -0.063182   seed 51 -0.017765
```

| check | F1 | bpc |
|---|---:|---:|
| mean | +0.015629 | −0.061932 |
| SD | 0.012970 | 0.041446 |
| SE | 0.004102 | 0.013106 |
| t(9) | +3.810 | −4.725 |
| t p (two-sided) | 0.004150 | 0.001081 |
| Cohen's dz | +1.20 | −1.49 |
| t 95% CI | [+0.006351, +0.024907] | [−0.091578, −0.032285] |
| bootstrap percentile 95% | [+0.008252, +0.023460] | [−0.086346, −0.037760] |
| BCa 95% | [+0.008602, +0.023875] | [−0.086435, −0.037854] |
| exact sign-flip, mean | p = 0.003906 | p = 0.001953 |
| exact sign-flip, t | p = 0.003906 | p = 0.001953 |
| exact Wilcoxon | p = 0.003906 | p = 0.001953 |
| exact sign test | 9/1, p = 0.021484 | 0/10, p = 0.001953 |

The F1 result is not at the permutation floor (4/1024 = 0.003906), but the sign
test is much weaker (p = 0.0215) because it ignores magnitudes. The bpc result
is 10/10 and at the floor.

Drop-one and drop-both:

| subset | mean F1 | BCa F1 | mean bpc | BCa bpc |
|---|---:|---:|---:|---:|
| full (n=10) | +0.015629 | [+0.008602, +0.023875] | −0.061932 | [−0.086435, −0.037854] |
| drop best (seed 47) | +0.013356 | [+0.007132, +0.021406] | −0.055669 | [−0.080569, −0.032992] |
| drop worst (seed 42) | +0.017931 | [+0.012102, +0.026352] | −0.068686 | [−0.092049, −0.045964] |
| drop both (n=8) | +0.015662 | [+0.010808, +0.024378] | −0.062486 | [−0.086598, −0.041202] |

All drop-one F1 intervals still exclude zero, but the F1 sign test is 8/1
(p = 0.0391) whenever any positive seed is dropped; only dropping the negative
seed 42 gives 9/0 (p = 0.0039). The F1 advantage is carried by the magnitudes of
a few large wins (seeds 47 and 44), not by a consistent 10/10 win record. The bpc
arm is consistently negative and remains so under every drop.

### 3.3 Compact robustness table

```text
morph-frequency/F1: mean=+0.009243 t_p=0.000109 pct=[+0.006506,+0.011769] bca=[+0.006188,+0.011525]
   drop_best:  mean=+0.008537 t_p=0.000257 CI95=[+0.005366,+0.011709] bca=[+0.005369,+0.010640]
   drop_worst: mean=+0.010254 t_p=0.000015 CI95=[+0.007690,+0.012818] bca=[+0.008230,+0.012275]
   drop_both:  mean=+0.009587 t_p=0.000030 CI95=[+0.007203,+0.011971] bca=[+0.007545,+0.011344]
morph-frequency/bpc: mean=-0.025788 t_p=0.000442 pct=[-0.034602,-0.016843] bca=[-0.034489,-0.016678]
   drop_best:  mean=-0.023160 t_p=0.000851 CI95=[-0.033486,-0.012834] bca=[-0.030849,-0.014276]
   drop_worst: mean=-0.028542 t_p=0.000185 CI95=[-0.038646,-0.018439] bca=[-0.036728,-0.020484]
   drop_both:  mean=-0.025930 t_p=0.000334 CI95=[-0.035364,-0.016497] bca=[-0.032879,-0.018181]
morph-random_closed/F1: mean=+0.015629 t_p=0.004150 pct=[+0.008252,+0.023460] bca=[+0.008602,+0.023875]
   drop_best:  mean=+0.013356 t_p=0.008094 CI95=[+0.004553,+0.022159] bca=[+0.007132,+0.021406]
   drop_worst: mean=+0.017931 t_p=0.001493 CI95=[+0.009180,+0.026683] bca=[+0.012102,+0.026352]
   drop_both:  mean=+0.015662 t_p=0.002666 CI95=[+0.007504,+0.023821] bca=[+0.010808,+0.024378]
morph-random_closed/bpc: mean=-0.061932 t_p=0.001081 pct=[-0.086346,-0.037760] bca=[-0.086435,-0.037854]
   drop_best:  mean=-0.055669 t_p=0.002530 CI95=[-0.085354,-0.025985] bca=[-0.080569,-0.032992]
   drop_worst: mean=-0.068686 t_p=0.000595 CI95=[-0.097644,-0.039729] bca=[-0.092049,-0.045964]
   drop_both:  mean=-0.062486 t_p=0.001486 CI95=[-0.091769,-0.033203] bca=[-0.086598,-0.041202]
```

---

## 4. Findings

**F1. The headline morph − frequency effect is seed-robust.** Ten of ten F1
deltas are positive and ten of ten bpc deltas are negative. The BCa intervals
are [+0.0062, +0.0115] F1 and [−0.0345, −0.0167] bpc. Dropping the best seed
leaves +0.0085 / −0.0232; dropping the worst leaves +0.0103 / −0.0285; dropping
both leaves +0.0096 / −0.0259. No single seed is necessary for the sign or the
5% significance.

**F2. The permutation test is at its resolution floor, so the t p-values
overstate the permutation evidence.** With 10 pairs the exact two-sided sign-flip
p cannot go below 2/1024 = 0.00195. Both headline metrics hit that floor. The
t-tests give 1.1e-4 and 4.4e-4 under a normality model; the exact permutation
test only establishes p < 0.002. This is not a contradiction, but the report
should not present 1e-4 as a distribution-free result. Normality is not rejected
(Shapiro–Wilk p = 0.28 for F1, 0.73 for bpc), and the t and bootstrap intervals
agree, so the t interval is defensible.

**F3. The morph − random_closed F1 result is the weak link.** It is 9/1, the
sign test is p = 0.0215, and after dropping the best seed it is 8/1 with p =
0.0391. The bootstrap and BCa intervals still exclude zero, and the exact
sign-flip p is 0.0039, but the evidence is carried by a few large wins rather
than a consistent win record. The bpc arm is much stronger (10/10, BCa
[−0.0864, −0.0379], all drops negative).

**F4. The random control is a single draw per seed, and it is slightly
under-budget.** `assign_levels(..., mode="random_closed", seed=seed*1000+512)`
produces one random topological prefix per seed. Its level-1 size is 294 in
7/10 seeds, 293 in seeds 44 and 46, and 292 in seed 50, versus 294 for
morph/frequency. The SD of the morph − random_closed *delta* is 0.0130 F1 /
0.0414 bpc, about 2.8× the morph − frequency delta SD; the between-seed SD of
random_closed itself is 0.0416 F1 / 0.0489 bpc. A single draw cannot distinguish
"morph beats a generic random assignment" from "morph beats these particular
random draws." This is the single most important design check remaining (§7).

**F5. The effect is small relative to between-seed variation; the pairing is
doing the work.** The between-seed SD is 0.0451 F1 / 0.0639 bpc for morph and
0.0420 / 0.0539 for frequency. The paired delta SD is only 0.0045 / 0.0151.
Effect / between-seed SD is +0.212 F1 and −0.436 bpc. An unpaired comparison
would have SE ≈ 0.0195 / 0.0264 and would not detect the effect (t ≈ 0.47 /
−0.98). The result is a within-corpus paired effect, not evidence that a morph
tokenizer is uniformly better in absolute terms across corpora.

**F6. F1 and bpc are not independent evidence.** The across-seed correlation
between the paired ΔF1 and Δbpc is −0.911 (morph − frequency) and −0.908 (morph −
random_closed). Both metrics move together because they are two views of the same
tokenization change. Agreement is still useful, but it should not be counted as
two independent replications.

**F7. The headline survives multiplicity; the random_closed F1 arm does not
survive the broadest correction.** Over the 6 v512 comparisons, Holm and
Bonferroni leave all four morph comparisons significant at 0.05
(morph − frequency F1: 0.00066; bpc: 0.00266; morph − random_closed bpc: 0.00649;
F1: 0.02490). Over a worst-case 18-comparison family (3 vocabularies × 3 pairs ×
2 metrics), the headline morph − frequency tests still pass (0.00196 and
0.00796), but morph − random_closed F1 becomes 0.0747. The v512 window itself is
post hoc, so the 18-test correction is a sensitivity check, not a pre-registered
analysis.

**F8. The result is conditional on the synthetic generator.** The corpus has
clean prefix–root–suffix boundaries, and the morph assignment uses the same
provenance/boundary definition that the F1 metric evaluates. That is the intended
controlled ablation, but it also means the F1 gain could shrink on a real corpus
with noisy or differently defined morpheme boundaries. The v1024/v2048 exact
ties (3 seeds, Δ = 0.000) support a budget-window interpretation but do not
establish it.

---

## 5. Does the v512 result survive?

| claim | verdict | basis |
|---|---|---|
| morph − frequency F1 = +0.0092 | **Survives, robust** | 10/10 wins; BCa [+0.0062, +0.0115]; drop-best/worst/both positive; sign/permutation p = 0.001953 (floor); survives 6- and 18-test correction |
| morph − frequency bpc = −0.0258 | **Survives, robust** | 10/10 wins; BCa [−0.0345, −0.0167]; drop-best/worst/both negative; sign/permutation p = 0.001953 (floor); survives 6- and 18-test correction |
| morph − random_closed bpc = −0.0619 | **Survives, robust** | 10/10 wins; BCa [−0.0864, −0.0379]; drop-best/worst/both negative; p = 0.001953 (floor) |
| morph − random_closed F1 = +0.0156 | **Survives statistically, provisional** | 9/1; BCa [+0.0086, +0.0239]; sign p = 0.0215; drop-best sign p = 0.0391; one random draw per seed; fails 18-test Bonferroni (0.075) |
| effect confined to v512 | **Confined in this dataset; general window not established** | v1024/v2048 exact ties on only 3 seeds; v512 selected after seeing them |
| "morphological provenance, not generic reordering" | **Supported against one random draw, not decisive** | morph beats one random_closed draw per seed; needs repeated random controls |

---

## 6. Commands and outputs

All commands were run from the project root. The analysis script is
`%TEMP%\opencode\w1_19_bootstrap.py`; the full output is
`%TEMP%\opencode\w1_19_output_final.txt`.

```powershell
python %TEMP%\opencode\w1_19_bootstrap.py
```

The script reads `results/level_assignment_v512_10seeds.csv`, builds the paired
deltas, runs the t interval, 200,000-resample percentile/BCa bootstrap, exact
2^10 sign-flip test (mean and t), exact Wilcoxon, exact sign test, leave-one-out,
drop-best/worst/both, and the multiplicity summaries. It writes only to stdout.

Key output (compact robustness table):

```text
morph-frequency/F1: mean=+0.009243 t_p=0.000109 pct=[+0.006506,+0.011769] bca=[+0.006188,+0.011525] sf_p=0.001953
   drop_best:  mean=+0.008537 t_p=0.000257 CI95=[+0.005366,+0.011709] bca=[+0.005369,+0.010640] sf_p=0.003906
   drop_worst: mean=+0.010254 t_p=0.000015 CI95=[+0.007690,+0.012818] bca=[+0.008230,+0.012275] sf_p=0.003906
   drop_both:  mean=+0.009587 t_p=0.000030 CI95=[+0.007203,+0.011971] bca=[+0.007545,+0.011344] sf_p=0.007812
morph-frequency/bpc: mean=-0.025788 t_p=0.000442 pct=[-0.034602,-0.016843] bca=[-0.034489,-0.016678] sf_p=0.001953
   drop_best:  mean=-0.023160 t_p=0.000851 CI95=[-0.033486,-0.012834] bca=[-0.030849,-0.014276] sf_p=0.003906
   drop_worst: mean=-0.028542 t_p=0.000185 CI95=[-0.038646,-0.018439] bca=[-0.036728,-0.020484] sf_p=0.003906
   drop_both:  mean=-0.025930 t_p=0.000334 CI95=[-0.035364,-0.016497] bca=[-0.032879,-0.018181] sf_p=0.007812
morph-random_closed/F1: mean=+0.015629 t_p=0.004150 pct=[+0.008252,+0.023460] bca=[+0.008602,+0.023875] sf_p=0.003906
   drop_best:  mean=+0.013356 t_p=0.008094 CI95=[+0.004553,+0.022159] bca=[+0.007132,+0.021406] sf_p=0.007812
   drop_worst: mean=+0.017931 t_p=0.001493 CI95=[+0.009180,+0.026683] bca=[+0.012102,+0.026352] sf_p=0.003906
   drop_both:  mean=+0.015662 t_p=0.002666 CI95=[+0.007504,+0.023821] bca=[+0.010808,+0.024378] sf_p=0.007812
morph-random_closed/bpc: mean=-0.061932 t_p=0.001081 pct=[-0.086346,-0.037760] bca=[-0.086435,-0.037854] sf_p=0.001953
   drop_best:  mean=-0.055669 t_p=0.002530 CI95=[-0.085354,-0.025985] bca=[-0.080569,-0.032992] sf_p=0.003906
   drop_worst: mean=-0.068686 t_p=0.000595 CI95=[-0.097644,-0.039729] bca=[-0.092049,-0.045964] sf_p=0.003906
   drop_both:  mean=-0.062486 t_p=0.001486 CI95=[-0.091769,-0.033203] bca=[-0.086598,-0.041202] sf_p=0.007812
```

Independent SciPy cross-check (t p-values, Wilcoxon, sign test):

```text
t=6.518 df=9 scipy_two=0.00010914
t=5.384 df=9 scipy_two=0.00044217
t=3.810 df=9 scipy_two=0.00415312
t=4.725 df=9 scipy_two=0.00108146
t=2.262 df=9 scipy_two=0.05001285
wilcoxon morph-random F1: statistic=1.0 pvalue=0.00390625
binom 9/10 pvalue=0.021484375
```

Synthetic test-set inspection (independence of seeds):

```text
seed=42 n_test_words=2304 n_train_words=9216 n_test_roots=120 test_tokens=2000
...
seed=51 n_test_words=2304 n_train_words=9216 n_test_roots=120 test_tokens=2000
pairwise root overlap: 0-5 roots out of 120; total distinct roots across seeds: 1133
```

Reproduction check (not a full rerun): on this machine the pipeline is ~10×
slower than the CSV's recorded `level_seconds`. For seed 42:

```text
gen 3.95s; train 55.41s; provenance 93.76s; eval flat 61.14s
flat F1=0.866507523  bpc=2.353869481   # exact match to the CSV row
```

The flat metrics reproduce exactly. A full 10-seed retrain plus even 10 random
control draws per seed would take hours on this machine, so the random-control
repetition in §7 was not run; the report is a reanalysis of the committed CSV,
not an independent rerun.

---

## 7. Limitations and the single check that would most change the verdict

**Limitations.**

1. **Seed-level, not word-level, bootstrap.** The bootstrap captures
   corpus-to-corpus and held-out-set-to-held-out-set variation (each seed draws a
   new corpus and test set), which is the broader generalization target. It does
   not give a word-level interval for a fixed corpus. A word-level bootstrap
   would require re-running the pipeline and retaining per-word F1; it would
   answer a different (fixed-corpus) question and cannot be computed from the
   CSV.
2. **One synthetic generator, 10 seeds.** The 10 seeds are independent draws
   from the same generator (nearly disjoint root inventories), not independent
   languages or corpora. The result is conditional on this generator's clean
   prefix–root–suffix structure.
3. **One random control draw per seed.** The `random_closed` control is a
   single random topological prefix per seed. Its level-1 size is 294 in 7/10
   seeds, 293 in two, and 292 in one, versus 294 for morph/frequency. The
   morph − random_closed F1 arm is the least robust part of the result.
4. **bpc is a unigram model.** The secondary metric uses add-α 0.1 on a unigram
   distribution over tokens, not a context model. It is a legitimate compression
   proxy, but a different model could change the bpc effect.
5. **The v512 window is post hoc.** v1024/v2048 show exact 0.000 deltas on 3
   seeds. That is consistent with a budget window, but it is not an independent
   confirmation of one.

**The single check that would most change the verdict:** repeat
`random_closed` with 20–50 independent random orderings per seed and test morph
against the *distribution* of random controls, not one draw per seed. Compute
the average random-control metric per seed, the paired morph − average-random
delta, and the fraction of individual random draws that beat morph. If the
average random control closes the +0.0156 F1 gap, the claim "morphological
provenance, not generic reordering" fails, and the paper should fall back to the
morph − frequency result. The morph − random_closed F1 arm is the only
comparison whose significance is not already at the permutation floor and whose
control is a single draw. The headline morph − frequency effect would probably
survive this check; the stronger mechanistic claim would not.

For external validity, the complementary check is a real morphological corpus
with independently annotated boundaries, evaluated with the same
morph/frequency/random_closed ablation. That is the check most likely to change
the scientific conclusion, but it is a different study, not a statistical
robustness check on the existing data.
