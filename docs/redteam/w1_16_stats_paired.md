# W1-16 — Audit of the paired t-test (level assignment, v512)

**Auditor:** W1-16 (red team, statistics)
**Date:** 2026-11-22
**Inputs:** `results/level_assignment_v512_10seeds.csv`,
`results/level_assignment_stats_v512.csv`, `experiments/analyze_levels.py`,
`experiments/level_assignment.py`, `ramified/synthetic.py`, `ramified/levels.py`,
`ramified/metrics.py`.
**Status:** read-only audit. No source file was modified by W1-16; the only
writes were temporary CSVs/scripts outside the repository and this report. Note
that `ramified/metrics.py` changed on disk during the broader red-team session
(see the source-state note in §1).

## Bottom line

The paired t-test is implemented correctly and reproduces the published table
exactly. For the primary `morph − frequency` contrast the assumptions are not
violated: the ten seeds are independent draws from the generator, the deltas are
consistent with normality (Shapiro p = 0.28 for F1, p = 0.73 for bpc), and exact
nonparametric tests confirm the result (Wilcoxon/sign-flip p = 0.00195, 10/10
signs, 9 clear + 1 near-tie). The v512 result **survives the statistical audit
as a narrow, conditional finding**.

It does **not** survive as a general claim about morphological tokenization.
The t statistic is not inflated by a code bug or test leakage, but it is
conditional on a fixed, oracle-labelled synthetic generator. In a 10-seed
probe, changing the root inventory from 120 to 60 removes the effect
(`ΔF1 = +0.000087`, 2/10 wins, 8 exact ties), and in the available 3-seed data
at v1024/2048 the effect is exactly zero (3/3 ties). Within the ten
published seeds, the F1 gain falls by 0.000265 for every cross-morpheme rule
left in level 1 (`R² = 0.61`, p = 0.008). The effect is a budget-purity /
construction window, not a stable morphological advantage.

The supported sentence is therefore: *on this synthetic generator, at v512,
with oracle morphological provenance and a 294/196 level budget, morph
assignment improves boundary F1 by about 0.9 points and bpc by about 0.026 over
flat/frequency; the effect is absent at v1024/2048 and at 60 roots, and remains
to be tested on a real corpus.* The paper's stronger sentence is not supported.

## 1. What the published test does

`experiments/analyze_levels.py`:

- `paired()` builds `(seed, mode)` pairs and computes `a − b` for each seed. It
  silently drops a seed if either mode is missing, but there are no missing
  pairs here.
- `describe()` computes the mean, sample SD (`ddof=1`), `SE = SD/√n`,
  `t = mean/SE`, and a two-sided 95% CI using the hard-coded `T95[n]` table.
  For n = 10 it uses 2.262, the correct two-sided t critical value at df = 9.
- The test is two-sided and paired. The code is numerically correct.
- It does not report an effect size, power, or multiplicity adjustment. It
  prints six comparisons (2 metrics × 3 mode pairs).

The published CSV is reproduced exactly:

```text
vocab=512
  boundary_f1
    morph         - frequency  mean=+0.0092 std=0.0045 t=+6.52 CI95=[+0.0060,+0.0125] W/L/T=10/0/0
    random_closed - frequency  mean=-0.0064 std=0.0117 t=-1.73 CI95=[-0.0147,+0.0020] W/L/T=3/7/0
    morph         - random_closed mean=+0.0156 std=0.0130 t=+3.81 CI95=[+0.0064,+0.0249] W/L/T=9/1/0
  bits_per_char
    morph         - frequency  mean=-0.0258 std=0.0151 t=-5.38 CI95=[-0.0366,-0.0150] W/L/T=0/10/0
    random_closed - frequency  mean=+0.0361 std=0.0376 t=+3.04 CI95=[+0.0092,+0.0631] W/L/T=8/2/0
    morph         - random_closed mean=-0.0619 std=0.0414 t=-4.73 CI95=[-0.0916,-0.0323] W/L/T=0/10/0
```

The 40-row input has no duplicate `(seed, vocab, mode)` rows, no missing modes,
and no level-size/count inconsistencies. `flat` and `frequency` are
bit-identical on every metric (max absolute difference 0), confirming that the
budget partition is a no-op and that `morph − frequency` is the same contrast as
`morph − flat`.

### Source state

SHA-256 (first 16 hex chars) at the end of this audit:

| file | hash |
|---|---|
| `ramified/metrics.py` | `14DBF6B1794ABFD9` |
| `ramified/bpe.py` | `73093A3E62FC66A5` |
| `ramified/levels.py` | `14420E923EE6C403` |
| `ramified/synthetic.py` | `E7D4E5FBB01E86C4` |
| `experiments/analyze_levels.py` | `1204507E4EA9E1DB` |
| `experiments/level_assignment.py` | `167C55CC81EE3736` |

`ramified/metrics.py` changed on disk during the broader red-team session
(W1-17): the one-sided-empty precision/recall convention changed from 1.0 to
0.0, and `boundary_report` was added. This audit did not make that change. The
v512 F1 column is invariant because no test word has an empty predicted or true
boundary set (W1-17 checked all 92,160 tokenizer/word cases); seed 42 reproduces
exactly under the current code. The bpc estimator in `level_assignment.py`
(add-0.1) is unaffected; W1-18 documents the separate add-1 divergence in
`metrics.py`.

## 2. Independence of seeds

The ten seeds 42–51 are independent PRNG streams; they are not resamples of one
word list. Test-word Jaccard overlap across seed pairs averages 0.0015 (max
0.0048), and test-root Jaccard averages 0.0072 (max 0.0213). There is no seed
order trend: Spearman ρ(seed, ΔF1) = +0.333 (p = 0.347), lag-1 r = +0.196;
for bpc ρ = −0.394 (p = 0.260), lag-1 r = +0.182. The leave-one-seed-out t
statistics stay in [5.76, 9.22] for F1 and [−6.51, −4.73] for bpc, so no single
seed drives the result. W1-20’s held-out-seed split (45–51, not used to choose
the mode family) replicates 7/7 on both metrics; the cleanest block 47–51 is 5/5
(t p = 0.0011 F1, 0.0033 bpc).

Two dependence caveats matter:

1. **The same ten seeds are reused across modes, metrics and vocabulary sizes.**
   The six printed tests are not independent. The F1 and bpc deltas correlate
   −0.91 (p = 0.0002), so they are two readouts of one effect, not independent
   confirmations. Treat the family as three mode comparisons, not six.
2. **The test words are clustered by root.** Each seed has 2,304 held-out words
   drawn from 120 roots, i.e. 19.2 words per root. The seed mean is the correct
   unit; a word-level n = 2,304 would be pseudo-replication. The t-test uses
   n = 10, so this is handled correctly, but “2,304 held-out words” must not be
   read as 2,304 independent observations.

Independence at the seed level is therefore satisfied. The limitation is the
population, not the iid assumption: the ten seeds are ten draws from **one fixed
generator**, not ten draws from morphological corpora. Formally, the t-test
treats the ten seed effects as iid normal draws; the exact sign-flip test
relaxes normality but still treats the seeds as exchangeable. Neither test
samples new generators, so neither can estimate generator-level uncertainty.

## 3. Normality of the deltas

| metric | comparison | skew | excess kurtosis | Shapiro W (p) | Anderson A² (5% crit 0.684) | JB p |
|---|---|---:|---:|---:|---:|---:|
| boundary_f1 | morph − frequency | −0.74 | +1.04 | 0.910 (0.283) | 0.514 | 0.725 |
| boundary_f1 | random_closed − frequency | −0.98 | +0.23 | 0.864 (0.085) | 0.675 | 0.548 |
| boundary_f1 | morph − random_closed | +0.44 | −0.25 | 0.916 (0.328) | 0.449 | 0.810 |
| bits_per_char | morph − frequency | +0.13 | −0.23 | 0.955 (0.728) | 0.265 | 0.899 |
| bits_per_char | random_closed − frequency | +0.45 | −1.61 | 0.878 (0.125) | 0.511 | 0.571 |
| bits_per_char | morph − random_closed | −0.08 | −1.36 | 0.946 (0.619) | 0.220 | 0.695 |

The primary morph deltas show no evidence of non-normality. The only borderline
case is the null `random_closed − frequency` F1 comparison (Shapiro p = 0.085,
Anderson A² = 0.675 vs 0.684), which is not part of the claim. With n = 10 the
normality tests have low power; the exact nonparametric tests in §5 are the
safeguard, and they agree. F1 is bounded and discrete, but the observed values
are 0.80–0.94, away from the ceiling; the delta–baseline correlation is
**positive** (+0.650, p = 0.042), so the gain is not a ceiling artefact.

## 4. Effect size

| metric | comparison | mean Δ | relative | d_z (95% CI) | d_av | P(superiority) |
|---|---|---:|---:|---:|---:|---:|
| boundary_f1 | morph − frequency | +0.009243 | +1.06% | +2.061 [+1.241, +3.977] | +0.212 | 10/10 [0.692, 1.0] |
| boundary_f1 | random_closed − frequency | −0.006386 | −0.73% | −0.547 [−1.480, +0.076] | −0.153 | 3/10 [0.067, 0.652] |
| boundary_f1 | morph − random_closed | +0.015629 | +1.80% | +1.205 [+0.550, +2.520] | +0.360 | 9/10 [0.555, 0.997] |
| bits_per_char | morph − frequency | −0.025788 | −1.12% | −1.702 [−3.357, −0.961] | −0.436 | 10/10 lower |
| bits_per_char | random_closed − frequency | +0.036144 | +1.57% | +0.961 [+0.331, +2.123] | +0.702 | 8/10 |
| bits_per_char | morph − random_closed | −0.061932 | −2.65% | −1.494 [−3.003, −0.793] | −1.088 | 10/10 lower |

`d_z` is the paired standardized effect, `mean(Δ)/SD(Δ)`. `d_av` is
`mean(Δ)/√((SD_a² + SD_b²)/2)`, the standardized effect with the between-corpus
SD in the denominator. The distinction is decisive here:

- `corr(morph F1, frequency F1)` across seeds is **0.9971**; pairing removes
  **99.5%** of the variance of the difference for F1 (96.7% for bpc).
- The between-seed SD of baseline F1 is 0.0420, **9.4×** the SD of the paired
  difference (0.00448).
- Consequently `d_z = 2.06` but `d_av = 0.21`. The former is a within-corpus
  effect; the latter is the effect relative to corpus-to-corpus variation.
  Reporting `d_z = 2.06` without this context overstates the practical effect.

The raw effect is small: F1 0.8742 → 0.8835, i.e. boundary error 12.58% →
11.65%, a **7.35% relative error reduction**; bpc 2.3003 → 2.2745, a **1.12%
relative reduction**. Tokens per character barely move (0.30353 → 0.30357,
+0.01%), so the bpc gain is not a token-length artefact. The median and
Hodges–Lehmann estimates agree with the mean (F1: median +0.00968, HL +0.00964,
10% trimmed +0.00959), so the mean is not driven by the high seed 48.

For calibration only, an unpaired Welch test on the same ten seeds gives F1
t = 0.47 (p = 0.641) and bpc t = −0.98 (p = 0.343). The unpaired test is **not**
the right test for the paired design; it answers the different, between-corpus
question. But it shows that the numerator is small relative to corpus-to-corpus
variation: almost all of the paired t = 6.52 comes from the tiny paired
denominator, not from a large absolute gain.

The bpc arm is estimator-specific. W1-18 finds the sign survives every
smoothing, denominator and unseen-token variant, but the gain reverses on
in-distribution text (`+0.0096` bpc, 8/2) and shrinks by 86–92% under an
interpolated bigram code. It should be reported as “held-out unigram bpc at
v512”, not as a general compression gain.

One consistency caveat: seed 46 has `ΔF1 = +0.000145`, essentially one F1
quantum, and `Δbpc = −0.000997`. The “10/10” claim is therefore **9 clear wins
plus 1 near-tie**, not ten independent meaningful wins. A sign test on
`|ΔF1| > 0.001` gives 9/10, p = 0.0215; the exact sign test on signs alone still
gives 10/10, p = 0.00195.

## 5. Nonparametric recomputation

All tests are two-sided, n = 10. “Sign-flip” is the exact paired permutation
test over all 2¹⁰ sign assignments; “bootstrap” is the percentile bootstrap over
seeds (50,000 resamples).

| metric | comparison | t p | Wilcoxon exact p | sign test | exact sign-flip p | bootstrap 95% CI |
|---|---|---:|---:|---:|---:|---|
| boundary_f1 | morph − frequency | 0.000109 | **0.00195** | 10/10, p=0.00195 | **0.00195** | [+0.00651, +0.01175] |
| boundary_f1 | random_closed − frequency | 0.118 | 0.1309 | 3/10, p=0.344 | 0.1133 | [−0.01361, −0.00004] |
| boundary_f1 | morph − random_closed | 0.00415 | **0.00391** | 9/10, p=0.0215 | **0.00391** | [+0.00828, +0.02344] |
| bits_per_char | morph − frequency | 0.000442 | **0.00195** | 10/10 lower (0/10 positive), p=0.00195 | **0.00195** | [−0.03452, −0.01692] |
| bits_per_char | random_closed − frequency | 0.0141 | 0.0137 | 8/10, p=0.109 | 0.0117 | [+0.01488, +0.05869] |
| bits_per_char | morph − random_closed | 0.00108 | **0.00195** | 10/10 lower (0/10 positive), p=0.00195 | **0.00195** | [−0.08626, −0.03799] |

The primary morph contrasts survive the nonparametric recomputation. Note the
**p-value floor**: with n = 10, the smallest attainable two-sided exact
sign/Wilcoxon p is 2/2¹⁰ = 0.00195. The t-test’s smaller p (0.000109) relies on
normal tails; the rank test cannot go below 0.00195. This does not change the
decision, but the conservative number to report is 0.00195, not 0.0001.

The `random_closed − frequency` F1 comparison is null under every rank test. Its
percentile bootstrap CI happens to exclude zero, a reminder that the bootstrap
is unreliable with n = 10 and a skewed delta; the t, Wilcoxon and sign tests all
agree on no significant difference. The `random_closed − frequency` bpc
comparison is marginal: t/Wilcoxon agree near 0.014, but the sign test gives
p = 0.109. Under Bonferroni over six tests (α = 0.00833) it fails; under Holm it
passes. It is a fragile control difference, not part of the primary claim.

Multiplicity, explicitly:

| comparison | metric | raw p | Bonferroni α=0.00833 | Holm | Survives? |
|---|---|---:|---:|---:|---|
| morph − frequency | F1 | 0.000109 | pass | pass | yes |
| morph − frequency | bpc | 0.000442 | pass | pass | yes |
| morph − random_closed | F1 | 0.00415 | pass | pass | yes |
| morph − random_closed | bpc | 0.00108 | pass | pass | yes |
| random_closed − frequency | bpc | 0.0141 | fail | pass | fragile |
| random_closed − frequency | F1 | 0.118 | fail | fail | no |

The four primary morph tests pass the strict Bonferroni threshold over this
six-test family. Because the two metrics correlate −0.91, treating them as one
effect is more honest than treating them as two independent confirmations; the
conclusion is unchanged. W1-20 enumerates the larger 24-test family across
vocabularies and mode pairs: under that family the two `morph − frequency` tests
still survive (Bonferroni p = 0.0026 F1, 0.0106 bpc), as does
`morph − random_closed` bpc (0.0260), while `morph − random_closed` F1 does not
(0.0996). The primary claim is robust; the secondary F1 claim is provisional at
the full-family level.

## 6. Power

At n = 10, two-sided α = 0.05, the minimum detectable paired effect at 80% power
is **|d_z| ≈ 0.996**. The study can detect only large within-pair effects.

| d_z | n=10 | n=20 | n=30 | n=50 | n=100 |
|---:|---:|---:|---:|---:|---:|
| 0.2 | 0.09 | 0.14 | 0.19 | 0.28 | 0.51 |
| 0.3 | 0.14 | 0.25 | 0.36 | 0.55 | 0.84 |
| 0.5 | 0.29 | 0.56 | 0.75 | 0.93 | 1.00 |
| 0.8 | 0.62 | 0.92 | 0.99 | 1.00 | 1.00 |
| 1.0 | 0.80 | 0.99 | 1.00 | 1.00 | 1.00 |

The observed `d_z = 2.06` gives post-hoc power ≈ 1.0, but post-hoc power is a
monotone function of the observed p-value and is circular; it should not be
reported as validation. The useful statements are the MDE (≈ 1.0) and the CI
(`d_z` ∈ [1.24, 3.98]).

The design implication is more important. The synthetic pairing correlation is
0.997, which is why ten seeds suffice. If a real-corpus replication has the
same `d_av ≈ 0.21` but a lower pairing correlation:

| pairing r | implied d_z | power at n=10 | seeds for 80% |
|---:|---:|---:|---:|
| 0.997 | 2.74 | 1.00 | 4 |
| 0.99 | 1.50 | 0.99 | 6 |
| 0.95 | 0.67 | 0.47 | 20 |
| 0.90 | 0.47 | 0.27 | 37 |
| 0.80 | 0.34 | 0.16 | 72 |

(Tables assume equal SDs; the observed unequal-SD case is slightly less
optimistic, `d_z = 2.06` rather than 2.74 at r = 0.997.) Ten seeds are enough
only because the synthetic corpus makes the two tokenizers almost identical.
A real-corpus replication should plan for ~20–40 seeds, not 10, or use a more
efficient design.

Sample-size history: the three-seed v512 result was not significant (F1
t = 4.23, p = 0.052, CI includes zero; bpc t = −2.69, p = 0.115). The ten-seed
result becomes significant at k = 4 and stays significant (F1 p = 0.0097 at
k = 4; bpc p = 0.0296). The direction was already stable at k = 3, so this is a
power increase, not a sign reversal. It is still worth recording that the
published significance depends on the seed extension from 3 to 10.

## 7. Is the t statistic inflated by the synthetic corpus construction?

### 7.1 No code bug and no test leakage

The t is not numerically inflated. The paired function, SD, SE and CI are
correct, and the published table reproduces exactly. `collect_provenance` is
computed on `syn.train_text` / `syn.train_word_objects` only; the test words are
held out of the assignment procedure. There is no test-set leakage. The
`flat`/`frequency` identity is exact, so the comparison changes only the level
assignment of a fixed rule set.

### 7.2 Oracle supervision

`collect_provenance` uses the generator’s ground-truth `Word.boundaries` to
count intra- vs cross-morpheme occurrences of every merge on the training words.
`morph` then puts high-intra rules in level 1. This is **oracle morphological
provenance**, not an unsupervised tokenizer. The t-test compares an
oracle-informed assignment with an unsupervised frequency baseline. It is an
ablation/upper bound, not a deployable method. The claim must say “with oracle
morphological provenance”; otherwise it overstates what was demonstrated.

### 7.3 A clean, fixed generator

The synthetic corpus is eight prefixes × 120 roots × twelve suffixes, with the
word exactly `prefix + root + suffix`, no allomorphy, no spelling changes, no
noise, and boundaries at deterministic character positions. Every test
component is attested in training, and the test text is uniform over held-out
types while training is Zipfian. This is a best-case morphology: the boundary
task is exactly the concatenation seam, and the two tokenizers are almost
identical on the same corpus. The ten seeds vary the random roots, the
train/test split, the Zipf order and the token sample; they do not vary the
generator’s structural choices. The seed-to-seed SD therefore measures PRNG
variation within one generator, not generator-level uncertainty.

### 7.4 Hard construction windows

- **Vocabulary:** in the available 3-seed data at v1024 and v2048,
  `morph = frequency` exactly: ΔF1 = 0, Δbpc = 0, 3/3 ties. These are structural
  no-ops at the fixed 60% fraction: W1-18 shows the level-1 sets converge to
  rank order (Jaccard 1.0 at v2048, 0.98 at v1024), and W1-20 notes the paired
  v512−v1024 difference is only p = 0.052 (F1). They show the effect is not
  present in this design at the larger sizes, not that morphology cannot help
  there. They do show that the v512 result is not vocabulary-independent.
- **Root inventory:** a probe with the same pipeline but `n_roots=60`, all ten
  seeds, gives `ΔF1 = +0.000087` (2/10 wins, 8 exact ties; t = 1.50, p = 0.168;
  baseline F1 0.871, essentially the same as the 120-root baseline 0.874). With
  `n_roots=240` the effect is larger (`ΔF1 = +0.0117`, 5/5), but that corpus is
  much harder (baseline F1 0.702), so the 60-vs-120 contrast is the clean one:
  same difficulty, no effect at 60 roots. The probe is sensitive enough to see
  an effect of the published size: the 120-root seeds give `ΔF1 = +0.00924`
  (t = 6.52), while the 60-root seeds give `+0.000087` (t = 1.50, p = 0.168,
  8/10 exact ties).
- **Mechanism:** the fixed level-1 budget is 294 rules (0.6 × 490). At 60 roots,
  level 1 contains ~120 cross-morpheme rules; at 120 roots, ~30; at 240 roots,
  ~23. Within the ten published seeds, `ΔF1 = 0.00705 − 0.000265 × l1_cross`
  (`R² = 0.610`, p = 0.0077); `Δbpc = −0.04655 + 0.000864 × l1_cross`
  (`R² = 0.567`, p = 0.0119). The effect is largest when the fixed budget is
  nearly pure intra-morpheme. This is a dose-response in assignment purity, not
  a constant morphological advantage.
- **Budget fraction:** W1-20’s specification curve varies the 0.6 default. At
  `level1_fraction = 0.5` the F1 effect reverses (`−0.01558`, 4/6, p = 0.081);
  at 0.4 it is `−0.02495` (0/10, p = 0.0017); at 0.6 it is `+0.00924`;
  at 0.7 `+0.00139`; at 0.8 `+0.00004`. The 0.6 default is the smallest tested
  grid point above the pure-intra rule fraction (~252/490 = 0.514). This is the
  strongest evidence that the v512 result is a tuned-window effect, not a
  robust morphological one.

### 7.5 What “inflated” means here

The t is **not** inflated by a statistical bug. It is inflated relative to a
general claim because:

- the numerator uses oracle boundary labels and a clean concatenative corpus;
- the denominator omits generator-level variance (one generator, ten seeds);
- the evaluation distribution is uniform over held-out types, not natural text;
- the effect exists in one vocabulary/root-inventory/budget window.

The unpaired Welch p = 0.64 is the clearest calibration: the same mean
difference is not distinguishable from zero when between-corpus variation is in
the denominator. The paired test answers “does morph beat frequency on the same
corpus?”; the paper’s broader claim is closer to “does morph beat frequency on
new corpora?” The data answer the first, not the second.

## 8. Per-seed data and the L1 purity mechanism

| seed | L1 cross rules | baseline F1 | ΔF1 (pp) | Δbpc |
|---:|---:|---:|---:|---:|
| 42 | 43 | 0.8665 | +4.12 | −0.00545 |
| 43 | 38 | 0.8011 | +9.72 | −0.02414 |
| 44 | 15 | 0.9142 | +9.64 | −0.02989 |
| 45 | 26 | 0.8935 | +9.98 | −0.02368 |
| 46 | 60 | 0.8029 | +0.14 | −0.00100 |
| 47 | 24 | 0.8691 | +9.14 | −0.02481 |
| 48 | 21 | 0.8943 | +15.60 | −0.04944 |
| 49 | 28 | 0.9255 | +14.54 | −0.03447 |
| 50 | 25 | 0.8902 | +10.79 | −0.04448 |
| 51 | 23 | 0.8848 | +8.75 | −0.02053 |

Seed 46 is the near-tie; seed 48 is the high outlier. The LOO analysis shows
that neither drives the significance. The regression in §7.4 shows that the
cross-rule count, not baseline difficulty, is the dominant predictor of the
gain (in the two-predictor model, `l1_cross` p = 0.105, baseline p = 0.797;
with only ten points the individual coefficients are imprecise, but the
cross-rule term has the larger standardized effect).

## 9. What claim the data support

**Supported (conditional):**

- On this synthetic generator, at v512, 120 roots, 0.6 level-1 fraction, with
  oracle morphological provenance, `morph − frequency` raises boundary F1 by
  **+0.00924** (95% CI +0.00604 to +0.01245; exact Wilcoxon p = 0.00195; 10/10
  signs, 9 clear + 1 near-tie) and lowers held-out unigram bpc by **−0.02579**
  (95% CI −0.03662 to −0.01495; exact Wilcoxon p = 0.00195; 10/10). Both
  precision and recall improve; decode remains exact.
- `morph − random_closed` is also positive: F1 +0.01563 (p = 0.00391), bpc
  −0.06193 (p = 0.00195). This supports specificity relative to the average
  random assignment, with the caveat that `random_closed` is 1–2 rules short of
  294 in 3/10 seeds (the duplicate-dependent shortfall documented in W1-12) and
  is highly variable. W1-14’s 10-draw ensemble gives morph − random mean
  +0.0223 F1 (10/10) and −0.0887 bpc, so the published single draw is
  conservative for morph; the control is noisy, not favourable to the claim.
  Excluding the three size-mismatched seeds and the high seed 48 leaves n = 6,
  F1 mean +0.0127 (p = 0.070) and bpc mean −0.0505 (p = 0.048): the direction
  persists, but the single-draw F1 evidence weakens. Under W1-20’s 24-test
  family the single-draw F1 arm is provisional (p = 0.0996), while the bpc arm
  survives (p = 0.0260); W1-14’s 10-draw ensemble suggests the morph-vs-random
  F1 effect is larger than the single draw, so this is primarily control noise,
  not evidence against specificity.
- The two `morph − frequency` contrasts survive leave-one-seed-out,
  median/Hodges–Lehmann, cumulative-seed, exact nonparametric and W1-20’s full
  24-test Bonferroni/Holm checks.

**Not supported:**

- A general claim that morphological level assignment improves tokenization, or
  that it would generalize to real morphological corpora.
- An unsupervised-method claim: `morph` uses oracle training boundaries.
- A large practical effect: the raw gain is ~1% relative, `d_av ≈ 0.21` (F1),
  and the unpaired between-corpus comparison is p = 0.64.
- A general compression claim: the bpc gain is a held-out unigram effect; it
  reverses on in-distribution text and shrinks by 86–92% under a bigram code
  (W1-18).
- “Ten independent corpora”: the ten seeds are ten draws from one generator.
- A robust `random_closed − frequency` bpc difference: it is marginal
  (sign test p = 0.109, Bonferroni fail).
- A v1024/2048 effect: morph is an exact no-op there in the available 3-seed
  data (a structural no-op at the fixed 60% fraction, not a tested absence).
- “All ten seeds improve” without the seed-46 near-tie qualification.

## 10. Does the v512 result survive?

**As a statistical result:** yes. The paired t-test is correct, the deltas pass
the normality diagnostics, the exact Wilcoxon/sign-flip tests reject at
p = 0.00195, the bootstrap CIs exclude zero, LOO and cumulative analyses are
stable, and the two `morph − frequency` contrasts pass Bonferroni over both the
six-test family and W1-20’s 24-test family. The secondary
`morph − random_closed` bpc arm also survives the 24-test family; its F1 arm is
provisional (W1-20 p = 0.0996).

**As a general claim:** no. The effect is conditional on the generator, the
oracle provenance, the 294/196 budget and the v512 vocabulary. The morph
assignment is an exact no-op at v1024/2048 in the available 3-seed data, and the
F1 effect is absent at 60 roots; W1-20’s budget-fraction curve reverses it at
0.5 and 0.4. The t = 6.52 is a valid within-corpus statistic, not evidence of a
general morphological advantage.

Recommended wording for the paper:

> On a controlled synthetic generator at vocabulary 512, with oracle
> morphological provenance and a 294/196 level budget, assigning merges to two
> levels by morphological provenance improves macro boundary F1 by 0.9 points
> and held-out unigram bpc by 0.026 over flat BPE/frequency assignment (10 seeds;
> exact Wilcoxon p = 0.002). The morph assignment is a no-op at vocabulary
> 1024/2048 in the available 3-seed data and the F1 effect is absent at 60 roots;
> the bpc gain is estimator-specific. The result remains to be tested on a real
> morphological corpus.

Do not write “the branched tokenizer generalizes better” or “morphology improves
compression” without the conditional clause. The data support a controlled
ablation with an oracle upper bound, not a deployable tokenizer.

## 11. Commands and outputs

Reproduce the published statistics (writes only the requested output path):

```powershell
python -m experiments.analyze_levels `
  --csv results/level_assignment_v512_10seeds.csv `
  --out C:\Users\acese\AppData\Local\Temp\opencode\w1_16_repro_stats.csv
```

Output: the six lines quoted in §1, with the same means, SDs, t values, CIs and
win/loss/tie counts as `results/level_assignment_stats_v512.csv` to the printed
precision.

Nonparametric and effect-size audit (run from the repository root; the audit
script is outside the repo and is read-only with respect to it):

```powershell
python C:\Users\acese\AppData\Local\Temp\opencode\w1_16_stats.py
python C:\Users\acese\AppData\Local\Temp\opencode\w1_16_addenda.py
python C:\Users\acese\AppData\Local\Temp\opencode\w1_16_variance.py
python C:\Users\acese\AppData\Local\Temp\opencode\w1_16_power.py
```

The generator probe used the same `train_tokenizer` → `collect_provenance` →
`assign_levels` → `evaluate` pipeline with
`generate_synthetic(seed=s, n_roots=60|120|240, holdout=0.2)`. Provenance was
recomputed by encoding each unique training word once and weighting by its text
frequency; this is algebraically identical to the repo’s per-occurrence count
and was validated on seed 42: flat F1 = 0.866508, morph F1 = 0.870631,
ΔF1 = +0.004123, Δbpc = −0.005449, L1 = 294 with 250/43/1 intra/cross/unseen,
matching the published row exactly.

```powershell
python C:\Users\acese\AppData\Local\Temp\opencode\w1_16_fast_sensitivity.py `
  --n-roots 60 --holdout 0.2 --seeds 42,43,44,45,46,47,48,49,50,51
python C:\Users\acese\AppData\Local\Temp\opencode\w1_16_fast_sensitivity.py `
  --n-roots 240 --holdout 0.2 --seeds 42,43,44,45,46
```

Probe outputs:

```text
n_roots=60:  mean ΔF1=+0.000087  sd=0.000183  t=+1.500  p=0.168  W/L/T=2/0/8
             mean Δbpc=-0.000090  sd=0.000210  t=-1.361  p=0.206  W/L/T=0/2/8
             baseline F1 mean=0.871279  mean L1 cross=119.8
n_roots=120: published 10 seeds: mean ΔF1=+0.009243  sd=0.004484  t=+6.518  p=0.000109
             mean Δbpc=-0.025788  sd=0.015148  t=-5.384  p=0.000442
             baseline F1 mean=0.8742  mean L1 cross=30.3
n_roots=240: mean ΔF1=+0.011712  sd=0.004478  t=+5.848  p=0.00427  W/L=5/0
             mean Δbpc=-0.017129  sd=0.009882  t=-3.876  p=0.0179  W/L=0/5
             baseline F1 mean=0.702307  mean L1 cross=22.6
```

The 60-root probe uses all ten published seeds; the 120-root column is the
published ten; the 240-root probe uses five. The 60-root null is not a power
artefact: eight seeds are exact ties and the two non-ties differ by less than
0.0005 F1, with a tiny SD.

### Minimal reproduction snippet

The key nonparametric table can be reproduced from the repository root with:

```python
import csv
import numpy as np
from scipy import stats

rows = list(csv.DictReader(open("results/level_assignment_v512_10seeds.csv")))
idx = {(int(r["seed"]), r["mode"]): r for r in rows}
seeds = sorted({int(r["seed"]) for r in rows})

for metric in ("boundary_f1", "bits_per_char"):
    for a, b in (("morph", "frequency"),
                 ("random_closed", "frequency"),
                 ("morph", "random_closed")):
        d = np.array([float(idx[(s, a)][metric]) -
                      float(idx[(s, b)][metric]) for s in seeds])
        t, p = stats.ttest_1samp(d, 0)
        w, pw = stats.wilcoxon(d, method="exact")
        k = int((d > 0).sum())
        ps = stats.binomtest(k, len(d), 0.5).pvalue
        print(f"{metric:14s} {a}-{b}: mean={d.mean():+.6f} "
              f"t={t:+.3f} p_t={p:.3g} p_wilcoxon={pw:.5f} "
              f"sign={k}/{len(d)} p_sign={ps:.5f}")
```

For the exact paired sign-flip test, add:

```python
import itertools
p_perm = np.mean([abs(np.mean(d * np.array(sgn))) >= abs(d.mean()) - 1e-15
                  for sgn in itertools.product([-1, 1], repeat=len(d))])
```

For the effect sizes, `d_z = d.mean()/d.std(ddof=1)`; `d_av` uses the average of
the two condition variances in the denominator, as in §4.

## 12. Limitations and the check that would change the verdict

- The generator probe is exploratory: the 60-root setting uses all ten seeds,
  the 240-root setting uses five; one generator family, and a reimplemented
  (validated) provenance pass. It shows non-robustness; it does not estimate a
  precise generator-level effect.
- n = 10 limits the power of the normality diagnostics and of the exact tests
  (the nonparametric p is at its floor). The t-test's p = 0.0001 is not
  reproducible nonparametrically; report 0.00195.
- The real-corpus question is untested. The synthetic task has no allomorphy,
  no noise, no ambiguous boundaries, uniform test types and oracle training
  labels.
- No pre-registration is available for the seed extension or the mode family;
  the p-values should be read with that in mind, even though W1-20 finds the
  primary result survives the full-family correction.

**Confidence.** High for the reproduction, the correctness of the paired t-test,
the nonparametric recomputation, and the effect-size arithmetic. Moderate for
the generator-sensitivity probe (one generator family; five seeds at 240 roots;
a reimplemented but validated provenance pass). High that the data do not
support a general morphological claim.

**Single check that would most change the verdict:** repeat the same 10-seed
`morph − frequency` comparison on a real morphological corpus with independent,
non-oracle boundary annotations, holding the 294/196 budget and the same
evaluation protocol. A positive, nonparametric-significant effect there (with a
between-corpus `d_av` around 0.2 and a pairing correlation below 0.95) would
overturn the external-validity verdict and support the general claim. A null
there would confirm that the v512 result is a synthetic-window artefact.
