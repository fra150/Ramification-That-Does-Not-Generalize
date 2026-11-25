# W1-14 — the `random_closed` control at v512: 10 topological orders per seed

Red-team note. No source files were modified. The published v512 CSV is
reproduced exactly (F1 difference 0, bpc difference < 2e-14), and the fast
evaluator used for the extra draws was validated against
`experiments.level_assignment.evaluate` to 5.3e-15.

**Scope.** 10 synthetic-corpus seeds (42–51), vocabulary 512, the same flat BPE
as `docs/findings-v0.2.md`. For each seed: `flat`, `morph`, the published
canonical `random_closed` draw, and **10 fresh independent random topological
orders**. The question is the one the v0.2 report could not answer with one
draw per seed: how wide is the `random_closed` distribution, is `morph` outside
it, and is the morph effect larger than that spread?

**Artifacts.**

| file | what |
|---|---|
| `results/redteam/w1_14_run_random_closed.py` | runner: 10 draws + canonical + flat/morph |
| `results/redteam/w1_14_random_closed_draws.csv` | long CSV, 130 rows (10 seeds × 13 modes/draws) |
| `results/redteam/w1_14_run.log` | full stdout |
| `results/redteam/w1_14_analyze.py` | per-seed and pooled analysis |
| `results/redteam/w1_14_analysis.txt` | full analysis stdout |
| `results/redteam/w1_14_per_seed.csv` | one row per seed, all summaries |
| `results/redteam/w1_14_corrected_sampler.py` | deduplicated random sampler (control-bug sensitivity) |
| `results/redteam/w1_14_corrected_sampler.csv` | corrected-sampler runs |
| `results/redteam/w1_14_corrected.log` | corrected-sampler stdout |
| `results/redteam/w1_14_size_corrected_pooled.py` | mixed size-294 / corrected pooled estimate |
| `results/redteam/w1_14_size_corrected_pooled.csv` | per-seed size-corrected deltas |

## TL;DR

1. **The v512 result survives the 10-order control.** Against the 10-draw mean,
   morph F1 is **+0.0223** (t = 7.23, 95% CI [+0.0153, +0.0293], 10/10 seeds)
   and bpc is **−0.0887** (t = −8.13, CI [−0.1133, −0.0640], 10/10 seeds). The
   published single-draw comparison gave +0.0156 / −0.0619, so the
   morph-vs-random gap is larger than published. The flat-relative headline is
   unchanged: +0.0092 F1 / −0.0258 bpc.
2. **Morph is outside the random distribution on most seeds.** It is better
   than all 10 draws on F1 in 7/10 seeds (exact binomial p = 4.8e-6 under
   exchangeability) and on bpc in 9/10 seeds (p = 3.9e-9). Across all 100
   draws, only **6 beat morph on F1** and **2 on bpc**; morph's mean rank is
   1.60/11 and 1.20/11, and its mean per-seed percentile is the 94th and 98th.
3. **But it is a shift, not a separation.** The within-seed random SD is 0.0130
   F1 / 0.0492 bpc. The morph−random gap is 1.72 / 1.80 of that SD. Individual
   random orders can match morph (seed 46: rank 5/11 F1 and 3/11 bpc), so
   "outside the distribution" is true on average and on most seeds, not on
   every seed.
4. **The random control is not neutral.** `random_closed` is worse than flat on
   10/10 seeds: ΔF1(random − flat) −0.0061 to −0.0256, Δbpc +0.0172 to +0.1065.
   Morph is at or better than flat on 10/10. Relative to flat, the v512 effect
   is only **+0.0092 F1 / −0.0258 bpc**, i.e. **0.71 / 0.52 of the within-seed
   random SD**. The 1.7–1.8 SD figure is the morph-vs-random gap, which includes
   the control's degradation relative to flat; it is not the flat-relative
   effect.
5. **The published single draw was a favourable draw.** Its mean percentile
   within the 10 draws is the 70th (F1) and 68th (bpc); the published
   morph−canonical gap is 0.0067 F1 and 0.0267 bpc smaller in magnitude than
   morph−10-draw-mean. The published effect is conservative on these seeds,
   not inflated.
6. **The control has a size bug, but it does not drive the result.** 37/100
   draws have 290–293 unique level-1 rules instead of 294 (three seeds have no
   294-rule draw at all). Restricting to the 63 size-294 draws gives +0.0215
   F1 / −0.0829 bpc. A deduplicated sampler on the three seeds with no 294-rule
   draw (44, 45, 50), validated on seed 42, gives a size-corrected pooled
   estimate of **+0.0220 F1 / −0.0845 bpc**, 10/10 seeds (buggy size-294 draws
   where available, corrected draws for 44/45/50). Per-seed effects move in
   both directions (seed 44 shrinks, seed 50 grows) and cancel. The bug is real
   and should be fixed, but it is not the source of the v512 effect.

**Verdict.** The v512 morph-vs-`random_closed` result survives the 10-order
control, with a larger morph-vs-random point estimate than the published one.
The stronger claim "morph is outside the random support" holds for bpc almost
everywhere and for F1 in 7/10 seeds, but not universally. The `random_closed`
control should still be fixed to be exactly size-matched.

## 1. Design

For every seed, one flat BPE at v512 is trained, provenance is collected once,
and the *same* flat tokenizer is reassigned:

- `flat`: the one-level reference;
- `morph`: the provenance assignment (deterministic);
- `random_closed_d00..d09`: ten independent topological orders from
  `ramified.levels.assign_levels(mode="random_closed")`, RNG seeds
  `seed*100_000 + 1_729 + 7_919*k`;
- `random_closed_canonical`: the published draw, RNG seed
  `seed*1000 + vocab`, used only to reproduce the paper row and to measure how
  representative it was.

All 10 draws per seed produce 10 distinct level-1 sets
(`w1_14_random_closed_draws.csv`, `level1_fingerprint`). Mean Jaccard to the
morph level-1 set is 0.622; to the frequency prefix, 0.525.

Commands:

```powershell
$env:PYTHONPATH="."
python -m results.redteam.w1_14_run_random_closed `
  --seeds 42,43,44,45,46,47,48,49,50,51 --draws 10 `
  --out results/redteam/w1_14_random_closed_draws.csv

python -m results.redteam.w1_14_analyze `
  --csv results/redteam/w1_14_random_closed_draws.csv `
  --published results/level_assignment_v512_10seeds.csv `
  --out-prefix results/redteam/w1_14
```

Reproduction check from the analysis:

```text
flat:    max |F1 diff| = 0.00e+00   max |bpc diff| = 1.15e-14
morph:   max |F1 diff| = 0.00e+00   max |bpc diff| = 5.33e-15
canonical: max |F1 diff| = 0.00e+00 max |bpc diff| = 1.82e-14
```

The cached-chunk evaluator was checked against the original `evaluate` with
`--validate-original` on seed 42; the largest difference was 5.33e-15
(`results/redteam/w1_14_pilot_fast.csv`).

## 2. Per-seed distribution

`dF1 = morph − mean(10 draws)` (positive favours morph);
`dBPC = morph − mean(10 draws)` (negative favours morph).
`rank` is morph's rank among morph plus the 10 fresh draws (1 = best); `beyond`
marks morph better than every fresh draw. The published canonical draw is not
part of this rank; it is analysed separately in §5.

F1 distribution:

| seed | morph F1 | random F1 mean ± SD | random range | dF1 | rank | beyond |
|---:|---:|---:|---|---:|---:|---:|
| 42 | 0.8706 | 0.8583 ± 0.0119 | [0.8329, 0.8713] | +0.0123 | 2 | no |
| 43 | 0.8108 | 0.7852 ± 0.0119 | [0.7634, 0.8020] | +0.0256 | 1 | yes |
| 44 | 0.9238 | 0.8886 ± 0.0207 | [0.8567, 0.9114] | +0.0352 | 1 | yes |
| 45 | 0.9035 | 0.8863 ± 0.0121 | [0.8713, 0.9108] | +0.0172 | 2 | no |
| 46 | 0.8030 | 0.7956 ± 0.0100 | [0.7766, 0.8066] | +0.0074 | 5 | no |
| 47 | 0.8782 | 0.8444 ± 0.0180 | [0.8127, 0.8628] | +0.0338 | 1 | yes |
| 48 | 0.9099 | 0.8761 ± 0.0099 | [0.8543, 0.8901] | +0.0339 | 1 | yes |
| 49 | 0.9401 | 0.9162 ± 0.0068 | [0.9045, 0.9289] | +0.0239 | 1 | yes |
| 50 | 0.9010 | 0.8842 ± 0.0124 | [0.8619, 0.8965] | +0.0169 | 1 | yes |
| 51 | 0.8935 | 0.8768 ± 0.0101 | [0.8572, 0.8896] | +0.0167 | 1 | yes |

bpc distribution (lower is better):

| seed | morph bpc | random bpc mean ± SD | random range | dBPC | rank |
|---:|---:|---:|---|---:|---:|
| 42 | 2.3484 | 2.3919 ± 0.0422 | [2.3521, 2.4914] | −0.0435 | 1 |
| 43 | 2.3678 | 2.4926 ± 0.0496 | [2.4259, 2.5745] | −0.1247 | 1 |
| 44 | 2.2104 | 2.3467 ± 0.0651 | [2.2413, 2.4277] | −0.1364 | 1 |
| 45 | 2.2317 | 2.3038 ± 0.0367 | [2.2467, 2.3602] | −0.0721 | 1 |
| 46 | 2.3661 | 2.4167 ± 0.0785 | [2.3479, 2.6181] | −0.0506 | 3 |
| 47 | 2.2481 | 2.3738 ± 0.0568 | [2.3003, 2.4833] | −0.1257 | 1 |
| 48 | 2.2333 | 2.3519 ± 0.0363 | [2.3041, 2.4075] | −0.1187 | 1 |
| 49 | 2.2068 | 2.2894 ± 0.0315 | [2.2377, 2.3475] | −0.0826 | 1 |
| 50 | 2.2427 | 2.3044 ± 0.0370 | [2.2621, 2.3760] | −0.0617 | 1 |
| 51 | 2.2902 | 2.3609 ± 0.0361 | [2.3084, 2.4181] | −0.0708 | 1 |

Pooled (paired over seeds, 9 df):

| metric | morph − random mean | t | 95% CI | W/L/T | draws better than morph | mean rank |
|---|---:|---:|---|---:|---:|---:|
| boundary F1 | **+0.02228** | +7.23 | [+0.01531, +0.02925] | 10/0/0 | 6/100 | 1.60/11 |
| bits per char | **−0.08867** | −8.13 | [−0.11334, −0.06401] | 10/0/0 | 2/100 | 1.20/11 |

The median-draw sensitivity agrees (F1 +0.02239, t = 6.97; bpc −0.08732,
t = −6.39). The six F1 draws that beat morph are one in seed 42 (0.8713 vs
0.8706), one in seed 45 (0.9108 vs 0.9035), and four in seed 46. The two bpc
draws that beat morph are both in seed 46 (2.3479 and 2.3586 vs 2.3661).

## 3. Is the effect larger than the random spread?

| quantity | F1 | bpc |
|---|---:|---:|
| within-seed SD, pooled | 0.01298 | 0.04918 |
| between-seed SD of random means | 0.04184 | 0.06090 |
| total SD, all 100 draws | 0.04177 | 0.07463 |
| abs(mean delta) / within-seed SD | **1.72** | **1.80** |
| abs(mean delta) / total SD | 0.53 | 1.19 |
| mean standardized shift, (morph − mean)/SD | +1.89 (t = 6.42) | +2.00 (t = 8.31) |
| per-seed z range | [+0.74, +3.54] | [+0.64, +3.27] |
| mean per-seed percentile of morph | 94th | 98th |
| margin beyond best draw / range | +0.17 | +0.22 |

The assignment-to-assignment noise is large: one random order moves F1 by
~0.013 and bpc by ~0.049, comparable to the published morph−frequency effect
(+0.0092 F1, −0.0258 bpc). Against that noise the morph effect is ~1.7–1.8 SD,
not a 5–10 SD separation. The distribution is also skewed (seed 46 has a
2.618 bpc draw), which is why the median-draw check is reported alongside the
mean.

So the answer to the literal question is: **the mean effect is larger than one
random SD, but morph is not outside the random support on every seed.** The
honest description is a consistent upper-tail shift: 7/10 seeds beyond the F1
maximum, 9/10 beyond the bpc minimum, 94/100 and 98/100 individual draws worse,
but 6/100 F1 draws (and 2/100 bpc draws) at least match it.

### The random control is not neutral

`random_closed` is worse than the flat tokenizer on every seed: ΔF1(random −
flat) ranges from −0.0061 to −0.0256, Δbpc from +0.0172 to +0.1065. Morph is at
or better than flat on every seed: ΔF1 +0.0001 to +0.0156, Δbpc −0.0010 to
−0.0494. The morph−random gap therefore combines the morph gain over flat with
the random control's degradation.

The published headline is morph vs flat/frequency: **+0.00924 F1 / −0.02579
bpc**. Relative to the within-seed random SDs, that is only **0.71 F1 / 0.52
bpc**. The answer therefore depends on the comparison:

| comparison | F1 | bpc |
|---|---:|---:|
| morph − random_closed mean, in random SD | 1.72 | 1.80 |
| morph − flat, in random SD | **0.71** | **0.52** |

The flat-relative effect is still statistically consistent on these seeds
(published paired t = 6.52 F1, −5.38 bpc); it is simply smaller than the
assignment-to-assignment noise. The larger numbers describe how bad an
arbitrary closed reordering is; they are not the effect of morphology over the
flat tokenizer. The v512 claim should keep morph−frequency/flat (+0.0092 F1 /
−0.0258 bpc) as the headline and use random_closed only as a control showing
that arbitrary reorderings do not reproduce it.

## 4. The `random_closed` control is not exactly size-matched

`ramified/levels.py:142-146` appends a child to `available` once per entry in
`dependents[dependency.new]`. When a rule depends twice on the same creator
(a self-pair `(X, X)` with `X` nonterminal), the child is appended twice and
the duplicate occupies two slots of the topological prefix. The prefix then
has fewer than `n1` unique rules. The published CSV already showed `n_l1` =
292–294 for `random_closed`; this run makes the size explicit:

```text
seed 42  {294: 10}
seed 43  {294: 10}
seed 44  {293: 10}
seed 45  {292: 2, 293: 8}
seed 46  {293: 7, 294: 3}
seed 47  {294: 10}
seed 48  {294: 10}
seed 49  {294: 10}
seed 50  {290: 1, 291: 2, 292: 3, 293: 4}
seed 51  {294: 10}
total    {290: 1, 291: 2, 292: 5, 293: 29, 294: 63}
```

37/100 draws are not size-matched to morph's 294. This is a bug relative to
the docstring's "closed level 1 of exactly n1 rules", not merely report
rounding.

Two checks say the effect is not a size artifact:

1. **Restrict to the 63 size-294 draws** (7 seeds; seeds 44, 45, 50 have none):
   F1 +0.02151 (t = 5.18, CI [+0.01134, +0.03168]); bpc −0.08294 (t = −5.07,
   CI [−0.12295, −0.04293]). Nearly identical to the all-draw result.
2. **Within-seed size regression**: each extra level-1 rule is worth only
   +0.00174 F1 (R² = 0.016) and −0.0158 bpc (R² = 0.103). At morph's 294
   rules, morph still exceeds the size prediction by +0.0214 F1 (t = 6.70) and
   −0.0810 bpc (t = −6.61).

A corrected sampler (deduplicated dependents, exactly 294 unique rules) was run
on the three seeds with no size-294 draws plus seed 42 as a control. See §6.

## 5. Why the published effect is smaller

The canonical published draw is not a typical draw: its mean percentile inside
the 10 fresh draws is the 70th for F1 and 68th for bpc, with mean z +0.64
(t = 3.39) and −0.63 (t = −2.55). The published deltas and the 10-draw deltas:

| metric | published (morph − canonical) | morph − 10-draw mean | difference (published − 10-draw) |
|---|---:|---:|---:|
| F1 | +0.01563 | +0.02228 | −0.00665 (t = −3.11) |
| bpc | −0.06193 | −0.08867 | +0.02674 (t = +2.49) |

The canonical draw is not more morph-like: its `l1_intra` z is +0.01 and its
Jaccard-to-morph z is −0.01. It is simply a favourable random assignment (best
of 10 on F1 in 1/10 seeds, best on bpc in 3/10, never the worst). On these
seeds the single-draw design was conservative; a 10-draw control is the more
honest and more powerful design.

## 6. Corrected-sampler sensitivity

The deduplicated sampler (`w1_14_corrected_sampler.py`) always emits exactly
294 unique rules. On seed 42, which has no duplicate dependency edge, it
reproduces the buggy-sampler mean within Monte Carlo (+0.0108 F1 / −0.0402 bpc
vs +0.0123 / −0.0435), validating the reimplementation. On the three seeds with
no size-294 draw:

| seed | buggy dF1 | corrected dF1 | buggy dBPC | corrected dBPC |
|---:|---:|---:|---:|---:|
| 44 | +0.0352 | **+0.0238** | −0.1364 | **−0.0797** |
| 45 | +0.0172 | **+0.0185** | −0.0721 | **−0.0757** |
| 50 | +0.0169 | **+0.0268** | −0.0617 | **−0.1086** |

The bug is not a one-way inflation: seed 44 shrinks, seed 50 grows, seed 45 is
unchanged. Which rules the duplicate displaced matters, not only the count.
Using buggy size-294 draws where available and corrected 294-rule draws for
seeds 44/45/50 gives:

```text
seed source            n  deltaF1   deltaBPC
42   buggy-294       10  +0.01230  -0.04349
43   buggy-294       10  +0.02557  -0.12474
44   corrected-294    6  +0.02380  -0.07966
45   corrected-294    6  +0.01852  -0.07565
46   buggy-294        3  +0.00445  -0.01461
47   buggy-294       10  +0.03380  -0.12570
48   buggy-294       10  +0.03385  -0.11867
49   buggy-294       10  +0.02392  -0.08262
50   corrected-294    6  +0.02677  -0.10861
51   buggy-294       10  +0.01667  -0.07075
pooled delta_f1:  mean=+0.02196  t=+7.53  CI95=[+0.01547,+0.02846] W/L=10/0
pooled delta_bpc: mean=-0.08445  t=-7.34  CI95=[-0.11010,-0.05880] W/L=10/0
```

The size-corrected estimate (+0.0220 F1 / −0.0845 bpc) is statistically
indistinguishable from the all-draw result (+0.0223 / −0.0887). The control
bug is real and should be fixed, but it does not drive the v512 result.

```powershell
python -m results.redteam.w1_14_corrected_sampler `
  --seeds 42,44,45,50 --draws 6 `
  --out results/redteam/w1_14_corrected_sampler.csv
python results/redteam/w1_14_size_corrected_pooled.py
```

## 7. Mechanism check: intra purity

The morph level 1 has 233–278 intra rules (mean 262.7); the random draws span
180–236 and never reach morph in any seed (10/10). A random closed prefix of
size 294 apparently cannot collect all the intra rules; morph's assignment can.

Within the random draws, purity predicts only a small part of the metric:

| metric | slope per +1 intra L1 rule | R² | predicted morph gain from 210→263 | morph residual |
|---|---:|---:|---:|---:|
| F1 | +0.000073 | 0.147 | +0.0039 | **+0.0186 (t = 6.17)** |
| bpc | −0.001296 | 0.241 | −0.069 | **−0.0232 (t = −2.30)** |

For F1 the gain is mostly *not* explained by how many intra rules are in level
1; for bpc purity explains most of the raw gap but a negative residual remains.
This is consistent with the v0.2 claim that the specific provenance assignment
matters, not just "put more frequent/intra rules early". It is also a warning:
the random control almost never samples morph's purity, so the comparison is
between morph and a distribution that is far away in level-1 composition.

The slopes are fit on draws with 180–236 intra rules; morph is at 233–278,
partly outside the fitted range. The residuals are therefore extrapolations,
not a causal decomposition. They motivate the purity-matched control below
rather than proving the mechanism.

## 8. Caveats

- **The sampler is not uniform over topological orders or closed sets.** The
  result is relative to this specific control. A uniformly sampled closed set
  of size 294 (or a random-priority linear extension) could have a different
  mean. The corrected sampler fixes the duplicate-size bug, not the
  non-uniformity.
- **Same corpus, same seeds.** This is a stronger control on the published
  data, not an independent replication. The synthetic corpus remains the
  external-validity limit; `docs/findings-v0.2.md` §4.1 is still open.
- **bpc is model-relative.** W1-18 shows the held-out unigram bpc is ~83%
  collinear with F1 and gives no gain under a fixed 9-bit/token code. Read the
  bpc numbers as "held-out unigram likelihood", not "better compression".
- **Ten draws per seed** gives limited tail resolution. The beyond-max counts
  and the pooled t are strong, but a 100-draw control would estimate the 95th
  percentile directly.
- The `random_closed` size bug should be fixed in source before the paper, even
  though it does not drive this result.

## Verdict

**The v512 result survives the 10-order random control, and the morph-vs-random
gap is larger than the published single-draw estimate.** Morph is a consistent
upper-tail assignment: +0.0223 F1 and −0.0887 bpc versus the random mean,
10/10 seeds, ~1.7–1.8 within-seed random SD, better than 94/100 and 98/100
individual draws, beyond the per-seed maximum in 7/10 and 9/10 seeds. It is not
universally outside the random support: seed 46 has four F1 draws and two bpc
draws at least as good as morph, and individual draws can match it elsewhere.
The result is a distribution shift with a robust mean, not a clean separation.
After fixing the sampler's size bug the pooled estimate is +0.0220 F1 /
−0.0845 bpc, still 10/10; individual seeds move by up to 0.011 F1 / 0.057 bpc,
so the correction matters for per-seed claims but not for the v512 conclusion.
Against the flat tokenizer—the paper's headline comparison—the effect remains
+0.0092 F1 / −0.0258 bpc, only 0.71 / 0.52 of the random SD. The large
morph−random gap is real but partly reflects how much an arbitrary closed
reordering costs; it should not replace the flat-relative effect in the paper.

The single check that would most change this verdict is a **purity-matched
random control**: same 294-rule size and the same number of intra rules as
morph, with the intra rules chosen at random rather than by provenance. If
morph still beats that control, the effect is specifically morphological
selection/ordering; if not, the v512 effect is "more intra rules in level 1"
and the provenance story needs weakening. The corrected-sampler run rules out
the simpler size bug but does not answer this question.
