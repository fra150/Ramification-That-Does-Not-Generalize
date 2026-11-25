# W1-13 — `level1_fraction` sweep at v512: a coverage window, not a robust effect

**Verdict: the v512 morph result survives at 0.6, but not as a fraction-robust
result.** The published 0.6 run reproduces exactly (max |Δ| = 0 on `n_l1`,
boundary F1, bpc and tok/ch), and it replicates on ten fresh seeds
(+0.0048 F1, 10/10; −0.0104 bpc, 10/10). But the morph advantage is confined to
a window of roughly **0.55–0.65**. At 0.4–0.52 morph is *worse* than the
frequency control (bpc significantly worse at every point; F1 significantly
worse at 0.4), and at 0.7–0.8 the advantage is essentially zero. The effect is
not robust across `level1_fraction` 0.4–0.8.

**0.6 is not a post-hoc maximum.** It is a pre-existing default inherited from
the old `hier2` budget weights `[0.6, 0.4]`, and it is not the best point:
0.55 beats 0.6 by +0.0094 F1 on the published seeds (t = 4.65, 10/10) and by
+0.0083 F1 on ten fresh seeds (t = 4.80, 10/10). The published effect is
therefore conservative, not cherry-picked upward. Selecting 0.55 as “the
optimal fraction” *would* be a post-hoc choice; it survives an out-of-sample
check on seeds 52–61, so it is a validated tuning result rather than an
in-sample artifact, but it still needs a real-corpus test.

**Confidence:** high (~0.9) that the effect is fraction-dependent on this
generator and that 0.55 > 0.6; high (~0.95) that 0.6 is a pre-existing default,
not a tuned maximum; medium (~0.6) that the 0.55–0.65 window transfers to a real
morphological corpus; low (~0.25) that a *fixed* fraction is the right way to
state the result at all.

All new scripts and data are under `results/redteam/`; this report is under
`docs/redteam/`. No source file was modified by this audit.

---

## 1. Question and design

`docs/findings-v0.2.md` reports the v512 level-assignment ablation at a single,
undisclosed fraction: the commands in §6 omit `--fraction`, so the CLI default
`0.6` is used (`experiments/level_assignment.py:80`; also
`ramified/levels.py:102`). The paper sentence claims a v512 effect without a
sensitivity analysis. Two red-team questions:

1. **Is the morph advantage specific to 0.6, or robust across fractions?**
   Sweep `level1_fraction ∈ {0.4, 0.5, 0.6, 0.7, 0.8}` at v512 on 10 seeds,
   with `frequency`, `morph` and `random_closed`.
2. **Is the “optimal” fraction a post-hoc choice?** Establish where 0.6 came
   from, whether it is the in-sample maximum, and whether the maximum (if
   different) validates on fresh seeds.

After the coarse sweep showed a sharp transition between 0.5 and 0.6, I added a
fine sweep `{0.52, 0.55, 0.58, 0.62, 0.65}` and two fresh-seed blocks
(52–61 at 0.6, 52–61 at 0.55, plus 52–61 at 0.5 as a below-threshold control).
The fine grid was chosen after seeing the coarse results, so it is an adaptive
design; the fresh seeds are the clean test.

Design is otherwise identical to the published ablation:

- synthetic corpus from `generate_synthetic(seed)`, 120k training tokens,
  20% held-out words, ground-truth prefix/root/suffix boundaries;
- one flat BPE per seed at `vocab=512` (490 merges), `split="chunks"`;
- the same flat rules and vocabulary for every mode; only the level assignment
  changes;
- `frequency` = rank prefix of size `int(fraction * 490)`; this is exactly the
  flat tokenization for every fraction (verified: zero within-seed spread);
- `morph` = top rules by intra-morpheme ratio, with dependency closure;
- `random_closed` = prefix of a random topological order of the same requested
  size (published control; see caveat in §8);
- primary metric: macro boundary F1 on held-out words; secondary: add-α 0.1
  unigram bits/char. Both are paired by seed.

Source state at run time (hashes in `results/redteam/provenance_hashes.txt`):

| file | mtime | SHA-256 (first 16) |
|---|---|---|
| `experiments/level_assignment.py` | 2026-11-22 22:23:04 | `167C55CC81EE3736` |
| `ramified/levels.py` | 2026-11-22 22:07:03 | `14420E923EE6C403` |
| `ramified/bpe.py` | 2026-11-22 22:02:02 | `73093A3E62FC66A5` |
| `ramified/synthetic.py` | — | `E7D4E5FBB01E86C4` |
| `ramified/metrics.py` | — | `14DBF6B1794ABFD9` |

`level_assignment.py` gained macro/micro columns from another session before
this run; `boundary_f1` is a backward-compatible alias for
`boundary_macro_f1`. Exact reproduction against the published CSV (§3) shows
this did not change behaviour.

## 2. Commands

Coarse sweep, five processes launched in parallel (same shape for each value):

```powershell
python -m experiments.level_assignment `
  --seeds 42,43,44,45,46,47,48,49,50,51 --vocabs 512 `
  --modes frequency,morph,random_closed --fraction 0.4 `
  --out results/redteam/level_frac_040_seeds42-51.csv
# repeated for 0.5, 0.6, 0.7, 0.8
# (logs: results/redteam/log_frac_0XX_seeds42-51.txt)
```

Fine sweep around the transition:

```powershell
# repeated for 0.52, 0.55, 0.58, 0.62, 0.65
python -m experiments.level_assignment `
  --seeds 42,43,44,45,46,47,48,49,50,51 --vocabs 512 `
  --modes frequency,morph,random_closed --fraction 0.55 `
  --out results/redteam/level_fine_055_seeds42-51.csv
```

Fresh seeds:

```powershell
python -m experiments.level_assignment `
  --seeds 52,53,54,55,56,57,58,59,60,61 --vocabs 512 `
  --modes frequency,morph,random_closed --fraction 0.6 `
  --out results/redteam/level_frac_060_seeds52-61.csv

# 0.55 was split into two foreground runs because the background job was
# CPU-throttled; the two files cover disjoint seeds.
python -m experiments.level_assignment `
  --seeds 52,53,54,55,56 --vocabs 512 `
  --modes frequency,morph,random_closed --fraction 0.55 `
  --out results/redteam/level_fine_055_seeds52-56.csv
python -m experiments.level_assignment `
  --seeds 57,58,59,60,61 --vocabs 512 `
  --modes frequency,morph,random_closed --fraction 0.55 `
  --out results/redteam/level_fine_055_seeds57-61.csv

python -m experiments.level_assignment `
  --seeds 52,53,54,55,56,57,58,59,60,61 --vocabs 512 `
  --modes frequency,morph,random_closed --fraction 0.5 `
  --out results/redteam/level_frac_050_seeds52-61.csv
```

Analysis:

```powershell
python results/redteam/analyze_fraction.py `
  --glob "results/redteam/level_f*_seeds42-51.csv" `
  --reference results/level_assignment_v512_10seeds.csv `
  --out results/redteam/fraction_summary_all_seeds42-51.txt

python results/redteam/analyze_fraction.py `
  --glob "results/redteam/level_*_seeds5*.csv" --reference none `
  --out results/redteam/fraction_summary_fresh52-61.txt
```

All intervals are paired 95% t intervals over seeds (df = n−1). `W/L/T` counts
per-seed signs; for bpc, a negative Δ is better. No multiplicity correction is
applied across the grid; see §8.

## 3. Sanity checks and exact reproduction

- **Frequency is invariant to the requested fraction.** For every seed, the
  `frequency` mode at 0.4–0.8 has identical F1, bpc and tok/ch (max within-seed
  spread = 0.0e+00). This is expected: a rank prefix of the flat merge order,
  applied before the remainder, is the flat tokenization. It makes `frequency`
  a fixed control and validates the harness.
- **Flat = frequency.** All rows match exactly.
- **Decode** is 1.0 for `decode_word` and `decode_doc` in every run and mode.
- **Published 0.6 reproduces exactly** on seeds 42–51: max |new − published| =
  0 for `n_l1`, `boundary_f1`, `bits_per_char`, `tokens_per_char`, macro
  precision/recall, `l1_intra`, `l1_cross`, `l2_intra`.

The 0.6 rows in the coarse sweep reproduce the published CSV exactly, and the
fresh-seed 0.6 run used the same source state. This confirms the sweep used one
consistent implementation.

## 4. Coarse sweep 0.4–0.8: the advantage is not robust

Seeds 42–51, v512. `ΔF1` and `Δbpc` are morph − frequency; `morph−random` is
the morphology-specific contrast. Published 0.6 is in bold.

| fraction | morph L1 | morph F1 | freq F1 | ΔF1 [95% CI] | t | W/L/T | Δbpc [95% CI] | t | morph−random ΔF1 |
|---:|---:|---:|---:|---|---:|---|---:|---:|---:|
| 0.4 | 176.5 | 0.8493 | 0.8742 | **−0.0249** [−0.0377, −0.0122] | −4.42 | 0/10/0 | **+0.1613** [+0.1002, +0.2225] | +5.97 | +0.0059 (ns) |
| 0.5 | 221.6 | 0.8586 | 0.8742 | **−0.0156** [−0.0335, +0.0024] | −1.96 | 4/6/0 | **+0.1294** [+0.0388, +0.2200] | +3.23 | +0.0007 (ns) |
| **0.6** | **294.0** | **0.8835** | **0.8742** | **+0.0092** [+0.0060, +0.0125] | **+6.52** | **10/0/0** | **−0.0258** [−0.0366, −0.0150] | **−5.38** | **+0.0156** |
| 0.7 | 343.0 | 0.8756 | 0.8742 | +0.0014 [+0.0006, +0.0022] | +4.00 | 8/0/2 | −0.0028 [−0.0044, −0.0012] | −3.91 | +0.0031 (ns) |
| 0.8 | 392.0 | 0.8743 | 0.8742 | +0.0000 [−0.0000, +0.0001] | +1.50 | 2/0/8 | −0.0001 [−0.0002, +0.0001] | −1.16 | +0.0006 (ns) |

Reading:

- At **0.4** morph is significantly *worse* on both metrics; all 10 seeds are
  worse on F1 and all 10 are worse on bpc.
- At **0.5** the F1 deficit is not significant at 5% (4/10 wins) but bpc is
  significantly worse (8/10 seeds), and the sign is negative.
- At **0.6** the published result is exact: +0.0092 F1, −0.0258 bpc, 10/10 on
  both.
- At **0.7** the effect is positive but ~6× smaller; at **0.8** it is zero.
- The morphology-specific contrast (`morph − random_closed`) is significant
  only at 0.6 in this grid; at 0.4, 0.5, 0.7 and 0.8 it is not.

So the answer to “specific to 0.6 or robust across 0.4–0.8?” is: **a window,
not a plateau across the requested grid**. The coarse grid alone cannot tell
whether 0.6 is a knife-edge or the centre of a wider window, hence the fine
sweep.

## 5. Fine sweep: the window is ~0.55–0.65 and the mechanism is intra-rule coverage

Seeds 42–51. `intra_cov` is the fraction of all majority-intra rules that end
up in L1; `L1 cross share` is cross-majority rules as a share of L1. Both are
computed from the CSV's `l1_intra`, `l1_cross`, `l2_intra` columns.

| fraction | intra_cov | L1 cross share | L2 intra (mean) | ΔF1 vs freq | t | Δbpc vs freq | t |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.40 | 0.672 | 0.000 | 86.2 | −0.0249 | −4.42 | +0.1613 | +5.97 |
| 0.50 | 0.839 | 0.005 | 42.3 | −0.0156 | −1.96 | +0.1294 | +3.23 |
| 0.52 | 0.876 | 0.010 | 32.5 | −0.0089 | −1.28 | +0.0959 | +3.34 |
| **0.55** | **0.977** | **0.027** | **6.0** | **+0.0186** | **+6.40** | **−0.0425** | **−4.06** |
| 0.58 | 1.000 | 0.072 | 0.0 | +0.0121 | +7.20 | −0.0370 | −6.60 |
| 0.60 | 1.000 | 0.103 | 0.0 | +0.0092 | +6.52 | −0.0258 | −5.38 |
| 0.62 | 1.000 | 0.130 | 0.0 | +0.0064 | +6.72 | −0.0165 | −5.39 |
| 0.65 | 1.000 | 0.171 | 0.0 | +0.0033 | +5.44 | −0.0069 | −4.76 |
| 0.70 | 1.000 | 0.232 | 0.0 | +0.0014 | +4.00 | −0.0028 | −3.91 |
| 0.80 | 1.000 | 0.328 | 0.0 | +0.0000 | +1.50 | −0.0001 | −1.16 |

The shape has a simple two-margin interpretation:

1. **Below ~0.55, the failure is stranded intra rules.** The morph ranking puts
   all majority-intra rules first, but truncating the list before the
   intra-majority set is exhausted leaves 12–33% of those rules in L2
   (`L2 intra` = 32–86 rules). Once an intra rule is in L2, lower-rank cross
   rules in L2 can fire before it and merge across the boundary. At 0.5 on
   seeds 42–51, the per-seed correlation between `L2 intra` and ΔF1 is
   **r = −0.53** (n = 10). The seed with 0 stranded rules (46) is positive, but
   the relation is not deterministic (seed 42 is also positive with 16 stranded
   rules). The fresh 0.5 block shows the same heterogeneity: the two seeds with
   full L1 (53, 55; `n_l1=245`, `L2 intra=0`) gain +0.015/+0.007, while most
   seeds with a closure shortfall lose 0.03–0.06 F1.
2. **Above ~0.55, the failure is cross contamination.** From 0.58 on, all intra
   rules are covered, `L1 intra` is fixed at ~263, and the only thing that
   changes is how many cross rules enter L1: 7 → 20 → 30 → 39 → 54 → 79 → 128
   as the fraction goes 0.55 → 0.80. F1 declines monotonically over the same
   range (+0.0186 → 0.0000). Adding early cross merges buys nothing once the
   intra rules are already first.

0.55 is the empirical balance: 97.7% intra coverage with only 2.7% cross
contamination. 0.6 is 100% coverage but 10.3% cross contamination; still a
clear gain, just a smaller one. 0.5 is the other side of the cliff (83.9%
coverage, 42 stranded intra rules), which is why a “natural” round value like
0.5 would have flipped the F1 sign and significantly worsened bpc.

The mechanism is consistent with the level-composition counts reported by
W1-12 (zero closure demotions at 0.6 on all ten published seeds) and with
W1-15's finding that level-1 size/closure effects dominate provenance
weighting: the operative variable is **which rules are in L1**, not the nominal
budget.

### 5a. Selected paired comparisons, seeds 42–51

| contrast | ΔF1 [95% CI] | t | W/L/T | Δbpc [95% CI] | t |
|---|---|---:|---|---:|---:|---:|
| 0.55 − 0.60 | +0.0094 [+0.0048, +0.0139] | +4.65 | 10/0/0 | −0.0167 [−0.0363, +0.0029] | −1.92 |
| 0.55 − 0.58 | +0.0065 [+0.0029, +0.0100] | +4.07 | 8/0/2 | — | — |
| 0.58 − 0.60 | +0.0029 [+0.0013, +0.0045] | +4.20 | 10/0/0 | — | — |
| 0.60 − 0.70 | +0.0079 [+0.0049, +0.0108] | +6.05 | 10/0/0 | — | — |
| 0.60 − 0.80 | +0.0092 [+0.0060, +0.0124] | +6.49 | 10/0/0 | — | — |

The 0.55 advantage over 0.6 is significant on F1 and all 10 seeds; the bpc
advantage is only marginal (t = −1.92, 8/10 seeds). So 0.55 is clearly the
better F1 point, not clearly the better compression point, on the discovery
seeds. The monotone decline from 0.55 to 0.8 is not noise: every adjacent
comparison is significant, with the fixed intra set and rising cross share.

## 6. Is 0.6 a post-hoc choice?

### 6.1 Where 0.6 came from

`grep` finds 0.6 in five relevant places:

- `experiments/level_assignment.py:80` — CLI `--fraction` default 0.6;
- `ramified/levels.py:102` — `level1_fraction=0.6` default;
- `experiments/run_matrix_v2.py:32` — `hier2` weights `[0.6, 0.4]`;
- `experiments/run_matrix.py:27` — the same `[0.6, 0.4]` in the v0.1 matrix;
- `experiments/check_levels.py:104` — `level_weights=[0.6, 0.4]`.

`docs/findings-v0.2.md` §6 runs `experiments.level_assignment` without
`--fraction`, so the published ablation inherited the old hierarchical budget
weight. No document reports an assignment-fraction sweep or a rationale for
0.6. The older `ablate_level1.py` sweeps the *hierarchical training* budget, not
the assignment fraction; its best tested v512 budget was 256/490 = 0.52. It
measures a different intervention, so it is not evidence that 0.6 was tuned for
this ablation.

Conclusion: **0.6 is an inherited default, not a value chosen after seeing the
v512 result.** That is the exculpatory half of the post-hoc question. The
repository has no git history, so “inherited” is inferred from the code paths
and the old `[0.6, 0.4]` weights, not from a commit log.

### 6.2 Is it the optimum?

No. The in-sample best on seeds 42–51 is 0.55 on both metrics
(+0.0186 F1, −0.0425 bpc). Leave-one-seed-out selection over the ten fractions
picks 0.55 in all ten folds, with held-out means equal to the in-sample means
because the ranking is unanimous. 0.6 is third of 10 on both metrics, behind
0.55 and 0.58. The published effect is therefore **not** inflated by picking
the best fraction; if anything it understates the achievable effect by ~2×.

But 0.55 as “the optimal fraction” is a post-hoc selection from a grid that was
refined after the coarse sweep. LOSO does not remove adaptive-grid selection.
The clean test is fresh seeds.

### 6.3 Fresh-seed validation (52–61)

| fraction | n_l1 morph | intra_cov | L1 cross share | morph F1 | freq F1 | ΔF1 [95% CI] | t | W/L/T | Δbpc [95% CI] | t | morph−random ΔF1 |
|---:|---:|---:|---:|---:|---:|---|---:|---|---:|---:|---:|
| 0.50 | 218.7 | 0.838 | 0.009 | 0.8325 | 0.8520 | −0.0195 [−0.0451, +0.0061] | −1.73 | 4/6/0 | +0.1392 [+0.0501, +0.2284] | +3.53 | +0.0125 (ns) |
| **0.55** | **267.5** | **0.992** | **0.039** | **0.8652** | **0.8520** | **+0.0132** [+0.0080, +0.0184] | **+5.75** | **10/0/0** | **−0.0362** [−0.0547, −0.0176] | **−4.40** | **+0.0338** |
| 0.60 | 294.0 | 1.000 | 0.118 | 0.8569 | 0.8520 | +0.0048 [+0.0021, +0.0076] | +3.99 | 10/0/0 | −0.0104 [−0.0193, −0.0015] | −2.65 | +0.0229 |

Selected fresh-seed paired comparisons:

| contrast | ΔF1 [95% CI] | t | W/L/T | Δbpc [95% CI] | t |
|---|---|---:|---|---:|---:|---:|
| 0.55 − 0.60 | +0.0083 [+0.0044, +0.0123] | +4.80 | 10/0/0 | −0.0257 [−0.0440, −0.0075] | −3.19 |
| 0.50 − 0.55 | −0.0327 [−0.0610, −0.0044] | −2.61 | 4/6/0 | +0.1754 [+0.0744, +0.2763] | +3.93 |
| 0.50 − 0.60 | −0.0243 [−0.0512, +0.0025] | −2.05 | 4/6/0 | +0.1496 [+0.0563, +0.2430] | +3.63 |

The fresh seeds confirm both sides:

- **0.6 survives out-of-sample**, attenuated to +0.0048 F1 / −0.0104 bpc, with
  10/10 wins on both metrics. The direction is real.
- **0.55 > 0.6 out-of-sample**, +0.0083 F1, 10/10, and −0.0257 bpc, 9/10. The
  0.55 optimum is not an in-sample artifact.
- **0.5 is on the wrong side of the cliff** out-of-sample too: F1 −0.0195
  (4/10 wins, not significant at 5%), bpc +0.1392 (8/10 worse, significant).
  0.5 vs 0.55 is −0.0327 F1, 4/10, with bpc significantly worse. The reversal
  is directionally robust, and the compression harm is statistically clear.
- **The morphology-specific contrast is larger than the frequency contrast.**
  At 0.55, morph − random_closed is +0.0338 F1 / −0.1123 bpc (10/10); at 0.6 it
  is +0.0229 / −0.0767 (9/10). The published random control is weaker than
  morph at every fraction in the window.

The fresh-seed 0.55 result also shows the coverage mechanism directly: on seeds
52–59 the requested `n_l1=269` is fully realized with `L2 intra=0`; on seed 60,
closure leaves 256 rules and 17 stranded intra rules, yet the seed still gains
+0.0126 F1; seed 61 leaves 267 and 4, gaining +0.0289. Coverage is the dominant
term, but it is not a deterministic per-seed law.

## 7. What survives, exactly

**Survives:**

1. The published 0.6 number: exact reproduction on seeds 42–51, and a
   statistically significant replication on seeds 52–61 (+0.0048 F1,
   −0.0104 bpc, 10/10 each).
2. The morphology-specific claim at 0.6: morph beats the size-matched random
   control by +0.0156 F1 on the published seeds and +0.0229 on fresh seeds.
3. 0.6 as a defensible inherited default: it lies inside the window, it is not
   the in-sample maximum, and no document reports it as tuned on these seeds.

**Does not survive:**

1. Any reading of the result as robust across `level1_fraction` 0.4–0.8. The
   effect is a window ~0.55–0.65. At 0.4 it is significantly negative; at 0.5
   and 0.52 it is negative on F1 and significantly negative on bpc; at 0.7 it is
   ~6× smaller; at 0.8 it is zero.
2. The implicit claim that a fixed fraction is a meaningful constant. The
   operative variable is intra-rule coverage: L1 must contain essentially all
   majority-intra rules (97–100%) and as few cross rules as possible. The
   coverage threshold here is ~0.54 of the budget (262.7/490), the empirical
   optimum sits just above it at 0.55, and the constant will move with the
   corpus's intra/total merge ratio; it is not a universal 0.6.
3. The phrase “morph improves at v512” without qualification. A reader would
   reasonably infer robustness to the one hyperparameter that controls the
   intervention; the sweep shows the sign flips at 0.5.

**The honest paper sentence:**

> At v512, when level 1 contains essentially all intra-morpheme merges
> (~55% of the merge budget on this synthetic corpus, 97–100% coverage), the
> morphological assignment improves boundary F1 by 0.5–1.9 points and lowers
> bpc by 0.010–0.043 against the frequency control, on 10/10 seeds in both the
> original and the fresh seed set. The effect shrinks to zero above ~0.7 and
> turns negative below ~0.52. The published 0.6 is a conservative point inside
> the window, not a tuned optimum.

Better still: define L1 by coverage (“all majority-intra rules plus dependency
closure”) and report its realized size, rather than fixing a fraction. That
removes the arbitrary constant and makes the mechanism the claim.

## 8. Caveats and relation to other W1 reports

- **One synthetic generator.** The 490 merges, the ~250–263 majority-intra
  rules and their ratio are properties of `generate_synthetic`. The window
  location is tied to that ratio. A real morphological corpus with a different
  intra/total ratio may put the optimum elsewhere; the coverage rule predicts
  where, but that prediction is untested.
- **Adaptive fine grid.** The coarse sweep motivated 0.52–0.65. LOSO over the
  combined grid is unanimous for 0.55, but LOSO cannot remove the choice of
  grid points. The fresh-seed 0.55 result is the strongest available check; it
  confirms the direction and the ordering, not the exact 0.55 value.
- **Multiplicity.** Ten fractions × three contrasts × two metrics, with
  per-comparison 95% intervals; no family-wise correction. The key results are
  not borderline: 0.55 vs 0.6 is 10/10 on both seed sets; 0.5 vs 0.55 is 4/10
  with a large negative bpc effect; the monotone 0.55→0.8 decline is
  significant at every adjacent step (8–10 of 10 seeds). The many “ns” cells in
  the coarse grid should not be read as evidence of no effect; they are
  underpowered at n=10 for small differences.
- **Intrinsic metrics only.** Boundary F1 and unigram bpc say nothing about
  downstream representation quality; the Phase-2 embedding test is still the
  right next step.
- **`random_closed` size bug (W1-12).** The published sampler occasionally
  installs 1–2 fewer L1 rules than requested (mean 293.6 at 0.6, 391.5 at 0.8;
  three published rows at 292–293). The shortfall is <0.5%; it affects the
  control, not the primary morph−frequency contrast. W1-14's corrected sampler
  gives a larger morph advantage (+0.0220 F1), consistent with our fresh-seed
  +0.0229.
- **W1-15 (provenance weighting)** finds that disjoint-word/root provenance
  gains are mostly level-1 size/closure effects and that unique vs
  frequency-weighted provenance selects the same 294 rules at 0.6. Our sweep
  explains why: at 0.6 the intra set is already fully covered, so provenance
  variants that change the ranking but not the coverage are nearly inert; the
  action is at the coverage margin, which we vary directly.
- **W1-12 (closure)** reports zero demotions at 0.6 on all ten published seeds.
  Our composition table matches: `n_l1=294`, `L2 intra=0` at 0.6. Below 0.55,
  closure demotions are exactly what creates the stranded intra rules and the
  performance cliff.

## 9. Artifacts

| path | contents |
|---|---|
| `results/redteam/level_frac_0{40,50,60,70,80}_seeds42-51.csv` | coarse sweep, 10 seeds |
| `results/redteam/level_fine_0{52,55,58,62,65}_seeds42-51.csv` | fine sweep, 10 seeds |
| `results/redteam/level_frac_060_seeds52-61.csv` | fresh 0.6, 10 seeds |
| `results/redteam/level_fine_055_seeds52-56.csv`, `..._seeds57-61.csv` | fresh 0.55, 10 seeds |
| `results/redteam/level_frac_050_seeds52-61.csv` | fresh 0.5 control, 10 seeds |
| `results/redteam/fraction_summary_all_seeds42-51.txt` | full paired analysis, 10 fractions |
| `results/redteam/fraction_summary_fresh52-61.txt` | full paired analysis, fresh seeds |
| `results/redteam/analyze_fraction.py` | analysis script (stdlib only) |
| `results/redteam/log_*.txt` | run logs |
| `results/redteam/provenance_hashes.txt` | source hashes at run time |

## 10. Bottom line for the paper

The v512 result is real at 0.6 and replicates on fresh seeds, but it is a
**coverage-window effect**, not a fraction-robust one. Reporting only 0.6 hides
that an equally natural 0.5 reverses the F1 sign and significantly worsens
compression, and that 0.55 is about twice as good as 0.6. The paper should
either (a) report the sweep and state the 0.55–0.65 window, or (b) replace the
fixed fraction with an explicit coverage rule. Option (b) is preferable: it is
the mechanism, it removes the post-hoc constant, and it predicts where the
window should move on a real corpus — which is the single check that would most
change this verdict.
