# W1-18 — audit of `bits_per_char` (v512 morph vs frequency)

Red-team note. No source files were modified. All numbers below were recomputed
from the corpus generator and the source modules; the published v512 CSV is
reproduced exactly.

**Scope.** 10 seeds (42–51), synthetic corpus, vocabulary 512, modes
`frequency`, `morph`, `random_closed`, plus `flat`. The question is the
secondary metric of `docs/findings-v0.2.md` §2: morph − frequency
Δ bpc = −0.0258 (t = −5.38, 10/10 seeds).

**Artifacts.**

| file | what |
|---|---|
| `results/redteam/w1_18_bpc_audit.py` | main recomputation + alternative estimators |
| `results/redteam/w1_18_bpc_audit.csv` | one row per (seed, mode), 34 columns |
| `results/redteam/w1_18_bpc_paired.csv` | paired stats per estimator and mode pair |
| `results/redteam/w1_18_bpc_audit.log` | full stdout of the main run |
| `results/redteam/w1_18_bpc_alt.py` | distribution / model-order checks |
| `results/redteam/w1_18_bpc_alt.csv` | per-seed in-distribution, bigram, KT bpc |
| `results/redteam/w1_18_indist_oov.py` | OOV check for the in-distribution split |
| `results/redteam/w1_18_indist_oov.csv` | per-seed unseen tokens/types, in-distribution |
| `results/redteam/w1_18_vocabcheck.py` | level-set / tokenization check at v512/1024/2048 |

## TL;DR

1. **Not an add-alpha, denominator, or unseen-token artifact.** With α from
   1.0 down to 1e-6, and with Witten–Bell, the paired difference is
   −0.0255 … −0.0259 bpc, 10/10 seeds. Only 3 unseen test tokens occur in
   59,319 (all in morph, seeds 44 and 50), and removing their cost makes the
   gain slightly larger (−0.0261). The normalised length is identical across
   modes within each seed, so the denominator cannot drive the paired delta.
2. **Not a token-count effect.** Δ tok/ch = +0.00004 (5/4/1, t = 0.25).
   A fixed 9-bit/token code gives +0.0003 bpc (no gain). The whole gain is a
   lower mean surprisal per token: −0.0867 bits/token (~1.1%).
3. **Not independent evidence for “compression”.** Δ bpc and Δ boundary F1
   correlate −0.911 across seeds; F1 explains 83% of the Δ bpc variance
   (R² = 0.830). The two metrics use different samples (bpc: 2,000 draws,
   1,324 distinct held-out types; F1: all 2,304 held-out types), so this is
   not mechanical identity — it is the same underlying alignment effect.
4. **Not robust to distribution or model order.** On natural-frequency
   held-out text (90/10 split of the training text), morph is *worse* by
   +0.0096 bpc (8/2, t = 3.65); on the training stream it is +0.0095 worse
   (9/1). An interpolated bigram code shrinks the uniform-test gain by
   86–92%, to −0.002 … −0.004 bpc. The −0.0258 is a unigram/held-out-
   recombination effect, not a general compression gain.
5. **Larger-vocabulary “null” is degenerate.** At v2048 the morph and
   frequency level-1 sets are literally identical (Jaccard 1.0). At v1024 they
   differ by 12 rules (Jaccard 0.980) but produce bit-identical tokenisation.
   The v1024/2048 zeros therefore say nothing about morphology at larger
   vocabularies; they only say the morph ordering has converged to rank order.

**Verdict.** The v512 bpc *sign* survives every smoothing / denominator /
unseen-token variant tested (−0.0255 … −0.0259, 10/10); it is not an
implementation artifact. The *magnitude and the compression interpretation* do
not survive: an interpolated bigram code removes 86–92% of the gain,
in-distribution text reverses its sign (+0.0096, 8/2), ~half of it is already
present when the morph stream is scored by the frequency model, and it is ~83%
collinear with F1. Report it as “held-out unigram bpc at v512”, with the
fixed-code, in-distribution and bigram baselines, not as “better compression”.

## 1. What the estimator actually computes

`experiments/level_assignment.py:33-42`:

```python
counts = Counter(train_ids); total = sum(counts.values())
denom = total + alpha * vocab_size            # alpha = 0.1
bits -= log2((counts.get(t, 0) + alpha) / denom)
return bits / char_len
```

- `train_ids` is the tokenisation of the *training* text by the same
  tokenizer; `test_ids` is the tokenisation of `syn.test_text`.
- `vocab_size = len(tok.expansions)` = 512 nominal (20 characters + `<unk>` +
  EOW + 490 merges). Only ~434 types are emitted in training (morph: ~413), so
  most of the 512 symbols receive only the α mass. The α·V term is
  51.2 / 208,661 = **2.45e-4** of the denominator: negligible.
- `char_len = len(normalize(test_text))` = 19,370–19,688, **identical for all
  modes within a seed** (the audit asserts this). It includes the 1,999
  inter-word spaces (10.3% of L). The token stream contains 2,000 EOW tokens
  (33.4% of ~5,900 tokens), so the space↔EOW mapping is roughly 1:1. The
  denominator is defensible, but the absolute number is “bits per normalised
  character including separators”, not a character-compressor rate and not
  comparable to the code-corpus bpc (data-review C1).
- This is an **add-0.1 Lidstone** estimator, not add-1. The doc's “add-alpha
  0.1” is accurate for the published v0.2 tables. However,
  `ramified/metrics.py:11-24` implements add-1 with denominator `N + V`, and
  that is the function exported by `ramified/__init__.py`. Rerunning the same
  data through it gives 2.353619 instead of 2.353869 on seed 42 flat, and a
  paired Δ of −0.02578 instead of −0.02579. The divergence is cosmetic here
  (N is large, unseen ≈ 0) but it is a replication trap; four copies of the
  α = 0.1 function exist (`level_assignment.py`, `run_matrix_v2.py`,
  `codec_compare.py`, `ablate_level1.py`).

**Reproduction.** `max |recomputed − results/level_assignment_v512_10seeds.csv|
= 6.7e-14` over all 40 rows.

## 2. Alpha, unseen tokens, denominator

Paired differences, morph − frequency, v512, 10 seeds (negative bpc = morph
better). W/L/T counts Δ > 0 / Δ < 0 / Δ = 0; for bpc, lower is better, so the
“losses” column is the desired direction.

| estimator | Δ (bpc unless noted) | t | 95% CI | W/L/T |
|---|---:|---:|---|---:|
| add-0.1 (published) | −0.02579 | −5.38 | [−0.03662, −0.01495] | 0/10/0 |
| add-1 (`metrics.py`) | −0.02578 | −5.38 | [−0.03662, −0.01494] | 0/10/0 |
| add-0.01 | −0.02574 | −5.39 | [−0.03656, −0.01493] | 0/10/0 |
| add-0.001 | −0.02569 | −5.39 | [−0.03648, −0.01490] | 0/10/0 |
| add-1e-6 (≈ MLE + floor) | −0.02554 | −5.39 | [−0.03626, −0.01482] | 0/10/0 |
| Witten–Bell | −0.02593 | −5.38 | [−0.03683, −0.01503] | 0/10/0 |
| fixed denominator (flat N) | −0.03342 | −5.24 | [−0.04786, −0.01898] | 0/10/0 |
| seen tokens only | −0.02611 | −5.37 | [−0.03711, −0.01511] | 0/10/0 |
| tpc × H_train (plug-in) | −0.02014 | −4.72 | [−0.02978, −0.01049] | 1/9/0 |
| tpc × H_test (plug-in) | −0.01556 | −5.16 | [−0.02239, −0.00874] | 1/9/0 |
| fixed 9-bit/token code | **+0.00033** | +0.25 | [−0.00272, +0.00338] | 5/4/1 |
| in-sample train stream | **+0.00947** | +3.73 | [+0.00373, +0.01521] | 9/1/0 |
| tok/ch (tokens/char) | +0.00004 | +0.25 | [−0.00030, +0.00038] | 5/4/1 |
| unseen tokens (count) | +0.30 | +1.41 | [−0.18, +0.78] | 2/0/8 |

Notes:

- **Unseen tokens.** Frequency has 0 unseen test tokens in all 10 seeds. Morph
  has 2 (seed 44) and 1 (seed 50), 3 / 59,319 = 0.005%. At ~21 bits each,
  their cost is 0.0021 and 0.0011 bpc in those two seeds. Removing unseen
  tokens from the numerator (`seen tokens only`) gives −0.02611, i.e. a
  slightly *larger* morph gain. The unseen treatment is conservative, not the
  source of the effect.
- **Denominator.** `char_len` is fixed within a seed. The only mode-dependent
  denominator is `N_train`: morph emits +1.77% more training tokens than
  frequency (212,358 vs 208,661), so the published estimator charges morph for
  its extra tokens. Holding the denominator at flat's N gives −0.03342, i.e.
  the published −0.0258 is the *conservative* choice. (The fixed-denom variant
  is a sensitivity check, not a valid normalised model: probabilities no
  longer sum to one for modes with different N.)
- **Vocab-size term.** α·V/N = 2.45e-4 at v512 (2.34e-4 for random_closed,
  2.41e-4 for morph). The α·V term cannot explain the delta or the cross-V
  non-monotonicity.

## 3. The unigram model: where the gain comes from

Decomposition of the published number:

```text
mean Δ bpc            = -0.02579
mean Δ tok/char       = +0.000037
mean Δ surprisal/token= -0.08665 bits   (~ -1.1%)
mean tpc              ≈  0.303
0.303 * (-0.08665)    ≈ -0.02625       (all of the gain, no token reduction;
                                        residual 0.00046 is the Δtpc covariance)
```

- **Fixed 9-bit code.** A token costs log2(512) = 9 bits regardless of
  frequency. Then bpc = 9 × tok/char and morph is +0.00033 bpc (5/4/1, not
  significant). The published gain therefore comes entirely from the fitted
  unigram distribution, not from emitting fewer or shorter tokens.
- **In-sample vs held-out.** On the training stream, under each mode's own
  counts, morph is +0.00947 bpc worse (9/1). On held-out recombinations it is
  −0.0258 better. This is the signature of a generalisation trade-off, not of
  a uniformly better code.
- **Own model vs competitor model.** Mean bpc of the 2×2 cross-score:

  | stream | model=frequency | model=morph |
  |---|---:|---:|
  | frequency | 2.30032 | 2.30884 |
  | morph | **2.28779** | **2.27454** |

  On the means, the morph *stream* is 0.0125 bpc cheaper even under the
  frequency model; its own model adds another 0.0133. So about half of the
  published gain is a property of the segmentation (the test stream is more
  likely under either model), and about half is own-model fit.
- **Entropy diagnostics.** The morph test token distribution is more
  concentrated: H_test is lower by 0.03–0.09 bits/token in 9/10 seeds, and
  `tpc × H_test` gives −0.0156. Training entropy gives −0.0201. Neither is a
  code length by itself (plug-in entropy is biased down), but both agree on
  the direction: morph's token stream is more predictable, not shorter.

## 4. Fairness across tokenizers and vocabularies

- **Within v512 everything is paired and fair.** Same corpus, same seed, same
  490 rules, same 512-symbol vocabulary, same test text, same L; only the
  assignment of rules to two levels changes. V is identical, so the α·V term
  is common-mode.
- **Across V, bpc is not a fair comparison.** From the 3-seed
  `results/level_assignment.csv`:

  | V | mean tok/ch | mean bits/token | mean bpc |
  |---:|---:|---:|---:|
  | 512 | 0.3054 | 7.624 | 2.3287 |
  | 1024 | 0.2770 | 8.494 | 2.3522 |
  | 2048 | 0.2285 | 9.961 | 2.2754 |

  Flat (= frequency) tokenizer, mean of the 3 seeds; morph and random_closed
  are omitted because they are the treatments. Token count falls, per-token
  surprisal rises, and bpc is non-monotone. The α·V term is ≤1.4e-3 of N, so
  this is token granularity, not smoothing. Do not read 2.35 → 2.38 → 2.33 as
  “compression at different vocabularies”.
- **The larger-V null is degenerate.** Recomputing the level sets for seed 42:

  | V | rules | L1 size | Jaccard(morph L1, freq L1) | rules differing | test tokenisation identical? |
  |---:|---:|---:|---:|---:|---|
  | 512 | 490 | 294 | 0.6945 | 106 | no (5,982 vs 5,993) |
  | 1024 | 1,002 | 601 | 0.9802 | 12 | **yes** (5,449 both) |
  | 2048 | 2,026 | 1,215 | **1.0000** | **0** | **yes** (4,544 both) |

  At v2048 morph and frequency are literally the same assignment. At v1024
  they differ in 12 rules, but none of those differences changes the greedy
  tokenisation of the held-out test text. The `docs/findings-v0.2.md` §3
  “budget window” reading is not tested by this null: the experiment cannot
  distinguish “morphology stops mattering” from “the morph ordering has become
  the rank ordering”. The v512 result is unaffected, but the mechanism
  sentence should be rewritten.
- **3-seed vs 10-seed.** The 3-seed v512 bpc CI was [−0.0516, +0.0119]
  (includes zero); the 10-seed extension is what makes it significant. That is
  a fragility/forking-path caution, not an error: the published 10-seed run is
  internally valid, but the effect is small enough that seed selection matters.

## 5. Statistical checks

- Paired t: −5.38 (9 df), CI [−0.03662, −0.01495], 10/10 seeds. Exact sign
  test and exact sign-flip permutation: p = 2/1024 = **0.00195**; paired t
  p = 4.4e-4.
- Word-level bootstrap of the test sample (2,000 resamples): the 95% CI
  excludes zero in 9/10 seeds; seed 46 is the exception
  ([−0.00206, +0.00011], P(Δ≥0) = 0.039). The mean is not carried by one
  outlier, but seed 46 is effectively a tie.
- F1: +0.00924, t = 6.52, CI [+0.00604, +0.01245], 10/10.
- Δ bpc vs Δ F1: Pearson −0.911, OLS Δbpc = +0.00265 − 3.077·ΔF1, R² = 0.830,
  residual sd = 0.00625. F1 and bpc use different samples (the bpc text covers
  1,324 of 2,304 held-out types, 57.5%), so this is not mechanical; it is the
  same segmentation property showing up in two metrics.
- Multiple comparisons: the findings table has 6 paired tests; the bpc
  morph−frequency two-sided p is ≈4.4e-4 (t = −5.38, 9 df), and ≈2.7e-3 after
  Bonferroni over 6 tests. The result survives correction, but it is a
  small-sample result with n = 10.
- Random control: `random_closed − frequency` bpc is +0.0361 (t = 3.04,
  8/2/0) under add-0.1, and +0.0361 under every smoothing variant. This is
  reassuring: the estimator does not simply reward “any reordering”. But note
  that random_closed emits +5.07% more training tokens than frequency and
  +0.0054 tok/ch on test; its bpc penalty is partly a token-count penalty
  (fixed-denom gives only +0.0142, t = 1.59). The paper's morph-vs-random
  comparison is therefore partly a token-efficiency comparison, not purely a
  likelihood comparison.

## 6. Does the v512 result survive?

**As a numerical result: yes.** The published −0.0258 bpc is reproducible to
1e-13, is stable at −0.0255 … −0.0259 across six smoothing schemes, is not
driven by unseen tokens (3 tokens total), is not driven by the L denominator
(fixed within seed), and gets larger, not smaller, under a fixed-denominator
sensitivity check. The per-seed sign is 10/10 (exact sign-flip p = 0.002;
paired t p = 4.4e-4).

**As evidence of better compression: no, not as stated.** Four facts point the
same way:

1. there is no token-count reduction (Δ tok/ch = +0.00004) and no gain under a
   fixed 9-bit/token code (+0.0003 bpc, ns);
2. morph is *worse* in-sample (+0.0095 bpc, 9/1) and *worse* on
   natural-frequency held-out text (+0.0096 bpc, 8/2), so the uniform held-out
   gain is a distribution-specific generalisation trade-off, not a uniformly
   better code;
3. an interpolated bigram code removes 86–92% of the gain (−0.0036 / −0.0022
   bpc, still 9/10 but ~0.1–0.2% of the baseline), so the effect is largely an
   artifact of the unigram modelling assumption;
4. 83% of the Δ bpc variance is explained by Δ F1, and about half of the gain
   survives when the morph stream is scored by the frequency model. The bpc
   number is a second view of the alignment effect, not independent
   compression evidence.

**What to write instead.** “At v512, on uniform held-out recombinations of seen
morphemes, the morph assignment lowers unigram bpc by 0.026 relative to the
frequency assignment (10/10 seeds; paired t = −5.38), with no change in token
count. The effect is a held-out likelihood/generalisation effect: it is
strongly collinear with the boundary-F1 gain, it reverses on natural-frequency
text (+0.0096 bpc), and 86–92% of it disappears under a bigram code. At
v1024/v2048 the two assignments produce identical tokenisation, so those cells
are not a test of the mechanism.” Add the fixed-code, in-distribution and
bigram baselines, and report OOV counts.

## 7. Distribution and model-order checks

The published estimator is a unigram code fitted on the training stream and
evaluated on `test_text`, which is 2,000 uniform draws from the 2,304 held-out
word types. Two checks the published estimator cannot answer:

- **In-distribution text.** Fit the unigram counts on the first 90% of the
  training chunks and score the last 10%. This is Zipfian text of *seen* words
  and periods, not uniform recombinations, i.e. the closest available proxy
  for natural-frequency text in this generator. The tokenizer itself is still
  trained on all chunks, as in the official setup; only the unigram counts are
  held out. Absolute bpc levels are not comparable to the uniform test
  (different text and predictability); only the paired delta is used.
- **Second model.** Interpolated bigram/unigram code on the official test
  stream: `p(t | prev) = λ·c(prev,t)/c(prev) + (1−λ)·p_uni(t)`, with add-0.1
  unigram backoff; λ = 0.5 and 0.8. Also an adaptive Krichevsky–Trofimov (KT)
  code with no α and no train/test counts at all.

| estimator | Δ (bpc unless noted) | t | 95% CI | W/L/T |
|---|---:|---:|---|---:|
| unigram, uniform held-out (published) | −0.02579 | −5.38 | [−0.03662, −0.01495] | 0/10/0 |
| unigram, in-distribution Zipfian 90/10 | **+0.00964** | +3.65 | [+0.00367, +0.01561] | 8/2/0 |
| bigram λ = 0.5, uniform held-out | −0.00359 | −5.39 | [−0.00510, −0.00208] | 1/9/0 |
| bigram λ = 0.8, uniform held-out | −0.00218 | −3.54 | [−0.00357, −0.00079] | 1/9/0 |
| adaptive KT (no α, no train counts) | −0.01521 | −5.12 | [−0.02193, −0.00849] | 1/9/0 |
| tok/ch, in-distribution (tokens/char) | +0.00313 | +4.38 | [+0.00151, +0.00475] | 10/0/0 |

W/L/T = Δ > 0 / Δ < 0 / Δ = 0; for bpc, lower is better.

- **In-distribution reverses the sign.** Morph is **worse** by +0.00964 bpc
  (8/2, p ≈ 0.005) on natural-frequency held-out text, and it emits +0.00313
  more tokens per character (10/10). The published −0.0258 is therefore
  specific to the uniform held-out-recombination distribution. Seed-level
  spread is large: the first two seeds happen to be near zero, so the 10-seed
  mean is what matters.
- **The bigram code removes 86–92% of the gain.** The morph advantage falls
  from −0.0258 to −0.0036 (λ = 0.5) / −0.0022 (λ = 0.8), i.e. from ~1.1% to
  ~0.1–0.2% of the baseline. The sign is still 9/10, but the magnitude is
  negligible. A context model absorbs most of what the unigram model rewards.
- **Adaptive KT keeps 59%.** With no arbitrary α and no train/test split,
  morph is −0.0152 (9/10), 59% of the published gain. So part of the effect is
  real under a universal code; it is not solely a smoothing or train-split
  artifact. But the bigram result shows the size is model-order dependent.
- **The controls still separate.** `random_closed` is worse than frequency
  under every alternative: in-distribution +0.0539 (10/0), bigram +0.0176
  (9/1), KT +0.0461 (9/1). The shrinking morph gain is not because the
  alternative estimators became noise.

- **In-distribution OOV is zero.** On the same 90/10 split, both modes have
  **0 unseen tokens and 0 unseen types in all 10 seeds**, so
  `bpc_add0.1` and `bpc_seen_only` are identical (+0.009637 both). The
  in-distribution reversal is therefore not an add-alpha penalty for unseen
  subwords: morph genuinely spends more bits per character on natural-
  frequency held-out text, and it also emits +0.00313 more tokens per
  character (10/10).
- **The reversal is conservative.** The morph level assignment is computed
  from provenance counts over the *full* training text, including the 10%
  held out from the unigram counts; the frequency assignment is rank-only. If
  anything, morph has a small information advantage on this split and still
  loses. The tokenizer itself is also trained on all chunks for both modes
  (shared advantage).

**Conclusion of §7.** The v512 morph bpc gain is not an implementation
artifact, but it is a *unigram-model* effect: a bigram code removes 86–92% of
it, and natural-frequency in-distribution text reverses its sign. This is the
failure mode data-review H4 warned about: “Do not treat a single unigram BPC
point as ‘compression’.”

## 8. Commands

```powershell
# main audit (10 seeds, v512; writes the CSV + log)
$env:PYTHONPATH=(Get-Location).Path
python results/redteam/w1_18_bpc_audit.py

# alternative distributions / models (10 seeds; may be split by --seeds)
python results/redteam/w1_18_bpc_alt.py
python results/redteam/w1_18_bpc_alt.py --seeds 42,43 --out results/redteam/w1_18_bpc_alt_42_43.csv
# ... five 2-seed shards were used for wall time, then combined:
python results/redteam/w1_18_bpc_alt_summary.py

# in-distribution OOV check (two 5-seed shards, then combined by the summary)
python results/redteam/w1_18_indist_oov.py --seeds 42,43,44,45,46 --out results/redteam/w1_18_indist_oov_a.csv
python results/redteam/w1_18_indist_oov.py --seeds 47,48,49,50,51 --out results/redteam/w1_18_indist_oov_b.csv

# level-set / tokenisation check at 512, 1024, 2048
python results/redteam/w1_18_vocabcheck.py
```

Key output of the main audit:

```text
max |recomputed - CSV| = 6.706e-14 over 40 rows

morph - frequency, v512, 10 seeds
bpc_official_a0.1        mean=-0.02579 t=  -5.38 CI95=[-0.03662,-0.01495] W/L/T=0/10/0
bpc_add1_metrics_py      mean=-0.02578 t=  -5.38 CI95=[-0.03662,-0.01494] W/L/T=0/10/0
bpc_add1e-6              mean=-0.02554 t=  -5.39 CI95=[-0.03626,-0.01482] W/L/T=0/10/0
bpc_witten_bell          mean=-0.02593 t=  -5.38 CI95=[-0.03683,-0.01503] W/L/T=0/10/0
bpc_fixed_denom_flatN    mean=-0.03342 t=  -5.24 CI95=[-0.04786,-0.01898] W/L/T=0/10/0
bpc_uniform9             mean=+0.00033 t=  +0.25 CI95=[-0.00272,+0.00338] W/L/T=5/4/1
bpc_train_stream_insample mean=+0.00947 t= +3.73 CI95=[+0.00373,+0.01521] W/L/T=9/1/0
tokens_per_char          mean=+0.00004 t=  +0.25 CI95=[-0.00030,+0.00038] W/L/T=5/4/1
unseen_tokens            mean=+0.30000 t=  +1.41 CI95=[-0.18280,+0.78280] W/L/T=2/0/8
boundary_f1              mean=+0.00924 t=  +6.52 CI95=[+0.00604,+0.01245] W/L/T=10/0/0
```

```text
corr(d_bpc,d_F1)=-0.911
OLS d_bpc = +0.00265 + -3.077*d_F1 ; R2=0.830
```

```text
v=512  n_rules=490  n_l1=294  jaccard_l1=0.6945  rules_level_diff=106
       test_tokenization_equal=False  n_test_tokens_f=5982 n_test_tokens_m=5993
v=1024 n_rules=1002 n_l1=601  jaccard_l1=0.9802  rules_level_diff=12
       test_tokenization_equal=True   n_test_tokens_f=5449 n_test_tokens_m=5449
v=2048 n_rules=2026 n_l1=1215 jaccard_l1=1.0000  rules_level_diff=0
       test_tokenization_equal=True   n_test_tokens_f=4544 n_test_tokens_m=4544
```

The alternative checks (run as five 2-seed shards for speed, then combined
with `w1_18_bpc_alt_summary.py`):

```text
official cross-check: max |alt - main| = 0.000e+00

morph - frequency, v512, 10 seeds
bpc_official_uniform_heldout     mean=-0.02579 t=  -5.38 CI95=[-0.03662,-0.01495] W/L/T=0/10/0
bpc_indist_zipf_90_10            mean=+0.00964 t=  +3.65 CI95=[+0.00367,+0.01561] W/L/T=8/2/0
bpc_bigram_lam0.5                mean=-0.00359 t=  -5.39 CI95=[-0.00510,-0.00208] W/L/T=1/9/0
bpc_bigram_lam0.8                mean=-0.00218 t=  -3.54 CI95=[-0.00357,-0.00079] W/L/T=1/9/0
bpc_kt_adaptive                  mean=-0.01521 t=  -5.12 CI95=[-0.02193,-0.00849] W/L/T=1/9/0
tpc_uniform_heldout              mean=+0.00004 t=  +0.25 CI95=[-0.00030,+0.00038] W/L/T=5/4/1
tpc_indist_heldout               mean=+0.00313 t=  +4.38 CI95=[+0.00151,+0.00475] W/L/T=10/0/0

in-distribution 90/10 OOV, 10 seeds
unseen_tokens    mean=+0.000000 sd=0.000000 W/L/T=0/0/10
unseen_types     mean=+0.000000 sd=0.000000 W/L/T=0/0/10
bpc_add0.1       mean=+0.009637 sd=0.008348 t=+3.65 CI95=[+0.003666,+0.015608] W/L/T=8/2/0
bpc_seen_only    mean=+0.009637 sd=0.008348 t=+3.65 CI95=[+0.003666,+0.015608] W/L/T=8/2/0
tpc              mean=+0.003131 sd=0.002261 t=+4.38 CI95=[+0.001514,+0.004748] W/L/T=10/0/0
```

## 9. Confidence and the one check that would change the verdict

- Confidence in the estimator mechanics: **high**. The published numbers are
  reproduced to 1e-13; the alpha sweep, unseen decomposition, fixed
  denominator, and word bootstrap all point the same way. Nothing in the
  implementation of `bpc_alpha` inflates the morph gain.
- Confidence that the published unigram number overstates a general
  “compression” gain: **high**. The in-distribution test has zero OOV for both
  modes and still reverses the sign (+0.0096, 8/2); the bigram code removes
  86–92% of the gain.
- Confidence that morph has *no* compression advantage on real text:
  **medium**. The synthetic test is uniform over recombinations of seen
  morphemes, which is exactly the distribution on which morpheme-aligned
  tokens should win; a real corpus could still show a small gain under a
  bigram code. The 10-seed spread is also large (per-seed Δ bpc from −0.001 to
  −0.049).
- **Single check that would most change the verdict:** rerun the same paired
  design on a real morphologically annotated corpus with natural word
  frequencies and report OOV plus a second-model (bigram/KN) bpc. If morph
  still wins there under the bigram code, a compression claim becomes
  defensible; if it loses or ties, the synthetic uniform-recombination test is
  the driver and the claim must stay a generalisation result.
