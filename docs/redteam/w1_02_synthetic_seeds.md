# W1-02 — Synthetic corpus across 20 seeds and the v512 morph result

**Status.** Red-team report, read-only with respect to project source. No source file
was modified. Date: 2026-11-22. Project root: `C:\Users\acese\Desktop\unibull`.

**Environment.** Windows, Python 3.13.13 (Anaconda), NumPy 2.2.6, SciPy 1.15.2,
16 logical CPUs, 16 GB RAM.

**Source revision pinned by SHA-256.**

| file | SHA-256 |
|---|---|
| `ramified/bpe.py` | `73093A3E62FC66A59594F29CDB8549F4523D358DF04EBAB46E412C7C47C5B0C2` |
| `ramified/synthetic.py` | `E7D4E5FBB01E86C41541785F3FA277A79AB49B545741C8453D296F8C8A0B6D22` |
| `ramified/levels.py` | `14420E923EE6C4034EDEA192DE161479772DD274406AD80DF95BC184E5CF7DED` |
| `ramified/metrics.py` | `14DBF6B1794ABFD9802C503BBF5C9E3D6B2922FE88E93F12246866A952FA4D7D` |
| `experiments/level_assignment.py` | `167C55CC81EE373672A88D018786996716013E76B5982F40072F7731DEEB4517` |

The BPE and synthetic-corpus hashes match the prefixes reported in
`docs/reviews/data-review.md`, so the reviewed generator and trainer are the same
files used here.

---

## 0. Verdict, short

**The v512 morph result survives a fresh 10-seed replication, but only as a small,
paired, corpus-conditional ordering effect.** The original 10 seeds (42–51) gave
`morph − frequency = +0.00924` macro F1 (t = 6.52, 10/10). A fresh 10 seeds
(52–61), not used in the original analysis, gives `+0.00483` (t = 3.99, 10/10,
95% CI `[+0.00209, +0.00758]`). The all-20 estimate is `+0.00704`
(95% CI `[+0.00486, +0.00921]`, 20/20; sign-test p = 1.9e-6). Micro F1
(`+0.00715`), exact word segmentation (`+0.01339`), and bits/char
(`−0.01811`) all move in the same direction on 20/20 seeds. So the sign and the
statistical significance survive; **the published magnitude does not** — the
original 10-seed value is optimistic.

The dominant fact is corpus difficulty. Across the 20 seeds × 3 distinct modes,
**94.4% of macro-F1 variance is between-seed** (ICC 0.961), 3.1% is mode, and
2.5% is interaction. For the specific frequency-vs-morph contrast, **99.1% of
variance is between-seed, 0.63% is the treatment, and 0.26% is interaction**.
The absolute F1 SD across seeds (0.0440) is **9.5×** the paired-delta SD
(0.00465). A counterfactual unpaired 10-vs-10 design would have t ≈ 0.49; the
paired design has t = 6.52. At this sample size, the effect is only detectable
because the same seed supplies both methods.

Finally, **the split is interpolation, not compositional generalisation**:
46,072/46,080 test words (99.98%) have all three morpheme pairs seen in training
types; only 8 words in seed 43 lack a root–suffix pair. Exact test-word overlap
is zero, but pairwise overlap is near-total. The result is about held-out
*combinations* of seen morphemes and seen pairs, not unseen morphemes or unseen
pairs.

---

## 1. Reproduction and method

### 1.1 Exact reproduction of the published 10-seed rows

`validate_fast.py` retrains seed 42 and reproduces the published seed-42 row
exactly:

```text
flat          n_l1=490 n_l2=0 tok/ch=0.308828084667 bpc=2.353869480859 macroF1=0.866507523148
morph         n_l1=294 n_l2=196 tok/ch=0.309395973154 bpc=2.348420660223 macroF1=0.870630787037
random_closed n_l1=294 n_l2=196 tok/ch=0.311822405782 bpc=2.349558098165 macroF1=0.875723379630
```

The 20-seed run was compared row-by-row with
`results/level_assignment_v512_10seeds.csv` for seeds 42–51 and all four modes:

```text
rows compared=40; max abs diff {'boundary_f1': 0.0, 'bits_per_char': 0.0, 'tokens_per_char': 0.0}
PASS
```

The project's `collect_provenance` walks every one of the 120k training tokens.
A type-weighted reimplementation walks each realised type once and multiplies by
its token frequency. It was checked against the original on seed 42:

```text
orig provenance seconds 58.44 keys 489
fast provenance seconds 5.34 keys 489
stats identical: True
```

This made the 20-seed run feasible without touching source. The fast version is
an exact integer-count rewrite, not an approximation.

### 1.2 The 20-seed run

Command:

```powershell
python -u C:\Users\acese\AppData\Local\Temp\opencode\w1_02\run_20seeds.py
```

Six worker processes, one flat v512 tokenizer per seed, then `frequency`,
`morph`, and `random_closed` assignments with the project's own
`assign_levels`. Output:

```text
done seed=42 elapsed=286.6s
...
done seed=61 elapsed=1154.0s
wrote C:\Users\acese\AppData\Local\Temp\opencode\w1_02\seeds_v512.csv 80
wrote C:\Users\acese\AppData\Local\Temp\opencode\w1_02\corpus_stats.csv 20
total seconds 1155.8
```

80 rows = 20 seeds × 4 modes (`flat`, `frequency`, `morph`, `random_closed`).
`flat` and `frequency` are identical by construction and serve as a check.

### 1.3 Analysis commands

All analysis scripts are in
`C:\Users\acese\AppData\Local\Temp\opencode\w1_02\` and read only the two CSVs
above (or regenerate the corpus in memory for coverage checks). Exact commands:

```powershell
python -u C:\Users\acese\AppData\Local\Temp\opencode\w1_02\validate_fast.py
python -u C:\Users\acese\AppData\Local\Temp\opencode\w1_02\run_20seeds.py
python -u C:\Users\acese\AppData\Local\Temp\opencode\w1_02\analyze.py
python -u C:\Users\acese\AppData\Local\Temp\opencode\w1_02\type_pair_coverage.py
python -u C:\Users\acese\AppData\Local\Temp\opencode\w1_02\probe_random_order.py
python -u C:\Users\acese\AppData\Local\Temp\opencode\w1_02\extra.py
python -u C:\Users\acese\AppData\Local\Temp\opencode\w1_02\extra2.py
python -u C:\Users\acese\AppData\Local\Temp\opencode\w1_02\extra3.py
python -u C:\Users\acese\AppData\Local\Temp\opencode\w1_02\extra4.py
python -u C:\Users\acese\AppData\Local\Temp\opencode\w1_02\extra5.py
python -u C:\Users\acese\AppData\Local\Temp\opencode\w1_02\extra6.py
python -u C:\Users\acese\AppData\Local\Temp\opencode\w1_02\make_tables.py
```

Raw outputs: `seeds_v512.csv`, `corpus_stats.csv` in the same temp directory.
The tables below are generated directly from those CSVs.

---

## 2. Corpus characterization, seeds 42–61

### 2.1 Size and split

The generator is fixed-size; only the lexicon, the Zipf rank assignment, the
20% holdout, and the realised training sample change with the seed.

| quantity | value across 20 seeds |
|---|---:|
| train word types | 9,216 (constant) |
| test word types | 2,304 (constant) |
| train word tokens | 120,000 (constant) |
| training punctuation tokens | ≈8,400 (8,275–8,550) |
| test word tokens | 2,000 (constant) |
| realised training types | 7,405–7,517 (mean 7,453; 80.3–81.6% of types) |
| test text distinct types | 1,305–1,369 (mean 1,342; 56.6–59.4% of test types) |
| test types never drawn for BPC | 935–999 |
| test words per root (type level) | 7–33 (mean 19.2, sd 3.86) |

The test text is 2,000 draws **with replacement** over the 2,304 held-out
types. BPC and boundary F1 therefore do not see the same sample. F1 sees all
2,304 types; BPC sees 56.6–59.4% of them. This is the known M2 issue, quantified
here across seeds.

### 2.2 Root/prefix/suffix coverage

| coverage level | roots | prefixes | suffixes |
|---|---:|---:|---:|
| training types | 120/120 | 8/8 | 12/12 |
| realised training text | 120/120 | 8/8 | 12/12 |
| test types | 120/120 | 8/8 | 12/12 |
| test words missing a component in realised text | 0 | 0 | 0 |

Single-morpheme coverage is perfect. Pair coverage is not quite:

- **Type level:** 19/20 seeds have all three pairs (prefix–root, root–suffix,
  prefix–suffix) attested for all 2,304 test words. Seed 43 has 8/2,304 test
  words whose root–suffix pair is absent from training types.
  Across all seeds: **46,072/46,080 = 99.98%** of test words have all three
  pairs seen.
- **Realised-text level:** the root–suffix pair is absent from the realised
  120k-token sample for **40/46,080 test words (0.087%)** in 6/20 seeds:
  43 (8), 48 (5), 52 (6), 55 (5), 59 (6), 60 (10). Prefix–root and
  prefix–suffix pairs are never missing.
- The generator's attestation filter is type-level, not realised-sample-level
  (the known L2 issue). A rare root–suffix pair can be in `train_words` but not
  in the 120k sampled tokens.

The 40 realised-pair misses are too few to explain the morph effect: even if
every one of those words were scored completely wrong, the macro-F1 change is
bounded by 40/46,080 < 0.0009, while the observed effect is +0.0070.

### 2.3 Exact-form leakage

| check | result, all 20 seeds |
|---|---:|
| test **types** whose exact string is in train types | 0/2,304 |
| test **tokens** whose exact string is in train types | 0/2,000 |
| string collisions across the 11,520 generated words | 0 |
| test words missing a component in realised train text | 0 |
| test words appearing as a **substring** of some train word | 9.2–11.9% (mean 10.3%) |

There is **no exact test-in-train leakage** and no string collision. The ~10%
substring overlap is expected: test words share morphemes with train words, and
a test word can be a substring of a longer train word. It is not exact leakage,
but it is another sign that the split is interpolation rather than
compositional extrapolation.

### 2.4 Class balance

| class | min | max | mean | sd |
|---|---:|---:|---:|---:|
| test words per root | 7 | 33 | 19.2 | 3.86 |
| test words per prefix | 251 | 327 | 288 | — |
| test words per suffix | 163 | 220 | 192 | — |

Every root, prefix, and suffix appears in test (120/8/12 classes). The random
20% holdout gives each root 7–33 test words (observed; the expected count is
19.2, since the full grid has 96 words per root). Macro F1 averages per word, so
roots with 33 test words carry more weight than roots with 7. This affects
absolute F1, but the paired design uses the same test words for every mode, so
it does not bias the morph-vs-frequency contrast.

### 2.5 Boundary positions

Each word has two true boundaries: `b1 = len(prefix)` and
`b2 = len(prefix) + len(root)`. Prefixes are 2–3 characters; roots are 3–5;
suffixes are 2.

| boundary position | mean count / 2,304 | share |
|---|---:|---:|
| b1 = 2 | 1,164.3 | 50.5% |
| b1 = 3 | 1,139.7 | 49.5% |
| b2 = 5 | 283.2 | 12.3% |
| b2 = 6 | 566.6 | 24.6% |
| b2 = 7 | 875.4 | 38.0% |
| b2 = 8 | 578.9 | 25.1% |

The **joint** boundary class `(b1, b2)` has entropy 2.48 bits on average
(range 2.42–2.55) against a maximum of log2(6) = 2.585: the six joint classes
are close to balanced. The marginal `b2` distribution is not: position 7 is
three times more common than position 5. Since every word contributes one b1
and one b2, this is not a class-weighting problem for F1, but it means
per-position recall at position 5 is based on fewer words.

The morph gain is concentrated at the **first** boundary: recall at `b1`
improves by +0.0114 (19/20 seeds), recall at `b2` by +0.0033 (16/20, 4 ties).
By position: +0.0116 at 2, +0.0112 at 3, and +0.0022 to +0.0048 at 5–8.

### 2.6 Token frequency distribution

Training word-token distribution, 20 seeds:

| statistic | mean | min | max | sd |
|---|---:|---:|---:|---:|
| Shannon entropy (bits) | 8.243 | 8.222 | 8.260 | 0.011 |
| effective vocabulary | 303.0 | 298.7 | 306.5 | 2.30 |
| top 1% of types' share | 62.4% | 62.2% | 62.5% | 0.11 pp |
| top 10 types' share | 40.8% | 40.6% | 41.1% | 0.14 pp |
| Gini | 0.8568 | 0.8550 | 0.8580 | 0.0006 |
| max count | 18,274 | 17,985 | 18,515 | 124 |
| singleton types | 2,176 | 2,057 | 2,251 | 43 |
| fitted Zipf slope (ranks 1–1000) | −1.089 | −1.096 | −1.081 | 0.004 |

The realised distribution is very stable across seeds. The generator's exponent
1.1 appears as a fitted slope near −1.09. The top 1% of types holds ~62% of
tokens; the distribution is heavy-tailed in every seed.

Test text: 2,000 draws, 1,305–1,369 distinct types, max repeat 4–7, 779–879
singletons, 935–999 types never drawn. This is a uniform-over-types stress set,
not the Zipfian training distribution.

### 2.7 What actually varies, and what does not

Aggregate distribution statistics are nearly seed-invariant, yet flat F1 ranges
**0.779–0.926** (mean 0.863, sd 0.044, range 0.146). Correlation of flat F1
with corpus statistics (Spearman, 20 seeds; full list in §7):

| feature | rho with flat F1 | p |
|---|---:|---:|
| Zipf slope (less negative = flatter) | +0.471 | 0.036 |
| Gini | −0.457 | 0.043 |
| train token entropy | +0.349 | 0.132 |
| top 10 share | −0.344 | 0.137 |
| test type coverage | +0.109 | 0.647 |
| roots ending with a suffix | −0.033 | 0.889 |
| test words missing root–suffix pair in realised text | −0.105 | 0.660 |

No aggregate feature explains more than ~22% of flat-F1 variance. The only
moderate axis is frequency inequality: flatter training distributions are
slightly easier. The large seed swings are mostly **idiosyncratic**: which
roots/words received the high Zipf ranks, and which triples landed in the 20%
holdout. This is why pairing is not optional — no simple corpus statistic can
be used to match corpora post hoc.

### 2.8 Per-seed corpus table

| seed | realized types | train cov | test distinct | test cov | rs miss | train H | Gini | top1% | zipf | L1 cross | roots end-sfx |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 42 | 7478 | 0.811 | 1324 | 0.575 | 0 | 8.251 | 0.857 | 0.623 | −1.084 | 43 (14.6%) | 18 |
| 43 | 7482 | 0.812 | 1342 | 0.582 | 8 | 8.229 | 0.858 | 0.625 | −1.096 | 38 (12.9%) | 15 |
| 44 | 7483 | 0.812 | 1339 | 0.581 | 0 | 8.250 | 0.857 | 0.624 | −1.082 | 15 (5.1%) | 14 |
| 45 | 7405 | 0.803 | 1317 | 0.572 | 0 | 8.254 | 0.855 | 0.622 | −1.091 | 26 (8.8%) | 20 |
| 46 | 7493 | 0.813 | 1361 | 0.591 | 0 | 8.244 | 0.857 | 0.623 | −1.092 | 60 (20.4%) | 15 |
| 47 | 7440 | 0.807 | 1341 | 0.582 | 0 | 8.246 | 0.857 | 0.624 | −1.096 | 24 (8.2%) | 11 |
| 48 | 7455 | 0.809 | 1352 | 0.587 | 5 | 8.257 | 0.857 | 0.622 | −1.081 | 21 (7.1%) | 20 |
| 49 | 7431 | 0.806 | 1349 | 0.586 | 0 | 8.243 | 0.856 | 0.624 | −1.088 | 28 (9.5%) | 20 |
| 50 | 7517 | 0.816 | 1348 | 0.585 | 0 | 8.254 | 0.857 | 0.624 | −1.086 | 25 (8.5%) | 12 |
| 51 | 7457 | 0.809 | 1368 | 0.594 | 0 | 8.245 | 0.857 | 0.623 | −1.089 | 23 (7.8%) | 17 |
| 52 | 7481 | 0.812 | 1357 | 0.589 | 6 | 8.233 | 0.857 | 0.625 | −1.088 | 36 (12.2%) | 10 |
| 53 | 7432 | 0.806 | 1305 | 0.566 | 0 | 8.239 | 0.857 | 0.623 | −1.087 | 55 (18.7%) | 26 |
| 54 | 7448 | 0.808 | 1369 | 0.594 | 0 | 8.222 | 0.858 | 0.625 | −1.089 | 24 (8.2%) | 12 |
| 55 | 7426 | 0.806 | 1332 | 0.578 | 5 | 8.230 | 0.857 | 0.625 | −1.090 | 63 (21.4%) | 17 |
| 56 | 7439 | 0.807 | 1326 | 0.576 | 0 | 8.235 | 0.857 | 0.625 | −1.094 | 36 (12.2%) | 22 |
| 57 | 7454 | 0.809 | 1346 | 0.584 | 0 | 8.260 | 0.857 | 0.622 | −1.088 | 28 (9.5%) | 16 |
| 58 | 7468 | 0.810 | 1352 | 0.587 | 0 | 8.245 | 0.857 | 0.624 | −1.091 | 32 (10.9%) | 18 |
| 59 | 7448 | 0.808 | 1323 | 0.574 | 6 | 8.256 | 0.856 | 0.623 | −1.084 | 29 (9.9%) | 18 |
| 60 | 7405 | 0.803 | 1331 | 0.578 | 10 | 8.243 | 0.856 | 0.624 | −1.090 | 20 (6.8%) | 15 |
| 61 | 7413 | 0.804 | 1351 | 0.586 | 0 | 8.225 | 0.857 | 0.625 | −1.090 | 22 (7.5%) | 11 |

`L1 cross` is the number of cross-majority rules forced into the morph level-1
budget of 294; it becomes the key moderator in §3.3.

---

## 3. How much of the seed variance is corpus difficulty?

### 3.1 Variance decomposition

Balanced design: 20 seeds × 3 distinct modes (`frequency`, `morph`,
`random_closed`). `frequency` equals `flat`. Two-way decomposition with seed and
mode main effects plus interaction:

| metric | seed SD | seed range | eta² seed | eta² mode | eta² interaction | ICC (seed) |
|---|---:|---:|---:|---:|---:|---:|
| macro boundary F1 | 0.04514 | 0.15050 | **0.9437** | 0.0309 | 0.0254 | **0.9607** |
| micro boundary F1 | 0.04524 | 0.15063 | **0.9125** | 0.0500 | 0.0375 | **0.9407** |
| exact word match | 0.08113 | 0.28559 | **0.9535** | 0.0290 | 0.0175 | **0.9730** |
| bits per char | 0.06259 | 0.22683 | **0.7185** | 0.1662 | 0.1152 | **0.7927** |

For macro F1, **94.4% of the observed variance is between-seed corpus
difficulty**, 3.1% is mode, and 2.5% is seed×mode interaction. The variance
component estimate agrees: ICC = 0.961. Micro F1 and exact match tell the same
story. BPC is noisier (seed 71.9%, mode 16.6%, interaction 11.5%) because the
mode spread is larger relative to the seed spread, but seed still dominates.

For the **specific** frequency-vs-morph contrast (2 conditions, 20 seeds,
balanced), the decomposition is starker:

```text
ss_total=0.078686  ss_seed=0.077986  ss_treat=0.00049540  ss_resid=0.000205
eta2_seed=0.9911   eta2_treat=0.0063   eta2_resid=0.0026
partial eta2 treatment = 0.7070
treatment variance as fraction of seed variance: 0.0064
```

The morph treatment accounts for **0.63% of total variance** and **0.64% of the
between-seed variance**. It is highly consistent after pairing (partial
eta² = 0.707), but tiny relative to the corpus.

### 3.2 Pairing is doing all the work

| group | flat F1 SD | paired ΔF1 SD | variance ratio | SD ratio |
|---|---:|---:|---:|---:|
| original 10 (42–51) | 0.04203 | 0.004484 | 87.8× | 9.4× |
| fresh 10 (52–61) | 0.04517 | 0.003834 | 138.8× | 11.8× |
| all 20 | 0.04396 | 0.004648 | 89.4× | 9.5× |

Counterfactual: an unpaired 10-vs-10 comparison with the original flat SD and
the observed +0.00924 effect would have **t ≈ 0.49** (not significant), versus
the paired **t = 6.52**. With 20 seeds, the unpaired t would still be ~0.51.
This is the single most important methodological fact in the report: the result
is only detectable because the same seed supplies both methods. Any table that
compares methods across different seeds, or reports absolute F1 without a
paired contrast, cannot support the claim.

### 3.3 The morph gain is corpus-conditional

The effect is not constant across seeds. It is larger on easier corpora and
collapses when the level-1 budget must be filled with cross-majority rules.

**Baseline difficulty.** OLS of the morph gain on flat F1:

```text
dMF = -0.043960 + 0.059085 * flatF1
r = 0.559, r2 = 0.312, p = 0.0104, se(slope) = 0.0207
predicted at flat F1 = 0.78: +0.00213
predicted at flat F1 = 0.93: +0.01099
```

| quartile of flat F1 | flat F1 mean | dMF mean | dRF mean |
|---|---:|---:|---:|
| Q1 (hardest) | 0.7976 | +0.00342 | −0.00902 |
| Q2 | 0.8606 | +0.00511 | −0.02144 |
| Q3 | 0.8866 | +0.01046 | −0.00669 |
| Q4 (easiest) | 0.9077 | +0.00916 | −0.01177 |

Hard half (flat F1 ≤ median): +0.00427, 10/10 wins. Easy half: +0.00981,
10/10 wins. The effect is positive in both halves; it is simply ~2.3× larger on
the easy half. The random control shows no such gradient (Spearman rho = 0.045,
p = 0.85), so this is specific to the morph treatment.

**Rule-level purity.** The mechanism is visible in the level composition. The
morph level 1 always has 294 rules. Of these, 230–278 are intra-majority,
15–63 are cross-majority (5.1–21.4%, mean 11.0%), and one has no provenance
statistics (the `l1_unseen=1` rule; §L4). All intra-majority rules are in level
1; level 2 has **zero** intra-majority rules. So the assignment is "all intra +
enough least-cross cross rules to fill the budget", not a pure intra/cross
partition.

The number of cross-majority rules in L1 predicts the morph gain:

```text
dMF = 0.014337 - 0.066224 * cross_share
r2 = 0.4263, p = 0.001804
predicted at cross_share = 0.05: +0.01103
predicted at cross_share = 0.21: +0.00043
```

Spearman(cross share, dMF) = **−0.694** (p = 0.0007);
Spearman(cross share, flat F1) = **−0.755** (p = 0.0001). In a two-predictor
model with flat F1, the cross-share coefficient is −0.066 (p = 0.084) and flat
F1 adds nothing (coefficient −0.0002, p = 0.995). The corpus's rule-level
morphological separability, not generic baseline difficulty, is the moderator.

Treatment dose (extra intra-majority rules in morph L1 vs frequency L1, 33–78
rules) also correlates with the gain (rho = +0.579, p = 0.0074), but less
strongly than the absolute number of intra-majority rules (rho = +0.694).

**Interpretation.** The v512 effect is largest when the flat BPE rule set has a
large clean intra-morpheme subset. When only ~230 of 490 rules are
intra-majority, L1 must include ~63 cross rules; the morph ordering is diluted
and the gain is only +0.0001 to +0.0029 (seeds 46, 53, 55). This is a more
precise version of the "budget window": it is a **purity window**, not just a
vocabulary-size window. It also predicts that a real corpus with noisier
morphology could show little or no gain.

---

## 4. Does the v512 morph result survive?

### 4.1 Paired effects

Macro boundary F1, morph − frequency:

| group | n | mean | SD | t | p | 95% CI | W/L/T | sign p | Wilcoxon p | dz |
|---|---:|---:|---:|---:|---:|---|---:|---:|---:|---:|
| original 10 (42–51) | 10 | +0.009243 | 0.004484 | +6.52 | 1.09e-4 | [+0.006035, +0.012451] | 10/0/0 | 0.00195 | 0.00195 | +2.06 |
| fresh 10 (52–61) | 10 | +0.004834 | 0.003834 | +3.99 | 0.00317 | [+0.002091, +0.007576] | 10/0/0 | 0.00195 | 0.00195 | +1.26 |
| all 20 | 20 | **+0.007038** | 0.004648 | **+6.77** | **1.81e-6** | **[+0.004863, +0.009214]** | **20/0/0** | **1.91e-6** | **1.91e-6** | +1.51 |

Bootstrap 95% CI for all 20: `[+0.005083, +0.009062]`. Leave-one-out mean range:
`[+0.006588, +0.007401]` — no single seed drives the all-20 result.

Other metrics, all 20 seeds, morph − frequency:

| metric | mean | t | p | 95% CI | direction |
|---|---:|---:|---:|---|---|
| micro boundary F1 | +0.007150 | +6.63 | 2.44e-6 | [+0.004891, +0.009409] | 20/20 better |
| exact word segmentation | +0.013390 | +6.44 | 3.60e-6 | [+0.009035, +0.017744] | 20/20 better |
| bits per char | −0.018109 | −5.18 | 5.31e-5 | [−0.025423, −0.010794] | 20/20 lower |
| tokens per char | +0.000114 | +1.29 | 0.214 | [−0.000072, +0.000299] | 10/5, ns |
| boundary precision (macro) | +0.006438 | +6.59 | 2.60e-6 | [+0.004395, +0.008481] | 19 wins / 0 losses / 1 tie |
| boundary recall (macro) | +0.007335 | +6.95 | 1.28e-6 | [+0.005125, +0.009545] | 20/20 better |
| mean symmetric boundary error | −0.028277 | −6.57 | 2.72e-6 | [−0.037282, −0.019272] | 20/20 lower |
| under-segmentation rate | −0.001910 | −4.61 | 1.93e-4 | [−0.002778, −0.001042] | 14 wins / 0 losses / 6 ties |
| over-segmentation rate | −0.001063 | −1.73 | 0.100 | [−0.002352, +0.000226] | 1 win / 9 losses / 10 ties (ns) |
| recall of first boundary | +0.011393 | +6.20 | 5.94e-6 | [+0.007544, +0.015242] | 19 wins / 1 loss |
| recall of second boundary | +0.003277 | +5.21 | 4.95e-5 | [+0.001961, +0.004592] | 16 wins / 0 losses / 4 ties |

The morph advantage is **not a macro-averaging artifact**: micro F1 moves the
same way (r = 0.998 between the macro and micro deltas), exact word match
improves, and the mean symmetric boundary error falls. It is **not a
token-count artifact**: morph uses slightly more tokens (not fewer), yet bpc is
lower. It is **not driven by over-segmentation**: morph mainly reduces
under-segmentation and recovers the first boundary.

BPC and F1 deltas correlate at r = −0.950 across seeds: seeds where morph
improves alignment also improve compression. Despite BPC and F1 using different
test samples, they do not tell contradictory stories for this treatment.

### 4.2 Random control

| comparison | original 10 | fresh 10 | all 20 |
|---|---:|---:|---:|
| random_closed − frequency, macro F1 | −0.006386 (t −1.73, 3/7) | −0.018074 (t −3.47, 1/9) | −0.012230 (t −3.61, 4/16) |
| morph − random_closed, macro F1 | +0.015629 (t 3.81, 9/1) | +0.022907 (t 4.09, 9/1) | +0.019268 (t 5.54, 18/2) |
| random_closed − frequency, bpc | +0.036144 (t 3.04, 8/2 worse) | +0.066228 (t 3.44, 9/1 worse) | +0.051186 (t 4.44, 17/3 worse) |
| morph − random_closed, bpc | −0.061932 (t −4.73, 10/0) | −0.076658 (t −3.85, 9/1) | −0.069295 (t −5.91, 19/1) |

The random control is worse than frequency on the fresh seeds and much worse
than morph. The morph-vs-random gap is larger on the new seeds than on the
original 10. Caveat: `random_closed` is not exactly size-matched in 10/20 seeds
(§5, M1); the morph-vs-frequency headline is unaffected.

### 4.3 Did the original effect replicate?

Yes in sign and significance; no in magnitude.

- Original 10: +0.009243 (10/10, CI +0.0060..+0.0125).
- Fresh 10: +0.004834 (10/10, CI +0.0021..+0.0076).
- All 20: +0.007038 (20/20, CI +0.0049..+0.0092).

The difference between the original and fresh means is +0.004410; Welch
t = 2.364, p = 0.0299; a 200k permutation test gives p = 0.0317. The fresh 10
are harder on average (flat F1 mean 0.8520 vs 0.8742). Controlling for flat F1,
the "fresh" coefficient is −0.003322 (se 0.001706, t = −1.95, p = 0.068), and
flat F1 is +0.049056 (p = 0.0247). So difficulty composition explains part of
the gap, but not all of it; the original 10-seed magnitude is probably
optimistic by ~25–50%. Note that the original point estimate +0.009243 lies
just outside the all-20 95% CI, whose upper bound is +0.009214.

**Best estimate for the paper:** `+0.0070` macro F1 (95% CI `+0.0049` to
`+0.0092`), 20/20 seeds, with the explicit caveat that the gain rises from
~+0.004 on hard corpora to ~+0.010 on easy ones and collapses when L1 must be
filled with many cross-majority rules. The old sentence "+0.9 points" should
become "+0.7 points (95% CI +0.5 to +0.9), and the effect is conditional on
corpus difficulty / rule purity."

**Independence caveat.** Seeds 52–61 are fresh corpora, but v512 was selected
using 42–51 (the effect is absent at 1024/2048). The replication therefore tests
the effect at the selected vocabulary, not the selection of the vocabulary. A
pre-registered v512 run, fixed before either set was seen, would be stronger
evidence for the "window" claim.

### 4.4 Per-seed evidence

| seed | flat F1 | freq F1 | morph F1 | random F1 | dMF | dRF | dbpc | dMicro | dExact |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 42 | 0.8665 | 0.8665 | 0.8706 | 0.8757 | +0.00412 | +0.00922 | −0.00545 | +0.00409 | +0.00868 |
| 43 | 0.8011 | 0.8011 | 0.8108 | 0.8017 | +0.00972 | +0.00062 | −0.02414 | +0.01060 | +0.01693 |
| 44 | 0.9142 | 0.9142 | 0.9238 | 0.8884 | +0.00964 | −0.02584 | −0.02989 | +0.01002 | +0.01823 |
| 45 | 0.8935 | 0.8935 | 0.9035 | 0.8916 | +0.00998 | −0.00190 | −0.02368 | +0.00994 | +0.02083 |
| 46 | 0.8029 | 0.8029 | 0.8030 | 0.7949 | +0.00014 | −0.00801 | −0.00100 | +0.00013 | +0.00043 |
| 47 | 0.8691 | 0.8691 | 0.8782 | 0.8421 | +0.00914 | −0.02694 | −0.02481 | +0.00923 | +0.01780 |
| 48 | 0.8943 | 0.8943 | 0.9099 | 0.8854 | +0.01560 | −0.00896 | −0.04944 | +0.01650 | +0.02908 |
| 49 | 0.9255 | 0.9255 | 0.9401 | 0.9234 | +0.01454 | −0.00218 | −0.03447 | +0.01495 | +0.02821 |
| 50 | 0.8902 | 0.8902 | 0.9010 | 0.8892 | +0.01079 | −0.00103 | −0.04448 | +0.01094 | +0.02214 |
| 51 | 0.8848 | 0.8848 | 0.8935 | 0.8859 | +0.00875 | +0.00116 | −0.02053 | +0.00844 | +0.01953 |
| 52 | 0.8220 | 0.8220 | 0.8241 | 0.7931 | +0.00211 | −0.02895 | −0.00191 | +0.00198 | +0.00347 |
| 53 | 0.7792 | 0.7792 | 0.7821 | 0.7763 | +0.00289 | −0.00289 | −0.00328 | +0.00307 | +0.00434 |
| 54 | 0.9039 | 0.9039 | 0.9098 | 0.8749 | +0.00583 | −0.02903 | −0.01537 | +0.00574 | +0.01172 |
| 55 | 0.7831 | 0.7831 | 0.7853 | 0.7772 | +0.00221 | −0.00587 | −0.00146 | +0.00242 | +0.00217 |
| 56 | 0.8441 | 0.8441 | 0.8491 | 0.7982 | +0.00496 | −0.04598 | −0.00407 | +0.00445 | +0.00651 |
| 57 | 0.8660 | 0.8660 | 0.8699 | 0.8578 | +0.00391 | −0.00819 | −0.00481 | +0.00377 | +0.00738 |
| 58 | 0.8572 | 0.8572 | 0.8606 | 0.8219 | +0.00343 | −0.03531 | −0.00843 | +0.00365 | +0.00608 |
| 59 | 0.9006 | 0.9006 | 0.9008 | 0.9078 | +0.00022 | +0.00719 | −0.00024 | +0.00022 | +0.00043 |
| 60 | 0.8932 | 0.8932 | 0.9033 | 0.8791 | +0.01011 | −0.01407 | −0.02882 | +0.01007 | +0.02040 |
| 61 | 0.8712 | 0.8712 | 0.8839 | 0.8536 | +0.01266 | −0.01764 | −0.03592 | +0.01281 | +0.02344 |

The near-zero seeds (46 and 59) and the small-effect seeds (52, 53, 55) are
mostly the seeds where L1 cross contamination is high (46: 20.4%, 53: 18.7%,
55: 21.4%) or the corpus is hard (52: 0.822, 53: 0.779, 55: 0.783). Seed 59 is
the exception: easy (0.9006) but almost no morph gain (+0.00022), and
`random_closed` beats morph there (+0.00719). The mechanism is strong but not
deterministic; there is residual seed heterogeneity.

---

## 5. Findings by severity

Severity scale as in `docs/reviews/data-review.md`: **Critical** = headline
number invalid or stated property not measured; **High** = can change a
scientific conclusion or leaves the central claim untested; **Medium** =
materially biases values, power, or reproducibility; **Low** = robustness or
hygiene.

| ID | Severity | Finding | Evidence |
|---|---|---|---|
| H1 | **High** | Corpus difficulty dominates boundary-F1 variance: 94.4% between-seed (ICC 0.961), 3.1% mode, 2.5% interaction. Absolute SD 0.0440 vs paired-delta SD 0.00465 (9.5×). A counterfactual unpaired 10-vs-10 design has t≈0.49; the paired design has t=6.52. At this sample size, only paired, same-seed contrasts are interpretable. | §3.1–3.2; flat F1 range 0.779–0.926 |
| H2 | **High** | The published 10-seed magnitude is optimistic. Fresh seeds 52–61 give +0.00483 (10/10, CI +0.0021..+0.0076) vs +0.00924 on 42–51; Welch p=0.030. All-20 estimate +0.00704 (CI +0.0049..+0.0092, 20/20). Sign and significance survive; magnitude shrinks ~25–50%. | §4.1, §4.3 |
| H3 | **High** | The morph gain is corpus-conditional and collapses with L1 cross-rule contamination. `dMF = 0.01434 − 0.06622 × cross_share` (R²=0.426, p=0.0018); predicted +0.0110 at 5% contamination, +0.0004 at 21%. Cross share beats flat F1 in a two-predictor model (flat F1 p=0.995). Hard half +0.0043 vs easy half +0.0098; both 10/10. | §3.3, §4.4 |
| H4 | **High** | The split is interpolation, not compositional generalisation. 46,072/46,080 test words (99.98%) have all three morpheme pairs attested in training types; only 8 words in seed 43 lack a root–suffix pair, and no test word has an unseen morpheme. Exact word overlap is 0, but pair overlap is near-total. The result is about unseen combinations of seen morphemes/pairs, not compositional extrapolation. | §2.2–2.3; `type_pair_coverage.py` |
| M1 | **Medium** | `random_closed` is not exactly size-matched in 10/20 seeds: `n_l1` = 291–293 instead of 294. Cause: the topological order can contain duplicate rule IDs when a rule's two child symbols were created by the same rule (e.g. `ku+ku`, `ip+ip`, `kuku+g`). Seed 58 probe: order length 493, distinct 490, 3 duplicate IDs, unique first 294 = 291. The bug is in `ramified/levels.py:140-146`. It weakens the random control, not the morph–frequency headline; the deficit correlates only weakly with `dRF` (rho −0.20, p=0.40). | §5.1; probe output |
| M2 | **Medium** | Morph level 1 is not purely intra-morpheme: 15–63 of 294 rules (5.1–21.4%, mean 11.0%) are cross-majority, and one has no provenance stats. All intra-majority rules (230–278) are in L1; L2 has zero intra. The paper's "intra→L1, cross→L2" is an oversimplification; the real rule is "all intra + least-cross cross rules to fill the 60% budget". This is also the H3 moderator. | §3.3, §4.4 |
| M3 | **Medium** | BPC and boundary F1 use different test samples: BPC sees 1,305–1,369 of 2,304 held-out types (56.6–59.4%); 935–999 types are never drawn. F1 uses all 2,304. The deltas correlate −0.95 across seeds, so the mismatch does not reverse the morph conclusion, but absolute BPC comparisons are noisy and not on the same sample as F1. | §2.1, §4.1 |
| M4 | **Medium** | Token-level pair coverage hole: the root–suffix pair is absent from the realised 120k training text for 40/46,080 test words (0.087%) in 6/20 seeds (max 10 in seed 60); at type level, 8/2,304 in seed 43. The generator's attestation filter is type-level, not realised-sample-level. Even total failure on these words changes macro F1 by <0.0009, so it cannot explain the +0.007 effect, but it should be reported and fixed. | §2.2 |
| L1 | **Low** | Exact test-in-train leakage is zero: 0/2,304 types, 0/2,000 tokens, 0 string collisions in all 20 seeds. Test words appearing as substrings of train words are 9.2–11.9% (mean 10.3%), expected from shared morphemes; this is not exact leakage but confirms interpolation. | §2.3 |
| L2 | **Low** | Class balance is uneven: test words per root 7–33 (mean 19.2, sd 3.86), per prefix 251–327, per suffix 163–220. Macro F1 weights roots unevenly. The joint boundary class `(b1,b2)` is near-uniform (entropy 2.48/2.585), though marginal B2 positions are imbalanced. The paired design cancels class imbalance for method comparisons. | §2.4–2.5 |
| L3 | **Low** | Aggregate token-frequency statistics are almost seed-invariant (entropy 8.22–8.26, Gini 0.855–0.858, top1% 62.2–62.5%), yet flat F1 ranges 0.779–0.926. No aggregate feature explains >22% of flat-F1 variance; difficulty is mostly idiosyncratic rank/holdout assignment. | §2.6–2.7 |
| L4 | **Low** | One learned rule per seed has no provenance stats (`l1_unseen=1`) and can land in L1 because unseen rules score 0.5. Morph and frequency L1 both have 294 rules in all 20 seeds; the unseen rule is a minor accounting edge, not a driver. | §4.4; CSV `l1_unseen` |
| L5 | **Low** | The reported macro precision is inflated by empty predictions (`boundary_scores` returns precision 1.0 when `pred` is empty), a known data-review M1 issue. Macro F1 is 0 for such words, so the F1 headline is not inflated; the precision column should not be read literally. Micro F1 and exact match confirm the morph effect. | §4.1; data-review M1 |

### 5.1 The `random_closed` size bug, exact probe

Command:

```powershell
python -u C:\Users\acese\AppData\Local\Temp\opencode\w1_02\probe_random_order.py
```

Output for seed 58 (the worst case, `n_l1=291` in the CSV):

```text
rules 490 requested n1 294
order length 493 distinct 490
duplicate rule ids 3 [(357, 2), (328, 2), (369, 2)]
unique rules in order[:n1] 291
dup rule 357 expansions 'ku' + 'ku' -> 'kuku' deps {92} creator(left) 92 creator(right) 92
dup rule 328 expansions 'ip' + 'ip' -> 'ipip' deps {167} creator(left) 167 creator(right) 167
dup rule 369 expansions 'kuku' + 'g' -> 'kukug' deps {357} creator(left) 357 creator(right) None
```

Mechanism: in `assign_levels` (`ramified/levels.py:140-146`), when a rule's left
and right symbols were created by the same earlier rule, the child is appended
to `available` twice; the loop appends it to `order` twice. `order[:n1]` then
contains fewer than `n1` distinct rules, so the L1 set is smaller. The fix is
to skip a child that is already in `emitted` **or already in `available`**, or
to build `order` from a set. This is a control bug, not a morph bug.

Affected seeds and deficits: 44, 46, 52, 53, 55, 56, 59 (deficit 1); 50, 57
(deficit 2); 58 (deficit 3). Ten of twenty seeds. The correlation between
deficit and `dRF` is rho = −0.201 (p = 0.395), so the bug is not the main reason
the random control loses; it should still be fixed before the paper.

---

## 6. What would change my mind

In descending order of importance:

1. **Pair-disjoint or zero-shot split.** Build a split that holds out entire
   roots, or entire root–suffix pairs, so test words contain an unseen morpheme
   or an unseen pair. Run the same 20-seed paired protocol. If morph's advantage
   disappears there, the current result is interpolation-specific and the
   compositional claim fails. This is the single check that would most change
   the verdict. The current split has 99.98% pair coverage, so it cannot answer
   this question.
2. **Real morphological corpus with verified boundaries.** The synthetic
   generator is controlled but artificial. A real corpus with a comparable
   baseline F1 and a paired morph/frequency/random comparison is the external
   validity test. If `dMF ≤ 0` or the CI includes 0, the result does not
   generalise.
3. **Adaptive L1 purity.** On high-cross seeds (46, 53, 55), set the L1 budget
   to the number of intra-majority rules instead of a fixed 60%, and compare
   against a frequency control with the same adaptive L1 size. If the gain
   returns when L1 is pure intra, the limitation is the fixed budget; if it
   still vanishes, the corpus is genuinely less amenable. This would turn H3
   from an observational moderator into a causal test.
4. **Corrected random control.** Fix the duplicate topological order, rerun the
   20 seeds, and confirm `morph − random_closed` remains positive. The
   morph–frequency result is unaffected, but the "not any reordering" claim
   should rest on a size-matched control.
5. **A harder corpus (flat F1 < 0.80).** The regression predicts `dMF ≈ +0.002`
   at flat F1 0.78. If the effect is absent there, the difficulty interaction is
   confirmed and the claim narrows to easier corpora. If it is still ~+0.007,
   the interaction is not causal.
6. **Common-test-set / resampled-training experiment.** Hold the lexicon and
   holdout fixed, resample only the 120k training tokens, and retrain. This
   would separate training-sample difficulty from test-set composition. The
   current "corpus difficulty" is the joint seed-specific train/test
   realisation; a common test set would make the label more precise.
7. **Pre-registered replication at v512 and at v1024/2048.** The fresh seeds
   confirm v512, but they do not test the "window" claim. If the v512 effect is
   a selection artifact, other vocabularies should show similar effects on fresh
   seeds; if the window is real, they should not. This is the test of the
   paper's conditional claim, not of the v512 sign.

**What would not change my mind:** a larger unpaired table, more absolute F1
numbers on different seeds, or a different macro/micro convention. The paired
contrast is already the right design; the open questions are external validity
and the purity moderator, not the arithmetic.

---

## 7. Appendix: exact outputs

### 7.1 Validation

```text
rows compared=40; max abs diff {'boundary_f1': 0.0, 'bits_per_char': 0.0, 'tokens_per_char': 0.0}
PASS
```

Fast provenance equivalence (seed 42):

```text
orig provenance seconds 58.44 keys 489
fast provenance seconds 5.34 keys 489
stats identical: True
```

### 7.2 Main paired effects (selected exact output from `analyze.py`)

The macro-F1 blocks are complete; the micro/exact/bpc blocks are truncated for
space after the fields needed for this report.

```text
macroF1 morph-frequency
  orig10 n=10 mean=+0.009243 sd=0.004484 t=+6.52 p=0.0001091 CI95=[+0.006035,+0.012451] boot=[+0.006476,+0.011722] W/L/T=10/0/0 sign_p=0.001953 wilcoxon_p=0.001953 dz=+2.06
  new10  n=10 mean=+0.004834 sd=0.003834 t=+3.99 p=0.003174 CI95=[+0.002091,+0.007576] boot=[+0.002782,+0.007195] W/L/T=10/0/0 sign_p=0.001953 wilcoxon_p=0.001953 dz=+1.26
  all20  n=20 mean=+0.007038 sd=0.004648 t=+6.77 p=1.813e-06 CI95=[+0.004863,+0.009214] boot=[+0.005083,+0.009062] W/L/T=20/0/0 sign_p=1.907e-06 wilcoxon_p=1.907e-06 dz=+1.51

macroF1 random-frequency
  orig10 n=10 mean=-0.006386 sd=0.011672 t=-1.73 p=0.1177 CI95=[-0.014735,+0.001964] boot=[-0.013590,-0.000039] W/L/T=3/7/0 sign_p=0.3438 wilcoxon_p=0.1309 dz=-0.55
  new10  n=10 mean=-0.018074 sd=0.016484 t=-3.47 p=0.00708 CI95=[-0.029866,-0.006281] boot=[-0.027928,-0.008434] W/L/T=1/9/0 sign_p=0.02148 wilcoxon_p=0.009766 dz=-1.10
  all20  n=20 mean=-0.012230 sd=0.015139 t=-3.61 p=0.001854 CI95=[-0.019315,-0.005144] boot=[-0.018890,-0.006016] W/L/T=4/16/0 sign_p=0.01182 wilcoxon_p=0.001432 dz=-0.81

macroF1 morph-random
  orig10 n=10 mean=+0.015629 sd=0.012970 t=+3.81 p=0.00415 CI95=[+0.006351,+0.024907] boot=[+0.008140,+0.023383] W/L/T=9/1/0 sign_p=0.02148 wilcoxon_p=0.003906 dz=+1.20
  new10  n=10 mean=+0.022907 sd=0.017716 t=+4.09 p=0.002721 CI95=[+0.010234,+0.035580] boot=[+0.012231,+0.033049] W/L/T=9/1/0 sign_p=0.02148 wilcoxon_p=0.005859 dz=+1.29
  all20  n=20 mean=+0.019268 sd=0.015566 t=+5.54 p=2.437e-05 CI95=[+0.011983,+0.026553] boot=[+0.012711,+0.026071] W/L/T=18/2/0 sign_p=0.0004025 wilcoxon_p=1.335e-05 dz=+1.24

microF1 morph-frequency
  orig10 n=10 mean=+0.009484 sd=0.004722 t=+6.35 p=0.0001326 CI95=[+0.006106,+0.012862] W/L/T=10/0/0
  new10  n=10 mean=+0.004816 sd=0.003844 t=+3.96 p=0.003295 CI95=[+0.002066,+0.007566] W/L/T=10/0/0
  all20  n=20 mean=+0.007150 sd=0.004827 t=+6.63 p=2.442e-06 CI95=[+0.004891,+0.009409] W/L/T=20/0/0

exactWord morph-frequency
  orig10 n=10 mean=+0.018186 sd=0.008504 t=+6.76 p=8.246e-05 CI95=[+0.012102,+0.024269] W/L/T=10/0/0
  new10  n=10 mean=+0.008594 sd=0.007701 t=+3.53 p=0.006427 CI95=[+0.003085,+0.014103] W/L/T=10/0/0
  all20  n=20 mean=+0.013390 sd=0.009304 t=+6.44 p=3.601e-06 CI95=[+0.009035,+0.017744] W/L/T=20/0/0

bpc morph-frequency
  orig10 n=10 mean=-0.025788 sd=0.015148 t=-5.38 p=0.0004425 CI95=[-0.036624,-0.014952] W/L/T=0/10/0
  new10  n=10 mean=-0.010430 sd=0.012456 t=-2.65 p=0.02657 CI95=[-0.019341,-0.001519] W/L/T=0/10/0
  all20  n=20 mean=-0.018109 sd=0.015629 t=-5.18 p=5.305e-05 CI95=[-0.025423,-0.010794] W/L/T=0/20/0
```

### 7.3 Variance decomposition (verbatim)

```text
boundary_f1    seed_sd=0.045140 range=0.150502 eta2_seed=0.9437 eta2_mode=0.0309 eta2_resid=0.0254 ICC=0.9607 sigma_seed=0.04484 sigma_resid=0.00907
micro_f1       seed_sd=0.045239 range=0.150628 eta2_seed=0.9125 eta2_mode=0.0500 eta2_resid=0.0375 ICC=0.9407 sigma_seed=0.04477 sigma_resid=0.01124
exact_word     seed_sd=0.081132 range=0.285590 eta2_seed=0.9535 eta2_mode=0.0290 eta2_resid=0.0175 ICC=0.9730 sigma_seed=0.08076 sigma_resid=0.01346
bits_per_char  seed_sd=0.062593 range=0.226826 eta2_seed=0.7185 eta2_mode=0.1662 eta2_resid=0.1152 ICC=0.7927 sigma_seed=0.06003 sigma_resid=0.03070

Absolute vs paired noise (macro F1):
  orig10 flat_sd=0.042031 flat_range=0.124465 delta_sd=0.004484 var_ratio=87.8x sd_ratio=9.4x
  new10  flat_sd=0.045170 flat_range=0.124783 delta_sd=0.003834 var_ratio=138.8x sd_ratio=11.8x
  all20  flat_sd=0.043962 flat_range=0.146383 delta_sd=0.004648 var_ratio=89.4x sd_ratio=9.5x
  counterfactual unpaired 10v10 t=0.49 (paired t=6.52)
```

### 7.4 Difficulty interaction (verbatim)

```text
OLS dMF = -0.043960 + 0.059085 * flatF1; r=0.559 r2=0.312 p=0.01043 se_slope=0.0207
predicted dMF at mean flat F1: 0.007038
predicted dMF at flat F1=0.78: 0.002127
predicted dMF at flat F1=0.93: 0.010990

Q1: flat mean=0.7976 dMF mean=+0.003417 sd=0.00367 seeds=[53, 55, 43, 46, 52]
Q2: flat mean=0.8606 dMF mean=+0.005113 sd=0.00232 seeds=[56, 58, 57, 42, 47]
Q3: flat mean=0.8866 dMF mean=+0.010460 sd=0.00143 seeds=[61, 51, 50, 60, 45]
Q4: flat mean=0.9077 dMF mean=+0.009164 sd=0.00636 seeds=[48, 59, 54, 44, 49]

Hard half (flat<=median): n=10 flat_mean=0.8291 dMF=+0.004265 sd=0.00303 wins=10/10
Easy half (flat>median):  n=10 flat_mean=0.8972 dMF=+0.009812 sd=0.00440 wins=10/10
```

### 7.5 Cross-share moderation (verbatim)

```text
l1_cross_count   rho(dMF)=-0.694 p=0.0007 range=[15.000,63.000]
l1_intra_count   rho(dMF)=+0.694 p=0.0007 range=[230.000,278.000]
l1_cross_share   rho(dMF)=-0.694 p=0.0007 range=[0.051,0.214]
l1_cross_share vs flatF1 rho=-0.755 p=0.0001

model cross_share: R2=0.4263
  beta0=+0.014337 se=0.002153 t=+6.66 p=3.016e-06
  beta1=-0.066224 se=0.018109 t=-3.66 p=0.001804
model flatF1: R2=0.3123
  beta0=-0.043960 se=0.017861 t=-2.46 p=0.02417
  beta1=+0.059085 se=0.020667 t=+2.86 p=0.01043
model both: R2=0.4263
  beta1(cross share)=-0.066423 se=0.036143 t=-1.84 p=0.08364
  beta2(flatF1)=-0.000241 se=0.037675 t=-0.01 p=0.995
```

### 7.6 Type-level and realised pair coverage (verbatim totals)

```text
exact_test_in_train 0
missing_component_type 0
missing_pr_type 0
missing_rs_type 8
missing_ps_type 0
missing_any_pair_type 8
```

Only seed 43 contributes the 8 type-level root–suffix misses. Realised-text
root–suffix misses (from `corpus_stats.csv`):

```text
test_missing_rs_realized sum 40
nonzero seeds [(43, 8), (48, 5), (52, 6), (55, 5), (59, 6), (60, 10)]
test_missing_pr_realized sum 0
test_missing_ps_realized sum 0
exact_test_types_in_train sum 0
exact_test_tokens_in_train sum 0
string_collisions sum 0
```

### 7.7 Replication gap and difficulty adjustment (verbatim)

```text
orig mean/sd 0.009243344907407424 0.004484453014083807
new mean/sd  0.004833622685185168 0.0038340633826599293
Welch t=2.364 p=0.0299; difference=+0.004410
permutation p 0.03168484157579212

flat F1 orig10 mean 0.8742158564814815 new10 mean 0.8520446841931217 all 0.8631302703373016
OLS with orig/new indicator (new=1)
  intercept beta=-0.033643 se=0.017442 t=-1.93 p=0.07063
  flatF1    beta=+0.049056 se=0.019908 t=+2.46 p=0.02469
  new       beta=-0.003322 se=0.001706 t=-1.95 p=0.06822
  R2 0.4377
```

### 7.8 Coverage of the original generator's guarantees

| guarantee | result |
|---|---|
| every test component attested in train types | 46,080/46,080 (all roots/prefixes/suffixes; 0 missing) |
| every test component in realised train text | 46,080/46,080 |
| every test word's exact form absent from train | 46,080/46,080 absent (0 exact overlap) |
| every test word's three pairs seen in train types | 46,072/46,080 (99.98%); seed 43 has 8 misses |
| every test word's three pairs in realised text | 46,040/46,080 (99.913%); 40 root–suffix misses |
| no string collisions across generated words | 0 collisions in all 20 seeds |
| all methods decode exactly (word and doc) | 1.0 in all 80 rows |

The first guarantee is the one the generator's filter enforces; the fourth and
fifth are stronger and are not enforced. The paper should state which one it
means.

---

## 8. Bottom line for the paper

The v512 morph result is real but small and conditional. The defensible sentence
is:

> Holding the flat BPE rules and vocabulary fixed, assigning merge rules to two
> levels by morphological provenance improves boundary F1 by +0.7 points
> (95% CI +0.5 to +0.9) and bits/char by −0.018 at v512 over 20 synthetic seeds,
> with 20/20 paired wins. The gain is a within-corpus effect: 94% of the variance
> in absolute F1 is between-seed corpus difficulty, and the treatment is 0.6% of
> the variance. The gain is larger on corpora whose BPE rules separate cleanly
> into intra- and cross-morpheme subsets, and is near zero when L1 must be filled
> with many cross-majority rules. It does not establish compositional
> generalisation: 99.98% of test words have all three morpheme pairs attested in
> training.

The indefensible sentences are: "+0.9 points" as a constant, "generalises to
unseen compounds" without a pair-disjoint/zero-shot split, and any comparison of
absolute F1 across seeds or across methods trained on different seeds. The
hierarchy itself remains a no-op; what the experiment tests is a principled
*ordering* of the same merge rules.

---

## 9. Files and reproducibility

- Report: `docs/redteam/w1_02_synthetic_seeds.md` (this file).
- Temp scripts and CSVs:
  `C:\Users\acese\AppData\Local\Temp\opencode\w1_02\`
  - `w1lib.py` — fast provenance + extended metrics
  - `validate_fast.py` — seed-42 exact reproduction and provenance check
  - `run_20seeds.py` — 20-seed v512 run
  - `seeds_v512.csv` — 80 rows
  - `corpus_stats.csv` — 20 rows
  - `analyze.py`, `extra*.py`, `type_pair_coverage.py`, `probe_random_order.py`,
    `make_tables.py`
- No source file under `ramified/` or `experiments/`, and no file under
  `results/`, was modified. The only repository write is this report; Python may
  have refreshed existing `__pycache__` bytecode while importing the project.

**Confidence.** High for the corpus statistics, exact reproduction, and variance
decomposition (deterministic, 20 seeds, published rows matched to 0.0). Medium-high
for the survival verdict (fresh 10-seed replication, p = 0.003, but one generator
and one vocabulary). Medium for the cross-share mechanism (20 seeds,
observational moderator, in-sample rule composition; plausible but not a causal
intervention).

**Single check that would most change the verdict.** A pair-disjoint or
zero-shot split (unseen root–suffix pairs, or unseen roots) with the same 20-seed
paired protocol. If the morph advantage disappears there, the result is
interpolation-specific and the compositional claim fails; if it survives, the
result is much stronger than the current interpolation-only evidence.
