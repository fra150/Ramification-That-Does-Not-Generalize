# W1-08 — Tie-break stability of the v512 morphological advantage

**Agent:** W1-08 (red team). **Date:** 2026-11-22.
**Question.** The v0.2 claim is that a morphological level assignment beats
`frequency` and `random_closed` at vocabulary 512. The BPE trainer keeps a lazy
heap of `(-count, pair)` entries; Python compares the pair tuple on ties, so an
equal-count merge is resolved by the smaller `(left, right)` symbol id. That
order is arbitrary. Does the v512 result depend on it?

**Short answer.** The v512 result survives for `morph - frequency`: the
five-seed paired advantage is positive in all 35 regime × seed cells, and the
paired contrast moves by at most **4.3e-4 F1** (largest within-seed range
across the seven regimes; the SD of the seven regime means is 3.9e-5) while
the effect is **+0.00672 F1 / -0.01683 bpc**. The `morph - random_closed` gap
is positive in all seven regime means and in 34/35 cells, but it is a mean
effect: the tie-break-induced range of that gap within a seed is
0.020–0.029 F1, larger than the +0.017 mean gap, and one published draw beats
morph.

## 1. Design

Seven tie-break regimes, all deterministic given the seed:

| regime | heap secondary key | what it stresses |
|---|---|---|
| `id` | `(left, right)` | the current behaviour (status quo) |
| `rev_id` | `(-left, -right)` | the opposite extreme among equal counts |
| `exp_short` | `(len(exp(left)) + len(exp(right)), left, right)` | shorter expanded symbols first |
| `exp_long` | `(-(len(exp(left)) + len(exp(right))), left, right)` | longer expanded symbols first |
| `rand101` | i.i.d. uniform priority per pair | random tie-breaking, draw 1 |
| `rand202` | i.i.d. uniform priority per pair | random tie-breaking, draw 2 |
| `rand303` | i.i.d. uniform priority per pair | random tie-breaking, draw 3 |

`exp_short` / `exp_long` use the character length of the fully expanded symbol
(the way `expansions` stores it). `EOW` contributes 5 characters, so pairs that
end a word get a +5 bias in these two regimes; this is deliberate, it is a
strong perturbation, not a neutral one.

Held fixed across regimes: the synthetic corpus generator (seeds 42–46), the
vocabulary target (512), the flat BPE rule count (490 merges), the level-1
fraction (0.6, giving the 294/196 split), the provenance statistics, the
metrics (`boundary_f1`, add-alpha 0.1 `bits_per_char`, `tokens_per_char`), and
the `random_closed` assignment seed (`seed * 1000 + vocab`). Only the heap
tie-break changes.

Implementation: `results/redteam/w1_08_tiebreak.py` contains a copy of the
fixed `train_bpe` with one change — the heap key — and installs it over
`ramified.bpe.train_bpe` for the duration of the run. No file under `ramified/`
or `experiments/` was edited. A unit test (`w1_08_verify_tiebreak.py`) checks
that `id` reproduces the source trainer on a tied toy sequence and that
`rev_id`, `exp_short`, `exp_long` flip the expected tie.

## 2. Commands

Run from the repository root, with the root on `PYTHONPATH`:

```powershell
$env:PYTHONPATH=(Get-Location).Path

# unit check of the variant semantics
python results/redteam/w1_08_verify_tiebreak.py

# 7 regimes x 5 seeds x v512; frequency / morph / random_closed + flat
python results/redteam/w1_08_tiebreak.py `
  --seeds 42,43,44,45,46 `
  --variants id,rev_id,exp_short,exp_long,rand101,rand202,rand303 `
  --vocab 512 --jobs 5 --tag full

# tables
python results/redteam/w1_08_analyze.py full

# mechanism check, seed 42, id vs rev_id
python results/redteam/w1_08_diagnose.py

# isolate random_closed draw noise on a fixed tokenizer (20 draws per seed, seeds 42-43)
python results/redteam/w1_08_random_noise.py
```

Raw stdout is in `results/redteam/w1_08_full_log.txt`; tables are written to
`results/redteam/w1_08_report_tables_full.txt`.

## 3. Validation of the harness

- `id` reproduces the published 10-seed CSV exactly on seeds 42–46 and all
  four modes: **max |ΔF1| = 0.0, max |Δbpc| = 0.0**. This is the strongest
  check that the monkey-patched copy is faithful.
- The unit check on tied toy sequences confirms that the variants act as
  specified:

  ```text
  source train_bpe   : [(1, 2, 1)]
  copy variant id    : [(1, 2, 1)]
  copy variant rev_id: [(3, 4, 1)]
  exp_short first merge: [(1, 2, 1)]
  exp_long  first merge: [(3, 5, 1)]
  OK: variants behave as specified and id reproduces the source
  ```

- `frequency` equals `flat` in all 35 runs (max |ΔF1| = max |Δbpc| = 0.0):
  a prefix of the rank order really is a no-op, as claimed in v0.2.
- `decode_word` and `decode_doc` are 1.0 in all 140 rows.
- Level sizes: the morph and expansion regimes split 294/196. The
  `random_closed` dependency closure sometimes demotes 1–3 rules, so across
  all runs n_l1 ∈ [291, 294] and n_l2 ∈ [196, 199].

Ties are not rare: **108–118 of the 490 merge steps (23.3–23.5%)** in every
regime. That is the point of the test — the status-quo order makes roughly a
hundred arbitrary decisions per v512 tokenizer.

## 4. Results

### 4.1 `morph - frequency`

The five-seed subset of the published run is the reference: **+0.00672 F1**
(95% CI [+0.00124, +0.01221], 5/0/0) and **-0.01683 bpc** (95% CI
[-0.03267, -0.00099], 5/0/0). The published ten-seed figures are larger
(+0.00924 F1, -0.02579 bpc); seeds 42–46 are a slightly weaker subset, seed 46
being nearly zero.

| regime | mean ΔF1 | SD (seeds) | t (df=4) | 95% CI | seeds Δ>0 |
|---|---:|---:|---:|---:|---:|
| `id` | +0.006722 | 0.004420 | +3.40 | [+0.001235, +0.012209] | 5/5 |
| `rev_id` | +0.006794 | 0.004348 | +3.49 | [+0.001396, +0.012192] | 5/5 |
| `exp_short` | +0.006722 | 0.004420 | +3.40 | [+0.001235, +0.012209] | 5/5 |
| `exp_long` | +0.006794 | 0.004348 | +3.49 | [+0.001396, +0.012192] | 5/5 |
| `rand101` | +0.006707 | 0.004408 | +3.40 | [+0.001235, +0.012179] | 5/5 |
| `rand202` | +0.006722 | 0.004420 | +3.40 | [+0.001235, +0.012209] | 5/5 |
| `rand303` | +0.006707 | 0.004408 | +3.40 | [+0.001235, +0.012179] | 5/5 |
| **all regimes** (per-seed mean) | **+0.006738** | **0.004395** | **+3.43** | **[+0.001282, +0.012194]** | **5/5** |

| regime | mean Δbpc | SD (seeds) | t (df=4) | 95% CI | seeds Δ<0 |
|---|---:|---:|---:|---:|---:|
| `id` | -0.016830 | 0.012758 | -2.95 | [-0.032669, -0.000991] | 5/5 |
| `rev_id` | -0.017010 | 0.012546 | -3.03 | [-0.032586, -0.001434] | 5/5 |
| `exp_short` | -0.016984 | 0.012480 | -3.04 | [-0.032477, -0.001491] | 5/5 |
| `exp_long` | -0.016853 | 0.012802 | -2.94 | [-0.032746, -0.000960] | 5/5 |
| `rand101` | -0.016862 | 0.012804 | -2.94 | [-0.032757, -0.000967] | 5/5 |
| `rand202` | -0.016986 | 0.012476 | -3.04 | [-0.032475, -0.001498] | 5/5 |
| `rand303` | -0.016877 | 0.012796 | -2.95 | [-0.032763, -0.000992] | 5/5 |
| **all regimes** (per-seed mean) | **-0.016915** | **0.012665** | **-2.99** | **[-0.032638, -0.001191]** | **5/5** |

All 35 regime × seed cells are positive on F1 and all 35 are negative on bpc.
The per-seed pattern shows where the (small) regime dependence lives:

| seed | id | rev_id | exp_short | exp_long | rand101 | rand202 | rand303 |
|---|---:|---:|---:|---:|---:|---:|---:|
| 42 | +0.004123 | +0.004557 | +0.004123 | +0.004557 | +0.004123 | +0.004123 | +0.004123 |
| 43 | +0.009722 | +0.009722 | +0.009722 | +0.009722 | +0.009722 | +0.009722 | +0.009722 |
| 44 | +0.009635 | +0.009563 | +0.009635 | +0.009563 | +0.009563 | +0.009635 | +0.009563 |
| 45 | +0.009983 | +0.009983 | +0.009983 | +0.009983 | +0.009983 | +0.009983 | +0.009983 |
| 46 | +0.000145 | +0.000145 | +0.000145 | +0.000145 | +0.000145 | +0.000145 | +0.000145 |

The seven regime means span [+0.006707, +0.006794] F1 and
[-0.017010, -0.016830] bpc. The SD of the seven means is **3.9e-5 F1** and
**7.5e-5 bpc**, i.e. the effect is **174×** and **224×** larger than the
between-regime SD. The largest within-seed range of the paired delta is
**4.34e-4 F1** (seed 42) and **8.65e-4 bpc** (seed 46); on F1, seeds 43, 45
and 46 are invariant to 1e-6 or exactly, and seed 44 moves by 7.2e-5. In short,
the level-assignment effect is not a tie-break artefact: swapping the
tie-break changes which merge rules exist, but the paired improvement moves by
less than a tenth of its own size.

### 4.2 `morph - random_closed`

This comparison is not the same kind of test. When the rule set changes, the
dependency graph changes, so the random topological order is effectively
re-drawn; the spread across regimes mixes tie-break noise with a fresh random
control. The 5-seed reference is already weak (+0.01190 F1, 95% CI
[-0.00637, +0.03017]); it is the ten-seed result (+0.01563, t=3.81, 9/1) that
carries the published claim.

**F1**

| regime | mean ΔF1 | SD (seeds) | t (df=4) | 95% CI | seeds Δ>0 |
|---|---:|---:|---:|---:|---:|
| `id` | +0.011903 | 0.014717 | +1.81 | [-0.006368, +0.030174] | 4/5 |
| `rev_id` | +0.019273 | 0.010640 | +4.05 | [+0.006063, +0.032483] | 5/5 |
| `exp_short` | +0.017809 | 0.016244 | +2.45 | [-0.002358, +0.037975] | 5/5 |
| `exp_long` | +0.014912 | 0.012497 | +2.67 | [-0.000602, +0.030426] | 5/5 |
| `rand101` | +0.019347 | 0.009662 | +4.48 | [+0.007352, +0.031342] | 5/5 |
| `rand202` | +0.018158 | 0.015351 | +2.64 | [-0.000900, +0.037215] | 5/5 |
| `rand303` | +0.020073 | 0.016073 | +2.79 | [+0.000120, +0.040027] | 5/5 |
| **all regimes** (per-seed mean) | **+0.017354** | **0.010952** | **+3.54** | **[+0.003757, +0.030950]** | **5/5** |

**bpc**

| regime | mean Δbpc | SD (seeds) | t (df=4) | 95% CI | seeds Δ<0 |
|---|---:|---:|---:|---:|---:|
| `id` | -0.049106 | 0.043511 | -2.52 | [-0.103124, +0.004912] | 5/5 |
| `rev_id` | -0.108912 | 0.067575 | -3.60 | [-0.192804, -0.025020] | 5/5 |
| `exp_short` | -0.076699 | 0.076068 | -2.25 | [-0.171135, +0.017737] | 5/5 |
| `exp_long` | -0.052385 | 0.043729 | -2.68 | [-0.106673, +0.001904] | 5/5 |
| `rand101` | -0.067837 | 0.033966 | -4.47 | [-0.110005, -0.025670] | 5/5 |
| `rand202` | -0.066190 | 0.051839 | -2.86 | [-0.130545, -0.001834] | 5/5 |
| `rand303` | -0.108127 | 0.108948 | -2.22 | [-0.243383, +0.027128] | 5/5 |
| **all regimes** (per-seed mean) | **-0.075608** | **0.044919** | **-3.76** | **[-0.131374, -0.019842]** | **5/5** |

The tie-break changes the random control much more than it changes morph. The
within-seed range of `morph - random_closed` across the seven regimes is
**0.0196–0.0290 F1** (0.086–0.234 bpc), i.e. larger than the +0.0174 mean gap.
The sign is not universal: the published `id` draw for seed 42 is the one
negative cell (-0.00509 F1), and `rev_id` flips that same seed to +0.02292.

To separate tie-break noise from draw noise, `w1_08_random_noise.py` fixes the
`id` flat tokenizer and draws 20 fresh `random_closed` assignments per seed on
seeds 42 and 43:

```text
seed 42: random_closed F1 mean=0.852824 sd=0.014426 range=0.055082
         morph - random F1 mean=+0.017806 sd=0.014426 wins=20/20
         morph - random bpc mean=-0.067252 wins=20/20
seed 43: random_closed F1 mean=0.790753 sd=0.007775 range=0.033507
         morph - random F1 mean=+0.020054 sd=0.007775 wins=20/20
         morph - random bpc mean=-0.098274 wins=20/20
```

Morph beats all 40 fresh draws on a fixed tokenizer. The published seed-42
draw (0.875723, RNG seed 42512) is a different draw: it is above all 20
seed-42 draws (max 0.867535), about 1.6 SD above their mean, and in the main
35-cell table it is the one cell where the random control beats morph. The
control's mean also moves when the base rule graph changes, which is why the
within-seed regime range (0.019–0.029 F1) is large. The secondary claim is a
mean over draws and regimes, not a per-draw guarantee across tokenizers.

### 4.3 Effect versus tie-break noise

| quantity | mean | SD of regime means | max within-seed range |
|---|---:|---:|---:|
| morph F1 | 0.862454 | 9.51e-5 | 7.67e-4 |
| frequency ( = flat) F1 | 0.855716 | 1.07e-4 | 7.96e-4 |
| **morph - frequency F1** | **+0.006738** | **3.90e-5** | **4.34e-4** |
| random_closed F1 | 0.845100 | 2.86e-3 | 2.90e-2 |
| morph - random_closed F1 | +0.017354 | 2.93e-3 | 2.90e-2 |
| morph bpc | 2.304797 | 2.78e-4 | 3.05e-3 |
| frequency ( = flat) bpc | 2.321712 | 2.25e-4 | 2.23e-3 |
| **morph - frequency bpc** | **-0.016915** | **7.5e-5** | **8.65e-4** |
| random_closed bpc | 2.380405 | 2.42e-2 | 2.34e-1 |
| morph - random_closed bpc | -0.075608 | 2.44e-2 | 2.34e-1 |

**Answer to the question.** For `morph - frequency`, yes: the paired effect
(+0.00672 F1, -0.01683 bpc) is 15–20× the largest within-seed tie-break
movement of the contrast and 174–224× the SD of the seven regime means. Even
the absolute morph metrics move 9× (F1) and 5.5× (bpc) less than the effect.
For `morph - random_closed`, no in the per-draw sense: the +0.017 mean gap is
smaller than the 0.019–0.029 F1 within-seed movement of the control, and the
random control's bpc movement (0.234) is 3.1× its mean gap.

## 5. Why the contrast is stable: rule drift is not segmentation drift

Symbol ids are not comparable across regimes (a different merge order renames
the symbols), so the diagnostic compares rules by their expansion strings. On
seed 42:

- `id` vs `rev_id`: first semantic divergence at merge rank **23**; **423**
  rules are shared of a **557**-rule union (134 symmetric-difference slots
  across the two 490-rule sets). Morph level-1: **229** shared of a
  **359**-rule union, i.e. 130 differing slots across the two 294-rule sets.
- Yet on the 2304 held-out words, only **6/2304** are segmented differently
  under the flat tokenizer and **4/2304** under the morph tokenizer.
- The per-word F1 contribution of `morph - frequency` is almost the same
  object: `id` improves 20 words and worsens none, `rev_id` improves 21;
  2303/2304 words have the same changed/unchanged status. Only one word
  differs, moving the mean from **+0.004123** to **+0.004557**.

So ties are abundant and they do change the rule set substantially, but on
this corpus the changed rules are nearly inert on the held-out vocabulary. The
morph assignment's effect is carried by a small, stable set of boundary
decisions, not by the rules that the tie-break swaps.

This also exposes the size of the effect: on seed 42, the whole +0.0041 F1
advantage is 20 of 2304 words, all improved. The effect is a small number of
boundary fixes, not a broad re-segmentation.

## 6. Findings

1. **The `morph - frequency` advantage survives every regime.** All 35
   regime × seed cells are positive on F1 and all 35 are negative on bpc. The
   seven regime means lie in [+0.006707, +0.006794] F1 and
   [-0.017010, -0.016830] bpc, versus the five-seed reference +0.006722 /
   -0.016830. The effect is not created or destroyed by the tie-break.
2. **The paired contrast is much more stable than the underlying vocabulary.**
   On seed 42, the `id` and `rev_id` flat rule sets differ in 134 rule slots
   (67 per set; 423 shared), and the two level-1 sets differ in 130 slots
   (65 per set; 229 shared), yet only 6/2304 flat and 4/2304 morph test-word
   segmentations differ; 2303/2304 words keep the same changed/unchanged
   per-word F1 status, and the mean moves by +0.00043.
3. **Tie-break is not the limiting uncertainty.** The between-seed SD is
   0.0044 F1, while the across-regime SD of the effect is 3.9e-5 (113×
   smaller) and the largest within-seed regime range is 4.3e-4. The five-seed
   CIs are wider than the ten-seed ones; the tie-break test does not add to
   that sampling uncertainty.
4. **`morph - random_closed` is a mean effect, not a per-draw guarantee.**
   All seven regime means are positive (+0.0119 to +0.0201 F1; 34/35 cells),
   but the control's own within-seed movement (0.019–0.029 F1) is larger than
   the +0.017 mean gap, the published seed-42 `id` cell is negative (-0.0051),
   and a fresh 40-draw check (20 each on seeds 42–43, fixed `id` tokenizer)
   gives 40/40 wins with mean gaps +0.0178 and +0.0201. A high draw can beat
   morph; a typical one does not.
5. **Mechanism caveat.** The invariance is real but partly corpus-specific:
   the test set only sees 4–6 changed segmentations out of 2304 even when 130
   level-1 rules differ. A corpus whose held-out words exercise the changed
   rules could be more sensitive.

## 7. Limits and the check that would most change the verdict

The test is synthetic, v512 only, five seeds, one flat vocabulary per seed and
regime. The text is Zipfian over 9216 training words, and the held-out set is
uniform over 2304 combinations of attested components. Ties already diverge
early — `rev_id` first differs from `id` at ranks 11–75 depending on seed — but
the swapped rules are rarely exercised by the held-out words, which is why
segmentation barely moves.

The single check that would most change the verdict: **rerun on a test
distribution that exercises the changed rules** — for example uniform-frequency
training, or held-out words built from rare roots and unseen affix
combinations — and check the sign and size of `morph - frequency`. If the
advantage stays positive there, the invariance is structural rather than a
property of a held-out sample that barely sees 130 changed level-1 rules.

## 8. Outputs

| file | contents |
|---|---|
| `results/redteam/w1_08_runs_full.csv` | 140 rows: 5 seeds × 7 regimes × 4 modes |
| `results/redteam/w1_08_pairwise_full.csv` | 105 paired deltas |
| `results/redteam/w1_08_summary_full.csv` | 42 paired statistics |
| `results/redteam/w1_08_ties_full.csv` | tie census per training |
| `results/redteam/w1_08_rules_full.csv` | id-keyed rule-sequence divergence |
| `results/redteam/w1_08_report_tables_full.txt` | tables printed in this report |
| `results/redteam/w1_08_diagnose_log.txt` | semantic rule/segmentation diagnostic |
| `results/redteam/w1_08_random_noise_log.txt` | 40 random_closed draws (20 each, seeds 42–43) on a fixed tokenizer |
