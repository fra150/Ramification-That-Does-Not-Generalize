# W1-15 — Morph level assignment: provenance weighting and disjoint-word estimates

**Agent:** W1-15 (red team, read-only) · **Date:** 2026-11-22 ·
**Python:** 3.13.13, Windows (PowerShell), numpy 2.2.6

**Verdict: the v512 advantage survives.** On seeds 42–46, with the flat BPE
rules and vocabulary fixed, a morph level assignment estimated from a *disjoint*
set of words still beats the frequency control on boundary F1 and bpc in 5/5
seeds. The original full-provenance result also reproduces exactly. The
advantage is **not** an artifact of frequency weighting: unique-word and
frequency-weighted provenance select the *same* 294 level-1 rules on every seed,
despite raw counts differing by ~15×.

The important qualification is that the larger gains from half/root-held-out
provenance are mostly a **level-1 size / dependency-closure effect**, not
evidence that held-out provenance estimates a better rule ranking. Full
provenance truncated to the same level-1 size matches the held-out variants
within +0.001 to +0.0034 F1 (no individual interval excludes zero), and itself
beats frequency by +0.015 F1 / −0.035 bpc. The morphology is still doing real
work: an approximately size-matched random-closed level 1 is *worse* than
frequency, and morph beats it by +0.027 F1 / −0.097 bpc.

**Confidence:** high (~0.85) that the direction survives disjoint-word/root
provenance on this synthetic v512; high (~0.9) that unique vs frequency-weighted
provenance is inert on this corpus; low (~0.3) that held-out provenance is a
better estimator beyond level-1 size/purity; low (~0.3) for generalization to
real morphology.

All scripts are new files under `results/redteam/`; no source file was modified.

---

## 1. Question and design

The published v512 ablation (`docs/findings-v0.2.md`, §2) computes rule
provenance with

```python
collect_provenance(flat, syn.train_word_objects, text=syn.train_text)
```

i.e. occurrence-frequency-weighted over the sampled training text, and assigns
the top 60% of rules by intra-morpheme ratio to level 1. Two red-team questions:

1. **Counting:** does the result depend on frequency-weighted text vs one count
   per unique training word?
2. **Disjointness:** does the result survive when provenance is estimated from
   words disjoint from the words used to train the merges / to evaluate? A
   word-level leave-one-out estimate at the rule-score level is algebraically a
   no-op here: `_score` uses aggregate `intra/(intra+cross)`, and summing
   leave-one-word-out counts over all words rescales every rule by `(N-1)/N`.
   The non-trivial implementations are therefore a 50/50 disjoint word split
   and 10-fold cross-fitting by root.

Design, identical to `experiments/level_assignment.py` except for the
provenance source:

- synthetic corpus, seeds **42, 43, 44, 45, 46**, `vocab=512`, `fraction=0.6`;
- one flat BPE tokenizer per seed; rules, vocabulary and rank order fixed;
- provenance variants:
  - `freq_all`: frequency-weighted training text (published design);
  - `unique_all`: each unique training word once (`text=None`);
  - `freq_halfA/B`, `unique_halfA/B`: 50/50 disjoint split of the unique
    training words; estimate on one half, evaluate on held-out test words and
    on the *other* training half;
  - `root_cv`: 10-fold cross-fitting by root. For each test word, provenance
    comes only from training words whose root is in a different fold, so the
    test word never shares a root with its provenance set;
- controls:
  - `frequency` / `flat`: rank-prefix budget split (identical tokenization);
  - `random_closed`: random topological prefix of the same size;
  - **size-matched frequency**: rank prefix with exactly the same level-1 size
    as the morph variant, to separate assignment quality from closure;
  - **full-top-k**: full-provenance morph assignment truncated to the same
    level-1 size as a held-out variant;
  - **random-closed at the same k**: any closed level-1 set of approximately
    the same size (within 2 rules; the sampler has the W1-12 duplicate-slot
    bug).

A one-time cache check verifies that aggregating per-unique-word provenance
counts and multiplying by text frequency is identical to the library's
`collect_provenance` call.

## 2. Commands

```powershell
# Main run: weighting, disjoint halves, root cross-fitting, size-matched controls
python -m results.redteam.w1_15_levels_weighting --seeds 42,43,44,45,46

# Full-provenance morph truncated to the held-out L1 size, plus random-closed
python -m results.redteam.w1_15_topk_control --seeds 42,43,44,45,46
python -m results.redteam.w1_15_topk_analysis

# Why unique and frequency-weighted provenance select the same rules
python -m results.redteam.w1_15_weighting_diagnostic --seeds 42,43,44,45,46
```

Logs: `results/redteam/w1_15_run.log`,
`results/redteam/w1_15_topk.log`,
`results/redteam/w1_15_topk_analysis.log`,
`results/redteam/w1_15_weighting_diagnostic.log`.

## 3. Sanity checks and reproduction

- Per-unique-word cache vs `collect_provenance`: **max count mismatch 0** on all
  seeds. The frequency-weighted reference was re-encoded once (120k
  occurrences) and also matched at 0.
- Published CSV (`results/level_assignment_v512_10seeds.csv`, seeds 42–46):
  **F1 difference 0.0** for flat, frequency, morph and random_closed on every
  seed; max bpc difference **2.9e-14**.
- Flat and frequency are identical in F1 and bpc on all seeds, as expected from
  the corrected budget-partition null.
- Training word types: 9,216 per seed; only 7,405–7,493 (80–81%) ever occur in
  the sampled text. The frequency-weighted estimate therefore gives zero weight
  to ~1,700–1,800 types that the unique estimate counts once. This is the
  strongest possible version of the weighting contrast.

## 4. Main result: the advantage survives disjoint provenance

Per-seed deltas vs the frequency control (5 seeds, paired):

| seed | flat/freq F1 | freq_all ΔF1 | half(A/B) mean ΔF1 | root_cv ΔF1 | freq_all Δbpc | half mean Δbpc | root_cv Δbpc |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 42 | 0.8665 | +0.0041 | +0.0225 | +0.0069 | −0.0054 | −0.0556 | −0.0121 |
| 43 | 0.8011 | +0.0097 | +0.0223 | +0.0107 | −0.0241 | −0.0270 | −0.0229 |
| 44 | 0.9142 | +0.0096 | +0.0130 | +0.0100 | −0.0299 | −0.0526 | −0.0345 |
| 45 | 0.8935 | +0.0100 | +0.0320 | +0.0235 | −0.0237 | −0.0598 | −0.0497 |
| 46 | 0.8029 | +0.0001 | +0.0056 | +0.0021 | −0.0010 | −0.0175 | −0.0069 |

Every F1 delta is positive (5/5) and every bpc delta is negative (5/5), but
seed 46 is essentially a tie for the published design (+0.0001 F1, −0.0010
bpc).

Paired 95% intervals (t critical 2.776, df=4):

| comparison | ΔF1 | t | 95% CI | W/L/T | Δbpc | t | 95% CI | W/L/T |
|---|---:|---:|---|---:|---:|---:|---|---:|
| freq_all (published) | **+0.0067** | 3.40 | [+0.0012, +0.0122] | 5/0/0 | **−0.0168** | −2.95 | [−0.0327, −0.0010] | 0/5/0 |
| unique_all | +0.0067 | 3.40 | [+0.0012, +0.0122] | 5/0/0 | −0.0168 | −2.95 | [−0.0327, −0.0010] | 0/5/0 |
| random_closed | −0.0052 | −0.88 | [−0.0214, +0.0111] | 2/3/0 | +0.0323 | 1.82 | [−0.0170, +0.0815] | 4/1/0 |
| **freq_half mean (test)** | **+0.0191** | 4.22 | [+0.0065, +0.0316] | 5/0/0 | **−0.0425** | −5.00 | [−0.0661, −0.0189] | 0/5/0 |
| unique_half mean (test) | +0.0182 | 4.15 | [+0.0060, +0.0304] | 5/0/0 | −0.0402 | −5.50 | [−0.0605, −0.0199] | 0/5/0 |
| **root_cv (10-fold)** | **+0.0107** | 2.99 | [+0.0008, +0.0205] | 5/0/0 | **−0.0252** | −3.26 | [−0.0467, −0.0037] | 0/5/0 |
| freq_half, held-out train words | +0.0199 | 4.40 | [+0.0074, +0.0325] | 5/0/0 | — | — | — | — |
| unique_half, held-out train words | +0.0190 | 4.50 | [+0.0072, +0.0307] | 5/0/0 | — | — | — | — |

Size-matched frequency controls give the same deltas as the ordinary frequency
control for the half variants, because a frequency prefix of 250–290 rules
tokenizes exactly like the full flat tokenizer on this corpus. The point is that
level-1 *size alone* does not explain the morph gain. The root_cv bpc against
the size-matched control is −0.0263 (t = −3.40), essentially unchanged.

**So: yes, the v512 advantage survives a disjoint-word and disjoint-root
provenance estimate, and the half-sample effect is larger, not smaller.**

## 5. Why the weighting axis is inert

Raw statistics differ substantially, but the *ranking* does not:

| seed | total counts unique | total counts freq | freq/unique | Pearson score | Spearman score | mean \|Δscore\| | max \|Δscore\| | L1 overlap |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 42 | 62,109 | 932,893 | 15.02 | 0.99738 | 0.999995 | 0.0045 | 0.4955 | 294/294 |
| 43 | 63,638 | 975,394 | 15.33 | 0.99949 | 0.999999 | 0.0017 | 0.2794 | 294/294 |
| 44 | 63,325 | 1,011,377 | 15.97 | 0.99877 | 0.999988 | 0.0032 | 0.3367 | 294/294 |
| 45 | 63,085 | 947,263 | 15.02 | 0.99853 | 0.999998 | 0.0029 | 0.3886 | 294/294 |
| 46 | 62,415 | 961,772 | 15.41 | 0.99945 | 0.999999 | 0.0015 | 0.3334 | 294/294 |

- The selected level-1 set is **identical (294/294)** on all five seeds, and the
  downstream F1/bpc are numerically identical. The `l1_intra`/`l1_cross` summary
  can differ by one borderline rule because it is classified with each variant's
  own counts, but membership is unchanged.
- The raw counts differ by ~15×; ~19% of training types have frequency zero.
  The near-invariance is consistent with a stable ratio ranking: every rule's
  occurrences are spread over many words, and the frequency weights are not
  concentrated enough to reorder the top 294.
- With only half the provenance, weighting has a small but visible effect:
  `freq_halfA` vs `unique_halfA` changes level-1 size by 0–4 rules and F1 by at
  most 0.004. The advantage over frequency is unchanged.

**Consequence:** this corpus cannot distinguish frequency-weighted from
type-weighted provenance because the two produce the same assignment. The
published result is not a frequency-weighting artifact, but it is also not
evidence *for* frequency weighting. The honest statement is "the assignment is
invariant to this counting choice on this corpus."

## 6. The held-out gain is mostly a level-1 size / closure effect

Held-out provenance is not a cleaner estimator of the rule ranking; it mainly
causes more dependency-closure demotions, shrinking level 1 from 294 to
~248–291 rules and making it nearly purely intra-morpheme (`l1_cross` often 0).
To test this, full-provenance morph was truncated to the **same level-1 size**
as each held-out variant:

| held-out variant | ΔF1 vs full-top-k (same size) | t | 95% CI | W/L/T | Δbpc vs full-top-k |
|---|---:|---:|---|---:|---:|
| freq_halfA | +0.0015 | 1.63 | [−0.0011, +0.0041] | 2/0/3 | −0.0047 |
| freq_halfB | +0.0010 | 1.40 | [−0.0010, +0.0029] | 2/0/3 | −0.0022 |
| unique_halfA | +0.0015 | 1.86 | [−0.0007, +0.0036] | 3/0/2 | −0.0047 |
| unique_halfB | +0.0034 | 2.15 | [−0.0010, +0.0079] | 3/0/2 | +0.0058 |
| root_cv | +0.0023 | 1.19 | [−0.0030, +0.0076] | 3/2/0 | −0.0033 |

Full-provenance morph at the held-out L1 sizes, vs frequency:

- F1 **+0.0150**, t = 4.02, 95% CI [+0.0046, +0.0254], 5/0/0;
- bpc **−0.0354**, t = −5.13, 95% CI [−0.0546, −0.0162], 0/5/0.

In other words, nearly all of the half/root gain is reproduced by a
full-provenance assignment that is simply truncated to the same level-1 size.
The held-out estimate adds at most ~+0.0034 F1, and every interval includes zero.
This is a **regularization/abstention effect**, not evidence of superior
held-out ranking.

Crucially, this does **not** invalidate the published v512 comparison: the
original morph and frequency tokenizers both have `n_l1 = 294`, so the published
+0.0067 F1 / −0.0168 bpc is size-matched by construction. The size effect only
inflates the held-out variants' apparent gain.

Morphology is still special. An approximately size-matched random closed
level-1 (within 2 rules) is worse than frequency on the per-seed mean over k
(ΔF1 −0.0121, t = −2.26; Δbpc +0.0614, t = 3.56, both 5/5 in the bad
direction), while full-provenance
morph beats the same-k random control by **+0.0271 F1** (t = 5.44, 5/0/0) and
**−0.0968 bpc** (t = −6.17, 0/5/0). "Any small closed level 1" is not enough;
selecting intra-morpheme rules is what works.

### 6.1 Relation to other W1 reports

- **W1-11** (`docs/redteam/w1_11_levels_provenance.md`) audited
  `collect_provenance` and found the raw `intra/(intra+cross)` score untuned:
  Laplace or Wilson smoothing changes which rules fill the 43 cross-majority
  level-1 slots, and on seed 42 improves F1. Our decomposition is consistent
  with that: those 43 cross rules are the liability, and truncating level 1 to
  the same smaller size recovers most of the held-out gain. We did not test
  smoothing; it is a separate route to the same purity effect.
- **W1-12** (`docs/redteam/w1_12_levels_closure.md`) found closure correct and
  **zero demotions** for the published full-provenance morph at v512
  (`n_l1 = 294`). We confirm `n_l1 = 294` and therefore no size gap in the
  published comparison. The held-out variants *do* demote (to 248–291 rules)
  because half/root samples leave some dependencies unseen; that is the
  mechanism behind their extra gain, not a closure defect.
- **W1-14** (`docs/redteam/w1_14_levels_random.md`) found morph outside the
  distribution of 10 random topological orders on most seeds. Our same-k
  random-closed control agrees in direction: random is worse than flat here
  (−0.0121 F1), and morph is far above it (+0.0271 F1). The random sampler's
  duplicate-slot bug is why our random sizes are within 2 rules of `k`; that
  cannot explain a gap of this size.

## 7. Limitations

1. **Five seeds, subset of the headline.** Seeds 42–46 give +0.0067 F1 /
   −0.0168 bpc, versus the 10-seed headline +0.0092 / −0.0258. The direction is
   5/5, but the effect is ~27% smaller for F1 and ~35% smaller for bpc on this
   subset; seed 46 is a near-tie. The remaining seeds 47–51 average +0.0118 F1,
   so the first-five subset understates the headline. The 5-seed rerun does not
   reproduce the headline effect size, only its sign and significance.
2. **Word/root disjointness is not morpheme disjointness.** Every test word
   recombines prefixes, roots and suffixes attested in training (the generator
   guarantees this). `root_cv` removes the shared root, but the shared prefix
   and suffix inventories remain. A truly disjoint-morpheme test is not
   possible with this generator.
3. **Root CV is a 10-fold cross-fit, not leave-one-root-out.** Each fold
   excludes ~12 roots; the aggregate F1 and bpc are cross-fitted across
   different tokenizers, not a single global tokenizer. The top-k comparison for
   root_cv uses the mean fold `n_l1`, so it is approximate.
4. **Unique-vs-frequency invariance is corpus-specific.** The synthetic corpus
   has only 8 prefixes, 12 suffixes and 120 roots, and the Zipf sampling gives
   every rule a broad, similar weight profile. A corpus with a stronger
   frequency–morphology interaction could make the counting choice matter.
5. **bpc is a unigram proxy.** No downstream embedding or representation test
   was run; the F1/bpc gain does not establish a representation gain.
6. **The held-out estimates are regularized assignments, not clean estimators.**
   Their larger effect should be reported as a size/purity effect, not as
   "held-out provenance generalizes better."

## 8. What survives, exactly

- **v512 direction and significance:** survives on 5/5 seeds under (a) 50/50
  disjoint word provenance (+0.0191 F1, −0.0425 bpc), (b) provenance/evaluation
  on disjoint training-word halves (+0.0199 F1), and (c) root-disjoint 10-fold
  cross-fitting (+0.0107 F1, −0.0252 bpc).
- **Original published comparison:** survives exactly as run (full provenance,
  `n_l1 = 294` both arms): +0.0067 F1 / −0.0168 bpc on seeds 42–46, 5/5 each,
  exact reproduction of the CSV. Effect size is smaller than the 10-seed
  headline; seed 46 is a near-tie.
- **Frequency-weighting artifact:** rejected. Unique and frequency-weighted
  provenance produce the identical level-1 set and identical metrics on all
  five seeds.
- **Size-only explanation:** rejected for the original comparison. The frequency
  prefix at the same smaller sizes tokenizes like flat, and a same-size random
  closed set is worse than flat. Morph selection is doing the work.
- **"Held-out provenance is a better estimator":** not supported. Its extra gain
  is mostly closure-driven level-1 shrinkage and is reproduced by full
  provenance truncated to the same size.

## 9. Artifacts

| file | contents |
|---|---|
| `results/redteam/w1_15_levels_weighting.py` | main experiment |
| `results/redteam/w1_15_levels_weighting_runs.csv` | per-seed, per-variant F1/bpc/level sizes |
| `results/redteam/w1_15_levels_weighting_stats.csv` | paired deltas and 95% CIs |
| `results/redteam/w1_15_levels_weighting_root_folds.csv` | per-root-fold morph/frequency/size-matched results |
| `results/redteam/w1_15_levels_weighting_similarity.csv` | level-1 Jaccard between assignments |
| `results/redteam/w1_15_levels_weighting_checks.csv` | cache checks and exact reproduction against the published CSV |
| `results/redteam/w1_15_topk_control.py` | full-provenance top-k and same-k random-closed controls |
| `results/redteam/w1_15_topk_control.csv` | per-seed top-k results |
| `results/redteam/w1_15_topk_analysis.py` / `.csv` | held-out vs same-size full-provenance comparisons |
| `results/redteam/w1_15_weighting_diagnostic.py` / `.csv` | raw score correlations and L1 overlap |
| `results/redteam/w1_15_run.log`, `w1_15_topk.log`, `w1_15_topk_analysis.log`, `w1_15_weighting_diagnostic.log` | full console output |

## 10. Bottom line for the paper

The v512 claim is robust to *how provenance is counted* and to *estimating it
on disjoint words/roots*. The effect is small and the 5-seed subset is weaker
than the 10-seed headline, but the direction is 5/5 and the mechanism is
morphological: same-size random or frequency-prefix level-1 assignments do not
reproduce it. The paper should not claim that held-out provenance is a better
estimator; the larger held-out gains are a level-1 size/closure effect. The
original size-matched v512 comparison is the defensible result.

**The single check that would most change this verdict** is a corpus with
*held-out morphemes*, i.e. unseen prefixes and suffixes rather than unseen
recombinations of a shared inventory, run with the same size-matched controls.
If the advantage disappears there, the result is limited to novel combinations
of familiar morphemes. This generator cannot make that test: every test word's
prefix, root and suffix are attested in training by construction.
