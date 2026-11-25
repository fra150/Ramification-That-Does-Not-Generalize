# W1-12 — Dependency closure in `assign_levels`

**Verdict: PASS on closure; conditional on the control.** For flat tokenizers
produced by `train_bpe`, the closure in `ramified/levels.py` is correct: no
level-1 rule depends on a level-2 rule, demotion preserves every rule, and the
v512 morph result is not altered by closure (zero demotions on all ten
published seeds). The strongest
defect is in the **control**, not in the closure: `random_closed` is not always
the advertised size. A rule with the same nonterminal on both sides
(`re+re`, `ba+ba`, `ot+ot`, `in+in`) is appended twice to the topological order;
the set conversion then silently loses a slot. Three of the ten published v512
`random_closed` rows have 292-293 level-1 rules instead of 294. The morph result
survives a re-analysis restricted to the seven exactly size-matched seeds
(+0.0144 F1), and the parallel W1-14 corrected sampler gives +0.0220 F1 over
all ten seeds; the documentation's "same size" claim still does not survive.

Audited files (mtimes at audit time):

| file | mtime | note |
|---|---|---|
| `ramified/levels.py` | 2026-11-22 22:07:03 | closure + `random_closed` |
| `ramified/bpe.py` | 2026-11-22 22:02:02 | `train_bpe` rank order |
| `experiments/level_assignment.py` | 2026-11-22 22:23:04 | changed during audit (micro/macro F1); `boundary_f1` alias unchanged |
| `results/level_assignment_v512_10seeds.csv` | 2026-11-22 22:18:51 | published v512 table |
| `results/level_assignment.csv` | 2026-11-22 22:13:20 | 3 seeds x 3 vocabularies |

No source file under `ramified/` or `experiments/` was edited by this audit.
The reproducible audit is `docs/redteam/w1_12_audit.py` (not imported by the
package); this report is the deliverable.

## 1. What was checked

For a flat tokenizer, let

- `creator[s]` be the unique rule whose `new == s`;
- `terminals` be the character ids plus `unk_id` and `eow_id`;
- `deps(r) = { creator[s].new : s in (r.left, r.right), s not terminal }`.

Four properties were checked for every assignment:

1. **Closure (P1):** for every rule in level 1, `deps(r) ⊆ level1`.
2. **Fixpoint (P2):** the level-1 set equals the brute-force greatest closed
   subset of the initial seed; the brute force removes rules until no rule has
   a dependency outside the set.
3. **Partition (P3):** `level1 ∪ level2` equals the flat rule set, the levels
   are disjoint, and no rule is missing or duplicated.
4. **Size honesty (P4):** installed `len(level1)`/`len(level2)` equal the CSV
   columns; the audit also computes the requested `n1 = int(0.6 * n_rules)` and
   the number of rules lost to demotion/shortfall (`n1 - len(level1)`).

The single-pass demotion is correct only if every dependency has a lower rank.
`train_bpe` guarantees this: a merge rule can only use symbols created by
earlier heap pops, and `rank` is the pop index. This invariant was checked
explicitly on the v512 tokenizer and on the random DAGs.

## 2. Synthetic tokenizer at v512, seed 42

Command:

```powershell
python -u docs/redteam/w1_12_audit.py
```

The v512 tokenizer has 20 characters, 490 merge rules, and a vocabulary of 512.
`n1 = int(0.6 * 490) = 294`.

```text
flat: chars=20 rules=490 vocab=512
rank order: violations=0 unique_ranks=True range=(0, 489)
requested n1 at fraction 0.6: 294
provenance: pairs=489 independent pairs=489 mismatches=0
  frequency     n1= 294 L1= 294 L2= 196 demote/short=   0 closure_bad=0 partition=ok brute=ok
    L1: size=294 strict_intra=197 tie=0 strict_cross=96 unseen=1
    L2: size=196 strict_intra=53 tie=0 strict_cross=143 unseen=0
  morph         n1= 294 L1= 294 L2= 196 demote/short=   0 closure_bad=0 partition=ok brute=ok
    L1: size=294 strict_intra=250 tie=0 strict_cross=43 unseen=1
    L2: size=196 strict_intra=0 tie=0 strict_cross=196 unseen=0
  random_closed n1= 294 L1= 294 L2= 196 demote/short=   0 closure_bad=0 partition=ok
    L1: size=294 strict_intra=214 tie=0 strict_cross=79 unseen=1
    L2: size=196 strict_intra=36 tie=0 strict_cross=160 unseen=0
  reverse       n1= 294 L1= 124 L2= 366 demote/short= 170 closure_bad=0 partition=ok brute=ok
    L1: size=124 strict_intra=54 tie=0 strict_cross=69 unseen=1
    L2: size=366 strict_intra=196 tie=0 strict_cross=170 unseen=0
```

- **P1 holds** in all four modes: zero level-1 rules have a level-2 dependency.
- **P2 holds** for `morph` and `reverse`, the two modes that use the demotion
  loop. `frequency` is closed by rank construction; `random_closed` is closed
  by its topological prefix.
- **P3 holds** in all modes; `reverse` demotes 170 rules into level 2 and loses
  none.
- **P4:** installed sizes match the CSV exactly for seed 42.

The provenance count was recomputed independently from decoded symbol spans
(not from `_walk`'s child offsets): all 489 observed pairs match exactly. The
one rule absent from the provenance is `'.' + EOW` (rank 30), because
`collect_provenance` filters `text.split()` through word objects and `.` is not
a word. It is unseen (score 0.5 in the scored modes) and lands in level 1 in
every mode; this is the `unseen=1` entry. It is not morphological, but it is
one rule and it has no dependencies.

The morph split is not a pure intra/cross split even though `L2` is pure: of
the 294 level-1 rules, 250 are strict intra, **43 are strict cross**, and 1 is
unseen. All 196 level-2 rules are strict cross. The 60% budget forces 43
cross-majority rules into level 1; the documentation's phrasing "intra to level
1, cross to level 2" is therefore approximate.

On the first 2000 unique training words, 277/294 morph level-1 rules fire; the
17 not observed on that sample are rare, not dropped (the partition check
proves all 490 rules are installed).

## 3. Hand-built rule DAGs

### 3a. Chain with two cascading demotions

`r0: a+b->X`, `r1: X+c->Y`, `r2: Y+d->Z`, plus independent rules. Scores put
`r1`, `r2`, `r4` in the seed but not their dependencies.

```text
rank rule            deps       score selected final
   0 a+b->ab   []         0.0   no       L2
   1 ab+c->abc  [12]       1.0   yes      L2
   2 abc+d->abcd [13]       1.0   yes      L2
   3 e+f->ef   []         0.0   no       L2
   4 ef+g->efg  [15]       1.0   yes      L2
   5 h+i->hi   []         1.0   yes      L1
   6 j+a->ja   []         1.0   yes      L1
   7 b+c->bc   []         1.0   yes      L1
   8 d+e->de   []         0.5   no       L2
   9 f+g->fg   []         0.5   no       L2
requested n1=6 seed L1=6 brute L1=3 actual L1=3 brute_match=True
closure violations=0 partition=ok (10 = 3 + 7)
```

The single pass demotes `r1`, then `r2` because `r1` was already demoted, and
`r4` because `r3` was never in level 1. This is the cascade that a naive
"check the original seed once" implementation would miss; the rank-order pass
handles it.

### 3b. Diamond with a demoted middle node

`r0: a+b->X`, `r1: X+c->Y`, `r2: X+d->Z`, `r3: Y+Z->W`. The seed selects
`r0`, `r1`, `r3`; `r2` is outside, so `r3` must cascade out.

```text
n1=3 seed=[12, 13, 15] expected=[12, 13] actual=[12, 13] brute_match=True
closure violations=0 partition=ok (6 = 2 + 4)
```

### 3c. Rank-order counterexample (invalid BPE input)

`assign_levels` assumes, but does not validate, that dependencies have lower
rank. The hand DAG `rA(rank 0) -> rB(rank 5) -> rC(rank 1)` produces a
level-1 rule whose dependency is demoted after it was examined:

```text
n1=4 seed=[13, 14, 15, 16] brute_expected=[15, 16] actual=[14, 15, 16]
closure violations=[(14, [13])]
rank_order_check={'rank_order_bad': 1, 'examples': [(1, 5, 14, 13)]}
```

This cannot happen with `train_bpe`, whose ranks are creation order. It matters
only if `assign_levels` is reused on a hand-built tokenizer. A one-line
`assert dep.rank < rule.rank` (or a post-condition closure assertion) would
turn a silent violation into a test failure.

## 4. Random DAG stress test

Three hundred random rule DAGs were generated with 5-40 rules, dependencies only
on earlier symbols, random intra/cross statistics, and a random fraction in
[0.25, 0.8]. Each was assigned in all four modes.

```text
dags=300 modes=4 checks per assignment
failures={'random_closed:size': 168}
random_closed size shortfalls=168
```

No closure, partition, unresolved-symbol, or brute-fixpoint failures occurred.
The only failure is the `random_closed` size shortfall described next. The
rate is much higher than in real BPE because the random generator creates many
repeated-nonterminal rules; the real v512 tokenizer has 0-2 such rules.

## 5. Strongest finding: `random_closed` is not size-matched

### 5a. The published CSV already shows it

The v512 CSV's `random_closed` level-1 sizes, seeds 42-51, are:

```text
294, 294, 293, 294, 293, 294, 294, 294, 292, 294
```

The requested size is 294. Seeds 44 and 46 lose one rule; seed 50 loses two.
At the other vocabularies the same seed 44 loses one: 600/601 at v1024,
1214/1215 at v2048. `morph` and `frequency` lose none.

### 5b. Mechanism

In `ramified/levels.py` the dependents list is built per symbol occurrence:

```python
dependents[dependency.new].append(rule.new)
```

If a rule has `left == right` and both are the same nonterminal, the dependent
is appended twice. When that dependency is emitted, the child is appended to
`available` twice:

```python
for child in dependents[rule.new]:
    if child not in emitted and deps[child].issubset(emitted):
        available.append(rule_by_new[child])
```

`order` then contains the same rule twice; `level1 = {m.new for m in order[:n1]}`
deduplicates it, so the prefix contains fewer than `n1` distinct rules. The
comment "gives a closed level 1 of exactly n1 rules" is false. Closure still
holds: a prefix of a topological order with duplicate entries is downward
closed, because every dependency still precedes its dependent.

### 5c. Direct reproduction on the real tokenizers

For seeds 44, 46, 50 the flat v512 tokenizer contains exactly the repeated
nonterminal rule(s) predicted by the shortfall:

```text
seed=44 rules=490 n1=294 order_len=491 prefix_distinct=293 shortfall=1
        dups=[('rere', 2)] repeated=[(416, 're+re->rere')] assign_L1=293 L2=197 csv_L1=293
seed=46 rules=490 n1=294 order_len=491 prefix_distinct=293 shortfall=1
        dups=[('baba', 2)] repeated=[(268, 'ba+ba->baba')] assign_L1=293 L2=197 csv_L1=293
seed=50 rules=490 n1=294 order_len=492 prefix_distinct=292 shortfall=2
        dups=[('otot', 2), ('inin', 2)]
        repeated=[(217, 'ot+ot->otot'), (359, 'in+in->inin')] assign_L1=292 L2=198 csv_L1=292
```

The shortfall equals the number of duplicated order entries and the number of
repeated-nonterminal rules. The CSV's `n_l1` values are exactly the
`assign_levels` outputs, so the published rows are reproducible and the cause
is confirmed.

### 5d. Does it change the headline?

No rule is dropped: all 490 rules remain in one of the two levels. But the
`random_closed` control is 0.3-0.7% smaller at the level that matters on three
seeds. Recomputing the paired comparison from the published CSV, excluding the
three non-matched seeds (44, 46, 50):

| comparison, v512 | n | mean Delta F1 | t | 95% CI | W/L/T | mean Delta bpc | t |
|---|---:|---:|---:|---|---:|---:|---:|
| morph - frequency, all 10 | 10 | +0.0092 | 6.52 | [+0.0060, +0.0125] | 10/0/0 | -0.0258 | -5.38 |
| morph - frequency, matched 7 | 7 | +0.0103 | 7.08 | [+0.0067, +0.0138] | 7/0/0 | -0.0261 | -5.13 |
| morph - random_closed, all 10 | 10 | +0.0156 | 3.81 | [+0.0064, +0.0249] | 9/1/0 | -0.0619 | -4.73 |
| morph - random_closed, matched 7 | 7 | +0.0144 | 2.90 | [+0.0022, +0.0266] | 6/1/0 | -0.0550 | -3.23 |
| morph - random_closed, mismatched 3 | 3 | +0.0185 | 2.16 | [+0.0017, +0.0353] | 3/0/0 | -0.0782 | -4.10 |

The direction survives, and the bpc comparison remains 7/0/0 on the matched
seeds, but the F1 comparison weakens (one loss, CI lower bound +0.0022). The
reported +0.0156 sits between our matched-only +0.0144 and W1-14's corrected
+0.0220; the three mismatched seeds happen to have larger morph deltas, but
they are not the source of the effect. The `morph - frequency` result, which
is the primary claim, is unaffected.

The `random_closed - frequency` control comparison is also slightly affected:
on the matched seven seeds it is -0.0041 F1 (t=-0.96, 3/4/0) and +0.0289 bpc
(t=2.03, 5/2/0), versus the all-seed -0.0064 F1 (t=-1.73, 3/7/0) and +0.0361
bpc (t=3.04, 8/2/0) in `docs/findings-v0.2.md`. The direction is unchanged, but
the F1 degradation of `random_closed` is not significant once the size
mismatch is removed.

**Cross-check with W1-14.** A parallel red-team analysis
(`docs/redteam/w1_14_levels_random.md`) independently found the same size bug
and built a deduplicated sampler. Its size-corrected pooled artifact
(`results/redteam/w1_14_size_corrected_pooled.csv`) gives **+0.0220 F1**
(t=7.53, 10/0/0) and **-0.0845 bpc** (t=-7.34, 10/0/0), using corrected
size-294 draws for seeds 44, 45, 50 and size-294 draws elsewhere. Recomputing
the mean from that CSV reproduces those figures. This is consistent with, and
stronger than, our matched-only +0.0144. We did not audit W1-14's sampler; its
report notes that it is not uniform over closed sets.

## 6. Level-size reporting

**What is honest.** Across all 40 v512 rows, `n_l1 + n_l2 = 490`, and
`l1_intra + l1_cross + l1_unseen = n_l1`, `l2_* = n_l2`; the audit found zero
inconsistencies. The installed sizes are the CSV sizes. `level_summary`
reports the installed level sizes and separates unseen rules.

**What is not.** The CSV records neither the requested `n1` nor the number of
demoted rules / duplicate slots. For `random_closed` the difference is a
topological duplicate, not a demotion; for `reverse` it is a real demotion.
Without `n1`, a reader cannot tell whether 292 is a bug or the intended
fraction. The documentation's design table explicitly says `random_closed` is
"della stessa dimensione"; that is false on 3/10 seeds.

**Stale rule metadata.** `assign_levels` reuses the existing `MergeRule`
objects and only partitions the lists; it does not rewrite `rule.level`. Every
rule still carries `level=0`, including the rules moved to level 2. No code in
the package reads `rule.level` (a grep for `.level` found no uses outside the
`Tokenizer.levels` lists), so the CSV is unaffected, but a downstream consumer
that trusts the field would mislabel level-2 rules. This is a low-severity
metadata bug, not a closure bug.

**`reverse` is not a size-controlled reorder.** At v512 it keeps only
124/141/154 of 294 level-1 rules. At v1024 and v2048 it keeps **zero**; the
tokenizer is then exactly flat (all rules in level 2 in rank order), and the
CSV's reverse F1 and bpc equal flat exactly. Calling this a "cross-first
reordering" control at those vocabularies is misleading. It is not part of the
10-seed v512 table, so the headline is unaffected, but it should not be used
as evidence for "any reordering helps" at larger vocabularies.

**Ties.** `level_summary` counts a rule as intra when `a >= b`, so ties are
counted as intra, contrary to the docstring's "by majority". On seed 42 there
were zero ties in every mode, so the published seed-42 composition is exact.
Other seeds were not independently tie-counted.

**Edge cases.** `assign_levels` clamps `n1` to `[1, n_rules-1]`: with one rule
the result is always `[1, 0]`; with five rules, fraction 0.0 gives `[1, 4]` and
fraction 1.0 gives `[4, 1]`. The experiment always uses 0.6, so this is an API
edge, not a published-number issue.

## 7. Does the v512 morph result survive?

**Yes, under the closure audit.**

1. `morph` has `n1 = 294` and `n_l1 = 294` on all ten published seeds. Since
   `assign_levels` starts from exactly `n1` rules and only discards, equal size
   proves **zero demotions**. The seed-42 run was reproduced directly; the
   published CSV's morph rows are unchanged by closure.
2. Seed 42 was regenerated with the current code and compared field by field
   with the committed row: `n_l1`, `n_l2`, `tokens_per_char`,
   `bits_per_char`, `boundary_f1`, `decode_word`, `decode_doc`, and all
   `l1_*`/`l2_*` fields matched with **0 mismatches**. Re-running
   `experiments.analyze_levels` reproduced
   `results/level_assignment_stats_v512.csv` with **0 mismatches**.
3. `morph - frequency` is unaffected by the control bug: +0.0092 F1
   (t=6.52, 10/0/0) and -0.0258 bpc (t=-5.38, 0/10/0), as documented.
4. `morph - random_closed` survives on the seven exactly matched seeds:
   +0.0144 F1 (t=2.90, 6/1/0) and -0.0550 bpc (t=-3.23, 7/0/0). The single F1
   loss is seed 42 (-0.0051 F1, though bpc is still -0.0011). The all-seed
   numbers in `docs/findings-v0.2.md` are arithmetically correct; the caveat is
   that three controls are 1-2 rules short.
5. The independent W1-14 corrected sampler strengthens the comparison:
   +0.0220 F1 (t=7.53, 10/0/0) and -0.0845 bpc (t=-7.34, 10/0/0) from
   `results/redteam/w1_14_size_corrected_pooled.csv`. This removes the
   size-mismatch caveat in favor of the morph result, subject to W1-14's
   sampler not being a uniform closed-set sampler.

W1-14 also reports that `random_closed` is worse than flat on 10/10 seeds, so
the morph-vs-random gap is not the flat-relative effect. The primary claim
should stay `morph - frequency`: +0.0092 F1 and -0.0258 bpc; the
morph-vs-random comparison is a control contrast, not the size of the
improvement over flat.

The demotion branch was never taken by the published morph assignment: the
60% intra-ratio prefix is already closed. The hand DAG and random DAG tests are
therefore the evidence for the closure implementation, not the v512 result.
That is a coverage gap in the experiment, not a defect in the result.

The morph treatment is also less pure than the prose: 43 of 294 level-1 rules
are cross-majority, and the punctuation/EOW rule is unseen. The contrast with
`random_closed` is still real (250 intra vs 214), but the paper should say
"the 60% with the highest intra ratio, then closure", not "intra rules go to
level 1".

## 8. Findings, ranked

1. **High — `random_closed` size bug.** Repeated-nonterminal rules duplicate a
   topological entry; the set conversion loses a slot. Published v512 sizes
   292-293/294 on 3/10 seeds; direct repro on seeds 44, 46, 50. No rule is
   dropped and closure still holds, but the control is not exactly
   size-matched. Fix: deduplicate `available`/`order` or assert
   `len(level1) == n1`. W1-14's corrected sampler already gives +0.0220 F1
   (10/0/0), so the bug does not drive the morph result, but the source bug
   should be fixed before the paper.
2. **Medium — `reverse` is not a reorder at v1024/2048.** Zero level-1 rules
   survive closure, so it is flat. At v512 it is also far from 60/40. It should
   not be reported as a size-controlled control, or it should be replaced by a
   closed cross-first set of the same size.
3. **Medium — rank-order assumption is unvalidated.** The single-pass
   demotion is complete only if every dependency has lower rank. `train_bpe`
   guarantees it, but `assign_levels` accepts any flat tokenizer. The hand DAG
   in section 3c shows a silent level-1 -> level-2 violation. Add an assertion
   and a closure post-condition.
4. **Low-medium — provenance reporting.** On the v512 seed-42 tokenizer,
   `collect_provenance` is exact on the 489 observed pairs, but the `.`+EOW
   rule is unseen and always lands in level 1; `level_summary` counts ties as
   intra; 15-60 cross rules sit in the morph level 1 across seeds (43 on seed
   42). The prose should describe the actual rule, not a pure split.
5. **Low — stale `MergeRule.level`.** Reused rule objects still say `level=0`
   after reassignment, including rules installed in level 2. No package code
   reads the field, but external consumers can be misled.
6. **Low — reporting gap.** The CSV does not record requested `n1`, demotions,
   or duplicate slots. Add these columns so the control size is auditable
   without re-running `assign_levels`.
7. **Positives.** Closure, brute-fixpoint equivalence, partition completeness,
   and decode exactness all pass on the v512 tokenizer, hand DAGs, and 300
   random DAGs. The v512 morph result is not an artifact of closure.

## 9. What would change my mind

- **Most decisive check still open:** a **uniformly sampled closed set of
  exactly 294 rules**, or an independent replication of W1-14's corrected
  sampler on fresh seeds. W1-14 has already run the deduplicated exact-size
  control and found +0.0220 F1 (t=7.53, 10/0/0) and -0.0845 bpc, so the
  size-mismatch caveat is resolved in favor of morph. If a uniform closed-set
  control removes the effect, the "specific morphological assignment beats
  random closed assignment" claim weakens, even though `morph - frequency`
  remains. Current matched-only evidence is +0.0144 F1
  (CI [+0.0022, +0.0266], 6/1/0); W1-14's corrected estimate is larger, but
  the sampler is not uniform.
- A **valid flat BPE tokenizer** (rank order consistent with dependencies) in
  which a level-1 rule depends on a level-2 rule would falsify the closure
  verdict. None was found; the only counterexample needs an invalid rank
  order that `train_bpe` cannot produce.
- A published `morph` row with `n_l1 < 294` whose metrics change under
  demotion would make closure load-bearing for the result. The current CSV has
  none, and seed 42 reproduced exactly.
- If an independent provenance convention (e.g., counting a merge that ends
  exactly at a morpheme boundary as cross) moved a large fraction of the 43
  cross rules, the treatment's interpretation would change. The implementation
  itself matches the current strict-interior convention exactly.

## 10. Reproduction

```powershell
# full audit (v512 seed 42, hand DAGs, 300 random DAGs, edge cases, CSV checks)
python -u docs/redteam/w1_12_audit.py

# seed 42 end-to-end, current experiment code
python -u -m experiments.level_assignment --seeds 42 --vocabs 512 `
  --modes frequency,morph,random_closed `
  --out C:\Users\acese\AppData\Local\Temp\opencode\w1_12_seed42.csv

# paired statistics from the published CSV
python -u -m experiments.analyze_levels `
  --csv results\level_assignment_v512_10seeds.csv `
  --out C:\Users\acese\AppData\Local\Temp\opencode\w1_12_stats.csv

# the three published random_closed shortfalls on the real tokenizers
python -u docs/redteam/w1_12_random_closed_repro.py --seeds 44,46,50
```

The seed-42 CSV matched the committed row with 0 field mismatches; the
regenerated statistics CSV matched the committed statistics with 0 mismatches.
The current `experiments/level_assignment.py` was edited by another process at
22:23 (after the 22:18 CSV) to add micro/macro boundary columns; the
`boundary_f1` alias remains the macro F1, and the reproduced seed-42 values are
identical to the committed ones.

**Confidence:** 0.90 for the closure/partition verdict on valid `train_bpe`
tokenizers; 0.85 for the control-impact estimate, which is based on the
published CSV and a matched-subsample re-analysis rather than a corrected
10-seed rerun.
