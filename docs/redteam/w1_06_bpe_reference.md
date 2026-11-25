# W1-06 — Independent reference check of the F1 fix in `ramified/bpe.py`

**Date:** 2026-11-22
**Snapshot checked:** `ramified/bpe.py`, 278 lines, SHA-256
`73093A3E62FC66A59594F29CDB8549F4523D358DF04EBAB46E412C7C47C5B0C2`
(mtime 2026-11-22 22:02:02). The hash was identical before and after every
run below.
**Review snapshot:** the adversarial review identifies the pre-fix revision by
SHA-256 `F4E6F2C8EA9C4C63F0AEE77CFA6AC6AA781B20953FF323F7FF2F3652F4CD2A17`
(255 lines). The current file is the fixed revision: `_update_pair_counts`
returns every changed pair and the caller pushes them (lines 40-65, 118-120,
133-135).
**Environment:** Python 3.13.13, Windows, PowerShell; all commands from
`C:\Users\acese\Desktop\unibull`.
**Source changes by this agent:** none. The harnesses live outside the repo in
`C:\Users\acese\AppData\Local\Temp\opencode\`. The only repo file written is
this report.

## Verdict

**F1 is fixed at the `train_bpe` level.** On every corpus tested, the fixed
implementation produced exactly the same merge sequence — same operands, new
ids, counts, ranks and final words — as an independent reference that
recomputes the full pair-count table from scratch at every step. There is no
first divergence to report for the fixed code. The reconstructed pre-fix code
diverges exactly where the review says it should, including merge 171 of 234 on
the 4,000-token synthetic corpus.

The only residual early-stop defect I found is **F7 in `train_tokenizer`**
(a zero-budget level still breaks the level loop and drops later budgets). It
is independent of F1 and does not affect the v512 experiments (flat uses
`n_levels=1`; `hier2` gets budgets 294/196, neither zero).

## 1. Contract tested

The reference implements the greedy BPE contract the review used:

1. For every word occurrence with frequency `f`, count each adjacent pair
   `(w[i], w[i+1])` with weight `f`.
2. Pick the pair with the largest count; ties are broken by the
   lexicographically smallest pair id, i.e. `min((-count, pair))`.
3. Replace all non-overlapping occurrences left to right with a fresh symbol.
4. Stop when the requested budget is reached or no pair has a positive count.

The comparison checks `(left, right, new, count, rank)` for every rule and the
entire final `words` list. Inputs are valid under the caller contract: positive
frequencies and `next_id > max(sequence ids)`. Invalid inputs (`freq <= 0`,
`next_id` collision, missing expansions) remain unvalidated; see F10 and
W1-06-6.

### Reference implementation

```python
from collections import Counter

def ref_train_bpe(sequences, freqs, n_merges, next_id):
    words = [list(s) for s in sequences]
    rules = []
    for step in range(n_merges):
        counts = Counter()
        for w, f in zip(words, freqs):
            for i in range(len(w) - 1):
                counts[(w[i], w[i + 1])] += f
        counts = {p: c for p, c in counts.items() if c > 0}
        if not counts:                       # genuine early stop
            break
        best_count = max(counts.values())
        pair = min(p for p, c in counts.items() if c == best_count)
        new = next_id + len(rules)
        rules.append((pair[0], pair[1], new, counts[pair], step))
        new_words = []
        for w in words:
            out, i = [], 0
            while i < len(w):
                if i + 1 < len(w) and (w[i], w[i + 1]) == pair:
                    out.append(new)
                    i += 2
                else:
                    out.append(w[i])
                    i += 1
            new_words.append(out)
        words = new_words
    return rules, words
```

The pre-fix control is the algorithm described in the review: `pair_counts` is
decremented on the negative pass without a heap push, and only pairs of the new
sequence are pushed. It reproduces the published minimal repro exactly.

## 2. Exact commands and outputs

### 2.1 Minimal repro and the pre-fix control

```powershell
python C:\Users\acese\AppData\Local\Temp\opencode\w1_06_refcheck.py --buggy-control --small
```

First part of the output:

```text
BUGGY_MINIMAL merges= [(1, 0, 3, 10, 0), (3, 2, 4, 1, 1)]
BUGGY_MINIMAL words= [[4], [3], [0, 2]]
BUGGY_MINIMAL pair_counts= {(0, 2): 4} heap_len= 0
FIXED_MINIMAL merges= [(1, 0, 3, 10), (0, 2, 4, 4), (3, 2, 5, 1)]
FIXED_MINIMAL words= [[5], [3], [4]]
```

The fixed run produces three merges for the five requested, and none of the
three words has an adjacent pair left. The pre-fix run produces two merges and
stops with `pair_counts == {(0, 2): 4}` and an empty heap — the exact failure
the review reported.

### 2.2 Forty-nine small corpora: 40 random + 9 structured

The suite includes alphabets of size 2–8, 1–35 words, word lengths 0–12,
frequencies 1–7, duplicate words, `n_merges` from 0 to 1000, empty words,
single-character words, all-tie corpora, overlapping `aaaa`/`abab`, and the
minimal repro. The output is the exact run output:

```text
SMALL_CASES total=49 pass=49 fail=0
SMALL_STEPS_COMPARED=535 TIE_STEPS=338
  rand00               req=3    got=3   ref=3   ties=1   PASS
  rand01               req=1000 got=1   ref=1   ties=0   PASS
  rand02               req=0    got=0   ref=0   ties=0   PASS
  rand03               req=40   got=13  ref=13  ties=4   PASS
  rand04               req=40   got=28  ref=28  ties=19  PASS
  rand05               req=1000 got=60  ref=60  ties=43  PASS
  rand06               req=0    got=0   ref=0   ties=0   PASS
  rand07               req=0    got=0   ref=0   ties=0   PASS
  rand08               req=80   got=17  ref=17  ties=9   PASS
  rand09               req=2    got=0   ref=0   ties=0   PASS
  rand10               req=0    got=0   ref=0   ties=0   PASS
  rand11               req=10   got=0   ref=0   ties=0   PASS
  rand12               req=3    got=3   ref=3   ties=2   PASS
  rand13               req=80   got=73  ref=73  ties=59  PASS
  rand14               req=1000 got=1   ref=1   ties=0   PASS
  rand15               req=5    got=5   ref=5   ties=2   PASS
  rand16               req=1000 got=54  ref=54  ties=39  PASS
  rand17               req=10   got=10  ref=10  ties=3   PASS
  rand18               req=3    got=3   ref=3   ties=0   PASS
  rand19               req=40   got=3   ref=3   ties=0   PASS
  rand20               req=2    got=2   ref=2   ties=0   PASS
  rand21               req=3    got=3   ref=3   ties=1   PASS
  rand22               req=1    got=1   ref=1   ties=0   PASS
  rand23               req=1000 got=33  ref=33  ties=19  PASS
  rand24               req=2    got=2   ref=2   ties=1   PASS
  rand25               req=1    got=0   ref=0   ties=0   PASS
  rand26               req=1000 got=8   ref=8   ties=1   PASS
  rand27               req=20   got=0   ref=0   ties=0   PASS
  rand28               req=80   got=75  ref=75  ties=60  PASS
  rand29               req=1    got=1   ref=1   ties=0   PASS
  rand30               req=1000 got=31  ref=31  ties=16  PASS
  rand31               req=20   got=0   ref=0   ties=0   PASS
  rand32               req=2    got=2   ref=2   ties=1   PASS
  rand33               req=20   got=20  ref=20  ties=14  PASS
  rand34               req=40   got=7   ref=7   ties=5   PASS
  rand35               req=80   got=12  ref=12  ties=6   PASS
  rand36               req=1    got=1   ref=1   ties=1   PASS
  rand37               req=40   got=40  ref=40  ties=26  PASS
  rand38               req=80   got=2   ref=2   ties=0   PASS
  rand39               req=3    got=3   ref=3   ties=1   PASS
  minimal-review       req=5    got=3   ref=3   ties=0   PASS
  all-ties             req=20   got=4   ref=4   ties=3   PASS
  overlap-aaaa         req=20   got=3   ref=3   ties=1   PASS
  overlap-abab         req=20   got=3   ref=3   ties=0   PASS
  one-word-one-token   req=20   got=0   ref=0   ties=0   PASS
  empty-words          req=20   got=0   ref=0   ties=0   PASS
  freq-skew            req=20   got=3   ref=3   ties=1   PASS
  no-pairs             req=5    got=0   ref=0   ties=0   PASS
  single-char-alphabet req=20   got=2   ref=2   ties=0   PASS
```

Thirty of the 49 cases stop before their requested budget; in every one the
stop step and final words match the reference, and the reference state has no
positive-count pair. Three hundred thirty-eight tie decisions all match the
lexicographic rule.

### 2.3 One thousand random cases

The same comparison on 1,000 additional random corpora, with alphabets 1–10,
0–15-long words, frequencies 1–100, and budgets 0–200:

```text
FUZZ cases=1000 fail=0 steps=8802 tie_steps=3462 early_stop_cases=543 max_merges=162
```

Five hundred forty-three of these stop early; all match the reference stop. No
failure was found. Including the 49-case suite, that is 535 + 8,802 = 9,337
small-corpus decisions, of which 3,800 are ties.

### 2.4 Direct heap-frontier invariant

At every `heappop` inside `train_bpe`, for every live pair `p` with count `c`,
the heap must contain at least one `(-c, p)` entry. The harness tracks heap
entries with a counter and checks this at every pop. It also checks that
`pair_words` is exact after every positive update and that final `pair_counts` /
`pair_words` match a recount from the final words.

```powershell
python C:\Users\acese\AppData\Local\Temp\opencode\w1_06_invariants.py
```

Output:

```text
FIXED invariant run
  cases=26 pops=740 missing_live_events=0 pair_words_violations=0 max_heap=70 positive_updates_checked=321
  minimal: final_counts_exact=True final_pair_words_exact=True extra_nonempty_buckets=0
  r00: final_counts_exact=True final_pair_words_exact=True extra_nonempty_buckets=0
  r01: final_counts_exact=True final_pair_words_exact=True extra_nonempty_buckets=0
  r02: final_counts_exact=True final_pair_words_exact=True extra_nonempty_buckets=0
  r03: final_counts_exact=True final_pair_words_exact=True extra_nonempty_buckets=0
  r04: final_counts_exact=True final_pair_words_exact=True extra_nonempty_buckets=0
BUGGY control run
  minimal: MISSING_LIVE_PAIR detected
  ...
  cases=26 pops=558 missing_live_events=264 max_heap=41
  FIRST MISSING (buggy_train_bpe): pairs=[(0, 2)] heap_len=2 counts={(0, 2): 4, (3, 2): 1}
```

The fixed run has zero missing-live events in 740 pops and zero `pair_words`
violations in 321 positive updates. The buggy control fails in 264 events; the
first is the minimal repro.

A separate scale run instrumented only the heap invariant on the 4,000-token
synthetic corpus at 234 merges. It completed with:

```text
HEAP_INVARIANT tokens=4300 v256 merges=234 checks=51297 missing_events=0 pushes=71323 pops=51297 max_heap=39836 seconds=214.1
```

I started the 490-merge variant of this same instrumented run, but the harness
was killed by a 300 s tool timeout after the 234-merge line; the 490-merge case
is covered by the full-sequence comparison in §2.6, which is a stronger check
(any missing frontier entry would change the selected rule).

### 2.5 Synthetic corpus, 4,000 tokens (the review's fast 234-merge repro)

Command:

```powershell
python C:\Users\acese\AppData\Local\Temp\opencode\w1_06_refcheck.py --synthetic --tokens 4000 --buggy-upto 512
```

Output:

```text
SYNTH tokens=4300 unique_words=1096 chars=20 unk=20 eow=21 sha256_text=f4b6967d893c35da
SYNTH v256 requested=234 got=234 ref=234 ties=121 status=PASS seconds=18.3
SYNTH v256 train_tokenizer rules=234 vocab_actual=256 target=256 rules_equal_ref=True
SYNTH v256 BUGGY merges=234 first_divergence=171 buggy_last=(7, 2, 255, 16, 233) ref_last=(7, 2, 255, 16, 233)
  buggy chose (24, 185, 193, 23, 171) vs ref (17, 60, 193, 23, 171)
SYNTH v512 requested=490 got=490 ref=490 ties=366 status=PASS seconds=31.7
SYNTH v512 train_tokenizer rules=490 vocab_actual=512 target=512 rules_equal_ref=True
SYNTH v512 BUGGY merges=490 first_divergence=171 buggy_last=(485, 59, 511, 5, 489) ref_last=(473, 40, 511, 5, 489)
  buggy chose (24, 185, 193, 23, 171) vs ref (17, 60, 193, 23, 171)
```

This is the exact case in the review: first divergence at merge 171 of 234.
The fixed code is 234/234 and 490/490 against the reference. Note that the
pre-fix choice `(24, 185)` and the reference choice `(17, 60)` both have count
23; this divergence is invisible to a count-only spot check and requires the
full merge sequence.

### 2.6 Synthetic corpus, default 120,000 tokens, 234 and 490 merges

Command:

```powershell
python C:\Users\acese\AppData\Local\Temp\opencode\w1_06_refcheck.py --synthetic --buggy-upto 256
```

Output:

```text
SYNTH tokens=128408 unique_words=7479 chars=20 unk=20 eow=21 sha256_text=ace1bc6d9b31c4c8
SYNTH v256 requested=234 got=234 ref=234 ties=30 status=PASS seconds=103.0
SYNTH v256 train_tokenizer rules=234 vocab_actual=256 target=256 rules_equal_ref=True
SYNTH v256 BUGGY merges=234 first_divergence=60 buggy_last=(79, 15, 255, 488, 233) ref_last=(14, 21, 255, 489, 233)
  buggy chose (58, 55, 82, 3226, 60) vs ref (28, 58, 82, 3247, 60)
SYNTH v512 requested=490 got=490 ref=490 ties=115 status=PASS seconds=176.4
SYNTH v512 train_tokenizer rules=490 vocab_actual=512 target=512 rules_equal_ref=True
```

The fixed code matches 234/234 and 490/490. The pre-fix code diverges at merge
60 already (it chooses a count-3,226 pair while the reference has a count-3,247
pair), so the v512 rule set was not a small perturbation of the correct one.
The `train_tokenizer` wrapper also returns exactly the reference rules and the
target vocabulary at both sizes.

Wall times are from a machine running many concurrent red-team agents; they
are not performance measurements.

Three additional ID edge cases also passed: non-contiguous ids
(`[[10,20,30],[10,20],[20,30]]`, `next_id=100`), ids whose lexicographic pair
order is not creation order (`[[500,1],[1,500],[500,1,2]]`, `next_id=501`), and
a very large `next_id` (`[[0,1],[2,3]]`, `next_id=1_000_000`):

```text
gapped PASS got 3 ref 3 ties 0
reverse-ids PASS got 3 ref 3 ties 0
huge-next PASS got 2 ref 2 ties 1
```

These are the 8 decisions that complete the 10,325 unique total below.

### 2.7 First-divergence summary

| Case | Fixed vs reference | Pre-fix control |
|---|---|---|
| Minimal repro | 3/3, no divergence | step 1: `(3,2)` count 1 vs `(0,2)` count 4; stops with `(0,2):4` |
| 4k synthetic, 234 merges | 234/234 | step 171: `(24,185)` count 23 vs `(17,60)` count 23 |
| 4k synthetic, 490 merges | 490/490 | step 171 (same prefix) |
| 120k synthetic, 234 merges | 234/234 | step 60: `(58,55)` 3226 vs `(28,58)` 3247 |
| 120k synthetic, 490 merges | 490/490 | not run (would be ≤60; 234 case already diverges) |
| 1,000 fuzz + 49 small | 0 failures | 264 missing-live events in 26 invariant cases |

### 2.8 Integration: budget hierarchy at v512

```powershell
python -m experiments.check_levels --corpus synthetic --vocab 512 --seed 42
```

Output:

```text
corpus=synthetic vocab=512 seed=42
flat levels=[490] rules=490
hier levels=[294, 196] rules=490
common rule triples: 490 / 490
rule sequences identical
level-1 budget b1=294; prefixes identical: True
top pairs at the boundary (flat state):
  'd' + 'ensi': 356
  'or' + 'ub': 355
  'unrin' + 'ex\x00</w>': 352
  'u' + 'reg': 352
  'pre' + 'k': 351
  'dis' + 'opogul\x00</w>': 349
  'inelud' + 'ul\x00</w>': 347
  'subk' + 'uspu': 333
  'dis' + 'gagbe': 331
  'dis' + 'k': 330
sample tokens flat/hier: 1540/1540 identical=True
```

This confirms the corrected flat BPE and the two-level budget partition are
the same merge sequence: the hierarchy is a partition of one greedy run, not a
constraint. That is the documented F9 caveat, not an F1 failure.

## 3. Tie-breaking

The implementation's heap entries are `(-count, pair)`, so the smallest tuple
pops first: maximum count, then lexicographically smallest `pair`. The
reference uses `min(p for p, c in counts.items() if c == max_count)`, i.e. the
same rule. The review's T3 reference uses `(-kv[1], kv[0])`, again the same.

Evidence:

- 338 tie decisions in the 49-case suite, all matched.
- 3,462 tie decisions in the 1,000-case fuzz suite, all matched.
- 121/366 tie decisions at the 4k synthetic 234/490 checkpoints, all matched.
- 30/115 tie decisions at the 120k synthetic 234/490 checkpoints, all matched.

No stale heap entry can change the tie outcome: entries are keyed by pair, and
the fixed code pushes a fresh entry for every count change (including
decrements). The invariant check in §2.4 confirms a valid entry exists for every
live pair at every pop in the fixed code.

**Caveat:** the repository does not specify a tie-break contract. Lexicographic
pair-id order is deterministic but arbitrary. A different tie-break is still a
valid greedy BPE, but produces a different rule sequence because new ids and
later pair identities change. This matters for reproducibility of the v512
morph result; see §5.3.

### 3.1 Alternate tie-break probe

To test whether the v512 morph effect is a tie-break artifact, I ran the
independent reference at v512/490 with the opposite deterministic rule (largest
pair id on a tie) on seed 42, then reused the same level-assignment and
evaluation code.

```powershell
python C:\Users\acese\AppData\Local\Temp\opencode\w1_06_alt_tie.py
```

Output:

```text
ALT_TIE merges=490 ties=114 seconds=72.4
ALT_TIE vocab=512 rules=490
ALT_TIE frequency     tok/ch=0.308725 bpc=2.353696 macroF1=0.866146 microF1=0.866586 L1=294 L2=196 t=21.8s
ALT_TIE morph         tok/ch=0.309344 bpc=2.348314 macroF1=0.870703 microF1=0.871021 L1=294 L2=196 t=21.9s
ALT_TIE random_closed tok/ch=0.320031 bpc=2.454134 macroF1=0.847782 microF1=0.837858 L1=294 L2=196 t=21.6s
ALT_TIE morph-frequency boundary_macro_f1: +0.004557
ALT_TIE morph-frequency boundary_micro_f1: +0.004434
ALT_TIE morph-frequency bits_per_char: -0.005382
```

For seed 42 the canonical result was +0.004123 macro F1, +0.004092 micro F1,
−0.005449 bpc. The alternate tie-break gives a very similar, still positive
effect. This is a one-seed probe, not a full 10-seed robustness test.

## 4. Early stopping

### 4.1 `train_bpe` is correct

- Thirty of the 49 small cases stop before the requested budget; all stop at the
  reference step and leave no positive-count pair.
- Five hundred forty-three of the 1,000 fuzz cases stop early; all match.
- The 4k and 120k synthetic runs reach exactly 234 and 490 merges; no early
  stop.
- The minimal repro fixes the old "heap empty, `pair_counts == {(0,2): 4}`"
  failure: the fixed run merges `(0,2)` second and finishes all three possible
  merges.

### 4.2 F7 remains in `train_tokenizer`

This is a separate, pre-existing bug. `train_tokenizer` still does (current
lines 250-257):

```python
if not merges:
    break
```

so a level with a zero budget is indistinguishable from a level with no
remaining pairs. Repro:

```powershell
python -c "
from ramified.bpe import train_tokenizer, split_budget
t = train_tokenizer(['abcd abcd abcd'], 7, n_levels=2, level_weights=[0.4, 0.6])
print('F7 case A: vocab_actual=', t.vocab_size, 'target=7 levels=', [len(r) for r in t.levels])
print('split_budget(1,[0.4,0.6]) =', split_budget(1,[0.4,0.6]))
t2 = train_tokenizer(['abcdefgh abcdefgh abcdefgh abcdefgh'], 30, n_levels=3, level_weights=[0.1,0.0,0.9])
print('F7 case B: vocab_actual=', t2.vocab_size, 'target=30 levels=', [len(r) for r in t2.levels])
print('split_budget(20,[0.1,0.0,0.9]) =', split_budget(20,[0.1,0.0,0.9]))
"
```

Output:

```text
F7 case A: vocab_actual= 6 target=7 levels= [0]
split_budget(1,[0.4,0.6]) = [0, 1]
F7 case B: vocab_actual= 12 target=30 levels= [2, 0]
split_budget(20,[0.1,0.0,0.9]) = [2, 0, 18]
```

This is not an F1 regression and does not affect the v512 flat/morph
experiments: flat uses one level, and the `hier2` run uses positive budgets
294/196. It does mean the "train_bpe is greedy" result should not be read as
"train_tokenizer always fills the vocabulary."

## 5. Does the v512 morph result survive?

**Yes, in the narrow sense that the archived result is reproducible on the
corrected greedy tokenizer and is not an artifact of F1.** It remains a small,
v512-only effect, and it does not turn the budget hierarchy into a constraint.

### 5.1 Full 10-seed reproduction

I reran the archived 10-seed level-assignment experiment at v512 with the
current code and compared every shared metric to
`results/level_assignment_v512_10seeds.csv`.

```powershell
python -m experiments.level_assignment --seeds 42,43,44 --vocabs 512 `
  --modes frequency,morph,random_closed `
  --out C:\Users\acese\AppData\Local\Temp\opencode\w1_06_la_repro.csv

python -m experiments.level_assignment --seeds 45,46,47,48,49,50,51 --vocabs 512 `
  --modes frequency,morph,random_closed `
  --out C:\Users\acese\AppData\Local\Temp\opencode\w1_06_la_v512_seeds45_51.csv

python C:\Users\acese\AppData\Local\Temp\opencode\w1_06_compare10.py
```

Comparison output:

```text
REPRO rows=40 expected=40 mismatches=0
PAIRED_STATS_REPRO
boundary_f1          morph        -frequency     mean=+0.009243 t=+6.518 CI95=[+0.006036,+0.012451] W/L/T=10/0/0
boundary_f1          random_closed-frequency     mean=-0.006386 t=-1.730 CI95=[-0.014735,+0.001963] W/L/T=3/7/0
boundary_f1          morph        -random_closed mean=+0.015629 t=+3.810 CI95=[+0.006351,+0.024907] W/L/T=9/1/0
boundary_macro_f1    morph        -frequency     mean=+0.009243 t=+6.518 CI95=[+0.006036,+0.012451] W/L/T=10/0/0
boundary_macro_f1    random_closed-frequency     mean=-0.006386 t=-1.730 CI95=[-0.014735,+0.001963] W/L/T=3/7/0
boundary_macro_f1    morph        -random_closed mean=+0.015629 t=+3.810 CI95=[+0.006351,+0.024907] W/L/T=9/1/0
boundary_micro_f1    morph        -frequency     mean=+0.009484 t=+6.352 CI95=[+0.006107,+0.012862] W/L/T=10/0/0
boundary_micro_f1    random_closed-frequency     mean=-0.010637 t=-2.273 CI95=[-0.021221,-0.000054] W/L/T=3/7/0
boundary_micro_f1    morph        -random_closed mean=+0.020122 t=+3.924 CI95=[+0.008521,+0.031722] W/L/T=9/1/0
bits_per_char        morph        -frequency     mean=-0.025788 t=-5.384 CI95=[-0.036623,-0.014952] W/L/T=0/10/0
bits_per_char        random_closed-frequency     mean=+0.036144 t=+3.038 CI95=[+0.009235,+0.063053] W/L/T=8/2/0
bits_per_char        morph        -random_closed mean=-0.061932 t=-4.725 CI95=[-0.091578,-0.032285] W/L/T=0/10/0
```

The 40 rows (10 seeds × 4 modes) have **zero mismatches** on all shared
columns: `n_levels`, `n_l1`, `n_l2`, `tokens_per_char`, `bits_per_char`,
`boundary_f1`, `decode_word`, `decode_doc`, and the six provenance counts. The
archived paired statistics in `results/level_assignment_stats_v512.csv` also
reproduce exactly. `boundary_macro_f1` is the same as the archived
`boundary_f1`; `boundary_micro_f1` was added to the current CSV by a concurrent
edit (see §7) and is not in the archived file. It is positive for
morph−frequency on all 10 seeds and for morph−random_closed on 9/10.

The archived `level_seconds` for flat is 4.78 s, while this run took 36–77 s
per seed under a 100-agent concurrent load. The timings are not comparable; the
metrics are exact.

### 5.2 The flat baseline is reference-correct

The morph result is a reassignment of a flat tokenizer's rules. That flat
tokenizer at v512/490 is the one checked in §2.6 for seed 42:
`train_tokenizer` returns exactly the reference rules, 490/490, and
`check_levels` shows flat and `hier2` are identical 490/490. Therefore the
morph result is built on the corrected greedy merge sequence, not on the pre-fix
frontier. The pre-fix flat tokenizer diverges at merge 60 on this corpus, so a
pre-fix version of this experiment would not be the same experiment.

### 5.3 Caveats that limit the claim

1. **Small effect, narrow window.** +0.0092 macro F1 and −0.0258 bpc at v512,
   with 10/10 seeds, but the findings document reports zero effect at
   v1024/2048. I did not rerun those sizes.
2. **Interpolation, not composition.** The synthetic test triples have all
   component pairs attested in training (README/diario). This measures
   morphological alignment on novel combinations with known parts, not
   compositional generalisation to unseen morphemes.
3. **One random control draw per seed.** `random_closed` uses
   `seed * 1000 + vocab`; seed 42 is the lone morph loss against random_closed.
   A distribution over random draws would be a stronger control.
4. **Tie-break dependence is possible but not observed at seed 42.** The
   alternate tie-break in §3.1 keeps the seed-42 morph effect positive, but a
   full 10-seed alternate run was not done.
5. **The hierarchy is still labels, not constraints (F9).** A positive morph
   result says rule assignment by provenance matters at v512; it does not say
   the budget hierarchy enforces structure.

## 6. Findings by severity

| ID | Severity | Finding | Status / evidence |
|---|---|---|---|
| W1-06-0 | — | **F1: heap frontier loses live pairs** | **Resolved.** 0 divergences in 10,325 unique reference-comparison decisions (10,793 including 234-merge prefix reruns); 52,037 fixed-code frontier checks, zero missing; pre-fix control fails at 171/234. |
| W1-06-1 | **Medium** | **F7 remains:** a zero-budget level breaks `train_tokenizer` and drops later levels' budgets | Vocab 6 vs target 7; vocab 12 vs target 30; `split_budget` allocations `[0,1]` and `[2,0,18]`. Not F1; unaffected flat/morph v512. |
| W1-06-2 | **Low** | No in-repo regression test or property test for F1 | No project `tests/` directory; all verification here is external. The review explicitly recommended adding one. |
| W1-06-3 | **Low** | Tie-break is unspecified and depends on pair-id order | 3,800 small/fuzz tie decisions match lexicographic `(-count, pair)`; alternate tie-break seed-42 probe stays positive; no full 10-seed alternate. |
| W1-06-4 | **Low (perf/memory)** | Heap grows without compaction; every touched pair is pushed on both passes | 4k/234 invariant run: 71,323 pushes, 51,297 pops, max heap 39,836. F5's `pair_words` pruning is fixed, but heap memory is still unbounded in the merge budget. No correctness impact found. |
| W1-06-5 | **Info** | F5 appears fixed as a side effect | `pair_words` exact after 321 positive updates; final counts/index exact in six checked runs; zero extra non-empty buckets. |
| W1-06-6 | **Info / out of scope** | F6, F10, F11 remain from the review | `split_budget` normalisation/empty/negative cases untouched; input validation, unknown-id, EOW/decode issues untouched. The v0.2 v512 results use the chunk codec and `n_levels=1`, so F2/F3 do not invalidate them. |

No Critical or High finding remains in `train_bpe`. The residual F7 is a
`train_tokenizer` bug, not a failure of the F1 fix.

## 7. Workspace drift observed during verification

The workspace is shared with other agents. During this run:

- `ramified/bpe.py` never changed: SHA-256 `7309...` before and after, 278
  lines, mtime 22:02:02.
- `ramified/metrics.py` changed at 22:22:42 and
  `experiments/level_assignment.py` at 22:23:04. The current
  `level_assignment.py` emits `boundary_macro_*` / `boundary_micro_*` columns
  and keeps `boundary_f1` as an alias for macro F1. The archived CSV has the
  older `boundary_precision` / `boundary_recall` columns. All shared metrics
  reproduce exactly.
- `metrics.bits_per_char` now uses add-1 smoothing and `total + vocab_size`;
  `level_assignment.py` uses its own local `bpc_alpha` (add 0.1), so the v512
  bpc reproduction is unaffected. Other experiments that import
  `metrics.bits_per_char` may not be comparable to older CSVs; that is outside
  this F1 scope.
- The concurrent 100-agent load makes wall times meaningless. No timing claim
  in this report is a performance result.

## 8. What would change my mind

For the **F1 verdict**:

- A full fixed-vs-reference comparison on the real code corpus
  (`ramified.corpora.load_code_corpus`) at a larger budget (v2048 or v4096),
  where the longer chunks and punctuation produce many more frontier updates.
  My synthetic tests reach 490 merges and 128k tokens; the real corpus is the
  remaining distribution shift.
- A second independent implementation that does not recompute from scratch —
  e.g. a linked-list plus versioned heap, or a trie — diverging from the fixed
  code on any corpus. That would show a shared blind spot between the current
  code and my reference.
- A failure of the frontier invariant at 490 merges on the default corpus with
  a longer-running instrumented harness. The 234-merge instrumented run passed
  51,297 checks; the 490 case passed behaviourally, which is stronger for
  selection but does not inspect internal entries.
- Any valid input outside the tested domain: non-positive frequencies,
  `next_id` collisions, or missing expansions. These are unvalidated (F10); if
  the API is expected to accept them, the greedy guarantee does not hold.

For the **v512 morph claim**:

- A full 10-seed alternate-tie-break run; if the morph−frequency effect
  disappears or reverses under a different valid tie-break, the result is not
  robust.
- A pair-disjoint split (hold out `(prefix, root)` or `(root, suffix)` pairs,
  not just triples); if the effect disappears, the current result is
  interpolation, not composition.
- Multiple random-control draws per seed; seed 42 is already a morph loss
  against one random draw, so the 9/1 aggregate may be sensitive to the draw.

**Single check that would most change the F1 verdict:** run the fixed code and
the recompute reference on the real code corpus at v2048/v4096 and compare the
complete rule sequence. If that passes, I would raise confidence from high to
very high; a divergence would reopen F1 despite all synthetic passes.

## 9. Reproduction notes

- The harnesses used here are:
  `C:\Users\acese\AppData\Local\Temp\opencode\w1_06_refcheck.py`,
  `w1_06_invariants.py`, `w1_06_heapinv.py`, `w1_06_fuzz.py`,
  `w1_06_alt_tie.py`, `w1_06_compare10.py`, `w1_06_deltas.py`,
  `w1_06_determinism.py`.
- The reference core is reproduced in §1; the buggy control is the algorithm
  from the review, reconstructed and validated by matching the published
  minimal repro and the 171/234 divergence.
- The comparison treats `next_id` as `max(sequence ids) + 1` for direct
  `train_bpe` tests, and uses the real `eow_id + 1` for `train_tokenizer`
  tests. New ids therefore match exactly.
- All random suites use fixed seeds; the exact output above is deterministic
  for the given Python and `bpe.py` hash. As a check that the set iteration
  order in `pair_words` does not leak into the result, the 4k/234 rule-sequence
  hash was identical under `PYTHONHASHSEED=0,1,12345`:
  `34f4943125d9b6b19fada287c6b42dc7` (234 rules) in all three runs.
- The 99 other red-team agents are not part of these results; no shared state
  is used except the project source and the synthetic generator.

## Bottom line

The F1 fix is correct at the level it was made. The fixed `train_bpe` is a
greedy weighted BPE under the tested contract: exact counts, valid frontier,
deterministic tie-break, and no early stop while a positive pair remains. The
pre-fix implementation still reproduces the review's failure, so the harness is
sensitive. The v512 morph result is reproducible on the corrected tokenizer
(40/40 rows, 10/10 seeds, 0 mismatches), but it remains a small, v512-only
rule-assignment effect; it is not evidence for the budget hierarchy as a
constraint. The only medium residual is F7 in `train_tokenizer`, which is
independent of F1.
