# W1-09 — Performance profile of the F1 heap fix

**Agent:** W1-09 (red team, read-only)
**Date:** 2026-11-22
**Snapshot profiled:** `ramified/bpe.py`, 278 lines, SHA-256
`73093A3E62FC66A59594F29CDB8549F4523D358DF04EBAB46E412C7C47C5B0C2`
(mtime 2026-11-22 22:02:02). This is the fixed revision: `_update_pair_counts`
returns every touched pair, the caller pushes every count change (including
decrements), and `pair_words` is pruned on negative/zero updates.
**Review snapshot:** the adversarial review describes the pre-fix revision,
255 lines, SHA-256 `F4E6F2C8...`. Line numbers in the review do not map to the
profiled revision.
**Companion reports:** W1-06 (independent reference check), W1-07 (findings
triage), W1-16/W1-20 (paired statistics). This report does not re-litigate
correctness beyond what is needed to interpret the timings.
**Constraint:** no project source was modified. All harnesses ran from
`C:\Users\acese\AppData\Local\Temp\opencode\w1_09\` and only wrote there. The
only repository write is this report.
**Environment:** Python 3.13.13, Windows 11, PowerShell, 16 logical cores,
16 GB RAM. The host was running many concurrent red-team agents. Wall-clock
times fluctuated between about 1.1× and 13× the process CPU time during the
session; the main uninstrumented curve (§2.1) was 6–13×. **Process CPU time is
the primary timing metric**; wall times are reported only to document the
noise.

---

## Verdict

1. **F1 is fixed, and the fix is greedy-correct in the profiled runs.** The
   current trainer matched a full-recount reference on 4/4 large-corpus
   truncations (synthetic and code, 234 and 490 merges), on 300/300 random
   small corpora, and on the minimal repro. At every tested size, final
   `pair_counts` matched an exact recount, and every positive pair had at least
   one valid heap entry. No remaining early-stop or invisible-pair event was
   found.
2. **F5's `pair_words` pruning works.** In all 10 profiled runs the index was
   exactly the live `(pair, word)` set: 0 stale keys, 0 missing keys, 0 stale
   entries, 0 missing entries. The review's T10 reproduction drops from 674
   keys / 1588 entries / 259 stale keys to 408 keys / 428 entries / 0 stale.
3. **Pushing every count change does not blow up the heap on the two benchmark
   corpora at 234/490/1002/2026/4074 merges.** Peak heap is 282,362 entries on
   synthetic and 532,090 on code, essentially constant across merge counts
   because the peak occurs early and later merges drain the frontier. Peak
   process RSS delta is ~51 MB / ~99 MB; tracemalloc peak at 490 merges is
   45.5 MB / 85.6 MB. This is ~2× the reconstructed buggy heap, not a blowup.
4. **The caveat is long chunks.** The frontier is not memory-bounded. A
   500-word × 120-character chunk stress reaches 1.64M heap entries at 500
   merges (66× the 24,747 live pairs) and 1.68M at 1,000 merges; it does not
   drain, and costs 52–60 CPU s for only 60k tokens. The benchmark corpora avoid
   this because their chunks are short (code mean 6.2, max 122; synthetic max
   10). Minified code or long unbroken lines remain a real memory/runtime cliff.
5. **The 20-agent matrix timings are invalid as scaling/performance evidence.**
   They are `time.perf_counter()` wall times measured while 20 agents ran in
   parallel (`run_matrix_v2.py:147-155`; `diario.md` §1c and “Errori da non
   ripetere” #4). Their medians are **3.4–4.3× the current serial CPU time** at
   the same corpus/merge count, while the fixed trainer is ~2× slower than the
   buggy one; the excess is therefore not algorithmic. Within-cell seed spread
   is up to 1.7× (code v2048: 35.92/59.17/60.90 s). Keep the matrix CSVs as
   history; do not use their `train_seconds` in the paper.
6. **The v512 morph result is computationally credible.** It uses the fixed
   flat tokenizer, is deterministic, and seed 42 reproduces exactly on every
   shared scientific field. Flat v512 training costs ~5.5 CPU s; the full
   10-seed experiment is ~1 CPU minute of training plus evaluation, not a
   computational barrier. The statistical/scientific caveats (small synthetic
   effect, one random-control draw, budget-fraction sensitivity) are covered by
   W1-16 and W1-20 and are not timing problems.

**Confidence:** high (~0.95) on F1/F5 correctness and benchmark heap behaviour;
high (~0.9) that the matrix timings are unusable; medium-high (~0.8) on the
long-chunk extrapolation, because the stress corpus is adversarial rather than
the project corpus; high (~0.9) that the v512 morph computation is reproducible
and cheap, medium on its scientific scope.

---

## 1. What was measured

### 1.1 Corpora

The profiler uses the same construction as the experiments, not a toy corpus.

| corpus | source | train chunks | chars | unique chunks | alphabet | merge targets (vocab actual) |
|---|---|---:|---:|---:|---:|---|
| synthetic | `generate_synthetic(seed=42)` default 120k tokens | 128,408 | 1,148,566 | 7,479 | 20 | 234/490/1002/2026/4074 → 256/512/1024/2048/4096 |
| code | `load_code_corpus()`, 90/10 line split | 39,232 | 360,014 | 9,564 | 94 | 234/490/1002/2026/4074 → 330/586/1098/2122/4170 |

`split="chunks"` throughout. The synthetic targets are exactly the v0.2
vocabulary cells (alphabet 20 + 2 + merges). The code targets use the requested
merge counts, which are slightly above the matrix code cells for v256/v512; for
the matrix comparison §2.6 additionally times code at 160/416/928/1952/4000
merges.

Chunk lengths explain the heap result: code mean 6.17, p99 30, max 122;
synthetic mean 7.94, p99 10, max 10. There are no long chunks in either
benchmark corpus.

### 1.2 Harness

`profile_one.py` monkeypatches `heapq.heapify`, `heapq.heappush` and
`heapq.heappop` to capture the actual list used by `train_bpe`, count pushes and
pops, and track its peak length. It also captures the live `Counter` and
`defaultdict` objects. After training it:

- recounts every adjacent pair from the final `words` and compares it with
  `pair_counts` (keys and values);
- builds the live `(pair, word)` set and compares it with `pair_words`;
- scans the final heap and checks that every live pair has a valid
  `(-count, pair)` entry;
- estimates heap/list bytes and samples process RSS with a 10 ms thread.

`time_one.py` runs the same trainer **without instrumentation** and records
`time.perf_counter()` and `time.process_time()`. `stress_one.py` applies the
same heap instrumentation to adversarial long chunks. None of these scripts
writes to the repository.

Timing caveat: the host was heavily loaded by the 100-agent red-team session.
For example, the uninstrumented synthetic 490-merge run used 5.5 CPU s but 44.5
wall s; the code 4074-merge run used 20.2 CPU s but 131.9 wall s. **Any
wall-clock number in this report, including the matrix’s, is contaminated.**
CPU time is stable to roughly 10–20% across repeated runs and is used for all
comparisons.

---

## 2. Results

### 2.1 Training time (uninstrumented)

| corpus | merges | vocab actual | CPU s | wall s | CPU ms/merge |
|---|---:|---:|---:|---:|---:|
| synthetic | 234 | 256 | 4.781 | 39.858 | 20.4 |
| synthetic | 490 | 512 | 5.500 | 44.534 | 11.2 |
| synthetic | 1002 | 1024 | 6.203 | 61.182 | 6.2 |
| synthetic | 2026 | 2048 | 6.125 | 78.558 | 3.0 |
| synthetic | 4074 | 4096 | 5.875 | 76.329 | 1.4 |
| code | 234 | 330 | 8.281 | 89.787 | 35.4 |
| code | 490 | 586 | 11.594 | 85.755 | 23.7 |
| code | 1002 | 1098 | 14.906 | 90.575 | 14.9 |
| code | 2026 | 2122 | 18.547 | 123.314 | 9.2 |
| code | 4074 | 4170 | 20.172 | 131.922 | 5.0 |

Observations:

- Synthetic training is **flat to sublinear** after 490 merges: 5.5 → 5.9 CPU s
  from 490 to 4074 merges. Late merges touch few surviving word types.
- Code training grows sublinearly, with a decreasing per-merge cost:
  8.3 → 20.2 CPU s from 234 to 4074 merges. No time blowup at the requested
  sizes.
- Wall/CPU ratios are 6–13× in this table; the wall columns are not usable.
  The matrix timings suffer from the same problem (§2.6).
- The fix costs CPU relative to the buggy reconstruction: fixed/buggy is
  1.87–2.04× on synthetic and 1.84–2.16× on code (buggy `train_bpe` CPU:
  synthetic 2.56/2.69/3.09 s at 234/490/4074; code 3.94/5.38/9.20 s at
  234/490/4000). That is the expected price of pushing every changed count and
  is small at these corpus sizes.

### 2.2 Heap size and frontier composition

Heap peak is the maximum length of the list `train_bpe` uses; heap final is its
length when the requested merge budget is exhausted. `peak/live` is peak heap
divided by the final number of live pairs (not measured at the same instant; it
is a scale indicator, not an instantaneous ratio). RSS delta is peak process
RSS minus pre-training RSS; tracemalloc peak is shown for the two 490-merge
runs.

| corpus | merges | heap peak | heap final | pushes | pops | live pairs | peak/live | stale at end | RSS delta MB |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| synthetic | 234 | 282,362 | 151,021 | 495,825 | 345,008 | 2,233 | 126× | 142,230 | 51.2 |
| synthetic | 490 | 282,362 | 92,165 | 530,298 | 438,337 | 3,371 | 84× | 87,604 | 51.5 |
| synthetic | 1002 | 282,362 | 55,298 | 538,552 | 483,458 | 5,071 | 56× | 49,407 | 51.8 |
| synthetic | 2026 | 282,362 | 34,405 | 547,612 | 513,411 | 6,634 | 43× | 27,397 | 51.9 |
| synthetic | 4074 | 282,362 | 14,260 | 549,499 | 535,443 | 4,897 | 58× | 9,334 | 51.4 |
| code | 234 | 532,090 | 513,545 | 985,442 | 474,815 | 8,358 | 64× | 415,741 | 98.7 |
| code | 490 | 532,090 | 483,971 | 1,162,931 | 681,878 | 10,740 | 50× | 392,602 | 99.0 |
| code | 1002 | 532,090 | 413,616 | 1,306,016 | 895,318 | 13,340 | 40× | 332,598 | 99.5 |
| code | 2026 | 532,090 | 323,316 | 1,413,817 | 1,093,419 | 15,067 | 35× | 256,257 | 98.8 |
| code | 4074 | 532,090 | 212,421 | 1,481,347 | 1,271,844 | 15,172 | 35× | 163,110 | 99.4 |

Instrumented runs have roughly 2–5 CPU s of wrapper overhead, so their CPU times
are not used in §2.1. Heap lengths and counts are unaffected by the wrappers.

Key observations:

- **No blowup at the requested merge counts.** Peak heap is constant at
  282,362 (synthetic) / 532,090 (code) for every merge budget. The peak is
  reached early, then the heap drains; the 4074-merge run ends with only
  14,260 / 212,421 entries.
- **But the frontier is mostly stale.** At the end of the 234-merge synthetic
  run, 142,230 of 151,021 entries are stale (94%). For code 234, 415,741 of
  513,545 are stale (81%). Every merge pushes both the old word’s pairs
  (negative pass) and the new word’s pairs (positive pass), so the heap is
  effectively an append-only event log that is filtered lazily.
- **Memory is modest on these corpora.** Tracemalloc peak: 45.5 MB synthetic
  490, 85.6 MB code 490. RSS delta is 51.5 / 99.0 MB and is flat across merge
  counts. The heap is the dominant allocation, but 0.28–0.53M entries is not a
  practical problem on this machine.
- **The fix roughly doubles heap work.** Buggy reconstructed peaks: 126,151
  (synthetic 490) and 270,429 (code 490); fixed peaks are 282,362 / 532,090
  (2.24× / 1.97×). Pushes are 240,114 / 558,531 buggy vs 530,298 / 1,162,931
  fixed (2.21× / 2.08×).
- The 4,000-token synthetic variant is much smaller: 39,836 peak / 20,230
  final / 3.7 MB RSS delta at 234 merges, matching W1-06’s independent
  `max_heap=39836`. It exhausts after 2,043 merges, at which point both
  `pair_counts` and the heap are empty (no early stop with live pairs).

### 2.3 Correctness invariants observed during profiling

For every row in §2.2:

| check | result |
|---|---|
| `pair_counts` vs exact recount (keys + values) | 0 mismatched pairs, max abs diff 0 |
| `pair_words` stale keys / missing keys | 0 / 0 |
| `pair_words` stale entries / missing entries | 0 / 0 |
| live pairs with no valid heap entry | 0 |
| selected merge counts monotone | true |

Additional checks:

- **Minimal repro** (`sequences=[[1,0,2],[1,0],[0,2]]`, `freqs=[1,9,4]`,
  5 merges): `[(1,0,3,10), (0,2,4,4), (3,2,5,1)]`, final `pair_counts={}`,
  `pair_words={}`, heap length 0. The old failure (2 of 5 merges, `(0,2):4`
  stranded) is gone.
- **Reference comparison**: synthetic 234/234 and 490/490, code 234/234 and
  490/490 selected the same `(left, right, count)` at every step as a
  full-recount greedy reference. No first divergence.
- **Property test**: 300 random corpora (alphabet 3–9, 30–200 words, word
  lengths 1–8, frequencies 1–5, 60 merges) matched the reference exactly:
  `trials=300 bad=0`.
- **`check_levels`** (synthetic v512 seed 42): flat 490 rules, hier2
  294+196; common triples 490/490; rule sequences identical; 1,540/1,540
  sample tokens identical. This reproduces the findings-v0.2 claim on the
  fixed revision.

### 2.4 `pair_words` pruning

The review’s T10 random corpus (300 words, alphabet 8, 100 merges) now gives:

```text
merges learned: 100
pair_words keys: 408 actual live pair keys: 408
pair_words (pair,word) entries: 428
keys whose set is empty: 0
stale keys (not in final adjacency set): 0
missing keys: 0
stale entries: 0 missing entries: 0
```

Compare the review snapshot: 674 keys, 1,588 entries, 259 stale keys. The
current negative pass discards `wi` for every old adjacent pair and pops
zero-count pairs; the positive pass re-adds surviving pairs. The index is exact.
W1-07’s independent run reports 409 keys / 433 entries / 0 stale, consistent
with this measurement. No further `pair_words` work is needed.

### 2.5 Long-chunk stress: the frontier is not bounded

The benchmark corpora have short chunks. To probe the mechanism, `stress_one.py`
uses 500 random chunks of length 120 over a 26-letter alphabet (frequencies
1–5), and a second corpus of 5,000 chunks of length 30 over 40 letters. These
are adversarial but plausible for minified code, base64, URLs, or long
identifiers.

| corpus | merges | pushes | heap peak | heap final | live pairs | peak/live | CPU s |
|---|---:|---:|---:|---:|---:|---:|---:|
| 500 × 120 | 500 | 4,058,557 | 1,636,566 | 1,636,566 | 24,747 | 66× | 52.4 |
| 500 × 120 | 1000 | 4,310,615 | 1,681,050 | 1,377,401 | not measured | — | 59.9 |
| 5,000 × 30 | 500 | 1,989,670 | 1,085,638 | 814,085 | 41,529 | 26× | 24.1 |

At 500 merges on 60k tokens the heap is 1.64M entries and **still at its peak
at the end**: it is not draining. At 1,000 merges it peaks at 1.68M and ends at
1.38M. For comparison, the entire 360k-character code corpus at 4,074 merges
peaks at 532k entries and ends at 212k. The long-chunk corpus costs 52–60 CPU s
for 60k input characters, one-sixth of the code corpus’s 360k characters.

Interpretation: the push-every-change policy is correctness-preserving but has
memory proportional to the number of frontier events, not to the number of live
pairs. On short-chunk corpora the event stream is small and the stale entries
are popped fast enough. On long chunks the event stream is ~2 × word length per
affected word, the low-count entries never reach the top, and the heap becomes a
large stale log. This is the F8 “versioned heap / periodic compaction” gap, not
a new correctness bug. At 4,074 merges on a production long-chunk corpus, a
multi-GB heap is plausible; the current benchmark does not exercise it.

### 2.6 The 20-agent matrix timings

Source: `results/matrix/flat_*/results.csv`, 3 seeds × 2 corpora per cell.
`run_matrix_v2.py:147-155` uses `time.perf_counter()` around
`train_tokenizer`; the harness reports `train_seconds=round(train_s, 2)`. There
is no CPU time, repetition within a seed, median or confidence interval.
`diario.md` §1c records that a cell is one of 20 agents and that 120 rows were
produced by those agents; “Errori da non ripetere” #4 explicitly says the
`train_seconds` of the matrix are not a scaling result.

Comparison with serial CPU at the same corpus/merge count:

| matrix cell | merges | matrix median wall s | fixed serial CPU s | ratio |
|---|---:|---:|---:|---:|
| synthetic v256 | 234 | 17.90 | 4.781 | 3.74× |
| synthetic v512 | 490 | 20.34 | 5.500 | 3.70× |
| synthetic v1024 | 1002 | 21.18 | 6.203 | 3.41× |
| synthetic v2048 | 2026 | 22.03 | 6.125 | 3.60× |
| synthetic v4096 | 4074 | 22.89 | 5.875 | 3.90× |
| code v256 | 160 | 27.97 | 6.484 | 4.31× |
| code v512 | 416 | 38.91 | 9.922 | 3.92× |
| code v1024 | 928 | 48.22 | 13.062 | 3.69× |
| code v2048 | 1952 | 59.17 | 15.734 | 3.76× |
| code v4096 | 4000 | 64.91 | 16.906 | 3.84× |

The excess is not caused by the F1 fix: the fixed trainer is the *slower* one
(1.8–2.2× the reconstructed buggy CPU), so a serial buggy run should be
*faster* than these fixed CPU numbers, not 3.4–4.3× slower. The matrix is also
the buggy revision, so its merge sequences are not the current ones (F1). It is
therefore invalid both as a timing measurement and as an algorithmic result;
the algorithmic invalidity is already the subject of findings-v0.2.

The hierarchical cells show the same range (e.g., hier2 v512 synthetic
16.36 / 20.02 / 19.85 s, code 33.70 / 34.59 / 37.80 s), so this is not a
flat-specific artifact. The seed spread inside a cell is itself larger than
most method effects: code v2048 is 35.92 / 59.17 / 60.90 s (1.7×), code v4096
is 42.54 / 64.91 / 68.77 s (1.6×), while a whole method comparison at a fixed
cell is typically <20%. No relative timing claim survives that noise.

### 2.7 v512 morph result: computational credibility

The morph result is produced by `experiments/level_assignment.py`: train one
flat tokenizer per seed at v512 (490 merges), collect provenance, then reassign
the same rules to two levels and evaluate. It does not retrain per mode.

Seed 42 reproduction (current code, current corpus generator):

| mode | tok/char | bpc | boundary F1 | boundary P | boundary R |
|---|---:|---:|---:|---:|---:|
| flat | 0.308828 | 2.353869 | 0.866508 | 0.888889 | 0.859375 |
| frequency | 0.308828 | 2.353869 | 0.866508 | 0.888889 | 0.859375 |
| morph | 0.309396 | 2.348421 | **0.870631** | 0.892578 | 0.863715 |
| random_closed | 0.311822 | 2.349558 | 0.875723 | 0.895110 | 0.871962 |

These values are bit-for-bit identical to the seed-42 rows of
`results/level_assignment_v512_10seeds.csv` on all shared scientific columns.
Only `level_seconds` differs (archived 0.0, re-run 0.01–0.02 s), which is just
timer resolution around the assignment step; it does not include training or
evaluation. A full 40-row reproduction is reported independently by W1-06 with
zero mismatches.

The published paired statistics also recompute exactly:

```text
morph - frequency    F1  +0.009243  t=+6.518  10/0/0  CI95 [+0.00604,+0.01245]
morph - frequency    bpc -0.025788  t=-5.384  10/0/0  CI95 [-0.03662,-0.01495]
morph - random_closed F1 +0.015629  t=+3.810   9/1/0  CI95 [+0.00635,+0.02491]
random_closed - frequency F1 -0.006386 t=-1.730 3/7/0
```

Seed 42 is the one published seed where `morph` loses to `random_closed` on F1
(0.8706 vs 0.8757); the 10-seed mean still favors morph by +0.0156. This is a
control-draw caveat, not a computational failure; W1-16 and W1-20 discuss its
statistical consequences.

Cost: flat v512 training is 5.5 CPU s (490 merges; 4.8 CPU s at v256, 6.2 at
v1024). Ten seeds therefore need roughly 55 CPU s of BPE training plus
provenance/evaluation. Even at the wall inflation observed in this session (up
to 13×), the complete 10-seed level-assignment experiment is a few minutes, not
hours. The result is computationally credible and easily reproducible.

Caveats that are *not* timing problems: the effect is small (+0.9 F1 points,
−0.026 bpc), synthetic-only, conditional on the 0.6 level-1 fraction, and the
secondary `morph - random_closed` contrast does not survive full multiplicity
correction (W1-20). Do not upgrade “computationally credible” to “scientifically
general.”

---

## 3. Findings

| ID | Severity | Finding |
|---|---|---|
| W1-09-1 | — | **F1 fix verified.** 4/4 large reference truncations, 300/300 random corpora, 0 count mismatches, 0 live pairs without a heap entry. |
| W1-09-2 | — | **F5 `pair_words` pruning exact.** 0 stale/missing keys or entries in 10/10 runs; T10 drops from 674/1588/259 to 408/428/0. |
| W1-09-3 | **Low (perf)** | **No benchmark heap blowup.** Peaks 282k/532k entries, 45–99 MB; flat across 234–4074 merges; ~2× the buggy heap. |
| W1-09-4 | **Medium (perf/memory)** | **Long-chunk frontier is unbounded.** 500×120 stress: 1.64M entries at 500 merges, 1.68M at 1000, 66× live pairs, 52–60 CPU s for 60k tokens. Benchmarks avoid it because chunks are short. A versioned heap or periodic compaction is still needed before claiming production robustness. |
| W1-09-5 | **Medium (validity)** | **Matrix timings invalid for scaling.** Wall-clock under 20-agent contention; 3.4–4.3× serial CPU; 1.6–1.7× seed spread; buggy revision. Do not use `train_seconds` from `results/matrix/`. |
| W1-09-6 | **Info** | **v512 morph is computationally credible.** Deterministic, exact seed-42 reproduction, ~5.5 CPU s per seed, ~1 CPU min for 10 seeds; statistical scope remains narrow (W1-16/W1-20). |
| W1-09-7 | **Info** | **Timing methodology.** `time.perf_counter()` alone is unusable on this shared host; record `time.process_time()` or run on an isolated machine for any future scaling table. |

No new correctness defect was found in the F1/F5 code paths. The residual risk
is operational: the frontier is a stale event log, and its size is governed by
chunk length and update volume rather than live-pair count.

---

## 4. What would change this verdict

The single most informative check is an **isolated-machine, 4074-merge run on
the project’s actual target corpus (or a 10× code corpus with realistic long
lines), recording process RSS and heap length**. If that stays near the
benchmark’s 0.28–0.53M entries and <300 MB RSS, W1-09-4 is a non-issue for the
paper and the long-chunk stress is only a hypothetical. If it reaches millions
of entries or forces swapping, the current fix needs a versioned/compacted heap
before any production claim.

Two secondary checks would sharpen other parts:

- A full fixed-vs-reference replay at code v2048/v4096 (not just 490) would
  extend W1-06’s correctness evidence to the largest code cell; a divergence
  there would change the F1 verdict.
- A serial rerun of the 120 matrix cells on an idle host, recording CPU time,
  would replace the invalid timing table; the expectation from this profile is
  ~4–6 s per synthetic cell and ~7–20 s per code cell, not 18–69 s.

---

## Appendix A — Commands

All commands from `C:\Users\acese\Desktop\unibull`; harnesses under
`C:\Users\acese\AppData\Local\Temp\opencode\w1_09\`.

```powershell
$env:PYTHONPATH=(Get-Location).Path

# minimal F1 repro + heap capture sanity
python $env:TEMP\opencode\w1_09\probe.py

# one heap/timing/exactness profile (example: synthetic 490, with reference)
python $env:TEMP\opencode\w1_09\profile_one.py --corpus synthetic --merges 490 `
  --reference 490 --out $env:TEMP\opencode\w1_09\synthetic_490.json

# uninstrumented time curve (both corpora)
python $env:TEMP\opencode\w1_09\time_one.py synthetic
python $env:TEMP\opencode\w1_09\time_one.py code

# exact matrix merge counts for code
python $env:TEMP\opencode\w1_09\time_one.py code 160,416,928,1952,4000

# long-chunk stress
python $env:TEMP\opencode\w1_09\stress_one.py 500 120 1000
python $env:TEMP\opencode\w1_09\stress_one.py 5000 30 500

# correctness cross-checks
python $env:TEMP\opencode\w1_09\property_test.py
python -m experiments.check_levels --corpus synthetic --vocab 512 --seed 42 --split chunks

# v512 morph reproduction
python -m experiments.level_assignment --seeds 42 --vocabs 512 `
  --modes frequency,morph,random_closed `
  --out $env:TEMP\opencode\w1_09\level_assignment_seed42.csv

# reconstructed buggy baseline (timing context only)
python $env:TEMP\opencode\w1_09\buggy_bpe.py synthetic 234,490,4074
python $env:TEMP\opencode\w1_09\buggy_bpe.py code 234,490,4096
```

The exact per-run JSONs are in `%TEMP%\opencode\w1_09\` (`synthetic_*.json`,
`code_*.json`, `timing_*.json`, `synthetic4k_*.json`).

## Appendix B — Key raw outputs

Minimal repro (fixed):

```text
F1 minimal repro merges: [(1, 0, 3, 10), (0, 2, 4, 4), (3, 2, 5, 1)]
words: [[5], [3], [4]]
pair_counts: {}
pair_words: {}
heap size: 0 pushes: 3 pops: 5 peak: 4
```

T10 replication:

```text
merges learned: 100
pair_words keys: 408 actual live pair keys: 408
pair_words (pair,word) entries: 428
keys whose set is empty: 0
stale keys (not in final adjacency set): 0
missing keys: 0
stale entries: 0 missing entries: 0
```

Reference check (large corpora):

```text
synthetic 234: reference.checked=234 first_divergence=null
synthetic 490: reference.checked=490 first_divergence=null
code      234: reference.checked=234 first_divergence=null
code      490: reference.checked=490 first_divergence=null
```

`check_levels`:

```text
corpus=synthetic vocab=512 seed=42
flat levels=[490] rules=490
hier levels=[294, 196] rules=490
common rule triples: 490 / 490
rule sequences identical
level-1 budget b1=294; prefixes identical: True
sample tokens flat/hier: 1540/1540 identical=True
```

Stress (1000 merges, 500 chunks × 120 chars):

```text
{"n_words": 500, "len": 120, "merges": 1000, "requested": 1000,
 "wall": 188.48, "cpu": 59.92, "pushes": 4310615, "pops": 2933890,
 "peak": 1681050, "final": 1377401,
 "samples": [[500000, 480026], [1000000, 894965], [1500000, 1217960],
             [2000000, 1437403], [2500000, 1541578], [3000000, 1368406],
             [3500000, 1372869], [4000000, 1598426]]}
```

v512 morph reproduction (console):

```text
seed=42 v=512 flat      tok/ch=0.309 bpc=2.354 F1m=0.867 F1u=0.867 L1=490 t=22.8s
seed=42 v=512 frequency tok/ch=0.309 bpc=2.354 F1m=0.867 F1u=0.867 L1=294 L2=196 t=0.0s
seed=42 v=512 morph     tok/ch=0.309 bpc=2.348 F1m=0.871 F1u=0.871 L1=294 L2=196 t=0.0s
seed=42 v=512 random_closed tok/ch=0.312 bpc=2.350 F1m=0.876 F1u=0.875 L1=294 L2=196 t=0.0s
```

The `t=22.8s` flat value is wall time under the current 100-agent load; the
uninstrumented CPU time for the same tokenizer is 5.5 s. Only the metrics are
used in the credibility verdict.
