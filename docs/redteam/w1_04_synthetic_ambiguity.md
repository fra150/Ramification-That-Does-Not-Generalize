# W1-04 - boundary ambiguity in the synthetic morphology

**Agent:** W1-04 (red team, read-only)
**Date:** 2026-11-22
**Python:** 3.13.13, Windows (PowerShell)
**Project:** `C:\Users\acese\Desktop\unibull`

**Files tested**

| file | SHA-256 |
|---|---|
| `ramified/synthetic.py` | `E7D4E5FBB01E86C41541785F3FA277A79AB49B545741C8453D296F8C8A0B6D22` |
| `ramified/levels.py` | `14420E923EE6C4034EDEA192DE161479772DD274406AD80DF95BC184E5CF7DED` |
| `ramified/bpe.py` | `73093A3E62FC66A59594F29CDB8549F4523D358DF04EBAB46E412C7C47C5B0C2` |
| `ramified/metrics.py` | `14DBF6B1794ABFD9802C503BBF5C9E3D6B2922FE88E93F12246866A952FA4D7D` |
| `results/level_assignment_v512_10seeds.csv` | `E813CFC019BA0D2D8B46233CCF29573AD7116F8EAEB5D2CCCD10B5A7FFF2F747` |

**Constraint.** No project source was modified. The only intentional project
write is this report. All probe scripts ran from `%TEMP%\opencode` and wrote only
there. The published v512 numbers are reproduced by an independent fast
evaluator, so the audit is not reading stale CSVs.

---

## Verdict

**The v512 result survives the boundary-ambiguity audit, but only as a
positional-slot effect.** Under the generator's exact grammar (one prefix, one
root from the 120-root lexicon, one suffix), **0 of 11,520 words have more than
one parse** on all 10 seeds. The two ground-truth cuts are recoverable from the
string with a trivial rule: longest prefix match + last two characters, **100%
exact on 11,520 words x 10 seeds**. The published +0.0092 F1 / -0.0258 bpc is
therefore not an artifact of ambiguous or mislabelled boundaries.

The interpretation does not survive intact. `collect_provenance` uses **only
the two cut positions**, never the prefix/root/suffix strings
(`ramified/levels.py:30-51,72-76`). On this generator those positions are a
fixed template: replacing the true boundaries with the constant cut
`{2, L-2}` reproduces the morph assignment **exactly on all 10 seeds** - same
level-1 set, same F1, same bpc. The 3-character prefixes `pre/dis/sub/con` are
therefore not used at all. Prefix-only boundaries beat morph on F1; random
two-cut boundaries recover ~64% of the F1 gain and ~46% of the bpc gain; a
clean intra-only level 1 beats morph on F1 10/10 seeds. The effect is real at
v512, but it is mostly a word-edge positional segmentation, not evidence about
morphological identity.

**Confidence:** high (~0.90) that v512 is not a boundary-ambiguity artifact;
high (~0.85) that `const2 == morph` on this generator; medium (~0.65) that most
of the flat-relative gain is positional; medium-low (~0.50) that it transfers
to real morphology.

---

## TL;DR

1. **Exact grammar: zero ambiguity.** 11,520/11,520 words have exactly one
   `prefix + root + suffix` parse on every seed; zero duplicate surface forms;
   exactly one inventory prefix matches each word.
2. **Relaxed grammar: small, seed-dependent ambiguity.** If a tokenizer allows
   two suffixes (the generator does not), 96-288/11,520 words (0.83-2.50%)
   have a second parse using the 120-root lexicon. If roots are accepted by the
   C/V templates rather than the lexicon, 192-1,248/11,520 words
   (1.67-10.83%) have a second parse; the true boundary set is unique for
   89.17-98.33% of words.
3. **The ambiguity never moves the true cuts.** Every alternative parse
   preserves both true cuts and adds an interior cut inside the root, e.g.
   `nutar + ul` -> `nut + ar + ul`. A tokenizer that chooses the alternative
   loses precision on an extra boundary, not recall of the true ones.
4. **Surface collisions are common but do not create parses.** 47.9-54.9% of
   words contain a prefix or suffix string at a non-morpheme position. The
   deterministic cases are `re` inside `pre` and `is` inside `dis` (12.5% of
   words each); suffix strings occur inside roots in 28.3% of seed-42 words,
   prefix strings inside roots in 10.8%.
5. **Boundaries are trivially recoverable.** `(longest prefix match, len-2)`
   is 100% exact on all seeds. A corpus-only learner sees 12 initial 2/3-grams
   above a wide frequency gap (true prefixes count ~1,128-1,179 types; the
   longest spurious extensions ~147-151) and also recovers 100%. The final
   2-gram set is exactly the 12 suffixes.
6. **The morph assignment does not use that full information.** `{2, L-2}`
   reproduces morph exactly on all 10 seeds. Only 8/490 rules change their
   intra/cross counts when the true 3-character prefix boundary is replaced by
   a cut at 2; 5-6 majority labels flip; all 8 affected affix rules stay in
   level 1 in both cases.
7. **The reported effect is a lower bound, not a maximum.** A clean
   intra-only level 1 (233-278 rules) beats the published 60%-quota morph on
   F1 in 10/10 seeds (+0.0130) and on bpc in 8/10 (-0.0287). The quota is
   filling level 1 with 15-60 majority-cross rules; removing them helps.
8. **Most of the gain is positional.** Prefix-only boundaries beat morph on
   F1 (+0.0067, 7/10). Uniform random two-cut boundaries recover +0.0059 F1
   (63.6% of morph) and -0.0119 bpc (46.1%). Morph is better than random cuts
   by only +0.0034 F1 / -0.0139 bpc.
9. **v512 verdict:** the numbers are valid; the phrase "e' la provenienza
   morfologica che conta" is too strong. The correct description is "a two-cut
   word-edge segmentation aligned with the synthetic slots, with a small
   residual effect of the true boundary positions."

---

## 1. What "ambiguous" means in this audit

The generator produces `word = prefix + root + suffix` with:

- 8 prefixes: `un re pre dis in sub ad con` (2 or 3 chars, prefix-free);
- 12 suffixes: `is or al ik os em at ul ar en ov ex` (all exactly 2 chars);
- 120 roots from `CVCVC / CVC / CVCCV / VCVC`;
- 11,520 word types = 8 x 120 x 12; 9,216 train, 2,304 held-out.

I separate three grammars, because they give different answers:

| grammar | allowed | intended use |
|---|---|---|
| **exact** | exactly 1 prefix + 1 root from the 120-root lexicon + 1 suffix | the generator's actual production rule |
| **relaxed** | 1-2 prefixes + 1 root from the lexicon + 1-2 suffixes | a tokenizer that does not know the generator forbids stacked affixes, but knows the root lexicon |
| **template** | 1-2 prefixes + any string matching a C/V root pattern + 1-2 suffixes | a tokenizer that does not know the root lexicon at all |

I also separate a **parse** (which prefix(es), root, suffix(es)) from a
**boundary set** (the cut positions). The task asks about both; they are not the
same. A word can have one boundary set and more than one parse, or (in
principle) two boundary sets.

Enumeration is exhaustive over all 11,520 words for each seed 42-51. No
sampling.

---

## 2. Exhaustive parse audit

### 2.1 Exact grammar: zero ambiguity

For every word, I enumerated all `p in PREFIXES`, `s in SUFFIXES`, and checked
whether the middle is in the 120-root lexicon.

| seed | words with >1 exact parse | words with >1 boundary set | prefix matches per word |
|---:|---:|---:|---:|
| 42-51 | **0 / 11,520** | **0 / 11,520** | always exactly 1 |

The same result holds if the middle is accepted by any C/V template rather than
the lexicon for the exact one-affix grammar: 11,520/11,520 unique. Duplicate
surface forms: 0. The reason is structural:

- the prefix set is prefix-free, so the first prefix is unique;
- all suffixes have length 2, so the suffix cut is always `len(word)-2`;
- with the prefix and suffix fixed, the root is the remaining middle string,
  and roots are unique.

There is no word like `unretalis` that can be parsed two ways under the exact
grammar. `un + retal + is` is the only parse, even though the substring `re`
also occurs at position 2.

### 2.2 Relaxed grammar: 0.83-2.50%

Allow 1-2 prefixes and 1-2 suffixes, keep the finite root lexicon.

| seed | multi-parse words | % of 11,520 |
|---:|---:|---:|
| 42 | 96 | 0.83% |
| 43 | 0 | 0.00% |
| 44 | 96 | 0.83% |
| 45 | 0 | 0.00% |
| 46 | 0 | 0.00% |
| 47 | 0 | 0.00% |
| 48 | 96 | 0.83% |
| 49 | 288 | 2.50% |
| 50 | 0 | 0.00% |
| 51 | 0 | 0.00% |

All alternatives are **two-suffix** splits of a 5-character `CVCVC` root that
ends in a suffix, where the residual 3-character `CVC` is also in the root
lexicon. Example from seed 42:

```text
renutarul
  true:      re + nutar + ul          cuts {2, 7}
  alternative: re + nut  + ar + ul    cuts {2, 5, 7}
```

There were **zero two-prefix alternatives** under the finite root lexicon on
all 10 seeds. The reason is that a root starting with a prefix is rare
(0.83-5.0% of words; 1-6 roots per seed), and the residual after stripping the
second prefix must also be in the 120-root lexicon; that coincidence did not
occur in this root draw.

### 2.3 Template grammar: 1.67-10.83%

Allow any middle matching `CVCVC / CVC / CVCCV / VCVC`, still 1-2 affixes.

| seed | multi-parse words | % | true boundary set unique |
|---:|---:|---:|---:|
| 42 | 288 | 2.50% | 97.50% |
| 43 | 1,056 | 9.17% | 90.83% |
| 44 | 768 | 6.67% | 93.33% |
| 45 | 576 | 5.00% | 95.00% |
| 46 | 480 | 4.17% | 95.83% |
| 47 | 384 | 3.33% | 96.67% |
| 48 | 768 | 6.67% | 93.33% |
| 49 | 1,152 | 10.00% | 90.00% |
| 50 | 192 | 1.67% | 98.33% |
| 51 | 1,248 | 10.83% | 89.17% |

Most of the ambiguity is again two-suffix splitting, now for any 5-character
`CVCVC` root ending in one of the 12 suffixes. Roots ending in a suffix:

| seed | CVCVC roots ending in suffix | CVC roots | VCVC roots | CVCCV roots |
|---:|---:|---:|---:|---:|
| 42 | 3 | 11 | 4 | 0 |
| 43 | 10 | 3 | 2 | 0 |
| 44 | 8 | 3 | 3 | 0 |
| 45 | 6 | 7 | 7 | 0 |
| 46 | 4 | 7 | 4 | 0 |
| 47 | 3 | 3 | 5 | 0 |
| 48 | 8 | 6 | 6 | 0 |
| 49 | 12 | 2 | 6 | 0 |
| 50 | 2 | 3 | 7 | 0 |
| 51 | 11 | 1 | 5 | 0 |

Each such root contributes 8 prefixes x 12 suffixes = 96 words. `CVCCV` roots
cannot contribute because they end `C V`, while all suffixes end `V C`. This is
a structural property of the generator, not a seed accident.

Template two-prefix alternatives exist for 0-192 word-parses on seeds
43/46/47/48/49/51. They add a cut after the second prefix and keep the true
first-prefix cut. All of them preserve the true boundary set (see next section).

### 2.4 The true cuts are a subset of every alternative

For every relaxed/template alternative in all 10 seeds:

```text
true cuts preserved: 100% of multi-parse words
true cuts lost:        0
```

A two-suffix alternative adds the cut between the two suffixes; a two-prefix
alternative adds the cut after the second prefix. Neither moves or deletes the
true cuts `{len(prefix), len(word)-2}`. Example:

```text
true:              {2, 7}
two-suffix:        {2, 5, 7}   (adds 5)
two-prefix:        {2, 4, 7}   (adds 4)
```

This is why the boundary-ambiguity audit does not damage boundary F1 recall:
the true boundaries are still the minimal/edge cuts. A tokenizer that picks an
alternative can only add a false positive boundary inside the root, not miss a
true one.

---

## 3. Surface collisions and spurious affix substrings

Boundary ambiguity is zero, but **surface ambiguity** is not. I counted every
occurrence of every prefix and suffix string at a position other than its true
morpheme span.

### 3.1 Seed 42 detail

| collision | words | % of 11,520 | occurrences |
|---|---:|---:|---:|
| prefix `re` inside prefix `pre` | 1,440 | 12.5% | 1,440 |
| suffix `is` inside prefix `dis` | 1,440 | 12.5% | 1,440 |
| prefix string inside root | 1,248 | 10.8% | 1,440 |
| suffix string inside root | 3,264 | 28.3% | 3,360 |
| prefix-like string overlapping prefix/root | 0 | 0.0% | 0 |
| prefix-like string overlapping root/suffix | 232 | 2.0% | 232 |
| suffix-like string overlapping prefix/root | 360 | 3.1% | 360 |
| **any spurious affix substring** | **6,322** | **54.9%** | **8,272** |

There is **no exact prefix/suffix string overlap** (`PREFIXES intersect SUFFIXES = empty`).
The deterministic near-overlaps are:

- `re` is a prefix and a proper substring of the prefix `pre` at offset 1;
- `is` is a suffix and a proper substring of the prefix `dis` at offset 1.

Neither changes a parse under the exact grammar, because the prefix `pre`/`dis`
is matched as a whole and the suffix is fixed at the last two characters. But
both are plausible false morpheme cues for a tokenizer. The suffix strings
occur inside roots very often (28.3% of words) because suffixes are common
`V C` bigrams and roots are random C/V strings.

### 3.2 Across seeds

| statistic | range (seeds 42-51) |
|---|---:|
| words with any spurious affix substring | 47.9-54.9% |
| test words with any spurious affix substring | 47.2-54.8% |
| words whose root starts with a prefix | 0.83-5.00% |
| words whose root ends with a suffix | 9.17-16.67% |

### 3.3 It does not become rule-level ambiguity

The spurious substrings are numerous, but the BPE rules themselves are mostly
clean. On the flat v512 tokenizer:

| statistic | range (10 seeds) |
|---|---:|
| rules with both intra and cross occurrences | 8-16 / 490 (1.6-3.3%) |
| strict intra-majority rules | 233-272 / 490 |
| strict cross-majority rules | 217-256 / 490 |
| ties | 0 |
| unseen (the punctuation rule) | 1 |
| duplicate expansion strings | 0 |

The dominant mixed rules are the affix rules `re` and `dis`; most other rules
are either entirely inside a morpheme or entirely across. So 55% of words carry
a false affix cue, but only ~2% of merge rules have genuinely mixed provenance.
The practical ambiguity is not "which side is this rule on?" but "the 60%
quota forces 15-60 pure-cross rules into level 1 anyway" (see section 5).

---

## 4. Boundary recovery from the string alone

### 4.1 The trivial exact rule

Because the prefix set is prefix-free and every suffix is length 2:

```text
p = longest string in PREFIXES that is a prefix of word
boundaries = (len(p), len(word) - 2)
```

Results on all words, all 10 seeds:

| metric | value |
|---|---:|
| exact boundary set | **1.0000** |
| prefix boundary exact | 1.0000 |
| suffix boundary exact | 1.0000 |
| macro F1 of `{2, L-2}` constant | 0.7500 |
| macro F1 of `{3, L-2}` constant | 0.7500 |
| macro F1 of lookup | **1.0000** |

The constant baselines lose exactly the words whose prefix length is not the
constant: half the prefixes are 2 characters, half are 3. The lookup uses one
bit - the first two characters - and is perfect. The first two characters
already identify the prefix length:

```text
un re in ad       -> length 2
pr di su co       -> first two chars of pre/dis/sub/con, length 3
pre dis sub con   -> length 3
```

Strictly, the lookup uses the 8-prefix inventory. section 4.2 shows that the inventory
itself is recoverable from the corpus with a wide margin, so the boundary
information is not hidden; it is a property of the string distribution.

### 4.2 A corpus-only learner also gets 100%

Using only the 9,216 training word types (no lexicon, no true boundaries):

1. Count initial 2- and 3-grams.
2. Keep candidates above a type-frequency threshold.
3. Longest match at the word start.
4. Suffix cut at `len(word)-2` (the final 2-gram set is exactly the 12 suffixes).

With threshold 500 (seed 42):

```text
candidates = ad co con di dis in pr pre re su sub un
true prefixes present = all 8
missing = none
final 2-grams = al ar at em en ex ik is or os ov ul  (exactly the 12 suffixes)
boundary exact = 1.0000
```

Threshold sensitivity across seeds:

| threshold | candidates | exact rate |
|---:|---|---:|
| 150 | 12-16 (extra `ret/int/unt/...`) | 0.933-0.967 |
| 200 | 12 | 1.000 |
| 300 | 12 | 1.000 |
| 500 | 12 | 1.000 |
| 800 | 12 | 1.000 |
| 1,000 | 12 | 1.000 |
| 1,100 | 12 | 1.000 |

The gap is wide: true prefix candidates occur ~1,128-1,179 times; the longest
spurious extension (e.g. `ret`, `int`, `unt`) occurs ~147-151 times. Any
threshold in 200-1,100 gives the same 12 candidates and 100% exact boundaries
on all 10 seeds.

**Caveat.** This learner is still tailored to the generator: it knows to look
at 2/3-grams and it knows the suffix is exactly 2 characters. A real
unsupervised segmenter would have to discover those lengths. But on this
corpus the discovery is trivial - the final 2-gram distribution has exactly 12
members and a huge frequency margin.

### 4.3 What is not recoverable from the string alone

The **parse** is not unique under the template grammar for up to 10.83% of
words. In particular, `nutar+ul` versus `nut+ar+ul` cannot be distinguished
without the root lexicon or the one-suffix rule. The **boundary set** is still
recoverable: both parses contain the true `{2, 7}` cuts; the alternative adds
`5`. This distinction matters:

- if the task is "recover the two ground-truth boundaries", the answer is
  100% from the string;
- if the task is "recover the morpheme identity/root", the answer is not unique
  for up to 10.83% of words under a template-only grammar.

`collect_provenance` only uses the two boundaries, so the synthetic
level-assignment experiment is in the first regime, not the second.

---

## 5. What the level assignment actually uses

### 5.1 It uses only boundary positions

`collect_provenance` walks each derivation node and asks whether any
`word.boundaries` cut lies strictly inside the node's span
(`ramified/levels.py:30-51`). It never looks at `word.prefix`, `word.root`, or
`word.suffix`. The only input from the morphology is the pair of integers
`(len(prefix), len(word)-2)`. The "morphological provenance" is therefore a
**boundary-position provenance**; any segmenter that supplies two cuts produces
the same signal.

### 5.2 Replacing true boundaries with `{2, L-2}` is exact

I recomputed provenance with every word's boundaries replaced by
`(2, len(word)-2)` and reran the morph assignment. On all 10 seeds:

| comparison | F1 | bpc | prefix F1 | suffix F1 | level-1 set |
|---|---:|---:|---:|---:|---:|
| `const2 - morph` | **0.000000** | **0.000000** | 0.000000 | 0.000000 | Jaccard **1.000** |

This is not a metric coincidence. On seeds 42-44 only **8/490 rules** change
their `(intra, cross)` counts when the 3-character prefix boundary is replaced
by a cut at 2; **5-6 majority labels flip**; but **all 8 affected rules remain
in level 1** in both assignments. The affected rules are exactly the high-rank
prefix/suffix rules:

```text
seed 42: is, on, con, re, ub, dis, sub, pre
seed 43: re, su, sub, pre, on, con, dis, sut
seed 44: re, su, sub, on, is, con, dis, pre
```

For example, seed 42:

```text
rank 2  con  true=(38006,0)      const2=(0,38006)      L1 true=L1 const2
rank 25 dis  true=(8785,0)       const2=(0,8785)       L1 true=L1 const2
rank 34 pre  true=(7680,0)       const2=(396,7284)     L1 true=L1 const2
rank 0  is   true=(46336,0)      const2=(37551,8785)   L1 true=L1 const2
```

The 3-character prefix boundary is real morphology, but the assignment never
needs it: the affected rules are so frequent that they are selected under
either label. The published "morph" result is therefore reproducible by a
constant cut that ignores `pre/dis/sub/con` entirely.

### 5.3 A cleaner intra-only level 1 is better

The 60% quota sets `n1 = int(0.6 x 490) = 294`. On every seed there are only
233-272 strict intra-majority rules, so the quota must be filled with 15-60
majority-cross rules (plus the punctuation rule). I tested the natural
counterfactual: **level 1 = every rule with `intra > cross`, then dependency
closure** (233-278 rules; no cross-majority rules). In the bpc rows of the next
table, W/L is counted with lower-is-better, so `9/1` means the left side is
better on 9 seeds and worse on 1.

| paired difference | F1 (mean, t, W/L) | bpc (mean, t, W/L) |
|---|---:|---:|
| `hard_intra - frequency` | **+0.02225**, t=10.68, 10/0 | **-0.05448**, t=-5.25, 9/1 |
| `hard_intra - morph` | **+0.01301**, t=5.58, 10/0 | **-0.02869**, t=-2.75, 8/2 |

So the published morph effect is not a maximum; it is a **diluted lower bound**.
The direction of the paper's claim is robust - indeed stronger - but the exact
60%-quota operationalization is not the best use of the provenance signal.
This corroborates W1-11's finding that the raw `intra/(intra+cross)` score is
untuned (their Laplace variants also improved on seed 42), but the hard-intra
result is simpler: it removes the cross-majority fill entirely and still wins
F1 10/10.

### 5.4 Prefix-only beats morph on F1; suffix-only is harmful

| assignment | F1 vs morph (mean, t, W/L) | bpc vs morph (mean, t, W/L) |
|---|---:|---:|
| `prefix_only - morph` | **+0.00665**, t=2.44, 7/3 | +0.00975, t=0.62, 5/5 |
| `suffix_only - morph` | **-0.03592**, t=-5.48, 0/10 | **+0.19480**, t=6.44, 10/0 |

Using only the prefix cut improves F1; using only the suffix cut is much worse.
The useful signal is concentrated at the **word-initial** boundary. The suffix
cut at `L-2` is trivially correct in the evaluation, but including it in
provenance moves suffix-internal rules to level 1 in a way that slightly hurts
F1 (prefix-only is better). This is a tradeoff: prefix-only has slightly worse
bpc (+0.0098, not significant), while morph is better for compression. But the
claim that both morphological boundaries are doing useful work is not supported
by F1.

### 5.5 Random two-cut boundaries recover most of the gain

I replaced every word's true boundaries with two uniformly random cuts
`1 <= i < j < len(word)` (same number of cuts as the truth), recomputed
provenance, and reran morph. Averaged over 5 draws per seed, 10 seeds:

| comparison | F1 | bpc |
|---|---:|---:|
| `random2 - frequency` | **+0.00588**, t=8.88, 10/0 | **-0.01188**, t=-3.85, 9/1 |
| `random2 - morph` | **-0.00337**, t=-2.81, 2/8 | +0.01390, t=2.75, 7/3 |
| share of morph-frequency gain | **63.6%** | **46.1%** |
| level-1 Jaccard with morph | ~0.80 | - |

Random cuts are not as good as true morphology, but they get most of the F1
gain and nearly half of the bpc gain. The **morphology-specific residual** is
only about +0.0034 F1 / -0.0139 bpc. This is the strongest evidence that the
flat-relative effect is largely "cut the word into chunks near the edges",
not "identify prefix/root/suffix".

For contrast, W1-14's random **rule-order** control is much worse than morph
(+0.0223 F1 for morph versus the random-order mean). Putting the two controls
together: the effect is not an arbitrary reordering (W1-14), but it is largely
reproduced by arbitrary **two-cut boundaries** (this audit). The boundary
structure matters; the exact morphology does not.

### 5.6 Summary table of level-assignment variants (10 seeds)

| variant | F1 vs frequency | bpc vs frequency | notes |
|---|---:|---:|---|
| `morph` (published) | +0.00924 (t=6.52, 10/0) | -0.02579 (t=-5.38, 10/0) | exact reproduction |
| `hard_intra` | **+0.02225** (t=10.68, 10/0) | **-0.05448** (t=-5.25, 9/1) | cleaner level 1 |
| `const2` | identical to morph | identical to morph | 10/10 exact |
| `prefix_only` | +0.01589 (morph +0.00665) | -0.01604 (morph +0.00975) | F1 better, bpc worse |
| `random2` | +0.00588 (t=8.88, 10/0) | -0.01188 (t=-3.85, 9/1) | 64% / 46% of morph |
| `suffix_only` | -0.02668 | +0.16901 | much worse |

The `prefix_only` and `suffix_only` values versus frequency are derived by
adding the paired `prefix_only - morph` / `suffix_only - morph` difference to
the directly measured `morph - frequency` difference; they are not separate
reruns against frequency.

---

## 6. Does the v512 result survive?

### 6.1 What survives

- **The published numbers reproduce to CSV precision.** My independent
  evaluator gives the same flat F1/bpc (seed 42: 0.866508 / 2.353869) and the
  same paired statistics as `results/level_assignment_stats_v512.csv`:

  ```text
  morph - frequency  F1  +0.009243  t=+6.52  10/0/0
  morph - frequency  bpc -0.025788  t=-5.38  0/10/0
  morph - random_closed F1 +0.015629 t=+3.81  9/1/0
  morph - random_closed bpc -0.061932 t=-4.73 0/10/0
  ```

- **The labels are not ambiguous.** Exact grammar: 0/11,520 multi-parse words
  on all seeds. Boundaries: 100% recoverable by longest prefix + last 2, and by
  a corpus-only frequency learner. The v512 numbers are not an artifact of
  mislabelled boundaries or duplicate surface forms.
- **The direction is robust and stronger under a cleaner assignment.**
  `hard_intra` beats frequency on F1 10/10 and bpc 9/10, and beats morph on F1
  10/10. So the paper's qualitative conclusion ("morphological provenance
  assignment helps in a budget window") is not overturned by this audit; if
  anything, the effect is larger than reported.
- **The effect is not an arbitrary reordering.** W1-14 already showed that
  random rule orders are much worse; this audit does not contradict that.

### 6.2 What does not survive

- **"It is morphological provenance that matters."** `collect_provenance`
  never uses prefix/root/suffix identity, only two cuts. `{2, L-2}` reproduces
  morph exactly on 10/10 seeds, including the level-1 set. The 3-character
  prefixes are not used. A fixed positional template is sufficient.
- **"Both boundaries matter."** Prefix-only beats morph on F1; suffix-only is
  much worse. The useful signal is the word-initial cut, not the full
  prefix/root/suffix structure.
- **"The gain is specific to true morphology."** Random two-cut boundaries
  recover 63.6% of the F1 gain and 46.1% of the bpc gain; the morphology-specific
  residual is small.
- **"A real tokenizer would need information it cannot have."** The true
  boundaries are recoverable from the corpus with a wide-margin threshold; a
  real tokenizer could implement the same cut. The experiment is an
  **oracle-segmentation ablation**, and the +0.0092 is an upper bound for a
  pipeline with perfect segmentation, not an end-to-end demonstration. Because
  segmentation is trivial here, the gap to an unsupervised implementation is
  probably small - but it is not measured.
- **External validity.** The generator is a slot code: fixed 2-character
  suffix, prefix-free 2/3-character prefixes, random roots. A real morphological
  corpus does not have a last-2 suffix slot or a first-2 prefix lookup that
  recovers 100% of boundaries. The result may be a property of this template,
  not of morphology.

### 6.3 The honest sentence

Replace:

> "E' la provenienza morfologica che conta."

with:

> "A two-cut word-edge segmentation aligned with the synthetic slots improves
> boundary F1 by ~0.9 points and bpc by ~0.026 at v512, over 10 seeds, against
> frequency and random-rule controls. Most of the F1 gain is reproduced by
> arbitrary two-cut boundaries; the exact prefix/root/suffix identity is not
> used, and the 3-character prefix boundary is irrelevant to the assignment."

The v512 result is real, but it is a **positional segmentation** result, not a
morphological-identity result. The paper should not present it as evidence that
the model learns prefix/root/suffix structure.

---

## 7. What would change my mind

1. **Variable-length morphology (the single most informative check).**
   Regenerate the same corpus with suffix lengths 1-3 and prefix lengths 1-4,
   keeping roots, vectors, splits and tokenizer budget fixed. Rerun
   `morph` vs `frequency` vs `const2` vs `random2`.
   - If `const2` still equals `morph`, the effect is a fixed-slot artifact.
   - If `const2` diverges and `morph` still beats both `frequency` and
     `random2`, the effect is genuinely boundary/morphology-specific.
   - If `morph` collapses to `frequency` or `random2`, the v512 result does not
     generalize beyond the slot template.

2. **A real morphological corpus with verified boundaries.** The project
   already lists this as the main open item. If morph beats frequency and
   random cuts there, with boundaries verified independently, the external
   validity objection weakens substantially.

3. **An unsupervised segmenter as the boundary source.** Replace the oracle
   boundaries with Morfessor/UniSeg/BPEmb-style segmentation (or the corpus-only
   learner in section 4.2, but with its thresholds learned on held-out data). If the
   gain survives the segmenter's errors, the "oracle information" critique is
   less serious. If it vanishes, the oracle segmentation is doing the work.

4. **A near-true random-boundary control.** My `random2` is uniform over
   positions. A stronger control would draw cuts near, but not at, the true
   positions (e.g. true +/- 1-2 characters) with the same per-word count. If
   morph beats that, exact boundary positions matter more than the current
   uniform control suggests. Conversely, if near-true cuts match morph, the
   effect is purely positional.

5. **A purity-matched random control** (W1-14's suggestion): same level-1 size
   and same number of intra rules as morph, with the intra rules chosen at
   random rather than by provenance. Combined with the boundary control, this
   would separate "more intra rules in level 1" from "specifically the true
   cuts."

6. **A corpus where the 3-character prefix rules are not top-ranked.** On this
   generator, `con/dis/sub/pre` are always high-frequency and always selected,
   so the 3-character boundary cannot change the level-1 set. At a different
   vocabulary size or with a different prefix inventory, `const2` might not
   equal morph. If it diverges there, the morphological detail could matter
   even though it does not at v512.

**Single check that would most change the verdict:** the variable-length
morphology generator in item 1. It directly removes the fixed `{2, L-2}` slot
that makes the current ground truth trivially recoverable and makes `const2`
exact.

---

## 8. Confidence

| claim | confidence |
|---|---:|
| Exact grammar has zero ambiguous parses (11,520 x 10 seeds) | **0.99** |
| True boundary set is recoverable by longest prefix + last 2 (100%) | **0.99** |
| Corpus-only 2/3-gram induction recovers 100% with threshold 200-1,100 | **0.95** |
| `const2` reproduces morph exactly on 10 seeds | **0.90** |
| Most of the flat-relative F1 gain is positional (random2/prefix-only/const2) | **0.70** |
| The v512 numbers are not a boundary-ambiguity artifact | **0.95** |
| The effect transfers to real morphology | **0.50** |

The high-confidence claims are exhaustive enumerations or exact reruns. The
medium-confidence claim is an interpretation: random cuts recover most, but not
all, of the gain, and hard-intra still improves, so a boundary-specific
component remains. The low-confidence claim is external validity, which this
corpus cannot settle.

---

## Appendix A - Commands

All commands were run from `C:\Users\acese\Desktop\unibull`; scripts live in
`%TEMP%\opencode` and write only there.

```powershell
# Exhaustive parse ambiguity, spurious affixes, boundary recovery, induction
python C:\Users\acese\AppData\Local\Temp\opencode\w1_04_ambiguity2.py

# Threshold sensitivity, two-prefix alternatives, true-cut geometry
python C:\Users\acese\AppData\Local\Temp\opencode\w1_04_extra.py

# Rule provenance, per-boundary F1, const2/prefix-only/suffix-only/random2,
# hard-intra counterfactual (seed 42 with perturbations)
python C:\Users\acese\AppData\Local\Temp\opencode\w1_04_levels2.py 42

# Same, seeds 43-51 (no perturbations)
python C:\Users\acese\AppData\Local\Temp\opencode\w1_04_levels2.py 43 44 45 46 47 48 49 50 51

# Compact cross-seed table and paired summaries
python C:\Users\acese\AppData\Local\Temp\opencode\w1_04_compact.py 42 43 44 45 46 47 48 49 50 51

# Stats-level check that const2 changes only 8 rules and no level-1 membership
python C:\Users\acese\AppData\Local\Temp\opencode\w1_04_const2_check.py
```

## Appendix B - Key output excerpts

Exact-grammar and relaxed ambiguity:

```text
SEED 42
  strict {1: 11520}  pattern {1: 11520}  prefix_matches {1: 11520}
  relaxed multi 96 (two-suffix), 0 (two-prefix strict)
  template multi 288, true boundary set unique 11232/11520 (97.50%)
```

Boundary recovery:

```text
seed 42 exact 1.0000 prefix 1.0000 suffix 1.0000
         F1 const2 0.7500 F1 const3 0.7500 F1 lookup 1.0000
seed 43 ... (same)
...
seed 51 ... (same)
```

Induction:

```text
threshold 200: exact 1.0000 n=12
  ['ad','co','con','di','dis','in','pr','pre','re','su','sub','un']
threshold 1100: exact 1.0000 n=12
final 2-grams = ['al','ar','at','em','en','ex','ik','is','or','os','ov','ul']
```

const2 identity:

```text
const2 - morph  boundary_f1        mean=+0.000000 sd=0.000000 W/L/T=0/0/10
const2 - morph  prefix_boundary_f1 mean=+0.000000 sd=0.000000 W/L/T=0/0/10
const2 - morph  suffix_boundary_f1 mean=+0.000000 sd=0.000000 W/L/T=0/0/10
const2 - morph  bits_per_char      mean=+0.000000 sd=0.000000 W/L/T=0/0/10
```

Stats-level const2 check:

```text
seed 42: rules changed 8/490, majority flips 6, L1 symmetric difference 0
seed 43: rules changed 8/490, majority flips 6, L1 symmetric difference 0
seed 44: rules changed 8/490, majority flips 5, L1 symmetric difference 0
```

Hard-intra and random-boundary controls:

```text
hard_intra - frequency  F1  mean=+0.022253 t=+10.68 W/L/T=10/0/0
hard_intra - frequency  bpc mean=-0.054477 t=-5.25  W/L/T=1/9/0
hard_intra - morph      F1  mean=+0.013009 t=+5.58  W/L/T=10/0/0
hard_intra - morph      bpc mean=-0.028689 t=-2.75  W/L/T=2/8/0
random2 - frequency     F1  mean=+0.005877 t=+8.88  W/L/T=10/0/0
random2 - frequency     bpc mean=-0.011884 t=-3.85  W/L/T=1/9/0
```

## Appendix C - Limitations

- The audit is on the synthetic generator only. It says nothing directly about
  real morphology.
- The corpus-only induction uses a hand-chosen threshold and the knowledge
  that affixes are 2/3 characters. The threshold range is wide, but the
  length prior is still a design choice.
- The `hard_intra` variant changes the level-1 size (233-278 vs 294), so it is
  a counterfactual, not a like-for-like replacement of the published method.
  It is reported as evidence that the 60% quota is not necessary, not as the
  new headline.
- The random-boundary control uses uniform positions and 5 draws per seed. A
  near-true or position-matched control would be stronger.
- The report relies on W1-11's verification that `collect_provenance` span
  arithmetic is exact; I did not re-audit the walker, only its inputs and
  outputs.
