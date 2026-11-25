# W1-05 — Vector ground truth audit: `P @ S @ r`

**Agent:** W1-05 (red team, 5 of 100)
**Date:** 2026-11-22
**Project:** `C:\Users\acese\Desktop\unibull`
**Scope:** `ramified/synthetic.py` (vector ground truth), `ramified/metrics.py`,
`experiments/level_assignment.py`, `docs/findings-v0.2.md`, `diario.md`,
`docs/ramified-v0.md`, `docs/embedding-experiment.md`, committed level-assignment CSVs.
**Mode:** read-only. No source file was modified. Audit scripts live in
`C:\Users\acese\AppData\Local\Temp\opencode\w1_05_*.py`; the two that train a
tokenizer use `PYTHONDONTWRITEBYTECODE=1`.
**Relation to W1-01:** W1-01 already established the exact-leakage and
triple-interpolation results and that `Word.vector` is never read. This report
confirms those two facts independently and spends its budget on the vector
algebra that W1-01 did not measure: conditioning, the QR determinant artifact,
injectivity/retrieval margins, component decodability, per-pair completion, and
a pair-disjoint alternative.

---

## TL;DR

The algebra is correct and numerically clean: all targets are unit norm to
`6.7e-16`, the component matrices are orthogonal to `1.1e-15`, condition number
is 1, and all 11,520 `(p,r,s) -> v` targets are distinct (max off-diagonal
cosine 0.933). Whole-word identity is therefore injective.

But the target is an **isometric bilinear code (a product of orthogonal
transforms), not a semantic space**. For a fixed root, changing the affixes
gives cosine `0.0036 ± 0.2499`; same-prefix
`0.0000`, same-suffix `-0.0001`, same-affix-pair-different-root `-0.0018`.
Component probes are at chance. Only the exact triple has cosine 1. The vector
is a random code for the triple, not a representation from which root, prefix,
suffix, or "meaning" can be retrieved. With the true affixes, inversion is exact
(cos 1.0); without them, root retrieval is 1.04% (chance 0.83%).

Phase 1 **never reads the vectors**. Static scan finds zero `.vector` reads;
a runtime run with every `Word.vector = None` reproduces the seed-42 v512 row
exactly (all metric differences `0.000e+00`). The v512 morph result is a
string-only tokenizer result.

The current 20% triple split is **trivially solvable without composition**
(given oracle morpheme segmentation, the same access as O2): all 96
prefix-suffix pairs are seen in training with 85-108 roots each, and a per-pair
bilinear model with no factorization (96 matrices + 120 root vectors) predicts
all 2,304 held-out targets at cosine 0.999998. A held-out-affix-pair
split is feasible and discriminative: a per-pair fallback is at chance (0.029),
a factored ALS reaches 0.975 (still underfitting: train = held-out = 0.975), and
oracle factorization reaches 1.000. The Phase 2 "HC" condition and baseline set
do not include this test.

Finally, boundary F1 is **trivially 1.000** on this corpus with a two-character
rule (prefix length from the first letter, suffix = last two characters), versus
0.8665 for flat BPE and 0.8706 for morph. The synthetic language is uniquely
decodable; there is no boundary ambiguity to resolve. The v512 effect is a
merge-priority effect on an orthographically transparent toy, not evidence of
morphological or compositional generalisation.

**Verdict:** the narrow v512 numbers survive; the vector/compositional reading
does not. The vector ground truth is unused in Phase 1, and the current split
cannot distinguish composition from per-pair interpolation.

---

## 1. Commands and raw evidence

### 1.1 Corpus, split, and string integrity

```powershell
$env:PYTHONDONTWRITEBYTECODE=1
python C:\Users\acese\AppData\Local\Temp\opencode\w1_05_corpus_audit.py
```

Verbatim (trimmed to seed 42 plus the aggregate rows):

```text
grid: |P|=8 |R|=120 |S|=12 total=11520
seed  train test  tot  uniq_str  str_coll  ambig  train/test_str_overlap  types_in_text  roots_missing  test_with_missing_root  pair_cov  test_types_in_test_text
  42   9216 2304 11520     11520         0     0                       0          7478              0                      0        96                   1324
  43   9216 2304 11520     11520         0     0                       0          7482              0                      0        96                   1342
...
  51   9216 2304 11520     11520         0     0                       0          7457              0                      0        96                   1368
```

Interpretation:

- 11,520 unique strings for 11,520 triples; zero collisions and zero
  ambiguous boundaries on seeds 42-51. The language is uniquely decodable:
  the eight prefixes have distinct first letters
  (`u,r,p,d,i,s,a,c`), and all twelve suffixes are distinct two-character
  strings, so any string determines prefix, suffix, and therefore root.
- Zero train/test string overlap across all seeds.
- Every root, prefix and suffix is attested in the realised 120,000-token
  training text (120/120 roots, 8/8 prefixes, 12/12 suffixes); the minimum
  number of distinct word forms per root is > 2. W1-01's stronger root-level
  counts (63-88 training types per root) agree.
- `pair_cov = 96` is **prefix-suffix** coverage; all 96 pairs are seen in
  training. W1-01 found a small root-suffix pocket on seed 43 (8/2,304 words),
  which does not affect the `P_p S_s` factorization but should still be
  disclosed.
- The realised `test_text` contains only 1,324 of the 2,304 held-out types
  (57.5%); the bpc evaluation is a 2,000-token subsample of the 2,304-type F1
  set. See M4.

### 1.2 Geometry: norms, orthogonality, conditioning, QR sign

```powershell
python C:\Users\acese\AppData\Local\Temp\opencode\w1_05_vector_audit.py
python C:\Users\acese\AppData\Local\Temp\opencode\w1_05_qr_and_seeds.py
python C:\Users\acese\AppData\Local\Temp\opencode\w1_05_root_geometry.py
```

Verbatim (seed 42 unless noted):

```text
=== norms / orthogonality ===
word norms: min=0.99999999999999944 max=1.0000000000000007 max|n-1|=6.661e-16
max |M^T M - I| over 20 matrices = 1.110e-15
condition numbers: min=1.0000000000000004 max=1.0000000000000013
determinants: min=-1.000000 max=-1.000000
max |Q^T Q - I| over 96 products = 1.554e-15
non-commutativity ||PS-SP||_F/sqrt(d): min=1.2735 mean=1.3811 max=1.4844
||Q-I||_F/sqrt(d): min=1.2501 mean=1.4021 max=1.5104

=== matrix similarity (Frobenius inner product / d) ===
P_p vs P_q: n_pairs=28  mean=+0.0174 sd=0.0687 min=-0.1379 max=+0.1633
S_s vs S_t: n_pairs=66  mean=+0.0242 sd=0.0748 min=-0.1166 max=+0.2029
Q_ps vs Q_p's': n_pairs=4560 mean=+0.0038 sd=0.0652 min=-0.2120 max=+0.2029

=== root constellation, d=16, n=120 ===
pairwise cosine: closest root pair = 0.8059 (distance 0.6231)
R singular values: min=1.8474 max=3.4752 cond=1.881
R^T R eigenvalues: min=3.4129 max=12.0772
per-pair design matrices R_obs (train roots):
n per pair: min=85 max=108 mean=96.00
cond(R_obs): min=1.868 max=2.391 mean=2.032
word covariance eigenvalues: min=0.059303 max=0.066210 ratio=1.12; trace=1.0
```

All 20 component matrices per seed have determinant `-1`; 200/200 over seeds
42-51. This is not Haar sampling of `O(16)`. The QR convention check:

```text
d= 2: det(Q) signs over 100 random = {-1: 100}; det(qr(I)) = +1
d= 3: det(Q) signs over 100 random = {1: 100};  det(qr(I)) = +1
d= 4: det(Q) signs over 100 random = {-1: 100}; det(qr(I)) = +1
d=16: det(Q) signs over 100 random = {-1: 100}; det(qr(I)) = +1
d=17: det(Q) signs over 100 random = {1: 100};  det(qr(I)) = +1
```

The sign is `(-1)^(d-1)` for random square input in this NumPy/LAPACK build.
Consequently every `P_p S_s` has determinant `+1`: the 96 products live in
`SO(16)`, not in a Haar sample of `O(16)`. Numerically harmless; see M1 for the
Phase 2 consequence.

### 1.3 Injectivity and retrieval

Same script as 1.2. Verbatim:

```text
=== injectivity of triple -> vector ===
distinct targets after 12-decimal rounding: 11520/11520
pairwise cosine: max off-diagonal=0.933290 (pair 417,8469)
vectors whose nearest other cosine >= .99/.95/.90: 0/0/32 (of 11520)

=== retrieval on 2304 held-out vectors (nearest other of 11520) ===
NN cosine: mean=0.8026 min=0.7134 max=0.9169
NN category: {'unrelated': 1812, 'same suffix only': 173, 'same prefix only': 261,
              'same affix pair, diff root': 35, 'same root, diff affixes': 23}

root retrieval from v alone: 0.0104 (chance 0.0083)
root retrieval after known-affix inverse: 1.0000; cos(recovered, true) min=1.000000000000

=== linear/5NN component probes trained on train, tested on held-out ===
  prefix  classes=  8 chance=0.1250 logreg=0.1363 5nn=0.1185
  suffix  classes= 12 chance=0.0833 logreg=0.0764 5nn=0.0781
  root    classes=120 chance=0.0083 logreg=0.0056 5nn=0.0091

=== whole-word R@1 under additive Gaussian noise (11,520 candidates) ===
  sigma=0.00: R@1=1.0000
  sigma=0.05: R@1=1.0000
  sigma=0.10: R@1=0.9961
  sigma=0.20: R@1=0.4579
  sigma=0.35: R@1=0.0603
  sigma=0.50: R@1=0.0113
```

The same-component cosine distribution is the cleaner diagnostic. Computed over
all 11,520 words:

```text
same root, diff affixes          n=1094400 mean=+0.0036 sd=0.2499 min=-0.8961 max=+0.8811
same prefix                      n=16577280 mean=+0.0000 sd=0.2497 min=-0.9047 max=+0.8911
same suffix                      n=11047680 mean=-0.0001 sd=0.2497 min=-0.9056 max=+0.8984
same affix pair, diff root       n=1370880 mean=-0.0018 sd=0.2478 min=-0.7552 max=+0.8059
```

There is no component-level metric structure. Every component-conditioned
cosine distribution is centred at zero with the same `1/sqrt(16)` spread as a
random pair. The vector encodes the triple, but it does not encode the
components in any retrievable geometric sense. The doc's root-centroid anchor
(0.2306, reproduced) is an oracle over the test set and is not a trainable
baseline.

### 1.4 Phase 1 usage: static and runtime

Static scan over project Python (excluding `.opencode`):

```powershell
# grep for '.vector' and 'vector=' in *.py
```

Only one hit: `ramified/synthetic.py:97` (`vector=vec`, the constructor write).
The field annotation is `synthetic.py:32`. There is no `.vector` read anywhere;
no CSV has a vector column.

Runtime proof: monkeypatch `generate_synthetic` in memory so every
`Word.vector = None`, then run the real `experiments.level_assignment` pipeline
for seed 42, v512, modes `frequency,morph,random_closed`:

```powershell
$env:PYTHONDONTWRITEBYTECODE=1
python C:\Users\acese\AppData\Local\Temp\opencode\w1_05_phase1_novec.py
```

Output (trimmed):

```text
seed=42 v=512 flat      tok/ch=0.309 bpc=2.354 F1m=0.867 F1u=0.867 L1=490
seed=42 v=512 frequency tok/ch=0.309 bpc=2.354 F1m=0.867 F1u=0.867 L1=294 L2=196
seed=42 v=512 morph     tok/ch=0.309 bpc=2.348 F1m=0.871 F1u=0.871 L1=294 L2=196
seed=42 v=512 random_closed tok/ch=0.312 bpc=2.350 F1m=0.876 F1u=0.875 L1=294 L2=196

=== compare with committed CSV (vectors present) ===
seed=42 mode=flat           tokens_per_char=+0.000e+00 bits_per_char=+0.000e+00 boundary_f1=+0.000e+00 decode_word=+0.000e+00 decode_doc=+0.000e+00
seed=42 mode=frequency      tokens_per_char=+0.000e+00 bits_per_char=+0.000e+00 boundary_f1=+0.000e+00 decode_word=+0.000e+00 decode_doc=+0.000e+00
seed=42 mode=morph          tokens_per_char=+0.000e+00 bits_per_char=+0.000e+00 boundary_f1=+0.000e+00 decode_word=+0.000e+00 decode_doc=+0.000e+00
seed=42 mode=random_closed  tokens_per_char=+0.000e+00 bits_per_char=+0.000e+00 boundary_f1=+0.000e+00 decode_word=+0.000e+00 decode_doc=+0.000e+00
```

This is a runtime proof, not just an AST argument. Phase 1 is invariant to the
vector branch.

### 1.5 Split triviality: per-pair bilinear completion

The current split leaves every prefix-suffix pair observed. A non-compositional
per-pair model, given the true morpheme identity of each word (the same oracle
segmentation access as O2), is

```text
v(p,s,k) = Q_(p,s) r_k
```

with 96 matrices `Q` and 120 root vectors `r` (26,496 parameters), trained on
the 9,216 observed triples (80% of roots per pair). Alternating least squares,
three random inits, 40 iterations:

```text
observed=11520 train=9216 held-out=2304
  init 0: held-out cosine min=0.999990 mean=0.999998 p1=0.999994 acc@0.9=1.0000
  init 1: held-out cosine min=0.999985 mean=0.999997 p1=0.999993 acc@0.9=1.0000
  init 2: held-out cosine min=0.999989 mean=0.999998 p1=0.999994 acc@0.9=1.0000
```

The current split cannot distinguish "learned composition `P_p S_s`" from
"learned 96 independent pair maps". The latter has no compositional rule at all
and is essentially perfect.

A stronger held-out-affix-pair split is feasible. For the vector target the
relevant held-out pair is `(prefix, suffix)`; for the boundary task W1-01
recommends `(prefix, root)` and `(root, suffix)` pairs. Hold out 7 of the 96
prefix-suffix pairs; keep one prefix and five suffixes fully observed as a
reference. Every held-out prefix and suffix still appears in other training
pairs.

```powershell
python C:\Users\acese\AppData\Local\Temp\opencode\w1_05_pair_split.py
python C:\Users\acese\AppData\Local\Temp\opencode\w1_05_pair_als_long.py
```

Results:

```text
=== held-out-pair split (7 of 96 pairs; every prefix/suffix still trained) ===
held-out pairs: [('un','or'), ('re','al'), ('pre','ik'), ('dis','os'),
                 ('in','em'), ('sub','at'), ('ad','ul')]
observed-pair Q estimates: 89; roots per pair ~97
held-out-pair prediction: n=840 cosine min=1.0000000000 mean=1.0000000000 p1=1.0000000000
root-only baseline cosine mean=+0.0382
global-mean baseline cosine mean=+0.0196
global-average-Q fallback cosine mean=+0.0291
```

The `1.0000000000` above is the oracle-root reconstruction
`Q_(p,s) = Q_(p,s0) Q_(p0,s0)^-1 Q_(p0,s)`, which proves the target is exactly
identifiable from observed pairs. A from-scratch factored ALS (learn `P`, `S`,
`r`; no oracle roots) reaches:

```text
it= 49 train_cos mean=0.946245 min=0.886790 | held_cos mean=0.945913 min=0.887746
it= 99 train_cos mean=0.963389 min=0.924318 | held_cos mean=0.963291 min=0.924905
it=149 train_cos mean=0.971030 min=0.939823 | held_cos mean=0.970980 min=0.942654
it=199 train_cos mean=0.975542 min=0.947927 | held_cos mean=0.975493 min=0.950155
```

(The 200-iteration run above is one init; the 80-iteration three-init runs
reached 0.938, 0.920 and 0.958 held-out cosine, with the same train = held-out
pattern.) Train and held-out differ by less than 0.0004 at every checkpoint:
the factored model is **underfitting**, not overfitting. The gap to 1.0 is
optimization, not identifiability. The important contrast is structural: a per-pair model has no
parameters for the held-out pairs and is at chance (0.029); the factored model
predicts them as well as it fits the observed pairs (0.975). This is what the
current HC split fails to test.

### 1.6 Boundary ceiling

```powershell
python C:\Users\acese\AppData\Local\Temp\opencode\w1_05_stats_and_ceiling.py
```

Verbatim (seed 42, 2,304 test words):

```text
=== boundary ceiling: orthographic rule on seed 42 test words ===
rule: prefix length from first letter, suffix = last 2 chars
n=2304 precision=1.000000 recall=1.000000 F1=1.000000 misses=0
first-letter -> prefix length map: {'u': 2, 'r': 2, 'p': 3, 'd': 3,
                                   'i': 2, 's': 3, 'a': 2, 'c': 3}
distinct (first-letter, last-2) keys: 96 for 2304 test words
```

The rule is exact by construction: all eight prefixes have distinct first
letters, all twelve suffixes have length 2, and the suffix boundary is
`len(word)-2`. It uses the known affix inventory (8 prefixes, 12 two-character
suffixes) but no learning. BPE flat
is 0.8665, frequency 0.8665, morph 0.8706, random_closed 0.8757 on this seed.
The morph gain is +0.0041 on a 0.1335 gap to the trivial rule (about 3.1%);
the 10-seed mean gain is +0.0092 on the same gap (about 6.9%).

### 1.7 v512 reproduction and per-word sparsity

Independent recomputation from `results/level_assignment_v512_10seeds.csv`
(no source code involved):

```text
boundary_f1:
  morph - frequency        : mean=+0.009243 sd=0.004484 t=+6.518 W/L/T=10/0/0
                             CI=[+0.006036,+0.012451]
  random_closed - frequency: mean=-0.006386 sd=0.011672 t=-1.730 W/L/T=3/7/0
                             CI=[-0.014735,+0.001963]
  morph - random_closed    : mean=+0.015629 sd=0.012970 t=+3.810 W/L/T=9/1/0
                             CI=[+0.006351,+0.024907]
bits_per_char:
  morph - frequency        : mean=-0.025788 sd=0.015148 t=-5.384 W/L/T=0/10/0
                             CI=[-0.036623,-0.014952]
  random_closed - frequency: mean=+0.036144 sd=0.037618 t=+3.038 W/L/T=8/2/0
  morph - random_closed    : mean=-0.061932 sd=0.041446 t=-4.725 W/L/T=0/10/0
per-seed F1 deltas morph-frequency:
  [0.004123, 0.009722, 0.009635, 0.009983, 0.000145,
   0.009144, 0.015596, 0.014540, 0.010793, 0.008753]
```

These match the committed stats CSV exactly. Seed 46 is effectively a tie
(+0.000145 F1, -0.000997 bpc); the "10/10" claim is threshold-sensitive.

Per-word check (seed 42, real tokenizer and real level assignment):

```powershell
$env:PYTHONDONTWRITEBYTECODE=1
python C:\Users\acese\AppData\Local\Temp\opencode\w1_05_perword.py
```

```text
mean F1 over 2304 test words:
  flat       0.866508
  frequency  0.866508
  morph      0.870631

substring group: n=232
  frequency  meanF1=0.825000
  morph      meanF1=0.846552
  morph-frequency mean delta=+0.021552  W/L=11/0
NOT substring: n=2072
  frequency  meanF1=0.871155
  morph      meanF1=0.873327
  morph-frequency mean delta=+0.002172  W/L=9/0
all test words: morph-frequency W/L/T = 20/0/2284
```

This independently confirms W1-01's M1: 232 of 2,304 test words are substrings
of a training word (almost all the `re`/`pre` relation), and that group carries
+0.0216 of the +0.0092 overall mean. Outside the group the effect is +0.0022.
Only 20 of 2,304 words change at all; 2,284 are exact ties.

The level composition is as intended (seed 42): morph L1 is 250 intra / 43
cross / 1 unseen, frequency L1 is 197/96/1, random_closed L1 is 214/79/1. The
assignment does move intra-morpheme rules earlier; the F1 gain is consistent
with that priority, not with a rule-count difference (all three have L1=294,
L2=196).

---

## 2. Findings by severity

### H1 — High — Phase 1 never reads the vector ground truth

**Where:** `ramified/synthetic.py:26-32,84-99` writes `Word.vector`;
`ramified/metrics.py:39-50` and `experiments/level_assignment.py:45-61` read
only `w.word` and `w.boundaries`; `ramified/levels.py:54-76` reads only strings
and boundaries.

**Evidence:** zero `.vector` reads in the project; runtime `vectors=None` run
reproduces every metric exactly (`0.000e+00` on all five columns).

**Why it matters:** the synthetic corpus is described as having a semantic
ground truth `P @ S @ r`, and `diario.md:51-53` mentions it in the Phase 1a
entry. The v512 result is a boundary/tokenization result. It cannot support any
claim about compositional meaning. `docs/findings-v0.2.md` is honest that
Phase 2 is future work; the paper must keep that separation explicit. This
confirms W1-01 H2; the new contribution here is the runtime invariance proof and
the geometry below.

### H2 — High — The current split is solvable without composition

**Where:** `ramified/synthetic.py:101-116` (random 20% triple split);
`docs/embedding-experiment.md:1007-1034` (HC/HR/HA conditions, no held-out-pair
condition); README:122-128 already flags that a pair-disjoint split is needed.

**Evidence:** all 96 prefix-suffix pairs are in training with 85-108 roots each
(mean 96; a 16x16 per-pair map is ~5x overdetermined). A per-pair bilinear model
with no factorization, using oracle morpheme segmentation (the same access as
O2), predicts all 2,304 held-out targets at cosine 0.999998 (three random
inits). The Phase 2 baseline set has flat mean, flat position
mean, GRU/Transformer, tree, and factored oracle O2, but no per-pair bilinear
oracle baseline. On a 7-pair held-out split, the per-pair fallback is at chance
(0.029), a from-scratch factored ALS reaches 0.975 (underfitting; train =
held-out), and oracle factorization reaches 1.0. The stronger test is both
feasible and discriminative.

**Why it matters:** the "HC" condition is described as the primary compositional
condition, but it only holds out the exact triple. Every pairwise interaction is
seen, so a lookup over 96 pair maps wins without a compositional rule. A Phase 2
result on HC would not distinguish "tree composition" from "per-pair
interpolation". Add a held-out-affix-pair condition (HP) and the per-pair
baseline before interpreting any cosine gain as composition.

### H3 — High (external validity) — Boundary F1 has a trivial ceiling

**Where:** `ramified/synthetic.py:18-22,45-49,85-99`; `ramified/metrics.py:39-50`.

**Evidence:** a two-character rule (prefix length from the first letter, suffix =
last two characters) gives precision = recall = F1 = 1.000000 on all 2,304
held-out words, zero misses. Flat BPE is 0.8665; morph is 0.8706. The language
is uniquely decodable (distinct prefix first letters; distinct two-character
suffixes; zero collisions/ambiguity across 10 seeds). W1-01's near-duplicate
substring group is a second, independent artefact: 232 test words are substrings
of training words, and the morph gain is +0.0216 inside that group vs +0.0022
outside.

**Why it matters:** the boundary task tests whether BPE learns a 20-entry
positional lookup, not whether it generalises morphological structure. The
morph condition keeps the same 490 rules and vocabulary and only partitions
them (`n_l1=294`, `n_l2=196`); the tokenization changes because level-1 rules
get priority. It is a two-stage merge-priority schedule, not a constraint on
which merges can fire. It may still be a valid tokenizer comparison, but
"migliora l'allineamento morfologico" should be read as "on this synthetic,
uniquely decodable language"; it is not evidence about ambiguous real
morphology.

### M1 — Medium — QR sign artifact: all component matrices are `det = -1`

**Where:** `ramified/synthetic.py:52-54` (`_orthogonal` via `np.linalg.qr`),
`:81-82`.

**Evidence:** 20/20 matrices per seed and 200/200 over seeds 42-51 have
determinant `-1`. Random square QR in this NumPy/LAPACK build gives
`det(Q) = (-1)^(d-1)` (100/100 at d=16; the identity is a special case and
returns +1). Every product `P_p S_s` therefore lies in `SO(16)`, not in a Haar
sample of `O(16)`.

**Why it matters:** a model whose composition operator is constrained to
`SO(16)` — a rotation parameterization, a matrix exponential of a skew-symmetric
matrix, an orthogonal recurrent layer — **cannot represent any `P_p` or `S_s`**
and will fail for a reason unrelated to composition. The doc's general linear
`M(e)` can represent them, but any orthogonal variant must be tested carefully.
Fix: multiply the first column of `Q` by a random sign (or draw from
`scipy.stats.ortho_group`) so the matrices are Haar on `O(16)`, or document the
coset explicitly. Numerically the artifact is harmless (orthogonality and
condition number are exact), but the distributional claim "random orthogonal"
is not literally Haar.

### M2 — Medium — The target is isometric and too stable

**Where:** `ramified/synthetic.py:52-54,81-89`.

**Evidence:** `P`, `S` orthogonal; all intermediate norms 1; condition number 1;
word covariance eigenvalues 0.0593-0.0662 (ratio 1.12); per-pair design
condition 1.87-2.39; root constellation condition 1.88; non-commutativity
`||PS-SP||_F/sqrt(d)` 1.27-1.48 (strong, so order does matter).

**Why it matters:** the benchmark cannot test the numerical failure modes that
motivate hierarchical composition: scale, ill-conditioning, vanishing/exploding
gradients. Any model that L2-normalises gets the norm for free; every
intermediate is already on the sphere. The doc treats orthogonality as a
feature ("normalising matches the target geometry"). It is a feature for
tractability and a limitation for external validity. A non-orthogonal,
ill-conditioned generator would be a harder and more realistic test.

### M3 — Medium — Injective for whole words, not for components or meaning

**Where:** `ramified/synthetic.py:89`; `ramified/metrics.py`; the Phase 2
retrieval design in `docs/embedding-experiment.md` §5.3, §7.

**Evidence:** 11,520/11,520 unique targets; max off-diagonal cosine 0.9333; no
pair >= 0.95; whole-word R@1 1.000 exact, 0.996 at sigma 0.10, 0.458 at sigma
0.20, 0.060 at sigma 0.35. Component probes at chance (prefix 0.136 vs 0.125;
suffix 0.076 vs 0.083; root 0.0056-0.0104 vs 0.0083); 5-NN at chance. Same-root
different-affix cosine 0.0036 ± 0.2499; same-prefix 0.0000; same-suffix
-0.0001; same-affix-pair -0.0018. With known affixes, inversion is exact
(cos 1.0, root retrieval 1.0).

**Why it matters:** "injective enough for retrieval" depends on the query. It is
injective for exact word identity and small perturbations, but not for
component identity, same-root grouping, or semantic similarity. The vector is a
random code for the triple, not a meaning space. Any Phase 2 evaluation that
uses cosine to the target is fine; any evaluation that expects root/prefix/suffix
clustering from the target alone is measuring an oracle, not a representation.
The 0.23 root-centroid anchor is oracle-only; a model at 0.23 has clustered
roots using test targets, not learned composition.

### M4 — Medium/Low — bpc and F1 use different supports

**Where:** `ramified/synthetic.py:134-136`; `experiments/level_assignment.py:45-61`.

**Evidence:** `test_words` has 2,304 types; `test_text` has 2,000 tokens sampled
with replacement, covering only 1,324 distinct types (57.5%). F1 is type-level
on all 2,304; bpc is token-level on the 2,000-token subsample.

**Why it matters:** the two metrics answer different questions and have
different noise. The paired comparison is still valid (same test text across
methods), but bpc is a noisy subsample and not directly aligned with F1. Report
both on the same support, or report the coverage explicitly.

### L1 — Low — The "10/10 seeds" wording is threshold-sensitive

Seed 46 has F1 +0.000145 and bpc -0.000997, effectively a tie. The 10/10 counts
use a `> 1e-15` threshold. Nine seeds show a clear gain; one is negligible.
Report "9 clear + 1 negligible" or a minimum effect size alongside the mean/CI.
The mean and paired t are unaffected.

### L2 — Low — F1 and bpc are not independent evidence

Both improve because the same level assignment changes the tokenization; they
are correlated consequences. "Both improve" is weaker than it sounds. A single
pre-registered primary metric with the other as secondary would be cleaner.

### L3 — Low — CLI papercut: `--modes flat` crashes

`experiments/level_assignment.py` always runs flat and then loops over the
requested modes; including `flat` in `--modes` raises
`ValueError: unknown mode: flat` at `ramified/levels.py:99`. Not vector-related,
but it bit the runtime proof. Either filter `flat` from `--modes` or document
that it is implicit.

### What checked out

- 11,520 unique targets; zero string collisions; zero ambiguous boundaries;
  zero train/test overlap on seeds 42-51.
- Unit norms (`6.7e-16`), orthogonality (`1.1e-15`), condition 1, full-rank
  isotropic covariance.
- Root constellation and per-pair design matrices well conditioned
  (cond 1.88, 1.87-2.39).
- The doc's geometry anchors reproduce: test pairwise cosine mean 0.000111,
  sd 0.249979, min -0.907951, max 0.898043; root centroid 0.230600; global
  retrieval chance 0.000434.
- The v512 statistics recompute exactly from the CSV.
- The per-pair ALS result is stable across three random inits.
- The held-out-pair oracle reconstruction is exact (`1.0000000000` on 840
  entries).

---

## 3. Does the v512 morph result survive?

**As a number: yes.** It recomputes exactly from
`results/level_assignment_v512_10seeds.csv`: +0.009243 F1 (t=6.518, 95% CI
[+0.006036,+0.012451], 10/0/0) and -0.025788 bpc (t=-5.384, 10/0/0), against
frequency; +0.015629 F1 and -0.061932 bpc against random_closed; the v1024/2048
null is exact. The runtime check with every vector set to `None` reproduces the
seed-42 row to `0.000e+00` on every metric. The effect is not a vector artefact:
Phase 1 never reads the vector.

**As evidence of compositional generalisation: no.** The current split is
triple interpolation; all 96 prefix-suffix pairs are seen; a per-pair bilinear
model with no composition and oracle segmentation reaches 0.999998 on the
held-out targets. The boundary
target has a trivial F1=1.0 ceiling via a two-character rule; the language is
uniquely decodable; the morph gain is ~3-7% of the gap to that ceiling. The
effect is sparse (20 of 2,304 words change on seed 42; 2,284 exact ties) and
partly inflated by the `re`/`pre` substring group (+0.0216 inside vs +0.0022
outside). Seed 46 is a practical tie.

**As a vector result: not applicable.** The vectors are dead data in Phase 1.
The v512 result says nothing about whether `P @ S @ r` is recoverable from the
token tree.

**Defensible paper sentence:** the existing sentence in
`docs/findings-v0.2.md` §5 is defensible if read narrowly: a two-level
assignment by morphological provenance changes tokenization and improves
boundary F1 by ~0.9 points and bpc by ~0.026 at v512, over 10 seeds, against
frequency and random controls, and the effect disappears at v1024/2048. Add:
(a) the split is unseen triples, not unseen pairs/components; (b) all pairwise
morpheme combinations are attested; (c) the effect is a handful of word types
per seed (20-50 on the four seeds W1-01 measured; not measured on all ten);
(d) the vector/compositional hypothesis is untested; (e) the `re`/`pre`
near-duplicate group should be removed or disclosed; (f) the boundary task is
trivially reachable by a two-character rule on this corpus. Without these, the
sentence overstates what the test shows.

### 3.1 Trivial or impossible?

Neither, in the absolute sense: the target is calibrated to the intended model
class.

- **Not impossible.** Unseen roots or affixes are information-theoretically
  unlearnable from spelling (the vector is independent of the string); the
  generator uses them as null controls, correctly (`docs/embedding-experiment.md`
  §6.4). A flat linear/mean model cannot recover components (probes at chance;
  same-component cosine ~0). A wrong parse `((p,r),s)` computes `S_s P_p r`,
  not `P_p S_s r`; with non-commutativity 1.27-1.48 the wrong parse is not
  representable by the same role-specific bilinear operator. A shared *linear*
  operator at d=16 cannot represent 20 independent matrices
  (`docs/embedding-experiment.md` §1.5; the general shared-bilinear case is
  left open there). These are design constraints, not bugs.
- **Trivial for the intended class.** A depth/role-specific bilinear tree with
  oracle segmentation can represent the target exactly; a per-pair bilinear
  model solves the current triple split at 0.999998 without any affix
  composition; a factored ALS on a held-out-pair split reaches 0.975 while
  underfitting, and oracle factorization reaches 1.0. The target is a rank-16
  bilinear code (96 orthogonal transforms of a 120-point root constellation),
  not a general compositional semantics.
- **Consequence.** The benchmark measures whether a model's inductive bias
  matches the generator (right-associated tree, role-specific matrices,
  oracle-ish segmentation), not whether it learns compositional meaning in a
  realistic sense. A positive result is a sanity check; a negative result
  indicts the architecture, not the hypothesis about language.

The formal homomorphism in `docs/ramified-v0.md` §3,
\(e(r(a_1,\dots,a_k)) = g(e(a_1),\dots,e(a_k))\), is realised exactly by the
bilinear target for the right-associated tree. The audit does not dispute the
algebra; it disputes the inference from a random orthogonal codomain to
"meaning".

---

## 4. What would change my mind

1. **Pair-disjoint Phase 2 with a per-pair baseline.** Hold out whole
   `(prefix, suffix)` pairs (each prefix and suffix still seen in other pairs),
   train the tree/factored model and a per-pair bilinear oracle baseline, and
   compare held-out cosine. If the factored/tree model wins clearly, the vector
   ground truth does test composition and the current "trivial" verdict is too
   harsh. If the per-pair baseline matches on the current HC split (it does at
   0.999998), the HC claim is unsupported.
2. **A real morphological corpus with ambiguous or non-positional boundaries.**
   If the morph gain persists there, the two-character ceiling is a property of
   the toy, not of the method. If it vanishes, the v512 window is a synthetic
   artefact.
3. **A non-orthogonal, ill-conditioned generator.** If the tree still wins with
   scale and conditioning in play, M2 is not doing the work. If the advantage
   disappears, orthogonality was carrying the result.
4. **A Phase 1 metric that actually reads the vectors** and correlates with the
   morph assignment. None exists today; if one did and showed the same effect,
   the vector/compositional reading would gain support.
5. **A per-pair model that fails on the current split.** It does not; the ALS
   gets 0.999998 across three inits. A failure would show the split is not
   trivial.

---

## 5. Bottom line

- `P @ S @ r` is algebraically correct, numerically stable, and injective as a
  triple code. It is **not** a semantic space: same-root words are as far apart
  as random words, and components are not linearly or metrically retrievable.
- Phase 1 never reads the vectors; the v512 morph result is string-only.
- The current triple split is solvable by 96 independent pair maps with no
  composition and oracle segmentation (cos 0.999998). A held-out-affix-pair
  split is feasible and
  discriminative; the per-pair model is at chance there, the factored model
  reaches 0.975 (underfitting), and oracle factorization reaches 1.0.
- Boundary F1 is trivially 1.0 with a two-character rule; the v512 effect is a
  small, sparse merge-priority effect on a uniquely decodable toy, partly
  inflated by the `re`/`pre` substring group.
- **Verdict:** the narrow v512 numbers survive; the compositional/vector
  interpretation does not. Confidence: high on the measurements, medium-high on
  the external-validity interpretation.
- **Single check that would most change the verdict:** a pair-disjoint Phase 2
  run (each prefix/suffix still seen) comparing the tree/factored model against
  a per-pair bilinear oracle baseline. A factored win there would overturn the
  "trivial/interpolation" reading; a per-pair win would confirm it.
