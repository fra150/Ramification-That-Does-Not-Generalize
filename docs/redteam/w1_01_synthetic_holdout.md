# W1-01 — Synthetic held-out audit

**Agent:** W1-01 (red team, 1 of 100)
**Date:** 2026-11-22
**Project:** `C:\Users\acese\Desktop\unibull`
**Scope:** `ramified/synthetic.py`, `experiments/level_assignment.py`,
`ramified/metrics.py`, `ramified/bpe.py`, `ramified/levels.py`, and the
committed level-assignment CSVs.
**Mode:** read-only. No source file was modified. Temporary audit scripts live
in `C:\Users\acese\AppData\Local\Temp\opencode\`.

## TL;DR

The v512 morph result **reproduces** as a paired, in-distribution segmentation
effect: on seed 42 the current code gives exactly the committed macro-F1 values,
and the 10-seed statistics recompute. There is **no exact train/test word
leakage**: 11,520 unique word strings, zero collisions, zero overlap, zero test
words in the training text.

But the 20% holdout is **triple interpolation, not compositional
generalisation**. Every test word has its prefix, root and suffix attested in
training, and for 2,296–2,304 of 2,304 test words *all three pairwise morpheme
combinations* are also attested. The median test word has **110 training words
that differ by exactly one morpheme**, and **96.8–98.0%** of test words are at
Levenshtein distance 1 from a training word. Only the exact triple is novel.

The compositional vector ground truth (`P @ S @ r`) is **never read in Phase 1**.
It is generated and then discarded; no experiment, metric, or CSV column uses
`Word.vector`. Phase 1 therefore cannot support any claim about compositional
meaning. The v512 result is a boundary-alignment result on seen morphemes, and
even there it is sparse: on seeds 42–45 only **158 of 9,216** per-word
evaluations change at all, all upward, with 20–50 improved word types per seed.

**Verdict:** the narrow v512 numbers survive; the compositional interpretation
does not. The result must be labelled “unseen combinations of seen morphemes”,
not “compositional generalisation”, and the vector hypothesis remains untested.

## 1. What I checked

1. Split integrity: total triples, uniqueness of strings/triples, exact
   train/test overlap, test words present in the realised training text.
2. Component attestation at the type level and in the realised 120,000-token
   training text, for seeds 42–51.
3. Pairwise morpheme coverage: whether `(prefix, root)`, `(root, suffix)` and
   `(prefix, suffix)` occur in training for each test word.
4. Near-duplicates: proper substring containment between test and train words,
   Levenshtein distance 1, one-morpheme neighbours, and whether any test word
   becomes an exact tokenizer token.
5. Whether the vector ground truth is consumed anywhere in Phase 1 (static AST
   scan of all project Python files, CSV header scan, and inspection of the
   evaluation path).
6. Reproduction of the v512 result on seed 42, the 10-seed paired statistics,
   the v1024/v2048 null, level sizes, and per-word sparsity.
7. Secondary checks: test-set root balance, F1-vs-bpc sample mismatch, and the
   attestation filter’s behaviour at the default split.

## 2. Commands and raw evidence

### 2.1 Split, component attestation, pairwise coverage, near-duplicates

Command:

```powershell
python C:\Users\acese\AppData\Local\Temp\opencode\w1_01_audit_synthetic.py
```

Per-seed output (verbatim; `pr/rs/ps` are min/median/max counts of the three
pairwise combinations in training types; `*_t` are the same counts in the
realised training text; `substr` is the number of test words that are a proper
substring of a training word; `edit1` is the number at Levenshtein distance 1;
`one_morph` is the number of training words differing in exactly one morpheme):

```text
seed=42 triples=11520 uniq=11520 dup=0 train=9216 test=2304 moved=0 exact_overlap=0 test_in_train_text=0 comp_fail=0/0 unsampled=1738 pair_all=2304/2304 triple_in_train=0 pr=(4, 9.0, 11) rs=(1, 6.0, 7) ps=(85, 95.0, 108) pr_t=(5, 35.0, 18249) rs_t=(2, 19.0, 18267) ps_t=(266, 677.0, 28046) one_morph=(94, 110.0, 125) substr=232 (pref/suff/mid=2/230/0) test_contains_train=260 edit1=2257 root_count=(10, 19.0, 31) valid_bnd=True bad_test_text=0 token_missing=0
seed=43 triples=11520 uniq=11520 dup=0 train=9216 test=2304 moved=0 exact_overlap=0 test_in_train_text=0 comp_fail=0/0 unsampled=1734 pair_all=2296/2296 triple_in_train=0 pr=(4, 9.0, 11) rs=(0, 6.0, 7) ps=(79, 95.0, 106) pr_t=(3, 32.0, 18241) rs_t=(0, 18.0, 18322) ps_t=(245, 649.0, 18840) one_morph=(88, 110.0, 123) substr=257 (pref/suff/mid=0/257/0) test_contains_train=229 edit1=2238 root_count=(10, 19.0, 29) valid_bnd=True bad_test_text=0 token_missing=0
seed=44 triples=11520 uniq=11520 dup=0 train=9216 test=2304 moved=0 exact_overlap=0 test_in_train_text=0 comp_fail=0/0 unsampled=1733 pair_all=2304/2304 triple_in_train=0 pr=(4, 9.0, 11) rs=(2, 6.0, 7) ps=(81, 96.0, 107) pr_t=(3, 35.0, 18248) rs_t=(1, 19.0, 18244) ps_t=(247, 603.0, 19033) one_morph=(91, 110.0, 124) substr=225 (pref/suff/mid=0/225/0) test_contains_train=244 edit1=2239 root_count=(11, 19.0, 33) valid_bnd=True bad_test_text=0 token_missing=0
seed=45 triples=11520 uniq=11520 dup=0 train=9216 test=2304 moved=0 exact_overlap=0 test_in_train_text=0 comp_fail=0/0 unsampled=1811 pair_all=2304/2304 triple_in_train=0 pr=(4, 9.0, 11) rs=(2, 6.0, 7) ps=(85, 96.0, 108) pr_t=(3, 33.0, 18207) rs_t=(1, 19.0, 18189) ps_t=(250, 724.0, 18800) one_morph=(93, 110.0, 125) substr=245 (pref/suff/mid=0/245/0) test_contains_train=216 edit1=2259 root_count=(8, 19.0, 29) valid_bnd=True bad_test_text=0 token_missing=0
seed=46 triples=11520 uniq=11520 dup=0 train=9216 test=2304 moved=0 exact_overlap=0 test_in_train_text=0 comp_fail=0/0 unsampled=1723 pair_all=2304/2304 triple_in_train=0 pr=(5, 9.0, 11) rs=(2, 6.0, 7) ps=(87, 95.0, 109) pr_t=(2, 33.0, 18364) rs_t=(1, 19.0, 18423) ps_t=(283, 706.0, 18599) one_morph=(98, 109.0, 127) substr=214 (pref/suff/mid=0/214/0) test_contains_train=237 edit1=2241 root_count=(10, 19.0, 29) valid_bnd=True bad_test_text=0 token_missing=0
seed=47 triples=11520 uniq=11520 dup=0 train=9216 test=2304 moved=0 exact_overlap=0 test_in_train_text=0 comp_fail=0/0 unsampled=1776 pair_all=2304/2304 triple_in_train=0 pr=(5, 9.0, 11) rs=(2, 6.0, 7) ps=(82, 95.0, 108) pr_t=(4, 33.0, 18413) rs_t=(1, 18.0, 18334) ps_t=(260, 663.0, 18789) one_morph=(93, 110.0, 125) substr=270 (pref/suff/mid=0/270/0) test_contains_train=231 edit1=2231 root_count=(11, 19.0, 29) valid_bnd=True bad_test_text=0 token_missing=0
seed=48 triples=11520 uniq=11520 dup=0 train=9216 test=2304 moved=0 exact_overlap=0 test_in_train_text=0 comp_fail=0/0 unsampled=1761 pair_all=2304/2299 triple_in_train=0 pr=(3, 9.0, 11) rs=(2, 6.0, 7) ps=(84, 95.0, 108) pr_t=(2, 33.0, 18025) rs_t=(0, 18.0, 17988) ps_t=(197, 691.0, 18387) one_morph=(94, 109.0, 125) substr=222 (pref/suff/mid=2/220/0) test_contains_train=273 edit1=2255 root_count=(10, 19.0, 30) valid_bnd=True bad_test_text=0 token_missing=0
seed=49 triples=11520 uniq=11520 dup=0 train=9216 test=2304 moved=0 exact_overlap=0 test_in_train_text=0 comp_fail=0/0 unsampled=1785 pair_all=2304/2304 triple_in_train=0 pr=(5, 9.0, 11) rs=(2, 6.0, 7) ps=(86, 96.0, 106) pr_t=(4, 32.0, 18239) rs_t=(1, 19.0, 18229) ps_t=(263, 669.0, 19566) one_morph=(96, 110.0, 124) substr=226 (pref/suff/mid=3/223/0) test_contains_train=291 edit1=2250 root_count=(9, 19.0, 31) valid_bnd=True bad_test_text=0 token_missing=0
seed=50 triples=11520 uniq=11520 dup=0 train=9216 test=2304 moved=0 exact_overlap=0 test_in_train_text=0 comp_fail=0/0 unsampled=1699 pair_all=2304/2304 triple_in_train=0 pr=(5, 9.0, 11) rs=(2, 6.0, 7) ps=(84, 95.0, 108) pr_t=(4, 33.0, 5461) rs_t=(1, 18.0, 18362) ps_t=(238, 614.0, 18797) one_morph=(95, 109.0, 125) substr=235 (pref/suff/mid=0/235/0) test_contains_train=230 edit1=2256 root_count=(12, 19.0, 32) valid_bnd=True bad_test_text=0 token_missing=0
seed=51 triples=11520 uniq=11520 dup=0 train=9216 test=2304 moved=0 exact_overlap=0 test_in_train_text=0 comp_fail=0/0 unsampled=1759 pair_all=2304/2304 triple_in_train=0 pr=(4, 9.0, 11) rs=(1, 6.0, 7) ps=(84, 94.0, 107) pr_t=(3, 34.0, 18361) rs_t=(1, 19.0, 18359) ps_t=(257, 677.0, 18861) one_morph=(93, 109.0, 124) substr=257 (pref/suff/mid=0/257/0) test_contains_train=211 edit1=2256 root_count=(10, 19.0, 30) valid_bnd=True bad_test_text=0 token_missing=0
```

Aggregate output:

```text
duplicate_strings: min=0 median=0.0 max=0
exact_overlap: min=0 median=0.0 max=0
test_strings_in_train_text: min=0 median=0.0 max=0
component_fail_type: min=0 median=0.0 max=0
component_fail_realized: min=0 median=0.0 max=0
pair_all: min=2296 median=2304.0 max=2304
pair_all_realized: min=2296 median=2304.0 max=2304
triple_in_train: min=0 median=0.0 max=0
substr_hits: min=214 median=233.5 max=270
test_contains_train: min=211 median=234.0 max=291
edit1: min=2231 median=2252.5 max=2259
unsampled_train_types: min=1699 median=1748.5 max=1811
pr: min_of_min=3 median_of_med=9.0 max_of_max=11
rs: min_of_min=0 median_of_med=6.0 max_of_max=7
ps: min_of_min=79 median_of_med=95.0 max_of_max=109
pr_realized: min_of_min=2 median_of_med=33.0 max_of_max=18462
rs_realized: min_of_min=0 median_of_med=19.0 max_of_max=18423
ps_realized: min_of_min=197 median_of_med=673.0 max_of_max=28046
one_morph: min_of_min=88 median_of_med=110.0 max_of_max=127
root_count: min_of_min=8 median_of_med=19.0 max_of_max=33
```

Examples:

```text
substr_examples: [('redamos', 'predamos'), ('retanluex', 'pretanluex'), ('relupis', 'prelupis')]
edit1_examples: [('subtopar', 'subtomar'), ('adbibdois', 'adbibdoik'), ('disarapar', 'disarapat')]
roots type/realized: 120/120 120/120
```

Component attestation counts in training types, seeds 42–51:
`w1_01_coverage.py` prints roots as minimum/median/maximum, prefixes and
suffixes as minimum/maximum, plus the test-text draw coverage:

```powershell
python C:\Users\acese\AppData\Local\Temp\opencode\w1_01_coverage.py
```

```text
seed root_min/median/max prefix_min/max suffix_min/max test_tokens distinct_types never_drawn
42 65/77/86 1128/1179 740/787 2000 1324 980
43 67/77/86 1114/1167 753/783 2000 1342 962
44 63/77/85 1118/1171 745/789 2000 1339 965
45 67/77/88 1122/1171 748/793 2000 1317 987
46 67/77/86 1130/1176 745/783 2000 1361 943
47 67/77/85 1113/1179 740/782 2000 1341 963
48 66/77/86 1123/1168 754/785 2000 1352 952
49 65/77/87 1118/1174 746/786 2000 1349 955
50 64/77/84 1138/1174 745/782 2000 1348 956
51 66/77/86 1120/1181 742/789 2000 1368 936
```

Interpretation: the single-component attestation filter is true but vacuous at
the default split. Every root is seen 63–88 times as a root; every prefix and
suffix hundreds of times. The filter moved **zero** words back to training on
all 10 seeds (`moved=0`), because the 80% training split already covers every
component by combinatorial redundancy.

### 2.2 The pairwise exception pockets

`pair_all` is 2,296 on seed 43 and `pair_all_realized` is 2,299 on seed 48.
I inspected them:

```powershell
python - <<'PY'
import sys, collections
sys.path.insert(0, r"C:\Users\acese\Desktop\unibull")
from ramified.synthetic import generate_synthetic
for seed in (43, 48):
    syn = generate_synthetic(seed=seed)
    train = syn.train_word_objects
    rs = collections.Counter((w.root, w.suffix) for w in train)
    pr = collections.Counter((w.prefix, w.root) for w in train)
    ps = collections.Counter((w.prefix, w.suffix) for w in train)
    bad = []
    for w in syn.test_words:
        a,b,c = pr[(w.prefix,w.root)], rs[(w.root,w.suffix)], ps[(w.prefix,w.suffix)]
        if not (a and b and c):
            bad.append((w.word, w.prefix, w.root, w.suffix, a, b, c))
    print("seed", seed, "bad", len(bad))
    for x in bad[:12]: print("  ", x)
PY
```

Output:

```text
seed 43 bad 8
   ('disanolik', 'dis', 'anol', 'ik', 10, 0, 89)
   ('inanolik', 'in', 'anol', 'ik', 9, 0, 99)
   ('adanolik', 'ad', 'anol', 'ik', 6, 0, 91)
   ('unanolik', 'un', 'anol', 'ik', 8, 0, 96)
   ('conanolik', 'con', 'anol', 'ik', 10, 0, 91)
   ('reanolik', 're', 'anol', 'ik', 10, 0, 92)
   ('preanolik', 'pre', 'anol', 'ik', 8, 0, 102)
   ('subanolik', 'sub', 'anol', 'ik', 9, 0, 93)
seed 48 bad 0
```

For seed 43, all eight prefixes of the `(anol, ik)` root-suffix pair landed in
the test set, so that pair is genuinely unseen — a small, accidental
pair-disjoint pocket (8/2,304 = 0.35%), not a leak. For seed 48, the five
`(mapgo, os)` test words have the pair in the training *type* list but zero
occurrences in the realised 120,000-token training text; that is a realised-text
coverage gap, not an exact leak. The other 8 seeds have full type-level pair
coverage.

### 2.3 The `re`/`pre` substring leak

The substring hits are not random. Classifying them by test prefix, train
prefix, and position:

```powershell
python - <<'PY'
import sys, collections
sys.path.insert(0, r"C:\Users\acese\Desktop\unibull")
from ramified.synthetic import generate_synthetic
for seed in range(42, 52):
    syn = generate_synthetic(seed=seed)
    test, train = syn.test_words, syn.train_word_objects
    ts = [w.word for w in test]
    lengths = sorted({len(x) for x in ts})
    sub = {}
    for tw in train:
        for L in lengths:
            if L >= len(tw.word): continue
            for i in range(len(tw.word)-L+1):
                sub.setdefault(tw.word[i:i+L], tw)
    c = collections.Counter()
    for w in test:
        tw = sub.get(w.word)
        if tw:
            pos = 'suffix' if tw.word.endswith(w.word) else ('prefix' if tw.word.startswith(w.word) else 'mid')
            c[(w.prefix, tw.prefix, pos)] += 1
    print(seed, 'hits', sum(c.values()), dict(c))
PY
```

Output:

```text
42 hits 232 {('re', 'pre', 'suffix'): 230, ('con', 'con', 'prefix'): 1, ('re', 're', 'prefix'): 1}
43 hits 257 {('re', 'pre', 'suffix'): 257}
44 hits 225 {('re', 'pre', 'suffix'): 225}
45 hits 245 {('re', 'pre', 'suffix'): 245}
46 hits 214 {('re', 'pre', 'suffix'): 214}
47 hits 270 {('re', 'pre', 'suffix'): 270}
48 hits 222 {('re', 'pre', 'suffix'): 220, ('con', 'con', 'prefix'): 1, ('un', 'un', 'prefix'): 1}
49 hits 226 {('re', 'pre', 'suffix'): 223, ('in', 'in', 'prefix'): 1, ('con', 'con', 'prefix'): 1, ('pre', 'pre', 'prefix'): 1}
50 hits 235 {('re', 'pre', 'suffix'): 235}
51 hits 257 {('re', 'pre', 'suffix'): 257}
```

The dominant channel is systematic: `PREFIXES` contains both `"re"` and
`"pre"`, and `"pre"` = `"p" + "re"`. Thus a test word `re + root + suffix`
appears verbatim as the suffix of the training word `pre + root + suffix`
whenever that training word is in the split (about 80% of the time). Example:
test `redamos` (`re|dam|os`) is a substring of train `predamos`
(`pre|dam|os`). The reverse containment is the same relation: a test word with
prefix `pre` contains the training word with prefix `re`. That is why
`test_contains_train` is 211–291 as well. There are a few residual root-prefix
cases, e.g. test `connutar` (`con|nut|ar`) inside train `connutarex`
(`con|nutar|ex`), but they are 0–3 per seed.

This is **not** exact word leakage: `predamos` is not `redamos`, and no test
word is a tokenizer token (checked on seed 42 flat: 0 exact-token matches,
0 single-token test words). It is a systematic near-duplicate: about 9.3–11.7%
of test strings occur contiguously inside a training string, and the
`re`/`pre` pair is the cause. It should be removed before calling the split
clean.

### 2.4 Vector ground truth is dead code in Phase 1

Command (AST scan of all project Python files, excluding `.opencode` and
`__pycache__`; plus CSV header scan):

```powershell
python C:\Users\acese\AppData\Local\Temp\opencode\w1_01_vector_usage.py
```

Output:

```text
python files scanned: 37
vector references outside synthetic.py: []
all vector references: [('ramified\\synthetic.py', 32, 'Name vector')]

csv headers containing 'vector':
  ablation_level1.csv: []
  level_assignment.csv: []
  level_assignment_smoke.csv: []
  level_assignment_stats.csv: []
  level_assignment_stats_v512.csv: []
  level_assignment_v512_10seeds.csv: []
any vector CSV column: False
```

The count is 37 at the final scan; it was 30 earlier in the same session because
other agents added scripts while this audit ran. The conclusion was unchanged:
zero readers outside `synthetic.py`.

The vector values themselves are correct:

```powershell
python C:\Users\acese\AppData\Local\Temp\opencode\w1_01_vector_norm.py
```

```text
norms 0.9999999999999994 1.0000000000000007
distinct 2304 of 2304
```

The only reference is the dataclass field annotation `vector: np.ndarray` at
`ramified/synthetic.py:32`. There is **no `.vector` attribute read anywhere**,
not even in `synthetic.py` (the generator only writes `vector=vec`). The Phase 1
evaluation path (`experiments/level_assignment.py::evaluate`) calls
`boundary_report`, `bpc_alpha`, `decode_exact` and `roundtrip_ok`; none touches
the vectors. `experiments/run_matrix_v2.py::eval_synthetic` is the same. The
`SyntheticCorpus` object exposes `test_words` and `train_word_objects`, both of
which carry vectors, but no experiment reads them. No CSV has a vector column.
`docs/embedding-experiment.md` is a design document; there is no
`experiments/embedding/` implementation in the tree.

The vector formula itself is correct: `P @ (S @ r)`, unit norm, 2,304 distinct
test vectors (confirmed by `docs/reviews/data-review.md` H2 and independently by
inspection). The problem is not the algebra; it is that Phase 1 never uses it.
Phase 1’s only metrics are boundary alignment, unigram bpc, tokens/char and
decode. A positive boundary-F1 result says nothing about whether a
tree-recursive embedding can recover `P @ S @ r`.

### 2.5 v512 reproduction

Command:

```powershell
python C:\Users\acese\AppData\Local\Temp\opencode\w1_01_repro_min.py
```

Output (seed 42, current source):

```text
generated syn 3.74s train_types=9216 test=2304
trained flat 54.64s rules=490
provenance 79.50s rules_with_stats=489
assign frequency 0.02s n_l1=294
assign morph 0.04s n_l1=294
assign random_closed 0.03s n_l1=294
eval flat 1.86s macro=0.866507523148 micro=0.866900175131 mean_boundary_f1=0.866507523148 P=0.888889 R=0.859375
eval frequency 1.87s macro=0.866507523148 micro=0.866900175131 mean_boundary_f1=0.866507523148 P=0.888889 R=0.859375
eval morph 1.90s macro=0.870630787037 micro=0.870992449940 mean_boundary_f1=0.870630787037 P=0.892578 R=0.863715
eval random_closed 1.86s macro=0.875723379630 micro=0.874809492706 mean_boundary_f1=0.875723379630 P=0.895110 R=0.871962
CSV flat         F1=0.8665075231481482 bpc=2.3538694808585707 n_l1=490
CSV frequency    F1=0.8665075231481482 bpc=2.3538694808585707 n_l1=294
CSV morph        F1=0.8706307870370371 bpc=2.3484206602229314 n_l1=294
CSV random_closed F1=0.8757233796296297 bpc=2.349558098164956 n_l1=294
```

The current code reproduces the committed CSV macro-F1 to 12 decimal places.
A separate identity check confirms that `frequency` is token-identical to
`flat`:

```text
frequency == flat on all test words: True
frequency == flat on all train types: True
n_l1 294 n_l2 196
```

The metrics module changed during the audit (macro/micro report, empty-prediction
precision 1.0 → 0.0), but macro-F1 is unaffected for these 2-boundary words and
still matches. The current `evaluate` writes `boundary_macro_f1` and a
backward-compatible `boundary_f1` alias; the committed CSV has only the old
`boundary_f1` column.

10-seed paired statistics recomputed from
`results/level_assignment_v512_10seeds.csv`:

```powershell
python C:\Users\acese\AppData\Local\Temp\opencode\w1_01_stats.py
```

Selected output (verbatim):

```text
boundary_f1 morph - frequency: n=10 mean=+0.009243344907 sd=0.004484453014 se=0.001418108558 t=6.518080 CI=[+0.006035360474,+0.012451329340] wins=10 losses=0 ties=0
boundary_f1 random_closed - frequency: n=10 mean=-0.006385582011 sd=0.011671806598 se=0.003690949326 t=-1.730065 CI=[-0.014735089465,+0.001963925444] wins=3 losses=7 ties=0
boundary_f1 morph - random_closed: n=10 mean=+0.015628926918 sd=0.012970343292 se=0.004101582684 t=3.810462 CI=[+0.006350502272,+0.024907351564] wins=9 losses=1 ties=0
bits_per_char morph - frequency: n=10 mean=-0.025787684384 sd=0.015147707932 se=0.004790125840 t=-5.383509 CI=[-0.036623701863,-0.014951666905] wins=0 losses=10 ties=0
bits_per_char random_closed - frequency: n=10 mean=+0.036143843625 sd=0.037618326540 se=0.011895959363 t=3.038329 CI=[+0.009233313945,+0.063054373306] wins=8 losses=2 ties=0
bits_per_char morph - random_closed: n=10 mean=-0.061931528010 sd=0.041446352956 se=0.013106487605 t=-4.725257 CI=[-0.091580462823,-0.032282593196] wins=0 losses=10 ties=0
tokens_per_char morph - frequency: n=10 mean=+0.000037019222 sd=0.000473802150 se=0.000149829395 t=0.247076 CI=[-0.000301918418,+0.000375956863] wins=5 losses=4 ties=1
```

These match `results/level_assignment_stats_v512.csv` to rounding. At v1024
and v2048, the three seeds 42–44 have **exactly** zero deltas for F1, bpc and
tokens/char between morph and frequency:

```text
1024 boundary_f1 deltas [0.0, 0.0, 0.0] all_zero True
1024 bits_per_char deltas [0.0, 0.0, 0.0] all_zero True
1024 tokens_per_char deltas [0.0, 0.0, 0.0] all_zero True
2048 boundary_f1 deltas [0.0, 0.0, 0.0] all_zero True
2048 bits_per_char deltas [0.0, 0.0, 0.0] all_zero True
2048 tokens_per_char deltas [0.0, 0.0, 0.0] all_zero True
```

Level sizes at v512 are matched for the primary comparison: `morph` and
`frequency` both have `n_l1=294`, `n_l2=196` on all 10 seeds. `random_closed`
has `n_l1=294` on 7 seeds and 292/293 on seeds 44/46/50 (a duplicate insertion
in its random topological order), which is a minor control imbalance. `morph`
level 1 is 233–278 intra rules vs 182–205 for `frequency`; the assignment does
what it claims.

### 2.6 The morph gain is sparse

Command:

```powershell
python C:\Users\acese\AppData\Local\Temp\opencode\w1_01_perword.py
```

Validation: the fast provenance implementation is exactly equal to
`collect_provenance` on a small corpus (`equal=True`, same 105 rules, same
19,246 counts). Per-seed output for seeds 43–45 (seed 42 is from the subgroup
run below):

```text
seed=43 F1 morph=0.810807292 freq=0.801085069 delta=+0.009722222 wins=42 losses=0 ties=2262 max_delta=+0.600000
  substr     n=  257 mean=+0.0412451 wins=19 losses=0 ties=238
  non_substr n= 2047 mean=+0.0057645 wins=23 losses=0 ties=2024
seed=44 F1 morph=0.923828125 freq=0.914192708 delta=+0.009635417 wins=46 losses=0 ties=2258 max_delta=+0.600000
  substr     n=  225 mean=+0.0048889 wins=2 losses=0 ties=223
  non_substr n= 2079 mean=+0.0101491 wins=44 losses=0 ties=2035
seed=45 F1 morph=0.903486690 freq=0.893504051 delta=+0.009982639 wins=50 losses=0 ties=2254 max_delta=+0.500000
  substr     n=  245 mean=+0.0414966 wins=22 losses=0 ties=223
  non_substr n= 2059 mean=+0.0062328 wins=28 losses=0 ties=2031
```

Seed 42 subgroup output:

```text
group                n         flat    frequency        morph random_closed
all               2304    0.8665075    0.8665075    0.8706308    0.8757234
substr             232    0.8250000    0.8250000    0.8465517    0.8630747
non_substr        2072    0.8711551    0.8711551    0.8733269    0.8771396
edit1             2257    0.8660168    0.8660168    0.8702260    0.8753508
non_edit1           47    0.8900709    0.8900709    0.8900709    0.8936170
in_test_text      1324    0.8611027    0.8611027    0.8663897    0.8714250
not_in_test_text   980    0.8738095    0.8738095    0.8763605    0.8815306

per-word morph-frequency F1 delta by group
all              n= 2304 mean=+0.0041233 median=+0.0000000 wins=20 losses=0 ties=2284
substr           n=  232 mean=+0.0215517 median=+0.0000000 wins=11 losses=0 ties=221
non_substr       n= 2072 mean=+0.0021718 median=+0.0000000 wins=9 losses=0 ties=2063
edit1            n= 2257 mean=+0.0042091 median=+0.0000000 wins=20 losses=0 ties=2237
non_edit1        n=   47 mean=+0.0000000 median=+0.0000000 wins=0 losses=0 ties=47
in_test_text     n= 1324 mean=+0.0052870 median=+0.0000000 wins=15 losses=0 ties=1309
not_in_test_text n=  980 mean=+0.0025510 median=+0.0000000 wins=5 losses=0 ties=975
```

On seed 42, only **20 of 2,304** word evaluations change; all 20 improve, 2,284
tie. Seeds 43–45 change 42/46/50 words respectively, all upward. Over the four
seeds, **158 of 9,216** per-word evaluations change, with **zero losses**. The
mean F1 deltas of ~+0.004 to +0.010 are therefore a handful of boundary flips,
not a broad alignment shift. Each winning word gains roughly 0.5 F1 (one
boundary decision), which is consistent with a small number of discrete
tokenization changes.

Excluding the substring group, morph still beats frequency on seeds 42–45
(+0.0022, +0.0058, +0.0101, +0.0062), so the `re`/`pre` near-duplicate is not
the sole cause; but the substring group has a much larger average gain on
seeds 42, 43 and 45, so it inflates the headline mean. On seed 44 the gain is
mostly outside the substring group.

### 2.7 Test-set balance and F1/bpc sample mismatch

Roots are not stratified: each seed’s test set has 8–33 words per root (median
19). Since `mean_boundary_f1` averages per word, roots with 33 test words carry
about four times the weight of roots with 8. The paired morph-vs-frequency
comparison uses the same corpus, so this does not bias the *difference*, but it
does affect the absolute F1 and seed variance.

F1 and bpc are also computed on different samples: F1 uses all 2,304 held-out
types; bpc uses the 2,000-token test text, which contains only 1,317–1,368
distinct types (936–987 types are never drawn). This is not a leak, but the two
metrics are not measuring the same test distribution. On seed 42, the F1 delta
on the bpc-covered 1,324 types is +0.00529, larger than the +0.00255 on the 980
never-drawn types, so the F1 result is not an artifact of the bpc subset; it is
still a type-level average over all held-out words.

## 3. Findings by severity

### H1 — High — The holdout is triple interpolation, not compositional generalisation

**Where:** `ramified/synthetic.py:101-116` (random triple split and
single-component filter), `synthetic.py:134-136` (test text), and the claims
that the synthetic split measures “compositional generalisation on unseen
words” (`synthetic.py:1-10`) / “composti mai visti” (`diario.md`).

**Evidence.** The split is a uniform random 20% of the 11,520 `(prefix, root,
suffix)` triples. The single-component filter is true but vacuous: every root,
prefix and suffix is seen, and zero words are moved back to training on all 10
seeds. More importantly, for 2,296–2,304 of 2,304 test words *all three
pairwise combinations* are attested in the training types; the realised
120,000-token training text gives the same range, with 2,299 on seed 48. The
median test word has 110 training words that differ by exactly one morpheme;
96.8–98.0% of test words are Levenshtein distance 1 from a training word. Only
the exact triple is novel.

**Why it matters.** A model that memorises pairwise morpheme interactions —
or a lookup table over `(prefix, root)` and `(root, suffix)` — can score well
without compositional rules. The test cannot distinguish “learned morphology”
from “seen two of the three morphemes”. This is interpolation over triples, not
systematic extrapolation to unseen components or unseen pairs. The result can
still be a valid tokenizer comparison, but it is not evidence for the
compositional-generalisation hypothesis.

**Fix.** Define the regime explicitly and add a pair-disjoint split (hold out
whole `(prefix, root)` and `(root, suffix)` pairs) and a zero-shot split (hold
out entire roots or affixes). Report pair coverage, not just single-component
coverage. Rename the current split “unseen triples of seen morphemes”.

### H2 — High (for the hypothesis) — The vector ground truth is never used in Phase 1

**Where:** `ramified/synthetic.py:26-32,84-99,138-145` generates `Word.vector`;
`experiments/level_assignment.py:45-61` and `experiments/run_matrix_v2.py:69-82`
evaluate only boundaries/tokens/bpc/decode.

**Evidence.** AST scan of 37 project Python files: zero `.vector` reads, zero
references outside the dataclass annotation in `synthetic.py`. No CSV has a
vector column. The Phase 1 evaluation path never reads a vector. The formula is
correct and the 2,304 test vectors are unit norm and distinct, but they are
dead data in Phase 1.

**Why it matters.** The v0.1 hypothesis has two parts: better morphological
alignment *and* a better compositional embedding on unseen compounds. Phase 1
tests only the first. The boundary-F1 result says nothing about whether
`P @ S @ r` is recoverable from the token tree. The v0.2 document is honest
that Phase 2 is future work (`docs/findings-v0.2.md` §4.2), so this is a
claim-discipline finding, not a hidden numerical error. But any sentence that
reads the boundary result as evidence for compositional meaning is unsupported.
The vector target is also exactly linear and noise-free; a future Phase 2 needs
a non-trivial compositional gap (pair-disjoint/zero-shot split, noise or
nonlinear composition) to be informative.

### M1 — Medium — Systematic `re`/`pre` near-duplicate leakage

**Where:** `ramified/synthetic.py:21` (`PREFIXES = ["un", "re", "pre", ...]`),
the concatenation at `synthetic.py:88`.

**Evidence.** 214–270 test words (9.3–11.7%) are proper substrings of a
training word. All but 0–3 per seed are exactly `re + root + suffix` inside
`pre + root + suffix`, because `"pre" = "p" + "re"`. Example:
`redamos` (`re|dam|os`) is a substring of `predamos` (`pre|dam|os`). The
reverse containment (test word contains a training word) is 211–291 test words,
mostly the same relation. In the forward direction, residual root-prefix cases
are 0–3 per seed; the reverse direction has more residual root-prefix cases
because a longer test root can contain a shorter training root.

**Why it matters.** The test string is present contiguously in training, so a
subword model can learn pieces of the exact test form. It is not exact word
leakage: the token-level check on seed 42 flat gives zero exact-token matches
and zero single-token test words.

```powershell
python C:\Users\acese\AppData\Local\Temp\opencode\w1_01_token_leak.py
```

```text
test words that are exact tokenizer tokens: 0
of those, in train words: 0
single-token test words flat: 0
examples: []
```

The substring group actually has
*lower* absolute F1 than the rest (0.825 vs 0.871 on seed 42). But it is an
unclean near-duplicate channel, and it inflates the morph headline on some
seeds: the substring group accounts for 11/20 seed-42 wins and 19/42 seed-43
wins, with mean deltas +0.0216/+0.0412 vs +0.0022/+0.0058 outside. Removing the
group leaves a positive but smaller effect.

**Fix.** Make the prefix set prefix-free (drop `"pre"` or `"re"`, or add a
separator), and reject any root that is a prefix of another root. Then rerun
the morph/frequency comparison. The residual root-prefix cases are small but
should also be removed for a clean split.

### M2 — Medium — The morph effect is sparse and should be reported per word

**Where:** the 10-seed table in `docs/findings-v0.2.md` §2, which reports only
mean ΔF1/Δbpc.

**Evidence.** On seeds 42–45, only 158 of 9,216 per-word evaluations change at
all (20, 42, 46, 50 per seed), all upward, with 2,254–2,284 ties per seed. The
mean ΔF1 of +0.0092 is a handful of boundary flips, not a broad shift. On seed
42, 2,284/2,304 words tie. The `random_closed` control beats morph on seed 42
(0.8757 vs 0.8706), which is the one loss in the 9/1 morph-vs-random comparison.

**Why it matters.** A mean with t=6.52 looks strong, but the distribution is
“a few words improve by ~0.5 F1, everything else identical”. That is a much
weaker claim than “morphological assignment improves alignment” as a general
statement. It is also fragile: a different seed or a slightly different test
set may change which 20–50 words flip. Report the win/loss/tie counts and the
per-word delta distribution, not only the mean. The bpc gain is similarly
narrow: tokens/char is essentially unchanged (+0.000037, 5W/4L/1T), so the
−0.0258 bpc is a unigram cross-entropy effect, not fewer tokens.

### M3 — Medium/Low — Test set is not stratified; F1 and bpc use different samples

**Evidence.** Test words per root range 8–33 (median 19), so type-averaged F1
weights roots unevenly. F1 uses all 2,304 types; bpc uses 1,317–1,368 distinct
types from the 2,000-token test text (936–987 types never drawn). On seed 42
the morph-frequency ΔF1 is +0.00529 on the bpc-covered types and +0.00255 on
the never-drawn types, so the direction is consistent, but the two metrics are
not on the same sample.

**Why it matters.** The paired comparison is still valid (same corpus for both
modes), but absolute F1 is sensitive to root composition, and bpc is noisier
than its 2,000-token denominator suggests. Stratify the holdout by root/affix
and use one evaluation set for every metric, or report both macro and
type-weighted results.

### L1 — Low — The attestation filter is vacuous at the default split, and uses types not realised text

**Evidence.** `moved=0` on all 10 seeds; every root is seen 63–88 times, every
prefix/suffix hundreds of times. The filter checks `train_roots` from the type
list before sampling, not the realised `train_text`. 1,699–1,811 of 9,216
training types are never sampled (18.4–19.6%). On seed 48, five test words have
a root-suffix pair that exists in training types but has zero realised tokens;
on seed 43, eight test words have an entire root-suffix block held out (the
`(anol, ik)` block), which is a stronger-than-intended pair-disjoint pocket.
The filter is order-dependent at high holdout (it appends rejected test words
without updating the component sets), and `holdout=0.0` and `holdout=1.0` both
raise `ValueError: high <= 0`. None of this
affects the default v512 numbers, but the filter is not providing the guarantee
it appears to provide.

### L2 — Low — `random_closed` level-1 size is not always exactly 294

**Evidence.** Seeds 44/46/50 have `n_l1=292/293`, `n_l2=197/198`, while
`frequency` and `morph` are 294/196 on all seeds. The cause is duplicate
insertion in the random topological order (`available` can receive the same
child twice), so the first 294 entries contain duplicates. A direct check on
seed 44 gives:

```text
rules 490 n1 294 order 491 unique order 490
duplicates 1 first [438]
unique first n1 293
```

The random control still loses, so the conclusion is not affected, but the
control is not exactly size-matched on 3/10 seeds. Fix the topological order to
deduplicate.

### L3 — Low/Note — Source changed during the audit; freeze a revision

`ramified/synthetic.py` (SHA256 `E7D4E5FBB01E86C41541785F3FA277A79AB49B545741C8453D296F8C8A0B6D22`,
mtime 22:04:54) and `ramified/bpe.py`
(`73093A3E62FC66A59594F29CDB8549F4523D358DF04EBAB46E412C7C47C5B0C2`,
22:02:02) were stable during the audit. `ramified/metrics.py`
(`14DBF6B1794ABFD9802C503BBF5C9E3D6B2922FE88E93F12246866A952FA4D7D`,
22:22:42) and `experiments/level_assignment.py`
(`167C55CC81EE373672A88D018786996716013E76B5982F40072F7731DEEB4517`,
22:23:04) changed while I was running checks (another process, not me). The current code reproduces the
seed-42 CSV macro-F1 exactly, so the committed v512 F1 numbers are not stale
for that seed; but the current runner writes extra macro/micro columns and the
committed CSV has the old schema. Freeze a commit/hash before the paper and
rerun the full 10 seeds with that revision.

## 4. Does the v512 morph result survive?

**As a paired in-distribution result: yes.** The current code reproduces seed
42’s committed macro-F1 to 12 decimals; the 10-seed mean ΔF1 is +0.00924
(95% CI [+0.00604, +0.01245], 10/10 wins), Δbpc is −0.02579 (10/10 wins), and
the v1024/v2048 deltas are exactly zero on the three tested seeds. The primary
comparison is size-matched (`morph` and `frequency` both 294/196 at v512), the
merge rules and vocabulary are fixed, and there is no exact train/test leakage.
The effect is not a bug artifact like the v0.1 heap issue.

**As evidence of compositional generalisation: no.** The test is interpolation
over triples of seen morphemes. All three pairwise combinations are seen for
2,296–2,304/2,304 words; 96.8–98.0% of test words are one edit from a training
word; the median test word has 110 one-morpheme neighbours; and 9.3–11.7% of
test strings are substrings of training strings via the `re`/`pre` relation.
The vector ground truth is unused. On the four seeds measured at per-word
level (42–45), the effect is also sparse: 20–50 word types per seed, all ties
otherwise. The result can be explained by pairwise morpheme memorisation or
near-neighbour segmentation, not by a compositional rule.

**What survives in the paper.** The sentence in `docs/findings-v0.2.md` §5 is
defensible if it is read narrowly: a two-level assignment by morphological
provenance changes tokenization and improves boundary F1 by ~0.9 points and bpc
by ~0.026 at v512, over 10 seeds, against frequency and random controls, and
the effect disappears at v1024/2048. The paper must add: (a) the split is
unseen triples, not unseen components/pairs; (b) all pairwise morpheme
combinations are attested; (c) on seeds 42–45 the effect is 20–50 word types per seed; (d) the
vector/compositional hypothesis is untested; (e) the `re`/`pre` near-duplicate
must be removed or disclosed. Without those, the sentence “migliora
l’allineamento morfologico” overstates what the test shows.

## 5. What would change my mind

1. **Pair-disjoint split.** Hold out whole `(prefix, root)` and/or
   `(root, suffix)` pairs so no test word contains a seen pair. If morph still
   beats frequency with a similar effect, the compositional interpretation
   gains real support. This is the single most informative check.
2. **Zero-shot morphemes.** Hold out entire roots (and separately prefixes or
   suffixes). If morph still wins, the result is about morphology, not
   pairwise interpolation.
3. **Prefix-free generator.** Remove `"pre"` or `"re"`, reject root-prefix
   relations, and confirm that no test word is a substring of a training word.
   If the effect survives, M1 is not the cause.
4. **Phase 2 with the vector ground truth.** Train the tree-recursive vs flat
   mean vs GRU/Transformer vs oracle models on the pair-disjoint/zero-shot
   split and measure cosine to `P @ S @ r`. A positive compositional gap would
   support the embedding half of the hypothesis; the current Phase 1 cannot.
5. **Nearest-neighbour / pair-lookup baseline.** A string-only baseline that
   copies boundaries from the nearest training word (or a pair lookup over
   `(prefix, root)` and `(root, suffix)`) should be reported next to morph. If
   it matches morph, the morph gain is not special.
6. **Per-word reporting on all 10 seeds.** If the 20–50-word sparsity on seeds
   42–45 is representative, report win/loss/tie and the delta distribution; if
   a different seed produces a broad shift, the effect is more general than
   this audit found.
7. **Stratified test set and one evaluation sample.** Equal test words per root
   and F1/bpc on the same words would reduce the composition and sample-mismatch
   concerns.

## 6. What checked out

- **No exact synthetic leakage.** 11,520 unique strings, zero duplicate
  strings/triples, zero train/test word-string overlap, zero test words in the
  training text, zero test tokens outside `test_words`, zero training tokens
  outside `train_words ∪ {"."}`. All 2,304 test triples are absent from
  training.
- **All components attested.** 0 component failures at type level and in the
  realised text, all 10 seeds; 120/120 roots, 8/8 prefixes, 12/12 suffixes.
  Minimum root attestation is 63–88 training types (median 77).
- **Boundaries valid.** Every test word is exactly `prefix + root + suffix`,
  with `boundaries == (len(prefix), len(prefix)+len(root))` and
  `0 < b0 < b1 < len(word)`.
- **No token-level test leak.** On seed 42 flat, zero test words equal a
  tokenizer expansion and zero test words tokenize as a single token.
- **Reproduction.** Seed 42 matches the committed CSV exactly; the 10-seed
  statistics recompute; the v1024/v2048 null is exact on seeds 42–44; the
  primary v512 comparison is level-size matched.
- **Vector formula.** `P @ (S @ r)` is correct, unit norm, distinct test
  vectors. The problem is usage, not algebra.
- **The v0.2 document is honest about Phase 2.** It lists the real morphological
  corpus, Phase 2 embeddings, and MorphBPE/TreeTokenizer baselines as open.
  The audit’s contribution is to make the interpolation/near-duplicate nature
  of the current split explicit before those claims are written.

## 7. Bottom line

The synthetic holdout is **clean of exact leakage and valid at the word and
boundary level**, but it is **not a compositional split**: the test words are
new triples of components and pairs that are all seen. The v512 morph result is
a real, reproducible, small in-distribution segmentation effect, concentrated
in 20–50 word types per seed on the four seeds measured (42–45) and partly
inflated by the systematic `re`/`pre` substring relation. The vector ground
truth is never used in Phase 1, so the
compositional-meaning half of the hypothesis remains untested. The paper should
either relabel the result as “unseen combinations of seen morphemes” and report
the per-word sparsity, or add a pair-disjoint/zero-shot split and a Phase 2
vector evaluation before claiming compositional generalisation.
