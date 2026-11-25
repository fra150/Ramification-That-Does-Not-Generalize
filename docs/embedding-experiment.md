# Phase 2 — Compositional embeddings on the ramified derivation forest

**Status:** design document. Nothing in this file is implemented. No source file is
changed by this phase's design. The document specifies an experiment that can be
implemented and run after the prerequisites in §8.1 are met.

**Critical finding about the current tokenizer (measured, current code).**
With the current `ramified/bpe.py`, hierarchical BPE is *exactly equivalent* to
flat BPE on this corpus. The levels are trained sequentially on the output of the
previous level, and each level continues the same deterministic greedy merge
sequence; the concatenation of the level merge lists equals the flat merge list.
Measured at vocabulary 512: flat `[490]` merges equals `hier3` `[245, 147, 98]`
concatenated, and the tokenizations of all train and test words are identical at
both holdout splits (9,216 + 2,304 words at holdout 0.2; 8,064 + 3,456 at holdout
0.3). The same holds at 256, 1024, 2048 and 4096: test and train diff counts
are 0, and the concatenated merge lists match. Boundary F1, `morph3`, forest
shape and top-level alignment are therefore identical across `n_levels`.

The "hierarchical advantage" quoted in an earlier calibration (boundary F1 0.893
versus 0.874 at vocabulary 512, `morph3` 0.84 versus 0.77) is **not reproducible
from the current code**. The cause is not established; it may reflect a
concurrent source change or an earlier tokenizer implementation. The current
measurements are the ones used throughout this document; the earlier numbers are
withdrawn.

Consequences:

- **C2 (ramification) is not testable as implemented.** `n_levels` and
  `level_weights` only partition the same merge sequence; they do not change the
  tokenization. Running the tokenizer axis now would be a null manipulation
  check, not a scientific comparison.
- **C1 (structure) is testable.** Flat BPE still produces a nontrivial binary
  merge tree; the tree model can compose it. Phase 2's primary endpoint is C1 on
  the shared tokenizer (flat BPE; `hier3` is identical in the current code).
- Before C2 can be tested, the tokenizer must be changed so that the level
  structure alters the merge sequence (prerequisite **P5** in §8.1). That is a
  real implementation task, not a hyperparameter search. Constraining merges
  with true morpheme boundaries would be oracle leakage and is not an acceptable
  fix.

**What Phase 1 established (context, corrected).** The tokenizer is a valid
finite presentation and is exactly decodable. It does **not** establish a
hierarchical advantage: the current implementation's levels are a no-op. Phase 2
must not claim C2 from the current code; it can test C1 and it must treat the
tokenizer axis as a manipulation check until P5 is done.

**What Phase 2 decides.** The v0.1 claim bundles two separable claims:

- **C1 (structure):** a model that composes the derivation forest recursively
  generalises better to unseen combinations than a model that sees the same
  tokens as a flat bag or a flat sequence.
- **C2 (ramification):** a hierarchical/ramified tokenizer produces a better
  compositional embedding than flat BPE at matched vocabulary and compression.
  **Blocked pending P5.**

(C1 and C2 are the two claims; KC1–KC11 in §7.2 are the kill criteria. The
labels are deliberately distinct.)

Phase 2 is a crossed design over *embedding model* × *tokenizer*; the headline
contrast is 2×2 (tree vs flat × flat BPE vs hierarchical). C1 can be true while
C2 is false, and vice versa. The headline result must report them separately.
Conflating them is the main way this experiment could produce a misleading
positive. With the current code the tokenizer factor is a no-op, so only the
model factor is scientifically live; the 2×2 is the design once P5 makes the
tokenizer factor real.

**Primary endpoint (pre-registered).** Held-out-combination cosine similarity at
vocabulary 512, comparing the best tree model against the best flat model
(selected on dev), using the shared tokenizer (flat BPE; `hier3` is identical in
the current code). This is the most favourable cell for C1: boundary F1 is
0.8665, 78% of test words contain all three morphemes as constituents, 72% have
exactly three top-level tokens at the two morpheme boundaries, and 73% have both
boundaries top-level, so a position-conditioned right fold can produce the
correct `(prefix, (root, suffix))` nesting. If the tree model does not win here,
C1 is falsified for this corpus. The rest of the vocabulary grid is reported as
secondary.

**Contents.** 1 Formal setup · 2 Ground truth · 3 Models · 4 Training objective
and data · 5 Metrics · 6 Ablations · 7 Failure modes and kill criteria ·
8 Implementation plan · 9 Threats to validity · Appendices A–B.

---

## 1. Formal setup

### 1.1 Symbols and the finite presentation

Let \(A\) be a finite terminal alphabet. In the prototype \(A\) is the set of
characters observed in the training text, plus two bookkeeping terminals:
`EOW` (end of word) and `UNK`. A finite presentation is

\[
\mathcal{G} = (A, R), \qquad R \subseteq A \times A^{*},\ R \text{ finite}.
\]

`bpe.py` restricts every rule to arity 2: a merge rule is a triple
\((a,b) \mapsto c\), organised in levels. The set of finite symbols is the
initial algebra

\[
S(\mathcal{G}) = \mu X.\; A + (X \times X)
\]

for the binary prototype (the general polynomial functor
\(A + \coprod_{\rho \in R} X^{|\rho|}\) is the intended mathematical object;
the code only implements the binary case). The carrier \(S(\mathcal{G})\) is the
set of all finite terms over \(A\) and the binary constructor. The tokenizer's
vocabulary is a **finite subset** of this carrier: the terminals plus the symbols
created by the learned rules. Every symbol \(s\) has a unique expansion
\(\sigma(s) \in A^{*}\):

\[
\sigma(a) = a \quad (a \in A), \qquad
\sigma(\mathrm{mk}(a_1,a_2)) = \sigma(a_1)\,\sigma(a_2).
\]

In code this is exactly `Tokenizer.expansions[s]`, with `EOW` expanding to a
sentinel and `UNK` expanding to `<unk>`. The expansion is a morphism from the
term algebra to the free monoid (with concatenation as the monoid operation):
\(\sigma(\mathrm{mk}(a_1,a_2)) = \sigma(a_1)\sigma(a_2)\). It is **not** injective in
general: two different merge paths can spell the same string. On the synthetic
corpus at all vocabularies tested, no two distinct symbol ids had the same
expansion string (measured collision count: 0), so the non-injective channel is
dormant here. Do not use this corpus to make claims about §5 of
`docs/ramified-v0.md`.

### 1.2 The derivation forest

For a word \(w\), `Tokenizer.encode_nodes(w, eow=True)` returns a list of roots
\(r_1,\dots,r_m\). Each root is a binary tree; reading the leaves left to right
spells \(w \cdot \mathrm{EOW}\). The list is a **forest** because a word is
usually split into several top-level symbols, not one. Internally, at each level
the encoder repeatedly applies the lowest-rank applicable merge, replacing two
adjacent nodes by their parent. The result is deterministic given the tokenizer.
It is a parse, not a distribution over parses.

Two consequences matter for the embedding design:

- The forest roots are exactly the token ids returned by `encode()`. All models
  compared in Phase 2 therefore see **the same token sequence**; the tree model
  additionally sees the binary structure inside each token.
- The forest is not a tree. A word embedding must aggregate several roots. The
  aggregation is part of the model and must be held fixed across the comparison,
  or ablated explicitly.

For span bookkeeping, assign `EOW` length 0 (it is a boundary marker, not a
character of \(w\)). `UNK` does not occur on this corpus because the alphabet is
closed; `forest.assert_forest` should treat an `UNK` leaf as an error rather
than silently assigning it zero characters. `Tokenizer.word_boundaries` uses
`len(expansions[s])` and therefore counts `EOW` as 5 characters; it happens to be
safe because it drops the last node, but Phase 2 must not use it. Use the node
spans defined in §8.2 instead.

### 1.3 The compositional embedding

Fix an embedding dimension \(d\). A terminal embedding is a map
\(E : A \to \mathbb{R}^{d}\). A composition operator is a map
\(g_\rho : \mathbb{R}^{d} \times \mathbb{R}^{d} \to \mathbb{R}^{d}\),
parameterised by a role \(\rho\) (see §1.5). The embedding
\(e : S(\mathcal{G}) \to \mathbb{R}^{d}\) is the unique fold

\[
e(a) = E(a), \qquad
e(\mathrm{mk}(a_1,a_2)) = g_\rho\big(e(a_1), e(a_2)\big).
\]

This is an algebra homomorphism from the term algebra
\((S(\mathcal{G}), \text{constructors})\) to
\((\mathbb{R}^{d}, g_\rho)\). It is **not** a monoid homomorphism: there is no
requirement that \(e(\sigma^{-1}(xy)) = e(x) + e(y)\), and \(g_\rho\) need not
be associative. The word "homomorphism" in `ramified-v0.md` §3 means the former,
not the latter. This distinction matters: a flat mean is a monoid-style
aggregation (sum/mean over tokens), while the tree model is a term-algebra fold.

For a forest \(r_1,\dots,r_m\), the word embedding is

\[
e(w) = h\big(e(r_1),\dots,e(r_m)\big),
\]

where \(h\) is a fixed aggregation. Phase 2 uses three choices:

- **flat mean:** \(h = \operatorname{mean}\). This is the flat baseline.
- **right fold:** \(h(x_1,\dots,x_m) = g_{\sigma(1)}(x_1, g_{\sigma(2)}(x_2, \dots))\)
  with a position-conditioned sequence role \(\sigma(i)\). For a three-root
  forest at the morpheme boundaries the sides \((R,L)\) produce \((p,(r,s))\),
  the nesting required by the target.
- **sequence encoder:** \(h = \mathrm{GRU}\) or a small Transformer over the root
  sequence. This is the flat sequence baseline.

A left fold \(((p,r),s)\) is the wrong nesting for this generator and is an
explicit ablation, not the default.

### 1.4 What the models are compared on

All models receive the same word string and the same frozen tokenizer. The
input to every model is the forest root sequence \(r_1,\dots,r_m\). The
differences are:

| Model | Sees token ids | Sees token order | Sees binary tree | Sees morpheme boundaries |
|---|---|---|---|---|
| Flat mean | yes | no (mean) | no | no |
| Flat position mean | yes | yes (position embeddings) | no | no |
| Flat GRU / Transformer | yes | yes | no | no |
| Tree (pure / hybrid) | yes | yes | yes | no |
| Oracle morpheme (O2) | yes (morpheme ids) | yes | yes (true 3-node tree) | yes |

The only model allowed to see `Word.boundaries` is O2. This is enforced by the
data API (§8.2), not by convention.

### 1.5 Representability: why the composition operator needs roles or depth

This subsection is not a philosophical aside. It determines the architecture and
predicts a specific failure mode.

Assume the tree is right-associated, \(\tau = (p,(r,s))\), and the operator at a
node is of the form

\[
g(x,y) = M(x)\,y,
\]

where \(x\) is the **operator child**, \(y\) is the **argument child**, and
\(M : \mathbb{R}^{d} \to \mathbb{R}^{d \times d}\) is linear in \(x\). The
operator child is selected by the node's role, not by a fixed left/right
convention: at the suffix node the right child (the suffix) is the operator; at
the prefix node the left child (the prefix) is the operator. This role
alternation is not cosmetic. With a fixed left-child-operator convention, the
suffix node would compute \(M(e_r)e_s\), not \(S_s e_r\), and the target would be
unrepresentable even with perfect segmentation. The model therefore needs an
operator-side parameter per role; it can be a discrete choice selected on dev or a
learned gate over the two orders (the ablation in §6.6). With a **shared** \(M\)
at both nodes,

\[
e(\tau) = M(e_p)\,M(e_s)\,e_r.
\]

The target is \(P_p S_s r\). Exact representation for all \((p,s)\) therefore
requires

\[
M(e_p)\,M(e_s) = P_p S_s \qquad \text{for all prefixes } p \text{ and suffixes } s.
\]

Fixing \(s\) and using invertibility of \(P_p\), this forces
\(M(e_s) = C^{-1} S_s\) for a constant \(C\), and then \(M(e_p) = P_p C\).
So the same linear map \(M\) must send the 12 suffix embeddings to the matrices
\(C^{-1}S_s\) and the 8 prefix embeddings to the matrices \(P_p C\). A linear map
from \(\mathbb{R}^{d}\) has an image of dimension at most \(d\); it can therefore
interpolate at most \(d\) linearly independent matrices. There are
\(8 + 12 = 20\) component matrices. For \(d = 16\), a shared operator cannot
represent the full target family exactly. For internal width \(w \ge 20\), it can
(in principle). This yields two testable predictions:

- **P1:** a shared-\(M\) tree at width 16 will underfit the training set, and its
  train cosine will be below that of a depth-specific tree at the same width.
- **P2:** widening the internal representation to 32 or 64 will close the
  shared-versus-depth gap, because the rank obstruction disappears.

With **depth-specific** operators \(M_{\mathrm{in}}\) for the suffix node and
\(M_{\mathrm{out}}\) for the prefix node, exact representation is possible at
\(d = 16\): each family has at most 12 members, and 12 ≤ 16. This is why the
primary tree model uses role/depth-specific composition. A shared operator is a
diagnostic ablation, not the primary model.

The lemma is stated for the operator form. A general bilinear map
\(g(x,y)_i = \sum_{jk} T_{ijk} x_j y_k\) has more freedom; whether a shared
general bilinear map can represent 20 independent component matrices at \(d=16\)
is not settled by the argument above. The shared-\(g\) ablation is the empirical
test. Be honest about this scope in the write-up.

There is a second, independent representability issue: **tree shape**. If the
parse groups the prefix and root first, \(((p,r),s)\), the inner node applies
the prefix and the outer node applies the suffix, producing \(S_s P_p r\) rather
than \(P_p S_s r\). For generic non-commuting orthogonal matrices these differ,
and no bilinear operator can undo the swap. The only fix is a non-bilinear model
with enough capacity to invert the mixing, or a different parse. This is the main
reason the experiment measures parse shape (§7.1), not just tokenizer quality.

---

## 2. Ground truth

### 2.1 Generator

`ramified/synthetic.py` builds a grid of words \(w = p\,r\,s\) with
\(|P| = 8\) prefixes, \(|R| = 120\) roots, \(|S| = 12\) suffixes, for
\(8 \times 120 \times 12 = 11{,}520\) words. For each prefix \(p\) and suffix
\(s\), a random orthogonal matrix is drawn by QR:

\[
P_p,\ S_s \in O(d), \qquad d = 16,
\]

and each root gets a unit vector \(r_r \in S^{d-1}\). The target is

\[
v(w) = P_p\,S_s\,r_r \;\in\; S^{d-1}.
\]

The code computes `prefix_mat[p] @ (suffix_mat[s] @ root_vec[r])`, which is the
same nesting: the suffix acts first, the prefix acts second. All targets are
unit norm (measured min/max: 0.9999999999999996 / 1.0000000000000007). The
composition is non-commutative by construction: \(P_p S_s \neq S_s P_p\) for
generic orthogonal matrices. A flat bag that loses order cannot recover the
target except by accident.

The orthogonality is important for the embedding design: every intermediate
\(S_s r_r\), \(P_p S_s r_r\) is also unit norm. A composition that normalises
each node is therefore not an arbitrary restriction; it matches the geometry of
the target. This is the justification for the normalised composition in §3.1.

### 2.2 Held-out combinations

The generator holds out whole \((p,r,s)\) triples, then filters the held-out set
so that every component of every held-out word appears in training. The
condition is therefore:

> a test word is an **unseen combination of seen components**.

This is exactly the compositional-generalisation condition. It is *not* a test
of unseen morphemes. With random orthogonal component matrices, an unseen root
or an unseen affix is information-theoretically unlearnable from its spelling:
the vector is independent of the string. Unseen-component splits are therefore
**null controls** (§6.4), not performance tests.

The default `holdout=0.2` gives 9,216 train words and 2,304 test words (measured,
seed 42; no string collisions, no train/test overlap). For Phase 2, use
`holdout=0.30`: 8,064 train words and 3,456 held-out words (measured, seed 42).
Split the held-out pool deterministically into 1,152 dev and 2,304 test words,
stratified by \((p,s)\) so that all 96 affix pairs are represented in both. This
keeps the test size at 2,304 while creating a clean dev set that the tokenizer
never saw. The exact recipe is in §4.3.

### 2.3 What the target contains, and what it does not

The target is exactly compositional with respect to the *morphemes*, but it is
**not** linearly decodable per component from a single vector. For a fixed
component class, the other components range over enough values that the class's
points span \(\mathbb{R}^{16}\) and their convex hull overlaps the other classes.
The empirical check on the exact ground truth (2,304 test words from the
holdout-0.2 split, 3-fold CV) confirms this:

| Probe target | Classes | Chance | Logistic regression | MLP (128) | 5-NN |
|---|---:|---:|---:|---:|---:|
| prefix | 8 | 0.125 | 0.118 | 0.102 | 0.115 |
| root | 120 | 0.008 | 0.011 | 0.009 | 0.007 |
| suffix | 12 | 0.083 | 0.086 | 0.099 | 0.085 |

All probes are at chance. This is a property of the generator, not a failure of
the model. It has a direct consequence for §5.3: **component-identity probes are
not a valid metric**. They are reported only as a negative control. The valid
linear probe is a ridge readout of the target vector itself, which measures
whether the model's representation is linearly aligned with the target.

### 2.4 Geometry anchors

Measured on the 2,304 test words (seed 42, holdout 0.2):

- Pairwise cosine between distinct targets: mean 0.0001, sd 0.2500,
  min −0.908, max 0.898. This is the expected \(1/\sqrt{d}\) behaviour for
  \(d=16\).
- Random unit vector versus target: mean cosine 0.007.
- Root centroid computed on the test set: mean cosine 0.231. This is an **oracle
  diagnostic**, not a trainable baseline: it uses the test targets to compute
  per-root centroids. It is the "knows the root, nothing else" reference. A model
  at cosine 0.23 has not demonstrated composition; it has only clustered roots.
- Global retrieval chance: \(1/2304 = 0.00043\) for R@1.

These anchors are used in §5.6 to interpret absolute scores. They are the reason
the experiment reports the root-centroid line alongside the trained baselines.

---

## 3. Models to compare

All models output a 16-dimensional vector, L2-normalised before the loss and
before evaluation. All models are trained with the same optimizer budget and the
same loss. Parameter counts are reported for every run; capacity is controlled
in §6.9.

### 3.1 (a) Tree-recursive composition

**Primary variant — role/depth-specific operator, pure recursion.**
For each internal node with children \((a,b)\), depth bucket \(\delta\), and an
operator side \(\omega \in \{L,R\}\) (which child supplies the matrix), define

\[
g_{\rho}(a,b) =
\mathrm{normalise}\big(M_{\rho}(x_\omega)\,x_{\bar\omega}\big),
\qquad \rho = (\omega,\delta),
\]

where \(x_L = a\), \(x_R = b\), and
\(M_{\rho} : \mathbb{R}^{d} \to \mathbb{R}^{d \times d}\) is a learned linear
map (equivalently, a rank-3 tensor of size \(d^3\)). The operator side is a
per-depth parameter: either a pre-registered discrete choice (primary: right at
the suffix node, left at the prefix node) or a learned gate over the two orders
(ablation). A fixed left-child convention is a **bug**, not a simplification: it
computes \(M(e_r)e_s\) at the suffix node and cannot represent \(S_s e_r\) (§1.5).
For the forest, use a right fold with a **position-conditioned sequence role**:

\[
u_{m-1} = e(r_{m-1}), \qquad
u_i = g_{\sigma(i)}\big(e(r_i), u_{i+1}\big),
\qquad
e(w) = u_0,
\]

where \(\sigma(i)\) is indexed by the position from the right. The sequence fold
shares one \(M\) across positions but uses an **operator side per position**: the
primary uses the string-order choice (\(R\) at the innermost step, \(L\) at the
outermost); a learned gate over the two orders is an ablation. A single
fixed-side sequence role is not enough: the prefix is a left operator and the
suffix is a right operator, the same role alternation as §1.5. For a three-root
flat forest \([p,r,s]\), the right fold with sides \((R,L)\) computes
\(g_L(p, g_R(r,s)) = P_p S_s r\), the correct nesting. A right fold with a fixed
left-operator side computes \(M(p)(M(r)s)\); a left fold with a left-operator
side computes \(M(M(p)r)s\). Neither is \(P_p S_s r\).

There is **no learned embedding for nonterminal symbols**: every nonterminal
value is computed from its children. This is the strong test of the
homomorphism claim. Terminals (characters, `EOW`) have learned vectors. Only the
word-level target is supervised; internal node values are latent and receive
gradients only through the final word loss. There is no per-morpheme
supervision, which would be oracle leakage.

Why this parameterisation: §1.5 shows that, for the operator form, a
depth/role-specific \(M\) can represent the target exactly at \(d=16\), while a
shared \(M\) cannot in general. A general bilinear shared map is the subject of
the ablation.
The depth bucket is capped at 5 (measured maximum tree depth is 5); the operator
side is \(L/R\). The sequence fold has its own \(M_{\mathrm{seq}}\) (one extra
role) and a side choice per forest position (primary string-order, gate ablation).
The number of \(M\) matrices is at most \(2 \times 6 + 1 = 13\).

Depth is a proxy for role. If a depth bucket mixes prefix and suffix nodes, one
side bit per depth is insufficient. The `level`-conditioned variant uses the
merge level from `tok.levels` (via `forest.symbol_levels`) as an additional role
key; it is an exploratory ablation, not the primary, because the level is a
tokenizer artefact and could correlate with frequency.

**Variants (ablations, not the primary model):**

- `tree_shared`: one \(M\) for all roles, with the same learned operator sides.
  This is the representability diagnostic; it isolates the rank constraint from
  the operator-side choice.
- `tree_leftfold`: forest aggregated left-to-right, \(((r_1,r_2),r_3)\). This is
  the wrong nesting for the generator and should lose when the forest is flat.
- `tree_hybrid(k)`: symbols with expansion length \(\le k\) use a learned
  embedding; longer symbols are composed. \(k \in \{0,1,2,3,\infty\}\), where
  \(k=0\) is pure recursion and \(k=\infty\) is a full lookup (a lookup-based
  model with tree aggregation, not the flat mean). This isolates "compose seen
  units into unseen combinations" from "reconstruct arbitrary sub-morphemic
  units from characters". The primary scientific claim uses the pure model; the
  hybrid is the practical variant and is included in the "best tree" selection
  for the kill criteria.
- `tree_width(w)`: internal width \(w \in \{16,32,64\}\), output projected to 16.
  This tests P1/P2 in §1.5.

**Normalisation:** each node output is L2-normalised. This is consistent with
the target's unit-norm intermediates (§2.1). An ablation without normalisation
is optional; it is not primary.

**EOW:** the `EOW` terminal is a learned leaf with zero character length. It
participates in the composition of the final token. All models see it, so it is
not a confound; it is a word-boundary cue, not a target leak.

### 3.2 (b) Flat mean of token embeddings

\[
e_{\mathrm{flat}}(w) = \mathrm{normalise}\left(\frac{1}{m}\sum_{i=1}^{m} E(r_i)\right),
\]

where \(r_i\) are the forest roots (the same token ids the tree model uses) and
\(E\) is a learned embedding table over the tokenizer vocabulary. This is the
canonical "bag of subwords" baseline. It has no order and no structure.

**Variant `flat_posmean`:** add a learned position embedding \(p_i\) to each
token before averaging. This is a stronger flat baseline that knows order but
not hierarchy. It is important because it separates "the tree model wins because
it knows order" from "the tree model wins because it knows structure".

Out-of-vocabulary tokens: the measured test OOV token rate is 0.000 at
vocabularies 512 and 1024, 0.001 at 2048, and 0.007 at 4096. Flat models are
therefore **not** OOV-handicapped in the primary cell. If an OOV token occurs,
it maps to a learned `<unk>` vector; report the rate. This removes the most
common unfair-comparison objection to flat baselines.

### 3.3 (c) Flat sequence model

**Primary sequence baseline: 1-layer GRU.** Token embeddings \(E(r_i)\), hidden
size 64, final hidden state projected to 16 and normalised. For sequences of
2–6 tokens (measured forest sizes: 2–6, mode 3 at vocabulary 512; mode 2 at
2048), the GRU is effectively a small order-sensitive bag. That is the point:
it is a strong, generic flat baseline with no tree.

**Optional Transformer:** 2 layers, 4 heads, \(d_{\text{model}}=64\), learned
positional embeddings, mean-pool over tokens, linear projection to 16. Include
only if the runtime budget allows; the GRU is the pre-registered sequence
baseline. With 2–3 tokens, a Transformer has little sequential structure to
exploit and is not expected to differ much from the GRU.

**Optional `flat_concat_mlp`:** concatenate the token embeddings in order
(padded to max length 8), apply a 2-layer MLP (64 hidden), project to 16. This
is a capacity control that sees order and token identity but no hierarchy. It is
often the strongest flat baseline at these sequence lengths. If it beats the
tree model, the structural claim fails even if the tree beats the flat mean.

### 3.4 (d) Oracle morpheme composition

Two distinct objects, often conflated:

- **O1 — exact oracle.** \(e(w) = v(w)\), the ground-truth vector. Cosine is
  1.000 by construction. This is a **pipeline check**, not an upper bound. If it
  is not 1.0 to numerical precision, the metric or the data alignment is broken.
- **O2 — oracle segmentation, learned composition.** The model is given the true
  morpheme spans \((0,|p|)\) and \((|p|,|p|+|r|)\), the morpheme strings, and
  learns an embedding for each distinct prefix/root/suffix plus a depth-specific
  composition operator. It is *not* given \(P_p\), \(S_s\), or \(r_r\). Its tree
  is the true \((p,(r,s))\) tree. This is the informative upper bound: it
  isolates "can the architecture learn the composition at all, given perfect
  segmentation?" from "can the tokenizer find the segmentation?"
  
  O2 has 140 morpheme embeddings (8 + 120 + 12) and two composition maps. At
  \(d=16\) it is exactly representable by §1.5. It is trained on the same 8,064
  words, with the same loss and budget, and evaluated on the same test set.
  
  O2's test score is the ceiling for any model that respects morphemes. The
  decomposition
  \[
  \underbrace{\text{O2} - \text{tree}}_{\text{tokenizer/structure cost}}
  \quad\text{and}\quad
  \underbrace{\text{tree} - \text{flat}}_{\text{structural benefit}}
  \]
  is the main quantitative story. If O2 itself is low, the bottleneck is the
  architecture or optimization, not the tokenizer; kill criterion KC1 fires.

**Null-control version of O2:** on the held-out-root and held-out-affix splits
(§6.4), O2 has no embedding for the unseen component and must use `<unk>`. Its
score should be at chance. If O2 scores above chance on an unseen root, the
root vector is somehow predictable from the string; that would indicate a
generator bug or a leak, not a success.

### 3.5 Parameter and compute matching

Approximate parameter counts at vocabulary 512, \(d=16\):

| Model | Parameters (order) |
|---|---:|
| Flat mean | \(512 \times 16 \approx 8\text{k}\) |
| Flat position mean | \(8\text{k} + 8 \times 16 \approx 8\text{k}\) |
| Flat GRU (hidden 64) | \(8\text{k} + 3 \times 16 \times 64 + 3 \times 64^2 + 64 \times 16 \approx 24\text{k}\) |
| Flat concat MLP | \(8\text{k} + 8 \times 16 \times 64 + 64^2 + 64 \times 16 \approx 21\text{k}\) |
| Tree, depth/role-specific \(M\) | \(13 \times 16^3 \approx 53\text{k}\) (fewer if roles are bucketed) |
| Tree, shared \(M\) | \(16^3 \approx 4\text{k}\) + terminals |
| O2 oracle | \(140 \times 16 + 2 \times 16^3 \approx 10\text{k}\) |

The tree model is larger than the flat mean at small vocabulary. This is a real
confound. Two controls:

1. **Capacity control:** widen the flat token embeddings to 64 dimensions and
   add a linear map to 16, matching the tree's parameter count. If the tree's
   advantage disappears, it was capacity.
2. **Shared-\(M\) control:** the shared tree has far fewer parameters and is
   expected to underfit (§1.5). If the shared tree matches the depth-specific
   tree, the representability story is wrong; investigate before interpreting
   the main result.

Do not claim a structural effect from a comparison that only differs in
parameter count. Report parameter counts in every table.

---

## 4. Training objective and data

### 4.1 Objective

Primary loss, averaged over a batch:

\[
\mathcal{L} = \frac{1}{B}\sum_{i=1}^{B}
\left(1 - \cos\big(\hat e(w_i),\, v(w_i)\big)\right),
\qquad \hat e = \frac{e}{\|e\|_2}.
\]

Why cosine: the targets are unit norm; cosine is scale-free; it is the primary
evaluation metric, so training and evaluation are aligned. Mean squared error on
normalised outputs is equivalent up to the monotone transform
\(\|\hat e - v\|^2 = 2 - 2\cos(\hat e,v)\); there is no reason to train on MSE
and evaluate on cosine.

Secondary loss (one ablation, not primary): symmetric InfoNCE with temperature
\(\tau = 0.07\), using the other words in the batch as negatives:

\[
\mathcal{L}_{\text{NCE}} = -\frac{1}{B}\sum_i
\log \frac{\exp(\cos(\hat e_i, v_i)/\tau)}
{\sum_j \exp(\cos(\hat e_i, v_j)/\tau)}.
\]

This is legitimate: in-batch negatives are other training words, not test words.
It is expected to help retrieval more than cosine, but it changes the objective;
keep cosine as the pre-registered primary.

Optimizer and schedule, fixed across all models:

- AdamW, learning rate \(10^{-3}\), weight decay \(10^{-4}\).
- Linear warmup 50 steps, cosine decay to \(10^{-5}\).
- Batch size 256, 800 optimizer steps (≈25 epochs over 8,064 words).
- Early stopping on dev cosine, patience 100 steps, max 800 steps.
- No dropout in the tree; dropout 0.1 in the GRU/Transformer.
- Seeds: model init and batch order from the model seed; data split and
  tokenizer from the data seed.
- Gradient clipping at norm 1.0 (relevant for the normalised bilinear tree).

Report actual steps for every run; the same patience and max apply to all models,
so an early-stopped run is a fair outcome, not a confound.

The step budget, not the epoch count, is held fixed. With 8,064 words and batch
256, one epoch is 32 steps; 800 steps is 25 epochs. This is enough for the
oracle and the flat models; the tree may need more. Run a learning-curve check
on dev for the first cell before fixing 800. If the tree is still improving at
800, raise the budget for **all** models equally, and report the change.

### 4.2 What is learned versus fixed

| Object | Status | Notes |
|---|---|---|
| Tokenizer merges, levels, expansions | **fixed** | trained on `train_text` only, per (data seed, method, vocabulary); frozen before any embedding training |
| Forest parse | **fixed** | deterministic from the tokenizer; no model may re-parse |
| Target vectors \(v(w)\) | **fixed** | generated; never fitted except through the loss |
| Train/dev/test split | **fixed** | pre-registered; test evaluated once |
| \(d=16\), batch, steps, optimizer | **fixed** | same for all models |
| Terminal embeddings | **learned** | all models; tree terminals only in the pure variant |
| Nonterminal embeddings | **learned in hybrid/flat** | absent in the pure tree; present for symbols with expansion length \(\le k\) in the hybrid; present for all roots in the flat models |
| Composition operator \(g_\rho\) | **learned** | role/depth-specific, shared, or MLP, depending on variant |
| Sequence encoder | **learned** | GRU/Transformer/MLP |
| Oracle morpheme embeddings | **learned** | O2 only |
| Morpheme boundaries | **forbidden** | O2 only; all other models receive only the string and the tokenizer |
| \(P_p, S_s, r_r\) | **forbidden** | never exposed |
| Hyperparameters | **selected on dev** | pre-registered grid, not on test |

### 4.3 Splits

Use the current generator with `holdout=0.30`, seed \(s_d \in \{42,43,44\}\):

1. `syn = generate_synthetic(seed=s_d, holdout=0.30)`.
2. `train_words = syn.train_words` (8,064 strings; measured, seed 42).
3. `heldout = syn.test_words` (3,456 `Word` objects; measured, seed 42).
4. Split `heldout` deterministically with a separate `split_seed` (e.g., 1000+s_d),
   stratified by `(prefix, suffix)`: 1/3 dev (1,152), 2/3 test (2,304).
5. Assert:
   - `len(train_words) == 8064`, `len(dev) == 1152`, `len(test) == 2304`
     for seed 42; report the exact counts for every seed.
   - train/dev/test word strings are pairwise disjoint.
   - every dev/test component (prefix, root, suffix) appears in train.
   - no dev/test word appears in `train_text` (tokenize and compare strings).
   - the tokenizer was trained on `train_text` only.

**Blocker:** `SyntheticCorpus` currently returns `train_words` as a list of
strings, not `Word` objects. The embedding experiment needs the target vector
for every train word. The clean fix is a 2-line additive change in
`ramified/synthetic.py`: add `all_words: list[Word]` to `SyntheticCorpus` and
return the shuffled full grid. The experiment can then derive
`train_objs = [w for w in all_words if (w.prefix,w.root,w.suffix) not in dev_triples | test_triples]`.
Do not reconstruct the grid by re-implementing `_make_root`/`_orthogonal`; that
duplicates private RNG logic and is brittle. This is prerequisite P0 in §8.1.

Why `holdout=0.30`: it creates a large enough held-out pool for a dev set
without changing the test size, and it keeps the tokenizer blind to both dev and
test. A 50/50 dev/test split of the existing 20% test set would work but leaves
only 1,152 test words, halving the power. Regenerating a separate test set with
a different seed would change the component geometry and is not comparable.

The `test_text` field is **not used** in Phase 2. The embedding task is
type-level, not token-level. Do not train the tokenizer on `test_text`; that is
the most damaging possible leak (test words become single symbols).

### 4.4 Training distribution

Primary: **uniform over training word types**, one pass per epoch over the 8,064
train words. The hypothesis is about representing word types compositionally,
not about language modelling. Uniform-over-types also makes the fit set and the
test set comparable: both are (nearly) uniform over combinations, so the
seen-versus-unseen gap is not confounded with frequency.

Secondary ablation: sample training words with the Zipfian weights used to build
`train_text` (or simply sample from `train_text`). This matches the corpus
statistics but confounds the gap with frequency. If the Zipfian model is much
worse on test, that is informative about the tokenizer's frequency bias; it is
not the primary result.

Report, for every run, the number of train word types that never appear in
`train_text` (7,478 of 9,216 distinct word types appear at holdout 0.2, excluding
the `.` separator; the analogous number at 0.3 should be reported). These words
are seen by the embedding model but not by the tokenizer's merge statistics. They
are a useful stress subset.

### 4.5 Leakage controls

These are assertions in the harness, not guidelines.

| ID | Control | Failure action |
|---|---|---|
| L1 | Train/dev/test word strings are pairwise disjoint | abort |
| L2 | Tokenizer is fitted on `train_text` only; no dev/test string reaches `train_tokenizer` | abort |
| L3 | Every dev/test component is attested in train | abort |
| L4 | `Word.boundaries` is passed only to O2; other models receive a view without it | abort |
| L5 | Hyperparameters are selected on dev; test is evaluated once per run | abort |
| L6 | Any target normalisation (mean/PCA/whitening) is fitted on train only | abort |
| L7 | OOV token rate is reported; flat models use a learned `<unk>`, not a composition fallback | report |
| L8 | No symbol collision is silently merged; report duplicate expansions | report |
| L9 | Data seeds are fixed in advance; all seeds are reported; no seed is dropped | abort |
| L10 | **Label-shuffle canary:** train the best model on permuted train targets; test cosine must be ≈ 0 (|mean| < 0.05) | abort |
| L11 | **Unseen-component null:** O2 and all models must be at chance on held-out roots/affixes (|mean cos| < 0.15) | abort |
| L12 | **O1 exact oracle** cosine = 1.000000 on test | abort |
| L13 | The training loader receives only train examples; `all_words` (which contains dev/test targets) is used only to construct splits and is never passed to the model factory | abort |

L10 is the strongest pipeline check. If a model trained on shuffled targets
scores above 0.05 on test, the evaluation set or the feature pipeline is leaking
(for example, test words appearing in the tokenizer's training text, or a
target-derived normalisation fitted on test). L11 is the strongest
generator/leak check: an unseen root's vector is independent of its spelling, so
any above-chance score is a bug.

---

## 5. Metrics

All metrics are computed on the 2,304 test words, per seed, and then aggregated.
Report per-prefix (8) and per-suffix (12) breakdowns; a single bad affix should
not be hidden by the mean.

### 5.1 Cosine similarity to ground truth

\[
\cos_i = \frac{\langle \hat e(w_i), v(w_i)\rangle}
{\|\hat e(w_i)\|_2\,\|v(w_i)\|_2}.
\]

Report mean, median, 10th and 90th percentiles, and the fraction with
\(\cos > 0.5\). The mean alone can hide a bimodal distribution (some words
memorised, some at chance); the percentiles are required.

Interpretation anchors (§2.4): random ≈ 0.00 ± 0.25; root-only ≈ 0.23;
0.5 is moderate; 0.8 is strong; 0.95 is near-ceiling. Do not describe a score of
0.3 as "compositional success"; it is barely above root clustering.

### 5.2 Retrieval@k

For each test word \(i\), rank all candidate test targets \(v_j\) by
\(\cos(\hat e_i, v_j)\). Include \(j=i\); R@k is 1 if the true target is in the
top \(k\). Chance is \(k/N\).

- **Global R@1, R@5, R@10, R@50, MRR** over all 2,304 candidates.
- **Hard-negative same-root R@1:** candidates are test words sharing the root
  (≈19 per root); chance ≈ 1/19 = 0.053. This tests whether the model uses the
  affixes, not just the root.
- **Hard-negative same-affix R@1:** candidates share the prefix and suffix but
  differ in the root (≈24 per affix pair in the 20% test split, since
  \(2304/96 \approx 24\)); chance is the inverse of the candidate count. This
  tests root recovery.

Global retrieval saturates quickly. The measured cosine-to-R@1 calibration on
this test set (prediction = \(\alpha v + \sqrt{1-\alpha^2}\,\text{noise}\), unit
norm) is:

| Cosine | Global R@1 | R@10 | R@50 |
|---:|---:|---:|---:|
| 0.20 | 0.006 | 0.034 | 0.105 |
| 0.30 | 0.009 | 0.051 | 0.179 |
| 0.40 | 0.022 | 0.111 | 0.307 |
| 0.50 | 0.053 | 0.227 | 0.516 |
| 0.60 | 0.118 | 0.457 | 0.756 |
| 0.70 | 0.306 | 0.786 | 0.971 |
| 0.80 | 0.765 | 1.000 | 1.000 |
| 0.90 | 1.000 | 1.000 | 1.000 |

This is a property of the metric, not a model result. It shows that global R@1
is uninformative below cosine ≈ 0.5 and saturated above ≈ 0.8. The hard-negative
R@1 is the primary retrieval metric in the 0.5–0.8 band.

### 5.3 Linear probe

Two probes, one valid and one a negative control.

**Valid linear probe — ridge readout of the target.** Fit
\(W \in \mathbb{R}^{16 \times 16}\) on train predictions and targets:

\[
W = \arg\min_{W} \| \hat E_{\text{train}} W - V_{\text{train}}\|_F^2
+ \lambda \|W\|_F^2,
\]

with \(\lambda\) selected on dev. Evaluate \(R^2\) and
\(\cos(\hat E_{\text{test}} W, V_{\text{test}})\). If the model learned the
right structure up to a rotation, \(R^2\) is high even when the unrotated cosine
is low; the gap between \(R^2\) and \(\cos^2\) is a measure of misalignment. This
is a meaningful diagnostic and is independent of the loss (the loss is on the
unrotated output).

Optional nonlinear probe: a 2-layer MLP (64 hidden) from \(\hat e\) to \(v\),
early-stopped on dev. Report \(\Delta R^2 = R^2_{\text{MLP}} - R^2_{\text{ridge}}\).
A large positive \(\Delta\) means the information is present but not linearly
readable.

**Negative control — component identity probe.** A logistic regression from
\(\hat e(w)\) to prefix / root / suffix identity, trained on train, tested on
test. Report accuracy and macro-F1 alongside the ground-truth ceiling from
§2.3. The exact target is at chance for all three components, so this probe
cannot distinguish a good model from a perfect one. It is included because the
experiment brief asks for a linear probe, but it is **pre-registered as
uninformative**: it must not appear in a kill criterion or a headline claim.
KC11 formalises this.

### 5.4 Compositional-generalisation gap

Let \(\overline{\cos}_{\text{fit}}\) be the mean cosine on the 8,064 training
words and \(\overline{\cos}_{\text{test}}\) the mean on the 2,304 held-out
combinations. Define

\[
\Delta = \overline{\cos}_{\text{fit}} - \overline{\cos}_{\text{test}}.
\]

A positive \(\Delta\) is the expected direction: training words are memorised,
held-out combinations are not. \(\Delta\) is **memorisation-inclusive** by
design; that is the point. Also report:

- \(\overline{\cos}_{\text{fit}}\) and \(\overline{\cos}_{\text{test}}\)
  separately. A small gap with a low test score is not a win.
- \(\Delta_{\text{tree}} - \Delta_{\text{flat}}\), paired by word.
- The oracle gap \(\Delta_{\text{O2}}\), which calibrates how much gap is due to
  the architecture versus the data.
- The gap within each prefix and suffix group, and within frequency bins of the
  fit words (rare vs frequent in `train_text`).

A model that memorises training words perfectly and scores at chance on test has
\(\Delta \approx 1\) if chance is zero, or \(\Delta \approx 1 - 0.23 = 0.77\) if
its test score sits at the root-centroid anchor. A model that generalises
perfectly has \(\Delta \approx 0\). The tree hypothesis predicts a smaller
\(\Delta\) than the flat mean, *and* a higher absolute test score.

### 5.5 Statistics

The test words are not independent: each root appears in ≈19 test words, each
prefix in ≈288, each suffix in ≈192. Word-level standard errors are therefore
too small. Use:

- **Cluster bootstrap over roots** (120 clusters), 10,000 resamples, percentile
  95% CI. For a paired model contrast, resample roots with replacement and keep
  all test words belonging to each sampled root; compute the paired difference
  on the resulting multiset. This preserves within-root correlation.
- Report the word-level bootstrap as an optimistic secondary, clearly labelled.
- Report per-seed results and the mean ± sd across seeds. The seed-level sd
  captures initialization and batch-order variance; the cluster bootstrap
  captures test-set/data variance. Do not conflate them.
- Primary estimate: mean over data seeds of the per-seed mean cosine, with a
  95% CI from the seed-level t distribution (df = seeds − 1). With 3 data seeds,
  this CI is wide; treat it as a sanity bound, not a precise interval.
- Effect size: require ≥ 0.05 absolute cosine for a claim. Differences below
  0.03 are not interpretable at this scale; the measured random sd is 0.25 and
  the root-cluster SE is ≈ 0.023.
- Multiplicity: one pre-registered primary endpoint (above); everything else is
  secondary or exploratory. Do not select the best cell on test and report it as
  confirmatory.

### 5.6 What counts as a win

C1 (structure) is supported only if, in the primary cell:

1. the best tree variant beats the **best flat variant** (not merely the flat
   mean) by ≥ 0.05 mean cosine on held-out combinations, with a paired 95% CI
   excluding 0;
2. the tree's gap \(\Delta\) is at least 0.05 smaller than the best flat
   variant's gap;
3. the sign replicates on at least 2 of the 3 data seeds;
4. the tree's absolute test cosine is at least 0.10 above the root-centroid
   anchor (0.23), i.e. ≥ 0.33. Otherwise it is not meaningfully compositional.

C2 (ramification) is **not evaluated in the current implementation.** The
hierarchical tokenizer's tokenizations are identical to flat BPE on every train
and test word, so there is no treatment to compare. The C2 gate is KC5: first
implement P5 (a tokenizer whose levels alter the merge sequence) and verify that
the tokenizations differ on a held-out set. Once P5 exists, C2 will be supported
only if the hierarchical tokenizer beats flat BPE at matched vocabulary and
model by ≥ 0.02 cosine with a CI excluding 0, and shows at least a +0.01
boundary-F1 advantage. Until then, a "C2 null" is an implementation fact, not a
scientific result.

If C1 passes and a future C2 passes, the result is "tree composition helps and
ramification helps." If C1 passes and C2 is null (after P5), the result is "tree
composition helps; ramification does not (on this corpus)." If C2 passes but C1
fails, the result is "the tokenizer's segmentation helps a flat model; recursive
composition adds nothing." All three are more informative than a bundled claim.

---

## 6. Ablations

Marked **P** (primary), **S** (secondary), **E** (exploratory). Only P
ablations enter the kill criteria.

### 6.1 Tree versus flat (P)

The headline contrast. Compare, at the same tokenizer and vocabulary:

- `tree_pure` (role/depth-specific, no nonterminal lookup)
- `tree_hybrid(k)` for \(k \in \{0,1,2,3,\infty\}\)
- `flat_mean`
- `flat_posmean`
- `flat_gru`
- `flat_concat_mlp` (if runtime permits)

The primary comparator is the best flat variant selected on dev. Report all test
scores, not just the winner. Prediction: `tree_hybrid(2)` ≥ `tree_pure` on seen
words (it can memorise affixes); both tree variants ≥ `flat_mean` on unseen
combinations when the parse is right-associated; `flat_gru` is the strongest flat
baseline and may win at high vocabulary.

### 6.2 Tokenizer construction (C2; blocked)

`n_levels ∈ {1, 2, 3}` at fixed total vocabulary and default level weights
(`[1.0]`, `[0.6,0.4]`, `[0.5,0.3,0.2]`). Add one control with equal weights
`[1/3,1/3,1/3]` at 3 levels, because `split_budget` allocates merges by weight.

**The current implementation makes this a null manipulation.** The levels are
trained sequentially on the output of the previous level, so each level continues
the same deterministic greedy merge sequence. The concatenation of the level
merge lists equals the flat merge list, and the tokenizations are identical for
all train and test words at every vocabulary tested (measured: 0 differing words
at 256, 512, 1024, 2048, 4096). `n_levels` and `level_weights` only partition the
same merge sequence; they do not change the tokenizer. Running this ablation now
is a **manipulation check**, not a C2 test: it must produce identical
tokenizations and statistically indistinguishable embedding results (same
distribution, not bit-identical model weights). If it does not, the tokenizer
implementation changed and the equivalence must be re-verified.

To make C2 testable, a new tokenizer constructor is required (P5). It must make
the level structure alter the merge sequence without using morpheme labels. Three
legitimate options, in increasing order of implementation cost:

1. **Frozen lower levels:** train level 0 on characters, then freeze its symbols
   and train level 1 on a *different* corpus or weighting, so the higher-level
   merges are not simply the continuation of the same sequence.
2. **Different objective per level:** level 0 optimises unigram compression,
   level 1 optimises a different criterion (e.g., PMI or skip-gram), so the merge
   order changes at the boundary.
3. **Orthography-aware but label-free:** use surface cues (e.g., vowel/consonant
   transitions, affix-like character n-grams) to initialise a level, without
   using `Word.boundaries`. This is a heuristic, not oracle supervision, but it
   must be pre-registered and ablated against the label-free baseline.

Constraining merges with true morpheme boundaries is **oracle leakage** and is
not an acceptable fix. Until P5 is implemented and verified to produce different
tokenizations on a held-out set, C2 is not evaluated. Report C1 only.

### 6.3 Vocabulary size (P for C1)

`vocab_size ∈ {256, 512, 1024, 2048, 4096}`. Report the actual vocabulary size
(the tokenizer stops early if merges run out; it does not here), tokens per word,
forest size, single-token rate, boundary F1, `morph3`, and top-level alignment.

Measured at seed 42, current code. These values are identical for `flat`,
`hier2` and `hier3`; the table uses `flat` because that is the shared tokenizer
used by the primary C1 comparison.

| Vocab | Tokens/word | Single-token | Forest roots (mode) | Max depth | Boundary F1 | `morph3` | Top-level aligned | Right-grouped | Left-grouped | Flat forest |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 512 | 2.97 | 0.00 | 3 | 5 | 0.8665 | 0.78 | 0.73 | 0.01 | 0.06 | 0.73 |
| 1024 | 2.71 | 0.00 | 3 | 5 | 0.8199 | 0.80 | 0.56 | 0.04 | 0.25 | 0.56 |
| 2048 | 2.25 | 0.00 | 2 | 5 | 0.7072 | 0.80 | 0.19 | 0.08 | 0.65 | 0.19 |
| 4096 | 2.04 | 0.00 | 2 | 5 | 0.6552 | 0.80 | 0.03 | 0.12 | 0.81 | 0.03 |

The remainder in each row is the “other” category (neither grouping is a
constituent, and the two morpheme boundaries are not both top-level): 0.19,
0.15, 0.08, 0.04 respectively. The three grouping columns plus “other” sum to 1.0
up to rounding.

“Right-grouped” means \((p,(r,s))\) is a constituent; “left-grouped” means
\(((p,r),s)\) is; “flat” means both morpheme boundaries are top-level and
neither grouping is present. Of the 0.73 flat forests at vocabulary 512, 0.72
have exactly three top-level tokens (the root is a single top-level token); the
rest split the root at the top level. At 512, a position-conditioned right fold
over three roots gives the correct nesting. At 2048, 65% are left-grouped and
only 19% are flat: the parse has already applied the prefix before the suffix,
which is the wrong order. This is why the vocabulary ablation is predicted to be
**non-monotonic** for the tree model: it should peak at 512 and decline at
2048+, even though tokens per word falls. The flat mean should degrade
monotonically with vocabulary (longer tokens are more specific, and a bag of
longer tokens is a worse composition); the flat GRU may be flat because it can
learn the order from the token sequence.

Single-token rate is 0.00 at every vocabulary tested: no test word is a single
token, so the tree model is never a pure lookup on test. This removes the
triviality objection at these vocabularies. The **train** single-token rate is
different, because the tokenizer is fitted on `train_text`: at holdout 0.3 it is
1.1% at vocabulary 512, 9.5% at 2048, and 31.7% at 4096 (measured). At 4096, a
third of the fit words are one token; the pure tree must reconstruct them from
characters while the flat model looks them up. This inflates the tree's fit
difficulty and complicates the gap at high vocabulary. The primary cell (512) is
nearly free of this confound. If a future corpus produces >50% single-token test
words, invalidate that cell (KC6).

### 6.4 Seen versus unseen components (P)

Three protocols:

- **HC — held-out combinations (primary):** all components seen, the triple
  unseen (§2.2).
- **HR — held-out roots:** a set of roots is removed from training; test words
  use those roots with seen affixes.
- **HA — held-out affixes:** one prefix and one suffix are removed from
  training; test words use them with seen roots and other affixes.

HR and HA are **null controls**, not performance tests. With random orthogonal
component matrices, an unseen component's vector is independent of its spelling;
no model, including O2, can predict it. The expected cosine is 0 for every
model. The purpose is to detect leakage: if any model scores above 0.15 on HR or
HA, something is wrong (KC7). O2 is the cleanest check: it has no embedding for
the unseen component and must fall back to `<unk>`; if it scores above chance,
the generator or the pipeline is leaking.

To generate HR/HA, add a new function in `ramified/splits.py` (prerequisite P2)
that builds the grid with a designated component excluded from training, and
bypasses the component-attestation filter (which would otherwise move the
held-out words back to train). Do not hack the existing `holdout` parameter; it
holds out words, not components.

A positive unseen-component result would require a different generator in which
component vectors are a compositional function of their surface form (e.g.,
\(r_r = f(\text{characters of } r)\)). That is Phase 3, not this experiment.

### 6.5 Shared versus depth/role-specific composition (P)

`shared_M`, `depth_M`, `side×depth_M`, and `width ∈ {16, 32, 64}`. This is the
direct test of §1.5:

- At width 16, `shared_M` should underfit training data relative to
  `side×depth_M` (P1).
- At width 32 or 64, the gap should close (P2).
- If `shared_M` at width 16 matches `side×depth_M`, the representability
  argument's assumptions are violated by the learned embeddings; investigate
  before drawing conclusions about structure.

This ablation also guards against a false negative: if the primary tree used a
shared operator, it would underfit for a reason unrelated to tokenization or
generalisation, and the experiment would wrongly reject C1. The primary model is
therefore depth/role-specific, and `shared_M` is diagnostic.

### 6.6 Forest aggregation and fold direction (P)

`left_fold`, `right_fold`, `side_gate`, `learned_sequence_combiner` (a small GRU
over the forest roots, used only at the forest level, with the internal tree
composition unchanged). The primary `right_fold` uses position-conditioned
operator sides (§3.1); `side_gate` replaces the discrete side choice with a
learned convex combination of the two orders (differentiable, more parameters). A
fixed-side right fold and a fixed-side left fold are separate diagnostics.
Prediction: the position-conditioned right fold wins when the forest is flat at
the morpheme boundaries (vocabulary 512); a fixed-side left fold matches the
parse's wrong order at 2048+, so it may fit the training data but should not
improve held-out target cosine. The learned combiner is a hedge: it can learn the
right order from data, but it also has more capacity, so report parameter counts.

The operator side is conditioned on position from the right, not on forest size.
A 2-root forest and a 3-root forest share the outermost position, even though
they may need different sides. A variant conditioning the side on
\((\text{forest size}, \text{position})\) is an exploratory ablation; it has more
parameters and should be reported separately.

### 6.7 Hybrid cutoff (S)

\(k \in \{0,1,2,3,\infty\}\), where symbols with expansion length \(\le k\) are
looked up and longer symbols are composed. \(k=0\) is pure recursion; \(k=\infty\)
is a full lookup (a lookup-based model with tree aggregation, not the flat mean).
Prediction: a sweet
spot at \(k=2\) or 3, because prefixes and suffixes are 2–3 characters and roots
are 3–5; the model can memorise affixes and compose roots, which is close to the
oracle segmentation without being given boundaries. This is a legitimate
architectural choice, not leakage: symbol length is visible from the tokenizer.
Report the parameter count and the fraction of symbols looked up.

### 6.8 Training distribution (S)

Uniform over word types (primary) versus Zipfian sampling from `train_text`. The
Zipfian run matches the tokenizer's statistics but confounds the gap with
frequency. Prediction: uniform training improves rare-combination performance;
Zipfian training inflates `cos_fit` on frequent words and widens the gap.

### 6.9 Capacity control (S)

Widen the flat models to match the tree's parameter count (embedding dimension
64 plus a linear map to 16; or a wider GRU). If the tree's advantage disappears,
it was capacity. This is not optional in the write-up; it is the standard
objection to any "structure helps" claim.

### 6.10 Loss and width (E)

Cosine versus InfoNCE; internal width 16/32/64; normalised versus unnormalised
nodes. These are exploratory. InfoNCE is expected to improve retrieval more than
cosine; a large improvement would mean the cosine-trained model is
under-optimised for the retrieval metric, not that the structure is better.

### 6.11 Data seeds and root count (S)

Data seeds 42, 43, 44 (the generator's component geometry changes with the
seed). Optional: `n_roots=240` to test data scaling; the grid doubles to 23,040
words, so this is a separate run, not part of the core matrix.

---

## 7. Expected failure modes and explicit kill criteria

### 7.1 Expected failure modes

**F1 — Parse order and grouping mismatch (highest risk).** The BPE tree is built
by frequency, not morphology. At vocabulary 512, 72% of test words have exactly
three top-level tokens at the morpheme boundaries (73% have both boundaries
top-level), which is the best case for a right fold. At 2048, 65% are
left-grouped \(((p,r),s)\), the wrong order for \(P_pS_sr\), and only 19% are
flat. A bilinear tree cannot undo a wrong grouping. Prediction: the tree model
peaks at 512 and declines at 2048+, even though tokens per word falls. Measure:
correlate the tree's test cosine with `topOK`/`morph3` across the vocabulary
cells.

**F2 — Shared-operator representability failure.** A shared \(M\) at width 16
cannot interpolate 20 independent component matrices (§1.5). If the primary tree
were shared, it would underfit training data and produce a false negative for
C1. Prediction: `shared_M` train cosine < `side×depth_M` train cosine at width
16, and the gap closes at width 32/64. This is a designed diagnostic, not a
surprise.

**F3 — Sub-morphemic root reconstruction.** The root is a constituent in 78–80%
of test words, but it is never a single leaf: 100% of test words have a root span
containing more than one leaf. The pure tree must build an arbitrary random root
vector out of characters/sub-tokens. That is a memorisation burden unrelated to
composition. The burden is worse on the fit set at high vocabulary: 31.7% of
train words are single tokens at vocabulary 4096 (9.5% at 2048, 1.1% at 512), and
the pure tree must reconstruct them from characters while the flat model looks
them up. Prediction: pure recursion underperforms the hybrid on *seen* words; the
hybrid is the practical model. If the pure tree fails while the hybrid succeeds,
the bottleneck is sub-morphemic reconstruction, not composition.

**F4 — No easy partial credit.** The target is a composition of three random
orthogonal actions. Knowing only the root gives cosine ≈ 0.23 (root centroid);
knowing only the prefix or suffix gives ≈ 0. Partial models do not get partial
credit. All models may sit in the 0.2–0.4 band, where the metric is noisy.
This is why the experiment uses paired tests, cluster bootstraps, and the
hard-negative retrieval metric. A difference of 0.02 is not a result.

**F5 — The flat GRU wins.** The sequences are short (2–3 roots at the primary
vocabulary), so a GRU is almost a position-tagged bag, but it can learn the
operator order from data. With 8,064 training words and 20 components, it may
match or beat the tree. If it does, the structural inductive bias is not
necessary for this task; C1 fails even if the tree beats the flat mean. This is
the most likely "negative" outcome and must be reported as such.

**F6 — Target geometry is too hard / too easy.** Random orthogonal matrices are
a hard, non-semantic composition: there is no similarity structure between
components. O2 should still fit (it is exactly representable), but the flat
models may be near chance. Conversely, if O2 reaches 0.99 on test, the task is
easy and differences between models may be small. Both outcomes are informative
but limit external validity.

**F7 — The tokenizer axis is a no-op.** In the current implementation,
`n_levels` and `level_weights` only partition the same flat merge sequence; the
tokenizations are identical at every vocabulary tested (0 differing words at 256,
512, 1024, 2048, 4096). C2 is therefore not testable, and running the tokenizer
axis now would produce identical embeddings by construction. Do not interpret a
null C2 as evidence for C1; it is a property of the implementation, not a
scientific result. Implement P5 before testing C2.

**F8 — Probe invalidity.** Component-identity probes are at chance even on the
exact target (§2.3). Any result that relies on them is void. Use the ridge target
probe and the retrieval metrics.

**F9 — Cluster correlation / false precision.** 2,304 test words share 120 roots
and 8/12 affixes. Naive word-level SEs are ≈ 0.005; root-cluster SEs are
≈ 0.023. Do not report the former as if it were the latter. Two-way clustering
(root and affix) is even wider; the root-cluster bootstrap is the conservative
primary.

**F10 — EOW and span bookkeeping.** `Tokenizer.word_boundaries` treats `EOW` as
5 characters. It is safe for the current last-node logic but fragile. Phase 2
uses explicit node spans with `EOW` length 0. A forest assertion checks that the
leaf expansions concatenate to `w + EOW` and that every internal node has exactly
two children.

**F11 — Non-injectivity is dormant.** No duplicate expansions were found on this
corpus. Do not use Phase 2 to claim anything about the steganographic channel;
the phenomenon is not exercised.

**F12 — Optimisation instability of the normalised bilinear tree.** If the
operator output collapses to zero, normalisation amplifies noise. Use a small
linear residual term, gradient clipping, and a learning-rate warmup. Monitor the
norm of the pre-normalised output; if it falls below \(10^{-3}\), the run is
broken. This is a practical failure mode, not a scientific one.

### 7.2 Kill criteria and pre-registered invalidations

All criteria are pre-registered. “Stop” means do not interpret the downstream
comparison; fix the pipeline or report the negative.

| ID | Condition | Action |
|---|---|---|
| **KC0** | O1 exact-oracle cosine < 0.999999 on test; target norms outside \(1 \pm 10^{-6}\); any split overlap; any dev/test component unattested; tokenizer saw any dev/test word | Stop; pipeline bug |
| **KC1** | O2 oracle-segmentation test cosine < 0.80 or train cosine < 0.95 | Stop; the composition parameterisation cannot learn the target even with true boundaries. Fix \(g\)/optimisation before comparing models. |
| **KC2** | Best parse-tree model's train cosine < 0.80 while O2 train > 0.95 | Stop; the parse destroys learnability, or the tree optimiser fails. Inspect the parse and the train fit before looking at test. |
| **KC3** | In the primary cell (shared tokenizer, vocabulary 512), best tree test cosine < best flat test cosine + 0.05, or the paired 95% CI includes 0 | C1 falsified for this corpus. Report the negative; do not reframe as a win on a weaker baseline. |
| **KC4** | Tree gap \(\Delta\) is not at least 0.05 smaller than the best flat gap | The mechanism (structure → generalisation) is absent. A higher absolute score with the same gap is a capacity/fit effect, not a generalisation effect. |
| **KC5** | **C2 gate:** the hierarchical tokenizer's tokenizations are identical to flat BPE on all train and test words in the current implementation | C2 is not tested; report C1 only. Implement P5 (a tokenizer whose levels change the merge sequence) before testing C2. Once P5 exists, require a measurable alignment advantage (boundary F1 ≥ +0.01) and a ≥ 0.02 cosine gain over flat BPE. |
| **KC6** | > 50% of test words are single tokens | Invalidate that cell; the tree model is a lookup. |
| **KC7** | Any model scores > 0.15 mean cosine on HR or HA; or the label-shuffle canary scores > 0.05 | Stop; leakage or generator bug. Audit L1–L12. |
| **KC8** | Paired 95% CI half-width for KC3 > 0.05 | Underpowered; add data seeds/roots or reduce variance. Do not interpret. |
| **KC9** | `flat_gru` or `flat_concat_mlp` beats the best tree by ≥ 0.05 on held-out combinations | C1 fails even if the tree beats the flat mean. Reframe: the benefit is sequence capacity, not tree structure. |
| **KC10** | Best model hard-negative R@1 > 0.95 and global R@1 > 0.99 | Test saturated; increase roots/dimension. Do not claim a win on a ceiling. |
| **KC11** | Ground-truth component probe is within 1 SE of chance (expected) | Component probes are uninformative; no claim may use them. Use the ridge target probe. |

**Decision rule.** C1 survives only if KC3 and KC4 pass and KC9 does not fire.
C2 is **not evaluated** until P5 is implemented; KC5 is a gate, not a scientific
comparison. A C1 result is a statement about recursive composition on a shared
tokenizer, not about ramification. Report the tokenizer axis as a manipulation
check (identical tokenizations) until P5 exists.

---

## 8. Concrete implementation plan

### 8.1 Prerequisites

| ID | Prerequisite | Why | Risk |
|---|---|---|---|
| **P0** | Add `all_words: list[Word]` to `SyntheticCorpus` and return the shuffled full grid (2-line additive change in `ramified/synthetic.py`) | Train words are currently returned as strings; the embedding task needs their target vectors. Without this, the experiment cannot train on the train split. | Blocker. Do not reconstruct the grid by duplicating `_make_root`/`_orthogonal`. |
| **P1** | New `ramified/forest.py` with span/depth/flatten utilities | `bpe.py` has no span API; `word_boundaries` has the EOW-length quirk. | Low; pure functions, no changes to `bpe.py`. |
| **P2** | New `ramified/splits.py` with the `holdout=0.30` dev/test split and the HR/HA component controls | The current generator has no dev split and no component holdout. | Medium; HR/HA needs a new grid builder that bypasses the component-attestation filter. |
| **P3** | New `experiments/embedding/` package | Keeps the embedding experiment separate from the Phase 1 tokenizer matrix. | Low. |
| **P4** | New `tests/test_embedding_contracts.py` | The contract tests in §8.4 catch the most likely silent bugs. | Low. |
| **P5** | New tokenizer constructor that makes levels alter the merge sequence | The current `n_levels`/`level_weights` only partition the same flat merge sequence; tokenizations are identical (measured). C2 is untestable without this. | **Blocker for C2.** Must not use `Word.boundaries` (oracle leakage). Verify that flat and hierarchical tokenizations differ on a held-out set before running C2. |

P0 and P5 are the two source-level blockers. P0 is additive and does not alter
tokenizer training. P5 is a new constructor; it must be validated against a
manipulation check (different tokenizations, not different bookkeeping) before
C2. The main C1 protocol needs P0, P1, P2 and P3; it does not need P5, because it
uses the shared tokenizer and the derivation forest, which flat BPE already
provides.

### 8.2 Files and functions

```
ramified/
  bpe.py                         unchanged (flat tokenizer)
  synthetic.py                   + all_words field (P0)
  forest.py                      new (P1)
  splits.py                      new (P2)
  bpe_levels.py                  new (P5, C2 only; must alter merge sequence)

experiments/embedding/
  __init__.py                    new
  config.py                      ExperimentConfig, grid definitions, run IDs
  data.py                        Example, NodeIndex, build_examples, vocab, collate
  models.py                      TreeComposer, FlatMean, FlatPositionMean,
                                 FlatGRU, FlatConcatMLP, OracleMorpheme, ExactOracle
  train.py                       train_one, losses, seeding, early stopping
  evaluate.py                    cosine_stats, retrieval, hard_negative,
                                 ridge_probe, mlp_probe, component_probe,
                                 cluster_bootstrap, compositional_gap
  io.py                          tokenizer cache, prediction cache, CSV/JSON, resume

experiments/
  run_embeddings.py              CLI and run matrix

tests/
  test_embedding_contracts.py    new (P4)

results/embed/                   outputs (git-ignored)
```

**`ramified/forest.py`**

- `spans(nodes, tok) -> list[Span]` — every node's `(lo, hi, symbol, depth, parent, side)`, with `EOW` length 0.
- `leaf_spans(nodes, tok) -> list[tuple[int,int]]` — terminal spans only.
- `forest_depth(nodes) -> int` — maximum node depth (measured max 5).
- `symbol_levels(tok) -> dict[int,int]` — symbol id to merge level, from `tok.levels`; terminals = −1.
- `right_fold_order(m) -> list[int]` — an index order for the forest aggregator; `left_fold_order(m)` is the mirror.
- `assert_forest(nodes, word, tok)` — every internal node has exactly 2 children; leaf expansions concatenate to `word + EOW`; spans are contiguous and non-overlapping; `EOW` contributes 0 characters.

**`ramified/splits.py`**

- `make_splits(data_seed, holdout=0.30, dev_frac=1/3, split_seed=None) -> Split` — uses `generate_synthetic`, splits the held-out pool stratified by `(prefix, suffix)`, returns `train/dev/test` as `list[Word]`; asserts L1–L3.
- `component_control(data_seed, component="root", n_held=20) -> Split` — builds a grid with the designated component absent from train; bypasses the attestation filter; asserts the held-out component is absent from train and present in dev/test.
- `assert_no_leak(split, train_text, tok)` — L1–L3 and the `train_text` check.

**`ramified/bpe_levels.py` (P5; C2 only)**

- `train_level_bpe(texts, vocab_size, level_spec) -> Tokenizer` — a level-aware
  constructor whose levels alter the merge sequence (see the three legitimate
  options in §6.2). It must not read `Word.boundaries`.
- `assert_levels_change_tokenization(flat_tok, hier_tok, words)` — the C2
  manipulation check: at least 1% of held-out words must tokenize differently.
  If they do not, C2 is not run.

**`experiments/embedding/config.py`**

- `ExperimentConfig` — data seed, model seed, tokenizer, vocabulary, method,
  width, roles, side mode, hybrid cutoff, steps, batch size, protocol.
- `PRIMARY_CELL` — `(tokenizer="flat", vocab=512)`; `grid()` returns the run list.

**`experiments/embedding/data.py`**

- `Example` — `word`, `target`, `roots: list[int]`, `node_index: NodeIndex`, `meta` (prefix/root/suffix, frequency, split). Non-oracle examples have **no** `boundaries` field.
- `OracleExample` — extends `Example` with `boundaries`; constructed only by `build_oracle_examples`.
- `NodeIndex` — flattened arrays: `symbol`, `left`, `right`, `depth`, `level`, `side`, `root_ids`, `word_id`, `node_word`. This is the key to fast batching; do not recurse per word in the training loop.
- `build_examples(words, tok, cache_key, oracle=False)` — encodes each word once, builds the node table, caches it under `results/embed/cache/{cache_key}.npz`.
- `build_symbol_vocab(train_examples)` — symbols seen in training; `<unk>` index; report OOV.
- `collate(batch)` — padded root sequences (max 8), tree arrays by depth, targets.

**`experiments/embedding/models.py`**

- `TreeComposer(vocab, d=16, width=16, roles="side_depth", side_mode="fixed", share_g=False, hybrid_max_len=0)`.
- `FlatMean(vocab, d=16)`, `FlatPositionMean(vocab, d=16, max_len=8)`.
- `FlatGRU(vocab, d=16, hidden=64)`, `FlatConcatMLP(vocab, d=16, max_len=8, hidden=64)`.
- `OracleMorpheme(n_prefix, n_root, n_suffix, d=16, width=16)`.
- `ExactOracle(targets)` — returns the target; used only for KC0.
- `build_model(name, cfg, data)` — one factory; no model-specific training code.

**`experiments/embedding/train.py`**

- `train_one(cfg, split, tok, model_name, model_seed) -> RunResult`.
- `cosine_loss(pred, target)`, `info_nce(pred, target, tau=0.07)`.
- `set_seed(seed)` — Python, NumPy, Torch; `torch.use_deterministic_algorithms(True)` where possible.
- `evaluate_on(model, examples) -> np.ndarray` — returns normalised predictions; no metric code here.

**`experiments/embedding/evaluate.py`**

- `cosine_stats(pred, target, meta) -> dict` — mean, median, p10, p90, fraction > 0.5, per-prefix, per-suffix.
- `retrieval(pred, target, meta, hard="none") -> dict` — global and hard-negative R@1/5/10/50, MRR; returns the candidate counts and chance levels.
- `ridge_probe(train_pred, train_y, test_pred, test_y) -> dict` — \(R^2\), cosine, lambda.
- `mlp_probe(...) -> dict` — optional; same protocol, early stop on dev.
- `component_probe(...) -> dict` — accuracy, macro-F1, chance, and the ground-truth ceiling; labelled `uninformative` in the output.
- `cluster_bootstrap(metric_fn, meta, cluster="root", n=10000) -> dict` — percentile CI; paired contrast support.
- `compositional_gap(fit_scores, test_scores, meta) -> dict` — \(\Delta\), per-group gaps, paired difference.

**`experiments/embedding/io.py`**

- `save_tokenizer(tok, path)`, `load_tokenizer(path)` — cache the frozen tokenizer.
- `save_run(result, path)`, `load_run(path)`, `config_hash(cfg)` — CSV/JSON/NPZ outputs and resume.

**`experiments/run_embeddings.py`**

- CLI flags: `--protocol`, `--methods`, `--tokenizers`, `--vocab-sizes`, `--widths`, `--data-seeds`, `--model-seeds`, `--steps`, `--batch-size`, `--out`, `--resume`, `--smoke`, `--dry-run`.
- `tree_hybrid` defaults to \(k=2\); the hybrid-cutoff ablation varies \(k\).
- `--dry-run` prints the run count and estimated time; use it before every long run.
- One CSV row per run; one `.npz` per run with test predictions and targets; a `summary.json` with the pre-registered endpoint and CIs.
- Resume: skip a run if its row and `.npz` exist and the config hash matches.

### 8.3 Commands

Run from the project root. The existing package layout supports
`python -m experiments.run_embeddings` because `experiments/__init__.py` exists.

```powershell
# 0. contract tests
python -m pytest tests/test_embedding_contracts.py -q

# 1. smoke: one cell, three models, 200 steps
python -m experiments.run_embeddings --protocol heldout_combination `
  --methods tree_hybrid,flat_mean,oracle_seg --tokenizers flat --vocab-sizes 512 `
  --data-seeds 42 --model-seeds 0 --steps 200 --out results/embed_smoke

# 2. representability gate (oracle tree only; no tokenizer uncertainty)
python -m experiments.run_embeddings --protocol oracle_tree `
  --methods oracle_seg,tree_shared,tree_pure --widths 16,32,64 `
  --data-seeds 42 --model-seeds 0,1 --steps 800 --out results/embed_gate

# 3. C1 core matrix (shared tokenizer; this is the scientifically live axis)
python -m experiments.run_embeddings --protocol heldout_combination `
  --methods tree_pure,tree_hybrid,flat_mean,flat_posmean,flat_gru,oracle_seg `
  --tokenizers flat --vocab-sizes 512,1024,2048 `
  --data-seeds 42,43,44 --model-seeds 0,1 --steps 800 --out results/embed_core

# 3b. tokenizer manipulation check (C2 is blocked until P5)
python -m experiments.run_embeddings --protocol heldout_combination `
  --methods flat_mean --tokenizers flat,hier2,hier3 --vocab-sizes 512 `
  --data-seeds 42 --model-seeds 0 --steps 200 --out results/embed_tokenizer_check

# 4. controls
python -m experiments.run_embeddings --protocol heldout_root,heldout_affix,label_shuffle `
  --methods tree_hybrid,flat_gru,oracle_seg --tokenizers flat --vocab-sizes 512 `
  --data-seeds 42 --model-seeds 0 --steps 800 --out results/embed_controls

# 5. ablations
python -m experiments.run_embeddings --protocol heldout_combination `
  --ablations shared_g,left_fold,side_gate,hybrid_cutoffs,zipf,widths `
  --tokenizers flat --vocab-sizes 512,1024 --data-seeds 42 --model-seeds 0 `
  --steps 800 --out results/embed_ablations
```

The exact flag names are an interface proposal; the important part is the
matrix, not the spelling. A dry run must print the number of runs before
execution.

### 8.4 Contract tests

`tests/test_embedding_contracts.py`:

1. `test_forest_covers_word` — for 100 random train words, the leaf expansions
   concatenate to `word + EOW`; every internal node has 2 children; spans are
   contiguous; `EOW` contributes 0 characters.
2. `test_no_split_leak` — train/dev/test strings are disjoint; every dev/test
   component is in train; no dev/test word occurs in `train_text`.
3. `test_oracle_exact` — O1 cosine on test is 1.0 within \(10^{-6}\).
4. `test_flat_mean_single_root` — if a word is one root, the flat mean equals its
   normalised embedding.
5. `test_tree_matches_manual` — a hand-built 3-node tree composed by
   `TreeComposer` matches the manual application of \(g\).
6. `test_metrics_chance` — random unit predictions give mean cosine ≈ 0 and
   global R@1 ≈ \(1/N\); hard-negative R@1 ≈ \(1/N_h\).
7. `test_cluster_bootstrap` — on synthetic clustered data, the 95% CI covers the
   true mean at the nominal rate in a small simulation.
8. `test_shared_vs_depth_representability` — on the oracle tree, 200 steps of
   depth-specific training fit the train set better than shared-\(M\) at width
   16; the gap shrinks at width 32. This is a slow test; mark it `slow`.
9. `test_hierarchy_is_noop` — `n_levels=1` and `n_levels=3` produce identical
   tokenizations for all test words and identical concatenated merge lists. This
   documents the current implementation and catches accidental changes.
10. `test_p5_changes_tokenization` — once P5 exists, assert that the new
    hierarchical tokenizer differs from flat BPE on at least 1% of held-out
    words. This is the C2 manipulation check; without it, C2 is not run.

### 8.5 Runtime estimates

Environment: CPU only (`torch 2.9.0+cpu`, 16 logical cores, no CUDA), batch 256,
800 steps, forest size ≤ 6, depth ≤ 5, \(d=16\).

Measured anchors:

- Tokenizer training: 2.2–2.7 s per (method, vocabulary, data seed) on the
  120k-token synthetic corpus.
- Naive per-node tree loop: **0.32 s/step** for batch 256 (≈ 258 s for 800
  steps). This is unacceptable and must not be the implementation. The measured
  0.32 s/step is dominated by Python per-node dispatch, not tensor math.
- Vectorised per-depth tree encoder target: ~1–5 ms/step, i.e. ~1–5 s for 800
  steps. The node table is static per tokenizer, so the index arrays are built
  once and reused.
- GRU: 25 s for 800 steps at batch 256 (measured).
- Flat mean / oracle: < 5 s per run.
- Evaluation: retrieval is a 2,304×2,304 cosine matrix (< 0.1 s); the ridge and
  component probes are 10–30 s per run; cache predictions and run probes once.

Estimated matrix sizes and wall time (single process):

| Tier | Runs | Estimate |
|---|---:|---:|
| Smoke | ~6 | 2–5 min |
| Representability gate | ~12 | 5–10 min |
| C1 core (3 data seeds × 2 model seeds × 3 vocabs × 6 methods, shared tokenizer) | ≈96–114 | 30–60 min |
| Tokenizer manipulation check (C2 gate; 3 tokenizers × 1 vocab × 1 seed) | ~3–9 | 2–5 min |
| Controls (HR/HA/shuffle) | ~9–27 | 5–20 min |
| Ablations | ~100 | 30–60 min |
| **Full C1** | **~220–250** | **1.5–2.5 h single process; 45 min–1.5 h with 4 workers** |
| **Full C1 + C2 (once P5 exists)** | **~450–520** | **3–5 h single process; 1.5–2.5 h with 4 workers** |

The oracle O2 depends only on the data seed and model seed, not on the tokenizer
or vocabulary; run it once per (data seed, model seed) and reuse the result. The
C1 core count is `3 data seeds × 2 model seeds × [3 vocabs × 5 tokenizer-dependent
methods + 1 oracle]` = 6 × 16 = 96, plus a few repeats. The larger 450–520 figure
is the full design including the tokenizer axis once P5 makes it real.

Assumptions: vectorised tree encoder; 4–8 worker processes with
`torch.set_num_threads(1)` each to avoid thread contention; predictions cached;
tokenizers cached (9 for the C1 core, 27 for the full design; ≈25–70 s total).
If the naive tree loop is used, the C1 core alone is ≈ 7 h; that is the main
engineering risk. Memory is < 2 GB.

If the first cell's dev learning curve is still rising at 800 steps, increase
the budget to 1,600 for all models and roughly double the estimates. Do not
increase it for the tree alone; that would confound capacity with compute.

### 8.6 Outputs and reproducibility

Per run, write:

- `config.json` — all hyperparameters, git-ish code version, library versions,
  data seed, model seed, tokenizer hash.
- `metrics.csv` — one row: run id, protocol, tokenizer, vocabulary, method,
  width, roles, params, steps, train/dev/test cosine, gap, global and
  hard-negative retrieval, ridge \(R^2\), component probes, OOV rate,
  tokens/word, forest size, wall time.
- `preds.npz` — test predictions, test targets, and metadata (prefix/root/suffix
  ids). Keep this so probes and bootstraps can be recomputed without retraining.
- `summary.json` — the pre-registered endpoint, paired contrasts, CIs, and the
  KC0–KC11 checks.

Set `PYTHONHASHSEED=0`, `OMP_NUM_THREADS=1` per worker, and record
`numpy`, `torch`, `scikit-learn` versions. Fix the data seeds in advance; report
all of them. Do not drop a seed because it is inconvenient.

---

## 9. Threats to validity — what this experiment cannot show

1. **Synthetic geometry.** Random orthogonal matrices have no semantic
   similarity structure. A model that succeeds here has recovered an exact
   algebraic composition; a model that fails may still work on natural language,
   where components share substructure. The experiment is a falsification test
   of the mechanism, not a benchmark of language quality.
2. **The target is exactly compositional by construction.** A positive result is
   weak evidence for natural language (the target is too clean); a negative
   result is strong evidence that the architecture cannot recover even an exact
   composition. State this asymmetry explicitly.
3. **Small component inventory.** 8 prefixes, 12 suffixes, 120 roots. The
   representability argument depends on the counts (20 component matrices versus
   \(d=16\)); a larger inventory makes the shared-operator obstruction worse, not
   better.
4. **Short words, shallow trees.** 2–6 roots, depth ≤ 5. The GRU/Transformer
   baselines have little sequence to model; the tree model has few levels. Results
   may not transfer to longer words or deeper morphology.
5. **No allomorphy, polysemy, context, or frequency effects in the target.** The
   generator is a clean prefix-root-suffix system. Real morphology is messier.
6. **The tokenizer is trained on a Zipfian sample, the embedding model on uniform
   types.** This mismatch is deliberate (it isolates type-level composition), but
   it is not how a deployed model would be trained.
7. **The oracle O2 is an upper bound only for models that respect morphemes.**
   It is not an upper bound for a model that could learn a better segmentation
   from data.
8. **Unseen components are null controls, not tests.** With random component
   vectors, no model can generalise to an unseen root or affix. Do not report
   HR/HA as evidence for or against composition; report them as leakage checks.
9. **Non-injectivity is dormant.** No duplicate expansions were found; this
   corpus cannot test the channel/steganography claims of `ramified-v0.md` §5.
10. **Compression is not varied.** For C1 the tokenizer is shared across models,
    so compression is matched by construction. For C2 (once P5 exists), the v0.1
    hypothesis requires equal vocabulary *and* compression; this design matches
    vocabulary, not compression. Add bits-per-character as a covariate and report
    it, but do not claim an equal-compression C2 result unless the tokenizers are
    matched on it (Phase 1's `bits_per_char` can be reused).
11. **One generator, one language.** The synthetic alphabet and affix inventory
    are fixed. A single negative result does not falsify the mathematical claim;
    it falsifies the claim for this presentation. Report it that way.

---

## Appendix A. Measured calibration (seed 42, current code)

These numbers are measurements, not predictions. They are included so the design
is grounded in the actual corpus and so the expected effect sizes are calibrated.

### A.1 Corpus and splits

- Grid: 8 prefixes × 120 roots × 12 suffixes = 11,520 words.
- `holdout=0.2`: 9,216 train, 2,304 test; no overlap, no string collisions.
- `holdout=0.3`: 8,064 train, 3,456 held-out; no overlap, no string collisions.
- `train_text`: 120,000 sampled word tokens, 1,148,566 characters; 7,478 distinct
  word types appear at holdout 0.2 (excluding the `.` separator).
- `test_text`: 2,000 tokens, 19,370 characters; **not used** in Phase 2.
- Target norms: 1.0 within \(10^{-15}\).

### A.2 Tokenizer statistics (test split, current code; flat = hier2 = hier3)

| Vocab | Tokens/word | Single-token | Forest roots (mode) | Max depth | Boundary F1 | `morph3` | Top-level aligned | Right-grouped | Left-grouped | Flat forest | Test OOV |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 512 | 2.97 | 0.00 | 3 | 5 | 0.8665 | 0.78 | 0.73 | 0.01 | 0.06 | 0.73 | 0.000 |
| 1024 | 2.71 | 0.00 | 3 | 5 | 0.8199 | 0.80 | 0.56 | 0.04 | 0.25 | 0.56 | 0.000 |
| 2048 | 2.25 | 0.00 | 2 | 5 | 0.7072 | 0.80 | 0.19 | 0.08 | 0.65 | 0.19 | 0.001 |
| 4096 | 2.04 | 0.00 | 2 | 5 | 0.6552 | 0.80 | 0.03 | 0.12 | 0.81 | 0.03 | 0.007 |

The remainder in each row is the “other” category (neither grouping is a
constituent, and the two morpheme boundaries are not both top-level): 0.19,
0.15, 0.08, 0.04 respectively. The three grouping columns plus “other” sum to 1.0
up to rounding.

**Important:** these values are identical for `flat`, `hier2` and `hier3` in the
current implementation. The concatenated level merge lists equal the flat merge
list, and the tokenizations of all train and test words are identical at every
vocabulary (0 differing words measured). There is no hierarchical advantage to
report; the earlier claim of one (0.893 vs 0.874 at 512, `morph3` 0.84 vs 0.77)
is withdrawn as unreproducible. Reproduce by training `n_levels=1` and
`n_levels=3` on the same `train_text` and comparing the concatenated
`(left,right,new)` merge lists and the `encode(w)` sequences. Leaf purity is 1.00
at every tested vocabulary: no token crosses a morpheme boundary on the test
split. The failure mode is grouping and order, not boundary crossing.

### A.3 Target geometry and metric calibration (holdout-0.2 test set)

- Pairwise target cosine: mean 0.0001, sd 0.2500, min −0.908, max 0.898.
- Random unit versus target: 0.007.
- Root centroid versus target: 0.231.
- Global R@1 chance: 0.00043.
- Cosine-to-global-R@1 table: §5.2.

### A.4 Component probes on the exact target

Logistic regression / MLP(128) / 5-NN, 3-fold CV on 2,304 test words from the
holdout-0.2 split:

| Component | Chance | Linear | MLP | 5-NN |
|---|---:|---:|---:|---:|
| prefix | 0.125 | 0.118 | 0.102 | 0.115 |
| root | 0.008 | 0.011 | 0.009 | 0.007 |
| suffix | 0.083 | 0.086 | 0.099 | 0.085 |

All at chance. This is why §5.3 demotes the component probe to a negative
control.

### A.5 Symbol collisions and EOW

- Distinct symbol ids with identical expansion strings: 0 at vocabularies 256,
  512, 1024, 2048, 4096. The non-injective channel is dormant.
- `EOW` expands to `"\u0000</w>"` (5 characters); it must be treated as length 0
  for word spans. `word_boundaries` drops the last node and is safe for the
  current tests, but Phase 2 uses `forest.spans` instead.
- `UNK` does not occur on the synthetic corpus (the alphabet is closed); report
  it as 0 and keep the `<unk>` vector for safety.

### A.6 Runtime anchors

- Tokenizer training: 2.2–2.7 s per (method, vocabulary, data seed).
- Naive tree loop: 0.32 s/step at batch 256 (258 s per 800-step run). Replace
  with the vectorised per-depth encoder.
- GRU: 25 s per 800-step run at batch 256.
- Flat/oracle: < 5 s per run.
- Retrieval: < 0.1 s per run; probes: 10–30 s per run.

---

## Appendix B. Notation summary

| Symbol | Meaning |
|---|---|
| \(A\) | terminal alphabet (characters, plus `EOW`/`UNK`) |
| \(R\) | finite set of binary merge rules |
| \(S(\mathcal{G})\) | finite symbols (initial algebra) |
| \(\sigma\) | expansion \(S(\mathcal{G}) \to A^{*}\) |
| \(w\) | a word string |
| \(r_1,\dots,r_m\) | derivation forest roots returned by `encode_nodes` |
| \(e\) | compositional embedding \(S(\mathcal{G}) \to \mathbb{R}^{d}\) |
| \(g_\rho\) | composition operator for role \(\rho\) |
| \(d\) | embedding dimension (16) |
| \(P_p, S_s\) | prefix/suffix orthogonal matrices in the ground truth |
| \(r_r\) | root unit vector |
| \(v(w) = P_p S_s r_r\) | ground-truth word vector |
| \(\Delta\) | compositional-generalisation gap, \(\cos_{\text{fit}} - \cos_{\text{test}}\) |
| HC / HR / HA | held-out combinations / roots / affixes |
| O1 / O2 | exact oracle / oracle segmentation with learned composition |
