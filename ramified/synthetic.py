"""Synthetic morphologically-rich corpus with ground-truth composition.

Each word is prefix + root + suffix. The meaning is a vector:

    v(word) = P(prefix) @ S(suffix) @ r(root)

The composition is non-commutative on purpose: a flat bag of subwords loses
the order, a tree can preserve it. Some prefix/root/suffix combinations are
held out, so we can measure compositional generalisation on unseen words.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import random
import numpy as np

CONSONANTS = "ptkbdgmnslr"
VOWELS = "aeiou"

PREFIXES = ["un", "re", "pre", "dis", "in", "sub", "ad", "con"]
SUFFIXES = ["is", "or", "al", "ik", "os", "em", "at", "ul", "ar", "en", "ov", "ex"]

# Variable-length morphology, for the positional control in Fase 1f. Each
# inventory is prefix-free (resp. suffix-free): no item is a prefix (resp.
# suffix) of another, so the boundary position depends on the morpheme's
# identity and not on a fixed slot. This is what makes ``const2`` a real
# control instead of a copy of the ground truth.
PREFIXES_BY_LENGTH = {
    1: ["e", "o", "t", "g"],
    2: ["un", "re", "in", "ad"],
    3: ["pre", "dis", "sub", "con"],
    4: ["anti", "auto", "mono", "poly"],
}
SUFFIXES_BY_LENGTH = {
    1: ["a", "o", "e", "u"],
    2: ["is", "or", "al", "ik"],
    3: ["ost", "ent", "ark", "ump"],
}

ROOT_PATTERNS_FIXED = ["CVCVC", "CVC", "CVCCV", "VCVC"]
ROOT_PATTERNS_VARIABLE = ["CVC", "CVCV", "CVCVC", "CVCCV", "VCVC", "VCVCV", "CCVCV"]


@dataclass
class Word:
    word: str
    prefix: str
    root: str
    suffix: str
    boundaries: tuple[int, ...]
    vector: np.ndarray


@dataclass
class SyntheticCorpus:
    train_text: str
    test_text: str
    test_words: list[Word]
    train_words: list[str]
    train_word_objects: list[Word]
    dim: int
    morphology: str = "fixed"
    split_report: dict = field(default_factory=dict)


def _make_root(rng: random.Random, patterns=ROOT_PATTERNS_FIXED) -> str:
    pattern = rng.choice(patterns)
    return "".join(
        rng.choice(CONSONANTS) if p == "C" else rng.choice(VOWELS) for p in pattern
    )


def _orthogonal(rng: np.random.Generator, d: int) -> np.ndarray:
    q, _ = np.linalg.qr(rng.normal(size=(d, d)))
    return q


def generate_synthetic(
    seed: int = 42,
    n_roots: int = 120,
    dim: int = 16,
    holdout: float = 0.2,
    n_train_tokens: int = 120_000,
    n_test_tokens: int = 2_000,
    holdout_mode: str = "triple",
    pair_holdout: float = 0.10,
    morphology: str = "fixed",
) -> SyntheticCorpus:
    rng = random.Random(seed)
    np_rng = np.random.default_rng(seed)

    if morphology == "fixed":
        prefixes = list(PREFIXES)
        suffixes = list(SUFFIXES)
        root_patterns = ROOT_PATTERNS_FIXED
    elif morphology == "variable":
        prefixes = [
            p for length in sorted(PREFIXES_BY_LENGTH) for p in PREFIXES_BY_LENGTH[length]
        ]
        suffixes = [
            s for length in sorted(SUFFIXES_BY_LENGTH) for s in SUFFIXES_BY_LENGTH[length]
        ]
        root_patterns = ROOT_PATTERNS_VARIABLE
    else:
        raise ValueError(f"unknown morphology: {morphology}")

    roots = []
    seen = set()
    while len(roots) < n_roots:
        r = _make_root(rng, root_patterns)
        if r not in seen:
            seen.add(r)
            roots.append(r)

    root_vec = {}
    for r in roots:
        v = np_rng.normal(size=dim)
        root_vec[r] = v / np.linalg.norm(v)

    prefix_mat = {p: _orthogonal(np_rng, dim) for p in prefixes}
    suffix_mat = {s: _orthogonal(np_rng, dim) for s in suffixes}

    all_words = []
    by_string = {}
    duplicates = 0
    for p in prefixes:
        for r in roots:
            for s in suffixes:
                w = p + r + s
                if w in by_string:
                    # Two different decompositions producing the same string
                    # would make the ground truth ambiguous. Keep the first
                    # and count the collision; the caller can inspect it.
                    if (by_string[w].prefix, by_string[w].root, by_string[w].suffix) != (p, r, s):
                        duplicates += 1
                    continue
                vec = prefix_mat[p] @ (suffix_mat[s] @ root_vec[r])
                obj = Word(
                    word=w,
                    prefix=p,
                    root=r,
                    suffix=s,
                    boundaries=(len(p), len(p) + len(r)),
                    vector=vec,
                )
                by_string[w] = obj
                all_words.append(obj)

    split_report = {
        "n_duplicate_strings": duplicates,
        "n_dropped_test": 0,
    }

    if holdout_mode == "triple":
        rng.shuffle(all_words)
        n_test = int(len(all_words) * holdout)
        test_words = all_words[:n_test]
        train_words_objs = all_words[n_test:]
    elif holdout_mode == "pair":
        # Hold out whole (prefix, root) and (root, suffix) pairs. A test word
        # contains at least one unseen pair, so the exact triple is not the only
        # novelty. Components remain attested because each prefix/root/suffix
        # still occurs in other pairs.
        pr_pairs = [(p, r) for p in prefixes for r in roots]
        rs_pairs = [(r, s) for r in roots for s in suffixes]
        rng.shuffle(pr_pairs)
        rng.shuffle(rs_pairs)
        held_pr = set(pr_pairs[: int(len(pr_pairs) * pair_holdout)])
        held_rs = set(rs_pairs[: int(len(rs_pairs) * pair_holdout)])
        split_report["n_held_pr"] = len(held_pr)
        split_report["n_held_rs"] = len(held_rs)

        def is_test(w):
            return (w.prefix, w.root) in held_pr or (w.root, w.suffix) in held_rs

        test_words = [w for w in all_words if is_test(w)]
        train_words_objs = [w for w in all_words if not is_test(w)]
    else:
        raise ValueError(f"unknown holdout_mode: {holdout_mode}")

    # Ensure every test component is attested in training. Do not move an
    # unattested test word into training: that would re-admit a held pair and
    # leak the split. Drop it instead and report the count.
    train_roots = {w.root for w in train_words_objs}
    train_prefixes = {w.prefix for w in train_words_objs}
    train_suffixes = {w.suffix for w in train_words_objs}
    kept_test = []
    for w in test_words:
        if w.root in train_roots and w.prefix in train_prefixes and w.suffix in train_suffixes:
            kept_test.append(w)
    split_report["n_dropped_test"] = len(test_words) - len(kept_test)
    test_words = kept_test
    if not test_words:
        raise ValueError("holdout produced no test words; lower pair_holdout or change seed")

    train_pr = {(w.prefix, w.root) for w in train_words_objs}
    train_rs = {(w.root, w.suffix) for w in train_words_objs}
    split_report["n_test_unseen_pr"] = sum(
        1 for w in test_words if (w.prefix, w.root) not in train_pr
    )
    split_report["n_test_unseen_rs"] = sum(
        1 for w in test_words if (w.root, w.suffix) not in train_rs
    )
    split_report["n_test_all_pairs_seen"] = sum(
        1
        for w in test_words
        if (w.prefix, w.root) in train_pr and (w.root, w.suffix) in train_rs
    )
    split_report["n_train_words"] = len(train_words_objs)
    split_report["n_test_words"] = len(test_words)

    train_words = [w.word for w in train_words_objs]

    # Zipfian sampling of training tokens.
    order = list(range(len(train_words)))
    rng.shuffle(order)
    probs = np.array([1.0 / (rank + 1) ** 1.1 for rank in order], dtype=np.float64)
    probs /= probs.sum()
    sampled_idx = np_rng.choice(len(train_words), size=n_train_tokens, p=probs)

    tokens = []
    for i, idx in enumerate(sampled_idx):
        tokens.append(train_words[idx])
        if rng.random() < 0.07:
            tokens.append(".")
    train_text = " ".join(tokens)

    # Held-out text: uniform over test words, so rare combinations are stressed.
    test_tokens = [test_words[i].word for i in np_rng.integers(0, len(test_words), size=n_test_tokens)]
    test_text = " ".join(test_tokens)

    return SyntheticCorpus(
        train_text=train_text,
        test_text=test_text,
        test_words=test_words,
        train_words=train_words,
        train_word_objects=train_words_objs,
        dim=dim,
        morphology=morphology,
        split_report=split_report,
    )
