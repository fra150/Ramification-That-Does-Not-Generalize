"""Sanity checks for the synthetic corpus and its splits.

Reports, per seed and generator variant:

- word counts and duplicate strings;
- whether every test component is attested in training;
- whether a pair-disjoint split actually has zero test words with all
  adjacent pairs seen in training;
- how often the true boundaries coincide with the fixed ``{2, L-2}`` slot
  (if this is 1.0, the positional control is a copy of the ground truth).

Usage:
    python -m experiments.split_check --seeds 42,43,44
    python -m experiments.split_check --morphologies fixed,variable `
        --holdout-modes triple,pair
"""

from __future__ import annotations

import argparse
from collections import Counter

from ramified.synthetic import generate_synthetic


def check(seed, morphology, holdout_mode, pair_holdout):
    syn = generate_synthetic(
        seed=seed,
        morphology=morphology,
        holdout_mode=holdout_mode,
        pair_holdout=pair_holdout,
    )
    train_pr = {(w.prefix, w.root) for w in syn.train_word_objects}
    train_rs = {(w.root, w.suffix) for w in syn.train_word_objects}
    train_prefixes = {w.prefix for w in syn.train_word_objects}
    train_roots = {w.root for w in syn.train_word_objects}
    train_suffixes = {w.suffix for w in syn.train_word_objects}

    missing_component = sum(
        1
        for w in syn.test_words
        if w.prefix not in train_prefixes
        or w.root not in train_roots
        or w.suffix not in train_suffixes
    )
    all_pairs_seen = sum(
        1
        for w in syn.test_words
        if (w.prefix, w.root) in train_pr and (w.root, w.suffix) in train_rs
    )
    const2_exact = sum(
        1
        for w in syn.test_words
        if tuple(w.boundaries) == (2, len(w.word) - 2)
    )

    print(
        f"seed={seed} morphology={morphology:<8} split={holdout_mode:<6} "
        f"train={len(syn.train_word_objects):>6} test={len(syn.test_words):>6} "
        f"dup={syn.split_report.get('n_duplicate_strings', 0):>3} "
        f"dropped={syn.split_report.get('n_dropped_test', 0):>3} "
        f"missing_component={missing_component:>3} "
        f"all_pairs_seen={all_pairs_seen:>5} "
        f"const2_exact={const2_exact:>5}/{len(syn.test_words):<5} "
        f"({const2_exact / max(1, len(syn.test_words)):.3f})"
    )
    print(
        "  "
        f"held_pr={syn.split_report.get('n_held_pr', '-')} "
        f"held_rs={syn.split_report.get('n_held_rs', '-')} "
        f"unseen_pr={syn.split_report.get('n_test_unseen_pr', '-')} "
        f"unseen_rs={syn.split_report.get('n_test_unseen_rs', '-')} "
        f"prefix_len={dict(sorted(Counter(len(w.prefix) for w in syn.test_words).items()))} "
        f"suffix_len={dict(sorted(Counter(len(w.suffix) for w in syn.test_words).items()))}"
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="42,43,44")
    ap.add_argument("--morphologies", default="fixed,variable")
    ap.add_argument("--holdout-modes", default="triple,pair")
    ap.add_argument("--pair-holdout", type=float, default=0.10)
    args = ap.parse_args()

    for seed in [int(s) for s in args.seeds.split(",") if s.strip()]:
        for morphology in [m.strip() for m in args.morphologies.split(",") if m.strip()]:
            for mode in [m.strip() for m in args.holdout_modes.split(",") if m.strip()]:
                check(seed, morphology, mode, args.pair_holdout)


if __name__ == "__main__":
    main()
