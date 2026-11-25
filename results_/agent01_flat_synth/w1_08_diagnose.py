"""W1-08 mechanism check on seed 42: what changes when the tie-break flips?

Trains the id and rev_id flat tokenizers, builds frequency/morph level
assignments for both, and compares:
  - semantic rule sets (rules keyed by expansion strings, not symbol ids);
  - morphological level-1 semantic sets;
  - per-test-word boundary predictions and per-word F1 contributions.

This explains why the aggregate morph - frequency delta can be invariant even
though the underlying rule sets differ.
"""

from __future__ import annotations

import os
import sys
from collections import Counter

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)

from ramified.bpe import train_tokenizer  # noqa: E402
from ramified.levels import assign_levels, collect_provenance  # noqa: E402
from ramified.metrics import boundary_scores  # noqa: E402
from ramified.synthetic import generate_synthetic  # noqa: E402
from w1_08_tiebreak import install_variant  # noqa: E402

SEED = 42
VOCAB = 512
FRACTION = 0.6


def sem(rule, tok):
    return (
        tok.expansions[rule.left],
        tok.expansions[rule.right],
        tok.expansions[rule.new],
    )


def first_diff(a, b):
    for i, (x, y) in enumerate(zip(a, b)):
        if x != y:
            return i
    return None if len(a) == len(b) else min(len(a), len(b))


def main():
    syn = generate_synthetic(seed=SEED)
    words = syn.test_words
    toks = {}
    for variant in ("id", "rev_id"):
        install_variant(variant)
        flat = train_tokenizer(
            [syn.train_text], vocab_size=VOCAB, n_levels=1, split="chunks"
        )
        stats = collect_provenance(flat, syn.train_word_objects, text=syn.train_text)
        morph = assign_levels(
            flat, stats, mode="morph", level1_fraction=FRACTION, seed=SEED * 1000 + VOCAB
        )
        freq = assign_levels(
            flat,
            stats,
            mode="frequency",
            level1_fraction=FRACTION,
            seed=SEED * 1000 + VOCAB,
        )
        toks[variant] = {"flat": flat, "morph": morph, "frequency": freq}

    a, b = toks["id"], toks["rev_id"]

    print("=== semantic rule sets (seed 42, v512) ===")
    for variant in ("id", "rev_id"):
        tok = toks[variant]["flat"]
        seq = [sem(m, tok) for m in tok.levels[0]]
        print(f"  {variant}: {len(seq)} flat rules")
    a_seq = [sem(m, a["flat"]) for m in a["flat"].levels[0]]
    b_seq = [sem(m, b["flat"]) for m in b["flat"].levels[0]]
    common = len(set(a_seq) & set(b_seq))
    print(
        f"  id/rev_id flat semantic overlap: {common}/{len(set(a_seq) | set(b_seq))} "
        f"union; first semantic divergence at rank {first_diff(a_seq, b_seq)}"
    )
    a_l1 = {sem(m, a["morph"]) for m in a["morph"].levels[0]}
    b_l1 = {sem(m, b["morph"]) for m in b["morph"].levels[0]}
    print(
        f"  morph level-1 semantic overlap: {len(a_l1 & b_l1)}/{len(a_l1 | b_l1)} "
        f"union; sizes {len(a_l1)} vs {len(b_l1)}"
    )

    print("\n=== test-word boundary predictions ===")
    words_sorted = [w.word for w in words]
    pred = {}
    for variant in ("id", "rev_id"):
        for mode in ("flat", "frequency", "morph"):
            tok = toks[variant][mode]
            pred[(variant, mode)] = {
                w.word: tuple(tok.word_boundaries(w.word)) for w in words
            }
    for mode in ("flat", "frequency", "morph"):
        pa = pred[("id", mode)]
        pb = pred[("rev_id", mode)]
        diff = sum(1 for w in words_sorted if pa[w] != pb[w])
        print(f"  {mode:<9} words segmented differently id vs rev_id: {diff}/{len(words)}")

    print("\n=== per-word F1 contribution: morph - frequency ===")
    true = {w.word: w.boundaries for w in words}
    contrib = {}
    for variant in ("id", "rev_id"):
        vals = {}
        for w in words_sorted:
            _, _, f_m = boundary_scores(pred[(variant, "morph")][w], true[w])
            _, _, f_f = boundary_scores(pred[(variant, "frequency")][w], true[w])
            vals[w] = (f_m, f_f, f_m - f_f)
        contrib[variant] = vals
        changed = sum(1 for w in words_sorted if vals[w][2] != 0)
        better = sum(1 for w in words_sorted if vals[w][2] > 0)
        worse = sum(1 for w in words_sorted if vals[w][2] < 0)
        total = sum(vals[w][2] for w in words_sorted) / len(words_sorted)
        print(
            f"  {variant:<7} changed {changed}/{len(words)} words "
            f"(better {better}, worse {worse}), mean dF1 = {total:+.9f}"
        )

    ca = Counter(round(contrib["id"][w][2], 12) for w in words_sorted)
    cb = Counter(round(contrib["rev_id"][w][2], 12) for w in words_sorted)
    if ca == cb:
        print("  per-word F1-delta multiset is IDENTICAL between id and rev_id")
    else:
        only_a = sum((ca - cb).values())
        only_b = sum((cb - ca).values())
        print(
            f"  per-word F1-delta multiset differs: {only_a} words only in id, "
            f"{only_b} only in rev_id"
        )
    same_words = sum(
        1 for w in words_sorted if (contrib["id"][w][2] != 0) == (contrib["rev_id"][w][2] != 0)
    )
    print(f"  same set of changed words: {same_words}/{len(words)}")


if __name__ == "__main__":
    main()
