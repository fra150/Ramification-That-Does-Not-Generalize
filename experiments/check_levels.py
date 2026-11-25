"""Check whether hierarchical BPE is more than a partition of the flat merge sequence.

The v0.2 implementation trains level 1 for b1 merges, then level 2 on the
level-1 output for b2 merges. If the algorithm is deterministic, level 2 is
just the continuation of the same greedy merge sequence; the levels only change
the heap tie-breaking. This script tests that claim directly.

It compares:
- the rule sequence of flat BPE vs hier2, by expansion strings;
- the first divergence;
- the pair counts at the level boundary;
- whether the divergent rule is involved in a tie.

Usage:
    python -m experiments.check_levels --corpus synthetic --vocab 512 --seed 42
    python -m experiments.check_levels --corpus code --vocab 512 --seed 42
"""

from __future__ import annotations

import argparse
import re
from collections import Counter

from ramified.bpe import EOW, train_bpe, train_tokenizer
from ramified.codec import tokenize_chunks
from ramified.corpora import load_code_corpus
from ramified.synthetic import generate_synthetic


def rule_triples(tok):
    return [
        (tok.expansions[m.left], tok.expansions[m.right], tok.expansions[m.new])
        for level in tok.levels
        for m in level
    ]


def rule_counts(tok):
    return [
        (tok.expansions[m.left], tok.expansions[m.right], m.count)
        for level in tok.levels
        for m in level
    ]


def first_divergence(a, b):
    for i, (x, y) in enumerate(zip(a, b)):
        if x != y:
            return i, x, y
    if len(a) != len(b):
        return min(len(a), len(b)), None, None
    return None, None, None


def pair_counts_at_boundary(text, b1, split):
    """Run flat BPE for b1 merges and return the top pairs of the resulting state."""
    words = text.split() if split == "chunks" else re.findall(r"\w+|[^\w\s]", text)
    freq = Counter(words)
    chars = sorted(set("".join(freq.keys())))
    char_to_id = {c: i for i, c in enumerate(chars)}
    unk_id = len(chars)
    eow_id = unk_id + 1
    expansions = {i: c for c, i in char_to_id.items()}
    expansions[unk_id] = "<unk>"
    expansions[eow_id] = EOW
    sequences = [[char_to_id[c] for c in w] + [eow_id] for w in freq]
    freqs = list(freq.values())
    merges, out = train_bpe(sequences, freqs, b1, eow_id + 1, 0, expansions)
    counts = Counter()
    for seq, f in zip(out, freqs):
        for i in range(len(seq) - 1):
            counts[(seq[i], seq[i + 1])] += f
    top = counts.most_common(10)
    return [(expansions[a], expansions[b], c) for (a, b), c in top]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", choices=["synthetic", "code"], default="synthetic")
    ap.add_argument("--vocab", type=int, default=512)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--split", default="chunks")
    args = ap.parse_args()

    if args.corpus == "synthetic":
        syn = generate_synthetic(seed=args.seed)
        train_text, test_text = syn.train_text, syn.test_text
    else:
        code = load_code_corpus()
        split = int(len(code) * 0.9)
        nl = code.find("\n", split)
        if nl > 0:
            split = nl
        train_text, test_text = code[:split], code[split:]

    flat = train_tokenizer(
        [train_text], vocab_size=args.vocab, n_levels=1, split=args.split
    )
    hier = train_tokenizer(
        [train_text],
        vocab_size=args.vocab,
        n_levels=2,
        level_weights=[0.6, 0.4],
        split=args.split,
    )

    rf = rule_triples(flat)
    rh = rule_triples(hier)
    i, x, y = first_divergence(rf, rh)

    print(f"corpus={args.corpus} vocab={args.vocab} seed={args.seed}")
    print(f"flat levels={[len(l) for l in flat.levels]} rules={len(rf)}")
    print(f"hier levels={[len(l) for l in hier.levels]} rules={len(rh)}")
    print(f"common rule triples: {len(set(rf) & set(rh))} / {len(rf)}")
    if i is None:
        print("rule sequences identical")
    else:
        print(f"first divergence at rule {i}: flat={x} hier={y}")
        print(f"flat rules around: {rf[max(0,i-2):i+3]}")
        print(f"hier rules around: {rh[max(0,i-2):i+3]}")

    # Compare the level-1 prefix with the flat prefix.
    b1 = len(hier.levels[0]) if hier.levels else 0
    prefix_flat = rf[:b1]
    prefix_hier = rule_triples(hier)[:b1]
    print(f"level-1 budget b1={b1}; prefixes identical: {prefix_flat == prefix_hier}")

    # Pair counts at the boundary, from the flat run.
    top = pair_counts_at_boundary(train_text, b1, args.split)
    print("top pairs at the boundary (flat state):")
    for a, b, c in top:
        print(f"  {a!r} + {b!r}: {c}")

    # Tokenization agreement on a sample.
    a_ids = tokenize_chunks(test_text[:5000], flat)
    b_ids = tokenize_chunks(test_text[:5000], hier)
    print(f"sample tokens flat/hier: {len(a_ids)}/{len(b_ids)} identical={a_ids == b_ids}")

    # Count how many rule counts at the divergence point are tied.
    cf = rule_counts(flat)
    if i is not None and i < len(cf):
        target = cf[i][2]
        tied = [r for r in cf if r[2] == target]
        print(f"flat rule {i} count={target}; rules with same count={len(tied)}")


if __name__ == "__main__":
    main()
