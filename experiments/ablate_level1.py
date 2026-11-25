"""Ablation: how much merge budget should level 1 get?

At vocab 1024 the 2-level tokenizer gives ~half of all merges to level 1 and
over-merges morphemes (boundary recall collapses). This sweep fixes the
level-1 budget and gives the rest to level 2. Flat BPE is included as a
control at each vocabulary size.

Usage:
    python -m experiments.ablate_level1
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import re
from collections import Counter

from ramified.bpe import train_tokenizer, tokenize_normalized
from ramified.codec import roundtrip_ok
from ramified.metrics import decode_exact, mean_boundary_f1
from ramified.synthetic import generate_synthetic


def bpc_alpha(train_ids, test_ids, vocab_size, test_char_len, alpha=0.1):
    if test_char_len == 0:
        return float("nan")
    counts = Counter(train_ids)
    total = sum(counts.values())
    denom = total + alpha * vocab_size
    bits = 0.0
    for t in test_ids:
        p = (counts.get(t, 0) + alpha) / denom
        bits -= math.log2(p)
    return bits / test_char_len


def evaluate(tok, syn):
    train_ids = tokenize_normalized(syn.train_text, tok)
    test_ids = tokenize_normalized(syn.test_text, tok)
    p, r, f1 = mean_boundary_f1(tok, syn.test_words)
    return {
        "tokens_per_char": len(test_ids) / max(1, len(syn.test_text)),
        "bits_per_char_a01": bpc_alpha(train_ids, test_ids, tok.vocab_size, len(syn.test_text)),
        "boundary_precision": p,
        "boundary_recall": r,
        "boundary_f1": f1,
        "decode_exact": decode_exact(tok, syn.test_words),
        "doc_roundtrip": roundtrip_ok(syn.test_text, tok),
        "n_merges_l1": len(tok.levels[0]) if tok.levels else 0,
        "n_merges_l2": len(tok.levels[1]) if len(tok.levels) > 1 else 0,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--vocabs", default="512,1024")
    ap.add_argument("--budgets", default="32,64,128,256,512")
    ap.add_argument("--out", default="results/ablation_level1.csv")
    args = ap.parse_args()

    syn = generate_synthetic(seed=args.seed)
    words = re.findall(r"\w+|[^\w\s]", syn.train_text)
    alphabet = len(set("".join(words)))

    vocabs = [int(x) for x in args.vocabs.split(",")]
    budgets = [int(x) for x in args.budgets.split(",")]

    rows = []
    for vocab in vocabs:
        total = vocab - alphabet - 2
        print(f"\nvocab={vocab} total_merges={total} alphabet={alphabet}")

        # Flat control.
        tok = train_tokenizer([syn.train_text], vocab_size=vocab, n_levels=1)
        m = evaluate(tok, syn)
        row = {"method": "flat", "vocab": vocab, "budget_l1": None, **m}
        rows.append(row)
        print(
            f"  flat      tok/ch={m['tokens_per_char']:.3f} bpc={m['bits_per_char_a01']:.3f} "
            f"F1={m['boundary_f1']:.3f} P={m['boundary_precision']:.3f} "
            f"R={m['boundary_recall']:.3f} doc={m['doc_roundtrip']}"
        )

        for budget in budgets:
            if budget >= total:
                continue
            w1 = budget / total
            tok = train_tokenizer(
                [syn.train_text],
                vocab_size=vocab,
                n_levels=2,
                level_weights=[w1, 1.0 - w1],
            )
            m = evaluate(tok, syn)
            row = {"method": "hier2", "vocab": vocab, "budget_l1": budget, **m}
            rows.append(row)
            print(
                f"  hier2 b1={budget:<4} tok/ch={m['tokens_per_char']:.3f} "
                f"bpc={m['bits_per_char_a01']:.3f} F1={m['boundary_f1']:.3f} "
                f"P={m['boundary_precision']:.3f} R={m['boundary_recall']:.3f} "
                f"L1={m['n_merges_l1']} L2={m['n_merges_l2']} doc={m['doc_roundtrip']}"
            )

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nwrote {args.out} ({len(rows)} runs)")


if __name__ == "__main__":
    main()
