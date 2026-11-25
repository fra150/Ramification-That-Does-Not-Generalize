"""W1-18 follow-up: distribution and model-order sensitivity of bpc.

Two checks the official estimator cannot answer:
  1. In-distribution text: estimate the unigram model on 90% of the training
     chunks and score the other 10% (Zipfian word distribution, seen words),
     instead of the uniform held-out-recombination test set.
  2. Bigram code: interpolated bigram/unigram cross-entropy on the official
     test set, instead of a unigram code.

No source files are modified.

Usage:
    python results/redteam/w1_18_bpc_alt.py
"""

from __future__ import annotations

import argparse
import csv
import math
import statistics
import sys
from collections import Counter

import numpy as np

sys.path.insert(0, ".")

from ramified.bpe import train_tokenizer
from ramified.codec import normalize
from ramified.levels import assign_levels
from ramified.synthetic import generate_synthetic
from results.redteam.w1_18_bpc_audit import MODES, describe, provenance_weighted


def bpc_unigram(train_ids, test_ids, V, char_len, alpha=0.1):
    c = Counter(train_ids)
    n = sum(c.values())
    denom = n + alpha * V
    return sum(-math.log2((c.get(t, 0) + alpha) / denom) for t in test_ids) / char_len


def bpc_bigram(train_ids, test_ids, V, char_len, lam, alpha=0.1):
    uni = Counter(train_ids)
    n = sum(uni.values())
    uniden = n + alpha * V
    bi = Counter(zip(train_ids, train_ids[1:]))
    prev_counts = Counter(train_ids[:-1])
    bits = 0.0
    prev = None
    for t in test_ids:
        pu = (uni.get(t, 0) + alpha) / uniden
        if prev is None or prev_counts.get(prev, 0) == 0:
            p = pu
        else:
            p = lam * bi.get((prev, t), 0) / prev_counts[prev] + (1.0 - lam) * pu
        bits -= math.log2(p)
        prev = t
    return bits / char_len


def bpc_kt_adaptive(test_ids, V, char_len):
    """Adaptive Krichevsky-Trofimov code length: no alpha, no train counts."""
    counts = Counter()
    n = 0
    bits = 0.0
    for t in test_ids:
        p = (counts.get(t, 0) + 0.5) / (n + 0.5 * V)
        bits -= math.log2(p)
        counts[t] += 1
        n += 1
    return bits / char_len


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="42,43,44,45,46,47,48,49,50,51")
    ap.add_argument("--out", default="results/redteam/w1_18_bpc_alt.csv")
    args = ap.parse_args()
    seeds = [int(x) for x in args.seeds.split(",") if x.strip()]
    rows = []
    for seed in seeds:
        syn = generate_synthetic(seed=seed)
        train_chunks = syn.train_text.split()
        flat = train_tokenizer(
            [syn.train_text], vocab_size=512, n_levels=1, split="chunks"
        )
        stats = provenance_weighted(flat, syn.train_word_objects, syn.train_text)
        toks = {"flat": flat}
        for mode in ("frequency", "morph", "random_closed"):
            toks[mode] = assign_levels(
                flat, stats, mode=mode, level1_fraction=0.6, seed=seed * 1000 + 512
            )

        test_chunks = syn.test_text.split()
        test_len = len(normalize(syn.test_text))
        cut = int(0.9 * len(train_chunks))
        fit_chunks, held_chunks = train_chunks[:cut], train_chunks[cut:]
        held_len = len(normalize(" ".join(held_chunks)))

        for mode in MODES:
            tok = toks[mode]
            vocab_words = set(train_chunks) | set(held_chunks) | set(test_chunks)
            cache = {w: tok.encode(w, eow=True) for w in vocab_words}

            def join_counts(chunks):
                out = []
                for w, c in Counter(chunks).items():
                    out.extend(cache[w] * c)
                return out

            def join(chunks):
                return [t for w in chunks for t in cache[w]]

            train_ids = join_counts(train_chunks)
            test_ids = join(test_chunks)
            fit_ids = join_counts(fit_chunks)
            held_ids = join(held_chunks)
            rows.append({
                "seed": seed, "mode": mode,
                "bpc_official_uniform_heldout": bpc_unigram(
                    train_ids, test_ids, tok.vocab_size, test_len
                ),
                "bpc_indist_zipf_90_10": bpc_unigram(
                    fit_ids, held_ids, tok.vocab_size, held_len
                ),
                "bpc_bigram_lam0.5": bpc_bigram(
                    train_ids, test_ids, tok.vocab_size, test_len, 0.5
                ),
                "bpc_bigram_lam0.8": bpc_bigram(
                    train_ids, test_ids, tok.vocab_size, test_len, 0.8
                ),
                "bpc_kt_adaptive": bpc_kt_adaptive(
                    test_ids, tok.vocab_size, test_len
                ),
                "tpc_uniform_heldout": len(test_ids) / test_len,
                "tpc_indist_heldout": len(held_ids) / held_len,
            })
        print(f"seed={seed} done", flush=True)

    with open(args.out, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    metrics = [k for k in rows[0] if k.startswith("bpc_") or k.startswith("tpc_")]
    print("\n== morph - frequency, paired over 10 seeds ==")
    for metric in metrics:
        d = [
            next(r[metric] for r in rows if r["seed"] == s and r["mode"] == "morph")
            - next(r[metric] for r in rows if r["seed"] == s and r["mode"] == "frequency")
            for s in seeds
        ]
        st = describe(d)
        print(
            f"{metric:<32} mean={st['mean']:+.5f} t={st['t']:+7.2f} "
            f"CI95=[{st['ci_low']:+.5f},{st['ci_high']:+.5f}] "
            f"W/L/T={st['wins']}/{st['losses']}/{st['ties']}"
        )

    print("\n== same, random_closed - frequency ==")
    for metric in metrics:
        d = [
            next(r[metric] for r in rows if r["seed"] == s and r["mode"] == "random_closed")
            - next(r[metric] for r in rows if r["seed"] == s and r["mode"] == "frequency")
            for s in seeds
        ]
        st = describe(d)
        print(
            f"{metric:<32} mean={st['mean']:+.5f} t={st['t']:+7.2f} "
            f"CI95=[{st['ci_low']:+.5f},{st['ci_high']:+.5f}] "
            f"W/L/T={st['wins']}/{st['losses']}/{st['ties']}"
        )


if __name__ == "__main__":
    main()
