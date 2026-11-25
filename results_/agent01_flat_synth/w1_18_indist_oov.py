"""W1-18: OOV check for the in-distribution 90/10 bpc reversal.

For each seed, fit the unigram counts on the first 90% of training chunks and
score the last 10%. Reports unseen tokens/types and the seen-only bpc, so the
reversal cannot be blamed on add-alpha OOV treatment.

Usage:
    python results/redteam/w1_18_indist_oov.py --seeds 42,43,44,45,46 \
        --out results/redteam/w1_18_indist_oov_a.csv
"""

from __future__ import annotations

import argparse
import csv
import math
import statistics
import sys
from collections import Counter

sys.path.insert(0, ".")

from ramified.bpe import train_tokenizer
from ramified.codec import normalize
from ramified.levels import assign_levels
from ramified.synthetic import generate_synthetic
from results.redteam.w1_18_bpc_audit import provenance_weighted


def bpc(counts, n, ids, V, L, alpha=0.1):
    denom = n + alpha * V
    return sum(-math.log2((counts.get(t, 0) + alpha) / denom) for t in ids) / L


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="42,43,44,45,46,47,48,49,50,51")
    ap.add_argument("--out", default="results/redteam/w1_18_indist_oov.csv")
    args = ap.parse_args()
    seeds = [int(x) for x in args.seeds.split(",") if x.strip()]
    rows = []
    for seed in seeds:
        syn = generate_synthetic(seed=seed)
        chunks = syn.train_text.split()
        cut = int(0.9 * len(chunks))
        fit_chunks, held_chunks = chunks[:cut], chunks[cut:]
        held_text = " ".join(held_chunks)
        held_len = len(normalize(held_text))
        flat = train_tokenizer(
            [syn.train_text], vocab_size=512, n_levels=1, split="chunks"
        )
        stats = provenance_weighted(flat, syn.train_word_objects, syn.train_text)
        for mode in ("frequency", "morph"):
            tok = assign_levels(
                flat, stats, mode=mode, level1_fraction=0.6, seed=seed * 1000 + 512
            )
            cache = {
                w: tok.encode(w, eow=True)
                for w in set(fit_chunks) | set(held_chunks)
            }
            fit_ids, held_ids = [], []
            for w, c in Counter(fit_chunks).items():
                fit_ids.extend(cache[w] * c)
            for w in held_chunks:
                held_ids.extend(cache[w])
            counts = Counter(fit_ids)
            n = len(fit_ids)
            unseen = sum(1 for t in held_ids if t not in counts)
            unseen_types = len({t for t in held_ids if t not in counts})
            seen_ids = [t for t in held_ids if t in counts]
            seen_len = len(held_ids)
            rows.append({
                "seed": seed, "mode": mode,
                "held_tokens": seen_len,
                "held_chars": held_len,
                "fit_tokens": n,
                "unseen_tokens": unseen,
                "unseen_types": unseen_types,
                "unseen_share": unseen / seen_len,
                "bpc_add0.1": bpc(counts, n, held_ids, tok.vocab_size, held_len),
                "bpc_seen_only": (
                    sum(-math.log2((counts[t] + 0.1) / (n + 0.1 * tok.vocab_size))
                        for t in seen_ids) / held_len
                ),
                "tpc": seen_len / held_len,
            })
        print(f"seed={seed} done", flush=True)

    with open(args.out, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    idx = {(r["seed"], r["mode"]): r for r in rows}
    print("\n== morph - frequency, in-distribution 90/10 ==")
    for metric in ("unseen_tokens", "bpc_add0.1", "bpc_seen_only", "tpc"):
        d = [
            idx[(s, "morph")][metric] - idx[(s, "frequency")][metric]
            for s in seeds
        ]
        mean = statistics.mean(d)
        sd = statistics.stdev(d) if len(d) > 1 else 0.0
        se = sd / math.sqrt(len(d)) if len(d) > 1 else 0.0
        print(
            f"{metric:<16} mean={mean:+.6f} sd={sd:.6f} "
            f"t={mean / se if se else float('inf'):+7.2f} "
            f"W/L/T={sum(x > 0 for x in d)}/{sum(x < 0 for x in d)}/{sum(x == 0 for x in d)}"
        )


if __name__ == "__main__":
    main()
