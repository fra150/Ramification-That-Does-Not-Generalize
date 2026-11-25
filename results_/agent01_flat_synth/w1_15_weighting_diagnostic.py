"""Why does unique-word provenance give the same level assignment as
frequency-weighted provenance? Compare the raw per-rule morph scores."""

from __future__ import annotations

import argparse
import csv
import statistics
from collections import Counter

from ramified.bpe import train_tokenizer
from ramified.levels import assign_levels, collect_provenance
from ramified.synthetic import generate_synthetic

from results.redteam.w1_15_levels_weighting import agg, jaccard, level1_set


def ranks(xs):
    order = sorted(range(len(xs)), key=lambda i: xs[i])
    out = [0.0] * len(xs)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and xs[order[j + 1]] == xs[order[i]]:
            j += 1
        mean_rank = (i + j) / 2.0
        for k in range(i, j + 1):
            out[order[k]] = mean_rank
        i = j + 1
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="42,43,44,45,46")
    ap.add_argument("--vocab", type=int, default=512)
    ap.add_argument("--fraction", type=float, default=0.6)
    ap.add_argument("--out", default="results/redteam/w1_15_weighting_diagnostic.csv")
    args = ap.parse_args()

    out = []
    for seed in [int(x) for x in args.seeds.split(",") if x.strip()]:
        syn = generate_synthetic(seed=seed)
        flat = train_tokenizer(
            [syn.train_text], vocab_size=args.vocab, n_levels=1, split="chunks"
        )
        words = list(syn.train_word_objects)
        counts = Counter(syn.train_text.split())
        weights = [counts.get(w.word, 0) for w in words]
        wc = [collect_provenance(flat, [w]) for w in words]
        idx = list(range(len(words)))
        stats_u = agg(wc, idx)
        stats_f = agg(wc, idx, weights)

        rules = sorted(flat.levels[0], key=lambda m: m.rank)
        keys = [(m.left, m.right) for m in rules]
        su, sf = [], []
        for key in keys:
            a, b = stats_u.get(key, (0, 0))
            su.append(a / (a + b) if a + b else None)
            a, b = stats_f.get(key, (0, 0))
            sf.append(a / (a + b) if a + b else None)

        common = [(x, y) for x, y in zip(su, sf) if x is not None and y is not None]
        xs = [x for x, _ in common]
        ys = [y for _, y in common]
        diffs = [abs(x - y) for x, y in common]
        rho = statistics.correlation(xs, ys)
        rho_s = statistics.correlation(ranks(xs), ranks(ys))

        tu = sum(a + b for a, b in stats_u.values())
        tf = sum(a + b for a, b in stats_f.values())

        # Reproduce the two level-1 sets at the design fraction.
        tu_tok = assign_levels(flat, stats_u, mode="morph",
                               level1_fraction=args.fraction, seed=seed)
        tf_tok = assign_levels(flat, stats_f, mode="morph",
                               level1_fraction=args.fraction, seed=seed)
        overlap = len(level1_set(tu_tok) & level1_set(tf_tok))
        n_l1 = len(level1_set(tf_tok))

        row = {
            "seed": seed,
            "n_rules_with_both_scores": len(common),
            "pearson_score": rho,
            "spearman_score": rho_s,
            "mean_abs_score_diff": statistics.mean(diffs),
            "max_abs_score_diff": max(diffs),
            "n_score_diff_gt_0.01": sum(1 for d in diffs if d > 0.01),
            "n_score_diff_gt_0.05": sum(1 for d in diffs if d > 0.05),
            "total_count_unique": tu,
            "total_count_freq": tf,
            "freq_over_unique": tf / tu if tu else float("nan"),
            "n_l1": n_l1,
            "l1_overlap": overlap,
            "l1_jaccard": jaccard(level1_set(tu_tok), level1_set(tf_tok)),
        }
        out.append(row)
        print(
            f"seed={seed} pearson={rho:.6f} spearman={rho_s:.6f} "
            f"mean|dscore|={statistics.mean(diffs):.4f} "
            f"max|dscore|={max(diffs):.4f} "
            f"L1 overlap={overlap}/{n_l1} "
            f"total counts u/f={tu}/{tf}",
            flush=True,
        )

    keys = list(out[0].keys())
    with open(args.out, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        w.writerows(out)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
