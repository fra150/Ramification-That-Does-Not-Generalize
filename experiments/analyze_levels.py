"""Paired statistics for the level-assignment experiment.

Reads a CSV produced by experiments.level_assignment and reports, for each
vocabulary size, the paired differences between modes across seeds. The same
seed uses the same synthetic corpus, so the comparison is paired.

Usage:
    python -m experiments.analyze_levels --csv results/level_assignment_v512_10seeds.csv
"""

from __future__ import annotations

import argparse
import csv
import math
import statistics
from collections import defaultdict

# Two-sided 95% t critical values for small samples.
T95 = {3: 4.303, 4: 3.182, 5: 2.776, 6: 2.571, 7: 2.447, 8: 2.365, 9: 2.306, 10: 2.262}


def load(path):
    rows = []
    with open(path, encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            rows.append(row)
    return rows


def paired(rows, vocab, mode_a, mode_b, metric):
    idx = {(int(r["seed"]), r["mode"]): r for r in rows if int(r["vocab"]) == vocab}
    deltas = []
    for (seed, m), row in idx.items():
        if m != mode_a:
            continue
        other = idx.get((seed, mode_b))
        if other is None:
            continue
        a = float(row[metric])
        b = float(other[metric])
        deltas.append(a - b)
    return deltas


def describe(deltas):
    n = len(deltas)
    mean = statistics.mean(deltas)
    std = statistics.stdev(deltas) if n > 1 else 0.0
    se = std / math.sqrt(n) if n > 1 else 0.0
    t = mean / se if se > 0 else float("inf")
    crit = T95.get(n, 1.96)
    return {
        "n": n,
        "mean": mean,
        "std": std,
        "se": se,
        "t": t,
        "ci_low": mean - crit * se,
        "ci_high": mean + crit * se,
        "wins": sum(1 for d in deltas if d > 0),
        "losses": sum(1 for d in deltas if d < 0),
        "ties": sum(1 for d in deltas if d == 0),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="results/level_assignment_v512_10seeds.csv")
    ap.add_argument("--out", default="results/level_assignment_stats.csv")
    args = ap.parse_args()

    rows = load(args.csv)
    vocabs = sorted({int(r["vocab"]) for r in rows})
    modes = sorted({r["mode"] for r in rows})
    pairs = [
        ("morph", "frequency"),
        ("random_closed", "frequency"),
        ("morph", "random_closed"),
        ("reverse", "frequency"),
        ("morph_const2", "frequency"),
        ("morph_random2", "frequency"),
        ("morph_near2", "frequency"),
        ("morph", "morph_const2"),
        ("morph", "morph_random2"),
        ("morph", "morph_near2"),
    ]

    out = []
    for vocab in vocabs:
        print(f"\nvocab={vocab}")
        for metric in (
            "boundary_f1",
            "boundary_macro_f1",
            "boundary_micro_f1",
            "bits_per_char",
        ):
            if not all(metric in r for r in rows):
                continue
            print(f"  {metric}")
            for a, b in pairs:
                if a not in modes or b not in modes:
                    continue
                stats = describe(paired(rows, vocab, a, b, metric))
                print(
                    f"    {a:<13} - {b:<10} "
                    f"mean={stats['mean']:+.4f} std={stats['std']:.4f} "
                    f"t={stats['t']:+.2f} "
                    f"CI95=[{stats['ci_low']:+.4f},{stats['ci_high']:+.4f}] "
                    f"W/L/T={stats['wins']}/{stats['losses']}/{stats['ties']}"
                )
                out.append(
                    {
                        "vocab": vocab,
                        "metric": metric,
                        "mode_a": a,
                        "mode_b": b,
                        **stats,
                    }
                )

    with open(args.out, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(out[0].keys()))
        writer.writeheader()
        writer.writerows(out)
    print(f"\nwrote {args.out} ({len(out)} comparisons)")


if __name__ == "__main__":
    main()
