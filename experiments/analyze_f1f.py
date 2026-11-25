"""Compact paired analysis for the Fase 1f design.

Reads the four CSVs of the 2 x 2 (morphology x split) and reports, per cell:

- the mean of each mode on boundary F1 (macro), bpc and tokens/char;
- the paired difference of the comparisons that decide the criterion:
  morph - frequency, morph - const2, morph - near2, morph - random2,
  random_closed - frequency.

The criterion is pre-registered in ``diario.md`` (2026-11-24). This script
only computes; it does not decide.

Usage:
    python -m experiments.analyze_f1f
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import statistics

T95 = {2: 12.706, 3: 4.303, 4: 3.182, 5: 2.776, 6: 2.571, 7: 2.447, 8: 2.365, 9: 2.306, 10: 2.262}

CELLS = [
    ("variable/triple", "results/f1f_variable_triple.csv"),
    ("variable/pair", "results/f1f_variable_pair.csv"),
    ("fixed/triple", "results/f1f_fixed_triple.csv"),
    ("fixed/pair", "results/f1f_fixed_pair.csv"),
]

COMPARISONS = [
    ("morph", "frequency"),
    ("morph", "morph_const2"),
    ("morph", "morph_near2"),
    ("morph", "morph_random2"),
    ("random_closed", "frequency"),
    ("morph_const2", "frequency"),
    ("morph_near2", "frequency"),
]

METRICS = ["boundary_f1", "boundary_macro_f1", "boundary_micro_f1", "bits_per_char"]


def load(path):
    rows = []
    with open(path, encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            rows.append(row)
    return rows


def index(rows):
    """(seed, mode) -> row. One CSV holds one cell, so vocab is constant."""
    return {(int(r["seed"]), r["mode"]): r for r in rows}


def describe(deltas):
    n = len(deltas)
    if n == 0:
        return None
    mean = statistics.mean(deltas)
    std = statistics.stdev(deltas) if n > 1 else 0.0
    se = std / math.sqrt(n) if n > 1 else 0.0
    t = mean / se if se > 0 else float("inf")
    crit = T95.get(n, 1.96)
    return {
        "n": n,
        "mean": mean,
        "std": std,
        "t": t,
        "ci_low": mean - crit * se,
        "ci_high": mean + crit * se,
        "wins": sum(1 for d in deltas if d > 0),
        "losses": sum(1 for d in deltas if d < 0),
        "ties": sum(1 for d in deltas if d == 0),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/f1f_summary.csv")
    args = ap.parse_args()

    all_stats = []
    for cell, path in CELLS:
        if not os.path.exists(path):
            print(f"missing: {path}")
            continue
        rows = load(path)
        idx = index(rows)
        seeds = sorted({int(r["seed"]) for r in rows})
        modes = sorted({r["mode"] for r in rows})
        print(f"\n=== {cell} ({len(seeds)} seeds) ===")

        print(f"{'mode':<16}" + "".join(f"{m:>12}" for m in METRICS))
        for mode in modes:
            vals = []
            for metric in METRICS:
                xs = [float(idx[(s, mode)][metric]) for s in seeds if (s, mode) in idx]
                vals.append(f"{statistics.mean(xs):>12.4f}" if xs else f"{'-':>12}")
            print(f"{mode:<16}" + "".join(vals))

        for metric in METRICS:
            print(f"  {metric}")
            for a, b in COMPARISONS:
                deltas = []
                for s in seeds:
                    ra, rb = idx.get((s, a)), idx.get((s, b))
                    if ra is None or rb is None:
                        continue
                    deltas.append(float(ra[metric]) - float(rb[metric]))
                stats = describe(deltas)
                if stats is None:
                    continue
                print(
                    f"    {a:<14} - {b:<14} "
                    f"mean={stats['mean']:+.4f} sd={stats['std']:.4f} "
                    f"t={stats['t']:+.2f} "
                    f"CI95=[{stats['ci_low']:+.4f},{stats['ci_high']:+.4f}] "
                    f"W/L/T={stats['wins']}/{stats['losses']}/{stats['ties']}"
                )
                all_stats.append({"cell": cell, "metric": metric, "a": a, "b": b, **stats})

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(all_stats[0].keys()))
        writer.writeheader()
        writer.writerows(all_stats)
    print(f"\nwrote {args.out} ({len(all_stats)} comparisons)")


if __name__ == "__main__":
    main()
