"""Paired seed analysis for the v0.2 matrix.

The same synthetic corpus is used by every method within a seed, so the
scientific comparison is the paired difference per seed, not the difference of
means. This script computes, for every (corpus, vocab, method):

    delta_seed = metric(method, seed) - metric(flat, seed)

then reports mean, sample standard deviation, number of seeds, and win count.

Code rows are deterministic given the corpus and vocabulary: all seeds produce
the same metrics, so n_effective is 1 even though 3 rows exist. The script
reports that explicitly instead of pretending the code std is evidence.

Usage:
    python -m experiments.paired
    python -m experiments.paired --pattern "results/matrix/**/results.csv"
"""

from __future__ import annotations

import argparse
import csv
import glob
import math
import os
import statistics
from collections import defaultdict


def _float(value):
    if value in ("", None):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def load_rows(pattern):
    rows = []
    for path in sorted(glob.glob(pattern, recursive=True)):
        with open(path, encoding="utf-8") as fh:
            for row in csv.DictReader(fh):
                rows.append(row)
    return rows


def index(rows):
    """(corpus, method, vocab, seed) -> row."""
    out = {}
    for r in rows:
        key = (r["corpus"], r["method"], int(r["vocab_target"]), int(r["seed"]))
        out[key] = r
    return out


def _decode_word(row):
    if "decode_word" in row:
        return _float(row.get("decode_word"))
    return _float(row.get("decode_exact"))


def metric(row, name):
    if name == "decode_word":
        return _decode_word(row)
    return _float(row.get(name))


def paired(rows, metric_name, higher_is_better):
    idx = index(rows)
    groups = defaultdict(list)
    for (corpus, method, vocab, seed), row in idx.items():
        if method == "flat":
            continue
        base = idx.get((corpus, "flat", vocab, seed))
        if base is None:
            continue
        a = metric(row, metric_name)
        b = metric(base, metric_name)
        if a is None or b is None:
            continue
        groups[(corpus, method, vocab)].append((seed, a - b))

    out = []
    for (corpus, method, vocab), pairs in sorted(groups.items()):
        deltas = [d for _, d in pairs]
        # Distinct metric values, to expose the deterministic code corpus.
        vals = sorted({round(d, 12) for d in deltas})
        out.append(
            {
                "corpus": corpus,
                "method": method,
                "vocab_target": vocab,
                "n_seeds": len(deltas),
                "n_effective": len(vals),
                "delta_mean": statistics.mean(deltas),
                "delta_std": statistics.stdev(deltas) if len(deltas) > 1 else 0.0,
                "wins": sum(1 for d in deltas if (d > 0) == higher_is_better and d != 0),
                "losses": sum(1 for d in deltas if (d < 0) == higher_is_better and d != 0),
            }
        )
    return out


def print_table(title, rows, unit):
    print(f"\n{title}")
    header = (
        f"{'corpus':<9} {'method':<11} {'vocab':>5} {'n':>2} {'n_eff':>5} "
        f"{'mean_delta':>11} {'std':>7} {'wins':>5} {'losses':>6}"
    )
    print(header)
    print("-" * len(header))
    for r in rows:
        print(
            f"{r['corpus']:<9} {r['method']:<11} {r['vocab_target']:>5} "
            f"{r['n_seeds']:>2} {r['n_effective']:>5} "
            f"{r['delta_mean']:>+11.4f} {r['delta_std']:>7.4f} "
            f"{r['wins']:>5} {r['losses']:>6}"
        )
    print(f"({unit})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pattern", default="results/matrix/**/results.csv")
    ap.add_argument("--out", default="results/paired_summary.csv")
    args = ap.parse_args()

    rows = load_rows(args.pattern)
    if not rows:
        print(f"no results found for pattern: {args.pattern}")
        return

    f1 = paired(rows, "boundary_f1", higher_is_better=True)
    bpc = paired(rows, "bits_per_char", higher_is_better=False)
    tok = paired(rows, "tokens_per_char", higher_is_better=False)

    print_table(
        "boundary_f1: method minus flat, paired by seed (positive is better)",
        f1,
        "positive delta = method better",
    )
    print_table(
        "bits_per_char: method minus flat, paired by seed (negative is better)",
        bpc,
        "negative delta = method better",
    )
    print_table(
        "tokens_per_char: method minus flat, paired by seed (negative is better)",
        tok,
        "negative delta = method better",
    )

    # Merge by key, not by position: code rows have no boundary_f1, so the
    # f1 list is shorter than the bpc/tok lists.
    f1_by_key = {(r["corpus"], r["method"], r["vocab_target"]): r for r in f1}
    tok_by_key = {(r["corpus"], r["method"], r["vocab_target"]): r for r in tok}
    combined = []
    for b in bpc:
        key = (b["corpus"], b["method"], b["vocab_target"])
        a = f1_by_key.get(key)
        c = tok_by_key[key]
        combined.append(
            {
                "corpus": b["corpus"],
                "method": b["method"],
                "vocab_target": b["vocab_target"],
                "n_seeds": b["n_seeds"],
                "n_effective": b["n_effective"],
                "df1_mean": a["delta_mean"] if a else float("nan"),
                "df1_std": a["delta_std"] if a else float("nan"),
                "df1_wins": a["wins"] if a else "",
                "df1_losses": a["losses"] if a else "",
                "dbpc_mean": b["delta_mean"],
                "dbpc_std": b["delta_std"],
                "dbpc_wins": b["wins"],
                "dbpc_losses": b["losses"],
                "dtok_mean": c["delta_mean"],
                "dtok_std": c["delta_std"],
                "dtok_wins": c["wins"],
                "dtok_losses": c["losses"],
            }
        )
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(combined[0].keys()))
        writer.writeheader()
        writer.writerows(combined)
    print(f"\nwrote {args.out} ({len(combined)} groups)")


if __name__ == "__main__":
    main()
