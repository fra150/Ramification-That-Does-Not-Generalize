"""Aggregate all result CSVs into one summary, with deltas versus flat BPE.

Handles both schemas:
- v0.1: decode_exact (word-level)
- v0.2: decode_word + decode_doc

Reports mean and sample standard deviation across seeds, so the 3-seed matrix
can be read without pretending that a single run is a result.

Usage:
    python -m experiments.aggregate
    python -m experiments.aggregate --pattern "results/matrix/**/results.csv"
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


def _mean(values):
    vals = [v for v in values if v is not None]
    return statistics.mean(vals) if vals else float("nan")


def _std(values):
    vals = [v for v in values if v is not None]
    return statistics.stdev(vals) if len(vals) > 1 else 0.0


def _min(values):
    vals = [v for v in values if v is not None]
    return min(vals) if vals else float("nan")


def load_rows(pattern):
    rows = []
    for path in sorted(glob.glob(pattern, recursive=True)):
        with open(path, encoding="utf-8") as fh:
            for row in csv.DictReader(fh):
                row["_source"] = path
                rows.append(row)
    return rows


def _decode_word(row):
    if "decode_word" in row:
        return _float(row.get("decode_word"))
    return _float(row.get("decode_exact"))


def summarize(rows):
    groups = defaultdict(list)
    for r in rows:
        key = (r["corpus"], r["method"], int(r["vocab_target"]))
        groups[key].append(r)

    summary = []
    for (corpus, method, vocab), rs in sorted(groups.items()):
        summary.append(
            {
                "corpus": corpus,
                "method": method,
                "vocab_target": vocab,
                "n": len(rs),
                "tokens_per_char": _mean([_float(r.get("tokens_per_char")) for r in rs]),
                "tokens_per_char_std": _std([_float(r.get("tokens_per_char")) for r in rs]),
                "bits_per_char": _mean([_float(r.get("bits_per_char")) for r in rs]),
                "bits_per_char_std": _std([_float(r.get("bits_per_char")) for r in rs]),
                "boundary_f1": _mean([_float(r.get("boundary_f1")) for r in rs]),
                "boundary_f1_std": _std([_float(r.get("boundary_f1")) for r in rs]),
                "decode_word": _mean([_decode_word(r) for r in rs]),
                "decode_doc_min": _min([_float(r.get("decode_doc")) for r in rs]),
                "train_seconds": _mean([_float(r.get("train_seconds")) for r in rs]),
            }
        )

    # Deltas versus the flat baseline at the same corpus/vocab.
    flat = {
        (r["corpus"], r["vocab_target"]): r
        for r in summary
        if r["method"] == "flat"
    }
    for r in summary:
        base = flat.get((r["corpus"], r["vocab_target"]))
        if base is None or r["method"] == "flat":
            r["df1_vs_flat"] = 0.0 if r["method"] == "flat" else float("nan")
            r["dbpc_vs_flat"] = 0.0 if r["method"] == "flat" else float("nan")
        else:
            r["df1_vs_flat"] = r["boundary_f1"] - base["boundary_f1"]
            r["dbpc_vs_flat"] = r["bits_per_char"] - base["bits_per_char"]
    return summary


def print_table(summary):
    header = (
        f"{'corpus':<9} {'method':<11} {'vocab':>5} {'n':>2} "
        f"{'tok/ch':>7} {'bpc':>7} {'bF1':>6} {'dF1':>7} {'dBPC':>7} "
        f"{'dW':>5} {'dD':>4} {'t(s)':>6}"
    )
    print(header)
    print("-" * len(header))
    for r in summary:
        bf1 = "-" if math.isnan(r["boundary_f1"]) else f"{r['boundary_f1']:.3f}"
        df1 = "-" if math.isnan(r["df1_vs_flat"]) else f"{r['df1_vs_flat']:+.3f}"
        dbpc = "-" if math.isnan(r["dbpc_vs_flat"]) else f"{r['dbpc_vs_flat']:+.3f}"
        dD = r["decode_doc_min"]
        dD_s = "-" if math.isnan(dD) else f"{dD:.0f}"
        print(
            f"{r['corpus']:<9} {r['method']:<11} {r['vocab_target']:>5} {r['n']:>2} "
            f"{r['tokens_per_char']:>7.3f} {r['bits_per_char']:>7.3f} {bf1:>6} "
            f"{df1:>7} {dbpc:>7} {r['decode_word']:>5.3f} {dD_s:>4} {r['train_seconds']:>6.1f}"
        )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pattern", default="results/**/results.csv")
    ap.add_argument("--out", default="results/summary.csv")
    args = ap.parse_args()

    rows = load_rows(args.pattern)
    if not rows:
        print(f"no results found for pattern: {args.pattern}")
        return
    summary = summarize(rows)
    print_table(summary)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(summary[0].keys()))
        writer.writeheader()
        writer.writerows(summary)
    print(f"\nwrote {args.out} ({len(summary)} groups from {len(rows)} runs)")


if __name__ == "__main__":
    main()
