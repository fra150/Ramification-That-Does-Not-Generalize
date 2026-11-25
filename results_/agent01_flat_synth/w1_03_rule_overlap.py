"""W1-03 follow-up: do morph and frequency pick the same level-1 rules?

Retrains baseline and uniform seed 42, assigns frequency and morph levels, and
reports the Jaccard overlap of the level-1 rule sets plus the level-2 intra
count. Writes results/redteam/w1_03_rule_overlap.csv.
"""

from __future__ import annotations

import csv
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "results", "redteam"))

import w1_03_synthetic_sampling as w
from ramified.levels import assign_levels, level_summary


def rule_set(tok, level):
    return {(r.left, r.right) for r in tok.levels[level]}


def main():
    syn, texts = w.make_corpora(42)
    rows = []
    for condition in ("baseline", "uniform", "noperiod", "uniform_noperiod"):
        flat = w.train_tokenizer_fast(
            [texts[condition]], vocab_size=512, n_levels=1, split="chunks"
        )
        stats = w.collect_provenance_cached(
            flat, syn.train_word_objects, text=texts[condition]
        )
        freq = assign_levels(flat, stats, mode="frequency", seed=42512)
        morph = assign_levels(flat, stats, mode="morph", seed=42512)
        fset = rule_set(freq, 0)
        mset = rule_set(morph, 0)
        inter = len(fset & mset)
        union = len(fset | mset)
        freq_summary = {r["level"]: r for r in level_summary(freq, stats)}
        morph_summary = {r["level"]: r for r in level_summary(morph, stats)}
        rows.append(
            {
                "seed": 42,
                "condition": condition,
                "l1_frequency": len(fset),
                "l1_morph": len(mset),
                "l1_intersection": inter,
                "l1_union": union,
                "l1_jaccard": inter / union if union else float("nan"),
                "l1_only_frequency": len(fset - mset),
                "l1_only_morph": len(mset - fset),
                "l2_intra_frequency": freq_summary[1]["intra"],
                "l2_cross_frequency": freq_summary[1]["cross"],
                "l2_intra_morph": morph_summary[1]["intra"],
                "l2_cross_morph": morph_summary[1]["cross"],
            }
        )
        print(rows[-1], flush=True)

    out = os.path.join(ROOT, "results", "redteam", "w1_03_rule_overlap.csv")
    with open(out, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
