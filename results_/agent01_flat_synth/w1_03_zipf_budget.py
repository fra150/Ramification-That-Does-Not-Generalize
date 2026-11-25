"""W1-03 follow-up: Zipf with a 1M-token budget (coverage-matched to uniform).

The main 2x2 ablation confounds Zipfian skew with type coverage: at 120k
tokens the Zipf corpus observes 81% of the training inventory, while the
uniform corpus observes 100%. This runner keeps the Zipf(1.1) law and the same
held-out words, but samples 1M training tokens, which covers ~99.9% of the
inventory. If the morph effect persists here, skew is the driver; if it
vanishes, type coverage is.

Boundary F1 and bpc are evaluated on the 120k baseline's test text and test
words, so the comparison against the main baseline rows is paired by seed.
"""

from __future__ import annotations

import csv
import math
import os
import statistics
import sys

import numpy as np
from scipy import stats as st

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "results", "redteam"))

import w1_03_synthetic_sampling as w
from ramified.levels import assign_levels
from ramified.synthetic import generate_synthetic

N_TOKENS = 1_000_000


def paired(rows, a, b, metric):
    by_seed = {}
    for row in rows:
        by_seed.setdefault(row["seed"], {})[row["mode"]] = row[metric]
    deltas = [v[a] - v[b] for v in by_seed.values() if a in v and b in v]
    n = len(deltas)
    mean = statistics.mean(deltas)
    sd = statistics.stdev(deltas) if n > 1 else 0.0
    se = sd / math.sqrt(n) if n else float("nan")
    t = mean / se if se else float("nan")
    crit = st.t.ppf(0.975, df=n - 1) if n > 1 else float("nan")
    if metric == "bits_per_char":
        wins = sum(1 for d in deltas if d < 0)
        losses = sum(1 for d in deltas if d > 0)
    else:
        wins = sum(1 for d in deltas if d > 0)
        losses = sum(1 for d in deltas if d < 0)
    return {
        "metric": metric,
        "mode_a": a,
        "mode_b": b,
        "n": n,
        "mean": mean,
        "std": sd,
        "se": se,
        "t": t,
        "p": 2 * st.t.sf(abs(t), df=n - 1) if n > 1 else float("nan"),
        "ci_low": mean - crit * se if n > 1 else float("nan"),
        "ci_high": mean + crit * se if n > 1 else float("nan"),
        "wins": wins,
        "losses": losses,
        "ties": n - wins - losses,
    }


def main():
    seeds = [42, 43, 44, 45, 46]
    rows = []
    corpus_rows = []
    for seed in seeds:
        syn120 = generate_synthetic(seed=seed)
        syn1m = generate_synthetic(seed=seed, n_train_tokens=N_TOKENS)
        stats_text = w.corpus_stats(syn1m.train_text, syn1m.train_words)
        corpus_rows.append({"seed": seed, **stats_text})
        print(
            f"seed={seed} zipf1m words={stats_text['train_word_tokens']} "
            f"types={stats_text['train_types']} coverage={stats_text['type_coverage']:.4f} "
            f"top1={stats_text['top1_share']:.3f}",
            flush=True,
        )

        flat = w.train_tokenizer_fast(
            [syn1m.train_text], vocab_size=512, n_levels=1, split="chunks"
        )
        stats = w.collect_provenance_cached(
            flat, syn1m.train_word_objects, text=syn1m.train_text
        )
        for mode in ("flat", "frequency", "morph", "random_closed"):
            if mode == "flat":
                tok = flat
            else:
                tok = assign_levels(
                    flat, stats, mode=mode, level1_fraction=0.6,
                    seed=seed * 1000 + 512,
                )
            metrics = w.evaluate(
                tok, syn1m.train_text, syn120.test_text, syn120.test_words
            )
            row = {
                "seed": seed,
                "condition": "zipf1m",
                "vocab": 512,
                "mode": mode,
                "train_seconds": 0.0,
                "n_merges": len(flat.levels[0]),
                **metrics,
                **w.summary_fields(tok, stats),
                **stats_text,
            }
            rows.append(row)
            print(
                f"  seed={seed} zipf1m {mode:<12} "
                f"tok/ch={metrics['tokens_per_char']:.4f} "
                f"bpc={metrics['bits_per_char']:.4f} "
                f"F1={metrics['boundary_f1']:.4f} L1={len(tok.levels[0])}",
                flush=True,
            )

    stat_rows = []
    for metric in ("boundary_f1", "bits_per_char"):
        for a, b in (
            ("morph", "frequency"),
            ("morph", "random_closed"),
            ("random_closed", "frequency"),
        ):
            stat_rows.append(paired(rows, a, b, metric))

    # Effect shift vs the 120k baseline rows from the main run.
    main_path = os.path.join(ROOT, "results", "redteam", "w1_03_runs.csv")
    if os.path.exists(main_path):
        with open(main_path, newline="", encoding="utf-8") as fh:
            main_rows = [
                r for r in csv.DictReader(fh) if r["condition"] == "baseline"
            ]
        for r in main_rows:
            r["seed"] = int(r["seed"])
            for m in ("boundary_f1", "bits_per_char"):
                r[m] = float(r[m])
        for metric in ("boundary_f1", "bits_per_char"):
            high = {}
            for row in rows:
                high.setdefault(row["seed"], {})[row["mode"]] = row[metric]
            base = {}
            for row in main_rows:
                base.setdefault(row["seed"], {})[row["mode"]] = row[metric]
            common = sorted(set(high) & set(base))
            if not common:
                continue
            deltas = [
                (high[s]["morph"] - high[s]["frequency"])
                - (base[s]["morph"] - base[s]["frequency"])
                for s in common
            ]
            n = len(deltas)
            mean = statistics.mean(deltas)
            sd = statistics.stdev(deltas) if n > 1 else 0.0
            se = sd / math.sqrt(n) if n else float("nan")
            t = mean / se if se else float("nan")
            crit = st.t.ppf(0.975, df=n - 1) if n > 1 else float("nan")
            stat_rows.append(
                {
                    "metric": metric,
                    "mode_a": "zipf1m_minus_baseline120k",
                    "mode_b": "morph-frequency",
                    "n": n,
                    "mean": mean,
                    "std": sd,
                    "se": se,
                    "t": t,
                    "p": 2 * st.t.sf(abs(t), df=n - 1) if n > 1 else float("nan"),
                    "ci_low": mean - crit * se if n > 1 else float("nan"),
                    "ci_high": mean + crit * se if n > 1 else float("nan"),
                    "wins": sum(1 for d in deltas if d > 0),
                    "losses": sum(1 for d in deltas if d < 0),
                    "ties": sum(1 for d in deltas if d == 0),
                }
            )

    out_dir = os.path.join(ROOT, "results", "redteam")
    for name, items in (
        ("w1_03_zipf1m_runs.csv", rows),
        ("w1_03_zipf1m_stats.csv", stat_rows),
        ("w1_03_zipf1m_corpus.csv", corpus_rows),
    ):
        path = os.path.join(out_dir, name)
        fieldnames = []
        for item in items:
            for key in item:
                if key not in fieldnames:
                    fieldnames.append(key)
        with open(path, "w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(items)
        print(f"wrote {path}")


if __name__ == "__main__":
    main()
