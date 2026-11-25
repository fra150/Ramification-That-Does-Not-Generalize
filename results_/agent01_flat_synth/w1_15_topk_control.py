"""W1-15 follow-up: is the held-out-provenance gain in level assignment, or
just the result of dependency closure shrinking level 1 to the confident
intra-morpheme rules?

For each seed, read the actual level-1 size of each held-out morph variant
from the main run, then build a full-provenance morph assignment truncated to
that same level-1 size. If F1 matches the held-out variant, the held-out
estimate adds nothing beyond a more conservative level-1 size; if it does not,
the held-out ranking itself is better.
"""

from __future__ import annotations

import argparse
import csv
import os
import time
from collections import Counter

from ramified.bpe import train_tokenizer
from ramified.codec import normalize
from ramified.levels import assign_levels, collect_provenance, level_summary
from ramified.synthetic import generate_synthetic

from results.redteam.w1_15_levels_weighting import (
    agg, bpc_counts, f1_words, jaccard, level1_set,
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="42,43,44,45,46")
    ap.add_argument("--vocab", type=int, default=512)
    ap.add_argument("--fraction", type=float, default=0.6)
    ap.add_argument("--runs", default="results/redteam/w1_15_levels_weighting_runs.csv")
    ap.add_argument("--out", default="results/redteam/w1_15_topk_control.csv")
    args = ap.parse_args()

    seeds = [int(x) for x in args.seeds.split(",") if x.strip()]
    with open(args.runs, encoding="utf-8") as fh:
        runs = list(csv.DictReader(fh))

    def get(seed, variant, mode="morph", eval_set="test_all"):
        for r in runs:
            if (int(r["seed"]) == seed and r["variant"] == variant
                    and r["mode"] == mode and r["eval_set"] == eval_set):
                return r
        return None

    out = []
    for seed in seeds:
        t0 = time.perf_counter()
        syn = generate_synthetic(seed=seed)
        flat = train_tokenizer(
            [syn.train_text], vocab_size=args.vocab, n_levels=1, split="chunks"
        )
        words = list(syn.train_word_objects)
        train_counts = Counter(syn.train_text.split())
        test_counts = Counter(syn.test_text.split())
        char_len = len(normalize(syn.test_text))
        weights = [train_counts.get(w.word, 0) for w in words]
        wc = [collect_provenance(flat, [w]) for w in words]
        stats_freq = agg(wc, list(range(len(words))), weights)

        freq_tok = get(seed, "rank", "frequency")
        freq_f1 = float(freq_tok["boundary_f1"])
        freq_bpc = float(freq_tok["bits_per_char"])

        targets = {}
        for name in ("freq_halfA", "freq_halfB", "unique_halfA", "unique_halfB",
                     "root_cv"):
            r = get(seed, name, "morph", "test_all")
            if r is not None:
                targets.setdefault(int(r["n_l1"]), []).append(name)

        for k in sorted(targets):
            frac = min(0.99, (k + 0.1) / len(flat.levels[0]))
            tok = assign_levels(
                flat, stats_freq, mode="morph",
                level1_fraction=frac,
                seed=seed * 1000 + args.vocab,
            )
            f1, n = f1_words(tok, syn.test_words)
            bpc = bpc_counts(tok, train_counts, test_counts, char_len)
            summary = level_summary(tok, stats_freq)

            rand_tok = assign_levels(
                flat, stats_freq, mode="random_closed",
                level1_fraction=frac,
                seed=seed * 1000 + args.vocab,
            )
            rand_f1, _ = f1_words(rand_tok, syn.test_words)
            rand_bpc = bpc_counts(rand_tok, train_counts, test_counts, char_len)
            rand_summary = level_summary(rand_tok, stats_freq)

            out.append({
                "seed": seed,
                "target_k": k,
                "actual_n_l1": len(tok.levels[0]),
                "variants_at_k": ",".join(targets[k]),
                "boundary_f1": f1,
                "frequency_f1": freq_f1,
                "delta_vs_frequency": f1 - freq_f1,
                "bits_per_char": bpc,
                "frequency_bpc": freq_bpc,
                "delta_bpc_vs_frequency": bpc - freq_bpc,
                "l1_intra": summary[0]["intra"],
                "l1_cross": summary[0]["cross"],
                "l1_unseen": summary[0]["unseen"],
                "random_closed_f1": rand_f1,
                "random_closed_bpc": rand_bpc,
                "delta_random_vs_frequency": rand_f1 - freq_f1,
                "delta_morph_vs_random": f1 - rand_f1,
                "random_n_l1": len(rand_tok.levels[0]),
                "random_l1_intra": rand_summary[0]["intra"],
                "random_l1_cross": rand_summary[0]["cross"],
                "random_l1_unseen": rand_summary[0]["unseen"],
                "jaccard_vs_freq_all_morph": jaccard(
                    level1_set(tok),
                    level1_set(assign_levels(
                        flat, stats_freq, mode="morph",
                        level1_fraction=args.fraction,
                        seed=seed * 1000 + args.vocab,
                    )),
                ),
            })
            print(
                f"seed={seed} k={k} actual={len(tok.levels[0])} "
                f"F1={f1:.4f} (flat {freq_f1:.4f}, "
                f"delta {f1 - freq_f1:+.4f}) bpc={bpc:.4f} "
                f"rand F1={rand_f1:.4f} (delta {rand_f1 - freq_f1:+.4f}) "
                f"L1 intra/cross/unseen="
                f"{summary[0]['intra']}/{summary[0]['cross']}/{summary[0]['unseen']} "
                f"for {targets[k]}",
                flush=True,
            )

        # Report the held-out variants next to the truncated full-provenance
        # control, so the report can compare them directly.
        for k in sorted(targets):
            control = next(r for r in out if r["seed"] == seed and r["target_k"] == k)
            for name in targets[k]:
                r = get(seed, name, "morph", "test_all")
                if r is None:
                    continue
                print(
                    f"  compare seed={seed} {name}: held-out F1="
                    f"{float(r['boundary_f1']):.4f} vs full-top{k}="
                    f"{control['boundary_f1']:.4f} "
                    f"(heldout-fulltop={float(r['boundary_f1']) - control['boundary_f1']:+.4f})",
                    flush=True,
                )
        print(f"seed={seed} done in {time.perf_counter() - t0:.1f}s", flush=True)

    keys = list(out[0].keys())
    with open(args.out, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        w.writerows(out)
    print(f"\nwrote {args.out} ({len(out)} rows)")


if __name__ == "__main__":
    main()
