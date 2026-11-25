"""W1-14: distribution of the random_closed control at v512.

For every corpus seed we train one flat BPE at v512, then evaluate:
  - flat
  - morph (deterministic assignment)
  - N independent random_closed topological orders
  - the canonical random_closed draw used in
    results/level_assignment_v512_10seeds.csv (seed*1000+vocab)

No source files are modified. Output is a long CSV with one row per
(seed, mode/draw).

Usage:
    python results/redteam/w1_14_run_random_closed.py --seeds 42,43 --draws 2
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import math
import os
import statistics
import time
from collections import Counter

from experiments.level_assignment import bpc_alpha, evaluate
from ramified.bpe import train_tokenizer
from ramified.codec import normalize, roundtrip_ok
from ramified.levels import assign_levels, collect_provenance, level_summary
from ramified.metrics import decode_exact, mean_boundary_f1
from ramified.synthetic import generate_synthetic


def cached_ids(text, tok):
    """Exact tokenization of ``text``: encode every distinct chunk once.

    tokenize_chunks normalises, splits on spaces, and encodes each chunk with
    EOW. Chunk tokenizations are independent, so weighting by chunk count
    gives the identical id multiset (and therefore identical bpc).
    """
    counts = Counter(normalize(text).split(" "))
    ids = []
    for chunk, n in counts.items():
        ids.extend(tok.encode(chunk, eow=True) * n)
    return ids


def evaluate_fast(tok, syn):
    """Same metrics as experiments.level_assignment.evaluate, cached chunks."""
    train_ids = cached_ids(syn.train_text, tok)
    test_ids = cached_ids(syn.test_text, tok)
    norm_len = len(normalize(syn.test_text))
    precision, recall, f1 = mean_boundary_f1(tok, syn.test_words)
    return {
        "n_levels": len(tok.levels),
        "n_l1": len(tok.levels[0]) if tok.levels else 0,
        "n_l2": len(tok.levels[1]) if len(tok.levels) > 1 else 0,
        "tokens_per_char": len(test_ids) / max(1, norm_len),
        "bits_per_char": bpc_alpha(train_ids, test_ids, tok.vocab_size, norm_len),
        "boundary_precision": precision,
        "boundary_recall": recall,
        "boundary_f1": f1,
        "decode_word": decode_exact(tok, syn.test_words),
        "decode_doc": 1.0 if roundtrip_ok(syn.test_text, tok) else 0.0,
    }


def level1_ids(tok):
    return tuple(sorted(m.new for m in tok.levels[0])) if tok.levels else ()


def fingerprint(ids):
    return hashlib.sha1(",".join(str(i) for i in ids).encode()).hexdigest()[:12]


def jaccard(a, b):
    sa, sb = set(a), set(b)
    if not sa and not sb:
        return 1.0
    return len(sa & sb) / len(sa | sb)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="42,43,44,45,46,47,48,49,50,51")
    ap.add_argument("--vocab", type=int, default=512)
    ap.add_argument("--draws", type=int, default=10)
    ap.add_argument(
        "--out", default="results/redteam/w1_14_random_closed_draws.csv"
    )
    ap.add_argument(
        "--validate-original",
        action="store_true",
        help="on the first seed, compare evaluate_fast with the original evaluate",
    )
    args = ap.parse_args()

    seeds = [int(x) for x in args.seeds.split(",") if x.strip()]
    vocab = args.vocab
    rows = []

    def add_row(seed, tag, draw, rng_seed, tok, metrics, stats, morph_ids, freq_ids):
        ids = level1_ids(tok)
        summary = level_summary(tok, stats)
        row = {
            "seed": seed,
            "vocab": vocab,
            "tag": tag,
            "draw": draw,
            "rng_seed": "" if rng_seed is None else rng_seed,
            "n_l1": len(tok.levels[0]) if tok.levels else 0,
            "n_l2": len(tok.levels[1]) if len(tok.levels) > 1 else 0,
            "level1_fingerprint": fingerprint(ids),
            "jaccard_morph": round(jaccard(ids, morph_ids), 4),
            "jaccard_frequency": round(jaccard(ids, freq_ids), 4),
            **metrics,
        }
        for row_summary in summary:
            prefix = f"l{row_summary['level'] + 1}"
            row[f"{prefix}_intra"] = row_summary["intra"]
            row[f"{prefix}_cross"] = row_summary["cross"]
            row[f"{prefix}_unseen"] = row_summary["unseen"]
        rows.append(row)

    for seed in seeds:
        t_seed = time.perf_counter()
        syn = generate_synthetic(seed=seed)
        t0 = time.perf_counter()
        flat = train_tokenizer(
            [syn.train_text], vocab_size=vocab, n_levels=1, split="chunks"
        )
        train_seconds = time.perf_counter() - t0
        stats = collect_provenance(flat, syn.train_word_objects, text=syn.train_text)

        # Deterministic level-1 sets for reference.
        morph_tok = assign_levels(
            flat, stats, mode="morph", level1_fraction=0.6, seed=seed * 1000 + vocab
        )
        freq_tok = assign_levels(
            flat, stats, mode="frequency", level1_fraction=0.6,
            seed=seed * 1000 + vocab,
        )
        morph_ids = level1_ids(morph_tok)
        freq_ids = level1_ids(freq_tok)

        t0 = time.perf_counter()
        flat_metrics = evaluate_fast(flat, syn)
        flat_eval_seconds = time.perf_counter() - t0
        if args.validate_original and seed == seeds[0]:
            slow = evaluate(flat, syn)
            diffs = {
                k: abs(slow[k] - flat_metrics[k])
                for k in ("tokens_per_char", "bits_per_char", "boundary_f1")
            }
            print(f"  validation evaluate vs evaluate_fast: {diffs}", flush=True)
            assert max(diffs.values()) < 1e-12, diffs
        add_row(seed, "flat", -2, None, flat, flat_metrics, stats, morph_ids, freq_ids)

        t0 = time.perf_counter()
        morph_metrics = evaluate_fast(morph_tok, syn)
        morph_eval_seconds = time.perf_counter() - t0
        if args.validate_original and seed == seeds[0]:
            slow = evaluate(morph_tok, syn)
            diffs = {
                k: abs(slow[k] - morph_metrics[k])
                for k in ("tokens_per_char", "bits_per_char", "boundary_f1")
            }
            print(f"  validation evaluate vs evaluate_fast (morph): {diffs}",
                  flush=True)
            assert max(diffs.values()) < 1e-12, diffs
        add_row(seed, "morph", -1, None, morph_tok, morph_metrics, stats,
                morph_ids, freq_ids)

        print(
            f"seed={seed} flat tok/ch={flat_metrics['tokens_per_char']:.4f} "
            f"bpc={flat_metrics['bits_per_char']:.4f} "
            f"F1={flat_metrics['boundary_f1']:.4f} | "
            f"morph bpc={morph_metrics['bits_per_char']:.4f} "
            f"F1={morph_metrics['boundary_f1']:.4f} | "
            f"train={train_seconds:.1f}s flat_eval={flat_eval_seconds:.1f}s "
            f"morph_eval={morph_eval_seconds:.1f}s",
            flush=True,
        )

        # Canonical draw first, then independent draws. The canonical draw
        # must reproduce the published random_closed row for this seed.
        draw_specs = [("random_closed_canonical", -1, seed * 1000 + vocab)]
        for k in range(args.draws):
            # Well-separated RNG seeds; no relation to the canonical one.
            rng_seed = seed * 100_000 + 1_729 + 7_919 * k
            draw_specs.append((f"random_closed_d{k:02d}", k, rng_seed))

        for tag, draw, rng_seed in draw_specs:
            t0 = time.perf_counter()
            tok = assign_levels(
                flat,
                stats,
                mode="random_closed",
                level1_fraction=0.6,
                seed=rng_seed,
            )
            level_seconds = time.perf_counter() - t0
            t0 = time.perf_counter()
            metrics = evaluate_fast(tok, syn)
            eval_seconds = time.perf_counter() - t0
            add_row(seed, tag, draw, rng_seed, tok, metrics, stats,
                    morph_ids, freq_ids)
            rows[-1]["level_seconds"] = round(level_seconds, 3)
            rows[-1]["eval_seconds"] = round(eval_seconds, 2)
            if draw >= 0:
                print(
                    f"  draw={draw:02d} nL1={len(tok.levels[0])} "
                    f"J(morph)={jaccard(level1_ids(tok), morph_ids):.3f} "
                    f"bpc={metrics['bits_per_char']:.4f} "
                    f"F1={metrics['boundary_f1']:.4f} "
                    f"t={eval_seconds:.1f}s",
                    flush=True,
                )
            else:
                print(
                    f"  canonical nL1={len(tok.levels[0])} "
                    f"bpc={metrics['bits_per_char']:.4f} "
                    f"F1={metrics['boundary_f1']:.4f}",
                    flush=True,
                )

        print(
            f"seed={seed} done in {time.perf_counter() - t_seed:.1f}s",
            flush=True,
        )

    fieldnames = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nwrote {args.out} ({len(rows)} runs)")

    # Quick within-seed spread printout.
    print("\nwithin-seed random_closed spread")
    for seed in seeds:
        f1s = [r["boundary_f1"] for r in rows
               if r["seed"] == seed and r["tag"].startswith("random_closed_d")]
        bpcs = [r["bits_per_char"] for r in rows
                if r["seed"] == seed and r["tag"].startswith("random_closed_d")]
        if len(f1s) > 1:
            print(
                f"seed={seed} F1 mean={statistics.mean(f1s):.4f} "
                f"sd={statistics.stdev(f1s):.4f} "
                f"range=[{min(f1s):.4f},{max(f1s):.4f}] | "
                f"bpc mean={statistics.mean(bpcs):.4f} "
                f"sd={statistics.stdev(bpcs):.4f} "
                f"range=[{min(bpcs):.4f},{max(bpcs):.4f}]"
            )


if __name__ == "__main__":
    main()
