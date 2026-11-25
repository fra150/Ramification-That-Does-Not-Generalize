"""Does morphological level assignment change anything?

Holds the flat BPE merge rules and vocabulary fixed, reassigns the rules to two
levels (morph / random / reverse), and measures boundary F1, bits per char and
exact decode on the synthetic held-out words.

The flat tokenizer is the control. If ``morph`` beats both ``flat`` and
``random``, the level assignment carries morphological information. If
``reverse`` does just as well, any reordering is enough. If all are equal, the
ramification is decorative on this corpus.

Usage:
    python -m experiments.level_assignment
    python -m experiments.level_assignment --vocabs 512,1024 --seeds 42,43,44
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import random
import time
from collections import Counter
from dataclasses import replace

from ramified.bpe import train_tokenizer
from ramified.codec import normalize, roundtrip_ok, tokenize_chunks
from ramified.levels import assign_levels, collect_provenance, level_summary
from ramified.metrics import boundary_report, decode_exact
from ramified.synthetic import generate_synthetic


def bpc_alpha(train_ids, test_ids, vocab_size, char_len, alpha=0.1):
    if char_len == 0:
        return float("nan")
    counts = Counter(train_ids)
    total = sum(counts.values())
    denom = total + alpha * vocab_size
    bits = 0.0
    for t in test_ids:
        bits -= math.log2((counts.get(t, 0) + alpha) / denom)
    return bits / char_len


def control_words(words, kind, seed):
    """Replace the true boundaries with a control boundary source.

    ``const2`` uses the fixed slot {2, L-2} that reproduced the old morph
    assignment on the fixed-length generator. ``random2`` draws two uniform
    cuts. ``near2`` perturbs each true boundary by -1/0/+1 and rejects the
    no-perturbation draw, so it is a near-true positional control. The metric
    is always evaluated against the true boundaries; only provenance changes.
    """
    rng = random.Random(seed)
    out = []
    for w in words:
        length = len(w.word)
        true = tuple(w.boundaries)
        if kind == "const2":
            cuts = (2, length - 2)
        elif kind == "random2":
            cuts = tuple(sorted(rng.sample(range(1, length), 2)))
        elif kind == "near2":
            while True:
                delta = (rng.choice((-1, 0, 1)), rng.choice((-1, 0, 1)))
                if delta == (0, 0):
                    continue
                b1 = min(max(1, true[0] + delta[0]), length - 2)
                b2 = min(max(b1 + 1, true[1] + delta[1]), length - 1)
                if b1 < b2:
                    cuts = (b1, b2)
                    break
        else:
            raise ValueError(f"unknown boundary control: {kind}")
        out.append(replace(w, boundaries=cuts))
    return out


def evaluate(tok, syn):
    train_ids = tokenize_chunks(syn.train_text, tok)
    test_ids = tokenize_chunks(syn.test_text, tok)
    norm_len = len(normalize(syn.test_text))
    report = boundary_report(tok, syn.test_words)
    return {
        "n_levels": len(tok.levels),
        "n_l1": len(tok.levels[0]) if tok.levels else 0,
        "n_l2": len(tok.levels[1]) if len(tok.levels) > 1 else 0,
        "tokens_per_char": len(test_ids) / max(1, norm_len),
        "bits_per_char": bpc_alpha(train_ids, test_ids, tok.vocab_size, norm_len),
        **report,
        # Backward-compatible alias: the macro F1 used in earlier tables.
        "boundary_f1": report["boundary_macro_f1"],
        "decode_word": decode_exact(tok, syn.test_words),
        "decode_doc": 1.0 if roundtrip_ok(syn.test_text, tok) else 0.0,
    }


def summary_fields(tok, stats):
    summary = level_summary(tok, stats)
    out = {}
    for row in summary:
        prefix = f"l{row['level'] + 1}"
        out[f"{prefix}_intra"] = row["intra"]
        out[f"{prefix}_cross"] = row["cross"]
        out[f"{prefix}_unseen"] = row["unseen"]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="42,43,44")
    ap.add_argument("--vocabs", default="512,1024,2048")
    ap.add_argument("--modes", default="frequency,morph,random_closed,reverse")
    ap.add_argument("--fraction", type=float, default=0.6)
    ap.add_argument("--holdout-mode", choices=["triple", "pair"], default="triple")
    ap.add_argument("--pair-holdout", type=float, default=0.10)
    ap.add_argument("--morphology", choices=["fixed", "variable"], default="fixed")
    ap.add_argument(
        "--boundary-controls",
        default="",
        help="comma-separated control boundary sources: const2,random2,near2",
    )
    ap.add_argument("--out", default="results/level_assignment.csv")
    args = ap.parse_args()

    seeds = [int(x) for x in args.seeds.split(",") if x.strip()]
    vocabs = [int(x) for x in args.vocabs.split(",") if x.strip()]
    modes = [x.strip() for x in args.modes.split(",") if x.strip()]
    controls = [x.strip() for x in args.boundary_controls.split(",") if x.strip()]
    control_offsets = {"const2": 11, "random2": 23, "near2": 37}

    rows = []
    for seed in seeds:
        syn = generate_synthetic(
            seed=seed,
            holdout_mode=args.holdout_mode,
            pair_holdout=args.pair_holdout,
            morphology=args.morphology,
        )
        print(
            f"seed={seed} morphology={args.morphology} split={args.holdout_mode} "
            f"{syn.split_report}",
            flush=True,
        )
        for vocab in vocabs:
            t0 = time.perf_counter()
            flat = train_tokenizer(
                [syn.train_text], vocab_size=vocab, n_levels=1, split="chunks"
            )
            flat_seconds = time.perf_counter() - t0
            stats = collect_provenance(
                flat, syn.train_word_objects, text=syn.train_text
            )

            metrics = evaluate(flat, syn)
            row = {
                "seed": seed,
                "vocab": vocab,
                "mode": "flat",
                "morphology": args.morphology,
                "holdout_mode": args.holdout_mode,
                "boundary_source": "none",
                "level_seconds": round(flat_seconds, 2),
                **metrics,
                **summary_fields(flat, stats),
            }
            rows.append(row)
            print(
                f"seed={seed} v={vocab} flat      "
                f"tok/ch={metrics['tokens_per_char']:.3f} "
                f"bpc={metrics['bits_per_char']:.3f} "
                f"F1m={metrics['boundary_macro_f1']:.3f} "
                f"F1u={metrics['boundary_micro_f1']:.3f} "
                f"L1={len(flat.levels[0])} t={flat_seconds:.1f}s",
                flush=True,
            )

            for mode in modes:
                t0 = time.perf_counter()
                tok = assign_levels(
                    flat,
                    stats,
                    mode=mode,
                    level1_fraction=args.fraction,
                    seed=seed * 1000 + vocab,
                )
                level_seconds = time.perf_counter() - t0
                metrics = evaluate(tok, syn)
                row = {
                    "seed": seed,
                    "vocab": vocab,
                    "mode": mode,
                    "morphology": args.morphology,
                    "holdout_mode": args.holdout_mode,
                    "boundary_source": "true" if mode == "morph" else "none",
                    "level_seconds": round(level_seconds, 2),
                    **metrics,
                    **summary_fields(tok, stats),
                }
                rows.append(row)
                print(
                    f"seed={seed} v={vocab} {mode:<9} "
                    f"tok/ch={metrics['tokens_per_char']:.3f} "
                    f"bpc={metrics['bits_per_char']:.3f} "
                    f"F1m={metrics['boundary_macro_f1']:.3f} "
                    f"F1u={metrics['boundary_micro_f1']:.3f} "
                    f"L1={len(tok.levels[0])} L2={len(tok.levels[1])} "
                    f"t={level_seconds:.1f}s",
                    flush=True,
                )

            for control in controls:
                ctrl = control_words(
                    syn.train_word_objects,
                    control,
                    seed=seed * 1_000_003 + vocab * 101 + control_offsets[control],
                )
                ctrl_stats = collect_provenance(flat, ctrl, text=syn.train_text)
                t0 = time.perf_counter()
                tok = assign_levels(
                    flat,
                    ctrl_stats,
                    mode="morph",
                    level1_fraction=args.fraction,
                    seed=seed * 1000 + vocab,
                )
                level_seconds = time.perf_counter() - t0
                metrics = evaluate(tok, syn)
                row = {
                    "seed": seed,
                    "vocab": vocab,
                    "mode": f"morph_{control}",
                    "morphology": args.morphology,
                    "holdout_mode": args.holdout_mode,
                    "boundary_source": control,
                    "level_seconds": round(level_seconds, 2),
                    **metrics,
                    **summary_fields(tok, ctrl_stats),
                }
                rows.append(row)
                print(
                    f"seed={seed} v={vocab} morph_{control:<6} "
                    f"tok/ch={metrics['tokens_per_char']:.3f} "
                    f"bpc={metrics['bits_per_char']:.3f} "
                    f"F1m={metrics['boundary_macro_f1']:.3f} "
                    f"F1u={metrics['boundary_micro_f1']:.3f} "
                    f"L1={len(tok.levels[0])} L2={len(tok.levels[1])} "
                    f"t={level_seconds:.1f}s",
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


if __name__ == "__main__":
    main()
