"""W1-14 sensitivity: random_closed without the duplicate-emission bug.

ramified.levels.assign_levels(mode="random_closed") builds ``dependents`` as a
list and can append the same child twice when a rule depends twice on the same
creator (left == right nonterminal). The duplicate then occupies two slots of
the topological prefix, so level 1 can have 292-294 unique rules instead of the
requested 294.

This script reimplements the same sampler with a deduplicated dependent set,
producing exactly n1 unique rules, and evaluates a few seeds. It does not
modify source files.

Usage:
    python -m results.redteam.w1_14_corrected_sampler --seeds 42,43,44 --draws 5
"""

from __future__ import annotations

import argparse
import csv
import os
import random
import statistics
from collections import defaultdict

from ramified.bpe import Tokenizer, train_tokenizer
from ramified.levels import assign_levels, collect_provenance
from ramified.synthetic import generate_synthetic

from results.redteam.w1_14_run_random_closed import evaluate_fast


def _graph(tok):
    rules = sorted(tok.levels[0], key=lambda m: m.rank)
    creator = {rule.new: rule for rule in rules}
    terminals = set(tok.char_to_id.values()) | {tok.unk_id, tok.eow_id}
    deps = {rule.new: set() for rule in rules}
    dependents = defaultdict(set)
    duplicate_edges = 0
    for rule in rules:
        seen = set()
        for symbol in (rule.left, rule.right):
            if symbol in terminals:
                continue
            dependency = creator.get(symbol)
            if dependency is None:
                continue
            if dependency.new in seen:
                duplicate_edges += 1
            seen.add(dependency.new)
            deps[rule.new].add(dependency.new)
            dependents[dependency.new].add(rule.new)
    return rules, deps, dependents, duplicate_edges


def assign_random_closed_fixed(tok, seed, level1_fraction=0.6):
    if len(tok.levels) != 1:
        raise ValueError("expects a flat tokenizer")
    rules, deps, dependents, _ = _graph(tok)
    n1 = int(level1_fraction * len(rules))
    n1 = max(1, min(len(rules) - 1, n1))
    rule_by_new = {m.new: m for m in rules}

    rng = random.Random(seed)
    available = [m for m in rules if not deps[m.new]]
    rng.shuffle(available)
    emitted = set()
    order = []
    while available:
        idx = rng.randrange(len(available))
        rule = available.pop(idx)
        order.append(rule)
        emitted.add(rule.new)
        for child in dependents[rule.new]:
            if child not in emitted and deps[child].issubset(emitted):
                available.append(rule_by_new[child])
    unique = {m.new for m in order}
    assert len(order) == len(rules), (len(order), len(rules))
    assert len(unique) == len(rules), (len(unique), len(rules))
    level1 = {m.new for m in order[:n1]}
    l1 = [m for m in rules if m.new in level1]
    l2 = [m for m in rules if m.new not in level1]
    assert len(l1) == n1, (len(l1), n1)
    out = Tokenizer(
        expansions=dict(tok.expansions),
        char_to_id=dict(tok.char_to_id),
        eow_id=tok.eow_id,
        levels=[l1, l2],
        unk_id=tok.unk_id,
    )
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="42,43,44")
    ap.add_argument("--vocab", type=int, default=512)
    ap.add_argument("--draws", type=int, default=5)
    ap.add_argument(
        "--out", default="results/redteam/w1_14_corrected_sampler.csv"
    )
    args = ap.parse_args()
    seeds = [int(x) for x in args.seeds.split(",") if x.strip()]
    rows = []

    for seed in seeds:
        syn = generate_synthetic(seed=seed)
        flat = train_tokenizer(
            [syn.train_text], vocab_size=args.vocab, n_levels=1, split="chunks"
        )
        stats = collect_provenance(flat, syn.train_word_objects, text=syn.train_text)
        target_n1 = int(0.6 * len(flat.levels[0]))
        _, _, _, duplicate_edges = _graph(flat)
        morph = assign_levels(
            flat, stats, mode="morph", level1_fraction=0.6,
            seed=seed * 1000 + args.vocab,
        )
        flat_m = evaluate_fast(flat, syn)
        morph_m = evaluate_fast(morph, syn)
        print(
            f"seed={seed} flat F1={flat_m['boundary_f1']:.4f} "
            f"bpc={flat_m['bits_per_char']:.4f} | morph "
            f"F1={morph_m['boundary_f1']:.4f} bpc={morph_m['bits_per_char']:.4f} "
            f"| target_n1={target_n1} duplicate_dep_edges={duplicate_edges}",
            flush=True,
        )
        f1s, bpcs = [], []
        for k in range(args.draws):
            rng_seed = seed * 100_000 + 50_000 + 7_919 * k
            tok = assign_random_closed_fixed(flat, rng_seed, 0.6)
            assert len(tok.levels[0]) == target_n1, len(tok.levels[0])
            m = evaluate_fast(tok, syn)
            f1s.append(m["boundary_f1"])
            bpcs.append(m["bits_per_char"])
            rows.append(
                {
                    "seed": seed,
                    "vocab": args.vocab,
                    "draw": k,
                    "rng_seed": rng_seed,
                    "n_l1": len(tok.levels[0]),
                    "duplicate_dep_edges": duplicate_edges,
                    "boundary_f1": m["boundary_f1"],
                    "bits_per_char": m["bits_per_char"],
                }
            )
            print(
                f"  d{k}: nL1={len(tok.levels[0])} "
                f"F1={m['boundary_f1']:.4f} bpc={m['bits_per_char']:.4f}",
                flush=True,
            )
        rows.append(
            {
                "seed": seed,
                "vocab": args.vocab,
                "draw": "summary",
                "n_l1": target_n1,
                "duplicate_dep_edges": duplicate_edges,
                "boundary_f1": statistics.mean(f1s),
                "bits_per_char": statistics.mean(bpcs),
                "f1_sd": statistics.stdev(f1s),
                "bpc_sd": statistics.stdev(bpcs),
                "morph_f1": morph_m["boundary_f1"],
                "morph_bpc": morph_m["bits_per_char"],
                "delta_f1": morph_m["boundary_f1"] - statistics.mean(f1s),
                "delta_bpc": morph_m["bits_per_char"] - statistics.mean(bpcs),
            }
        )
        print(
            f"seed={seed} corrected-random F1={statistics.mean(f1s):.4f}"
            f"+-{statistics.stdev(f1s):.4f} bpc={statistics.mean(bpcs):.4f}"
            f"+-{statistics.stdev(bpcs):.4f} deltaF1="
            f"{morph_m['boundary_f1']-statistics.mean(f1s):+.4f} "
            f"deltaBPC={morph_m['bits_per_char']-statistics.mean(bpcs):+.4f}",
            flush=True,
        )

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    fieldnames = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with open(args.out, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {args.out} ({len(rows)} rows)")


if __name__ == "__main__":
    main()
