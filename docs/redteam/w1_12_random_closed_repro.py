"""Reproduce the published random_closed size shortfall on real tokenizers.

For each synthetic seed, train the flat v512 tokenizer, replicate the
topological-order construction in ramified.levels.assign_levels, and show the
duplicate order entry that makes level 1 smaller than the requested n1.

Run from the repository root:

    python -u docs/redteam/w1_12_random_closed_repro.py --seeds 44,46,50
"""

from __future__ import annotations

import argparse
import csv
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from ramified.bpe import train_tokenizer  # noqa: E402
from ramified.levels import assign_levels  # noqa: E402
from ramified.synthetic import generate_synthetic  # noqa: E402


def reproduce(seed: int, vocab: int, fraction: float):
    syn = generate_synthetic(seed=seed)
    tok = train_tokenizer(
        [syn.train_text], vocab_size=vocab, n_levels=1, split="chunks"
    )
    rules = sorted(tok.levels[0], key=lambda m: m.rank)
    terminals = set(tok.char_to_id.values()) | {tok.unk_id, tok.eow_id}
    creator = {r.new: r for r in rules}
    deps = {r.new: set() for r in rules}
    dependents = defaultdict(list)
    for rule in rules:
        for symbol in (rule.left, rule.right):
            if symbol in terminals:
                continue
            dep = creator.get(symbol)
            if dep is not None:
                deps[rule.new].add(dep.new)
                dependents[dep.new].append(rule.new)

    n1 = max(1, min(len(rules) - 1, int(fraction * len(rules))))
    rng = random.Random(seed * 1000 + vocab)
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
                available.append(creator[child])
    order.extend(m for m in rules if m.new not in emitted)

    prefix = [m.new for m in order[:n1]]
    dups = [(k, v) for k, v in Counter(m.new for m in order).items() if v > 1]
    repeated = [
        (r.rank, tok.expansions[r.left], tok.expansions[r.right], tok.expansions[r.new])
        for r in rules
        if r.left == r.right and r.left not in terminals
    ]
    out = assign_levels(
        tok, {}, mode="random_closed", level1_fraction=fraction,
        seed=seed * 1000 + vocab,
    )
    return {
        "seed": seed,
        "rules": len(rules),
        "n1": n1,
        "order_len": len(order),
        "prefix_distinct": len(set(prefix)),
        "shortfall": n1 - len(set(prefix)),
        "dups": [(tok.expansions[k], v) for k, v in dups],
        "repeated": [
            (rank, f"{left}+{right}->{new}") for rank, left, right, new in repeated
        ],
        "assign_l1": len(out.levels[0]),
        "assign_l2": len(out.levels[1]),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="44,46,50")
    ap.add_argument("--vocab", type=int, default=512)
    ap.add_argument("--fraction", type=float, default=0.6)
    args = ap.parse_args()

    seeds = [int(x) for x in args.seeds.split(",") if x.strip()]
    published = {}
    path = ROOT / "results" / "level_assignment_v512_10seeds.csv"
    if path.exists() and args.vocab == 512:
        with open(path, encoding="utf-8") as fh:
            for row in csv.DictReader(fh):
                if row["mode"] == "random_closed":
                    published[int(row["seed"])] = int(row["n_l1"])

    for seed in seeds:
        row = reproduce(seed, args.vocab, args.fraction)
        csv_l1 = published.get(seed)
        print(
            f"seed={row['seed']} rules={row['rules']} n1={row['n1']} "
            f"order_len={row['order_len']} prefix_distinct={row['prefix_distinct']} "
            f"shortfall={row['shortfall']} dups={row['dups']} "
            f"repeated={row['repeated']} assign_L1={row['assign_l1']} "
            f"L2={row['assign_l2']} csv_L1={csv_l1}"
        )


if __name__ == "__main__":
    main()
