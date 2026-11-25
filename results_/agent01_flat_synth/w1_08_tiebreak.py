"""W1-08: is the v512 morph level-assignment result stable to BPE heap ties?

The source files are NOT edited. The fixed ``train_bpe`` from ramified/bpe.py is
copied here, with one change: the heap key's secondary component. The copy is
installed over ``ramified.bpe.train_bpe`` for the duration of the run, so
``train_tokenizer`` uses it.

Tie-break variants (all deterministic given the seed):
  id        (-count, left, right)                -- current behaviour
  rev_id    (-count, -left, -right)              -- reverse pair id
  exp_short (-count, len(exp(a))+len(exp(b)), a, b)
  exp_long  (-count, -(len(exp(a))+len(exp(b))), a, b)
  randNNN   (-count, random priority per pair)   -- NN N = tie-break seed

For every variant we rerun flat / frequency / morph / random_closed at v512 on
the same synthetic corpus seeds, and report paired morph - frequency and
morph - random_closed deltas. Tie steps are counted exactly: a step is tied
when at least two distinct pairs share the winning count in the valid state.
"""

from __future__ import annotations

import argparse
import csv
import heapq
import os
import random
import statistics
import time
from collections import Counter, defaultdict

from ramified import bpe
from ramified.bpe import train_tokenizer
from ramified.levels import assign_levels, collect_provenance
from ramified.synthetic import generate_synthetic

from experiments.level_assignment import evaluate, summary_fields
from experiments.analyze_levels import describe

RESULTS_DIR = os.path.dirname(os.path.abspath(__file__))


def _update_pair_counts(seq, delta, pair_counts, pair_words, wi, count_pairs):
    """Exact copy of ramified.bpe._update_pair_counts plus count_pairs upkeep.

    count_pairs[c] = number of distinct pairs whose current count is c. It is
    only used to count tie steps; it does not affect the merge choice.
    """
    touched = []
    for i in range(len(seq) - 1):
        p = (seq[i], seq[i + 1])
        old = pair_counts.get(p, 0)
        new_count = old + delta
        if new_count <= 0:
            if old > 0:
                count_pairs[old] -= 1
                if count_pairs[old] <= 0:
                    del count_pairs[old]
            pair_counts.pop(p, None)
            pair_words.pop(p, None)
            continue
        pair_counts[p] = new_count
        if old > 0:
            count_pairs[old] -= 1
            if count_pairs[old] <= 0:
                del count_pairs[old]
        count_pairs[new_count] += 1
        if delta > 0:
            pair_words[p].add(wi)
        else:
            bucket = pair_words.get(p)
            if bucket is not None:
                bucket.discard(wi)
        touched.append((p, new_count))
    return touched


def train_bpe_variant(
    sequences,
    freqs,
    n_merges,
    next_id,
    level,
    expansions,
    variant="id",
    stats=None,
):
    """Greedy BPE identical to the fixed source, except the heap tie-break."""
    words = [list(s) for s in sequences]
    pair_counts = Counter()
    pair_words = defaultdict(set)
    count_pairs = Counter()

    for wi, seq in enumerate(words):
        f = freqs[wi]
        for i in range(len(seq) - 1):
            p = (seq[i], seq[i + 1])
            pair_counts[p] += f
            pair_words[p].add(wi)
    for p, c in pair_counts.items():
        if c > 0:
            count_pairs[c] += 1

    rng = random.Random(f"w1-08-{variant}")
    rand_key: dict[tuple[int, int], float] = {}

    def heap_key(p, c):
        a, b = p
        if variant == "id":
            return (-c, a, b)
        if variant == "rev_id":
            return (-c, -a, -b)
        if variant == "exp_short":
            return (-c, len(expansions[a]) + len(expansions[b]), a, b)
        if variant == "exp_long":
            return (-c, -(len(expansions[a]) + len(expansions[b])), a, b)
        if variant.startswith("rand"):
            k = rand_key.get(p)
            if k is None:
                k = rng.random()
                rand_key[p] = k
            return (-c, k, a, b)
        raise ValueError(f"unknown variant: {variant}")

    heap = [(heap_key(p, c), p) for p, c in pair_counts.items() if c > 0]
    heapq.heapify(heap)

    merges = []
    tie_steps = 0
    merges_done = 0
    for step in range(n_merges):
        pair = None
        count = None
        tied = False
        while heap:
            key, p = heapq.heappop(heap)
            c = -key[0]
            if pair_counts.get(p, 0) == c and c > 0:
                pair = p
                count = c
                tied = count_pairs.get(c, 0) > 1
                break
        if pair is None:
            break

        merges_done += 1
        if tied:
            tie_steps += 1

        a, b = pair
        new_id = next_id
        next_id += 1
        rule = bpe.MergeRule(a, b, new_id, step, level, count)
        merges.append(rule)
        expansions[new_id] = expansions[a] + expansions[b]

        for wi in list(pair_words.get(pair, ())):
            seq = words[wi]
            if not any(
                seq[i] == a and seq[i + 1] == b for i in range(len(seq) - 1)
            ):
                pair_words[pair].discard(wi)
                continue

            f = freqs[wi]

            touched = _update_pair_counts(
                seq, -f, pair_counts, pair_words, wi, count_pairs
            )
            for p2, c2 in touched:
                heapq.heappush(heap, (heap_key(p2, c2), p2))

            new_seq = []
            i = 0
            while i < len(seq):
                if i < len(seq) - 1 and seq[i] == a and seq[i + 1] == b:
                    new_seq.append(new_id)
                    i += 2
                else:
                    new_seq.append(seq[i])
                    i += 1

            words[wi] = new_seq
            touched = _update_pair_counts(
                new_seq, +f, pair_counts, pair_words, wi, count_pairs
            )
            for p2, c2 in touched:
                heapq.heappush(heap, (heap_key(p2, c2), p2))

    if stats is not None:
        stats["merges"] = merges_done
        stats["tie_steps"] = tie_steps
        stats["final_pairs"] = len(pair_counts)

    return merges, words


def install_variant(variant):
    def patched(sequences, freqs, n_merges, next_id, level, expansions):
        # Per-training stats land in a dict keyed by the caller's own record.
        patched.last_stats = {}
        return train_bpe_variant(
            sequences,
            freqs,
            n_merges,
            next_id,
            level,
            expansions,
            variant=variant,
            stats=patched.last_stats,
        )

    bpe.train_bpe = patched
    return patched


def rule_key(rule):
    return (rule.left, rule.right)


def run_seed(task):
    """Train every variant for one seed. Runs in a worker process."""
    seed, variants, vocab, fraction = task
    syn = generate_synthetic(seed=seed)
    runs = []
    ties = []
    rules_out = []
    flat_by_variant = {}

    for variant in variants:
        t0 = time.perf_counter()
        patched = install_variant(variant)
        flat = train_tokenizer(
            [syn.train_text], vocab_size=vocab, n_levels=1, split="chunks"
        )
        train_seconds = time.perf_counter() - t0
        tie = dict(patched.last_stats)
        tie.update({"seed": seed, "variant": variant, "vocab": vocab})
        ties.append(tie)
        flat_by_variant[variant] = flat

        stats = collect_provenance(flat, syn.train_word_objects, text=syn.train_text)
        base = evaluate(flat, syn)
        runs.append(
            {
                "seed": seed,
                "vocab": vocab,
                "variant": variant,
                "mode": "flat",
                "train_seconds": round(train_seconds, 2),
                **base,
                **summary_fields(flat, stats),
            }
        )
        print(
            f"seed={seed} v={vocab} {variant:<9} flat      "
            f"F1={base['boundary_f1']:.4f} bpc={base['bits_per_char']:.4f} "
            f"ties={tie['tie_steps']}/{tie['merges']} "
            f"rules={len(flat.levels[0])} t={train_seconds:.1f}s",
            flush=True,
        )

        for mode in ("frequency", "morph", "random_closed"):
            row_t = time.perf_counter()
            tok = assign_levels(
                flat,
                stats,
                mode=mode,
                level1_fraction=fraction,
                seed=seed * 1000 + vocab,
            )
            metrics = evaluate(tok, syn)
            runs.append(
                {
                    "seed": seed,
                    "vocab": vocab,
                    "variant": variant,
                    "mode": mode,
                    "train_seconds": round(time.perf_counter() - row_t, 2),
                    **metrics,
                    **summary_fields(tok, stats),
                }
            )
            print(
                f"seed={seed} v={vocab} {variant:<9} {mode:<13} "
                f"F1={metrics['boundary_f1']:.4f} "
                f"bpc={metrics['bits_per_char']:.4f} "
                f"L1={len(tok.levels[0])} L2={len(tok.levels[1])}",
                flush=True,
            )

    # Rule-set drift versus the current tie-break, same seed.
    ref = flat_by_variant["id"]
    ref_rules = [rule_key(m) for m in ref.levels[0]]
    ref_set = set(ref_rules)
    for variant in variants:
        tok = flat_by_variant[variant]
        rules = [rule_key(m) for m in tok.levels[0]]
        first_diff = None
        for i, (x, y) in enumerate(zip(ref_rules, rules)):
            if x != y:
                first_diff = i
                break
        if first_diff is None and len(ref_rules) != len(rules):
            first_diff = min(len(ref_rules), len(rules))
        common = len(ref_set & set(rules))
        rules_out.append(
            {
                "seed": seed,
                "vocab": vocab,
                "variant": variant,
                "n_rules": len(rules),
                "common_with_id": common,
                "jaccard": common / max(1, len(ref_set | set(rules))),
                "first_diff_rank": "" if first_diff is None else first_diff,
            }
        )
    return {"runs": runs, "ties": ties, "rules": rules_out}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="42,43,44,45,46")
    ap.add_argument("--vocab", type=int, default=512)
    ap.add_argument("--variants", default="id,rev_id,exp_short,exp_long,rand101,rand202,rand303")
    ap.add_argument("--fraction", type=float, default=0.6)
    ap.add_argument("--jobs", type=int, default=1)
    ap.add_argument("--tag", default="")
    args = ap.parse_args()

    seeds = [int(x) for x in args.seeds.split(",") if x.strip()]
    variants = [x.strip() for x in args.variants.split(",") if x.strip()]
    vocab = args.vocab
    suffix = f"_{args.tag}" if args.tag else ""

    tasks = [(seed, variants, vocab, args.fraction) for seed in seeds]
    t_start = time.perf_counter()
    if args.jobs > 1 and len(tasks) > 1:
        from concurrent.futures import ProcessPoolExecutor

        with ProcessPoolExecutor(max_workers=min(args.jobs, len(tasks))) as ex:
            results = list(ex.map(run_seed, tasks))
    else:
        results = [run_seed(task) for task in tasks]

    runs = [row for result in results for row in result["runs"]]
    tie_rows = [row for result in results for row in result["ties"]]
    rule_rows = [row for result in results for row in result["rules"]]

    # ---- pairwise deltas within (variant, seed) ----
    idx = {(r["seed"], r["variant"], r["mode"]): r for r in runs}
    pairwise = []
    for seed in seeds:
        for variant in variants:
            f = idx.get((seed, variant, "frequency"))
            m = idx.get((seed, variant, "morph"))
            rc = idx.get((seed, variant, "random_closed"))
            if not f or not m or not rc:
                continue
            for metric in ("boundary_f1", "bits_per_char", "tokens_per_char"):
                pairwise.append(
                    {
                        "seed": seed,
                        "vocab": vocab,
                        "variant": variant,
                        "metric": metric,
                        "morph_minus_frequency": float(m[metric]) - float(f[metric]),
                        "morph_minus_random": float(m[metric]) - float(rc[metric]),
                        "random_minus_frequency": float(rc[metric]) - float(f[metric]),
                        "morph": float(m[metric]),
                        "frequency": float(f[metric]),
                        "random_closed": float(rc[metric]),
                    }
                )

    # ---- per-variant paired statistics ----
    summary = []
    for variant in variants:
        for metric in ("boundary_f1", "bits_per_char"):
            for name, key in (
                ("morph-frequency", "morph_minus_frequency"),
                ("morph-random_closed", "morph_minus_random"),
                ("random_closed-frequency", "random_minus_frequency"),
            ):
                deltas = [
                    row[key]
                    for row in pairwise
                    if row["variant"] == variant and row["metric"] == metric
                ]
                st = describe(deltas)
                summary.append(
                    {
                        "variant": variant,
                        "metric": metric,
                        "comparison": name,
                        **st,
                    }
                )

    # ---- tie-break noise of each metric and of the paired contrast ----
    noise_rows = []
    for metric in ("boundary_f1", "bits_per_char", "tokens_per_char"):
        for quantity in ("morph", "frequency", "random_closed",
                         "morph_minus_frequency", "morph_minus_random",
                         "random_minus_frequency"):
            per_seed_sd = []
            per_seed_mean = []
            for seed in seeds:
                vals = [
                    row[quantity]
                    for row in pairwise
                    if row["seed"] == seed and row["metric"] == metric
                ]
                if len(vals) > 1:
                    per_seed_sd.append(statistics.stdev(vals))
                per_seed_mean.append(statistics.mean(vals))
            means_by_variant = []
            for variant in variants:
                vals = [
                    row[quantity]
                    for row in pairwise
                    if row["variant"] == variant and row["metric"] == metric
                ]
                means_by_variant.append(statistics.mean(vals))
            noise_rows.append(
                {
                    "metric": metric,
                    "quantity": quantity,
                    "mean_over_variants": statistics.mean(means_by_variant),
                    "sd_of_variant_means": statistics.stdev(means_by_variant)
                    if len(means_by_variant) > 1
                    else 0.0,
                    "mean_within_seed_sd_across_variants": statistics.mean(per_seed_sd)
                    if per_seed_sd
                    else 0.0,
                    "max_within_seed_sd_across_variants": max(per_seed_sd)
                    if per_seed_sd
                    else 0.0,
                }
            )

    # ---- write ----
    os.makedirs(RESULTS_DIR, exist_ok=True)

    def write_csv(name, rows):
        if not rows:
            return
        path = os.path.join(RESULTS_DIR, name)
        fields = []
        for row in rows:
            for k in row:
                if k not in fields:
                    fields.append(k)
        with open(path, "w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
        print(f"wrote {path} ({len(rows)} rows)")

    write_csv(f"w1_08_runs{suffix}.csv", runs)
    write_csv(f"w1_08_pairwise{suffix}.csv", pairwise)
    write_csv(f"w1_08_summary{suffix}.csv", summary)
    write_csv(f"w1_08_ties{suffix}.csv", tie_rows)
    write_csv(f"w1_08_rules{suffix}.csv", rule_rows)

    # ---- console report ----
    print("\n=== paired deltas by variant (5 seeds) ===")
    for variant in variants:
        print(f"\n{variant}")
        for metric in ("boundary_f1", "bits_per_char"):
            for name in ("morph-frequency", "morph-random_closed",
                         "random_closed-frequency"):
                st = next(
                    s for s in summary
                    if s["variant"] == variant
                    and s["metric"] == metric
                    and s["comparison"] == name
                )
                print(
                    f"  {metric:<14} {name:<25} "
                    f"mean={st['mean']:+.5f} sd={st['std']:.5f} "
                    f"t={st['t']:+.2f} "
                    f"CI=[{st['ci_low']:+.5f},{st['ci_high']:+.5f}] "
                    f"W/L/T={st['wins']}/{st['losses']}/{st['ties']}"
                )

    print("\n=== tie steps in BPE training ===")
    for row in tie_rows:
        print(
            f"  seed={row['seed']} {row['variant']:<9} "
            f"ties={row['tie_steps']:>3}/{row['merges']:<3} "
            f"({100.0 * row['tie_steps'] / max(1, row['merges']):.1f}%)"
        )

    print("\n=== tie-break noise (SD across variants) ===")
    for row in noise_rows:
        if row["metric"] in ("boundary_f1", "bits_per_char"):
            print(
                f"  {row['metric']:<14} {row['quantity']:<22} "
                f"mean={row['mean_over_variants']:+.5f} "
                f"sd_variant_means={row['sd_of_variant_means']:.5f} "
                f"mean_within_seed_sd={row['mean_within_seed_sd_across_variants']:.5f}"
            )

    print(f"\ntotal wall time {time.perf_counter() - t_start:.1f}s")


if __name__ == "__main__":
    main()
