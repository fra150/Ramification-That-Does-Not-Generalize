"""W1-12 red-team audit: dependency closure in ramified.levels.assign_levels.

Read-only with respect to the package.  Builds flat tokenizers, reassigns their
rules to two levels, and checks four claims:

1. no level-1 rule depends on a level-2 rule (directly or transitively);
2. the demotion pass equals the brute-force greatest closed subset of the
   initial level-1 seed;
3. every rule lands in exactly one level (no silent drops);
4. the reported level sizes equal the rules actually installed, and the
   requested size / demotion count are recoverable.

Run from the repository root:

    python docs/redteam/w1_12_audit.py
"""

from __future__ import annotations

import csv
import random
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from ramified.bpe import EOW, MergeRule, Tokenizer, train_tokenizer  # noqa: E402
from ramified.levels import (  # noqa: E402
    _creator_map,
    _score,
    assign_levels,
    collect_provenance,
    level_summary,
)
from ramified.metrics import decode_exact, mean_boundary_f1  # noqa: E402
from ramified.synthetic import generate_synthetic  # noqa: E402


# ---------------------------------------------------------------------------
# generic graph helpers
# ---------------------------------------------------------------------------


def flat_rules(tok: Tokenizer) -> list[MergeRule]:
    if len(tok.levels) != 1:
        raise ValueError("expected a flat tokenizer")
    return sorted(tok.levels[0], key=lambda m: m.rank)


def own_creator_map(tok: Tokenizer):
    creator = {}
    for level in tok.levels:
        for rule in level:
            creator[rule.new] = rule
    terminals = set(tok.char_to_id.values()) | {tok.unk_id, tok.eow_id}
    return creator, terminals


def dep_map(tok: Tokenizer):
    """new id -> set of new ids that must be created before it can fire."""
    creator, terminals = own_creator_map(tok)
    deps = {}
    unresolved = {}
    for level in tok.levels:
        for rule in level:
            ds = set()
            for symbol in (rule.left, rule.right):
                if symbol in terminals:
                    continue
                dep = creator.get(symbol)
                if dep is None:
                    unresolved.setdefault(rule.new, []).append(symbol)
                else:
                    ds.add(dep.new)
            deps[rule.new] = ds
    return deps, unresolved


def requested_n1(n_rules: int, fraction: float) -> int:
    if n_rules == 0:
        return 0
    n1 = int(fraction * n_rules)
    return max(1, min(n_rules - 1, n1))


def brute_closure(seed, deps):
    """Greatest subset of `seed` closed under deps (the demotion fixpoint)."""
    keep = set(seed)
    changed = True
    while changed:
        changed = False
        for rid in sorted(keep):
            if not deps[rid].issubset(keep):
                keep.remove(rid)
                changed = True
    return keep


def seed_level1(tok: Tokenizer, stats, mode: str, fraction: float, seed: int):
    """Replicate the initial level-1 selection before closure for morph/reverse."""
    rules = flat_rules(tok)
    n1 = requested_n1(len(rules), fraction)
    if mode == "frequency":
        return {m.new for m in rules[:n1]}, n1
    if mode in ("morph", "reverse"):
        rng = random.Random(seed)
        scored = [(m, _score(m, stats, mode, rng)) for m in rules]
        order = sorted(scored, key=lambda pair: (-pair[1], pair[0].rank))
        return {m.new for m, _ in order[:n1]}, n1
    raise ValueError(f"no seed replication for {mode}")


def closure_violations(tok: Tokenizer):
    deps, unresolved = dep_map(tok)
    level1 = {r.new for r in tok.levels[0]}
    bad = []
    for rid in sorted(level1):
        outside = deps.get(rid, set()) - level1
        if outside:
            bad.append((rid, sorted(outside)))
    return bad, unresolved


def rank_order_check(tok: Tokenizer):
    """The single-pass demotion is correct only if deps have lower rank."""
    creator, _ = own_creator_map(tok)
    rules = [r for level in tok.levels for r in level]
    bad = []
    for rule in rules:
        for symbol in (rule.left, rule.right):
            dep = creator.get(symbol)
            if dep is not None and dep.rank >= rule.rank:
                bad.append((rule.rank, dep.rank, rule.new, dep.new))
    ranks = [r.rank for r in rules]
    return {
        "rank_order_bad": len(bad),
        "examples": bad[:5],
        "unique_ranks": len(set(ranks)) == len(ranks),
        "rank_min_max": (min(ranks), max(ranks)) if ranks else None,
    }


def partition_check(flat: Tokenizer, assigned: Tokenizer):
    flat_ids = {r.new for r in flat.levels[0]}
    l1 = {r.new for r in assigned.levels[0]}
    l2 = {r.new for r in assigned.levels[1]}
    return {
        "union_ok": (l1 | l2) == flat_ids,
        "disjoint": not (l1 & l2),
        "count_ok": len(l1) + len(l2) == len(flat_ids),
        "missing": sorted(flat_ids - (l1 | l2))[:5],
        "extra": sorted((l1 | l2) - flat_ids)[:5],
        "n_flat": len(flat_ids),
        "n_l1": len(l1),
        "n_l2": len(l2),
    }


def rule_level_of(tok: Tokenizer):
    return {r.new: i for i, level in enumerate(tok.levels) for r in level}


def fired_by_level(tok: Tokenizer, words):
    used = set()
    for word in words:
        stack = list(tok.encode_nodes(word.word, eow=True))
        while stack:
            node = stack.pop()
            if node.children:
                used.add(node.symbol)
                stack.extend(node.children)
    level_of = rule_level_of(tok)
    l1 = sum(1 for s in used if level_of.get(s) == 0)
    l2 = sum(1 for s in used if level_of.get(s) == 1)
    return l1, l2, len(used)


def summary_counts(tok: Tokenizer, stats):
    out = []
    for level in tok.levels:
        strict_intra = tie = strict_cross = unseen = 0
        for rule in level:
            a, b = stats.get((rule.left, rule.right), (0, 0))
            if a + b == 0:
                unseen += 1
            elif a > b:
                strict_intra += 1
            elif a == b:
                tie += 1
            else:
                strict_cross += 1
        out.append((len(level), strict_intra, tie, strict_cross, unseen))
    return out


def summarize_assignment(flat, stats, mode, fraction, seed, words=None):
    tok = assign_levels(flat, stats, mode=mode, level1_fraction=fraction, seed=seed)
    part = partition_check(flat, tok)
    bad, unresolved = closure_violations(tok)
    n1 = requested_n1(len(flat_rules(flat)), fraction) if flat.levels[0] else 0
    row = {
        "mode": mode,
        "n1_requested": n1,
        "n_l1": part["n_l1"],
        "n_l2": part["n_l2"],
        "demoted_or_short": n1 - part["n_l1"],
        "union_ok": part["union_ok"],
        "disjoint": part["disjoint"],
        "count_ok": part["count_ok"],
        "closure_violations": len(bad),
        "unresolved_symbols": sum(len(v) for v in unresolved.values()),
        "summary": summary_counts(tok, stats),
        "level_summary": level_summary(tok, stats),
        "vocab_ok": tok.vocab_size == flat.vocab_size,
    }
    if mode in ("morph", "reverse"):
        seed_set, _ = seed_level1(flat, stats, mode, fraction, seed)
        expected = brute_closure(seed_set, dep_map(flat)[0])
        actual = {r.new for r in tok.levels[0]}
        row["brute_match"] = expected == actual
        row["brute_size"] = len(expected)
        row["seed_size"] = len(seed_set)
    if words is not None:
        row["decode_word"] = decode_exact(tok, words)
        row["fired_l1"], row["fired_l2"], row["fired_total"] = fired_by_level(
            tok, words
        )
        flat_ids = [flat.encode(w.word, eow=True) for w in words]
        tok_ids = [tok.encode(w.word, eow=True) for w in words]
        row["words_tokenized_differently"] = sum(
            1 for a, b in zip(flat_ids, tok_ids) if a != b
        )
        row["n_words"] = len(words)
    return tok, row


def print_assignment_row(row):
    print(
        f"  {row['mode']:<13} n1={row['n1_requested']:>4} "
        f"L1={row['n_l1']:>4} L2={row['n_l2']:>4} "
        f"demote/short={row['demoted_or_short']:>4} "
        f"closure_bad={row['closure_violations']} "
        f"partition={'ok' if row['union_ok'] and row['disjoint'] and row['count_ok'] else 'FAIL'} "
        f"vocab={'ok' if row['vocab_ok'] else 'FAIL'}"
        + (f" brute={'ok' if row['brute_match'] else 'FAIL'}" if "brute_match" in row else "")
    )
    for i, (n, si, tie, sc, un) in enumerate(row["summary"]):
        print(
            f"    L{i + 1}: size={n} strict_intra={si} tie={tie} "
            f"strict_cross={sc} unseen={un}"
        )
    if "decode_word" in row:
        print(
            f"    decode_word={row['decode_word']:.3f} "
            f"fired L1/L2={row['fired_l1']}/{row['fired_l2']} "
            f"of {row['fired_total']} rules used; "
            f"tokenization differs on {row['words_tokenized_differently']}/{row['n_words']} words"
        )


# ---------------------------------------------------------------------------
# 1. v512 synthetic, seed 42, all four modes
# ---------------------------------------------------------------------------


def audit_v512_seed42():
    print("\n=== 1. v512 synthetic, seed 42 ===")
    syn = generate_synthetic(seed=42)
    flat = train_tokenizer([syn.train_text], vocab_size=512, n_levels=1, split="chunks")
    stats = collect_provenance(flat, syn.train_word_objects, text=syn.train_text)
    rules = flat_rules(flat)
    print(f"flat: chars={len(flat.char_to_id)} rules={len(rules)} vocab={flat.vocab_size}")
    rank_check = rank_order_check(flat)
    print(
        f"rank order: violations={rank_check['rank_order_bad']} "
        f"unique_ranks={rank_check['unique_ranks']} "
        f"range={rank_check['rank_min_max']} examples={rank_check['examples']}"
    )
    priv_creator, priv_term = _creator_map(flat)
    own_creator, own_term = own_creator_map(flat)
    print(
        f"creator/terminal helpers agree: "
        f"{set(priv_creator) == set(own_creator) and priv_term == own_term}"
    )
    print(f"requested n1 at fraction 0.6: {requested_n1(len(rules), 0.6)}")

    # Provenance independently recomputed from expansion strings.
    indep = independent_provenance(flat, syn.train_word_objects, syn.train_text)
    diff = {
        k: (stats.get(k), indep.get(k))
        for k in set(stats) | set(indep)
        if stats.get(k) != indep.get(k)
    }
    print(
        f"provenance: pairs={len(stats)} independent pairs={len(indep)} "
        f"mismatches={len(diff)}"
    )
    if diff:
        for k, v in list(diff.items())[:5]:
            print(f"  MISMATCH {k}: collect={v[0]} independent={v[1]}")

    rows = []
    # experiments.level_assignment uses seed * 1000 + vocab for assign_levels.
    assign_seed = 42 * 1000 + 512
    for mode in ("frequency", "morph", "random_closed", "reverse"):
        tok, row = summarize_assignment(
            flat, stats, mode, 0.6, assign_seed, words=syn.test_words
        )
        rows.append(row)
        print_assignment_row(row)
        summary = row["level_summary"]
        sums_ok = all(
            s["rules"] == row["n_l1"] if s["level"] == 0 else s["rules"] == row["n_l2"]
            for s in summary
        )
        print(f"    level_summary sizes match installed levels: {sums_ok}")
        train_l1, train_l2, train_total = fired_by_level(
            tok, syn.train_word_objects[:2000]
        )
        print(
            f"    first 2000 train words: rules fired L1/L2={train_l1}/{train_l2} "
            f"of {train_total}; L1 rules never fired on this sample="
            f"{row['n_l1'] - train_l1}"
        )
    return flat, stats, rows


def independent_provenance(tok, word_objects, text):
    """Recompute provenance from decoded symbol spans, not from child offsets."""
    by_word = {w.word: w for w in word_objects}
    items = [by_word[t] for t in text.split() if t in by_word]
    counts = defaultdict(lambda: [0, 0])

    def walk(node, start):
        expansion = tok.expansions[node.symbol]
        if not node.children:
            if expansion == EOW:
                return start
            return start + len(expansion)
        end = start
        for child in node.children:
            end = walk(child, end)
        if len(node.children) == 2:
            key = (node.children[0].symbol, node.children[1].symbol)
            stripped = expansion[: -len(EOW)] if expansion.endswith(EOW) else expansion
            span_end = start + len(stripped)
            crosses = any(start < b < span_end for b in word.boundaries)
            if crosses:
                counts[key][1] += 1
            else:
                counts[key][0] += 1
        return end

    for word in items:
        pos = 0
        for node in tok.encode_nodes(word.word, eow=True):
            pos = walk(node, pos)
    return {k: tuple(v) for k, v in counts.items()}


# ---------------------------------------------------------------------------
# 2. all ten published v512 seeds: does morph ever demote?
# ---------------------------------------------------------------------------


def audit_v512_all_seeds():
    print("\n=== 2. v512 synthetic, seeds 42-51: morph closure ===")
    published = load_csv(ROOT / "results" / "level_assignment_v512_10seeds.csv")
    pub = {(int(r["seed"]), r["mode"]): r for r in published}
    print("seed rules n1 L1 L2 demoted closure_bad brute L1_intra L1_cross L1_unseen")
    for seed in range(42, 52):
        syn = generate_synthetic(seed=seed)
        flat = train_tokenizer(
            [syn.train_text], vocab_size=512, n_levels=1, split="chunks"
        )
        stats = collect_provenance(flat, syn.train_word_objects, text=syn.train_text)
        tok, row = summarize_assignment(flat, stats, "morph", 0.6, seed)
        p = pub[(seed, "morph")]
        same_n = int(p["n_l1"]) == row["n_l1"]
        print(
            f"{seed:>4} {len(flat_rules(flat)):>5} {row['n1_requested']:>2} "
            f"{row['n_l1']:>2} {row['n_l2']:>2} {row['demoted_or_short']:>7} "
            f"{row['closure_violations']:>11} "
            f"{'ok' if row['brute_match'] else 'FAIL':>5} "
            f"{row['summary'][0][1]:>8} {row['summary'][0][3]:>9} "
            f"{row['summary'][0][4]:>10} csv_n_l1_match={same_n}"
        )
        if row["demoted_or_short"] != 0:
            print("  !! demotion occurred in a published morph run")


# ---------------------------------------------------------------------------
# 3. hand-built DAGs
# ---------------------------------------------------------------------------


def make_tokenizer(specs, terminals="abcdefghij", ranks=None):
    """specs: list of (left_symbol, right_symbol, new_symbol)."""
    chars = list(terminals)
    char_to_id = {c: i for i, c in enumerate(chars)}
    unk_id = len(chars)
    eow_id = unk_id + 1
    expansions = {i: c for c, i in char_to_id.items()}
    expansions[unk_id] = "<unk>"
    expansions[eow_id] = EOW
    rules = []
    for pos, (left, right, new) in enumerate(specs):
        if new in expansions:
            raise ValueError(f"duplicate symbol {new}")
        expansions[new] = expansions[left] + expansions[right]
        rank = pos if ranks is None else ranks[pos]
        rules.append(MergeRule(left, right, new, rank, 0))
    return Tokenizer(
        expansions=expansions,
        char_to_id=char_to_id,
        eow_id=eow_id,
        levels=[rules],
        unk_id=unk_id,
    )


def stats_from_scores(specs, scores):
    """scores: iterable of (intra, cross), one per rule."""
    stats = {}
    for (left, right, _), (intra, cross) in zip(specs, scores):
        stats[(left, right)] = (intra, cross)
    return stats


def audit_hand_chain():
    print("\n=== 3. hand-built chain/diamond DAG, morph ===")
    # terminals a..j are ids 0..9.  new symbols start at 12.
    specs = [
        (0, 1, 12),   # r0 a+b -> X
        (12, 2, 13),  # r1 X+c -> Y   depends r0
        (13, 3, 14),  # r2 Y+d -> Z   depends r1
        (4, 5, 15),   # r3 e+f -> U
        (15, 6, 16),  # r4 U+g -> V   depends r3
        (7, 8, 17),   # r5 h+i -> W
        (9, 0, 18),   # r6 j+a -> Q
        (1, 2, 19),   # r7 b+c -> R
        (3, 4, 20),   # r8 d+e -> S
        (5, 6, 21),   # r9 f+g -> T
    ]
    scores = [
        (0, 1),  # r0 cross, not selected
        (1, 0),  # r1 selected, must demote
        (1, 0),  # r2 selected, must cascade
        (0, 1),  # r3 not selected
        (1, 0),  # r4 selected, must demote
        (1, 0),  # r5 selected, stays
        (1, 0),  # r6 selected, stays
        (1, 0),  # r7 selected, stays
        (0, 0),  # r8 unseen 0.5
        (0, 0),  # r9 unseen 0.5
    ]
    tok = make_tokenizer(specs)
    stats = stats_from_scores(specs, scores)
    deps, _ = dep_map(tok)
    seed, _ = seed_level1(tok, stats, "morph", 0.6, 7)
    expected = brute_closure(seed, deps)
    out = assign_levels(tok, stats, mode="morph", level1_fraction=0.6, seed=7)
    level1 = {r.new for r in out.levels[0]}
    print("rank rule            deps       score selected final")
    for rank, ((l, r, new), (a, b)) in enumerate(zip(specs, scores)):
        score = 0.5 if a + b == 0 else a / (a + b)
        print(
            f"{rank:>4} {tok.expansions[l]}+{tok.expansions[r]}->{tok.expansions[new]:<4} "
            f"{str(sorted(deps[new])):<10} {score:<5} "
            f"{'yes' if new in seed else 'no':<8} "
            f"{'L1' if new in level1 else 'L2'}"
        )
    print(f"requested n1={requested_n1(len(specs), 0.6)} seed L1={len(seed)} "
          f"brute L1={len(expected)} actual L1={len(level1)} "
          f"brute_match={expected == level1}")
    print(f"closure violations={len(closure_violations(out)[0])} "
          f"partition={partition_check(tok, out)}")


def audit_hand_diamond():
    print("\n=== 3b. hand-built diamond with a demoted middle node ===")
    # r0 -> X; r1: X+a -> Y; r2: X+b -> Z; r3: Y+Z -> W.
    # Select r0, r1, r3; r2 is not selected, so r3 must cascade out.
    specs = [
        (0, 1, 12),   # r0 a+b -> X
        (12, 2, 13),  # r1 X+c -> Y  dep r0
        (12, 3, 14),  # r2 X+d -> Z  dep r0
        (13, 14, 15),  # r3 Y+Z -> W dep r1,r2
        (4, 5, 16),   # r4 e+f -> U
        (6, 7, 17),   # r5 g+h -> V
    ]
    scores = [(1, 0), (1, 0), (0, 1), (1, 0), (1, 0), (0, 0)]
    tok = make_tokenizer(specs)
    stats = stats_from_scores(specs, scores)
    out = assign_levels(tok, stats, mode="morph", level1_fraction=0.5, seed=11)
    # n1 = int(0.5*6) = 3: scores put r0,r1,r3 in the seed; r2 is cross.
    seed, n1 = seed_level1(tok, stats, "morph", 0.5, 11)
    expected = brute_closure(seed, dep_map(tok)[0])
    level1 = {r.new for r in out.levels[0]}
    print(f"n1={n1} seed={sorted(seed)} expected={sorted(expected)} "
          f"actual={sorted(level1)} brute_match={expected == level1}")
    print(f"closure violations={len(closure_violations(out)[0])} "
          f"partition={partition_check(tok, out)}")


# ---------------------------------------------------------------------------
# 4. random DAG stress test
# ---------------------------------------------------------------------------


def audit_rank_violation():
    print("\n=== 3c. hand DAG with dependency rank > dependent rank ===")
    # Valid BPE never produces this, but assign_levels does not validate it.
    # rC (rank 1) depends on rB (rank 5); rB is demoted after rC is processed,
    # so the single pass leaves rC in L1 with a level-2 dependency.
    specs = [
        (0, 1, 12),    # rA rank 0: a+b -> X, not selected
        (12, 2, 13),   # rB rank 5: X+c -> Y, selected then demoted
        (13, 3, 14),   # rC rank 1: Y+d -> Z, selected before rB is examined
        (4, 5, 15),    # filler, selected
        (6, 7, 16),    # filler, selected
    ]
    ranks = [0, 5, 1, 2, 3]
    scores = [(0, 1), (1, 0), (1, 0), (1, 0), (1, 0)]
    tok = make_tokenizer(specs, ranks=ranks)
    stats = stats_from_scores(specs, scores)
    out = assign_levels(tok, stats, mode="morph", level1_fraction=0.8, seed=3)
    level1 = {r.new for r in out.levels[0]}
    seed, n1 = seed_level1(tok, stats, "morph", 0.8, 3)
    expected = brute_closure(seed, dep_map(tok)[0])
    bad, _ = closure_violations(out)
    print(f"n1={n1} seed={sorted(seed)} brute_expected={sorted(expected)} "
          f"actual={sorted(level1)}")
    print(f"closure violations={bad}")
    print(f"rank_order_check={rank_order_check(tok)}")


def random_dag(seed, n_rules=20, n_terminals=8):
    rng = random.Random(seed)
    terminals = list(range(n_terminals))
    expansions = {i: chr(ord("a") + i) for i in terminals}
    unk_id = n_terminals
    eow_id = n_terminals + 1
    expansions[unk_id] = "<unk>"
    expansions[eow_id] = EOW
    next_id = eow_id + 1
    nonterms = []
    specs = []
    for _ in range(n_rules):
        def pick():
            if nonterms and rng.random() < 0.6:
                return rng.choice(nonterms)
            return rng.choice(terminals + nonterms)

        left, right = pick(), pick()
        new = next_id
        next_id += 1
        specs.append((left, right, new))
        expansions[new] = f"<{new}>"
        nonterms.append(new)
    rules = [
        MergeRule(l, r, n, rank, 0) for rank, (l, r, n) in enumerate(specs)
    ]
    char_to_id = {chr(ord("a") + i): i for i in terminals}
    tok = Tokenizer(expansions, char_to_id, eow_id, [rules], unk_id)
    stats = {}
    for (l, r, _) in specs:
        if rng.random() < 0.85:
            stats[(l, r)] = (rng.randrange(0, 6), rng.randrange(0, 6))
    return tok, stats


def audit_random_dags(n_dags=300):
    print("\n=== 4. random DAG stress test ===")
    failures = Counter()
    examples = []
    rcd_short = 0
    for s in range(n_dags):
        n_rules = random.Random(1000 + s).randrange(5, 41)
        tok, stats = random_dag(s, n_rules=n_rules, n_terminals=8)
        rc = rank_order_check(tok)
        if rc["rank_order_bad"]:
            failures["dag:rank_order"] += 1
        fraction = random.Random(2000 + s).uniform(0.25, 0.8)
        for mode in ("frequency", "morph", "reverse", "random_closed"):
            try:
                out = assign_levels(
                    tok, stats, mode=mode, level1_fraction=fraction, seed=s
                )
            except Exception as exc:  # noqa: BLE001
                failures[f"{mode}:exception:{type(exc).__name__}"] += 1
                if len(examples) < 8:
                    examples.append((s, mode, "exception", repr(exc)))
                continue
            part = partition_check(tok, out)
            if not (part["union_ok"] and part["disjoint"] and part["count_ok"]):
                failures[f"{mode}:partition"] += 1
                if len(examples) < 8:
                    examples.append((s, mode, "partition", part))
            bad, unresolved = closure_violations(out)
            if bad:
                failures[f"{mode}:closure"] += 1
                if len(examples) < 8:
                    examples.append((s, mode, "closure", bad[:3]))
            if unresolved:
                failures[f"{mode}:unresolved"] += 1
            n1 = requested_n1(n_rules, fraction)
            if mode in ("morph", "reverse"):
                seed_set, _ = seed_level1(tok, stats, mode, fraction, s)
                expected = brute_closure(seed_set, dep_map(tok)[0])
                actual = {r.new for r in out.levels[0]}
                if expected != actual:
                    failures[f"{mode}:brute"] += 1
                    if len(examples) < 8:
                        examples.append(
                            (s, mode, "brute", (sorted(expected), sorted(actual)))
                        )
            if mode in ("frequency", "random_closed") and len(out.levels[0]) != n1:
                failures[f"{mode}:size"] += 1
                if mode == "random_closed":
                    rcd_short += 1
                if len(examples) < 8:
                    examples.append(
                        (s, mode, "size", (n1, len(out.levels[0]), n_rules))
                    )
    print(f"dags={n_dags} modes=4 checks per assignment")
    print(f"failures={dict(failures) if failures else 'none'}")
    print(f"random_closed size shortfalls={rcd_short}")
    for ex in examples:
        print(f"  example seed={ex[0]} mode={ex[1]} kind={ex[2]}: {ex[3]}")
    return failures


# ---------------------------------------------------------------------------
# 5. repeated-child topological duplicate in random_closed
# ---------------------------------------------------------------------------


def audit_random_closed_duplicate():
    print("\n=== 5. random_closed repeated-child repro ===")
    # r0: a+b -> X ; r1: X+X -> Y ; r2..r4 independent.
    # dependents[r0] contains r1 twice.  A random linear extension can emit
    # r1 twice back to back, so order[:n1] holds fewer than n1 distinct rules.
    specs = [
        (0, 1, 12),   # r0
        (12, 12, 13),  # r1, left == right nonterminal
        (2, 3, 14),   # r2
        (4, 5, 15),   # r3
        (6, 7, 16),   # r4
    ]
    tok = make_tokenizer(specs)
    stats = stats_from_scores(specs, [(1, 0)] * 5)
    n1 = requested_n1(len(specs), 0.75)
    print(f"rules={len(specs)} fraction=0.75 requested n1={n1}")
    found = None
    for seed in range(200):
        out = assign_levels(
            tok, stats, mode="random_closed", level1_fraction=0.75, seed=seed
        )
        if len(out.levels[0]) != n1:
            found = (seed, out)
            break
    if found is None:
        print("no shortfall in 200 seeds")
        return
    seed, out = found
    print(f"first seed with shortfall: {seed}; "
          f"actual L1={len(out.levels[0])} requested={n1} "
          f"L2={len(out.levels[1])}")
    print(f"closure violations={len(closure_violations(out)[0])} "
          f"partition={partition_check(tok, out)}")
    # Show the dependency fan-out that causes the duplicate.
    deps, _ = dep_map(tok)
    print(f"deps: " + ", ".join(
        f"{tok.expansions[n]}<-{sorted(tok.expansions[x] for x in d)}"
        for n, d in deps.items()
    ))


# ---------------------------------------------------------------------------
# 6. edge cases and CSV honesty
# ---------------------------------------------------------------------------


def audit_edges():
    print("\n=== 6. edge cases ===")
    zero = make_tokenizer([])
    out = assign_levels(zero, {}, mode="morph", level1_fraction=0.6, seed=1)
    print(f"0 rules: levels={[len(l) for l in out.levels]} (no crash)")
    one = make_tokenizer([(0, 1, 12)])
    for frac in (0.0, 0.6, 1.0):
        out = assign_levels(one, {}, mode="frequency", level1_fraction=frac, seed=1)
        print(f"1 rule, fraction={frac}: requested={requested_n1(1, frac)} "
              f"levels={[len(l) for l in out.levels]}")
    five = make_tokenizer([(0, 1, 12), (2, 3, 13), (4, 5, 14), (6, 7, 15), (8, 9, 16)])
    for frac in (0.0, 0.6, 1.0):
        out = assign_levels(five, {}, mode="frequency", level1_fraction=frac, seed=1)
        print(f"5 rules, fraction={frac}: requested={requested_n1(5, frac)} "
              f"L1/L2={[len(l) for l in out.levels]}")


def load_csv(path):
    with open(path, encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def audit_csv():
    print("\n=== 7. published CSV internal consistency ===")
    rows = load_csv(ROOT / "results" / "level_assignment_v512_10seeds.csv")
    print(f"rows={len(rows)}")
    bad = 0
    for r in rows:
        n1, n2 = int(r["n_l1"]), int(r["n_l2"])
        l1 = sum(int(r[f"l1_{k}"]) for k in ("intra", "cross", "unseen"))
        l2 = (
            sum(int(r[f"l2_{k}"]) for k in ("intra", "cross", "unseen"))
            if n2
            else 0
        )
        if n1 + n2 != 490 or l1 != n1 or l2 != n2:
            bad += 1
    print(f"v512 rows with size/count inconsistency: {bad}")
    for mode in ("frequency", "morph", "random_closed"):
        modes = [r for r in rows if r["mode"] == mode]
        sizes = [int(r["n_l1"]) for r in modes]
        print(f"  {mode:<13} L1 sizes={sorted(set(sizes))} requested=294 "
              f"per-seed={sizes} demotions={[294 - s for s in sizes]}")
    print("  morph all L2 strict-cross:",
          all(int(r["l2_intra"]) == 0 for r in rows if r["mode"] == "morph"))
    tri = load_csv(ROOT / "results" / "level_assignment.csv")
    print(f"3-vocab rows={len(tri)}")
    for vocab in (512, 1024, 2048):
        n_rules = 490 if vocab == 512 else (1002 if vocab == 1024 else 2026)
        req = requested_n1(n_rules, 0.6)
        for mode in ("frequency", "morph", "random_closed", "reverse"):
            sel = [r for r in tri if int(r["vocab"]) == vocab and r["mode"] == mode]
            sizes = [int(r["n_l1"]) for r in sel]
            print(f"  v{vocab:<4} {mode:<13} requested={req} L1={sizes} "
                  f"demoted/short={[req - s for s in sizes]}")


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------


def main():
    print(f"cwd={Path.cwd()}")
    audit_v512_seed42()
    audit_hand_chain()
    audit_hand_diamond()
    audit_rank_violation()
    failures = audit_random_dags()
    audit_random_closed_duplicate()
    audit_edges()
    audit_csv()
    print("\n=== overall ===")
    print(f"random DAG failures: {dict(failures) if failures else 'none'}")


if __name__ == "__main__":
    main()
