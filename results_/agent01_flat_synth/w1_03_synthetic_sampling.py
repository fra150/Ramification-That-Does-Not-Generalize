"""W1-03 red team: does the v512 morph effect depend on the synthetic corpus?

The published v512 result uses a synthetic corpus with two generator choices:

1. training words are sampled with a Zipf(1.1) rank-frequency law;
2. a period is inserted after ~7% of word tokens.

This script rebuilds the same word inventory and the same held-out test text,
then varies only those two choices:

    baseline           zipf + periods   (the published corpus)
    uniform            uniform + periods
    noperiod           zipf, no periods
    uniform_noperiod   uniform, no periods

For each corpus it reruns the level-assignment ablation at v512 with
``frequency``, ``morph`` and ``random_closed`` (plus ``flat`` as a reference
and ``morph_type`` as a diagnostic that drops the token-frequency weighting of
provenance). Five seeds, paired by seed and by test text.

Outputs go to results/redteam/; nothing under results/ is overwritten.

Usage:
    python results/redteam/w1_03_synthetic_sampling.py
"""

from __future__ import annotations

import argparse
import csv
import heapq
import math
import os
import random
import statistics
import sys
import time
from collections import Counter, defaultdict

import numpy as np
from scipy import stats as st

# The script lives in results/redteam/, three levels below the project root.
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from ramified.bpe import EOW, MergeRule, Tokenizer, train_tokenizer
from ramified.codec import normalize, roundtrip_ok, tokenize_chunks
from ramified.levels import assign_levels, collect_provenance, level_summary
from ramified.metrics import decode_exact, mean_boundary_f1
from ramified.synthetic import generate_synthetic

PERIOD_RATE = 0.07


def _pair_counter(seq):
    counts = Counter()
    for i in range(len(seq) - 1):
        counts[(seq[i], seq[i + 1])] += 1
    return counts


def train_bpe_fast(
    sequences, freqs, n_merges, next_id, level, expansions, rebuild_every=25
):
    """Drop-in equivalent of ``ramified.bpe.train_bpe``, without duplicate pushes.

    The source trainer pushes every changed count onto the lazy heap, including
    repeated pushes of the same pair inside one word update. That is correct but
    quadratic under near-uniform word frequencies: every word contributes a
    push for the same common pair, and the heap fills with stale duplicates.
    This version aggregates the old and new pairs of a word, updates the counts
    once, and pushes one current entry per affected pair. It also rebuilds the
    heap periodically, which only discards stale entries.

    It returns exactly the same merges as the source trainer; the red-team
    harness checks rule equality field by field before using it.
    """
    words = [list(s) for s in sequences]
    pair_counts = Counter()
    pair_words = defaultdict(set)

    for wi, seq in enumerate(words):
        f = freqs[wi]
        for p, n in _pair_counter(seq).items():
            pair_counts[p] += n * f
            pair_words[p].add(wi)

    heap = [(-c, p) for p, c in pair_counts.items() if c > 0]
    heapq.heapify(heap)
    merges = []

    for step in range(n_merges):
        pair = None
        count = None
        while heap:
            negc, p = heapq.heappop(heap)
            if pair_counts.get(p, 0) == -negc and -negc > 0:
                pair = p
                count = -negc
                break
        if pair is None:
            break

        a, b = pair
        new_id = next_id
        next_id += 1
        merges.append(MergeRule(a, b, new_id, step, level, count))
        expansions[new_id] = expansions[a] + expansions[b]

        for wi in list(pair_words.get(pair, ())):
            seq = words[wi]
            if not any(
                seq[i] == a and seq[i + 1] == b for i in range(len(seq) - 1)
            ):
                pair_words[pair].discard(wi)
                continue

            f = freqs[wi]
            old = _pair_counter(seq)
            for p, n in old.items():
                new_count = pair_counts.get(p, 0) - n * f
                if new_count <= 0:
                    pair_counts.pop(p, None)
                    pair_words.pop(p, None)
                else:
                    pair_counts[p] = new_count
                    bucket = pair_words.get(p)
                    if bucket is not None:
                        bucket.discard(wi)

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

            new = _pair_counter(new_seq)
            for p, n in new.items():
                pair_counts[p] = pair_counts.get(p, 0) + n * f
                pair_words[p].add(wi)

            for p in set(old) | set(new):
                c = pair_counts.get(p, 0)
                if c > 0:
                    heapq.heappush(heap, (-c, p))

        if rebuild_every and (step + 1) % rebuild_every == 0:
            heap = [(-c, p) for p, c in pair_counts.items() if c > 0]
            heapq.heapify(heap)

    return merges, words


def train_tokenizer_fast(texts, vocab_size, n_levels=1, split="chunks"):
    """Same setup as ``ramified.bpe.train_tokenizer`` with the fast inner loop."""
    words = []
    for text in texts:
        if split == "chunks":
            words.extend(text.split())
        else:
            import re

            words.extend(re.findall(r"\w+|[^\w\s]", text))

    freq = Counter(words)
    if not freq:
        raise ValueError("empty corpus")

    chars = sorted(set("".join(freq.keys())))
    char_to_id = {c: i for i, c in enumerate(chars)}
    unk_id = len(chars)
    eow_id = unk_id + 1
    expansions = {i: c for c, i in char_to_id.items()}
    expansions[unk_id] = "<unk>"
    expansions[eow_id] = EOW

    next_id = eow_id + 1
    sequences = [[char_to_id[c] for c in w] + [eow_id] for w in freq]
    freqs = list(freq.values())
    total_merges = max(0, vocab_size - len(chars) - 2)

    levels = []
    for level in range(n_levels):
        merges, sequences = train_bpe_fast(
            sequences, freqs, total_merges, next_id, level, expansions
        )
        next_id += len(merges)
        levels.append(merges)
        if not merges:
            break

    return Tokenizer(expansions, char_to_id, eow_id, levels, unk_id)


def verify_trainers(text, vocab, test_words):
    """Compare source and fast trainers: rules, expansions, tokenization."""
    src = train_tokenizer([text], vocab_size=vocab, n_levels=1, split="chunks")
    fast = train_tokenizer_fast([text], vocab_size=vocab, n_levels=1, split="chunks")
    rule_fields = ("left", "right", "new", "rank", "level", "count")
    src_rules = [
        tuple(getattr(r, f) for f in rule_fields) for r in src.levels[0]
    ]
    fast_rules = [
        tuple(getattr(r, f) for f in rule_fields) for r in fast.levels[0]
    ]
    same_expansions = src.expansions == fast.expansions
    same_encoding = all(
        src.encode(w.word, eow=True) == fast.encode(w.word, eow=True)
        for w in test_words
    )
    return {
        "vocab": vocab,
        "n_rules_src": len(src_rules),
        "n_rules_fast": len(fast_rules),
        "same_rules": src_rules == fast_rules,
        "same_expansions": same_expansions,
        "same_test_encoding": same_encoding,
    }


def encode_ids_fast(tok, word, eow=True):
    """Source-equivalent integer encoder that does not build Node objects."""
    nodes = [tok.char_to_id.get(c, tok.unk_id) for c in word]
    if eow:
        nodes.append(tok.eow_id)
    for level in range(len(tok.levels)):
        ranks = tok.level_rank[level]
        mapping = tok.level_map[level]
        while True:
            best_rank = None
            best_idx = -1
            for i in range(len(nodes) - 1):
                r = ranks.get((nodes[i], nodes[i + 1]))
                if r is not None and (best_rank is None or r < best_rank):
                    best_rank = r
                    best_idx = i
            if best_idx < 0:
                break
            left = nodes[best_idx]
            right = nodes[best_idx + 1]
            nodes[best_idx:best_idx + 2] = [mapping[(left, right)]]
    return nodes


def token_counts_cached(tok, text):
    """Counter of token ids over ``text``; counts only, order is irrelevant."""
    counts = Counter()
    for chunk, n in Counter(normalize(text).split(" ")).items():
        if not chunk:
            continue
        for token_id in encode_ids_fast(tok, chunk, eow=True):
            counts[token_id] += n
    return counts


def bpc_from_counts(train_counts, test_counts, vocab_size, char_len, alpha=0.1):
    if char_len == 0:
        return float("nan")
    total = sum(train_counts.values())
    if total == 0:
        return float("nan")
    denom = total + alpha * vocab_size
    bits = 0.0
    for token_id, n in test_counts.items():
        bits -= n * math.log2((train_counts.get(token_id, 0) + alpha) / denom)
    return bits / char_len


def _walk(node, start, tok, boundaries, stats):
    """Copy of ramified.levels._walk, kept local so provenance caching is exact."""
    if not node.children:
        expansion = tok.expansions[node.symbol]
        if expansion == EOW:
            return start
        return start + len(expansion)

    end = start
    for child in node.children:
        end = _walk(child, end, tok, boundaries, stats)

    if len(node.children) == 2:
        left, right = node.children
        key = (left.symbol, right.symbol)
        crosses = any(start < b < end for b in boundaries)
        intra, cross = stats[key]
        if crosses:
            stats[key] = (intra, cross + 1)
        else:
            stats[key] = (intra + 1, cross)
    return end


def collect_provenance_cached(tok, word_objects, text=None):
    """Same result as ``ramified.levels.collect_provenance``, one walk per type.

    The source walks every token occurrence. Provenance counts are linear in
    occurrences, so walking each unique word once and multiplying by its text
    frequency is exact. This is the difference between ~9k and ~120k tree
    walks per condition.
    """
    if len(tok.levels) != 1:
        raise ValueError("collect_provenance_cached expects a flat tokenizer")
    by_word = {w.word: w for w in word_objects}
    if text is None:
        items = Counter(w.word for w in word_objects)
    else:
        items = Counter(t for t in normalize(text).split(" ") if t in by_word)

    totals = defaultdict(lambda: (0, 0))
    for word_str, n in items.items():
        word = by_word[word_str]
        local = defaultdict(lambda: (0, 0))
        pos = 0
        for node in tok.encode_nodes(word.word, eow=True):
            pos = _walk(node, pos, tok, word.boundaries, local)
        for key, (intra, cross) in local.items():
            ti, tc = totals[key]
            totals[key] = (ti + intra * n, tc + cross * n)
    return dict(totals)


def evaluate(tok, train_text, test_text, test_words):
    train_counts = token_counts_cached(tok, train_text)
    test_counts = token_counts_cached(tok, test_text)
    norm_len = len(normalize(test_text))
    precision, recall, f1 = mean_boundary_f1(tok, test_words)
    return {
        "n_levels": len(tok.levels),
        "n_l1": len(tok.levels[0]) if tok.levels else 0,
        "n_l2": len(tok.levels[1]) if len(tok.levels) > 1 else 0,
        "tokens_per_char": sum(test_counts.values()) / max(1, norm_len),
        "bits_per_char": bpc_from_counts(
            train_counts, test_counts, tok.vocab_size, norm_len
        ),
        "boundary_precision": precision,
        "boundary_recall": recall,
        "boundary_f1": f1,
        "decode_word": decode_exact(tok, test_words),
        "decode_doc": 1.0 if roundtrip_ok(test_text, tok) else 0.0,
    }


def summary_fields(tok, stats):
    out = {}
    for row in level_summary(tok, stats):
        prefix = f"l{row['level'] + 1}"
        out[f"{prefix}_intra"] = row["intra"]
        out[f"{prefix}_cross"] = row["cross"]
        out[f"{prefix}_unseen"] = row["unseen"]
    return out


def corpus_stats(text, train_words):
    tokens = text.split()
    words = [t for t in tokens if t != "."]
    counts = Counter(words)
    n = len(words)
    probs = [c / n for c in counts.values()]
    entropy = -sum(p * math.log2(p) for p in probs if p > 0)
    ordered = counts.most_common()
    return {
        "train_tokens": len(tokens),
        "train_word_tokens": n,
        "train_periods": len(tokens) - n,
        "period_fraction": (len(tokens) - n) / max(1, len(tokens)),
        "train_types": len(counts),
        "train_word_inventory": len(train_words),
        "type_coverage": len(counts) / max(1, len(train_words)),
        "top1_share": ordered[0][1] / n if ordered else 0.0,
        "top10_share": sum(c for _, c in ordered[:10]) / n if ordered else 0.0,
        "entropy_bits_per_word": entropy,
        "mean_word_len": sum(len(w) for w in words) / max(1, n),
        "alphabet_size": len(set("".join(counts))),
        "singleton_types": sum(1 for c in counts.values() if c == 1),
    }


def make_corpora(seed):
    """Return the four training texts plus shared held-out data.

    ``generate_synthetic`` gives the published corpus. The no-period variant
    removes the period tokens from it, so the Zipf-sampled word sequence is
    untouched. The uniform variants draw the same number of word tokens
    uniformly from the same 9,216-word training inventory. Test text and test
    words are shared across all four conditions so every comparison is paired.
    """
    syn = generate_synthetic(seed=seed)
    tokens = syn.train_text.split()
    zipf_words = [t for t in tokens if t != "."]
    n_words = len(zipf_words)

    rng_np = np.random.default_rng(seed * 1_000_003 + 11)
    uni_idx = rng_np.integers(0, len(syn.train_words), size=n_words)
    rng_py = random.Random(seed * 1_000_003 + 12)
    uniform_words = []
    for i in uni_idx:
        uniform_words.append(syn.train_words[int(i)])
        if rng_py.random() < PERIOD_RATE:
            uniform_words.append(".")

    texts = {
        "baseline": " ".join(tokens),
        "uniform": " ".join(uniform_words),
        "noperiod": " ".join(zipf_words),
        "uniform_noperiod": " ".join(w for w in uniform_words if w != "."),
    }
    return syn, texts


def paired_stats(rows, condition, mode_a, mode_b, metric):
    by_seed = {}
    for row in rows:
        if row["condition"] != condition:
            continue
        by_seed.setdefault(row["seed"], {})[row["mode"]] = row[metric]
    deltas = [v[mode_a] - v[mode_b] for v in by_seed.values() if mode_a in v and mode_b in v]
    n = len(deltas)
    mean = statistics.mean(deltas)
    sd = statistics.stdev(deltas) if n > 1 else 0.0
    se = sd / math.sqrt(n) if n > 0 else float("nan")
    t = mean / se if se and se > 0 else float("nan")
    p = 2 * st.t.sf(abs(t), df=n - 1) if n > 1 else float("nan")
    crit = st.t.ppf(0.975, df=n - 1) if n > 1 else float("nan")
    if metric == "bits_per_char":
        wins = sum(1 for d in deltas if d < 0)
        losses = sum(1 for d in deltas if d > 0)
    else:
        wins = sum(1 for d in deltas if d > 0)
        losses = sum(1 for d in deltas if d < 0)
    ties = n - wins - losses
    return {
        "kind": "paired_mode",
        "condition": condition,
        "metric": metric,
        "mode_a": mode_a,
        "mode_b": mode_b,
        "n": n,
        "mean": mean,
        "std": sd,
        "se": se,
        "t": t,
        "p": p,
        "ci_low": mean - crit * se if n > 1 else float("nan"),
        "ci_high": mean + crit * se if n > 1 else float("nan"),
        "wins": wins,
        "losses": losses,
        "ties": ties,
    }


def effect_shift_stats(rows, metric, condition_a, condition_b, mode_a="morph", mode_b="frequency"):
    """Paired by seed: is the morph-minus-control effect different in A vs B?"""
    per_cond = {}
    for condition in (condition_a, condition_b):
        by_seed = {}
        for row in rows:
            if row["condition"] != condition:
                continue
            by_seed.setdefault(row["seed"], {})[row["mode"]] = row[metric]
        per_cond[condition] = {
            seed: v[mode_a] - v[mode_b] for seed, v in by_seed.items()
        }
    common = sorted(set(per_cond[condition_a]) & set(per_cond[condition_b]))
    deltas = [per_cond[condition_a][s] - per_cond[condition_b][s] for s in common]
    n = len(deltas)
    mean = statistics.mean(deltas)
    sd = statistics.stdev(deltas) if n > 1 else 0.0
    se = sd / math.sqrt(n) if n > 0 else float("nan")
    t = mean / se if se and se > 0 else float("nan")
    p = 2 * st.t.sf(abs(t), df=n - 1) if n > 1 else float("nan")
    crit = st.t.ppf(0.975, df=n - 1) if n > 1 else float("nan")
    return {
        "kind": "effect_shift",
        "condition": f"{condition_a} - {condition_b}",
        "metric": metric,
        "mode_a": f"{mode_a}-{mode_b}",
        "mode_b": "",
        "n": n,
        "mean": mean,
        "std": sd,
        "se": se,
        "t": t,
        "p": p,
        "ci_low": mean - crit * se if n > 1 else float("nan"),
        "ci_high": mean + crit * se if n > 1 else float("nan"),
        "wins": sum(1 for d in deltas if d > 0),
        "losses": sum(1 for d in deltas if d < 0),
        "ties": sum(1 for d in deltas if d == 0),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="42,43,44,45,46")
    ap.add_argument("--vocab", type=int, default=512)
    ap.add_argument("--out-dir", default="results/redteam")
    ap.add_argument(
        "--modes", default="frequency,morph,random_closed,morph_type"
    )
    ap.add_argument("--fraction", type=float, default=0.6)
    ap.add_argument(
        "--trainer", choices=("fast", "source"), default="fast",
        help="fast is a verified drop-in for source; see --verify-trainer",
    )
    ap.add_argument("--verify-trainer", action="store_true")
    ap.add_argument("--verify-cache", action="store_true")
    args = ap.parse_args()

    seeds = [int(x) for x in args.seeds.split(",") if x.strip()]
    modes = [x.strip() for x in args.modes.split(",") if x.strip()]
    conditions = ["baseline", "uniform", "noperiod", "uniform_noperiod"]
    os.makedirs(args.out_dir, exist_ok=True)
    train_fn = train_tokenizer_fast if args.trainer == "fast" else train_tokenizer

    rows = []
    corpus_rows = []
    started = time.perf_counter()

    if args.verify_trainer:
        verify_rows = []
        syn, texts = make_corpora(seeds[0])
        for condition, vocab in (
            ("baseline", args.vocab),
            ("baseline", 256),
            ("uniform", 256),
            ("noperiod", args.vocab),
            ("uniform_noperiod", 256),
        ):
            check = verify_trainers(texts[condition], vocab, syn.test_words)
            verify_rows.append(
                {"seed": seeds[0], "condition": condition, **check}
            )
            print(f"verify {condition} v{vocab}: {check}", flush=True)
        with open(
            os.path.join(args.out_dir, "w1_03_trainer_check.csv"),
            "w",
            newline="",
            encoding="utf-8",
        ) as fh:
            writer = csv.DictWriter(fh, fieldnames=list(verify_rows[0]))
            writer.writeheader()
            writer.writerows(verify_rows)
        if not all(
            r["same_rules"] and r["same_expansions"] and r["same_test_encoding"]
            for r in verify_rows
        ):
            raise SystemExit("fast trainer is not equivalent; aborting")

    if args.verify_cache:
        syn, texts = make_corpora(seeds[0])
        for condition in ("baseline", "uniform"):
            text = texts[condition]
            flat = train_tokenizer_fast(
                [text], vocab_size=args.vocab, n_levels=1, split="chunks"
            )
            cached_unweighted = collect_provenance_cached(
                flat, syn.train_word_objects
            )
            source_unweighted = collect_provenance(flat, syn.train_word_objects)
            sample = " ".join(text.split()[:20_000])
            cached_sample = collect_provenance_cached(
                flat, syn.train_word_objects, text=sample
            )
            source_sample = collect_provenance(
                flat, syn.train_word_objects, text=sample
            )
            print(
                f"cache check {condition}: unweighted_equal="
                f"{cached_unweighted == source_unweighted} "
                f"weighted20k_equal={cached_sample == source_sample}",
                flush=True,
            )
            if cached_unweighted != source_unweighted or cached_sample != source_sample:
                raise SystemExit("provenance cache is not equivalent; aborting")

    for seed in seeds:
        syn, texts = make_corpora(seed)
        for condition in conditions:
            train_text = texts[condition]
            stats_text = corpus_stats(train_text, syn.train_words)
            corpus_rows.append({"seed": seed, "condition": condition, **stats_text})
            print(
                f"seed={seed} {condition:<16} words={stats_text['train_word_tokens']} "
                f"periods={stats_text['train_periods']} types={stats_text['train_types']} "
                f"top1={stats_text['top1_share']:.3f}",
                flush=True,
            )

            t0 = time.perf_counter()
            flat = train_fn(
                [train_text], vocab_size=args.vocab, n_levels=1, split="chunks"
            )
            flat_seconds = time.perf_counter() - t0
            print(
                f"  seed={seed} {condition:<16} trained in {flat_seconds:.1f}s "
                f"({len(flat.levels[0])} merges)",
                flush=True,
            )
            stats = collect_provenance_cached(
                flat, syn.train_word_objects, text=train_text
            )
            stats_unweighted = collect_provenance_cached(
                flat, syn.train_word_objects
            )
            enc_ok = all(
                tuple(encode_ids_fast(flat, chunk, eow=True))
                == tuple(flat.encode(chunk, eow=True))
                for chunk in set(normalize(train_text).split(" "))
            )
            if not enc_ok:
                raise SystemExit("fast encoder diverged from Tokenizer.encode")

            toks = {"flat": flat}
            for mode in modes:
                if mode == "morph_type":
                    tok = assign_levels(
                        flat,
                        stats_unweighted,
                        mode="morph",
                        level1_fraction=args.fraction,
                        seed=seed * 1000 + args.vocab,
                    )
                else:
                    tok = assign_levels(
                        flat,
                        stats,
                        mode=mode,
                        level1_fraction=args.fraction,
                        seed=seed * 1000 + args.vocab,
                    )
                toks[mode] = tok

            # How much does each assignment actually change the test
            # tokenization relative to the frequency control?
            freq_tok = toks["frequency"]
            freq_enc = {
                w.word: tuple(encode_ids_fast(freq_tok, w.word, eow=True))
                for w in syn.test_words
            }

            for mode in ["flat"] + modes:
                tok = toks[mode]
                metrics = evaluate(tok, train_text, syn.test_text, syn.test_words)
                if mode == "frequency":
                    diff_frac = 0.0
                else:
                    diff = sum(
                        1
                        for w in syn.test_words
                        if tuple(encode_ids_fast(tok, w.word, eow=True))
                        != freq_enc[w.word]
                    )
                    diff_frac = diff / len(syn.test_words)
                row = {
                    "seed": seed,
                    "condition": condition,
                    "vocab": args.vocab,
                    "mode": mode,
                    "level_seconds": round(flat_seconds if mode == "flat" else 0.0, 2),
                    "train_seconds": round(flat_seconds, 2),
                    "n_merges": len(flat.levels[0]),
                    "frac_test_words_diff_vs_frequency": diff_frac,
                    **metrics,
                    **summary_fields(tok, stats),
                    **stats_text,
                }
                rows.append(row)
                print(
                    f"  seed={seed} {condition:<16} {mode:<11} "
                    f"tok/ch={metrics['tokens_per_char']:.4f} "
                    f"bpc={metrics['bits_per_char']:.4f} "
                    f"F1={metrics['boundary_f1']:.4f} "
                    f"L1={len(tok.levels[0])} "
                    f"diff={diff_frac:.3f}",
                    flush=True,
                )

    # --- statistics -------------------------------------------------------
    stat_rows = []
    for condition in conditions:
        present = {r["mode"] for r in rows if r["condition"] == condition}
        for metric in ("boundary_f1", "bits_per_char"):
            for a, b in (
                ("morph", "frequency"),
                ("morph", "random_closed"),
                ("random_closed", "frequency"),
                ("morph_type", "frequency"),
                ("morph_type", "morph"),
            ):
                if a in present and b in present:
                    stat_rows.append(paired_stats(rows, condition, a, b, metric))

    for metric in ("boundary_f1", "bits_per_char"):
        for cond_a, cond_b in (
            ("baseline", "uniform"),
            ("baseline", "noperiod"),
            ("baseline", "uniform_noperiod"),
            ("uniform", "uniform_noperiod"),
            ("noperiod", "uniform_noperiod"),
        ):
            stat_rows.append(
                effect_shift_stats(rows, metric, cond_a, cond_b)
            )

    # --- write ------------------------------------------------------------
    def write_csv(path, items):
        fieldnames = []
        for item in items:
            for key in item:
                if key not in fieldnames:
                    fieldnames.append(key)
        with open(path, "w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(items)
        print(f"wrote {path} ({len(items)} rows)")

    write_csv(os.path.join(args.out_dir, "w1_03_runs.csv"), rows)
    write_csv(os.path.join(args.out_dir, "w1_03_stats.csv"), stat_rows)
    write_csv(os.path.join(args.out_dir, "w1_03_corpus_stats.csv"), corpus_rows)

    # --- baseline reproduction check against the published CSV ------------
    published_path = os.path.join(ROOT, "results", "level_assignment_v512_10seeds.csv")
    check_rows = []
    if os.path.exists(published_path):
        with open(published_path, newline="", encoding="utf-8") as fh:
            published = list(csv.DictReader(fh))
        lookup = {
            (int(r["seed"]), r["mode"]): r
            for r in published
            if int(r["vocab"]) == args.vocab
        }
        for row in rows:
            if row["condition"] != "baseline":
                continue
            key = (row["seed"], row["mode"])
            if key not in lookup:
                continue
            ref = lookup[key]
            for metric in ("boundary_f1", "bits_per_char", "tokens_per_char"):
                got = float(row[metric])
                want = float(ref[metric])
                check_rows.append(
                    {
                        "seed": row["seed"],
                        "mode": row["mode"],
                        "metric": metric,
                        "reproduced": got,
                        "published": want,
                        "abs_diff": abs(got - want),
                    }
                )
        write_csv(os.path.join(args.out_dir, "w1_03_baseline_check.csv"), check_rows)
        worst = max((r["abs_diff"] for r in check_rows), default=float("nan"))
        print(f"baseline reproduction: worst abs diff vs published = {worst:.3e}")
    else:
        print("published CSV not found; skipped baseline reproduction check")

    elapsed = time.perf_counter() - started
    print(f"total wall time {elapsed:.1f}s")


if __name__ == "__main__":
    main()
